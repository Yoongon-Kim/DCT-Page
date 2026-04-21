#!/usr/bin/env python3
"""
Attention-mass recall on RULER — Quest-only, dense-trajectory reference.

Differences from ``attention_mass_recall_ruler.py``:

  1) Dense baseline drives decoding. No DCT-Page / Quest / ShadowKV monkey-patch.
     ``model.generate`` runs standard full attention; the recording wrapper we
     install only observes Q, K, V post-RoPE / post-cache-update and never
     alters the decode path. This isolates the selection rule from
     trajectory contamination.

  2) No sink and no explicit recent window. The KV cache is treated as a
     sequence of fixed-size pages; Quest's only unconditional keep is the
     **last page** (matches ``kv_indices_without_last`` + ``kv_last_page_idx``
     in ``baselines/quest_attn/utils/controller.py:106``).

  3) Layers 0 and 1 are excluded from analysis. Quest's
     ``_quest_skip_layer=2`` (``baselines/quest_attn/models/llama.py:538``,
     ``qwen3.py:420``) runs those layers as full attention, so recall is
     trivially 1.0 there.

  4) Quest-only metrics plus two ceilings:

     mass_recall_quest      = (Σ m[p∈QuestTopK] + m[last_page]) / 1
     mass_recall_oracle_max = (Σ m[p∈OracleTopK] + m[last_page]) / 1
     mass_recall_mass_topk  = (Σ m[p∈MassTopK]  + m[last_page]) / 1   (ceiling)
     mass_recall_last_page  = m[last_page]                           (always-kept floor)
     output_fidelity_quest      = cos(full_attn_out, quest_drop_out)
     output_fidelity_oracle_max = cos(full_attn_out, oracle_drop_out)

     Mass sum invariant: page_mass.sum(-1) + last_page_mass = 1 per head.

Quest scoring (Tang et al., MLSys 2024) — reused from the sibling script:
    score[p] = (1/√d) · Σ_d max(q[d]·K_max[p, d], q[d]·K_min[p, d])
Matches the CUDA kernel at ``baselines/quest_attn/ops/csrc/estimate.cu:68-76``.

Usage:
    python oracle/attention_mass_recall_ruler_quest.py \\
        --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --tasks niah_single_1 --num_samples 2 --seq_len 32768 \\
        --page_size 16 --top_k 128 --num_decode_steps 2 \\
        --output_dir results_quest_mass_recall --run_name smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ruler import infer_model_family
from oracle.attention_mass_recall_ruler import (
    compute_output_fidelity,
    compute_per_page_mass,
    compute_quest_scores,
)
from oracle.compare_proxy_oracle_slices import (
    ALL_TASKS,
    load_task_configs,
)


MASS_METRIC_KEYS = [
    "mass_recall_last_page",
    "mass_recall_quest",
    "mass_recall_oracle_max",
    "mass_recall_mass_topk",
]

FIDELITY_METRIC_KEYS = [
    "output_fidelity_quest",
    "output_fidelity_oracle_max",
]

METRIC_KEYS = MASS_METRIC_KEYS + FIDELITY_METRIC_KEYS


# ---------------------------------------------------------------------------
# Recording wrapper — replaces attention forward to capture post-RoPE Q, K, V
# without altering the decode path.
# ---------------------------------------------------------------------------
_recording_hook: Optional[Callable[[dict], None]] = None


def set_recording_hook(hook: Optional[Callable[[dict], None]]) -> None:
    global _recording_hook
    _recording_hook = hook


def _install_recording_forward(model: torch.nn.Module, family: str) -> None:
    """Monkey-patch attention forward on every decoder layer.

    The replacement mirrors HF's own forward body 1:1 (transformers 5.2) and
    only adds a side-effect call to the recorder after ``past_key_values.update``.
    Decoding results are unchanged.
    """
    if family == "llama":
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        target_cls = LlamaAttention
        has_qk_norm = False
        extra_attn_kwargs: dict[str, Any] = {}
    elif family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        target_cls = Qwen3Attention
        has_qk_norm = True
        extra_attn_kwargs = {}  # sliding_window handled per-instance below
    elif family == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Attention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        target_cls = Qwen2Attention
        has_qk_norm = False
        extra_attn_kwargs = {}
    else:
        raise ValueError(f"unsupported family {family!r}")

    def make_forward(_target_cls):
        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            if has_qk_norm:
                query_states = self.q_norm(
                    self.q_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                key_states = self.k_norm(
                    self.k_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
            else:
                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs,
                )

            # ---- Side-effect recorder call (decode only) -------------------
            if query_states.shape[-2] == 1 and _recording_hook is not None:
                _recording_hook({
                    "layer_idx": int(self.layer_idx),
                    "query_states": query_states,
                    "key_states_full": key_states,
                    "value_states_full": value_states,
                    "num_kv_groups": int(self.num_key_value_groups),
                })
            # ---------------------------------------------------------------

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward,
            )
            qwen3_extra = {}
            if family == "qwen3":
                qwen3_extra["sliding_window"] = getattr(self, "sliding_window", None)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **qwen3_extra,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        return forward

    target_cls.forward = make_forward(target_cls)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class QuestMassRecallRecorder:
    """Per-decode-step recorder that computes Quest mass-recall metrics inline.

    Drops records for ``layer_idx in skip_layers``. Drops records where the
    KV cache hasn't grown large enough for Quest to sparsify (num_pages <=
    top_k), matching Quest's ``need_estimate()`` semantics.
    """

    def __init__(
        self,
        num_decode_steps: int,
        page_size: int,
        top_k: int,
        skip_layers: set[int],
        group_agg_method: str,
    ):
        self.num_decode_steps = num_decode_steps
        self.page_size = page_size
        self.top_k = top_k
        self.skip_layers = set(skip_layers)
        self.group_agg_method = group_agg_method
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        if layer_idx in self.skip_layers:
            return

        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step >= self.num_decode_steps:
            return

        query_states = payload["query_states"]    # [1, H_q, 1, d]
        key_full = payload["key_states_full"]     # [1, H_kv, kv_len, d]
        value_full = payload["value_states_full"] # [1, H_kv, kv_len, d]
        num_kv_groups = int(payload["num_kv_groups"])

        bsz, H_kv, kv_len, d = key_full.shape
        _, H_q, q_len, _ = query_states.shape
        assert bsz == 1 and q_len == 1, f"expected decode step, got {query_states.shape}"
        assert H_q == H_kv * num_kv_groups

        # Quest layout: pages of size page_size; last page is always kept.
        num_pages = (kv_len + self.page_size - 1) // self.page_size
        if num_pages <= max(2, self.top_k):
            # Quest wouldn't sparsify here (need_estimate() would be False).
            return

        P = num_pages - 1                       # pages eligible for top-K
        full_region_len = P * self.page_size    # first P * page_size tokens
        tail_len = kv_len - full_region_len     # last page, 1..page_size tokens
        actual_top_k = min(self.top_k, P)

        paged_k = key_full[:, :, :full_region_len, :].view(
            bsz, H_kv, P, self.page_size, d,
        )
        paged_v = value_full[:, :, :full_region_len, :].view(
            bsz, H_kv, P, self.page_size, d,
        )
        tail_k = key_full[:, :, full_region_len:, :]    # [1, H_kv, tail_len, d]
        tail_v = value_full[:, :, full_region_len:, :]

        # Empty sink tensors (shape preserves dims for torch.cat inside helpers).
        sink_k = key_full.new_zeros(bsz, H_kv, 0, d)
        sink_v = value_full.new_zeros(bsz, H_kv, 0, d)

        device = key_full.device

        with torch.no_grad():
            # Ground-truth softmax mass per page; last-page mass returns as "recent".
            page_mass_gpu, sink_mass_gpu, tail_mass_gpu = compute_per_page_mass(
                query_states, sink_k, paged_k, tail_k, num_kv_groups,
            )
            page_mass = page_mass_gpu.float().cpu()      # [H_q, P]
            tail_mass = tail_mass_gpu.float().cpu()      # [H_q]
            # sink_mass is zero by construction; ignore.

            quest_scores_gpu = compute_quest_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )  # [H_kv, P]

            # Oracle ceiling: per-kv-head max Q·K over all tokens in page.
            # Reuse the same group-agg path by computing max over the page
            # axis of the exact dot products.
            scale = 1.0 / math.sqrt(d)
            k_exp = paged_k.repeat_interleave(num_kv_groups, dim=1).float()
            q = query_states.float()                     # [1, H_q, 1, d]
            # [1, H_q, P, S]
            qk = torch.einsum("bhqd,bhpsd->bhps", q, k_exp) * scale
            qk_max_per_page = qk.amax(dim=-1)            # [1, H_q, P]
            # Reduce query-group to kv-head (mirror compute_quest_scores agg).
            qk_max_g = qk_max_per_page.view(1, H_kv, num_kv_groups, P)
            if self.group_agg_method == "max":
                oracle_scores_gpu = qk_max_g.amax(dim=2).squeeze(0)   # [H_kv, P]
            else:
                oracle_scores_gpu = qk_max_g.mean(dim=2).squeeze(0)

            # Selections (on GPU). All at kv-head granularity.
            quest_topk_gpu = torch.topk(quest_scores_gpu, actual_top_k, dim=-1).indices
            oracle_topk_gpu = torch.topk(oracle_scores_gpu, actual_top_k, dim=-1).indices
            # mass_topk ceiling uses page_mass aggregated to kv-head level.
            page_mass_kv = page_mass_gpu.view(H_kv, num_kv_groups, P).mean(dim=1)
            mass_topk_gpu = torch.topk(page_mass_kv, actual_top_k, dim=-1).indices

            # V-aware output fidelity (full vs. "Quest drop") per selection.
            fidelity_gpu = compute_output_fidelity(
                query_states, sink_k, sink_v, paged_k, paged_v, tail_k, tail_v,
                {
                    "output_fidelity_quest": quest_topk_gpu,
                    "output_fidelity_oracle_max": oracle_topk_gpu,
                },
                num_kv_groups,
            )
            fidelity = {k: v.float().cpu() for k, v in fidelity_gpu.items()}

        # Mass recall: gather selected pages in the query-head space and add
        # the always-kept last-page mass (tail_mass).
        def _recall(topk_kv: torch.Tensor) -> torch.Tensor:
            topk_q = topk_kv.repeat_interleave(num_kv_groups, dim=0).cpu()  # [H_q, K]
            return page_mass.gather(-1, topk_q).sum(-1) + tail_mass         # [H_q]

        metrics = {
            "mass_recall_last_page": tail_mass,
            "mass_recall_quest": _recall(quest_topk_gpu),
            "mass_recall_oracle_max": _recall(oracle_topk_gpu),
            "mass_recall_mass_topk": _recall(mass_topk_gpu),
            **fidelity,
        }

        # Invariants.
        for key, tensor in metrics.items():
            lo, hi = float(tensor.min()), float(tensor.max())
            if key in FIDELITY_METRIC_KEYS:
                low_bound, high_bound = -1.0, 1.0
            else:
                low_bound, high_bound = 0.0, 1.0
            if lo < low_bound - 1e-4 or hi > high_bound + 1e-4:
                raise AssertionError(
                    f"{key} out of [{low_bound}, {high_bound}]: "
                    f"min={lo:.6f}, max={hi:.6f} "
                    f"(layer={layer_idx}, step={decode_step})"
                )

        self.records.append({
            "layer_idx": layer_idx,
            "decode_step": decode_step,
            "num_pages": int(num_pages),
            "tail_len": int(tail_len),
            "actual_top_k": int(actual_top_k),
            "num_kv_groups": num_kv_groups,
            "H_q": int(H_q),
            **{k: metrics[k].tolist() for k in METRIC_KEYS},
        })


def generate_with_mass_traces(
    model, tokenizer, sample: dict[str, Any], recorder: QuestMassRecallRecorder,
) -> tuple[list[dict[str, Any]], int]:
    """Run generate() with the Quest mass-recall hook installed."""
    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    set_recording_hook(recorder)
    try:
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=recorder.num_decode_steps,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        set_recording_hook(None)

    return recorder.records, int(input_ids.shape[1])


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def load_model(args: argparse.Namespace):
    yarn_kwargs = {}
    if "qwen3" in args.base_model.lower():
        yarn_kwargs = {
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            "max_position_embeddings": 131072,
        }
    return AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": args.cuda_device},
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
        **yarn_kwargs,
    ).eval()


def cleanup_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_family(base_model: str) -> str:
    name = base_model.lower()
    if "qwen3" in name:
        return "qwen3"
    if "qwen" in name:
        return "qwen2"
    if "llama" in name or "mistral" in name:
        return "llama"
    raise ValueError(f"unsupported model: {base_model}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Quest-only attention-mass recall vs full-attention softmax on RULER. "
            "Dense baseline drives decoding; no sink, no recent; layers 0-1 skipped."
        )
    )
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--data_root", type=Path,
                   default=Path("benchmark/data/ruler_data"))

    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128,
                   help="Quest page_budget (token_budget = page_size * top_k).")
    p.add_argument("--group_agg_method", type=str, default="max",
                   choices=["mean", "max"])

    p.add_argument("--skip_layers", type=str, default="0,1",
                   help="Comma-separated layer indices Quest runs as dense "
                        "(default '0,1' matches _quest_skip_layer=2).")

    p.add_argument("--num_decode_steps", type=int, default=20)

    p.add_argument("--output_dir", type=Path,
                   default=Path("results_quest_mass_recall"))
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _aggregate_metric_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {k: 0.0 for k in METRIC_KEYS}
    return {k: _mean([d[k] for d in dicts if k in d]) for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    start_time = time.time()
    torch.manual_seed(42)

    run_name = args.run_name or f"quest_ps{args.page_size}_topk{args.top_k}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = run_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    skip_layers = {int(x) for x in args.skip_layers.split(",") if x.strip()}

    family = _model_family(args.base_model)
    print(f"Installing recording wrapper on family={family}")
    _install_recording_forward(None, family)  # patches the class, model arg unused

    print(f"Loading model (dense): {args.base_model}")
    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _, tokenizer_family = infer_model_family(args.base_model)
    task_configs = load_task_configs()

    per_task_results: dict[str, Any] = {}

    try:
        for task in args.tasks:
            if task not in task_configs:
                print(f"  WARNING: task {task!r} not in RULER configs, skipping")
                continue
            print(f"\n{'=' * 60}\nTASK: {task}\n{'=' * 60}")

            data_path = (
                args.data_root / tokenizer_family / str(args.seq_len)
                / task / "validation.jsonl"
            )
            if not data_path.exists():
                print(f"  WARNING: data not found at {data_path}, skipping")
                continue

            with data_path.open("r", encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]
            if args.num_samples > 0:
                samples = samples[: args.num_samples]

            task_overall_records: list[dict[str, float]] = []
            task_per_layer: dict[int, list[dict[str, float]]] = {}

            sample_fp = (per_sample_dir / f"{task}.jsonl").open(
                "w", encoding="utf-8", buffering=1,
            )

            for sample_idx, sample in enumerate(
                tqdm(samples, desc=f"  {task}"), start=1,
            ):
                recorder = QuestMassRecallRecorder(
                    num_decode_steps=args.num_decode_steps,
                    page_size=args.page_size,
                    top_k=args.top_k,
                    skip_layers=skip_layers,
                    group_agg_method=args.group_agg_method,
                )
                records, input_len = generate_with_mass_traces(
                    model, tokenizer, sample, recorder,
                )
                if not records:
                    print(f"  WARNING: no traces for sample {sample['index']} "
                          f"(input_len={input_len}); skipping")
                    continue

                per_head_rows: list[dict[str, Any]] = []
                per_layer_buckets: dict[int, list[dict[str, float]]] = {}

                for rec in records:
                    layer_idx = rec["layer_idx"]
                    decode_step = rec["decode_step"]
                    num_kv_groups = rec["num_kv_groups"]
                    H_q = rec["H_q"]

                    for q in range(H_q):
                        flat = {k: rec[k][q] for k in METRIC_KEYS}
                        row = {
                            "layer_idx": layer_idx,
                            "decode_step": decode_step,
                            "q_head": q,
                            "kv_head": q // num_kv_groups,
                            "actual_top_k": rec["actual_top_k"],
                            "tail_len": rec["tail_len"],
                            **flat,
                        }
                        per_head_rows.append(row)
                        per_layer_buckets.setdefault(layer_idx, []).append(flat)
                        task_overall_records.append(flat)
                        task_per_layer.setdefault(layer_idx, []).append(flat)

                per_layer_mean = {
                    str(lyr): _aggregate_metric_dicts(bucket)
                    for lyr, bucket in sorted(per_layer_buckets.items())
                }

                sample_record = {
                    "sample_index": int(sample["index"]),
                    "input_len": input_len,
                    "num_records": len(records),
                    "per_layer_mean": per_layer_mean,
                    "per_head": per_head_rows,
                }
                sample_fp.write(
                    json.dumps(sample_record, ensure_ascii=False) + "\n"
                )

                if sample_idx % 5 == 0 or sample_idx == len(samples):
                    o = _aggregate_metric_dicts(task_overall_records)
                    print(
                        f"  [{sample_idx}/{len(samples)}] "
                        f"last_page={o['mass_recall_last_page']:.3f}  "
                        f"mass[q/o/c] = "
                        f"{o['mass_recall_quest']:.3f}/"
                        f"{o['mass_recall_oracle_max']:.3f}/"
                        f"{o['mass_recall_mass_topk']:.3f}  "
                        f"fid[q/o] = "
                        f"{o['output_fidelity_quest']:.3f}/"
                        f"{o['output_fidelity_oracle_max']:.3f}"
                    )

            sample_fp.close()

            per_task_results[task] = {
                "num_samples": len(samples),
                "overall": _aggregate_metric_dicts(task_overall_records),
                "per_layer": {
                    str(lyr): _aggregate_metric_dicts(bucket)
                    for lyr, bucket in sorted(task_per_layer.items())
                },
            }
            o = per_task_results[task]["overall"]
            print(
                f"  TASK SUMMARY\n"
                f"    last_page (always-kept floor) = {o['mass_recall_last_page']:.3f}\n"
                f"    mass   [quest/oracle/ceil]    = "
                f"{o['mass_recall_quest']:.3f} / {o['mass_recall_oracle_max']:.3f} / "
                f"{o['mass_recall_mass_topk']:.3f}\n"
                f"    fidelity [quest/oracle]       = "
                f"{o['output_fidelity_quest']:.3f} / {o['output_fidelity_oracle_max']:.3f}"
            )

        overall_task_means = [r["overall"] for r in per_task_results.values()]
        overall = _aggregate_metric_dicts(overall_task_means)

        summary = {
            "config": {
                "base_model": args.base_model,
                "seq_len": args.seq_len,
                "num_samples": args.num_samples,
                "num_decode_steps": args.num_decode_steps,
                "page_size": args.page_size,
                "top_k": args.top_k,
                "group_agg_method": args.group_agg_method,
                "skip_layers": sorted(skip_layers),
            },
            "per_task": per_task_results,
            "overall": overall,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        elapsed = (time.time() - start_time) / 60
        print(f"\n{'=' * 60}\nOVERALL RESULTS\n{'=' * 60}")
        for k in METRIC_KEYS:
            print(f"  {k:30s} = {overall[k]:.3f}")
        print(f"\n  Results: {run_dir}")
        print(f"  Total time: {elapsed:.1f} min")

    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
