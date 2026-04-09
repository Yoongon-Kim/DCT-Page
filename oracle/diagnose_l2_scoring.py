#!/usr/bin/env python3
"""
Diagnose page scoring methods by comparing them against a configurable ground truth.

All methods use group_agg=max. For each sample, captures paged_k, paged_v and
query_states at the first decode step of the last layer, then computes per-page
per-head scores:

  Ground truths (--ground_truth):
    oracle_max:          full tokens, scoring=max             max_i <q, k_i>
    output_contribution: per-page contribution to attention output
                         || sum_{i in page} softmax(s_i) * v_i ||

  Methods compared against ground truth:
    oracle_max:    full tokens, scoring=max
    oracle_mean:   full tokens, scoring=mean         mean_i <q, k_i>
    proxy_max:     DCT compressed tokens, scoring=max   max_c <q, comp_k_c>
    proxy_mean:    DCT compressed tokens, scoring=mean  mean_c <q, comp_k_c>
    l2_energy:     full tokens, L2 scoring           sqrt(sum_i <q, k_i>^2)

For each method reports:
  - recall: fraction of GT top-k pages also selected by this method
  - false positives / false negatives
  - neg_gt_in_topk: pages in method's top-k with negative GT score
  - fn_rank: average rank (in this method) of missed GT pages
  - fn_gt_score: average GT score of missed pages
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

GROUND_TRUTHS = ["oracle_max", "output_contribution"]


# ---------------------------------------------------------------------------
# Debug hook recorder: captures first decode step of last layer
# ---------------------------------------------------------------------------
class L2DiagnosticRecorder:
    """Capture paged_k, paged_v and query_states at the first decode step of the last layer."""

    def __init__(self):
        self.result: dict[str, Any] | None = None
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step != 0:
            return

        paged_k = payload.get("paged_k")
        paged_v = payload.get("paged_v")
        query_states = payload.get("query_states")
        if paged_k is None or query_states is None:
            return

        self.result = {
            "layer_idx": layer_idx,
            "num_pages": int(payload["num_pages"]),
            "actual_top_k": int(payload["actual_top_k"]),
            "page_size": int(payload["page_size"]),
            "num_kv_groups": int(payload["num_kv_groups"]),
            "sink_k": payload.get("sink_k"),    # [bsz, kv_heads, sink_size, head_dim]
            "sink_v": payload.get("sink_v"),
            "paged_k": paged_k,                 # [bsz, kv_heads, num_pages, page_size, head_dim]
            "paged_v": paged_v,                 # [bsz, kv_heads, num_pages, page_size, head_dim]
            "recent_k": payload.get("recent_k"),  # [bsz, kv_heads, recent_size, head_dim]
            "recent_v": payload.get("recent_v"),
            "query_states": query_states,       # [bsz, num_heads, 1, head_dim]
        }

    def reset(self):
        self.result = None
        self._step_by_layer.clear()


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def compute_per_token_scores(
    query_states: torch.Tensor,   # [bsz, num_heads, 1, head_dim]
    paged_k: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    num_kv_groups: int,
) -> torch.Tensor:
    """Compute per-token Q·K scores with group_agg=max.

    Returns: [bsz, kv_heads, num_pages, page_size] float32
    """
    bsz, num_heads, _, head_dim = query_states.shape
    _, kv_heads, num_pages, page_size, _ = paged_k.shape
    scaling = head_dim ** -0.5

    q = query_states.squeeze(2).reshape(bsz, kv_heads, num_kv_groups, head_dim).float()
    k = paged_k.float()

    # dots: [bsz, kv_heads, num_kv_groups, num_pages, page_size]
    dots = torch.einsum('bhgd,bhnsd->bhgns', q * scaling, k)

    # group_agg=max
    return dots.max(dim=2).values  # [bsz, kv_heads, num_pages, page_size]


def compute_output_contribution(
    query_states: torch.Tensor,   # [bsz, num_heads, 1, head_dim]
    paged_k: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    paged_v: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    num_kv_groups: int,
    sink_k: torch.Tensor | None = None,    # [bsz, kv_heads, sink_size, head_dim]
    sink_v: torch.Tensor | None = None,
    recent_k: torch.Tensor | None = None,  # [bsz, kv_heads, recent_size, head_dim]
    recent_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-page output contribution: || sum_{i in page} softmax(s_i) * v_i ||.

    Softmax is computed over ALL tokens (sink + paged + recent), so the weights
    reflect the real competition. Then contributions are summed per page and the
    L2 norm is taken. Sink/recent tokens participate in softmax but we only
    measure page contributions.

    For GQA, we compute per query head and then take max across the group
    (group_agg=max) to match the other methods.

    Returns: [bsz, kv_heads, num_pages] float32
    """
    bsz, num_heads, _, head_dim = query_states.shape
    _, kv_heads, num_pages, page_size, _ = paged_k.shape
    scaling = head_dim ** -0.5

    q = query_states.squeeze(2).reshape(bsz, kv_heads, num_kv_groups, head_dim).float()

    # Flatten paged keys/values: [bsz, kv_heads, num_pages * page_size, head_dim]
    paged_k_flat = paged_k.float().reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    paged_v_flat = paged_v.float().reshape(bsz, kv_heads, num_pages * page_size, head_dim)

    # Concatenate all keys for global softmax: sink + paged + recent
    all_k_parts = []
    all_v_parts = []
    if sink_k is not None and sink_k.shape[2] > 0:
        all_k_parts.append(sink_k.float())
        all_v_parts.append(sink_v.float())
    all_k_parts.append(paged_k_flat)
    all_v_parts.append(paged_v_flat)
    if recent_k is not None and recent_k.shape[2] > 0:
        all_k_parts.append(recent_k.float())
        all_v_parts.append(recent_v.float())

    # all_k: [bsz, kv_heads, total_len, head_dim]
    all_k = torch.cat(all_k_parts, dim=2)

    # Q·K for all tokens: [bsz, kv_heads, num_kv_groups, total_len]
    all_dots = torch.einsum('bhgd,bhtd->bhgt', q * scaling, all_k)

    # Global softmax over total_len
    all_attn = F.softmax(all_dots, dim=-1)

    # Extract only the paged portion of attention weights
    sink_len = sink_k.shape[2] if sink_k is not None else 0
    paged_start = sink_len
    paged_end = paged_start + num_pages * page_size
    # [bsz, kv_heads, num_kv_groups, num_pages * page_size]
    paged_attn = all_attn[:, :, :, paged_start:paged_end]
    # Reshape to [bsz, kv_heads, num_kv_groups, num_pages, page_size]
    paged_attn = paged_attn.reshape(bsz, kv_heads, num_kv_groups, num_pages, page_size)

    # Weighted values: [bsz, kv_heads, num_kv_groups, num_pages, page_size, head_dim]
    paged_v_5d = paged_v.float()  # [bsz, kv_heads, num_pages, page_size, head_dim]
    weighted_v = paged_attn.unsqueeze(-1) * paged_v_5d.unsqueeze(2)

    # Sum within each page: [bsz, kv_heads, num_kv_groups, num_pages, head_dim]
    page_output = weighted_v.sum(dim=4)

    # L2 norm: [bsz, kv_heads, num_kv_groups, num_pages]
    page_contrib = page_output.norm(dim=-1)

    # group_agg=max
    return page_contrib.max(dim=2).values  # [bsz, kv_heads, num_pages]


def compute_proxy_scores(
    query_states: torch.Tensor,   # [bsz, num_heads, 1, head_dim]
    paged_k: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    num_kv_groups: int,
    comp_size: int,
) -> dict[str, torch.Tensor]:
    """Simulate proxy scoring with DCT compression, group_agg=max.

    Returns dict with proxy_max and proxy_mean: [bsz, kv_heads, num_pages] float32.
    """
    from dct_page_attention import _build_dct_projection_matrix, _dct_page_cfg

    bsz, num_heads, _, head_dim = query_states.shape
    _, kv_heads, num_pages, page_size, _ = paged_k.shape
    scaling = head_dim ** -0.5

    layout = _dct_page_cfg.proxy_frequency_layout if _dct_page_cfg is not None else "low"
    M = _build_dct_projection_matrix(page_size, comp_size, paged_k.device, paged_k.dtype, layout)

    comp_k = torch.einsum('cs,bhnsd->bhncd', M.float(), paged_k.float())

    q = query_states.squeeze(2).reshape(bsz, kv_heads, num_kv_groups, head_dim).float()

    # dots: [bsz, kv_heads, num_kv_groups, num_pages, comp_size]
    dots = torch.einsum('bhgd,bhncd->bhgnc', q * scaling, comp_k)

    # group_agg=max
    dots_gagg = dots.max(dim=2).values  # [bsz, kv_heads, num_pages, comp_size]

    proxy_max = dots_gagg.max(dim=-1).values
    proxy_mean = dots_gagg.mean(dim=-1)

    return {"proxy_max": proxy_max, "proxy_mean": proxy_mean}


def compute_all_scores(
    per_token_scores: torch.Tensor,  # [bsz, kv_heads, num_pages, page_size]
    proxy_scores: dict[str, torch.Tensor],
    output_contrib: torch.Tensor | None,  # [bsz, kv_heads, num_pages] or None
    lambdas: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all page scoring variants. All use group_agg=max.

    Returns dict of [bsz, kv_heads, num_pages] tensors.
    """
    from dct_page_attention import dct

    scores = {
        "oracle_max": per_token_scores.max(dim=-1).values,
        "oracle_mean": per_token_scores.mean(dim=-1),
        "proxy_max": proxy_scores["proxy_max"],
        "proxy_mean": proxy_scores["proxy_mean"],
        "l2_energy": per_token_scores.pow(2).sum(dim=-1).sqrt(),
    }

    # Oracle DC+AC: apply DCT to per-token scores along page_size dim.
    # By linearity: DCT(⟨q, K⟩)[k] = ⟨q, K̃[k]⟩
    dct_scores = dct(per_token_scores.float())  # [bsz, kv_heads, num_pages, page_size]
    dc = dct_scores[..., 0]                          # signed
    ac = dct_scores[..., 1:].pow(2).sum(-1).sqrt()   # unsigned
    for lam in (lambdas or [0.5, 1.0, 2.0]):
        scores[f"dc_ac_{lam}"] = dc + lam * ac

    if output_contrib is not None:
        scores["output_contribution"] = output_contrib
    return scores


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyze_method_vs_gt(
    method_scores: torch.Tensor,   # [num_pages]
    gt_scores: torch.Tensor,       # [num_pages]
    gt_topk_idx: set[int],
    actual_top_k: int,
) -> dict[str, Any]:
    """Analyze one method against ground truth for one head."""
    method_topk_idx = set(method_scores.topk(actual_top_k).indices.tolist())
    overlap = len(method_topk_idx & gt_topk_idx)
    false_positives = method_topk_idx - gt_topk_idx
    false_negatives = gt_topk_idx - method_topk_idx

    method_ranks = method_scores.argsort(descending=True).argsort()

    neg_gt = sum(1 for idx in method_topk_idx if gt_scores[idx].item() < 0)
    fp_gt_scores = [gt_scores[idx].item() for idx in false_positives]
    fn_ranks = [method_ranks[idx].item() for idx in false_negatives]
    fn_gt_scores = [gt_scores[idx].item() for idx in false_negatives]

    return {
        "recall": overlap / actual_top_k,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "neg_gt_in_topk": neg_gt,
        "fp_gt_score_mean": (
            sum(fp_gt_scores) / len(fp_gt_scores) if fp_gt_scores else 0.0
        ),
        "fn_rank_mean": (
            sum(fn_ranks) / len(fn_ranks) if fn_ranks else 0.0
        ),
        "fn_gt_score_mean": (
            sum(fn_gt_scores) / len(fn_gt_scores) if fn_gt_scores else 0.0
        ),
    }


def analyze_one_sample(
    all_scores: dict[str, torch.Tensor],  # each [bsz, kv_heads, num_pages]
    ground_truth: str,
    methods: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Analyze all methods vs ground truth for a single sample across all heads."""
    gt = all_scores[ground_truth][0]  # [kv_heads, num_pages]

    kv_heads, num_pages = gt.shape
    actual_top_k = min(top_k, num_pages)

    head_results = []
    for h in range(kv_heads):
        gt_h = gt[h]
        gt_topk_idx = set(gt_h.topk(actual_top_k).indices.tolist())

        head_row = {}
        for method in methods:
            result = _analyze_method_vs_gt(
                all_scores[method][0, h], gt_h, gt_topk_idx, actual_top_k,
            )
            for k, v in result.items():
                head_row[f"{method}_{k}"] = v
        head_results.append(head_row)

    n = len(head_results)
    agg = {
        "num_heads": n,
        "actual_top_k": actual_top_k,
        "num_pages": num_pages,
        "ground_truth": ground_truth,
    }
    metric_keys = [k for k in head_results[0] if isinstance(head_results[0][k], (int, float))]
    for k in metric_keys:
        agg[f"{k}_avg"] = sum(r[k] for r in head_results) / n
    agg["head_results"] = head_results
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose page scoring methods")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--data_root", type=Path, default=Path("benchmark/data/ruler_data"))
    p.add_argument("--output_dir", type=Path, default=Path("results/results_ruler/l2_diagnostic"))
    p.add_argument("--tasks", default="cwe", help="Comma-separated task names or 'all'")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=2,
                   help="Need >=2: token 1 comes from prefill, token 2 triggers first decode step")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--ground_truth",
        default="oracle_max",
        choices=GROUND_TRUTHS,
        help="Ground truth for top-k comparison: "
             "oracle_max (max per-token QK score) or "
             "output_contribution (per-page attention output norm)",
    )

    # DCT page config
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--lambdas", default="0.5,1.0,2.0,2.5,3.0,4.0",
                   help="Comma-separated lambda values for DC+AC scoring")
    return p.parse_args()


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def apply_patch(args: argparse.Namespace) -> None:
    model_name = args.model_name_or_path.lower()
    common_kwargs = dict(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=args.compress_ratio,
        scoring_method="max",
        group_agg_method="max",
        unselected_mode="drop",
        compression_method="haar",
        use_triton=True,
    )
    if "qwen3" in model_name:
        from dct_page_attention import replace_qwen3_attn
        replace_qwen3_attn(**common_kwargs)
    elif "qwen" in model_name:
        from dct_page_attention import replace_qwen2_attn
        replace_qwen2_attn(**common_kwargs)
    elif "llama" in model_name:
        from dct_page_attention import replace_llama_attn
        replace_llama_attn(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model: {args.model_name_or_path}")


def load_model(args: argparse.Namespace):
    yarn_kwargs = {}
    if "qwen3" in args.model_name_or_path.lower():
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
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.cuda_device},
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
        **yarn_kwargs,
    ).eval()


def resolve_model_family(model_name_or_path: str) -> str:
    name = model_name_or_path.lower().split("/")[-1]
    if "qwen3" in name:
        return "qwen3"
    elif "qwen2" in name:
        return "qwen2"
    elif "llama-3" in name or "llama3" in name:
        return "llama3"
    elif "llama" in name:
        return "llama"
    else:
        return name


def cleanup_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    gt_name = args.ground_truth
    run_dir = args.output_dir / f"ps{args.page_size}_topk{args.top_k}_gt_{gt_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # Methods to compare (exclude the ground truth itself from the comparison list)
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    all_methods = ["oracle_max", "oracle_mean", "proxy_max", "proxy_mean", "l2_energy",
                   "output_contribution"]
    all_methods += [f"dc_ac_{lam}" for lam in lambdas]
    methods = [m for m in all_methods if m != gt_name]

    print(f"Applying DCT page attention patch (scoring=max, group_agg=max)...")
    print(f"Ground truth: {gt_name}")
    apply_patch(args)

    from dct_page_attention import set_dct_page_debug_hook

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name_or_path}")
    model = load_model(args)

    recorder = L2DiagnosticRecorder()

    try:
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"TASK: {task}")
            print(f"{'='*60}")

            model_family = resolve_model_family(args.model_name_or_path)
            data_path = args.data_root / model_family / str(args.context_len) / task / "validation.jsonl"
            if not data_path.exists():
                print(f"  WARNING: data not found at {data_path}, skipping")
                continue

            with data_path.open("r", encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]
            samples = samples[:args.num_samples]

            sample_results = []
            for idx, sample in enumerate(tqdm(samples, desc=f"  {task}"), start=1):
                recorder.reset()
                set_dct_page_debug_hook(recorder)

                device = next(model.parameters()).device
                encoded = tokenizer(sample["input"], return_tensors="pt")
                input_ids = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)

                with torch.no_grad():
                    model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                set_dct_page_debug_hook(None)

                if recorder.result is None:
                    print(f"  WARNING: no trace for sample {sample['index']}, skipping")
                    continue

                r = recorder.result
                comp_size = max(1, int(r["page_size"] * args.compress_ratio))
                per_token_scores = compute_per_token_scores(
                    r["query_states"], r["paged_k"], r["num_kv_groups"],
                )
                proxy_scores = compute_proxy_scores(
                    r["query_states"], r["paged_k"], r["num_kv_groups"], comp_size,
                )

                # Compute output_contribution if needed (as GT or as a method to compare)
                output_contrib = None
                if gt_name == "output_contribution" or "output_contribution" in methods:
                    output_contrib = compute_output_contribution(
                        r["query_states"], r["paged_k"], r["paged_v"],
                        r["num_kv_groups"],
                        sink_k=r["sink_k"], sink_v=r["sink_v"],
                        recent_k=r["recent_k"], recent_v=r["recent_v"],
                    )

                all_scores = compute_all_scores(
                    per_token_scores, proxy_scores, output_contrib, lambdas,
                )
                analysis = analyze_one_sample(
                    all_scores, gt_name, methods, r["actual_top_k"],
                )
                analysis["sample_index"] = sample["index"]
                sample_results.append(analysis)

                if idx % 5 == 0 or idx == len(samples):
                    recalls = " ".join(
                        f"{m}={analysis[f'{m}_recall_avg']:.3f}" for m in methods
                    )
                    print(f"  [{idx}/{len(samples)}] recall: {recalls}")

            if not sample_results:
                print(f"  No results for {task}")
                continue

            # Aggregate across samples
            n = len(sample_results)
            task_summary = {
                "task": task,
                "ground_truth": gt_name,
                "num_samples": n,
                "num_pages": sample_results[0]["num_pages"],
                "actual_top_k": sample_results[0]["actual_top_k"],
            }
            avg_keys = [k for k in sample_results[0] if k.endswith("_avg")]
            for k in avg_keys:
                task_summary[k] = sum(r[k] for r in sample_results) / n

            print(f"\n  === {task} Summary (vs {gt_name}, group_agg=max) ===")
            print(f"  Pages: {task_summary['num_pages']}, Top-k: {task_summary['actual_top_k']}")
            for m in methods:
                print(
                    f"  {m:22s}  "
                    f"recall={task_summary[f'{m}_recall_avg']:.3f}  "
                    f"FP={task_summary[f'{m}_false_positive_count_avg']:.1f}  "
                    f"FN={task_summary[f'{m}_false_negative_count_avg']:.1f}  "
                    f"neg_gt={task_summary[f'{m}_neg_gt_in_topk_avg']:.2f}  "
                    f"FN_rank={task_summary[f'{m}_fn_rank_mean_avg']:.1f}  "
                    f"FN_gt_score={task_summary[f'{m}_fn_gt_score_mean_avg']:.4f}"
                )

            output = {
                "summary": task_summary,
                "samples": [
                    {k: v for k, v in r.items() if k != "head_results"}
                    for r in sample_results
                ],
            }
            (run_dir / f"{task}.json").write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"  Results saved to: {run_dir / f'{task}.json'}")

    finally:
        cleanup_model(model)

    print(f"\nAll results in: {run_dir}")


if __name__ == "__main__":
    main()
