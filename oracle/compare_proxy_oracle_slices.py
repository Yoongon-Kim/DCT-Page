#!/usr/bin/env python3
"""
Proxy-vs-oracle top-K selection overlap, sliced by proxy-score rank.

For each RULER sample, runs generation with the normal proxy_max page-scoring
code path (haar or dct compression, configurable comp_kv_quant), captures the
debug-hook payloads emitted at every layer for the first --num_decode_steps
decode steps, and reports per-layer per-head:

  Overall:
    overall_recall   = |P ∩ O| / |O|
    overall_jaccard  = |P ∩ O| / |P ∪ O|

  Top slice P_top (top ⌈K/2⌉ of proxy-selected by proxy score):
    top_hit_full     = |P_top ∩ O|     / |P_top|
    top_jac_full     = |P_top ∩ O|     / |P_top ∪ O|
    top_recall_half  = |P_top ∩ O_top| / |O_top|      (O_top = top ⌈K/2⌉ of O)
    top_jac_half     = |P_top ∩ O_top| / |P_top ∪ O_top|

  Bottom slice P_bot (bottom ⌊K/2⌋ of proxy-selected by proxy score):
    bot_hit_full     = |P_bot ∩ O|     / |P_bot|
    bot_jac_full     = |P_bot ∩ O|     / |P_bot ∪ O|
    bot_recall_half  = |P_bot ∩ O_bot| / |O_bot|      (O_bot = bottom ⌊K/2⌋ of O)
    bot_jac_half     = |P_bot ∩ O_bot| / |P_bot ∪ O_bot|

Hypothesis: when proxy_max drives selection well on RULER, the top slice's
hit rate against O should be much higher than the bottom slice's.

Usage:
    python oracle/compare_proxy_oracle_slices.py \\
        --base_model Qwen/Qwen3-8B \\
        --tasks niah_single_1 --num_samples 5 --seq_len 32768 \\
        --page_size 16 --top_k 128 --compress_ratio 0.125 \\
        --comp_kv_quant none --num_decode_steps 4 \\
        --output_dir results_proxy_slice_overlap --run_name smoke
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ruler import infer_model_family

ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

METRIC_KEYS = [
    "overall_recall", "overall_jaccard",
    "top_hit_full", "top_jac_full", "top_recall_half", "top_jac_half",
    "bot_hit_full", "bot_jac_full", "bot_recall_half", "bot_jac_half",
]


# ---------------------------------------------------------------------------
# Debug-hook recorder
# ---------------------------------------------------------------------------
class AllStepRecorder:
    """Captures debug-hook payloads for all layers, first N decode steps."""

    def __init__(self, num_decode_steps: int):
        self.num_decode_steps = num_decode_steps
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step >= self.num_decode_steps:
            return

        self.records.append({
            "layer_idx": layer_idx,
            "decode_step": decode_step,
            "num_pages": int(payload["num_pages"]),
            "actual_top_k": int(payload["actual_top_k"]),
            # page_scores / oracle_page_scores arrive float CPU already
            "page_scores": payload["page_scores"],        # [bsz, H, num_pages]
            "oracle_page_scores": payload["oracle_page_scores"],
            "selected_indices": payload["selected_indices"],  # [bsz, H, K]
        })


def generate_with_traces(model, tokenizer, sample: dict[str, Any], num_decode_steps: int):
    """Generate exactly `num_decode_steps` new tokens with the hook attached."""
    from dct_page_attention import set_dct_page_debug_hook

    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder = AllStepRecorder(num_decode_steps)
    set_dct_page_debug_hook(recorder)
    try:
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_decode_steps,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        set_dct_page_debug_hook(None)

    return recorder.records, int(input_ids.shape[1])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _indices_to_mask(indices: torch.Tensor, num_pages: int) -> torch.Tensor:
    """indices: [..., M] → bool mask [..., num_pages]."""
    shape = list(indices.shape[:-1]) + [num_pages]
    mask = torch.zeros(shape, dtype=torch.bool, device=indices.device)
    mask.scatter_(-1, indices.long(), True)
    return mask


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(num, dtype=torch.float32)
    valid = den > 0
    out[valid] = num[valid].float() / den[valid].float()
    return out


def compute_slice_metrics(
    page_scores: torch.Tensor,        # [H, num_pages]
    oracle_scores: torch.Tensor,      # [H, num_pages]
    selected_indices: torch.Tensor,   # [H, K]  (proxy's top-K)
    actual_top_k: int,
) -> dict[str, torch.Tensor]:
    """Compute the 10 metrics per head. Returns dict of [H] float32 tensors."""
    H, P = page_scores.shape
    K = actual_top_k
    K_top = (K + 1) // 2   # ⌈K/2⌉
    K_bot = K // 2         # ⌊K/2⌋

    page_scores = page_scores.float()
    oracle_scores = oracle_scores.float()
    selected_indices = selected_indices.long()

    # --- Oracle top-K and its halves (by oracle score) ---
    O_indices = torch.topk(oracle_scores, K, dim=-1).indices            # [H, K]
    o_at_O = torch.gather(oracle_scores, -1, O_indices)                 # [H, K]
    O_order = torch.argsort(o_at_O, dim=-1, descending=True)
    O_sorted = torch.gather(O_indices, -1, O_order)                     # [H, K]
    O_top = O_sorted[:, :K_top]                                         # [H, K_top]
    O_bot = O_sorted[:, K_top:K_top + K_bot]                            # [H, K_bot]

    # --- Proxy top-K and its halves (by proxy score) ---
    P_indices = selected_indices                                        # [H, K]
    p_at_P = torch.gather(page_scores, -1, P_indices)                   # [H, K]
    P_order = torch.argsort(p_at_P, dim=-1, descending=True)
    P_sorted = torch.gather(P_indices, -1, P_order)                     # [H, K]
    P_top = P_sorted[:, :K_top]                                         # [H, K_top]
    P_bot = P_sorted[:, K_top:K_top + K_bot]                            # [H, K_bot]

    # --- Masks over pages ---
    mO = _indices_to_mask(O_indices, P)
    mO_top = _indices_to_mask(O_top, P)
    mO_bot = _indices_to_mask(O_bot, P) if K_bot > 0 else torch.zeros_like(mO)
    mP = _indices_to_mask(P_indices, P)
    mP_top = _indices_to_mask(P_top, P)
    mP_bot = _indices_to_mask(P_bot, P) if K_bot > 0 else torch.zeros_like(mP)

    def inter(a, b): return (a & b).sum(-1)
    def union(a, b): return (a | b).sum(-1)
    def sz(a): return a.sum(-1)

    # Overall
    pi = inter(mP, mO)
    overall_recall = _safe_div(pi, sz(mO))
    overall_jaccard = _safe_div(pi, union(mP, mO))

    # Top slice
    t_full_i = inter(mP_top, mO)
    t_half_i = inter(mP_top, mO_top)
    top_hit_full = _safe_div(t_full_i, sz(mP_top))
    top_jac_full = _safe_div(t_full_i, union(mP_top, mO))
    top_recall_half = _safe_div(t_half_i, sz(mO_top))
    top_jac_half = _safe_div(t_half_i, union(mP_top, mO_top))

    # Bottom slice
    b_full_i = inter(mP_bot, mO)
    b_half_i = inter(mP_bot, mO_bot)
    bot_hit_full = _safe_div(b_full_i, sz(mP_bot))
    bot_jac_full = _safe_div(b_full_i, union(mP_bot, mO))
    bot_recall_half = _safe_div(b_half_i, sz(mO_bot))
    bot_jac_half = _safe_div(b_half_i, union(mP_bot, mO_bot))

    return {
        "overall_recall": overall_recall,
        "overall_jaccard": overall_jaccard,
        "top_hit_full": top_hit_full,
        "top_jac_full": top_jac_full,
        "top_recall_half": top_recall_half,
        "top_jac_half": top_jac_half,
        "bot_hit_full": bot_hit_full,
        "bot_jac_full": bot_jac_full,
        "bot_recall_half": bot_recall_half,
        "bot_jac_half": bot_jac_half,
    }


# ---------------------------------------------------------------------------
# RULER task-config loading (data path only; we skip task accuracy here)
# ---------------------------------------------------------------------------
def load_task_configs() -> dict[str, dict]:
    ruler_dir = str(_REPO_ROOT / "benchmark" / "eval_ruler")
    sys.path.insert(0, os.path.join(ruler_dir, "data"))
    data_constants = importlib.import_module("synthetic.constants")
    data_tasks = data_constants.TASKS
    if "synthetic.constants" in sys.modules:
        del sys.modules["synthetic.constants"]
    sys.path.insert(0, os.path.join(ruler_dir, "eval"))
    eval_constants = importlib.import_module("synthetic.constants")
    eval_tasks = eval_constants.TASKS
    with open(os.path.join(ruler_dir, "synthetic.yaml"), "r") as f:
        yaml_tasks = yaml.safe_load(f)
    configs = {}
    for task_name, yaml_cfg in yaml_tasks.items():
        base_task = yaml_cfg["task"]
        cfg = dict(yaml_cfg)
        cfg.update(data_tasks[base_task])
        cfg.update(eval_tasks[base_task])
        configs[task_name] = cfg
    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Proxy-vs-oracle top-K slice recall/Jaccard under comp_kv_quant"
    )
    # Model
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    # RULER
    p.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--data_root", type=Path,
                   default=Path("benchmark/data/ruler_data"))

    # DCT page config
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--scoring_method", type=str, default="max",
                   choices=["mean", "max", "sum"])
    p.add_argument("--group_agg_method", type=str, default="max",
                   choices=["mean", "max", "topp"])
    p.add_argument("--unselected_mode", type=str, default="drop",
                   choices=["drop", "compressed"])
    p.add_argument("--compression_method", type=str, default="dct",
                   choices=["haar", "dct"])
    p.add_argument("--compressed_token_rope", type=str, default="mixed",
                   choices=["mixed", "block_center"])
    p.add_argument("--proxy_frequency_layout", type=str, default="low")

    # comp_kv_quant
    p.add_argument("--comp_kv_quant", type=str, default="fp8_e4m3",
                   choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    p.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                   choices=["per_page", "per_comp_token"])

    # Analysis
    p.add_argument("--num_decode_steps", type=int, default=20,
                   help="Number of decode steps per sample to record.")

    # Output
    p.add_argument("--output_dir", type=Path,
                   default=Path("results_proxy_slice_overlap"))
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def apply_monkey_patch(args: argparse.Namespace) -> None:
    model_name = args.base_model.lower()
    common_kwargs = dict(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        compression_method=args.compression_method,
        compressed_token_rope=args.compressed_token_rope,
        proxy_frequency_layout=args.proxy_frequency_layout,
        comp_kv_quant=args.comp_kv_quant,
        comp_kv_quant_granularity=args.comp_kv_quant_granularity,
        use_triton=True,
    )
    if "llama" in model_name:
        from dct_page_attention import replace_llama_attn
        replace_llama_attn(**common_kwargs)
    elif "qwen3" in model_name:
        from dct_page_attention import replace_qwen3_attn
        replace_qwen3_attn(**common_kwargs)
    elif "qwen" in model_name:
        from dct_page_attention import replace_qwen2_attn
        replace_qwen2_attn(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model: {args.base_model}")


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
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Aggregation helpers
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

    # Output dir
    run_name = args.run_name or (
        f"{args.compression_method}_ps{args.page_size}_topk{args.top_k}"
        f"_cr{args.compress_ratio}_quant_{args.comp_kv_quant}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = run_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("Applying DCT page attention patch...")
    apply_monkey_patch(args)
    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Resolve RULER data path family
    _, tokenizer_family = infer_model_family(args.base_model)

    # load_task_configs validates task existence
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

            # Per-task accumulators: flat list of per-head metric dicts (overall)
            # and per-layer buckets.
            task_overall_records: list[dict[str, float]] = []
            task_per_layer: dict[int, list[dict[str, float]]] = {}

            sample_fp = (per_sample_dir / f"{task}.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )

            for sample_idx, sample in enumerate(
                tqdm(samples, desc=f"  {task}"), start=1
            ):
                records, input_len = generate_with_traces(
                    model, tokenizer, sample, args.num_decode_steps,
                )
                if not records:
                    print(f"  WARNING: no traces for sample {sample['index']} "
                          f"(input_len={input_len}); skipping")
                    continue

                per_head_rows: list[dict[str, Any]] = []
                per_layer_buckets: dict[int, list[dict[str, float]]] = {}

                for rec in records:
                    page_scores = rec["page_scores"][0]        # [H, P]
                    oracle_scores = rec["oracle_page_scores"][0]
                    selected = rec["selected_indices"][0]      # [H, K]
                    actual_top_k = rec["actual_top_k"]
                    layer_idx = rec["layer_idx"]
                    decode_step = rec["decode_step"]

                    metrics = compute_slice_metrics(
                        page_scores, oracle_scores, selected, actual_top_k,
                    )
                    # metrics[k] shape: [H]
                    H = metrics["overall_recall"].shape[0]
                    metric_arrs = {k: metrics[k].tolist() for k in METRIC_KEYS}

                    for h in range(H):
                        row = {
                            "layer_idx": layer_idx,
                            "decode_step": decode_step,
                            "kv_head": h,
                            "actual_top_k": actual_top_k,
                            **{k: metric_arrs[k][h] for k in METRIC_KEYS},
                        }
                        per_head_rows.append(row)
                        per_layer_buckets.setdefault(layer_idx, []).append(
                            {k: metric_arrs[k][h] for k in METRIC_KEYS}
                        )
                        task_overall_records.append(
                            {k: metric_arrs[k][h] for k in METRIC_KEYS}
                        )
                        task_per_layer.setdefault(layer_idx, []).append(
                            {k: metric_arrs[k][h] for k in METRIC_KEYS}
                        )

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
                        f"recall={o['overall_recall']:.3f} "
                        f"top_hit={o['top_hit_full']:.3f} "
                        f"bot_hit={o['bot_hit_full']:.3f}"
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
                f"  TASK SUMMARY  recall={o['overall_recall']:.3f} "
                f"jaccard={o['overall_jaccard']:.3f} "
                f"top_hit={o['top_hit_full']:.3f} "
                f"bot_hit={o['bot_hit_full']:.3f}"
            )

        # Overall across tasks (mean of per-task overall means)
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
                "sink_size": args.sink_size,
                "recent_size": args.recent_size,
                "compress_ratio": args.compress_ratio,
                "compression_method": args.compression_method,
                "compressed_token_rope": args.compressed_token_rope,
                "proxy_frequency_layout": args.proxy_frequency_layout,
                "scoring_method": args.scoring_method,
                "group_agg_method": args.group_agg_method,
                "unselected_mode": args.unselected_mode,
                "comp_kv_quant": args.comp_kv_quant,
                "comp_kv_quant_granularity": args.comp_kv_quant_granularity,
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
            print(f"  {k:20s} = {overall[k]:.3f}")
        print(f"\n  Results: {run_dir}")
        print(f"  Total time: {elapsed:.1f} min")

    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
