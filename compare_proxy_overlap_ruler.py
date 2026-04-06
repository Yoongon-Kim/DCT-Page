#!/usr/bin/env python3
"""
Compare Haar vs DCT proxy page selection against oracle on RULER benchmark.

For each sample, runs inference twice (Haar proxy, DCT proxy) with the debug
hook enabled.  Both runs capture proxy scores and oracle scores at the first
decode step of the last layer.  Computes three overlap comparisons:
  - Haar-vs-Oracle
  - DCT-vs-Oracle
  - Haar-vs-DCT

Usage:
    # Quick smoke test
    python compare_proxy_overlap_ruler.py --tasks niah_single_1 --num_samples 5

    # Full run (all 13 tasks, 25 samples)
    python compare_proxy_overlap_ruler.py
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Reuse from existing analysis scripts
# ---------------------------------------------------------------------------
from analyze_selection_oracle_overlap import (
    FirstStepRecorder,
    generate_with_first_step_traces,
    oracle_topk_from_scores,
    summarize_last_layer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Haar vs DCT proxy page selection overlap against oracle on RULER"
    )
    # Model
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    # RULER
    p.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--tokenizer_family", type=str, default="qwen3")
    p.add_argument("--num_samples", type=int, default=25)

    # DCT page config
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--scoring_method", type=str, default="max",
                    choices=["mean", "max", "sum"])
    p.add_argument("--group_agg_method", type=str, default="mean",
                    choices=["mean", "max", "topp"])
    p.add_argument("--unselected_mode", type=str, default="drop",
                    choices=["drop", "compressed"])

    # Output
    p.add_argument("--output_dir", type=Path,
                    default=Path("results_proxy_overlap_ruler"))
    p.add_argument("--tag", type=str, default="proxy_overlap")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Task config loading (merges data + eval constants, adapted from eval_ruler.py)
# ---------------------------------------------------------------------------
def load_task_configs() -> dict[str, dict]:
    ruler_dir = str(Path(__file__).resolve().parent / "eval_ruler")
    # Data constants (tokens_to_generate, template, etc.)
    sys.path.insert(0, os.path.join(ruler_dir, "data"))
    data_constants = importlib.import_module("synthetic.constants")
    data_tasks = data_constants.TASKS
    # Eval constants (metric_fn)
    if "synthetic.constants" in sys.modules:
        del sys.modules["synthetic.constants"]
    sys.path.insert(0, os.path.join(ruler_dir, "eval"))
    eval_constants = importlib.import_module("synthetic.constants")
    eval_tasks = eval_constants.TASKS
    # YAML customization
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


def postprocess_pred(predict_str: str) -> str:
    predict_str = predict_str.strip()
    np_pattern = re.compile(r"[\x00-\x1f]")
    return np_pattern.sub("\n", predict_str).strip()


def score_predictions(metric_fn, rows: list[dict[str, Any]]) -> float:
    preds = [postprocess_pred(row["text"]) for row in rows]
    refs = [row["gold"] for row in rows]
    return metric_fn(preds, refs)


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
        compression_method="haar",  # initial; toggled at runtime
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
# Haar-vs-DCT direct overlap
# ---------------------------------------------------------------------------
def compute_cross_overlap(
    haar_trace: dict[str, Any],
    dct_trace: dict[str, Any],
) -> dict[str, Any]:
    """Compare which pages Haar and DCT proxy methods select (last layer, step 0)."""
    haar_selected = haar_trace["selected_indices"][0]  # [num_kv_heads, top_k]
    dct_selected = dct_trace["selected_indices"][0]
    actual_top_k = int(haar_trace["actual_top_k"])

    head_rows = []
    for head_idx in range(len(haar_selected)):
        haar_set = set(int(x) for x in haar_selected[head_idx])
        dct_set = set(int(x) for x in dct_selected[head_idx])
        overlap = len(haar_set & dct_set)
        head_rows.append({
            "kv_head": head_idx,
            "overlap_count": overlap,
            "overlap_rate": overlap / actual_top_k if actual_top_k else 0.0,
            "exact_set_match": haar_set == dct_set,
        })

    overlap_rates = [r["overlap_rate"] for r in head_rows]
    return {
        "actual_top_k": actual_top_k,
        "num_heads": len(head_rows),
        "mean_overlap_rate": sum(overlap_rates) / len(overlap_rates),
        "median_overlap_rate": statistics.median(overlap_rates),
        "heads_with_exact_set_match": sum(
            1 for r in head_rows if r["exact_set_match"]
        ),
        "head_rows": head_rows,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate proxy-vs-oracle summaries (from summarize_last_layer) across samples."""
    if not summaries:
        return {}
    return {
        "mean_overlap_rate": sum(s["mean_overlap_rate"] for s in summaries) / len(summaries),
        "median_overlap_rate": statistics.median(s["median_overlap_rate"] for s in summaries),
        "mean_exact_set_match_heads": sum(s["heads_with_exact_set_match"] for s in summaries) / len(summaries),
        "mean_oracle_top1_proxy_rank": sum(s["mean_oracle_top1_proxy_rank"] for s in summaries) / len(summaries),
    }


def aggregate_cross_summaries(summaries: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate Haar-vs-DCT cross-overlap summaries across samples."""
    if not summaries:
        return {}
    return {
        "mean_overlap_rate": sum(s["mean_overlap_rate"] for s in summaries) / len(summaries),
        "median_overlap_rate": statistics.median(s["median_overlap_rate"] for s in summaries),
        "mean_exact_set_match_heads": sum(s["heads_with_exact_set_match"] for s in summaries) / len(summaries),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    start_time = time.time()
    torch.manual_seed(42)

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.tag}_{args.unselected_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = run_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    # Save config
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # Apply monkey-patch and load model
    print("Applying DCT page attention patch...")
    apply_monkey_patch(args)
    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from dct_page_attention import _dct_page_cfg

    task_configs = load_task_configs()
    per_task_results = {}

    try:
        for task in args.tasks:
            print(f"\n{'=' * 60}")
            print(f"TASK: {task}")
            print("=" * 60)

            # Load data
            data_path = (
                Path("ruler_data") / args.tokenizer_family
                / str(args.seq_len) / task / "validation.jsonl"
            )
            if not data_path.exists():
                print(f"  WARNING: data not found at {data_path}, skipping")
                continue
            with data_path.open("r", encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]
            if args.num_samples > 0:
                samples = samples[: args.num_samples]

            # Load metric and tokens_to_generate
            task_config = task_configs[task]
            metric_fn = task_config["metric_fn"]
            tokens_to_generate = task_config["tokens_to_generate"]

            # Per-sample collection
            haar_oracle_summaries = []
            dct_oracle_summaries = []
            cross_summaries = []
            haar_gen_rows = []
            dct_gen_rows = []

            sample_fp = (per_sample_dir / f"{task}.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )

            for sample_idx, sample in enumerate(
                tqdm(samples, desc=f"  {task}"), start=1
            ):
                # --- Haar run ---
                _dct_page_cfg.compression_method = "haar"
                haar_gen, haar_traces = generate_with_first_step_traces(
                    model, tokenizer, sample, tokens_to_generate,
                )
                haar_gen_rows.append(haar_gen)

                # --- DCT run ---
                _dct_page_cfg.compression_method = "dct"
                dct_gen, dct_traces = generate_with_first_step_traces(
                    model, tokenizer, sample, tokens_to_generate,
                )
                dct_gen_rows.append(dct_gen)

                if not haar_traces or not dct_traces:
                    print(f"  WARNING: no traces for sample {sample['index']}, skipping")
                    continue

                # Last-layer traces
                haar_last = max(haar_traces, key=lambda t: t["layer_idx"])
                dct_last = max(dct_traces, key=lambda t: t["layer_idx"])

                # Proxy-vs-oracle overlap
                haar_summary = summarize_last_layer(haar_last)
                dct_summary = summarize_last_layer(dct_last)
                haar_oracle_summaries.append(haar_summary)
                dct_oracle_summaries.append(dct_summary)

                # Haar-vs-DCT direct overlap
                cross = compute_cross_overlap(haar_last, dct_last)
                cross_summaries.append(cross)

                # Write per-sample detail (without bulky head_rows)
                sample_record = {
                    "sample_index": sample["index"],
                    "haar_vs_oracle": {
                        "mean_overlap_rate": haar_summary["mean_overlap_rate"],
                        "median_overlap_rate": haar_summary["median_overlap_rate"],
                        "exact_set_match_heads": haar_summary["heads_with_exact_set_match"],
                    },
                    "dct_vs_oracle": {
                        "mean_overlap_rate": dct_summary["mean_overlap_rate"],
                        "median_overlap_rate": dct_summary["median_overlap_rate"],
                        "exact_set_match_heads": dct_summary["heads_with_exact_set_match"],
                    },
                    "haar_vs_dct": {
                        "mean_overlap_rate": cross["mean_overlap_rate"],
                        "median_overlap_rate": cross["median_overlap_rate"],
                        "exact_set_match_heads": cross["heads_with_exact_set_match"],
                    },
                    "haar_pred": haar_gen["text"],
                    "dct_pred": dct_gen["text"],
                    "gold": sample["outputs"],
                }
                sample_fp.write(
                    json.dumps(sample_record, ensure_ascii=False) + "\n"
                )

                if sample_idx % 5 == 0 or sample_idx == len(samples):
                    print(
                        f"  [{sample_idx}/{len(samples)}] "
                        f"haar-oracle={haar_summary['mean_overlap_rate']:.3f} "
                        f"dct-oracle={dct_summary['mean_overlap_rate']:.3f} "
                        f"haar-dct={cross['mean_overlap_rate']:.3f}"
                    )

            sample_fp.close()

            # Task accuracy
            haar_score = score_predictions(metric_fn, haar_gen_rows) if haar_gen_rows else 0.0
            dct_score = score_predictions(metric_fn, dct_gen_rows) if dct_gen_rows else 0.0

            task_result = {
                "num_samples": len(haar_oracle_summaries),
                "haar_vs_oracle": aggregate_summaries(haar_oracle_summaries),
                "dct_vs_oracle": aggregate_summaries(dct_oracle_summaries),
                "haar_vs_dct": aggregate_cross_summaries(cross_summaries),
                "task_accuracy": {"haar": round(haar_score, 2), "dct": round(dct_score, 2)},
            }
            per_task_results[task] = task_result

            print(f"  Task accuracy: haar={haar_score:.2f}, dct={dct_score:.2f}")
            print(
                f"  Overlap (avg): "
                f"haar-oracle={task_result['haar_vs_oracle'].get('mean_overlap_rate', 0):.3f} "
                f"dct-oracle={task_result['dct_vs_oracle'].get('mean_overlap_rate', 0):.3f} "
                f"haar-dct={task_result['haar_vs_dct'].get('mean_overlap_rate', 0):.3f}"
            )

        # Overall summary
        all_haar = [r["haar_vs_oracle"]["mean_overlap_rate"]
                     for r in per_task_results.values() if r["haar_vs_oracle"]]
        all_dct = [r["dct_vs_oracle"]["mean_overlap_rate"]
                    for r in per_task_results.values() if r["dct_vs_oracle"]]
        all_cross = [r["haar_vs_dct"]["mean_overlap_rate"]
                      for r in per_task_results.values() if r["haar_vs_dct"]]
        all_haar_acc = [r["task_accuracy"]["haar"]
                         for r in per_task_results.values()]
        all_dct_acc = [r["task_accuracy"]["dct"]
                        for r in per_task_results.values()]

        overall = {
            "haar_vs_oracle_mean_overlap": sum(all_haar) / len(all_haar) if all_haar else 0,
            "dct_vs_oracle_mean_overlap": sum(all_dct) / len(all_dct) if all_dct else 0,
            "haar_vs_dct_mean_overlap": sum(all_cross) / len(all_cross) if all_cross else 0,
            "haar_accuracy_avg": sum(all_haar_acc) / len(all_haar_acc) if all_haar_acc else 0,
            "dct_accuracy_avg": sum(all_dct_acc) / len(all_dct_acc) if all_dct_acc else 0,
        }

        summary = {
            "overall": overall,
            "config": {
                "base_model": args.base_model,
                "seq_len": args.seq_len,
                "page_size": args.page_size,
                "top_k": args.top_k,
                "compress_ratio": args.compress_ratio,
                "sink_size": args.sink_size,
                "recent_size": args.recent_size,
                "scoring_method": args.scoring_method,
                "group_agg_method": args.group_agg_method,
                "num_samples": args.num_samples,
            },
            "per_task": per_task_results,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Print final summary
        elapsed = (time.time() - start_time) / 60
        print(f"\n{'=' * 60}")
        print("OVERALL RESULTS")
        print("=" * 60)
        print(f"  Haar-vs-Oracle mean overlap: {overall['haar_vs_oracle_mean_overlap']:.3f}")
        print(f"  DCT-vs-Oracle  mean overlap: {overall['dct_vs_oracle_mean_overlap']:.3f}")
        print(f"  Haar-vs-DCT    mean overlap: {overall['haar_vs_dct_mean_overlap']:.3f}")
        print(f"  Haar accuracy avg: {overall['haar_accuracy_avg']:.2f}")
        print(f"  DCT  accuracy avg: {overall['dct_accuracy_avg']:.2f}")
        print(f"\n  Results saved to: {run_dir}")
        print(f"  Total time: {elapsed:.1f} minutes")

    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
