#!/usr/bin/env python3
"""
Compare proxy page selection against oracle on RULER benchmark.

For each sample, runs inference for 4 variants of (compression_method, compressed_token_rope):
  - (haar, mixed)
  - (haar, block_center)
  - (dct,  mixed)
  - (dct,  block_center)

The debug hook captures proxy scores and oracle scores at the first decode step of the
last layer for each variant. We then compute proxy-vs-oracle overlap for each variant,
plus pairwise cross-overlap between variants.

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

# Bootstrap sys.path so this script runs from any cwd. The project root holds
# `dct_page_attention.py`, which `apply_monkey_patch` imports lazily.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ruler import infer_model_family

# ---------------------------------------------------------------------------
# Helpers (originally lived in the deleted analyze_selection_oracle_overlap.py)
# ---------------------------------------------------------------------------
class FirstStepRecorder:
    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step != 0:
            return

        self.records.append(
            {
                "layer_idx": layer_idx,
                "decode_step": decode_step,
                "kv_len": int(payload["kv_len"]),
                "num_pages": int(payload["num_pages"]),
                "actual_top_k": int(payload["actual_top_k"]),
                "page_size": int(payload["page_size"]),
                "sink_size": int(payload["sink_size"]),
                "recent_size": int(payload["recent_size"]),
                "page_scores": payload["page_scores"].tolist(),
                "oracle_page_scores": payload["oracle_page_scores"].tolist(),
                "selected_indices": payload["selected_indices"].tolist(),
            }
        )


def generate_with_first_step_traces(model, tokenizer, sample: dict[str, Any], max_new_tokens: int):
    from dct_page_attention import set_dct_page_debug_hook

    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder = FirstStepRecorder()
    set_dct_page_debug_hook(recorder)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        set_dct_page_debug_hook(None)

    generated_ids = output_ids[0, input_ids.shape[1] :].cpu()
    return {
        "index": sample["index"],
        "input_len": int(input_ids.shape[1]),
        "gen_ids": generated_ids.tolist(),
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "gold": sample["outputs"],
    }, recorder.records


def oracle_topk_from_scores(head_scores: list[float], top_k: int) -> list[int]:
    ranked = sorted(range(len(head_scores)), key=lambda idx: head_scores[idx], reverse=True)
    return sorted(ranked[:top_k])


def summarize_last_layer(record: dict[str, Any]) -> dict[str, Any]:
    page_scores = record["page_scores"][0]
    oracle_scores = record["oracle_page_scores"][0]
    selected_indices = record["selected_indices"][0]
    actual_top_k = int(record["actual_top_k"])

    head_rows = []
    for head_idx, (proxy_head_scores, oracle_head_scores) in enumerate(zip(page_scores, oracle_scores)):
        proxy_selected = sorted(int(x) for x in selected_indices[head_idx])
        oracle_selected = oracle_topk_from_scores(oracle_head_scores, actual_top_k)
        overlap = len(set(proxy_selected).intersection(oracle_selected))
        head_rows.append(
            {
                "kv_head": head_idx,
                "proxy_selected": proxy_selected,
                "oracle_selected": oracle_selected,
                "overlap_count": overlap,
                "overlap_rate": overlap / actual_top_k if actual_top_k else 0.0,
                "exact_set_match": proxy_selected == oracle_selected,
                "oracle_only_pages": sorted(set(oracle_selected) - set(proxy_selected)),
                "proxy_only_pages": sorted(set(proxy_selected) - set(oracle_selected)),
                "oracle_top1_page": int(max(range(len(oracle_head_scores)), key=lambda idx: oracle_head_scores[idx])),
                "oracle_top1_proxy_rank": sorted(
                    range(len(proxy_head_scores)),
                    key=lambda idx: proxy_head_scores[idx],
                    reverse=True,
                ).index(
                    max(range(len(oracle_head_scores)), key=lambda idx: oracle_head_scores[idx])
                )
                + 1,
            }
        )

    overlap_rates = [row["overlap_rate"] for row in head_rows]
    overlap_counts = [row["overlap_count"] for row in head_rows]
    oracle_top1_ranks = [row["oracle_top1_proxy_rank"] for row in head_rows]
    return {
        "layer_idx": record["layer_idx"],
        "actual_top_k": actual_top_k,
        "num_heads": len(head_rows),
        "mean_overlap_rate": sum(overlap_rates) / len(overlap_rates),
        "median_overlap_rate": statistics.median(overlap_rates),
        "mean_overlap_count": sum(overlap_counts) / len(overlap_counts),
        "heads_with_exact_set_match": sum(1 for row in head_rows if row["exact_set_match"]),
        "heads_with_full_overlap": sum(1 for row in head_rows if row["overlap_count"] == actual_top_k),
        "mean_oracle_top1_proxy_rank": sum(oracle_top1_ranks) / len(oracle_top1_ranks),
        "median_oracle_top1_proxy_rank": statistics.median(oracle_top1_ranks),
        "head_rows": head_rows,
    }

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
    p.add_argument("--num_samples", type=int, default=25)

    # DCT page config
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128)
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
    ruler_dir = str(_REPO_ROOT / "benchmark" / "eval_ruler")
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
    run_dir = args.output_dir / f"ps{args.page_size}_topk_{args.top_k}cr_{args.compress_ratio}"
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
            _, tokenizer_family = infer_model_family(args.base_model)
            data_path = (
                Path("ruler_data") / tokenizer_family
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

            # 4 variants to compare per sample
            VARIANTS = [
                ("haar", "mixed"),
                ("haar", "block_center"),
                ("dct",  "mixed"),
                ("dct",  "block_center"),
            ]

            def variant_name(comp_method: str, token_rope: str) -> str:
                return f"{comp_method}_{token_rope}"

            # Per-sample collection (one entry per variant)
            oracle_summaries: dict[str, list] = {variant_name(c, t): [] for c, t in VARIANTS}
            gen_rows: dict[str, list] = {variant_name(c, t): [] for c, t in VARIANTS}

            sample_fp = (per_sample_dir / f"{task}.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )

            for sample_idx, sample in enumerate(
                tqdm(samples, desc=f"  {task}"), start=1
            ):
                variant_traces: dict[str, dict] = {}
                missing = False

                for comp_method, token_rope in VARIANTS:
                    name = variant_name(comp_method, token_rope)
                    _dct_page_cfg.compression_method = comp_method
                    _dct_page_cfg.compressed_token_rope = token_rope
                    gen_row, traces = generate_with_first_step_traces(
                        model, tokenizer, sample, tokens_to_generate,
                    )
                    gen_rows[name].append(gen_row)
                    if not traces:
                        missing = True
                        break
                    variant_traces[name] = max(traces, key=lambda t: t["layer_idx"])

                if missing:
                    print(f"  WARNING: no traces for sample {sample['index']}, skipping")
                    continue

                # Proxy-vs-oracle overlap per variant
                variant_oracle_summary: dict[str, dict] = {}
                for name, last_trace in variant_traces.items():
                    s = summarize_last_layer(last_trace)
                    oracle_summaries[name].append(s)
                    variant_oracle_summary[name] = s

                # Pairwise cross-overlap between variants
                variant_names = list(variant_traces.keys())
                cross_overlaps: dict[str, dict] = {}
                for i in range(len(variant_names)):
                    for j in range(i + 1, len(variant_names)):
                        a = variant_names[i]
                        b = variant_names[j]
                        cross_overlaps[f"{a}__vs__{b}"] = compute_cross_overlap(
                            variant_traces[a], variant_traces[b]
                        )

                # Write per-sample detail (without bulky head_rows)
                sample_record = {
                    "sample_index": sample["index"],
                    "vs_oracle": {
                        name: {
                            "mean_overlap_rate": s["mean_overlap_rate"],
                            "median_overlap_rate": s["median_overlap_rate"],
                            "exact_set_match_heads": s["heads_with_exact_set_match"],
                        }
                        for name, s in variant_oracle_summary.items()
                    },
                    "cross_overlap": {
                        key: {
                            "mean_overlap_rate": c["mean_overlap_rate"],
                            "median_overlap_rate": c["median_overlap_rate"],
                            "exact_set_match_heads": c["heads_with_exact_set_match"],
                        }
                        for key, c in cross_overlaps.items()
                    },
                    "preds": {
                        name: gen_rows[name][-1]["text"]
                        for name in variant_names
                    },
                    "gold": sample["outputs"],
                }
                sample_fp.write(
                    json.dumps(sample_record, ensure_ascii=False) + "\n"
                )

                if sample_idx % 5 == 0 or sample_idx == len(samples):
                    summary_str = " ".join(
                        f"{name}={variant_oracle_summary[name]['mean_overlap_rate']:.3f}"
                        for name in variant_names
                    )
                    print(f"  [{sample_idx}/{len(samples)}] {summary_str}")

            sample_fp.close()

            # Task accuracy per variant
            task_accuracy = {
                name: round(score_predictions(metric_fn, gen_rows[name]), 2)
                if gen_rows[name] else 0.0
                for name in oracle_summaries
            }

            task_result = {
                "num_samples": len(next(iter(oracle_summaries.values()))),
                "vs_oracle": {
                    name: aggregate_summaries(summaries)
                    for name, summaries in oracle_summaries.items()
                },
                "task_accuracy": task_accuracy,
            }
            per_task_results[task] = task_result

            print(f"  Task accuracy: " + ", ".join(f"{n}={s:.2f}" for n, s in task_accuracy.items()))
            print(
                "  Overlap (avg): "
                + " ".join(
                    f"{name}={task_result['vs_oracle'][name].get('mean_overlap_rate', 0):.3f}"
                    for name in task_result['vs_oracle']
                )
            )

        # Overall summary across tasks (per variant)
        variant_names = ["haar_mixed", "haar_block_center", "dct_mixed", "dct_block_center"]
        overall = {}
        for name in variant_names:
            overlap_vals = [r["vs_oracle"][name]["mean_overlap_rate"]
                             for r in per_task_results.values()
                             if name in r["vs_oracle"] and r["vs_oracle"][name]]
            acc_vals = [r["task_accuracy"][name]
                         for r in per_task_results.values()
                         if name in r["task_accuracy"]]
            overall[f"{name}_vs_oracle_mean_overlap"] = (
                sum(overlap_vals) / len(overlap_vals) if overlap_vals else 0
            )
            overall[f"{name}_accuracy_avg"] = (
                sum(acc_vals) / len(acc_vals) if acc_vals else 0
            )

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
        for name in variant_names:
            print(f"  {name:24s}  vs-oracle={overall[f'{name}_vs_oracle_mean_overlap']:.3f}  "
                  f"accuracy={overall[f'{name}_accuracy_avg']:.2f}")
        print(f"\n  Results saved to: {run_dir}")
        print(f"  Total time: {elapsed:.1f} minutes")

    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
