#!/usr/bin/env python3
"""
Analyze how closely proxy-based page selection matches oracle top-k selection.

This is task-agnostic: unlike the NIAH-specific analyzers, it does not require a
known gold page. Instead it compares the proxy-selected page set against the
oracle-selected page set on the first decode step, typically at the last layer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from analyze_niah_page_ranks import (
    RunLogger,
    apply_dct_patch,
    cleanup_model,
    load_model,
    load_samples,
    make_run_dir,
    write_jsonl_line,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze proxy-vs-oracle selection overlap")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("results/debug_compare"))
    parser.add_argument("--tag", type=str, default="selection_oracle_overlap")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--num_samples", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--dct_page_size", type=int, default=32)
    parser.add_argument("--dct_top_k", type=int, default=64)
    parser.add_argument("--dct_sink_size", type=int, default=4)
    parser.add_argument("--dct_recent_size", type=int, default=128)
    parser.add_argument("--dct_compress_ratio", type=float, default=0.03125)
    parser.add_argument(
        "--dct_proxy_frequency_layout",
        type=str,
        default="low",
        choices=["low", "low_high", "low_mid_high", "spread"],
    )
    parser.add_argument(
        "--dct_scoring_method",
        type=str,
        default="max",
        choices=["mean", "max", "sum"],
    )
    parser.add_argument(
        "--dct_group_agg_method",
        type=str,
        default="mean",
        choices=["mean", "max", "topp"],
    )
    parser.add_argument(
        "--dct_unselected_mode",
        type=str,
        default="drop",
        choices=["drop", "compressed"],
    )
    parser.add_argument("--dct_compression_method", type=str, default="haar", choices=["haar", "dct"])
    parser.add_argument("--dct_compressed_token_rope", type=str, default="mixed", choices=["mixed", "block_center"])
    parser.add_argument("--dct_score_use_direct_spectral_proxy", action="store_true")
    parser.add_argument(
        "--dct_score_use_haar_proxy",
        dest="dct_score_use_haar_proxy",
        action="store_true",
        help="Use Haar lowpass score proxies (default).",
    )
    parser.add_argument(
        "--dct_score_use_low_proxy",
        dest="dct_score_use_haar_proxy",
        action="store_false",
        help="Use the original low-frequency DCT IDCT score proxy instead of Haar.",
    )
    parser.add_argument("--dct_score_use_haar_mixed_proxy", action="store_true")
    parser.add_argument("--dct_score_use_hadamard_proxy", action="store_true")
    parser.add_argument("--dct_continuous_rope", action="store_true",
                        help="Temporarily disabled — raises error if used")
    parser.add_argument("--dct_no_triton", action="store_true")
    parser.set_defaults(dct_score_use_haar_proxy=True)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root, args.tag)
    logger = RunLogger(run_dir / "run.log")

    raw_trace_path = run_dir / "raw_traces.jsonl"
    sample_path = run_dir / "sample_analysis.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    config_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    raw_trace_fp = raw_trace_path.open("a", encoding="utf-8", buffering=1)
    sample_fp = sample_path.open("a", encoding="utf-8", buffering=1)

    logger.log(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.log(f"Set pad_token_id to eos_token_id={tokenizer.pad_token_id}")

    samples = load_samples(args.data_path, args.num_samples)
    logger.log(f"Loaded {len(samples)} samples from {args.data_path}")

    logger.log("Applying DCT patch")
    apply_dct_patch(args)
    logger.log("Loading DCT model")
    model = load_model(args)

    sample_summaries = []
    last_layer_summaries = []
    try:
        for sample_idx, sample in enumerate(samples, start=1):
            generation, traces = generate_with_first_step_traces(model, tokenizer, sample, args.max_new_tokens)
            for trace in traces:
                write_jsonl_line(raw_trace_fp, {"sample_index": sample["index"], **trace})

            if not traces:
                raise RuntimeError(f"No first decode-step traces were recorded for sample {sample['index']}.")

            last_trace = max(traces, key=lambda item: item["layer_idx"])
            last_summary = summarize_last_layer(last_trace)
            sample_summary = {
                "sample_index": sample["index"],
                "input_len": generation["input_len"],
                "generated_text": generation["text"],
                "gold": sample["outputs"],
                "first_step_last_layer": last_summary,
            }
            write_jsonl_line(sample_fp, sample_summary)
            sample_summaries.append(sample_summary)
            last_layer_summaries.append(last_summary)
            logger.log(
                f"sample {sample_idx}/{len(samples)} index={sample['index']} "
                f"mean_overlap={last_summary['mean_overlap_rate']:.3f} "
                f"exact_heads={last_summary['heads_with_exact_set_match']}/{last_summary['num_heads']} "
                f"oracle_top1_proxy_rank={last_summary['median_oracle_top1_proxy_rank']}"
            )

        num_heads = last_layer_summaries[0]["num_heads"] if last_layer_summaries else 0
        summary = {
            "num_samples": len(sample_summaries),
            "dct_top_k": args.dct_top_k,
            "dct_compress_ratio": args.dct_compress_ratio,
            "dct_page_size": args.dct_page_size,
            "dct_score_use_haar_proxy": args.dct_score_use_haar_proxy,
            "first_step_last_layer": {
                "mean_overlap_rate_avg": sum(item["mean_overlap_rate"] for item in last_layer_summaries)
                / len(last_layer_summaries),
                "median_overlap_rate_median": statistics.median(
                    item["median_overlap_rate"] for item in last_layer_summaries
                ),
                "mean_overlap_count_avg": sum(item["mean_overlap_count"] for item in last_layer_summaries)
                / len(last_layer_summaries),
                "mean_heads_with_exact_set_match": sum(
                    item["heads_with_exact_set_match"] for item in last_layer_summaries
                )
                / len(last_layer_summaries),
                "samples_with_all_heads_exact_set_match": sum(
                    1 for item in last_layer_summaries if item["heads_with_exact_set_match"] == num_heads
                ),
                "samples_with_any_head_exact_set_match": sum(
                    1 for item in last_layer_summaries if item["heads_with_exact_set_match"] > 0
                ),
                "mean_oracle_top1_proxy_rank_avg": sum(
                    item["mean_oracle_top1_proxy_rank"] for item in last_layer_summaries
                )
                / len(last_layer_summaries),
                "median_oracle_top1_proxy_rank_median": statistics.median(
                    item["median_oracle_top1_proxy_rank"] for item in last_layer_summaries
                ),
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.log(f"Wrote summary to {summary_path}")
    finally:
        raw_trace_fp.close()
        sample_fp.close()
        cleanup_model(model)
        logger.close()


if __name__ == "__main__":
    main()
