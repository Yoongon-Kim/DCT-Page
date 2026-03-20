#!/usr/bin/env python3
"""
Compare proxy page scores against full-page oracle scores on NIAH samples.

This script:
1. Finds the gold needle page(s).
2. Runs DCT-Page generation with decode-time traces enabled.
3. Records first-step proxy page scores and full-page oracle scores.
4. Summarizes whether the compressed proxy loses ranking recall on the gold page.
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
    classify_token_position,
    cleanup_model,
    extract_niah_needle,
    load_model,
    load_samples,
    locate_token_span,
    make_run_dir,
    write_jsonl_line,
)


class FirstStepOracleRecorder:
    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step != 0:
            return

        cache_position = payload["cache_position"]
        if cache_position is not None:
            cache_position = cache_position.tolist()

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
                "cache_position": cache_position,
                "page_scores": payload["page_scores"].tolist(),
                "oracle_page_scores": payload["oracle_page_scores"].tolist(),
                "selected_indices": payload["selected_indices"].tolist(),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze proxy-vs-oracle page score quality on NIAH"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Prepared NIAH jsonl file with `input` and `outputs` fields.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/debug_compare"),
        help="Root directory where timestamped analysis results are written.",
    )
    parser.add_argument("--tag", type=str, default="proxy_oracle_page_scores")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--num_samples", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--dct_page_size", type=int, default=128)
    parser.add_argument("--dct_top_k", type=int, default=8)
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
    parser.add_argument("--dct_score_use_direct_spectral_proxy", action="store_true")
    parser.add_argument("--dct_score_use_haar_proxy", action="store_true")
    parser.add_argument("--dct_score_use_haar_mixed_proxy", action="store_true")
    parser.add_argument("--dct_score_use_hadamard_proxy", action="store_true")
    parser.add_argument("--dct_no_continuous_rope", action="store_true")
    parser.add_argument("--dct_no_triton", action="store_true")
    return parser.parse_args()


def summarize_rank_record(
    record: dict[str, Any], target_pages: list[int], score_key: str
) -> dict[str, Any]:
    page_scores = record[score_key][0]
    head_ranks = []
    heads_in_top_k = 0
    actual_top_k = int(record["actual_top_k"])
    for head_idx, head_scores in enumerate(page_scores):
        ranked_pages = sorted(
            range(len(head_scores)),
            key=lambda page_idx: head_scores[page_idx],
            reverse=True,
        )
        best_rank = min(ranked_pages.index(page_idx) + 1 for page_idx in target_pages)
        in_top_k = best_rank <= actual_top_k
        if in_top_k:
            heads_in_top_k += 1
        head_ranks.append(
            {
                "kv_head": head_idx,
                "best_rank": best_rank,
                "in_top_k": in_top_k,
                "target_scores": {
                    str(page_idx): float(head_scores[page_idx]) for page_idx in target_pages
                },
            }
        )

    rank_values = [item["best_rank"] for item in head_ranks]
    return {
        "layer_idx": record["layer_idx"],
        "decode_step": record["decode_step"],
        "kv_len": record["kv_len"],
        "num_pages": record["num_pages"],
        "actual_top_k": actual_top_k,
        "target_pages": target_pages,
        "min_rank": min(rank_values),
        "median_rank": statistics.median(rank_values),
        "max_rank": max(rank_values),
        "heads_in_top_k": heads_in_top_k,
        "num_heads": len(head_ranks),
        "head_ranks": head_ranks,
    }


def summarize_proxy_oracle_gap(proxy_summary: dict[str, Any], oracle_summary: dict[str, Any]) -> dict[str, Any]:
    oracle_hit_proxy_miss = 0
    oracle_better_rank_heads = 0
    proxy_better_rank_heads = 0
    tied_rank_heads = 0
    for proxy_head, oracle_head in zip(
        proxy_summary["head_ranks"], oracle_summary["head_ranks"]
    ):
        if oracle_head["in_top_k"] and not proxy_head["in_top_k"]:
            oracle_hit_proxy_miss += 1
        if oracle_head["best_rank"] < proxy_head["best_rank"]:
            oracle_better_rank_heads += 1
        elif oracle_head["best_rank"] > proxy_head["best_rank"]:
            proxy_better_rank_heads += 1
        else:
            tied_rank_heads += 1

    return {
        "median_rank_gap": proxy_summary["median_rank"] - oracle_summary["median_rank"],
        "min_rank_gap": proxy_summary["min_rank"] - oracle_summary["min_rank"],
        "max_rank_gap": proxy_summary["max_rank"] - oracle_summary["max_rank"],
        "heads_proxy_in_top_k": proxy_summary["heads_in_top_k"],
        "heads_oracle_in_top_k": oracle_summary["heads_in_top_k"],
        "heads_oracle_hit_proxy_miss": oracle_hit_proxy_miss,
        "heads_oracle_better_rank": oracle_better_rank_heads,
        "heads_proxy_better_rank": proxy_better_rank_heads,
        "heads_tied_rank": tied_rank_heads,
    }


def generate_with_first_step_traces(model, tokenizer, sample: dict[str, Any], max_new_tokens: int):
    from dct_page_attention import set_dct_page_debug_hook

    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder = FirstStepOracleRecorder()
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
    generation = {
        "index": sample["index"],
        "input_len": int(input_ids.shape[1]),
        "gen_ids": generated_ids.tolist(),
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "gold": sample["outputs"],
    }
    return generation, recorder.records


def build_summary_block(values: list[dict[str, Any]], num_heads: int) -> dict[str, Any]:
    return {
        "median_of_median_ranks": statistics.median(item["median_rank"] for item in values),
        "mean_of_median_ranks": sum(item["median_rank"] for item in values) / len(values),
        "samples_with_all_heads_in_top_k": sum(
            1 for item in values if item["heads_in_top_k"] == num_heads
        ),
        "samples_with_any_head_in_top_k": sum(
            1 for item in values if item["heads_in_top_k"] > 0
        ),
    }


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root, args.tag)
    logger = RunLogger(run_dir / "run.log")

    raw_trace_path = run_dir / "raw_traces.jsonl"
    sample_path = run_dir / "sample_analysis.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2, default=str)
        fp.write("\n")
        fp.flush()

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
    proxy_last_layer = []
    oracle_last_layer = []
    gap_last_layer = []
    num_heads = None
    pageable_samples = 0
    recent_only_samples = 0
    content_match_count = 0

    try:
        for sample_idx, sample in enumerate(samples, start=1):
            needle_info = extract_niah_needle(sample)
            token_span, tokenized_len = locate_token_span(
                tokenizer,
                sample["input"],
                needle_info["start_char"],
                needle_info["end_char"],
            )

            generation, traces = generate_with_first_step_traces(
                model, tokenizer, sample, args.max_new_tokens
            )
            content_match = needle_info["value"] in generation["text"]
            if content_match:
                content_match_count += 1

            for trace in traces:
                write_jsonl_line(
                    raw_trace_fp,
                    {
                        "sample_index": sample["index"],
                        **trace,
                    },
                )

            if not traces:
                raise RuntimeError(
                    f"No first decode-step traces were recorded for sample {sample['index']}."
                )

            first_trace = traces[0]
            region_records = [
                classify_token_position(
                    token_idx=token_idx,
                    kv_len=first_trace["kv_len"],
                    sink_size=first_trace["sink_size"],
                    page_size=first_trace["page_size"],
                    recent_size=first_trace["recent_size"],
                )
                for token_idx in token_span
            ]
            target_pages = sorted(
                {
                    item["page_idx"]
                    for item in region_records
                    if item["region"] == "page" and item["page_idx"] is not None
                }
            )

            if not target_pages:
                sample_summary = {
                    "sample_index": sample["index"],
                    "query_key": needle_info["key"],
                    "gold_value": needle_info["value"],
                    "generated_text": generation["text"],
                    "content_match": content_match,
                    "target_pages": [],
                    "selection_needed": False,
                    "selection_reason": "needle_in_recent_region",
                }
                write_jsonl_line(sample_fp, sample_summary)
                sample_summaries.append(sample_summary)
                recent_only_samples += 1
                logger.log(
                    "sample "
                    f"{sample_idx}/{len(samples)} index={sample['index']} key={needle_info['key']} "
                    "needle lives in recent region, so page selection is not needed"
                )
                continue

            layer_summaries = []
            for record in traces:
                proxy_summary = summarize_rank_record(record, target_pages, "page_scores")
                oracle_summary = summarize_rank_record(
                    record, target_pages, "oracle_page_scores"
                )
                gap_summary = summarize_proxy_oracle_gap(proxy_summary, oracle_summary)
                layer_summaries.append(
                    {
                        "layer_idx": record["layer_idx"],
                        "proxy": {
                            "median_rank": proxy_summary["median_rank"],
                            "heads_in_top_k": proxy_summary["heads_in_top_k"],
                            "num_heads": proxy_summary["num_heads"],
                        },
                        "oracle": {
                            "median_rank": oracle_summary["median_rank"],
                            "heads_in_top_k": oracle_summary["heads_in_top_k"],
                            "num_heads": oracle_summary["num_heads"],
                        },
                        "gap": gap_summary,
                    }
                )

            last_trace = max(traces, key=lambda item: item["layer_idx"])
            last_proxy = summarize_rank_record(last_trace, target_pages, "page_scores")
            last_oracle = summarize_rank_record(last_trace, target_pages, "oracle_page_scores")
            last_gap = summarize_proxy_oracle_gap(last_proxy, last_oracle)

            sample_summary = {
                "sample_index": sample["index"],
                "query_key": needle_info["key"],
                "gold_value": needle_info["value"],
                "input_len": generation["input_len"],
                "tokenized_len": tokenized_len,
                "generated_text": generation["text"],
                "content_match": content_match,
                "target_pages": target_pages,
                "needle_token_span": [token_span[0], token_span[-1]],
                "selection_needed": True,
                "page_layout": {
                    "page_size": first_trace["page_size"],
                    "sink_size": first_trace["sink_size"],
                    "recent_size": first_trace["recent_size"],
                    "num_pages": first_trace["num_pages"],
                },
                "first_step_last_layer_proxy": last_proxy,
                "first_step_last_layer_oracle": last_oracle,
                "first_step_last_layer_gap": last_gap,
                "first_step_layers": layer_summaries,
            }
            write_jsonl_line(sample_fp, sample_summary)

            sample_summaries.append(sample_summary)
            proxy_last_layer.append(last_proxy)
            oracle_last_layer.append(last_oracle)
            gap_last_layer.append(last_gap)
            num_heads = last_proxy["num_heads"]
            pageable_samples += 1

            logger.log(
                "sample "
                f"{sample_idx}/{len(samples)} index={sample['index']} key={needle_info['key']} "
                f"proxy_last_median_rank={last_proxy['median_rank']} "
                f"oracle_last_median_rank={last_oracle['median_rank']} "
                f"oracle_hit_proxy_miss_heads={last_gap['heads_oracle_hit_proxy_miss']}/"
                f"{last_proxy['num_heads']} content_match={content_match}"
            )

        summary = {
            "num_samples": len(sample_summaries),
            "num_pageable_samples": pageable_samples,
            "num_recent_only_samples": recent_only_samples,
            "content_match_samples": content_match_count,
            "dct_top_k": args.dct_top_k,
            "dct_compress_ratio": args.dct_compress_ratio,
            "dct_proxy_frequency_layout": args.dct_proxy_frequency_layout,
            "dct_score_use_direct_spectral_proxy": args.dct_score_use_direct_spectral_proxy,
            "dct_score_use_haar_proxy": args.dct_score_use_haar_proxy,
            "dct_score_use_haar_mixed_proxy": args.dct_score_use_haar_mixed_proxy,
            "dct_score_use_hadamard_proxy": args.dct_score_use_hadamard_proxy,
            "first_step_last_layer": {
                "proxy": build_summary_block(proxy_last_layer, num_heads),
                "oracle": build_summary_block(oracle_last_layer, num_heads),
                "gap": {
                    "median_of_median_rank_gaps": statistics.median(
                        item["median_rank_gap"] for item in gap_last_layer
                    ),
                    "mean_of_median_rank_gaps": sum(
                        item["median_rank_gap"] for item in gap_last_layer
                    ) / len(gap_last_layer),
                    "samples_where_oracle_median_rank_is_better": sum(
                        1 for item in gap_last_layer if item["median_rank_gap"] > 0
                    ),
                    "samples_where_oracle_any_head_hits_but_proxy_any_head_misses": sum(
                        1
                        for proxy_item, oracle_item in zip(proxy_last_layer, oracle_last_layer)
                        if oracle_item["heads_in_top_k"] > 0 and proxy_item["heads_in_top_k"] == 0
                    ),
                    "samples_where_oracle_all_heads_hit_but_proxy_not_all": sum(
                        1
                        for proxy_item, oracle_item in zip(proxy_last_layer, oracle_last_layer)
                        if oracle_item["heads_in_top_k"] == num_heads
                        and proxy_item["heads_in_top_k"] < num_heads
                    ),
                    "mean_heads_oracle_hit_proxy_miss": sum(
                        item["heads_oracle_hit_proxy_miss"] for item in gap_last_layer
                    ) / len(gap_last_layer),
                },
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        logger.log(f"Wrote summary to {summary_path}")
    finally:
        raw_trace_fp.close()
        sample_fp.close()
        cleanup_model(model)
        logger.close()


if __name__ == "__main__":
    main()
