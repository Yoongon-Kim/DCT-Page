#!/usr/bin/env python3
"""
Analyze how highly the gold NIAH needle page ranks during DCT-Page decode.

This script:
1. Maps the gold needle sentence to token positions and page indices.
2. Runs greedy generation with DCT-Page enabled.
3. Records per-layer page scores and selected pages during decode.
4. Summarizes where the gold page ranks, especially at the first decode step.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTION_RE = re.compile(
    r"What is the special magic (?P<needle_type>\w+) for (?P<key>.+?) "
    r"mentioned in the provided text\?"
)


class RunLogger:
    def __init__(self, log_path: Path):
        self._fp = log_path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


def write_jsonl_line(fp, payload: dict[str, Any]) -> None:
    fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fp.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NIAH needle page ranks")
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
    parser.add_argument("--tag", type=str, default="niah_page_rank")
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
    parser.add_argument("--dct_compress_ratio", type=float, default=1.0)
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
    parser.add_argument("--dct_score_with_original_rope", action="store_true")
    parser.add_argument("--dct_score_use_direct_spectral_proxy", action="store_true")
    parser.add_argument("--dct_score_use_haar_proxy", action="store_true")
    parser.add_argument("--dct_score_use_haar_mixed_proxy", action="store_true")
    parser.add_argument("--dct_score_use_hadamard_proxy", action="store_true")
    parser.add_argument("--dct_score_proxy_with_block_position_rope", action="store_true")
    parser.add_argument("--dct_score_proxy_with_shared_block_center_rope", action="store_true")
    parser.add_argument("--dct_score_proxy_with_shared_block_start_rope", action="store_true")
    parser.add_argument("--dct_no_continuous_rope", action="store_true")
    parser.add_argument("--dct_no_triton", action="store_true")
    return parser.parse_args()


def make_run_dir(output_root: Path, tag: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def cleanup_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_samples(path: Path, num_samples: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        rows = [json.loads(line) for line in fp if line.strip()]
    return rows[:num_samples]


def load_model(args: argparse.Namespace):
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.cuda_device},
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).eval()


def apply_dct_patch(args: argparse.Namespace):
    model_name = args.model_name_or_path.lower()
    common_kwargs = dict(
        page_size=args.dct_page_size,
        top_k=args.dct_top_k,
        sink_size=args.dct_sink_size,
        recent_size=args.dct_recent_size,
        compress_ratio=args.dct_compress_ratio,
        proxy_frequency_layout=args.dct_proxy_frequency_layout,
        scoring_method=args.dct_scoring_method,
        group_agg_method=args.dct_group_agg_method,
        unselected_mode=args.dct_unselected_mode,
        continuous_rope=not args.dct_no_continuous_rope,
        score_with_original_rope=args.dct_score_with_original_rope,
        score_use_direct_spectral_proxy=args.dct_score_use_direct_spectral_proxy,
        score_use_haar_proxy=args.dct_score_use_haar_proxy,
        score_use_haar_mixed_proxy=args.dct_score_use_haar_mixed_proxy,
        score_use_hadamard_proxy=args.dct_score_use_hadamard_proxy,
        score_proxy_with_block_position_rope=args.dct_score_proxy_with_block_position_rope,
        score_proxy_with_shared_block_center_rope=args.dct_score_proxy_with_shared_block_center_rope,
        score_proxy_with_shared_block_start_rope=args.dct_score_proxy_with_shared_block_start_rope,
        use_triton=not args.dct_no_triton,
    )
    if "llama" in model_name:
        from dct_page_attention import replace_llama_attn

        replace_llama_attn(**common_kwargs)
    elif "qwen" in model_name:
        from dct_page_attention import replace_qwen2_attn

        replace_qwen2_attn(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model for DCT patch: {args.model_name_or_path}")


class PageTraceRecorder:
    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
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
                "selected_indices": payload["selected_indices"].tolist(),
            }
        )


def extract_niah_needle(sample: dict[str, Any]) -> dict[str, Any]:
    match = QUESTION_RE.search(sample["input"])
    if match is None:
        raise ValueError(f"Could not parse NIAH question from sample {sample['index']}")
    if len(sample["outputs"]) != 1:
        raise ValueError("This analyzer currently expects exactly one gold output.")

    needle_type = match.group("needle_type")
    key = match.group("key")
    value = sample["outputs"][0]
    candidate_types = [needle_type]
    if not needle_type.endswith("s"):
        candidate_types.append(f"{needle_type}s")

    needle_sentence = None
    start_char = -1
    for candidate_type in candidate_types:
        candidate_sentence = (
            f"One of the special magic {candidate_type} for {key} is: {value}."
        )
        start_char = sample["input"].find(candidate_sentence)
        if start_char >= 0:
            needle_sentence = candidate_sentence
            break

    if needle_sentence is None:
        relaxed = re.search(
            rf"One of the special magic \w+ for {re.escape(key)} is: {re.escape(value)}\.",
            sample["input"],
        )
        if relaxed is None:
            raise ValueError(
                f"Could not find gold needle sentence for sample {sample['index']}."
            )
        needle_sentence = relaxed.group(0)
        start_char = relaxed.start()

    return {
        "key": key,
        "value": value,
        "needle_type": needle_type,
        "needle_sentence": needle_sentence,
        "start_char": start_char,
        "end_char": start_char + len(needle_sentence),
    }


def locate_token_span(tokenizer, text: str, start_char: int, end_char: int) -> tuple[list[int], int]:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded["offset_mapping"]
    token_indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > start_char and tok_start < end_char
    ]
    if not token_indices:
        raise ValueError("Failed to map needle span to tokenizer offsets.")
    return token_indices, len(encoded["input_ids"])


def classify_token_position(
    token_idx: int,
    kv_len: int,
    sink_size: int,
    page_size: int,
    recent_size: int,
) -> dict[str, Any]:
    pageable_len = kv_len - sink_size - recent_size
    num_pages = max(pageable_len // page_size, 0)
    pages_end = sink_size + num_pages * page_size
    if token_idx < sink_size:
        return {"region": "sink", "page_idx": None}
    if token_idx >= pages_end:
        return {"region": "recent", "page_idx": None}
    return {
        "region": "page",
        "page_idx": (token_idx - sink_size) // page_size,
    }


def summarize_rank_record(record: dict[str, Any], target_pages: list[int]) -> dict[str, Any]:
    page_scores = record["page_scores"][0]
    selected_indices = record["selected_indices"][0]
    head_ranks = []
    heads_in_topk = 0
    for head_idx, head_scores in enumerate(page_scores):
        ranked_pages = sorted(
            range(len(head_scores)),
            key=lambda page_idx: head_scores[page_idx],
            reverse=True,
        )
        best_rank = min(ranked_pages.index(page_idx) + 1 for page_idx in target_pages)
        in_topk = any(page_idx in selected_indices[head_idx] for page_idx in target_pages)
        if in_topk:
            heads_in_topk += 1
        head_ranks.append(
            {
                "kv_head": head_idx,
                "best_rank": best_rank,
                "in_top_k": in_topk,
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
        "actual_top_k": record["actual_top_k"],
        "target_pages": target_pages,
        "min_rank": min(rank_values),
        "median_rank": statistics.median(rank_values),
        "max_rank": max(rank_values),
        "heads_in_top_k": heads_in_topk,
        "num_heads": len(head_ranks),
        "head_ranks": head_ranks,
    }


def generate_with_traces(
    model,
    tokenizer,
    sample: dict[str, Any],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from dct_page_attention import set_dct_page_debug_hook

    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder = PageTraceRecorder()
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
    first_step_layer_medians = []
    first_step_layer_topk_hits = []
    first_step_best_layer_median = []
    first_step_best_layer_hits = []
    num_heads = None
    pageable_samples = 0
    recent_only_samples = 0

    try:
        for sample_idx, sample in enumerate(samples, start=1):
            needle_info = extract_niah_needle(sample)
            token_span, tokenized_len = locate_token_span(
                tokenizer,
                sample["input"],
                needle_info["start_char"],
                needle_info["end_char"],
            )

            generation, traces = generate_with_traces(
                model,
                tokenizer,
                sample,
                args.max_new_tokens,
            )

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
                    f"No DCT decode traces were recorded for sample {sample['index']}."
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
                    "needle_sentence": needle_info["needle_sentence"],
                    "input_len": generation["input_len"],
                    "tokenized_len": tokenized_len,
                    "generated_text": generation["text"],
                    "target_pages": [],
                    "needle_token_span": [token_span[0], token_span[-1]],
                    "first_decode_kv_len": first_trace["kv_len"],
                    "page_layout": {
                        "page_size": first_trace["page_size"],
                        "sink_size": first_trace["sink_size"],
                        "recent_size": first_trace["recent_size"],
                        "num_pages": first_trace["num_pages"],
                    },
                    "region_records": region_records,
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

            first_step_records = [record for record in traces if record["decode_step"] == 0]
            if not first_step_records:
                raise RuntimeError(
                    f"No first decode-step traces recorded for sample {sample['index']}."
                )

            first_step_summaries = [
                summarize_rank_record(record, target_pages) for record in first_step_records
            ]
            best_first_step = min(
                first_step_summaries,
                key=lambda item: (item["median_rank"], -item["heads_in_top_k"]),
            )
            last_layer_first_step = max(first_step_summaries, key=lambda item: item["layer_idx"])

            sample_summary = {
                "sample_index": sample["index"],
                "query_key": needle_info["key"],
                "gold_value": needle_info["value"],
                "needle_sentence": needle_info["needle_sentence"],
                "input_len": generation["input_len"],
                "tokenized_len": tokenized_len,
                "generated_text": generation["text"],
                "target_pages": target_pages,
                "needle_token_span": [token_span[0], token_span[-1]],
                "first_decode_kv_len": first_trace["kv_len"],
                "selection_needed": True,
                "page_layout": {
                    "page_size": first_trace["page_size"],
                    "sink_size": first_trace["sink_size"],
                    "recent_size": first_trace["recent_size"],
                    "num_pages": first_trace["num_pages"],
                },
                "region_records": region_records,
                "first_step_last_layer": {
                    "layer_idx": last_layer_first_step["layer_idx"],
                    "min_rank": last_layer_first_step["min_rank"],
                    "median_rank": last_layer_first_step["median_rank"],
                    "max_rank": last_layer_first_step["max_rank"],
                    "heads_in_top_k": last_layer_first_step["heads_in_top_k"],
                    "num_heads": last_layer_first_step["num_heads"],
                },
                "first_step_best_layer": {
                    "layer_idx": best_first_step["layer_idx"],
                    "min_rank": best_first_step["min_rank"],
                    "median_rank": best_first_step["median_rank"],
                    "max_rank": best_first_step["max_rank"],
                    "heads_in_top_k": best_first_step["heads_in_top_k"],
                    "num_heads": best_first_step["num_heads"],
                },
                "first_step_layers": first_step_summaries,
            }
            write_jsonl_line(sample_fp, sample_summary)

            sample_summaries.append(sample_summary)
            first_step_layer_medians.append(last_layer_first_step["median_rank"])
            first_step_layer_topk_hits.append(last_layer_first_step["heads_in_top_k"])
            first_step_best_layer_median.append(best_first_step["median_rank"])
            first_step_best_layer_hits.append(best_first_step["heads_in_top_k"])
            num_heads = last_layer_first_step["num_heads"]
            pageable_samples += 1

            logger.log(
                "sample "
                f"{sample_idx}/{len(samples)} index={sample['index']} key={needle_info['key']} "
                f"target_pages={target_pages} last_layer_median_rank="
                f"{last_layer_first_step['median_rank']} "
                f"last_layer_heads_in_topk={last_layer_first_step['heads_in_top_k']}/"
                f"{last_layer_first_step['num_heads']} "
                f"best_layer={best_first_step['layer_idx']} "
                f"best_layer_median_rank={best_first_step['median_rank']}"
            )

        summary = {
            "num_samples": len(sample_summaries),
            "num_pageable_samples": pageable_samples,
            "num_recent_only_samples": recent_only_samples,
            "dct_top_k": args.dct_top_k,
            "dct_compress_ratio": args.dct_compress_ratio,
            "dct_score_use_haar_proxy": args.dct_score_use_haar_proxy,
            "dct_score_use_haar_mixed_proxy": args.dct_score_use_haar_mixed_proxy,
            "dct_score_use_hadamard_proxy": args.dct_score_use_hadamard_proxy,
            "first_step_last_layer": {
                "median_of_median_ranks": statistics.median(first_step_layer_medians),
                "mean_of_median_ranks": sum(first_step_layer_medians) / len(first_step_layer_medians),
                "samples_with_all_heads_in_top_k": sum(
                    1 for value in first_step_layer_topk_hits if value == num_heads
                ),
                "samples_with_any_head_in_top_k": sum(
                    1 for value in first_step_layer_topk_hits if value > 0
                ),
            },
            "first_step_best_layer": {
                "median_of_median_ranks": statistics.median(first_step_best_layer_median),
                "mean_of_median_ranks": sum(first_step_best_layer_median)
                / len(first_step_best_layer_median),
                "samples_with_all_heads_in_top_k": sum(
                    1 for value in first_step_best_layer_hits if value == num_heads
                ),
                "samples_with_any_head_in_top_k": sum(
                    1 for value in first_step_best_layer_hits if value > 0
                ),
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
