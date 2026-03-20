#!/usr/bin/env python3
"""
Compare greedy generation outputs between vanilla baseline and DCT-Page.

Writes baseline outputs, DCT outputs, per-sample comparisons, and a run log to a
timestamped directory under results/. Every line is flushed immediately so
long-running runs can be monitored while they are still in progress.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs DCT-Page outputs")
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Prepared RULER jsonl file with `input` and `outputs` fields.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/debug_compare"),
        help="Root directory where timestamped run results are written.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline_vs_dct",
        help="Prefix used for the timestamped result directory.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--benchmark", type=str, default="synthetic")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name for metric lookup. Defaults to the parent directory name of data_path.",
    )
    parser.add_argument("--num_samples", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--dct_page_size", type=int, default=128)
    parser.add_argument("--dct_top_k", type=int, default=8)
    parser.add_argument("--dct_sink_size", type=int, default=4)
    parser.add_argument("--dct_recent_size", type=int, default=128)
    parser.add_argument("--dct_compress_ratio", type=float, default=1.0)
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
    parser.add_argument("--dct_no_continuous_rope", action="store_true")
    parser.add_argument("--dct_no_triton", action="store_true")
    return parser.parse_args()


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


def diff_pos(left: list[int], right: list[int]) -> int | None:
    shared = min(len(left), len(right))
    for idx in range(shared):
        if left[idx] != right[idx]:
            return idx
    if len(left) != len(right):
        return shared
    return None


def infer_task_name(args: argparse.Namespace) -> str:
    if args.task is not None:
        return args.task
    return args.data_path.parent.name


def load_metric_fn(benchmark: str, task_name: str):
    curr_folder = Path(__file__).resolve().parent / "eval_ruler" / "eval"
    if str(curr_folder) not in sys.path:
        sys.path.insert(0, str(curr_folder))
    module = importlib.import_module(f"{benchmark}.constants")
    tasks_base = module.TASKS
    with (curr_folder.parent / f"{benchmark}.yaml").open("r", encoding="utf-8") as fp:
        tasks_customized = yaml.safe_load(fp)
    task_config = tasks_customized[task_name]
    task_config.update(tasks_base[task_config["task"]])
    return task_config["metric_fn"], task_config


def score_predictions(metric_fn, rows: list[dict[str, Any]]) -> float:
    preds = [row["text"] for row in rows]
    refs = [row["gold"] for row in rows]
    return metric_fn(preds, refs)


def load_samples(path: Path, num_samples: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        rows = [json.loads(line) for line in fp if line.strip()]
    return rows[:num_samples]


def cleanup_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(args: argparse.Namespace):
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.cuda_device},
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).eval()


def generate_one(
    model,
    tokenizer,
    sample: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1] :].cpu()
    return {
        "index": sample["index"],
        "input_len": int(input_ids.shape[1]),
        "gen_ids": generated_ids.tolist(),
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "gold": sample["outputs"],
    }


def apply_dct_patch(args: argparse.Namespace) -> None:
    model_name = args.model_name_or_path.lower()
    common_kwargs = dict(
        page_size=args.dct_page_size,
        top_k=args.dct_top_k,
        sink_size=args.dct_sink_size,
        recent_size=args.dct_recent_size,
        compress_ratio=args.dct_compress_ratio,
        scoring_method=args.dct_scoring_method,
        group_agg_method=args.dct_group_agg_method,
        unselected_mode=args.dct_unselected_mode,
        continuous_rope=not args.dct_no_continuous_rope,
        score_with_original_rope=args.dct_score_with_original_rope,
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


def make_run_dir(output_root: Path, tag: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root, args.tag)
    logger = RunLogger(run_dir / "run.log")
    task_name = infer_task_name(args)
    metric_fn, task_config = load_metric_fn(args.benchmark, task_name)

    baseline_path = run_dir / "baseline.jsonl"
    dct_path = run_dir / "dct.jsonl"
    compare_path = run_dir / "compare.jsonl"
    baseline_eval_path = run_dir / "baseline_eval.jsonl"
    dct_eval_path = run_dir / "dct_eval.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2, default=str)
        fp.write("\n")
        fp.flush()

    baseline_fp = baseline_path.open("a", encoding="utf-8", buffering=1)
    dct_fp = dct_path.open("a", encoding="utf-8", buffering=1)
    compare_fp = compare_path.open("a", encoding="utf-8", buffering=1)
    baseline_eval_fp = baseline_eval_path.open("a", encoding="utf-8", buffering=1)
    dct_eval_fp = dct_eval_path.open("a", encoding="utf-8", buffering=1)

    start_time = time.time()
    try:
        samples = load_samples(args.data_path, args.num_samples)
        logger.log(f"Loaded {len(samples)} samples from {args.data_path}")
        logger.log(f"Using task={task_name} benchmark={args.benchmark}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            local_files_only=args.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.log(f"Set pad_token_id to eos_token_id={tokenizer.pad_token_id}")

        baseline_rows = []
        logger.log("Loading baseline model")
        baseline_model = load_model(args)
        logger.log("Running baseline generation")
        for idx, sample in enumerate(samples, start=1):
            row = generate_one(
                baseline_model,
                tokenizer,
                sample,
                max_new_tokens=args.max_new_tokens,
            )
            baseline_rows.append(row)
            write_jsonl_line(baseline_fp, row)
            write_jsonl_line(
                baseline_eval_fp,
                {
                    "index": sample["index"],
                    "pred": row["text"],
                    "input": sample["input"],
                    "outputs": sample["outputs"],
                    "others": sample.get("others", {}),
                    "length": sample.get("length", -1),
                },
            )
            logger.log(
                f"baseline sample {idx}/{len(samples)} index={row['index']} "
                f"input_len={row['input_len']} text={row['text']!r}"
            )
        cleanup_model(baseline_model)
        baseline_score = score_predictions(metric_fn, baseline_rows)
        logger.log(f"Baseline score={baseline_score}")

        logger.log("Applying DCT patch")
        apply_dct_patch(args)

        logger.log("Loading DCT model")
        dct_model = load_model(args)
        logger.log("Running DCT generation")

        matches = 0
        dct_rows = []
        mismatch_examples = []
        for idx, (sample, baseline_row) in enumerate(zip(samples, baseline_rows), start=1):
            dct_row = generate_one(
                dct_model,
                tokenizer,
                sample,
                max_new_tokens=args.max_new_tokens,
            )
            dct_rows.append(dct_row)
            write_jsonl_line(dct_fp, dct_row)
            write_jsonl_line(
                dct_eval_fp,
                {
                    "index": sample["index"],
                    "pred": dct_row["text"],
                    "input": sample["input"],
                    "outputs": sample["outputs"],
                    "others": sample.get("others", {}),
                    "length": sample.get("length", -1),
                },
            )

            same = baseline_row["gen_ids"] == dct_row["gen_ids"]
            if same:
                matches += 1

            compare_row = {
                "index": sample["index"],
                "input_len": baseline_row["input_len"],
                "exact_match": same,
                "first_diff": diff_pos(baseline_row["gen_ids"], dct_row["gen_ids"]),
                "gold": sample["outputs"],
                "baseline_text": baseline_row["text"],
                "dct_text": dct_row["text"],
                "baseline_gen_ids": baseline_row["gen_ids"],
                "dct_gen_ids": dct_row["gen_ids"],
            }
            write_jsonl_line(compare_fp, compare_row)

            if not same and len(mismatch_examples) < 5:
                mismatch_examples.append(compare_row)

            logger.log(
                f"dct sample {idx}/{len(samples)} index={sample['index']} "
                f"match={same} first_diff={compare_row['first_diff']} "
                f"baseline={baseline_row['text']!r} dct={dct_row['text']!r}"
            )

        cleanup_model(dct_model)
        dct_score = score_predictions(metric_fn, dct_rows)
        logger.log(f"DCT score={dct_score}")

        summary = {
            "data_path": str(args.data_path),
            "model_name_or_path": args.model_name_or_path,
            "benchmark": args.benchmark,
            "task": task_name,
            "num_samples": len(samples),
            "exact_matches": matches,
            "exact_match_rate": matches / len(samples) if samples else 0.0,
            "baseline_score": baseline_score,
            "dct_score": dct_score,
            "dct": {
                "page_size": args.dct_page_size,
                "top_k": args.dct_top_k,
                "sink_size": args.dct_sink_size,
                "recent_size": args.dct_recent_size,
                "compress_ratio": args.dct_compress_ratio,
                "scoring_method": args.dct_scoring_method,
                "group_agg_method": args.dct_group_agg_method,
                "unselected_mode": args.dct_unselected_mode,
                "continuous_rope": not args.dct_no_continuous_rope,
                "use_triton": not args.dct_no_triton,
            },
            "artifacts": {
                "baseline": str(baseline_path),
                "dct": str(dct_path),
                "compare": str(compare_path),
                "baseline_eval": str(baseline_eval_path),
                "dct_eval": str(dct_eval_path),
                "log": str(run_dir / "run.log"),
            },
            "mismatch_examples": mismatch_examples,
            "elapsed_sec": time.time() - start_time,
        }
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
            fp.flush()

        logger.log(
            f"Finished comparison: exact_match={matches}/{len(samples)} "
            f"summary={summary_path}"
        )
    finally:
        baseline_fp.close()
        dct_fp.close()
        compare_fp.close()
        baseline_eval_fp.close()
        dct_eval_fp.close()
        logger.close()


if __name__ == "__main__":
    main()
