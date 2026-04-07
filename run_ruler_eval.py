#!/usr/bin/env python3
"""
Run a single RULER evaluation mode and accumulate task outputs in one flat run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path

from compare_baseline_dct import (
    RunLogger,
    apply_dct_patch,
    cleanup_model,
    generate_one,
    load_metric_fn,
    load_model,
    load_samples,
)


TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one RULER mode into a flat task-jsonl directory")
    p.add_argument("--mode", choices=["baseline", "page_attention"], required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, required=True)
    p.add_argument("--data_root", type=Path, default=Path("ruler_data"))
    p.add_argument("--output_root", type=Path, default=Path("results_ruler_oracle/ruler_runs"))
    p.add_argument("--tag", default="ruler_run")
    p.add_argument("--run_dir", type=Path, default=None)
    p.add_argument("--benchmark", default="synthetic")
    p.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--dct_page_size", type=int, default=32)
    p.add_argument("--dct_top_k", type=int, default=64)
    p.add_argument("--dct_sink_size", type=int, default=4)
    p.add_argument("--dct_recent_size", type=int, default=128)
    p.add_argument("--dct_compress_ratio", type=float, default=0.03125)
    p.add_argument(
        "--dct_proxy_frequency_layout",
        type=str,
        default="low",
        choices=["low", "low_high", "low_mid_high", "spread"],
    )
    p.add_argument("--dct_scoring_method", type=str, default="max", choices=["mean", "max", "sum"])
    p.add_argument(
        "--dct_group_agg_method",
        type=str,
        default="mean",
        choices=["mean", "max", "topp"],
    )
    p.add_argument(
        "--dct_unselected_mode",
        type=str,
        default="drop",
        choices=["drop", "compressed"],
    )
    p.add_argument("--dct_compression_method", type=str, default="haar", choices=["haar", "dct"])
    p.add_argument("--dct_compressed_token_rope", type=str, default="mixed", choices=["mixed", "block_center"])
    p.add_argument("--dct_score_use_direct_spectral_proxy", action="store_true")
    p.add_argument(
        "--dct_score_use_haar_proxy",
        dest="dct_score_use_haar_proxy",
        action="store_true",
        help="Use Haar lowpass score proxies (default).",
    )
    p.add_argument(
        "--dct_score_use_low_proxy",
        dest="dct_score_use_haar_proxy",
        action="store_false",
        help="Use the original low-frequency DCT IDCT score proxy instead of Haar.",
    )
    p.add_argument("--dct_score_use_haar_mixed_proxy", action="store_true")
    p.add_argument("--dct_score_use_hadamard_proxy", action="store_true")
    p.add_argument("--dct_select_with_oracle_page_scores", action="store_true")
    p.add_argument("--dct_continuous_rope", action="store_true",
                   help="Temporarily disabled — raises error if used")
    p.add_argument("--dct_no_triton", action="store_true")
    p.set_defaults(dct_score_use_haar_proxy=True)
    return p.parse_args()


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
        return name.split("-")[0]


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


def make_run_dir(output_root: Path, page_size: int, top_k: int, compress_ratio: float) -> Path:
    run_dir = output_root / f"ps{page_size}_topk{top_k}_cr{compress_ratio}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def score_predictions(metric_fn, rows: list[dict]) -> float:
    preds = [row["pred"] for row in rows]
    refs = [row["outputs"] for row in rows]
    return metric_fn(preds, refs)


def make_eval_row(sample: dict, row: dict) -> dict:
    return {
        "index": sample["index"],
        "pred": row["text"],
        "input": sample["input"],
        "outputs": sample["outputs"],
        "others": sample.get("others", {}),
        "length": sample.get("length", -1),
        "input_len": row["input_len"],
        "gen_ids": row["gen_ids"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        fp.flush()


def write_summary(run_dir: Path, manifest: dict, summary_rows: list[dict]) -> None:
    summary_rows = sorted(summary_rows, key=lambda r: TASKS.index(r["task"]))
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["task", "score", "num_samples", "output_jsonl", "elapsed_sec"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    overall = sum(row["score"] for row in summary_rows) / len(summary_rows) if summary_rows else 0.0
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "rows": summary_rows,
                "overall_score_avg": overall,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    run_dir = args.run_dir if args.run_dir is not None else make_run_dir(args.output_root, args.dct_page_size, args.dct_top_k, args.dct_compress_ratio)
    if args.run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": args.mode,
        "model_name_or_path": args.model_name_or_path,
        "context_len": args.context_len,
        "benchmark": args.benchmark,
        "tasks": tasks,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "cuda_device": args.cuda_device,
        "dct": {
            "page_size": args.dct_page_size,
            "top_k": args.dct_top_k,
            "sink_size": args.dct_sink_size,
            "recent_size": args.dct_recent_size,
            "compress_ratio": args.dct_compress_ratio,
            "proxy_frequency_layout": args.dct_proxy_frequency_layout,
            "scoring_method": args.dct_scoring_method,
            "group_agg_method": args.dct_group_agg_method,
            "unselected_mode": args.dct_unselected_mode,
            "compression_method": args.dct_compression_method,
            "compressed_token_rope": args.dct_compressed_token_rope,
            "continuous_rope": args.dct_continuous_rope,
            "score_use_direct_spectral_proxy": args.dct_score_use_direct_spectral_proxy,
            "score_use_haar_proxy": args.dct_score_use_haar_proxy,
            "score_use_haar_mixed_proxy": args.dct_score_use_haar_mixed_proxy,
            "score_use_hadamard_proxy": args.dct_score_use_hadamard_proxy,
            "select_with_oracle_page_scores": args.dct_select_with_oracle_page_scores,
            "use_triton": not args.dct_no_triton,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + " ".join(shlex.quote(arg) for arg in [sys.executable, __file__, *sys.argv[1:]])
        + "\n",
        encoding="utf-8",
    )

    logger = RunLogger(run_dir / "run.log")
    model = None
    try:
        if args.mode == "page_attention":
            logger.log("Applying page-attention patch")
            apply_dct_patch(args)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            local_files_only=args.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.log(f"Set pad_token_id to eos_token_id={tokenizer.pad_token_id}")

        logger.log(f"Loading model for mode={args.mode}")
        model = load_model(args)

        summary_rows: list[dict] = []
        run_start = time.time()
        for task in tasks:
            task_start = time.time()
            model_family = resolve_model_family(args.model_name_or_path)
            data_path = args.data_root / model_family / str(args.context_len) / task / "validation.jsonl"
            samples = load_samples(data_path, args.num_samples)
            metric_fn, _ = load_metric_fn(args.benchmark, task)
            logger.log(f"Running task={task} samples={len(samples)} data={data_path}")

            eval_rows = []
            for idx, sample in enumerate(samples, start=1):
                gen_row = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                )
                eval_rows.append(make_eval_row(sample, gen_row))
                logger.log(
                    f"{task} sample {idx}/{len(samples)} index={sample['index']} "
                    f"input_len={gen_row['input_len']} pred={gen_row['text']!r}"
                )

            output_path = run_dir / f"{task}.jsonl"
            write_jsonl(output_path, eval_rows)
            score = score_predictions(metric_fn, eval_rows)
            summary_rows.append(
                {
                    "task": task,
                    "score": score,
                    "num_samples": len(eval_rows),
                    "output_jsonl": str(output_path),
                    "elapsed_sec": time.time() - task_start,
                }
            )
            write_summary(run_dir, manifest, summary_rows)
            logger.log(
                f"Finished task={task} score={score:.4f} "
                f"elapsed={time.time() - task_start:.1f}s overall_elapsed={time.time() - run_start:.1f}s"
            )
    finally:
        cleanup_model(model)
        logger.close()


if __name__ == "__main__":
    main()
