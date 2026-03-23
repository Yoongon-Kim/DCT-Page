#!/usr/bin/env python3
"""
Run RULER oracle-selection upper-bound sweeps and summarize results.

This orchestrates flat per-page-size runs built on run_ruler_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


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
    p = argparse.ArgumentParser(description="Run RULER oracle-selection upper-bound sweeps")
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--data_root", type=Path, default=Path("results_ruler/data/synthetic"))
    p.add_argument("--output_root", type=Path, default=Path("results/ruler_oracle_selection"))
    p.add_argument("--tag", default="oracle_selection_ruler")
    p.add_argument("--page_sizes", default="32,64,128")
    p.add_argument("--selected_token_budget", type=int, default=2048)
    p.add_argument("--compress_ratio", type=float, default=0.03125)
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    p.add_argument("--cuda_device", type=int, default=0, help="Used for all runs in this process")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


def make_run_root(output_root: Path, tag: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = output_root / f"{tag}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def build_run_cmd(
    repo_root: Path,
    model_name_or_path: str,
    run_dir: Path,
    context_len: int,
    tasks: list[str],
    page_size: int,
    top_k: int,
    compress_ratio: float,
    num_samples: int,
    max_new_tokens: int,
    cuda_device: int,
    local_files_only: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "run_ruler_eval.py"),
        "--mode",
        "page_attention",
        "--run_dir",
        str(run_dir),
        "--model_name_or_path",
        model_name_or_path,
        "--context_len",
        str(context_len),
        "--tasks",
        ",".join(tasks),
        "--num_samples",
        str(num_samples),
        "--max_new_tokens",
        str(max_new_tokens),
        "--cuda_device",
        str(cuda_device),
        "--dct_page_size",
        str(page_size),
        "--dct_top_k",
        str(top_k),
        "--dct_sink_size",
        "4",
        "--dct_recent_size",
        "128",
        "--dct_compress_ratio",
        str(compress_ratio),
        "--dct_scoring_method",
        "max",
        "--dct_group_agg_method",
        "mean",
        "--dct_unselected_mode",
        "drop",
        "--dct_select_with_oracle_page_scores",
    ]
    if local_files_only:
        cmd.append("--local_files_only")
    return cmd


def collect_page_summary(page_size: int, top_k: int, run_dir: Path) -> list[dict]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as fp:
        summary = json.load(fp)
    comp_size = int(round(page_size * summary["manifest"]["dct"]["compress_ratio"]))
    rows = []
    for row in summary["rows"]:
        rows.append(
            {
                "task": row["task"],
                "page_size": page_size,
                "top_k": top_k,
                "comp_size": comp_size,
                "oracle_score": float(row["score"]),
                "output_jsonl": row["output_jsonl"],
                "run_dir": str(run_dir),
            }
        )
    return rows


def write_summary_files(run_root: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["page_size"], TASKS.index(r["task"])))

    with (run_root / "summary.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "task",
                "page_size",
                "top_k",
                "comp_size",
                "oracle_score",
                "output_jsonl",
                "run_dir",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    avg_rows = []
    for page_size in sorted({r["page_size"] for r in rows}):
        group = [r for r in rows if r["page_size"] == page_size]
        avg_rows.append(
            {
                "page_size": page_size,
                "top_k": group[0]["top_k"],
                "comp_size": group[0]["comp_size"],
                "oracle_score_avg": sum(r["oracle_score"] for r in group) / len(group),
            }
        )

    with (run_root / "summary_avg.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["page_size", "top_k", "comp_size", "oracle_score_avg"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(avg_rows)

    (run_root / "summary.json").write_text(
        json.dumps({"rows": rows, "averages": avg_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    page_sizes = parse_csv_ints(args.page_sizes)
    tasks = resolve_tasks(args.tasks)

    run_root = make_run_root(args.output_root, args.tag)
    manifest = {
        "model_name_or_path": args.model_name_or_path,
        "context_len": args.context_len,
        "page_sizes": page_sizes,
        "selected_token_budget": args.selected_token_budget,
        "compress_ratio": args.compress_ratio,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "cuda_device": args.cuda_device,
        "tasks": tasks,
        "assumption": "Keep selected full-token budget fixed while varying page_size.",
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict] = []
    command_lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for page_size in page_sizes:
        if args.selected_token_budget % page_size != 0:
            raise ValueError(
                f"selected_token_budget={args.selected_token_budget} must be divisible by page_size={page_size}"
            )
        top_k = args.selected_token_budget // page_size
        page_root = run_root / f"ps{page_size}_topk{top_k}"
        cmd = build_run_cmd(
            repo_root=repo_root,
            model_name_or_path=args.model_name_or_path,
            run_dir=page_root,
            context_len=args.context_len,
            tasks=tasks,
            page_size=page_size,
            top_k=top_k,
            compress_ratio=args.compress_ratio,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            cuda_device=args.cuda_device,
            local_files_only=args.local_files_only,
        )
        command_lines.append(" ".join(shlex.quote(part) for part in cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=repo_root, check=True)
        rows.extend(collect_page_summary(page_size, top_k, page_root))

    (run_root / "commands.sh").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Dry run manifest written to: {run_root}")
        return

    write_summary_files(run_root, rows)
    print(f"Results written to: {run_root}")


if __name__ == "__main__":
    main()
