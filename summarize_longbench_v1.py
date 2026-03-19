"""Summarize LongBench v1 result JSONL files.

Prints per-task and overall scores, with optional breakdown by input length.

Usage:
    python summarize_longbench_v1.py results_longbench_v1/
    python summarize_longbench_v1.py results_longbench_v1/llama_page_attn_0.032_topk8_mean_max_compressed_continuous_rope/
    python summarize_longbench_v1.py results_longbench_v1/llama_page_attn_*/  # multiple runs
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_task(records):
    scores = [r["score"] for r in records]
    input_lens = [r["input_len"] for r in records]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Breakdown by input length buckets
    buckets = {"<8K": [], "8K-16K": [], "16K-32K": [], "32K+": []}
    for r in records:
        l = r["input_len"]
        if l < 8192:
            buckets["<8K"].append(r["score"])
        elif l < 16384:
            buckets["8K-16K"].append(r["score"])
        elif l < 32768:
            buckets["16K-32K"].append(r["score"])
        else:
            buckets["32K+"].append(r["score"])

    return {
        "n_samples": len(records),
        "avg_score": avg_score,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "avg_input_len": sum(input_lens) / len(input_lens) if input_lens else 0,
        "buckets": {
            k: {"n": len(v), "avg": sum(v) / len(v) if v else 0.0}
            for k, v in buckets.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize LongBench v1 results")
    parser.add_argument("dirs", nargs="+", help="Result directories (or a parent folder containing run subdirectories)")
    args = parser.parse_args()

    # Resolve dirs: if a single directory is given that contains subdirectories
    # with JSONL files (rather than JSONL files directly), treat it as a parent
    # folder and summarize all subdirectories.
    dirs = []
    for d in args.dirs:
        p = Path(d)
        if p.is_dir() and not list(p.glob("*.jsonl")) and list(p.iterdir()):
            dirs.extend(sorted(sub for sub in p.iterdir() if sub.is_dir()))
        else:
            dirs.append(p)

    for dir_path in dirs:
        if not dir_path.is_dir():
            print(f"Skipping {dir_path} (not a directory)")
            continue

        jsonl_files = sorted(dir_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files in {dir_path}")
            continue

        run_name = dir_path.name
        print(f"\n{'=' * 70}")
        print(f"Run: {run_name}")
        print(f"{'=' * 70}")

        all_scores = []
        task_results = {}

        for f in jsonl_files:
            task = f.stem
            records = load_jsonl(f)
            summary = summarize_task(records)
            task_results[task] = summary
            all_scores.extend([r["score"] for r in records])

        # Per-task table
        print(f"\n  {'Task':<20} {'N':>5} {'Score':>8} {'Avg Len':>10}  Length Breakdown")
        print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*10}  {'-'*30}")

        for task in sorted(task_results):
            s = task_results[task]
            bucket_str = "  ".join(
                f"{k}:{b['avg']:.0%}({b['n']})"
                for k, b in s["buckets"].items()
                if b["n"] > 0
            )
            print(
                f"  {task:<20} {s['n_samples']:>5} {s['avg_score']:>7.1%} "
                f"{s['avg_input_len']:>10,.0f}  {bucket_str}"
            )

        # Overall
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"  {'-'*20} {'-'*5} {'-'*8}")
        print(f"  {'OVERALL':<20} {len(all_scores):>5} {overall:>7.1%}")
        print()


if __name__ == "__main__":
    main()
