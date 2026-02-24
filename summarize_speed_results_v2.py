#!/usr/bin/env python3
"""Summarize LongBench v2 JSONL result files matching a glob pattern."""

import json
import sys
import glob
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results_longbench"


def summarize_file(path: str) -> None:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    if total == 0:
        print(f"  (empty file: {path})")
        return

    failed = sum(1 for r in records if r.get("predicted") is None or r.get("predicted") == "")

    def stats(subset):
        n = len(subset)
        correct = sum(1 for r in subset if r.get("correct"))
        return correct, n

    overall_c, overall_n = stats(records)
    easy_c,   easy_n   = stats([r for r in records if r.get("difficulty") == "easy"])
    hard_c,   hard_n   = stats([r for r in records if r.get("difficulty") == "hard"])
    short_c,  short_n  = stats([r for r in records if r.get("length") == "short"])
    medium_c, medium_n = stats([r for r in records if r.get("length") == "medium"])
    long_c,   long_n   = stats([r for r in records if r.get("length") == "long"])

    def pct(c, n):
        return f"{100 * c / n:.1f}%" if n else "N/A"

    name = Path(path).stem
    width = max(60, len(name) + 4)
    sep = "=" * width

    print(sep)
    print(f"LongBench v2 Results — {name}")
    print(sep)
    print(f"  Overall : {pct(overall_c, overall_n):>6}  ({overall_c}/{overall_n})")
    print(f"  Easy    : {pct(easy_c,    easy_n):>6}  ({easy_c}/{easy_n})")
    print(f"  Hard    : {pct(hard_c,    hard_n):>6}  ({hard_c}/{hard_n})")
    print(f"  Short   : {pct(short_c,   short_n):>6}  ({short_c}/{short_n})")
    print(f"  Medium  : {pct(medium_c,  medium_n):>6}  ({medium_c}/{medium_n})")
    print(f"  Long    : {pct(long_c,    long_n):>6}  ({long_c}/{long_n})")
    print(f"  Failed answer extraction: {failed}/{total}")
    print(sep)
    print()


def main():
    # Args are filename glob patterns applied inside RESULTS_DIR (no path needed)
    # Default: all llama_page_attn_0.03_topk32 files
    name_patterns = sys.argv[1:] if len(sys.argv) > 1 else ["llama_page_attn_0.03_topk8*.jsonl"]

    files = []
    for pattern in name_patterns:
        matched = sorted(glob.glob(str(RESULTS_DIR / pattern)))
        if not matched:
            print(f"Warning: no files matched: {RESULTS_DIR / pattern}", file=sys.stderr)
        files.extend(matched)

    if not files:
        print("No files found. Usage:", file=sys.stderr)
        print("  python summarize_results.py [filename_pattern ...]", file=sys.stderr)
        print("  python summarize_results.py llama_page_attn_0.03_topk32*.jsonl", file=sys.stderr)
        sys.exit(1)

    for path in files:
        summarize_file(path)


if __name__ == "__main__":
    main()