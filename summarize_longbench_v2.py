"""Summarize LongBench v2 result JSONL files.

Prints per-run accuracy with breakdowns by domain, difficulty, and length.

Usage:
    python summarize_longbench_v2.py results_longbench_v2/
    python summarize_longbench_v2.py results_longbench_v2/llama_baseline.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def accuracy(records):
    if not records:
        return 0.0, 0
    correct = sum(1 for r in records if r["correct"])
    return correct / len(records), len(records)


def summarize_run(records):
    # By domain
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    # By difficulty
    by_difficulty = defaultdict(list)
    for r in records:
        by_difficulty[r["difficulty"]].append(r)

    # By length
    by_length = defaultdict(list)
    for r in records:
        by_length[r["length"]].append(r)

    overall_acc, n = accuracy(records)

    return {
        "n_samples": n,
        "overall_acc": overall_acc,
        "by_domain": {k: accuracy(v) for k, v in sorted(by_domain.items())},
        "by_difficulty": {k: accuracy(v) for k, v in sorted(by_difficulty.items())},
        "by_length": {k: accuracy(v) for k, v in sorted(by_length.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize LongBench v2 results")
    parser.add_argument("paths", nargs="+", help="Result JSONL files or a directory containing them")
    args = parser.parse_args()

    # Resolve paths: collect all JSONL files (excluding summary files)
    jsonl_files = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            jsonl_files.extend(sorted(
                f for f in p.glob("*.jsonl") if not f.stem.endswith("_summary")
            ))
        elif p.is_file() and p.suffix == ".jsonl":
            jsonl_files.append(p)
        else:
            print(f"Skipping {p}")

    if not jsonl_files:
        print("No JSONL files found.")
        return

    for jsonl_path in jsonl_files:
        run_name = jsonl_path.stem
        records = load_jsonl(jsonl_path)
        if not records:
            print(f"No records in {jsonl_path}")
            continue

        s = summarize_run(records)

        print(f"\n{'=' * 70}")
        print(f"Run: {run_name}  ({s['n_samples']} samples)")
        print(f"{'=' * 70}")

        # By domain
        print(f"\n  {'Domain':<35} {'N':>5} {'Acc':>8}")
        print(f"  {'-'*35} {'-'*5} {'-'*8}")
        for domain, (acc, n) in s["by_domain"].items():
            print(f"  {domain:<35} {n:>5} {acc:>7.1%}")

        # By difficulty
        print(f"\n  {'Difficulty':<35} {'N':>5} {'Acc':>8}")
        print(f"  {'-'*35} {'-'*5} {'-'*8}")
        for diff, (acc, n) in s["by_difficulty"].items():
            print(f"  {diff:<35} {n:>5} {acc:>7.1%}")

        # By length
        print(f"\n  {'Length':<35} {'N':>5} {'Acc':>8}")
        print(f"  {'-'*35} {'-'*5} {'-'*8}")
        for length, (acc, n) in s["by_length"].items():
            print(f"  {length:<35} {n:>5} {acc:>7.1%}")

        # Overall
        print(f"\n  {'-'*35} {'-'*5} {'-'*8}")
        print(f"  {'OVERALL':<35} {s['n_samples']:>5} {s['overall_acc']:>7.1%}")
        print()


if __name__ == "__main__":
    main()
