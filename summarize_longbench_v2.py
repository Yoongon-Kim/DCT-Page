"""Summarize LongBench v2 result JSONL files.

Prints per-run accuracy with breakdowns by domain, difficulty, and length, and
can also write machine-readable summary files next to each JSONL.

Usage:
    python summarize_longbench_v2.py results/longbench_v2/
    python summarize_longbench_v2.py results/longbench_v2/llama_baseline.jsonl
"""

import argparse
import csv
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


def to_detailed_summary_dict(run_name, records, base_summary=None):
    s = summarize_run(records)
    detailed = dict(base_summary or {})
    detailed.update(
        {
            "run_name": run_name,
            "num_samples": s["n_samples"],
            "overall_accuracy": round(s["overall_acc"] * 100, 2),
            "by_difficulty": {
                k: {"accuracy": round(acc * 100, 2), "num_samples": n}
                for k, (acc, n) in s["by_difficulty"].items()
            },
            "by_length": {
                k: {"accuracy": round(acc * 100, 2), "num_samples": n}
                for k, (acc, n) in s["by_length"].items()
            },
            "by_domain": {
                k: {"accuracy": round(acc * 100, 2), "num_samples": n}
                for k, (acc, n) in s["by_domain"].items()
            },
        }
    )
    return detailed


def write_summary_files(jsonl_path, detailed_summary):
    summary_json_path = jsonl_path.with_name(f"{jsonl_path.stem}_summary.json")
    summary_csv_path = jsonl_path.with_name(f"{jsonl_path.stem}_summary.csv")

    summary_json_path.write_text(
        json.dumps(detailed_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = [
        {
            "group": "overall",
            "label": "overall",
            "accuracy": detailed_summary["overall_accuracy"],
            "num_samples": detailed_summary["num_samples"],
        }
    ]
    for group_name in ("by_difficulty", "by_length", "by_domain"):
        for label, payload in detailed_summary[group_name].items():
            rows.append(
                {
                    "group": group_name.removeprefix("by_"),
                    "label": label,
                    "accuracy": payload["accuracy"],
                    "num_samples": payload["num_samples"],
                }
            )

    with summary_csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["group", "label", "accuracy", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)

    return summary_json_path, summary_csv_path


def main():
    parser = argparse.ArgumentParser(description="Summarize LongBench v2 results")
    parser.add_argument("paths", nargs="+", help="Result JSONL files or a directory containing them")
    parser.add_argument("--write", action="store_true",
                        help="Write <run>_summary.json and <run>_summary.csv next to each JSONL")
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

        base_summary = {}
        summary_json_path = jsonl_path.with_name(f"{run_name}_summary.json")
        if summary_json_path.exists():
            with summary_json_path.open("r", encoding="utf-8") as fp:
                base_summary = json.load(fp)

        s = summarize_run(records)
        detailed = to_detailed_summary_dict(run_name, records, base_summary=base_summary)

        if args.write:
            out_json, out_csv = write_summary_files(jsonl_path, detailed)
            print(f"Wrote {out_json}")
            print(f"Wrote {out_csv}")

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
