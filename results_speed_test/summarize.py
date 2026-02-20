"""
Summarize speed test results from all subdirectories.
Reads each summary.json and prints a comparison table.
Optionally saves to CSV with --csv flag.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent


def load_summaries(results_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        with open(summary_path) as f:
            data = json.load(f)

        variant = summary_path.parent.name

        row = {
            "variant": variant,
            "label": data.get("label", ""),
            "overall_tok_per_s": data.get("overall_decode_tok_per_s"),
            "overall_ms_per_tok": data.get("overall_avg_decode_ms_per_tok"),
            # DCT-specific params (None for baseline)
            "compress_ratio": data.get("compress_ratio"),
            "top_k": data.get("top_k"),
            "scoring_method": data.get("scoring_method"),
            "unselected_mode": data.get("unselected_mode"),
            "page_size": data.get("page_size"),
            "sink_size": data.get("sink_size"),
            "recent_size": data.get("recent_size"),
        }

        # Per-task metrics
        for task, metrics in data.get("per_task", {}).items():
            row[f"{task}_tok_per_s"] = metrics.get("decode_tok_per_s")
            row[f"{task}_ms_per_tok"] = metrics.get("avg_decode_ms_per_tok")
            row[f"{task}_prefill_ms"] = metrics.get("avg_prefill_ms")

        rows.append(row)

    return rows


def find_baseline(rows: list[dict]) -> dict | None:
    for row in rows:
        if row["label"] == "baseline":
            return row
    return None


def print_table(rows: list[dict], baseline: dict | None) -> None:
    tasks = ["gov_report", "multifieldqa_en", "narrativeqa", "qasper"]

    # Sort: baseline first, then by overall tok/s descending
    def sort_key(r):
        return (0 if r["label"] == "baseline" else 1, -(r["overall_tok_per_s"] or 0))

    rows = sorted(rows, key=sort_key)

    # --- Overall summary table ---
    header = f"{'Variant':<55} {'tok/s':>7} {'ms/tok':>8}"
    if baseline:
        header += f"  {'speedup':>8}"
    print("\n=== Overall Decode Speed ===")
    print(header)
    print("-" * len(header))

    baseline_tps = baseline["overall_tok_per_s"] if baseline else None

    for row in rows:
        tps = row["overall_tok_per_s"]
        mpt = row["overall_ms_per_tok"]
        line = f"{row['variant']:<55} {tps:>7.2f} {mpt:>8.3f}"
        if baseline and baseline_tps:
            speedup = (tps / baseline_tps) if tps else float("nan")
            line += f"  {speedup:>7.2f}x"
        print(line)

    # --- Per-task table ---
    for task in tasks:
        print(f"\n=== Per-task: {task} ===")
        col_w = 55
        hdr = f"{'Variant':<{col_w}} {'tok/s':>7} {'ms/tok':>8} {'prefill_ms':>11}"
        baseline_task_tps = baseline.get(f"{task}_tok_per_s") if baseline else None
        if baseline_task_tps:
            hdr += f"  {'speedup':>8}"
        print(hdr)
        print("-" * len(hdr))
        for row in rows:
            tps = row.get(f"{task}_tok_per_s")
            mpt = row.get(f"{task}_ms_per_tok")
            pre = row.get(f"{task}_prefill_ms")
            tps_s = f"{tps:>7.2f}" if tps is not None else f"{'N/A':>7}"
            mpt_s = f"{mpt:>8.3f}" if mpt is not None else f"{'N/A':>8}"
            pre_s = f"{pre:>11.1f}" if pre is not None else f"{'N/A':>11}"
            line = f"{row['variant']:<{col_w}} {tps_s} {mpt_s} {pre_s}"
            if baseline_task_tps:
                speedup = (tps / baseline_task_tps) if tps else float("nan")
                line += f"  {speedup:>7.2f}x"
            print(line)


def save_csv(rows: list[dict], output_path: Path) -> None:
    import csv

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV to: {output_path}")


def main():
    save_to_csv = "--csv" in sys.argv

    rows = load_summaries(RESULTS_DIR)
    if not rows:
        print("No summary.json files found.")
        sys.exit(1)

    baseline = find_baseline(rows)
    print_table(rows, baseline)

    if save_to_csv:
        save_csv(rows, RESULTS_DIR / "summary_all.csv")


if __name__ == "__main__":
    main()