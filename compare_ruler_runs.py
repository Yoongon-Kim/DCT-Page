#!/usr/bin/env python3
"""
Compare multiple flat RULER run directories after the fact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple RULER run directories")
    p.add_argument("--run_dirs", required=True, help="Comma-separated run directories")
    p.add_argument("--labels", default=None, help="Comma-separated labels. Defaults to directory names.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--tag", default="compare")
    return p.parse_args()


def load_summary(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(x.strip()) for x in args.run_dirs.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",")] if args.labels else [p.name for p in run_dirs]
    if len(run_dirs) != len(labels):
        raise ValueError("run_dirs and labels must have the same length")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_rows: dict[str, dict] = {}
    avg_rows: list[dict] = []
    for label, run_dir in zip(labels, run_dirs):
        summary = load_summary(run_dir)
        avg_rows.append(
            {
                "label": label,
                "overall_score_avg": summary["overall_score_avg"],
                "run_dir": str(run_dir),
            }
        )
        for row in summary["rows"]:
            task = row["task"]
            task_rows.setdefault(task, {"task": task})[label] = row["score"]

    row_list = [task_rows[k] for k in sorted(task_rows.keys())]

    with (args.output_dir / f"{args.tag}.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["task", *labels], delimiter="\t")
        writer.writeheader()
        writer.writerows(row_list)

    with (args.output_dir / f"{args.tag}_avg.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["label", "overall_score_avg", "run_dir"], delimiter="\t")
        writer.writeheader()
        writer.writerows(avg_rows)

    (args.output_dir / f"{args.tag}.json").write_text(
        json.dumps({"rows": row_list, "averages": avg_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
