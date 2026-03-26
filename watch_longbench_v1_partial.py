"""Periodically refresh LongBench v1 partial summaries while runs are active."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def refresh(run_dirs: list[Path], summarize_script: Path) -> None:
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        jsonl_files = list(run_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue
        subprocess.run(
            ["python3", str(summarize_script), str(run_dir)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="fix*_full root containing v1_* dirs")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-idle-rounds", type=int, default=3)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    summarize_script = repo_dir / "summarize_longbench_v1.py"
    root = Path(args.root)
    run_dirs = [
        root / "v1_baseline" / "baseline",
        root / "v1_drop" / "drop_ps32_top64_comp1_haar",
        root / "v1_hybrid" / "hybrid_ps32_top64_comp1_haar",
    ]

    idle_rounds = 0
    prev_counts = None
    while True:
        refresh(run_dirs, summarize_script)
        counts = tuple(sum(1 for _ in run_dir.glob("*.jsonl")) for run_dir in run_dirs)
        if counts == prev_counts:
            idle_rounds += 1
        else:
            idle_rounds = 0
        prev_counts = counts

        if idle_rounds >= args.max_idle_rounds:
            refresh(run_dirs, summarize_script)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
