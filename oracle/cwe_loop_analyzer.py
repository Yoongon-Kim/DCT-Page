"""Quantify generation loop / repetition in cwe predictions.

For each cwe sample, parses the prediction into the 10 numbered slots (e.g.
``1. cannon 2. ...``), counts:
  - run_len_max: longest streak of identical consecutive slots
  - distinct_words: distinct tokens across the 10 slots
  - looped: run_len_max >= 3 (heuristic)
  - hits: how many of the 10 GT words appear in the prediction slots

Then compares against the official score (under the same per-sample order).
Usage:
    python oracle/cwe_loop_analyzer.py <pred1.jsonl> [<pred2.jsonl> ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


WORD_PATTERN = re.compile(r"\s*(\d+)\s*[.):]?\s*\*?\*?([\w\-']+)\*?\*?")


def _extract_slots(pred: str) -> list[str]:
    """Best-effort: pull out the 1..10 slot words from a numbered list response."""
    slots: dict[int, str] = {}
    for m in WORD_PATTERN.finditer(pred):
        idx = int(m.group(1))
        word = m.group(2).lower()
        if 1 <= idx <= 50 and idx not in slots:
            slots[idx] = word
        if len(slots) >= 10:
            break
    ordered = [slots[i] for i in sorted(slots) if i <= 10]
    return ordered


def _max_run(words: list[str]) -> int:
    if not words:
        return 0
    best = cur = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _hits(pred_words: list[str], gt: list[str]) -> int:
    gt_set = {g.lower() for g in gt}
    return sum(1 for w in pred_words if w in gt_set)


def analyze(path: Path) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pred = d.get("prediction") or d.get("pred") or ""
        gt = d.get("outputs") or d.get("gt") or d.get("answer") or []
        if isinstance(gt, str):
            gt = [gt]
        slots = _extract_slots(pred)
        run = _max_run(slots)
        rows.append({
            "n_slots": len(slots),
            "distinct": len(set(slots)),
            "run_max": run,
            "hits": _hits(slots, gt),
            "looped": run >= 3,
        })
    n = len(rows) or 1
    return {
        "path": str(path),
        "n_samples": len(rows),
        "looped_count": sum(r["looped"] for r in rows),
        "looped_frac": sum(r["looped"] for r in rows) / n,
        "run_max_mean": sum(r["run_max"] for r in rows) / n,
        "distinct_mean": sum(r["distinct"] for r in rows) / n,
        "hits_mean": sum(r["hits"] for r in rows) / n,
        "hits_std_dev": (sum((r["hits"] - sum(rr["hits"] for rr in rows)/n) ** 2 for r in rows) / n) ** 0.5,
        "per_sample": rows,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(f"{'config':<60s} {'N':>3s} {'loop%':>6s} {'rmax':>5s} {'distinct':>8s} {'hits':>5s}")
    print("-" * 100)
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"  ! missing: {p}")
            continue
        a = analyze(p)
        label = p.parent.parent.parent.parent.name  # run_name
        print(
            f"{label:<60s} {a['n_samples']:>3d} "
            f"{a['looped_frac']*100:>5.1f}% "
            f"{a['run_max_mean']:>5.2f} "
            f"{a['distinct_mean']:>8.2f} "
            f"{a['hits_mean']:>5.2f}"
        )


if __name__ == "__main__":
    main()
