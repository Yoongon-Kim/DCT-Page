#!/usr/bin/env python3
"""Summarize decode-speed benchmark results from results_speed_test_dummy/."""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results_speed_test_dummy"


def load_summaries(results_dir: Path) -> dict:
    """Load all summary.json files, keyed by run name."""
    summaries = {}
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        run_name = summary_path.parent.name
        with open(summary_path) as f:
            summaries[run_name] = json.load(f)
    return summaries


def find_baseline(summaries: dict) -> str | None:
    """Return the key of the baseline run."""
    for name, data in summaries.items():
        if data.get("label") == "baseline":
            return name
    return None


def short_name(run_name: str) -> str:
    """Shorten run name for display."""
    name = run_name.replace("llama_page_attn_dummy_", "dct_")
    name = name.replace("llama_baseline_dummy", "baseline")
    return name


def print_table(headers: list[str], rows: list[list[str]], col_align: list[str] | None = None):
    """Print a formatted table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    if col_align is None:
        col_align = ["<"] + [">"] * (len(headers) - 1)

    def fmt_row(cells):
        parts = []
        for cell, w, a in zip(cells, widths, col_align):
            if a == ">":
                parts.append(cell.rjust(w))
            else:
                parts.append(cell.ljust(w))
        return "  ".join(parts)

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR

    summaries = load_summaries(results_dir)
    if not summaries:
        print(f"No summary.json files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    baseline_key = find_baseline(summaries)
    baseline = summaries.get(baseline_key) if baseline_key else None

    # Collect context lengths from any run
    ctx_lengths = sorted(
        int(k) for k in next(iter(summaries.values()))["per_length"]
    )

    # ── Overall summary ──
    print("=" * 80)
    print("DECODE SPEED SUMMARY  (dummy inputs)")
    print("=" * 80)
    print()

    headers = ["Run"] + [f"{c // 1024}K" for c in ctx_lengths] + ["Overall"]
    rows = []
    for name in summaries:
        data = summaries[name]
        row = [short_name(name)]
        for c in ctx_lengths:
            pl = data["per_length"].get(str(c), {})
            row.append(f"{pl.get('decode_tok_per_s', 0):.1f}")
        row.append(f"{data['decode_tok_per_s']:.1f}")
        rows.append(row)

    print("Decode speed (tok/s):")
    print()
    print_table(headers, rows)

    # ── Speedup vs baseline ──
    if baseline:
        print()
        print("-" * 80)
        print()
        print(f"Speedup vs baseline ({short_name(baseline_key)}):")
        print()

        headers_sp = ["Run"] + [f"{c // 1024}K" for c in ctx_lengths] + ["Overall"]
        rows_sp = []
        for name in summaries:
            if name == baseline_key:
                continue
            data = summaries[name]
            row = [short_name(name)]
            for c in ctx_lengths:
                bl_tok = baseline["per_length"][str(c)]["decode_tok_per_s"]
                dc_tok = data["per_length"].get(str(c), {}).get("decode_tok_per_s", 0)
                ratio = dc_tok / bl_tok if bl_tok else 0
                row.append(f"{ratio:.2f}x")
            bl_overall = baseline["decode_tok_per_s"]
            dc_overall = data["decode_tok_per_s"]
            row.append(f"{dc_overall / bl_overall:.2f}x" if bl_overall else "N/A")
            rows_sp.append(row)

        print_table(headers_sp, rows_sp)

    # ── Prefill latency ──
    print()
    print("-" * 80)
    print()
    print("Prefill latency (ms):")
    print()

    headers_pf = ["Run"] + [f"{c // 1024}K" for c in ctx_lengths] + ["Overall"]
    rows_pf = []
    for name in summaries:
        data = summaries[name]
        row = [short_name(name)]
        for c in ctx_lengths:
            pl = data["per_length"].get(str(c), {})
            row.append(f"{pl.get('avg_prefill_ms', 0):.0f}")
        row.append(f"{data['avg_prefill_ms']:.0f}")
        rows_pf.append(row)

    print_table(headers_pf, rows_pf)

    # ── DCT config details ──
    dct_runs = {k: v for k, v in summaries.items() if v.get("label") != "baseline"}
    if dct_runs:
        print()
        print("-" * 80)
        print()
        print("DCT configurations:")
        print()
        cfg_headers = ["Run", "page", "top_k", "sink", "recent", "compress", "scoring", "unselected"]
        cfg_rows = []
        for name, data in dct_runs.items():
            cfg_rows.append([
                short_name(name),
                str(data.get("page_size", "")),
                str(data.get("top_k", "")),
                str(data.get("sink_size", "")),
                str(data.get("recent_size", "")),
                str(data.get("compress_ratio", "")),
                data.get("scoring_method", ""),
                data.get("unselected_mode", ""),
            ])
        print_table(cfg_headers, cfg_rows)

    print()
    print("=" * 80)
    print(f"Source: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
