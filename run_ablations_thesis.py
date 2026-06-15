#!/usr/bin/env python3
"""
Run RULER ablations for Bachelor's-thesis Chapter 5 (Tables 5.6–5.9) on
Llama-3.1-8B-Instruct, drop mode only, 32K context.

Default config (matches eval_ruler.py CLI defaults; sanity RULER Avg = 86.26):
  P=32, top_k=64 (=S+B+R total, S=1, R=4 → middle B=59), C=4 (compress_ratio=0.125),
  unselected_mode=drop, scoring_method=max, group_agg_method=max,
  attention_backend=upstream_flashinfer, comp_kv_quant=none.

NOTE on --top_k semantics: eval_ruler.py interprets --top_k as the TOTAL page
budget (sink + middle + recent), NOT just middle pages. The user's "B" in the
table headers maps directly to --top_k. With S=1 and R=4 the middle budget is
B - 5.

Subcommands:
  run         — execute the accuracy grid via eval_ruler.py (skip_existing)
  throughput  — execute the page-size throughput sweep for Table 5.6
  print       — parse results and emit the 4 markdown tables to stdout
  dry-run     — print the exact commands that would run, without executing

Output layout under <output_dir>:
  <output_dir>/abl_ps{P}_tk{B}_cr{CR}_sm{SM}_ga{GA}_drop/summary.json
  <output_dir>/throughput/ps{P}_tk{B}.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/ruler" / "ablations"

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEQ_LEN = 32768
NUM_SAMPLES = 25
NUM_SINK_PAGES = 1
NUM_RECENT_PAGES = 4  # matches eval_ruler.py CLI default; user spec S=1, R=4
UNSELECTED_MODE = "drop"

# Defaults that pin one variable while sweeping another.
DEF_P = 32
DEF_B = 64           # --top_k total (sink + middle + recent)
DEF_CR = 0.125       # C = page_size * compress_ratio = 4
DEF_SM = "max"       # scoring_method
DEF_GA = "max"       # group_agg_method


# ---------------------------------------------------------------------------
# Run-name helpers
# ---------------------------------------------------------------------------
def fmt_cr(cr: float) -> str:
    """Format a compress_ratio without trailing zeros (0.125, 0.03125, ...)."""
    s = f"{cr:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def run_name(P: int, B: int, cr: float, sm: str, ga: str) -> str:
    return f"abl_ps{P}_tk{B}_cr{fmt_cr(cr)}_sm{sm}_ga{ga}_drop"


def throughput_log_name(P: int, B: int) -> str:
    return f"ps{P}_tk{B}.log"


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    P: int
    B: int
    cr: float
    sm: str
    ga: str

    def name(self) -> str:
        return run_name(self.P, self.B, self.cr, self.sm, self.ga)


def table_56_configs() -> list[Cfg]:
    """Page size sweep, selected-token budget B*P = 2048 fixed."""
    return [
        Cfg(P=16,  B=128, cr=DEF_CR, sm=DEF_SM, ga=DEF_GA),
        Cfg(P=32,  B=64,  cr=DEF_CR, sm=DEF_SM, ga=DEF_GA),
        Cfg(P=64,  B=32,  cr=DEF_CR, sm=DEF_SM, ga=DEF_GA),
        Cfg(P=128, B=16,  cr=DEF_CR, sm=DEF_SM, ga=DEF_GA),
    ]


def table_57_configs() -> list[Cfg]:
    """Budget sweep with P=32 fixed (budgets 1024, 2048, 4096)."""
    return [
        Cfg(P=DEF_P, B=32,  cr=DEF_CR, sm=DEF_SM, ga=DEF_GA),
        Cfg(P=DEF_P, B=64,  cr=DEF_CR, sm=DEF_SM, ga=DEF_GA),
        Cfg(P=DEF_P, B=128, cr=DEF_CR, sm=DEF_SM, ga=DEF_GA),
    ]


def table_58_configs() -> list[Cfg]:
    """Compression-ratio sweep with P=32, B=64 fixed."""
    return [
        Cfg(P=DEF_P, B=DEF_B, cr=1/32, sm=DEF_SM, ga=DEF_GA),   # C=1
        Cfg(P=DEF_P, B=DEF_B, cr=2/32, sm=DEF_SM, ga=DEF_GA),   # C=2
        Cfg(P=DEF_P, B=DEF_B, cr=4/32, sm=DEF_SM, ga=DEF_GA),   # C=4 (default)
        Cfg(P=DEF_P, B=DEF_B, cr=8/32, sm=DEF_SM, ga=DEF_GA),   # C=8
    ]


def table_59_configs() -> list[Cfg]:
    """Scoring / GQA-group aggregator sweep with P=32, B=64, C=4 fixed."""
    return [
        Cfg(P=DEF_P, B=DEF_B, cr=DEF_CR, sm="max",  ga="max"),
        Cfg(P=DEF_P, B=DEF_B, cr=DEF_CR, sm="max",  ga="mean"),
        Cfg(P=DEF_P, B=DEF_B, cr=DEF_CR, sm="mean", ga="max"),
        Cfg(P=DEF_P, B=DEF_B, cr=DEF_CR, sm="mean", ga="mean"),
    ]


def all_unique_configs() -> list[Cfg]:
    seen: set[Cfg] = set()
    ordered: list[Cfg] = []
    for cfgs in (table_56_configs(), table_57_configs(), table_58_configs(), table_59_configs()):
        for c in cfgs:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
    return ordered


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------
def eval_ruler_cmd(cfg: Cfg, output_dir: Path, num_samples: int, prepare: bool) -> list[str]:
    cmd = [
        sys.executable, str(REPO_ROOT / "eval_ruler.py"),
        "--mode", "page_attention",
        "--base_model", BASE_MODEL,
        "--seq_lengths", str(SEQ_LEN),
        "--num_samples", str(num_samples),
        "--output_dir", str(output_dir),
        "--run_name", cfg.name(),
        "--page_size", str(cfg.P),
        "--top_k", str(cfg.B),
        "--num_sink_pages", str(NUM_SINK_PAGES),
        "--num_recent_pages", str(NUM_RECENT_PAGES),
        "--compress_ratio", f"{cfg.cr}",
        "--scoring_method", cfg.sm,
        "--group_agg_method", cfg.ga,
        "--unselected_mode", UNSELECTED_MODE,
        "--comp_kv_quant", "fp8_e5m2",
        "--comp_kv_quant_granularity", "per_page",
        "--skip_existing",
    ]
    if prepare:
        cmd.append("--prepare")
    return cmd


def throughput_cmd(P: int, B: int, num_decode_steps: int, warmup_steps: int) -> list[str]:
    return [
        sys.executable, str(REPO_ROOT / "speed" / "profile_decode_upstream_flash_infer.py"),
        "--model", BASE_MODEL,
        "--context_length", str(SEQ_LEN),
        "--num_decode_steps", str(num_decode_steps),
        "--warmup_steps", str(warmup_steps),
        "--mode", "dct_upstream_flashinfer",
        "--page_size", str(P),
        "--top_k", str(B),
        "--num_sink_pages", str(NUM_SINK_PAGES),
        "--num_recent_pages", str(NUM_RECENT_PAGES),
        "--compress_ratio", f"{DEF_CR}",
        "--scoring_method", DEF_SM,
        "--group_agg_method", DEF_GA,
        "--unselected_mode", UNSELECTED_MODE,
        "--comp_kv_quant", "none",
    ]


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def child_env() -> dict[str, str]:
    """Prepend the active interpreter's bin dir to PATH so subprocesses (e.g.
    FlashInfer's ninja JIT) find conda-env binaries even when the env is not
    activated in the parent shell."""
    env = os.environ.copy()
    py_bin = str(Path(sys.executable).parent)
    if py_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")
    return env


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------
def load_overall(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    return data.get("overall")


TOK_S_RE = re.compile(r"Model total:\s*[\d.]+\s*ms/step\s*\(([\d.]+)\s*tok/s\)")


def parse_tok_s(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    # Use the last match: print_profile is called once per mode; in our single-mode
    # invocation that's exactly one line, but be defensive about future re-prints.
    matches = TOK_S_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def parse_shard(spec: str | None) -> tuple[int, int]:
    """Parse '--shard i/N' (1-indexed shard out of N). Returns (i-1, N)."""
    if not spec:
        return 0, 1
    try:
        i_str, n_str = spec.split("/")
        i, n = int(i_str), int(n_str)
        assert 1 <= i <= n and n >= 1
    except Exception as e:
        raise SystemExit(f"Invalid --shard {spec!r} (expected 'i/N', 1-indexed): {e}")
    return i - 1, n


def run_accuracy(output_dir: Path, num_samples: int, prepare: bool, dry_run: bool,
                 shard: str | None = None) -> int:
    cfgs_all = all_unique_configs()
    shard_idx, shard_n = parse_shard(shard)
    cfgs = [c for i, c in enumerate(cfgs_all) if i % shard_n == shard_idx]
    shard_tag = f" shard {shard_idx+1}/{shard_n}" if shard_n > 1 else ""
    print(f"# Accuracy sweep:{shard_tag} {len(cfgs)}/{len(cfgs_all)} configurations")
    rc_any = 0
    for i, cfg in enumerate(cfgs, 1):
        summary = output_dir / cfg.name() / "summary.json"
        if summary.exists():
            print(f"[{i}/{len(cfgs)}] SKIP (summary exists): {summary}")
            continue
        cmd = eval_ruler_cmd(cfg, output_dir, num_samples, prepare=(prepare and i == 1))
        print(f"\n[{i}/{len(cfgs)}] {cfg.name()}")
        print(f"  $ {quote_cmd(cmd)}")
        if dry_run:
            continue
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=child_env())
        if rc != 0:
            print(f"  ERROR: exit code {rc}")
            rc_any = rc_any or rc
    return rc_any


def run_throughput(output_dir: Path, num_decode_steps: int, warmup_steps: int,
                   dry_run: bool) -> int:
    log_dir = output_dir / "throughput"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = table_56_configs()
    print(f"# Throughput sweep: {len(rows)} page sizes")
    rc_any = 0
    for i, cfg in enumerate(rows, 1):
        log_path = log_dir / throughput_log_name(cfg.P, cfg.B)
        if log_path.exists() and parse_tok_s(log_path) is not None:
            print(f"[{i}/{len(rows)}] SKIP (log has tok/s): {log_path}")
            continue
        cmd = throughput_cmd(cfg.P, cfg.B, num_decode_steps, warmup_steps)
        print(f"\n[{i}/{len(rows)}] P={cfg.P} top_k={cfg.B}")
        print(f"  $ {quote_cmd(cmd)}")
        print(f"  log: {log_path}")
        if dry_run:
            continue
        with log_path.open("w") as f:
            rc = subprocess.call(cmd, cwd=str(REPO_ROOT), stdout=f,
                                 stderr=subprocess.STDOUT, env=child_env())
        if rc != 0:
            print(f"  ERROR: exit code {rc}; see {log_path}")
            rc_any = rc_any or rc
    return rc_any


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------
def fmt_score(x: float | None) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def fmt_toks(x: float | None) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else "—"


def get_score(output_dir: Path, cfg: Cfg) -> float | None:
    return load_overall(output_dir / cfg.name() / "summary.json")


def emit_tables(output_dir: Path) -> None:
    # --- Sanity check ---
    sanity_cfg = Cfg(P=DEF_P, B=DEF_B, cr=DEF_CR, sm=DEF_SM, ga=DEF_GA)
    sanity = get_score(output_dir, sanity_cfg)
    expected = 86.26
    print(f"<!-- Sanity check: default (P={DEF_P}, top_k={DEF_B}, C=4, max/max) "
          f"RULER Avg = {fmt_score(sanity)}; expected {expected:.2f} -->")
    if sanity is not None and abs(sanity - expected) > 1.0:
        print(f"<!-- WARNING: default RULER Avg deviates from expected {expected:.2f} "
              f"by more than 1.0 pt — check config alignment -->")

    # --- Table 5.6 ---
    rows56 = table_56_configs()
    tps_dir = output_dir / "throughput"
    print()
    print("### Table 5.6 — Page size P sweep (selected-token budget B·P = 2048)")
    print()
    print("| P   | top_k (=B) | RULER Avg | Tok/s @ 32K |")
    print("|----:|-----------:|----------:|------------:|")
    for c in rows56:
        score = get_score(output_dir, c)
        tps = parse_tok_s(tps_dir / throughput_log_name(c.P, c.B))
        print(f"| {c.P:<3} | {c.B:<10} | {fmt_score(score):>9} | {fmt_toks(tps):>11} |")

    # --- Table 5.7 ---
    rows57 = table_57_configs()
    print()
    print("### Table 5.7 — Selected-token budget sweep (P = 32)")
    print()
    print("| Budget tokens | top_k | RULER Avg |")
    print("|--------------:|------:|----------:|")
    for c in rows57:
        budget = c.P * c.B
        score = get_score(output_dir, c)
        print(f"| {budget:<13} | {c.B:<5} | {fmt_score(score):>9} |")

    # --- Table 5.8 ---
    rows58 = table_58_configs()
    print()
    print("### Table 5.8 — Compression-ratio sweep (P = 32, top_k = 64)")
    print()
    print("| C | compress_ratio  | RULER Avg |")
    print("|--:|----------------:|----------:|")
    for c in rows58:
        C = round(c.P * c.cr)
        score = get_score(output_dir, c)
        # Display compress_ratio as a fraction-ish label
        cr_label = f"{c.cr:.5f} (1/{round(1/c.cr)})" if c.cr > 0 else "0"
        print(f"| {C} | {cr_label:>15} | {fmt_score(score):>9} |")

    # --- Table 5.9 ---
    rows59 = table_59_configs()
    print()
    print("### Table 5.9 — Scoring / GQA-group aggregator (P = 32, top_k = 64, C = 4)")
    print()
    print("| scoring_method | group_agg_method | RULER Avg |")
    print("|:---------------|:-----------------|----------:|")
    for c in rows59:
        score = get_score(output_dir, c)
        print(f"| {c.sm:<14} | {c.ga:<16} | {fmt_score(score):>9} |")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subcommand", choices=["run", "throughput", "print", "dry-run", "dry-run-throughput"])
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Root dir for ablation results (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--prepare", action="store_true",
                   help="Pass --prepare to the first eval_ruler.py invocation "
                        "(RULER data prep). Safe to keep on for the first run.")
    p.add_argument("--num_decode_steps", type=int, default=128,
                   help="Throughput: number of timed decode steps (default 128).")
    p.add_argument("--warmup_steps", type=int, default=8,
                   help="Throughput: warmup steps before timing (default 8).")
    p.add_argument("--shard", type=str, default=None,
                   help="For accuracy: process only the i-th of N shards "
                        "(format 'i/N', 1-indexed). Each shard takes every "
                        "Nth config. Use with CUDA_VISIBLE_DEVICES to "
                        "distribute across GPUs. Skip_existing makes shards "
                        "idempotent with respect to each other.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.subcommand == "run":
        return run_accuracy(args.output_dir, args.num_samples, args.prepare,
                            dry_run=False, shard=args.shard)
    if args.subcommand == "throughput":
        return run_throughput(args.output_dir, args.num_decode_steps, args.warmup_steps, dry_run=False)
    if args.subcommand == "dry-run":
        run_accuracy(args.output_dir, args.num_samples, args.prepare,
                     dry_run=True, shard=args.shard)
        return 0
    if args.subcommand == "dry-run-throughput":
        run_throughput(args.output_dir, args.num_decode_steps, args.warmup_steps, dry_run=True)
        return 0
    if args.subcommand == "print":
        emit_tables(args.output_dir)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
