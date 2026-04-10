#!/usr/bin/env python3
"""
Run RULER oracle-selection + hybrid unselected-mode sweeps and summarize results.

Uses oracle page scores for selection, but keeps unselected pages as
Haar lowpass proxy KV cache (hybrid mode) instead of dropping them.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RULER oracle+hybrid sweeps")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--data_root", type=Path, default=Path("benchmark/data/ruler_data"),
                   help="Root dir containing <model_family>/<context_len>/<task>/validation.jsonl. "
                        "Forwarded to run_ruler_eval.py as --data_root.")
    p.add_argument("--output_root", type=Path, default=Path("results/results_ruler/oracle_hybrid"))
    p.add_argument("--tag", default="oracle_hybrid")
    p.add_argument("--combos", nargs="+", default=["16,8", "32,4", "64,2"],
                   help="Space-separated page_size,top_k pairs. "
                        "E.g. '16,128 32,64 64,32'.")
    p.add_argument("--compress_ratio", type=float, default=0.0625)
    p.add_argument(
        "--weight_pop",
        default="0,1",
        help="Comma-separated 0/1 values to sweep --dct_weight_compressed_by_population. "
             "E.g. '0', '1', or '0,1' to run both.",
    )
    p.add_argument("--compression_method", type=str, default="dct", choices=["haar", "dct"],
                   help="Compression method for unselected pages: 'haar' (default) or 'dct'.")
    p.add_argument("--scoring_method", type=str, default="max",
                   help="'mean'|'max'|'sum'|'proxy_dc_ac_{lam}'|'spread_dc_ac_{lam}'")
    p.add_argument("--group_agg_method", type=str, default="max", choices=["mean", "max", "topp"],
                   help="Cross-GQA-head aggregation: 'mean' (default), 'max', or 'topp'.")
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def parse_combos(values: list[str]) -> list[tuple[int, int]]:
    """Parse 'page_size,top_k' pairs, e.g. ['16,128', '32,64', '64,32']."""
    combos = []
    for v in values:
        parts = v.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Each combo must be page_size,top_k — got {v!r}")
        combos.append((int(parts[0]), int(parts[1])))
    return combos


def parse_weight_pop(value: str) -> list[bool]:
    out = []
    for x in value.split(","):
        x = x.strip()
        if not x:
            continue
        if x not in ("0", "1"):
            raise ValueError(f"--weight_pop entries must be 0 or 1, got {x!r}")
        out.append(x == "1")
    if not out:
        raise ValueError("--weight_pop must contain at least one of 0,1")
    return out


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


def make_run_root(
    output_root: Path,
    page_size: int,
    top_k: int,
    compress_ratio: float,
    compression_method: str,
    weight_pop: bool,
) -> Path:
    pop_tag = "popw" if weight_pop else "nopopw"
    run_root = output_root / f"ps{page_size}_topk{top_k}_cr{compress_ratio}_{compression_method}_{pop_tag}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def build_run_cmd(
    script_path: Path,
    model_name_or_path: str,
    run_dir: Path,
    context_len: int,
    tasks: list[str],
    page_size: int,
    top_k: int,
    compress_ratio: float,
    compression_method: str,
    scoring_method: str,
    group_agg_method: str,
    weight_pop: bool,
    data_root: Path,
    num_samples: int,
    max_new_tokens: int,
    cuda_device: int,
    local_files_only: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        "page_attention",
        "--run_dir",
        str(run_dir),
        "--data_root",
        str(data_root),
        "--model_name_or_path",
        model_name_or_path,
        "--context_len",
        str(context_len),
        "--tasks",
        ",".join(tasks),
        "--num_samples",
        str(num_samples),
        "--max_new_tokens",
        str(max_new_tokens),
        "--cuda_device",
        str(cuda_device),
        "--dct_page_size",
        str(page_size),
        "--dct_top_k",
        str(top_k),
        "--dct_sink_size",
        "4",
        "--dct_recent_size",
        "128",
        "--dct_compress_ratio",
        str(compress_ratio),
        "--dct_scoring_method",
        scoring_method,
        "--dct_group_agg_method",
        group_agg_method,
        "--dct_unselected_mode",
        "compressed",
        "--dct_compression_method",
        compression_method,
        *(["--dct_score_use_low_proxy"] if compression_method == "dct" else []),
        "--dct_select_with_oracle_page_scores",
    ]
    if weight_pop:
        cmd.append("--dct_weight_compressed_by_population")
    if local_files_only:
        cmd.append("--local_files_only")
    return cmd


def collect_page_summary(page_size: int, top_k: int, run_dir: Path) -> list[dict]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as fp:
        summary = json.load(fp)
    comp_size = int(round(page_size * summary["manifest"]["dct"]["compress_ratio"]))
    rows = []
    for row in summary["rows"]:
        rows.append(
            {
                "task": row["task"],
                "page_size": page_size,
                "top_k": top_k,
                "comp_size": comp_size,
                "score": float(row["score"]),
                "output_jsonl": row["output_jsonl"],
                "run_dir": str(run_dir),
            }
        )
    return rows


def write_summary_files(run_root: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["page_size"], TASKS.index(r["task"])))

    with (run_root / "summary.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "task",
                "page_size",
                "top_k",
                "comp_size",
                "score",
                "output_jsonl",
                "run_dir",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    avg_rows = []
    for page_size in sorted({r["page_size"] for r in rows}):
        group = [r for r in rows if r["page_size"] == page_size]
        avg_rows.append(
            {
                "page_size": page_size,
                "top_k": group[0]["top_k"],
                "comp_size": group[0]["comp_size"],
                "score_avg": sum(r["score"] for r in group) / len(group),
            }
        )

    with (run_root / "summary_avg.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["page_size", "top_k", "comp_size", "score_avg"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(avg_rows)

    (run_root / "summary.json").write_text(
        json.dumps({"rows": rows, "averages": avg_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    # Sibling script lives next to us in oracle/, but the child run_ruler_eval.py
    # expects to be invoked with cwd at the project root so its relative data
    # paths resolve correctly.
    script_path = Path(__file__).resolve().parent / "run_ruler_eval.py"
    repo_root = Path(__file__).resolve().parent.parent
    combos = parse_combos(args.combos)
    weight_pops = parse_weight_pop(args.weight_pop)
    tasks = resolve_tasks(args.tasks)

    for weight_pop in weight_pops:
        for page_size, top_k in combos:
            run_root = make_run_root(
                args.output_root, page_size, top_k, args.compress_ratio, args.compression_method, weight_pop
            )

            if (run_root / "summary.json").exists():
                print(
                    f"\n[oracle_hybrid] SKIP page_size={page_size} top_k={top_k} "
                    f"cr={args.compress_ratio} weight_pop={int(weight_pop)} "
                    f"— summary already exists in {run_root}",
                    flush=True,
                )
                continue

            manifest = {
                "model_name_or_path": args.model_name_or_path,
                "context_len": args.context_len,
                "page_size": page_size,
                "top_k": top_k,
                "compress_ratio": args.compress_ratio,
                "weight_compressed_by_population": weight_pop,
                "compression_method": args.compression_method,
                "scoring_method": args.scoring_method,
                "group_agg_method": args.group_agg_method,
                "unselected_mode": "compressed",
                "select_with_oracle_page_scores": True,
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "cuda_device": args.cuda_device,
                "tasks": tasks,
                "assumption": "Oracle page selection + hybrid mode: selected pages use full tokens, "
                f"unselected pages use {args.compression_method} compressed proxy KV cache.",
            }
            (run_root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            cmd = build_run_cmd(
                script_path=script_path,
                model_name_or_path=args.model_name_or_path,
                run_dir=run_root,
                context_len=args.context_len,
                tasks=tasks,
                page_size=page_size,
                top_k=top_k,
                compress_ratio=args.compress_ratio,
                compression_method=args.compression_method,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                weight_pop=weight_pop,
                data_root=args.data_root,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                cuda_device=args.cuda_device,
                local_files_only=args.local_files_only,
            )
            (run_root / "command.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n\n"
                + " ".join(shlex.quote(part) for part in cmd)
                + "\n",
                encoding="utf-8",
            )
            print(
                f"\n[oracle_hybrid] page_size={page_size} top_k={top_k} "
                f"cr={args.compress_ratio} weight_pop={int(weight_pop)} -> {run_root}",
                flush=True,
            )
            print("  cmd: " + " ".join(shlex.quote(part) for part in cmd), flush=True)
            if args.dry_run:
                print(f"Dry run: {run_root}", flush=True)
                continue
            subprocess.run(cmd, cwd=repo_root, check=True)
            rows = collect_page_summary(page_size, top_k, run_root)
            write_summary_files(run_root, rows)
            print(f"Results written to: {run_root}")


if __name__ == "__main__":
    main()