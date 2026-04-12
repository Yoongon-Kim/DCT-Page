#!/usr/bin/env python3
"""
Evaluate dc_ac scoring (full DCT spectrum) on RULER benchmark.

Sweeps over lambda values and optionally (page_size, top_k) combos.
Each combination is run as a subprocess via oracle/run_ruler_eval.py.

Unlike proxy_dc_ac (which uses only comp_size DCT coefficients, typically 1),
dc_ac uses ALL page_size coefficients for scoring:
    score = DC + lambda * sqrt(sum(AC[1:]^2))
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
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
    p = argparse.ArgumentParser(
        description="Run RULER evaluation with full dc_ac scoring (all DCT coefficients)"
    )
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("benchmark/data/ruler_data"),
        help="Root dir containing <model_family>/<context_len>/<task>/validation.jsonl. "
        "Forwarded to run_ruler_eval.py as --data_root.",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/results_ruler/dc_ac"),
    )
    p.add_argument(
        "--lambdas",
        nargs="+",
        default=["0.25", "0.5", "1.0", "2.0", "4.0"],
        help="Lambda values to sweep for dc_ac scoring.",
    )
    p.add_argument(
        "--combos",
        nargs="+",
        default=["32,64"],
        help="Space-separated page_size,top_k pairs. E.g. '32,64 16,128'.",
    )
    p.add_argument(
        "--group_agg_method",
        type=str,
        default="max",
        choices=["mean", "max", "topp"],
    )
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def parse_combos(values: list[str]) -> list[tuple[int, int]]:
    combos = []
    for v in values:
        parts = v.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Each combo must be page_size,top_k -- got {v!r}")
        combos.append((int(parts[0]), int(parts[1])))
    return combos


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


def make_run_dir(output_root: Path, page_size: int, top_k: int, lam: str) -> Path:
    run_dir = output_root / f"ps{page_size}_topk{top_k}_dc_ac_{lam}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_cmd(
    script_path: Path,
    model_name_or_path: str,
    run_dir: Path,
    context_len: int,
    tasks: list[str],
    page_size: int,
    top_k: int,
    lam: str,
    group_agg_method: str,
    data_root: Path,
    num_samples: int,
    max_new_tokens: int,
    cuda_device: int,
    local_files_only: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--mode", "page_attention",
        "--run_dir", str(run_dir),
        "--data_root", str(data_root),
        "--model_name_or_path", model_name_or_path,
        "--context_len", str(context_len),
        "--tasks", ",".join(tasks),
        "--num_samples", str(num_samples),
        "--max_new_tokens", str(max_new_tokens),
        "--cuda_device", str(cuda_device),
        "--dct_page_size", str(page_size),
        "--dct_top_k", str(top_k),
        "--dct_sink_size", "4",
        "--dct_recent_size", "128",
        "--dct_scoring_method", f"dc_ac_{lam}",
        "--dct_group_agg_method", group_agg_method,
        "--dct_unselected_mode", "drop",
    ]
    if local_files_only:
        cmd.append("--local_files_only")
    return cmd


def collect_run_summary(
    page_size: int, top_k: int, lam: str, run_dir: Path,
) -> list[dict]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as fp:
        summary = json.load(fp)
    rows = []
    for row in summary["rows"]:
        rows.append({
            "task": row["task"],
            "page_size": page_size,
            "top_k": top_k,
            "lambda": lam,
            "score": float(row["score"]),
            "output_jsonl": row["output_jsonl"],
            "run_dir": str(run_dir),
        })
    return rows


def write_summary_files(output_root: Path, all_rows: list[dict]) -> None:
    all_rows = sorted(
        all_rows,
        key=lambda r: (r["page_size"], float(r["lambda"]), TASKS.index(r["task"])),
    )

    with (output_root / "summary.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["task", "page_size", "top_k", "lambda", "score", "output_jsonl", "run_dir"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    avg_rows = []
    seen = set()
    for r in all_rows:
        key = (r["page_size"], r["top_k"], r["lambda"])
        if key in seen:
            continue
        seen.add(key)
        group = [x for x in all_rows if (x["page_size"], x["top_k"], x["lambda"]) == key]
        avg_rows.append({
            "page_size": r["page_size"],
            "top_k": r["top_k"],
            "lambda": r["lambda"],
            "score_avg": sum(x["score"] for x in group) / len(group),
        })

    with (output_root / "summary_avg.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["page_size", "top_k", "lambda", "score_avg"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(avg_rows)

    (output_root / "summary.json").write_text(
        json.dumps({"rows": all_rows, "averages": avg_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent / "run_ruler_eval.py"
    repo_root = Path(__file__).resolve().parent.parent
    combos = parse_combos(args.combos)
    tasks = resolve_tasks(args.tasks)

    all_rows: list[dict] = []

    for page_size, top_k in combos:
        for lam in args.lambdas:
            run_dir = make_run_dir(args.output_root, page_size, top_k, lam)

            if (run_dir / "summary.json").exists():
                print(
                    f"\n[dc_ac] SKIP page_size={page_size} top_k={top_k} lambda={lam} "
                    f"-- summary already exists in {run_dir}",
                    flush=True,
                )
                all_rows.extend(collect_run_summary(page_size, top_k, lam, run_dir))
                continue

            manifest = {
                "model_name_or_path": args.model_name_or_path,
                "context_len": args.context_len,
                "page_size": page_size,
                "top_k": top_k,
                "lambda": lam,
                "scoring_method": f"dc_ac_{lam}",
                "group_agg_method": args.group_agg_method,
                "unselected_mode": "drop",
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "cuda_device": args.cuda_device,
                "tasks": tasks,
                "note": "Full dc_ac scoring: uses all page_size DCT coefficients, "
                "not just comp_size proxy coefficients.",
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            cmd = build_run_cmd(
                script_path=script_path,
                model_name_or_path=args.model_name_or_path,
                run_dir=run_dir,
                context_len=args.context_len,
                tasks=tasks,
                page_size=page_size,
                top_k=top_k,
                lam=lam,
                group_agg_method=args.group_agg_method,
                data_root=args.data_root,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                cuda_device=args.cuda_device,
                local_files_only=args.local_files_only,
            )
            (run_dir / "command.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n\n"
                + " ".join(shlex.quote(part) for part in cmd)
                + "\n",
                encoding="utf-8",
            )
            print(
                f"\n[dc_ac] page_size={page_size} top_k={top_k} lambda={lam} -> {run_dir}",
                flush=True,
            )
            print("  cmd: " + " ".join(shlex.quote(part) for part in cmd), flush=True)
            if args.dry_run:
                print(f"  Dry run: skipping execution", flush=True)
                continue
            subprocess.run(cmd, cwd=repo_root, check=True)
            all_rows.extend(collect_run_summary(page_size, top_k, lam, run_dir))

    if all_rows:
        write_summary_files(args.output_root, all_rows)
        print(f"\nCross-lambda summary written to: {args.output_root}")

    print("\n[dc_ac] All runs finished.", flush=True)


if __name__ == "__main__":
    main()
