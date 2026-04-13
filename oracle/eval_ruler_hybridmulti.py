#!/usr/bin/env python3
"""
Evaluate hybrid_multi scoring on RULER benchmark.

Sweeps over M (number of AC-energy highlight tokens) and alpha (spectral
scaling factor), and optionally (page_size, top_k) combos.
Each combination is run as a subprocess via oracle/run_ruler_eval.py.

The hybrid_multi{M}_ac_max_a{alpha} scoring method allocates the scoring
budget as (page_size - M) DCT coefficients + M AC-energy-selected tokens:
    spectral_max = max_t( sum_{j=1..c_multi} a_j * phi_j(t) )
    multi_hi     = max_m( Q . K_m )   for top-M AC-energy tokens
    page_score   = max(alpha * spectral_max, multi_hi)

Modes:
  - drop (default): unselected pages are discarded entirely.
  - compressed (hybrid): unselected pages keep DCT-IDCT compressed KV tokens,
    with optional population weighting (--weight_pop).
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
        description="Run RULER evaluation with hybrid_multi scoring "
        "(M AC-energy tokens + spectral reconstruction)"
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
        default=Path("results/results_ruler/hybrid_multi"),
    )
    p.add_argument(
        "--m_values",
        nargs="+",
        default=["1", "2", "3", "4"],
        help="M values (number of highlight tokens) to sweep.",
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        default=["0.25", "0.5", "1.0", "2.0", "4.0"],
        help="Alpha (spectral scaling) values to sweep.",
    )
    p.add_argument(
        "--combos",
        nargs="+",
        default=["32,64"],
        help="Space-separated page_size,top_k pairs. E.g. '32,64 16,128'.",
    )
    p.add_argument(
        "--unselected_mode",
        type=str,
        default="drop",
        choices=["drop", "compressed"],
        help="'drop' discards unselected pages; 'compressed' (hybrid) keeps "
        "DCT-IDCT compressed KV for unselected pages.",
    )
    p.add_argument(
        "--compress_ratio",
        type=float,
        default=0.0625,
        help="Compression ratio for unselected pages in compressed mode.",
    )
    p.add_argument(
        "--compression_method",
        type=str,
        default="dct",
        choices=["haar", "dct"],
        help="Compression method for unselected pages in compressed mode.",
    )
    p.add_argument(
        "--weight_pop",
        default="1",
        help="Comma-separated 0/1 values to sweep --dct_weight_compressed_by_population. "
        "E.g. '0', '1', or '0,1' to run both. Only effective in compressed mode.",
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


def make_run_dir(
    output_root: Path,
    page_size: int,
    top_k: int,
    M: str,
    alpha: str,
    unselected_mode: str,
    compress_ratio: float = 0.0,
    compression_method: str = "",
    weight_pop: bool = False,
) -> Path:
    name = f"ps{page_size}_topk{top_k}_hybrid_multi{M}_ac_max_a{alpha}_cr{compress_ratio}"
    if unselected_mode == "compressed":
        pop_tag = "popw" if weight_pop else "nopopw"
        name += f"_{compression_method}_{pop_tag}"
    run_dir = output_root / name
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
    M: str,
    alpha: str,
    group_agg_method: str,
    unselected_mode: str,
    compress_ratio: float,
    compression_method: str,
    weight_pop: bool,
    data_root: Path,
    num_samples: int,
    max_new_tokens: int,
    cuda_device: int,
    local_files_only: bool,
) -> list[str]:
    scoring_method = f"hybrid_multi{M}_ac_max_a{alpha}"
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
        "--dct_scoring_method", scoring_method,
        "--dct_group_agg_method", group_agg_method,
        "--dct_unselected_mode", unselected_mode,
        "--dct_compress_ratio", str(compress_ratio),
        "--dct_no_select_with_oracle_page_scores",
    ]
    if unselected_mode == "compressed":
        cmd.extend([
            "--dct_compression_method", compression_method,
        ])
        if compression_method == "dct":
            cmd.append("--dct_score_use_low_proxy")
        if weight_pop:
            cmd.append("--dct_weight_compressed_by_population")
    if local_files_only:
        cmd.append("--local_files_only")
    return cmd


def collect_run_summary(
    page_size: int,
    top_k: int,
    M: str,
    alpha: str,
    run_dir: Path,
    unselected_mode: str,
    compress_ratio: float,
    compression_method: str,
    weight_pop: bool,
) -> list[dict]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as fp:
        summary = json.load(fp)
    rows = []
    for row in summary["rows"]:
        entry = {
            "task": row["task"],
            "page_size": page_size,
            "top_k": top_k,
            "M": M,
            "alpha": alpha,
            "compress_ratio": compress_ratio,
            "unselected_mode": unselected_mode,
            "score": float(row["score"]),
            "output_jsonl": row["output_jsonl"],
            "run_dir": str(run_dir),
        }
        if unselected_mode == "compressed":
            entry["compression_method"] = compression_method
            entry["weight_pop"] = int(weight_pop)
        rows.append(entry)
    return rows


def write_summary_files(output_root: Path, all_rows: list[dict]) -> None:
    all_rows = sorted(
        all_rows,
        key=lambda r: (
            r["page_size"],
            r["unselected_mode"],
            r.get("weight_pop", 0),
            float(r["compress_ratio"]),
            int(r["M"]),
            float(r["alpha"]),
            TASKS.index(r["task"]),
        ),
    )

    fieldnames = [
        "task", "page_size", "top_k", "M", "alpha", "unselected_mode",
        "compress_ratio", "compression_method", "weight_pop",
        "score", "output_jsonl", "run_dir",
    ]
    with (output_root / "summary.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    avg_rows = []
    seen = set()
    for r in all_rows:
        key = (r["page_size"], r["top_k"], r["M"], r["alpha"],
               r["compress_ratio"], r["unselected_mode"], r.get("weight_pop", 0))
        if key in seen:
            continue
        seen.add(key)
        group = [
            x for x in all_rows
            if (x["page_size"], x["top_k"], x["M"], x["alpha"],
                x["compress_ratio"], x["unselected_mode"], x.get("weight_pop", 0)) == key
        ]
        entry = {
            "page_size": r["page_size"],
            "top_k": r["top_k"],
            "M": r["M"],
            "alpha": r["alpha"],
            "compress_ratio": r["compress_ratio"],
            "unselected_mode": r["unselected_mode"],
            "score_avg": sum(x["score"] for x in group) / len(group),
        }
        if r["unselected_mode"] == "compressed":
            entry["compression_method"] = r.get("compression_method")
            entry["weight_pop"] = r.get("weight_pop")
        avg_rows.append(entry)

    avg_fieldnames = [
        "page_size", "top_k", "M", "alpha", "unselected_mode",
        "compress_ratio", "compression_method", "weight_pop", "score_avg",
    ]
    with (output_root / "summary_avg.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=avg_fieldnames, delimiter="\t",
                                extrasaction="ignore")
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
    weight_pops = parse_weight_pop(args.weight_pop)

    # In drop mode, weight_pop is irrelevant — run once with False.
    if args.unselected_mode == "drop":
        weight_pops = [False]

    all_rows: list[dict] = []

    for page_size, top_k in combos:
        for M in args.m_values:
            for alpha in args.alphas:
                for weight_pop in weight_pops:
                    run_dir = make_run_dir(
                        args.output_root, page_size, top_k, M, alpha,
                        args.unselected_mode, args.compress_ratio,
                        args.compression_method, weight_pop,
                    )

                    mode_tag = args.unselected_mode
                    if args.unselected_mode == "compressed":
                        pop_str = "popw" if weight_pop else "nopopw"
                        mode_tag = (
                            f"compressed cr={args.compress_ratio} "
                            f"{args.compression_method} {pop_str}"
                        )

                    scoring_method = f"hybrid_multi{M}_ac_max_a{alpha}"

                    if (run_dir / "summary.json").exists():
                        print(
                            f"\n[hybrid_multi] SKIP page_size={page_size} top_k={top_k} "
                            f"M={M} alpha={alpha} cr={args.compress_ratio} {mode_tag} "
                            f"-- summary already exists in {run_dir}",
                            flush=True,
                        )
                        all_rows.extend(collect_run_summary(
                            page_size, top_k, M, alpha, run_dir,
                            args.unselected_mode, args.compress_ratio,
                            args.compression_method, weight_pop,
                        ))
                        continue

                    manifest = {
                        "model_name_or_path": args.model_name_or_path,
                        "context_len": args.context_len,
                        "page_size": page_size,
                        "top_k": top_k,
                        "M": M,
                        "alpha": alpha,
                        "scoring_method": scoring_method,
                        "group_agg_method": args.group_agg_method,
                        "compress_ratio": args.compress_ratio,
                        "unselected_mode": args.unselected_mode,
                        "num_samples": args.num_samples,
                        "max_new_tokens": args.max_new_tokens,
                        "cuda_device": args.cuda_device,
                        "tasks": tasks,
                        "note": "Hybrid multi-highlight scoring: M AC-energy tokens + "
                        f"c=page_size*compress_ratio={int(page_size * args.compress_ratio)} "
                        "DCT coefficients.",
                    }
                    if args.unselected_mode == "compressed":
                        manifest.update({
                            "compression_method": args.compression_method,
                            "weight_compressed_by_population": weight_pop,
                        })
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
                        M=M,
                        alpha=alpha,
                        group_agg_method=args.group_agg_method,
                        unselected_mode=args.unselected_mode,
                        compress_ratio=args.compress_ratio,
                        compression_method=args.compression_method,
                        weight_pop=weight_pop,
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
                        f"\n[hybrid_multi] page_size={page_size} top_k={top_k} "
                        f"M={M} alpha={alpha} cr={args.compress_ratio} {mode_tag} -> {run_dir}",
                        flush=True,
                    )
                    print("  cmd: " + " ".join(shlex.quote(part) for part in cmd), flush=True)
                    if args.dry_run:
                        print(f"  Dry run: skipping execution", flush=True)
                        continue
                    subprocess.run(cmd, cwd=repo_root, check=True)
                    all_rows.extend(collect_run_summary(
                        page_size, top_k, M, alpha, run_dir,
                        args.unselected_mode, args.compress_ratio,
                        args.compression_method, weight_pop,
                    ))

    if all_rows:
        write_summary_files(args.output_root, all_rows)
        print(f"\nCross-sweep summary written to: {args.output_root}")

    print("\n[hybrid_multi] All runs finished.", flush=True)


if __name__ == "__main__":
    main()
