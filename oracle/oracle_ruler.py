#!/usr/bin/env python3
"""
Run a single RULER evaluation mode and accumulate task outputs in one flat run directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Bootstrap sys.path so this script runs from any cwd. The project root holds
# `dct_page_attention.py`, which `apply_dct_patch` imports lazily below.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Helpers (originally lived in the deleted compare_baseline_dct.py)
# ---------------------------------------------------------------------------
class RunLogger:
    def __init__(self, log_path: Path):
        self._fp = log_path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


def load_samples(path: Path, num_samples: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        rows = [json.loads(line) for line in fp if line.strip()]
    return rows[:num_samples]


def load_metric_fn(benchmark: str, task_name: str):
    eval_root = _REPO_ROOT / "benchmark" / "eval_ruler" / "eval"
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    module = importlib.import_module(f"{benchmark}.constants")
    tasks_base = module.TASKS
    with (eval_root.parent / f"{benchmark}.yaml").open("r", encoding="utf-8") as fp:
        tasks_customized = yaml.safe_load(fp)
    task_config = tasks_customized[task_name]
    task_config.update(tasks_base[task_config["task"]])
    return task_config["metric_fn"], task_config


def cleanup_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(args: argparse.Namespace):
    yarn_kwargs = {}
    if "qwen3" in args.model_name_or_path.lower():
        yarn_kwargs = {
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            "max_position_embeddings": 131072,
        }
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": args.cuda_device},
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
        **yarn_kwargs,
    ).eval()


def generate_one(
    model,
    tokenizer,
    sample: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1] :].cpu()
    return {
        "index": sample["index"],
        "input_len": int(input_ids.shape[1]),
        "gen_ids": generated_ids.tolist(),
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "gold": sample["outputs"],
    }


def apply_dct_patch(args: argparse.Namespace) -> None:
    model_name = args.model_name_or_path.lower()
    common_kwargs = dict(
        page_size=args.dct_page_size,
        top_k=args.dct_top_k,
        sink_size=args.dct_sink_size,
        recent_size=args.dct_recent_size,
        compress_ratio=args.dct_compress_ratio,
        scoring_method=args.dct_scoring_method,
        group_agg_method=args.dct_group_agg_method,
        unselected_mode=args.dct_unselected_mode,
        compressed_token_rope=args.dct_compressed_token_rope,
        continuous_rope=args.dct_continuous_rope,
        select_with_oracle_page_scores=args.dct_select_with_oracle_page_scores,
        use_triton=not args.dct_no_triton,
        weight_compressed_by_population=True,
        comp_kv_quant=args.dct_comp_kv_quant,
        comp_kv_quant_granularity=args.dct_comp_kv_quant_granularity,
    )
    if "llama" in model_name:
        from dct_page_attention import replace_llama_attn

        replace_llama_attn(**common_kwargs)
    elif "qwen3" in model_name:
        from dct_page_attention import replace_qwen3_attn

        replace_qwen3_attn(**common_kwargs)
    elif "qwen" in model_name:
        from dct_page_attention import replace_qwen2_attn

        replace_qwen2_attn(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model for DCT patch: {args.model_name_or_path}")


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
    p = argparse.ArgumentParser(description="Run one RULER mode into a flat task-jsonl directory")
    p.add_argument("--mode", choices=["baseline", "page_attention"], required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--data_root", type=Path, default=Path("benchmark/data/ruler_data"))
    p.add_argument("--output_root", type=Path, default=Path("results/results_ruler/oracle"))
    p.add_argument("--tag", default="ruler_run")
    p.add_argument("--run_dir", type=Path, default=None)
    p.add_argument("--benchmark", default="synthetic")
    p.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--dct_page_size", type=int, default=32)
    p.add_argument("--dct_top_k", type=int, default=64)
    p.add_argument("--dct_sink_size", type=int, default=4)
    p.add_argument("--dct_recent_size", type=int, default=128)
    p.add_argument("--dct_compress_ratio", type=float, default=0.125)
    p.add_argument("--dct_scoring_method", type=str, default="max",
                   choices=["mean", "max"])
    p.add_argument(
        "--dct_group_agg_method",
        type=str,
        default="max",
        choices=["mean", "max"],
    )
    p.add_argument(
        "--dct_unselected_mode",
        type=str,
        default="drop",
        choices=["drop", "compressed"],
    )
    p.add_argument("--dct_compressed_token_rope", type=str, default="mixed", choices=["mixed", "block_center"])
    p.add_argument("--dct_select_with_oracle_page_scores", action="store_true", default=True)
    p.add_argument("--dct_no_select_with_oracle_page_scores", dest="dct_select_with_oracle_page_scores", action="store_false")
    p.add_argument("--dct_continuous_rope", action="store_true",
                   help="Temporarily disabled — raises error if used")
    p.add_argument("--dct_comp_kv_quant", type=str, default="none",
                   choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"],
                   help="Fake-quantization of compressed K/V at write time "
                        "(precision study; no real byte-level storage change)")
    p.add_argument("--dct_comp_kv_quant_granularity", type=str, default="per_page",
                   choices=["per_page", "per_comp_token"],
                   help="Scale granularity for --dct_comp_kv_quant")
    p.add_argument("--dct_no_triton", action="store_true")
    p.add_argument("--sweep_combos", nargs="+", default=None,
                   help="Space-separated page_size,top_k pairs to sweep. "
                        "E.g. --sweep_combos 16,128 32,64 64,32. "
                        "Each combo runs as a separate subprocess with its own run directory.")
    return p.parse_args()


def resolve_model_family(model_name_or_path: str) -> str:
    name = model_name_or_path.lower().split("/")[-1]
    if "qwen3" in name:
        return "qwen3"
    elif "qwen2" in name:
        return "qwen2"
    elif "llama-3" in name or "llama3" in name:
        return "llama"
    elif "llama" in name:
        return "llama"
    else:
        return name.split("-")[0]


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


def parse_combos(values: list[str]) -> list[tuple[int, int]]:
    """Parse 'page_size,top_k' pairs, e.g. ['16,128', '32,64', '64,32']."""
    combos = []
    for v in values:
        parts = v.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Each combo must be page_size,top_k — got {v!r}")
        combos.append((int(parts[0]), int(parts[1])))
    return combos


def make_run_dir(
    output_root: Path,
    page_size: int,
    top_k: int,
    compress_ratio: float,
    comp_kv_quant: str = "none",
) -> Path:
    quant_tag = "noquant" if comp_kv_quant == "none" else comp_kv_quant
    run_dir = output_root / f"ps{page_size}_topk{top_k}_cr{compress_ratio}_{quant_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def score_predictions(metric_fn, rows: list[dict]) -> float:
    preds = [row["pred"] for row in rows]
    refs = [row["outputs"] for row in rows]
    return metric_fn(preds, refs)


def make_eval_row(sample: dict, row: dict) -> dict:
    return {
        "index": sample["index"],
        "pred": row["text"],
        "input": sample["input"],
        "outputs": sample["outputs"],
        "others": sample.get("others", {}),
        "length": sample.get("length", -1),
        "input_len": row["input_len"],
        "gen_ids": row["gen_ids"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        fp.flush()


def write_summary(run_dir: Path, manifest: dict, summary_rows: list[dict]) -> None:
    summary_rows = sorted(summary_rows, key=lambda r: TASKS.index(r["task"]))
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["task", "score", "num_samples", "output_jsonl", "elapsed_sec"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    overall = sum(row["score"] for row in summary_rows) / len(summary_rows) if summary_rows else 0.0
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "rows": summary_rows,
                "overall_score_avg": overall,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def sweep_main(args: argparse.Namespace) -> None:
    """Re-invoke this script as a subprocess for each (page_size, top_k) combo."""
    combos = parse_combos(args.sweep_combos)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    # Build the base argv, stripping --sweep_combos and its values,
    # and stripping --dct_page_size / --dct_top_k (we override per combo).
    skip_args = {"--sweep_combos", "--dct_page_size", "--dct_top_k", "--run_dir"}
    base_argv = []
    it = iter(sys.argv[1:])
    for arg in it:
        if arg in skip_args:
            # Skip this flag and its value(s)
            if arg == "--sweep_combos":
                # consume all following non-flag tokens (nargs="+")
                for remaining in it:
                    if remaining.startswith("-"):
                        base_argv.append(remaining)
                        break
            else:
                next(it, None)  # skip the single value
        else:
            base_argv.append(arg)

    for page_size, top_k in combos:
        run_dir = make_run_dir(
            args.output_root,
            page_size,
            top_k,
            args.dct_compress_ratio,
            args.dct_comp_kv_quant,
        )
        cmd = [
            sys.executable,
            str(script_path),
            *base_argv,
            "--dct_page_size", str(page_size),
            "--dct_top_k", str(top_k),
            "--run_dir", str(run_dir),
        ]
        print(
            f"\n[sweep] page_size={page_size} top_k={top_k} -> {run_dir}",
            flush=True,
        )
        print("  cmd: " + " ".join(shlex.quote(part) for part in cmd), flush=True)
        subprocess.run(cmd, cwd=repo_root, check=True)

    print("\n[sweep] All combos finished.", flush=True)


def main() -> None:
    args = parse_args()

    if args.sweep_combos:
        sweep_main(args)
        return

    tasks = resolve_tasks(args.tasks)
    run_dir = args.run_dir if args.run_dir is not None else make_run_dir(
        args.output_root,
        args.dct_page_size,
        args.dct_top_k,
        args.dct_compress_ratio,
        args.dct_comp_kv_quant,
    )
    if args.run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": args.mode,
        "model_name_or_path": args.model_name_or_path,
        "context_len": args.context_len,
        "benchmark": args.benchmark,
        "tasks": tasks,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "cuda_device": args.cuda_device,
        "dct": {
            "page_size": args.dct_page_size,
            "top_k": args.dct_top_k,
            "sink_size": args.dct_sink_size,
            "recent_size": args.dct_recent_size,
            "compress_ratio": args.dct_compress_ratio,
            "scoring_method": args.dct_scoring_method,
            "group_agg_method": args.dct_group_agg_method,
            "unselected_mode": args.dct_unselected_mode,
            "compressed_token_rope": args.dct_compressed_token_rope,
            "continuous_rope": args.dct_continuous_rope,
            "select_with_oracle_page_scores": args.dct_select_with_oracle_page_scores,
            "use_triton": not args.dct_no_triton,
            "comp_kv_quant": args.dct_comp_kv_quant,
            "comp_kv_quant_granularity": args.dct_comp_kv_quant_granularity,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + " ".join(shlex.quote(arg) for arg in [sys.executable, __file__, *sys.argv[1:]])
        + "\n",
        encoding="utf-8",
    )

    logger = RunLogger(run_dir / "run.log")
    model = None
    try:
        if args.mode == "page_attention":
            logger.log("Applying page-attention patch")
            apply_dct_patch(args)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            local_files_only=args.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.log(f"Set pad_token_id to eos_token_id={tokenizer.pad_token_id}")

        logger.log(f"Loading model for mode={args.mode}")
        model = load_model(args)

        summary_rows: list[dict] = []
        run_start = time.time()
        for task in tasks:
            task_start = time.time()
            model_family = resolve_model_family(args.model_name_or_path)
            data_path = args.data_root / model_family / str(args.context_len) / task / "validation.jsonl"
            samples = load_samples(data_path, args.num_samples)
            metric_fn, _ = load_metric_fn(args.benchmark, task)
            logger.log(f"Running task={task} samples={len(samples)} data={data_path}")

            eval_rows = []
            for idx, sample in enumerate(samples, start=1):
                gen_row = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                )
                eval_rows.append(make_eval_row(sample, gen_row))
                logger.log(
                    f"{task} sample {idx}/{len(samples)} index={sample['index']} "
                    f"input_len={gen_row['input_len']} pred={gen_row['text']!r}"
                )

            output_path = run_dir / f"{task}.jsonl"
            write_jsonl(output_path, eval_rows)
            score = score_predictions(metric_fn, eval_rows)
            summary_rows.append(
                {
                    "task": task,
                    "score": score,
                    "num_samples": len(eval_rows),
                    "output_jsonl": str(output_path),
                    "elapsed_sec": time.time() - task_start,
                }
            )
            write_summary(run_dir, manifest, summary_rows)
            logger.log(
                f"Finished task={task} score={score:.4f} "
                f"elapsed={time.time() - task_start:.1f}s overall_elapsed={time.time() - run_start:.1f}s"
            )
    finally:
        cleanup_model(model)
        logger.close()


if __name__ == "__main__":
    main()
