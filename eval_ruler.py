"""
Unified RULER benchmark evaluation.

Supports four attention modes (baseline, page_attention, seer_attention,
multipole_attention) with optional data preparation.  Mirrors the mode-based
dispatch pattern of eval_longbench_v1.py.

Usage examples:
    # Prepare data + run baseline
    python eval_ruler.py --mode baseline \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --prepare --model_template_type llama-3 \
        --output_dir results_ruler --run_name baseline

    # Run seer attention (data already prepared)
    python eval_ruler.py --mode seer_attention \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results_ruler --run_name seer_budget1024
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# RULER task list (matches eval_ruler/synthetic.yaml)
# ---------------------------------------------------------------------------
ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

DEFAULT_SEQ_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

RULER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ruler")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified RULER Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "page_attention", "seer_attention",
                                 "multipole_attention"])

    # Model
    parser.add_argument("--base_model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")

    # Data preparation
    parser.add_argument("--prepare", action="store_true",
                        help="Run data preparation before prediction (skips if data exists)")
    parser.add_argument("--model_template_type", type=str, default="llama-3",
                        help="Template type for prepare.py (e.g. llama-3, qwen-3)")
    parser.add_argument("--tokenizer_family", type=str, default="llama",
                        choices=["llama", "qwen2", "qwen3"],
                        help="Tokenizer family name for data directory (models in same family share data)")

    # RULER config
    parser.add_argument("--seq_lengths", type=int, nargs="+",
                        default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_samples", type=int, default=25)

    # Output
    parser.add_argument("--output_dir", type=str, default="results_ruler")
    parser.add_argument("--run_name", type=str, default=None)

    # DCT Page Attention params
    parser.add_argument("--page_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--compress_ratio", type=float, default=0.25)
    parser.add_argument("--scoring_method", type=str, default="max",
                        choices=["mean", "max", "sum"])
    parser.add_argument("--group_agg_method", type=str, default="mean",
                        choices=["mean", "max", "topp"])
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--no_continuous_rope", action="store_true")
    parser.add_argument("--no_triton", action="store_true")

    args = parser.parse_args()
    args.continuous_rope = not args.no_continuous_rope

    if args.run_name is None:
        if args.mode == "baseline":
            args.run_name = "baseline"
        elif args.mode == "page_attention":
            args.run_name = (f"page_attn_topk{args.top_k}_cr{args.compress_ratio}"
                             f"_ps{args.page_size}_{args.unselected_mode}")
        elif args.mode == "seer_attention":
            args.run_name = "seer_attention"
        elif args.mode == "multipole_attention":
            args.run_name = "multipole_attention"

    return args


# ---------------------------------------------------------------------------
# Load RULER task configs (tokens_to_generate, metric_fn, etc.)
# ---------------------------------------------------------------------------
def load_task_configs():
    """Load task configurations from eval_ruler YAML and constants."""
    # Data constants (tokens_to_generate, template, etc.)
    sys.path.insert(0, os.path.join(RULER_DIR, "data"))
    data_constants = importlib.import_module("synthetic.constants")
    data_tasks = data_constants.TASKS

    # Eval constants (metric_fn)
    # Remove cached data version so we can load the eval version
    if "synthetic.constants" in sys.modules:
        del sys.modules["synthetic.constants"]
    sys.path.insert(0, os.path.join(RULER_DIR, "eval"))
    eval_constants = importlib.import_module("synthetic.constants")
    eval_tasks = eval_constants.TASKS

    # YAML customization
    with open(os.path.join(RULER_DIR, "synthetic.yaml"), "r") as f:
        yaml_tasks = yaml.safe_load(f)

    configs = {}
    for task_name, yaml_cfg in yaml_tasks.items():
        base_task = yaml_cfg["task"]
        cfg = dict(yaml_cfg)
        cfg.update(data_tasks[base_task])
        cfg.update(eval_tasks[base_task])
        configs[task_name] = cfg

    return configs


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(args):
    """Run eval_ruler/data/prepare.py for tasks that don't have data yet."""
    model_family = args.tokenizer_family
    prepare_script = os.path.join(RULER_DIR, "data", "prepare.py")

    for seq_len in args.seq_lengths:
        for task in args.tasks:
            data_file = Path("ruler_data") / model_family / str(seq_len) / task / "validation.jsonl"
            if data_file.exists():
                print(f"  Data exists, skipping: {data_file}")
                continue

            data_file.parent.mkdir(parents=True, exist_ok=True)
            save_dir = str(Path("ruler_data").resolve() / model_family / str(seq_len))

            cmd = [
                sys.executable, prepare_script,
                "--save_dir", save_dir,
                "--benchmark", "synthetic",
                "--task", task,
                "--tokenizer_path", args.base_model,
                "--tokenizer_type", "hf",
                "--max_seq_length", str(seq_len),
                "--model_template_type", args.model_template_type,
                "--num_samples", str(args.num_samples),
            ]
            print(f"  Preparing {task} @ seq_len={seq_len}...")
            result = subprocess.run(cmd, cwd=RULER_DIR, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR preparing {task}: {result.stderr}")
            else:
                print(f"  Done: {data_file}")


# ---------------------------------------------------------------------------
# Monkey-patching
# ---------------------------------------------------------------------------
def apply_monkey_patch(args):
    """Apply attention monkey-patch before model loading."""
    if args.mode == "page_attention":
        model_name_lower = args.base_model.lower()
        if "llama" in model_name_lower:
            from dct_page_attention import replace_llama_attn
            replace_llama_attn(
                page_size=args.page_size,
                top_k=args.top_k,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
            )
        else:
            from dct_page_attention import replace_qwen2_attn
            replace_qwen2_attn(
                page_size=args.page_size,
                top_k=args.top_k,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
            )
    elif args.mode == "multipole_attention":
        from multipole_attn import replace_attn_multipole
        from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
        MULTIPOLE_ATTN_CONFIG["base_model"] = args.base_model
        replace_attn_multipole(MULTIPOLE_ATTN_CONFIG)
    elif args.mode == "baseline":
        print("Baseline mode: full attention (no monkey-patch)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(args):
    """Load model and tokenizer based on mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        from seer_attn import SeerDecodingQwen3ForCausalLM

        seer_model = SEER_ATTN_CONFIG["seer_model"]
        print(f"Loading SeerAttention-R model: {seer_model}")
        model = SeerDecodingQwen3ForCausalLM.from_pretrained(
            seer_model,
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method=SEER_ATTN_CONFIG["sparsity_method"],
            seerattn_token_budget=SEER_ATTN_CONFIG["token_budget"],
            seerattn_threshold=SEER_ATTN_CONFIG["threshold"],
            seerattn_start_layer=SEER_ATTN_CONFIG["start_layer"],
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            max_position_embeddings=131072,
        ).cuda()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    else:
        attn_impl = "sdpa"
        print(f"Loading model: {args.base_model} (attn: {attn_impl})")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        yarn_kwargs = {}
        if "qwen3" in args.base_model.lower():
            yarn_kwargs = {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 1000000.0,
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                },
                "max_position_embeddings": 131072,
            }
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            **yarn_kwargs,
        )
        model.eval()
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        if args.mode == "multipole_attention":
            from multipole_attn import init_multipole_layers
            init_multipole_layers(model)
            print("Multipole attention layers initialized.")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction for a single task
# ---------------------------------------------------------------------------
def predict_task(model, tokenizer, task, task_config, data_dir, pred_dir, args):
    """Generate predictions for one RULER task. Returns list of result dicts."""
    data_file = data_dir / task / "validation.jsonl"
    if not data_file.exists():
        print(f"  WARNING: data file not found: {data_file}, skipping")
        return []

    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    if args.num_samples > 0 and len(data) > args.num_samples:
        data = data[:args.num_samples]

    tokens_to_generate = task_config["tokens_to_generate"]

    # Resume support
    pred_file = pred_dir / f"{task}.jsonl"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    completed_indices = set()
    if pred_file.exists():
        with open(pred_file, "r") as f:
            for line in f:
                if line.strip():
                    completed_indices.add(json.loads(line)["index"])
        print(f"  Resuming {task}: {len(completed_indices)} already done")

    remaining = [s for s in data if s["index"] not in completed_indices]
    if not remaining:
        print(f"  {task}: all samples already completed")
        # Return all results for scoring
        with open(pred_file, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(pred_file, "a", encoding="utf-8", buffering=1) as fout:
        for sample in tqdm(remaining, desc=f"  {task}"):
            prompt = sample["input"]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                if args.mode == "seer_attention":
                    output_ids, _ = model.batch_exist_generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_length=input_len + tokens_to_generate,
                        do_sample=False,
                    )
                else:
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=tokens_to_generate,
                        do_sample=False,
                        use_cache=True,
                    )

            generated_ids = output_ids[0, input_len:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            result = {
                "index": sample["index"],
                "pred": pred_text,
                "input": prompt,
                "outputs": sample["outputs"],
                "others": sample.get("others", {}),
                "truncation": sample.get("truncation", -1),
                "length": sample.get("length", -1),
            }
            fout.write(json.dumps(result) + "\n")

    # Reload all results for scoring
    with open(pred_file, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Evaluation (scoring)
# ---------------------------------------------------------------------------
def postprocess_pred(predict_str):
    """Clean prediction string (matches eval_ruler/eval/evaluate.py)."""
    import re
    predict_str = predict_str.strip()
    np_pattern = re.compile(r"[\x00-\x1f]")
    predict_str = np_pattern.sub("\n", predict_str).strip()
    return predict_str


def evaluate_task(predictions, task_config):
    """Score predictions for one task. Returns (score, nulls_str)."""
    preds = [postprocess_pred(p["pred"]) for p in predictions]
    refs = [p["outputs"] for p in predictions]

    nulls = f"{sum(len(x) == 0 for x in preds)}/{len(preds)}"

    if len(refs) > 0 and refs[0][0] is not None:
        score = task_config["metric_fn"](preds, refs)
    else:
        score = 0.0

    return score, nulls


def write_summary_csv(eval_results, pred_dir):
    """Write summary.csv to pred_dir (same format as eval_ruler/eval/evaluate.py)."""
    import pandas as pd

    tasks = list(eval_results.keys())
    scores = [eval_results[t]["score"] for t in tasks]
    nulls = [eval_results[t]["nulls"] for t in tasks]

    dfs = [
        ["Tasks"] + tasks,
        ["Score"] + scores,
        ["Nulls"] + nulls,
    ]
    output_file = pred_dir / ("summary.csv" if len(tasks) > 1 else f"summary-{tasks[0]}.csv")
    df = pd.DataFrame(dfs)
    df.to_csv(output_file, index=False)
    print(f"  Summary saved to {output_file}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    torch.manual_seed(42)

    task_configs = load_task_configs()

    # Validate requested tasks
    for task in args.tasks:
        if task not in task_configs:
            raise ValueError(f"Unknown task: {task}. Available: {list(task_configs.keys())}")

    # Step 1: Data preparation (optional)
    if args.prepare:
        print("=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        prepare_data(args)

    # Step 2: Monkey-patch + load model
    print("\n" + "=" * 60)
    print(f"LOADING MODEL (mode={args.mode})")
    print("=" * 60)
    apply_monkey_patch(args)
    model, tokenizer = load_model_and_tokenizer(args)

    model_family = args.tokenizer_family

    # Step 3: Predict + evaluate per sequence length
    all_seq_results = {}  # {seq_len: {task: score}}

    for seq_len in args.seq_lengths:
        print(f"\n{'=' * 60}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print("=" * 60)

        data_dir = Path("ruler_data") / model_family / str(seq_len)
        pred_dir = Path(args.output_dir) / args.run_name / "synthetic" / str(seq_len) / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)

        eval_results = {}
        for task in args.tasks:
            if task not in task_configs:
                continue

            predictions = predict_task(
                model, tokenizer, task, task_configs[task],
                data_dir, pred_dir, args,
            )

            if predictions:
                score, nulls = evaluate_task(predictions, task_configs[task])
                eval_results[task] = {"score": score, "nulls": nulls}
                print(f"  {task}: score={score:.2f}, nulls={nulls}")

        if eval_results:
            write_summary_csv(eval_results, pred_dir)
            all_seq_results[seq_len] = {t: r["score"] for t, r in eval_results.items()}

    # Step 4: Consolidated summary
    print(f"\n{'=' * 60}")
    print(f"RULER Results — {args.run_name}")
    print("=" * 60)

    if all_seq_results:
        # Header
        seq_lens = sorted(all_seq_results.keys())
        header = f"{'Task':24s}" + "".join(f" {sl:>8d}" for sl in seq_lens) + "     Avg"
        print(header)
        print("-" * len(header))

        task_avgs = {}
        for task in args.tasks:
            scores = []
            row = f"{task:24s}"
            for sl in seq_lens:
                if task in all_seq_results.get(sl, {}):
                    s = all_seq_results[sl][task]
                    scores.append(s)
                    row += f" {s:>8.2f}"
                else:
                    row += f" {'N/A':>8s}"
            avg = sum(scores) / len(scores) if scores else 0
            task_avgs[task] = avg
            row += f" {avg:>7.2f}"
            print(row)

        # Overall average
        print("-" * len(header))
        overall_scores = []
        row = f"{'AVERAGE':24s}"
        for sl in seq_lens:
            sl_scores = list(all_seq_results.get(sl, {}).values())
            if sl_scores:
                sl_avg = sum(sl_scores) / len(sl_scores)
                overall_scores.append(sl_avg)
                row += f" {sl_avg:>8.2f}"
            else:
                row += f" {'N/A':>8s}"
        overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        row += f" {overall:>7.2f}"
        print(row)
        print("=" * len(header))

        # Save consolidated summary JSON
        summary_path = Path(args.output_dir) / args.run_name / "summary.json"
        summary = {
            "mode": args.mode,
            "base_model": args.base_model,
            "run_name": args.run_name,
            "seq_length_scores": {str(sl): all_seq_results[sl] for sl in seq_lens},
            "task_averages": {t: round(a, 2) for t, a in task_avgs.items()},
            "overall": round(overall, 2),
        }
        if args.mode == "page_attention":
            summary["top_k"] = args.top_k
            summary["page_size"] = args.page_size
            summary["compress_ratio"] = args.compress_ratio
            summary["scoring_method"] = args.scoring_method
            summary["group_agg_method"] = args.group_agg_method
            summary["unselected_mode"] = args.unselected_mode
        elif args.mode == "seer_attention":
            from seer_attn.config import SEER_ATTN_CONFIG
            summary["seer_attn_config"] = SEER_ATTN_CONFIG
        elif args.mode == "multipole_attention":
            from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
            summary["multipole_attn_config"] = MULTIPOLE_ATTN_CONFIG
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

    elapsed = (time.time() - start_time) / 60
    print(f"Total time: {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
