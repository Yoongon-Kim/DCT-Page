"""
MATH-500 evaluation for DCT Page Attention and baseline attention mechanisms.

Evaluates Qwen3-8B (default) on MATH-500 (500 competition-math problems,
free-form short-answer). The prompt template and grader match
SeerAttention/eval/reasoning_tasks (LIMO-derived):

    {problem}
    Please reason step by step, and put your final answer within \\boxed{{}}.

Grading uses math_grader.check_is_correct (vendored from SeerAttention's
Utils/grader.py:math_equal — strip_string + numeric/symbolic equivalence).
Gold answers are NOT coerced to integers (MATH-500 contains fractions,
latex tuples like '(3,\\frac{\\pi}{2})', symbolic expressions like 'p - q').

Outputs:
    {output_dir}/{run_name}/results.jsonl       per-sample records
    {output_dir}/{run_name}/summary.json        overall + per-level + per-subject
    {output_dir}/{run_name}/summary.csv         tabular summary

Usage:
    python eval_math500.py --mode baseline \
        --num_samples 25 --output_dir results/math500 --run_name smoke

    python eval_math500.py --mode page_attention \
        --page_size 32 --top_k 64 --unselected_mode drop \
        --output_dir results/math500 --run_name page_topk64
"""

import os
import sys
import json
import csv
import random
import argparse
from collections import defaultdict

# Ensure baselines/ packages are importable
_BASELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

import torch
from tqdm import tqdm

from eval_ruler import (
    model_name_tag,
    apply_monkey_patch,
    load_model_and_tokenizer,
    _resolve_middle_top_k,
    _validate_upstream_fi_args,
)
from math_grader import (
    extract_answer as _grader_extract_answer,
    check_is_correct as _grader_check_is_correct,
)


# ---------------------------------------------------------------------------
# Mode / model-family compatibility (mirrors eval_aime25.py)
# ---------------------------------------------------------------------------
QWEN3_SUPPORTED_MODES = {
    "baseline", "page_attention", "seer_attention", "multipole_attention",
}
LLAMA_ONLY_MODES = {"inf_llm"}
UNSUPPORTED_MODES = {
    "seer_prefill", "quest_attention",
}


def _assert_mode_model_compatible(mode: str, base_model: str) -> None:
    if mode in UNSUPPORTED_MODES:
        raise ValueError(f"--mode {mode!r} is not supported by eval_math500.")
    name = base_model.lower()
    if mode in LLAMA_ONLY_MODES and "llama" not in name:
        raise ValueError(f"--mode {mode!r} is Llama-only; got {base_model!r}.")
    if mode in QWEN3_SUPPORTED_MODES and "qwen3" not in name:
        raise ValueError(f"--mode {mode!r} requires Qwen3; got {base_model!r}.")
    if mode not in (UNSUPPORTED_MODES | LLAMA_ONLY_MODES | QWEN3_SUPPORTED_MODES):
        raise ValueError(f"Unknown --mode {mode!r}.")


# ---------------------------------------------------------------------------
# Prompt template — matches SeerAttention reasoning_tasks/eval_hf.py:184-188.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def format_math500_sample(item):
    """Return (prompt, gold_str, level, subject)."""
    problem = item["problem"]
    if not isinstance(problem, str):
        problem = str(problem)
    problem = problem.strip()

    gold = item["answer"]
    if not isinstance(gold, str):
        gold = str(gold)
    gold = gold.strip()

    level = str(item.get("level", "")).strip() or "unknown"
    subject = str(item.get("subject", "")).strip() or "unknown"

    prompt = PROMPT_TEMPLATE.format(problem=problem)
    return prompt, gold, level, subject


# ---------------------------------------------------------------------------
# Tokenize (Qwen3 thinking ON for reasoning evals)
# ---------------------------------------------------------------------------
def tokenize_prompt(prompt_text, tokenizer, max_input_len, args=None):
    messages = [{"role": "user", "content": prompt_text}]
    chat_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    base_model = (args.base_model if args is not None else "")
    if "qwen3" in base_model.lower() and "enable_thinking" in (tokenizer.chat_template or ""):
        chat_kwargs["enable_thinking"] = True
    input_ids = tokenizer.apply_chat_template(messages, **chat_kwargs)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    if input_ids.shape[1] > max_input_len:
        half = max_input_len // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
    return input_ids


# ---------------------------------------------------------------------------
# Answer extraction & grading — delegated to math_grader.
# ---------------------------------------------------------------------------
def extract_answer(response: str):
    """Return the boxed-content string, or None when no boxed expression."""
    if not response:
        return None
    pred = _grader_extract_answer(response)
    return pred if pred else None


def is_correct_answer(predicted, gold) -> bool:
    if predicted is None:
        return False
    return _grader_check_is_correct(predicted, gold)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MATH-500 Evaluation (Qwen3-8B)")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "page_attention", "seer_attention",
                                 "seer_prefill",
                                 "multipole_attention", "quest_attention",
                                 "inf_llm"])

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")

    # Dataset
    parser.add_argument("--math500_dataset", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--math500_split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 = all 500)")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Reserved for future pass@k. Only 1 is implemented.")
    parser.add_argument("--seed", type=int, default=42)

    # Generation. SeerAttention's reasoning_tasks defaults --max_tokens=32768
    # (which is total max_length, not new tokens). MATH-500 problems are far
    # shorter than AIME and rarely need >16K thinking tokens; 32768 new tokens
    # is comfortable headroom and matches the Qwen3 model-card recommendation
    # for non-AIME math/reasoning.
    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Sampling on by default (Qwen3 thinking mode requirement).")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k_sampling", type=int, default=20,
                        help="Sampling top_k (distinct from DCT-Page --top_k page budget).")
    parser.add_argument("--min_p", type=float, default=0.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="results/math500")
    parser.add_argument("--run_name", type=str, default=None)

    # DCT Page Attention params (mirror eval_aime25)
    parser.add_argument("--page_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64,
                        help="TOTAL page budget (sink + middle + recent).")
    parser.add_argument("--num_sink_pages", type=int, default=1)
    parser.add_argument("--num_recent_pages", type=int, default=4)
    parser.add_argument("--compress_ratio", type=float, default=0.125)
    parser.add_argument("--scoring_method", type=str, default="max",
                        choices=["mean", "max"])
    parser.add_argument("--group_agg_method", type=str, default="max",
                        choices=["mean", "max"])
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--compressed_token_rope", type=str, default="mixed",
                        choices=["mixed", "block_center"])
    parser.add_argument("--continuous_rope", action="store_true")
    parser.add_argument("--score_use_quest_minmax", action="store_true")
    parser.add_argument("--no_triton", action="store_true")
    parser.add_argument("--attention_backend", type=str, default="upstream_flashinfer",
                        choices=["sdpa", "upstream_flashinfer"])
    parser.add_argument("--verify_upstream_fi", action="store_true")
    parser.add_argument("--comp_kv_quant", type=str, default="fp8_e5m2",
                        choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    parser.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                        choices=["per_page", "per_comp_token"])

    # InfLLM (Llama-only)
    parser.add_argument("--inf_llm_n_init", type=int, default=128)
    parser.add_argument("--inf_llm_repr_topk", type=int, default=4)
    parser.add_argument("--inf_llm_max_cached_block", type=int, default=128)
    parser.add_argument("--inf_llm_chunk_size", type=int, default=8192)

    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    _assert_mode_model_compatible(args.mode, args.base_model)

    if args.num_repeats != 1:
        raise NotImplementedError("--num_repeats > 1 (pass@k) is not implemented yet.")

    if args.run_name is None:
        tag = model_name_tag(args.base_model)
        suffix = "math500"
        if args.mode == "baseline":
            args.run_name = f"{tag}_baseline_{suffix}"
        elif args.mode == "page_attention":
            args.run_name = (f"{tag}_page_attn_topk{args.top_k}T_cr{args.compress_ratio}"
                             f"_ps{args.page_size}_{args.unselected_mode}_{args.comp_kv_quant}"
                             f"_{suffix}")
        elif args.mode == "seer_attention":
            args.run_name = f"{tag}_seer_attention_{suffix}"
        elif args.mode == "multipole_attention":
            args.run_name = f"{tag}_multipole_attention_{suffix}"
        elif args.mode == "inf_llm":
            args.run_name = (f"{tag}_inf_llm_nini{args.inf_llm_n_init}"
                             f"_repr{args.inf_llm_repr_topk}_{suffix}")

    if args.skip_existing:
        summary_path = os.path.join(args.output_dir, args.run_name, "summary.json")
        if os.path.exists(summary_path):
            print(f"SKIP (already exists): {summary_path}")
            sys.exit(0)

    if args.num_samples == 0:
        print(args.run_name)
        sys.exit(0)

    return args


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, dataset, args):
    model.eval()

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "results.jsonl")

    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed_ids.add(r["_id"])
        print(f"Resuming: {len(completed_ids)} samples already completed")

    samples = list(dataset)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]

    out_f = open(output_path, "a")

    for idx, item in enumerate(tqdm(samples, desc="MATH500")):
        sample_id = f"{idx}"
        if sample_id in completed_ids:
            continue

        prompt_text, gold, level, subject = format_math500_sample(item)

        input_ids = tokenize_prompt(prompt_text, tokenizer, args.max_input_len, args=args).to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            if args.mode == "seer_attention":
                output_ids, _ = model.batch_exist_generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_length=input_len + args.max_new_tokens,
                    do_sample=False,
                )
            elif args.mode == "inf_llm":
                output_ids = args._inf_llm_generator.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                )
            elif args.mode == "page_attention" and args.attention_backend == "upstream_flashinfer":
                from dct_page_attention import _generate_with_upstream_fi
                if args.verify_upstream_fi:
                    for _m in model.modules():
                        if hasattr(_m, "q_proj") and hasattr(_m, "k_proj"):
                            _m._verify_upstream = True

                def _harvest_verify_diffs(_model, _output_ids):
                    if not args.verify_upstream_fi:
                        return
                    all_diffs = []
                    for _m in _model.modules():
                        diffs = getattr(_m, "_verify_diffs", None)
                        if diffs:
                            all_diffs.extend(diffs)
                    if all_diffs:
                        import numpy as _np
                        arr = _np.array(all_diffs)
                        print(f"  [verify] _verify_diffs n={len(all_diffs)} "
                              f"max={arr.max():.4f} "
                              f"p99={_np.percentile(arr, 99):.4f} "
                              f"p50={_np.percentile(arr, 50):.4f}")

                output_ids = _generate_with_upstream_fi(
                    model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    on_post_generate=_harvest_verify_diffs,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k_sampling,
                    min_p=args.min_p,
                    use_cache=True,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k_sampling,
                    min_p=args.min_p,
                    use_cache=True,
                )

        generated_ids = output_ids[0, input_len:]
        if args.mode == "inf_llm":
            args._inf_llm_generator.clear()
        del input_ids, output_ids
        torch.cuda.empty_cache()

        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = extract_answer(response)
        is_correct = is_correct_answer(predicted, gold)

        result = {
            "_id": sample_id,
            "unique_id": item.get("unique_id", sample_id),
            "level": level,
            "subject": subject,
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "response": response,
            "input_len": input_len,
        }
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

    out_f.close()

    print(f"[mem] peak_alloc_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)

    results = []
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _accuracy_breakdown(results, key):
    """Return list of {label, accuracy, num_samples} sorted by label."""
    buckets = defaultdict(lambda: [0, 0])  # label -> [correct, total]
    for r in results:
        label = str(r.get(key, "unknown"))
        buckets[label][1] += 1
        if r["correct"]:
            buckets[label][0] += 1
    rows = []
    for label in sorted(buckets.keys()):
        c, t = buckets[label]
        rows.append({
            "label": label,
            "accuracy": round(c / t * 100, 2) if t else 0.0,
            "num_samples": t,
        })
    return rows


def print_summary(results, run_name):
    total = len(results)
    if total == 0:
        print("No results to summarise.")
        return
    correct = sum(1 for r in results if r["correct"])
    print("\n" + "=" * 60)
    print(f"MATH-500 Results — {run_name}")
    print("=" * 60)
    print(f"  Pass@1 : {correct / total * 100:5.1f}%  ({correct}/{total})")
    no_answer = sum(1 for r in results if r["predicted"] is None)
    if no_answer > 0:
        print(f"  Failed answer extraction: {no_answer}/{total}")

    print("\n  By level:")
    for row in _accuracy_breakdown(results, "level"):
        print(f"    L{row['label']:>7}  {row['accuracy']:5.1f}%  ({row['num_samples']})")
    print("\n  By subject:")
    for row in _accuracy_breakdown(results, "subject"):
        print(f"    {row['label']:>22}  {row['accuracy']:5.1f}%  ({row['num_samples']})")
    print("=" * 60)


def build_summary(results, args):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    overall_acc = round(correct / total * 100, 2) if total else 0.0

    summary = {
        "mode": args.mode,
        "model": args.base_model,
        "dataset": args.math500_dataset,
        "run_name": args.run_name,
        "num_samples": total,
        "overall_accuracy": overall_acc,
        "extraction_failures": sum(1 for r in results if r["predicted"] is None),
        "max_new_tokens": args.max_new_tokens,
        "by_level": _accuracy_breakdown(results, "level"),
        "by_subject": _accuracy_breakdown(results, "subject"),
        "per_problem": [
            {"_id": r["_id"], "unique_id": r.get("unique_id", r["_id"]),
             "level": r.get("level", ""), "subject": r.get("subject", ""),
             "gold": r["gold"], "predicted": r["predicted"], "correct": r["correct"]}
            for r in results
        ],
    }

    if args.mode == "page_attention":
        summary["top_k"] = args.top_k
        summary["middle_top_k"] = _resolve_middle_top_k(args)
        summary["page_size"] = args.page_size
        summary["compress_ratio"] = args.compress_ratio
        summary["scoring_method"] = args.scoring_method
        summary["group_agg_method"] = args.group_agg_method
        summary["unselected_mode"] = args.unselected_mode
        summary["comp_kv_quant"] = args.comp_kv_quant
    elif args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        summary["seer_attn_config"] = SEER_ATTN_CONFIG
    elif args.mode == "multipole_attention":
        from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
        summary["multipole_attn_config"] = MULTIPOLE_ATTN_CONFIG
    elif args.mode == "inf_llm":
        from infllm.config import INF_LLM_CONFIG
        summary["inf_llm_config"] = INF_LLM_CONFIG

    return summary


def write_summary_files(results, args):
    summary = build_summary(results, args)
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(run_dir, "summary.csv")
    rows = [{
        "group": "overall",
        "label": "overall",
        "accuracy": summary["overall_accuracy"],
        "num_samples": summary["num_samples"],
    }]
    for row in summary["by_level"]:
        rows.append({
            "group": "level",
            "label": f"L{row['label']}",
            "accuracy": row["accuracy"],
            "num_samples": row["num_samples"],
        })
    for row in summary["by_subject"]:
        rows.append({
            "group": "subject",
            "label": row["label"],
            "accuracy": row["accuracy"],
            "num_samples": row["num_samples"],
        })
    for entry in summary["per_problem"]:
        rows.append({
            "group": "problem",
            "label": str(entry["unique_id"]),
            "accuracy": 100.0 if entry["correct"] else 0.0,
            "num_samples": 1,
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "label", "accuracy", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)

    return summary_path, csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    _validate_upstream_fi_args(args)

    print("=" * 60)
    print(f"LOADING MODEL (mode={args.mode})")
    print("=" * 60)
    apply_monkey_patch(args)
    model, tokenizer = load_model_and_tokenizer(args)

    from datasets import load_dataset
    print(f"\nLoading MATH-500: {args.math500_dataset} (split={args.math500_split})")
    dataset = load_dataset(args.math500_dataset, split=args.math500_split)
    print(f"Loaded {len(dataset)} samples")

    results = evaluate(model, tokenizer, dataset, args)
    print_summary(results, args.run_name)
    summary_path, csv_path = write_summary_files(results, args)
    print(f"\nSummary: {summary_path}")
    print(f"CSV    : {csv_path}")


if __name__ == "__main__":
    main()
