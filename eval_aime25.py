"""
AIME 2025 evaluation for DCT Page Attention and baseline attention mechanisms.

Evaluates Qwen3-8B (the default reasoning model in this harness) on the
AIME 2025 math competition (30 integer-answer problems) across the attention
modes exposed by eval_ruler.py. Modes that do not support Qwen3
(seer_prefill, quest_attention, duo_attention, shadowkv) are still listed in
--mode for parity, but the script raises ValueError before model load if one
is selected. inf_llm is Llama-only (transformers==4.37.2 env pinning) and
requires --base_model pointing to a Llama-family checkpoint.

Default dataset is yentinglin/aime_2025. Prompting is chain-of-thought; final
answer is extracted from the innermost \\boxed{...} expression and compared
to the gold integer with exact match.

Outputs:
    {output_dir}/{run_name}.jsonl              per-sample records
    {output_dir}/{run_name}_summary.json       aggregate accuracy + config
    {output_dir}/{run_name}_summary.csv        tabular summary

Usage:
    python eval_aime25.py --mode baseline \
        --num_samples 5 --output_dir results_aime25 --run_name smoke_baseline

    python eval_aime25.py --mode page_attention \
        --page_size 32 --top_k 64 --unselected_mode drop \
        --output_dir results_aime25 --run_name qwen_page_topk64
"""

import os
import sys
import json
import re
import csv
import random
import argparse

# Ensure baselines/ packages are importable
_BASELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

import torch
from tqdm import tqdm

from eval_ruler import model_name_tag, apply_monkey_patch, load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Mode / model-family compatibility
# ---------------------------------------------------------------------------
# Modes that run on Qwen3-8B (current default reasoning model).
QWEN3_SUPPORTED_MODES = {
    "baseline", "page_attention", "seer_attention", "multipole_attention",
}
# Modes that are Llama-only in this harness (e.g. InfLLM's transformers==4.37
# shim and Llama-only RoPE replacement). When --mode is in this set, --base_model
# must be a Llama-family ID.
LLAMA_ONLY_MODES = {"inf_llm"}
# Modes intentionally listed in --mode for argparse parity but unsupported here.
UNSUPPORTED_MODES = {
    "seer_prefill", "quest_attention", "duo_attention", "shadowkv",
}


def _assert_mode_model_compatible(mode: str, base_model: str) -> None:
    # Intentionally duplicates the family check in baselines/infllm/__init__.py:33-46
    # (`assert_llama_only`) — we want early failure *before* model load and config
    # rope stripping. The (mode × family) cross-axis lives only here.
    if mode in UNSUPPORTED_MODES:
        raise ValueError(f"--mode {mode!r} is not supported by eval_aime25.")
    name = base_model.lower()
    if mode in LLAMA_ONLY_MODES and "llama" not in name:
        raise ValueError(f"--mode {mode!r} is Llama-only; got {base_model!r}.")
    if mode in QWEN3_SUPPORTED_MODES and "qwen3" not in name:
        raise ValueError(f"--mode {mode!r} requires Qwen3; got {base_model!r}.")
    if mode not in (UNSUPPORTED_MODES | LLAMA_ONLY_MODES | QWEN3_SUPPORTED_MODES):
        raise ValueError(f"Unknown --mode {mode!r}.")


# ---------------------------------------------------------------------------
# Prompt template (CoT, math)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Solve the following problem. Reason step by step, then place your final integer answer (0-999) inside \\boxed{{}}.

Problem: {problem}"""


def format_aime25_sample(item):
    """Return (prompt, gold_str)."""
    problem = item["problem"].strip() if isinstance(item.get("problem"), str) else str(item["problem"]).strip()
    answer_raw = item["answer"]
    gold = str(int(answer_raw)) if isinstance(answer_raw, (int, float)) else str(answer_raw).strip()
    prompt = PROMPT_TEMPLATE.format(problem=problem)
    return prompt, gold


# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------
def tokenize_prompt(prompt_text, tokenizer, max_input_len, args=None):
    messages = [{"role": "user", "content": prompt_text}]
    chat_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Qwen3 chat template: keep thinking ON for reasoning evals.
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
# Answer extraction (innermost \boxed{...} with brace-balance scan)
# ---------------------------------------------------------------------------
_LAST_INT_RE = re.compile(r"(-?\d+)(?!.*\d)", re.DOTALL)


def _find_innermost_boxed(text: str):
    """Return the contents of the last well-balanced \\boxed{...}, or None."""
    last_content = None
    i = 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 1
        j = idx + len("\\boxed{")
        start = j
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_content = text[start:j]
                    break
            j += 1
        if depth != 0:
            break
        i = j + 1
    return last_content


def _normalize_int(s: str):
    """Strip $, commas, whitespace, leading +; return canonical int string or None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("\\,", "")
    s = s.strip("+ ")
    m = re.fullmatch(r"-?\d+", s)
    if m:
        return str(int(m.group(0)))
    # Try to pluck a trailing integer from the boxed content (e.g. "= 042" -> 42)
    m = _LAST_INT_RE.search(s)
    if m:
        return str(int(m.group(1)))
    return None


def extract_answer(response: str):
    """Return canonical integer string or None."""
    if not response:
        return None
    boxed = _find_innermost_boxed(response)
    if boxed is not None:
        ans = _normalize_int(boxed)
        if ans is not None:
            return ans
    # Fallback: last integer in the response.
    m = _LAST_INT_RE.search(response)
    if m:
        return str(int(m.group(1)))
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AIME 2025 Evaluation (Qwen3-8B)")

    # Mode (all visible for parity; model-family guard runs after parsing)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "page_attention", "seer_attention",
                                 "seer_prefill",
                                 "multipole_attention", "quest_attention",
                                 "duo_attention",
                                 "shadowkv",
                                 "inf_llm"])

    # Model — Qwen3-8B default; Llama-family accepted for inf_llm mode
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen3-8B")

    # Dataset
    parser.add_argument("--aime25_dataset", type=str, default="yentinglin/aime_2025")
    parser.add_argument("--aime25_split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 = all 30)")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Reserved for future pass@k. Only 1 is implemented.")
    parser.add_argument("--seed", type=int, default=42)

    # Generation
    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=16384,
                        help="AIME solutions can be very long; default 16384 tokens")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_aime25")
    parser.add_argument("--run_name", type=str, default=None)

    # DCT Page Attention params
    parser.add_argument("--page_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--num_sink_pages", type=int, default=1)
    parser.add_argument("--num_recent_pages", type=int, default=5)
    parser.add_argument("--compress_ratio", type=float, default=0.03125)
    parser.add_argument("--scoring_method", type=str, default="max",
                        choices=["mean", "max"])
    parser.add_argument("--group_agg_method", type=str, default="mean",
                        choices=["mean", "max"])
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--compressed_token_rope", type=str, default="mixed",
                        choices=["mixed", "block_center"])
    parser.add_argument("--continuous_rope", action="store_true")
    parser.add_argument("--no_triton", action="store_true")
    parser.add_argument("--comp_kv_quant", type=str, default="none",
                        choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    parser.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                        choices=["per_page", "per_comp_token"])

    # eval_ruler.py expects these on args even though AIME doesn't loop over seq_lengths.
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[32768])

    # ShadowKV baseline params (here for argparse parity only)
    parser.add_argument("--shadowkv_cache_mode", type=str, default="shadowkv_cpu",
                        choices=["shadowkv", "shadowkv_cpu"])
    parser.add_argument("--sparse_budget", type=int, default=2192)
    parser.add_argument("--rank", type=int, default=160)
    parser.add_argument("--chunk_size", type=int, default=8)

    # InfLLM baseline params (only used when --mode inf_llm). Llama 3.x only.
    parser.add_argument("--inf_llm_n_init", type=int, default=128,
                        help="InfLLM: sink token count.")
    parser.add_argument("--inf_llm_repr_topk", type=int, default=4,
                        help="InfLLM: representative tokens per block.")
    parser.add_argument("--inf_llm_max_cached_block", type=int, default=128,
                        help="InfLLM: GPU block cache size. MUST be >= "
                             "INF_LLM_CONFIG['topk'] (default 64); upstream "
                             "MemoryCache.alloc hard-asserts this.")
    parser.add_argument("--inf_llm_chunk_size", type=int, default=8192,
                        help="InfLLM: prefill chunk size for GreedySearch.")

    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    _assert_mode_model_compatible(args.mode, args.base_model)

    if args.num_repeats != 1:
        raise NotImplementedError("--num_repeats > 1 (pass@k) is not implemented yet.")

    if args.run_name is None:
        tag = model_name_tag(args.base_model)
        suffix = "aime25"
        if args.mode == "baseline":
            args.run_name = f"{tag}_baseline_{suffix}"
        elif args.mode == "page_attention":
            args.run_name = (f"{tag}_page_attn_topk{args.top_k}_cr{args.compress_ratio}"
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
        summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
        if os.path.exists(summary_path):
            print(f"SKIP (already exists): {summary_path}")
            sys.exit(0)

    # Diagnostic short-circuit: --num_samples 0 prints the auto-generated
    # run_name to stdout and exits 0 without loading the model. Mirrors the
    # --skip_existing short-circuit above.
    if args.num_samples == 0:
        print(args.run_name)
        sys.exit(0)

    return args


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, dataset, args):
    model.eval()

    output_path = os.path.join(args.output_dir, f"{args.run_name}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

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
    correct = 0
    total = 0

    for idx, item in enumerate(tqdm(samples, desc="AIME25")):
        sample_id = f"{idx}"
        if sample_id in completed_ids:
            continue

        prompt_text, gold = format_aime25_sample(item)

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
                # InfLLM uses a stateful ContextManager KV cache that HF generate()
                # cannot round-trip. Use the GreedySearch adapter built in
                # load_model_and_tokenizer. AIME extracts numeric \boxed{...}, so
                # we omit extra_end_token_ids (mirrors eval_longbench_v2.py:280-283).
                output_ids = args._inf_llm_generator.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

        generated_ids = output_ids[0, input_len:]
        if args.mode == "inf_llm":
            # ContextManager persists past_kv across samples; reset it.
            args._inf_llm_generator.clear()
        del input_ids, output_ids
        torch.cuda.empty_cache()

        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = extract_answer(response)
        is_correct = (predicted == gold) if predicted is not None else False

        result = {
            "_id": sample_id,
            "problem_id": item.get("id", item.get("ID", sample_id)),
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "response": response,
            "input_len": input_len,
        }
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

        if total == 0:
            sample_path = os.path.join(args.output_dir, f"{args.run_name}_sample0.txt")
            with open(sample_path, "w", encoding="utf-8") as sf:
                sf.write(response[:200])

        if is_correct:
            correct += 1
        total += 1

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
def print_summary(results, run_name):
    total = len(results)
    if total == 0:
        print("No results to summarise.")
        return
    correct = sum(1 for r in results if r["correct"])
    print("\n" + "=" * 60)
    print(f"AIME 2025 Results — {run_name}")
    print("=" * 60)
    print(f"  Pass@1 : {correct / total * 100:5.1f}%  ({correct}/{total})")
    no_answer = sum(1 for r in results if r["predicted"] is None)
    if no_answer > 0:
        print(f"  Failed answer extraction: {no_answer}/{total}")
    print("=" * 60)


def build_summary(results, args):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    overall_acc = round(correct / total * 100, 2) if total else 0.0

    summary = {
        "mode": args.mode,
        "model": args.base_model,
        "dataset": args.aime25_dataset,
        "run_name": args.run_name,
        "num_samples": total,
        "overall_accuracy": overall_acc,
        "extraction_failures": sum(1 for r in results if r["predicted"] is None),
        "max_new_tokens": args.max_new_tokens,
        "per_problem": [
            {"_id": r["_id"], "problem_id": r.get("problem_id", r["_id"]),
             "gold": r["gold"], "predicted": r["predicted"], "correct": r["correct"]}
            for r in results
        ],
    }

    if args.mode == "page_attention":
        summary["top_k"] = args.top_k
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
    summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.output_dir, f"{args.run_name}_summary.csv")
    rows = [{
        "group": "overall",
        "label": "overall",
        "accuracy": summary["overall_accuracy"],
        "num_samples": summary["num_samples"],
    }]
    for entry in summary["per_problem"]:
        rows.append({
            "group": "problem",
            "label": str(entry["problem_id"]),
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

    print("=" * 60)
    print(f"LOADING MODEL (mode={args.mode})")
    print("=" * 60)
    apply_monkey_patch(args)
    model, tokenizer = load_model_and_tokenizer(args)

    from datasets import load_dataset
    print(f"\nLoading AIME25: {args.aime25_dataset} (split={args.aime25_split})")
    dataset = load_dataset(args.aime25_dataset, split=args.aime25_split)
    print(f"Loaded {len(dataset)} samples")

    results = evaluate(model, tokenizer, dataset, args)
    print_summary(results, args.run_name)
    summary_path, csv_path = write_summary_files(results, args)
    print(f"\nSummary: {summary_path}")
    print(f"CSV    : {csv_path}")


if __name__ == "__main__":
    main()
