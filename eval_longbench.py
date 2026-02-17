"""
LongBench v2 evaluation for DCT Page Attention.

Compares baseline (full attention) vs page attention on 503 multiple-choice
questions. Reports accuracy overall, by difficulty (easy/hard), and by
context length (short/medium/long).
"""

import os
import sys
import json
import re
import math
import argparse
import random

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Prompt template (matches official LongBench v2 0shot.txt)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)"."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LongBench v2 Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "page_attention", "rope_gap"])

    # Model
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_input_len", type=int, default=120000,
                        help="Truncate tokenised input if it exceeds this length")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Data
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 = all 503)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_longbench")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (auto-generated if not given)")

    # RoPE Gap params (only used when mode=rope_gap)
    parser.add_argument("--num_gaps", type=int, default=8,
                        help="Number of position gaps to insert")
    parser.add_argument("--gap_size", type=int, default=128,
                        help="Size of each position gap (multiple of page_size)")

    # DCT Page Attention params (only used when mode=page_attention)
    parser.add_argument("--page_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--compress_ratio", type=float, default=0.25)
    parser.add_argument("--scoring_method", type=str, default="max",
                        choices=["mean", "max", "sum"])
    parser.add_argument("--group_agg_method", type=str, default="mean",
                        choices=["mean", "max", "topp"],
                        help="How to aggregate per-head scores within a GQA group")
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--continuous_rope", action="store_true",
                        help="Store KV before RoPE, apply continuous RoPE after assembly")

    args = parser.parse_args()

    if args.run_name is None:
        if args.mode == "baseline":
            args.run_name = "baseline"
        elif args.mode == "rope_gap":
            args.run_name = f"rope_gap_{args.num_gaps}x{args.gap_size}"
        else:
            args.run_name = f"page_attn_topk{args.top_k}"

    return args


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(item):
    return PROMPT_TEMPLATE.format(
        context=item["context"].strip(),
        question=item["question"].strip(),
        choice_A=item["choice_A"].strip(),
        choice_B=item["choice_B"].strip(),
        choice_C=item["choice_C"].strip(),
        choice_D=item["choice_D"].strip(),
    )


# ---------------------------------------------------------------------------
# Tokenise + truncate (official LongBench v2 approach: first-half + last-half)
# ---------------------------------------------------------------------------
def tokenize_and_truncate(prompt_text, tokenizer, max_input_len):
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Some transformers versions return BatchEncoding instead of a tensor
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    if input_ids.shape[1] > max_input_len:
        half = max_input_len // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
    return input_ids


# ---------------------------------------------------------------------------
# Answer extraction (verbatim from official LongBench v2 pred.py)
# ---------------------------------------------------------------------------
def compute_effective_len(input_len, args):
    """Compute effective KV length during decode after page selection (first decode step)."""
    if args.mode != "page_attention":
        return input_len

    kv_len = input_len + 1  # first decode step adds one token to cache

    min_len_for_paging = args.sink_size + args.page_size * (args.top_k + 1) + args.recent_size
    if kv_len < min_len_for_paging:
        return kv_len

    pageable_len = kv_len - args.sink_size - args.recent_size
    num_pages = pageable_len // args.page_size
    leftover = pageable_len % args.page_size
    actual_recent = args.recent_size + leftover
    top_k = min(args.top_k, num_pages)

    if args.unselected_mode == "drop":
        return args.sink_size + top_k * args.page_size + actual_recent
    elif args.unselected_mode == "compressed":
        comp_size = max(1, int(args.page_size * args.compress_ratio))
        num_unselected = num_pages - top_k
        return args.sink_size + top_k * args.page_size + num_unselected * comp_size + actual_recent
    else:
        return kv_len


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, dataset, args):
    model.eval()

    output_path = os.path.join(args.output_dir, f"{args.run_name}.jsonl")

    # Resume support: skip already-completed samples
    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                r = json.loads(line)
                completed_ids.add(r["_id"])
        print(f"Resuming: {len(completed_ids)} samples already completed")

    samples = list(dataset)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    out_f = open(output_path, "a")

    correct = 0
    total = 0

    for item in tqdm(samples, desc="Evaluating"):
        if item["_id"] in completed_ids:
            continue

        prompt_text = build_prompt(item)
        input_ids = tokenize_and_truncate(prompt_text, tokenizer, args.max_input_len)
        input_ids = input_ids.to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = output_ids[0, input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predicted = extract_answer(response)
        gold = item["answer"]
        is_correct = (predicted == gold) if predicted else False

        result = {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "response": response,
            "input_len": input_len,
            "effective_len": compute_effective_len(input_len, args),
        }

        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

        if is_correct:
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  Progress: {total} done, accuracy so far: {correct / total * 100:.1f}%")

    out_f.close()

    # Reload all results (including previously completed) for final stats
    results = []
    with open(output_path, "r") as f:
        for line in f:
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

    def acc(subset):
        if not subset:
            return 0.0, 0, 0
        c = sum(1 for r in subset if r["correct"])
        return c / len(subset) * 100, c, len(subset)

    easy = [r for r in results if r["difficulty"] == "easy"]
    hard = [r for r in results if r["difficulty"] == "hard"]
    short = [r for r in results if r["length"] == "short"]
    medium = [r for r in results if r["length"] == "medium"]
    long_ = [r for r in results if r["length"] == "long"]

    print("\n" + "=" * 60)
    print(f"LongBench v2 Results — {run_name}")
    print("=" * 60)
    for label, subset in [("Overall", results), ("Easy", easy), ("Hard", hard),
                          ("Short", short), ("Medium", medium), ("Long", long_)]:
        a, c, n = acc(subset)
        print(f"  {label:8s}: {a:5.1f}%  ({c}/{n})")

    no_answer = sum(1 for r in results if r["predicted"] is None)
    if no_answer > 0:
        print(f"  Failed answer extraction: {no_answer}/{total}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load dataset
    print("Loading LongBench v2 dataset...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    print(f"Loaded {len(dataset)} samples")

    # Conditionally apply monkey-patch
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
            )
    elif args.mode == "rope_gap":
        from rope_gap_attention import replace_qwen2_with_rope_gaps
        replace_qwen2_with_rope_gaps(
            num_gaps=args.num_gaps,
            gap_size=args.gap_size,
        )
    else:
        print("Baseline mode: full attention (no monkey-patch)")

    # Use sdpa for both modes — prefill needs memory-efficient attention for
    # long sequences; the DCT page attention monkey-patch handles decode itself.
    attn_impl = "sdpa"

    # Load tokenizer + model
    print(f"Loading model: {args.base_model} (attn: {attn_impl})")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Evaluate
    results = evaluate(model, tokenizer, dataset, args)

    # Print summary
    print_summary(results, args.run_name)

    # # Save summary JSON
    # summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    # overall_acc = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else 0
    # summary = {
    #     "mode": args.mode,
    #     "model": args.base_model,
    #     "run_name": args.run_name,
    #     "num_samples": len(results),
    #     "overall_accuracy": round(overall_acc, 2),
    # }
    # if args.mode == "page_attention":
    #     summary["top_k"] = args.top_k
    #     summary["page_size"] = args.page_size
    #     summary["scoring_method"] = args.scoring_method
    #     summary["group_agg_method"] = args.group_agg_method
    #     summary["unselected_mode"] = args.unselected_mode
    # with open(summary_path, "w") as f:
    #     json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {os.path.join(args.output_dir, args.run_name + '.jsonl')}")
    # print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
