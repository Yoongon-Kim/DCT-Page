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
import csv

# Ensure baselines/ packages (seer_attn, multipole_attn, quest_attn) are importable
_BASELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ruler import model_name_tag


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
                        choices=["baseline", "page_attention", "rope_gap", "seer_attention", "multipole_attention", "quest_attention", "duo_attention"])

    # Model
    parser.add_argument("--base_model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_input_len", type=int, default=120000,
                        help="Truncate tokenised input if it exceeds this length")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Data
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 = all 503)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_longbench_v2/page_attention")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (auto-generated if not given)")

    # RoPE Gap params (only used when mode=rope_gap)
    parser.add_argument("--num_gaps", type=int, default=8,
                        help="Number of position gaps to insert")
    parser.add_argument("--gap_size", type=int, default=128,
                        help="Size of each position gap (multiple of page_size)")

    # DCT Page Attention params (only used when mode=page_attention)
    parser.add_argument("--page_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--compress_ratio", type=float, default=0.03125)
    parser.add_argument("--scoring_method", type=str, default="max",
                        help="'mean'|'max'|'sum'|'proxy_dc_ac_{lam}'|'spread_dc_ac_{lam}'")
    parser.add_argument("--group_agg_method", type=str, default="mean",
                        choices=["mean", "max", "topp"],
                        help="How to aggregate per-head scores within a GQA group")
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--compression_method", type=str, default="haar",
                        choices=["haar", "dct"],
                        help="Compression method for unselected pages (used when unselected_mode=compressed)")
    parser.add_argument("--compressed_token_rope", type=str, default="mixed",
                        choices=["mixed", "block_center"],
                        help="RoPE handling for compressed tokens")
    parser.add_argument("--continuous_rope", action="store_true",
                        help="Temporarily disabled — raises error if used")
    parser.add_argument("--weight_compressed_by_population", action="store_true",
                        help="In compressed mode, scale each unselected-page rep's softmax mass "
                             "by page_size/comp_size via a log(n) bias on QK logits "
                             "(multipole-style population weighting). No-op for drop mode.")
    parser.add_argument("--no_triton", action="store_true",
                        help="Disable Triton kernels (use pure PyTorch for comparison)")
    parser.add_argument("--comp_kv_quant", type=str, default="none",
                        choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"],
                        help="Fake-quantization of compressed K/V at write time "
                             "(precision study; no real byte-level storage change)")
    parser.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                        choices=["per_page", "per_comp_token"],
                        help="Scale granularity for comp_kv_quant")

    # Chunked prefill (useful for multipole_attention on single GPU)
    parser.add_argument("--prefill_chunk_size", type=int, default=0,
                        help="Chunk size for prefill (0 = no chunking)")

    args = parser.parse_args()

    if args.run_name is None:
        tag = model_name_tag(args.base_model)
        if args.mode == "baseline":
            args.run_name = f"{tag}_baseline"
        elif args.mode == "rope_gap":
            args.run_name = f"{tag}_rope_gap_{args.num_gaps}x{args.gap_size}"
        elif args.mode == "seer_attention":
            args.run_name = f"{tag}_seer_attention"
        elif args.mode == "multipole_attention":
            args.run_name = f"{tag}_multipole_attention"
        elif args.mode == "quest_attention":
            args.run_name = f"{tag}_quest_attention"
        elif args.mode == "duo_attention":
            args.run_name = f"{tag}_duo_attention"
        else:
            args.run_name = f"{tag}_page_attn_topk{args.top_k}"

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
    chat_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(tokenizer, "chat_template") and "enable_thinking" in (tokenizer.chat_template or ""):
        chat_kwargs["enable_thinking"] = False
    input_ids = tokenizer.apply_chat_template(
        messages,
        **chat_kwargs,
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

    max_gen = args.max_new_tokens

    for item in tqdm(samples, desc="Evaluating"):
        if item["_id"] in completed_ids:
            continue

        prompt_text = build_prompt(item)
        input_ids = tokenize_and_truncate(prompt_text, tokenizer, args.max_input_len)
        input_ids = input_ids.to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            if args.mode == "seer_attention":
                output_ids, _ = model.batch_exist_generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_length=input_len + max_gen,
                    do_sample=False,
                )
            else:
                gen_kwargs = dict(
                    max_new_tokens=max_gen,
                    do_sample=False,
                    use_cache=True,
                )
                if args.prefill_chunk_size > 0:
                    gen_kwargs["prefill_chunk_size"] = args.prefill_chunk_size
                output_ids = model.generate(input_ids, **gen_kwargs)

        generated_ids = output_ids[0, input_len:]
        del input_ids, output_ids
        torch.cuda.empty_cache()

        if args.mode == "quest_attention":
            model.quest_clear()

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


def build_summary(results, args):
    total = len(results)

    def subset_stats(records):
        if not records:
            return {"accuracy": 0.0, "num_samples": 0}
        correct = sum(1 for r in records if r["correct"])
        return {
            "accuracy": round(correct / len(records) * 100, 2),
            "num_samples": len(records),
        }

    easy = [r for r in results if r["difficulty"] == "easy"]
    hard = [r for r in results if r["difficulty"] == "hard"]
    short = [r for r in results if r["length"] == "short"]
    medium = [r for r in results if r["length"] == "medium"]
    long_ = [r for r in results if r["length"] == "long"]

    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)

    overall_acc = round((sum(1 for r in results if r["correct"]) / total * 100), 2) if total else 0.0

    summary = {
        "mode": args.mode,
        "model": args.base_model,
        "run_name": args.run_name,
        "num_samples": total,
        "overall_accuracy": overall_acc,
        "by_difficulty": {
            "easy": subset_stats(easy),
            "hard": subset_stats(hard),
        },
        "by_length": {
            "short": subset_stats(short),
            "medium": subset_stats(medium),
            "long": subset_stats(long_),
        },
        "by_domain": {
            domain: subset_stats(records)
            for domain, records in sorted(by_domain.items())
        },
    }

    if args.mode == "page_attention":
        summary["top_k"] = args.top_k
        summary["page_size"] = args.page_size
        summary["scoring_method"] = args.scoring_method
        summary["group_agg_method"] = args.group_agg_method
        summary["unselected_mode"] = args.unselected_mode
    elif args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        summary["seer_attn_config"] = SEER_ATTN_CONFIG
    elif args.mode == "multipole_attention":
        from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
        summary["multipole_attn_config"] = MULTIPOLE_ATTN_CONFIG
    elif args.mode == "quest_attention":
        from quest_attn.config import QUEST_ATTN_CONFIG
        summary["quest_attn_config"] = QUEST_ATTN_CONFIG
    elif args.mode == "duo_attention":
        from duo_attn_baseline.config import DUO_ATTN_CONFIG
        summary["duo_attn_config"] = DUO_ATTN_CONFIG

    return summary


def write_summary_files(results, args):
    summary = build_summary(results, args)
    summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.output_dir, f"{args.run_name}_summary.csv")
    rows = [
        {
            "group": "overall",
            "label": "overall",
            "accuracy": summary["overall_accuracy"],
            "num_samples": summary["num_samples"],
        }
    ]
    for group_name in ("by_difficulty", "by_length", "by_domain"):
        for label, payload in summary[group_name].items():
            rows.append(
                {
                    "group": group_name.removeprefix("by_"),
                    "label": label,
                    "accuracy": payload["accuracy"],
                    "num_samples": payload["num_samples"],
                }
            )

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
                compression_method=args.compression_method,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=args.weight_compressed_by_population,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
            )
        elif "qwen3" in model_name_lower:
            from dct_page_attention import replace_qwen3_attn
            replace_qwen3_attn(
                page_size=args.page_size,
                top_k=args.top_k,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compression_method=args.compression_method,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=args.weight_compressed_by_population,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
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
                compression_method=args.compression_method,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=args.weight_compressed_by_population,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
            )
    elif args.mode == "rope_gap":
        from rope_gap_attention import replace_qwen2_with_rope_gaps
        replace_qwen2_with_rope_gaps(
            num_gaps=args.num_gaps,
            gap_size=args.gap_size,
        )
    elif args.mode == "multipole_attention":
        from multipole_attn import replace_attn_multipole
        from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
        MULTIPOLE_ATTN_CONFIG["base_model"] = args.base_model
        replace_attn_multipole(MULTIPOLE_ATTN_CONFIG)
    elif args.mode == "duo_attention":
        pass  # DuoAttention patches per-instance forwards post-load
    elif args.mode not in ("seer_attention", "quest_attention"):
        print("Baseline mode: full attention (no monkey-patch)")

    # Load tokenizer + model
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
        tokenizer = AutoTokenizer.from_pretrained(model.config.base_model)
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    elif args.mode == "quest_attention":
        from quest_attn.config import QUEST_ATTN_CONFIG
        from quest_attn import LlamaForCausalLM as QuestLlamaForCausalLM

        base_model = QUEST_ATTN_CONFIG["base_model"]
        model_name_lower = base_model.lower()
        if not any(fam in model_name_lower for fam in ["llama", "mistral"]):
            raise ValueError(
                f"Quest only supports LLaMA-family models (Llama-2, Llama-3.x, Mistral), "
                f"got: {base_model}"
            )
        print(f"Loading Quest model: {base_model}")
        model = QuestLlamaForCausalLM.from_pretrained(
            base_model,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        model.quest_init(
            page_size=QUEST_ATTN_CONFIG["page_size"],
            max_seq_len=QUEST_ATTN_CONFIG["max_seq_len"],
            token_budget=QUEST_ATTN_CONFIG["token_budget"],
            dtype=torch.float16,
            device=torch.device("cuda:0"),
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    else:
        # DuoAttention's replacement forward assumes eager-style Q/K/V signatures.
        attn_impl = "eager" if args.mode == "duo_attention" else "sdpa"
        print(f"Loading model: {args.base_model} (attn: {attn_impl})")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # Multipole attention requires all layers on a single GPU (the original
        # MultipoleAttention repo uses model.to(device)); device_map="auto"
        # spreads layers across GPUs and breaks per-layer clustering state.
        device_map = "cuda:0" if args.mode == "multipole_attention" else "auto"
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
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn_impl,
            **yarn_kwargs,
        )
        model.eval()
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        if args.mode == "multipole_attention":
            from multipole_attn import init_multipole_layers
            init_multipole_layers(model)
            print("Multipole attention layers initialized.")

        if args.mode == "duo_attention":
            from duo_attn_baseline import init_duo_attention, assert_llama
            from duo_attn_baseline.config import DUO_ATTN_CONFIG
            assert_llama(args.base_model)
            DUO_ATTN_CONFIG["base_model"] = args.base_model
            init_duo_attention(model, DUO_ATTN_CONFIG)

    # Evaluate
    results = evaluate(model, tokenizer, dataset, args)

    # Print summary
    print_summary(results, args.run_name)

    summary_path, csv_path = write_summary_files(results, args)

    print(f"\nResults saved to: {os.path.join(args.output_dir, args.run_name + '.jsonl')}")
    print(f"Summary saved to: {summary_path}")
    print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
