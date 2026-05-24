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

from eval_ruler import model_name_tag, _resolve_middle_top_k, _str2bool


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
                        choices=["baseline", "page_attention", "rope_gap", "seer_attention", "multipole_attention", "quest_attention", "duo_attention", "inf_llm"])

    # Model
    parser.add_argument("--base_model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_input_len", type=int, default=127500,
                        help="Truncate tokenised input if it exceeds this length")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Data
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 = all 503)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/longbench_v2")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (auto-generated if not given)")

    # Quest baseline (--mode quest_attention) — separate from DCT's --page_size/--top_k.
    # token_budget = quest_page_size * quest_top_k.
    parser.add_argument("--quest_page_size", type=int, default=32,
                        help="Quest baseline: tokens per KV page (Quest paper default: 16)")
    parser.add_argument("--quest_top_k", type=int, default=64,
                        help="Quest baseline: page budget (=token_budget/page_size). "
                             "Default 128 → token_budget=2048 with quest_page_size=16")

    # RoPE Gap params (only used when mode=rope_gap)
    parser.add_argument("--num_gaps", type=int, default=8,
                        help="Number of position gaps to insert")
    parser.add_argument("--gap_size", type=int, default=128,
                        help="Size of each position gap (multiple of page_size)")

    # DCT Page Attention params (only used when mode=page_attention)
    parser.add_argument("--page_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64,
                        help="Total selected page budget (sink + middle + recent). "
                             "DCTPageConfig receives total - sink - recent as its internal top_k.")
    parser.add_argument("--num_sink_pages", type=int, default=1)
    parser.add_argument("--num_recent_pages", type=int, default=4)
    parser.add_argument("--compress_ratio", type=float, default=0.125)
    parser.add_argument("--scoring_method", type=str, default="max",
                        choices=["mean", "max"])
    parser.add_argument("--group_agg_method", type=str, default="max",
                        choices=["mean", "max"],
                        help="How to aggregate per-head scores within a GQA group")
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--compressed_token_rope", type=str, default="mixed",
                        choices=["mixed", "block_center"],
                        help="RoPE handling for compressed tokens")
    parser.add_argument("--continuous_rope", action="store_true",
                        help="Temporarily disabled — raises error if used")
    parser.add_argument("--no_triton", action="store_true",
                        help="Disable Triton kernels (use pure PyTorch for comparison)")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="upstream_flashinfer",
        choices=["sdpa", "upstream_flashinfer"],
        help=(
            "Attention backend for page_attention mode. "
            "'sdpa' (default): assemble + torch.scaled_dot_product_attention "
            "(unchanged production path). "
            "'upstream_flashinfer': stock FlashInfer paged-decode kernel via "
            "virtual-batch-per-(batch, KV head) layout (drop mode only). "
            "Ignored for non-page_attention modes."
        ),
    )
    parser.add_argument(
        "--verify_upstream_fi",
        action="store_true",
        help=(
            "When --attention_backend upstream_flashinfer, run a per-layer SDPA "
            "shadow comparison and log the per-step max-abs-diff distribution. "
            "bf16 noise floor on this hardware is 0.05 — see project memory "
            "project_upstream_fi_multibatch.md."
        ),
    )
    parser.add_argument("--comp_kv_quant", type=str, default="fp8_e5m2",
                        choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"],
                        help="Fake-quantization of compressed K/V at write time "
                             "(precision study; no real byte-level storage change)")
    parser.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                        choices=["per_page", "per_comp_token"],
                        help="Scale granularity for comp_kv_quant")

    # Chunked prefill (useful for multipole_attention on single GPU)
    parser.add_argument("--prefill_chunk_size", type=int, default=0,
                        help="Chunk size for prefill (0 = no chunking)")

    # InfLLM baseline params (only used when --mode inf_llm). All accuracy-relevant
    # knobs (topk, block_size, n_local, n_init, repr_topk) plus max_cached_block /
    # chunk_size / exc_block_size are exposed as CLI overrides on top of
    # baselines/infllm/config.py.
    parser.add_argument("--inf_llm_topk", type=int, default=64,
                        help="InfLLM: blocks attended per decode step (main sparsity dial).")
    parser.add_argument("--inf_llm_block_size", type=int, default=32,
                        help="InfLLM: tokens per block (retrieval granularity).")
    parser.add_argument("--inf_llm_n_local", type=int, default=128,
                        help="InfLLM: sliding-window of always-attended recent tokens. "
                             "Upstream Llama-3 config uses 4096; shrinking this below ~1k "
                             "tanks retrieval scores.")
    parser.add_argument("--inf_llm_n_recent", type=int, default=None,
                        help="InfLLM: output sliding window (defaults to n_local — byte-identical "
                             "to upstream behavior). Must be <= n_local. When set < n_local, "
                             "decouples the local-output window from the block-scoring horizon "
                             "and ATTENUATES the local-attention output (output mass over keys "
                             "in [n_recent, n_local) is zeroed; softmax is NOT renormalized), "
                             "which can significantly degrade retrieval at long context.")
    parser.add_argument("--inf_llm_n_init", type=int, default=32,
                        help="InfLLM: sink token count (upstream Llama-3 default: 128).")
    parser.add_argument("--inf_llm_repr_topk", type=int, default=4,
                        help="InfLLM: representative tokens per block.")
    parser.add_argument("--inf_llm_max_cached_block", type=int, default=64,
                        help="InfLLM: GPU block cache size (must be >= --inf_llm_topk).")
    parser.add_argument("--inf_llm_chunk_size", type=int, default=8192,
                        help="InfLLM: prefill chunk size for GreedySearch.")
    parser.add_argument("--inf_llm_exc_block_size", type=int, default=None,
                        help="InfLLM: per-iteration global-attention block size during prefill. "
                             "Upstream asserts exc_block_size <= n_local. None => "
                             "min(INF_LLM_CONFIG['exc_block_size'], n_local).")

    # SeerAttention-R overrides (only used when --mode seer_attention).
    # NOTE: "seer_attention" in this script refers to the DECODE-time SeerAttention-R
    # (AttnGates) baseline, not the prefill-sparse variant. CLI takes precedence
    # over baselines/seer_attn/config.py.
    parser.add_argument("--seer_model", type=str,
                        default="SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates",
                        help="SeerAttention-R AttnGates checkpoint (HF Hub ID or local path).")
    parser.add_argument("--seerattn_sparsity_method", type=str, default="token_budget",
                        choices=["token_budget", "threshold"],
                        help="Decode sparsity method. "
                             "'token_budget' selects top-k blocks per step (k = "
                             "token_budget // gate_block_size). 'threshold' keeps blocks whose "
                             "gate softmax score exceeds SEER_ATTN_CONFIG['threshold']. "
                             "Note: 'nz_ratio' is prefill-only and rejected here.")
    parser.add_argument("--seerattn_token_budget", type=int, default=2048,
                        help="Active tokens per decode step (only used when "
                             "sparsity_method='token_budget'). Internally rounded to "
                             "block_budget = token_budget // gate_block_size; pass a multiple "
                             "of gate_block_size (default 64) to avoid truncation.")
    parser.add_argument("--seerattn_gate_block_size", type=int, default=64,
                        choices=[16, 32, 64],
                        help="SeerAttention-R gate block size (= sparse-decode tile size). "
                             "Default 64 matches the released SeerAttention/SeerAttention-Decode-* "
                             "AttnGates checkpoints. Overriding to 16 or 32 feeds the gate K-pool "
                             "reps at a different granularity than it was trained on, so accuracy "
                             "degrades. Useful only as an ablation/sanity sweep.")

    # Multipole Attention overrides (only used when --mode multipole_attention).
    # CLI takes precedence over baselines/multipole_attn/config.py.
    # use_centroids is intentionally not exposed: it is always True for this mode.
    parser.add_argument("--multipole_percent_clusters_lst", type=float, nargs="+", default=[3.125],
                        help="Multipole: percentage of keys retained as centroids per hierarchy "
                             "level (one float per level; e.g. '6.25' for single-level, "
                             "'6.25 25.0' for two-level). "
                             "3.125 => 100/3.125=32 tok/page.")
    parser.add_argument("--multipole_percentiles_lst", type=int, nargs="+", default=[2048],
                        help="Multipole: token-budget threshold per level "
                             "(same length as --multipole_percent_clusters_lst). "
                             "=> token budget.")
    parser.add_argument("--multipole_use_replacement", type=_str2bool, default=False,
                        help="Multipole: if True, unselected tokens contribute via centroid "
                             "value approximation; if False, dropped.")
    parser.add_argument("--multipole_cluster_interval", type=int, default=32,
                        help="Multipole: tokens between re-clusterings during generation.")

    args = parser.parse_args()

    if args.run_name is None:
        tag = model_name_tag(args.base_model)
        if args.mode == "baseline":
            args.run_name = f"{tag}_baseline"
        elif args.mode == "rope_gap":
            args.run_name = f"{tag}_rope_gap_{args.num_gaps}x{args.gap_size}"
        elif args.mode == "seer_attention":
            args.run_name = f"{tag}_seer_attention"
            if args.seerattn_sparsity_method is not None:
                args.run_name += f"_{args.seerattn_sparsity_method}"
            if args.seerattn_token_budget is not None:
                args.run_name += f"_b{args.seerattn_token_budget}"
            if args.seerattn_gate_block_size != 64:
                args.run_name += f"_bs{args.seerattn_gate_block_size}"
        elif args.mode == "multipole_attention":
            args.run_name = f"{tag}_multipole_attention"
            if args.multipole_percent_clusters_lst is not None:
                pct_str = "_".join(str(p) for p in args.multipole_percent_clusters_lst)
                args.run_name += f"_pct{pct_str}"
            if args.multipole_percentiles_lst is not None:
                ptl_str = "_".join(str(p) for p in args.multipole_percentiles_lst)
                args.run_name += f"_ptl{ptl_str}"
            if args.multipole_use_replacement is not None:
                args.run_name += f"_repl{args.multipole_use_replacement}"
            if args.multipole_cluster_interval is not None:
                args.run_name += f"_ci{args.multipole_cluster_interval}"
        elif args.mode == "quest_attention":
            args.run_name = f"{tag}_quest_ps{args.quest_page_size}_pb{args.quest_top_k}"
        elif args.mode == "duo_attention":
            args.run_name = f"{tag}_duo_attention"
        elif args.mode == "inf_llm":
            args.run_name = (f"{tag}_inf_llm_topk{args.inf_llm_topk}"
                             f"_bs{args.inf_llm_block_size}"
                             f"_nlocal{args.inf_llm_n_local}"
                             f"_nini{args.inf_llm_n_init}"
                             f"_repr{args.inf_llm_repr_topk}")
            if args.inf_llm_n_recent is not None:
                args.run_name += f"_nrec{args.inf_llm_n_recent}"
        else:
            args.run_name = f"{tag}_page_attn_topk{args.top_k}T_{args.comp_kv_quant}"

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

    min_len_for_paging = (args.top_k + 1) * args.page_size
    if kv_len < min_len_for_paging:
        return kv_len

    sink_tokens = args.num_sink_pages * args.page_size
    recent_tokens_min = args.num_recent_pages * args.page_size
    pageable_len = kv_len - sink_tokens - recent_tokens_min
    num_pages = pageable_len // args.page_size
    actual_recent = kv_len - sink_tokens - num_pages * args.page_size
    top_k = min(args.top_k - args.num_sink_pages - args.num_recent_pages, num_pages)

    if args.unselected_mode == "drop":
        return sink_tokens + top_k * args.page_size + actual_recent
    elif args.unselected_mode == "compressed":
        comp_size = max(1, int(args.page_size * args.compress_ratio))
        num_unselected = num_pages - top_k
        return sink_tokens + top_k * args.page_size + num_unselected * comp_size + actual_recent
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

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "results.jsonl")

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
            elif args.mode == "inf_llm":
                # InfLLM uses a stateful ContextManager KV cache that HF
                # generate() cannot round-trip. Use our GreedySearch adapter.
                output_ids = args._inf_llm_generator.generate(
                    input_ids,
                    max_new_tokens=max_gen,
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
                    max_new_tokens=max_gen,
                    on_post_generate=_harvest_verify_diffs,
                    do_sample=False,
                    use_cache=True,
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
        elif args.mode == "inf_llm":
            # ContextManager persists past_kv across samples; reset it.
            args._inf_llm_generator.clear()

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
        summary["middle_top_k"] = _resolve_middle_top_k(args)
        summary["page_size"] = args.page_size
        summary["scoring_method"] = args.scoring_method
        summary["group_agg_method"] = args.group_agg_method
        summary["unselected_mode"] = args.unselected_mode
    elif args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        summary["seer_attn_config"] = dict(SEER_ATTN_CONFIG)
        if args.seer_model is not None:
            summary["seer_attn_config"]["seer_model"] = args.seer_model
        if args.seerattn_sparsity_method is not None:
            summary["seer_attn_config"]["sparsity_method"] = args.seerattn_sparsity_method
        if args.seerattn_token_budget is not None:
            summary["seer_attn_config"]["token_budget"] = args.seerattn_token_budget
        if args.seerattn_gate_block_size is not None:
            summary["seer_attn_config"]["seerattn_gate_block_size"] = args.seerattn_gate_block_size
    elif args.mode == "multipole_attention":
        summary["multipole_attn_config"] = getattr(args, "_multipole_cfg", None)
    elif args.mode == "quest_attention":
        from quest_attn.config import QUEST_ATTN_CONFIG
        summary["quest_attn_config"] = {
            **QUEST_ATTN_CONFIG,
            "page_size": args.quest_page_size,
            "page_budget": args.quest_top_k,
            "token_budget": args.quest_page_size * args.quest_top_k,
        }
    elif args.mode == "duo_attention":
        from duo_attn.config import DUO_ATTN_CONFIG
        summary["duo_attn_config"] = DUO_ATTN_CONFIG
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
# Upstream-FlashInfer preflight validation
# ---------------------------------------------------------------------------
# NOTE: keep this function name/logic IDENTICAL across eval_ruler.py,
# eval_longbench_v1.py, and eval_longbench_v2.py so future maintainers can grep.
def _validate_upstream_fi_args(args):
    """Hard preflights for --attention_backend upstream_flashinfer.

    Three checks (all hard errors, no silent downgrade):
      1. compressed-mode reject: upstream FI does not implement compressed mode.
      2. 64K memory preflight: refuses to start when projected paged-KV exceeds
         0.9 * total GPU memory (uses approximate model dims for known families).
      3. greedy-only assert: beam search / sampling break the lazy-init contract.
    """
    backend = getattr(args, "attention_backend", "sdpa")
    if backend != "upstream_flashinfer":
        return
    if getattr(args, "mode", None) != "page_attention":
        return

    # 1. Compressed-mode hard error.
    if getattr(args, "unselected_mode", "drop") != "drop":
        raise SystemExit(
            f"--attention_backend upstream_flashinfer requires "
            f"--unselected_mode drop (got {args.unselected_mode!r}). "
            f"Use --attention_backend sdpa for compressed mode. "
            f"To sweep both modes, run two passes."
        )

    # 2. Greedy-only hard assert. Eval scripts default to greedy; this is defensive.
    do_sample = getattr(args, "do_sample", False)
    num_beams = getattr(args, "num_beams", 1)
    num_return_sequences = getattr(args, "num_return_sequences", 1)
    if do_sample or num_beams > 1 or num_return_sequences > 1:
        raise SystemExit(
            f"--attention_backend upstream_flashinfer requires greedy decoding "
            f"(do_sample=False, num_beams=1, num_return_sequences=1). "
            f"Got do_sample={do_sample}, num_beams={num_beams}, "
            f"num_return_sequences={num_return_sequences}. "
            f"The lazy-init cache contract assumes one forward per decode step."
        )

    # 3. 64K paged-KV memory preflight. Use approximate dims for known families.
    # LongBench v2: prompts can run up to args.max_input_len tokens.
    max_seq_len = int(getattr(args, "max_input_len", 0) or 0)
    if max_seq_len <= 0:
        return
    max_decode_steps = max(int(getattr(args, "max_new_tokens", 0)), 256)
    page_size = int(getattr(args, "page_size", 32))
    num_sink_pages = int(getattr(args, "num_sink_pages", 1))
    num_recent_pages = int(getattr(args, "num_recent_pages", 5))

    name = (getattr(args, "base_model", "") or "").lower()
    if "llama" in name:
        num_kv_heads, num_layers, head_dim = 8, 32, 128
    elif "qwen3" in name:
        num_kv_heads, num_layers, head_dim = 8, 36, 128
    else:
        print(f"  [preflight] WARNING: unknown model family for {name!r}; "
              f"skipping upstream-FI memory preflight.")
        return

    pages_per_head = (
        math.ceil((max_seq_len + max_decode_steps) / page_size)
        + num_sink_pages
        + (num_recent_pages + 1)
    )
    bsz = 1  # eval scripts process one sample at a time
    proj_bytes = (
        bsz * num_kv_heads * pages_per_head * 2 * page_size * head_dim * 2 * num_layers
    )
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception as e:
        print(f"  [preflight] WARNING: could not query GPU memory ({e}); "
              f"skipping upstream-FI memory preflight.")
        return
    threshold = 0.9 * total_bytes
    if proj_bytes > threshold:
        raise SystemExit(
            f"--attention_backend upstream_flashinfer projected paged-KV memory "
            f"({proj_bytes / 1e9:.2f} GiB) exceeds 0.9 * total GPU memory "
            f"({threshold / 1e9:.2f} GiB of {total_bytes / 1e9:.2f} GiB). "
            f"Reduce --max_input_len (={max_seq_len}) or run on a larger GPU."
        )
    print(f"  [preflight] upstream-FI projected paged-KV: "
          f"{proj_bytes / 1e9:.2f} GiB (threshold {threshold / 1e9:.2f} GiB) — OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    _validate_upstream_fi_args(args)

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
                top_k=_resolve_middle_top_k(args),
                num_sink_pages=args.num_sink_pages,
                num_recent_pages=args.num_recent_pages,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=True,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
                attention_backend=args.attention_backend,
            )
        elif "qwen3" in model_name_lower:
            from dct_page_attention import replace_qwen3_attn
            replace_qwen3_attn(
                page_size=args.page_size,
                top_k=_resolve_middle_top_k(args),
                num_sink_pages=args.num_sink_pages,
                num_recent_pages=args.num_recent_pages,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=True,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
                attention_backend=args.attention_backend,
            )
        else:
            from dct_page_attention import replace_qwen2_attn
            replace_qwen2_attn(
                page_size=args.page_size,
                top_k=_resolve_middle_top_k(args),
                num_sink_pages=args.num_sink_pages,
                num_recent_pages=args.num_recent_pages,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=True,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
                attention_backend=args.attention_backend,
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
        cfg = dict(MULTIPOLE_ATTN_CONFIG)
        cfg["base_model"] = args.base_model
        if args.multipole_percent_clusters_lst is not None:
            cfg["percent_clusters_lst"] = list(args.multipole_percent_clusters_lst)
        if args.multipole_percentiles_lst is not None:
            cfg["percentiles_lst"] = list(args.multipole_percentiles_lst)
        if args.multipole_use_replacement is not None:
            cfg["use_replacement"] = args.multipole_use_replacement
        if args.multipole_cluster_interval is not None:
            cfg["cluster_interval"] = args.multipole_cluster_interval
        args._multipole_cfg = cfg
        replace_attn_multipole(cfg)
    elif args.mode == "duo_attention":
        pass  # DuoAttention patches per-instance forwards post-load
    elif args.mode == "inf_llm":
        pass  # InfLLM patches per-instance forwards post-load
    elif args.mode not in ("seer_attention", "quest_attention"):
        print("Baseline mode: full attention (no monkey-patch)")

    # Load tokenizer + model
    if args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        from seer_attn import SeerDecodingQwen3ForCausalLM

        seer_model = args.seer_model or SEER_ATTN_CONFIG["seer_model"]
        sparsity_method = args.seerattn_sparsity_method or SEER_ATTN_CONFIG["sparsity_method"]
        token_budget = (args.seerattn_token_budget
                        if args.seerattn_token_budget is not None
                        else SEER_ATTN_CONFIG["token_budget"])
        print(f"Loading SeerAttention-R model: {seer_model}")
        if args.seer_model is not None:
            print(f"  Override: seer_model={args.seer_model}")
        if args.seerattn_sparsity_method is not None:
            print(f"  Override: seerattn_sparsity_method={args.seerattn_sparsity_method}")
        if args.seerattn_token_budget is not None:
            print(f"  Override: seerattn_token_budget={args.seerattn_token_budget}")
        if args.seerattn_gate_block_size != 64:
            print(f"  Override: seerattn_gate_block_size={args.seerattn_gate_block_size} "
                  f"(checkpoint default is 64; ablation only)")
        model = SeerDecodingQwen3ForCausalLM.from_pretrained(
            seer_model,
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method=sparsity_method,
            seerattn_token_budget=token_budget,
            seerattn_threshold=SEER_ATTN_CONFIG["threshold"],
            seerattn_start_layer=SEER_ATTN_CONFIG["start_layer"],
            seerattn_gate_block_size=args.seerattn_gate_block_size,
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

        base_model = QUEST_ATTN_CONFIG["base_model"]
        page_size = args.quest_page_size
        token_budget = args.quest_page_size * args.quest_top_k  # quest_top_k = page_budget
        model_name_lower = base_model.lower()
        if "qwen3" in model_name_lower:
            from quest_attn import Qwen3ForCausalLM as QuestModel
        elif any(fam in model_name_lower for fam in ["llama", "mistral"]):
            from quest_attn import LlamaForCausalLM as QuestModel
        else:
            raise ValueError(
                f"Quest supports LLaMA-family (Llama-2, Llama-3.x, Mistral) and Qwen3 models, "
                f"got: {base_model}"
            )
        print(f"Loading Quest model: {base_model} (page_size={page_size}, page_budget={args.quest_top_k}, token_budget={token_budget})")
        model = QuestModel.from_pretrained(
            base_model,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        model.quest_init(
            page_size=page_size,
            max_seq_len=QUEST_ATTN_CONFIG["max_seq_len"],
            token_budget=token_budget,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    else:
        # DuoAttention's and InfLLM's replacement forwards assume eager-style Q/K/V signatures.
        attn_impl = "eager" if args.mode in {"duo_attention", "inf_llm"} else "sdpa"
        print(f"Loading model: {args.base_model} (attn: {attn_impl})")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # Multipole attention requires all layers on a single GPU (the original
        # MultipoleAttention repo uses model.to(device)); device_map="auto"
        # spreads layers across GPUs and breaks per-layer clustering state.
        device_map = "cuda:0" if args.mode == "multipole_attention" else "auto"
        yarn_kwargs = {}
        # InfLLM overrides RoPE and is Llama-only, so skip Qwen3-yarn injection
        # (its old-transformers env also doesn't accept rope_parameters=).
        if "qwen3" in args.base_model.lower() and args.mode != "inf_llm":
            yarn_kwargs = {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 1000000.0,
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                },
                "max_position_embeddings": 131072,
            }
        # duo_attention is pinned to transformers 4.45 (torch_dtype=); everything else
        # (including inf_llm post-5.2.0 migration) is on the main DCT-Page env (dtype=).
        dtype_kwarg = (
            {"torch_dtype": torch.bfloat16}
            if args.mode == "duo_attention"
            else {"dtype": torch.bfloat16}
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            **dtype_kwarg,
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
            from duo_attn import init_duo_attention, assert_llama
            from duo_attn.config import DUO_ATTN_CONFIG
            assert_llama(args.base_model)
            DUO_ATTN_CONFIG["base_model"] = args.base_model
            init_duo_attention(model, DUO_ATTN_CONFIG)

        if args.mode == "inf_llm":
            from infllm import (
                assert_llama_only,
                build_inf_llm_generator,
                init_inf_llm,
            )
            from infllm.config import INF_LLM_CONFIG
            assert_llama_only(args.base_model)
            INF_LLM_CONFIG["base_model"] = args.base_model
            INF_LLM_CONFIG["topk"] = args.inf_llm_topk
            INF_LLM_CONFIG["block_size"] = args.inf_llm_block_size
            INF_LLM_CONFIG["n_local"] = args.inf_llm_n_local
            INF_LLM_CONFIG["n_init"] = args.inf_llm_n_init
            INF_LLM_CONFIG["repr_topk"] = args.inf_llm_repr_topk
            INF_LLM_CONFIG["max_cached_block"] = args.inf_llm_max_cached_block
            INF_LLM_CONFIG["chunk_size"] = args.inf_llm_chunk_size
            INF_LLM_CONFIG["n_recent"] = args.inf_llm_n_recent
            requested_exc = (args.inf_llm_exc_block_size
                             if args.inf_llm_exc_block_size is not None
                             else INF_LLM_CONFIG["exc_block_size"])
            INF_LLM_CONFIG["exc_block_size"] = min(requested_exc, args.inf_llm_n_local)
            init_inf_llm(model, INF_LLM_CONFIG)
            args._inf_llm_generator = build_inf_llm_generator(model, tokenizer, INF_LLM_CONFIG)

    # Seed per-attention-module `_upstream_fi_build_kwargs` AFTER from_pretrained.
    # `replace_*_attn` runs BEFORE model load by project convention, so this
    # post-load walk is the only safe time to stash per-instance state.
    if args.mode == "page_attention" and getattr(args, "attention_backend", "sdpa") == "upstream_flashinfer":
        from dct_page_attention import _init_upstream_fi_build_kwargs
        _init_upstream_fi_build_kwargs(model)

    # Evaluate
    results = evaluate(model, tokenizer, dataset, args)

    # Print summary
    print_summary(results, args.run_name)

    summary_path, csv_path = write_summary_files(results, args)

    print(f"\nResults saved to: {os.path.join(args.output_dir, args.run_name, 'results.jsonl')}")
    print(f"Summary saved to: {summary_path}")
    print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
