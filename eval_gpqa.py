"""
GPQA evaluation for DCT Page Attention and baseline attention mechanisms.

Evaluates Qwen3-8B (the only supported reasoning model in this harness) on the
GPQA multiple-choice benchmark across the same eight attention modes exposed by
eval_ruler.py. Attention modes that do not support Qwen3 (seer_prefill,
quest_attention, duo_attention, shadowkv) are still listed in --mode for parity,
but the script raises ValueError before model load if one is selected.

Default subset is gpqa_diamond (198 hardest questions); --gpqa_subset switches
to main or extended. Prompting is chain-of-thought; answers are extracted as
A/B/C/D via regex, scored with exact match.

Outputs:
    {output_dir}/{run_name}.jsonl              per-sample records
    {output_dir}/{run_name}_summary.json       aggregate accuracy + config
    {output_dir}/{run_name}_summary.csv        tabular summary

Usage:
    python eval_gpqa.py --mode baseline \
        --gpqa_subset diamond --num_samples 5 \
        --output_dir results_gpqa --run_name smoke_baseline

    python eval_gpqa.py --mode page_attention \
        --page_size 32 --top_k 64 --unselected_mode drop \
        --output_dir results_gpqa --run_name qwen_page_topk64
"""

import os
import sys
import json
import re
import csv
import random
import argparse

# Ensure baselines/ packages are importable (matches eval_longbench_v2.py:19-21)
_BASELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

import torch
from tqdm import tqdm
from datasets import load_dataset

from eval_ruler import model_name_tag, apply_monkey_patch, load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Qwen3 compatibility
# ---------------------------------------------------------------------------
QWEN3_SUPPORTED_MODES = {
    "baseline", "page_attention", "seer_attention", "multipole_attention",
}
QWEN3_UNSUPPORTED_MODES = {
    "seer_prefill", "quest_attention", "duo_attention", "shadowkv",
}


def _assert_qwen3_compatible(mode: str) -> None:
    if mode in QWEN3_UNSUPPORTED_MODES:
        raise ValueError(
            f"--mode {mode!r} does not support Qwen3. "
            f"Supported modes for Qwen3-8B: {sorted(QWEN3_SUPPORTED_MODES)}."
        )
    if mode not in QWEN3_SUPPORTED_MODES:
        raise ValueError(f"Unknown --mode {mode!r}.")


# ---------------------------------------------------------------------------
# Prompt template (simple-evals / OpenAI GPQA format, CoT)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the form "Answer: $LETTER" (without quotes) where LETTER is one of A, B, C, D. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def format_gpqa_sample(item, rng):
    """Shuffle the four answers into A/B/C/D and return (prompt, gold_letter)."""
    correct = item["Correct Answer"].strip()
    incorrect = [
        item["Incorrect Answer 1"].strip(),
        item["Incorrect Answer 2"].strip(),
        item["Incorrect Answer 3"].strip(),
    ]
    choices = [correct] + incorrect
    rng.shuffle(choices)
    gold_letter = "ABCD"[choices.index(correct)]
    prompt = PROMPT_TEMPLATE.format(
        question=item["Question"].strip(),
        A=choices[0], B=choices[1], C=choices[2], D=choices[3],
    )
    return prompt, gold_letter


# ---------------------------------------------------------------------------
# Tokenize + truncate (mirrors eval_longbench_v2.py:161-180)
# ---------------------------------------------------------------------------
def tokenize_prompt(prompt_text, tokenizer, max_input_len):
    messages = [{"role": "user", "content": prompt_text}]
    chat_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Qwen3 chat template supports enable_thinking; keep it ON for reasoning evals.
    if hasattr(tokenizer, "chat_template") and "enable_thinking" in (tokenizer.chat_template or ""):
        chat_kwargs["enable_thinking"] = True
    input_ids = tokenizer.apply_chat_template(messages, **chat_kwargs)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    if input_ids.shape[1] > max_input_len:
        half = max_input_len // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
    return input_ids


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_ANSWER_PATTERNS = [
    re.compile(r"Answer:\s*\$?([A-D])\$?", re.IGNORECASE),
    re.compile(r"answer is\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*([A-D])\s*\}", re.IGNORECASE),
    re.compile(r"\b([A-D])\)\s*$"),
]


def extract_answer(response: str):
    """Return 'A'..'D' or None."""
    if not response:
        return None
    cleaned = response.replace("*", "")
    for pat in _ANSWER_PATTERNS:
        m = pat.search(cleaned)
        if m:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GPQA Evaluation (Qwen3-8B)")

    # Mode (all 8 visible for parity with eval_ruler.py; Qwen3 guard runs after parsing)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "page_attention", "seer_attention",
                                 "seer_prefill",
                                 "multipole_attention", "quest_attention",
                                 "duo_attention",
                                 "shadowkv"])

    # Model — Qwen3-8B only
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen3-8B",
                        choices=["Qwen/Qwen3-8B"])

    # Dataset
    parser.add_argument("--gpqa_dataset", type=str, default="Idavidrein/gpqa")
    parser.add_argument("--gpqa_subset", type=str, default="diamond",
                        choices=["diamond", "main", "extended"])
    parser.add_argument("--gpqa_split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 = all)")
    parser.add_argument("--seed", type=int, default=42)

    # Generation
    parser.add_argument("--max_input_len", type=int, default=8192,
                        help="Truncate tokenised input if it exceeds this length")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Reasoning chains can be long; default 8192 tokens")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_gpqa")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (auto-generated if not given)")

    # DCT Page Attention params (mirror eval_ruler.py:124-156)
    parser.add_argument("--page_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--compress_ratio", type=float, default=0.03125)
    parser.add_argument("--scoring_method", type=str, default="max")
    parser.add_argument("--group_agg_method", type=str, default="mean",
                        choices=["mean", "max", "topp"])
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--compression_method", type=str, default="haar",
                        choices=["haar", "dct"])
    parser.add_argument("--compressed_token_rope", type=str, default="mixed",
                        choices=["mixed", "block_center"])
    parser.add_argument("--continuous_rope", action="store_true")
    parser.add_argument("--weight_compressed_by_population", action="store_true")
    parser.add_argument("--no_triton", action="store_true")
    parser.add_argument("--comp_kv_quant", type=str, default="none",
                        choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    parser.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                        choices=["per_page", "per_comp_token"])

    # eval_ruler.py expects these on args even though GPQA doesn't loop over seq_lengths.
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[32768],
                        help="Unused by GPQA but consumed by ShadowKV's max_length sizing in eval_ruler helpers")

    # ShadowKV baseline params (only validated above; here for argparse parity)
    parser.add_argument("--shadowkv_cache_mode", type=str, default="shadowkv_cpu",
                        choices=["shadowkv", "shadowkv_cpu"])
    parser.add_argument("--sparse_budget", type=int, default=2192)
    parser.add_argument("--rank", type=int, default=160)
    parser.add_argument("--chunk_size", type=int, default=8)

    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip run if summary.json already exists in output dir")

    args = parser.parse_args()

    # Fail fast on unsupported modes — before any model load.
    _assert_qwen3_compatible(args.mode)

    if args.run_name is None:
        tag = model_name_tag(args.base_model)
        suffix = f"gpqa_{args.gpqa_subset}"
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

    if args.skip_existing:
        summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
        if os.path.exists(summary_path):
            print(f"SKIP (already exists): {summary_path}")
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

    for idx, item in enumerate(tqdm(samples, desc="GPQA")):
        sample_id = f"{idx}"
        if sample_id in completed_ids:
            continue

        # Per-sample seeded RNG so the A/B/C/D shuffle is identical across modes.
        rng = random.Random(args.seed + idx)
        prompt_text, gold = format_gpqa_sample(item, rng)

        input_ids = tokenize_prompt(prompt_text, tokenizer, args.max_input_len).to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            if args.mode == "seer_attention":
                output_ids, _ = model.batch_exist_generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_length=input_len + args.max_new_tokens,
                    do_sample=False,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

        generated_ids = output_ids[0, input_len:]
        del input_ids, output_ids
        torch.cuda.empty_cache()

        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = extract_answer(response)
        is_correct = (predicted == gold) if predicted else False

        result = {
            "_id": sample_id,
            "domain": item.get("High-level domain", ""),
            "subdomain": item.get("Subdomain", ""),
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "response": response,
            "input_len": input_len,
        }
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

        if is_correct:
            correct += 1
        total += 1

        if total > 0 and total % 25 == 0:
            print(f"  Progress: {total} done, accuracy so far: {correct / total * 100:.1f}%")

    out_f.close()

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
    print(f"GPQA Results — {run_name}")
    print("=" * 60)
    print(f"  Overall : {correct / total * 100:5.1f}%  ({correct}/{total})")

    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"] or "(unknown)", []).append(r)
    for domain, recs in sorted(by_domain.items()):
        c = sum(1 for r in recs if r["correct"])
        print(f"  {domain[:30]:30s}: {c / len(recs) * 100:5.1f}%  ({c}/{len(recs)})")

    no_answer = sum(1 for r in results if r["predicted"] is None)
    if no_answer > 0:
        print(f"  Failed answer extraction: {no_answer}/{total}")
    print("=" * 60)


def build_summary(results, args):
    total = len(results)

    def subset_stats(records):
        if not records:
            return {"accuracy": 0.0, "num_samples": 0}
        c = sum(1 for r in records if r["correct"])
        return {"accuracy": round(c / len(records) * 100, 2), "num_samples": len(records)}

    overall_acc = round((sum(1 for r in results if r["correct"]) / total * 100), 2) if total else 0.0

    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"] or "(unknown)", []).append(r)

    summary = {
        "mode": args.mode,
        "model": args.base_model,
        "dataset": args.gpqa_dataset,
        "subset": args.gpqa_subset,
        "run_name": args.run_name,
        "num_samples": total,
        "overall_accuracy": overall_acc,
        "extraction_failures": sum(1 for r in results if r["predicted"] is None),
        "by_domain": {
            domain: subset_stats(records)
            for domain, records in sorted(by_domain.items())
        },
        "max_new_tokens": args.max_new_tokens,
    }

    if args.mode == "page_attention":
        summary["top_k"] = args.top_k
        summary["page_size"] = args.page_size
        summary["compress_ratio"] = args.compress_ratio
        summary["scoring_method"] = args.scoring_method
        summary["group_agg_method"] = args.group_agg_method
        summary["unselected_mode"] = args.unselected_mode
        summary["compression_method"] = args.compression_method
        summary["comp_kv_quant"] = args.comp_kv_quant
    elif args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        summary["seer_attn_config"] = SEER_ATTN_CONFIG
    elif args.mode == "multipole_attention":
        from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
        summary["multipole_attn_config"] = MULTIPOLE_ATTN_CONFIG

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
    for label, payload in summary["by_domain"].items():
        rows.append({
            "group": "domain",
            "label": label,
            "accuracy": payload["accuracy"],
            "num_samples": payload["num_samples"],
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

    print(f"\nLoading GPQA: {args.gpqa_dataset} (subset=gpqa_{args.gpqa_subset})")
    dataset = load_dataset(args.gpqa_dataset, f"gpqa_{args.gpqa_subset}", split=args.gpqa_split)
    print(f"Loaded {len(dataset)} samples")

    results = evaluate(model, tokenizer, dataset, args)
    print_summary(results, args.run_name)
    summary_path, csv_path = write_summary_files(results, args)
    print(f"\nSummary: {summary_path}")
    print(f"CSV    : {csv_path}")


if __name__ == "__main__":
    main()
