"""
LongBench v1 evaluation for DCT Page Attention.

Compares baseline (full attention) vs page attention across 16 English tasks
spanning single-doc QA, multi-doc QA, summarization, few-shot learning,
synthetic tasks, and code completion.  Reports per-task scores using official
metrics (F1, ROUGE-L, accuracy, code similarity).
"""

import os
import json
import re
import string
import argparse
import random
import difflib
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------
ENGLISH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]

TASK_CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-Shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

# Official per-task max generation tokens
TASK_MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}

# Tasks where only the first line of generation is used for scoring
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum"}

# Task -> metric type
TASK_METRIC = {
    "narrativeqa": "f1", "qasper": "f1", "multifieldqa_en": "f1",
    "hotpotqa": "f1", "2wikimqa": "f1", "musique": "f1",
    "triviaqa": "f1",
    "gov_report": "rouge", "qmsum": "rouge", "multi_news": "rouge",
    "samsum": "rouge",
    "trec": "classification",
    "passage_count": "count",
    "passage_retrieval_en": "retrieval",
    "lcc": "code_sim", "repobench-p": "code_sim",
}


# ---------------------------------------------------------------------------
# Prompt templates (official LongBench v1)
# ---------------------------------------------------------------------------
TASK_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you "
        "can, using a single phrase or sentence if possible. If the question "
        "cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation."
        "\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\n"
        "Report: {context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript: {context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or "
        "more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news.\n\n"
        "{context}\n\n"
        "Now, write a one-page summary of all the news passages above.\n\n"
        "Summary:"
    ),
    "trec": "{context}\n{input}",
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "samsum": "{context}\n{input}",
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them "
        "may be duplicates. Please carefully read these paragraphs and "
        "determine how many unique paragraphs there are after removing "
        "duplicates. In other words, how many non-repeating paragraphs are "
        "there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing "
        "duplicates. The output format should only contain the number, such as "
        "1, 2, 3, and so on.\n\nOutput:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The abstract is from Paragraph"
    ),
    "lcc": (
        "Please complete the code given below.\n"
        "{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n"
        "{context}{input}"
    ),
}


# ---------------------------------------------------------------------------
# Metrics (official LongBench implementations)
# ---------------------------------------------------------------------------
def normalize_answer(s):
    """Normalize text for QA F1 scoring."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction, ground_truth):
    """Token-level F1 (official LongBench metric for QA tasks)."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(x, y):
    """Longest common subsequence length (space-optimised)."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l_f1(prediction, reference):
    """ROUGE-L F1 score (word-level LCS)."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def classification_score(prediction, ground_truth, all_classes):
    """Classification accuracy (official LongBench)."""
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match in em_match_list:
        if match in ground_truth:
            return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth):
    """Passage retrieval accuracy (official LongBench)."""
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = int(matches[0])
    numbers = re.findall(r"\d+", prediction)
    right = sum(1 for n in numbers if int(n) == ground_truth_id)
    return min(right, 1)


def count_score(prediction, ground_truth):
    """Passage count accuracy (official LongBench)."""
    numbers = re.findall(r"\d+", prediction)
    right = sum(1 for n in numbers if int(n) == int(ground_truth))
    return 1.0 if right > 0 else 0.0


def code_sim_score(prediction, ground_truth):
    """Code edit-similarity using SequenceMatcher."""
    return difflib.SequenceMatcher(None, prediction, ground_truth).ratio()


def score_single(prediction, answers, task, all_classes=None):
    """Score one prediction against all gold answers; return max."""
    if task in FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]

    metric = TASK_METRIC[task]
    scores = []
    for ans in answers:
        if metric == "f1":
            scores.append(qa_f1_score(prediction, ans))
        elif metric == "rouge":
            scores.append(rouge_l_f1(prediction, ans))
        elif metric == "classification":
            scores.append(classification_score(prediction, ans, all_classes or []))
        elif metric == "retrieval":
            scores.append(retrieval_score(prediction, ans))
        elif metric == "count":
            scores.append(count_score(prediction, ans))
        elif metric == "code_sim":
            scores.append(code_sim_score(prediction, ans))
    return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(context, input_text, task):
    template = TASK_PROMPTS[task]
    return template.format(context=context, input=input_text)


# ---------------------------------------------------------------------------
# Tokenise + truncate (first-half + last-half, same as official)
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
        input_ids = torch.cat(
            [input_ids[:, :half], input_ids[:, -half:]], dim=1
        )
    return input_ids


# ---------------------------------------------------------------------------
# Effective KV length computation
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LongBench v1 Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "page_attention"])

    # Model
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_input_len", type=int, default=120000)
    parser.add_argument("--max_new_tokens_override", type=int, default=-1,
                        help="Override per-task max_new_tokens (-1 = use official)")

    # Tasks
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Tasks to evaluate (default: all 16 English tasks)")

    # Data
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Max samples per task (-1 = all)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_longbench_v1")
    parser.add_argument("--run_name", type=str, default=None)

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
    parser.add_argument("--no_triton", action="store_true",
                        help="Disable Triton kernels (use pure PyTorch for comparison)")

    args = parser.parse_args()

    if args.tasks is None:
        args.tasks = ENGLISH_TASKS

    if args.run_name is None:
        args.run_name = ("baseline" if args.mode == "baseline"
                         else f"page_attn_topk{args.top_k}")

    return args


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------
def evaluate_task(model, tokenizer, task, dataset, args):
    """Evaluate a single LongBench task. Returns list of result dicts."""
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, f"{task}.jsonl")

    max_gen = (args.max_new_tokens_override
               if args.max_new_tokens_override > 0
               else TASK_MAX_NEW_TOKENS.get(task, 64))

    # Resume support
    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                r = json.loads(line)
                completed_ids.add(r["_id"])
        print(f"  Resuming {task}: {len(completed_ids)} already done")

    samples = list(dataset)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]

    out_f = open(output_path, "a")

    for item in tqdm(samples, desc=f"  {task}"):
        if item["_id"] in completed_ids:
            continue

        prompt_text = build_prompt(item["context"], item["input"], task)
        input_ids = tokenize_and_truncate(
            prompt_text, tokenizer, args.max_input_len
        )
        input_ids = input_ids.to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_gen,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = output_ids[0, input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        sc = score_single(
            response, item["answers"], task,
            all_classes=item.get("all_classes"),
        )

        result = {
            "_id": item["_id"],
            "task": task,
            "score": sc,
            "response": response,
            "gold": item["answers"],
            "input_len": input_len,
            "effective_len": compute_effective_len(input_len, args),
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()

    # Reload all results (including previously completed) for final stats
    all_results = []
    with open(output_path, "r") as f:
        for line in f:
            all_results.append(json.loads(line))

    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(all_task_results, run_name):
    """Print per-task and category-level summary."""
    print("\n" + "=" * 65)
    print(f"LongBench v1 Results \u2014 {run_name}")
    print("=" * 65)

    task_scores = {}
    for task, results in all_task_results.items():
        if not results:
            continue
        avg = sum(r["score"] for r in results) / len(results) * 100
        task_scores[task] = avg

    # Per-task
    for task in ENGLISH_TASKS:
        if task in task_scores:
            n = len(all_task_results[task])
            print(f"  {task:24s}: {task_scores[task]:5.1f}%  (n={n})")

    # Per-category
    print("-" * 65)
    for cat, tasks in TASK_CATEGORIES.items():
        cat_vals = [task_scores[t] for t in tasks if t in task_scores]
        if cat_vals:
            cat_avg = sum(cat_vals) / len(cat_vals)
            print(f"  {cat:24s}: {cat_avg:5.1f}%")

    # Overall
    if task_scores:
        overall = sum(task_scores.values()) / len(task_scores)
        print("-" * 65)
        print(f"  {'Overall':24s}: {overall:5.1f}%")
    print("=" * 65)

    return task_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Conditionally apply monkey-patch (before model loading)
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

    # Evaluate each task
    all_task_results = {}
    for task in args.tasks:
        print(f"\n--- Loading task: {task} ---")
        ds = load_dataset("THUDM/LongBench", task, split="test")
        print(f"  {len(ds)} samples")
        all_task_results[task] = evaluate_task(model, tokenizer, task, ds, args)

    # Summary
    task_scores = print_summary(all_task_results, args.run_name)

    # Save summary JSON
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")
    summary = {
        "mode": args.mode,
        "model": args.base_model,
        "run_name": args.run_name,
        "task_scores": {k: round(v, 2) for k, v in task_scores.items()},
        "overall": round(
            sum(task_scores.values()) / len(task_scores), 2
        ) if task_scores else 0,
    }
    if args.mode == "page_attention":
        summary["top_k"] = args.top_k
        summary["page_size"] = args.page_size
        summary["scoring_method"] = args.scoring_method
        summary["group_agg_method"] = args.group_agg_method
        summary["unselected_mode"] = args.unselected_mode
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-task results in: {run_dir}/")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
