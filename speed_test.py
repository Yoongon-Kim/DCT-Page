"""
Decode speed benchmark: Baseline vs DCT Page Attention on LongBench v1.

Prefill and decode steps are timed separately using a manual decode loop,
so decode latency is measured cleanly without prefill noise.

Only samples with tokenized length >= --min_context_len are used to ensure
the DCT paging path is active (not the full-attention fallback).

Per-task timing is saved as .jsonl files (one line per sample) and a
summary.json is written per run, mirroring the results_longbench_v1 layout:

    results_speed_test/
        baseline/
            narrativeqa.jsonl
            qasper.jsonl
            ...
            summary.json
        dct_topk8/
            narrativeqa.jsonl
            ...
            summary.json

Usage:
    # Both modes back-to-back on the same model (saves GPU memory)
    python benchmark_decode.py

    # One mode at a time
    python benchmark_decode.py --mode baseline
    python benchmark_decode.py --mode dct --top_k 8 --page_size 128

    # Custom model / tasks
    python benchmark_decode.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --tasks 2wikimqa hotpotqa --num_samples 20
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse prompt templates and helpers from the existing eval script
from eval_longbench_v1 import (
    ENGLISH_TASKS,
    TASK_MAX_NEW_TOKENS,
    build_prompt,
    tokenize_and_truncate,
)

# Default tasks for speed benchmarking: mix of QA, summarization, and long-form
# to cover a range of decode budgets (32–512 tokens).
SPEED_TASKS = [
    "narrativeqa",      # single-doc QA,  max_new_tokens=128
    "qasper",           # single-doc QA,  max_new_tokens=128
    "multifieldqa_en",  # single-doc QA,  max_new_tokens=64
    #"2wikimqa",         # multi-doc QA,   max_new_tokens=32
    #"triviaqa",         # few-shot QA,    max_new_tokens=32
    "gov_report",       # summarization,  max_new_tokens=512
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LongBench v1 decode speed benchmark")

    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HuggingFace model name or local path")
    p.add_argument("--mode", choices=["baseline", "dct", "both"], default="both",
                   help="Which mode(s) to benchmark")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="LongBench tasks to benchmark (default: 6 speed-test tasks)")
    p.add_argument("--num_samples", type=int, default=20,
                   help="Samples per task")
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Override per-task decode budget (None = use official limits)")
    p.add_argument("--max_input_len", type=int, default=120000,
                   help="Truncate inputs longer than this (tokens)")
    p.add_argument("--min_context_len", type=int, default=4356,
                   help="Skip samples shorter than this (default: auto from DCT config)")
    p.add_argument("--warmup_steps", type=int, default=3,
                   help="Decode steps to discard at the start of each sample")
    p.add_argument("--output_dir", default="results_speed_test",
                   help="Root directory for results (default: results_speed_test)")
    p.add_argument("--run_name", default=None,
                   help="Subdirectory name for this run (default: auto from mode/config)")

    # DCT page config
    dct = p.add_argument_group("DCT Page Attention config")
    dct.add_argument("--page_size", type=int, default=128)
    dct.add_argument("--top_k", type=int, default=8)
    dct.add_argument("--sink_size", type=int, default=4)
    dct.add_argument("--recent_size", type=int, default=128)
    dct.add_argument("--compress_ratio", type=float, default=0.03)
    dct.add_argument("--scoring_method", default="max",
                     choices=["mean", "max"])
    dct.add_argument("--group_agg_method", default="mean",
                     choices=["mean", "max", "topp"])
    dct.add_argument("--unselected_mode", default="drop",
                     choices=["drop", "compressed"])
    dct.add_argument("--selection_mode", default="standard",
                     choices=["standard", "hierarchical"])
    dct.add_argument("--continuous_rope", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_family(model_name):
    """Extract a short model family tag from the model name."""
    name = model_name.lower()
    if "llama" in name:
        return "llama"
    elif "qwen" in name:
        return "qwen2"
    else:
        return model_name.split("/")[-1].lower()


def make_run_name(label, args):
    """
    Auto-generate a descriptive run name.

    Examples:
        baseline  ->  llama_baseline
        dct       ->  llama_page_attn_0.03_topk8_mean_max_drop
    """
    family = model_family(args.model)
    if label == "baseline":
        return f"{family}_baseline"
    parts = [
        family,
        "page_attn",
        str(args.compress_ratio),
        f"topk{args.top_k}",
        args.scoring_method,
        args.group_agg_method,
        args.unselected_mode,
    ]
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded: {model_name} ({n_params:.2f}B params)")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Attention patch helpers
# ---------------------------------------------------------------------------
def get_original_forward(model_name):
    """Return the unpatched attention forward for the model family."""
    import transformers
    if "llama" in model_name.lower():
        return transformers.models.llama.modeling_llama.LlamaAttention.forward
    else:
        return transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward


def restore_forward(model_name, original_forward):
    import transformers
    if "llama" in model_name.lower():
        transformers.models.llama.modeling_llama.LlamaAttention.forward = original_forward
    else:
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = original_forward


def apply_dct_patch(args):
    if "llama" in args.model.lower():
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
            selection_mode=args.selection_mode,
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
            selection_mode=args.selection_mode,
            continuous_rope=args.continuous_rope,
        )


# ---------------------------------------------------------------------------
# Per-sample timing
# ---------------------------------------------------------------------------
def time_sample(model, tokenizer, input_ids, max_new_tokens, warmup_steps):
    """
    Time one sample, separating prefill and decode.

    Uses a manual decode loop so each decode step is individually timed.
    The first `warmup_steps` decode steps are discarded from timing to let
    CUDA kernels warm up.

    Returns:
        prefill_time_s  (float)
        step_times_s    (list[float]) — one entry per timed decode step
        n_generated     (int) — total steps including warmup
    """
    device = input_ids.device
    prefill_len = input_ids.shape[1]

    # Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)  # [bsz, 1]

    # Decode loop
    step_times = []
    step = 0
    for step in range(max_new_tokens):
        cache_position = torch.tensor([prefill_len + step], device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)

        if step >= warmup_steps:
            step_times.append(elapsed)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return prefill_time, step_times, step + 1


# ---------------------------------------------------------------------------
# Sample pre-selection
# ---------------------------------------------------------------------------
def preselect_samples(task_data, tasks, tokenizer, args):
    """
    Pre-select sample indices per task using min_context_len.

    Called once before any mode runs so that baseline, topk32, and topk8
    all benchmark the identical set of samples.

    Returns:
        {task: [sample_idx, ...]}  — exactly num_samples indices per task
    """
    print(f"\nPre-selecting {args.num_samples} samples per task "
          f"(min_context_len={args.min_context_len} tokens)...")
    task_selected = {}
    for task in tasks:
        selected = []
        for i, sample in enumerate(task_data[task]):
            prompt_text = build_prompt(sample["context"], sample["input"], task)
            input_ids = tokenize_and_truncate(prompt_text, tokenizer, args.max_input_len)
            seq_len = input_ids.shape[1]
            if seq_len >= args.min_context_len:
                selected.append(i)
            if len(selected) >= args.num_samples:
                break
        task_selected[task] = selected
        if len(selected) < args.num_samples:
            print(f"  {task}: WARNING only {len(selected)}/{args.num_samples} samples "
                  f"meet min_context_len={args.min_context_len}")
        else:
            print(f"  {task}: {len(selected)} samples selected")
    return task_selected


# ---------------------------------------------------------------------------
# Task benchmark
# ---------------------------------------------------------------------------
def benchmark_task(model, tokenizer, task, samples, args, label, selected_indices):
    """
    Benchmark one LongBench task.

    selected_indices: list of dataset indices to use (pre-selected in main so
                      all modes benchmark the same samples).

    Returns:
        stats       (dict)       — aggregated timing for this task
        per_sample  (list[dict]) — one record per sample, for .jsonl output
    """
    max_gen = args.max_new_tokens or TASK_MAX_NEW_TOKENS.get(task, 64)
    device = next(model.parameters()).device

    prefill_times = []
    all_step_times = []
    per_sample = []
    n_target = args.num_samples

    for i in selected_indices:
        if len(per_sample) >= n_target:
            break

        sample = samples[i]
        prompt_text = build_prompt(sample["context"], sample["input"], task)
        input_ids = tokenize_and_truncate(prompt_text, tokenizer, args.max_input_len)
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]

        prefill_time, step_times, n_generated = time_sample(
            model, tokenizer, input_ids, max_gen, args.warmup_steps
        )

        prefill_times.append(prefill_time)
        all_step_times.extend(step_times)

        if step_times:
            avg_ms = sum(step_times) / len(step_times) * 1000
            tok_s = 1.0 / (sum(step_times) / len(step_times))
        else:
            avg_ms = tok_s = float("nan")

        per_sample.append({
            "sample_idx": i,
            "ctx_len": seq_len,
            "prefill_ms": round(prefill_time * 1000, 2),
            "avg_decode_ms_per_tok": round(avg_ms, 3),
            "decode_tok_per_s": round(tok_s, 2),
            "n_decode_steps": len(step_times),
        })

        print(f"  [{label}] {task} {len(per_sample)}/{n_target}: "
              f"ctx={seq_len}, "
              f"prefill={prefill_time*1000:.0f}ms, "
              f"decode={avg_ms:.1f}ms/tok ({tok_s:.1f} tok/s), "
              f"steps={n_generated}")

    if len(per_sample) < n_target:
        print(f"  [{label}] {task}: WARNING only {len(per_sample)}/{n_target} samples processed")

    n = len(prefill_times)
    stats = {
        "task": task,
        "label": label,
        "n_samples": n,
        "avg_prefill_ms": round(sum(prefill_times) / n * 1000, 2) if n else None,
        "avg_decode_ms_per_tok": (
            round(sum(all_step_times) / len(all_step_times) * 1000, 3)
            if all_step_times else None
        ),
        "decode_tok_per_s": (
            round(len(all_step_times) / sum(all_step_times), 2)
            if all_step_times else None
        ),
        "total_decode_steps": len(all_step_times),
    }
    return stats, per_sample


# ---------------------------------------------------------------------------
# Saving results (mirrors results_longbench_v1 layout)
# ---------------------------------------------------------------------------
def save_task_results(per_sample, task, run_dir):
    """Write per-sample timing records to {run_dir}/{task}.jsonl."""
    path = Path(run_dir) / f"{task}.jsonl"
    with open(path, "w") as f:
        for record in per_sample:
            f.write(json.dumps(record) + "\n")


def save_summary(task_stats_list, run_dir, args, label):
    """Write aggregated summary to {run_dir}/summary.json."""
    valid = [s for s in task_stats_list if s["decode_tok_per_s"] is not None]
    overall_tok_s = (
        sum(s["decode_tok_per_s"] for s in valid) / len(valid) if valid else None
    )
    overall_ms = (
        sum(s["avg_decode_ms_per_tok"] for s in valid) / len(valid) if valid else None
    )

    summary = {
        "label": label,
        "model": args.model,
        "overall_decode_tok_per_s": round(overall_tok_s, 2) if overall_tok_s else None,
        "overall_avg_decode_ms_per_tok": round(overall_ms, 3) if overall_ms else None,
        "per_task": {
            s["task"]: {
                "n_samples": s["n_samples"],
                "avg_prefill_ms": s["avg_prefill_ms"],
                "avg_decode_ms_per_tok": s["avg_decode_ms_per_tok"],
                "decode_tok_per_s": s["decode_tok_per_s"],
                "total_decode_steps": s["total_decode_steps"],
            }
            for s in task_stats_list
        },
    }
    if label != "baseline":
        summary.update({
            "page_size": args.page_size,
            "top_k": args.top_k,
            "sink_size": args.sink_size,
            "recent_size": args.recent_size,
            "compress_ratio": args.compress_ratio,
            "scoring_method": args.scoring_method,
            "unselected_mode": args.unselected_mode,
            "selection_mode": args.selection_mode,
        })

    path = Path(run_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to: {path}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(results):
    print("\n" + "=" * 65)
    print("DECODE SPEED SUMMARY")
    print("=" * 65)

    for label, task_results in results.items():
        valid = [r for r in task_results if r["decode_tok_per_s"] is not None]
        if not valid:
            continue
        avg_tok_s = sum(r["decode_tok_per_s"] for r in valid) / len(valid)
        avg_ms = sum(r["avg_decode_ms_per_tok"] for r in valid) / len(valid)
        n_samples = sum(r["n_samples"] for r in valid)
        n_steps = sum(r["total_decode_steps"] for r in valid)
        print(f"\n  {label.upper()}")
        print(f"    {avg_tok_s:.1f} tok/s  |  {avg_ms:.2f} ms/tok")
        print(f"    {n_samples} samples, {n_steps} decode steps across {len(valid)} tasks")
        print(f"    Per-task breakdown:")
        for r in valid:
            print(f"      {r['task']:24s}: {r['decode_tok_per_s']:.1f} tok/s  "
                  f"({r['avg_decode_ms_per_tok']:.2f} ms/tok,  "
                  f"prefill={r['avg_prefill_ms']:.0f}ms)")

    if "baseline" in results and "dct" in results:
        base = [r for r in results["baseline"] if r["decode_tok_per_s"]]
        dct = [r for r in results["dct"] if r["decode_tok_per_s"]]
        if base and dct:
            base_tok_s = sum(r["decode_tok_per_s"] for r in base) / len(base)
            dct_tok_s = sum(r["decode_tok_per_s"] for r in dct) / len(dct)
            speedup = dct_tok_s / base_tok_s
            print(f"\n  Decode speedup (DCT / baseline): {speedup:.2f}x")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    tasks = args.tasks or SPEED_TASKS

    # Auto-compute min_context_len so every benchmarked sample triggers paging
    if args.min_context_len is None:
        args.min_context_len = (
            args.sink_size + args.page_size * (args.top_k + 1) + args.recent_size
        )
        print(f"min_context_len = {args.min_context_len} tokens "
              f"(sink + page_size*(top_k+1) + recent)")

    # Save the unpatched forward before any monkey-patching happens
    original_forward = get_original_forward(args.model)

    # Load model once; we swap the forward method between modes
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load all task data up front (full dataset, so we can filter by min_context_len
    # and still collect enough valid samples)
    print("\nLoading LongBench datasets...")
    task_data = {}
    for task in tasks:
        ds = load_dataset("THUDM/LongBench", task, split="test")
        task_data[task] = list(ds)
        print(f"  {task}: {len(task_data[task])} samples in pool")

    # Pre-select samples once — all modes use the identical set
    task_selected = preselect_samples(task_data, tasks, tokenizer, args)

    results = {}

    def run_mode(label, patch=False):
        run_name = args.run_name or make_run_name(label, args)
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        restore_forward(args.model, original_forward)
        if patch:
            apply_dct_patch(args)

        task_stats_list = []
        for task in tasks:
            print(f"\n--- {task} ---")
            stats, per_sample = benchmark_task(
                model, tokenizer, task, task_data[task], args, label,
                selected_indices=task_selected[task],
            )
            task_stats_list.append(stats)
            save_task_results(per_sample, task, run_dir)

        save_summary(task_stats_list, run_dir, args, label)
        results[label] = task_stats_list
        print(f"\nResults written to: {run_dir}/")

    if args.mode in ("baseline", "both"):
        print("\n" + "=" * 65)
        print("BASELINE (full attention)")
        print("=" * 65)
        run_mode("baseline", patch=False)

    if args.mode in ("dct", "both"):
        print("\n" + "=" * 65)
        print("DCT PAGE ATTENTION")
        print("=" * 65)
        run_mode("dct", patch=True)

    print_summary(results)


if __name__ == "__main__":
    main()
