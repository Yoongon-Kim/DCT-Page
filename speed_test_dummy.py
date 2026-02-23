"""
Decode speed benchmark with dummy (random) inputs: Baseline vs DCT Page Attention.

Generates random token sequences of configurable lengths, avoiding any dataset
dependency.  Measures prefill and decode speed separately.

Results are saved under:
    results_speed_test_dummy/<run_name>/
        samples.jsonl   — per-(length, repeat) timing records
        summary.json    — aggregated stats grouped by context length

Usage:
    python speed_test_dummy.py --context_lengths 4096,8192,16384 --mode both
    python speed_test_dummy.py --context_lengths 32768 --mode baseline --num_repeats 5
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speed_test_v2 import (
    load_model_and_tokenizer,
    get_original_forward,
    restore_forward,
    apply_dct_patch,
    model_family,
)


# ---------------------------------------------------------------------------
# Per-sample timing (no EOS stopping — always generates max_new_tokens)
# ---------------------------------------------------------------------------
def time_sample(model, tokenizer, input_ids, max_new_tokens, warmup_steps):
    device = input_ids.device
    prefill_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)

    step_times = []
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

        # No EOS check — always generate max_new_tokens for consistent measurement

    return prefill_time, step_times, max_new_tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Dummy-input decode speed benchmark")

    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--mode", choices=["baseline", "dct", "both"], default="both")
    p.add_argument("--context_lengths", type=str, default="4096,8192,16384,32768",
                   help="Comma-separated context lengths to benchmark")
    p.add_argument("--num_repeats", type=int, default=3,
                   help="Repeats per context length for averaging")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--output_dir", default="results_speed_test_dummy")
    p.add_argument("--run_name", default=None)

    dct = p.add_argument_group("DCT Page Attention config")
    dct.add_argument("--page_size", type=int, default=128)
    dct.add_argument("--top_k", type=int, default=8)
    dct.add_argument("--sink_size", type=int, default=4)
    dct.add_argument("--recent_size", type=int, default=128)
    dct.add_argument("--compress_ratio", type=float, default=0.03)
    dct.add_argument("--scoring_method", default="max", choices=["mean", "max"])
    dct.add_argument("--group_agg_method", default="mean",
                     choices=["mean", "max", "topp"])
    dct.add_argument("--unselected_mode", default="drop",
                     choices=["drop", "compressed"])
    dct.add_argument("--selection_mode", default="standard",
                     choices=["standard", "hierarchical"])
    dct.add_argument("--continuous_rope", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Run name
# ---------------------------------------------------------------------------
def make_run_name(label, args):
    family = model_family(args.model)
    if label == "baseline":
        return f"{family}_baseline_dummy"
    parts = [
        family, "page_attn_dummy",
        str(args.compress_ratio),
        f"topk{args.top_k}",
        args.scoring_method,
        args.group_agg_method,
        args.unselected_mode,
    ]
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Benchmark with dummy inputs
# ---------------------------------------------------------------------------
def benchmark_dummy(model, tokenizer, args, label, context_lengths):
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    # Warmup run: triggers CUDA kernel compilation and memory allocation
    warmup_len = min(context_lengths)
    warmup_ids = torch.randint(0, vocab_size, (1, warmup_len), dtype=torch.long, device=device)
    print(f"  [{label}] Warmup run (ctx={warmup_len})...")
    time_sample(model, tokenizer, warmup_ids, min(16, args.max_new_tokens), 0)
    del warmup_ids
    torch.cuda.empty_cache()

    all_records = []
    # stats grouped by context length: {ctx_len: {prefill_times, step_times}}
    per_length_stats = {}

    total_runs = len(context_lengths) * args.num_repeats
    run_idx = 0

    for ctx_len in context_lengths:
        per_length_stats[ctx_len] = {"prefill_times": [], "step_times": []}

        for repeat in range(args.num_repeats):
            run_idx += 1

            # Generate random input
            input_ids = torch.randint(
                0, vocab_size, (1, ctx_len), dtype=torch.long, device=device
            )

            prefill_time, step_times, n_generated = time_sample(
                model, tokenizer, input_ids, args.max_new_tokens, args.warmup_steps
            )

            per_length_stats[ctx_len]["prefill_times"].append(prefill_time)
            per_length_stats[ctx_len]["step_times"].extend(step_times)

            if step_times:
                avg_ms = sum(step_times) / len(step_times) * 1000
                tok_s = 1.0 / (sum(step_times) / len(step_times))
            else:
                avg_ms = tok_s = float("nan")

            record = {
                "context_length": ctx_len,
                "repeat": repeat,
                "prefill_ms": round(prefill_time * 1000, 2),
                "avg_decode_ms_per_tok": round(avg_ms, 3),
                "decode_tok_per_s": round(tok_s, 2),
                "n_decode_steps": len(step_times),
            }
            all_records.append(record)

            print(f"  [{label}] {run_idx}/{total_runs}: "
                  f"ctx={ctx_len}, repeat={repeat}, "
                  f"prefill={prefill_time*1000:.0f}ms, "
                  f"decode={avg_ms:.1f}ms/tok ({tok_s:.1f} tok/s), "
                  f"steps={n_generated}")

            # Free KV cache memory
            del input_ids
            torch.cuda.empty_cache()

    # Build per-length summary
    length_summaries = {}
    all_prefill = []
    all_step = []

    for ctx_len in context_lengths:
        s = per_length_stats[ctx_len]
        all_prefill.extend(s["prefill_times"])
        all_step.extend(s["step_times"])

        n = len(s["prefill_times"])
        length_summaries[ctx_len] = {
            "n_repeats": n,
            "avg_prefill_ms": round(sum(s["prefill_times"]) / n * 1000, 2) if n else None,
            "avg_decode_ms_per_tok": (
                round(sum(s["step_times"]) / len(s["step_times"]) * 1000, 3)
                if s["step_times"] else None
            ),
            "decode_tok_per_s": (
                round(len(s["step_times"]) / sum(s["step_times"]), 2)
                if s["step_times"] else None
            ),
            "total_decode_steps": len(s["step_times"]),
        }

    # Overall stats
    overall = {
        "label": label,
        "n_total_runs": len(all_records),
        "avg_prefill_ms": round(sum(all_prefill) / len(all_prefill) * 1000, 2) if all_prefill else None,
        "avg_decode_ms_per_tok": (
            round(sum(all_step) / len(all_step) * 1000, 3)
            if all_step else None
        ),
        "decode_tok_per_s": (
            round(len(all_step) / sum(all_step), 2)
            if all_step else None
        ),
        "total_decode_steps": len(all_step),
        "per_length": length_summaries,
    }

    return overall, all_records


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_results(records, run_dir):
    path = Path(run_dir) / "samples.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def save_summary(stats, run_dir, args, label):
    summary = dict(stats)
    summary["model"] = args.model
    summary["context_lengths"] = [int(x) for x in args.context_lengths.split(",")]
    summary["num_repeats"] = args.num_repeats
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
def print_summary(results, context_lengths):
    print("\n" + "=" * 75)
    print("DECODE SPEED SUMMARY  (dummy inputs)")
    print("=" * 75)

    # Per-length comparison table
    has_baseline = "baseline" in results
    has_dct = "dct" in results

    header = f"{'ctx_len':>10}"
    if has_baseline:
        header += f" | {'baseline (tok/s)':>18} {'prefill (ms)':>14}"
    if has_dct:
        header += f" | {'dct (tok/s)':>18} {'prefill (ms)':>14}"
    if has_baseline and has_dct:
        header += f" | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for ctx_len in context_lengths:
        row = f"{ctx_len:>10}"
        b_tok = d_tok = None

        if has_baseline:
            bl = results["baseline"]["per_length"].get(ctx_len, {})
            b_tok = bl.get("decode_tok_per_s")
            b_pre = bl.get("avg_prefill_ms")
            row += f" | {b_tok:>18.1f} {b_pre:>14.0f}" if b_tok else " | {'N/A':>18} {'N/A':>14}"

        if has_dct:
            dl = results["dct"]["per_length"].get(ctx_len, {})
            d_tok = dl.get("decode_tok_per_s")
            d_pre = dl.get("avg_prefill_ms")
            row += f" | {d_tok:>18.1f} {d_pre:>14.0f}" if d_tok else " | {'N/A':>18} {'N/A':>14}"

        if has_baseline and has_dct and b_tok and d_tok:
            row += f" | {d_tok/b_tok:>7.2f}x"

        print(row)

    # Overall
    print()
    for label, stats in results.items():
        tok_s = stats.get("decode_tok_per_s")
        ms = stats.get("avg_decode_ms_per_tok")
        if tok_s is None:
            continue
        print(f"  {label.upper()} overall: "
              f"{tok_s:.1f} tok/s  |  {ms:.2f} ms/tok  |  "
              f"{stats['total_decode_steps']} decode steps")

    if has_baseline and has_dct:
        b = results["baseline"].get("decode_tok_per_s")
        d = results["dct"].get("decode_tok_per_s")
        if b and d:
            print(f"\n  Overall decode speedup (DCT / baseline): {d/b:.2f}x")

    print("=" * 75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    print(f"Context lengths: {context_lengths}")
    print(f"Repeats per length: {args.num_repeats}")

    original_forward = get_original_forward(args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)

    results = {}

    def run_mode(label, patch=False):
        run_name = args.run_name or make_run_name(label, args)
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        restore_forward(args.model, original_forward, model)
        if patch:
            apply_dct_patch(args, model)

        stats, records = benchmark_dummy(
            model, tokenizer, args, label, context_lengths
        )
        save_results(records, run_dir)
        save_summary(stats, run_dir, args, label)
        results[label] = stats
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

    print_summary(results, context_lengths)


if __name__ == "__main__":
    main()
