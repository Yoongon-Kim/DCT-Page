"""
Decode speed benchmark with dummy (random) inputs: Baseline vs Multipole Attention.

Generates random token sequences of configurable lengths, avoiding any dataset
dependency.  Measures prefill time, decode speed, and attention-only time
separately.

Results are saved under:
    results/speed_test_dummy_multipole/<run_name>/
        samples.jsonl   — per-(length, repeat) timing records
        summary.json    — aggregated stats grouped by context length

Usage:
    python speed_test_dummy_multipole.py --mode both --context_lengths 4096,8192,16384
    python speed_test_dummy_multipole.py --mode baseline --context_lengths 4096 --num_repeats 1
    python speed_test_dummy_multipole.py --mode multipole --percent_clusters 6.25 --percentiles 2180
"""

import argparse
import importlib
import json
import time
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap (run from any cwd)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT, _REPO_ROOT / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speed_test_dummy import (
    load_model_and_tokenizer,
    model_family,
    PreAllocatedLayer,
    pre_allocate_cache,
)


# ---------------------------------------------------------------------------
# Architecture-aware forward helpers (handles llama, qwen2, qwen3)
# ---------------------------------------------------------------------------
def _detect_arch(model_name: str) -> str:
    name = model_name.lower()
    if "qwen3" in name:
        return "qwen3"
    if "qwen2" in name:
        return "qwen2"
    if "llama" in name:
        return "llama"
    raise ValueError(f"Cannot detect architecture from '{model_name}'")


_ARCH_MODULES = {
    "llama": "transformers.models.llama.modeling_llama",
    "qwen2": "transformers.models.qwen2.modeling_qwen2",
    "qwen3": "transformers.models.qwen3.modeling_qwen3",
}

_ARCH_ATTN_CLS_NAME = {
    "llama": "LlamaAttention",
    "qwen2": "Qwen2Attention",
    "qwen3": "Qwen3Attention",
}


def _get_attn_cls(model_name: str):
    arch = _detect_arch(model_name)
    mod = importlib.import_module(_ARCH_MODULES[arch])
    return getattr(mod, _ARCH_ATTN_CLS_NAME[arch])


def get_original_forward(model_name: str):
    return _get_attn_cls(model_name).forward


def restore_forward(model_name: str, original_forward):
    _get_attn_cls(model_name).forward = original_forward


# ---------------------------------------------------------------------------
# Multipole config & patching
# ---------------------------------------------------------------------------
def build_multipole_config(args):
    return {
        "base_model": args.model,
        "use_centroids": True,
        "percent_clusters_lst": [float(x) for x in args.percent_clusters.split(",")],
        "percentiles_lst": [int(x) for x in args.percentiles.split(",")],
        "use_replacement": args.use_replacement,
        "cluster_interval": args.cluster_interval,
        "inference_tp": 1,
    }


def apply_multipole_patch(model, config_dict):
    from multipole_attn import replace_attn_multipole, init_multipole_layers
    replace_attn_multipole(config_dict)
    init_multipole_layers(model)


# ---------------------------------------------------------------------------
# Chunked prefill (multipole-compatible, no RoPE unrotation needed)
# ---------------------------------------------------------------------------
def chunked_prefill_multipole(model, input_ids, chunk_size):
    """Prefill in chunks for memory efficiency.

    Unlike the DCT version, multipole uses standard attention during prefill
    internally, so no forward-swapping or RoPE unrotation is needed.
    """
    seq_len = input_ids.shape[1]
    if chunk_size <= 0 or seq_len <= chunk_size:
        out = model(input_ids, use_cache=True)
        return out

    past_key_values = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        cache_position = torch.arange(start, end, device=input_ids.device)
        out = model(
            chunk,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        past_key_values = out.past_key_values
    return out


# ---------------------------------------------------------------------------
# Attention-only timing via CUDA events
# ---------------------------------------------------------------------------
_pending_attn_events = []   # (start_event, end_event) per layer per step
_attn_step_totals = []      # total attention ms per decode step
_attn_timing_enabled = False


def _make_timed_forward(original_forward):
    """Wrap any attention forward with CUDA event timing.

    Works for both baseline (e.g. LlamaAttention.forward) and multipole
    (multipole_attention_forward).  Only instruments decode steps
    (hidden_states.shape[1] == 1) and only when _attn_timing_enabled is True.
    """
    def timed_forward(self, *args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not _attn_timing_enabled or hidden_states.shape[1] > 1:
            return original_forward(self, *args, **kwargs)

        dev = hidden_states.device
        stream = torch.cuda.current_stream(dev)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        start_ev.record(stream)
        result = original_forward(self, *args, **kwargs)
        end_ev.record(stream)

        _pending_attn_events.append((start_ev, end_ev))
        return result

    return timed_forward


def _flush_attn_events():
    """Compute total attention ms from pending events and append to step totals.

    Must be called after torch.cuda.synchronize() so all events are complete.
    """
    total_ms = 0.0
    for s, e in _pending_attn_events:
        total_ms += s.elapsed_time(e)
    _attn_step_totals.append(total_ms)
    _pending_attn_events.clear()


def _reset_attn_timing():
    """Clear all timing state between runs."""
    _pending_attn_events.clear()
    _attn_step_totals.clear()


# ---------------------------------------------------------------------------
# Per-sample timing
# ---------------------------------------------------------------------------
def time_sample(model, input_ids, max_new_tokens, warmup_steps,
                chunk_size=0, measure_attn=False):
    """Measure prefill time, per-step decode times, and optionally attention-only times.

    Always generates exactly max_new_tokens (no EOS stopping) for consistent
    measurement.  Returns:
        (prefill_time_s, step_times, attn_step_times,
         first_decode_s, first_decode_attn_ms, n_generated)
    """
    global _attn_timing_enabled

    device = input_ids.device
    prefill_len = input_ids.shape[1]

    # --- Prefill ---
    _attn_timing_enabled = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = chunked_prefill_multipole(model, input_ids, chunk_size)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)

    # Pre-allocate KV cache for O(1) decode append
    extra = max_new_tokens + 16
    past_key_values = pre_allocate_cache(past_key_values, extra_tokens=extra)

    # --- Decode ---
    _reset_attn_timing()
    step_times = []
    attn_step_times = []
    first_decode_time = None
    first_decode_attn_ms = None

    for step in range(max_new_tokens):
        cache_position = torch.tensor([prefill_len + step], device=device)

        _attn_timing_enabled = measure_attn

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

        if measure_attn:
            _flush_attn_events()
            attn_ms = _attn_step_totals[-1]

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)

        # Record first decode step separately (includes clustering init for multipole)
        if step == 0:
            first_decode_time = elapsed
            first_decode_attn_ms = attn_ms if measure_attn else None

        if step >= warmup_steps:
            step_times.append(elapsed)
            if measure_attn:
                attn_step_times.append(attn_ms)

    _attn_timing_enabled = False
    return (prefill_time, step_times, attn_step_times,
            first_decode_time, first_decode_attn_ms, max_new_tokens)


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def benchmark_dummy(model, tokenizer, args, label, context_lengths, measure_attn):
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    all_records = []
    per_length_stats = {}

    total_runs = len(context_lengths) * args.num_repeats
    run_idx = 0

    for ctx_len in context_lengths:
        per_length_stats[ctx_len] = {
            "prefill_times": [],
            "step_times": [],
            "attn_step_times": [],
            "first_decode_times": [],
            "first_decode_attn_times": [],
        }

        # Warmup run (absorbs kernel compilation / allocator costs)
        warmup_ids = torch.randint(
            0, vocab_size, (1, ctx_len), dtype=torch.long, device=device
        )
        print(f"  [{label}] Warmup run (ctx={ctx_len})...")
        time_sample(
            model, warmup_ids, args.max_new_tokens, 0,
            chunk_size=args.chunk_size, measure_attn=False,
        )
        del warmup_ids
        torch.cuda.empty_cache()

        for repeat in range(args.num_repeats):
            run_idx += 1

            input_ids = torch.randint(
                0, vocab_size, (1, ctx_len), dtype=torch.long, device=device
            )

            (prefill_time, step_times, attn_step_times,
             first_decode_time, first_decode_attn_ms,
             n_generated) = time_sample(
                model, input_ids, args.max_new_tokens, args.warmup_steps,
                chunk_size=args.chunk_size, measure_attn=measure_attn,
            )

            stats = per_length_stats[ctx_len]
            stats["prefill_times"].append(prefill_time)
            stats["step_times"].extend(step_times)
            if first_decode_time is not None:
                stats["first_decode_times"].append(first_decode_time)
            if first_decode_attn_ms is not None:
                stats["first_decode_attn_times"].append(first_decode_attn_ms)
            if attn_step_times:
                stats["attn_step_times"].extend(attn_step_times)

            # Per-sample metrics
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

            if attn_step_times:
                avg_attn_ms = sum(attn_step_times) / len(attn_step_times)
                record["avg_attn_ms_per_tok"] = round(avg_attn_ms, 3)
                record["attn_fraction_pct"] = round(avg_attn_ms / avg_ms * 100, 1)

            if first_decode_time is not None:
                record["first_decode_ms"] = round(first_decode_time * 1000, 2)
            if first_decode_attn_ms is not None:
                record["first_decode_attn_ms"] = round(first_decode_attn_ms, 3)

            all_records.append(record)

            attn_info = ""
            if attn_step_times:
                attn_info = f", attn={record['avg_attn_ms_per_tok']:.1f}ms ({record['attn_fraction_pct']:.0f}%)"

            print(f"  [{label}] {run_idx}/{total_runs}: "
                  f"ctx={ctx_len}, repeat={repeat}, "
                  f"prefill={prefill_time*1000:.0f}ms, "
                  f"decode={avg_ms:.1f}ms/tok ({tok_s:.1f} tok/s)"
                  f"{attn_info}, "
                  f"steps={n_generated}")

            del input_ids
            torch.cuda.empty_cache()

    # Build per-length summary
    length_summaries = {}
    all_prefill = []
    all_step = []
    all_attn = []

    for ctx_len in context_lengths:
        s = per_length_stats[ctx_len]
        all_prefill.extend(s["prefill_times"])
        all_step.extend(s["step_times"])
        all_attn.extend(s["attn_step_times"])

        n = len(s["prefill_times"])
        ls = {
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

        if s["attn_step_times"]:
            avg_attn = sum(s["attn_step_times"]) / len(s["attn_step_times"])
            ls["avg_attn_ms_per_tok"] = round(avg_attn, 3)
            if ls["avg_decode_ms_per_tok"]:
                ls["attn_fraction_pct"] = round(
                    avg_attn / ls["avg_decode_ms_per_tok"] * 100, 1
                )

        if s["first_decode_times"]:
            ls["avg_first_decode_ms"] = round(
                sum(s["first_decode_times"]) / len(s["first_decode_times"]) * 1000, 2
            )
        if s["first_decode_attn_times"]:
            ls["avg_first_decode_attn_ms"] = round(
                sum(s["first_decode_attn_times"]) / len(s["first_decode_attn_times"]), 3
            )

        length_summaries[ctx_len] = ls

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

    if all_attn:
        avg_attn = sum(all_attn) / len(all_attn)
        overall["avg_attn_ms_per_tok"] = round(avg_attn, 3)
        if overall["avg_decode_ms_per_tok"]:
            overall["attn_fraction_pct"] = round(
                avg_attn / overall["avg_decode_ms_per_tok"] * 100, 1
            )

    return overall, all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Dummy-input decode speed benchmark: Baseline vs Multipole Attention"
    )

    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--mode", choices=["baseline", "multipole", "both"], default="both")
    p.add_argument("--context_lengths", type=str, default="4096,8192,16384,32768",
                   help="Comma-separated context lengths to benchmark")
    p.add_argument("--num_repeats", type=int, default=3,
                   help="Repeats per context length for averaging")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Chunked prefill size (0 = single-pass). "
                        "Use e.g. 8192 to reduce peak memory for long contexts.")
    p.add_argument("--output_dir", default="results/speed_test_dummy_multipole")
    p.add_argument("--run_name", default=None)
    p.add_argument("--no_measure_attn", action="store_true",
                   help="Disable attention-only timing (enabled by default)")

    mp = p.add_argument_group("Multipole Attention config")
    mp.add_argument("--percent_clusters", type=str, default="6.25",
                    help="Comma-separated percent_clusters_lst values")
    mp.add_argument("--percentiles", type=str, default="2180",
                    help="Comma-separated percentiles_lst values (token budgets)")
    mp.add_argument("--use_replacement", action="store_true", default=False,
                    help="Use centroid value approximation for non-selected tokens")
    mp.add_argument("--cluster_interval", type=int, default=128,
                    help="Tokens between re-clustering during generation")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Run name
# ---------------------------------------------------------------------------
def make_run_name(label, args):
    family = model_family(args.model)
    if label == "baseline":
        return f"{family}_baseline_dummy"
    parts = [
        family, "multipole_dummy",
        f"pct{'_'.join(args.percent_clusters.split(','))}",
        f"ptl{'_'.join(args.percentiles.split(','))}",
        f"ci{args.cluster_interval}",
        "repl" if args.use_replacement else "norepl",
    ]
    return "_".join(parts)


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
            "percent_clusters_lst": [float(x) for x in args.percent_clusters.split(",")],
            "percentiles_lst": [int(x) for x in args.percentiles.split(",")],
            "use_replacement": args.use_replacement,
            "cluster_interval": args.cluster_interval,
        })
    path = Path(run_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to: {path}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(results, context_lengths):
    print("\n" + "=" * 90)
    print("DECODE SPEED SUMMARY  (dummy inputs)")
    print("=" * 90)

    has_baseline = "baseline" in results
    has_mp = "multipole" in results

    # Build header
    header = f"{'ctx_len':>10}"
    if has_baseline:
        header += f" | {'bl tok/s':>10} {'prefill':>10} {'attn%':>7}"
    if has_mp:
        header += f" | {'mp tok/s':>10} {'prefill':>10} {'attn%':>7}"
    if has_baseline and has_mp:
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
            b_attn = bl.get("attn_fraction_pct")
            if b_tok is not None:
                attn_str = f"{b_attn:.0f}%" if b_attn is not None else "N/A"
                row += f" | {b_tok:>10.1f} {b_pre:>9.0f}ms {attn_str:>7}"
            else:
                row += f" | {'N/A':>10} {'N/A':>10} {'N/A':>7}"

        if has_mp:
            ml = results["multipole"]["per_length"].get(ctx_len, {})
            d_tok = ml.get("decode_tok_per_s")
            d_pre = ml.get("avg_prefill_ms")
            d_attn = ml.get("attn_fraction_pct")
            if d_tok is not None:
                attn_str = f"{d_attn:.0f}%" if d_attn is not None else "N/A"
                row += f" | {d_tok:>10.1f} {d_pre:>9.0f}ms {attn_str:>7}"
            else:
                row += f" | {'N/A':>10} {'N/A':>10} {'N/A':>7}"

        if has_baseline and has_mp and b_tok and d_tok:
            row += f" | {d_tok/b_tok:>7.2f}x"

        print(row)

    # Overall
    print()
    for label, stats in results.items():
        tok_s = stats.get("decode_tok_per_s")
        ms = stats.get("avg_decode_ms_per_tok")
        pre_ms = stats.get("avg_prefill_ms")
        attn_pct = stats.get("attn_fraction_pct")
        if tok_s is None:
            continue
        attn_str = f"  |  attn {attn_pct:.0f}%" if attn_pct is not None else ""
        print(f"  {label.upper()} overall: "
              f"{tok_s:.1f} tok/s  |  {ms:.2f} ms/tok  |  "
              f"prefill {pre_ms:.0f}ms  |  "
              f"{stats['total_decode_steps']} decode steps"
              f"{attn_str}")

    if has_baseline and has_mp:
        b = results["baseline"].get("decode_tok_per_s")
        d = results["multipole"].get("decode_tok_per_s")
        if b and d:
            print(f"\n  Overall decode speedup (multipole / baseline): {d/b:.2f}x")

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    measure_attn = not args.no_measure_attn

    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    print(f"Context lengths: {context_lengths}")
    print(f"Repeats per length: {args.num_repeats}")
    print(f"Attention timing: {'enabled' if measure_attn else 'disabled'}")

    original_forward = get_original_forward(args.model)
    attn_cls = _get_attn_cls(args.model)

    model, tokenizer = load_model_and_tokenizer(args.model)

    multipole_config = build_multipole_config(args) if args.mode != "baseline" else None

    results = {}

    def run_mode(label):
        run_name = args.run_name
        if run_name is not None:
            if args.mode == "both":
                run_name = f"{run_name}_{label}"
        else:
            run_name = make_run_name(label, args)
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Restore to baseline forward first
        restore_forward(args.model, original_forward)

        if label == "multipole":
            apply_multipole_patch(model, multipole_config)

        # Install timing wrapper
        if measure_attn:
            current_forward = attn_cls.forward
            attn_cls.forward = _make_timed_forward(current_forward)

        stats, records = benchmark_dummy(
            model, tokenizer, args, label, context_lengths, measure_attn
        )
        save_results(records, run_dir)
        save_summary(stats, run_dir, args, label)
        results[label] = stats
        print(f"\nResults written to: {run_dir}/")

        # Unwrap timing (restore to untimed forward)
        if measure_attn:
            attn_cls.forward = current_forward

    if args.mode in ("baseline", "both"):
        print("\n" + "=" * 65)
        print("BASELINE (full attention)")
        print("=" * 65)
        run_mode("baseline")

    if args.mode in ("multipole", "both"):
        print("\n" + "=" * 65)
        print("MULTIPOLE ATTENTION")
        print("=" * 65)
        run_mode("multipole")

    print_summary(results, context_lengths)


if __name__ == "__main__":
    main()
