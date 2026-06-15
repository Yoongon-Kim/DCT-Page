"""
Decode speed benchmark with dummy (random) inputs.

Modes: Baseline (full attention), DCT Page Attention, and Multipole Attention.
Generates random token sequences of configurable lengths, avoiding any dataset
dependency.  Measures prefill and decode speed separately.

Results are saved under:
    results/speed/<run_name>/
        samples.jsonl   — per-(length, repeat) timing records
        summary.json    — aggregated stats grouped by context length

Usage:
    python speed_test_dummy.py --context_lengths 4096,8192,16384 --mode dct
    python speed_test_dummy.py --context_lengths 32768 --mode baseline --num_repeats 5
    # Multipole (standalone) — typically on Qwen3:
    python speed_test_dummy.py --mode multipole --model Qwen/Qwen3-8B \\
        --percent_clusters 6.25 --percentiles 2048
"""

import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicLayer


# ---------------------------------------------------------------------------
# Shared utilities (formerly in speed_test_v2.py)
# ---------------------------------------------------------------------------
def model_family(model_name):
    name = model_name.lower()
    if "llama" in name:
        return "llama"
    elif "qwen" in name:
        return "qwen2"
    return model_name.split("/")[-1].lower()


def load_model_and_tokenizer(model_name, attn_implementation="sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded: {model_name} ({n_params:.2f}B params, attn={attn_implementation})")
    return model, tokenizer


# Architecture-aware attention-class resolution (handles llama, qwen2, qwen3).
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


def _detect_arch(model_name):
    name = model_name.lower()
    if "qwen3" in name:
        return "qwen3"
    if "qwen2" in name:
        return "qwen2"
    if "llama" in name:
        return "llama"
    raise ValueError(f"Cannot detect architecture from '{model_name}'")


def _get_attn_cls(model_name):
    arch = _detect_arch(model_name)
    mod = importlib.import_module(_ARCH_MODULES[arch])
    return getattr(mod, _ARCH_ATTN_CLS_NAME[arch])


def get_original_forward(model_name):
    return _get_attn_cls(model_name).forward


def restore_forward(model_name, original_forward, model=None):
    attn_cls = _get_attn_cls(model_name)
    attn_cls.forward = original_forward
    if model is not None:
        for module in model.modules():
            if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                module._old_forward = types.MethodType(original_forward, module)


def apply_dct_patch(args, model=None):
    patch_kwargs = dict(
        page_size=args.page_size,
        top_k=args.top_k,
        num_sink_pages=args.num_sink_pages,
        num_recent_pages=args.num_recent_pages,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        compressed_token_rope=args.compressed_token_rope,
        use_triton=not getattr(args, 'no_triton', False),
        weight_compressed_by_population=True,
    )
    if "llama" in args.model.lower():
        import transformers
        from dct_page_attention import replace_llama_attn, dct_page_attention_forward
        replace_llama_attn(**patch_kwargs)
        if model is not None:
            attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(dct_page_attention_forward, module)
    else:
        import transformers
        from dct_page_attention import replace_qwen2_attn, dct_page_attention_forward
        replace_qwen2_attn(**patch_kwargs)
        if model is not None:
            attn_cls = transformers.models.qwen2.modeling_qwen2.Qwen2Attention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(dct_page_attention_forward, module)


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
# Pre-allocated KV cache (avoids torch.cat during decode)
# ---------------------------------------------------------------------------
class PreAllocatedLayer(DynamicLayer):
    """Drop-in replacement for DynamicLayer that uses pre-allocated buffers."""

    @classmethod
    def from_dynamic_layer(cls, layer, extra_tokens):
        new_layer = cls()
        k, v = layer.keys, layer.values
        bsz, heads, seq_len, dim = k.shape

        alloc_len = seq_len + extra_tokens
        new_layer.keys = torch.empty(bsz, heads, alloc_len, dim,
                                     dtype=k.dtype, device=k.device)
        new_layer.values = torch.empty(bsz, heads, alloc_len, dim,
                                       dtype=v.dtype, device=v.device)
        new_layer.keys[:, :, :seq_len, :] = k
        new_layer.values[:, :, :seq_len, :] = v

        new_layer._seen = seq_len
        new_layer._alloc_len = alloc_len
        new_layer.is_initialized = True
        new_layer.dtype = k.dtype
        new_layer.device = k.device
        return new_layer

    def update(self, key_states, value_states, cache_kwargs=None):
        seq_len = key_states.shape[-2]
        start = self._seen
        end = start + seq_len

        self.keys[:, :, start:end, :] = key_states
        self.values[:, :, start:end, :] = value_states
        self._seen = end

        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_seq_length(self):
        return self._seen


def pre_allocate_cache(cache, extra_tokens=256):
    """Convert a DynamicCache (after prefill) to use pre-allocated layers."""
    for i, layer in enumerate(cache.layers):
        cache.layers[i] = PreAllocatedLayer.from_dynamic_layer(layer, extra_tokens)
    return cache


# ---------------------------------------------------------------------------
# Per-sample timing (no EOS stopping — always generates max_new_tokens)
# ---------------------------------------------------------------------------
def chunked_prefill(model, input_ids, chunk_size):
    """Prefill by processing input_ids in chunks to reduce peak activation memory.

    Returns the same (output, past_key_values) as a single forward pass would,
    but with much lower peak GPU memory usage.

    When a DCT-patched attention forward is active, temporarily restores the
    original forward for prefill: the DCT forward attends only over the current
    chunk's keys, which shape-mismatches the causal mask on chunk 2+. The
    original forward stores RoPE-rotated keys — exactly what DCT decode expects
    (RoPE is baked into the cached keys; decode reads them directly).
    """
    seq_len = input_ids.shape[1]
    if chunk_size <= 0 or seq_len <= chunk_size:
        out = model(input_ids, use_cache=True)
        return out

    # Detect if a DCT-patched forward is active and swap to original.
    original_forward = getattr(model, '_original_attn_forward', None)
    attn_cls = None
    patched_forward = None

    if original_forward is not None:
        for module in model.modules():
            if type(module).__name__ in ('LlamaAttention', 'Qwen2Attention', 'Qwen3Attention'):
                attn_cls = type(module)
                break
        if attn_cls is not None:
            current_forward = attn_cls.__dict__.get('forward')
            if current_forward is not None and current_forward is not original_forward:
                patched_forward = current_forward
                attn_cls.forward = original_forward

    try:
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
    finally:
        # Always restore the patched forward
        if attn_cls is not None and patched_forward is not None:
            attn_cls.forward = patched_forward

    return out


def time_sample(model, tokenizer, input_ids, max_new_tokens, warmup_steps,
                use_pre_alloc=False, chunk_size=0):
    device = input_ids.device
    prefill_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = chunked_prefill(model, input_ids, chunk_size)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)

    if use_pre_alloc:
        extra = max_new_tokens + 16
        past_key_values = pre_allocate_cache(past_key_values, extra_tokens=extra)

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
    p.add_argument("--mode", choices=["baseline", "dct", "multipole"], default="dct",
                   help="Which attention to benchmark (one per run).")
    p.add_argument("--context_lengths", type=str, default="4096,8192,16384,32768",
                   help="Comma-separated context lengths to benchmark")
    p.add_argument("--num_repeats", type=int, default=3,
                   help="Repeats per context length for averaging")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Chunked prefill size (0 = single-pass prefill). "
                        "Use e.g. 8192 to reduce peak memory for long contexts.")
    p.add_argument("--baseline_dynamic_cache", action="store_true",
                   help="Use HF DynamicLayer (torch.cat every decode step) for the baseline. "
                        "Default: baseline uses PreAllocatedLayer, matching the DCT path, "
                        "so the measured speedup isolates the attention-sparsity gain "
                        "rather than conflating it with cache-management overhead.")
    p.add_argument("--output_dir", default="results/speed")
    p.add_argument("--run_name", default=None)

    dct = p.add_argument_group("DCT Page Attention config")
    dct.add_argument("--page_size", type=int, default=32)
    dct.add_argument("--top_k", type=int, default=64)
    dct.add_argument("--num_sink_pages", type=int, default=1)
    dct.add_argument("--num_recent_pages", type=int, default=5)
    dct.add_argument("--compress_ratio", type=float, default=0.125)
    dct.add_argument("--scoring_method", default="max",
                     choices=["mean", "max"])
    dct.add_argument("--group_agg_method", default="max",
                     choices=["mean", "max"])
    dct.add_argument("--unselected_mode", default="drop",
                     choices=["drop", "compressed"])
    dct.add_argument("--compressed_token_rope", default="mixed", choices=["mixed", "block_center"])
    dct.add_argument("--no_triton", action="store_true",
                     help="Disable Triton kernels (use pure PyTorch for comparison)")

    mp = p.add_argument_group("Multipole Attention config (--mode multipole)")
    mp.add_argument("--percent_clusters", type=str, default="6.25",
                    help="Comma-separated percent_clusters_lst values")
    mp.add_argument("--percentiles", type=str, default="2048",
                    help="Comma-separated percentiles_lst values (token budgets)")
    mp.add_argument("--use_replacement", action="store_true", default=False,
                    help="Use centroid value approximation for non-selected tokens")
    mp.add_argument("--cluster_interval", type=int, default=128,
                    help="Tokens between re-clustering during generation")

    args = p.parse_args()
    return args


# ---------------------------------------------------------------------------
# Run name
# ---------------------------------------------------------------------------
def make_run_name(label, args):
    family = model_family(args.model)
    if label == "baseline":
        return f"{family}_baseline_dummy"
    if label == "multipole":
        parts = [
            family, "multipole_dummy",
            f"pct{'_'.join(args.percent_clusters.split(','))}",
            f"ptl{'_'.join(args.percentiles.split(','))}",
            f"ci{args.cluster_interval}",
            "repl" if args.use_replacement else "norepl",
        ]
        return "_".join(parts)
    triton_tag = "notriton" if getattr(args, 'no_triton', False) else "triton"
    parts = [
        family, "page_attn_dummy",
        str(args.compress_ratio),
        f"topk{args.top_k}",
        args.scoring_method,
        args.group_agg_method,
        args.unselected_mode,
        "nocrope",
        triton_tag,
    ]
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Benchmark with dummy inputs
# ---------------------------------------------------------------------------
def benchmark_dummy(model, tokenizer, args, label, context_lengths):
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size
    if label == "dct":
        use_pre_alloc = not getattr(args, 'no_triton', False)
    elif label == "baseline":
        use_pre_alloc = not getattr(args, 'baseline_dynamic_cache', False)
    else:
        use_pre_alloc = False

    all_records = []
    # stats grouped by context length: {ctx_len: {prefill_times, step_times}}
    per_length_stats = {}

    total_runs = len(context_lengths) * args.num_repeats
    run_idx = 0

    for ctx_len in context_lengths:
        per_length_stats[ctx_len] = {"prefill_times": [], "step_times": []}

        # Warm each context length separately so the first measured repeat for a
        # given length does not absorb shape-specific kernel compilation or
        # allocator/setup costs.
        warmup_ids = torch.randint(
            0, vocab_size, (1, ctx_len), dtype=torch.long, device=device
        )
        print(f"  [{label}] Warmup run (ctx={ctx_len})...")
        time_sample(
            model,
            tokenizer,
            warmup_ids,
            args.max_new_tokens,
            0,
            use_pre_alloc=use_pre_alloc,
            chunk_size=args.chunk_size,
        )
        del warmup_ids
        torch.cuda.empty_cache()

        for repeat in range(args.num_repeats):
            run_idx += 1

            # Generate random input
            input_ids = torch.randint(
                0, vocab_size, (1, ctx_len), dtype=torch.long, device=device
            )

            prefill_time, step_times, n_generated = time_sample(
                model, tokenizer, input_ids, args.max_new_tokens, args.warmup_steps,
                use_pre_alloc=use_pre_alloc, chunk_size=args.chunk_size,
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
    if label == "dct":
        summary.update({
            "page_size": args.page_size,
            "top_k": args.top_k,
            "num_sink_pages": args.num_sink_pages,
            "num_recent_pages": args.num_recent_pages,
            "compress_ratio": args.compress_ratio,
            "scoring_method": args.scoring_method,
            "unselected_mode": args.unselected_mode,
            "use_triton": not getattr(args, 'no_triton', False),
        })
    elif label == "multipole":
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
    print("\n" + "=" * 75)
    print("DECODE SPEED SUMMARY  (dummy inputs)")
    print("=" * 75)

    # Per-length comparison table. Baseline + any sparse method(s) present.
    has_baseline = "baseline" in results
    methods = [m for m in ("dct", "multipole") if m in results]

    header = f"{'ctx_len':>10}"
    if has_baseline:
        header += f" | {'baseline (tok/s)':>18} {'prefill (ms)':>14}"
    for m in methods:
        header += f" | {m + ' (tok/s)':>18} {'prefill (ms)':>14}"
    if has_baseline and len(methods) == 1:
        header += f" | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for ctx_len in context_lengths:
        row = f"{ctx_len:>10}"

        b_tok = None
        if has_baseline:
            bl = results["baseline"]["per_length"].get(ctx_len, {})
            b_tok = bl.get("decode_tok_per_s")
            b_pre = bl.get("avg_prefill_ms")
            row += f" | {b_tok:>18.1f} {b_pre:>14.0f}" if b_tok else f" | {'N/A':>18} {'N/A':>14}"

        m_tok_last = None
        for m in methods:
            ml = results[m]["per_length"].get(ctx_len, {})
            m_tok = ml.get("decode_tok_per_s")
            m_pre = ml.get("avg_prefill_ms")
            row += f" | {m_tok:>18.1f} {m_pre:>14.0f}" if m_tok else f" | {'N/A':>18} {'N/A':>14}"
            m_tok_last = m_tok

        if has_baseline and len(methods) == 1 and b_tok and m_tok_last:
            row += f" | {m_tok_last/b_tok:>7.2f}x"

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

    if has_baseline:
        b = results["baseline"].get("decode_tok_per_s")
        for m in methods:
            d = results[m].get("decode_tok_per_s")
            if b and d:
                print(f"\n  Overall decode speedup ({m} / baseline): {d/b:.2f}x")

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
    model._original_attn_forward = original_forward

    results = {}

    def run_mode(label):
        run_name = args.run_name if args.run_name is not None else make_run_name(label, args)
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        restore_forward(args.model, original_forward, model)
        if label == "dct":
            apply_dct_patch(args, model)
        elif label == "multipole":
            apply_multipole_patch(model, build_multipole_config(args))

        stats, records = benchmark_dummy(
            model, tokenizer, args, label, context_lengths
        )
        save_results(records, run_dir)
        save_summary(stats, run_dir, args, label)
        results[label] = stats
        print(f"\nResults written to: {run_dir}/")

    if args.mode == "baseline":
        print("\n" + "=" * 65)
        print("BASELINE (full attention)")
        print("=" * 65)
        run_mode("baseline")

    if args.mode == "dct":
        print("\n" + "=" * 65)
        print("DCT PAGE ATTENTION")
        print("=" * 65)
        run_mode("dct")

    if args.mode == "multipole":
        print("\n" + "=" * 65)
        print("MULTIPOLE ATTENTION")
        print("=" * 65)
        run_mode("multipole")

    print_summary(results, context_lengths)


if __name__ == "__main__":
    main()
