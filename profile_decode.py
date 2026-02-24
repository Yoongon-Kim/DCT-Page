"""
Profile DCT Page Attention decode path: break down time per operation.

Instruments the dct_page_attention_forward to measure GPU time for each step
in the decode path using CUDA events. Reports per-layer and aggregated timing.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python profile_decode.py --context_length 65536
    CUDA_VISIBLE_DEVICES=0,1 python profile_decode.py --context_length 32768 --top_k 32
"""

import argparse
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speed_test_v2 import (
    load_model_and_tokenizer,
    get_original_forward,
    restore_forward,
    apply_dct_patch,
)
from dct_page_attention import (
    _dct_page_cfg,
    segment_kv,
    _update_comp_cache,
    score_pages,
    assemble_kv,
    repeat_kv,
    apply_rotary_pos_emb,
    _apply_rope,
    _compute_rope_cos_sin,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timing storage
# ---------------------------------------------------------------------------
_step_timings = defaultdict(list)  # step_name -> list of ms
_pending_events = []  # (name, start_event, end_event) — flushed after each decode step
_enabled = False
_current_layer = 0


def _record(name, start_event, end_event):
    """Record end event and defer elapsed-time computation (no sync)."""
    end_event.record()
    _pending_events.append((name, start_event, end_event))


def _flush_events():
    """Sync once, then compute all deferred elapsed times."""
    torch.cuda.synchronize()
    for name, s, e in _pending_events:
        _step_timings[name].append(s.elapsed_time(e))
    _pending_events.clear()


# ---------------------------------------------------------------------------
# Instrumented DCT Page attention forward
# ---------------------------------------------------------------------------
def profiled_dct_page_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Instrumented version of dct_page_attention_forward for profiling."""
    global _current_layer

    from dct_page_attention import _dct_page_cfg as cfg
    from typing import Callable

    input_shape = hidden_states.shape[:-1] # (bsz, q_len)
    hidden_shape = (*input_shape, -1, self.head_dim) # (bsz, q_len, num_heads, head_dim)
    bsz, q_len = input_shape

    if q_len > 1:
        # Prefill path — always use standard attention (no profiling)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_heads, q_len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa

        cos, sin = position_embeddings
        if cfg.continuous_rope:
            # Compute RoPE'd Q/K for prefill/short-seq attention only
            query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} # Q. Why do we need sin, cos, and cache_position? We will eventually change them in the end! A. on line 151 
                key_cached, value_cached = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_cached, value_cached = key_states, value_states
            attn_q, attn_k, attn_v = query_rope, key_rope, value_states
        else:
            # Standard: apply RoPE then cache
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            attn_q, attn_k, attn_v = query_states, key_states, value_states

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, attn_q, attn_k, attn_v, attention_mask,
            dropout=0.0, scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ---- DECODE PATH (q_len == 1) — always runs DCT, records events only when _enabled ----
    device = hidden_states.device

    # Step 1: QKV projection
    if _enabled:
        s1 = torch.cuda.Event(enable_timing=True) # Create a start marker
        e1 = torch.cuda.Event(enable_timing=True) # Create an end marker
        s1.record() # Drop start marker into the GPU command stream

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # These are for the last generated token
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _enabled:
        _record("1_qkv_proj", s1, e1) # Drop end marker(e1) into the GPU stream

    # Step 2: RoPE + KV cache update
    if _enabled:
        s2 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s2.record()

    cos, sin = position_embeddings
    if cfg.continuous_rope:
        # query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin) # this is only for short kv_len fallback
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} # cache_position: tells the cache where to store the new token / sin, cos: not neede for this cache type, but required by the API contract
            key_cached, value_cached = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            key_cached, value_cached = key_states, value_states
        kv_len = key_cached.shape[2]
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        kv_len = key_states.shape[2]

    if _enabled:
        _record("2_rope_cache", s2, e2)

    # Check if DCT path is active
    min_len_for_paging = cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size
    if kv_len < min_len_for_paging:
        if cfg.continuous_rope:
            all_pos = torch.arange(kv_len, device=key_cached.device)
            cos_all, sin_all = _compute_rope_cos_sin(
                all_pos, self.head_dim, self.config.rope_parameters['rope_theta'],
                key_cached.device, key_cached.dtype
            )
            attn_q = query_rope
            attn_k = _apply_rope(key_cached, cos_all, sin_all)
            attn_v = value_cached
        else:
            attn_q, attn_k, attn_v = query_states, key_states, value_states

        attention_interface = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, attn_q, attn_k, attn_v, attention_mask,
            dropout=0.0, scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        _current_layer += 1
        return attn_output, attn_weights

    # Use pre-RoPE KV for page building in continuous_rope mode
    if cfg.continuous_rope:
        key_states = key_cached
        value_states = value_cached

    # Step 5: Segment KV
    if _enabled:
        s5 = torch.cuda.Event(enable_timing=True)
        e5 = torch.cuda.Event(enable_timing=True)
        s5.record()

    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    (sink_k, sink_v, paged_k, paged_v,
     recent_k, recent_v, num_pages, actual_recent) = segment_kv(
        key_states, value_states, cfg
    )

    if _enabled:
        _record("5_segment_kv", s5, e5)

    # Step 5b: Update compressed cache
    if _enabled:
        s5b = torch.cuda.Event(enable_timing=True)
        e5b = torch.cuda.Event(enable_timing=True)
        s5b.record()

    comp_k, comp_v = _update_comp_cache(self, paged_k, paged_v, num_pages, comp_size)

    if _enabled:
        _record("5b_update_comp_cache", s5b, e5b)

    # Step 6: Score pages
    if _enabled:
        s6 = torch.cuda.Event(enable_timing=True)
        e6 = torch.cuda.Event(enable_timing=True)
        s6.record()

    selected_indices, _page_scores = score_pages(
        query_states, comp_k, cfg, self.num_key_value_groups
    )

    if _enabled:
        _record("6_score_pages", s6, e6)

    # Step 7: Assemble KV
    if _enabled:
        s7 = torch.cuda.Event(enable_timing=True)
        e7 = torch.cuda.Event(enable_timing=True)
        s7.record()

    final_k, final_v = assemble_kv(
        sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
        recent_k, recent_v, selected_indices, cfg, num_pages
    )

    if _enabled:
        _record("7_assemble_kv", s7, e7)

    # Step 7.5: Continuous RoPE (apply RoPE to assembled K and Q)
    if cfg.continuous_rope:
        if _enabled:
            s75 = torch.cuda.Event(enable_timing=True)
            e75 = torch.cuda.Event(enable_timing=True)
            s75.record()

        assembled_len = final_k.shape[2]

        # Cache RoPE cos/sin table on the module to avoid recomputing
        # trig every decode step.  In drop mode the assembled length is
        # nearly constant (grows by 1 per step, resets at page boundaries),
        # so a single cached table (with headroom) covers all steps.
        cached_len = getattr(self, '_rope_cache_len', 0)
        if assembled_len > cached_len:
            # (Re)compute with headroom so we don't recompute every step
            max_len = assembled_len + cfg.page_size
            positions = torch.arange(max_len, device=final_k.device)
            cos_cached, sin_cached = _compute_rope_cos_sin(
                positions, self.head_dim,
                self.config.rope_parameters['rope_theta'],
                final_k.device, final_k.dtype,
            )
            self._rope_cos_cache = cos_cached
            self._rope_sin_cache = sin_cached
            self._rope_cache_len = max_len

        # Slice from cache (just views, zero kernels)
        cos_k = self._rope_cos_cache[:, :, :assembled_len]
        sin_k = self._rope_sin_cache[:, :, :assembled_len]
        cos_q = self._rope_cos_cache[:, :, assembled_len - 1:assembled_len]
        sin_q = self._rope_sin_cache[:, :, assembled_len - 1:assembled_len]
        final_k = _apply_rope(final_k, cos_k, sin_k)
        query_states = _apply_rope(query_states, cos_q, sin_q)

        if _enabled:
            _record("7.5_continuous_rope", s75, e75)

    # Step 8: Attention (SDPA with GQA — fused, no repeat_kv needed)
    if _enabled:
        s8 = torch.cuda.Event(enable_timing=True)
        e8 = torch.cuda.Event(enable_timing=True)
        s8.record()

    attn_output = F.scaled_dot_product_attention(
        query_states, final_k, final_v,
        is_causal=False,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    if _enabled:
        _record("8_attention", s8, e8)

    # Step 9: Output projection
    if _enabled:
        s9 = torch.cuda.Event(enable_timing=True)
        e9 = torch.cuda.Event(enable_timing=True)
        s9.record()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if _enabled:
        _record("9_o_proj", s9, e9)

    _current_layer += 1
    return attn_output, None


# ---------------------------------------------------------------------------
# Profiled baseline forward
# ---------------------------------------------------------------------------
def profiled_baseline_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Instrumented baseline (standard) attention for comparison."""
    from typing import Callable

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    if not _enabled or q_len > 1:
        # Prefill — standard path
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ---- Instrumented decode path ----
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    _record("1_qkv_proj", s, e)

    s2 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    s2.record()

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    _record("2_rope_cache", s2, e2)

    s_attn = torch.cuda.Event(enable_timing=True)
    e_attn = torch.cuda.Event(enable_timing=True)
    s_attn.record()

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0, scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None), **kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    _record("8_sdpa_attention", s_attn, e_attn)

    s9 = torch.cuda.Event(enable_timing=True)
    e9 = torch.cuda.Event(enable_timing=True)
    s9.record()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    _record("9_o_proj", s9, e9)

    return attn_output, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Profile DCT Page decode path")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--context_length", type=int, default=65536)
    p.add_argument("--num_decode_steps", type=int, default=32,
                   help="Decode steps to profile (after warmup)")
    p.add_argument("--warmup_steps", type=int, default=8,
                   help="Warmup decode steps (not profiled)")
    p.add_argument("--mode", choices=["dct", "baseline", "both"], default="both")

    # DCT config
    p.add_argument("--page_size", type=int, default=128)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.03)
    p.add_argument("--scoring_method", default="mean")
    p.add_argument("--group_agg_method", default="max")
    p.add_argument("--unselected_mode", default="compressed")
    p.add_argument("--selection_mode", default="standard")
    p.add_argument("--continuous_rope", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run profiled decode
# ---------------------------------------------------------------------------
def run_profiled_decode(model, tokenizer, args, mode):
    global _step_timings, _enabled, _current_layer

    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    # Generate dummy input
    input_ids = torch.randint(0, vocab_size, (1, args.context_length),
                              dtype=torch.long, device=device)

    # Prefill (not profiled)
    print(f"  Prefilling ({args.context_length} tokens)...")
    _enabled = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prefill done: {prefill_ms:.0f}ms")

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)
    prefill_len = args.context_length

    # Warmup decode steps (not profiled)
    print(f"  Warming up ({args.warmup_steps} steps)...")
    for step in range(args.warmup_steps):
        cache_position = torch.tensor([prefill_len + step], device=device)
        with torch.no_grad():
            out = model(next_token, past_key_values=past_key_values,
                        use_cache=True, cache_position=cache_position)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()

    # Profiled decode steps
    print(f"  Profiling ({args.num_decode_steps} steps)...")
    _step_timings.clear()
    _enabled = True

    total_times = []
    for step in range(args.num_decode_steps):
        _current_layer = 0
        cache_position = torch.tensor(
            [prefill_len + args.warmup_steps + step], device=device
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(next_token, past_key_values=past_key_values,
                        use_cache=True, cache_position=cache_position)
        _flush_events()  # single sync + compute all step elapsed times
        total_ms = (time.perf_counter() - t0) * 1000
        total_times.append(total_ms)

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)

    _enabled = False

    # Summarize
    avg_total = sum(total_times) / len(total_times)
    tok_s = 1000.0 / avg_total

    return avg_total, tok_s, dict(_step_timings), total_times


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_profile(mode, avg_total, tok_s, timings, num_layers=32):
    print(f"\n{'=' * 70}")
    print(f"PROFILE: {mode.upper()}")
    print(f"{'=' * 70}")
    print(f"  Overall: {avg_total:.2f} ms/tok  ({tok_s:.1f} tok/s)")
    print()

    # Each step is called num_layers times per decode step
    # timings[step_name] has num_decode_steps * num_layers entries
    print(f"  {'Step':<25} {'Per-layer (ms)':>15} {'Per-token (ms)':>15} {'% of total':>12}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    step_order = sorted(timings.keys())
    accounted = 0.0
    for step_name in step_order:
        vals = timings[step_name]
        n_calls = len(vals)
        avg_per_call = sum(vals) / n_calls if vals else 0.0
        # Per-token = average across all calls for one decode step
        # = avg_per_call * num_layers (calls per token)
        n_decode_steps = n_calls / num_layers if num_layers > 0 else 1
        per_token = sum(vals) / n_decode_steps if n_decode_steps > 0 else 0.0
        pct = per_token / avg_total * 100 if avg_total > 0 else 0.0
        accounted += per_token

        print(f"  {step_name:<25} {avg_per_call:>15.4f} {per_token:>15.3f} {pct:>11.1f}%")

    unaccounted = avg_total - accounted
    pct_unaccounted = unaccounted / avg_total * 100 if avg_total > 0 else 0.0
    print(f"  {'(unaccounted)':<25} {'':>15} {unaccounted:>15.3f} {pct_unaccounted:>11.1f}%")
    print(f"  {'TOTAL':<25} {'':>15} {avg_total:>15.3f} {'100.0':>11}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    import types
    import transformers

    original_forward = get_original_forward(args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)

    num_layers = model.config.num_hidden_layers
    print(f"Model layers: {num_layers}")
    print(f"Context length: {args.context_length}")

    results = {}

    if args.mode in ("baseline", "both"):
        print(f"\n{'=' * 65}")
        print("BASELINE (profiled)")
        print(f"{'=' * 65}")

        # Patch with profiled baseline forward
        restore_forward(args.model, original_forward, model)
        attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
        attn_cls.forward = profiled_baseline_forward
        for module in model.modules():
            if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                module._old_forward = types.MethodType(profiled_baseline_forward, module)

        avg_total, tok_s, timings, total_times = run_profiled_decode(
            model, tokenizer, args, "baseline"
        )
        print_profile("baseline", avg_total, tok_s, timings, num_layers)
        results["baseline"] = (avg_total, tok_s, timings)

        # Clean up KV cache
        del total_times
        torch.cuda.empty_cache()

    if args.mode in ("dct", "both"):
        print(f"\n{'=' * 65}")
        print(f"DCT PAGE ATTENTION (profiled, top_k={args.top_k})")
        print(f"{'=' * 65}")

        # Restore original, then apply DCT patch, then override with profiled version
        restore_forward(args.model, original_forward, model)
        apply_dct_patch(args, model)

        # Now override with profiled version
        attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
        attn_cls.forward = profiled_dct_page_attention_forward
        for module in model.modules():
            if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                module._old_forward = types.MethodType(profiled_dct_page_attention_forward, module)

        avg_total, tok_s, timings, total_times = run_profiled_decode(
            model, tokenizer, args, "dct"
        )
        print_profile("dct", avg_total, tok_s, timings, num_layers)
        results["dct"] = (avg_total, tok_s, timings)

        del total_times
        torch.cuda.empty_cache()

    # Comparison
    if "baseline" in results and "dct" in results:
        b_avg, b_tok, b_timings = results["baseline"]
        d_avg, d_tok, d_timings = results["dct"]

        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        print(f"  Baseline: {b_avg:.2f} ms/tok ({b_tok:.1f} tok/s)")
        print(f"  DCT:      {d_avg:.2f} ms/tok ({d_tok:.1f} tok/s)")
        print(f"  Ratio:    {b_avg/d_avg:.2f}x (baseline/DCT)")
        print()

        # Compare shared steps
        all_steps = sorted(set(list(b_timings.keys()) + list(d_timings.keys())))
        print(f"  {'Step':<25} {'Baseline (ms)':>15} {'DCT (ms)':>15} {'Diff (ms)':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

        for step in all_steps:
            b_vals = b_timings.get(step, [])
            d_vals = d_timings.get(step, [])
            n_layers = 32

            b_per_tok = sum(b_vals) / (len(b_vals) / n_layers) if b_vals else 0
            d_per_tok = sum(d_vals) / (len(d_vals) / n_layers) if d_vals else 0

            b_str = f"{b_per_tok:.3f}" if b_vals else "—"
            d_str = f"{d_per_tok:.3f}" if d_vals else "—"

            if b_vals and d_vals:
                diff = d_per_tok - b_per_tok
                diff_str = f"{diff:+.3f}"
            else:
                diff_str = "—"

            print(f"  {step:<25} {b_str:>15} {d_str:>15} {diff_str:>12}")


if __name__ == "__main__":
    main()
