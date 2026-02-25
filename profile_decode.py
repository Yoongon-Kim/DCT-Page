"""
Profile DCT Page Attention decode path: break down time per operation.

Instruments the dct_page_attention_forward to measure GPU time for each step
in the decode path using CUDA events. Reports per-layer and aggregated timing.

Uses chained CUDA events (end of step N = start of step N+1) so the total
attention time equals exactly the sum of individual steps — zero unaccounted
overhead.

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
    """Instrumented version of dct_page_attention_forward for profiling.

    Uses 8 chained CUDA events so that total attention = sum of steps 1-8
    with zero gaps.
    """
    global _current_layer

    from dct_page_attention import _dct_page_cfg as cfg
    from typing import Callable

    input_shape = hidden_states.shape[:-1]  # (bsz, q_len)
    hidden_shape = (*input_shape, -1, self.head_dim)  # (bsz, q_len, num_heads, head_dim)
    bsz, q_len = input_shape

    if q_len > 1:
        # Prefill path — always use standard attention (no profiling)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if cfg.continuous_rope:
            query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_cached, value_cached = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_cached, value_cached = key_states, value_states
            attn_q, attn_k, attn_v = query_rope, key_rope, value_states
        else:
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

    # ---- DECODE PATH (q_len == 1) ----
    # Chained CUDA events: ev[i] is the boundary between step i and step i+1.
    # Total attention = ev[0] → ev[8] = sum of steps 1-8 by construction.
    if _enabled:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
        ev[0].record()

    # Step 1: QKV projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _enabled:
        ev[1].record()

    # Step 2: RoPE + KV cache update
    cos, sin = position_embeddings
    if cfg.continuous_rope:
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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
        ev[2].record()

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

    # Step 3: Segment KV
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    (sink_k, sink_v, paged_k, paged_v,
     recent_k, recent_v, num_pages, actual_recent) = segment_kv(
        key_states, value_states, cfg
    )

    if _enabled:
        ev[3].record()

    # Step 4: DCT compression
    comp_k, comp_v = _update_comp_cache(self, paged_k, paged_v, num_pages, comp_size)

    if _enabled:
        ev[4].record()

    # Step 5: Score pages
    selected_indices, _page_scores = score_pages(
        query_states, comp_k, cfg, self.num_key_value_groups
    )

    if _enabled:
        ev[5].record()

    # Step 6: Assemble KV
    final_k, final_v = assemble_kv(
        sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
        recent_k, recent_v, selected_indices, cfg, num_pages
    )

    if _enabled:
        ev[6].record()

    # Step 7: Continuous RoPE
    if cfg.continuous_rope:
        assembled_len = final_k.shape[2]
        cached_len = getattr(self, '_rope_cache_len', 0)
        if assembled_len > cached_len:
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

        cos_k = self._rope_cos_cache[:, :, :assembled_len]
        sin_k = self._rope_sin_cache[:, :, :assembled_len]
        cos_q = self._rope_cos_cache[:, :, assembled_len - 1:assembled_len]
        sin_q = self._rope_sin_cache[:, :, assembled_len - 1:assembled_len]
        final_k = _apply_rope(final_k, cos_k, sin_k)
        query_states = _apply_rope(query_states, cos_q, sin_q)

    if _enabled:
        ev[7].record()

    # Step 8: Attention + output projection
    attn_output = F.scaled_dot_product_attention(
        query_states, final_k, final_v,
        is_causal=False,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if _enabled:
        ev[8].record()
        step_names = [
            "1_qkv_proj", "2_rope_cache", "3_segment_kv",
            "4_dct_compress", "5_score_pages", "6_assemble_kv",
            "7_continuous_rope", "8_attn_output",
        ]
        for i, name in enumerate(step_names):
            _pending_events.append((name, ev[i], ev[i + 1]))

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
    """Instrumented baseline (standard) attention for comparison.

    Uses 3 chained CUDA events with step names matching the DCT forward
    (1_qkv_proj, 2_rope_cache, 8_attn_output) for direct comparison.
    """
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

    # ---- Instrumented decode path (chained events) ----
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ev[0].record()

    # Step 1: QKV projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    ev[1].record()

    # Step 2: RoPE + KV cache update
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    ev[2].record()

    # Step 8: Attention + output projection (named "8" for comparison with DCT)
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0, scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None), **kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    ev[3].record()

    step_names = ["1_qkv_proj", "2_rope_cache", "8_attn_output"]
    for i, name in enumerate(step_names):
        _pending_events.append((name, ev[i], ev[i + 1]))

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
    p.add_argument("--no_triton", action="store_true",
                   help="Disable Triton kernels (use pure PyTorch for comparison)")
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
    avg_model_total = sum(total_times) / len(total_times)
    tok_s = 1000.0 / avg_model_total

    return avg_model_total, tok_s, dict(_step_timings), total_times


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_profile(mode, avg_model_total, tok_s, timings, num_layers=32):
    print(f"\n{'=' * 70}")
    print(f"PROFILE: {mode.upper()}")
    print(f"{'=' * 70}")

    # Compute per-token time for each step
    step_order = sorted(timings.keys())
    step_per_token = {}
    for step_name in step_order:
        vals = timings[step_name]
        n_decode_steps = len(vals) / num_layers if num_layers > 0 else 1
        step_per_token[step_name] = sum(vals) / n_decode_steps if n_decode_steps > 0 else 0.0

    # Attention total = exact sum of all steps (guaranteed by chained events)
    attn_total = sum(step_per_token.values())

    print(f"  Attention total: {attn_total:.2f} ms/tok")
    print(f"  Model total:     {avg_model_total:.2f} ms/tok  ({tok_s:.1f} tok/s)")
    print()

    print(f"  {'Step':<25} {'Per-layer (ms)':>15} {'Per-token (ms)':>15} {'% of attn':>12}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    for step_name in step_order:
        vals = timings[step_name]
        avg_per_call = sum(vals) / len(vals) if vals else 0.0
        per_token = step_per_token[step_name]
        pct = per_token / attn_total * 100 if attn_total > 0 else 0.0
        print(f"  {step_name:<25} {avg_per_call:>15.4f} {per_token:>15.3f} {pct:>11.1f}%")

    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
    print(f"  {'TOTAL':<25} {'':>15} {attn_total:>15.3f} {'100.0':>11}%")


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

        def _per_token(timings_dict, step, n_layers):
            vals = timings_dict.get(step, [])
            if not vals:
                return 0.0
            return sum(vals) / (len(vals) / n_layers)

        b_attn = sum(_per_token(b_timings, s, num_layers) for s in b_timings)
        d_attn = sum(_per_token(d_timings, s, num_layers) for s in d_timings)

        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        print(f"  Baseline attn: {b_attn:.2f} ms/tok ({b_tok:.1f} tok/s)")
        print(f"  DCT attn:      {d_attn:.2f} ms/tok ({d_tok:.1f} tok/s)")
        if d_attn > 0:
            print(f"  Ratio:         {b_attn/d_attn:.2f}x (baseline/DCT)")
        print()

        all_steps = sorted(set(list(b_timings.keys()) + list(d_timings.keys())))
        print(f"  {'Step':<25} {'Baseline (ms)':>15} {'DCT (ms)':>15} {'Diff (ms)':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

        for step in all_steps:
            b_per_tok = _per_token(b_timings, step, num_layers)
            d_per_tok = _per_token(d_timings, step, num_layers)

            b_str = f"{b_per_tok:.3f}" if step in b_timings else "\u2014"
            d_str = f"{d_per_tok:.3f}" if step in d_timings else "\u2014"

            if step in b_timings and step in d_timings:
                diff = d_per_tok - b_per_tok
                diff_str = f"{diff:+.3f}"
            else:
                diff_str = "\u2014"

            print(f"  {step:<25} {b_str:>15} {d_str:>15} {diff_str:>12}")


if __name__ == "__main__":
    main()