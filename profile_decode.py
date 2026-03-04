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

from speed_test_dummy import (
    load_model_and_tokenizer,
    get_original_forward,
    restore_forward,
)


def apply_dct_patch(args, model=None):
    """Apply DCT patch (imports from dct_page_attention)."""
    import types
    import transformers

    patch_kwargs = dict(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        continuous_rope=args.continuous_rope,
        use_triton=not getattr(args, 'no_triton', False),
    )
    if "llama" in args.model.lower():
        from dct_page_attention import replace_llama_attn, dct_page_attention_forward
        replace_llama_attn(**patch_kwargs)
        if model is not None:
            attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(dct_page_attention_forward, module)
    else:
        from dct_page_attention import replace_qwen2_attn, dct_page_attention_forward
        replace_qwen2_attn(**patch_kwargs)
        if model is not None:
            attn_cls = transformers.models.qwen2.modeling_qwen2.Qwen2Attention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(dct_page_attention_forward, module)
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
# New fused kernels imported inside profiled_dct_page_attention_forward
from triton_kernels import apply_rope_q_direct
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicLayer


# ---------------------------------------------------------------------------
# Pre-allocated KV cache (avoids torch.cat during decode)
# ---------------------------------------------------------------------------
class PreAllocatedLayer(DynamicLayer):
    """Drop-in replacement for DynamicLayer that uses pre-allocated buffers.

    Instead of torch.cat (O(seq_len) alloc+copy per step), uses index
    assignment into a pre-allocated buffer (O(1) write per step).
    """

    @classmethod
    def from_dynamic_layer(cls, layer, extra_tokens):
        """Convert a populated DynamicLayer into a pre-allocated version."""
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

        # Return view of valid portion (zero-copy)
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_seq_length(self):
        return self._seen


def pre_allocate_cache(cache, extra_tokens=256):
    """Convert a DynamicCache (after prefill) to use pre-allocated layers."""
    for i, layer in enumerate(cache.layers):
        cache.layers[i] = PreAllocatedLayer.from_dynamic_layer(layer, extra_tokens)
    return cache


# ---------------------------------------------------------------------------
# Timing storage
# ---------------------------------------------------------------------------
_step_timings = defaultdict(list)  # step_name -> list of ms
_pending_events = []  # (name, start_event, end_event) — flushed after each decode step
_enabled = False
_current_layer = 0


def _flush_events():
    """Sync all devices, then compute all deferred elapsed times."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
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

    Uses 8 chained CUDA events so that total attention = sum of steps 1-7
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
    # Total attention = ev[0] → ev[9] = sum of steps 1-7 by construction.
    if _enabled:
        # Record events on the correct device's stream (critical for multi-GPU)
        _dev = hidden_states.device
        _stream = torch.cuda.current_stream(_dev)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
        ev[0].record(_stream)

    # Step 1: QKV projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _enabled:
        ev[1].record(_stream)

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
        ev[2].record(_stream)

    # Check if DCT path is active
    min_len_for_paging = cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size
    if kv_len < min_len_for_paging:
        if cfg.continuous_rope:
            all_pos = torch.arange(kv_len, device=key_cached.device)
            cos_all, sin_all = _compute_rope_cos_sin(
                all_pos, self.head_dim, getattr(self.config, 'rope_theta', self.config.rope_parameters['rope_theta'] if hasattr(self.config, 'rope_parameters') else 500000.0),
                key_cached.device, key_cached.dtype
            )
            attn_q, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
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
        ev[3].record(_stream)

    # Step 4: DCT compression
    comp_k, comp_v = _update_comp_cache(self, paged_k, paged_v, num_pages, comp_size)

    if _enabled:
        ev[4].record(_stream)

    # Step 5: Score pages (Triton kernel 1)
    from triton_kernels import score_pages_triton, assemble_kv_full_triton, assemble_kv_split_triton, apply_rope_q_triton
    page_scores = score_pages_triton(
        query_states, comp_k,
        cfg.scoring_method, cfg.group_agg_method,
        self.num_key_value_groups,
    )

    if _enabled:
        ev[5].record(_stream)

    # Step 6a: TopK + sort
    actual_top_k = min(cfg.top_k, num_pages)
    _, top_indices = torch.topk(page_scores, actual_top_k, dim=-1)
    selected_indices, _ = top_indices.sort(dim=-1)

    num_unselected = num_pages - actual_top_k
    middle_len = actual_top_k * cfg.page_size + num_unselected * comp_size
    assembled_len = cfg.sink_size + middle_len + actual_recent

    cos_table = None
    sin_table = None
    if cfg.continuous_rope:
        cached_len = getattr(self, '_rope_cache_len', 0)
        if assembled_len > cached_len:
            max_len = assembled_len + cfg.page_size
            positions = torch.arange(max_len, device=comp_k.device)
            cos_cached, sin_cached = _compute_rope_cos_sin(
                positions, self.head_dim,
                getattr(self.config, 'rope_theta', self.config.rope_parameters['rope_theta'] if hasattr(self.config, 'rope_parameters') else 500000.0),
                comp_k.device, comp_k.dtype,
            )
            self._rope_cos_cache = cos_cached
            self._rope_sin_cache = sin_cached
            self._rope_cache_len = max_len
        cos_table = self._rope_cos_cache[0, 0, :assembled_len].contiguous()
        sin_table = self._rope_sin_cache[0, 0, :assembled_len].contiguous()

    if _enabled:
        ev[6].record(_stream)

    # Step 6b: Assemble + K-RoPE (Triton kernel 2 — split version)
    final_k, final_v = assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
        cos_table, sin_table,
    )

    if _enabled:
        ev[7].record(_stream)

    # Step 6c: Q-RoPE (zero-overhead wrapper — pre-flattened cos/sin, pre-allocated buf)
    if cfg.continuous_rope:
        cos_q_flat = self._rope_cos_cache[0, 0, assembled_len - 1]  # [head_dim], contiguous
        sin_q_flat = self._rope_sin_cache[0, 0, assembled_len - 1]  # [head_dim], contiguous
        if not hasattr(self, '_q_rope_buf') or self._q_rope_buf.shape != query_states.shape:
            self._q_rope_buf = torch.empty_like(query_states)
        query_states = apply_rope_q_direct(query_states, cos_q_flat, sin_q_flat, self._q_rope_buf)

    if _enabled:
        ev[8].record(_stream)

    # Step 7: Attention + output projection
    # from flash_attn import flash_attn_func
    # attn_output = flash_attn_func(
    #     query_states.transpose(1, 2),
    #     final_k.transpose(1, 2),
    #     final_v.transpose(1, 2),
    #     causal=False,
    # )
    attn_output = F.scaled_dot_product_attention(
        query_states, final_k, final_v,
        is_causal=False,
        enable_gqa=True,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    if _enabled:
        ev[9].record(_stream)

    attn_output = self.o_proj(attn_output)

    if _enabled:
        ev[10].record(_stream)
        step_names = [
            "1_qkv_proj", "2_rope_cache", "3_segment_kv",
            "4_dct_compress", "5_score_pages", "6a_topk",
            "6b_assemble", "6c_q_rope", "7a_sdpa", "7b_o_proj",
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
    (1_qkv_proj, 2_rope_cache, 7_attn_output) for direct comparison.
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
    _dev = hidden_states.device
    _stream = torch.cuda.current_stream(_dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    ev[0].record(_stream)

    # Step 1: QKV projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    ev[1].record(_stream)

    # Step 2: RoPE + KV cache update
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    ev[2].record(_stream)

    # Step 7: Attention + output projection (named "7" for comparison with DCT)
    # from flash_attn import flash_attn_func
    # attn_output = flash_attn_func(
    #     query_states.transpose(1, 2),
    #     key_states.transpose(1, 2),
    #     value_states.transpose(1, 2),
    #     causal=True,
    # )
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        is_causal=False,  # q_len=1 decode: no future positions to mask
        enable_gqa=True,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    ev[3].record(_stream)

    attn_output = self.o_proj(attn_output)

    ev[4].record(_stream)

    step_names = ["1_qkv_proj", "2_rope_cache", "7a_sdpa", "7b_o_proj"]
    for i, name in enumerate(step_names):
        _pending_events.append((name, ev[i], ev[i + 1]))

    return attn_output, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Profile DCT Page decode path")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--context_length", type=int, default=32768)
    p.add_argument("--num_decode_steps", type=int, default=128,
                   help="Decode steps to profile (after warmup)")
    p.add_argument("--warmup_steps", type=int, default=8,
                   help="Warmup decode steps (not profiled)")
    p.add_argument("--mode", choices=["dct", "baseline", "both"], default="both")

    # DCT config
    p.add_argument("--page_size", type=int, default=128)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=4/128)
    p.add_argument("--scoring_method", default="mean")
    p.add_argument("--group_agg_method", default="max")
    p.add_argument("--unselected_mode", default="compressed")
    p.add_argument("--no_continuous_rope", action="store_true",
                   help="Disable continuous RoPE (enabled by default)")
    p.add_argument("--no_triton", action="store_true",
                   help="Disable Triton kernels (use pure PyTorch for comparison)")
    args = p.parse_args()
    args.continuous_rope = not args.no_continuous_rope
    return args


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

    # Pre-allocated cache avoids torch.cat during decode.
    # Only useful for DCT mode — baseline needs contiguous KV for SDPA anyway,
    # so the .contiguous() copy in step 7 would negate the savings.
    if mode == "dct":
        extra = args.warmup_steps + args.num_decode_steps + 16
        past_key_values = pre_allocate_cache(past_key_values, extra_tokens=extra)
        print(f"  Converted to pre-allocated cache (+{extra} tokens)")

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

        # Step 1: apply DCT patch (sets _dct_page_cfg + non-profiled forward)
        restore_forward(args.model, original_forward, model)
        apply_dct_patch(args, model)

        # Step 2: overwrite with profiled version — same pattern as baseline
        attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
        attn_cls.forward = profiled_dct_page_attention_forward
        for module in model.modules():
            if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                module._old_forward = types.MethodType(profiled_dct_page_attention_forward, module)

        # Verify
        assert attn_cls.__dict__['forward'] is profiled_dct_page_attention_forward, \
            "Patching failed: LlamaAttention.forward is not profiled_dct_page_attention_forward"

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