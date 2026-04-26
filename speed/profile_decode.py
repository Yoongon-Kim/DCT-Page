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
import sys
import time
from collections import defaultdict
from pathlib import Path

# Bootstrap sys.path so this script runs from any cwd. The sibling
# `speed_test_dummy.py` lives in this dir; `dct_page_attention.py` and
# `triton_kernels.py` live at the project root one level up.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speed_test_dummy import (
    load_model_and_tokenizer,
    get_original_forward,
    restore_forward,
    chunked_prefill,
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
        compressed_token_rope=getattr(args, 'compressed_token_rope', 'mixed'),
        use_triton=not getattr(args, 'no_triton', False),
        weight_compressed_by_population=True,
        comp_kv_quant=args.comp_kv_quant,
        comp_kv_quant_granularity=args.comp_kv_quant_granularity,
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
    _update_comp_cache as _update_comp_cache_original,
    repeat_kv,
    apply_rotary_pos_emb,
)


# New fused kernels imported inside profiled_dct_page_attention_forward
from triton_kernels import assemble_kv_drop_triton
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicLayer

import contextlib
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile as _torch_profile, ProfilerActivity


_SDPA_BACKEND_MAP = {
    "auto":         None,
    "flash":        [SDPBackend.FLASH_ATTENTION],
    "mem_efficient":[SDPBackend.EFFICIENT_ATTENTION],
    "math":         [SDPBackend.MATH],
    "cudnn":        [SDPBackend.CUDNN_ATTENTION],
}


def _sdpa_backend_context(backend_str):
    """Return a context manager that pins SDPA to the chosen backend.

    `auto` keeps PyTorch's default dispatcher (all backends allowed). Any
    other value whitelists ONLY that backend — SDPA will raise rather than
    silently fall back, which is what we want when benchmarking.
    """
    backends = _SDPA_BACKEND_MAP.get(backend_str)
    if backends is None:
        return contextlib.nullcontext()
    return sdpa_kernel(backends)


_SDPA_KERNEL_SIGNATURES = [
    # (label, substrings — any match wins)
    ("FlashAttention-2",     ("flash_fwd", "flash::flash_fwd", "flash_attn")),
    ("memory-efficient",     ("fmha_cutlass", "cutlassf", "mem_efficient_attention")),
    ("cuDNN attention",      ("cudnn_attention", "cudnn_fused_attention")),
    ("math (unfused)",       ("scaled_dot_product_attention_math", "sdp_math")),
]


def _probe_sdpa_backend(model, past_key_values, next_token, current_pos,
                        backend_ctx, mode_label):
    """Run ONE decode step under torch.profiler, grep kernel names, and
    print which SDPA backend actually fired. `backend_ctx` is a context
    manager for the first `with` block; a second one is constructed via
    the stashed `_requested_label -> backend_str` map for the profile run
    (sdpa_kernel isn't re-entrant on a single instance)."""
    device = next_token.device
    static_pos = torch.tensor([current_pos], device=device, dtype=torch.long)
    # Resolve the backend string from the requested_label so we can build a
    # second fresh context for the profiled call.
    requested_label = getattr(_probe_sdpa_backend, "_requested_label", None)
    label_to_str = {
        "FlashAttention-2": "flash", "memory-efficient": "mem_efficient",
        "math (unfused)": "math", "cuDNN attention": "cudnn", None: "auto",
    }
    backend_str = label_to_str.get(requested_label, "auto")

    # Warmup (cuDNN + FA pick plans on first call; skip that cost from the trace).
    with backend_ctx, torch.no_grad():
        model(next_token, past_key_values=past_key_values,
              use_cache=True, cache_position=static_pos)
    torch.cuda.synchronize(device)

    with _sdpa_backend_context(backend_str), torch.no_grad(), _torch_profile(
        activities=[ProfilerActivity.CUDA], record_shapes=False,
    ) as prof:
        model(next_token, past_key_values=past_key_values,
              use_cache=True, cache_position=static_pos)
    torch.cuda.synchronize(device)

    kernel_names = []
    for ev in prof.events():
        dur = getattr(ev, "cuda_time_total", 0) or getattr(ev, "self_cuda_time_total", 0)
        if dur > 0 and ev.name:
            kernel_names.append(ev.name.lower())

    hits = {label: [] for label, _ in _SDPA_KERNEL_SIGNATURES}
    unmatched_attn_like = []
    for name in kernel_names:
        matched = False
        for label, needles in _SDPA_KERNEL_SIGNATURES:
            if any(n in name for n in needles):
                hits[label].append(name)
                matched = True
                break
        if not matched and ("attention" in name or "sdpa" in name or "fmha" in name):
            unmatched_attn_like.append(name)

    print(f"\n  [SDPA probe: {mode_label}]")
    any_hit = False
    for label, _ in _SDPA_KERNEL_SIGNATURES:
        if hits[label]:
            any_hit = True
            sample = hits[label][0]
            print(f"    -> {label}  (e.g. {sample[:90]})  count={len(hits[label])}")
    if not any_hit:
        print(f"    -> NO known SDPA kernel matched. "
              f"Attention-like kernels observed: {list(set(unmatched_attn_like))[:5]}")
    # Extra: print a warning if FA2 was requested but not observed.
    requested = getattr(_probe_sdpa_backend, "_requested_label", None)
    if requested == "FlashAttention-2" and not hits["FlashAttention-2"]:
        print(f"    [WARN] FA2 was requested via --sdpa_backend flash but "
              f"no flash_fwd kernels were observed.")
    elif requested == "FlashAttention-2":
        print(f"    [OK] FA2 confirmed.")


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

    def get_seq_length(self, cache_position=None):
        return self._seen


def _get_attention_interface(attn_module):
    if attn_module.config._attn_implementation == "eager":
        return eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS[attn_module.config._attn_implementation]


def pre_allocate_cache(cache, extra_tokens=256):
    """Convert a DynamicCache (after prefill) to use pre-allocated layers."""
    for i, layer in enumerate(cache.layers):
        cache.layers[i] = PreAllocatedLayer.from_dynamic_layer(layer, extra_tokens)
    return cache


# ---------------------------------------------------------------------------
# Timing storage
# ---------------------------------------------------------------------------
_step_timings = defaultdict(list)  # step_name -> list of ms
_cpu_timings = defaultdict(list)   # step_name -> list of ms (CPU wall-clock with sync)
_pending_events = []  # (name, start_event, end_event) — flushed after each decode step
_enabled = False
_sync_mode = False  # when True, add torch.cuda.synchronize() between steps for CPU timing
_current_layer = 0


class _ProfileTopKImpl:
    """Tiny mutable holder so the forward can read the chosen topk impl
    without threading a kwarg through each call."""
    value = 'auto'


_profile_topk_impl = _ProfileTopKImpl()


class _ProfileAttnBackend:
    """Holds the attention backend selection ('sdpa' or 'quest') and the
    optional verify flag. Set from CLI in main()."""
    value = 'sdpa'
    verify = False


_profile_attn_backend = _ProfileAttnBackend()
_quest_cache_ref = [None]  # single-element list; stashed after post-prefill build


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

    from dct_page_attention import _dct_page_cfg as cfg, _maybe_reset_dct_runtime_state
    from typing import Callable

    input_shape = hidden_states.shape[:-1]  # (bsz, q_len)
    hidden_shape = (*input_shape, -1, self.head_dim)  # (bsz, q_len, num_heads, head_dim)
    bsz, q_len = input_shape
    _maybe_reset_dct_runtime_state(self, past_key_values)

    if q_len > 1:
        # Prefill path — always use standard attention (no profiling)
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

        attention_interface = _get_attention_interface(self)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ---- DECODE PATH (q_len == 1) ----
    # Peek at KV length for the short-KV fallback without running step 1/2
    # (matches dct_page_attention.py:1152).
    if past_key_values is not None:
        prev_len = int(past_key_values.layers[self.layer_idx].get_seq_length())
    else:
        prev_len = 0
    projected_kv_len = prev_len + q_len

    min_len_for_paging = max(
        cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size,
        getattr(cfg, "min_decode_kv_len_for_paging", 0),
    )
    if projected_kv_len < min_len_for_paging:
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
        attention_interface = _get_attention_interface(self)
        attn_output, _ = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        _current_layer += 1
        return attn_output, None

    # Chained CUDA events: ev[i] is the boundary between step i and step i+1.
    # Total attention = ev[0] → ev[-1] = exact sum of all profiled steps.
    if _enabled:
        _dev = hidden_states.device
        _stream = torch.cuda.current_stream(_dev)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
        _cpu_ts = []

        def _rec(i):
            if _sync_mode:
                torch.cuda.synchronize(_dev)
            ev[i].record(_stream)
            if _sync_mode:
                _cpu_ts.append(time.perf_counter())

        _rec(0)

    # Step 1: qkv
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _enabled:
        _rec(1)

    # Step 2: RoPE + KV cache update (post-RoPE stored in cache)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    kv_len = key_states.shape[2]

    if _enabled:
        _rec(2)

    # Step 3: segment
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    (sink_k, sink_v, paged_k, paged_v,
     recent_k, recent_v, num_pages, actual_recent) = segment_kv(
        key_states, value_states, cfg
    )

    if _enabled:
        _rec(3)

    # Step 4: compressed page cache maintenance. comp_k is always needed for
    # scoring; comp_v is only stored in "compressed" mode (drop mode returns
    # comp_v=None).
    comp_k, comp_v = _update_comp_cache_original(
        self, paged_k, paged_v, num_pages, comp_size, cfg,
    )

    if _enabled:
        _rec(4)

    # Step 5a: score cache update
    from triton_kernels import (
        score_pages_triton,
        topk_sort,
        topk_sort_triton,
        topk_sort_twostage_triton,
        topk_sort_torch,
    )

    _num_kv_heads = self.config.num_key_value_heads  # 8 for Llama-3.1-8B
    page_scores_buf = getattr(self, '_page_scores_buf', None)
    if (
        page_scores_buf is None
        or page_scores_buf.shape[0] != bsz
        or page_scores_buf.shape[1] != _num_kv_heads
        or page_scores_buf.shape[2] != num_pages
    ):
        self._page_scores_buf = torch.empty(
            bsz, _num_kv_heads, num_pages,
            dtype=torch.float32, device=paged_k.device,
        )

    score_query_states = query_states
    score_comp_k = comp_k

    if _enabled:
        _rec(5)

    # Step 5b: score pages kernel
    page_scores = score_pages_triton(
        score_query_states, score_comp_k,
        cfg.scoring_method, cfg.group_agg_method,
        self.num_key_value_groups,
        out=self._page_scores_buf,
    )

    if _enabled:
        _rec(6)

    # Step 6: topk
    actual_top_k = min(cfg.top_k, num_pages)

    topk_buf = getattr(self, '_topk_out_buf', None)
    if (
        topk_buf is None
        or topk_buf.shape[0] != bsz
        or topk_buf.shape[1] != _num_kv_heads
        or topk_buf.shape[2] != actual_top_k
    ):
        self._topk_out_buf = torch.empty(
            bsz, _num_kv_heads, actual_top_k, dtype=torch.int32, device=paged_k.device
        )
    # Scratch for two-stage dispatch; large enough for NUM_CHUNKS=8.
    scratch_n = 8 * actual_top_k
    scratch_buf = getattr(self, '_topk_scratch_buf', None)
    if (
        scratch_buf is None
        or scratch_buf.shape[0] != bsz
        or scratch_buf.shape[1] != _num_kv_heads
        or scratch_buf.shape[2] < scratch_n
    ):
        self._topk_scratch_buf = torch.empty(
            bsz, _num_kv_heads, scratch_n, dtype=torch.int64, device=paged_k.device
        )

    topk_impl = getattr(_profile_topk_impl, 'value', 'auto')
    _sort_ascending = (cfg.unselected_mode == "compressed")
    if topk_impl == 'fused':
        selected_indices = topk_sort_triton(
            page_scores, actual_top_k, out=self._topk_out_buf,
            sort_ascending=_sort_ascending,
        )
    elif topk_impl == 'twostage':
        selected_indices = topk_sort_twostage_triton(
            page_scores, actual_top_k,
            out=self._topk_out_buf,
            scratch=self._topk_scratch_buf[:, :, :scratch_n],
            sort_ascending=_sort_ascending,
        )
    elif topk_impl == 'torch':
        selected_indices = topk_sort_torch(
            page_scores, actual_top_k, out=self._topk_out_buf,
            sort_ascending=_sort_ascending,
        )
    else:  # auto
        selected_indices = topk_sort(
            page_scores, actual_top_k,
            out=self._topk_out_buf,
            scratch=self._topk_scratch_buf[:, :, :scratch_n],
            sort_ascending=_sort_ascending,
        )

    if _enabled:
        _rec(7)

    if cfg.unselected_mode == "drop":
        assembled_len = cfg.sink_size + actual_top_k * cfg.page_size + actual_recent
    else:
        num_unselected = num_pages - actual_top_k
        middle_len = actual_top_k * cfg.page_size + num_unselected * comp_size
        assembled_len = cfg.sink_size + middle_len + actual_recent

    # Pre-allocate or expand output buffers (avoids torch.empty per step).
    # Only needed for the SDPA path — Quest reads directly from its own cache.
    _backend = _profile_attn_backend.value
    if _backend == "sdpa" or _profile_attn_backend.verify:
        _buf_len = getattr(self, '_assemble_buf_len', 0)
        if assembled_len > _buf_len:
            _max_len = assembled_len + cfg.page_size
            _nkv = _num_kv_heads
            self._final_k_buf = torch.empty(bsz, _nkv, _max_len, self.head_dim, dtype=paged_k.dtype, device=paged_k.device)
            self._final_v_buf = torch.empty_like(self._final_k_buf)
            self._sel_idx_buf = torch.empty(bsz, _nkv, actual_top_k, dtype=torch.int32, device=paged_k.device)
            self._assemble_buf_len = _max_len

    # Step 7: assemble KV (SDPA) or pack Quest page indices. Cache is
    # post-RoPE, so no RoPE is re-applied during assembly.
    if _backend == "quest":
        import quest_backend
        qcache = _quest_cache_ref[0]
        # Append this step's K/V into Quest's paged layout (in-place, O(1)).
        # key_states/value_states include the just-inserted token at position -1.
        quest_backend.append_quest_cache(
            qcache, key_states[:, :, -1:, :], value_states[:, :, -1:, :], self.layer_idx,
        )
        # pack_indices is alloc-free and produces a fixed-width index tensor
        # (qcache.max_total_selected). sink / top_k / recent page counts are
        # all build-time constants on qcache — per-step args are unused.
        packed, indptr = quest_backend.pack_indices(qcache, selected_indices[0])
        # Optionally also run the SDPA path for verification (writes to same
        # buffers; result discarded unless verify).
        if _profile_attn_backend.verify and cfg.unselected_mode == "drop":
            final_k, final_v = assemble_kv_drop_triton(
                paged_k, paged_v,
                sink_k, sink_v, recent_k, recent_v,
                selected_indices,
                None, None,
                out_k=self._final_k_buf,
                out_v=self._final_v_buf,
                out_sel_idx=self._sel_idx_buf,
                original_position_rope=False,
            )
    else:
        if cfg.unselected_mode == "drop":
            final_k, final_v = assemble_kv_drop_triton(
                paged_k, paged_v,
                sink_k, sink_v, recent_k, recent_v,
                selected_indices,
                None, None,
                out_k=self._final_k_buf,
                out_v=self._final_v_buf,
                out_sel_idx=self._sel_idx_buf,
                original_position_rope=False,
            )
        else:
            raise NotImplementedError("profile_decode only supports the current drop-mode default path.")

    if _enabled:
        _rec(8)

    # Step 8: attention (SDPA or Quest paged kernel).
    if _backend == "quest":
        import quest_backend
        rope_theta = float(getattr(self.config, "rope_theta", 1e4))
        quest_out = quest_backend.quest_decode_attention(
            query_states, _quest_cache_ref[0], packed, indptr, self.layer_idx,
            rope_scale=1.0, rope_theta=rope_theta,
        )  # (1, num_qo_heads, 1, head_dim)

        if _profile_attn_backend.verify:
            sdpa_out = F.scaled_dot_product_attention(
                query_states, final_k, final_v,
                is_causal=False, enable_gqa=True,
            )
            max_diff = (quest_out - sdpa_out).abs().max().item()
            if not hasattr(self, "_verify_diffs"):
                self._verify_diffs = []
            self._verify_diffs.append(max_diff)
            # Use SDPA output for downstream layers so errors do not compound.
            attn_output = sdpa_out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        else:
            attn_output = quest_out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    else:
        attn_output = F.scaled_dot_product_attention(
            query_states, final_k, final_v,
            is_causal=False, enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

    if _enabled:
        _rec(9)

    attn_output = self.o_proj(attn_output)

    if _enabled:
        _rec(10)
        step7_label = "7_pack_indices" if _backend == "quest" else "7_assemble_drop"
        step8_label = "8_quest_attn" if _backend == "quest" else "8_sdpa"
        step_names = [
            "1_qkv_proj", "2_rope_and_cache_append", "3_segment",
            "4_compress", "5a_score_cache_update", "5b_score_pages_kernel",
            "6_topk", step7_label,
            step8_label, "9_o_proj",
        ]
        for i, name in enumerate(step_names):
            _pending_events.append((name, ev[i], ev[i + 1]))
        if _sync_mode:
            for i, name in enumerate(step_names):
                cpu_ms = (_cpu_ts[i + 1] - _cpu_ts[i]) * 1000
                _cpu_timings[name].append(cpu_ms)

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

    Chained CUDA events with step names matching the DCT forward
    (1_qkv_proj, 2_rope_and_cache_append, 8_sdpa, 9_o_proj) for direct
    comparison. Run with pre-allocated KV cache (same backend as DCT) so
    2_rope_and_cache_append is apples-to-apples.
    """
    from typing import Callable

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    if q_len > 1:
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

        attention_interface = _get_attention_interface(self)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ---- Decode path (fast SDPA; events only when _enabled) ----
    if _enabled:
        _dev = hidden_states.device
        _stream = torch.cuda.current_stream(_dev)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        ev[0].record(_stream)

    # Step 1: QKV projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _enabled:
        ev[1].record(_stream)

    # Step 2: RoPE + KV cache update
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if _enabled:
        ev[2].record(_stream)

    # Step 8: SDPA (matches DCT's step 8 label for the comparison table)
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        is_causal=False,  # q_len=1 decode: no future positions to mask
        enable_gqa=True,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    if _enabled:
        ev[3].record(_stream)

    attn_output = self.o_proj(attn_output)

    if _enabled:
        ev[4].record(_stream)
        step_names = ["1_qkv_proj", "2_rope_and_cache_append", "8_sdpa", "9_o_proj"]
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
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--scoring_method", default="max", choices=["mean", "max"])
    p.add_argument("--group_agg_method", default="max", choices=["mean", "max"])
    p.add_argument("--unselected_mode", default="drop",
                   choices=["drop", "compressed"])
    p.add_argument("--compressed_token_rope", default="mixed", choices=["mixed", "block_center"])
    p.add_argument("--no_triton", action="store_true",
                   help="Disable Triton kernels (use pure PyTorch for comparison)")
    p.add_argument("--comp_kv_quant", type=str, default="none",
                   choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"],
                   help="Fake-quantization of compressed K/V at write time "
                        "(precision study; no real byte-level storage change)")
    p.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                   choices=["per_page", "per_comp_token"],
                   help="Scale granularity for comp_kv_quant")
    p.add_argument("--sync", action="store_true",
                   help="Add torch.cuda.synchronize() between steps to get CPU timing breakdown")
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Chunked prefill size (0 = single-pass prefill). "
                        "Use e.g. 8192 to reduce peak memory for long contexts.")
    p.add_argument("--topk_impl", choices=["auto", "fused", "twostage", "torch"],
                   default="auto",
                   help="Which top-k implementation to use in the decode step. "
                        "auto = single-stage for num_pages<=1024, two-stage above.")
    p.add_argument("--benchmark_topk", action="store_true",
                   help="Run a standalone micro-benchmark of the three topk "
                        "implementations over representative shapes and exit.")
    p.add_argument("--cudagraph", action="store_true",
                   help="After the per-step profile, also capture one decode step "
                        "into a CUDA graph and benchmark replay throughput. "
                        "Measures the kernel-launch-overhead-free ceiling at the "
                        "given context length. During replay the KV cache is "
                        "overwritten in-place at the same slot each iteration, "
                        "so the output is not used — pure perf measurement.")
    p.add_argument("--cudagraph_replays", type=int, default=0,
                   help="Number of graph replays to time (0 = use --num_decode_steps).")
    p.add_argument("--attention_backend", choices=["sdpa", "quest"], default="sdpa",
                   help="Attention kernel used in the DCT decode path. "
                        "'sdpa' assembles selected pages then runs "
                        "F.scaled_dot_product_attention (current default). "
                        "'quest' skips the gather and calls Quest's FlashInfer "
                        "paged decode kernel directly on selected pages.")
    p.add_argument("--verify_quest", action="store_true",
                   help="Run both SDPA and Quest paths per layer and log "
                        "max-abs-diff (bf16 tolerance). Implies --attention_backend quest.")
    p.add_argument("--sdpa_backend",
                   choices=["auto", "flash", "mem_efficient", "math", "cudnn"],
                   default="flash",
                   help="Pin PyTorch's SDPA dispatcher to a single backend for "
                        "fair comparison. 'flash' = FA2 only (raises if SDPA "
                        "can't route there). 'auto' = PyTorch's default heuristic. "
                        "A one-step torch.profiler probe before the measured loop "
                        "prints which backend actually fired.")
    args = p.parse_args()
    if args.verify_quest:
        args.attention_backend = "quest"
    return args


def run_benchmark_topk():
    """Standalone micro-benchmark: fused vs twostage vs torch across shapes.

    Reports average per-call time and validates that all three kernels produce
    the same output.
    """
    from triton_kernels import (
        topk_sort_triton, topk_sort_twostage_triton, topk_sort_torch,
    )

    def bench(fn, *args, trials=1000, warmup=50):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(trials):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / trials * 1000  # us

    print("Top-K micro-benchmark (bsz=1, num_kv_heads=8, top_k=64)")
    print(f"{'num_pages':>10} {'fused us':>10} {'2stage-4':>10} {'2stage-8':>10} {'torch':>8} {'ok':>5}")
    torch.manual_seed(0)
    top_k = 64
    for num_pages in [512, 1020, 1024, 1025, 2040, 2048]:
        scores = torch.randn(1, 8, num_pages, dtype=torch.float32, device='cuda')
        out_buf = torch.empty(1, 8, top_k, dtype=torch.int32, device='cuda')
        scratch4 = torch.empty(1, 8, 4 * top_k, dtype=torch.int64, device='cuda')
        scratch8 = torch.empty(1, 8, 8 * top_k, dtype=torch.int64, device='cuda')

        us_fused = bench(topk_sort_triton, scores, top_k, out_buf)
        us_ts4 = bench(topk_sort_twostage_triton, scores, top_k, out_buf, scratch4, 4)
        us_ts8 = bench(topk_sort_twostage_triton, scores, top_k, out_buf, scratch8, 8)
        us_torch = bench(topk_sort_torch, scores, top_k, out_buf)

        a = topk_sort_triton(scores, top_k).clone()
        b = topk_sort_twostage_triton(scores, top_k, num_chunks=4)
        c = topk_sort_twostage_triton(scores, top_k, num_chunks=8)
        d = topk_sort_torch(scores, top_k)
        ok = torch.equal(a, b) and torch.equal(a, c) and torch.equal(a, d)

        print(f"{num_pages:>10} {us_fused:>10.2f} {us_ts4:>10.2f} {us_ts8:>10.2f} {us_torch:>8.2f} {str(ok):>5}")


# ---------------------------------------------------------------------------
# CUDA graph capture + replay benchmark
# ---------------------------------------------------------------------------
def _capture_and_benchmark(model, past_key_values, next_token, current_pos,
                           num_replays):
    """Capture one decode step into a CUDA graph, benchmark N replays.

    Called after warmup/profile so kernels are JIT'd and buffers pre-allocated.
    The returned tok/s is the steady-state ceiling at the current context
    length: every replay overwrites the same KV slot, so real text generation
    would diverge numerically — this is strictly a performance measurement.
    """
    global _enabled
    _enabled = False  # no CUDA event recording inside capture/replay

    device = next_token.device
    static_input = next_token.clone()
    static_pos = torch.tensor([current_pos], device=device, dtype=torch.long)

    # Prime on a side stream (torch.cuda.graph requires this for correct
    # allocator state before capture).
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                model(static_input, past_key_values=past_key_values,
                      use_cache=True, cache_position=static_pos)
    torch.cuda.current_stream(device).wait_stream(s)
    torch.cuda.synchronize(device)

    print(f"  Capturing CUDA graph...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            model(static_input, past_key_values=past_key_values,
                  use_cache=True, cache_position=static_pos)

    # Warm-up replays before timing (first few replays have minor one-time cost)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize(device)

    print(f"  Replaying graph ({num_replays} steps)...")
    t0 = time.perf_counter()
    for _ in range(num_replays):
        g.replay()
    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000
    per_replay_ms = total_ms / num_replays
    tok_s = 1000.0 / per_replay_ms
    return per_replay_ms, tok_s


# ---------------------------------------------------------------------------
# Run profiled decode
# ---------------------------------------------------------------------------
def run_profiled_decode(model, tokenizer, args, mode):
    global _step_timings, _cpu_timings, _enabled, _current_layer

    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    # Generate dummy input
    input_ids = torch.randint(0, vocab_size, (1, args.context_length),
                              dtype=torch.long, device=device)

    # Prefill (not profiled)
    chunk_size = getattr(args, 'chunk_size', 0)
    if chunk_size > 0:
        print(f"  Prefilling ({args.context_length} tokens, chunk_size={chunk_size})...")
    else:
        print(f"  Prefilling ({args.context_length} tokens)...")
    _enabled = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = chunked_prefill(model, input_ids, chunk_size)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prefill done: {prefill_ms:.0f}ms")

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)
    prefill_len = args.context_length

    # Pre-allocated cache: both DCT and baseline use it so step 2
    # (2_rope_and_cache_append) is apples-to-apples. Both paths get an O(1)
    # index-write instead of DynamicCache's torch.cat per step.
    extra = args.warmup_steps + args.num_decode_steps + 16
    past_key_values = pre_allocate_cache(past_key_values, extra_tokens=extra)
    print(f"  Converted to pre-allocated cache (+{extra} tokens)")

    # Build Quest paged KV cache (post-prefill one-time copy) if this is the
    # DCT run and the Quest attention backend was requested. The baseline run
    # never uses Quest.
    _quest_cache_ref[0] = None
    if mode == "dct" and args.attention_backend == "quest":
        import quest_backend
        cfg_model = model.config
        num_kv_heads = cfg_model.num_key_value_heads
        num_qo_heads = cfg_model.num_attention_heads
        head_dim = cfg_model.hidden_size // num_qo_heads
        num_layers = cfg_model.num_hidden_layers
        num_sink_pages = (args.sink_size + args.page_size - 1) // args.page_size
        # Fixed number of EXPLICIT recent pages per step (strictly before
        # last_page_idx; the partially-filled last page is attended via the
        # kernel's paged_kv_last_page_* path on top of this). The variable
        # count would oscillate between ceil(recent/ps) - 1 and ceil(recent/ps)
        # + 1 over a page_size-cycle; fixing to the max keeps
        # total_selected constant (required for CUDA-graph capture) and
        # overshoots the recent window by ≤1 page on the partial-page steps.
        num_recent_pages_fixed = (args.recent_size + args.page_size - 1) // args.page_size + 1
        max_total_selected = num_sink_pages + args.top_k + num_recent_pages_fixed
        max_decode_steps = args.warmup_steps + args.num_decode_steps + 16
        print(f"  Building Quest paged cache (layers={num_layers}, "
              f"page_size={args.page_size}, num_sink_pages={num_sink_pages}, "
              f"top_k={args.top_k}, num_recent_pages_fixed={num_recent_pages_fixed}, "
              f"max_total_selected={max_total_selected})...")
        _quest_cache_ref[0] = quest_backend.build_quest_paged_cache(
            preallocated_layers=past_key_values.layers,
            prefill_len=prefill_len,
            page_size=args.page_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_layers=num_layers,
            max_decode_steps=max_decode_steps,
            dtype=past_key_values.layers[0].keys.dtype,
            device=device,
            num_sink_pages=num_sink_pages,
            top_k=args.top_k,
            num_recent_pages_fixed=num_recent_pages_fixed,
        )
        print(f"  Quest cache ready: {_quest_cache_ref[0].capacity_pages} pages "
              f"allocated, cur_seqlen={_quest_cache_ref[0].cur_seqlen}, "
              f"last_page_idx={_quest_cache_ref[0].last_page_idx}, "
              f"last_page_len={_quest_cache_ref[0].last_page_len}")

    # SDPA backend context: pins PyTorch's SDPA dispatcher to the user-chosen
    # backend for the whole measured region (warmup + profile + cudagraph
    # capture). 'auto' = default dispatcher. Anything else = whitelist ONE
    # backend so SDPA raises rather than silently falling back. A fresh
    # context manager is created at each `with` site (sdpa_kernel isn't
    # guaranteed re-entrant across multiple `with` blocks on the same object).
    sdpa_backend_str = getattr(args, 'sdpa_backend', 'flash')
    _label_map = {
        "flash": "FlashAttention-2", "mem_efficient": "memory-efficient",
        "math": "math (unfused)", "cudnn": "cuDNN attention", "auto": None,
    }
    _probe_sdpa_backend._requested_label = _label_map.get(sdpa_backend_str)
    print(f"  SDPA backend: --sdpa_backend={sdpa_backend_str} "
          f"(requested: {_probe_sdpa_backend._requested_label or 'PyTorch default'})")

    # Probe once (before warmup) to report which kernel actually fires.
    try:
        _probe_sdpa_backend(
            model, past_key_values, next_token,
            current_pos=prefill_len,
            backend_ctx=_sdpa_backend_context(sdpa_backend_str),
            mode_label=mode,
        )
    except Exception as e:
        print(f"  [SDPA probe skipped: {type(e).__name__}: {e}]")

    with _sdpa_backend_context(sdpa_backend_str):
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
        _cpu_timings.clear()
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

    # Verify-mode summary: aggregate per-layer max-abs-diff collected inside
    # the forward on `self._verify_diffs`.
    if _profile_attn_backend.verify and mode == "dct":
        per_layer_max = {}
        per_layer_mean = {}
        for mod in model.modules():
            diffs = getattr(mod, "_verify_diffs", None)
            if diffs:
                lid = getattr(mod, "layer_idx", None)
                per_layer_max[lid] = max(diffs)
                per_layer_mean[lid] = sum(diffs) / len(diffs)
        if per_layer_max:
            print(f"\n  [verify] Quest vs SDPA max-abs-diff across {len(per_layer_max)} layers:")
            for lid in sorted(k for k in per_layer_max if k is not None):
                print(f"    layer {lid:>2}: max={per_layer_max[lid]:.4f}  mean={per_layer_mean[lid]:.4f}")
            worst = max(per_layer_max.values())
            print(f"  [verify] worst layer max-abs-diff = {worst:.4f} "
                  f"(bf16 tolerance ~0.01)")

    # Optional: CUDA graph capture + replay benchmark on the same warmed state.
    # Pre-allocated KV cache has headroom for the +3 priming + 1 capture steps
    # (extra = warmup + num_decode_steps + 16). Captured UNDER the SDPA backend
    # context so the baked-in SDPA kernel matches the measured loop.
    graph_stats = None
    if getattr(args, 'cudagraph', False):
        num_replays = args.cudagraph_replays or args.num_decode_steps
        current_pos = prefill_len + args.warmup_steps + args.num_decode_steps
        try:
            with _sdpa_backend_context(sdpa_backend_str):
                graph_stats = _capture_and_benchmark(
                    model, past_key_values, next_token, current_pos, num_replays
                )
        except Exception as e:
            print(f"  CUDA graph benchmark failed: {type(e).__name__}: {e}")
            graph_stats = None

    return avg_model_total, tok_s, dict(_step_timings), total_times, dict(_cpu_timings), graph_stats


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_profile(mode, avg_model_total, tok_s, timings, num_layers=32, cpu_timings=None):
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

    has_cpu = cpu_timings and len(cpu_timings) > 0
    if has_cpu:
        print(f"  {'Step':<25} {'GPU (ms/tok)':>12} {'CPU+sync (ms/tok)':>18} {'GPU kern (µs)':>14} {'% of attn':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*18} {'-'*14} {'-'*10}")
    else:
        print(f"  {'Step':<25} {'Per-layer (ms)':>15} {'Per-token (ms)':>15} {'% of attn':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    for step_name in step_order:
        vals = timings[step_name]
        avg_per_call = sum(vals) / len(vals) if vals else 0.0
        per_token = step_per_token[step_name]
        pct = per_token / attn_total * 100 if attn_total > 0 else 0.0

        if has_cpu and step_name in cpu_timings:
            cpu_vals = cpu_timings[step_name]
            cpu_per_tok = sum(cpu_vals) / (len(cpu_vals) / num_layers) if cpu_vals else 0.0
            print(f"  {step_name:<25} {per_token:>12.3f} {cpu_per_tok:>18.3f} {avg_per_call*1000:>14.1f} {pct:>9.1f}%")
        elif has_cpu:
            print(f"  {step_name:<25} {per_token:>12.3f} {'—':>18} {avg_per_call*1000:>14.1f} {pct:>9.1f}%")
        else:
            print(f"  {step_name:<25} {avg_per_call:>15.4f} {per_token:>15.3f} {pct:>11.1f}%")

    print(f"  {'-'*25} {'-'*12 if has_cpu else '-'*15} {'-'*18 if has_cpu else '-'*15} {'-'*14 if has_cpu else '-'*12}")
    total_label = 'TOTAL'
    if has_cpu:
        cpu_total = sum(sum(v) / (len(v) / num_layers) for v in cpu_timings.values() if v)
        print(f"  {total_label:<25} {attn_total:>12.3f} {cpu_total:>18.3f} {'':>14} {'100.0':>9}%")
    else:
        print(f"  {total_label:<25} {'':>15} {attn_total:>15.3f} {'100.0':>11}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _sync_mode
    args = parse_args()
    _sync_mode = getattr(args, 'sync', False)

    if getattr(args, 'benchmark_topk', False):
        run_benchmark_topk()
        return

    _profile_topk_impl.value = args.topk_impl
    _profile_attn_backend.value = args.attention_backend
    _profile_attn_backend.verify = args.verify_quest

    import types
    import transformers

    original_forward = get_original_forward(args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)
    model._original_attn_forward = original_forward

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

        avg_total, tok_s, timings, total_times, cpu_timings, graph_stats = run_profiled_decode(
            model, tokenizer, args, "baseline"
        )
        print_profile("baseline", avg_total, tok_s, timings, num_layers, cpu_timings)
        if graph_stats is not None:
            gp, gts = graph_stats
            print(f"\n  CUDA graph: {gp:.3f} ms/step  ({gts:.1f} tok/s)")
            print(f"  Speedup (graph vs profiled): {avg_total / gp:.2f}x  "
                  f"(saved {avg_total - gp:.2f} ms/step)")
        results["baseline"] = (avg_total, tok_s, timings, graph_stats)

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

        avg_total, tok_s, timings, total_times, cpu_timings, graph_stats = run_profiled_decode(
            model, tokenizer, args, "dct"
        )
        print_profile("dct", avg_total, tok_s, timings, num_layers, cpu_timings)
        if graph_stats is not None:
            gp, gts = graph_stats
            print(f"\n  CUDA graph: {gp:.3f} ms/step  ({gts:.1f} tok/s)")
            print(f"  Speedup (graph vs profiled): {avg_total / gp:.2f}x  "
                  f"(saved {avg_total - gp:.2f} ms/step)")
        results["dct"] = (avg_total, tok_s, timings, graph_stats)

        del total_times
        torch.cuda.empty_cache()

    # Comparison
    if "baseline" in results and "dct" in results:
        b_avg, b_tok, b_timings, b_graph = results["baseline"]
        d_avg, d_tok, d_timings, d_graph = results["dct"]

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

        if b_graph is not None and d_graph is not None:
            b_gp, b_gts = b_graph
            d_gp, d_gts = d_graph
            print(f"\n  {'CUDA GRAPH':<25} {'Baseline':>15} {'DCT':>15} {'Diff':>12}")
            print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
            print(f"  {'ms/step':<25} {b_gp:>15.3f} {d_gp:>15.3f} {d_gp-b_gp:>+12.3f}")
            print(f"  {'tok/s':<25} {b_gts:>15.1f} {d_gts:>15.1f} {d_gts-b_gts:>+12.1f}")


if __name__ == "__main__":
    main()
