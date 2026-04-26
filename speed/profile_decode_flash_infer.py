"""
Phase 2b Stage 8 — DCT + FlashInfer eager-mode profiling driver.

Sibling to `speed/profile_decode.py`. Adds a `dct_flashinfer` mode that wires
the Stage 7 forward (`dct_page_attention_forward_flashinfer`) with a
`FlashInferPagedKVCache` built post-prefill, and surfaces `torch.profiler`
trace export so eager-mode kernel times can be measured without the
chained-CUDA-event CPU-dispatch-lag bias noted in
`feedback_profile_decode_eager_bias.md`.

Four modes (selectable via `--mode`, default `dct_flashinfer`):
  - baseline        : full-KV FlashInfer (fair comparison against dct_flashinfer —
                      same attention kernel, no page selection).
  - dct_sdpa        : DCT + SDPA (reuses `profile_decode.profiled_dct_page_attention_forward`
                      with `--attention_backend sdpa`)
  - dct_quest       : DCT + Quest (same profiled forward, `--attention_backend quest`)
  - dct_flashinfer  : DCT + FlashInfer (new profiled forward in this file)

Phase 2b is eager-only. `torch.cuda.graph()` / `torch.cuda.CUDAGraph()` are
never called here — the `--cudagraph` flag from `profile_decode.py` is NOT
carried over.

Usage:
    CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_flash_infer.py \\
        --context_length 32768 --page_size 32 --top_k 64 \\
        --sink_size 32 --recent_size 128 \\
        --num_decode_steps 128 --warmup_steps 8 \\
        --batch_size 1 \\
        --torch_profiler_trace results/phase2b_stage8/flashinfer_eager.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn.functional as F
import transformers

# Shared helpers from the Phase 2 profiler. Import-as-module so we can share
# its module-level state (`_enabled`, `_pending_events`, `_step_timings`,
# `_cpu_timings`, `_sync_mode`, `_current_layer`, `_profile_topk_impl`,
# `_profile_attn_backend`, `_quest_cache_ref`) across modes without duplicating
# the event infrastructure.
import profile_decode as _pd
from profile_decode import (
    PreAllocatedLayer,
    pre_allocate_cache,
    print_profile,
    profiled_dct_page_attention_forward,
)

from speed_test_dummy import (
    load_model_and_tokenizer,
    get_original_forward,
    restore_forward,
    chunked_prefill,
)

import dct_page_attention as _dpa
from dct_page_attention import (
    _flashinfer_cache_ref,
    apply_rotary_pos_emb,
    dct_page_attention_forward,
    dct_page_attention_forward_flashinfer,
    replace_llama_attn,
    segment_kv,
    _maybe_reset_dct_runtime_state,
    _update_comp_cache,
)
from triton_kernels import (
    score_pages_triton,
    topk_sort_and_pack_triton,
)
from flashinfer_backend import (
    FlashInferPagedKVCache,
    _pack_preallocated_to_paged,
    append_flashinfer_cache,
    build_flashinfer_paged_cache,
    flashinfer_decode_attention,
)
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# NVTX / Nsight Systems helpers
# ---------------------------------------------------------------------------
# `torch.profiler`'s kernel-bucket view (see _STAGE_BUCKETS below) can only
# split stages whose kernel names are unique — flashinfer, topk, score_pages,
# compress. It CANNOT tell `1_qkv_proj` from `9_o_proj` (both are cuBLAS gemms)
# or `3_segment` from other view/slice ops. For a complete per-stage breakdown
# under CUDA graph, we emit NVTX range markers alongside the chained-CUDA-event
# boundaries in the profiled forwards. NVTX push/pop calls are captured by
# `torch.cuda.graph()` capture as stream markers, so they replay with the
# kernels and Nsight Systems aggregates per-range GPU time across replays.
#
# Usage (captures only the measurement region when --nsys_range is set):
#   nsys profile --trace=cuda,nvtx \
#       --capture-range=cudaProfilerApi --capture-range-end=stop \
#       python speed/profile_decode_flash_infer.py \
#           --mode dct_flashinfer --cudagraph --nvtx --nsys_range \
#           --context_length 32768 --page_size 32 --top_k 64
#
# Extract per-stage GPU time:
#   nsys stats --report nvtx_gpu_proj_sum <trace>.nsys-rep
#
# Cost outside nsys: each push/pop is O(100 ns) on the CPU — negligible
# at ~1 ms/step even across 32 layers and 8 stages.
_NVTX_ENABLED = False


def _set_nvtx_enabled(on: bool) -> None:
    global _NVTX_ENABLED
    _NVTX_ENABLED = bool(on)


def _nvtx_push(name: str) -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_pop()


def _nvtx_advance(next_name):
    """Close the current NVTX range and open `next_name`. Pass `next_name=None`
    to just close (end-of-forward). No-op when --nvtx is off."""
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_pop()
        if next_name is not None:
            torch.cuda.nvtx.range_push(next_name)


# ---------------------------------------------------------------------------
# Full-KV FlashInfer baseline cache
# ---------------------------------------------------------------------------
# For fair comparison against the DCT+FlashInfer mode, the "baseline" path also
# runs FlashInfer paged decode, but on the FULL KV (no page selection).
#
# Layout mirrors `FlashInferPagedKVCache` from `flashinfer_backend.py`:
#   buf: (num_layers, capacity_pages, 2, page_size, num_kv_heads, head_dim)
#
# Differences vs the DCT path:
#   - No sink / top_k / recent structure; indices = arange(capacity_pages).
#   - indptr[1] grows by 1 each time a new physical page opens (every
#     page_size decode steps). `wrapper.plan()` is re-called at those
#     boundaries; between boundaries only `last_page_len_buf` changes.
#   - `use_cuda_graph=True` still pins buffers so replan is a cheap
#     in-place refresh of the scheduler.
# ---------------------------------------------------------------------------
@dataclass
class FullKVFlashInferCache:
    buf: torch.Tensor
    wrapper: BatchDecodeWithPagedKVCacheWrapper

    page_size: int
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int
    num_layers: int
    capacity_pages: int

    dtype: torch.dtype
    device: torch.device

    indices_buf: torch.Tensor          # (capacity_pages,) int32, arange
    indptr_buf: torch.Tensor           # (2,) int32 = [0, num_active_pages]
    last_page_len_buf: torch.Tensor    # (1,) int32

    cur_seqlen: int = 0
    last_page_idx_py: int = 0
    last_page_len_py: int = 0
    num_active_pages: int = 0


def build_fi_baseline_cache(
    preallocated_layers,
    prefill_len: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    num_qo_heads: int,
    num_layers: int,
    max_decode_steps: int,
    dtype: torch.dtype,
    device: torch.device,
    workspace_bytes: int = 128 * 1024 * 1024,
) -> FullKVFlashInferCache:
    """Pack prefill into an FI paged buffer and plan a full-KV decode wrapper.
    No page selection — indices_buf is pre-filled as arange(capacity_pages)
    and `indptr_buf[1]` tracks the number of actually-filled pages. `plan()`
    is re-called each time that count changes (inside the forward)."""
    prefill_pages = (prefill_len + page_size - 1) // page_size
    decode_pages = (max_decode_steps + page_size - 1) // page_size
    capacity_pages = prefill_pages + decode_pages + 4

    buf = torch.zeros(
        num_layers, capacity_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    _pack_preallocated_to_paged(
        buf, preallocated_layers, prefill_len, page_size,
        num_layers, num_kv_heads, head_dim, dtype,
    )

    indices_buf = torch.arange(capacity_pages, dtype=torch.int32, device=device)
    indptr_buf = torch.zeros(2, dtype=torch.int32, device=device)
    indptr_buf[1] = prefill_pages

    last_open_page = (prefill_len - 1) // page_size
    last_open_len = prefill_len - last_open_page * page_size  # in [1, page_size]
    last_page_len_buf = torch.full(
        (1,), last_open_len, dtype=torch.int32, device=device,
    )

    float_workspace_buffer = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device,
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_page_len_buf,
    )
    # Plan with a slice so the scheduler sees exactly the active page count.
    # The wrapper's internal `_paged_kv_indices_buf` IS `indices_buf`, so the
    # plan()-side `.copy_` is an in-place self-copy (no-op, cheap).
    wrapper.plan(
        indptr_buf, indices_buf[:prefill_pages], last_page_len_buf,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_data_type=dtype, kv_data_type=dtype,
    )

    return FullKVFlashInferCache(
        buf=buf, wrapper=wrapper,
        page_size=page_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        num_qo_heads=num_qo_heads, num_layers=num_layers,
        capacity_pages=capacity_pages,
        dtype=dtype, device=device,
        indices_buf=indices_buf, indptr_buf=indptr_buf,
        last_page_len_buf=last_page_len_buf,
        cur_seqlen=prefill_len,
        last_page_idx_py=last_open_page,
        last_page_len_py=last_open_len,
        num_active_pages=prefill_pages,
    )


# Module-level ref, populated post-prefill in _run_one_mode.
_fi_baseline_cache_ref = [None]


# ---------------------------------------------------------------------------
# Instrumented DCT + FlashInfer forward
# ---------------------------------------------------------------------------
def profiled_dct_flashinfer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Instrumented `dct_page_attention_forward_flashinfer` for per-stage
    timing. Mirrors the Stage 7 forward structure but emits chained CUDA
    events across 8 boundaries:

        1_qkv_proj
        2_rope_and_cache_append        (RoPE + past_key_values.update +
                                        FI-cache counter advance)
        3_segment                      (segment_kv)
        4_compress                     (_update_comp_cache)
        5_score_pages_kernel           (score_pages_triton, native bf16)
        6_topk_and_pack                (FUSED topk_sort_and_pack_triton)
        7_flashinfer_run               (FI-cache K/V write + FI decode run)
        8_o_proj                       (o_proj)

    Prefill (q_len > 1) and the short-KV fallback delegate to the original
    `dct_page_attention_forward` — same path as Stage 7, no profiling there.

    Verify path: when `self._verify_flashinfer == True`, the SAME pages FI
    used are gathered from `cache.buf[layer_idx]` via `cache.indices_buf` and
    SDPA is run on them. Max-abs-diff is appended to `self._verify_diffs`.
    Gathered outside the event window (between ev[7] and ev[8]) to avoid
    biasing the attention-kernel timing with the verification cost.
    """
    cfg = _dpa._dct_page_cfg
    if cfg.unselected_mode != "drop":
        raise NotImplementedError(
            "profiled_dct_flashinfer_forward supports drop mode only"
        )
    if cfg.continuous_rope:
        raise NotImplementedError("continuous_rope=True is temporarily disabled")

    input_shape = hidden_states.shape[:-1]
    bsz, q_len = input_shape
    _maybe_reset_dct_runtime_state(self, past_key_values)

    # Prefill: delegate to the non-profiled SDPA forward. Prefill is NOT
    # profiled; all event emission lives in the decode branch.
    if q_len > 1:
        return dct_page_attention_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    # Short-KV fallback: delegate to the SDPA decode path. No events emitted
    # in this branch — caller should set `min_decode_kv_len_for_paging` low
    # enough (or context long enough) that profiled steps always hit the
    # paged branch.
    min_len_for_paging = max(
        cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size,
        getattr(cfg, "min_decode_kv_len_for_paging", 0),
    )
    if past_key_values is not None:
        prev_len = int(past_key_values.layers[self.layer_idx].get_seq_length())
    else:
        prev_len = 0
    if prev_len + q_len < min_len_for_paging:
        return dct_page_attention_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    hidden_shape = (*input_shape, -1, self.head_dim)
    _has_qk_norm = hasattr(self, "q_norm") and hasattr(self, "k_norm")

    if _pd._enabled:
        _dev = hidden_states.device
        _stream = torch.cuda.current_stream(_dev)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
        _cpu_ts = []

        def _rec(i):
            if _pd._sync_mode:
                torch.cuda.synchronize(_dev)
            ev[i].record(_stream)
            if _pd._sync_mode:
                _cpu_ts.append(time.perf_counter())

        _rec(0)

    _nvtx_push("1_qkv_proj")

    # Step 1: QKV projection (+ optional Qwen3 q_norm/k_norm)
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if _has_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _pd._enabled:
        _rec(1)
    _nvtx_advance("2_rope_and_cache_append")

    # Step 2: RoPE + DCT cache update + FI cache counter advance (layer 0 only).
    # The counter advance is part of the "cache append" logical step — its
    # per-step cost is a couple of Python-side int bumps plus two `fill_` ops
    # on layer 0. The actual K/V copy into cache.buf happens in Step 7 so that
    # FI-bookkeeping cost stays paired with the FI kernel launch.
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    cache = _flashinfer_cache_ref[0]
    if cache is None:
        raise RuntimeError(
            "FlashInfer cache is not set. Build via "
            "speed.flashinfer_backend.build_flashinfer_paged_cache(...) "
            "post-prefill and assign to "
            "dct_page_attention._flashinfer_cache_ref[0] before decode."
        )
    if cache.top_k != cfg.top_k:
        raise RuntimeError(
            f"cfg.top_k ({cfg.top_k}) != FlashInfer cache.top_k "
            f"({cache.top_k}); cache top_k is fixed at build time."
        )

    if self.layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1
        cache.last_page_idx.fill_(cache.last_page_idx_py)
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    if _pd._enabled:
        _rec(2)
    _nvtx_advance("3_segment")

    # Step 3: segment DCT's cache into [sink | paged | recent].
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    (sink_k, sink_v, paged_k, paged_v,
     recent_k, recent_v, num_pages, actual_recent) = segment_kv(
        key_states, value_states, cfg,
    )

    if _pd._enabled:
        _rec(3)
    _nvtx_advance("4_compress")

    # Step 4: compressed page cache maintenance (DCT proxy for scoring).
    comp_k, comp_v = _update_comp_cache(
        self, paged_k, paged_v, num_pages, comp_size, cfg,
    )

    if _pd._enabled:
        _rec(4)
    _nvtx_advance("5_score_pages_kernel")

    # Step 5: score pages (native bf16, no fp16 cast).
    _num_kv_heads = self.config.num_key_value_heads
    page_scores_buf = getattr(self, '_page_scores_buf', None)
    if (
        page_scores_buf is None
        or page_scores_buf.shape[0] != bsz
        or page_scores_buf.shape[1] != _num_kv_heads
        or page_scores_buf.shape[2] < num_pages
    ):
        self._page_scores_buf = torch.empty(
            bsz, _num_kv_heads, num_pages,
            dtype=torch.float32, device=paged_k.device,
        )
    page_scores = score_pages_triton(
        query_states, comp_k,
        cfg.scoring_method, cfg.group_agg_method,
        self.num_key_value_groups,
        out=self._page_scores_buf[:, :, :num_pages],
    )

    if _pd._enabled:
        _rec(5)
    _nvtx_advance("6_topk_and_pack")

    # Step 6: fused topk + pack. Writes middle + recent slices of
    # `cache.indices_buf`; sink slice is static (filled once at cache init).
    # num_middle_pages uses the FI-aligned page count (may be num_pages - 1
    # on cycle-boundary steps — see Stage 7 SUMMARY "Alignment quirk").
    num_middle_pages = (
        cache.last_page_idx_py - cache.num_sink_pages
        - cache.num_recent_pages_fixed + 1
    )
    if num_middle_pages < cache.top_k:
        raise RuntimeError(
            f"num_middle_pages ({num_middle_pages}) < cache.top_k "
            f"({cache.top_k}). Configure min_decode_kv_len_for_paging to "
            f"keep profiled steps in the steady-state paged regime."
        )
    effective_num_pages = min(num_pages, num_middle_pages)
    eff_scores = page_scores[:, :, :effective_num_pages]
    topk_sort_and_pack_triton(
        eff_scores,
        cache.indices_buf,
        num_sink_pages=cache.num_sink_pages,
        top_k=cache.top_k,
        last_page_idx=cache.last_page_idx,
        recent_offsets=cache.recent_offsets,
        sort_ascending=False,
    )

    if _pd._enabled:
        _rec(6)
    _nvtx_advance("7_flashinfer_run")

    # Step 7: FI-cache K/V write + FI decode run. Both charged to the same
    # bucket so the FlashInfer-side cost is a single atomic measurement.
    page_idx = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    if page_idx >= cache.capacity_pages:
        raise RuntimeError(
            f"FlashInferPagedKVCache overflow: page_idx={page_idx} >= "
            f"capacity_pages={cache.capacity_pages}"
        )
    k_flat = key_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
    v_flat = value_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
    cache.buf[self.layer_idx, page_idx, 0, slot].copy_(k_flat)
    cache.buf[self.layer_idx, page_idx, 1, slot].copy_(v_flat)
    attn_output_fi = flashinfer_decode_attention(query_states, cache, self.layer_idx)

    if _pd._enabled:
        _rec(7)
    # Close 7_flashinfer_run here so the optional verify block (below) does
    # NOT fall inside any NVTX stage range. 8_o_proj is opened right before
    # the o_proj line so it excludes verify cost for nsys, exactly like it
    # already does for the chained CUDA events.
    _nvtx_advance(None)

    # Optional verify (NOT inside the event window — adds ~tens of µs of
    # SDPA work plus a Python-side per-head gather that we don't want to
    # charge against `7_flashinfer_run`).
    if getattr(self, "_verify_flashinfer", False):
        buf_l = cache.buf[self.layer_idx]
        page_budget = cache.page_budget
        last_page_len = cache.last_page_len_py
        full_len = (page_budget - 1) * cache.page_size + last_page_len
        k_pages = []
        v_pages = []
        for h in range(_num_kv_heads):
            sel_h = cache.indices_buf[0, h].long()
            kv_h = buf_l[sel_h][:, :, :, h, :]
            k_h = kv_h[:, 0].reshape(page_budget * cache.page_size, self.head_dim)
            v_h = kv_h[:, 1].reshape(page_budget * cache.page_size, self.head_dim)
            k_pages.append(k_h[:full_len])
            v_pages.append(v_h[:full_len])
        k_flat = torch.stack(k_pages, dim=0).unsqueeze(0)
        v_flat = torch.stack(v_pages, dim=0).unsqueeze(0)
        sdpa_out = F.scaled_dot_product_attention(
            query_states, k_flat, v_flat,
            is_causal=False, enable_gqa=True,
        )
        max_diff = (attn_output_fi.float() - sdpa_out.float()).abs().max().item()
        if not hasattr(self, "_verify_diffs"):
            self._verify_diffs = []
        self._verify_diffs.append(max_diff)

    # Step 8: output projection.
    _nvtx_push("8_o_proj")
    attn_output = attn_output_fi.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if _pd._enabled:
        _rec(8)
        step_names = [
            "1_qkv_proj",
            "2_rope_and_cache_append",
            "3_segment",
            "4_compress",
            "5_score_pages_kernel",
            "6_topk_and_pack",
            "7_flashinfer_run",
            "8_o_proj",
        ]
        for i, name in enumerate(step_names):
            _pd._pending_events.append((name, ev[i], ev[i + 1]))
        if _pd._sync_mode:
            for i, name in enumerate(step_names):
                cpu_ms = (_cpu_ts[i + 1] - _cpu_ts[i]) * 1000
                _pd._cpu_timings[name].append(cpu_ms)
    _nvtx_advance(None)  # close 8_o_proj

    _pd._current_layer += 1
    return attn_output, None


# ---------------------------------------------------------------------------
# Instrumented full-KV FlashInfer baseline forward
# ---------------------------------------------------------------------------
def profiled_baseline_flashinfer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Baseline (no page selection) decode that runs attention via FlashInfer
    paged decode on the FULL KV. For fair apples-to-apples comparison against
    the DCT+FlashInfer mode.

    4 chained events:
        1_qkv_proj
        2_rope_and_cache_append   (RoPE + FI cache K/V write; layer-0 also
                                   advances counters and replans when a new
                                   physical page opens)
        8_flashinfer_run          (wrapper.run)
        9_o_proj

    Prefill delegates to the standard attention interface unchanged (no
    profiling there; prefill is not the focus of this tool).
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    if q_len > 1:
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

        attention_interface = _pd._get_attention_interface(self)
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
    cache = _fi_baseline_cache_ref[0]
    if cache is None:
        raise RuntimeError(
            "Full-KV FlashInfer cache is not set. Build via "
            "build_fi_baseline_cache(...) post-prefill and assign to "
            "_fi_baseline_cache_ref[0] before decode."
        )

    if _pd._enabled:
        _dev = hidden_states.device
        _stream = torch.cuda.current_stream(_dev)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        ev[0].record(_stream)

    _nvtx_push("1_qkv_proj")

    # Step 1: QKV projection.
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _pd._enabled:
        ev[1].record(_stream)
    _nvtx_advance("2_rope_and_cache_append")

    # Step 2: RoPE + FI cache append (+ counter advance + optional replan on
    # layer 0 only).
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            # New page opens — num_active_pages grows by 1.
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
            cache.num_active_pages += 1
            if cache.num_active_pages > cache.capacity_pages:
                raise RuntimeError(
                    f"FullKVFlashInferCache overflow: "
                    f"num_active_pages={cache.num_active_pages} > "
                    f"capacity_pages={cache.capacity_pages}. Increase "
                    f"max_decode_steps at build time."
                )
            cache.indptr_buf[1].fill_(cache.num_active_pages)
            # Replan: scheduler needs fresh indptr. indices/last_page_len
            # buffers are already pinned; plan() copies into the pre-allocated
            # buffers (self-copy, no-op).
            cache.wrapper.plan(
                cache.indptr_buf,
                cache.indices_buf[:cache.num_active_pages],
                cache.last_page_len_buf,
                cache.num_qo_heads, cache.num_kv_heads, cache.head_dim,
                cache.page_size,
                q_data_type=cache.dtype, kv_data_type=cache.dtype,
            )
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    page_idx = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    k_flat = key_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
    v_flat = value_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
    cache.buf[self.layer_idx, page_idx, 0, slot].copy_(k_flat)
    cache.buf[self.layer_idx, page_idx, 1, slot].copy_(v_flat)

    if _pd._enabled:
        ev[2].record(_stream)
    _nvtx_advance("8_flashinfer_run")

    # Step 8 (label kept from SDPA baseline for comparison-table alignment):
    # FlashInfer paged decode on the full KV.
    q_flat = query_states.reshape(1, cache.num_qo_heads, cache.head_dim)
    attn_output = cache.wrapper.run(q_flat, cache.buf[self.layer_idx])
    attn_output = attn_output.view(1, cache.num_qo_heads, 1, cache.head_dim)

    if _pd._enabled:
        ev[3].record(_stream)
    _nvtx_advance("9_o_proj")

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if _pd._enabled:
        ev[4].record(_stream)
        step_names = [
            "1_qkv_proj",
            "2_rope_and_cache_append",
            "8_flashinfer_run",
            "9_o_proj",
        ]
        for i, name in enumerate(step_names):
            _pd._pending_events.append((name, ev[i], ev[i + 1]))
    _nvtx_advance(None)  # close 9_o_proj

    return attn_output, None


# ---------------------------------------------------------------------------
# FI-mode graph-capture alignment
# ---------------------------------------------------------------------------
# Both FI wrappers were built with `use_cuda_graph=True`, so `wrapper.run()` is
# graph-capturable. `wrapper.plan()` is NOT — it touches the scheduler's
# internal Python state and mutates `indptr_buf`. The DCT+FI path calls plan()
# exactly once at build time, so nothing to worry about there. The full-KV
# baseline calls plan() inside `profiled_baseline_flashinfer_forward` whenever
# a new physical page opens (line ~595). To keep plan() out of the captured
# graph, we run extra eager steps until the cache has a full safe page ahead
# (3 priming + 1 capture = 4 advances must stay mid-page).
def _fi_cache_for_mode(mode):
    if mode == "dct_flashinfer":
        return _flashinfer_cache_ref[0]
    if mode == "baseline":
        return _fi_baseline_cache_ref[0]
    return None


# ---------------------------------------------------------------------------
# CUDA graph capture + torch.profiler per-kernel breakdown
# ---------------------------------------------------------------------------
# Mirrors profile_decode._capture_and_benchmark's prime+capture+replay
# sequence, but splits the replay into two passes:
#   1. untraced throughput pass    → clean ms/step, tok/s
#   2. torch.profiler CUDA pass    → per-kernel self-GPU-time aggregated
#                                    across replays (pure kernel time, no
#                                    CPU dispatch lag in the numbers)
#
# Kept local to this driver so profile_decode.py is untouched. The graph
# object is reused for both passes, so capture cost is paid only once.
_STAGE_BUCKETS = [
    # Ordered — first substring match wins. Case-insensitive on kernel name.
    ("7_flashinfer",          ["BatchDecodeWithPagedKV", "merge_state", "MergeAttentionStates"]),
    ("6_topk_and_pack",       ["topk_sort", "pack_indices"]),
    ("5_score_pages",         ["score_pages"]),
    ("4_compress",            ["compress", "dct_proj"]),
    ("2_rope",                ["rotary", "rope"]),
    ("2_cache_append",        ["index_put", "copy_", "CatArrayBatched"]),
    ("sdpa",                  ["flash_fwd", "scaled_dot_product", "fmha", "mha_fwd"]),
    ("gemm",                  ["gemm", "xmma", "matmul", "addmm", "cutlass", "ampere_", "sm_", "gemv"]),
    ("norm",                  ["layer_norm", "rms_norm", "RMSNorm"]),
    ("elementwise",           ["elementwise", "vectorized", "reduce", "fill_", "zero_"]),
]


def _bucket_for_kernel(name):
    lower = name.lower()
    for bucket, substrs in _STAGE_BUCKETS:
        for s in substrs:
            if s.lower() in lower:
                return bucket
    return "other"


def _capture_and_benchmark_with_trace(
    model, past_key_values, next_token, current_pos, num_replays,
    trace_path=None, num_profiled_replays=32, nsys_range=False,
):
    """Capture one decode step into a CUDA graph, replay N times untraced
    (for tok/s), then replay M times under `torch.profiler` (for per-kernel
    self-CUDA-time). Returns (per_replay_ms, tok_s, per_kernel_list) where
    `per_kernel_list` is [(name, us_per_replay), ...] sorted desc by time.

    `num_profiled_replays` is capped at `num_replays` internally. 32 replays
    is enough to get stable kernel-level numbers at ~3–15 µs per kernel.

    When `nsys_range=True`, brackets the throughput replay pass with
    `torch.cuda.profiler.start/stop` so Nsight Systems captures the clean
    graph-replay window (skipping priming + capture + torch.profiler pass).
    The inner per-stage NVTX ranges (pushed from the forwards during capture)
    are replayed on every `g.replay()`, so `nsys stats --report
    nvtx_gpu_proj_sum` gives per-stage GPU time averaged across replays.
    """
    _pd._enabled = False
    device = next_token.device
    static_input = next_token.clone()
    static_pos = torch.tensor([current_pos], device=device, dtype=torch.long)

    # Prime on side stream (allocator + JIT warmup for the captured sequence).
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
    _nvtx_push("graph_capture")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            model(static_input, past_key_values=past_key_values,
                  use_cache=True, cache_position=static_pos)
    _nvtx_pop()

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize(device)

    # Pass 1: untraced replay for clean throughput.
    print(f"  Replaying graph ({num_replays} steps) for throughput...")
    if nsys_range:
        torch.cuda.synchronize(device)
        torch.cuda.profiler.start()
    _nvtx_push("graph_replay_throughput")
    t0 = time.perf_counter()
    for _ in range(num_replays):
        g.replay()
    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000
    per_replay_ms = total_ms / num_replays
    tok_s = 1000.0 / per_replay_ms
    _nvtx_pop()
    if nsys_range:
        torch.cuda.synchronize(device)
        torch.cuda.profiler.stop()

    # Pass 2: profiled replay for per-kernel breakdown. CUDA-only activities
    # keep the overhead minimal and avoid cluttering the trace with CPU ops.
    # Run OUTSIDE the nsys capture window — nsys already has the data from
    # Pass 1 and torch.profiler's overhead would distort the Pass 1 numbers.
    prof_replays = min(num_replays, num_profiled_replays)
    print(f"  Profiling graph replay ({prof_replays} steps)...")
    _nvtx_push("graph_replay_torch_profiler")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(prof_replays):
            g.replay()
        torch.cuda.synchronize(device)
    _nvtx_pop()

    if trace_path is not None:
        if trace_path.endswith(".json"):
            graph_trace = trace_path[:-5] + "_graph.json"
        else:
            graph_trace = trace_path + "_graph.json"
        os.makedirs(os.path.dirname(graph_trace) or ".", exist_ok=True)
        prof.export_chrome_trace(graph_trace)
        print(f"  Graph-mode Chrome trace → {graph_trace}")

    per_kernel = []
    for evt in prof.key_averages():
        t_us = evt.self_device_time_total
        if t_us > 0:
            per_kernel.append((evt.key, t_us / prof_replays))
    per_kernel.sort(key=lambda x: -x[1])

    return per_replay_ms, tok_s, per_kernel


def _print_graph_kernel_table(per_kernel, top_n=25):
    if not per_kernel:
        return
    total_us = sum(t for _, t in per_kernel)

    buckets = {}
    for name, us in per_kernel:
        b = _bucket_for_kernel(name)
        buckets[b] = buckets.get(b, 0.0) + us

    print(f"\n  {'Stage bucket':<24} {'us/step':>10} {'% of kern':>10}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 10}")
    for b in sorted(buckets, key=lambda k: -buckets[k]):
        us = buckets[b]
        pct = us / total_us * 100 if total_us > 0 else 0
        print(f"  {b:<24} {us:>10.2f} {pct:>9.1f}%")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 10}")
    print(f"  {'TOTAL (kernel GPU)':<24} {total_us:>10.2f} {100.0:>9.1f}%")

    print(f"\n  Top {min(top_n, len(per_kernel))} kernels (graph replay, self-CUDA time):")
    print(f"  {'Kernel':<70} {'us/step':>10} {'% kern':>8}")
    print(f"  {'-' * 70} {'-' * 10} {'-' * 8}")
    for name, us in per_kernel[:top_n]:
        pct = us / total_us * 100 if total_us > 0 else 0
        nm = name if len(name) <= 70 else name[:67] + "..."
        print(f"  {nm:<70} {us:>10.2f} {pct:>7.1f}%")
    if len(per_kernel) > top_n:
        rest_us = sum(t for _, t in per_kernel[top_n:])
        rest_pct = rest_us / total_us * 100 if total_us > 0 else 0
        print(f"  {'... (' + str(len(per_kernel) - top_n) + ' more)':<70} "
              f"{rest_us:>10.2f} {rest_pct:>7.1f}%")


# ---------------------------------------------------------------------------
# FlashInfer cache lifecycle
# ---------------------------------------------------------------------------
def _build_fi_cache(model, past_key_values, prefill_len, args):
    """Build + publish the FI cache after prefill. Lives on
    `_flashinfer_cache_ref[0]` so the forward can reach it per decode step.
    Matches the recipe from `test_forward_flashinfer.py`.
    """
    cfg_model = model.config
    num_kv_heads = cfg_model.num_key_value_heads
    num_qo_heads = cfg_model.num_attention_heads
    head_dim = cfg_model.hidden_size // num_qo_heads
    num_layers = cfg_model.num_hidden_layers
    num_sink_pages = (args.sink_size + args.page_size - 1) // args.page_size
    # +1 absorbs the currently-open page (Stage 6 contract — FI's last entry in
    # indices IS the open page). Different from Quest's +1 (which was an
    # oscillation buffer).
    num_recent_pages_fixed = (
        (args.recent_size + args.page_size - 1) // args.page_size + 1
    )
    max_decode_steps = args.warmup_steps + args.num_decode_steps + 16
    if args.cudagraph:
        max_decode_steps += 64
    page_budget = num_sink_pages + args.top_k + num_recent_pages_fixed
    print(
        f"  Building FlashInfer cache: layers={num_layers}, "
        f"num_sink_pages={num_sink_pages}, top_k={args.top_k}, "
        f"num_recent_pages_fixed={num_recent_pages_fixed}, "
        f"page_budget={page_budget}..."
    )
    device = next(model.parameters()).device
    cache = build_flashinfer_paged_cache(
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
    _flashinfer_cache_ref[0] = cache
    print(
        f"  FI cache ready: capacity_pages={cache.capacity_pages}, "
        f"cur_seqlen={cache.cur_seqlen}, "
        f"last_page_idx={cache.last_page_idx_py}, "
        f"last_page_len={cache.last_page_len_py}"
    )
    return cache


def _reset_mode_state():
    """Clear the module-level backend refs. Called between modes so neither
    the Quest cache nor the FI cache leaks into the next run (where it may
    still be read and point at freed memory)."""
    _flashinfer_cache_ref[0] = None
    _fi_baseline_cache_ref[0] = None
    _pd._quest_cache_ref[0] = None
    _pd._step_timings.clear()
    _pd._cpu_timings.clear()
    _pd._pending_events.clear()
    _pd._enabled = False
    _pd._current_layer = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Profile DCT + FlashInfer decode path (eager mode)"
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--context_length", type=int, default=32768)
    p.add_argument("--num_decode_steps", type=int, default=128,
                   help="Decode steps to profile (after warmup)")
    p.add_argument("--warmup_steps", type=int, default=8,
                   help="Warmup decode steps (not profiled)")
    p.add_argument(
        "--mode",
        choices=["baseline", "dct_sdpa", "dct_quest", "dct_flashinfer", "all"],
        default="dct_flashinfer",
    )

    # DCT config (same defaults as profile_decode.py for comparability).
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--sink_size", type=int, default=32,
                   help="Sink tokens (matches page_size by default so sink "
                        "aligns cleanly with page 0).")
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--scoring_method", default="max", choices=["mean", "max"])
    p.add_argument("--group_agg_method", default="max", choices=["mean", "max"])
    p.add_argument("--unselected_mode", default="drop", choices=["drop"])
    p.add_argument("--compressed_token_rope", default="mixed",
                   choices=["mixed", "block_center"])
    p.add_argument("--comp_kv_quant", default="fp8_e5m2",
                   choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    p.add_argument("--comp_kv_quant_granularity", default="per_page",
                   choices=["per_page", "per_comp_token"])
    p.add_argument("--no_triton", action="store_true")
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Chunked prefill size (0 = single-pass).")
    p.add_argument("--sync", action="store_true",
                   help="torch.cuda.synchronize() between steps for CPU "
                        "breakdown (high overhead, biased numbers).")
    p.add_argument("--topk_impl", choices=["auto", "fused", "twostage", "torch"],
                   default="auto")
    p.add_argument("--cudagraph", action="store_true",
                   help="After the per-step profile, also capture one decode "
                        "step into a CUDA graph and benchmark replay "
                        "throughput. For FI-based modes (baseline / "
                        "dct_flashinfer), the FI cache page counter is aligned "
                        "before capture so wrapper.plan() is never called "
                        "inside the graph. Replay overwrites the same KV slot "
                        "each iteration — pure perf measurement.")
    p.add_argument("--cudagraph_replays", type=int, default=0,
                   help="Number of graph replays to time (0 = use "
                        "--num_decode_steps).")

    # Stage 8-specific flags.
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size. Only 1 is supported by the FlashInfer "
                        "backend in Phase 2b; the flag is wired for parity "
                        "with future work.")
    p.add_argument("--verify_flashinfer", action="store_true",
                   help="Toggle per-layer FI-vs-SDPA shadow verification in "
                        "the dct_flashinfer profiled forward. Threshold 0.02.")
    p.add_argument("--torch_profiler_trace", default=None,
                   help="Path to export a Chrome trace via torch.profiler. "
                        "Wraps warmup + profiled decode with a schedule that "
                        "skips warmup; only the profiled window is recorded.")
    p.add_argument("--verify_threshold", type=float, default=0.02)

    # Nsight Systems / NVTX range emission. Enables per-stage GPU-time
    # accounting under CUDA graph (the kernel-bucket table cannot distinguish
    # qkv_proj from o_proj, or 3_segment from other ops — NVTX markers can).
    p.add_argument("--nvtx", action="store_true",
                   help="Emit NVTX range markers around each forward stage "
                        "(1_qkv_proj, 2_rope_and_cache_append, 3_segment, "
                        "4_compress, 5_score_pages_kernel, 6_topk_and_pack, "
                        "7_flashinfer_run, 8_o_proj/9_o_proj). NVTX markers are "
                        "captured by torch.cuda.graph(), so per-stage GPU time "
                        "is visible under --cudagraph when run under Nsight "
                        "Systems. Cost without nsys attached: ~100 ns/push-pop.")
    p.add_argument("--nsys_range", action="store_true",
                   help="Call torch.cuda.profiler.start() / stop() to bracket "
                        "the measurement region (profiled eager decode + graph "
                        "replay). Use with "
                        "`nsys profile --capture-range=cudaProfilerApi "
                        "--capture-range-end=stop` so the nsys trace skips "
                        "model load, prefill and warmup. Implies --nvtx.")

    args = p.parse_args()
    if args.nsys_range and not args.nvtx:
        # NVTX ranges are what makes --nsys_range useful; auto-enable so users
        # don't have to pass both flags.
        args.nvtx = True

    if args.batch_size != 1 and args.mode in ("dct_flashinfer", "all"):
        print(
            f"WARN: --batch_size={args.batch_size} > 1 is not supported by "
            f"the FlashInfer backend in Phase 2b; the dct_flashinfer path "
            f"will fail. See flashinfer_backend module docstring."
        )

    return args


# ---------------------------------------------------------------------------
# Patching helpers (per mode)
# ---------------------------------------------------------------------------
def _rebind_instance_forward(model, attn_cls, forward_fn):
    """Rebind `_old_forward` on every attention module to `forward_fn`. Needed
    under transformers 5.x + accelerate: the class-level patch alone doesn't
    hit already-wrapped forwards."""
    for module in model.modules():
        if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
            module._old_forward = types.MethodType(forward_fn, module)


def _patch_baseline(model, args, original_forward):
    """Patch the attention forward to the FlashInfer full-KV baseline. The
    SDPA baseline is still reachable via the sibling `profile_decode.py`;
    this file wires baseline = FI for fair comparison against dct_flashinfer."""
    restore_forward(args.model, original_forward, model)
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    attn_cls.forward = profiled_baseline_flashinfer_forward
    _rebind_instance_forward(model, attn_cls, profiled_baseline_flashinfer_forward)


def _patch_dct_sdpa(model, args, original_forward):
    restore_forward(args.model, original_forward, model)
    replace_llama_attn(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        compressed_token_rope=args.compressed_token_rope,
        use_triton=not args.no_triton,
        weight_compressed_by_population=True,
        comp_kv_quant=args.comp_kv_quant,
        comp_kv_quant_granularity=args.comp_kv_quant_granularity,
        attention_backend="sdpa",
    )
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    attn_cls.forward = profiled_dct_page_attention_forward
    _rebind_instance_forward(model, attn_cls, profiled_dct_page_attention_forward)


def _patch_dct_quest(model, args, original_forward):
    _patch_dct_sdpa(model, args, original_forward)


def _patch_dct_flashinfer(model, args, original_forward):
    restore_forward(args.model, original_forward, model)
    replace_llama_attn(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        compressed_token_rope=args.compressed_token_rope,
        use_triton=not args.no_triton,
        weight_compressed_by_population=True,
        comp_kv_quant=args.comp_kv_quant,
        comp_kv_quant_granularity=args.comp_kv_quant_granularity,
        attention_backend="flashinfer",
    )
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    attn_cls.forward = profiled_dct_flashinfer_forward
    _rebind_instance_forward(model, attn_cls, profiled_dct_flashinfer_forward)


# ---------------------------------------------------------------------------
# Run-one-mode driver
# ---------------------------------------------------------------------------
def _run_one_mode(model, tokenizer, args, mode, original_forward):
    """Prefill + (optional FI-cache build) + warmup + profiled decode, all for
    one mode. Returns (avg_total_ms, tok_s, timings_dict, graph_stats=None)
    for feeding into the comparison table. `graph_stats` is always None here
    (Phase 2b is eager-only).
    """
    _reset_mode_state()
    _pd._sync_mode = args.sync
    _pd._profile_topk_impl.value = args.topk_impl

    if mode == "baseline":
        _pd._profile_attn_backend.value = "sdpa"
        _pd._profile_attn_backend.verify = False
        _patch_baseline(model, args, original_forward)
    elif mode == "dct_sdpa":
        _pd._profile_attn_backend.value = "sdpa"
        _pd._profile_attn_backend.verify = False
        _patch_dct_sdpa(model, args, original_forward)
    elif mode == "dct_quest":
        _pd._profile_attn_backend.value = "quest"
        _pd._profile_attn_backend.verify = False
        _patch_dct_quest(model, args, original_forward)
    elif mode == "dct_flashinfer":
        # Quest state cleared by _reset_mode_state above.
        _pd._profile_attn_backend.value = "sdpa"  # unused by the FI forward
        _pd._profile_attn_backend.verify = False
        _patch_dct_flashinfer(model, args, original_forward)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size
    bsz = args.batch_size

    torch.manual_seed(0)
    input_ids = torch.randint(
        0, vocab_size, (bsz, args.context_length), dtype=torch.long, device=device,
    )

    chunk_size = args.chunk_size
    if chunk_size > 0:
        print(f"  Prefilling ({args.context_length} tokens, chunk_size={chunk_size}, bsz={bsz})...")
    else:
        print(f"  Prefilling ({args.context_length} tokens, bsz={bsz})...")
    _pd._enabled = False
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = chunked_prefill(model, input_ids, chunk_size)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prefill done: {prefill_ms:.0f}ms")

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)
    prefill_len = args.context_length

    # `+16` absorbs the `+16` slack in FI cache capacity_pages. With
    # --cudagraph we also need headroom for alignment-to-safe-page (up to
    # page_size-1 extra eager steps) plus 3 priming + 1 capture. 64 covers
    # both comfortably for any page_size up to 64.
    extra = args.warmup_steps + args.num_decode_steps + 16
    if args.cudagraph:
        extra += 64
    past_key_values = pre_allocate_cache(past_key_values, extra_tokens=extra)
    print(f"  Converted to pre-allocated cache (+{extra} tokens)")

    # Mode-specific post-prefill setup.
    if mode == "dct_quest":
        import quest_backend
        cfg_model = model.config
        num_kv_heads = cfg_model.num_key_value_heads
        num_qo_heads = cfg_model.num_attention_heads
        head_dim = cfg_model.hidden_size // num_qo_heads
        num_layers = cfg_model.num_hidden_layers
        num_sink_pages = (args.sink_size + args.page_size - 1) // args.page_size
        num_recent_pages_fixed = (
            (args.recent_size + args.page_size - 1) // args.page_size + 1
        )
        max_total_selected = num_sink_pages + args.top_k + num_recent_pages_fixed
        max_decode_steps = extra
        print(
            f"  Building Quest paged cache (layers={num_layers}, "
            f"page_size={args.page_size}, num_sink_pages={num_sink_pages}, "
            f"top_k={args.top_k}, num_recent_pages_fixed={num_recent_pages_fixed}, "
            f"max_total_selected={max_total_selected})..."
        )
        _pd._quest_cache_ref[0] = quest_backend.build_quest_paged_cache(
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
        print(
            f"  Quest cache ready: {_pd._quest_cache_ref[0].capacity_pages} pages, "
            f"cur_seqlen={_pd._quest_cache_ref[0].cur_seqlen}"
        )
    elif mode == "dct_flashinfer":
        _build_fi_cache(model, past_key_values, prefill_len, args)
    elif mode == "baseline":
        cfg_model = model.config
        num_kv_heads = cfg_model.num_key_value_heads
        num_qo_heads = cfg_model.num_attention_heads
        head_dim = cfg_model.hidden_size // num_qo_heads
        num_layers = cfg_model.num_hidden_layers
        max_decode_steps = extra
        print(
            f"  Building full-KV FlashInfer baseline cache (layers={num_layers}, "
            f"page_size={args.page_size})..."
        )
        _fi_baseline_cache_ref[0] = build_fi_baseline_cache(
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
        )
        _bc = _fi_baseline_cache_ref[0]
        print(
            f"  FI baseline cache ready: capacity_pages={_bc.capacity_pages}, "
            f"num_active_pages={_bc.num_active_pages}, "
            f"last_page_idx={_bc.last_page_idx_py}, "
            f"last_page_len={_bc.last_page_len_py}"
        )

    # Verify flag — only meaningful for dct_flashinfer, but clear on all
    # modules so the state doesn't bleed across modes.
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    for m in model.modules():
        if isinstance(m, attn_cls):
            m._verify_flashinfer = (
                mode == "dct_flashinfer" and args.verify_flashinfer
            )
            if hasattr(m, "_verify_diffs"):
                del m._verify_diffs

    # Verify cost should not inflate warmup — flip it off for the warmup
    # window and restore it before profiled steps.
    saved_verify = {}
    if args.verify_flashinfer and mode == "dct_flashinfer":
        for m in model.modules():
            if isinstance(m, attn_cls):
                saved_verify[id(m)] = m._verify_flashinfer
                m._verify_flashinfer = False

    total_times = []

    def _do_one_decode_step(step_idx, profiled):
        """Run a single decode step. When `profiled`, time it and let the
        chained CUDA events from the forward flow into the step timings
        dict."""
        nonlocal next_token, past_key_values
        if profiled:
            _pd._current_layer = 0
        cache_position = torch.tensor([prefill_len + step_idx], device=device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter() if profiled else None
        with torch.no_grad():
            out = model(
                next_token, past_key_values=past_key_values,
                use_cache=True, cache_position=cache_position,
            )
        if profiled:
            _pd._flush_events()
            total_times.append((time.perf_counter() - t0) * 1000)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)

    def _run_warmup_and_profiled(prof=None):
        """Warmup (no events, no verify) then profiled decode. When `prof` is
        set, call prof.step() after each iteration — warmup falls inside the
        schedule's `warmup` window and is not recorded."""
        # Warmup
        _pd._enabled = False
        _nvtx_push("warmup")
        for step in range(args.warmup_steps):
            _do_one_decode_step(step, profiled=False)
            if prof is not None:
                prof.step()
        torch.cuda.synchronize(device)
        _nvtx_pop()

        # Restore verify for the profiled window.
        if args.verify_flashinfer and mode == "dct_flashinfer":
            for m in model.modules():
                if isinstance(m, attn_cls):
                    m._verify_flashinfer = saved_verify.get(id(m), True)
                    m._verify_diffs = []

        # Profiled decode.
        print(f"  Profiling ({args.num_decode_steps} steps)...")
        _pd._step_timings.clear()
        _pd._cpu_timings.clear()
        _pd._pending_events.clear()
        _pd._enabled = True

        # --nsys_range: open the nsys capture window here so the trace skips
        # model load, prefill, cache build and warmup. Closed after the
        # profiled window. When --cudagraph also runs, a second start/stop
        # pair opens around the graph phase in _capture_and_benchmark_with_trace.
        if args.nsys_range:
            torch.cuda.synchronize(device)
            torch.cuda.profiler.start()

        _nvtx_push("profiled_decode_eager")
        for step in range(args.num_decode_steps):
            _do_one_decode_step(args.warmup_steps + step, profiled=True)
            if prof is not None:
                prof.step()
        _nvtx_pop()

        if args.nsys_range:
            torch.cuda.synchronize(device)
            torch.cuda.profiler.stop()

        _pd._enabled = False

    print(f"  Warming up ({args.warmup_steps} steps)...")

    if args.torch_profiler_trace and mode == "dct_flashinfer":
        # Only wrap the target mode — otherwise the trace balloons with
        # unrelated work and the Chrome trace becomes hard to read.
        # Schedule: warmup_steps calls to prof.step() fall inside the
        # profiler's own warmup window (recorded but discarded); the next
        # num_decode_steps calls are the active (recorded) window.
        trace_path = args.torch_profiler_trace
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        print(f"  torch.profiler active → {trace_path}")
        schedule = torch.profiler.schedule(
            wait=0,
            warmup=args.warmup_steps,
            active=args.num_decode_steps,
            repeat=1,
        )

        def _on_trace_ready(p):
            p.export_chrome_trace(trace_path)
            print(f"  torch.profiler trace written to {trace_path}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            schedule=schedule,
            on_trace_ready=_on_trace_ready,
        ) as prof:
            _run_warmup_and_profiled(prof=prof)
    else:
        _run_warmup_and_profiled(prof=None)

    torch.cuda.synchronize(device)

    avg_total = sum(total_times) / len(total_times)
    tok_s = 1000.0 / avg_total

    # Verify summary (FI-only). Mirrors the Stage 7 test harness.
    verify_ok = None
    if args.verify_flashinfer and mode == "dct_flashinfer":
        per_layer_diffs = {}
        for m in model.modules():
            if isinstance(m, attn_cls) and getattr(m, "_verify_diffs", None):
                lid = getattr(m, "layer_idx", None)
                per_layer_diffs[lid] = list(m._verify_diffs)
        if per_layer_diffs:
            all_steps = max(len(v) for v in per_layer_diffs.values())
            per_step_worst = [0.0] * all_steps
            per_step_layer = [-1] * all_steps
            for lid, diffs in per_layer_diffs.items():
                for s, d in enumerate(diffs):
                    if d > per_step_worst[s]:
                        per_step_worst[s] = d
                        per_step_layer[s] = lid
            worst = max(max(v) for v in per_layer_diffs.values())
            print(
                f"  [verify_flashinfer] worst max-abs-diff across "
                f"{len(per_layer_diffs)} layers × {all_steps} steps = "
                f"{worst:.3e} (threshold = {args.verify_threshold:.0e})"
            )
            # Print the first few per-step maxima for quick eyeballing.
            head = min(8, all_steps)
            for s in range(head):
                ok = per_step_worst[s] < args.verify_threshold
                print(
                    f"    step {s}: {per_step_worst[s]:.3e}  "
                    f"worst layer={per_step_layer[s]:>2}  {'OK' if ok else 'FAIL'}"
                )
            if all_steps > head:
                print(f"    ... ({all_steps - head} more steps)")
            verify_ok = worst < args.verify_threshold
            print(
                f"  [verify_flashinfer] overall: "
                f"{'PASS' if verify_ok else 'FAIL'}"
            )

    # Optional CUDA graph capture + replay benchmark.
    graph_stats = None
    graph_kernels = None
    if args.cudagraph:
        # For FI-based modes, align the cache page counter so wrapper.plan()
        # does not fire during priming or capture. Priming advances counter
        # by 3, capture advances by 1 more — so we need last_page_len_py + 4
        # <= page_size. If too close to the boundary, run extra eager steps
        # to wrap past it (resets counter to 1 of a new page).
        fi_cache = _fi_cache_for_mode(mode)
        align_steps = 0
        if fi_cache is not None:
            ps = fi_cache.page_size
            while fi_cache.last_page_len_py > ps - 4:
                _do_one_decode_step(
                    args.warmup_steps + args.num_decode_steps + align_steps,
                    profiled=False,
                )
                align_steps += 1
                if align_steps > ps + 1:  # safety: never loop forever
                    break
            if align_steps:
                print(
                    f"  Graph alignment: {align_steps} eager step(s) "
                    f"(last_page_len={fi_cache.last_page_len_py}/{ps})"
                )

        # Disable verify before capture — the per-layer gather + Python-side
        # diff accumulation is not graph-safe.
        if args.verify_flashinfer and mode == "dct_flashinfer":
            for m in model.modules():
                if isinstance(m, attn_cls):
                    m._verify_flashinfer = False

        torch.cuda.synchronize(device)
        num_replays = args.cudagraph_replays or args.num_decode_steps
        current_pos = (
            prefill_len + args.warmup_steps + args.num_decode_steps + align_steps
        )
        try:
            per_replay_ms, graph_tok_s, graph_kernels = (
                _capture_and_benchmark_with_trace(
                    model, past_key_values, next_token, current_pos, num_replays,
                    trace_path=args.torch_profiler_trace,
                    nsys_range=args.nsys_range,
                )
            )
            graph_stats = (per_replay_ms, graph_tok_s)
        except Exception as e:
            print(f"  CUDA graph benchmark failed: {type(e).__name__}: {e}")
            graph_stats = None
            graph_kernels = None

    return (
        avg_total, tok_s,
        dict(_pd._step_timings), dict(_pd._cpu_timings),
        verify_ok, graph_stats, graph_kernels,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    _set_nvtx_enabled(args.nvtx)
    if args.nvtx:
        if args.nsys_range:
            print(
                "  [nvtx] range markers ON; profiler.start/stop brackets the "
                "profiled eager decode window AND the CUDA-graph throughput "
                "replay pass. Run under:"
            )
            print(
                "    nsys profile --trace=cuda,nvtx "
                "--capture-range=cudaProfilerApi --capture-range-end=stop \\"
            )
            print(
                "        -o <output_prefix> python speed/profile_decode_flash_infer.py \\"
            )
            print("        " + " ".join(sys.argv[1:]))
            print(
                "  [nvtx] Per-stage GPU time: "
                "`nsys stats --report nvtx_gpu_proj_sum <prefix>.nsys-rep`"
            )
        else:
            print(
                "  [nvtx] range markers ON. --nsys_range is OFF, so the nsys "
                "trace will include model load, prefill and warmup as well as "
                "the measurement region."
            )

    original_forward = get_original_forward(args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)
    model._original_attn_forward = original_forward

    num_layers = model.config.num_hidden_layers
    print(f"Model layers: {num_layers}")
    print(f"Context length: {args.context_length}  batch_size: {args.batch_size}")

    modes_order = ["baseline", "dct_sdpa", "dct_quest", "dct_flashinfer"]
    if args.mode == "all":
        modes_to_run = modes_order
    else:
        modes_to_run = [args.mode]

    results = {}
    verify_state = {}
    for mode in modes_to_run:
        print(f"\n{'=' * 70}")
        print(f"MODE: {mode}")
        print(f"{'=' * 70}")
        (avg_total, tok_s, timings, cpu_timings,
         verify_ok, graph_stats, graph_kernels) = _run_one_mode(
            model, tokenizer, args, mode, original_forward,
        )
        print_profile(mode, avg_total, tok_s, timings, num_layers, cpu_timings)
        if graph_stats is not None:
            gp, gts = graph_stats
            print(f"\n  CUDA graph: {gp:.3f} ms/step  ({gts:.1f} tok/s)")
            print(
                f"  Speedup (graph vs profiled): {avg_total / gp:.2f}x  "
                f"(saved {avg_total - gp:.2f} ms/step)"
            )
            if graph_kernels:
                _print_graph_kernel_table(graph_kernels)
        results[mode] = (avg_total, tok_s, timings, graph_stats)
        if verify_ok is not None:
            verify_state[mode] = verify_ok

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Comparison table (only when >= 2 modes ran).
    if len(results) >= 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        header = f"  {'Mode':<20} {'ms/tok':>10} {'tok/s':>10} {'vs baseline':>14}"
        print(header)
        print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 14}")
        base = results.get("baseline")
        for mode in modes_order:
            if mode not in results:
                continue
            avg, tok, _, _graph = results[mode]
            if base is not None and mode != "baseline":
                delta_pct = (tok - base[1]) / base[1] * 100
                vs_str = f"{delta_pct:+.1f}%"
            elif mode == "baseline":
                vs_str = "(ref)"
            else:
                vs_str = "—"
            print(f"  {mode:<20} {avg:>10.2f} {tok:>10.1f} {vs_str:>14}")

        # Graph-mode comparison table, if any mode ran with --cudagraph.
        any_graph = any(r[3] is not None for r in results.values())
        if any_graph:
            print(f"\n  {'Mode (graph)':<20} {'ms/tok':>10} {'tok/s':>10} {'vs baseline':>14}")
            print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 14}")
            base_graph = results.get("baseline", (None,) * 4)[3]
            for mode in modes_order:
                if mode not in results or results[mode][3] is None:
                    continue
                gp, gts = results[mode][3]
                if base_graph is not None and mode != "baseline":
                    delta_pct = (gts - base_graph[1]) / base_graph[1] * 100
                    vs_str = f"{delta_pct:+.1f}%"
                elif mode == "baseline":
                    vs_str = "(ref)"
                else:
                    vs_str = "—"
                print(f"  {mode:<20} {gp:>10.2f} {gts:>10.1f} {vs_str:>14}")

        # Per-step stage comparison table — only when dct_flashinfer ran.
        if "dct_flashinfer" in results:
            fi_timings = results["dct_flashinfer"][2]
            all_steps = sorted(fi_timings.keys())
            print(
                f"\n  {'Step':<28} {'DCT+FI (ms/tok)':>18}"
            )
            print(f"  {'-' * 28} {'-' * 18}")

            def _per_token(timings_dict, step):
                vals = timings_dict.get(step, [])
                if not vals:
                    return 0.0
                return sum(vals) / (len(vals) / num_layers)

            total = 0.0
            for step in all_steps:
                pt = _per_token(fi_timings, step)
                total += pt
                print(f"  {step:<28} {pt:>18.3f}")
            print(f"  {'-' * 28} {'-' * 18}")
            print(f"  {'TOTAL':<28} {total:>18.3f}")

    if verify_state:
        print()
        for mode, ok in verify_state.items():
            tag = "PASS" if ok else "FAIL"
            print(f"  [verify_flashinfer] {mode}: {tag}")


if __name__ == "__main__":
    main()
