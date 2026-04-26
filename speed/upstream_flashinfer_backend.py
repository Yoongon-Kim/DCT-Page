"""
Upstream-FlashInfer paged decode adapter (no custom-fork patch).

Sibling of `speed/flashinfer_backend.py`. The other adapter depends on the
per-head `indices` patch (`page_budget` kwarg in plan/run) living in the
DCT-Page fork at `/home/yoongonkim/flashinfer-dct`. This adapter takes the
"virtual batch = KV head" approach so the 2-D indices API in stock
FlashInfer is enough:

  - Physical KV pool is partitioned into `num_kv_heads` contiguous chunks
    of `pages_per_head` pages each. Each physical page stores data for one
    KV head only (shape `(page_size, 1, head_dim)`).
  - Q is reshaped `(1, num_qo_heads, d) -> (num_kv_heads, group_size, d)`.
    Virtual batch `h` carries the `group_size` Q heads that belong to KV
    head `h` in the GQA grouping.
  - `wrapper.plan()` is called with `num_qo_heads=group_size,
    num_kv_heads=1` — each virtual batch attends its own 1 KV head over
    its own slice of pages.

The attention math is identical to the per-head fork path: each Q head's
softmax is independent, and virtual-batching is just a parallelism axis.

Layout at decode time (per layer, flat view):
  buf_flat: (num_kv_heads * pages_per_head, 2, page_size, 1, head_dim)
    physical page (h * pages_per_head + p) holds head h's tokens for
    logical page p.
  buf_7d:   view as (num_kv_heads, pages_per_head, 2, page_size, 1, head_dim)
    for convenient append writes.

Indices buffers:
  indices_buf_3d: (1, num_kv_heads, page_budget) int32 — HEAD-LOCAL indices,
    exactly what `topk_sort_and_pack_triton` writes today (shared with the
    fork path). Sink slice prefilled once.
  indices_flat_buf: (num_kv_heads * page_budget,) int32 — PHYSICAL indices
    that FlashInfer's wrapper reads. Refreshed per decode step via a
    cheap `torch.add` that applies the per-head offset
    (h * pages_per_head).

Sink alignment: same contract as the fork backend — `sink_size >= page_size`
with `num_sink_pages = ceil(sink_size / page_size)` cleanly partitioning
page 0 within each head's pool. `sink_size=32` (page_size) keeps things
clean for the Phase 2b benchmarks.

plan() lifecycle: called ONCE at build time. Stock scheduler uses only
indptr + last_page_len + shape metadata — none of which change during
decode. `last_page_len_buf` mutates in-place; `indices_flat_buf` is
refreshed in-place each step. Both are pinned via `use_cuda_graph=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


@dataclass
class UpstreamFlashInferPagedKVCache:
    """Upstream-FlashInfer paged KV cache (virtual-batch = KV head).

    Storage:
      `buf` is the flat (FlashInfer-facing) 6-D view;
      `buf_7d` is the same underlying storage viewed 7-D so append writes
      can address (layer, head, page, 0/1, slot, 0, :) directly.

    Python-side counters (`cur_seqlen`, `last_page_idx_py`, `last_page_len_py`)
    mirror the GPU tensors (`last_page_idx`, `last_page_len_buf`). The
    Python-side values drive append-path indexing; the GPU tensors feed
    the Stage 5 fused kernel and FlashInfer's run().
    """

    buf: torch.Tensor                 # flat 6-D: (L, H*P, 2, ps, 1, d)
    buf_7d: torch.Tensor              # view: (L, H, P, 2, ps, 1, d)
    wrapper: BatchDecodeWithPagedKVCacheWrapper

    page_size: int
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int                 # total Q heads = num_kv_heads * group_size
    group_size: int                   # num_qo_heads // num_kv_heads
    num_layers: int

    # Per-head capacity — each head has its own pool of this many physical pages.
    pages_per_head: int
    total_pages: int                  # num_kv_heads * pages_per_head

    dtype: torch.dtype
    device: torch.device

    num_sink_pages: int
    top_k: int
    num_recent_pages_fixed: int
    page_budget: int                  # num_sink_pages + top_k + num_recent_pages_fixed

    # Pre-allocated wrapper-owned buffers.
    float_workspace_buffer: torch.Tensor
    indices_buf_3d: torch.Tensor      # (1, num_kv_heads, page_budget) int32, head-local
    indices_flat_buf: torch.Tensor    # (num_kv_heads * page_budget,) int32, physical
    indptr_buf: torch.Tensor          # (num_kv_heads + 1,) int32
    last_page_len_buf: torch.Tensor   # (num_kv_heads,) int32
    recent_offsets: torch.Tensor      # (num_recent_pages_fixed,) int32
    last_page_idx: torch.Tensor       # (1,) int32, head-local scalar

    head_offset: torch.Tensor         # (num_kv_heads, 1) int32, values h*pages_per_head

    # Python-side state.
    cur_seqlen: int = 0
    last_page_idx_py: int = 0
    last_page_len_py: int = 0


def _pack_preallocated_to_paged_upstream(
    buf_7d: torch.Tensor,
    preallocated_layers,
    prefill_len: int,
    page_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Pack DCT's (1, nkv, T, d) prefill cache into the virtual-batch-per-head
    7-D buffer `(L, H, P, 2, ps, 1, d)`. Each head's rows land in its own
    pool slice. Returns `prefill_pages` (physical pages per head used for
    the prefill, including the partial last page).
    """
    prefill_pages = (prefill_len + page_size - 1) // page_size
    pad = prefill_pages * page_size - prefill_len
    for l, layer in enumerate(preallocated_layers):
        k = layer.keys[0, :, :prefill_len, :]    # (H, T, d)
        v = layer.values[0, :, :prefill_len, :]
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        # (H, P*ps, d) -> (H, P, ps, d). Destination expects (H, P, ps, 1, d).
        k = k.view(num_kv_heads, prefill_pages, page_size, head_dim).to(dtype)
        v = v.view(num_kv_heads, prefill_pages, page_size, head_dim).to(dtype)
        # buf_7d[l]: (H, P_full, 2, ps, 1, d)
        buf_7d[l, :, :prefill_pages, 0, :, 0, :].copy_(k)
        buf_7d[l, :, :prefill_pages, 1, :, 0, :].copy_(v)
    return prefill_pages


def build_upstream_flashinfer_paged_cache(
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
    num_sink_pages: int,
    top_k: int,
    num_recent_pages_fixed: int,
    workspace_bytes: int = 128 * 1024 * 1024,
) -> UpstreamFlashInferPagedKVCache:
    """Build a virtual-batch-per-head cache populated from DCT's prefilled
    `preallocated_layers`. `num_recent_pages_fixed` INCLUDES the open page
    (same contract as `flashinfer_backend.build_flashinfer_paged_cache`).
    """
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads "
            f"({num_kv_heads}) for GQA virtual-batch layout."
        )
    group_size = num_qo_heads // num_kv_heads
    if num_sink_pages < 1:
        raise ValueError("num_sink_pages must be >= 1.")
    if num_recent_pages_fixed < 1:
        raise ValueError("num_recent_pages_fixed must be >= 1 (includes open page).")

    page_budget = num_sink_pages + top_k + num_recent_pages_fixed

    prefill_pages = (prefill_len + page_size - 1) // page_size
    decode_pages = (max_decode_steps + page_size - 1) // page_size
    pages_per_head = prefill_pages + decode_pages + 4  # slack matches fork backend
    total_pages = num_kv_heads * pages_per_head

    # Allocate 7-D so append writes are a single indexed copy per layer.
    # The flat view fed to FlashInfer shares storage.
    buf_7d = torch.zeros(
        num_layers, num_kv_heads, pages_per_head, 2, page_size, 1, head_dim,
        dtype=dtype, device=device,
    )
    buf = buf_7d.view(num_layers, total_pages, 2, page_size, 1, head_dim)

    _pack_preallocated_to_paged_upstream(
        buf_7d, preallocated_layers, prefill_len, page_size,
        num_layers, num_kv_heads, head_dim, dtype,
    )

    # Head-local indices, shape matches `topk_sort_and_pack_triton` contract
    # (same fused kernel drives both backends).
    indices_buf_3d = torch.zeros(
        1, num_kv_heads, page_budget, dtype=torch.int32, device=device,
    )
    # Sink slice is static per head (head-local: [0, num_sink_pages)).
    # Bias to physical pages is applied each step into `indices_flat_buf`.
    sink_ids = torch.arange(num_sink_pages, dtype=torch.int32, device=device)
    indices_buf_3d[:, :, :num_sink_pages] = sink_ids

    # Physical indices buffer (what FI reads via `paged_kv_indices_buffer`).
    indices_flat_buf = torch.zeros(
        num_kv_heads * page_budget, dtype=torch.int32, device=device,
    )
    head_offset = (
        torch.arange(num_kv_heads, dtype=torch.int32, device=device)
        * pages_per_head
    ).view(num_kv_heads, 1)

    # Seed indices_flat_buf with the post-bias sink IDs (topk/recent regions
    # start as zeros in indices_buf_3d and get filled by the Stage 5 kernel
    # each decode step). This seeding is overwritten by the per-step
    # torch.add before every run() anyway — we do it here for plan() to see
    # sensible values during the one-time scheduler setup.
    torch.add(
        indices_buf_3d[0], head_offset,
        out=indices_flat_buf.view(num_kv_heads, page_budget),
    )

    # indptr: one row per virtual batch, each spanning `page_budget` pages.
    indptr_buf = (
        torch.arange(num_kv_heads + 1, dtype=torch.int32, device=device)
        * page_budget
    )

    last_open_page = (prefill_len - 1) // page_size
    last_open_len = prefill_len - last_open_page * page_size
    last_page_len_buf = torch.full(
        (num_kv_heads,), last_open_len, dtype=torch.int32, device=device,
    )
    last_page_idx = torch.full(
        (1,), last_open_page, dtype=torch.int32, device=device,
    )

    recent_offsets = torch.arange(
        -num_recent_pages_fixed + 1, 1, dtype=torch.int32, device=device,
    )

    float_workspace_buffer = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device,
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_flat_buf,
        paged_kv_last_page_len_buffer=last_page_len_buf,
    )
    # NOTE: no `page_budget=` kwarg → stock (upstream-compatible) code path.
    # With vbsz = num_kv_heads, FI handles the per-head selection naturally
    # via indptr + indices without needing the custom per-head patch.
    wrapper.plan(
        indptr_buf,
        indices_flat_buf,
        last_page_len_buf,
        group_size,       # num_qo_heads per virtual batch
        1,                # num_kv_heads per virtual batch
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    return UpstreamFlashInferPagedKVCache(
        buf=buf,
        buf_7d=buf_7d,
        wrapper=wrapper,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        group_size=group_size,
        num_layers=num_layers,
        pages_per_head=pages_per_head,
        total_pages=total_pages,
        dtype=dtype,
        device=device,
        num_sink_pages=num_sink_pages,
        top_k=top_k,
        num_recent_pages_fixed=num_recent_pages_fixed,
        page_budget=page_budget,
        float_workspace_buffer=float_workspace_buffer,
        indices_buf_3d=indices_buf_3d,
        indices_flat_buf=indices_flat_buf,
        indptr_buf=indptr_buf,
        last_page_len_buf=last_page_len_buf,
        recent_offsets=recent_offsets,
        last_page_idx=last_page_idx,
        head_offset=head_offset,
        cur_seqlen=prefill_len,
        last_page_idx_py=last_open_page,
        last_page_len_py=last_open_len,
    )


def append_upstream_flashinfer_cache(
    cache: UpstreamFlashInferPagedKVCache,
    new_k: torch.Tensor,     # (1, num_kv_heads, 1, head_dim) bf16, post-RoPE
    new_v: torch.Tensor,
    layer_idx: int,
) -> None:
    """Append one decode step's K/V into the virtual-batch-per-head cache.
    Only layer 0 advances the shared counters. Last-page-len/idx mirrors
    are broadcast across all virtual batches (they share the same logical
    open-page position).
    """
    if layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1

        cache.last_page_idx.fill_(cache.last_page_idx_py)
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    page_idx = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    if page_idx >= cache.pages_per_head:
        raise RuntimeError(
            f"UpstreamFlashInferPagedKVCache overflow: page_idx={page_idx} >= "
            f"pages_per_head={cache.pages_per_head}. Increase "
            f"max_decode_steps at build time."
        )

    k_flat = new_k.reshape(cache.num_kv_heads, cache.head_dim)
    v_flat = new_v.reshape(cache.num_kv_heads, cache.head_dim)
    # buf_7d: (L, H, P, 2, ps, 1, d). Touch all heads at (page_idx, slot).
    cache.buf_7d[layer_idx, :, page_idx, 0, slot, 0, :].copy_(k_flat)
    cache.buf_7d[layer_idx, :, page_idx, 1, slot, 0, :].copy_(v_flat)


def refresh_upstream_indices_flat(
    cache: UpstreamFlashInferPagedKVCache,
) -> None:
    """Apply per-head page-pool bias to the head-local scratch indices and
    write the result into the FI-facing flat buffer. Must be called AFTER
    `topk_sort_and_pack_triton` populates `indices_buf_3d` for this step
    and BEFORE `wrapper.run()`.

    Operation: `indices_flat[h, :] = indices_buf_3d[0, h, :] + h * pages_per_head`.
    One fused `torch.add` with a pre-allocated output view; no intermediate.
    """
    torch.add(
        cache.indices_buf_3d[0],
        cache.head_offset,
        out=cache.indices_flat_buf.view(cache.num_kv_heads, cache.page_budget),
    )


def upstream_flashinfer_decode_attention(
    query_states: torch.Tensor,   # (1, num_qo_heads, 1, head_dim) bf16
    cache: UpstreamFlashInferPagedKVCache,
    layer_idx: int,
) -> torch.Tensor:
    """Run upstream FlashInfer paged decode with the virtual-batch-per-head
    layout.

    Q reshape: `(1, num_qo_heads, 1, head_dim)` → `(num_kv_heads, group_size,
    head_dim)`. FlashInfer interprets this as `num_kv_heads` batch entries,
    each carrying `group_size` Q heads against its own 1 KV head over its
    own slice of indices.

    Returns `(1, num_qo_heads, 1, head_dim)` to match the SDPA pre-transpose
    convention (same as `flashinfer_decode_attention` in the fork adapter).

    IMPORTANT: `refresh_upstream_indices_flat(cache)` must be called before
    this function, after the Stage 5 kernel has written fresh head-local
    indices.
    """
    nkv = cache.num_kv_heads
    gs = cache.group_size
    d = cache.head_dim

    q_flat = query_states.reshape(nkv, gs, d)
    out = cache.wrapper.run(q_flat, cache.buf[layer_idx])  # (nkv, gs, d)
    return out.view(1, nkv * gs, 1, d)
