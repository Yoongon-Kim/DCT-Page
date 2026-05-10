"""
FlashInfer paged decode adapter for DCT-Page (Phase 2b Stage 6).

Bridges DCT-Page's decode path into upstream FlashInfer's
`BatchDecodeWithPagedKVCacheWrapper` (v0.6.7.post3 fork at
`/home/yoongonkim/flashinfer-dct`) with the per-head indices patch landed in
Stages 2–3. Scope: drop mode, native bf16 end-to-end (no fp16 cast).

Mirrors the shape of `speed/quest_backend.py` (build/append/decode) but on the
upstream plan()/run() API with:

- Native bf16 KV cache (Phase 2 fp16 cast path removed).
- 3-D `indices_buf: (bsz, num_kv_heads, page_budget)` int32 — Triton/DCT-facing
  view. The underlying storage is HEAD-MAJOR
  `(num_kv_heads, bsz, page_budget)` flat so that the fork's per-head
  `get_phys_page(page_iter, head_idx) = indices[h * indptr[bsz] + page_iter]`
  formula (page.cuh post Phase 2b multibatch patch) lands at the right
  (head, batch, p_local) slot. The Triton-facing view is `storage.permute(1,
  0, 2)` — non-contiguous (batch_stride=page_budget, head_stride=bsz*page_budget),
  but the topk wrapper passes both strides explicitly so writes still land
  at `[b, h, :]` correctly. The flat storage view is handed to the wrapper
  via `use_cuda_graph=True` at construction — this pins pre-allocated
  buffers without requiring graph capture at runtime.
- `page_budget` kwarg in plan() (Stage 3 addition) so the per-head branch of
  `get_phys_page(page_iter, head_idx)` fires at run time.
- Per-call pack of sink indices at cache init (static across the session) —
  the fused Stage 5 kernel never rewrites that slice.

Multibatch contract (bsz>=1):
- Each batch b owns physical pages [b*pages_per_batch, (b+1)*pages_per_batch)
  in cache.buf[layer]. The page IDs stored in indices_buf[b, h, :] and in
  last_page_idx[b] are ALWAYS pre-biased by `b * pages_per_batch`, so the
  fork's per-head `get_phys_page` lookup returns batch-correct physical IDs
  unchanged.
- Lockstep prefill: every batch shares the same `cur_seqlen`, `last_page_idx_py`
  (logical, pre-bias) and `last_page_len_py`. The Python scalars are the
  per-batch logical state; the GPU `last_page_idx` tensor stores the bias-
  applied per-batch IDs. Ragged prefill is NOT supported here.

Key layout choice (pages aligned with token 0 within each batch):
  page 0 (logical) = tokens [0, page_size) of that batch
  page p (logical) = tokens [p*page_size, (p+1)*page_size)
  physical page  = b * pages_per_batch + logical_page

"Sink", "topk", "recent" are logical regions within `indices_buf[b, h, :]`:

  indices_buf[b, h, :]:
    [0, num_sink_pages)                        : sink page IDs of batch b
                                                 (= [0..S) + b*pages_per_batch,
                                                 static)
    [num_sink_pages, num_sink_pages + top_k)   : topk middle pages of batch b
                                                 (already batch-biased)
    [num_sink_pages + top_k, page_budget)      : recent pages of batch b
                                                 (last entry = last_page_idx[b],
                                                 the currently-open page)

`num_recent_pages_fixed` EXCLUDES the open page; the open page is implicit
(+1) and recent_offsets span [-R, 0] (length R+1, last entry = 0 selects the
currently-open page). `last_page_len_buf` tracks the current length of the
open page directly — FlashInfer's standard contract (last entry in indices
is the last page; `paged_kv_last_page_len` is its valid length).

Sink contract: page 0 (per-batch logical) is sink iff `num_sink_pages >= 1`.
With `num_sink_pages=1` the entire first physical page (page_size tokens) of
each batch is unconditionally attended. The page-unit contract makes sink
alignment trivial — there is no `sink_size` to compare against `page_size`.

Plan lifecycle: plan() is called ONCE at cache build time. The wrapper's
scheduler uses only indptr + last_page_len + shape metadata (none of which
change across decode steps in drop mode), so replanning per step is
unnecessary. `last_page_len_buf` and `indices_buf` are read live at run()
time via pre-allocated buffers — the Triton fused kernel writes new topk
indices and `append_flashinfer_cache` updates `last_page_len_buf` each step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


@dataclass
class FlashInferPagedKVCache:
    """FlashInfer NHD paged KV cache + wrapper state.

    Layout of `buf`: list[Tensor], one (capacity_pages, 2, page_size,
    num_kv_heads, head_dim) per layer. `buf[l][p, 0] = K`, `buf[l][p, 1] = V`,
    same as Quest's layout. With bsz>=1, `capacity_pages = bsz *
    pages_per_batch`; physical page IDs are pre-biased by `b * pages_per_batch`
    in `indices_buf` and `last_page_idx`. All layers share the page-advance
    counters (`cur_seqlen`, `last_page_idx_py`, `last_page_len_py`); only
    layer 0 advances them. Lockstep across batch — every batch advances
    together, ragged is not supported.

    Fixed-width indices: `page_budget = num_sink_pages + top_k +
    num_recent_pages_fixed + 1` entries per (batch, kv_head) row (the +1 is
    the open page, implicit on top of `num_recent_pages_fixed` full recent
    pages). `plan()` is
    called once at build time with this budget; `run()` reads the latest
    indices/last_page_len out of the pre-allocated buffers.

    CPU/GPU state mirror: `last_page_idx_py` / `last_page_len_py` are Python
    ints used for in-kernel-free Python-side indexing in
    `append_flashinfer_cache`. `last_page_idx_py` is the per-batch LOGICAL
    page index (no batch bias); `last_page_idx` is its (bsz,) int32 GPU
    mirror with batch bias applied (`last_page_idx_py + arange(bsz) *
    pages_per_batch`). The fused Stage 5 kernel reads `last_page_idx[b]` and
    writes already-biased physical IDs into `indices_buf[b, h, :]`. Python
    scalar and GPU tensor must stay in sync.
    """

    buf: list  # list[torch.Tensor], one (capacity_pages, 2, ps, nkv, d) per layer
    wrapper: BatchDecodeWithPagedKVCacheWrapper

    bsz: int
    page_size: int
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int
    num_layers: int
    capacity_pages: int
    pages_per_batch: int  # capacity_pages == bsz * pages_per_batch

    dtype: torch.dtype   # bf16 end-to-end
    device: torch.device

    num_sink_pages: int
    top_k: int
    num_recent_pages_fixed: int  # full recent pages, EXCLUDES the open page (open is implicit, +1)
    page_budget: int

    # Pre-allocated wrapper-owned buffers.
    float_workspace_buffer: torch.Tensor
    indices_buf_storage: torch.Tensor    # (num_kv_heads, bsz, page_budget) int32 contiguous (FI-facing flat = .view(-1))
    indices_buf: torch.Tensor            # (bsz, num_kv_heads, page_budget) int32 — Triton-facing permuted view of storage
    indptr_buf: torch.Tensor             # (bsz + 1,) int32, arange * page_budget
    last_page_len_buf: torch.Tensor      # (bsz,) int32, uniform under lockstep
    recent_offsets: torch.Tensor         # (num_recent_pages_fixed + 1,) int32 (full recent + open page at offset 0)
    last_page_idx: torch.Tensor          # (bsz,) int32, batch-biased physical IDs

    # Reusable scratch for batched H2D refresh of last_page_idx every decode
    # step: holds `arange(bsz, int32, device) * pages_per_batch` once;
    # `append_flashinfer_cache` builds the new tensor with `add(scratch, scalar, out=...)`.
    _batch_offsets: torch.Tensor = None  # (bsz,) int32

    # Python-side state (updated together with the GPU tensors). Lockstep
    # across batch — these are the SHARED logical (pre-bias) values.
    cur_seqlen: int = 0
    last_page_idx_py: int = 0
    last_page_len_py: int = 0


def _build_paged_buf_per_layer(
    preallocated_layers,
    prefill_len: int,
    page_size: int,
    capacity_pages: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    bsz: int = 1,
    pages_per_batch: int = None,
) -> list:
    """Build the per-layer paged KV buffer list and pack the prefill into it
    one layer at a time. Frees each layer's flat (bsz, nkv, alloc_len, d) cache
    immediately after pack so peak transient memory stays ~one layer's worth
    above the steady-state paged size (instead of 2× when buf and flat coexist
    as monolithic tensors).

    Multibatch: each batch b owns physical pages [b*pages_per_batch,
    (b+1)*pages_per_batch). The prefill of batch b is packed into pages
    [b*pages_per_batch, b*pages_per_batch + prefill_pages). Lockstep prefill
    (same prefill_len across batches).
    """
    prefill_pages = (prefill_len + page_size - 1) // page_size
    pad = prefill_pages * page_size - prefill_len
    if pages_per_batch is None:
        pages_per_batch = capacity_pages // max(bsz, 1)
    bufs = []
    for layer in preallocated_layers:
        layer_buf = torch.zeros(
            capacity_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        for b in range(bsz):
            k = layer.keys[b, :, :prefill_len, :]    # (nkv, T, d)
            v = layer.values[b, :, :prefill_len, :]
            if pad:
                k = torch.nn.functional.pad(k, (0, 0, 0, pad))
                v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            # (nkv, P*ps, d) -> (nkv, P, ps, d) -> (P, ps, nkv, d). Keep dtype.
            k = k.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(dtype)
            v = v.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(dtype)
            base = b * pages_per_batch
            layer_buf[base:base + prefill_pages, 0].copy_(k)
            layer_buf[base:base + prefill_pages, 1].copy_(v)
        bufs.append(layer_buf)
        # Free this layer's flat KV NOW — k/v are views; clearing the source
        # releases ~bsz * prefill_len * num_kv_heads * head_dim * 2 (K+V) bytes.
        layer._fi_mode = True
        layer.keys = None
        layer.values = None
    return bufs


# Legacy single-tensor pack — used by the full-KV FlashInfer baseline
# (`profile_decode_flash_infer.build_fi_baseline_cache`), which does NOT free
# flat keys/values and stays on a monolithic 6-D `buf`. The DCT path uses
# `_build_paged_buf_per_layer` instead to keep transient peak low.
def _pack_preallocated_to_paged(
    buf: torch.Tensor,
    preallocated_layers,
    prefill_len: int,
    page_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    prefill_pages = (prefill_len + page_size - 1) // page_size
    for l, layer in enumerate(preallocated_layers):
        k = layer.keys[0, :, :prefill_len, :]    # (nkv, T, d)
        v = layer.values[0, :, :prefill_len, :]
        pad = prefill_pages * page_size - prefill_len
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        k = k.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(dtype)
        v = v.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(dtype)
        buf[l, :prefill_pages, 0].copy_(k)
        buf[l, :prefill_pages, 1].copy_(v)
    return prefill_pages


def build_flashinfer_paged_cache(
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
    bsz: int = 1,
    workspace_bytes: int = 128 * 1024 * 1024,
) -> FlashInferPagedKVCache:
    """Build a FlashInferPagedKVCache populated from `preallocated_layers`
    (list of PreAllocatedLayer, one per model layer).

    `num_recent_pages_fixed` EXCLUDES the currently-open page (contract of
    this backend — see module docstring). It is the same `cfg.num_recent_pages`
    from `DCTPageConfig`. The open page is implicit and always allocated as
    +1, so total recent slots = `num_recent_pages_fixed + 1`.

    `bsz` controls the per-batch disjoint physical-page pool layout: each
    batch b owns pages [b * pages_per_batch, (b+1) * pages_per_batch). All
    indices written into `indices_buf` and `last_page_idx` are pre-biased
    by `b * pages_per_batch` so they index `cache.buf[layer]` directly.
    Lockstep prefill (uniform `prefill_len` across the batch) is required.
    """
    if num_sink_pages < 1:
        raise ValueError(
            f"num_sink_pages ({num_sink_pages}) must be >= 1. With "
            f"num_sink_pages=1 page 0 is fully attended (page_size tokens "
            f"of unconditional attention)."
        )
    if num_recent_pages_fixed < 0:
        raise ValueError(
            "num_recent_pages_fixed must be >= 0 (excludes the currently-open page; "
            "the open page is implicit, +1)."
        )
    if bsz < 1:
        raise ValueError(f"bsz ({bsz}) must be >= 1")

    page_budget = num_sink_pages + top_k + num_recent_pages_fixed + 1

    # Per-batch capacity in physical pages.
    prefill_pages = (prefill_len + page_size - 1) // page_size
    decode_pages = (max_decode_steps + page_size - 1) // page_size
    pages_per_batch = prefill_pages + decode_pages + 4  # per-batch slack
    capacity_pages = bsz * pages_per_batch

    # Paged KV buffer (bf16 NHD): one (capacity_pages, 2, page_size,
    # num_kv_heads, head_dim) tensor per layer, packed and flat-freed one
    # layer at a time. `cache.buf[layer_idx]` returns the 5-D tensor that
    # FlashInfer's wrapper.run() and DCT's paged_views_from_buf() consume.
    buf = _build_paged_buf_per_layer(
        preallocated_layers, prefill_len, page_size, capacity_pages,
        num_kv_heads, head_dim, dtype, device,
        bsz=bsz, pages_per_batch=pages_per_batch,
    )
    torch.cuda.empty_cache()

    # Indices buffer storage is HEAD-MAJOR (num_kv_heads, bsz, page_budget)
    # so the fork formula `indices[h * indptr[bsz] + page_iter]` (page.cuh
    # post Phase 2b multibatch patch) addresses (h, b, p_local) correctly.
    # `indices_buf` is the Triton/DCT-facing (bsz, num_kv_heads, page_budget)
    # permuted view (non-contiguous strides preserved for kernel writes).
    indices_buf_storage = torch.zeros(
        num_kv_heads, bsz, page_budget, dtype=torch.int32, device=device,
    )
    indices_buf = indices_buf_storage.permute(1, 0, 2)  # (bsz, nkv, pb), non-contig
    # Sink slice is static across the session — fill once, never rewritten
    # (per Stage 5 fused-kernel contract). For batch b, sink IDs are
    # arange(num_sink_pages) + b * pages_per_batch.
    if num_sink_pages > 0:
        sink_ids = torch.arange(num_sink_pages, dtype=torch.int32, device=device)
        batch_offsets = (
            torch.arange(bsz, dtype=torch.int32, device=device) * pages_per_batch
        )
        # Write through the permuted view: shape (bsz, nkv, num_sink_pages)
        # broadcast assignment of sink_ids + b*pages_per_batch.
        indices_buf[:, :, :num_sink_pages] = (
            sink_ids[None, None, :] + batch_offsets[:, None, None]
        )

    # indptr: per-head mode pre-bias semantics — the scheduler iterates
    # `page_budget` pages per batch; bias is applied inside indices_buf.
    # `indptr[bsz] = bsz * page_budget` is the head stride the patched
    # `get_phys_page` reads to address its (h, b, p) layout.
    indptr_buf = (
        torch.arange(bsz + 1, dtype=torch.int32, device=device) * page_budget
    )

    # Compute last-page state from the prefill. "Last page" here is the page
    # holding the currently-open slot; `last_page_len` is its valid token
    # count (1..page_size). Lockstep — same logical value for every batch.
    last_open_page = (prefill_len - 1) // page_size
    last_open_len = prefill_len - last_open_page * page_size  # in [1, page_size]

    last_page_len_buf = torch.full(
        (bsz,), last_open_len, dtype=torch.int32, device=device,
    )
    # last_page_idx is the BATCH-BIASED physical ID per batch.
    batch_offsets = (
        torch.arange(bsz, dtype=torch.int32, device=device) * pages_per_batch
    )
    last_page_idx = batch_offsets + last_open_page

    # recent_offsets: spans [-R, 0] (length R+1), so recent region covers R
    # full recent pages plus the currently-open page (offset 0 = last_page_idx
    # itself). Stage 5 kernel adds `last_page_idx[b]` (already batch-biased)
    # then stores into indices_buf[b, h, num_sink+top_k:].
    recent_offsets = torch.arange(
        -num_recent_pages_fixed, 1, dtype=torch.int32, device=device,
    )

    # Wrapper with pre-allocated buffers. use_cuda_graph=True forces static
    # shape / buffer identity for run() — we DO NOT actually call
    # torch.cuda.graph() anywhere; eager is the measured path for Phase 2b.
    float_workspace_buffer = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device,
    )
    # FI-facing flat indices view = head-major contiguous storage (.view(-1)).
    # The Triton-facing `indices_buf` view writes here through permuted strides.
    indices_flat_buf = indices_buf_storage.view(-1)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_flat_buf,
        paged_kv_last_page_len_buffer=last_page_len_buf,
    )

    # Plan ONCE at build time. The scheduler's partition depends only on
    # indptr + last_page_len + shape — none of which change after this.
    # Plan with last_page_len_buf already initialized from prefill; subsequent
    # updates to last_page_len_buf (in-place) and indices_buf (fused kernel)
    # are picked up live at run() via the pre-allocated buffers.
    wrapper.plan(
        indptr_buf,
        indices_flat_buf,
        last_page_len_buf,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_data_type=dtype, kv_data_type=dtype,
        page_budget=page_budget,
    )

    cache = FlashInferPagedKVCache(
        buf=buf,
        wrapper=wrapper,
        bsz=bsz,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_layers=num_layers,
        capacity_pages=capacity_pages,
        pages_per_batch=pages_per_batch,
        dtype=dtype,
        device=device,
        num_sink_pages=num_sink_pages,
        top_k=top_k,
        num_recent_pages_fixed=num_recent_pages_fixed,
        page_budget=page_budget,
        float_workspace_buffer=float_workspace_buffer,
        indices_buf_storage=indices_buf_storage,
        indices_buf=indices_buf,
        indptr_buf=indptr_buf,
        last_page_len_buf=last_page_len_buf,
        recent_offsets=recent_offsets,
        last_page_idx=last_page_idx,
        _batch_offsets=batch_offsets,
        cur_seqlen=prefill_len,
        last_page_idx_py=last_open_page,
        last_page_len_py=last_open_len,
    )
    return cache


def append_flashinfer_cache(
    cache: FlashInferPagedKVCache,
    new_k: torch.Tensor,   # (bsz, num_kv_heads, 1, head_dim) bf16, post-RoPE
    new_v: torch.Tensor,
    layer_idx: int,
) -> None:
    """In-place append of one decode step's K/V into the FlashInfer paged
    cache. Only layer 0 advances the shared counters; other layers just write
    to the already-advanced slot.

    Lockstep across batch — every batch advances together with the same
    `last_page_idx_py` (logical) and `last_page_len_py`. The GPU tensor
    `cache.last_page_idx` holds per-batch BIASED physical IDs (logical +
    `b * pages_per_batch`).

    When the currently-open page fills (last_page_len == page_size), wrap to
    a fresh page at `last_page_idx_py + 1` with length 1. Python-side counters
    and their GPU mirrors (`last_page_idx`, `last_page_len_buf`) are kept in
    sync so both the Stage 5 fused kernel and FlashInfer's run() see the
    latest state.
    """
    bsz = cache.bsz
    if layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1

        # Mirror to the GPU tensors. Refresh last_page_idx as
        # `_batch_offsets + last_page_idx_py` (broadcast scalar add). Tiny
        # H2D from a Python int per step at layer 0 only.
        torch.add(
            cache._batch_offsets, cache.last_page_idx_py, out=cache.last_page_idx,
        )
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    logical_page = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    if logical_page >= cache.pages_per_batch:
        raise RuntimeError(
            f"FlashInferPagedKVCache overflow: logical_page={logical_page} >= "
            f"pages_per_batch={cache.pages_per_batch}. Increase max_decode_steps "
            f"at build time."
        )

    # (bsz, num_kv_heads, head_dim) — K/V last-token slice across the batch.
    k_flat = new_k.reshape(bsz, cache.num_kv_heads, cache.head_dim)
    v_flat = new_v.reshape(bsz, cache.num_kv_heads, cache.head_dim)
    layer_buf = cache.buf[layer_idx]
    if bsz == 1:
        # Hot path stays alloc-free: scalar physical page index.
        phys_page = logical_page  # batch 0, no bias
        layer_buf[phys_page, 0, slot].copy_(k_flat[0])
        layer_buf[phys_page, 1, slot].copy_(v_flat[0])
    else:
        # Multibatch: write all bsz batches into their per-batch physical
        # pages with one advanced-indexing write each (no Python loop in the
        # fast path). `cache.last_page_idx` is the (bsz,) batch-biased GPU
        # tensor refreshed at layer 0; convert to long for fancy indexing.
        # NOTE: layer_buf[phys, 0, slot] writes a (bsz, num_kv_heads, head_dim)
        # tile in one indirect indexed_put.
        phys = cache.last_page_idx.long()
        layer_buf[phys, 0, slot] = k_flat
        layer_buf[phys, 1, slot] = v_flat


def flashinfer_decode_attention(
    query_states: torch.Tensor,   # (bsz, num_qo_heads, 1, head_dim) bf16
    cache: FlashInferPagedKVCache,
    layer_idx: int,
) -> torch.Tensor:
    """Run FlashInfer paged decode attention on the selected pages.

    NO pack_indices call here — the fused Stage 5 kernel
    `topk_sort_and_pack_triton` is responsible for writing the topk + recent
    slices of `cache.indices_buf` before this function is called. This
    function just dispatches `wrapper.run(q, cache.buf[layer_idx])`.

    Input / output shape matches `quest_decode_attention` for call-site
    symmetry: (bsz, num_qo_heads, 1, head_dim) bf16 in, (bsz, num_qo_heads, 1,
    head_dim) bf16 out.
    """
    bsz = cache.bsz
    # FlashInfer's run expects q as (batch_size, num_qo_heads, head_dim).
    # query_states is (bsz, num_qo_heads, 1, head_dim); may be non-contiguous
    # after the attention-forward's transpose, so reshape+contiguous handles
    # that for bsz>1 (FI requires contiguous q across the batch dim).
    q_flat = query_states.reshape(bsz, cache.num_qo_heads, cache.head_dim).contiguous()
    # Single 5-D tensor form of the paged cache: (capacity_pages, 2, page_size,
    # num_kv_heads, head_dim). FlashInfer interprets [:, 0] as K, [:, 1] as V.
    out = cache.wrapper.run(q_flat, cache.buf[layer_idx])
    # (bsz, num_qo_heads, head_dim) -> (bsz, num_qo_heads, 1, head_dim) to
    # match the shape the outer forward expects (pre-transpose SDPA output).
    return out.view(bsz, cache.num_qo_heads, 1, cache.head_dim)
