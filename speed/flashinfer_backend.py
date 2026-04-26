"""
FlashInfer paged decode adapter for DCT-Page (Phase 2b Stage 6).

Bridges DCT-Page's decode path into upstream FlashInfer's
`BatchDecodeWithPagedKVCacheWrapper` (v0.6.7.post3 fork at
`/home/yoongonkim/flashinfer-dct`) with the per-head indices patch landed in
Stages 2–3. Scope: drop mode, bsz=1, native bf16 end-to-end (no fp16 cast).

Mirrors the shape of `speed/quest_backend.py` (build/append/decode) but on the
upstream plan()/run() API with:

- Native bf16 KV cache (Phase 2 fp16 cast path removed).
- 3-D `indices_buf: (bsz, num_kv_heads, page_budget)` int32 written in-place by
  the fused Stage 5 kernel `topk_sort_and_pack_triton`. A flat view is handed
  to the wrapper via `use_cuda_graph=True` at construction — this pins
  pre-allocated buffers without requiring graph capture at runtime.
- `page_budget` kwarg in plan() (Stage 3 addition) so the per-head branch of
  `get_phys_page(page_iter, head_idx)` fires at run time.
- Per-call pack of sink indices at cache init (static across the session) —
  the fused Stage 5 kernel never rewrites that slice.

Key layout choice (pages aligned with token 0):
  page 0 = tokens [0, page_size)
  page p = tokens [p*page_size, (p+1)*page_size)

"Sink", "topk", "recent" are logical regions within `indices_buf`:

  indices_buf[b, h, :]:
    [0, num_sink_pages)                        : sink page IDs (static)
    [num_sink_pages, num_sink_pages + top_k)   : topk middle pages
    [num_sink_pages + top_k, page_budget)      : recent pages (last tail
                                                 entry = last_page_idx itself,
                                                 i.e. the currently-open page)

`num_recent_pages_fixed` INCLUDES the open page (recent_offsets span
[-R+1, 0]). This lets `last_page_len_buf` track the current length of the open
page directly — FlashInfer's standard contract (last entry in indices is the
last page; `paged_kv_last_page_len` is its valid length).

Sink alignment requirement: this backend targets `sink_size >= page_size` and
a clean `num_sink_pages = ceil(sink_size / page_size)` partition. When
`sink_size < page_size`, the first physical page contains (sink + first
few middle tokens) — we still treat page 0 as "sink" but a handful of middle
tokens end up attended unconditionally (same ~28-token overshoot noted in
`speed/quest_backend.py`). For Phase 2b benchmarks, configure `sink_size=32`
(= page_size) to avoid this.

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

    Layout of `buf`: (num_layers, capacity_pages, 2, page_size, num_kv_heads,
    head_dim). `buf[l, p, 0] = K`, `buf[l, p, 1] = V`, same as Quest's layout.
    All layers share the page-advance counters (`cur_seqlen`,
    `last_page_idx_py`, `last_page_len_py`); only layer 0 advances them.

    Fixed-width indices: `page_budget = num_sink_pages + top_k +
    num_recent_pages_fixed` entries per (batch, kv_head) row. `plan()` is
    called once at build time with this budget; `run()` reads the latest
    indices/last_page_len out of the pre-allocated buffers.

    CPU/GPU state mirror: `last_page_idx_py` / `last_page_len_py` are Python
    ints used for in-kernel-free Python-side indexing in
    `append_flashinfer_cache`. `last_page_idx` / `last_page_len_buf` are int32
    GPU tensors used by the fused Stage 5 kernel and FlashInfer's run path.
    They must stay in sync.
    """

    buf: torch.Tensor
    wrapper: BatchDecodeWithPagedKVCacheWrapper

    page_size: int
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int
    num_layers: int
    capacity_pages: int

    dtype: torch.dtype   # bf16 end-to-end
    device: torch.device

    num_sink_pages: int
    top_k: int
    num_recent_pages_fixed: int  # includes the currently-open page
    page_budget: int

    # Pre-allocated wrapper-owned buffers.
    float_workspace_buffer: torch.Tensor
    indices_buf: torch.Tensor            # (1, num_kv_heads, page_budget) int32
    indptr_buf: torch.Tensor             # (2,) int32 = [0, page_budget]
    last_page_len_buf: torch.Tensor      # (1,) int32
    recent_offsets: torch.Tensor         # (num_recent_pages_fixed,) int32
    last_page_idx: torch.Tensor          # (1,) int32, tensor mirror

    # Python-side state (updated together with the GPU tensors).
    cur_seqlen: int = 0
    last_page_idx_py: int = 0
    last_page_len_py: int = 0


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
    """One-time post-prefill copy from DCT's flat (1, nkv, alloc_len, d) cache
    to FlashInfer's NHD paged buffer. Returns number of physical pages used
    for the prefill (including the partially-filled last page).
    """
    prefill_pages = (prefill_len + page_size - 1) // page_size
    for l, layer in enumerate(preallocated_layers):
        k = layer.keys[0, :, :prefill_len, :]    # (nkv, T, d)
        v = layer.values[0, :, :prefill_len, :]
        pad = prefill_pages * page_size - prefill_len
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        # (nkv, P*ps, d) -> (nkv, P, ps, d) -> (P, ps, nkv, d). Keep dtype.
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
    workspace_bytes: int = 128 * 1024 * 1024,
) -> FlashInferPagedKVCache:
    """Build a FlashInferPagedKVCache populated from `preallocated_layers`
    (list of PreAllocatedLayer, one per model layer).

    `num_recent_pages_fixed` INCLUDES the currently-open page (contract of
    this backend — see module docstring). Callers migrating from Quest's
    adapter should pass `quest_num_recent_pages_fixed` unchanged; Quest also
    uses a +1 page of "recent overshoot" so the token budgets are comparable.
    """
    if num_sink_pages < 1:
        raise ValueError(
            f"num_sink_pages ({num_sink_pages}) must be >= 1. With sink_size "
            f"< page_size, set num_sink_pages=1 and accept page 0 absorbing "
            f"some early middle tokens as unconditional attention."
        )
    if num_recent_pages_fixed < 1:
        raise ValueError(
            "num_recent_pages_fixed must be >= 1 (includes the currently-open page)."
        )

    bsz = 1  # Phase 2b scope
    page_budget = num_sink_pages + top_k + num_recent_pages_fixed

    # Capacity in physical pages for the whole session.
    prefill_pages = (prefill_len + page_size - 1) // page_size
    decode_pages = (max_decode_steps + page_size - 1) // page_size
    capacity_pages = prefill_pages + decode_pages + 4  # a little slack

    # Paged KV buffer (bf16 NHD, single 5-D tensor per layer).
    buf = torch.zeros(
        num_layers, capacity_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    _pack_preallocated_to_paged(
        buf, preallocated_layers, prefill_len, page_size,
        num_layers, num_kv_heads, head_dim, dtype,
    )

    # Per-head indices buffer. Layout is (bsz, num_kv_heads, page_budget);
    # flat view matches what FlashInfer's per-head `get_phys_page` kernel
    # expects: indices[head_idx * page_budget + page_iter] at bsz=1.
    indices_buf = torch.zeros(
        bsz, num_kv_heads, page_budget, dtype=torch.int32, device=device,
    )
    # Sink slice is static across the session — fill once, never rewritten
    # (per Stage 5 fused-kernel contract).
    if num_sink_pages > 0:
        sink_ids = torch.arange(num_sink_pages, dtype=torch.int32, device=device)
        indices_buf[:, :, :num_sink_pages] = sink_ids

    # indptr: for bsz=1 with per-head indices, the wrapper's scheduler sees
    # `page_budget` pages per batch (regardless of num_kv_heads); the per-head
    # stride is applied inside `get_phys_page` via the kernel-side patch.
    indptr_buf = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
    indptr_buf[1] = page_budget

    # Compute last-page state from the prefill. "Last page" here is the page
    # holding the currently-open slot; `last_page_len` is its valid token
    # count (1..page_size).
    last_open_page = (prefill_len - 1) // page_size
    last_open_len = prefill_len - last_open_page * page_size  # in [1, page_size]

    last_page_len_buf = torch.full(
        (bsz,), last_open_len, dtype=torch.int32, device=device,
    )
    last_page_idx = torch.full(
        (bsz,), last_open_page, dtype=torch.int32, device=device,
    )

    # recent_offsets: spans [-R+1, 0], so recent region ends at last_page_idx
    # itself (the currently-open page). Stage 5 kernel adds `last_page_idx[b]`
    # then stores into indices_buf[b, h, num_sink_pages+top_k:].
    recent_offsets = torch.arange(
        -num_recent_pages_fixed + 1, 1, dtype=torch.int32, device=device,
    )

    # Wrapper with pre-allocated buffers. use_cuda_graph=True forces static
    # shape / buffer identity for run() — we DO NOT actually call
    # torch.cuda.graph() anywhere; eager is the measured path for Phase 2b.
    float_workspace_buffer = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device,
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf.view(-1),
        paged_kv_last_page_len_buffer=last_page_len_buf,
    )

    # Plan ONCE at build time. The scheduler's partition depends only on
    # indptr + last_page_len + shape — none of which change after this.
    # Plan with last_page_len_buf already initialized from prefill; subsequent
    # updates to last_page_len_buf (in-place) and indices_buf (fused kernel)
    # are picked up live at run() via the pre-allocated buffers.
    wrapper.plan(
        indptr_buf,
        indices_buf.view(-1),
        last_page_len_buf,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_data_type=dtype, kv_data_type=dtype,
        page_budget=page_budget,
    )

    cache = FlashInferPagedKVCache(
        buf=buf,
        wrapper=wrapper,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_layers=num_layers,
        capacity_pages=capacity_pages,
        dtype=dtype,
        device=device,
        num_sink_pages=num_sink_pages,
        top_k=top_k,
        num_recent_pages_fixed=num_recent_pages_fixed,
        page_budget=page_budget,
        float_workspace_buffer=float_workspace_buffer,
        indices_buf=indices_buf,
        indptr_buf=indptr_buf,
        last_page_len_buf=last_page_len_buf,
        recent_offsets=recent_offsets,
        last_page_idx=last_page_idx,
        cur_seqlen=prefill_len,
        last_page_idx_py=last_open_page,
        last_page_len_py=last_open_len,
    )
    return cache


def append_flashinfer_cache(
    cache: FlashInferPagedKVCache,
    new_k: torch.Tensor,   # (1, num_kv_heads, 1, head_dim) bf16, post-RoPE
    new_v: torch.Tensor,
    layer_idx: int,
) -> None:
    """In-place append of one decode step's K/V into the FlashInfer paged
    cache. Only layer 0 advances the shared counters; other layers just write
    to the already-advanced slot.

    When the currently-open page fills (last_page_len == page_size), wrap to
    a fresh page at `last_page_idx + 1` with length 1. Python-side counters
    and their GPU mirrors (`last_page_idx`, `last_page_len_buf`) are kept in
    sync so both the Stage 5 fused kernel and FlashInfer's run() see the
    latest state.
    """
    if layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1

        # Mirror to the GPU tensors. `fill_` is a tiny H2D copy but it
        # happens once per step (at layer 0), not per layer.
        cache.last_page_idx.fill_(cache.last_page_idx_py)
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    page_idx = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    if page_idx >= cache.capacity_pages:
        raise RuntimeError(
            f"FlashInferPagedKVCache overflow: page_idx={page_idx} >= "
            f"capacity_pages={cache.capacity_pages}. Increase max_decode_steps "
            f"at build time."
        )

    k_flat = new_k.reshape(cache.num_kv_heads, cache.head_dim)
    v_flat = new_v.reshape(cache.num_kv_heads, cache.head_dim)
    cache.buf[layer_idx, page_idx, 0, slot].copy_(k_flat)
    cache.buf[layer_idx, page_idx, 1, slot].copy_(v_flat)


def flashinfer_decode_attention(
    query_states: torch.Tensor,   # (1, num_qo_heads, 1, head_dim) bf16
    cache: FlashInferPagedKVCache,
    layer_idx: int,
) -> torch.Tensor:
    """Run FlashInfer paged decode attention on the selected pages.

    NO pack_indices call here — the fused Stage 5 kernel
    `topk_sort_and_pack_triton` is responsible for writing the topk + recent
    slices of `cache.indices_buf` before this function is called. This
    function just dispatches `wrapper.run(q, cache.buf[layer_idx])`.

    Input / output shape matches `quest_decode_attention` for call-site
    symmetry: (1, num_qo_heads, 1, head_dim) bf16 in, (1, num_qo_heads, 1,
    head_dim) bf16 out.
    """
    # FlashInfer's run expects q as (batch_size, num_qo_heads, head_dim).
    # query_states is (1, num_qo_heads, 1, head_dim); may be non-contiguous
    # after the attention-forward's transpose, so reshape handles that.
    q_flat = query_states.reshape(1, cache.num_qo_heads, cache.head_dim)
    # Single 5-D tensor form of the paged cache: (capacity_pages, 2, page_size,
    # num_kv_heads, head_dim). FlashInfer interprets [:, 0] as K, [:, 1] as V.
    out = cache.wrapper.run(q_flat, cache.buf[layer_idx])
    # (1, num_qo_heads, head_dim) -> (1, num_qo_heads, 1, head_dim) to match
    # the shape the outer forward expects (pre-transpose SDPA output layout).
    return out.view(1, cache.num_qo_heads, 1, cache.head_dim)
