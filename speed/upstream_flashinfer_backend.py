"""
Upstream-FlashInfer paged decode adapter (no custom-fork patch).

Sibling of `speed/flashinfer_backend.py`. The other adapter depends on the
per-head `indices` patch (`page_budget` kwarg in plan/run) living in the
DCT-Page fork at `/home/yoongonkim/flashinfer-dct`. This adapter takes the
"virtual batch = (real batch, KV head)" approach so the 2-D indices API in
stock FlashInfer is enough.

Layout (post-Phase-1 multibatch — vbsz = B*H, v = b*H + h):

  Physical KV pool is partitioned into `bsz * num_kv_heads` contiguous chunks
  of `pages_per_head` pages each. Each physical page stores data for one
  (batch, KV head) pair (shape `(page_size, 1, head_dim)`).

  Q reshape `(B, num_qo_heads, 1, d) → (B, H, group_size, d) → (B*H, group_size, d)`.
  Virtual batch v = b*H + h carries the `group_size` Q heads that belong to
  KV head `h` within batch `b`.

  `wrapper.plan()` is called with `num_qo_heads=group_size, num_kv_heads=1` —
  each virtual batch attends its own 1 KV head over its own slice of pages.

The attention math is identical to the per-head fork path: each Q head's
softmax is independent, and virtual-batching is just a parallelism axis.

Layout at decode time (per layer):
  buf_8d:  (num_layers, B, H, pages_per_head, 2, page_size, 1, head_dim)
  buf:     view as (num_layers, B*H*pages_per_head, 2, page_size, 1, head_dim)
           — what FlashInfer's wrapper.run reads.

  Physical page id for (b, h, p_local) = (b*H + h) * pages_per_head + p_local.

Indices buffers:
  indices_buf_3d:    (B, H, page_budget) int32 — HEAD-LOCAL indices, exactly
    what `topk_sort_and_pack_triton` writes. Sink slice prefilled once.
  indices_flat_buf:  (B*H * page_budget,) int32 — PHYSICAL indices that
    FlashInfer's wrapper reads. Refreshed per decode step via a single
    `torch.add` that applies the per-vbatch offset `(b*H + h) * pages_per_head`.

Sink contract: page 0 (head-local) is sink iff `num_sink_pages >= 1`. With
`num_sink_pages=1`, page 0 within each (b, h) pool is unconditionally
attended (page_size tokens of that batch / KV head).

plan() lifecycle: called ONCE at build time. Stock scheduler uses only
indptr + last_page_len + shape metadata — none of which change during
decode. `last_page_len_buf` mutates in-place; `indices_flat_buf` is
refreshed in-place each step. Both are pinned via `use_cuda_graph=True`.

Multibatch contract:
- Lockstep across batch — every batch advances together with the same
  `last_page_idx_py` (head-local logical) and `last_page_len_py`. Ragged
  prefill is NOT supported.
- bsz=1 path is bit-equivalent to the pre-multibatch implementation: the
  new 8-D buf collapses to today's 7-D layout when B=1, and the indices
  bias formula `(b*H + h) * pages_per_head` reduces to `h * pages_per_head`.
- Build-time invariants are asserted (see `build_upstream_flashinfer_paged_cache`)
  to catch silent vbatch-ordering aliasing before any kernel sees stale layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import torch

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


@dataclass
class UpstreamFlashInferPagedKVCache:
    """Upstream-FlashInfer paged KV cache (virtual-batch = (batch, KV head)).

    Storage:
      `buf` is the flat (FlashInfer-facing) 6-D view;
      `buf_8d` is the same underlying storage viewed 8-D so append writes
      can address (layer, batch, head, page, 0/1, slot, 0, :) directly.

    Python-side counters (`cur_seqlen`, `last_page_idx_py`, `last_page_len_py`)
    are scalars under lockstep — every (b, h) virtual batch shares the same
    head-local logical position. The GPU tensors `last_page_idx` (B,) and
    `last_page_len_buf` (B*H,) are filled with these scalars; lockstep makes
    `.fill_()` correct.
    """

    # Per-layer storage: list of `num_layers` tensors. `buf[l]` is the
    # FI-facing 5-D view; `buf_views[l]` is the same storage viewed 7-D for
    # append writes. Per-layer allocation (mirroring fork's
    # `_build_paged_buf_per_layer`) caps peak transient memory at one
    # layer's flat KV + one layer's paged buf, instead of the all-layers
    # × {flat, paged} double allocation that OOMs on A6000 at bsz>=2.
    buf: list                         # list[Tensor], each (B*H*P, 2, ps, 1, d)
    buf_views: list                   # list[Tensor], each (B, H, P, 2, ps, 1, d)
    wrapper: BatchDecodeWithPagedKVCacheWrapper

    bsz: int
    page_size: int
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int                 # total Q heads = num_kv_heads * group_size
    group_size: int                   # num_qo_heads // num_kv_heads
    num_layers: int

    # Per-(b, h) capacity — each virtual batch has its own pool of this many
    # physical pages.
    pages_per_head: int
    total_pages: int                  # bsz * num_kv_heads * pages_per_head

    dtype: torch.dtype
    device: torch.device

    num_sink_pages: int
    top_k: int
    num_recent_pages_fixed: int
    page_budget: int                  # num_sink_pages + top_k + num_recent_pages_fixed + 1 (the +1 is the implicit open page)

    # Pre-allocated wrapper-owned buffers.
    float_workspace_buffer: torch.Tensor
    indices_buf_3d: torch.Tensor      # (B, H, page_budget) int32, HEAD-LOCAL
    indices_flat_buf: torch.Tensor    # (B*H * page_budget,) int32, PHYSICAL
    indptr_buf: torch.Tensor          # (B*H + 1,) int32
    last_page_len_buf: torch.Tensor   # (B*H,) int32
    recent_offsets: torch.Tensor      # (num_recent_pages_fixed + 1,) int32 (full recent + open page at offset 0)
    last_page_idx: torch.Tensor       # (B,) int32, broadcast head-local logical

    head_offset: torch.Tensor         # (B*H, 1) int32, value (b*H + h) * pages_per_head

    # Python-side state.
    cur_seqlen: int = 0
    last_page_idx_py: int = 0
    last_page_len_py: int = 0


def _build_paged_bufs_per_layer_upstream(
    preallocated_layers,
    prefill_len: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    pages_per_head: int,
    dtype: torch.dtype,
    device: torch.device,
    bsz: int,
    free_flat_kv: bool = True,
):
    """Per-layer alloc-pack-free, mirroring the fork's
    `_build_paged_buf_per_layer` (`speed/flashinfer_backend.py:181-204`).

    For each layer:
      1. Allocate the per-layer paged buf (B, H, pages_per_head, 2, ps, 1, d).
      2. Pack from `layer.keys` / `layer.values` (per-batch loop).
      3. Drop the flat KV (set keys/values to None, mark _fi_mode=True).

    Peak transient memory: one layer's flat KV + one layer's paged buf
    (~268 MiB each at bsz=2/32K), instead of all-layers × {flat, paged}
    (~17 GiB) which OOMs on A6000.

    Returns `(buf_list, buf_views_list, prefill_pages)` where:
      - `buf_list[l]` has shape `(B*H*pages_per_head, 2, ps, 1, d)` — FI-facing
      - `buf_views_list[l]` is the same storage viewed `(B, H, pages_per_head, 2, ps, 1, d)` —
        append-friendly for `[:, :, page_idx, 0/1, slot, 0, :]`.

    Lockstep prefill: every batch shares the same `prefill_len`. Ragged
    prefill is not supported.
    """
    prefill_pages = (prefill_len + page_size - 1) // page_size
    pad = prefill_pages * page_size - prefill_len

    buf_list = []
    buf_views_list = []
    for layer in preallocated_layers:
        # Allocate this layer's paged buf as 7-D so append writes land via
        # named axes; the 5-D view shares storage and is what FI consumes.
        layer_buf_7d = torch.zeros(
            bsz, num_kv_heads, pages_per_head, 2, page_size, 1, head_dim,
            dtype=dtype, device=device,
        )
        layer_buf_5d = layer_buf_7d.view(
            bsz * num_kv_heads * pages_per_head, 2, page_size, 1, head_dim,
        )

        # Pack prefill from this layer's flat KV.
        for b in range(bsz):
            k = layer.keys[b, :, :prefill_len, :]    # (H, T, d)
            v = layer.values[b, :, :prefill_len, :]
            if pad:
                k = torch.nn.functional.pad(k, (0, 0, 0, pad))
                v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            k = k.view(num_kv_heads, prefill_pages, page_size, head_dim).to(dtype)
            v = v.view(num_kv_heads, prefill_pages, page_size, head_dim).to(dtype)
            layer_buf_7d[b, :, :prefill_pages, 0, :, 0, :].copy_(k)
            layer_buf_7d[b, :, :prefill_pages, 1, :, 0, :].copy_(v)

        buf_list.append(layer_buf_5d)
        buf_views_list.append(layer_buf_7d)

        if free_flat_kv:
            # Drop this layer's flat KV NOW so the next layer's allocation
            # reuses freed bytes. k/v above are views; clearing the source
            # releases the layer's KV bytes immediately.
            layer._fi_mode = True
            layer.keys = None
            layer.values = None

    return buf_list, buf_views_list, prefill_pages


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
    bsz: int = 1,
    workspace_bytes: int = 128 * 1024 * 1024,
) -> UpstreamFlashInferPagedKVCache:
    """Build a virtual-batch-per-(b,h) cache populated from DCT's prefilled
    `preallocated_layers`. `num_recent_pages_fixed` EXCLUDES the open page;
    the open page is implicit (+1) — same contract as
    `flashinfer_backend.build_flashinfer_paged_cache`.

    Vbatch ordering: `v = b*H + h` (h-contiguous within each batch). This is
    asserted at build via `head_offset == arange(B*H) * pages_per_head`.

    Lockstep across batch.
    """
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads "
            f"({num_kv_heads}) for GQA virtual-batch layout."
        )
    group_size = num_qo_heads // num_kv_heads
    if num_sink_pages < 1:
        raise ValueError("num_sink_pages must be >= 1.")
    if num_recent_pages_fixed < 0:
        raise ValueError(
            "num_recent_pages_fixed must be >= 0 (excludes the currently-open page; "
            "the open page is implicit, +1)."
        )
    if bsz < 1:
        raise ValueError(f"bsz ({bsz}) must be >= 1")

    page_budget = num_sink_pages + top_k + num_recent_pages_fixed + 1
    vbsz = bsz * num_kv_heads

    prefill_pages = (prefill_len + page_size - 1) // page_size
    decode_pages = (max_decode_steps + page_size - 1) // page_size
    pages_per_head = prefill_pages + decode_pages + 4  # slack matches fork backend
    total_pages = bsz * num_kv_heads * pages_per_head

    # Per-layer alloc-pack-free (mirrors fork's `_build_paged_buf_per_layer`).
    # `buf` is a list of (B*H*pages_per_head, 2, ps, 1, d) tensors — one per
    # layer; each is FI's source of truth via `wrapper.run(q, cache.buf[l])`.
    # `buf_views[l]` is the same storage viewed (B, H, P, 2, ps, 1, d) for
    # append writes.
    buf, buf_views, _prefill_pages = _build_paged_bufs_per_layer_upstream(
        preallocated_layers,
        prefill_len=prefill_len,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        pages_per_head=pages_per_head,
        dtype=dtype,
        device=device,
        bsz=bsz,
    )
    torch.cuda.empty_cache()

    # Head-local indices, shape matches `topk_sort_and_pack_triton` contract
    # at bsz>=1 (the same fused kernel drives both backends).
    indices_buf_3d = torch.zeros(
        bsz, num_kv_heads, page_budget, dtype=torch.int32, device=device,
    )
    # Sink slice is static per (b, h) (head-local: [0, num_sink_pages)).
    # Bias to physical pages is applied each step into `indices_flat_buf`.
    sink_ids = torch.arange(num_sink_pages, dtype=torch.int32, device=device)
    indices_buf_3d[:, :, :num_sink_pages] = sink_ids   # broadcasts across (B, H)

    # Physical indices buffer (what FI reads via `paged_kv_indices_buffer`).
    indices_flat_buf = torch.zeros(
        vbsz * page_budget, dtype=torch.int32, device=device,
    )
    # head_offset: (B*H, 1) with values (b*H + h) * pages_per_head — the
    # per-vbatch bias added at refresh time to translate head-local IDs to
    # physical IDs in the flat pool.
    head_offset = (
        torch.arange(vbsz, dtype=torch.int32, device=device) * pages_per_head
    ).view(vbsz, 1)

    # ---- Build-time invariant asserts (Architect rec #2) ----
    # These catch the silent vbatch-ordering aliasing class entirely. The
    # plan's correctness hinges on three coupled invariants:
    #   1. Each per-layer `buf_views[l]` is C-contiguous so its
    #      `(B*H*P, 2, ps, 1, d)` flat view (= `buf[l]`) iterates physical
    #      pages in v=b*H+h order.
    #   2. `indices_buf_3d` is C-contiguous so its `(B*H, page_budget)` view
    #      matches the same ordering.
    #   3. `head_offset.view(-1)` equals `arange(B*H) * pages_per_head` — i.e.
    #      the bias tensor encodes v=b*H+h, not h*B+b.
    assert all(v.is_contiguous() for v in buf_views), \
        "every per-layer buf_views[l] must be C-contiguous"
    assert indices_buf_3d.is_contiguous(), "indices_buf_3d must be C-contiguous"
    expected_offsets = (
        torch.arange(vbsz, dtype=torch.int32, device=device) * pages_per_head
    ).view(vbsz, 1)
    assert head_offset.eq(expected_offsets).all(), (
        f"head_offset must equal arange(B*H={vbsz}) * pages_per_head "
        f"({pages_per_head}); vbatch ordering is v=b*H+h (row-major over (B, H))."
    )

    # Seed indices_flat_buf with the post-bias sink IDs (topk/recent regions
    # start as zeros in indices_buf_3d and get filled by the Stage 5 kernel
    # each decode step). This seeding is overwritten by the per-step
    # torch.add before every run() anyway — we do it here for plan() to see
    # sensible values during the one-time scheduler setup.
    torch.add(
        indices_buf_3d.view(vbsz, page_budget), head_offset,
        out=indices_flat_buf.view(vbsz, page_budget),
    )

    # indptr: one row per virtual batch, each spanning `page_budget` pages.
    indptr_buf = (
        torch.arange(vbsz + 1, dtype=torch.int32, device=device) * page_budget
    )

    last_open_page = (prefill_len - 1) // page_size
    last_open_len = prefill_len - last_open_page * page_size
    last_page_len_buf = torch.full(
        (vbsz,), last_open_len, dtype=torch.int32, device=device,
    )
    # last_page_idx is keyed on the REAL batch dim (the topk kernel's `bsz`
    # axis). Broadcast head-local logical page across batches under lockstep.
    last_page_idx = torch.full(
        (bsz,), last_open_page, dtype=torch.int32, device=device,
    )

    recent_offsets = torch.arange(
        -num_recent_pages_fixed, 1, dtype=torch.int32, device=device,
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
    # Per-vbatch shape args (group_size, 1) stay the same regardless of vbsz;
    # vbsz comes implicitly from `len(indptr_buf) - 1`.
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

    # CUDA graph sanity (Architect rec #5). Print scheduler partition info at
    # high vbsz so degenerate splits are visible. Gated by env to keep noise
    # off by default. Use `_plan_info` if FI exposes it; otherwise dump the
    # wrapper's __dict__ keys for triage.
    if os.environ.get("OMC_FI_PLAN_DEBUG") == "1":
        info = getattr(wrapper, "_plan_info", None)
        if info is None:
            info = {k: v for k, v in wrapper.__dict__.items()
                    if not k.startswith("_buf") and not torch.is_tensor(v)}
        print(
            f"[upstream-FI plan] vbsz={vbsz} page_budget={page_budget} "
            f"pages_per_head={pages_per_head} plan_info={info}"
        )

    return UpstreamFlashInferPagedKVCache(
        buf=buf,
        buf_views=buf_views,
        wrapper=wrapper,
        bsz=bsz,
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
    new_k: torch.Tensor,     # (B, num_kv_heads, 1, head_dim) bf16, post-RoPE
    new_v: torch.Tensor,
    layer_idx: int,
) -> None:
    """Append one decode step's K/V into the virtual-batch-per-(b,h) cache.
    Only layer 0 advances the shared counters. last_page_idx/last_page_len
    mirrors are broadcast across all virtual batches via `.fill_()` (lockstep
    means every (b, h) shares the same head-local open-page position).
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

    bsz = cache.bsz
    k_flat = new_k.reshape(bsz, cache.num_kv_heads, cache.head_dim)
    v_flat = new_v.reshape(bsz, cache.num_kv_heads, cache.head_dim)
    # buf_views[l]: (B, H, P, 2, ps, 1, d). Touch all (b, h) at (page_idx, slot).
    cache.buf_views[layer_idx][:, :, page_idx, 0, slot, 0, :].copy_(k_flat)
    cache.buf_views[layer_idx][:, :, page_idx, 1, slot, 0, :].copy_(v_flat)


def refresh_upstream_indices_flat(
    cache: UpstreamFlashInferPagedKVCache,
) -> None:
    """Apply per-(b, h) page-pool bias to the head-local scratch indices and
    write the result into the FI-facing flat buffer. Must be called AFTER
    `topk_sort_and_pack_triton` populates `indices_buf_3d` for this step
    and BEFORE `wrapper.run()`.

    Operation: `indices_flat[v, :] = indices_buf_3d_view[v, :] + head_offset[v]`
    where `v = b*H + h`. One fused `torch.add` with a pre-allocated output
    view; no intermediate.
    """
    vbsz = cache.bsz * cache.num_kv_heads
    torch.add(
        cache.indices_buf_3d.view(vbsz, cache.page_budget),
        cache.head_offset,
        out=cache.indices_flat_buf.view(vbsz, cache.page_budget),
    )


def upstream_flashinfer_decode_attention(
    query_states: torch.Tensor,   # (B, num_qo_heads, 1, head_dim) bf16
    cache: UpstreamFlashInferPagedKVCache,
    layer_idx: int,
) -> torch.Tensor:
    """Run upstream FlashInfer paged decode with the virtual-batch-per-(b,h)
    layout.

    Q reshape: `(B, num_qo_heads, 1, head_dim)` → `(B, H, group_size, d)` →
    `(B*H, group_size, d)`. FlashInfer interprets this as `B*H` batch entries,
    each carrying `group_size` Q heads against its own 1 KV head over its
    own slice of indices.

    Returns `(B, num_qo_heads, 1, head_dim)` to match the SDPA pre-transpose
    convention (same as `flashinfer_decode_attention` in the fork adapter).

    IMPORTANT: `refresh_upstream_indices_flat(cache)` must be called before
    this function, after the Stage 5 kernel has written fresh head-local
    indices.
    """
    bsz = cache.bsz
    H = cache.num_kv_heads
    gs = cache.group_size
    d = cache.head_dim

    # Two-stage reshape encodes vbatch ordering v = b*H + h.
    # `.contiguous()` is defensive (Critic G): if `query_states` arrives
    # non-contiguous (e.g. after a transpose), FI would otherwise silently
    # see strided q. Negligible cost vs the FI run.
    q_flat = (
        query_states
        .reshape(bsz, H, gs, d)
        .reshape(bsz * H, gs, d)
        .contiguous()
    )

    # Optional bsz=1 parity gate (Critic B). At bsz=1, the new chain reduces
    # to today's (H, gs, d) reshape — bit-equality is the gate. Gated by env;
    # remove after one clean run.
    if bsz == 1 and os.environ.get("OMC_FI_PARITY_GATE") == "1":
        q_old = query_states.reshape(H, gs, d).contiguous()
        max_diff = (q_flat - q_old).abs().max().item()
        assert max_diff == 0.0, (
            f"[parity gate] Q reshape mismatch at bsz=1: max_diff={max_diff}"
        )

    out = cache.wrapper.run(q_flat, cache.buf[layer_idx])  # (B*H, gs, d)
    return out.view(bsz, H * gs, 1, d)
