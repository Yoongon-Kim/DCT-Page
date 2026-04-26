"""
Quest / FlashInfer paged decode adapter for DCT-Page.

Bridges DCT-Page's decode path into Quest's `BatchDecodeWithPagedKVCacheWrapper`.
Scope: `profile_decode.py` only, drop mode, bsz=1.

Precision: Quest's kernel dispatches fp16 only (adding bf16 requires upstream
surgery in `decode_page.cuh` and RAFT vectorized.cuh, out of scope for
Phase 1). We therefore hold the Quest KV cache in fp16 and cast query states
to fp16 per decode step; the fp16 attention output is cast back to bf16 to
match DCT-Page's dtype. Precision drift from the cast is tiny for Llama and
verified empirically via `--verify_quest`.

Quest's paged KV layout (NHD): (num_layers, capacity_pages, 2, page_size,
num_kv_heads, head_dim). DCT-Page's PreAllocatedLayer stores flat (1,
num_kv_heads, alloc_len, head_dim) per layer with separate K and V tensors.

Alignment note: DCT-Page's logical segmentation places the sink region at
tokens [0, sink_size), and "page 0" of the paged region starts at
token sink_size. This adapter reframes the whole cache as pages aligned with
token 0, so Quest's page 0 = tokens [0, page_size) covers both DCT's sink
and the first (page_size - sink_size) tokens of DCT's paged region. Quest
page (i + num_sink_pages) maps approximately to DCT's page i, with a within-
page token offset equal to (num_sink_pages * page_size - sink_size). For
sink_size=4, page_size=32, that offset is 28 tokens — acceptable for a
Phase-1 proof-of-concept; we quantify the drift empirically via --verify_quest.
"""
from __future__ import annotations

import sys
import types
from typing import Optional

import torch

# Pre-register quest_attn as a namespace package to bypass its __init__.py,
# which eagerly imports quest_attn.models (broken by transformers version drift).
_QA_DIR = "/home/yoongonkim/DCT-Page/baselines/quest_attn"
if "quest_attn" not in sys.modules:
    _stub = types.ModuleType("quest_attn")
    _stub.__path__ = [_QA_DIR]
    sys.modules["quest_attn"] = _stub
if "quest_attn.utils" not in sys.modules:
    _ustub = types.ModuleType("quest_attn.utils")
    _ustub.__path__ = [_QA_DIR + "/utils"]
    sys.modules["quest_attn.utils"] = _ustub

from quest_attn.utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper  # noqa: E402


class QuestPagedKVCache:
    """Quest NHD paged KV cache + wrapper state.

    Layout: buf[l, p, 0, t, h, d] = K, buf[l, p, 1, t, h, d] = V
    All layers share cur_seqlen / last_page_idx / last_page_len.

    Fixed-width indices: per-step `pack_indices` always produces exactly
    `num_sink_pages + top_k + num_recent_pages_fixed` entries per head.
    `begin_forward` is called exactly once at cache build time (planning the
    kernel workspace for this fixed count), which lets the whole decode path
    be captured into a CUDA graph without hitting `cudaMallocAsync`. See
    `pack_indices` for the in-place index writes that make this work.
    """

    def __init__(
        self,
        buf: torch.Tensor,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_qo_heads: int,
        num_layers: int,
        capacity_pages: int,
        dtype: torch.dtype,      # kernel dtype (fp16)
        model_dtype: torch.dtype,  # model / outer-world dtype (bf16)
        device: torch.device,
        num_sink_pages: int,
        top_k: int,
        num_recent_pages_fixed: int,
    ):
        self.buf = buf
        self.wrapper = wrapper
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_qo_heads = num_qo_heads
        self.num_layers = num_layers
        self.capacity_pages = capacity_pages
        self.dtype = dtype
        self.model_dtype = model_dtype
        self.device = device
        self.cur_seqlen = 0
        self.last_page_idx = -1  # becomes 0 on first append
        self.last_page_len = 0   # 0 means "no open page"; becomes 1..page_size as tokens accumulate

        # Fixed index layout: [sink | dct_middle | recent].
        self.num_sink_pages = num_sink_pages
        self.top_k = top_k
        self.num_recent_pages_fixed = num_recent_pages_fixed
        self.max_total_selected = num_sink_pages + top_k + num_recent_pages_fixed

        self.num_kv_groups = num_qo_heads // num_kv_heads

        # Preallocated scratch: one int32 tensor reused across steps.
        # Kernel requires paged_kv_indices.size(0) == num_qo_heads: each QO head
        # gets its own (possibly distinct) page list, though GQA kernel only
        # reads the first num_kv_heads rows. Sink columns are filled once in
        # prime_wrapper and never touched again; DCT middle and recent are
        # rewritten per step via in-place ops (no fresh allocations — required
        # for CUDA graph capture).
        self.packed_indices_buf = torch.empty(
            num_qo_heads, self.max_total_selected, dtype=torch.int32, device=device
        )
        # indptr is (2,) for bsz=1: fixed at [0, max_total_selected].
        self.indptr = torch.zeros(2, dtype=torch.int32, device=device)

        # Preallocated index templates for alloc-free pack_indices.
        # Sink IDs are [0, 1, ..., num_sink_pages - 1].
        self._sink_ids = torch.arange(
            num_sink_pages, dtype=torch.int32, device=device,
        )
        # Recent offset template: [-R, -R+1, ..., -1]. Adding last_page_idx
        # produces [last_page_idx - R, ..., last_page_idx - 1] — the explicit
        # recent pages strictly before the currently-open last page.
        self._recent_offsets = torch.arange(
            -num_recent_pages_fixed, 0, dtype=torch.int32, device=device,
        )

        # Output buffer for the attention result (fp16, cast back on exit).
        self.o_buf = torch.empty(1, num_qo_heads, head_dim, dtype=dtype, device=device)
        # Reusable query fp16 buffer (avoid allocating each decode step).
        self.q_fp16_buf = torch.empty(1, num_qo_heads, head_dim, dtype=dtype, device=device)

    def prime_wrapper(self):
        """One-time workspace plan + static index layout initialization.

        Plans the FlashInfer decode kernel for `max_total_selected` pages per
        KV head, pre-fills sink indices (they never change), and sets the
        indptr tensor. After this, every decode step can reuse the planned
        workspace — no `cudaMallocAsync`, so the decode forward is CUDA-graph
        capturable.
        """
        self.indptr[0] = 0
        self.indptr[1] = self.max_total_selected

        # Pre-fill sink columns (the same IDs [0..num_sink_pages) for every head).
        nkv = self.num_kv_heads
        if self.num_sink_pages > 0:
            self.packed_indices_buf[:nkv, :self.num_sink_pages] = self._sink_ids

        self.wrapper.begin_forward(
            self.indptr,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.dtype,
        )


def build_quest_paged_cache(
    preallocated_layers: list,
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
) -> QuestPagedKVCache:
    """One-time post-prefill conversion from DCT-Page's PreAllocatedLayer stack
    to Quest's NHD paged layout. Allocates enough pages for prefill + decode.

    preallocated_layers: list of PreAllocatedLayer (one per model layer). Each
    has .keys/.values of shape (1, num_kv_heads, alloc_len, head_dim).

    The returned cache is held in fp16 regardless of `dtype` (the DCT-Page
    model dtype). Quest's kernel only dispatches fp16, so we cast once at
    build time and keep a parallel fp16 cache.
    """
    # Capacity: prefill_pages + decode_pages + slack.
    prefill_pages = (prefill_len + page_size - 1) // page_size
    decode_pages = (max_decode_steps + page_size - 1) // page_size
    capacity_pages = prefill_pages + decode_pages + 4  # a little slack

    kernel_dtype = torch.float16  # Quest kernel dispatch restriction
    buf = torch.zeros(
        num_layers, capacity_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=kernel_dtype, device=device,
    )

    # Populate from each layer's flat (1, nkv, alloc_len, d) cache.
    for l, layer in enumerate(preallocated_layers):
        k = layer.keys[0, :, :prefill_len, :]   # (nkv, T, d)
        v = layer.values[0, :, :prefill_len, :]
        # Pad T up to a multiple of page_size, then reshape + permute.
        pad = prefill_pages * page_size - prefill_len
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        # (nkv, P*ps, d) -> (nkv, P, ps, d) -> (P, ps, nkv, d). Cast to fp16.
        k = k.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(kernel_dtype)
        v = v.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(kernel_dtype)
        buf[l, :prefill_pages, 0].copy_(k)
        buf[l, :prefill_pages, 1].copy_(v)

    wrapper = BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD")

    cache = QuestPagedKVCache(
        buf=buf,
        wrapper=wrapper,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_layers=num_layers,
        capacity_pages=capacity_pages,
        dtype=kernel_dtype,  # fp16 for kernel dispatch
        model_dtype=dtype,   # bf16 for I/O casting back
        device=device,
        num_sink_pages=num_sink_pages,
        top_k=top_k,
        num_recent_pages_fixed=num_recent_pages_fixed,
    )
    cache.cur_seqlen = prefill_len
    # last_page_idx: index of the page that holds the currently-open slot.
    # If prefill_len is a multiple of page_size, we've just closed page
    # (prefill_pages-1) and need to open a new one on next append. Encode
    # that as last_page_len=page_size (the open page is full).
    cache.last_page_idx = prefill_pages - 1
    if prefill_len == prefill_pages * page_size:
        cache.last_page_len = page_size  # page is full; next append opens a new one
    else:
        cache.last_page_len = prefill_len - (prefill_pages - 1) * page_size

    cache.prime_wrapper()
    return cache


def append_quest_cache(
    cache: QuestPagedKVCache,
    new_k: torch.Tensor,   # (1, num_kv_heads, 1, head_dim) bf16, post-RoPE
    new_v: torch.Tensor,
    layer_idx: int,
):
    """In-place append of one decode step's K/V into Quest's paged layout.

    Only layer 0 advances cur_seqlen / last_page_* counters; other layers
    write to the same slot. The attention forward must call this BEFORE
    calling quest_decode_attention for that layer (so the new token is
    visible).
    """
    if layer_idx == 0:
        # Advance position
        if cache.last_page_len == cache.page_size:
            cache.last_page_idx += 1
            cache.last_page_len = 0
        cache.last_page_len += 1
        cache.cur_seqlen += 1

    slot = cache.last_page_len - 1
    # new_k shape: (1, nkv, 1, d) -> (nkv, d). Cast bf16 -> fp16 on copy.
    k_flat = new_k.view(cache.num_kv_heads, cache.head_dim)
    v_flat = new_v.view(cache.num_kv_heads, cache.head_dim)
    # copy_ handles dtype conversion automatically.
    cache.buf[layer_idx, cache.last_page_idx, 0, slot].copy_(k_flat)
    cache.buf[layer_idx, cache.last_page_idx, 1, slot].copy_(v_flat)


def pack_indices(
    cache: QuestPagedKVCache,
    dct_selected: torch.Tensor,  # (num_kv_heads, top_k) int32, in DCT middle-page space
    num_sink_pages: int = None,         # ignored — cache carries the fixed count
    num_recent_pages_explicit: int = None,  # ignored — cache carries the fixed count
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write the [sink | dct_middle | recent] indices into the preallocated
    packed buffer using only in-place ops (no fresh allocations). Width is
    always `cache.max_total_selected`, so the whole op sequence is CUDA-graph
    capturable.

    Layout per KV head row:
      cols [0, num_sink_pages)                             : sink page IDs (static, pre-filled)
      cols [num_sink_pages, num_sink_pages + top_k)        : dct_selected + num_sink_pages
      cols [.. + top_k, .. + top_k + num_recent_pages_fixed) : last_page_idx + recent_offsets

    The currently-open last page is still attended via the kernel's
    `paged_kv_last_page_idx` / `paged_kv_last_page_len` path and must NOT
    appear in this list. `num_recent_pages_fixed` therefore counts only full
    pages strictly before `last_page_idx` — and is held constant across
    steps (using the max over the decode cycle, so the recent window
    overshoots by ≤1 page's worth of tokens on the partial-page steps).

    `num_sink_pages` / `num_recent_pages_explicit` args are kept for
    call-site stability (the previous variable-width API) but are ignored;
    the build-time fixed counts on `cache` are authoritative.
    """
    nkv = cache.num_kv_heads
    top_k = dct_selected.shape[-1]
    assert top_k == cache.top_k, (
        f"DCT top_k changed ({top_k}) vs cache-fixed ({cache.top_k}); "
        f"fixed-width packer requires constant top_k across steps."
    )
    out = cache.packed_indices_buf  # (num_qo_heads, max_total_selected)

    # Sink cols [0, num_sink_pages) are static — filled once in prime_wrapper.

    # DCT middle: out[:nkv, mid_start:mid_end] = dct_selected + num_sink_pages
    mid_start = cache.num_sink_pages
    mid_end = mid_start + top_k
    out_mid = out[:nkv, mid_start:mid_end]
    out_mid.copy_(dct_selected)
    if cache.num_sink_pages != 0:
        out_mid.add_(cache.num_sink_pages)

    # Recent: out[:nkv, rs:re] = last_page_idx + recent_offsets
    # recent_offsets = [-R, -R+1, ..., -1] so the resulting cols are
    # [last_page_idx - R, ..., last_page_idx - 1].
    if cache.num_recent_pages_fixed > 0:
        rs = mid_end
        re = rs + cache.num_recent_pages_fixed
        out_rec = out[:nkv, rs:re]
        out_rec.fill_(cache.last_page_idx)
        out_rec.add_(cache._recent_offsets)

    # indptr is fixed at [0, max_total_selected] — set once in prime_wrapper,
    # never touched per-step. `out` is the full buffer (contiguous).
    return out, cache.indptr


def quest_decode_attention(
    query_states: torch.Tensor,  # (1, num_qo_heads, 1, head_dim)
    cache: QuestPagedKVCache,
    packed_indices: torch.Tensor,
    indptr: torch.Tensor,
    layer_idx: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
) -> torch.Tensor:
    """Run Quest's paged decode attention on the selected pages.

    Returns attention output shaped (1, num_qo_heads, 1, head_dim) in the
    model dtype (bf16). Internally casts query to fp16, runs the fp16 kernel,
    and casts the output back to bf16.
    """
    # Wrapper expects q as (1, num_qo_heads, head_dim). query_states is
    # (1, num_qo_heads, 1, head_dim) and usually non-contiguous after the
    # transpose in the forward; reshape handles that. Cast bf16 -> fp16.
    q_bf16 = query_states.reshape(1, cache.num_qo_heads, cache.head_dim)
    cache.q_fp16_buf.copy_(q_bf16)
    q = cache.q_fp16_buf
    o = cache.o_buf  # (1, num_qo_heads, head_dim) fp16

    # begin_forward was called once in build_quest_paged_cache (workspace
    # + partition info planned for the fixed max_total_selected). Because
    # pack_indices writes exactly max_total_selected entries and indptr
    # stays at [0, max_total_selected], the plan is always current — no
    # per-step replan, no cudaMallocAsync inside the captured region.
    cache.wrapper.forward(
        q, o, cache.buf[layer_idx],
        packed_indices, indptr,
        cache.last_page_len, cache.last_page_idx,
        rope_scale, rope_theta,
    )

    # Reshape + cast fp16 -> bf16 back to SDPA output layout.
    return o.view(1, cache.num_qo_heads, 1, cache.head_dim).to(cache.model_dtype)
