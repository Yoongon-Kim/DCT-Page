"""
Fused Triton kernels for DCT Page Attention (v2).

Kernel 1: _score_pages_fused_kernel  — scores all pages
Kernel 2: _assemble_kv_full_kernel   — assemble + sink/recent + K-RoPE (topk done by torch.topk)
Kernel 3: _apply_rope_q_kernel       — RoPE for single decode query token

Total decode-path launches: Score(1) + torch.topk(1) + Assemble(1) + Q-RoPE(1) + SDPA(1) = 5
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Page scoring (kept from v1)
# ---------------------------------------------------------------------------
@triton.jit
def _score_pages_fused_kernel(
    # --- pointers ---
    query_ptr,         # [bsz, num_kv_heads, num_kv_groups, head_dim]
    comp_keys_ptr,     # [bsz, num_kv_heads, num_pages, comp_size, head_dim]
    out_scores_ptr,    # [bsz, num_kv_heads, num_pages]
    # --- query strides ---
    q_stride_b, q_stride_h, q_stride_g,
    # --- comp keys strides ---
    ck_stride_b, ck_stride_h, ck_stride_p, ck_stride_c,
    # --- output strides ---
    os_stride_b, os_stride_h,
    # --- runtime dims ---
    num_kv_heads,
    num_pages,
    head_dim,
    scaling,
    # --- compile-time constants ---
    NUM_KV_GROUPS: tl.constexpr,
    COMP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    SCORING_METHOD: tl.constexpr,   # 0=max, 1=mean, 2=sum
    GROUP_AGG_METHOD: tl.constexpr, # 0=mean, 1=max
):
    """One program per (batch, kv_head, page_tile)."""
    pid_bh = tl.program_id(0)
    pid_pt = tl.program_id(1)

    h = pid_bh % num_kv_heads
    b = pid_bh // num_kv_heads

    p_start = pid_pt * BLOCK_P
    p_offsets = tl.arange(0, BLOCK_P)
    p_indices = p_start + p_offsets
    p_mask = p_indices < num_pages

    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < head_dim

    q_base = query_ptr + b * q_stride_b + h * q_stride_h
    ck_base = comp_keys_ptr + b * ck_stride_b + h * ck_stride_h

    g_idx = tl.arange(0, NUM_KV_GROUPS)[:, None]

    if SCORING_METHOD == 0:
        group_ps = tl.full([NUM_KV_GROUPS, BLOCK_P], float('-inf'), dtype=tl.float32)
    else:
        group_ps = tl.zeros([NUM_KV_GROUPS, BLOCK_P], dtype=tl.float32)

    for c in range(COMP_SIZE):
        k_ptrs = ck_base + p_indices[:, None] * ck_stride_p + c * ck_stride_c + d_idx[None, :]
        mask_2d = p_mask[:, None] & d_mask[None, :]
        k = tl.load(k_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

        for g in range(NUM_KV_GROUPS):
            q = tl.load(q_base + g * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32)
            q = q * scaling
            dots = tl.sum(k * q[None, :], axis=1)
            g_mask = (g_idx == g)
            if SCORING_METHOD == 0:
                group_ps = tl.where(g_mask, tl.maximum(group_ps, dots[None, :]), group_ps)
            else:
                group_ps = tl.where(g_mask, group_ps + dots[None, :], group_ps)

    if SCORING_METHOD == 1:
        group_ps = group_ps / COMP_SIZE

    if GROUP_AGG_METHOD == 0:
        agg_scores = tl.sum(group_ps, axis=0) / NUM_KV_GROUPS
    else:
        agg_scores = tl.max(group_ps, axis=0)

    out_ptrs = out_scores_ptr + b * os_stride_b + h * os_stride_h + p_indices
    tl.store(out_ptrs, agg_scores, mask=p_mask)


def score_pages_triton(
    query_states: torch.Tensor,
    compressed_keys: torch.Tensor,
    scoring_method: str,
    group_agg_method: str,
    num_kv_groups: int,
) -> torch.Tensor:
    """Score all pages. Returns page_scores only (topk deferred to assemble kernel).

    Args:
        query_states:    [bsz, num_heads, 1, head_dim]
        compressed_keys: [bsz, num_kv_heads, num_pages, comp_size, head_dim]

    Returns:
        page_scores: [bsz, num_kv_heads, num_pages] (float32)
    """
    bsz, _num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, num_pages, comp_size, _ = compressed_keys.shape
    scaling = head_dim ** -0.5

    assert q_len == 1, "score_pages_triton only supports decode (q_len=1)"

    query = query_states.squeeze(2).reshape(bsz, num_kv_heads, num_kv_groups, head_dim).contiguous()
    compressed_keys = compressed_keys.contiguous()

    page_scores = torch.empty(
        bsz, num_kv_heads, num_pages,
        dtype=torch.float32, device=query.device,
    )

    SCORING = {"max": 0, "mean": 1, "sum": 2}[scoring_method]
    GROUP_AGG = {"mean": 0, "max": 1}[group_agg_method]
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_P = 32

    num_page_tiles = (num_pages + BLOCK_P - 1) // BLOCK_P
    grid = (bsz * num_kv_heads, num_page_tiles)

    # Ensure correct CUDA device context for Triton kernel launch (multi-GPU)
    with torch.cuda.device(query.device):
        _score_pages_fused_kernel[grid](
            query, compressed_keys, page_scores,
            query.stride(0), query.stride(1), query.stride(2),
            compressed_keys.stride(0), compressed_keys.stride(1),
            compressed_keys.stride(2), compressed_keys.stride(3),
            page_scores.stride(0), page_scores.stride(1),
            num_kv_heads, num_pages, head_dim,
            scaling,
            NUM_KV_GROUPS=num_kv_groups,
            COMP_SIZE=comp_size,
            BLOCK_D=BLOCK_D,
            BLOCK_P=BLOCK_P,
            SCORING_METHOD=SCORING,
            GROUP_AGG_METHOD=GROUP_AGG,
        )

    return page_scores


# ---------------------------------------------------------------------------
# Kernel 2: Fused Assemble + K-RoPE + sink/recent
# ---------------------------------------------------------------------------
@triton.jit
def _assemble_kv_full_kernel(
    # --- K/V source pointers ---
    paged_k_ptr, comp_k_ptr, sink_k_ptr, recent_k_ptr,
    paged_v_ptr, comp_v_ptr, sink_v_ptr, recent_v_ptr,
    # --- output pointers ---
    out_k_ptr, out_v_ptr,
    # --- RoPE tables ---
    cos_ptr, sin_ptr,
    # --- pre-sorted selected page indices ---
    sel_indices_ptr,
    # --- strides for paged [bsz, kv_heads, num_pages, page_size, head_dim] ---
    paged_stride_b, paged_stride_h, paged_stride_p, paged_stride_t,
    # --- strides for comp [bsz, kv_heads, num_pages, comp_size, head_dim] ---
    comp_stride_b, comp_stride_h, comp_stride_p, comp_stride_t,
    # --- strides for sink [bsz, kv_heads, sink_len, head_dim] ---
    sink_stride_b, sink_stride_h, sink_stride_t,
    # --- strides for recent [bsz, kv_heads, recent_len, head_dim] ---
    recent_stride_b, recent_stride_h, recent_stride_t,
    # --- strides for output [bsz, kv_heads, total_len, head_dim] ---
    out_stride_b, out_stride_h, out_stride_t,
    # --- strides for cos/sin [total_len, head_dim] ---
    rope_stride_t,
    # --- strides for sel_indices [bsz, kv_heads, top_k] ---
    si_stride_b, si_stride_h,
    # --- runtime dims ---
    num_kv_heads,
    num_pages,
    head_dim,
    sink_len,
    recent_len,
    total_len,
    # --- compile-time constants ---
    PAGE_SIZE: tl.constexpr,
    COMP_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    APPLY_ROPE: tl.constexpr,   # 1 = apply K-RoPE, 0 = straight copy
):
    """One program per (batch, kv_head, segment, token-tile).

    Segments: 0=sink, 1..num_pages=pages, num_pages+1=recent.
    Each program:
      1. Loads pre-sorted selected_indices (TOP_K int32s) from L2
      2. Copies a [BLOCK_T, head_dim] tile from the correct source
      3. For K: applies RoPE during store (if APPLY_ROPE)
      4. For V: straight copy
    """
    pid_bhs = tl.program_id(0)   # (batch, head, segment)
    pid_tile = tl.program_id(1)  # token tile within segment

    num_segments = num_pages + 2
    seg = pid_bhs % num_segments
    tmp = pid_bhs // num_segments
    h = tmp % num_kv_heads
    b = tmp // num_kv_heads

    # ---- Load pre-sorted selected indices (tiny: TOP_K int32s) ----
    si_base = sel_indices_ptr + b * si_stride_b + h * si_stride_h
    k_idx = tl.arange(0, TOP_K)
    sel_indices = tl.load(si_base + k_idx).to(tl.int32)

    # ---- Determine segment type and source ----
    is_sink = (seg == 0)
    is_recent = (seg == num_pages + 1)
    page_idx = seg - 1  # only valid when not sink/recent

    # Check if this page is selected (linear scan over TOP_K elements)
    is_selected = tl.sum((sel_indices == page_idx).to(tl.int32)) > 0
    count_before = tl.sum((sel_indices < page_idx).to(tl.int32))

    # Token count and write offset for this segment
    # Sink: sink_len tokens, starts at 0
    # Page p: page_size (selected) or comp_size (unselected)
    #         write_start = sink_len + p * comp_size + count_selected_before * (page_size - comp_size)
    # Recent: recent_len tokens, starts at total_len - recent_len

    if is_sink:
        num_tokens = sink_len
        write_start = 0
    elif is_recent:
        num_tokens = recent_len
        write_start = total_len - recent_len
    else:
        num_tokens = tl.where(is_selected, PAGE_SIZE, COMP_SIZE)
        write_start = sink_len + page_idx * COMP_SIZE + count_before * (PAGE_SIZE - COMP_SIZE)

    # Early exit for tiles beyond this segment's token count
    t_start = pid_tile * BLOCK_T
    if t_start >= num_tokens:
        return

    # ---- 2D tile indices ----
    t_idx = tl.arange(0, BLOCK_T)
    d_idx = tl.arange(0, BLOCK_D)
    mask = ((t_start + t_idx) < num_tokens)[:, None] & (d_idx < head_dim)[None, :]

    # ---- Destination offsets ----
    dst_offsets = (write_start + t_start + t_idx)[:, None] * out_stride_t + d_idx[None, :]
    k_dst = out_k_ptr + b * out_stride_b + h * out_stride_h
    v_dst = out_v_ptr + b * out_stride_b + h * out_stride_h

    # ---- RoPE setup (precompute outside branches) ----
    if APPLY_ROPE == 1:
        out_pos = (write_start + t_start + t_idx)  # [BLOCK_T]
        half_d = head_dim // 2
        cos_offsets = out_pos[:, None] * rope_stride_t + d_idx[None, :]
        cos_vals = tl.load(cos_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
        d_rot = tl.where(d_idx < half_d, d_idx + half_d, d_idx - half_d)
        rot_sign = tl.where(d_idx[None, :] < half_d, -1.0, 1.0)

    # ---- Load from source, apply K-RoPE, store ----
    if is_sink:
        sk_base = sink_k_ptr + b * sink_stride_b + h * sink_stride_h
        sv_base = sink_v_ptr + b * sink_stride_b + h * sink_stride_h
        src_offsets = (t_start + t_idx)[:, None] * sink_stride_t + d_idx[None, :]
        k_vals = tl.load(sk_base + src_offsets, mask=mask, other=0.0)
        v_vals = tl.load(sv_base + src_offsets, mask=mask, other=0.0)
        if APPLY_ROPE == 1:
            rot_offsets = (t_start + t_idx)[:, None] * sink_stride_t + d_rot[None, :]
            k_rot_raw = tl.load(sk_base + rot_offsets, mask=mask, other=0.0)
            k_rotated = rot_sign * k_rot_raw.to(tl.float32)
            k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)
    elif is_recent:
        rk_base = recent_k_ptr + b * recent_stride_b + h * recent_stride_h
        rv_base = recent_v_ptr + b * recent_stride_b + h * recent_stride_h
        src_offsets = (t_start + t_idx)[:, None] * recent_stride_t + d_idx[None, :]
        k_vals = tl.load(rk_base + src_offsets, mask=mask, other=0.0)
        v_vals = tl.load(rv_base + src_offsets, mask=mask, other=0.0)
        if APPLY_ROPE == 1:
            rot_offsets = (t_start + t_idx)[:, None] * recent_stride_t + d_rot[None, :]
            k_rot_raw = tl.load(rk_base + rot_offsets, mask=mask, other=0.0)
            k_rotated = rot_sign * k_rot_raw.to(tl.float32)
            k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)
    elif is_selected:
        src_off = (b * paged_stride_b + h * paged_stride_h
                   + page_idx * paged_stride_p + t_start * paged_stride_t)
        src_offsets = t_idx[:, None] * paged_stride_t + d_idx[None, :]
        k_vals = tl.load(paged_k_ptr + src_off + src_offsets, mask=mask, other=0.0)
        v_vals = tl.load(paged_v_ptr + src_off + src_offsets, mask=mask, other=0.0)
        if APPLY_ROPE == 1:
            rot_offsets = t_idx[:, None] * paged_stride_t + d_rot[None, :]
            k_rot_raw = tl.load(paged_k_ptr + src_off + rot_offsets, mask=mask, other=0.0)
            k_rotated = rot_sign * k_rot_raw.to(tl.float32)
            k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)
    else:
        src_off = (b * comp_stride_b + h * comp_stride_h
                   + page_idx * comp_stride_p + t_start * comp_stride_t)
        src_offsets = t_idx[:, None] * comp_stride_t + d_idx[None, :]
        k_vals = tl.load(comp_k_ptr + src_off + src_offsets, mask=mask, other=0.0)
        v_vals = tl.load(comp_v_ptr + src_off + src_offsets, mask=mask, other=0.0)
        if APPLY_ROPE == 1:
            rot_offsets = t_idx[:, None] * comp_stride_t + d_rot[None, :]
            k_rot_raw = tl.load(comp_k_ptr + src_off + rot_offsets, mask=mask, other=0.0)
            k_rotated = rot_sign * k_rot_raw.to(tl.float32)
            k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)

    # Store K (with RoPE if applied above) and V (always straight copy)
    tl.store(k_dst + dst_offsets, k_vals, mask=mask)
    tl.store(v_dst + dst_offsets, v_vals, mask=mask)


def assemble_kv_full_triton(
    paged_k: torch.Tensor,
    paged_v: torch.Tensor,
    comp_k: torch.Tensor,
    comp_v: torch.Tensor,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    selected_indices: torch.Tensor,
    cos_table: torch.Tensor = None,
    sin_table: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused assemble: sink/middle/recent copy + K-RoPE in one kernel.

    TopK is done externally via torch.topk before calling this function.

    Args:
        paged_k/v:        [bsz, num_kv_heads, num_pages, page_size, head_dim]
        comp_k/v:         [bsz, num_kv_heads, num_pages, comp_size, head_dim]
        sink_k/v:         [bsz, num_kv_heads, sink_len, head_dim]
        recent_k/v:       [bsz, num_kv_heads, recent_len, head_dim]
        selected_indices: [bsz, num_kv_heads, top_k] (int64, sorted ascending)
        cos_table:        [total_len, head_dim] (optional, for K-RoPE)
        sin_table:        [total_len, head_dim] (optional, for K-RoPE)

    Returns:
        final_k, final_v: [bsz, num_kv_heads, total_len, head_dim]
    """
    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    comp_size = comp_k.shape[3]
    sink_len = sink_k.shape[2]
    recent_len = recent_k.shape[2]
    top_k = selected_indices.shape[2]
    num_unselected = num_pages - top_k
    middle_len = top_k * page_size + num_unselected * comp_size
    total_len = sink_len + middle_len + recent_len

    # Ensure contiguous (DCT compression can produce transposed tensors)
    if not comp_k.is_contiguous():
        comp_k = comp_k.contiguous()
    if not comp_v.is_contiguous():
        comp_v = comp_v.contiguous()

    # Ensure int32 for Triton (torch.topk returns int64)
    sel_idx = selected_indices.to(torch.int32).contiguous()

    apply_rope = cos_table is not None and sin_table is not None

    # Allocate output
    final_k = torch.empty(bsz, num_kv_heads, total_len, head_dim,
                           dtype=paged_k.dtype, device=paged_k.device)
    final_v = torch.empty_like(final_k)

    # --- Diagnostic: time wrapper vs kernel ---
    _diag = getattr(assemble_kv_full_triton, '_diag', False)
    if _diag:
        _ev_pre = torch.cuda.Event(enable_timing=True)
        _ev_post = torch.cuda.Event(enable_timing=True)
        _ev_pre.record()

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_T = 32

    # Segments: sink(1) + pages(num_pages) + recent(1)
    num_segments = num_pages + 2
    max_seg_len = max(sink_len, page_size, recent_len)
    num_tiles = (max_seg_len + BLOCK_T - 1) // BLOCK_T
    grid = (bsz * num_kv_heads * num_segments, num_tiles)

    # RoPE strides (or dummy if not applying)
    if apply_rope:
        cos_table = cos_table.contiguous()
        sin_table = sin_table.contiguous()
        rope_stride_t = cos_table.stride(0)
        cos_ptr = cos_table
        sin_ptr = sin_table
    else:
        # Pass dummy pointers and stride
        cos_ptr = final_k  # won't be accessed
        sin_ptr = final_k
        rope_stride_t = 0

    # Ensure correct CUDA device context for Triton kernel launch (multi-GPU)
    with torch.cuda.device(paged_k.device):
        _assemble_kv_full_kernel[grid](
            paged_k, comp_k, sink_k, recent_k,
            paged_v, comp_v, sink_v, recent_v,
            final_k, final_v,
            cos_ptr, sin_ptr,
            sel_idx,
            # paged strides
            paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3),
            # comp strides
            comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3),
            # sink strides
            sink_k.stride(0), sink_k.stride(1), sink_k.stride(2),
            # recent strides
            recent_k.stride(0), recent_k.stride(1), recent_k.stride(2),
            # output strides
            final_k.stride(0), final_k.stride(1), final_k.stride(2),
            # rope stride
            rope_stride_t,
            # sel_indices strides
            sel_idx.stride(0), sel_idx.stride(1),
            # runtime dims
            num_kv_heads, num_pages, head_dim,
            sink_len, recent_len, total_len,
            # constexprs
            PAGE_SIZE=page_size, COMP_SIZE=comp_size, TOP_K=top_k,
            BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T,
            APPLY_ROPE=1 if apply_rope else 0,
        )

    if _diag:
        _ev_post.record()
        # Deferred: don't synchronize here — collect events for later
        _diag_events = getattr(assemble_kv_full_triton, '_diag_events', [])
        _diag_events.append((_ev_pre, _ev_post))
        assemble_kv_full_triton._diag_events = _diag_events

    return final_k, final_v


# ---------------------------------------------------------------------------
# Kernel 2b: Copy sink + selected pages + recent (pure Triton, no PyTorch mixing)
# ---------------------------------------------------------------------------
@triton.jit
def _copy_full_segments_kernel(
    # --- K/V source pointers ---
    paged_k_ptr, paged_v_ptr,
    sink_k_ptr, sink_v_ptr,
    recent_k_ptr, recent_v_ptr,
    # --- output pointers ---
    out_k_ptr, out_v_ptr,
    # --- RoPE tables ---
    cos_ptr, sin_ptr,
    # --- pre-sorted selected page indices ---
    sel_indices_ptr,
    # --- strides for paged [bsz, kv_heads, num_pages, page_size, head_dim] ---
    paged_stride_b, paged_stride_h, paged_stride_p, paged_stride_t,
    # --- strides for sink [bsz, kv_heads, sink_len, head_dim] ---
    sink_stride_b, sink_stride_h, sink_stride_t,
    # --- strides for recent [bsz, kv_heads, recent_len, head_dim] ---
    recent_stride_b, recent_stride_h, recent_stride_t,
    # --- strides for output [bsz, kv_heads, total_len, head_dim] ---
    out_stride_b, out_stride_h, out_stride_t,
    # --- strides for cos/sin [total_len, head_dim] ---
    rope_stride_t,
    # --- strides for sel_indices [bsz, kv_heads, top_k] ---
    si_stride_b, si_stride_h,
    # --- runtime dims ---
    num_kv_heads,
    top_k,
    head_dim,
    sink_len,
    recent_len,
    total_len,
    # --- compile-time constants ---
    COMP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    """Copy sink + selected pages + recent to output with optional K-RoPE.

    Grid: (bsz * kv_heads * (top_k + 2), tiles_per_full_segment)
    Work items: 0=sink, 1..top_k=selected pages, top_k+1=recent.
    Rank for selected pages is directly (work_id - 1) — no selected_indices scan.
    """
    pid_bhw = tl.program_id(0)
    pid_tile = tl.program_id(1)

    num_work_items = top_k + 2
    w = pid_bhw % num_work_items
    tmp = pid_bhw // num_work_items
    h = tmp % num_kv_heads
    b = tmp // num_kv_heads

    is_sink = (w == 0)
    is_recent = (w == top_k + 1)

    # Load selected page index (1 int32). Clamped rank is safe for all work items.
    rank = w - 1
    safe_rank = tl.maximum(tl.minimum(rank, top_k - 1), 0)
    page_idx = tl.load(sel_indices_ptr + b * si_stride_b + h * si_stride_h + safe_rank).to(tl.int32)

    # Determine token count and write position
    if is_sink:
        num_tokens = sink_len
        write_start = 0
    elif is_recent:
        num_tokens = recent_len
        write_start = total_len - recent_len
    else:
        num_tokens = PAGE_SIZE
        write_start = sink_len + page_idx * COMP_SIZE + rank * (PAGE_SIZE - COMP_SIZE)

    # Early exit for tiles beyond this segment's token count
    t_start = pid_tile * BLOCK_T
    if t_start >= num_tokens:
        return

    # ---- 2D tile indices ----
    t_idx = tl.arange(0, BLOCK_T)
    d_idx = tl.arange(0, BLOCK_D)
    mask = ((t_start + t_idx) < num_tokens)[:, None] & (d_idx < head_dim)[None, :]

    # ---- Destination offsets ----
    dst_offsets = (write_start + t_start + t_idx)[:, None] * out_stride_t + d_idx[None, :]
    k_dst = out_k_ptr + b * out_stride_b + h * out_stride_h
    v_dst = out_v_ptr + b * out_stride_b + h * out_stride_h

    # ---- RoPE setup (precompute outside branches) ----
    if APPLY_ROPE == 1:
        out_pos = (write_start + t_start + t_idx)
        half_d = head_dim // 2
        cos_offsets = out_pos[:, None] * rope_stride_t + d_idx[None, :]
        cos_vals = tl.load(cos_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
        d_rot = tl.where(d_idx < half_d, d_idx + half_d, d_idx - half_d)
        rot_sign = tl.where(d_idx[None, :] < half_d, -1.0, 1.0)

    # ---- Load from source, apply K-RoPE, store ----
    if is_sink:
        sk_base = sink_k_ptr + b * sink_stride_b + h * sink_stride_h
        sv_base = sink_v_ptr + b * sink_stride_b + h * sink_stride_h
        src_offsets = (t_start + t_idx)[:, None] * sink_stride_t + d_idx[None, :]
        k_vals = tl.load(sk_base + src_offsets, mask=mask, other=0.0)
        v_vals = tl.load(sv_base + src_offsets, mask=mask, other=0.0)
        if APPLY_ROPE == 1:
            rot_offsets = (t_start + t_idx)[:, None] * sink_stride_t + d_rot[None, :]
            k_rot_raw = tl.load(sk_base + rot_offsets, mask=mask, other=0.0)
            k_rotated = rot_sign * k_rot_raw.to(tl.float32)
            k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)
    elif is_recent:
        rk_base = recent_k_ptr + b * recent_stride_b + h * recent_stride_h
        rv_base = recent_v_ptr + b * recent_stride_b + h * recent_stride_h
        src_offsets = (t_start + t_idx)[:, None] * recent_stride_t + d_idx[None, :]
        k_vals = tl.load(rk_base + src_offsets, mask=mask, other=0.0)
        v_vals = tl.load(rv_base + src_offsets, mask=mask, other=0.0)
        if APPLY_ROPE == 1:
            rot_offsets = (t_start + t_idx)[:, None] * recent_stride_t + d_rot[None, :]
            k_rot_raw = tl.load(rk_base + rot_offsets, mask=mask, other=0.0)
            k_rotated = rot_sign * k_rot_raw.to(tl.float32)
            k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)
    else:
        src_off = (b * paged_stride_b + h * paged_stride_h
                   + page_idx * paged_stride_p + t_start * paged_stride_t)
        src_offsets = t_idx[:, None] * paged_stride_t + d_idx[None, :]
        k_vals = tl.load(paged_k_ptr + src_off + src_offsets, mask=mask, other=0.0)
        v_vals = tl.load(paged_v_ptr + src_off + src_offsets, mask=mask, other=0.0)
        if APPLY_ROPE == 1:
            rot_offsets = t_idx[:, None] * paged_stride_t + d_rot[None, :]
            k_rot_raw = tl.load(paged_k_ptr + src_off + rot_offsets, mask=mask, other=0.0)
            k_rotated = rot_sign * k_rot_raw.to(tl.float32)
            k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)

    # Store K (with RoPE if applied above) and V (always straight copy)
    tl.store(k_dst + dst_offsets, k_vals, mask=mask)
    tl.store(v_dst + dst_offsets, v_vals, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2c: Copy unselected compressed pages (in-kernel detection + offset)
# ---------------------------------------------------------------------------
@triton.jit
def _copy_unselected_pages_kernel(
    # --- source pointers ---
    comp_k_ptr, comp_v_ptr,
    # --- output pointers ---
    out_k_ptr, out_v_ptr,
    # --- RoPE tables ---
    cos_ptr, sin_ptr,
    # --- selected indices [bsz, kv_heads, top_k] (sorted) ---
    sel_indices_ptr,
    # --- strides for comp [bsz, kv_heads, num_pages, comp_size, head_dim] ---
    comp_stride_b, comp_stride_h, comp_stride_p, comp_stride_t,
    # --- strides for output [bsz, kv_heads, total_len, head_dim] ---
    out_stride_b, out_stride_h, out_stride_t,
    # --- strides for cos/sin [total_len, head_dim] ---
    rope_stride_t,
    # --- strides for sel_indices [bsz, kv_heads, top_k] ---
    sel_stride_b, sel_stride_h,
    # --- runtime dims ---
    num_kv_heads, num_pages, head_dim, sink_len,
    # --- compile-time constants ---
    COMP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    """Copy unselected compressed pages to output with optional K-RoPE.
    Grid: (bsz * kv_heads * num_pages,)  — selected pages skip immediately.
    count_before computed in-kernel via loop over TOP_K (fully unrolled).
    """
    pid = tl.program_id(0)

    p = pid % num_pages
    tmp = pid // num_pages
    h = tmp % num_kv_heads
    b = tmp // num_kv_heads

    # Vectorized: load all TOP_K selected indices, check membership + count
    sel_base = sel_indices_ptr + b * sel_stride_b + h * sel_stride_h
    k_idx = tl.arange(0, TOP_K)
    sel_all = tl.load(sel_base + k_idx).to(tl.int32)
    is_selected = tl.sum((sel_all == p).to(tl.int32))
    count_before = tl.sum((sel_all < p).to(tl.int32))

    if is_selected > 0:
        return

    # Compute write offset in-kernel
    write_start = sink_len + p * COMP_SIZE + count_before * (PAGE_SIZE - COMP_SIZE)

    t_idx = tl.arange(0, BLOCK_T)
    d_idx = tl.arange(0, BLOCK_D)
    mask = (t_idx < COMP_SIZE)[:, None] & (d_idx < head_dim)[None, :]
    t_safe = tl.minimum(t_idx, COMP_SIZE - 1)

    # Source: comp_k[b, h, p, :comp_size, :]
    src_off = b * comp_stride_b + h * comp_stride_h + p * comp_stride_p
    src_offsets = t_safe[:, None] * comp_stride_t + d_idx[None, :]
    k_vals = tl.load(comp_k_ptr + src_off + src_offsets, mask=mask, other=0.0)
    v_vals = tl.load(comp_v_ptr + src_off + src_offsets, mask=mask, other=0.0)

    # Destination
    dst_offsets = (write_start + t_safe)[:, None] * out_stride_t + d_idx[None, :]
    k_dst = out_k_ptr + b * out_stride_b + h * out_stride_h
    v_dst = out_v_ptr + b * out_stride_b + h * out_stride_h

    # K-RoPE
    if APPLY_ROPE == 1:
        out_pos = write_start + t_safe
        half_d = head_dim // 2
        cos_offsets = out_pos[:, None] * rope_stride_t + d_idx[None, :]
        cos_vals = tl.load(cos_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
        d_rot = tl.where(d_idx < half_d, d_idx + half_d, d_idx - half_d)
        rot_sign = tl.where(d_idx[None, :] < half_d, -1.0, 1.0)
        rot_offsets = t_safe[:, None] * comp_stride_t + d_rot[None, :]
        k_rot_raw = tl.load(comp_k_ptr + src_off + rot_offsets, mask=mask, other=0.0)
        k_rotated = rot_sign * k_rot_raw.to(tl.float32)
        k_vals = (k_vals.to(tl.float32) * cos_vals + k_rotated * sin_vals).to(k_vals.dtype)

    tl.store(k_dst + dst_offsets, k_vals, mask=mask)
    tl.store(v_dst + dst_offsets, v_vals, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper: Split assemble (2 pure-Triton kernels, no PyTorch/Triton mixing)
# ---------------------------------------------------------------------------
def assemble_kv_split_triton(
    paged_k: torch.Tensor,
    paged_v: torch.Tensor,
    comp_k: torch.Tensor,
    comp_v: torch.Tensor,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    selected_indices: torch.Tensor,
    cos_table: torch.Tensor = None,
    sin_table: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble KV using split kernels. Same API as assemble_kv_full_triton.

    Kernel A: sink + selected pages + recent (all inside Triton)
    Kernel B: unselected compressed pages (branch-free)
    No PyTorch tensor ops between kernel launches.
    """
    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    comp_size = comp_k.shape[3]
    sink_len = sink_k.shape[2]
    recent_len = recent_k.shape[2]
    top_k = selected_indices.shape[2]
    num_unselected = num_pages - top_k
    middle_len = top_k * page_size + num_unselected * comp_size
    total_len = sink_len + middle_len + recent_len

    if not comp_k.is_contiguous():
        comp_k = comp_k.contiguous()
    if not comp_v.is_contiguous():
        comp_v = comp_v.contiguous()

    apply_rope = cos_table is not None and sin_table is not None
    device = paged_k.device
    dtype = paged_k.dtype

    # Allocate output
    final_k = torch.empty(bsz, num_kv_heads, total_len, head_dim, dtype=dtype, device=device)
    final_v = torch.empty_like(final_k)

    # Int32 indices for Triton
    sel_idx = selected_indices.to(torch.int32).contiguous()

    # RoPE setup
    BLOCK_D = triton.next_power_of_2(head_dim)
    if apply_rope:
        cos_table = cos_table.contiguous()
        sin_table = sin_table.contiguous()
        rope_stride_t = cos_table.stride(0)
        cos_ptr = cos_table
        sin_ptr = sin_table
    else:
        cos_ptr = final_k  # won't be accessed
        sin_ptr = final_k
        rope_stride_t = 0

    # Ensure correct CUDA device context for Triton kernel launch (multi-GPU)
    with torch.cuda.device(device):
        # ---- Kernel A: sink + selected + recent ----
        BLOCK_T_FULL = 32
        max_seg_len = max(sink_len, page_size, recent_len)
        tiles_per_seg = (max_seg_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
        grid_full = (bsz * num_kv_heads * (top_k + 2), tiles_per_seg)

        _copy_full_segments_kernel[grid_full](
            paged_k, paged_v,
            sink_k, sink_v,
            recent_k, recent_v,
            final_k, final_v,
            cos_ptr, sin_ptr,
            sel_idx,
            # paged strides
            paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3),
            # sink strides
            sink_k.stride(0), sink_k.stride(1), sink_k.stride(2),
            # recent strides
            recent_k.stride(0), recent_k.stride(1), recent_k.stride(2),
            # output strides
            final_k.stride(0), final_k.stride(1), final_k.stride(2),
            # rope stride
            rope_stride_t,
            # sel_indices strides
            sel_idx.stride(0), sel_idx.stride(1),
            # runtime dims
            num_kv_heads, top_k, head_dim, sink_len, recent_len, total_len,
            # constexprs
            COMP_SIZE=comp_size, PAGE_SIZE=page_size,
            BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_FULL,
            APPLY_ROPE=1 if apply_rope else 0,
        )

        # ---- Kernel B: unselected compressed pages ----
        if num_unselected > 0:
            BLOCK_T_COMP = max(triton.next_power_of_2(comp_size), 4)
            grid_comp = (bsz * num_kv_heads * num_pages,)
            _copy_unselected_pages_kernel[grid_comp](
                comp_k, comp_v, final_k, final_v,
                cos_ptr, sin_ptr,
                sel_idx,
                comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3),
                final_k.stride(0), final_k.stride(1), final_k.stride(2),
                rope_stride_t,
                sel_idx.stride(0), sel_idx.stride(1),
                num_kv_heads, num_pages, head_dim, sink_len,
                COMP_SIZE=comp_size, PAGE_SIZE=page_size, TOP_K=top_k,
                BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_COMP,
                APPLY_ROPE=1 if apply_rope else 0,
            )

    return final_k, final_v


# ---------------------------------------------------------------------------
# Kernel 3: Q-RoPE (tiny — 1 program per head)
# ---------------------------------------------------------------------------
@triton.jit
def _apply_rope_q_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_h,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    """Apply RoPE to a single-token tensor [bsz, num_heads, 1, head_dim].
    Grid: (bsz * num_heads,)
    """
    pid = tl.program_id(0)
    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < head_dim
    half_d = head_dim // 2

    x_base = x_ptr + pid * stride_h
    x = tl.load(x_base + d_idx, mask=d_mask, other=0.0).to(tl.float32)

    cos = tl.load(cos_ptr + d_idx, mask=d_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + d_idx, mask=d_mask, other=0.0).to(tl.float32)

    # Rotated x: swap halves, negate first half
    d_rot = tl.where(d_idx < half_d, d_idx + half_d, d_idx - half_d)
    x_rot = tl.load(x_base + d_rot, mask=d_mask, other=0.0).to(tl.float32)
    rot_sign = tl.where(d_idx < half_d, -1.0, 1.0)
    x_rotated = rot_sign * x_rot

    out = x * cos + x_rotated * sin

    out_base = out_ptr + pid * stride_h
    tl.store(out_base + d_idx, out.to(tl.bfloat16), mask=d_mask)


def apply_rope_q_triton(
    query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to decode query [bsz, num_heads, 1, head_dim].

    Args:
        query: [bsz, num_heads, 1, head_dim]
        cos:   [1, 1, 1, head_dim]
        sin:   [1, 1, 1, head_dim]

    Returns:
        query_roped: [bsz, num_heads, 1, head_dim]
    """
    bsz, num_heads, _, head_dim = query.shape
    out = torch.empty_like(query)

    BLOCK_D = triton.next_power_of_2(head_dim)

    # Flatten cos/sin to [head_dim]
    cos_flat = cos.view(-1)[:head_dim].contiguous()
    sin_flat = sin.view(-1)[:head_dim].contiguous()

    # query is [bsz, num_heads, 1, head_dim] — stride over heads
    q_flat = query.view(bsz * num_heads, head_dim)
    o_flat = out.view(bsz * num_heads, head_dim)

    # Ensure correct CUDA device context for Triton kernel launch (multi-GPU)
    with torch.cuda.device(query.device):
        grid = (bsz * num_heads,)
        _apply_rope_q_kernel[grid](
            q_flat, cos_flat, sin_flat, o_flat,
            q_flat.stride(0),
            head_dim,
            BLOCK_D=BLOCK_D,
        )

    return out


def apply_rope_q_direct(
    query: torch.Tensor,
    cos_flat: torch.Tensor,
    sin_flat: torch.Tensor,
    out_buf: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to decode query — zero-overhead wrapper.

    Unlike apply_rope_q_triton, this expects pre-flattened cos/sin
    and a pre-allocated output buffer to avoid .contiguous() copies
    and torch.empty_like allocation on every call.

    Args:
        query:    [bsz, num_heads, 1, head_dim]
        cos_flat: [head_dim] — already contiguous (e.g. rope_cache[0, 0, pos])
        sin_flat: [head_dim] — already contiguous
        out_buf:  [bsz, num_heads, 1, head_dim] — pre-allocated output

    Returns:
        out_buf with RoPE applied
    """
    bsz, num_heads, _, head_dim = query.shape
    q_flat = query.view(bsz * num_heads, head_dim)
    o_flat = out_buf.view(bsz * num_heads, head_dim)
    BLOCK_D = triton.next_power_of_2(head_dim)

    with torch.cuda.device(query.device):
        _apply_rope_q_kernel[(bsz * num_heads,)](
            q_flat, cos_flat, sin_flat, o_flat,
            q_flat.stride(0),
            head_dim,
            BLOCK_D=BLOCK_D,
        )

    return out_buf
