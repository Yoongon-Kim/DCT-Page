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


_SCORING_MAP = {"max": 0, "mean": 1, "sum": 2}
_GROUP_AGG_MAP = {"mean": 0, "max": 1}


def score_pages_triton(
    query_states: torch.Tensor,
    compressed_keys: torch.Tensor,
    scoring_method: str,
    group_agg_method: str,
    num_kv_groups: int,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Score all pages. Returns page_scores only (topk deferred to assemble kernel).

    Args:
        query_states:    [bsz, num_heads, 1, head_dim]
        compressed_keys: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
        out: optional pre-allocated [bsz, num_kv_heads, capacity] float32 buffer
             (capacity >= num_pages). Stride(1) may differ from num_pages.

    Returns:
        page_scores: [bsz, num_kv_heads, num_pages] (float32)
    """
    bsz, _num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, num_pages, comp_size, _ = compressed_keys.shape
    scaling = head_dim ** -0.5

    assert q_len == 1, "score_pages_triton only supports decode (q_len=1)"

    query = query_states.squeeze(2).reshape(bsz, num_kv_heads, num_kv_groups, head_dim).contiguous()
    # The score cache may be a non-contiguous prefix slice of a growable page-capacity
    # buffer. Consume its runtime strides directly to avoid copying the full cache
    # every decode step.
    assert compressed_keys.stride(-1) == 1, "score_pages_triton expects head_dim-contiguous keys"

    if out is not None:
        page_scores = out[:, :, :num_pages]
    else:
        page_scores = torch.empty(
            bsz, num_kv_heads, num_pages,
            dtype=torch.float32, device=query.device,
        )

    BLOCK_D = triton.next_power_of_2(head_dim)

    q_stride_0 = query.stride(0)
    q_stride_1 = query.stride(1)
    q_stride_2 = query.stride(2)
    ck_stride_0 = compressed_keys.stride(0)
    ck_stride_1 = compressed_keys.stride(1)
    ck_stride_2 = compressed_keys.stride(2)
    ck_stride_3 = compressed_keys.stride(3)
    ps_stride_0 = page_scores.stride(0)
    ps_stride_bh = page_scores.stride(1)

    SCORING = _SCORING_MAP[scoring_method]
    GROUP_AGG = _GROUP_AGG_MAP[group_agg_method]
    BLOCK_P = 32
    num_page_tiles = (num_pages + BLOCK_P - 1) // BLOCK_P
    grid = (bsz * num_kv_heads, num_page_tiles)

    with torch.cuda.device(query.device):
        _score_pages_fused_kernel[grid](
            query, compressed_keys, page_scores,
            q_stride_0, q_stride_1, q_stride_2,
            ck_stride_0, ck_stride_1, ck_stride_2, ck_stride_3,
            ps_stride_0, ps_stride_bh,
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
# Kernel 1b: Fused TopK + Sort (replaces torch.topk + .sort + .to(int32))
# ---------------------------------------------------------------------------
@triton.jit
def _topk_sort_kernel(
    scores_ptr,       # [bsz * num_kv_heads, num_pages] (flat)
    out_ptr,          # [bsz * num_kv_heads, top_k] (flat)
    s_stride_bh,      # stride for (batch*head) dim of scores
    o_stride_bh,      # stride for (batch*head) dim of output
    num_pages,
    TOP_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """One program per (batch, kv_head). Finds top-k page indices sorted ascending.

    Algorithm: two vectorized sorts via tl.sort —
      1) Sort (score descending, index) packed into int64 to find top-k,
      2) Sort the top-k indices ascending.
    O(n log²n + n log²n) vectorized ops vs. O(k·n + k²) serial ops.
    """
    pid = tl.program_id(0)
    base_s = scores_ptr + pid * s_stride_bh
    base_o = out_ptr + pid * o_stride_bh

    p_idx = tl.arange(0, BLOCK_P)
    mask = p_idx < num_pages
    scores = tl.load(base_s + p_idx, mask=mask, other=float('-inf'))

    # --- Phase 1: Sort by score descending to find top-k ---
    # Convert float32 to signed int32 that preserves total float ordering:
    #   positive floats: bits already sort correctly as signed int32
    #   negative floats: XOR with 0x7FFFFFFF reverses their internal order
    bits = scores.to(tl.int32, bitcast=True)
    sortable = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    # Negate so ascending int64 sort = descending float order
    desc_key = -sortable

    # Pack (desc_key, page_index) into int64; ascending sort yields
    # highest scores first, ties broken by lowest page index.
    packed = (desc_key.to(tl.int64) << 32) | p_idx.to(tl.int64)
    sorted_packed = tl.sort(packed)

    # --- Phase 2: Sort top-k indices ascending ---
    # Extract page indices (lower 32 bits), sentinel the tail with INT_MAX
    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    top_indices = tl.where(p_idx < TOP_K, sorted_indices, 2147483647)
    sorted_top = tl.sort(top_indices)

    # Store first TOP_K results (mask keeps writes in-bounds)
    tl.store(base_o + p_idx, sorted_top, mask=p_idx < TOP_K)


def topk_sort_triton(
    page_scores: torch.Tensor,
    top_k: int,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Fused top-k selection + ascending sort + int32 cast in one kernel.

    Replaces: torch.topk() + .sort() + .to(int32)

    Args:
        page_scores: [bsz, num_kv_heads, num_pages] float32
        top_k: number of pages to select
        out: optional pre-allocated [bsz, num_kv_heads, top_k] int32 buffer.
             Strides may be larger than the logical shape.

    Returns:
        selected_indices: [bsz, num_kv_heads, top_k] int32, sorted ascending
    """
    bsz, num_kv_heads, num_pages = page_scores.shape
    BLOCK_P = triton.next_power_of_2(num_pages)

    if out is not None:
        selected = out
    else:
        selected = torch.empty(
            bsz, num_kv_heads, top_k,
            dtype=torch.int32, device=page_scores.device,
        )

    grid = (bsz * num_kv_heads,)
    s_stride_bh = page_scores.stride(1)
    o_stride_bh = selected.stride(1)

    with torch.cuda.device(page_scores.device):
        _topk_sort_kernel[grid](
            page_scores, selected,
            s_stride_bh,
            o_stride_bh,
            num_pages,
            TOP_K=top_k,
            BLOCK_P=BLOCK_P,
        )

    # --- DEBUG: validate against torch.topk reference ---
    # _, ref_indices = torch.topk(page_scores, top_k, dim=-1)
    # ref_sorted = ref_indices.sort(dim=-1).values.to(torch.int32)
    # if not torch.equal(selected, ref_sorted):
    #     # Find first mismatch for diagnostics
    #     diff_mask = selected != ref_sorted
    #     b, h, k = diff_mask.nonzero(as_tuple=False)[0].tolist()
    #     raise AssertionError(
    #         f"topk_sort_triton mismatch at (b={b}, h={h}, k={k}): "
    #         f"triton={selected[b, h].tolist()}, "
    #         f"ref={ref_sorted[b, h].tolist()}"
    #     )
    # --- END DEBUG ---

    return selected


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
    BLOCK_K: tl.constexpr,
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
    k_idx = tl.arange(0, BLOCK_K)
    k_mask = k_idx < TOP_K
    sel_indices = tl.load(si_base + k_idx, mask=k_mask, other=2147483647).to(tl.int32)

    # ---- Determine segment type and source ----
    is_sink = (seg == 0)
    is_recent = (seg == num_pages + 1)
    page_idx = seg - 1  # only valid when not sink/recent

    # Check if this page is selected (linear scan over TOP_K elements)
    is_selected = tl.sum((sel_indices == page_idx).to(tl.int32)) > 0
    count_before = tl.sum((sel_indices < page_idx).to(tl.int32))

    # Token count and write offset for this segment
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
    out_k: torch.Tensor = None,
    out_v: torch.Tensor = None,
    out_sel_idx: torch.Tensor = None,
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
        out_k/v:          Pre-allocated output buffers (optional)
        out_sel_idx:      Pre-allocated int32 index buffer (optional)

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

    apply_rope = cos_table is not None and sin_table is not None
    device = paged_k.device
    dtype = paged_k.dtype

    # Reuse pre-allocated output buffers if provided and large enough
    if out_k is not None and out_k.shape[2] >= total_len:
        final_k = out_k[:, :, :total_len, :]
        final_v = out_v[:, :, :total_len, :]
    else:
        final_k = torch.empty(bsz, num_kv_heads, total_len, head_dim,
                               dtype=dtype, device=device)
        final_v = torch.empty_like(final_k)

    # Use selected_indices directly if already int32; avoid unnecessary copy
    if selected_indices.dtype == torch.int32:
        sel_idx = selected_indices
    elif out_sel_idx is not None and out_sel_idx.shape[2] >= top_k:
        sel_idx = out_sel_idx[:, :, :top_k]
        sel_idx.copy_(selected_indices)
    else:
        sel_idx = selected_indices.to(torch.int32)

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
            BLOCK_K=triton.next_power_of_2(top_k),
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
    # --- Q-RoPE pointers (fused into Kernel A to avoid separate launch) ---
    q_ptr, q_out_ptr, q_rope_cos_ptr, q_rope_sin_ptr,
    q_stride_h,
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
    num_pages,
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
    ORIGINAL_POS_ROPE: tl.constexpr,
    FUSE_Q_ROPE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
):
    """Copy sink + selected pages + recent to output with optional K-RoPE.

    Grid: (bsz * kv_heads, tiles_per_head)
    tiles_per_head = sink_tiles + top_k * page_tiles + recent_tiles (no wasted tiles).
    Flat tile index maps to segment type via cumulative tile counts.
    """
    pid_bh = tl.program_id(0)
    pid_tile = tl.program_id(1)

    h = pid_bh % num_kv_heads
    b = pid_bh // num_kv_heads

    # Flat tile -> segment mapping (zero wasted tiles)
    sink_tiles = (sink_len + BLOCK_T - 1) // BLOCK_T
    page_tiles = (PAGE_SIZE + BLOCK_T - 1) // BLOCK_T  # constexpr
    page_end = sink_tiles + top_k * page_tiles

    # Segment flags — comparisons, not Python bools (Triton runtime values)
    is_sink = (pid_tile < sink_tiles)
    is_recent = (pid_tile >= page_end)

    # Page-local offset — unconditional, safe for all segments (clamped >= 0)
    local = tl.maximum(pid_tile - sink_tiles, 0)
    rank = local // page_tiles  # only meaningful for pages; clamped via safe_rank elsewhere

    # Load selected page index unconditionally (clamped rank is safe for all segments)
    safe_rank = tl.maximum(tl.minimum(rank, top_k - 1), 0)
    page_idx = tl.load(sel_indices_ptr + b * si_stride_b + h * si_stride_h + safe_rank).to(tl.int32)

    # Single if/elif/else for ALL per-segment values (one merge point)
    if is_sink:
        t_start = pid_tile * BLOCK_T
        num_tokens = sink_len
        write_start = 0
    elif is_recent:
        t_start = (pid_tile - page_end) * BLOCK_T
        num_tokens = recent_len
        write_start = total_len - recent_len
    else:
        t_start = (local % page_tiles) * BLOCK_T
        num_tokens = PAGE_SIZE
        write_start = sink_len + page_idx * COMP_SIZE + rank * (PAGE_SIZE - COMP_SIZE)

    # Early exit for partial last tile of each segment
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
        if ORIGINAL_POS_ROPE == 1:
            if is_sink:
                rope_pos = t_start + t_idx
            elif is_recent:
                rope_pos = sink_len + num_pages * PAGE_SIZE + t_start + t_idx
            else:
                rope_pos = sink_len + page_idx * PAGE_SIZE + t_start + t_idx
        else:
            rope_pos = write_start + t_start + t_idx
        half_d = head_dim // 2
        cos_offsets = rope_pos[:, None] * rope_stride_t + d_idx[None, :]
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

    # ---- Fused Q-RoPE: program 0 applies RoPE to all Q heads ----
    # Q is [bsz, num_q_heads, 1, head_dim]. Since data is tiny (32 heads × 128 dims),
    # a single program handles all Q heads via a loop. This avoids a separate kernel launch.
    if FUSE_Q_ROPE == 1:
        if pid_bh == 0 and pid_tile == 0:
            q_d_idx = tl.arange(0, BLOCK_D)
            q_d_mask = q_d_idx < head_dim
            q_half_d = head_dim // 2
            q_d_rot = tl.where(q_d_idx < q_half_d, q_d_idx + q_half_d, q_d_idx - q_half_d)
            q_rot_sign = tl.where(q_d_idx < q_half_d, -1.0, 1.0)
            # cos/sin for Q position: last position in assembled sequence
            q_cos = tl.load(q_rope_cos_ptr + q_d_idx, mask=q_d_mask, other=0.0).to(tl.float32)
            q_sin = tl.load(q_rope_sin_ptr + q_d_idx, mask=q_d_mask, other=0.0).to(tl.float32)
            for qh in tl.static_range(0, NUM_Q_HEADS):
                q_base = q_ptr + qh * q_stride_h
                q_val = tl.load(q_base + q_d_idx, mask=q_d_mask, other=0.0).to(tl.float32)
                q_rot_val = tl.load(q_base + q_d_rot, mask=q_d_mask, other=0.0).to(tl.float32)
                q_rotated = q_rot_sign * q_rot_val
                q_out = q_val * q_cos + q_rotated * q_sin
                q_out_base = q_out_ptr + qh * q_stride_h
                tl.store(q_out_base + q_d_idx, q_out.to(tl.bfloat16), mask=q_d_mask)


# ---------------------------------------------------------------------------
# Kernel 2c: Copy unselected compressed pages (batched: multiple pages per program)
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
    num_kv_heads, num_pages, head_dim, sink_len, num_groups,
    # --- compile-time constants ---
    COMP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PAGES_PER_PROG: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    """Copy unselected compressed pages to output with optional K-RoPE.
    Grid: (bsz * kv_heads * num_groups,)  where num_groups = ceil(num_pages / PAGES_PER_PROG).
    Each program handles PAGES_PER_PROG consecutive pages, skipping selected ones.
    """
    pid = tl.program_id(0)

    g = pid % num_groups
    tmp = pid // num_groups
    h = tmp % num_kv_heads
    b = tmp // num_kv_heads

    # Load selected indices once per program (reused across all pages in batch)
    sel_base = sel_indices_ptr + b * sel_stride_b + h * sel_stride_h
    k_idx = tl.arange(0, BLOCK_K)
    k_mask = k_idx < TOP_K
    sel_all = tl.load(sel_base + k_idx, mask=k_mask, other=2147483647).to(tl.int32)

    # Precompute d_idx and RoPE constants (shared across pages)
    d_idx = tl.arange(0, BLOCK_D)
    t_idx = tl.arange(0, BLOCK_T)
    t_safe = tl.minimum(t_idx, COMP_SIZE - 1)
    mask = (t_idx < COMP_SIZE)[:, None] & (d_idx < head_dim)[None, :]

    # Base pointers for this (b, h)
    k_dst = out_k_ptr + b * out_stride_b + h * out_stride_h
    v_dst = out_v_ptr + b * out_stride_b + h * out_stride_h
    src_base = b * comp_stride_b + h * comp_stride_h

    if APPLY_ROPE == 1:
        half_d = head_dim // 2
        d_rot = tl.where(d_idx < half_d, d_idx + half_d, d_idx - half_d)
        rot_sign = tl.where(d_idx[None, :] < half_d, -1.0, 1.0)

    p_start = g * PAGES_PER_PROG
    for p_off in tl.static_range(0, PAGES_PER_PROG):
        p = p_start + p_off
        if p < num_pages:
            # Check if selected
            is_selected = tl.sum((sel_all == p).to(tl.int32))
            if is_selected == 0:
                count_before = tl.sum((sel_all < p).to(tl.int32))
                write_start = sink_len + p * COMP_SIZE + count_before * (PAGE_SIZE - COMP_SIZE)

                # Source
                src_off = src_base + p * comp_stride_p
                src_offsets = t_safe[:, None] * comp_stride_t + d_idx[None, :]
                k_vals = tl.load(comp_k_ptr + src_off + src_offsets, mask=mask, other=0.0)
                v_vals = tl.load(comp_v_ptr + src_off + src_offsets, mask=mask, other=0.0)

                # Destination
                dst_offsets = (write_start + t_safe)[:, None] * out_stride_t + d_idx[None, :]

                # K-RoPE
                if APPLY_ROPE == 1:
                    out_pos = write_start + t_safe
                    cos_offsets = out_pos[:, None] * rope_stride_t + d_idx[None, :]
                    cos_vals = tl.load(cos_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
                    sin_vals = tl.load(sin_ptr + cos_offsets, mask=mask, other=0.0).to(tl.float32)
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
    out_k: torch.Tensor = None,
    out_v: torch.Tensor = None,
    out_sel_idx: torch.Tensor = None,
    query_states: torch.Tensor = None,
    q_rope_cos: torch.Tensor = None,
    q_rope_sin: torch.Tensor = None,
    q_rope_buf: torch.Tensor = None,
    cached_params: dict = None,
    stride_cache: dict = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble KV using split kernels. Same API as assemble_kv_full_triton.

    Kernel A: sink + selected pages + recent (all inside Triton)
    Kernel B: unselected compressed pages (branch-free)
    No PyTorch tensor ops between kernel launches.

    Optional pre-allocated buffers (out_k, out_v, out_sel_idx) can be passed
    to avoid per-step memory allocation.

    If cached_params is provided, skips all Python overhead (shape lookups,
    stride computations, grid calculations) and launches kernels directly.
    Use build_assemble_cache() to create the cache on first call.

    If stride_cache is provided, skips stride computations but recomputes
    shape-dependent values (num_pages, total_len, grids) each call.
    Use build_assemble_stride_cache() to create. Safe with changing seq lengths.
    """
    if cached_params is not None:
        return _assemble_kv_split_cached(
            paged_k, paged_v, comp_k, comp_v,
            sink_k, sink_v, recent_k, recent_v,
            selected_indices, cos_table, sin_table,
            out_k, out_v, query_states, q_rope_cos, q_rope_sin, q_rope_buf,
            cached_params,
        )

    if stride_cache is not None:
        return _assemble_kv_split_stride_cached(
            paged_k, paged_v, comp_k, comp_v,
            sink_k, sink_v, recent_k, recent_v,
            selected_indices, cos_table, sin_table,
            out_k, out_v, query_states, q_rope_cos, q_rope_sin, q_rope_buf,
            stride_cache,
        )

    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    comp_size = comp_k.shape[3]
    sink_len = sink_k.shape[2]
    recent_len = recent_k.shape[2]
    top_k = selected_indices.shape[2]
    num_unselected = num_pages - top_k
    middle_len = top_k * page_size + num_unselected * comp_size
    total_len = sink_len + middle_len + recent_len

    apply_rope = cos_table is not None and sin_table is not None
    device = paged_k.device
    dtype = paged_k.dtype

    # Reuse pre-allocated output buffers if provided and large enough
    if out_k is not None and out_k.shape[2] >= total_len:
        final_k = out_k[:, :, :total_len, :]
        final_v = out_v[:, :, :total_len, :]
    else:
        final_k = torch.empty(bsz, num_kv_heads, total_len, head_dim, dtype=dtype, device=device)
        final_v = torch.empty_like(final_k)

    # Use selected_indices directly if already int32; avoid unnecessary copy
    if selected_indices.dtype == torch.int32:
        sel_idx = selected_indices
    elif out_sel_idx is not None and out_sel_idx.shape[2] >= top_k:
        sel_idx = out_sel_idx[:, :, :top_k]
        sel_idx.copy_(selected_indices)
    else:
        sel_idx = selected_indices.to(torch.int32)

    # RoPE setup
    BLOCK_D = triton.next_power_of_2(head_dim)
    if apply_rope:
        rope_stride_t = cos_table.stride(0)
        cos_ptr = cos_table
        sin_ptr = sin_table
    else:
        cos_ptr = final_k  # won't be accessed
        sin_ptr = final_k
        rope_stride_t = 0

    # Q-RoPE fusion setup
    fuse_q_rope = query_states is not None and q_rope_cos is not None and q_rope_sin is not None
    if fuse_q_rope:
        if q_rope_buf is None:
            q_rope_buf = torch.empty_like(query_states)
        q_flat = query_states.view(bsz * query_states.shape[1], head_dim)
        q_out_flat = q_rope_buf.view(bsz * query_states.shape[1], head_dim)
        num_q_heads = query_states.shape[1]
        q_stride_h = q_flat.stride(0)
    else:
        # Dummy pointers (won't be accessed)
        q_flat = final_k
        q_out_flat = final_k
        q_rope_cos = final_k
        q_rope_sin = final_k
        q_stride_h = 0

    # Ensure correct CUDA device context for Triton kernel launch (multi-GPU)
    with torch.cuda.device(device):
        # ---- Kernel A: sink + selected + recent (+ fused Q-RoPE) ----
        BLOCK_T_FULL = 64
        sink_tiles = (sink_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
        page_tiles = (page_size + BLOCK_T_FULL - 1) // BLOCK_T_FULL
        recent_tiles = (recent_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
        tiles_per_head = sink_tiles + top_k * page_tiles + recent_tiles
        grid_full = (bsz * num_kv_heads, tiles_per_head)

        _copy_full_segments_kernel[grid_full](
            paged_k, paged_v,
            sink_k, sink_v,
            recent_k, recent_v,
            final_k, final_v,
            cos_ptr, sin_ptr,
            sel_idx,
            # Q-RoPE args
            q_flat, q_out_flat, q_rope_cos, q_rope_sin,
            q_stride_h,
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
            num_kv_heads, num_pages, top_k, head_dim, sink_len, recent_len, total_len,
            # constexprs
            COMP_SIZE=comp_size, PAGE_SIZE=page_size,
            BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_FULL,
            APPLY_ROPE=1 if apply_rope else 0,
            ORIGINAL_POS_ROPE=0,
            FUSE_Q_ROPE=1 if fuse_q_rope else 0,
            NUM_Q_HEADS=num_q_heads if fuse_q_rope else 0,
        )

        # ---- Kernel B: unselected compressed pages (batched) ----
        if num_unselected > 0:
            BLOCK_T_COMP = max(triton.next_power_of_2(comp_size), 4)
            PAGES_PER_PROG = 16
            num_groups = (num_pages + PAGES_PER_PROG - 1) // PAGES_PER_PROG
            grid_comp = (bsz * num_kv_heads * num_groups,)
            _copy_unselected_pages_kernel[grid_comp](
                comp_k, comp_v, final_k, final_v,
                cos_ptr, sin_ptr,
                sel_idx,
                comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3),
                final_k.stride(0), final_k.stride(1), final_k.stride(2),
                rope_stride_t,
                sel_idx.stride(0), sel_idx.stride(1),
                num_kv_heads, num_pages, head_dim, sink_len, num_groups,
                COMP_SIZE=comp_size, PAGE_SIZE=page_size, TOP_K=top_k,
                BLOCK_K=triton.next_power_of_2(top_k),
                PAGES_PER_PROG=PAGES_PER_PROG,
                BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_COMP,
                APPLY_ROPE=1 if apply_rope else 0,
            )

    if fuse_q_rope:
        return final_k, final_v, q_rope_buf
    return final_k, final_v


def build_assemble_stride_cache(
    paged_k, comp_k, sink_k, recent_k, selected_indices=None,
    cos_table=None, out_k=None, query_states=None,
):
    """Precompute tensor strides and fixed constants only.

    Unlike build_assemble_cache(), this does NOT cache shape-dependent values
    (num_pages, total_len, grids) that change as sequence length grows.
    Safe to reuse across all decode steps.
    """
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape
    comp_size = comp_k.shape[3]
    sink_len = sink_k.shape[2]
    recent_len = recent_k.shape[2]

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_T_FULL = 64
    BLOCK_T_COMP = max(triton.next_power_of_2(comp_size), 4)
    PAGES_PER_PROG = 16

    apply_rope = cos_table is not None
    rope_stride_t = cos_table.stride(0) if apply_rope else 0

    fuse_q_rope = query_states is not None
    num_q_heads = query_states.shape[1] if fuse_q_rope else 0
    q_stride_h = head_dim if fuse_q_rope else 0

    sink_tiles = (sink_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
    page_tiles = (page_size + BLOCK_T_FULL - 1) // BLOCK_T_FULL

    return {
        'device': paged_k.device,
        # Fixed shapes
        'bsz': bsz, 'num_kv_heads': num_kv_heads,
        'page_size': page_size, 'head_dim': head_dim, 'comp_size': comp_size,
        'sink_len': sink_len, 'recent_len': recent_len,
        # ALL strides cached (requires PreAllocatedLayer + pre-allocated comp buffers):
        'paged_strides': (paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3)),
        'sink_strides': (sink_k.stride(0), sink_k.stride(1), sink_k.stride(2)),
        'recent_strides': (recent_k.stride(0), recent_k.stride(1), recent_k.stride(2)),
        'out_strides': (out_k.stride(0), out_k.stride(1), out_k.stride(2)),
        # comp_strides NOT cached — comp_k from torch.cat changes strides every step
        'sel_strides': (selected_indices.stride(0), selected_indices.stride(1)) if selected_indices is not None else (0, 0),
        'rope_stride_t': rope_stride_t,
        'q_stride_h': q_stride_h,
        # Fixed constexprs
        'BLOCK_D': BLOCK_D, 'BLOCK_T_FULL': BLOCK_T_FULL,
        'BLOCK_T_COMP': BLOCK_T_COMP, 'PAGES_PER_PROG': PAGES_PER_PROG,
        'APPLY_ROPE': 1 if apply_rope else 0,
        'FUSE_Q_ROPE': 1 if fuse_q_rope else 0,
        'NUM_Q_HEADS': num_q_heads,
        'sink_tiles': sink_tiles,
        'page_tiles': page_tiles,
        # Flags
        'fuse_q_rope': fuse_q_rope,
    }


def _assemble_kv_split_stride_cached(
    paged_k, paged_v, comp_k, comp_v,
    sink_k, sink_v, recent_k, recent_v,
    selected_indices, cos_table, sin_table,
    out_k, out_v, query_states, q_rope_cos, q_rope_sin, q_rope_buf,
    c,
):
    """Fast path using cached strides. Recomputes shape-dependent values each call."""
    head_dim = c['head_dim']

    # Recompute shape-dependent values (cheap integer ops)
    num_pages = paged_k.shape[2]
    recent_len = recent_k.shape[2]  # live — actual_recent changes every step
    top_k = selected_indices.shape[2]
    num_unselected = num_pages - top_k
    middle_len = top_k * c['page_size'] + num_unselected * c['comp_size']
    total_len = c['sink_len'] + middle_len + recent_len

    final_k = out_k[:, :, :total_len, :]
    final_v = out_v[:, :, :total_len, :]
    sel_idx = selected_indices

    cos_ptr = cos_table if cos_table is not None else final_k
    sin_ptr = sin_table if sin_table is not None else final_k

    if c['fuse_q_rope']:
        q_flat = query_states.view(-1, head_dim)
        q_out_flat = q_rope_buf.view(-1, head_dim)
    else:
        q_flat = final_k
        q_out_flat = final_k
        q_rope_cos = final_k
        q_rope_sin = final_k

    # Strides: most cached, comp_strides computed live (torch.cat changes them each step)
    ps = c['paged_strides']
    ss = c['sink_strides']
    rs = c['recent_strides']
    os = c['out_strides']
    cs = (comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3))
    si = c['sel_strides']

    # Recompute grids (cheap integer ops)
    recent_tiles = (recent_len + c['BLOCK_T_FULL'] - 1) // c['BLOCK_T_FULL']
    tiles_per_head = c['sink_tiles'] + top_k * c['page_tiles'] + recent_tiles
    grid_full = (c['bsz'] * c['num_kv_heads'], tiles_per_head)
    num_groups = (num_pages + c['PAGES_PER_PROG'] - 1) // c['PAGES_PER_PROG']

    with torch.cuda.device(c['device']):
        _copy_full_segments_kernel[grid_full](
            paged_k, paged_v,
            sink_k, sink_v,
            recent_k, recent_v,
            final_k, final_v,
            cos_ptr, sin_ptr,
            sel_idx,
            q_flat, q_out_flat, q_rope_cos, q_rope_sin,
            c['q_stride_h'],
            ps[0], ps[1], ps[2], ps[3],
            ss[0], ss[1], ss[2],
            rs[0], rs[1], rs[2],
            os[0], os[1], os[2],
            c['rope_stride_t'],
            si[0], si[1],
            c['num_kv_heads'], num_pages, top_k, head_dim, c['sink_len'], recent_len, total_len,
            COMP_SIZE=c['comp_size'], PAGE_SIZE=c['page_size'],
            BLOCK_D=c['BLOCK_D'], BLOCK_T=c['BLOCK_T_FULL'],
            APPLY_ROPE=c['APPLY_ROPE'],
            ORIGINAL_POS_ROPE=0,
            FUSE_Q_ROPE=c['FUSE_Q_ROPE'],
            NUM_Q_HEADS=c['NUM_Q_HEADS'],
        )

        if num_unselected > 0:
            grid_comp = (c['bsz'] * c['num_kv_heads'] * num_groups,)
            _copy_unselected_pages_kernel[grid_comp](
                comp_k, comp_v, final_k, final_v,
                cos_ptr, sin_ptr,
                sel_idx,
                cs[0], cs[1], cs[2], cs[3],
                os[0], os[1], os[2],
                c['rope_stride_t'],
                si[0], si[1],
                c['num_kv_heads'], num_pages, head_dim, c['sink_len'], num_groups,
                COMP_SIZE=c['comp_size'], PAGE_SIZE=c['page_size'], TOP_K=top_k,
                BLOCK_K=triton.next_power_of_2(top_k),
                PAGES_PER_PROG=c['PAGES_PER_PROG'],
                BLOCK_D=c['BLOCK_D'], BLOCK_T=c['BLOCK_T_COMP'],
                APPLY_ROPE=c['APPLY_ROPE'],
            )

    if c['fuse_q_rope']:
        return final_k, final_v, q_rope_buf
    return final_k, final_v


def build_assemble_cache(
    paged_k, comp_k, sink_k, recent_k, selected_indices,
    cos_table, out_k, out_v, query_states=None,
):
    """Precompute all strides, grid dims, and constants for assemble_kv_split_triton.

    Call once when shapes are known (first decode step). Pass the returned dict
    as cached_params= on all subsequent calls to skip Python overhead.
    """
    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    comp_size = comp_k.shape[3]
    sink_len = sink_k.shape[2]
    recent_len = recent_k.shape[2]
    top_k = selected_indices.shape[2]
    num_unselected = num_pages - top_k
    middle_len = top_k * page_size + num_unselected * comp_size
    total_len = sink_len + middle_len + recent_len

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_T_FULL = 64
    sink_tiles = (sink_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
    page_tiles = (page_size + BLOCK_T_FULL - 1) // BLOCK_T_FULL
    recent_tiles = (recent_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
    tiles_per_head = sink_tiles + top_k * page_tiles + recent_tiles
    BLOCK_T_COMP = max(triton.next_power_of_2(comp_size), 4)
    PAGES_PER_PROG = 16
    num_groups = (num_pages + PAGES_PER_PROG - 1) // PAGES_PER_PROG

    apply_rope = cos_table is not None
    rope_stride_t = cos_table.stride(0) if apply_rope else 0

    fuse_q_rope = query_states is not None
    num_q_heads = query_states.shape[1] if fuse_q_rope else 0
    q_stride_h = head_dim if fuse_q_rope else 0  # contiguous view stride

    return {
        # Device
        'device': paged_k.device,
        # Shapes
        'bsz': bsz, 'num_kv_heads': num_kv_heads, 'num_pages': num_pages,
        'page_size': page_size, 'head_dim': head_dim, 'comp_size': comp_size,
        'sink_len': sink_len, 'recent_len': recent_len, 'top_k': top_k,
        'num_unselected': num_unselected, 'total_len': total_len,
        # Strides (these don't change if tensor shapes/layout are fixed)
        'paged_strides': (paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3)),
        'sink_strides': (sink_k.stride(0), sink_k.stride(1), sink_k.stride(2)),
        'recent_strides': (recent_k.stride(0), recent_k.stride(1), recent_k.stride(2)),
        'out_strides': (out_k.stride(0), out_k.stride(1), out_k.stride(2)),
        # comp_strides NOT cached — comp_k from torch.cat changes strides every step
        'sel_strides': (selected_indices.stride(0), selected_indices.stride(1)),
        'rope_stride_t': rope_stride_t,
        'q_stride_h': q_stride_h,
        # Grid dims
        'grid_full': (bsz * num_kv_heads, tiles_per_head),
        'grid_comp': (bsz * num_kv_heads * num_groups,),
        'num_groups': num_groups,
        # Constexprs
        'BLOCK_D': BLOCK_D, 'BLOCK_T_FULL': BLOCK_T_FULL,
        'BLOCK_T_COMP': BLOCK_T_COMP, 'PAGES_PER_PROG': PAGES_PER_PROG,
        'APPLY_ROPE': 1 if apply_rope else 0,
        'FUSE_Q_ROPE': 1 if fuse_q_rope else 0,
        'NUM_Q_HEADS': num_q_heads,
        # Flags
        'fuse_q_rope': fuse_q_rope,
    }


def _assemble_kv_split_cached(
    paged_k, paged_v, comp_k, comp_v,
    sink_k, sink_v, recent_k, recent_v,
    selected_indices, cos_table, sin_table,
    out_k, out_v, query_states, q_rope_cos, q_rope_sin, q_rope_buf,
    c,
):
    """Fast path: launch kernels using precomputed params. Minimal Python overhead."""
    total_len = c['total_len']
    head_dim = c['head_dim']

    final_k = out_k[:, :, :total_len, :]
    final_v = out_v[:, :, :total_len, :]
    sel_idx = selected_indices  # assumed int32 from caller

    cos_ptr = cos_table if cos_table is not None else final_k
    sin_ptr = sin_table if sin_table is not None else final_k

    if c['fuse_q_rope']:
        q_flat = query_states.view(-1, head_dim)
        q_out_flat = q_rope_buf.view(-1, head_dim)
    else:
        q_flat = final_k
        q_out_flat = final_k
        q_rope_cos = final_k
        q_rope_sin = final_k

    ps = c['paged_strides']
    ss = c['sink_strides']
    rs = c['recent_strides']
    os = c['out_strides']
    cs = (comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3))
    si = c['sel_strides']

    # Ensure correct CUDA device context for Triton kernel launch (multi-GPU)
    with torch.cuda.device(c['device']):
        # Kernel A
        _copy_full_segments_kernel[c['grid_full']](
            paged_k, paged_v,
            sink_k, sink_v,
            recent_k, recent_v,
            final_k, final_v,
            cos_ptr, sin_ptr,
            sel_idx,
            q_flat, q_out_flat, q_rope_cos, q_rope_sin,
            c['q_stride_h'],
            ps[0], ps[1], ps[2], ps[3],
            ss[0], ss[1], ss[2],
            rs[0], rs[1], rs[2],
            os[0], os[1], os[2],
            c['rope_stride_t'],
            si[0], si[1],
            c['num_kv_heads'], c['num_pages'], c['top_k'], head_dim, c['sink_len'], c['recent_len'], total_len,
            COMP_SIZE=c['comp_size'], PAGE_SIZE=c['page_size'],
            BLOCK_D=c['BLOCK_D'], BLOCK_T=c['BLOCK_T_FULL'],
            APPLY_ROPE=c['APPLY_ROPE'],
            ORIGINAL_POS_ROPE=0,
            FUSE_Q_ROPE=c['FUSE_Q_ROPE'],
            NUM_Q_HEADS=c['NUM_Q_HEADS'],
        )

        # Kernel B
        if c['num_unselected'] > 0:
            _copy_unselected_pages_kernel[c['grid_comp']](
                comp_k, comp_v, final_k, final_v,
                cos_ptr, sin_ptr,
                sel_idx,
                cs[0], cs[1], cs[2], cs[3],
                os[0], os[1], os[2],
                c['rope_stride_t'],
                si[0], si[1],
                c['num_kv_heads'], c['num_pages'], head_dim, c['sink_len'], c['num_groups'],
                COMP_SIZE=c['comp_size'], PAGE_SIZE=c['page_size'], TOP_K=c['top_k'],
                BLOCK_K=triton.next_power_of_2(c['top_k']),
                PAGES_PER_PROG=c['PAGES_PER_PROG'],
                BLOCK_D=c['BLOCK_D'], BLOCK_T=c['BLOCK_T_COMP'],
                APPLY_ROPE=c['APPLY_ROPE'],
            )

    if c['fuse_q_rope']:
        return final_k, final_v, q_rope_buf
    return final_k, final_v


# ---------------------------------------------------------------------------
# Wrapper: Drop-mode assemble (Kernel A only, COMP_SIZE=0)
# ---------------------------------------------------------------------------
def build_assemble_drop_stride_cache(
    paged_k,
    sink_k,
    recent_k,
    selected_indices,
    out_k,
    cos_table=None,
    original_position_rope=False,
):
    """Precompute tensor strides and fixed constants for drop-mode assembly."""
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape
    sink_len = sink_k.shape[2]

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_T_FULL = 64
    sink_tiles = (sink_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
    page_tiles = (page_size + BLOCK_T_FULL - 1) // BLOCK_T_FULL

    apply_rope = cos_table is not None
    rope_stride_t = cos_table.stride(0) if apply_rope else 0

    return {
        "device": paged_k.device,
        "bsz": bsz,
        "num_kv_heads": num_kv_heads,
        "page_size": page_size,
        "head_dim": head_dim,
        "sink_len": sink_len,
        "paged_strides": (
            paged_k.stride(0),
            paged_k.stride(1),
            paged_k.stride(2),
            paged_k.stride(3),
        ),
        "sink_strides": (sink_k.stride(0), sink_k.stride(1), sink_k.stride(2)),
        "recent_strides": (recent_k.stride(0), recent_k.stride(1), recent_k.stride(2)),
        "out_strides": (out_k.stride(0), out_k.stride(1), out_k.stride(2)),
        "sel_strides": (selected_indices.stride(0), selected_indices.stride(1)),
        "rope_stride_t": rope_stride_t,
        "BLOCK_D": BLOCK_D,
        "BLOCK_T_FULL": BLOCK_T_FULL,
        "sink_tiles": sink_tiles,
        "page_tiles": page_tiles,
        "APPLY_ROPE": 1 if apply_rope else 0,
        "ORIGINAL_POS_ROPE": 1 if original_position_rope else 0,
    }


def _assemble_kv_drop_stride_cached(
    paged_k,
    paged_v,
    sink_k,
    sink_v,
    recent_k,
    recent_v,
    selected_indices,
    cos_table,
    sin_table,
    out_k,
    out_v,
    out_sel_idx,
    num_pages,
    c,
):
    """Fast path using cached strides for drop-mode assembly."""
    recent_len = recent_k.shape[2]
    top_k = selected_indices.shape[2]
    total_len = c["sink_len"] + top_k * c["page_size"] + recent_len

    final_k = out_k[:, :, :total_len, :]
    final_v = out_v[:, :, :total_len, :]

    if selected_indices.dtype == torch.int32:
        sel_idx = selected_indices
    elif out_sel_idx is not None and out_sel_idx.shape[2] >= top_k:
        sel_idx = out_sel_idx[:, :, :top_k]
        sel_idx.copy_(selected_indices)
    else:
        sel_idx = selected_indices.to(torch.int32)

    ps = c["paged_strides"]
    ss = c["sink_strides"]
    rs = c["recent_strides"]
    os = c["out_strides"]
    si = c["sel_strides"]

    cos_ptr = cos_table if cos_table is not None else final_k
    sin_ptr = sin_table if sin_table is not None else final_k

    recent_tiles = (recent_len + c["BLOCK_T_FULL"] - 1) // c["BLOCK_T_FULL"]
    tiles_per_head = c["sink_tiles"] + top_k * c["page_tiles"] + recent_tiles
    grid_full = (c["bsz"] * c["num_kv_heads"], tiles_per_head)

    with torch.cuda.device(c["device"]):
        _copy_full_segments_kernel[grid_full](
            paged_k, paged_v,
            sink_k, sink_v,
            recent_k, recent_v,
            final_k, final_v,
            cos_ptr, sin_ptr,
            sel_idx,
            final_k, final_k, final_k, final_k,
            0,
            ps[0], ps[1], ps[2], ps[3],
            ss[0], ss[1], ss[2],
            rs[0], rs[1], rs[2],
            os[0], os[1], os[2],
            c["rope_stride_t"],
            si[0], si[1],
            c["num_kv_heads"], num_pages, top_k, c["head_dim"], c["sink_len"], recent_len, total_len,
            COMP_SIZE=0, PAGE_SIZE=c["page_size"],
            BLOCK_D=c["BLOCK_D"], BLOCK_T=c["BLOCK_T_FULL"],
            APPLY_ROPE=c["APPLY_ROPE"],
            ORIGINAL_POS_ROPE=c["ORIGINAL_POS_ROPE"],
            FUSE_Q_ROPE=0,
            NUM_Q_HEADS=0,
        )

    return final_k, final_v


def assemble_kv_drop_triton(
    paged_k: torch.Tensor,
    paged_v: torch.Tensor,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    selected_indices: torch.Tensor,
    cos_table: torch.Tensor = None,
    sin_table: torch.Tensor = None,
    out_k: torch.Tensor = None,
    out_v: torch.Tensor = None,
    out_sel_idx: torch.Tensor = None,
    stride_cache: dict = None,
    num_pages: int = None,
    original_position_rope: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble KV for drop mode: sink + selected full pages + recent.

    Reuses _copy_full_segments_kernel with COMP_SIZE=0 so that selected pages
    are packed contiguously (write_start = sink_len + rank * PAGE_SIZE).
    No Kernel B launch needed — unselected pages are simply dropped.
    """
    bsz, num_kv_heads, inferred_num_pages, page_size, head_dim = paged_k.shape
    if num_pages is None:
        num_pages = inferred_num_pages
    sink_len = sink_k.shape[2]
    recent_len = recent_k.shape[2]
    top_k = selected_indices.shape[2]
    total_len = sink_len + top_k * page_size + recent_len

    apply_rope = cos_table is not None and sin_table is not None
    device = paged_k.device
    dtype = paged_k.dtype

    if stride_cache is not None:
        return _assemble_kv_drop_stride_cached(
            paged_k,
            paged_v,
            sink_k,
            sink_v,
            recent_k,
            recent_v,
            selected_indices,
            cos_table,
            sin_table,
            out_k,
            out_v,
            out_sel_idx,
            num_pages,
            stride_cache,
        )

    if out_k is not None and out_k.shape[2] >= total_len:
        final_k = out_k[:, :, :total_len, :]
        final_v = out_v[:, :, :total_len, :]
    else:
        final_k = torch.empty(bsz, num_kv_heads, total_len, head_dim, dtype=dtype, device=device)
        final_v = torch.empty_like(final_k)

    if selected_indices.dtype == torch.int32:
        sel_idx = selected_indices
    elif out_sel_idx is not None and out_sel_idx.shape[2] >= top_k:
        sel_idx = out_sel_idx[:, :, :top_k]
        sel_idx.copy_(selected_indices)
    else:
        sel_idx = selected_indices.to(torch.int32)

    BLOCK_D = triton.next_power_of_2(head_dim)
    if apply_rope:
        rope_stride_t = cos_table.stride(0)
        cos_ptr = cos_table
        sin_ptr = sin_table
    else:
        cos_ptr = final_k
        sin_ptr = final_k
        rope_stride_t = 0

    with torch.cuda.device(device):
        BLOCK_T_FULL = 64
        sink_tiles = (sink_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
        page_tiles = (page_size + BLOCK_T_FULL - 1) // BLOCK_T_FULL
        recent_tiles = (recent_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
        tiles_per_head = sink_tiles + top_k * page_tiles + recent_tiles
        grid_full = (bsz * num_kv_heads, tiles_per_head)

        _copy_full_segments_kernel[grid_full](
            paged_k, paged_v,
            sink_k, sink_v,
            recent_k, recent_v,
            final_k, final_v,
            cos_ptr, sin_ptr,
            sel_idx,
            # Q-RoPE args (dummy — not fused in drop mode)
            final_k, final_k, final_k, final_k,
            0,  # q_stride_h
            paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3),
            sink_k.stride(0), sink_k.stride(1), sink_k.stride(2),
            recent_k.stride(0), recent_k.stride(1), recent_k.stride(2),
            final_k.stride(0), final_k.stride(1), final_k.stride(2),
            rope_stride_t,
            sel_idx.stride(0), sel_idx.stride(1),
            num_kv_heads, num_pages, top_k, head_dim, sink_len, recent_len, total_len,
            COMP_SIZE=0, PAGE_SIZE=page_size,
            BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_FULL,
            APPLY_ROPE=1 if apply_rope else 0,
            ORIGINAL_POS_ROPE=1 if original_position_rope else 0,
            FUSE_Q_ROPE=0,
            NUM_Q_HEADS=0,
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
