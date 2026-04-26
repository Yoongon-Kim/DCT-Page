"""
Fused Triton kernels for DCT Page Attention.

Score:    _score_pages_fused_kernel (+ specialized c4/g4, c1/g4 variants)
TopK:     _topk_sort_kernel (fused topk + ascending sort)
Assemble: _copy_full_segments_kernel (sink + selected + recent + fused Q-RoPE)
          _copy_unselected_pages_kernel (unselected compressed pages)
Q-RoPE:   _apply_rope_q_kernel (standalone decode query RoPE)
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


@triton.jit
def _score_pages_max_mean_c4_g4_kernel(
    query_ptr,         # [bsz, num_kv_heads, 4, head_dim]
    comp_keys_ptr,     # [bsz, num_kv_heads, num_pages, 4, head_dim]
    out_scores_ptr,    # [bsz, num_kv_heads, num_pages]
    q_stride_b, q_stride_h, q_stride_g,
    ck_stride_b, ck_stride_h, ck_stride_p, ck_stride_c,
    os_stride_b, os_stride_h,
    num_kv_heads,
    num_pages,
    head_dim,
    scaling,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Specialized exact path for scoring=max, group_agg=mean, comp_size=4, groups=4."""
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

    q0 = tl.load(q_base + 0 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_base + 1 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_base + 2 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_base + 3 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling

    s0 = tl.full([BLOCK_P], float("-inf"), dtype=tl.float32)
    s1 = tl.full([BLOCK_P], float("-inf"), dtype=tl.float32)
    s2 = tl.full([BLOCK_P], float("-inf"), dtype=tl.float32)
    s3 = tl.full([BLOCK_P], float("-inf"), dtype=tl.float32)

    mask_2d = p_mask[:, None] & d_mask[None, :]
    for c in range(4):
        k_ptrs = ck_base + p_indices[:, None] * ck_stride_p + c * ck_stride_c + d_idx[None, :]
        k = tl.load(k_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        s0 = tl.maximum(s0, tl.sum(k * q0[None, :], axis=1))
        s1 = tl.maximum(s1, tl.sum(k * q1[None, :], axis=1))
        s2 = tl.maximum(s2, tl.sum(k * q2[None, :], axis=1))
        s3 = tl.maximum(s3, tl.sum(k * q3[None, :], axis=1))

    agg_scores = 0.25 * (s0 + s1 + s2 + s3)
    out_ptrs = out_scores_ptr + b * os_stride_b + h * os_stride_h + p_indices
    tl.store(out_ptrs, agg_scores, mask=p_mask)


@triton.jit
def _score_pages_c1_g4_kernel(
    query_ptr,         # [bsz, num_kv_heads, 4, head_dim]
    comp_keys_ptr,     # [bsz, num_kv_heads, num_pages, 1, head_dim]
    out_scores_ptr,    # [bsz, num_kv_heads, num_pages]
    q_stride_b, q_stride_h, q_stride_g,
    ck_stride_b, ck_stride_h, ck_stride_p, ck_stride_c,
    os_stride_b, os_stride_h,
    num_kv_heads,
    num_pages,
    head_dim,
    scaling,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    GROUP_AGG_METHOD: tl.constexpr,  # 0=mean, 1=max
):
    """Specialized exact path for comp_size=1, groups=4."""
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

    q0 = tl.load(q_base + 0 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_base + 1 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_base + 2 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_base + 3 * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling

    k_ptrs = ck_base + p_indices[:, None] * ck_stride_p + 0 * ck_stride_c + d_idx[None, :]
    mask_2d = p_mask[:, None] & d_mask[None, :]
    k = tl.load(k_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    s0 = tl.sum(k * q0[None, :], axis=1)
    s1 = tl.sum(k * q1[None, :], axis=1)
    s2 = tl.sum(k * q2[None, :], axis=1)
    s3 = tl.sum(k * q3[None, :], axis=1)

    if GROUP_AGG_METHOD == 0:
        agg_scores = 0.25 * (s0 + s1 + s2 + s3)
    else:
        agg_scores = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))

    out_ptrs = out_scores_ptr + b * os_stride_b + h * os_stride_h + p_indices
    tl.store(out_ptrs, agg_scores, mask=p_mask)


@triton.jit
def _score_pages_max_max_kernel(
    query_ptr,         # [bsz, num_kv_heads, NUM_KV_GROUPS, head_dim]
    comp_keys_ptr,     # [bsz, num_kv_heads, num_pages, COMP_SIZE, head_dim]
    out_scores_ptr,    # [bsz, num_kv_heads, num_pages]
    q_stride_b, q_stride_h, q_stride_g,
    ck_stride_b, ck_stride_h, ck_stride_p, ck_stride_c,
    os_stride_b, os_stride_h,
    num_kv_heads,
    num_pages,
    head_dim,
    scaling,
    NUM_KV_GROUPS: tl.constexpr,
    COMP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Specialized kernel for scoring_method=max, group_agg_method=max.

    Collapses the intermediate [NUM_KV_GROUPS, BLOCK_P] buffer into a single
    [BLOCK_P] running max since both aggregations are max (associative /
    commutative across (c, g) pairs).

    Tuning: BLOCK_P=16, num_warps=4, num_stages=4 was the winner of a manual
    sweep at (page_size=32, comp_size=4, num_pages=1020) and (page_size=16,
    comp_size=2, num_pages=2040) — within noise across both target configs.
    @triton.autotune was tried but added ~40us/launch overhead even on cache
    hits, dominating the kernel itself; a fixed config is faster here.
    """
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

    best = tl.full([BLOCK_P], float("-inf"), dtype=tl.float32)
    mask_2d = p_mask[:, None] & d_mask[None, :]

    # Outer loop on c (loads large k tile once per c), inner loop on g (small
    # q vector). Both loops are compile-time unrolled via tl.static_range.
    for c in tl.static_range(COMP_SIZE):
        k_ptrs = ck_base + p_indices[:, None] * ck_stride_p + c * ck_stride_c + d_idx[None, :]
        k = tl.load(k_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        for g in tl.static_range(NUM_KV_GROUPS):
            q = tl.load(q_base + g * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32) * scaling
            dots = tl.sum(k * q[None, :], axis=1)
            best = tl.maximum(best, dots)

    out_ptrs = out_scores_ptr + b * os_stride_b + h * os_stride_h + p_indices
    tl.store(out_ptrs, best, mask=p_mask)


_SCORING_MAP = {"max": 0, "mean": 1, "sum": 2}
_GROUP_AGG_MAP = {"mean": 0, "max": 1, "topp": 1}


def _score_pages_torch_fallback(
    query_states: torch.Tensor,
    compressed_keys: torch.Tensor,
    scoring_method: str,
    group_agg_method: str,
    num_kv_groups: int,
    out: torch.Tensor = None,
) -> torch.Tensor:
    bsz, _num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, num_pages, _comp_size, _ = compressed_keys.shape
    assert q_len == 1, "_score_pages_torch_fallback only supports decode (q_len=1)"

    query_3d = query_states.squeeze(2)
    if query_3d.stride(-1) == 1 and query_3d.stride(1) == head_dim:
        query = query_3d.view(bsz, num_kv_heads, num_kv_groups, head_dim)
    else:
        query = query_3d.reshape(bsz, num_kv_heads, num_kv_groups, head_dim).contiguous()

    q = query.to(torch.float32) * (head_dim ** -0.5)
    k = compressed_keys.to(torch.float32)
    group_token_scores = torch.einsum("bhgd,bhpcd->bhgpc", q, k)

    if scoring_method == "max":
        group_page_scores = group_token_scores.max(dim=-1).values
    elif scoring_method == "mean":
        group_page_scores = group_token_scores.mean(dim=-1)
    elif scoring_method == "sum":
        group_page_scores = group_token_scores.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported scoring_method: {scoring_method}")

    if group_agg_method == "mean":
        page_scores = group_page_scores.mean(dim=2)
    elif group_agg_method == "max":
        page_scores = group_page_scores.max(dim=2).values
    elif group_agg_method == "topp":
        k_top = min(2, num_kv_groups)
        page_scores = group_page_scores.topk(k_top, dim=2).values.mean(dim=2)
    else:
        raise ValueError(f"Unsupported group_agg_method: {group_agg_method}")

    if out is not None:
        out_view = out[:, :, :num_pages]
        out_view.copy_(page_scores)
        return out_view
    return page_scores.contiguous()


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

    query_3d = query_states.squeeze(2)
    if query_3d.stride(-1) == 1 and query_3d.stride(1) == head_dim:
        query = query_3d.view(bsz, num_kv_heads, num_kv_groups, head_dim)
    else:
        query = query_3d.reshape(bsz, num_kv_heads, num_kv_groups, head_dim).contiguous()
    # The score cache may be a non-contiguous prefix slice of a growable page-capacity
    # buffer. Consume its runtime strides directly to avoid copying the full cache
    # every decode step.
    assert compressed_keys.stride(-1) == 1, "score_pages_triton expects head_dim-contiguous keys"
    assert query.stride(-1) == 1, "score_pages_triton expects head_dim-contiguous query"

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

    if group_agg_method == "topp":
        return _score_pages_torch_fallback(
            query_states,
            compressed_keys,
            scoring_method,
            group_agg_method,
            num_kv_groups,
            out=out,
        )

    SCORING = _SCORING_MAP[scoring_method]
    GROUP_AGG = _GROUP_AGG_MAP[group_agg_method]
    BLOCK_P = 32
    num_page_tiles = (num_pages + BLOCK_P - 1) // BLOCK_P
    grid = (bsz * num_kv_heads, num_page_tiles)

    use_specialized_max_max = (
        scoring_method == "max" and group_agg_method == "max"
    )
    use_specialized_c4_g4 = (
        scoring_method == "max"
        and group_agg_method == "mean"
        and num_kv_groups == 4
        and comp_size == 4
    )
    use_specialized_c1_g4 = (
        num_kv_groups == 4
        and comp_size == 1
        and group_agg_method in {"mean", "max"}
    )

    with torch.cuda.device(query.device):
        if use_specialized_max_max:
            MM_BLOCK_P = 16
            MM_NUM_WARPS = 4
            MM_NUM_STAGES = 4
            mm_grid = (bsz * num_kv_heads, (num_pages + MM_BLOCK_P - 1) // MM_BLOCK_P)
            _score_pages_max_max_kernel[mm_grid](
                query, compressed_keys, page_scores,
                q_stride_0, q_stride_1, q_stride_2,
                ck_stride_0, ck_stride_1, ck_stride_2, ck_stride_3,
                ps_stride_0, ps_stride_bh,
                num_kv_heads, num_pages, head_dim,
                scaling,
                NUM_KV_GROUPS=num_kv_groups,
                COMP_SIZE=comp_size,
                BLOCK_D=BLOCK_D,
                BLOCK_P=MM_BLOCK_P,
                num_warps=MM_NUM_WARPS,
                num_stages=MM_NUM_STAGES,
            )
        elif use_specialized_c4_g4:
            _score_pages_max_mean_c4_g4_kernel[grid](
                query, compressed_keys, page_scores,
                q_stride_0, q_stride_1, q_stride_2,
                ck_stride_0, ck_stride_1, ck_stride_2, ck_stride_3,
                ps_stride_0, ps_stride_bh,
                num_kv_heads, num_pages, head_dim,
                scaling,
                BLOCK_D=BLOCK_D,
                BLOCK_P=BLOCK_P,
            )
        elif use_specialized_c1_g4:
            _score_pages_c1_g4_kernel[grid](
                query, compressed_keys, page_scores,
                q_stride_0, q_stride_1, q_stride_2,
                ck_stride_0, ck_stride_1, ck_stride_2, ck_stride_3,
                ps_stride_0, ps_stride_bh,
                num_kv_heads, num_pages, head_dim,
                scaling,
                BLOCK_D=BLOCK_D,
                BLOCK_P=BLOCK_P,
                GROUP_AGG_METHOD=GROUP_AGG,
            )
        else:
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
    SORT_ASCENDING: tl.constexpr,
):
    """One program per (batch, kv_head). Finds top-k page indices.

    Phase 1 sorts `(−score, page_idx)` packed into int64 to locate the top-k.
    Phase 2 (when `SORT_ASCENDING=True`) sorts those top-k indices ascending.
    Drop-mode callers pass `SORT_ASCENDING=False` and skip phase 2 entirely —
    attention with `is_causal=False` is permutation-invariant over middle KV.
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

    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    if SORT_ASCENDING:
        # Sentinel the tail with INT_MAX, then sort ascending by page index.
        top_indices = tl.where(p_idx < TOP_K, sorted_indices, 2147483647)
        out_indices = tl.sort(top_indices)
    else:
        # Descending-score order is fine for drop mode; skip the second sort.
        out_indices = sorted_indices

    tl.store(base_o + p_idx, out_indices, mask=p_idx < TOP_K)


def topk_sort_triton(
    page_scores: torch.Tensor,
    top_k: int,
    out: torch.Tensor = None,
    sort_ascending: bool = True,
) -> torch.Tensor:
    """Fused top-k selection (+ optional ascending sort + int32 cast) in one kernel.

    Replaces: torch.topk() + .sort() + .to(int32)

    Args:
        page_scores: [bsz, num_kv_heads, num_pages] float32
        top_k: number of pages to select
        out: optional pre-allocated [bsz, num_kv_heads, top_k] int32 buffer.
             Strides may be larger than the logical shape.
        sort_ascending: if True (default), output indices are sorted ascending by
            page index (required for compressed mode). If False, output indices
            are in descending-score order — fine for drop mode, ~40-50% cheaper.

    Returns:
        selected_indices: [bsz, num_kv_heads, top_k] int32
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
            SORT_ASCENDING=sort_ascending,
            num_warps=1,
            num_stages=1,
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
# Kernel 1c: Two-stage TopK (for wide BLOCK_P, e.g. page_size=16)
# ---------------------------------------------------------------------------
@triton.jit
def _topk_local_kernel(
    scores_ptr,       # [bsz * num_kv_heads, num_pages]
    scratch_ptr,      # [bsz * num_kv_heads, NUM_CHUNKS * TOP_K]  int64 packed
    s_stride_bh, sc_stride_bh,
    num_pages,
    TOP_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Stage 1: sort each chunk of scores, write top TOP_K packed (desc_key, page_idx)."""
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs = tl.arange(0, CHUNK_SIZE)
    p_idx = pid_c * CHUNK_SIZE + offs
    mask = p_idx < num_pages

    base_s = scores_ptr + pid_bh * s_stride_bh
    scores = tl.load(base_s + p_idx, mask=mask, other=float('-inf'))

    bits = scores.to(tl.int32, bitcast=True)
    sortable = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    desc_key = -sortable
    # Use p_idx (global page index) so merge yields the correct answer.
    packed = (desc_key.to(tl.int64) << 32) | p_idx.to(tl.int64)
    sorted_packed = tl.sort(packed)  # ascending int64 = descending score

    sc_base = scratch_ptr + pid_bh * sc_stride_bh + pid_c * TOP_K
    write_mask = offs < TOP_K
    tl.store(sc_base + offs, sorted_packed, mask=write_mask)


@triton.jit
def _topk_merge_kernel(
    scratch_ptr,      # [bsz * num_kv_heads, TOTAL]  int64 packed
    out_ptr,          # [bsz * num_kv_heads, TOP_K]  int32
    sc_stride_bh, o_stride_bh,
    TOP_K: tl.constexpr,
    TOTAL: tl.constexpr,  # NUM_CHUNKS * TOP_K, must be power of 2
    SORT_ASCENDING: tl.constexpr,
):
    """Stage 2: merge all chunk top-k lists and emit final top-k.

    When `SORT_ASCENDING=True` the output is sorted ascending by page index
    (required for compressed mode). Otherwise it's left in descending-score
    order — fine for drop mode, skips a second bitonic sort on `TOTAL` lanes.
    """
    pid_bh = tl.program_id(0)
    idx = tl.arange(0, TOTAL)

    sc_base = scratch_ptr + pid_bh * sc_stride_bh
    packed = tl.load(sc_base + idx)
    sorted_packed = tl.sort(packed)

    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    if SORT_ASCENDING:
        top_indices = tl.where(idx < TOP_K, sorted_indices, 2147483647)
        out_indices = tl.sort(top_indices)
    else:
        out_indices = sorted_indices

    base_o = out_ptr + pid_bh * o_stride_bh
    tl.store(base_o + idx, out_indices, mask=idx < TOP_K)


def topk_sort_twostage_triton(
    page_scores: torch.Tensor,
    top_k: int,
    out: torch.Tensor = None,
    scratch: torch.Tensor = None,
    num_chunks: int = 4,
    sort_ascending: bool = True,
) -> torch.Tensor:
    """Two-stage parallel top-k: sort chunks in parallel, then merge.

    Produces identical output to topk_sort_triton. Wins when BLOCK_P would
    be very wide (e.g. num_pages >= 2048) because single-CTA bitonic sort
    is O(log²n) barrier-chained — halving n via chunking roughly halves
    the barrier depth.

    TOP_K must be a power of 2 so that NUM_CHUNKS * TOP_K is also a power
    of 2 (required by tl.sort in the merge kernel).

    `sort_ascending=False` (drop mode) skips the final ascending-by-index
    sort inside `_topk_merge_kernel` — output is in descending-score order.
    """
    bsz, num_kv_heads, num_pages = page_scores.shape
    assert (top_k & (top_k - 1)) == 0 and top_k > 0, "top_k must be a power of 2"
    assert (num_chunks & (num_chunks - 1)) == 0 and num_chunks > 0, "num_chunks must be a power of 2"

    # Each chunk must be at least TOP_K so it can contribute TOP_K entries.
    per_chunk = max(1, (num_pages + num_chunks - 1) // num_chunks)
    CHUNK_SIZE = max(top_k, triton.next_power_of_2(per_chunk))
    TOTAL = num_chunks * top_k

    if out is None:
        out = torch.empty(
            bsz, num_kv_heads, top_k,
            dtype=torch.int32, device=page_scores.device,
        )
    if scratch is None:
        scratch = torch.empty(
            bsz, num_kv_heads, TOTAL,
            dtype=torch.int64, device=page_scores.device,
        )

    s_stride_bh = page_scores.stride(1)
    sc_stride_bh = scratch.stride(1)
    o_stride_bh = out.stride(1)

    grid1 = (bsz * num_kv_heads, num_chunks)
    grid2 = (bsz * num_kv_heads,)

    with torch.cuda.device(page_scores.device):
        _topk_local_kernel[grid1](
            page_scores, scratch,
            s_stride_bh, sc_stride_bh,
            num_pages,
            TOP_K=top_k,
            CHUNK_SIZE=CHUNK_SIZE,
            num_warps=1,
            num_stages=1,
        )
        _topk_merge_kernel[grid2](
            scratch, out,
            sc_stride_bh, o_stride_bh,
            TOP_K=top_k,
            TOTAL=TOTAL,
            SORT_ASCENDING=sort_ascending,
            num_warps=1,
            num_stages=1,
        )

    return out


def topk_sort_torch(
    page_scores: torch.Tensor,
    top_k: int,
    out: torch.Tensor = None,
    sort_ascending: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch fallback for top-k page selection.

    Produces identical output contract to topk_sort_triton. When
    `sort_ascending=False` (drop mode) the extra `.sort()` is skipped and
    indices are returned in descending-score order.
    """
    _, indices = torch.topk(page_scores, top_k, dim=-1)
    if sort_ascending:
        indices, _ = indices.sort(dim=-1)
    indices = indices.to(torch.int32)
    if out is not None:
        out.copy_(indices)
        return out
    return indices


# Threshold: single-stage wins up to BLOCK_P=1024, two-stage wins above.
# Measured on H100: at num_pages=1020, fused=14us vs twostage-8=24us;
# at num_pages=2040, fused=28us vs twostage-8=23us.
_TOPK_TWOSTAGE_MIN_PAGES = 1025


def topk_sort(
    page_scores: torch.Tensor,
    top_k: int,
    out: torch.Tensor = None,
    scratch: torch.Tensor = None,
    sort_ascending: bool = True,
) -> torch.Tensor:
    """Auto-dispatch top-k: single-stage for num_pages <= 1024, two-stage above.

    The two-stage path needs a power-of-2 top_k (so NUM_CHUNKS * TOP_K is a
    power of 2 for tl.sort). If top_k is not a power of 2, falls back to
    single-stage regardless of num_pages.

    `sort_ascending=False` (drop mode) skips the final ascending-by-index sort.
    """
    num_pages = page_scores.shape[-1]
    is_pow2_top_k = top_k > 0 and (top_k & (top_k - 1)) == 0
    if num_pages >= _TOPK_TWOSTAGE_MIN_PAGES and is_pow2_top_k:
        return topk_sort_twostage_triton(
            page_scores, top_k, out=out, scratch=scratch, num_chunks=8,
            sort_ascending=sort_ascending,
        )
    return topk_sort_triton(
        page_scores, top_k, out=out, sort_ascending=sort_ascending,
    )


# ---------------------------------------------------------------------------
# Kernel 1d: Fused TopK + Pack (writes FlashInfer-ready indices_buf in one pass)
# ---------------------------------------------------------------------------
@triton.jit
def _topk_sort_and_pack_kernel(
    scores_ptr,            # [bsz * num_kv_heads, num_pages] (flat, fp32)
    indices_buf_ptr,       # [bsz * num_kv_heads, page_budget] (flat, int32)
    last_page_idx_ptr,     # [bsz] int32
    recent_offsets_ptr,    # [NUM_RECENT] int32 (static: typically [-R, -R+1, ..., -1])
    s_stride_bh,
    o_stride_bh,
    num_pages,
    NUM_KV_HEADS: tl.constexpr,
    NUM_SINK_PAGES: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    NUM_RECENT: tl.constexpr,
    NUM_RECENT_PADDED: tl.constexpr,   # next_pow2(NUM_RECENT), >= 1
    SORT_ASCENDING: tl.constexpr,
):
    """One program per (batch, kv_head). Finds top-k page indices and writes
    the middle+recent regions of a preallocated `indices_buf` in one pass.

    Layout produced:
      indices_buf[b, h, 0                       : NUM_SINK_PAGES)            : NOT TOUCHED (static, written once at cache init)
      indices_buf[b, h, NUM_SINK_PAGES          : NUM_SINK_PAGES + TOP_K)    : topk(scores[b, h]) + NUM_SINK_PAGES
      indices_buf[b, h, NUM_SINK_PAGES + TOP_K  : NUM_SINK_PAGES + TOP_K + NUM_RECENT) : last_page_idx[b] + recent_offsets

    Top-k phase is identical to `_topk_sort_kernel`; only the write-back
    target and offset differ. Register footprint matches the unfused kernel
    (no new accumulators) — we only add a handful of int32 loads/adds/stores
    for the recent tail.
    """
    pid = tl.program_id(0)
    b = pid // NUM_KV_HEADS
    base_s = scores_ptr + pid * s_stride_bh
    base_o = indices_buf_ptr + pid * o_stride_bh

    p_idx = tl.arange(0, BLOCK_P)
    mask = p_idx < num_pages
    scores = tl.load(base_s + p_idx, mask=mask, other=float('-inf'))

    # --- Phase 1: Sort by score descending to find top-k ---
    bits = scores.to(tl.int32, bitcast=True)
    sortable = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    desc_key = -sortable
    packed = (desc_key.to(tl.int64) << 32) | p_idx.to(tl.int64)
    sorted_packed = tl.sort(packed)

    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    if SORT_ASCENDING:
        top_indices = tl.where(p_idx < TOP_K, sorted_indices, 2147483647)
        topk_out = tl.sort(top_indices)
    else:
        topk_out = sorted_indices

    # --- Phase 2: Pack write — topk region with +NUM_SINK_PAGES offset ---
    topk_with_offset = topk_out + NUM_SINK_PAGES
    tl.store(
        base_o + NUM_SINK_PAGES + p_idx,
        topk_with_offset,
        mask=p_idx < TOP_K,
    )

    # --- Phase 3: Pack write — recent region = last_page_idx + recent_offsets ---
    if NUM_RECENT > 0:
        last_idx = tl.load(last_page_idx_ptr + b)  # int32 scalar
        r_idx = tl.arange(0, NUM_RECENT_PADDED)
        r_mask = r_idx < NUM_RECENT
        offsets = tl.load(recent_offsets_ptr + r_idx, mask=r_mask, other=0)
        recent_vals = last_idx + offsets
        tl.store(
            base_o + (NUM_SINK_PAGES + TOP_K) + r_idx,
            recent_vals,
            mask=r_mask,
        )


@triton.jit
def _topk_merge_and_pack_kernel(
    scratch_ptr,           # [bsz * num_kv_heads, TOTAL] int64 packed
    indices_buf_ptr,       # [bsz * num_kv_heads, page_budget] int32
    last_page_idx_ptr,     # [bsz] int32
    recent_offsets_ptr,    # [NUM_RECENT] int32
    sc_stride_bh,
    o_stride_bh,
    NUM_KV_HEADS: tl.constexpr,
    NUM_SINK_PAGES: tl.constexpr,
    TOP_K: tl.constexpr,
    TOTAL: tl.constexpr,         # NUM_CHUNKS * TOP_K, must be pow2
    NUM_RECENT: tl.constexpr,
    NUM_RECENT_PADDED: tl.constexpr,
    SORT_ASCENDING: tl.constexpr,
):
    """Stage 2 (two-stage topk) with fused pack-indices write.

    Same layout as `_topk_sort_and_pack_kernel`; only the local-chunk topk
    staging differs — Stage 1 (`_topk_local_kernel`) is reused unchanged.
    """
    pid = tl.program_id(0)
    b = pid // NUM_KV_HEADS
    idx = tl.arange(0, TOTAL)

    sc_base = scratch_ptr + pid * sc_stride_bh
    packed = tl.load(sc_base + idx)
    sorted_packed = tl.sort(packed)

    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    if SORT_ASCENDING:
        top_indices = tl.where(idx < TOP_K, sorted_indices, 2147483647)
        topk_out = tl.sort(top_indices)
    else:
        topk_out = sorted_indices

    base_o = indices_buf_ptr + pid * o_stride_bh
    topk_with_offset = topk_out + NUM_SINK_PAGES
    tl.store(
        base_o + NUM_SINK_PAGES + idx,
        topk_with_offset,
        mask=idx < TOP_K,
    )

    if NUM_RECENT > 0:
        last_idx = tl.load(last_page_idx_ptr + b)
        r_idx = tl.arange(0, NUM_RECENT_PADDED)
        r_mask = r_idx < NUM_RECENT
        offsets = tl.load(recent_offsets_ptr + r_idx, mask=r_mask, other=0)
        recent_vals = last_idx + offsets
        tl.store(
            base_o + (NUM_SINK_PAGES + TOP_K) + r_idx,
            recent_vals,
            mask=r_mask,
        )


def topk_sort_and_pack_triton(
    page_scores: torch.Tensor,           # (bsz, num_kv_heads, num_pages) fp32
    indices_buf: torch.Tensor,           # (bsz, num_kv_heads, page_budget) int32
    num_sink_pages: int,
    top_k: int,
    last_page_idx: torch.Tensor,         # (bsz,) int32
    recent_offsets: torch.Tensor,        # (num_recent,) int32 (may be empty)
    scratch: torch.Tensor = None,        # (bsz, num_kv_heads, >= 8*top_k) int64, for two-stage
    sort_ascending: bool = False,
    num_chunks: int = 8,
) -> torch.Tensor:
    """Fused top-k + pack-indices write. Replaces `topk_sort(...)` followed by
    a Python `pack_indices(...)` call that copies/offsets into a preallocated
    indices buffer.

    Layout written into `indices_buf[b, h, :]`:
      [0, num_sink_pages)                              : NOT TOUCHED — static, caller writes once
      [num_sink_pages, num_sink_pages + top_k)         : topk indices + num_sink_pages
      [num_sink_pages + top_k, page_budget)            : last_page_idx[b] + recent_offsets

    `indices_buf` must be shape (bsz, num_kv_heads, page_budget) int32 where
    `page_budget == num_sink_pages + top_k + recent_offsets.numel()`. The
    caller is responsible for pre-filling the sink region (e.g. at cache
    init) — doing it here would waste a store on every decode step.

    `sort_ascending=False` is the drop-mode default (order-invariant middle
    attention). With `sort_ascending=True` the topk slice is sorted ascending
    by page index — matches `topk_sort(sort_ascending=True)` semantics for
    bit-identical comparison against sequential topk + pack.

    Dispatch: single-stage for `num_pages <= 1024`; two-stage (reusing
    `_topk_local_kernel`) above that, same threshold as `topk_sort`.
    """
    bsz, num_kv_heads, num_pages = page_scores.shape
    page_budget = indices_buf.shape[-1]
    num_recent = int(recent_offsets.numel())

    assert indices_buf.shape == (bsz, num_kv_heads, page_budget), (
        f"indices_buf shape {tuple(indices_buf.shape)} does not match "
        f"(bsz={bsz}, num_kv_heads={num_kv_heads}, page_budget={page_budget})"
    )
    assert indices_buf.dtype == torch.int32, "indices_buf must be int32"
    assert page_budget == num_sink_pages + top_k + num_recent, (
        f"page_budget ({page_budget}) != num_sink_pages ({num_sink_pages}) "
        f"+ top_k ({top_k}) + num_recent ({num_recent})"
    )
    assert last_page_idx.shape == (bsz,) and last_page_idx.dtype == torch.int32, (
        "last_page_idx must be shape (bsz,) int32"
    )
    if num_recent > 0:
        assert recent_offsets.dtype == torch.int32, "recent_offsets must be int32"

    is_pow2_top_k = top_k > 0 and (top_k & (top_k - 1)) == 0
    use_twostage = num_pages >= _TOPK_TWOSTAGE_MIN_PAGES and is_pow2_top_k

    s_stride_bh = page_scores.stride(1)
    o_stride_bh = indices_buf.stride(1)
    # next_pow2(max(NUM_RECENT, 1)) so tl.arange is always valid.
    NUM_RECENT_PADDED = 1 if num_recent == 0 else triton.next_power_of_2(num_recent)

    if use_twostage:
        per_chunk = max(1, (num_pages + num_chunks - 1) // num_chunks)
        CHUNK_SIZE = max(top_k, triton.next_power_of_2(per_chunk))
        TOTAL = num_chunks * top_k

        if scratch is None:
            scratch = torch.empty(
                bsz, num_kv_heads, TOTAL,
                dtype=torch.int64, device=page_scores.device,
            )
        sc_stride_bh = scratch.stride(1)

        grid1 = (bsz * num_kv_heads, num_chunks)
        grid2 = (bsz * num_kv_heads,)

        with torch.cuda.device(page_scores.device):
            _topk_local_kernel[grid1](
                page_scores, scratch,
                s_stride_bh, sc_stride_bh,
                num_pages,
                TOP_K=top_k,
                CHUNK_SIZE=CHUNK_SIZE,
                num_warps=1,
                num_stages=1,
            )
            _topk_merge_and_pack_kernel[grid2](
                scratch, indices_buf,
                last_page_idx, recent_offsets,
                sc_stride_bh, o_stride_bh,
                NUM_KV_HEADS=num_kv_heads,
                NUM_SINK_PAGES=num_sink_pages,
                TOP_K=top_k,
                TOTAL=TOTAL,
                NUM_RECENT=num_recent,
                NUM_RECENT_PADDED=NUM_RECENT_PADDED,
                SORT_ASCENDING=sort_ascending,
                num_warps=1,
                num_stages=1,
            )
    else:
        BLOCK_P = max(top_k, triton.next_power_of_2(num_pages))
        grid = (bsz * num_kv_heads,)

        with torch.cuda.device(page_scores.device):
            _topk_sort_and_pack_kernel[grid](
                page_scores, indices_buf,
                last_page_idx, recent_offsets,
                s_stride_bh, o_stride_bh,
                num_pages,
                NUM_KV_HEADS=num_kv_heads,
                NUM_SINK_PAGES=num_sink_pages,
                TOP_K=top_k,
                BLOCK_P=BLOCK_P,
                NUM_RECENT=num_recent,
                NUM_RECENT_PADDED=NUM_RECENT_PADDED,
                SORT_ASCENDING=sort_ascending,
                num_warps=1,
                num_stages=1,
            )

    return indices_buf


# ---------------------------------------------------------------------------
# Kernel 2a: Copy sink + selected pages + recent (+ fused Q-RoPE)
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

    # Single if/elif/else for ALL per-segment values (one merge point).
    # sel_indices is only needed on page tiles — defer the load until then.
    if is_sink:
        t_start = pid_tile * BLOCK_T
        num_tokens = sink_len
        write_start = 0
        page_idx = tl.zeros([], dtype=tl.int32)
    elif is_recent:
        t_start = (pid_tile - page_end) * BLOCK_T
        num_tokens = recent_len
        write_start = total_len - recent_len
        page_idx = tl.zeros([], dtype=tl.int32)
    else:
        page_idx = tl.load(sel_indices_ptr + b * si_stride_b + h * si_stride_h + rank).to(tl.int32)
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
# Kernel 2b: Copy unselected compressed pages (batched: multiple pages per program)
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
    # --- population bias output [bsz, kv_heads, total_len] (dummy if WRITE_BIAS=0) ---
    bias_out_ptr,
    # --- strides for comp [bsz, kv_heads, num_pages, comp_size, head_dim] ---
    comp_stride_b, comp_stride_h, comp_stride_p, comp_stride_t,
    # --- strides for output [bsz, kv_heads, total_len, head_dim] ---
    out_stride_b, out_stride_h, out_stride_t,
    # --- strides for cos/sin [total_len, head_dim] ---
    rope_stride_t,
    # --- strides for sel_indices [bsz, kv_heads, top_k] ---
    sel_stride_b, sel_stride_h,
    # --- strides for bias_out [bsz, kv_heads, total_len] ---
    bias_stride_b, bias_stride_h, bias_stride_t,
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
    WRITE_BIAS: tl.constexpr,
    LOG_POP_WEIGHT: tl.constexpr,
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
    if WRITE_BIAS == 1:
        bias_dst = bias_out_ptr + b * bias_stride_b + h * bias_stride_h
        bias_t_mask = t_idx < COMP_SIZE

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

                if WRITE_BIAS == 1:
                    bias_offsets = (write_start + t_safe) * bias_stride_t
                    bias_vals = tl.full([BLOCK_T], LOG_POP_WEIGHT, dtype=tl.float32)
                    tl.store(bias_dst + bias_offsets, bias_vals, mask=bias_t_mask)


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
    stride_cache: dict = None,
    bias_out: torch.Tensor = None,
    log_pop_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble KV using split kernels.

    Kernel A: sink + selected pages + recent (all inside Triton)
    Kernel B: unselected compressed pages (branch-free)
    No PyTorch tensor ops between kernel launches.

    Optional pre-allocated buffers (out_k, out_v, out_sel_idx) can be passed
    to avoid per-step memory allocation.

    If stride_cache is provided, skips stride computations but recomputes
    shape-dependent values (num_pages, total_len, grids) each call.
    Use build_assemble_stride_cache() to create. Safe with changing seq lengths.
    """
    if stride_cache is not None:
        return _assemble_kv_split_stride_cached(
            paged_k, paged_v, comp_k, comp_v,
            sink_k, sink_v, recent_k, recent_v,
            selected_indices, cos_table, sin_table,
            out_k, out_v, query_states, q_rope_cos, q_rope_sin, q_rope_buf,
            stride_cache,
            bias_out=bias_out, log_pop_weight=log_pop_weight,
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
            write_bias = 1 if bias_out is not None else 0
            if write_bias == 1:
                bias_b_stride = bias_out.stride(0)
                bias_h_stride = bias_out.stride(1)
                bias_t_stride = bias_out.stride(2)
                bias_ptr = bias_out
            else:
                bias_b_stride = 0
                bias_h_stride = 0
                bias_t_stride = 0
                bias_ptr = final_k  # dummy, not accessed when WRITE_BIAS=0
            _copy_unselected_pages_kernel[grid_comp](
                comp_k, comp_v, final_k, final_v,
                cos_ptr, sin_ptr,
                sel_idx,
                bias_ptr,
                comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3),
                final_k.stride(0), final_k.stride(1), final_k.stride(2),
                rope_stride_t,
                sel_idx.stride(0), sel_idx.stride(1),
                bias_b_stride, bias_h_stride, bias_t_stride,
                num_kv_heads, num_pages, head_dim, sink_len, num_groups,
                COMP_SIZE=comp_size, PAGE_SIZE=page_size, TOP_K=top_k,
                BLOCK_K=triton.next_power_of_2(top_k),
                PAGES_PER_PROG=PAGES_PER_PROG,
                BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_COMP,
                APPLY_ROPE=1 if apply_rope else 0,
                WRITE_BIAS=write_bias,
                LOG_POP_WEIGHT=float(log_pop_weight),
            )

    if fuse_q_rope:
        return final_k, final_v, q_rope_buf
    return final_k, final_v


def build_assemble_stride_cache(
    paged_k, comp_k, sink_k, recent_k, selected_indices=None,
    cos_table=None, out_k=None, query_states=None,
    bias_out=None,
):
    """Precompute tensor strides and fixed constants only.

    Shape-dependent values (num_pages, total_len, grids) are recomputed
    each call, so this cache is safe to reuse across all decode steps.
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
        # Bias output strides (only meaningful when caller passes bias_out at call time)
        'bias_strides': (bias_out.stride(0), bias_out.stride(1), bias_out.stride(2)) if bias_out is not None else (0, 0, 0),
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
    bias_out=None, log_pop_weight=0.0,
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
            write_bias = 1 if bias_out is not None else 0
            if write_bias == 1:
                bs = c.get('bias_strides', (bias_out.stride(0), bias_out.stride(1), bias_out.stride(2)))
                bias_ptr = bias_out
            else:
                bs = (0, 0, 0)
                bias_ptr = final_k  # dummy, not accessed
            _copy_unselected_pages_kernel[grid_comp](
                comp_k, comp_v, final_k, final_v,
                cos_ptr, sin_ptr,
                sel_idx,
                bias_ptr,
                cs[0], cs[1], cs[2], cs[3],
                os[0], os[1], os[2],
                c['rope_stride_t'],
                si[0], si[1],
                bs[0], bs[1], bs[2],
                c['num_kv_heads'], num_pages, head_dim, c['sink_len'], num_groups,
                COMP_SIZE=c['comp_size'], PAGE_SIZE=c['page_size'], TOP_K=top_k,
                BLOCK_K=triton.next_power_of_2(top_k),
                PAGES_PER_PROG=c['PAGES_PER_PROG'],
                BLOCK_D=c['BLOCK_D'], BLOCK_T=c['BLOCK_T_COMP'],
                APPLY_ROPE=c['APPLY_ROPE'],
                WRITE_BIAS=write_bias,
                LOG_POP_WEIGHT=float(log_pop_weight),
            )

    if c['fuse_q_rope']:
        return final_k, final_v, q_rope_buf
    return final_k, final_v


# ---------------------------------------------------------------------------
# Wrapper: Drop-mode assemble (Kernel A only, COMP_SIZE=0)
# ---------------------------------------------------------------------------
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
        BLOCK_T_FULL = min(128, triton.next_power_of_2(page_size))
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


def apply_rope_q_direct(
    query: torch.Tensor,
    cos_flat: torch.Tensor,
    sin_flat: torch.Tensor,
    out_buf: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to decode query — zero-overhead wrapper.

    Expects pre-flattened cos/sin and a pre-allocated output buffer
    to avoid .contiguous() copies and torch.empty_like allocation on
    every call.

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


# ---------------------------------------------------------------------------
# Kernel 4: DCT page compression (replaces torch.einsum('cs,bhnsd->bhncd', ...))
# ---------------------------------------------------------------------------
@triton.jit
def _compress_pages_kernel(
    page_ptr,   # [bsz, kv_heads, n_new, PAGE_SIZE, head_dim]
    m_ptr,      # [COMP_SIZE, PAGE_SIZE]
    out_ptr,    # [bsz, kv_heads, n_new, COMP_SIZE, head_dim]
    # page strides (outer b, h, n, t dims; head_dim is contiguous)
    p_stride_b, p_stride_h, p_stride_n, p_stride_t,
    # projection matrix stride
    m_stride_c,
    # output strides
    o_stride_b, o_stride_h, o_stride_n, o_stride_c,
    # runtime dim
    num_kv_heads,
    head_dim,
    # constexprs
    PAGE_SIZE: tl.constexpr,
    COMP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Project one page per program via the precomputed DCT-lowpass-IDCT matrix.

    Grid: (bsz * num_kv_heads, n_new).
    Computes out[b, h, n, c, d] = sum_t(M[c, t] * page[b, h, n, t, d]).
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_bh // num_kv_heads
    h = pid_bh % num_kv_heads

    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < head_dim
    c_idx = tl.arange(0, COMP_SIZE)

    page_base = page_ptr + b * p_stride_b + h * p_stride_h + pid_n * p_stride_n

    # Accumulate [COMP_SIZE, BLOCK_D] in fp32. Unroll t-loop over PAGE_SIZE
    # (constexpr) so both M and page rows live in registers.
    out = tl.zeros([COMP_SIZE, BLOCK_D], dtype=tl.float32)
    for t in tl.static_range(PAGE_SIZE):
        # page_row[d] = page[b, h, pid_n, t, d]  ->  [BLOCK_D]
        page_row = tl.load(
            page_base + t * p_stride_t + d_idx,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        # m_col[c] = M[c, t]  ->  [COMP_SIZE]
        m_col = tl.load(m_ptr + c_idx * m_stride_c + t).to(tl.float32)
        out += m_col[:, None] * page_row[None, :]

    out_base = out_ptr + b * o_stride_b + h * o_stride_h + pid_n * o_stride_n
    out_offsets = c_idx[:, None] * o_stride_c + d_idx[None, :]
    out_mask = d_mask[None, :]
    tl.store(out_base + out_offsets, out.to(out_ptr.dtype.element_ty), mask=out_mask)


def compress_pages_triton(
    paged_x: torch.Tensor,  # [bsz, kv_heads, n_new, page_size, head_dim]
    M: torch.Tensor,        # [comp_size, page_size]
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Apply the DCT-lowpass-IDCT projection M to each page of paged_x.

    Equivalent to torch.einsum('cs,bhnsd->bhncd', M, paged_x) but avoids the
    einsum launch-dispatch cost on each new-page step.

    Shapes:
        paged_x: [bsz, kv_heads, n_new, page_size, head_dim]
        M:       [comp_size, page_size]
    Returns:
        [bsz, kv_heads, n_new, comp_size, head_dim] in paged_x.dtype.
    """
    bsz, num_kv_heads, n_new, page_size, head_dim = paged_x.shape
    comp_size = M.shape[0]
    assert M.shape[1] == page_size
    assert paged_x.stride(-1) == 1, "compress_pages_triton expects contiguous head_dim"

    if out is None:
        out = torch.empty(
            bsz, num_kv_heads, n_new, comp_size, head_dim,
            dtype=paged_x.dtype, device=paged_x.device,
        )
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (bsz * num_kv_heads, n_new)

    with torch.cuda.device(paged_x.device):
        _compress_pages_kernel[grid](
            paged_x, M, out,
            paged_x.stride(0), paged_x.stride(1), paged_x.stride(2), paged_x.stride(3),
            M.stride(0),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            num_kv_heads,
            head_dim,
            PAGE_SIZE=page_size,
            COMP_SIZE=comp_size,
            BLOCK_D=BLOCK_D,
        )
    return out
