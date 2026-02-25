"""
Fused Triton kernels for DCT Page Attention.

_assemble_kv_compressed_kernel: replaces the ~20-kernel PyTorch scatter/gather
pipeline in assemble_kv (compressed mode) with a single fused kernel that
handles both K and V in one launch.  Uses a 2D grid (page × token-tile)
so every SM stays busy.

_score_pages_fused_kernel: replaces the einsum → reduce → aggregate → topk
pipeline in score_pages with a single fused kernel (scoring) + small PyTorch
topk+sort.  Eliminates the large intermediate scores tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _assemble_kv_compressed_kernel(
    # --- K tensor pointers ---
    paged_k_ptr, comp_k_ptr, out_k_ptr,
    # --- V tensor pointers ---
    paged_v_ptr, comp_v_ptr, out_v_ptr,
    # --- selected indices ---
    selected_indices_ptr,
    # --- strides for paged (shared by K and V — same shape) ---
    paged_stride_b, paged_stride_h, paged_stride_p, paged_stride_t,
    # --- strides for comp (shared by K and V) ---
    comp_stride_b, comp_stride_h, comp_stride_p, comp_stride_t,
    # --- strides for out (shared by K and V) ---
    out_stride_b, out_stride_h, out_stride_t,
    # --- strides for selected_indices ---
    sel_stride_b, sel_stride_h,
    # --- runtime dimensions ---
    num_kv_heads,
    num_pages,
    head_dim,
    # --- compile-time constants ---
    PAGE_SIZE: tl.constexpr,
    COMP_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,    # >= TOP_K, power-of-2
    BLOCK_D: tl.constexpr,    # >= head_dim, power-of-2
    BLOCK_T: tl.constexpr,    # tokens per tile (e.g. 32)
):
    """One program per (batch, kv_head, page, token-tile).

    Each program copies a [BLOCK_T, head_dim] tile for BOTH K and V from
    either paged (selected) or comp (unselected) into the correct
    interleaved position.  Programs whose tile falls outside the page's
    token count early-exit.
    """
    # ---- axis 0 → (b, h, p);  axis 1 → token tile ----
    pid_bhp = tl.program_id(0)
    pid_tile = tl.program_id(1)

    p = pid_bhp % num_pages
    tmp = pid_bhp // num_pages
    h = tmp % num_kv_heads
    b = tmp // num_kv_heads

    # ---- load selected indices for this (b, h) ----
    sel_base = selected_indices_ptr + b * sel_stride_b + h * sel_stride_h
    k_idx = tl.arange(0, BLOCK_K)
    k_mask = k_idx < TOP_K
    sel_indices = tl.load(sel_base + k_idx, mask=k_mask, other=num_pages)

    # ---- selection status & write-offset ----
    is_selected = tl.sum((sel_indices == p).to(tl.int32)) > 0
    count_before = tl.sum((sel_indices < p).to(tl.int32))
    write_start = p * COMP_SIZE + count_before * (PAGE_SIZE - COMP_SIZE)

    # ---- early exit for tiles beyond this page's token count ----
    num_tokens = tl.where(is_selected, PAGE_SIZE, COMP_SIZE)
    t_start = pid_tile * BLOCK_T
    if t_start >= num_tokens:
        return

    # ---- 2-D tile indices: [BLOCK_T, BLOCK_D] ----
    t_idx = tl.arange(0, BLOCK_T)
    d_idx = tl.arange(0, BLOCK_D)
    mask = ((t_start + t_idx) < num_tokens)[:, None] & (d_idx < head_dim)[None, :]

    # ---- destination offsets (same layout for K and V) ----
    dst_offsets = t_idx[:, None] * out_stride_t + d_idx[None, :]
    dst_off = b * out_stride_b + h * out_stride_h + (write_start + t_start) * out_stride_t
    k_dst_base = out_k_ptr + dst_off
    v_dst_base = out_v_ptr + dst_off

    # ---- load from correct source & store (K and V together) ----
    if is_selected:
        src_offsets = t_idx[:, None] * paged_stride_t + d_idx[None, :]
        src_off = (b * paged_stride_b + h * paged_stride_h
                   + p * paged_stride_p + t_start * paged_stride_t)

        k_vals = tl.load(paged_k_ptr + src_off + src_offsets, mask=mask, other=0.0)
        tl.store(k_dst_base + dst_offsets, k_vals, mask=mask)
        v_vals = tl.load(paged_v_ptr + src_off + src_offsets, mask=mask, other=0.0)
        tl.store(v_dst_base + dst_offsets, v_vals, mask=mask)
    else:
        src_offsets = t_idx[:, None] * comp_stride_t + d_idx[None, :]
        src_off = (b * comp_stride_b + h * comp_stride_h
                   + p * comp_stride_p + t_start * comp_stride_t)

        k_vals = tl.load(comp_k_ptr + src_off + src_offsets, mask=mask, other=0.0)
        tl.store(k_dst_base + dst_offsets, k_vals, mask=mask)
        v_vals = tl.load(comp_v_ptr + src_off + src_offsets, mask=mask, other=0.0)
        tl.store(v_dst_base + dst_offsets, v_vals, mask=mask)


def assemble_kv_compressed_triton(
    paged_k: torch.Tensor,
    paged_v: torch.Tensor,
    comp_k: torch.Tensor,
    comp_v: torch.Tensor,
    selected_indices: torch.Tensor,
    num_pages: int,
    page_size: int,
    comp_size: int,
    top_k: int,
    sink_k: torch.Tensor = None,
    sink_v: torch.Tensor = None,
    recent_k: torch.Tensor = None,
    recent_v: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated compressed-mode KV assembly.

    Single fused kernel handles both K and V, eliminating the CPU dispatch
    gap between separate launches.

    When sink/recent tensors are provided, the kernel writes the middle
    portion directly into the pre-allocated final output tensor (via a view),
    eliminating both the intermediate middle_k/v allocation and the torch.cat.

    Args:
        paged_k/v:  [bsz, num_kv_heads, num_pages, page_size, head_dim]
        comp_k/v:   [bsz, num_kv_heads, num_pages, comp_size, head_dim]
        selected_indices: [bsz, num_kv_heads, top_k]  (sorted page indices)
        num_pages, page_size, comp_size, top_k: scalar dimensions
        sink_k/v:   [bsz, num_kv_heads, sink_len, head_dim]  (optional)
        recent_k/v: [bsz, num_kv_heads, recent_len, head_dim] (optional)

    Returns:
        If sink/recent provided: final_k, final_v (includes sink+middle+recent)
        Otherwise: middle_k, middle_v
    """
    bsz, num_kv_heads = paged_k.shape[:2]
    head_dim = paged_k.shape[-1]
    num_unselected = num_pages - top_k
    middle_len = top_k * page_size + num_unselected * comp_size

    # The kernel uses d_idx with implicit stride=1 for the head_dim dimension.
    # DCT compression can produce non-contiguous comp tensors (transposed last
    # two dims), so we must ensure stride(-1)==1 before launching the kernel.
    if not comp_k.is_contiguous():
        comp_k = comp_k.contiguous()
    if not comp_v.is_contiguous():
        comp_v = comp_v.contiguous()

    if sink_k is not None:
        # Fused path: write directly into pre-allocated final tensor
        sink_len = sink_k.shape[2]
        recent_len = recent_k.shape[2]
        total_len = sink_len + middle_len + recent_len

        final_k = torch.empty(
            bsz, num_kv_heads, total_len, head_dim,
            dtype=paged_k.dtype, device=paged_k.device,
        )
        final_v = torch.empty_like(final_k)

        # Copy sink and recent (tiny — 4 + 128 tokens)
        final_k[:, :, :sink_len, :] = sink_k
        final_v[:, :, :sink_len, :] = sink_v
        final_k[:, :, sink_len + middle_len:, :] = recent_k
        final_v[:, :, sink_len + middle_len:, :] = recent_v

        # View into middle portion — shares storage, kernel writes here directly
        out_k = final_k[:, :, sink_len:sink_len + middle_len, :]
        out_v = final_v[:, :, sink_len:sink_len + middle_len, :]
    else:
        # Original path: allocate standalone middle tensors
        out_k = torch.empty(
            bsz, num_kv_heads, middle_len, head_dim,
            dtype=paged_k.dtype, device=paged_k.device,
        )
        out_v = torch.empty_like(out_k)

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_K = triton.next_power_of_2(top_k)
    BLOCK_T = 32

    # 2-D grid: axis 0 = (batch, head, page), axis 1 = token tile
    num_tiles = (page_size + BLOCK_T - 1) // BLOCK_T
    grid = (bsz * num_kv_heads * num_pages, num_tiles)

    # Single launch for both K and V
    _assemble_kv_compressed_kernel[grid](
        paged_k, comp_k, out_k,
        paged_v, comp_v, out_v,
        selected_indices,
        paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3),
        comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3),
        out_k.stride(0), out_k.stride(1), out_k.stride(2),
        selected_indices.stride(0), selected_indices.stride(1),
        num_kv_heads, num_pages, head_dim,
        PAGE_SIZE=page_size, COMP_SIZE=comp_size, TOP_K=top_k,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T,
    )

    if sink_k is not None:
        return final_k, final_v
    return out_k, out_v


# ---------------------------------------------------------------------------
# Fused page scoring kernel
# ---------------------------------------------------------------------------
@triton.jit
def _score_pages_fused_kernel(
    # --- pointers ---
    query_ptr,         # [bsz, num_kv_heads, num_kv_groups, head_dim]
    comp_keys_ptr,     # [bsz, num_kv_heads, num_pages, comp_size, head_dim]
    out_scores_ptr,    # [bsz, num_kv_heads, num_pages]
    # --- query strides (head_dim dim is contiguous, stride=1) ---
    q_stride_b, q_stride_h, q_stride_g,
    # --- comp keys strides (head_dim dim is contiguous, stride=1) ---
    ck_stride_b, ck_stride_h, ck_stride_p, ck_stride_c,
    # --- output strides ---
    os_stride_b, os_stride_h,
    # --- runtime dims ---
    num_kv_heads,
    num_pages,
    head_dim,
    scaling,           # float: head_dim ** -0.5
    # --- compile-time constants ---
    NUM_KV_GROUPS: tl.constexpr,
    COMP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,        # >= head_dim, power-of-2
    BLOCK_P: tl.constexpr,        # pages per program (tiling factor)
    SCORING_METHOD: tl.constexpr,  # 0=max, 1=mean, 2=sum
    GROUP_AGG_METHOD: tl.constexpr,  # 0=mean, 1=max
):
    """One program per (batch, kv_head, page_tile).

    Comp tokens are the OUTER loop so each key tile [BLOCK_P, BLOCK_D] is
    loaded only once (not NUM_KV_GROUPS times).  GQA groups are the INNER
    loop; queries are small ([BLOCK_D]) and hit L1 cache.  Per-group
    accumulators are kept in a [NUM_KV_GROUPS, BLOCK_P] 2D tensor.
    All arithmetic is done in float32 for numerical stability.
    """
    # Grid: axis 0 = (batch, kv_head),  axis 1 = page tile
    pid_bh = tl.program_id(0)
    pid_pt = tl.program_id(1)

    h = pid_bh % num_kv_heads
    b = pid_bh // num_kv_heads

    # Page indices for this tile
    p_start = pid_pt * BLOCK_P
    p_offsets = tl.arange(0, BLOCK_P)
    p_indices = p_start + p_offsets
    p_mask = p_indices < num_pages

    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < head_dim

    q_base = query_ptr + b * q_stride_b + h * q_stride_h
    ck_base = comp_keys_ptr + b * ck_stride_b + h * ck_stride_h

    # Group index vector for masked 2D updates
    g_idx = tl.arange(0, NUM_KV_GROUPS)[:, None]  # [NUM_KV_GROUPS, 1]

    # Per-group page scores: [NUM_KV_GROUPS, BLOCK_P]
    if SCORING_METHOD == 0:     # max
        group_ps = tl.full([NUM_KV_GROUPS, BLOCK_P], float('-inf'), dtype=tl.float32)
    else:                       # mean or sum
        group_ps = tl.zeros([NUM_KV_GROUPS, BLOCK_P], dtype=tl.float32)

    # ---- Outer loop: comp tokens (keys loaded ONCE per token) ----
    for c in range(COMP_SIZE):
        # Load key tile for BLOCK_P pages at comp token c: [BLOCK_P, BLOCK_D]
        k_ptrs = ck_base + p_indices[:, None] * ck_stride_p + c * ck_stride_c + d_idx[None, :]
        mask_2d = p_mask[:, None] & d_mask[None, :]
        k = tl.load(k_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

        # ---- Inner loop: GQA groups (queries are small, hit L1) ----
        for g in range(NUM_KV_GROUPS):
            q = tl.load(q_base + g * q_stride_g + d_idx, mask=d_mask, other=0.0).to(tl.float32)
            q = q * scaling

            # Batched dot product: [BLOCK_P, BLOCK_D] · [BLOCK_D] → [BLOCK_P]
            dots = tl.sum(k * q[None, :], axis=1)  # [BLOCK_P]

            # Update only row g of the [NUM_KV_GROUPS, BLOCK_P] accumulator
            g_mask = (g_idx == g)  # [NUM_KV_GROUPS, 1], broadcasts to [NUM_KV_GROUPS, BLOCK_P]
            if SCORING_METHOD == 0:     # max
                group_ps = tl.where(g_mask, tl.maximum(group_ps, dots[None, :]), group_ps)
            else:                       # mean or sum
                group_ps = tl.where(g_mask, group_ps + dots[None, :], group_ps)

    # ---- Finalize per-group scores ----
    if SCORING_METHOD == 1:     # mean
        group_ps = group_ps / COMP_SIZE

    # ---- Aggregate across groups: [NUM_KV_GROUPS, BLOCK_P] → [BLOCK_P] ----
    if GROUP_AGG_METHOD == 0:   # mean
        agg_scores = tl.sum(group_ps, axis=0) / NUM_KV_GROUPS
    else:                       # max
        agg_scores = tl.max(group_ps, axis=0)

    # Store scores for all valid pages in this tile
    out_ptrs = out_scores_ptr + b * os_stride_b + h * os_stride_h + p_indices
    tl.store(out_ptrs, agg_scores, mask=p_mask)


def score_pages_fused_triton(
    query_states: torch.Tensor,
    compressed_keys: torch.Tensor,
    scoring_method: str,
    group_agg_method: str,
    top_k: int,
    num_kv_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused page scoring via Triton + PyTorch top-k.

    Replaces the einsum → reduce → aggregate → topk → sort pipeline in
    score_pages() with a single Triton kernel (scoring) followed by small
    PyTorch ops (topk, sort).  Eliminates the large intermediate scores
    tensor entirely.

    Supports scoring_method in {"max", "mean", "sum"} and
    group_agg_method in {"mean", "max"}.

    Args:
        query_states:    [bsz, num_heads, 1, head_dim]  (decode only)
        compressed_keys: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
        scoring_method:  "max" | "mean" | "sum"
        group_agg_method: "mean" | "max"
        top_k: int
        num_kv_groups: int

    Returns:
        selected_indices: [bsz, num_kv_heads, actual_top_k]  (sorted)
        page_scores:      [bsz, num_kv_heads, num_pages]     (float32)
    """
    bsz, _num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, num_pages, comp_size, _ = compressed_keys.shape
    scaling = head_dim ** -0.5

    assert q_len == 1, "Fused score_pages only supports decode (q_len=1)"

    # Reshape query: [bsz, num_heads, 1, head_dim] → [bsz, num_kv_heads, num_kv_groups, head_dim]
    query = query_states.squeeze(2).reshape(bsz, num_kv_heads, num_kv_groups, head_dim).contiguous()
    compressed_keys = compressed_keys.contiguous()

    # Allocate output (float32 for precision)
    page_scores = torch.empty(
        bsz, num_kv_heads, num_pages,
        dtype=torch.float32, device=query.device,
    )

    SCORING = {"max": 0, "mean": 1, "sum": 2}[scoring_method]
    GROUP_AGG = {"mean": 0, "max": 1}[group_agg_method]
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_P = 32  # pages per program — amortizes query load across tile

    num_page_tiles = (num_pages + BLOCK_P - 1) // BLOCK_P
    grid = (bsz * num_kv_heads, num_page_tiles)

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

    # Top-k + sort on the small [bsz, num_kv_heads, num_pages] tensor
    actual_top_k = min(top_k, num_pages)
    _, selected_indices = torch.topk(page_scores, actual_top_k, dim=-1)
    selected_indices, _ = selected_indices.sort(dim=-1)

    return selected_indices, page_scores
