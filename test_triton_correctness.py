"""
Diagnostic: compare Triton kernels vs PyTorch reference paths.

Run:  python test_triton_correctness.py
"""

import torch
import sys

# ---- Config matching eval defaults ----
BSZ = 1
NUM_KV_HEADS = 8
NUM_KV_GROUPS = 4   # Llama-3.1-8B: 32 heads / 8 kv_heads
NUM_HEADS = NUM_KV_HEADS * NUM_KV_GROUPS
HEAD_DIM = 128
PAGE_SIZE = 128
COMP_SIZE = 3        # int(128 * 0.03) = 3
NUM_PAGES = 244      # typical for ~31K input
TOP_K = 8
SCORING_METHOD = "mean"
GROUP_AGG_METHOD = "max"

DEVICE = "cuda"
DTYPE = torch.bfloat16


def test_score_pages():
    """Compare score_pages: Triton vs PyTorch reference."""
    print("=" * 60)
    print("TEST: score_pages")
    print("=" * 60)

    torch.manual_seed(42)

    # query_states: [bsz, num_heads, 1, head_dim]
    query_states = torch.randn(BSZ, NUM_HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    # compressed_keys: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
    compressed_keys = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM,
                                  device=DEVICE, dtype=DTYPE)

    # ---- PyTorch reference (from dct_page_attention.py score_pages) ----
    scaling = HEAD_DIM ** -0.5
    query_grouped = query_states.view(BSZ, NUM_KV_HEADS, NUM_KV_GROUPS, 1, HEAD_DIM)
    scores = torch.einsum(
        'bgiqd,bgncd->bgiqnc',
        query_grouped * scaling, compressed_keys
    )
    if SCORING_METHOD == "mean":
        page_scores_ref = scores.mean(dim=-1)
    elif SCORING_METHOD == "max":
        page_scores_ref = scores.max(dim=-1).values
    elif SCORING_METHOD == "sum":
        page_scores_ref = scores.sum(dim=-1)

    page_scores_ref = page_scores_ref.mean(dim=3)  # mean over q_len

    if GROUP_AGG_METHOD == "mean":
        page_scores_ref = page_scores_ref.mean(dim=2)
    elif GROUP_AGG_METHOD == "max":
        page_scores_ref = page_scores_ref.max(dim=2).values

    actual_top_k = min(TOP_K, NUM_PAGES)
    _, selected_ref = torch.topk(page_scores_ref, actual_top_k, dim=-1)
    selected_ref, _ = selected_ref.sort(dim=-1)

    # ---- Triton path ----
    from triton_kernels import score_pages_fused_triton
    selected_tri, page_scores_tri = score_pages_fused_triton(
        query_states, compressed_keys,
        SCORING_METHOD, GROUP_AGG_METHOD, TOP_K, NUM_KV_GROUPS,
    )

    # ---- Compare ----
    score_diff = (page_scores_ref.float() - page_scores_tri.float()).abs()
    max_score_diff = score_diff.max().item()
    mean_score_diff = score_diff.mean().item()

    print(f"  Page scores  max|diff|: {max_score_diff:.6f}")
    print(f"  Page scores mean|diff|: {mean_score_diff:.6f}")

    indices_match = torch.equal(selected_ref, selected_tri)
    print(f"  Selected indices match: {indices_match}")

    if not indices_match:
        for h in range(NUM_KV_HEADS):
            ref = selected_ref[0, h].tolist()
            tri = selected_tri[0, h].tolist()
            if ref != tri:
                print(f"    Head {h}: ref={ref}  triton={tri}")

    score_ok = max_score_diff < 0.01
    print(f"  RESULT: {'PASS' if score_ok and indices_match else 'FAIL'}")
    return score_ok and indices_match


def test_assemble_kv():
    """Compare assemble_kv: Triton vs PyTorch reference."""
    print()
    print("=" * 60)
    print("TEST: assemble_kv (compressed mode)")
    print("=" * 60)

    torch.manual_seed(42)

    paged_k = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, PAGE_SIZE, HEAD_DIM,
                           device=DEVICE, dtype=DTYPE)
    paged_v = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, PAGE_SIZE, HEAD_DIM,
                           device=DEVICE, dtype=DTYPE)
    comp_k = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM,
                          device=DEVICE, dtype=DTYPE)
    comp_v = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM,
                          device=DEVICE, dtype=DTYPE)
    sink_k = torch.randn(BSZ, NUM_KV_HEADS, 4, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    sink_v = torch.randn(BSZ, NUM_KV_HEADS, 4, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    recent_k = torch.randn(BSZ, NUM_KV_HEADS, 173, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    recent_v = torch.randn(BSZ, NUM_KV_HEADS, 173, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    # Make some sorted selected indices
    selected_indices = torch.sort(
        torch.randperm(NUM_PAGES, device=DEVICE)[:TOP_K]
    ).values.unsqueeze(0).unsqueeze(0).expand(BSZ, NUM_KV_HEADS, TOP_K).contiguous()

    # ---- PyTorch reference (from dct_page_attention.py assemble_kv) ----
    num_unselected = NUM_PAGES - TOP_K
    middle_len = TOP_K * PAGE_SIZE + num_unselected * COMP_SIZE

    selected_mask = torch.zeros(BSZ, NUM_KV_HEADS, NUM_PAGES, dtype=torch.bool, device=DEVICE)
    selected_mask.scatter_(2, selected_indices, True)

    token_counts = torch.where(selected_mask, PAGE_SIZE, COMP_SIZE)
    page_offsets = torch.zeros(BSZ, NUM_KV_HEADS, NUM_PAGES, dtype=torch.long, device=DEVICE)
    page_offsets[:, :, 1:] = token_counts[:, :, :-1].cumsum(dim=-1)

    idx_sel = selected_indices[:, :, :, None, None].expand(
        BSZ, NUM_KV_HEADS, TOP_K, PAGE_SIZE, HEAD_DIM
    )
    sel_k = torch.gather(paged_k, 2, idx_sel).reshape(BSZ, NUM_KV_HEADS, TOP_K * PAGE_SIZE, HEAD_DIM)
    sel_v = torch.gather(paged_v, 2, idx_sel).reshape(BSZ, NUM_KV_HEADS, TOP_K * PAGE_SIZE, HEAD_DIM)

    sort_perm = torch.argsort(selected_mask.int(), dim=-1, stable=True)
    unselected_indices = sort_perm[:, :, :num_unselected]

    idx_unsel = unselected_indices[:, :, :, None, None].expand(
        BSZ, NUM_KV_HEADS, num_unselected, COMP_SIZE, HEAD_DIM
    )
    unsel_k = torch.gather(comp_k, 2, idx_unsel).reshape(
        BSZ, NUM_KV_HEADS, num_unselected * COMP_SIZE, HEAD_DIM
    )
    unsel_v = torch.gather(comp_v, 2, idx_unsel).reshape(
        BSZ, NUM_KV_HEADS, num_unselected * COMP_SIZE, HEAD_DIM
    )

    sel_offsets = torch.gather(page_offsets, 2, selected_indices)
    sel_dst = (
        sel_offsets[:, :, :, None] + torch.arange(PAGE_SIZE, device=DEVICE)
    ).reshape(BSZ, NUM_KV_HEADS, TOP_K * PAGE_SIZE)

    unsel_offsets = torch.gather(page_offsets, 2, unselected_indices)
    unsel_dst = (
        unsel_offsets[:, :, :, None] + torch.arange(COMP_SIZE, device=DEVICE)
    ).reshape(BSZ, NUM_KV_HEADS, num_unselected * COMP_SIZE)

    middle_k_ref = torch.zeros(BSZ, NUM_KV_HEADS, middle_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    middle_v_ref = torch.zeros(BSZ, NUM_KV_HEADS, middle_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    middle_k_ref.scatter_(2, sel_dst[..., None].expand_as(sel_k), sel_k)
    middle_v_ref.scatter_(2, unsel_dst[..., None].expand_as(unsel_k), unsel_k)
    # Wait, this is wrong for v. Let me redo properly.

    middle_k_ref2 = torch.zeros(BSZ, NUM_KV_HEADS, middle_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    middle_v_ref2 = torch.zeros(BSZ, NUM_KV_HEADS, middle_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    middle_k_ref2.scatter_(2, sel_dst[..., None].expand_as(sel_k), sel_k)
    middle_k_ref2.scatter_(2, unsel_dst[..., None].expand_as(unsel_k), unsel_k)
    middle_v_ref2.scatter_(2, sel_dst[..., None].expand_as(sel_v), sel_v)
    middle_v_ref2.scatter_(2, unsel_dst[..., None].expand_as(unsel_v), unsel_v)

    final_k_ref = torch.cat([sink_k, middle_k_ref2, recent_k], dim=2)
    final_v_ref = torch.cat([sink_v, middle_v_ref2, recent_v], dim=2)

    # ---- Triton path (with fused sink/recent) ----
    from triton_kernels import assemble_kv_compressed_triton
    final_k_tri, final_v_tri = assemble_kv_compressed_triton(
        paged_k, paged_v, comp_k, comp_v,
        selected_indices, NUM_PAGES,
        PAGE_SIZE, COMP_SIZE, TOP_K,
        sink_k, sink_v, recent_k, recent_v,
    )

    # ---- Compare ----
    # Check sink region
    sink_len = 4
    recent_len = 173
    k_sink_ok = torch.equal(final_k_ref[:, :, :sink_len], final_k_tri[:, :, :sink_len])
    v_sink_ok = torch.equal(final_v_ref[:, :, :sink_len], final_v_tri[:, :, :sink_len])
    print(f"  Sink K match:   {k_sink_ok}")
    print(f"  Sink V match:   {v_sink_ok}")

    # Check recent region
    k_recent_ok = torch.equal(
        final_k_ref[:, :, sink_len + middle_len:],
        final_k_tri[:, :, sink_len + middle_len:]
    )
    v_recent_ok = torch.equal(
        final_v_ref[:, :, sink_len + middle_len:],
        final_v_tri[:, :, sink_len + middle_len:]
    )
    print(f"  Recent K match: {k_recent_ok}")
    print(f"  Recent V match: {v_recent_ok}")

    # Check middle region
    mid_k_ref = final_k_ref[:, :, sink_len:sink_len + middle_len]
    mid_k_tri = final_k_tri[:, :, sink_len:sink_len + middle_len]
    mid_v_ref = final_v_ref[:, :, sink_len:sink_len + middle_len]
    mid_v_tri = final_v_tri[:, :, sink_len:sink_len + middle_len]

    k_diff = (mid_k_ref.float() - mid_k_tri.float()).abs()
    v_diff = (mid_v_ref.float() - mid_v_tri.float()).abs()

    k_max_diff = k_diff.max().item()
    v_max_diff = v_diff.max().item()
    k_nonzero_mismatches = (k_diff > 0).sum().item()
    v_nonzero_mismatches = (v_diff > 0).sum().item()
    total_elements = mid_k_ref.numel()

    print(f"  Middle K max|diff|:  {k_max_diff:.6f}  ({k_nonzero_mismatches}/{total_elements} differ)")
    print(f"  Middle V max|diff|:  {v_max_diff:.6f}  ({v_nonzero_mismatches}/{total_elements} differ)")

    # Find first mismatching position for debugging
    if k_nonzero_mismatches > 0:
        mismatch_idx = (k_diff > 0).nonzero(as_tuple=False)[0]
        b, h, t, d = mismatch_idx.tolist()
        print(f"  First K mismatch at [b={b}, h={h}, t={t}, d={d}]:")
        print(f"    ref={mid_k_ref[b, h, t, d].item():.6f}  triton={mid_k_tri[b, h, t, d].item():.6f}")

        # Figure out which page this token belongs to
        cum = 0
        for p in range(NUM_PAGES):
            is_sel = p in selected_indices[0, h].tolist()
            toks = PAGE_SIZE if is_sel else COMP_SIZE
            if cum + toks > t:
                tok_in_page = t - cum
                print(f"    -> page {p} ({'selected' if is_sel else 'unselected'}), token {tok_in_page} within page")
                break
            cum += toks

    if v_nonzero_mismatches > 0:
        mismatch_idx = (v_diff > 0).nonzero(as_tuple=False)[0]
        b, h, t, d = mismatch_idx.tolist()
        print(f"  First V mismatch at [b={b}, h={h}, t={t}, d={d}]:")
        print(f"    ref={mid_v_ref[b, h, t, d].item():.6f}  triton={mid_v_tri[b, h, t, d].item():.6f}")

    ok = k_max_diff == 0 and v_max_diff == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_assemble_kv_no_fuse():
    """Compare assemble_kv middle-only (without sink/recent fusion)."""
    print()
    print("=" * 60)
    print("TEST: assemble_kv (middle only, no sink/recent)")
    print("=" * 60)

    torch.manual_seed(42)

    paged_k = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, PAGE_SIZE, HEAD_DIM,
                           device=DEVICE, dtype=DTYPE)
    paged_v = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, PAGE_SIZE, HEAD_DIM,
                           device=DEVICE, dtype=DTYPE)
    comp_k = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM,
                          device=DEVICE, dtype=DTYPE)
    comp_v = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM,
                          device=DEVICE, dtype=DTYPE)

    selected_indices = torch.sort(
        torch.randperm(NUM_PAGES, device=DEVICE)[:TOP_K]
    ).values.unsqueeze(0).unsqueeze(0).expand(BSZ, NUM_KV_HEADS, TOP_K).contiguous()

    # ---- Triton (middle only) ----
    from triton_kernels import assemble_kv_compressed_triton
    mid_k_tri, mid_v_tri = assemble_kv_compressed_triton(
        paged_k, paged_v, comp_k, comp_v,
        selected_indices, NUM_PAGES,
        PAGE_SIZE, COMP_SIZE, TOP_K,
    )

    # ---- PyTorch reference ----
    num_unselected = NUM_PAGES - TOP_K
    middle_len = TOP_K * PAGE_SIZE + num_unselected * COMP_SIZE

    selected_mask = torch.zeros(BSZ, NUM_KV_HEADS, NUM_PAGES, dtype=torch.bool, device=DEVICE)
    selected_mask.scatter_(2, selected_indices, True)

    token_counts = torch.where(selected_mask, PAGE_SIZE, COMP_SIZE)
    page_offsets = torch.zeros(BSZ, NUM_KV_HEADS, NUM_PAGES, dtype=torch.long, device=DEVICE)
    page_offsets[:, :, 1:] = token_counts[:, :, :-1].cumsum(dim=-1)

    idx_sel = selected_indices[:, :, :, None, None].expand(
        BSZ, NUM_KV_HEADS, TOP_K, PAGE_SIZE, HEAD_DIM
    )
    sel_k = torch.gather(paged_k, 2, idx_sel).reshape(BSZ, NUM_KV_HEADS, TOP_K * PAGE_SIZE, HEAD_DIM)
    sel_v = torch.gather(paged_v, 2, idx_sel).reshape(BSZ, NUM_KV_HEADS, TOP_K * PAGE_SIZE, HEAD_DIM)

    sort_perm = torch.argsort(selected_mask.int(), dim=-1, stable=True)
    unselected_indices = sort_perm[:, :, :num_unselected]

    idx_unsel = unselected_indices[:, :, :, None, None].expand(
        BSZ, NUM_KV_HEADS, num_unselected, COMP_SIZE, HEAD_DIM
    )
    unsel_k = torch.gather(comp_k, 2, idx_unsel).reshape(
        BSZ, NUM_KV_HEADS, num_unselected * COMP_SIZE, HEAD_DIM
    )
    unsel_v = torch.gather(comp_v, 2, idx_unsel).reshape(
        BSZ, NUM_KV_HEADS, num_unselected * COMP_SIZE, HEAD_DIM
    )

    sel_offsets = torch.gather(page_offsets, 2, selected_indices)
    sel_dst = (
        sel_offsets[:, :, :, None] + torch.arange(PAGE_SIZE, device=DEVICE)
    ).reshape(BSZ, NUM_KV_HEADS, TOP_K * PAGE_SIZE)

    unsel_offsets = torch.gather(page_offsets, 2, unselected_indices)
    unsel_dst = (
        unsel_offsets[:, :, :, None] + torch.arange(COMP_SIZE, device=DEVICE)
    ).reshape(BSZ, NUM_KV_HEADS, num_unselected * COMP_SIZE)

    mid_k_ref = torch.zeros(BSZ, NUM_KV_HEADS, middle_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    mid_v_ref = torch.zeros(BSZ, NUM_KV_HEADS, middle_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    mid_k_ref.scatter_(2, sel_dst[..., None].expand_as(sel_k), sel_k)
    mid_k_ref.scatter_(2, unsel_dst[..., None].expand_as(unsel_k), unsel_k)
    mid_v_ref.scatter_(2, sel_dst[..., None].expand_as(sel_v), sel_v)
    mid_v_ref.scatter_(2, unsel_dst[..., None].expand_as(unsel_v), unsel_v)

    # Compare
    k_diff = (mid_k_ref.float() - mid_k_tri.float()).abs()
    v_diff = (mid_v_ref.float() - mid_v_tri.float()).abs()

    k_max = k_diff.max().item()
    v_max = v_diff.max().item()
    k_mismatches = (k_diff > 0).sum().item()
    v_mismatches = (v_diff > 0).sum().item()
    total = mid_k_ref.numel()

    print(f"  Middle K max|diff|: {k_max:.6f}  ({k_mismatches}/{total} differ)")
    print(f"  Middle V max|diff|: {v_max:.6f}  ({v_mismatches}/{total} differ)")

    if k_mismatches > 0:
        idx = (k_diff > 0).nonzero(as_tuple=False)[0]
        b, h, t, d = idx.tolist()
        print(f"  First K mismatch at [b={b}, h={h}, t={t}, d={d}]:")
        print(f"    ref={mid_k_ref[b, h, t, d].item():.6f}  triton={mid_k_tri[b, h, t, d].item():.6f}")
        cum = 0
        for p in range(NUM_PAGES):
            is_sel = p in selected_indices[0, h].tolist()
            toks = PAGE_SIZE if is_sel else COMP_SIZE
            if cum + toks > t:
                print(f"    -> page {p} ({'selected' if is_sel else 'unselected'}), token {t - cum} within page")
                break
            cum += toks

    ok = k_max == 0 and v_max == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    all_pass = True
    all_pass &= test_score_pages()
    all_pass &= test_assemble_kv_no_fuse()
    all_pass &= test_assemble_kv()

    print()
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)