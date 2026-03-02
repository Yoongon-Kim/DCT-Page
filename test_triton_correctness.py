"""
Diagnostic: compare fused Triton kernels vs PyTorch reference paths.

Tests:
  1. Projection matrix equivalence (matmul vs FFT-based DCT)
  2. Score pages (Triton vs PyTorch)
  3. Assemble KV (Triton vs PyTorch, interleaved layout, no RoPE)
  4. Assemble KV with K-RoPE (fused vs separate apply)
  5. Q-RoPE kernel (Triton vs PyTorch)

Run:  python test_triton_correctness.py
"""

import torch
import sys

# ---- Config matching eval defaults ----
BSZ = 1
NUM_KV_HEADS = 8
NUM_KV_GROUPS = 4
NUM_HEADS = NUM_KV_HEADS * NUM_KV_GROUPS
HEAD_DIM = 128
PAGE_SIZE = 128
COMP_SIZE = 3
NUM_PAGES = 244
TOP_K = 8
SCORING_METHOD = "mean"
GROUP_AGG_METHOD = "max"
SINK_LEN = 4
RECENT_LEN = 173

DEVICE = "cuda"
DTYPE = torch.bfloat16


def test_projection_matrix():
    """Test that matmul projection matches FFT-based dct_compress_page."""
    print("=" * 60)
    print("TEST: projection matrix equivalence")
    print("=" * 60)

    from old_dct_page_attention import dct_compress_page
    from dct_page_attention import _build_dct_projection_matrix

    torch.manual_seed(42)

    M = _build_dct_projection_matrix(PAGE_SIZE, COMP_SIZE, DEVICE, DTYPE)
    print(f"  M shape: {M.shape}")  # [comp_size, page_size]

    # Random pages: [bsz, num_kv_heads, n_pages, page_size, head_dim]
    pages = torch.randn(BSZ, NUM_KV_HEADS, 5, PAGE_SIZE, HEAD_DIM,
                         device=DEVICE, dtype=DTYPE)

    # FFT reference
    flat = pages.reshape(BSZ * NUM_KV_HEADS * 5, 1, PAGE_SIZE, HEAD_DIM)
    ref = dct_compress_page(flat, COMP_SIZE)
    ref = ref.view(BSZ, NUM_KV_HEADS, 5, COMP_SIZE, HEAD_DIM)

    # Matmul path
    tri = torch.einsum('cs,bhnsd->bhncd', M, pages)

    diff = (ref.float() - tri.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  max|diff|:  {max_diff:.8f}")
    print(f"  mean|diff|: {mean_diff:.8f}")

    ok = max_diff < 0.01
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_score_pages():
    """Compare score_pages: Triton vs PyTorch reference."""
    print()
    print("=" * 60)
    print("TEST: score_pages")
    print("=" * 60)

    torch.manual_seed(42)

    query_states = torch.randn(BSZ, NUM_HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    compressed_keys = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM,
                                  device=DEVICE, dtype=DTYPE)

    # PyTorch reference
    scaling = HEAD_DIM ** -0.5
    query_grouped = query_states.view(BSZ, NUM_KV_HEADS, NUM_KV_GROUPS, 1, HEAD_DIM)
    scores = torch.einsum('bgiqd,bgncd->bgiqnc', query_grouped * scaling, compressed_keys)
    if SCORING_METHOD == "mean":
        page_scores_ref = scores.mean(dim=-1)
    elif SCORING_METHOD == "max":
        page_scores_ref = scores.max(dim=-1).values
    else:
        page_scores_ref = scores.sum(dim=-1)
    page_scores_ref = page_scores_ref.mean(dim=3)
    if GROUP_AGG_METHOD == "mean":
        page_scores_ref = page_scores_ref.mean(dim=2)
    else:
        page_scores_ref = page_scores_ref.max(dim=2).values

    # Triton path
    from triton_kernels import score_pages_triton
    page_scores_tri = score_pages_triton(
        query_states, compressed_keys,
        SCORING_METHOD, GROUP_AGG_METHOD, NUM_KV_GROUPS,
    )

    score_diff = (page_scores_ref.float() - page_scores_tri.float()).abs()
    max_diff = score_diff.max().item()
    mean_diff = score_diff.mean().item()
    print(f"  Page scores max|diff|:  {max_diff:.6f}")
    print(f"  Page scores mean|diff|: {mean_diff:.6f}")

    # Check topk agreement
    actual_top_k = min(TOP_K, NUM_PAGES)
    _, sel_ref = torch.topk(page_scores_ref, actual_top_k, dim=-1)
    sel_ref, _ = sel_ref.sort(dim=-1)
    _, sel_tri = torch.topk(page_scores_tri, actual_top_k, dim=-1)
    sel_tri, _ = sel_tri.sort(dim=-1)
    indices_match = torch.equal(sel_ref, sel_tri)
    print(f"  TopK indices match: {indices_match}")

    ok = max_diff < 0.01 and indices_match
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_assemble_kv_full():
    """Test Triton assemble produces correct interleaved layout (no RoPE)."""
    print()
    print("=" * 60)
    print("TEST: assemble_kv_full_triton (interleaved layout, no RoPE)")
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
    sink_k = torch.randn(BSZ, NUM_KV_HEADS, SINK_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    sink_v = torch.randn(BSZ, NUM_KV_HEADS, SINK_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    recent_k = torch.randn(BSZ, NUM_KV_HEADS, RECENT_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    recent_v = torch.randn(BSZ, NUM_KV_HEADS, RECENT_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    page_scores = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES,
                               dtype=torch.float32, device=DEVICE)
    actual_top_k = min(TOP_K, NUM_PAGES)
    _, selected_indices = torch.topk(page_scores, actual_top_k, dim=-1)
    selected_indices, _ = selected_indices.sort(dim=-1)

    num_unselected = NUM_PAGES - actual_top_k
    middle_len = actual_top_k * PAGE_SIZE + num_unselected * COMP_SIZE
    total_len = SINK_LEN + middle_len + RECENT_LEN

    # ---- Build expected interleaved output (PyTorch reference) ----
    # assemble_kv from dct_page_attention.py (compressed mode, interleaved order)
    from dct_page_attention import assemble_kv
    from config import DCTPageConfig
    dummy_cfg = DCTPageConfig(
        page_size=PAGE_SIZE, top_k=TOP_K, sink_size=SINK_LEN,
        recent_size=RECENT_LEN, compress_ratio=COMP_SIZE / PAGE_SIZE,
        unselected_mode="compressed", use_triton=False,
    )
    expected_k, expected_v = assemble_kv(
        sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
        recent_k, recent_v, selected_indices, dummy_cfg, NUM_PAGES,
    )

    # ---- Function under test ----
    from triton_kernels import assemble_kv_split_triton
    actual_k, actual_v = assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
    )

    print(f"  Output shape: {actual_k.shape} (expected [{BSZ}, {NUM_KV_HEADS}, {total_len}, {HEAD_DIM}])")

    k_diff = (expected_k.float() - actual_k.float()).abs()
    v_diff = (expected_v.float() - actual_v.float()).abs()
    k_max = k_diff.max().item()
    v_max = v_diff.max().item()

    print(f"  K max|diff|: {k_max:.6f}")
    print(f"  V max|diff|: {v_max:.6f}")
    print(f"  K exact match: {torch.equal(expected_k, actual_k)}")
    print(f"  V exact match: {torch.equal(expected_v, actual_v)}")

    ok = torch.equal(expected_k, actual_k) and torch.equal(expected_v, actual_v)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_assemble_kv_with_rope():
    """Test fused assemble + K-RoPE matches separate assemble + _apply_rope."""
    print()
    print("=" * 60)
    print("TEST: assemble_kv_full_triton with K-RoPE")
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
    sink_k = torch.randn(BSZ, NUM_KV_HEADS, SINK_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    sink_v = torch.randn(BSZ, NUM_KV_HEADS, SINK_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    recent_k = torch.randn(BSZ, NUM_KV_HEADS, RECENT_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    recent_v = torch.randn(BSZ, NUM_KV_HEADS, RECENT_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    actual_top_k = min(TOP_K, NUM_PAGES)
    num_unselected = NUM_PAGES - actual_top_k
    middle_len = actual_top_k * PAGE_SIZE + num_unselected * COMP_SIZE
    total_len = SINK_LEN + middle_len + RECENT_LEN

    # Build RoPE tables
    from old_dct_page_attention import _compute_rope_cos_sin, _apply_rope
    rope_theta = 500000.0
    positions = torch.arange(total_len, device=DEVICE)
    cos_full, sin_full = _compute_rope_cos_sin(positions, HEAD_DIM, rope_theta, DEVICE, DTYPE)
    # cos_full: [1, 1, total_len, HEAD_DIM]

    # Compute topk indices (shared by both paths)
    _, top_indices = torch.topk(
        torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, dtype=torch.float32, device=DEVICE),
        actual_top_k, dim=-1,
    )
    selected_indices, _ = top_indices.sort(dim=-1)

    # Reference: assemble without RoPE, then apply RoPE separately
    from triton_kernels import assemble_kv_split_triton
    final_k_no_rope, final_v_ref = assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
    )
    final_k_ref = _apply_rope(final_k_no_rope, cos_full, sin_full)

    # Fused: assemble + K-RoPE in one call
    cos_table = cos_full[0, 0].contiguous()  # [total_len, HEAD_DIM]
    sin_table = sin_full[0, 0].contiguous()
    final_k_fused, final_v_fused = assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
        cos_table, sin_table,
    )

    k_diff = (final_k_ref.float() - final_k_fused.float()).abs()
    v_diff = (final_v_ref.float() - final_v_fused.float()).abs()

    k_max = k_diff.max().item()
    v_max = v_diff.max().item()
    k_mis = (k_diff > 0.01).sum().item()
    total = k_diff.numel()

    print(f"  K max|diff|: {k_max:.6f} ({k_mis}/{total} exceed 0.01)")
    print(f"  V max|diff|: {v_max:.6f}")
    print(f"  V exact match: {torch.equal(final_v_ref, final_v_fused)}")

    # RoPE computation in bf16 can have small numerical differences
    ok = k_max < 0.1 and v_max == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_q_rope():
    """Test Q-RoPE Triton kernel matches PyTorch _apply_rope."""
    print()
    print("=" * 60)
    print("TEST: Q-RoPE kernel")
    print("=" * 60)

    torch.manual_seed(42)

    query = torch.randn(BSZ, NUM_HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    from old_dct_page_attention import _compute_rope_cos_sin, _apply_rope
    rope_theta = 500000.0
    positions = torch.tensor([1000], device=DEVICE)
    cos, sin = _compute_rope_cos_sin(positions, HEAD_DIM, rope_theta, DEVICE, DTYPE)

    # PyTorch reference
    ref = _apply_rope(query, cos, sin)

    # Triton
    from triton_kernels import apply_rope_q_triton
    tri = apply_rope_q_triton(query, cos, sin)

    diff = (ref.float() - tri.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  max|diff|:  {max_diff:.6f}")
    print(f"  mean|diff|: {mean_diff:.6f}")

    # bf16 RoPE has ~0.015 max diff (1 ULP) due to intermediate precision
    ok = max_diff < 0.05
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    all_pass = True
    all_pass &= test_projection_matrix()
    all_pass &= test_score_pages()
    all_pass &= test_assemble_kv_full()
    all_pass &= test_assemble_kv_with_rope()
    all_pass &= test_q_rope()

    print()
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
