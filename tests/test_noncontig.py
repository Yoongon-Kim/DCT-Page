"""Test _copy_full_segments_kernel with non-contiguous paged_k/v (mimics real model KV cache views)."""

import torch
from triton_kernels import assemble_kv_split_triton

DEVICE = "cuda"
DTYPE = torch.bfloat16

BSZ = 1
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 128
COMP_SIZE = 4
SINK_LEN = 4
RECENT_LEN = 128
TOP_K = 8

# Simulate real model: num_pages from a 32K context
# 32768 - 4 (sink) - 128 (recent) = 32636 pageable
# 32636 // 128 = 255 pages (with 12 leftover)
NUM_PAGES = 255
ACTUAL_RECENT = RECENT_LEN + 12  # leftover merged into recent

torch.manual_seed(42)

# Allocate a LARGE KV cache buffer (like the real model does)
# Real model pre-allocates for max_seq_len, creating padded strides
MAX_SEQ_LEN = 32768 + 256  # extra padding
kv_buffer_k = torch.randn(BSZ, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
kv_buffer_v = torch.randn(BSZ, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)

# Slice out segments (views, non-contiguous in dim 1 due to padded strides)
sink_k = kv_buffer_k[:, :, :SINK_LEN]
sink_v = kv_buffer_v[:, :, :SINK_LEN]

pages_end = SINK_LEN + NUM_PAGES * PAGE_SIZE
paged_k = kv_buffer_k[:, :, SINK_LEN:pages_end].view(BSZ, NUM_KV_HEADS, NUM_PAGES, PAGE_SIZE, HEAD_DIM)
paged_v = kv_buffer_v[:, :, SINK_LEN:pages_end].view(BSZ, NUM_KV_HEADS, NUM_PAGES, PAGE_SIZE, HEAD_DIM)

recent_k = kv_buffer_k[:, :, pages_end:pages_end + ACTUAL_RECENT]
recent_v = kv_buffer_v[:, :, pages_end:pages_end + ACTUAL_RECENT]

# Compressed pages (contiguous — from DCT computation)
comp_k = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM, device=DEVICE, dtype=DTYPE)
comp_v = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, COMP_SIZE, HEAD_DIM, device=DEVICE, dtype=DTYPE)

# Selected indices
page_scores = torch.randn(BSZ, NUM_KV_HEADS, NUM_PAGES, dtype=torch.float32, device=DEVICE)
_, top_indices = torch.topk(page_scores, TOP_K, dim=-1)
selected_indices, _ = top_indices.sort(dim=-1)

# Print tensor info
print(f"paged_k: shape={paged_k.shape}, strides={paged_k.stride()}, contiguous={paged_k.is_contiguous()}")
print(f"sink_k:  shape={sink_k.shape}, strides={sink_k.stride()}, contiguous={sink_k.is_contiguous()}")
print(f"recent_k: shape={recent_k.shape}, strides={recent_k.stride()}, contiguous={recent_k.is_contiguous()}")
print(f"sel_idx: {selected_indices[0, 0]}")

# Test WITHOUT RoPE first
print("\n--- Test 1: No RoPE ---")
try:
    final_k, final_v = assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
    )
    print(f"  Output shape: {final_k.shape}")
    print(f"  PASS (no crash)")
except Exception as e:
    print(f"  FAIL: {e}")

# Test WITH RoPE
print("\n--- Test 2: With RoPE ---")
try:
    from dct_page_attention import _compute_rope_cos_sin
    num_unselected = NUM_PAGES - TOP_K
    middle_len = TOP_K * PAGE_SIZE + num_unselected * COMP_SIZE
    assembled_len = SINK_LEN + middle_len + ACTUAL_RECENT

    positions = torch.arange(assembled_len, device=DEVICE)
    cos_full, sin_full = _compute_rope_cos_sin(positions, HEAD_DIM, 500000.0, DEVICE, DTYPE)
    cos_table = cos_full[0, 0].contiguous()
    sin_table = sin_full[0, 0].contiguous()

    final_k, final_v = assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
        cos_table, sin_table,
    )
    print(f"  Output shape: {final_k.shape}")
    print(f"  PASS (no crash)")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: Verify correctness vs old kernel
print("\n--- Test 3: Correctness vs assemble_kv_full_triton ---")
try:
    from triton_kernels import assemble_kv_full_triton
    expected_k, expected_v = assemble_kv_full_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
    )
    actual_k, actual_v = assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices,
    )
    k_diff = (expected_k.float() - actual_k.float()).abs().max().item()
    v_diff = (expected_v.float() - actual_v.float()).abs().max().item()
    print(f"  K max diff: {k_diff}")
    print(f"  V max diff: {v_diff}")
    ok = k_diff == 0 and v_diff == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"  FAIL: {e}")
