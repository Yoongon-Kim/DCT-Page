"""Minimal script to profile assemble kernels with ncu.

Creates realistic tensors matching Llama-3.1-8B decode config and runs
assemble_kv_split_triton in a loop. ncu captures per-kernel hardware metrics.

Usage:
    ncu --kernel-name "_copy_full_segments_kernel|_copy_unselected_pages_kernel" \
        --set full -o assemble_report \
        python profile_assemble_ncu.py

    # Or lightweight metrics only:
    ncu --kernel-name "_copy_full_segments_kernel|_copy_unselected_pages_kernel" \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
        python profile_assemble_ncu.py
"""

import torch
from triton_kernels import assemble_kv_split_triton, topk_sort_triton, score_pages_triton
from dct_page_attention import _compute_rope_cos_sin

# Config matching Llama-3.1-8B with context_length=32768
bsz = 1
num_kv_heads = 8
page_size = 128
top_k = 8
sink_size = 4
recent_size = 128
comp_size = 4
head_dim = 128
context_length = 32768

num_pages = (context_length - sink_size - recent_size) // page_size
num_unselected = num_pages - top_k
middle_len = top_k * page_size + num_unselected * comp_size
total_len = sink_size + middle_len + recent_size

device = "cuda"
dtype = torch.bfloat16

print(f"num_pages={num_pages}, top_k={top_k}, num_unselected={num_unselected}")
print(f"total_len={total_len}, middle_len={middle_len}")

# Create realistic tensors
paged_k = torch.randn(bsz, num_kv_heads, num_pages, page_size, head_dim, dtype=dtype, device=device)
paged_v = torch.randn_like(paged_k)
comp_k = torch.randn(bsz, num_kv_heads, num_pages, comp_size, head_dim, dtype=dtype, device=device)
comp_v = torch.randn_like(comp_k)
sink_k = torch.randn(bsz, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
sink_v = torch.randn_like(sink_k)
recent_k = torch.randn(bsz, num_kv_heads, recent_size, head_dim, dtype=dtype, device=device)
recent_v = torch.randn_like(recent_k)

# Generate selected indices (sorted, like topk_sort_triton output)
selected_indices = torch.sort(torch.randperm(num_pages, device=device)[:top_k])[0]
selected_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_kv_heads, -1).to(torch.int32)

# RoPE tables
positions = torch.arange(total_len, device=device)
cos_table, sin_table = _compute_rope_cos_sin(positions, head_dim, 500000.0, device, dtype)
cos_table = cos_table[0, 0]  # [total_len, head_dim]
sin_table = sin_table[0, 0]

# Pre-allocate output buffers
out_k = torch.empty(bsz, num_kv_heads, total_len + page_size, head_dim, dtype=dtype, device=device)
out_v = torch.empty_like(out_k)
out_sel_idx = torch.empty(bsz, num_kv_heads, top_k, dtype=torch.int32, device=device)

# Warmup (triggers Triton compilation)
print("Warmup...")
for _ in range(3):
    assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices, cos_table, sin_table,
        out_k=out_k, out_v=out_v, out_sel_idx=out_sel_idx,
    )
torch.cuda.synchronize()

# Profiled iterations (ncu captures these)
print("Profiling...")
for i in range(5):
    assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices, cos_table, sin_table,
        out_k=out_k, out_v=out_v, out_sel_idx=out_sel_idx,
    )
torch.cuda.synchronize()
print("Done.")
