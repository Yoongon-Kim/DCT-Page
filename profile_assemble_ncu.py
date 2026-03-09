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

# Profiled iterations
N = 100
print(f"Profiling {N} iterations...")

# Time with CUDA events (no sync overhead)
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(N)]

for i in range(N):
    start_events[i].record()
    assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices, cos_table, sin_table,
        out_k=out_k, out_v=out_v, out_sel_idx=out_sel_idx,
    )
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # microseconds
times.sort()
# Drop top/bottom 10%
trimmed = times[N//10 : -N//10]
avg_us = sum(trimmed) / len(trimmed)
print(f"Isolated assemble_kv_split_triton: {avg_us:.1f} µs (trimmed mean, {len(trimmed)} samples)")
print(f"  Min: {min(times):.1f} µs, Max: {max(times):.1f} µs, Median: {times[N//2]:.1f} µs")
print(f"  Per layer in full model would be: {avg_us:.1f} µs × 32 layers = {avg_us * 32 / 1000:.2f} ms/tok")
print()

# Break down: how much is Python wrapper vs actual GPU kernel time?
# Run bare kernels directly (bypass wrapper overhead)
import triton
from triton_kernels import _copy_full_segments_kernel, _copy_unselected_pages_kernel

BLOCK_D = triton.next_power_of_2(head_dim)
BLOCK_T_FULL = 32
BLOCK_T_COMP = max(triton.next_power_of_2(comp_size), 4)
apply_rope = True

# Prepare sel_idx (what the wrapper does)
sel_idx = out_sel_idx[:, :, :top_k]
sel_idx.copy_(selected_indices)

max_seg_len = max(sink_size, page_size, recent_size)
tiles_per_seg = (max_seg_len + BLOCK_T_FULL - 1) // BLOCK_T_FULL
grid_full = (bsz * num_kv_heads * (top_k + 2), tiles_per_seg)
PAGES_PER_PROG = 16  # Optimal from sweep with clean triton cache
num_groups = (num_pages + PAGES_PER_PROG - 1) // PAGES_PER_PROG
grid_comp = (bsz * num_kv_heads * num_groups,)

rope_stride_t = cos_table.stride(0)
final_k = out_k[:, :, :total_len, :]
final_v = out_v[:, :, :total_len, :]

# Warmup bare kernels
for _ in range(3):
    _copy_full_segments_kernel[grid_full](
        paged_k, paged_v, sink_k, sink_v, recent_k, recent_v,
        final_k, final_v, cos_table, sin_table, sel_idx,
        paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3),
        sink_k.stride(0), sink_k.stride(1), sink_k.stride(2),
        recent_k.stride(0), recent_k.stride(1), recent_k.stride(2),
        final_k.stride(0), final_k.stride(1), final_k.stride(2),
        rope_stride_t,
        sel_idx.stride(0), sel_idx.stride(1),
        num_kv_heads, top_k, head_dim, sink_size, recent_size, total_len,
        COMP_SIZE=comp_size, PAGE_SIZE=page_size,
        BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_FULL, APPLY_ROPE=1,
    )
    _copy_unselected_pages_kernel[grid_comp](
        comp_k, comp_v, final_k, final_v, cos_table, sin_table, sel_idx,
        comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3),
        final_k.stride(0), final_k.stride(1), final_k.stride(2),
        rope_stride_t,
        sel_idx.stride(0), sel_idx.stride(1),
        num_kv_heads, num_pages, head_dim, sink_size, num_groups,
        COMP_SIZE=comp_size, PAGE_SIZE=page_size, TOP_K=top_k,
        PAGES_PER_PROG=PAGES_PER_PROG,
        BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_COMP, APPLY_ROPE=1,
    )
torch.cuda.synchronize()

# Time Kernel A alone
ka_starts = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
ka_ends = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
for i in range(N):
    ka_starts[i].record()
    _copy_full_segments_kernel[grid_full](
        paged_k, paged_v, sink_k, sink_v, recent_k, recent_v,
        final_k, final_v, cos_table, sin_table, sel_idx,
        paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3),
        sink_k.stride(0), sink_k.stride(1), sink_k.stride(2),
        recent_k.stride(0), recent_k.stride(1), recent_k.stride(2),
        final_k.stride(0), final_k.stride(1), final_k.stride(2),
        rope_stride_t,
        sel_idx.stride(0), sel_idx.stride(1),
        num_kv_heads, top_k, head_dim, sink_size, recent_size, total_len,
        COMP_SIZE=comp_size, PAGE_SIZE=page_size,
        BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_FULL, APPLY_ROPE=1,
    )
    ka_ends[i].record()
torch.cuda.synchronize()
ka_times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(ka_starts, ka_ends)])
ka_trimmed = ka_times[N//10:-N//10]
ka_avg = sum(ka_trimmed) / len(ka_trimmed)

# Time Kernel B alone
kb_starts = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
kb_ends = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
for i in range(N):
    kb_starts[i].record()
    _copy_unselected_pages_kernel[grid_comp](
        comp_k, comp_v, final_k, final_v, cos_table, sin_table, sel_idx,
        comp_k.stride(0), comp_k.stride(1), comp_k.stride(2), comp_k.stride(3),
        final_k.stride(0), final_k.stride(1), final_k.stride(2),
        rope_stride_t,
        sel_idx.stride(0), sel_idx.stride(1),
        num_kv_heads, num_pages, head_dim, sink_size, num_groups,
        COMP_SIZE=comp_size, PAGE_SIZE=page_size, TOP_K=top_k,
        PAGES_PER_PROG=PAGES_PER_PROG,
        BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T_COMP, APPLY_ROPE=1,
    )
    kb_ends[i].record()
torch.cuda.synchronize()
kb_times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(kb_starts, kb_ends)])
kb_trimmed = kb_times[N//10:-N//10]
kb_avg = sum(kb_trimmed) / len(kb_trimmed)

print(f"Bare kernel timing (no Python wrapper):")
print(f"  Kernel A (full segments): {ka_avg:.1f} µs")
print(f"  Kernel B (unselected):    {kb_avg:.1f} µs")
print(f"  Sum bare kernels:         {ka_avg + kb_avg:.1f} µs")
print(f"  Wrapper overhead:         {avg_us - ka_avg - kb_avg:.1f} µs")
print()

# Grid analysis
print(f"Grid analysis:")
print(f"  Kernel A grid: ({grid_full[0]}, {grid_full[1]}) = {grid_full[0] * grid_full[1]} programs")
print(f"    = {bsz}×{num_kv_heads}×{top_k+2} work items × {tiles_per_seg} tiles")
print(f"  Kernel B grid: ({grid_comp[0]},) = {grid_comp[0]} programs")
print(f"    = {bsz}×{num_kv_heads}×{num_groups} groups ({PAGES_PER_PROG} pages/prog, was {bsz*num_kv_heads*num_pages})")
print()

# Memory analysis
selected_bytes = bsz * num_kv_heads * top_k * page_size * head_dim * 2 * 2  # K+V, bf16
unselected_bytes = bsz * num_kv_heads * num_unselected * comp_size * head_dim * 2 * 2
sink_bytes = bsz * num_kv_heads * sink_size * head_dim * 2 * 2
recent_bytes = bsz * num_kv_heads * recent_size * head_dim * 2 * 2
rope_read_bytes = total_len * head_dim * 2 * 4  # cos+sin, float32 or bf16
total_bytes = selected_bytes + unselected_bytes + sink_bytes + recent_bytes
print(f"Memory transfer analysis:")
print(f"  Selected pages:  {selected_bytes / 1024:.1f} KB ({top_k} × {page_size} tokens)")
print(f"  Unselected comp: {unselected_bytes / 1024:.1f} KB ({num_unselected} × {comp_size} tokens)")
print(f"  Sink:            {sink_bytes / 1024:.1f} KB ({sink_size} tokens)")
print(f"  Recent:          {recent_bytes / 1024:.1f} KB ({recent_size} tokens)")
print(f"  Total read+write: {total_bytes * 2 / 1024 / 1024:.2f} MB (read src + write dst)")
print(f"  + RoPE tables:   ~{rope_read_bytes / 1024:.1f} KB")
print(f"  + RoPE rotated K read: ~{total_bytes / 1024:.1f} KB (extra load for rotation)")
print()

# Bandwidth calculation
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}")
total_transfer = total_bytes * 3  # read src + read rotated + write dst
bw_achieved = total_transfer / (avg_us / 1e6) / 1e9  # GB/s
print(f"Achieved bandwidth: {bw_achieved:.1f} GB/s")
print("Done.")
