"""Profile Kernel A vs Kernel B timing with NVTX markers for nsys."""
import torch
import torch.cuda.nvtx as nvtx
from triton_kernels import assemble_kv_split_triton
import triton
import math

def _compute_rope_cos_sin(positions, head_dim, theta, device, dtype):
    """Standalone RoPE computation (avoids importing dct_page_attention)."""
    dim = head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = positions.float()
    freqs = torch.outer(t, freqs)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin

# Config matching Llama-3.1-8B with context_length=32768
bsz = 1
num_kv_heads = 8
num_q_heads = 32
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
assembled_len = total_len

device = "cuda"
dtype = torch.bfloat16

print(f"num_pages={num_pages}, top_k={top_k}, num_unselected={num_unselected}")
print(f"total_len={total_len}")

# Create tensors
paged_k = torch.randn(bsz, num_kv_heads, num_pages, page_size, head_dim, dtype=dtype, device=device)
paged_v = torch.randn_like(paged_k)
comp_k = torch.randn(bsz, num_kv_heads, num_pages, comp_size, head_dim, dtype=dtype, device=device)
comp_v = torch.randn_like(comp_k)
sink_k = torch.randn(bsz, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
sink_v = torch.randn_like(sink_k)
recent_k = torch.randn(bsz, num_kv_heads, recent_size, head_dim, dtype=dtype, device=device)
recent_v = torch.randn_like(recent_k)
query_states = torch.randn(bsz, num_q_heads, 1, head_dim, dtype=dtype, device=device)

selected_indices = torch.sort(torch.randperm(num_pages, device=device)[:top_k])[0]
selected_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_kv_heads, -1).to(torch.int32)

positions = torch.arange(total_len, device=device)
cos_table, sin_table = _compute_rope_cos_sin(positions, head_dim, 500000.0, device, dtype)
cos_table = cos_table[0, 0]
sin_table = sin_table[0, 0]

q_rope_cos = cos_table[assembled_len - 1]
q_rope_sin = sin_table[assembled_len - 1]
q_rope_buf = torch.empty_like(query_states)

out_k = torch.empty(bsz, num_kv_heads, total_len + page_size, head_dim, dtype=dtype, device=device)
out_v = torch.empty_like(out_k)
out_sel_idx = torch.empty(bsz, num_kv_heads, top_k, dtype=torch.int32, device=device)

# Warmup
print("Warmup...")
for _ in range(20):
    assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices, cos_table, sin_table,
        out_k=out_k, out_v=out_v, out_sel_idx=out_sel_idx,
        query_states=query_states,
        q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin, q_rope_buf=q_rope_buf,
    )
torch.cuda.synchronize()

# Profile
N = 50
print(f"Profiling {N} iterations...")
for i in range(N):
    nvtx.range_push(f"assemble_{i}")
    assemble_kv_split_triton(
        paged_k, paged_v, comp_k, comp_v,
        sink_k, sink_v, recent_k, recent_v,
        selected_indices, cos_table, sin_table,
        out_k=out_k, out_v=out_v, out_sel_idx=out_sel_idx,
        query_states=query_states,
        q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin, q_rope_buf=q_rope_buf,
    )
    nvtx.range_pop()

torch.cuda.synchronize()
print("Done.")