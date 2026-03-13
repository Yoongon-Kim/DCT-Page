"""Verify stride-cached assemble produces identical output to non-cached path."""
import torch
from triton_kernels import assemble_kv_split_triton, build_assemble_stride_cache

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

device = "cuda"
dtype = torch.bfloat16

def _compute_rope_cos_sin(positions, head_dim, theta, device, dtype):
    dim = head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = positions.float()
    freqs = torch.outer(t, freqs)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin

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

q_rope_cos = cos_table[total_len - 1]
q_rope_sin = sin_table[total_len - 1]

# Allocate separate output buffers for each path
buf_len = total_len + page_size
out_k_ref = torch.empty(bsz, num_kv_heads, buf_len, head_dim, dtype=dtype, device=device)
out_v_ref = torch.empty_like(out_k_ref)
q_rope_buf_ref = torch.empty_like(query_states)

out_k_cached = torch.empty(bsz, num_kv_heads, buf_len, head_dim, dtype=dtype, device=device)
out_v_cached = torch.empty_like(out_k_cached)
q_rope_buf_cached = torch.empty_like(query_states)

# --- Reference: no cache ---
final_k_ref, final_v_ref, q_ref = assemble_kv_split_triton(
    paged_k, paged_v, comp_k, comp_v,
    sink_k, sink_v, recent_k, recent_v,
    selected_indices, cos_table, sin_table,
    out_k=out_k_ref, out_v=out_v_ref,
    query_states=query_states,
    q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin, q_rope_buf=q_rope_buf_ref,
)

# --- Stride-cached path ---
stride_cache = build_assemble_stride_cache(
    paged_k, comp_k, sink_k, recent_k, selected_indices,
    cos_table, out_k_cached,
    query_states=query_states,
)

final_k_sc, final_v_sc, q_sc = assemble_kv_split_triton(
    paged_k, paged_v, comp_k, comp_v,
    sink_k, sink_v, recent_k, recent_v,
    selected_indices, cos_table, sin_table,
    out_k=out_k_cached, out_v=out_v_cached,
    query_states=query_states,
    q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin, q_rope_buf=q_rope_buf_cached,
    stride_cache=stride_cache,
)

torch.cuda.synchronize()

# --- Verify ---
k_match = torch.equal(final_k_ref, final_k_sc)
v_match = torch.equal(final_v_ref, final_v_sc)
q_match = torch.equal(q_ref, q_sc)

print(f"final_k exact match: {k_match}")
print(f"final_v exact match: {v_match}")
print(f"q_rope  exact match: {q_match}")

if k_match and v_match and q_match:
    print("\nAll outputs are IDENTICAL. Stride cache is correct.")
else:
    if not k_match:
        diff = (final_k_ref - final_k_sc).abs()
        print(f"  final_k max diff: {diff.max().item()}, num mismatches: {(diff > 0).sum().item()}")
    if not v_match:
        diff = (final_v_ref - final_v_sc).abs()
        print(f"  final_v max diff: {diff.max().item()}, num mismatches: {(diff > 0).sum().item()}")
    if not q_match:
        diff = (q_ref - q_sc).abs()
        print(f"  q_rope max diff: {diff.max().item()}, num mismatches: {(diff > 0).sum().item()}")