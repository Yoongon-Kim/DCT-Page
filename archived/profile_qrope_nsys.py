"""Isolated profiling of 6c_q_rope step for nsys analysis."""
import torch
import torch.cuda.nvtx as nvtx
import sys
sys.path.insert(0, "/home/yoongonkim/DCT-Page")
from triton_kernels import apply_rope_q_direct

# Llama-3.1-8B decode config
bsz = 1
num_heads = 32
head_dim = 128
device = "cuda"
dtype = torch.bfloat16

# Create tensors
query_states = torch.randn(bsz, num_heads, 1, head_dim, device=device, dtype=dtype)
cos_q_flat = torch.randn(head_dim, device=device, dtype=dtype)
sin_q_flat = torch.randn(head_dim, device=device, dtype=dtype)
q_rope_buf = torch.empty_like(query_states)

# Rope cos/sin cache (simulating self._rope_cos_cache)
assembled_len = 1156  # sink(4) + top_k(8)*page_size(128) + recent(128)
rope_cos_cache = torch.randn(1, 1, assembled_len, head_dim, device=device, dtype=dtype)
rope_sin_cache = torch.randn(1, 1, assembled_len, head_dim, device=device, dtype=dtype)

# Warmup
for _ in range(20):
    apply_rope_q_direct(query_states, cos_q_flat, sin_q_flat, q_rope_buf)
torch.cuda.synchronize()

# Profile with NVTX markers for nsys
for i in range(50):
    nvtx.range_push(f"q_rope_iter_{i}")

    # Simulate the full 6c_q_rope code path from profile_decode.py
    nvtx.range_push("cache_index")
    cos_q = rope_cos_cache[0, 0, assembled_len - 1]  # [head_dim]
    sin_q = rope_sin_cache[0, 0, assembled_len - 1]  # [head_dim]
    nvtx.range_pop()

    nvtx.range_push("triton_rope_kernel")
    query_states = apply_rope_q_direct(query_states, cos_q, sin_q, q_rope_buf)
    nvtx.range_pop()

    nvtx.range_pop()

torch.cuda.synchronize()
print("Done. Analyze with: nsys stats <report>.nsys-rep")