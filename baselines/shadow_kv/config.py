"""
ShadowKV baseline evaluation configuration.

Edit this file to change defaults before running eval_ruler.py with --mode
shadowkv. CLI flags (--shadowkv_cache_mode, --sparse_budget, --rank,
--chunk_size) override these values at run time.

ShadowKV (https://arxiv.org/abs/2410.21465) keeps a low-rank (SVD-compressed)
key cache on GPU and offloads the value cache to CPU pinned memory. At decode
time, the query attends to per-chunk landmark keys (mean post-RoPE key per
chunk of `chunk_size` tokens), selects top-k chunks until `sparse_budget`
tokens are reached, reconstructs K via U @ SV with per-position RoPE, and
gathers V from CPU.

Key parameters:
  - cache_mode:    "shadowkv_cpu" (production, V on CPU) or "shadowkv"
                   (GPU-only, batch=1 only).
  - sparse_budget: tokens attended to per decode step (must equal
                   select_sets * chunk_size).
  - rank:          SVD rank for the key cache (paper default: 160).
  - chunk_size:    tokens per landmark chunk (paper default: 8).
  - max_length:    KV cache buffer size; eval_ruler overwrites this with
                   max(seq_lengths) + 4096.
  - minference:    keep False for accuracy comparison; enabling requires
                   shipping MInference per-model pattern JSONs.

Note: Only Llama-3.x is supported by this baseline. ShadowKV's Qwen2 class
does not implement Qwen3's QK-norm.
"""

SHADOWKV_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "cache_mode": "shadowkv_cpu",
    "sparse_budget": 2048,
    "rank": 160,
    "chunk_size": 8,
    "max_length": 131072,
    "minference": False,
}
