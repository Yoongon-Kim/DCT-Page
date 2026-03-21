from dataclasses import dataclass


@dataclass
class DCTPageConfig:
    page_size: int = 128           # Tokens per page
    top_k: int = 8                 # Pages selected for full attention
    sink_size: int = 4             # Initial tokens always kept (attention sink)
    recent_size: int = 128         # Recent tokens always kept (absorbs last partial page)
    compress_ratio: float = 0.25   # Per-page DCT compression (128 -> 32 tokens)
    proxy_frequency_layout: str = "low"  # "low" | "low_high" | "low_mid_high" | "spread"
    scoring_method: str = "max"    # "mean" | "max" | "sum" — reduction over compressed tokens within a page
    group_agg_method: str = "mean" # "mean" | "max" | "topp" — aggregation of per-head scores within a GQA group
    unselected_mode: str = "drop"  # "drop" | "compressed"
    continuous_rope: bool = True       # Store KV before RoPE, apply continuous RoPE after assembly
    score_use_direct_spectral_proxy: bool = False  # Score with truncated DCT coefficients directly instead of IDCT-reconstructed proxies
    score_use_haar_proxy: bool = True  # Default score path: Haar lowpass block proxies instead of DCT IDCT proxies
    score_use_haar_mixed_proxy: bool = False  # Score with Haar mixed global/detail proxies instead of DCT IDCT proxies
    score_use_hadamard_proxy: bool = False  # Score with Walsh-Hadamard compressed proxies in original-position RoPE space
    select_with_oracle_page_scores: bool = False  # Debug/upper-bound mode: use full-page oracle scores for top-k selection
    use_triton: bool = True            # Use fused Triton kernels (False = pure PyTorch, for comparison)
