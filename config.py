from dataclasses import dataclass


@dataclass
class DCTPageConfig:
    page_size: int = 32            # Tokens per page
    top_k: int = 64                # Pages selected for full attention
    sink_size: int = 4             # Initial tokens always kept (attention sink)
    recent_size: int = 128         # Recent tokens always kept (absorbs last partial page)
    compress_ratio: float = 0.03125   # Per-page compression (32 -> 1 token)
    min_decode_kv_len_for_paging: int = 8192  # Fallback to baseline decode attention below this KV length
    proxy_frequency_layout: str = "low"  # "low" | "low_high" | "low_mid_high" | "spread"
    scoring_method: str = "max"    # "mean" | "max" | "sum" | "dc_ac_{lambda}" | "proxy_dc_ac_{lambda}" | "spread_dc_ac_{lambda}" | "hybrid_multi{M}_ac_max_a{alpha}" | "spectral_recon_max"
    group_agg_method: str = "mean" # "mean" | "max" | "topp" — aggregation of per-head scores within a GQA group
    unselected_mode: str = "drop"  # "drop" | "compressed"
    compression_method: str = "haar"  # "haar" | "dct" — compression method for unselected pages in compressed mode
    compressed_token_rope: str = "mixed"  # "mixed" | "block_center" — RoPE handling for compressed tokens
    continuous_rope: bool = False      # Temporarily disabled. Store KV before RoPE, apply continuous RoPE after assembly.
    score_use_direct_spectral_proxy: bool = False  # Score with truncated DCT coefficients directly instead of IDCT-reconstructed proxies
    score_use_haar_proxy: bool = True  # Default score path: Haar lowpass block proxies instead of DCT IDCT proxies
    score_use_haar_mixed_proxy: bool = False  # Score with Haar mixed global/detail proxies instead of DCT IDCT proxies
    score_use_hadamard_proxy: bool = False  # Score with Walsh-Hadamard compressed proxies in original-position RoPE space
    select_with_oracle_page_scores: bool = False  # Debug/upper-bound mode: use full-page oracle scores for top-k selection
    use_triton: bool = True            # Use fused Triton kernels (False = pure PyTorch, for comparison)
    weight_compressed_by_population: bool = False  # In compressed mode, scale each unselected-page rep's softmax mass by page_size/comp_size via a log(n) bias on QK logits (multipole-style population weighting). No-op for drop mode and for direct-spectral-proxy mode.
    max_unselected_compressed: int = -1  # Max unselected pages contributing compressed tokens (-1 = unlimited, 0 = drop all unselected, N = keep top-N by score)
    comp_kv_quant: str = "none"  # "none" | "fp8_e4m3" | "fp8_e5m2" | "int8" | "int4" — fake-quantization of compressed K and V at write time (for selection-precision study).
    comp_kv_quant_granularity: str = "per_page"  # "per_page" | "per_comp_token" — scale granularity for fake-quant. per_page: one scale per (bsz, kv_head, page); per_comp_token: one scale per (bsz, kv_head, page, comp_idx).
