from dataclasses import dataclass


@dataclass
class DCTPageConfig:
    page_size: int = 32            # Tokens per page (sink/recent are page-counted)
    top_k: int = 64                # Pages selected for full attention
    num_sink_pages: int = 0        # First N physical pages always attended (sink). 0 = no sink forcing.
    num_recent_pages: int = 0      # Number of full recent pages always attended; EXCLUDES the currently-open partial page (open is implicit, +1). 0 = only open page forced.
    compress_ratio: float = 0.03125   # Per-page compression (32 -> 1 token)
    proxy_method: str = "dct"         # "dct" (lowpass-IDCT) | "haar" (block-mean) | "dct_haar" (DCT+detail) | "harp" (adaptive Haar with detail-driven expansion)
    score_softmax: bool = False       # ShadowKV-style: per qo-head softmax over pages before GQA group reduction (head-magnitude normalization)
    qaware_lastq_topbins: int = 0     # >0 enables Q-aware lastq adaptive bin selection. Per page, pick top-N DCT bins by |Q_lastq · DCT(K)| (Q_lastq = first-decode-step Q or last-N prefill queries' mean, frozen per (sample, layer)), then score current Q via partial IDCT reconstruction.
    qaware_lastq_window: int = 1      # Window size for Q_lastq. 1 = first decode-step Q. >1 = mean of last-N prefill queries (InfLLM r_m–style local window, captured at prefill).
    haar_detail_per_block: int = 0    # 0 = pure block-mean; >0 = +N per-block orthonormal Haar detail rows (total comp = cs*(1+N))
    haar_detail_with_negation: bool = False  # If True, every detail row is duplicated with its negation so that max(Q·row, Q·-row) = |Q·row|
    harp_detail_topk: int = 4         # HARP only: # of blocks whose H_3 detail contributes to scoring (top-K by per-page H_3 L2 norm)
    min_decode_kv_len_for_paging: int = 8192  # Fallback to baseline decode attention below this KV length
    dense_first_n_layers: int = 0  # Skip sparse attention on the first N layers (use dense attention). 0 = all layers sparse. Quest baseline uses 2.
    scoring_method: str = "max"    # "mean" | "max" | "sum"
    group_agg_method: str = "mean" # "mean" | "max" | "topp" — aggregation of per-head scores within a GQA group
    unselected_mode: str = "drop"  # "drop" | "compressed"
    compressed_token_rope: str = "mixed"  # "mixed" | "block_center" — RoPE handling for compressed tokens
    continuous_rope: bool = False      # Temporarily disabled. Store KV before RoPE, apply continuous RoPE after assembly.
    score_use_quest_minmax: bool = False  # QUEST-style scoring: per-page per-channel min/max key metadata → sum_i max(q_i*max_i, q_i*min_i)
    score_combine_quest_dct: bool = False  # Best-rank fusion ensemble: score by -min(rank_dct_lowpass, rank_quest_minmax), takes top-K of fused ranks. Computes both selectors per step.
    km_quest_split: int = 0  # K-M union strategy: DCT picks top-(top_k - M) + Quest picks M pages NOT in DCT's set. Total = top_k. 0 = off, recommended M=8.
    select_with_oracle_page_scores: bool = False  # Debug/upper-bound mode: use full-page oracle scores for top-k selection
    oracle_score_mode: str = "max"  # "max" | "mass" — oracle scoring used when select_with_oracle_page_scores=True. "max" = max(q·K) per page (matches cs=page_size proxy). "mass" = Σ_t exp(q·K) per page (softmax-mass-based selection).
    use_triton: bool = True            # Use fused Triton kernels (False = pure PyTorch, for comparison)
    weight_compressed_by_population: bool = False  # In compressed mode, scale each unselected-page rep's softmax mass by page_size/comp_size via a log(n) bias on QK logits (multipole-style population weighting). No-op for drop mode.
    max_unselected_compressed: int = -1  # Max unselected pages contributing compressed tokens (-1 = unlimited, 0 = drop all unselected, N = keep top-N by score)
    comp_kv_quant: str = "none"  # "none" | "fp8_e4m3" | "fp8_e5m2" | "int8" | "int4" — fake-quantization of compressed K and V at write time (for selection-precision study).
    comp_kv_quant_granularity: str = "per_page"  # "per_page" | "per_comp_token" — scale granularity for fake-quant. per_page: one scale per (bsz, kv_head, page); per_comp_token: one scale per (bsz, kv_head, page, comp_idx).
    # Outlier bank: post-prefill detect M tokens per (layer, kv_head) and always
    # include them in decode attention alongside DCT-selected pages.
    outlier_budget: int = 0              # 0 = off; >0 = top-M outlier tokens per kv_head
    outlier_detector: str = "lastq_mean" # "knorm" | "lastq_mean" | "cluster_dyn"
    cluster_outlier_N: int = 256         # k-means cluster count (cluster_dyn only)
    cluster_outlier_iters: int = 5       # k-means Lloyd iterations
    cluster_outlier_top_k: int = 8       # # of top clusters per step; refined to M tokens by Q·K within
    cluster_outlier_q_agg: str = "mean"  # GQA aggregation for cluster scoring/refinement: "mean" | "max"
    cluster_outlier_scoring: str = "centroid"  # "centroid" (mean K) | "minmax" (Quest-style upper bound)
    # ---- Head-dim selection proxies (calibrated, FASA-aligned) ----
    # proxy_method: "pca_qaware" — dense head-dim PCA projection (D1 Pearson or D2 TopK-overlap basis).
    #               "fasa_fc"    — FASA's dominant FC channel subset.
    # When set, the proxy basis file is loaded from `proxy_basis_path`.
    proxy_basis_path: str = ""    # path to .pt file containing calibrated PCA M or FASA I_dom
    pca_cs_h: int = 0              # head_dim projection rank for pca_qaware (0 = use stored cs_h_max)
    fasa_n_tip: int = 0            # FASA dominant FC count for fasa_fc proxy (0 = use stored n_tip_max)
