"""
Multipole Attention evaluation configuration.

Edit this file to change model, clustering parameters, and attention behavior
before running eval_longbench_v1.py or eval_longbench_v2.py with --mode multipole_attention.

Key parameters:
  - percent_clusters_lst: percentage of keys to retain per hierarchy level
  - percentiles_lst:      importance threshold (token budget) per level
  - use_replacement:      if True, use centroid value approximation for non-selected tokens
  - cluster_interval:     number of new tokens between re-clustering during generation
"""

MULTIPOLE_ATTN_CONFIG = {
    # Base model (HF Hub ID or local path)
    "base_model": "Qwen/Qwen3-8B",

    # Enable centroid-based sparse attention
    "use_centroids": True,

    # Percentage of keys to retain per hierarchy level (list, one per level)
    "percent_clusters_lst": [6.25],

    # Importance threshold per level (token budget for centroid selection)
    "percentiles_lst": [2180],

    # Use centroid value approximation for non-selected tokens
    "use_replacement": False,

    # Number of new tokens between re-clustering during generation
    "cluster_interval": 128,

    # Tensor parallelism degree (1 = single GPU)
    "inference_tp": 1,
}
