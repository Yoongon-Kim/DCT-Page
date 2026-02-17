from dataclasses import dataclass


@dataclass
class DCTPageConfig:
    page_size: int = 128           # Tokens per page
    top_k: int = 8                 # Pages selected for full attention
    sink_size: int = 4             # Initial tokens always kept (attention sink)
    recent_size: int = 128         # Recent tokens always kept (absorbs last partial page)
    compress_ratio: float = 0.25   # Per-page DCT compression (128 -> 32 tokens)
    scoring_method: str = "max"    # "mean" | "max" | "sum" — reduction over compressed tokens within a page
    group_agg_method: str = "mean" # "mean" | "max" | "topp" — aggregation of per-head scores within a GQA group
    unselected_mode: str = "drop"  # "drop" | "compressed"
    selection_mode: str = "standard"  # "standard" | "hierarchical"
    continuous_rope: bool = False      # Store KV before RoPE, apply continuous RoPE after assembly
