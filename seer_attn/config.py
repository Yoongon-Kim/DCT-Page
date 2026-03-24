"""
SeerAttention-R evaluation configuration.

Edit this file to change model checkpoint, sparsity method, and budget/threshold
before running eval_ruler.py with --mode seer_attention.

Available HF checkpoints (SeerAttention-R, decode sparse only):
  - SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates
  - SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates
  - SeerAttention/SeerAttention-Decode-Qwen3-14B-AttnGates
  - SeerAttention/SeerAttention-Decode-DeepSeek-R1-Distill-Qwen-... (Qwen2-based)

Sparsity methods:
  - "token_budget": keep top-k tokens per decode step (controlled by token_budget)
  - "threshold":    keep blocks where gate score > threshold
"""

SEER_ATTN_CONFIG = {
    # HF Hub ID or local path to SeerAttention-R checkpoint
    "seer_model": "SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates",

    # Sparsity method: "token_budget" or "threshold"
    "sparsity_method": "token_budget",

    # Max active tokens per decode step (only used when sparsity_method="token_budget")
    "token_budget": 2180,

    # Gate score threshold (only used when sparsity_method="threshold")
    "threshold": 0.0,

    # First layer to apply sparse attention (layers below use dense); 0 = all layers
    "start_layer": 0,
}
