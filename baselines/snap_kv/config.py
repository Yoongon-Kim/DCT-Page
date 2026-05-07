"""
Default configuration for the SnapKV baseline.

Environment requirements:
  - transformers==4.37.2  (SnapKV patches LlamaFlashAttention2 from that release)
  - flash-attn pinned to a version compatible with transformers 4.37.x
  - Llama 3.x models only (the shim patches transformers.models.llama.*)

Do NOT run this baseline with Qwen3 or Mistral without extending the shim.
"""

SNAPKV_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "window_size": 32,
    "max_capacity_prompt": 2048,
    "kernel_size": 5,
    "pooling": "avgpool",  # "avgpool" | "maxpool"
}
