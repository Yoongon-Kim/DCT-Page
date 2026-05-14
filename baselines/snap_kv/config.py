"""Default configuration for the SnapKV baseline.

Environment requirements (LIVE PATH):
  - transformers==5.2.0   (DCT_Page conda env)
  - python>=3.12          (DCT_Page conda env)
  - Supports Llama 3.x AND Qwen3.

Reference path:
  - transformers==4.37.2  (snap_kv conda env)
  - Llama 3.x only.
"""
SNAPKV_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",  # or "Qwen/Qwen3-8B"
    "window_size": 32,
    "max_capacity_prompt": 2048,
    "kernel_size": 5,
    "pooling": "avgpool",  # "avgpool" | "maxpool"
}
