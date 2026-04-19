"""
DuoAttention evaluation configuration.

Edit this file to change model, pattern directory, and attention behavior
before running eval_ruler.py or eval_longbench_v{1,2}.py with --mode duo_attention.

DuoAttention requires transformers==4.45.2 and flash-attn==2.6.3. Activate the
dedicated conda env (e.g. `duo_env`) before invoking any eval script in this mode.

Key parameters:
  - pattern_root:      parent directory containing attn_patterns/ from the upstream repo
  - pattern_subdir:    model-specific subpath (from attn_patterns/{MODEL}/lr=...)
  - sparsity:          fraction of heads set to streaming (0..1); higher = more sparse
  - sink_size:         number of attention-sink tokens kept for streaming heads
  - recent_size:       number of recent tokens kept for streaming heads
"""

DUO_ATTN_CONFIG = {
    # Base model (HF Hub ID or local path). Llama only.
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",

    # Root of the upstream DuoAttention attn_patterns/ directory.
    "pattern_root": "/home/yoongonkim/duo-attention/attn_patterns",

    # Relative subpath to the specific pattern under pattern_root.
    "pattern_subdir": "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10",

    # Fraction of heads to push into streaming mode (0 = all full-attention, 1 = all streaming).
    "sparsity": 0.75,

    # Sink and recent window sizes for streaming heads.
    "sink_size": 64,
    "recent_size": 256,
}
