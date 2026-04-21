"""
InfLLM evaluation configuration.

Edit this file to change model, memory-block parameters, and attention behavior
before running eval_ruler.py or eval_longbench_v{1,2}.py with --mode inf_llm.

InfLLM requires transformers==4.37.2 (its patch.py targets pre-4.45 internals:
rotary_emb.base/.dim, tuple past_key_values, legacy attention-forward signature).
Use a dedicated conda env, e.g.:

    conda create -n inf_llm_env python=3.10
    conda activate inf_llm_env
    pip install torch>=2.1 accelerate>=0.27
    pip install transformers==4.37.2
    pip install omegaconf fschat
    pip install -e /home/yoongonkim/InfLLM
    pip install -e /home/yoongonkim/DCT-Page

Llama 3.x only. Qwen2/Mistral are in InfLLM's upstream whitelist but are not
enabled here; Qwen3 requires extra upstream support for q_norm/k_norm and is
explicitly out of scope.

Key parameters (see upstream config/llama-3-inf-llm.yaml):
  block_size         tokens per memory block
  n_init             sink tokens
  n_local            local sliding window
  topk               blocks retrieved for attention
  repr_topk          representative tokens per block
  max_cached_block   GPU block cache size
  exc_block_size     execution block size (<= n_local)
  chunk_size         prefill chunk size for GreedySearch
"""

INF_LLM_CONFIG = {
    # Base model (HF Hub ID or local path). Llama 3.x only.
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",

    # Which attention variant from inf_llm.attention.ATTN_FORWRAD to use.
    # One of: "inf-llm", "infinite-lm", "stream-llm", "origin".
    # Use "origin" (full attention) as a sanity-check baseline.
    "attn_type": "inf-llm",

    # Memory/block parameters.
    "block_size": 128,
    "n_init": 128,
    "n_local": 4096,
    "topk": 16,
    "repr_topk": 4,
    "max_cached_block": 32,
    "exc_block_size": 512,

    # Triton flash attention (requires nightly triton; leave False by default).
    "fattn": False,

    # RoPE. None => inherit from model's rotary_emb.base.
    "base": None,
    "distance_scale": 1.0,

    # Advanced knobs (kept at upstream defaults).
    "score_decay": None,
    "async_global_stream": False,
    "faiss": False,
    "perhead": False,

    # GreedySearch prefill chunk size (affects memory, not quality).
    "chunk_size": 8192,
}
