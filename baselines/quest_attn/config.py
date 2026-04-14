"""
Quest attention evaluation configuration.

Edit this file to change model, page size, max sequence length, and token budget
before running eval_ruler.py or eval_longbench_v1.py with --mode quest_attention.

Quest divides the KV cache into fixed-size pages and estimates page importance
using min/max key metadata per page. Only the top-k most important pages
(by token budget) are used for sparse attention during decoding.

Key parameters:
  - page_size:     tokens per KV cache page (Quest default: 16)
  - max_seq_len:   maximum context length for KV cache allocation
  - token_budget:  max tokens to attend to per decode step (= page_budget * page_size)

Note: Quest supports LLaMA-family models (Llama-2, Llama-3.x, Mistral) and Qwen3.
"""

QUEST_ATTN_CONFIG = {
    # Base model (HF Hub ID or local path) — LLaMA-family or Qwen3
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",

    # Tokens per KV cache page
    "page_size": 16,

    # Maximum context length (determines KV cache allocation)
    "max_seq_len": 131072,

    # Token budget per decode step (page_budget = token_budget // page_size)
    "token_budget": 2048,
}
