"""
InfLLM baseline wrapper for DCT-Page.

Thin shim around the upstream inf_llm package (pip install -e
/home/yoongonkim/InfLLM/) that:
  1. Applies inf_llm.patch_hf() to swap in retrieval-based block attention.
  2. Provides an InfLLMGenerator that mirrors HF generate()'s token-id
     output shape, built on top of inf_llm.utils.greedy_search.GreedySearch.

Requires transformers==4.37.2 (see config.py for full env setup). Llama 3.x
only in this wrapper; loosen assert_llama_only to re-enable Qwen2/Mistral.
"""

import torch


_ATTN_KWARGS_KEYS = (
    "block_size",
    "n_init",
    "n_local",
    "topk",
    "repr_topk",
    "max_cached_block",
    "exc_block_size",
    "fattn",
    "score_decay",
    "async_global_stream",
    "faiss",
    "perhead",
)


def assert_llama_only(base_model: str) -> None:
    """Fail fast if base_model is not a Llama variant.

    InfLLM's upstream patch_hf also supports Mistral and Qwen2, but this
    wrapper is scoped to Llama 3.x only. To extend: drop this assertion.
    Qwen3 is NOT supported upstream (Qwen3Attention has q_norm/k_norm that
    the patched forward does not re-apply).
    """
    if "llama" not in base_model.lower():
        raise ValueError(
            "InfLLM baseline wrapper only supports Llama models "
            f"(got base_model={base_model!r}). Qwen2/Mistral require loosening "
            "assert_llama_only; Qwen3 is not supported upstream."
        )


def load_llama_config_stripped_rope(base_model: str):
    """Load LlamaConfig with rope_scaling stripped.

    InfLLM pins transformers==4.37, which doesn't recognise Llama-3.1's
    rope_scaling (rope_type='llama3'); loading the config raises
    ValueError: `rope_scaling` must be a dictionary with ... `type` and `factor`.

    Stripping the key is safe because InfLLM replaces HF's rotary embedding
    with its own RotaryEmbeddingESM (see inf_llm.utils.patch.patch_hf), so
    the value is never consulted at runtime.
    """
    from transformers import LlamaConfig, PretrainedConfig

    config_dict, _ = PretrainedConfig.get_config_dict(base_model)
    config_dict.pop("rope_scaling", None)
    return LlamaConfig(**config_dict)


def init_inf_llm(model, cfg: dict):
    """Apply InfLLM's retrieval-block attention patch to a loaded Llama model."""
    from inf_llm import patch_hf

    attn_kwargs = {k: cfg[k] for k in _ATTN_KWARGS_KEYS if k in cfg}
    patch_hf(
        model,
        attn_type=cfg["attn_type"],
        attn_kwargs=attn_kwargs,
        base=cfg.get("base"),
        distance_scale=cfg.get("distance_scale", 1.0),
    )
    print(
        f"[inf_llm] attn_type={cfg['attn_type']} "
        f"n_init={cfg['n_init']} n_local={cfg['n_local']} "
        f"topk={cfg['topk']} block_size={cfg['block_size']} "
        f"repr_topk={cfg['repr_topk']} max_cached_block={cfg['max_cached_block']}"
    )
    return model


class InfLLMGenerator:
    """Adapter over inf_llm.utils.GreedySearch returning token-id tensors.

    The upstream GreedySearch._decode returns a list[str] of decoded text; our
    eval harness expects a [1, input_len + gen_len] id tensor (matching HF
    generate()). We replicate the prefill+decode loop here to avoid the lossy
    text round-trip. past_kv must be cleared between samples via clear().

    Assumes batch size 1.
    """

    def __init__(self, model, tokenizer, chunk_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.past_kv = None

    def clear(self) -> None:
        self.past_kv = None

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        extra_end_token_ids=None,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        assert input_ids.size(0) == 1, "InfLLMGenerator only supports batch size 1"

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        end_token_ids = list(extra_end_token_ids or [])
        if self.tokenizer.eos_token_id is not None:
            end_token_ids.append(self.tokenizer.eos_token_id)

        past_key_values = self.past_kv
        chunk_size = self.chunk_size

        for i in range(max_new_tokens + 1):
            if i == 0:
                # Prefill in chunks, leaving the last token for the unified step below.
                cs = chunk_size if chunk_size is not None else input_ids.size(1)
                for st in range(0, input_ids.size(1) - 1, cs):
                    ed = min(input_ids.size(1) - 1, st + cs)
                    out = self.model(
                        input_ids=input_ids[:, st:ed],
                        attention_mask=attention_mask[:, :ed],
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                    )
                    past_key_values = out.past_key_values

                out = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                )
            else:
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                )

            logits, past_key_values = out.logits, out.past_key_values
            word = logits[:, -1, :].argmax(dim=-1)
            if word.item() in end_token_ids or i == max_new_tokens:
                break

            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.size(0), 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=-1,
            )

        self.past_kv = past_key_values
        return input_ids


def build_inf_llm_generator(model, tokenizer, cfg: dict) -> InfLLMGenerator:
    return InfLLMGenerator(model, tokenizer, cfg["chunk_size"])


__all__ = [
    "assert_llama_only",
    "load_llama_config_stripped_rope",
    "init_inf_llm",
    "InfLLMGenerator",
    "build_inf_llm_generator",
]
