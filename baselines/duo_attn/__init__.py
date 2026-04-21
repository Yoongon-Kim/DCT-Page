"""
DuoAttention baseline wrapper for DCT-Page.

Thin shim around the upstream duo_attn package (pip install -e
/home/yoongonkim/duo-attention/) that:
  1. Resolves the pre-trained attention pattern directory.
  2. Loads + sparsifies the full/streaming head mask.
  3. Calls enable_duo_attention_eval() on the loaded model.
  4. Provides a manual greedy-generation loop (duo_generate_greedy)
     that matches DuoAttention's own eval pattern. transformers>=4.37
     wraps past_key_values in DynamicCache inside generate(), which is
     incompatible with DuoAttention's v4.34-style tuple KV cache.

Requires transformers==4.45.2; use a dedicated conda env. Llama only.
"""

import os

import torch


def assert_llama(base_model: str) -> None:
    """Fail fast if base_model is not a Llama variant."""
    if "llama" not in base_model.lower():
        raise ValueError(
            "DuoAttention baseline only ships pre-trained patterns for Llama "
            f"(got base_model={base_model!r}). Qwen/Mistral not supported here."
        )


def resolve_pattern_dir(cfg: dict) -> str:
    """Join pattern_root + pattern_subdir and verify the directory exists."""
    pattern_dir = os.path.join(cfg["pattern_root"], cfg["pattern_subdir"])
    if not os.path.isdir(pattern_dir):
        raise FileNotFoundError(
            f"DuoAttention pattern directory not found: {pattern_dir}. "
            "Check pattern_root/pattern_subdir in DUO_ATTN_CONFIG."
        )
    return pattern_dir


def init_duo_attention(model, cfg: dict) -> None:
    """Apply DuoAttention head-sparsity mask to a loaded Llama model."""
    from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
    from duo_attn.patch import enable_duo_attention_eval

    pattern_dir = resolve_pattern_dir(cfg)
    full_attention_heads, pattern_sink, pattern_recent = load_attn_pattern(pattern_dir)
    full_attention_heads, achieved_sparsity = sparsify_attention_heads(
        full_attention_heads, sparsity=cfg["sparsity"]
    )

    sink_size = cfg.get("sink_size", pattern_sink)
    recent_size = cfg.get("recent_size", pattern_recent)

    enable_duo_attention_eval(
        model,
        full_attention_heads,
        sink_size=sink_size,
        recent_size=recent_size,
    )
    print(
        f"[duo_attn] pattern={cfg['pattern_subdir']} "
        f"target_sparsity={cfg['sparsity']} achieved={achieved_sparsity:.3f} "
        f"sink={sink_size} recent={recent_size}"
    )


def duo_generate_greedy(model, input_ids, max_new_tokens, eos_token_ids=None):
    """Greedy generation using DuoAttention's tuple-KV-cache forward path.

    Bypasses transformers.generate() because generate() wraps past_key_values
    in a DynamicCache object, which DuoAttention's v4.34-style forward cannot
    index as past_key_values[0][0]. Mirrors the manual loop in
    /home/yoongonkim/duo-attention/eval/LongBench/pred.py.

    Assumes batch size 1 (matches DCT-Page's eval harness).

    Args:
        model: HF CausalLM with enable_duo_attention_eval applied.
        input_ids: [1, seq] prompt tokens on model.device.
        max_new_tokens: max number of tokens to generate.
        eos_token_ids: optional iterable of ints; stops early if any is produced.

    Returns:
        [1, seq + generated] concatenated token ids.
    """
    eos_set = set(eos_token_ids) if eos_token_ids is not None else set()
    generated = []

    with torch.no_grad():
        # Prefill
        output = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_tok = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok)

        for _ in range(max_new_tokens - 1):
            if next_tok.item() in eos_set:
                break
            output = model(
                input_ids=next_tok,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_tok = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)

    generated_tensor = torch.cat(generated, dim=1)
    return torch.cat([input_ids, generated_tensor], dim=1)


__all__ = ["init_duo_attention", "assert_llama", "resolve_pattern_dir",
           "duo_generate_greedy"]
