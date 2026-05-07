"""
SnapKV baseline wrapper for DCT-Page.

Thin shim around vendored SnapKV sources (_vendor.py) that:
  1. Applies replace_llama() to monkey-patch LlamaFlashAttention2 and
     LlamaForCausalLM with SnapKV's KV-compression variants.
  2. Provides init_snap_kv() which sets the per-model config attributes
     (window_size, max_capacity_prompt, kernel_size, pooling) before patching.

Requires transformers==4.37.2 + compatible flash-attn. Llama 3.x only.
"""

from ._vendor import init_snapkv as _init_snapkv, replace_llama as _replace_llama


def assert_llama_only(base_model: str) -> None:
    """Fail fast if base_model is not a Llama variant.

    SnapKV's upstream patch targets LlamaFlashAttention2 from transformers 4.37.
    Qwen3/Mistral are not supported by this shim.
    """
    if "llama" not in base_model.lower():
        raise ValueError(
            "SnapKV baseline wrapper only supports Llama models "
            f"(got base_model={base_model!r}). Qwen3/Mistral require extending "
            "the shim with the corresponding hijack modules from upstream SnapKV."
        )


def load_llama_config_stripped_rope(base_model: str):
    """Load LlamaConfig with rope_scaling stripped.

    SnapKV pins transformers==4.37, which doesn't recognise Llama-3.1's
    rope_scaling (rope_type='llama3'); loading the config raises
    ValueError: `rope_scaling` must be a dictionary with ... `type` and `factor`.

    Stripping the key is safe because SnapKV replaces the rotary embedding
    handling via the patched forward, so the value is never consulted at runtime.
    """
    from transformers import LlamaConfig, PretrainedConfig

    config_dict, _ = PretrainedConfig.get_config_dict(base_model)
    config_dict.pop("rope_scaling", None)
    return LlamaConfig(**config_dict)


def init_snap_kv(model, cfg: dict) -> None:
    """Apply SnapKV KV-compression patch to a loaded Llama model.

    Sets model.config.{window_size,max_capacity_prompt,kernel_size,pooling}
    from cfg BEFORE calling replace_llama() so every attention forward can
    read them via self.config.
    """
    model.config.window_size = cfg["window_size"]
    model.config.max_capacity_prompt = cfg["max_capacity_prompt"]
    model.config.kernel_size = cfg["kernel_size"]
    model.config.pooling = cfg["pooling"]
    _replace_llama()
    print(
        f"[snap_kv] window_size={cfg['window_size']} "
        f"max_capacity_prompt={cfg['max_capacity_prompt']} "
        f"kernel_size={cfg['kernel_size']} "
        f"pooling={cfg['pooling']}"
    )


__all__ = [
    "assert_llama_only",
    "load_llama_config_stripped_rope",
    "init_snap_kv",
]
