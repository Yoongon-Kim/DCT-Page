"""
SeerAttention prefill-sparse evaluation configuration.

Edit this file to change model checkpoint and sparsity knobs before running
eval_ruler.py with --mode seer_prefill.

Available HF checkpoints (SeerAttention, prefill-sparse AttnGates):
  Llama:
    - SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates
    - SeerAttention/SeerAttention-Llama-3.1-70B-AttnGates
  Qwen 2.5:
    - SeerAttention/SeerAttention-Qwen2.5-7B-AttnGates
    - SeerAttention/SeerAttention-Qwen2.5-14B-AttnGates
    - SeerAttention/SeerAttention-Qwen2.5-32B-AttnGates
  DeepSeek-R1-Distill (Qwen2-based):
    - SeerAttention/SeerAttention-DeepSeek-R1-Distill-Qwen-14B-AttnGates
    - SeerAttention/SeerAttention-DeepSeek-R1-Distill-Qwen-32B-AttnGates

Note: prefill sparsity is applied only during prefill. Decode runs dense with
a standard KV cache, so generation uses the default `model.generate(...)` path.

Sparsity methods:
  - "threshold": keep blocks where gate score > threshold (typical 5e-4 to 5e-3)
  - "nz_ratio":  keep top nz_ratio fraction of blocks per row
"""

SEER_PREFILL_CONFIG = {
    # HF Hub ID or local path to SeerAttention prefill AttnGates checkpoint
    "seer_model": "SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates",

    # Sparsity method: "threshold" or "nz_ratio"
    "sparsity_method": "nz_ratio",

    # Threshold (only used when sparsity_method="threshold")
    "threshold": 2e-3,

    # Non-zero ratio (only used when sparsity_method="nz_ratio")
    "nz_ratio": 0.066,

    # Keep last row blocks dense for the query (recommended True)
    "last_block_dense": True,

    # Use flash_attn's fused rotary kernel. Set False if your env has flash-attn
    # newer than torch (newer flash-attn calls torch.library.wrap_triton, only
    # available in torch >= 2.6). The fallback is a pure PyTorch RoPE; perf cost
    # for prefill is ~milliseconds vs. seconds of attention compute.
    "use_flash_rope": False,
}


def load_seer_prefill_model(cfg, torch_dtype):
    """
    Dispatch to the Llama or Qwen2 prefill-sparse class based on the
    `base_model` field in the AttnGates checkpoint's config.

    Returns (model, hf_config) so callers can reuse hf_config.base_model
    to load the correct tokenizer.
    """
    from transformers import AutoConfig
    from seer_attn import SeerAttnLlamaForCausalLM, SeerAttnQwen2ForCausalLM

    ckpt = cfg["seer_model"]
    hf_config = AutoConfig.from_pretrained(ckpt)
    base = hf_config.base_model.lower()

    kwargs = dict(
        torch_dtype=torch_dtype,
        seerattn_sparsity_method=cfg["sparsity_method"],
        seerattn_threshold=cfg["threshold"],
        seerattn_nz_ratio=cfg["nz_ratio"],
        seerattn_last_block_dense=cfg["last_block_dense"],
        use_flash_rope=cfg["use_flash_rope"],
    )
    if "llama" in base:
        model = SeerAttnLlamaForCausalLM.from_pretrained(ckpt, **kwargs)
    elif "qwen" in base:
        model = SeerAttnQwen2ForCausalLM.from_pretrained(ckpt, **kwargs)
    else:
        raise ValueError(
            f"Unsupported base model for seer_prefill: {hf_config.base_model!r}. "
            "Expected a Llama or Qwen2.x base."
        )
    return model, hf_config
