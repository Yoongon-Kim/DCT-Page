"""SnapKV baseline wrapper for DCT-Page.

Live path (transformers==5.2.0, DCT_Page conda env):
    baselines/snap_kv/__init__.py
        -> patch_v5.replace_{llama,qwen3}_v5(model)
        -> patch_v5._snapkv_attention_forward (unified)
        -> upstream/snapkv/monkeypatch/snapkv_utils.SnapKVCluster

Reference path (transformers==4.37.2, snap_kv conda env):
    baselines/snap_kv/upstream/__init__.py
        -> upstream/snapkv/monkeypatch/monkeypatch.replace_llama()
        -> upstream/snapkv/monkeypatch/llama_hijack_4_37.llama_flash_attn2_forward
        -> upstream/snapkv/monkeypatch/snapkv_utils.SnapKVCluster
"""
from .patch_v5 import (
    replace_llama_v5,
    replace_qwen3_v5,
    unpatch_llama_v5,
    unpatch_qwen3_v5,
)


def _assert_llama_or_qwen3(base_model: str) -> None:
    """Internal — fail fast if base_model is not a supported family."""
    bm = base_model.lower()
    if "llama" not in bm and "qwen3" not in bm:
        raise ValueError(
            "SnapKV wrapper supports only Llama 3.x and Qwen3 "
            f"(got base_model={base_model!r})."
        )


def _detect_family(model) -> str:
    mt = getattr(model.config, "model_type", "").lower()
    if mt == "llama":   return "llama"
    if mt == "qwen3":   return "qwen3"
    raise ValueError(f"Unsupported model_type for SnapKV: {mt!r}")


def init_snap_kv(model, cfg: dict) -> None:
    """Apply SnapKV KV-compression patch to a loaded model."""
    if "base_model" in cfg:
        _assert_llama_or_qwen3(cfg["base_model"])

    model.config.window_size         = cfg["window_size"]
    model.config.max_capacity_prompt = cfg["max_capacity_prompt"]
    model.config.kernel_size         = cfg["kernel_size"]
    model.config.pooling             = cfg["pooling"]

    fam = _detect_family(model)
    if fam == "llama":
        replace_llama_v5(model)
    else:
        replace_qwen3_v5(model)

    print(
        f"[snap_kv] family={fam} "
        f"window_size={cfg['window_size']} "
        f"max_capacity_prompt={cfg['max_capacity_prompt']} "
        f"kernel_size={cfg['kernel_size']} "
        f"pooling={cfg['pooling']}"
    )


__all__ = [
    "init_snap_kv",
    "replace_llama_v5",
    "replace_qwen3_v5",
    "unpatch_llama_v5",
    "unpatch_qwen3_v5",
]
