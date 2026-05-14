"""SnapKV — live patch for transformers==5.2.0 (Llama 3.x + Qwen3).

Live path:
    init_snap_kv(model, cfg)
      -> replace_llama_v5(model) | replace_qwen3_v5(model)
          - patches {Llama|Qwen3}Attention.forward = _snapkv_attention_forward
          - eagerly initializes layer.self_attn.kv_cluster for each layer

Algorithm reuse: SnapKVCluster from upstream/snapkv/monkeypatch/snapkv_utils.py
(model-, transformers-version-, AND head-count-agnostic, vendored as a museum).

GQA reduction strategy: (c) group-mean Q (plan §3.7). K and V stay un-repeated
at num_kv_heads; only Q is reduced for scoring. Per-kv-head K/V identity is
preserved; cluster output is directly writable to the cache.

Cache rewrite: only supports DynamicLayer (vanilla DynamicCache). Hard
runtime assert blocks Static/Hybrid layers — see plan §3.3.

No prepare_inputs_for_generation patch (plan §0.1, §3.6). _update_model_kwargs_
for_generation advances cache_position independently of get_seq_length();
RoPE's relative-rotation property makes the compressed K cache holding
original-position phases correct under generate().

DO NOT EDIT baselines/snap_kv/upstream/*. See VENDORING.md.
"""
import torch
import transformers
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
# Direct callable default for attention dispatch ("eager" is NOT registered
# in ALL_ATTENTION_FUNCTIONS._global_mapping — see plan §3.4).
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# Algorithm — verbatim from upstream museum. NEVER monkey-patch this file.
from .upstream.snapkv.monkeypatch.snapkv_utils import SnapKVCluster

# --- module globals (mirrors dct_page_attention.py:134) ---
_original_llama_forward = None
_original_qwen3_forward = None


def init_snapkv_v5(self_attn) -> None:
    """Eagerly register (or re-initialize) kv_cluster on a {Llama|Qwen3}Attention module.

    Called at replace_*_v5(model) boot time, NOT on the hot path.
    Reads window_size / max_capacity_prompt / kernel_size / pooling from
    self_attn.config (set by init_snap_kv in __init__.py before replace_*_v5).

    Always re-initializes the cluster so that a second call to init_snap_kv()
    with a different max_capacity_prompt / window_size takes effect immediately
    (e.g. smoke-test parity run followed by compression run on the same model).
    """
    cfg = self_attn.config
    if not hasattr(cfg, "window_size"):         cfg.window_size = 32
    if not hasattr(cfg, "max_capacity_prompt"): cfg.max_capacity_prompt = 2048
    if not hasattr(cfg, "kernel_size"):         cfg.kernel_size = 5
    if not hasattr(cfg, "pooling"):             cfg.pooling = "avgpool"
    if hasattr(self_attn, "kv_cluster"):
        # Re-use the existing object to avoid per-layer allocation churn.
        self_attn.kv_cluster.reset(
            window_size=cfg.window_size,
            max_capacity_prompt=cfg.max_capacity_prompt,
            kernel_size=cfg.kernel_size,
            pooling=cfg.pooling,
        )
    else:
        self_attn.kv_cluster = SnapKVCluster(
            window_size=cfg.window_size,
            max_capacity_prompt=cfg.max_capacity_prompt,
            kernel_size=cfg.kernel_size,
            pooling=cfg.pooling,
        )


def _snapkv_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """Unified replacement for both LlamaAttention.forward and Qwen3Attention.forward.

    Contract:
      - Prefill (q_len > 1 and the cache layer was empty before this update):
        compute QKV, apply RoPE, do standard cache append; if the resulting
        cache length >= max_capacity_prompt, group-mean Q across kv-groups,
        run SnapKVCluster on the (num_kv_heads-shaped) full K/V with the
        group-mean'd Q, overwrite the cache layer's .keys/.values with the
        compressed tensors.
      - Decode (q_len == 1, or short prefill below capacity): standard cache
        append + attention; no further compression.

    State-reset is INLINE — detected via `full_k.shape[-2] == q_len` after
    Cache.update returns. No per-module sticky flag. No prepare_inputs
    wrapper.

    Dispatches q_norm/k_norm via hasattr; sliding_window via getattr.
    """
    # Dev-only sanity (init_snapkv_v5 should have run at boot).
    assert hasattr(self, "kv_cluster"), \
        "kv_cluster missing — replace_{llama,qwen3}_v5(model) was not called."

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape
    _has_qk_norm = hasattr(self, "q_norm") and hasattr(self, "k_norm")

    # Step 1: Q/K/V projection (with Qwen3 q_norm/k_norm)
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states   = self.k_proj(hidden_states).view(hidden_shape)
    if _has_qk_norm:
        query_states = self.q_norm(query_states)
        key_states   = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)    # (B, num_heads,    q_len, D)
    key_states   = key_states.transpose(1, 2)      # (B, num_kv_heads, q_len, D)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Step 2: RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Step 3: SnapKV cluster + cache update
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Standard cache append. full_k/full_v are post-append (num_kv_heads-shaped).
        full_k, full_v = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

        # Inline state-reset: the layer was empty before this update iff the
        # post-append length equals q_len. True on fresh prefill. False on
        # any decode step (cache has prior tokens) and on multi-call usage
        # where the cache carries history. No external state needed.
        is_prefill_to_compress = (
            q_len > 1
            and full_k.shape[-2] == q_len
            and full_k.shape[-2] >= self.kv_cluster.max_capacity_prompt
        )

        if is_prefill_to_compress:
            # Runtime guard — Static/Hybrid layers preallocate fixed-shape
            # tensors; overwriting .keys/.values silently corrupts
            # subsequent index_copy_ calls. (plan §3.3)
            layer = past_key_values.layers[self.layer_idx]
            assert isinstance(layer, DynamicLayer), (
                f"SnapKV v5 requires DynamicLayer (DynamicCache); got "
                f"{type(layer).__name__}. Pass past_key_values=DynamicCache(...) "
                "explicitly to model.generate(), or omit it (default)."
            )

            # GQA-reduction strategy (c) — plan §3.7. Group-mean Q only;
            # K/V stay un-repeated at num_kv_heads. Cluster's
            # head-count-agnostic body (verified by snapkv_utils.py:41
            # T-axis-only assertion) handles num_kv_heads-shaped inputs.
            B, H, _q, D = query_states.shape
            G = self.num_key_value_groups
            query_for_cluster = query_states.view(B, H // G, G, _q, D).mean(dim=2)
            # query_for_cluster: (B, num_kv_heads, q_len, D)

            k_compressed, v_compressed = self.kv_cluster.update_kv(
                full_k, query_for_cluster, full_v, attention_mask, G
            )
            # k_compressed, v_compressed: (B, num_kv_heads, max_capacity_prompt, D)

            # Overwrite the cache layer (DynamicLayer public attribute contract;
            # cache_utils.py:33-34, :120-121).
            layer.keys   = k_compressed.contiguous()
            layer.values = v_compressed.contiguous()

            # Use the compressed tensors for this layer's own forward output.
            # attention_interface will call repeat_kv internally on them.
            key_states_attn   = layer.keys
            value_states_attn = layer.values
            # The causal attention_mask was built for q_len keys; after
            # compression the KV sequence is max_capacity_prompt tokens.
            # All compressed tokens are historical — no causal mask needed.
            # Pass None so eager/sdpa don't broadcast the wrong shape.
            attn_mask_for_interface = None
        else:
            # Decode (or short-prefill below capacity).
            key_states_attn       = full_k
            value_states_attn     = full_v
            attn_mask_for_interface = attention_mask
    else:
        # No cache — pure forward pass, no clustering (matches upstream).
        key_states_attn       = key_states
        value_states_attn     = value_states
        attn_mask_for_interface = attention_mask

    # Step 4: attention dispatch. Direct callable default (plan §3.4).
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    # Prevent duplicate-kwarg TypeError if upstream caller stuffed
    # sliding_window into kwargs.
    kwargs.pop("sliding_window", None)
    sw = getattr(self, "sliding_window", None)
    extra = {"sliding_window": sw} if sw is not None else {}

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states_attn,
        value_states_attn,
        attn_mask_for_interface,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **extra,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def replace_llama_v5(model) -> None:
    """Install SnapKV forward on LlamaAttention.
    Eagerly initializes kv_cluster on every layer."""
    global _original_llama_forward
    if _original_llama_forward is None:
        _original_llama_forward = transformers.models.llama.modeling_llama.LlamaAttention.forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = _snapkv_attention_forward

    for layer in model.model.layers:
        init_snapkv_v5(layer.self_attn)

    print(f"[snapkv] patched LlamaAttention.forward; "
          f"initialized kv_cluster on {len(model.model.layers)} layers")


def unpatch_llama_v5(model=None) -> None:
    """Restore the original LlamaAttention.forward. Idempotent."""
    global _original_llama_forward
    if _original_llama_forward is not None:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = _original_llama_forward
        _original_llama_forward = None


def replace_qwen3_v5(model) -> None:
    """Install SnapKV forward on Qwen3Attention."""
    global _original_qwen3_forward
    if _original_qwen3_forward is None:
        _original_qwen3_forward = transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = _snapkv_attention_forward

    for layer in model.model.layers:
        init_snapkv_v5(layer.self_attn)

    print(f"[snapkv] patched Qwen3Attention.forward; "
          f"initialized kv_cluster on {len(model.model.layers)} layers")


def unpatch_qwen3_v5(model=None) -> None:
    """Restore the original Qwen3Attention.forward. Idempotent."""
    global _original_qwen3_forward
    if _original_qwen3_forward is not None:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = _original_qwen3_forward
        _original_qwen3_forward = None
