import torch
from ..attention import RotaryEmbeddingESM, ATTN_FORWRAD


def huggingface_forward(forward):
    """Adapter from the upstream InfLLM attention forward to transformers 5.x's
    LlamaAttention.forward signature.

    The 5.x attention forward returns a 2-tuple (output, attn_weights). InfLLM's
    inner forward returns (o, past_key_value) when use_cache=True. We stash pkv on
    the attention module as self._infllm_kv so the patched LlamaModel.forward can
    read it back without breaking the 2-tuple return contract.
    """
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,        # (cos, sin); UNUSED - InfLLM uses RotaryEmbeddingESM via self.position_bias
        attention_mask=None,
        past_key_values=None,            # NOT a transformers.Cache - this is a ContextManager (or None on first call)
        cache_position=None,
        **kwargs,
    ):
        del position_embeddings, cache_position
        ret = forward(
            self, hidden_states, hidden_states,
            self.position_bias,          # custom RoPE injected by patch_hf
            True,                        # use_cache always True under this wrapper
            past_key_values,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.head_dim,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )
        if isinstance(ret, tuple):
            o, pkv = ret
        else:
            o, pkv = ret, None
        self._infllm_kv = pkv            # side-channel for model_forward to collect
        return o, None                   # 2-tuple matches LlamaDecoderLayer's unpack

    return hf_forward


def patch_hf(
    model,
    attn_type: str = "inf_llm",
    attn_kwargs: dict = {},
    base = None,
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.mistral.modeling_mistral import MistralAttention, MistralModel
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        cache_position = None,
        use_cache = None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        pkv = tuple() if use_cache else None
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            # past_key_values is a tuple of ContextManager (or None on first call).
            # We bypass HF Cache entirely; the patched hf_forward stashes the new
            # pkv at decoder_layer.self_attn._infllm_kv for us to collect below.
            decoder_layer.self_attn._infllm_kv = None
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=None,        # ignored by patched LlamaAttention.forward
                past_key_values=past_key_values[i] if past_key_values is not None else None,
                cache_position=None,
                use_cache=True,
            )
            # LlamaDecoderLayer.forward returns a single Tensor in 5.x, NOT a tuple.
            if use_cache:
                pkv = pkv + (decoder_layer.self_attn._infllm_kv,)
            decoder_layer.self_attn._infllm_kv = None  # clear stale state

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=None,
            attentions=None,
        )

    forward = huggingface_forward(ATTN_FORWRAD[attn_type](**attn_kwargs))

    if isinstance(model, LlamaForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, MistralForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, Qwen2ForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif model.__class__.__name__ == "MiniCPMForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    else:
        raise ValueError("Only supports llama, mistral and qwen2 models.")

    # 5.x: LlamaRotaryEmbedding lives on the model, not on each attention instance;
    # its `dim` and `base` attrs are gone — read from config instead.
    # rope_theta moved from `config.rope_theta` (4.37) to nested under
    # `config.rope_parameters` / `config.rope_scaling` (5.x). Walk the chain.
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    if base is None:
        cfg = model.config
        # Order: 5.x rope_parameters → 5.x rope_scaling fallback → legacy attr → default
        for src in (getattr(cfg, "rope_parameters", None), getattr(cfg, "rope_scaling", None)):
            if isinstance(src, dict) and "rope_theta" in src:
                rope_base = src["rope_theta"]
                break
        else:
            rope_base = getattr(cfg, "rope_theta", 10000.0)
    else:
        rope_base = base
    distance_scale = distance_scale if distance_scale is not None else 1.0
    rope = RotaryEmbeddingESM(head_dim, rope_base, distance_scale)
    print(f"[inf_llm.patch_hf] rope_base={rope_base} head_dim={head_dim}")
    model.model.position_bias = rope
    # One-shot per-attention assignment (hot-path-safe; position_bias never changes)
    for layer in model.model.layers:
        layer.self_attn.position_bias = rope

    # Class-level patching mirrors dct_page_attention.py:2529 and wins over
    # 5.x's `attention_interface` dispatch (which runs INSIDE LlamaAttention.forward).
    Attention.forward = forward
    Model.forward = model_forward

    return model
