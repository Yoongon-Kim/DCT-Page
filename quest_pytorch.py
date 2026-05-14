"""Pure-PyTorch Quest baseline (no custom CUDA kernels), gather-based sparse attention.

Monkey-patches HF Qwen3 / Llama Attention.forward with Quest scoring + gather-based
sparse attention. Two GQA modes:

  - ``per_qo_head``: 32 query heads each pick their own top-K pages (matches
    upstream Quest paper). Per-qo_head gather from native GQA KV (4× temporary
    materialization at attention time, but KV cache stays at num_kv_heads).
  - ``per_kv_head``: group_agg(max) over GQA group → 8 selections shared by 4
    query heads each. Same granularity as DCT-Page's selector — apples-to-apples
    selector comparison.

Both modes use REAL sparse attention via gather + SDPA (only K_top × page_size
tokens read/computed). Differs from a mask-based approach (which computes full
attention and -inf masks — no compute reduction).

Falls back to dense SDPA for prefill (q_len > 1), shallow layers
(layer_idx < skip_layers), or short context (kv_len ≤ min_kv_len_for_sparsity).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


_QUEST_PT_CFG = {
    "page_size": 32,
    "token_budget": 2048,
    "skip_layers": 2,
    "min_kv_len_for_sparsity": 2048,
    "gqa_mode": "per_qo_head",   # "per_qo_head" | "per_kv_head"
}


def _quest_pytorch_decode_gather(
    query_states: torch.Tensor,   # [B, H_q, 1, d]
    key_states: torch.Tensor,     # [B, H_kv, kv_seq, d]
    value_states: torch.Tensor,   # [B, H_kv, kv_seq, d]
    num_kv_groups: int,
    page_size: int,
    token_budget: int,
    gqa_mode: str,
) -> torch.Tensor:
    """Gather-based sparse Quest attention. Returns [B, H_q, 1, d]."""
    bsz, num_heads, _, d = query_states.shape
    num_kv_heads = key_states.shape[1]
    kv_seq = key_states.shape[2]

    # The "open chunk" = chunk containing the just-appended token (kv_seq-1).
    # We always attend to it (forced), so exclude from topk candidates.
    open_start = (kv_seq - 1) // page_size * page_size
    n_chunks_full = open_start // page_size
    if open_start < kv_seq:
        open_len = kv_seq - open_start
    else:
        open_len = 0

    # If no selectable chunks (context fits in single open chunk), dense attention.
    if n_chunks_full <= 0:
        return F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=False, scale=1.0 / math.sqrt(d), enable_gqa=True,
        )

    # Per-chunk K_max, K_min over selectable chunks (excludes open chunk).
    K_selectable = key_states[:, :, :open_start, :]
    K_chunked = K_selectable.view(bsz, num_kv_heads, n_chunks_full, page_size, d)
    K_max_per_chunk = K_chunked.amax(dim=-2)                                   # [B, H_kv, n_chunks_full, d]
    K_min_per_chunk = K_chunked.amin(dim=-2)

    # Quest scoring per-(B, H_kv, group_member, chunk) via grouped einsum.
    # Σ_d max(q[d]·K_max[d], q[d]·K_min[d]) = einsum(q+, K_max) + einsum(q-, K_min)
    q = query_states.squeeze(-2).float()                                       # [B, H_q, d]
    q_grouped = q.view(bsz, num_kv_heads, num_kv_groups, d)
    q_pos = q_grouped.clamp(min=0)
    q_neg = q_grouped.clamp(max=0)
    score_pos = torch.einsum('bhgd,bhcd->bhgc', q_pos, K_max_per_chunk.float())
    score_neg = torch.einsum('bhgd,bhcd->bhgc', q_neg, K_min_per_chunk.float())
    quest_scores_per_head = score_pos + score_neg                              # [B, H_kv, G, n_chunks_full]

    K_pages = min(token_budget // page_size, n_chunks_full)
    offsets = torch.arange(page_size, device=key_states.device)

    if gqa_mode == "per_kv_head":
        # Group aggregation (max) → per-kv_head selection.
        page_scores = quest_scores_per_head.amax(dim=2)                        # [B, H_kv, n_chunks_full]
        topk_idx = page_scores.topk(K_pages, dim=-1).indices                   # [B, H_kv, K_pages]
        token_idx = (topk_idx.unsqueeze(-1) * page_size + offsets).flatten(-2) # [B, H_kv, K_pages*ps]
        token_idx_exp = token_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        gathered_K = key_states.gather(2, token_idx_exp)                       # [B, H_kv, K_pages*ps, d]
        gathered_V = value_states.gather(2, token_idx_exp)

        # Append open chunk (forced inclusion, per-kv_head shared).
        if open_len > 0:
            open_K = key_states[:, :, open_start:kv_seq, :]
            open_V = value_states[:, :, open_start:kv_seq, :]
            gathered_K = torch.cat([gathered_K, open_K], dim=-2)
            gathered_V = torch.cat([gathered_V, open_V], dim=-2)

        # GQA SDPA: Q at H_q, K/V at H_kv. enable_gqa handles expansion in attention.
        return F.scaled_dot_product_attention(
            query_states, gathered_K, gathered_V,
            is_causal=False, scale=1.0 / math.sqrt(d), enable_gqa=True,
        )

    elif gqa_mode == "per_qo_head":
        # Per-query-head selection.
        per_qo_scores = quest_scores_per_head.reshape(bsz, num_heads, n_chunks_full)
        topk_idx = per_qo_scores.topk(K_pages, dim=-1).indices                 # [B, H_q, K_pages]
        token_idx = (topk_idx.unsqueeze(-1) * page_size + offsets).flatten(-2) # [B, H_q, K_pages*ps]

        # Per-qo_head gather from H_kv-storage. Each qo_head h fetches its
        # kv_head = h // num_kv_groups. Use advanced indexing — no temp expand.
        batch_idx = torch.arange(bsz, device=key_states.device).view(bsz, 1, 1)
        h_kv_idx = (torch.arange(num_heads, device=key_states.device) // num_kv_groups).view(1, num_heads, 1)
        batch_idx = batch_idx.expand_as(token_idx)
        h_kv_idx = h_kv_idx.expand_as(token_idx)
        gathered_K = key_states[batch_idx, h_kv_idx, token_idx]                # [B, H_q, K_pages*ps, d]
        gathered_V = value_states[batch_idx, h_kv_idx, token_idx]

        # Open chunk per qo_head's kv_head (broadcasted to qo_head granularity).
        if open_len > 0:
            open_K_kv = key_states[:, :, open_start:kv_seq, :]                 # [B, H_kv, open_len, d]
            open_V_kv = value_states[:, :, open_start:kv_seq, :]
            open_K_qo = open_K_kv.repeat_interleave(num_kv_groups, dim=1)      # [B, H_q, open_len, d]
            open_V_qo = open_V_kv.repeat_interleave(num_kv_groups, dim=1)
            gathered_K = torch.cat([gathered_K, open_K_qo], dim=-2)
            gathered_V = torch.cat([gathered_V, open_V_qo], dim=-2)

        # Per-head SDPA: K/V already at H_q granularity, no GQA expansion needed.
        return F.scaled_dot_product_attention(
            query_states, gathered_K, gathered_V,
            is_causal=False, scale=1.0 / math.sqrt(d),
        )

    else:
        raise ValueError(f"gqa_mode must be 'per_qo_head' or 'per_kv_head', got {gqa_mode}")


def quest_pytorch_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Replacement forward for Qwen3Attention / LlamaAttention (transformers v5)."""
    cfg = _QUEST_PT_CFG
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs,
        )

    kv_seq_len = key_states.shape[-2]
    use_dense = (
        q_len > 1
        or self.layer_idx < cfg["skip_layers"]
        or kv_seq_len <= cfg["min_kv_len_for_sparsity"]
    )

    if use_dense:
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            is_causal=(q_len > 1 and attention_mask is None),
            scale=1.0 / math.sqrt(self.head_dim),
            enable_gqa=True,
        )
    else:
        attn_output = _quest_pytorch_decode_gather(
            query_states, key_states, value_states,
            num_kv_groups=self.num_key_value_groups,
            page_size=cfg["page_size"],
            token_budget=cfg["token_budget"],
            gqa_mode=cfg["gqa_mode"],
        )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _patch_attention_class(attn_cls):
    attn_cls.forward = quest_pytorch_forward


def replace_qwen3_attn_quest_pytorch(page_size=32, token_budget=2048,
                                    skip_layers=2, min_kv_len_for_sparsity=2048,
                                    gqa_mode="per_qo_head"):
    global _QUEST_PT_CFG
    _QUEST_PT_CFG.update({
        "page_size": page_size,
        "token_budget": token_budget,
        "skip_layers": skip_layers,
        "min_kv_len_for_sparsity": min_kv_len_for_sparsity,
        "gqa_mode": gqa_mode,
    })
    from transformers.models.qwen3 import modeling_qwen3
    _patch_attention_class(modeling_qwen3.Qwen3Attention)
    print(f"[quest_pytorch] Qwen3Attention patched (page_size={page_size}, "
          f"token_budget={token_budget}, skip_layers={skip_layers}, "
          f"gqa_mode={gqa_mode})")


def replace_llama_attn_quest_pytorch(page_size=32, token_budget=2048,
                                    skip_layers=2, min_kv_len_for_sparsity=2048,
                                    gqa_mode="per_qo_head"):
    global _QUEST_PT_CFG
    _QUEST_PT_CFG.update({
        "page_size": page_size,
        "token_budget": token_budget,
        "skip_layers": skip_layers,
        "min_kv_len_for_sparsity": min_kv_len_for_sparsity,
        "gqa_mode": gqa_mode,
    })
    from transformers.models.llama import modeling_llama
    _patch_attention_class(modeling_llama.LlamaAttention)
    print(f"[quest_pytorch] LlamaAttention patched (page_size={page_size}, "
          f"token_budget={token_budget}, skip_layers={skip_layers}, "
          f"gqa_mode={gqa_mode})")
