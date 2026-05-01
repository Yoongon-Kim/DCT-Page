import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

import quest_attn.utils
from quest_attn.utils import rms_norm_forward


class _HeadRMSNorm(nn.Module):
    """RMSNorm on head_dim for QK-norm (Qwen3-style). Uses the same CUDA kernel."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_forward(x, self.weight, self.eps)

class QuestAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = getattr(config, 'pretraining_tp', 1)
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Optional QK-norm (Qwen3 applies RMSNorm on head_dim BEFORE RoPE)
        _model_type = getattr(config, 'model_type', 'llama')
        if _model_type == 'qwen3':
            self.q_norm = _HeadRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = _HeadRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self._init_rope()

    def _init_rope(self):
        self.rope_theta = getattr(self.config, 'rope_theta', 1e4)
        # rope_theta is default to 1e4, as set in RoPE kernel API.
        rope_scaling = getattr(self.config, 'rope_scaling', None)
        if rope_scaling is None:
            self.rotary_emb = None  # RoPE applied by FlashInfer kernel
            self.rope_scale = 1.0
            return

        scaling_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
        if scaling_type in (None, "default"):
            self.rotary_emb = None
            self.rope_scale = 1.0
        elif scaling_type == "linear":
            # support for Longchat-v1.5.
            self.rotary_emb = None
            self.rope_scale = rope_scaling["factor"]
        elif scaling_type in ("llama3", "dynamic", "yarn", "longrope"):
            # Non-uniform per-dim scaling cannot be expressed as a scalar
            # rope_scale. Fall back to HF's rotary embedding and skip the
            # fused FlashInfer RoPE kernel for this layer.
            self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
            self.rope_scale = None  # signals "use self.rotary_emb in forward"
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        iController: Optional[quest_attn.utils.InferenceController] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        assert bsz == 1, "QuestAttention only supports batch size 1."
        assert hasattr(self, 'layer_idx'), "QuestAttention requires layer_idx to inference."

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            torch.cuda.nvtx.range_push("qkv_proj")
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            torch.cuda.nvtx.range_pop()

        # Not transposed for Append kv cache NHD layout
        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        # Optional QK-norm (Qwen3 applies norm BEFORE RoPE)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        torch.cuda.nvtx.range_push("RoPE")
        if self.rope_scale is not None:
            quest_attn.utils.apply_rope_in_place(query_states, key_states, iController.kv_cache.seqlen - q_len, rope_scale=self.rope_scale, rope_theta=self.rope_theta)
        else:
            # Non-uniform RoPE (e.g. Llama-3 piecewise scaling): use HF rotary.
            past_len = iController.kv_cache.seqlen - q_len
            position_ids = torch.arange(
                past_len, past_len + q_len,
                device=query_states.device, dtype=torch.long,
            ).unsqueeze(0)
            # [N, H, D] -> [1, H, N, D] for HF apply_rotary_pos_emb
            q4d = query_states.unsqueeze(0).transpose(1, 2)
            k4d = key_states.unsqueeze(0).transpose(1, 2)
            cos, sin = self.rotary_emb(q4d, position_ids)
            q4d, k4d = apply_rotary_pos_emb(q4d, k4d, cos, sin)
            query_states = q4d.transpose(1, 2).squeeze(0).contiguous()
            key_states = k4d.transpose(1, 2).squeeze(0).contiguous()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("append_kv")
        # Quest manages KV-Cache internal (with PageAttention)
        # Here we do not concat / stack
        # We concat after RoPE
        quest_attn.utils.append_kv(
            key_states,
            value_states,
            iController,
            self.layer_idx,
        )
        torch.cuda.nvtx.range_pop()

        # Prefill/Decode kernels is different
        if q_len > 1:
            torch.cuda.nvtx.range_push("prefill_attn")
            attn_output = quest_attn.utils.prefill_forward(
                query_states,
                iController,
                self.layer_idx,
            )
            torch.cuda.nvtx.range_pop()
        else:
            # Skipping layers is controled by PAGE_BUDGET, which is set in LlamaModel.
            if iController.need_estimate() == False:
                torch.cuda.nvtx.range_push("full_attn")
                attn_output = quest_attn.utils.decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.kv_indices_without_last,
                )
                torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.nvtx.range_push("estimate")
                estimated_attn_score = quest_attn.utils.decode_estimate(
                    query_states,
                    iController,
                    self.layer_idx,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("topk")
                quest_attn.utils.decode_topk(
                    estimated_attn_score,
                    iController,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("approx_attn")
                attn_output = quest_attn.utils.decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.topk_dindices_buffer,
                )
                torch.cuda.nvtx.range_pop()

        attn_output = attn_output.unsqueeze(0) # unsqueeze the batch dimension
        # FlashInfer output is naturally NHD
        # Note that we manully control NHD. Should be more general
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        torch.cuda.nvtx.range_push("o_proj")
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
