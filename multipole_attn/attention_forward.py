"""
Multipole Attention monkey-patch for Qwen2Attention.

Ported from MultipoleAttention/transformers/src/transformers/models/qwen2/modeling_qwen2.py.
Uses a two-step approach:
  1. replace_qwen2_attn_multipole(config_dict) — call BEFORE model loading
  2. init_multipole_layers(model) — call AFTER model loading
"""

import math
import torch
import torch.distributed as dist
import transformers
from typing import Optional, Tuple

from multipole_attn.kernel_wrappers import (
    centroid_lookup,
    dynamic_sparse_attention,
    compute_k_idx_optimized,
    centroid_replacement_kernel,
)
from multipole_attn.clustering import (
    run_clustering_online,
    run_clustering_online_update,
)

# Module-level config set by replace_qwen2_attn_multipole()
_multipole_config = None


# ---------------------------------------------------------------------------
# RoPE helpers (copied from stock transformers to avoid import issues)
# ---------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Replacement forward for Qwen2Attention
# ---------------------------------------------------------------------------
def multipole_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # Derive past_seen_tokens from cache_position (works with stock transformers)
    past_seen_tokens = cache_position[0].item() if cache_position is not None else 0

    # Reset per-example state
    if past_seen_tokens == 0:
        self._mp_num_clusters_lst = None
        self._mp_cos_cache = None
        self._mp_sin_cache = None

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states_rope, key_states_rope = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Compute wRoPE embeddings for centroid lookup
    if self._mp_use_centroids:
        fixed_position = 0
        cos_wrope, sin_wrope = self._mp_rotary_emb(
            hidden_states, torch.tensor([[fixed_position]]).to(query_states.device)
        )
        _, key_states_wrope = _apply_rotary_pos_emb(query_states, key_states, cos_wrope, sin_wrope)

        fixed_position = 2048
        cos_wrope, sin_wrope = self._mp_rotary_emb(
            hidden_states, torch.tensor([[fixed_position]]).to(query_states.device)
        )
        query_states_wrope, _ = _apply_rotary_pos_emb(query_states, key_states, cos_wrope, sin_wrope)
    else:
        key_states_wrope = None
        query_states_wrope = None

    # Determine which path to take
    recompute_context = self._mp_use_centroids and (self._mp_num_clusters_lst is None)
    online_gen = self._mp_use_centroids and not recompute_context and query_states.shape[2] == 1

    # Check for re-clustering
    buffer_len = past_seen_tokens + 1 - self._mp_clustered_length - self._mp_clustered_offset
    recluster = online_gen and buffer_len >= 2 * self._mp_cluster_interval

    # Initial clustering setup
    if recompute_context:
        self._mp_clustered_offset = 10  # attention sink
        prompt_len = key_states.shape[2] - self._mp_clustered_offset
        num_blocks = (prompt_len // self._mp_cluster_interval - 1)
        self._mp_clustered_length = num_blocks * self._mp_cluster_interval
        self._mp_num_clusters_lst = [
            int(self._mp_clustered_length * pct)
            for pct in self._mp_percent_clusters_lst
        ]

    # Update clustered length on recluster
    if recluster:
        self._mp_clustered_length += self._mp_cluster_interval

    # --- Recompute centroids (prefill) ---
    if recompute_context:
        num_levels = len(self._mp_num_clusters_lst)
        rank = (
            dist.get_rank() if self._mp_inference_tp > 1
            else key_states.device.index
        )

        key_states_cluster = key_states_wrope[
            :, :,
            self._mp_clustered_offset : self._mp_clustered_length + self._mp_clustered_offset,
            :,
        ].contiguous()
        value_states_cluster = value_states[
            :, :,
            self._mp_clustered_offset : self._mp_clustered_length + self._mp_clustered_offset,
            :,
        ].contiguous()

        cluster_centers_list, cluster_labels_list, num_keys_list, cluster_labels_list_orig = (
            run_clustering_online(key_states_cluster, self._mp_num_clusters_lst, num_levels, rank=rank)
        )

        # Calculate value centroids
        if self._mp_use_replacement:
            value_centers_lst = []
            for cluster_labels, num_clusters in zip(cluster_labels_list, self._mp_num_clusters_lst):
                temp_value_centers = []
                for H in range(key_states.shape[1]):
                    value_data = value_states_cluster[0, H]
                    cluster_data = cluster_labels[0, H]
                    sums = torch.zeros(num_clusters, value_data.shape[-1], device=value_data.device, dtype=value_data.dtype)
                    sums.index_add_(0, cluster_data, value_data)
                    nk = torch.bincount(cluster_data, minlength=num_clusters)
                    centroids = sums / nk.clamp(min=1).unsqueeze(1)
                    temp_value_centers.append(centroids)
                value_centers_lst.append(torch.stack(temp_value_centers).unsqueeze(0))

        self._mp_cluster_centers_lst = cluster_centers_list
        self._mp_centroid_labels_lst = cluster_labels_list
        self._mp_num_keys_list = num_keys_list
        self._mp_cluster_labels_list_orig = cluster_labels_list_orig
        if self._mp_use_replacement:
            self._mp_value_centers_lst = value_centers_lst

    # Cache cos/sin for RoPE inversion during re-clustering
    if self._mp_use_centroids:
        if self._mp_cos_cache is None:
            self._mp_cos_cache = cos
            self._mp_sin_cache = sin
        else:
            self._mp_cos_cache = torch.cat([self._mp_cos_cache, cos], dim=-2)
            self._mp_sin_cache = torch.cat([self._mp_sin_cache, sin], dim=-2)

    # Cache correct keys (with RoPE applied)
    key_states = key_states_rope.clone()

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # --- Re-clustering during generation ---
    if recluster:
        num_levels = len(self._mp_num_clusters_lst)
        rank = (
            dist.get_rank() if self._mp_inference_tp > 1
            else key_states.device.index
        )

        num_new_clusters = [
            int(self._mp_clustered_length * pct)
            for pct in self._mp_percent_clusters_lst
        ]
        num_added_clusters = [
            new - old
            for new, old in zip(num_new_clusters, self._mp_num_clusters_lst)
        ]
        self._mp_num_clusters_lst = num_new_clusters

        num_added_tokens = [self._mp_cluster_interval]
        if num_levels > 1:
            num_added_tokens = num_added_clusters[1:] + num_added_tokens

        # Reverse RoPE to get raw keys for re-clustering
        key_states_cluster = key_states[
            :, :,
            : self._mp_clustered_offset + self._mp_clustered_length,
            :,
        ].contiguous()

        cos_inv = self._mp_cos_cache[:, : self._mp_clustered_offset + self._mp_clustered_length] / self._mp_attention_scaling
        sin_inv = self._mp_sin_cache[:, : self._mp_clustered_offset + self._mp_clustered_length] / self._mp_attention_scaling
        _, key_states_cluster = _apply_rotary_pos_emb(key_states_cluster, key_states_cluster, cos_inv, -sin_inv)

        key_states_cluster = key_states_cluster[
            :, :,
            self._mp_clustered_offset : self._mp_clustered_offset + self._mp_clustered_length,
            :,
        ]
        value_states_cluster = value_states[
            :, :,
            self._mp_clustered_offset : self._mp_clustered_offset + self._mp_clustered_length,
            :,
        ].contiguous()

        cluster_centers_list, cluster_labels_list, num_keys_list, cluster_labels_list_orig = (
            run_clustering_online_update(
                key_states_cluster,
                self._mp_cluster_centers_lst,
                self._mp_centroid_labels_lst,
                self._mp_num_keys_list,
                num_added_clusters,
                num_added_tokens,
                num_levels,
                rank=rank,
            )
        )

        # Recalculate value centroids
        if self._mp_use_replacement:
            value_centers_lst = []
            for cluster_labels, num_clusters in zip(cluster_labels_list, self._mp_num_clusters_lst):
                temp_value_centers = []
                for H in range(key_states.shape[1]):
                    value_data = value_states_cluster[0, H]
                    cluster_data = cluster_labels[0, H]
                    sums = torch.zeros(num_clusters, value_data.shape[-1], device=value_data.device, dtype=value_data.dtype)
                    sums.index_add_(0, cluster_data, value_data)
                    nk = torch.bincount(cluster_data, minlength=num_clusters)
                    centroids = sums / nk.clamp(min=1).unsqueeze(1)
                    temp_value_centers.append(centroids)
                value_centers_lst.append(torch.stack(temp_value_centers).unsqueeze(0))

        self._mp_cluster_centers_lst = cluster_centers_list
        self._mp_centroid_labels_lst = cluster_labels_list
        self._mp_num_keys_list = num_keys_list
        self._mp_cluster_labels_list_orig = cluster_labels_list_orig
        if self._mp_use_replacement:
            self._mp_value_centers_lst = value_centers_lst

    # --- Centroid lookup ---
    qks = []
    centroid_masks = []
    repl_masks = []

    if online_gen:
        query_states_w = query_states_wrope.clone()
        num_kv_heads = query_states_w.shape[1] // self.num_key_value_groups
        query_states_reshaped = query_states_w.reshape(1, num_kv_heads, self.num_key_value_groups, -1)

        masks = []
        for i in range(len(self._mp_cluster_centers_lst)):
            sm_scale = 1.0 / math.sqrt(query_states_w.size(-1))
            num_keys = self._mp_num_keys_list[i].clone()

            # Mask out centroids at next level
            if i > 0:
                cmask = centroid_masks[i - 1]
                labels = self._mp_cluster_labels_list_orig[i - 1].to(torch.long)
                cmask_expanded = torch.gather(cmask, dim=2, index=labels)
                num_keys[~cmask_expanded] = 0

            msk, centroid_msk, qk = centroid_lookup(
                query_states_reshaped,
                self._mp_cluster_centers_lst[i],
                self._mp_centroid_labels_lst[i],
                num_keys,
                sm_scale,
                self._mp_percentiles_lst[i],
                self.num_key_value_groups,
            )
            msk = msk.squeeze(0)
            masks.append(msk)
            qks.append(qk)
            centroid_masks.append(centroid_msk)

        mask = masks[0]
        for msk in masks[1:]:
            mask = mask & msk

        if self._mp_use_replacement:
            centroid_msk = centroid_masks[0]
            repl_masks.append(~centroid_msk)
            for i in range(1, len(centroid_masks)):
                cmask = centroid_masks[i - 1]
                labels = self._mp_cluster_labels_list_orig[i - 1].to(torch.long)
                cmask_expanded = torch.gather(cmask, dim=2, index=labels)
                centroid_msk = centroid_masks[i] & cmask_expanded
                not_centroid_msk = (~centroid_masks[i]) & cmask_expanded
                repl_masks.append(not_centroid_msk)

    # --- Attention computation ---
    if online_gen:
        assert attention_mask is None

        # Separate input KV: sink + unclustered suffix
        key_states_inp = key_states[:, :, self._mp_clustered_offset + self._mp_clustered_length :, :].contiguous()
        value_states_inp = value_states[:, :, self._mp_clustered_offset + self._mp_clustered_length :, :].contiguous()

        # Prepend attention sink
        key_states_inp = torch.cat((key_states[:, :, : self._mp_clustered_offset, :], key_states_inp), dim=-2)
        value_states_inp = torch.cat((value_states[:, :, : self._mp_clustered_offset, :], value_states_inp), dim=-2)

        # Clustered prefix KV
        key_states_cluster = key_states[
            :, :,
            self._mp_clustered_offset : self._mp_clustered_offset + self._mp_clustered_length,
            :,
        ].contiguous()
        value_states_cluster = value_states[
            :, :,
            self._mp_clustered_offset : self._mp_clustered_offset + self._mp_clustered_length,
            :,
        ].contiguous()

        sm_scale = 1.0 / math.sqrt(query_states.size(-1))

        # Centroid replacement for non-selected tokens
        if self._mp_use_replacement:
            q_w = query_states_wrope.clone()
            num_kv_heads = q_w.shape[1] // self.num_key_value_groups
            q_reshaped = q_w.reshape(1, num_kv_heads, self.num_key_value_groups, -1)

            o = torch.zeros_like(q_reshaped, dtype=torch.float32)
            l = torch.zeros((1, q_reshaped.shape[1], q_reshaped.shape[2]), device=q_reshaped.device, dtype=torch.float32)
            m = torch.zeros((1, q_reshaped.shape[1], q_reshaped.shape[2]), device=q_reshaped.device, dtype=torch.float32) - float("inf")

            for i in range(len(self._mp_num_clusters_lst)):
                num_keys = self._mp_num_keys_list[i].clone()
                if i > 1:
                    cmask = centroid_masks[i - 1]
                    labels = self._mp_cluster_labels_list_orig[i - 1].to(torch.long)
                    cmask_expanded = torch.gather(cmask, dim=2, index=labels)
                    num_keys[~cmask_expanded] = 0

                o, m, l = centroid_replacement_kernel(
                    q_reshaped,
                    self._mp_value_centers_lst[i].float(),
                    repl_masks[i],
                    num_keys,
                    sm_scale,
                    qks[i].float(),
                    o, m, l,
                )
        else:
            num_kv_heads = query_states_rope.shape[1] // self.num_key_value_groups
            q_reshaped = query_states_rope.reshape(1, num_kv_heads, self.num_key_value_groups, -1)
            o = torch.zeros_like(q_reshaped, dtype=torch.float32)
            l = torch.zeros((1, q_reshaped.shape[1], q_reshaped.shape[2]), device=q_reshaped.device, dtype=torch.float32)
            m = torch.zeros((1, q_reshaped.shape[1], q_reshaped.shape[2]), device=q_reshaped.device, dtype=torch.float32) - float("inf")

        # Causal attention to dynamic input tokens (sink + unclustered)
        q_rope = query_states_rope
        num_kv_heads = q_rope.shape[1] // self.num_key_value_groups
        q_reshaped = q_rope.reshape(1, num_kv_heads, self.num_key_value_groups, -1)

        dummy_mask = torch.ones(
            (1, key_states_inp.shape[1], key_states_inp.shape[2]),
            dtype=torch.bool,
            device=key_states_inp.device,
        )
        k_idx, head_kv_len, num_kv_blocks, head_start_block = compute_k_idx_optimized(dummy_mask)
        attn_output_inp, max_val, denom = dynamic_sparse_attention(
            q_reshaped, key_states_inp, value_states_inp,
            k_idx, head_kv_len, num_kv_blocks, head_start_block,
            o, m, l, sm_scale=sm_scale,
        )

        # Sparse attention to clustered prefix
        q_rope2 = query_states_rope.clone()
        num_kv_heads = q_rope2.shape[1] // self.num_key_value_groups
        q_reshaped2 = q_rope2.reshape(1, num_kv_heads, self.num_key_value_groups, -1)

        k_idx, head_kv_len, num_kv_blocks, head_start_block = compute_k_idx_optimized(mask.unsqueeze(0))
        attn_output, _, _ = dynamic_sparse_attention(
            q_reshaped2, key_states_cluster, value_states_cluster,
            k_idx, head_kv_len, num_kv_blocks, head_start_block,
            attn_output_inp, max_val, denom, sm_scale=sm_scale,
        )

        attn_output = attn_output.reshape(1, attn_output.shape[1] * attn_output.shape[2], 1, -1)
        attn_output = attn_output.transpose(1, 2).to(query_states.dtype)
        attn_weights = None

    else:
        # Prefill or non-centroid path: use standard attention
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        query_states_attn = query_states_rope
        if query_states_attn.shape[2] > 1:  # prefill
            key_states = key_states_rope

        sliding_window = getattr(self, "sliding_window", None)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get(
            self.config._attn_implementation, None
        )
        if attention_interface is None:
            # Fallback to eager
            from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states_attn,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Two-step monkey-patch API
# ---------------------------------------------------------------------------
def replace_qwen2_attn_multipole(config_dict):
    """
    Step 1: Monkey-patch Qwen2Attention.forward. Call BEFORE model loading.
    """
    global _multipole_config
    _multipole_config = config_dict

    print("Multipole Attention config:")
    print(f"  use_centroids={config_dict['use_centroids']}")
    print(f"  percent_clusters_lst={config_dict['percent_clusters_lst']}")
    print(f"  percentiles_lst={config_dict['percentiles_lst']}")
    print(f"  use_replacement={config_dict['use_replacement']}")
    print(f"  cluster_interval={config_dict['cluster_interval']}")
    print(f"  inference_tp={config_dict['inference_tp']}")
    print(f"  Multipole attention active during decode only (prefill uses full attention)")

    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = multipole_attention_forward


def init_multipole_layers(model):
    """
    Step 2: Inject per-layer state after model loading.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    config = _multipole_config

    for layer in model.model.layers:
        attn = layer.self_attn
        attn._mp_use_centroids = config["use_centroids"]
        attn._mp_use_replacement = config["use_replacement"]
        attn._mp_percentiles_lst = config["percentiles_lst"]
        attn._mp_percent_clusters_lst = [p / 100.0 for p in config["percent_clusters_lst"]]
        attn._mp_num_clusters_lst = None
        attn._mp_cluster_interval = config["cluster_interval"]
        attn._mp_clustered_length = 0
        attn._mp_clustered_offset = 0
        attn._mp_inference_tp = config["inference_tp"]
        attn._mp_rotary_emb = Qwen2RotaryEmbedding(config=model.config).to(attn.q_proj.weight.device)
        attn._mp_attention_scaling = attn._mp_rotary_emb.attention_scaling
        attn._mp_cos_cache = None
        attn._mp_sin_cache = None
