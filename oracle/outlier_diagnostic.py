"""Diagnose outlier-bank candidates before implementing.

For each decode step on a few RULER samples, install a dense Q/K/V recording
hook (re-using the infrastructure from attention_mass_recall_ruler_quest.py).
At step 0 per (sample, layer), build several candidate outlier sets of size
M tokens per kv_head from the pageable region only (positions outside
[0, sink_tokens) and (kv_len - recent_tokens, kv_len)):

    * knorm        : top-M L2 norm of K
    * lastq_max    : top-M max over qo-heads in group of (q · k)
    * lastq_mean   : top-M mean over qo-heads in group of (q · k)
    * random       : random M positions (sanity floor)

Plus one dynamic set recomputed every step:

    * oracle_step  : top-M (q_t · k)            (upper bound)

For every decode step we compute the true full-KV softmax mass per kv_head
(mean-aggregated across qo-heads in the GQA group), then sum mass over each
candidate set → coverage. We also bucket the static set indices by position
to see whether candidates cluster near the sink edge.

Output: results_outlier_diag/<run_name>/{summary.json, per_sample.jsonl}.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "oracle"))

from eval_ruler import infer_model_family  # noqa: E402
from oracle.attention_mass_recall_ruler import load_task_configs  # noqa: E402
from oracle.attention_mass_recall_ruler_quest import (  # noqa: E402
    _model_family,
    load_model,
    set_recording_hook,
)


# ---------------------------------------------------------------------------
# Dual recording forward: fires hook on prefill (with last-N Q) AND decode.
# ---------------------------------------------------------------------------
def _install_dual_recording_forward(model, family: str, prefill_last_n: int) -> None:
    if family == "llama":
        from transformers.models.llama.modeling_llama import (
            LlamaAttention, apply_rotary_pos_emb, eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        target_cls = LlamaAttention
        has_qk_norm = False
    elif family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention, apply_rotary_pos_emb, eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        target_cls = Qwen3Attention
        has_qk_norm = True
    elif family == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Attention, apply_rotary_pos_emb, eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        target_cls = Qwen2Attention
        has_qk_norm = False
    else:
        raise ValueError(f"unsupported family {family!r}")

    import oracle.attention_mass_recall_ruler_quest as _qmod

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_values=None,
                cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if has_qk_norm:
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        q_len = query_states.shape[-2]
        hook = _qmod._recording_hook
        if hook is not None:
            if q_len == 1:
                hook({
                    "phase": "decode",
                    "layer_idx": int(self.layer_idx),
                    "query_states": query_states,
                    "key_states_full": key_states,
                    "value_states_full": value_states,
                    "num_kv_groups": int(self.num_key_value_groups),
                })
            elif q_len > 1:
                last_n = min(prefill_last_n, q_len)
                hook({
                    "phase": "prefill",
                    "layer_idx": int(self.layer_idx),
                    "prefill_last_q": query_states[:, :, -last_n:, :].detach(),
                    "num_kv_groups": int(self.num_key_value_groups),
                })

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward,
        )
        qwen3_extra = {}
        if family == "qwen3":
            qwen3_extra["sliding_window"] = getattr(self, "sliding_window", None)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **qwen3_extra, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    target_cls.forward = forward


# Position bucket edges (left-inclusive) for the histogram.
POS_BUCKETS = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def _bucketize(positions: torch.Tensor, edges: list[int]) -> list[int]:
    """Counts in each [edges[i], edges[i+1]) bucket; last bucket is open-ended."""
    counts = [0] * len(edges)
    p = positions.flatten().tolist()
    for v in p:
        # Place in highest bucket whose left edge <= v
        idx = 0
        for i, e in enumerate(edges):
            if v >= e:
                idx = i
        counts[idx] += 1
    return counts


def _kmeans(K: torch.Tensor, N: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-kv-head k-means (single-level, hard assignment).

    Args:
        K: [H_kv, T, d] candidate K vectors
        N: number of clusters
        iters: number of Lloyd iterations
    Returns:
        centroids: [H_kv, N, d]
        cluster_ids: [H_kv, T] long
    """
    H_kv, T, d = K.shape
    g = torch.Generator(device=K.device).manual_seed(seed)
    # Random init: pick N distinct token indices per kv_head
    init_idx = torch.stack([
        torch.randperm(T, generator=g, device=K.device)[:N]
        for _ in range(H_kv)
    ])  # [H_kv, N]
    centroids = K.gather(1, init_idx.unsqueeze(-1).expand(-1, -1, d)).clone()  # [H_kv, N, d]
    cluster_ids = torch.zeros(H_kv, T, dtype=torch.long, device=K.device)
    for _ in range(iters):
        # Assign: each token to nearest centroid by L2 distance (= argmin -2 K·C + ||C||²).
        c_norm_sq = (centroids * centroids).sum(-1)               # [H_kv, N]
        dots = torch.einsum("htd, hnd -> htn", K, centroids)      # [H_kv, T, N]
        d_sq = c_norm_sq.unsqueeze(1) - 2.0 * dots                 # K·K constant per token, irrelevant for argmin
        cluster_ids = d_sq.argmin(dim=-1)                          # [H_kv, T]
        # Update centroids
        for h in range(H_kv):
            for c in range(N):
                mask = cluster_ids[h] == c
                if mask.any():
                    centroids[h, c] = K[h, mask].mean(dim=0)
    return centroids, cluster_ids


def _kmeans_vectorized(K: torch.Tensor, N: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Faster k-means via scatter_add for centroid update."""
    H_kv, T, d = K.shape
    g = torch.Generator(device=K.device).manual_seed(seed)
    init_idx = torch.stack([
        torch.randperm(T, generator=g, device=K.device)[:N]
        for _ in range(H_kv)
    ])  # [H_kv, N]
    centroids = K.gather(1, init_idx.unsqueeze(-1).expand(-1, -1, d)).clone().float()
    cluster_ids = torch.zeros(H_kv, T, dtype=torch.long, device=K.device)
    Kf = K.float()
    for _ in range(iters):
        c_norm_sq = (centroids * centroids).sum(-1)
        dots = torch.einsum("htd, hnd -> htn", Kf, centroids)
        d_sq = c_norm_sq.unsqueeze(1) - 2.0 * dots
        cluster_ids = d_sq.argmin(dim=-1)                         # [H_kv, T]
        # Scatter-add sums and counts per cluster, per kv_head
        sums = torch.zeros(H_kv, N, d, device=K.device, dtype=torch.float32)
        sums.scatter_add_(1, cluster_ids.unsqueeze(-1).expand(-1, -1, d), Kf)
        counts = torch.zeros(H_kv, N, device=K.device, dtype=torch.float32)
        counts.scatter_add_(1, cluster_ids, torch.ones_like(cluster_ids, dtype=torch.float32))
        # Avoid div-by-zero: keep old centroid where count=0
        mask = (counts > 0).unsqueeze(-1)
        new_centroids = torch.where(mask, sums / counts.unsqueeze(-1).clamp(min=1), centroids)
        centroids = new_centroids
    return centroids, cluster_ids


class OutlierDiagRecorder:
    """Per-layer, per-step recorder for outlier-set coverage analysis."""

    def __init__(
        self,
        num_decode_steps: int,
        M: int,
        sink_tokens: int,
        recent_tokens: int,
        skip_layers: set[int],
        prefill_last_ns: list[int],
        cluster_N: int = 256,
        cluster_iters: int = 5,
        cluster_top_clusters: int = 2,
    ):
        self.num_decode_steps = num_decode_steps
        self.M = M
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.skip_layers = set(skip_layers)
        self.prefill_last_ns = sorted(set(prefill_last_ns))
        self.cluster_N = cluster_N
        self.cluster_iters = cluster_iters
        self.cluster_top_clusters = cluster_top_clusters
        # state per layer
        self._step_by_layer: dict[int, int] = defaultdict(int)
        self._sets_by_layer: dict[int, dict[str, torch.Tensor]] = {}
        self._pos_hist_by_layer: dict[int, dict[str, list[int]]] = {}
        self._prefill_last_q: dict[int, torch.Tensor] = {}  # layer_idx -> [1, H_q, N_max, d] (post-RoPE)
        self._clusters: dict[int, dict[str, torch.Tensor]] = {}  # layer_idx -> {centroids, cluster_ids, cand_start, T_c}
        self._dyn_cluster_selections: dict[int, list[torch.Tensor]] = defaultdict(list)  # layer_idx -> [top_ids per step]
        # per-step records: layer, step, kv_len, coverages (per detector mean over kv_heads)
        self.records: list[dict[str, Any]] = []

    def _refine_cluster_members_by_qk(
        self,
        K_cand: torch.Tensor,        # [H_kv, T_c, d] (float)
        Q_g: torch.Tensor,           # [H_kv, d] (float, already scaled)
        cluster_ids: torch.Tensor,   # [H_kv, T_c]
        top_c: torch.Tensor,         # [H_kv, K_top]
        cand_start: int,
    ) -> torch.Tensor:
        """For each kv_head, gather positions in top-K clusters, rank by Q·K, take top-M.
        Returns [H_kv, M] absolute positions. Non-members are masked to -inf.
        If a kv-head has fewer than M total members, the topk falls back to non-members
        (= we end up over-attending), but at N=256 / K_top>=1 this is unlikely."""
        H_kv, T_c, _ = K_cand.shape
        M = self.M
        # Membership mask
        member_mask = (cluster_ids.unsqueeze(-1) == top_c.unsqueeze(1)).any(dim=-1)  # [H_kv, T_c]
        # Score every candidate by Q·K, mask non-members to -inf
        scores = torch.einsum("hd, htd -> ht", Q_g, K_cand)                          # [H_kv, T_c]
        scores = scores.masked_fill(~member_mask, float("-inf"))
        top_pos = scores.topk(M, dim=-1).indices                                     # [H_kv, M]
        return top_pos + cand_start

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        if layer_idx in self.skip_layers:
            return
        if payload.get("phase") == "prefill":
            # Cache last-N prefill Q for this layer (overwrites any prior; last call wins).
            self._prefill_last_q[layer_idx] = payload["prefill_last_q"]
            return
        step = self._step_by_layer[layer_idx]
        if step >= self.num_decode_steps:
            return
        self._step_by_layer[layer_idx] = step + 1

        Q = payload["query_states"]            # [1, H_q, 1, d]
        K = payload["key_states_full"]         # [1, H_kv, T, d]
        num_kv_groups = int(payload["num_kv_groups"])

        bsz, H_kv, T, d = K.shape
        _, H_q, q_len, _ = Q.shape
        assert bsz == 1 and q_len == 1 and H_q == H_kv * num_kv_groups
        scale = 1.0 / math.sqrt(d)

        cand_start = self.sink_tokens
        cand_end = T - self.recent_tokens
        if cand_end - cand_start < self.M:
            return  # not enough pageable tokens

        # --- Static set construction (once per layer, on first valid step) ---
        if layer_idx not in self._sets_by_layer:
            K_cand = K[:, :, cand_start:cand_end, :]
            T_c = K_cand.shape[2]

            knorm = K_cand.float().norm(dim=-1).squeeze(0)  # [H_kv, T_c]
            knorm_idx = knorm.topk(self.M, dim=-1).indices  # [H_kv, M] (rel)

            Q_g = Q.view(1, H_kv, num_kv_groups, d).float() * scale  # [1, H_kv, G, d]
            qk = torch.einsum(
                "bhgd, bhtd -> bhgt", Q_g, K_cand.float()
            ).squeeze(0)  # [H_kv, G, T_c]
            lastq_max = qk.amax(dim=1)
            lastq_mean = qk.mean(dim=1)
            lastq_max_idx = lastq_max.topk(self.M, dim=-1).indices
            lastq_mean_idx = lastq_mean.topk(self.M, dim=-1).indices

            sets_abs: dict[str, torch.Tensor] = {
                "knorm": knorm_idx + cand_start,
                "lastq_max": lastq_max_idx + cand_start,
                "lastq_mean": lastq_mean_idx + cand_start,
            }

            # Prefill-side multi-Q variants: average over last N prefill Qs.
            prefill_q = self._prefill_last_q.get(layer_idx)
            if prefill_q is not None:
                # prefill_q: [1, H_q, N_max, d]
                for n in self.prefill_last_ns:
                    n_eff = min(n, prefill_q.shape[2])
                    Q_pref = prefill_q[:, :, -n_eff:, :].float() * scale  # [1, H_q, n_eff, d]
                    # Reshape qo-head axis into (H_kv, G); average over both group and time.
                    Q_pref_g = Q_pref.view(1, H_kv, num_kv_groups, n_eff, d)
                    qkp = torch.einsum(
                        "bhgnd, bhtd -> bhgnt", Q_pref_g, K_cand.float()
                    )  # [1, H_kv, G, n_eff, T_c]
                    qkp = qkp.squeeze(0)  # [H_kv, G, n_eff, T_c]
                    score_mean = qkp.mean(dim=(1, 2))   # average across group and across last-N
                    score_max = qkp.amax(dim=(1, 2))    # max across group and across last-N
                    sets_abs[f"pref{n}_mean"] = score_mean.topk(self.M, dim=-1).indices + cand_start
                    sets_abs[f"pref{n}_max"] = score_max.topk(self.M, dim=-1).indices + cand_start

            g = torch.Generator(device=K.device).manual_seed(layer_idx + 1)
            rand_idx = torch.stack([
                torch.randperm(T_c, generator=g, device=K.device)[: self.M]
                for _ in range(H_kv)
            ])  # [H_kv, M]
            sets_abs["random"] = rand_idx + cand_start

            # Cluster: k-means once on K_cand, store centroids + cluster_ids for dynamic dispatch.
            centroids, cluster_ids = _kmeans_vectorized(
                K_cand.squeeze(0), N=self.cluster_N, iters=self.cluster_iters,
                seed=layer_idx + 12345,
            )  # centroids: [H_kv, N, d]; cluster_ids: [H_kv, T_c]
            self._clusters[layer_idx] = {
                "centroids": centroids,
                "cluster_ids": cluster_ids,  # rel to cand_start
                "cand_start": cand_start,
                "T_c": T_c,
            }
            # cluster_static: pick top-K clusters using the first-decode Q, refine within by Q·K.
            Q_g_first = Q.view(1, H_kv, num_kv_groups, d).mean(dim=2).squeeze(0).float() * scale  # [H_kv, d]
            cluster_scores = torch.einsum("hd, hnd -> hn", Q_g_first, centroids)                 # [H_kv, N]
            top_c = cluster_scores.topk(self.cluster_top_clusters, dim=-1).indices               # [H_kv, K]
            cluster_static_idx = self._refine_cluster_members_by_qk(
                K_cand.squeeze(0).float(), Q_g_first, cluster_ids, top_c, cand_start,
            )
            sets_abs[f"cluster_static_K{self.cluster_top_clusters}"] = cluster_static_idx

            self._sets_by_layer[layer_idx] = sets_abs

            # Position histograms (per detector, summed over kv_heads).
            self._pos_hist_by_layer[layer_idx] = {
                name: _bucketize(idx, POS_BUCKETS) for name, idx in sets_abs.items()
            }

        # --- Full-KV attention logits per qo-head, expressed as [H_kv, G, T] ---
        Q_g_step = Q.view(1, H_kv, num_kv_groups, d).float() * scale  # [1, H_kv, G, d]
        logits_kvgT = torch.einsum(
            "bhgd, bhtd -> bhgt", Q_g_step, K.float()
        ).squeeze(0)  # [H_kv, G, T]
        attn_qg = F.softmax(logits_kvgT, dim=-1)             # [H_kv, G, T]
        attn_kv = attn_qg.mean(dim=1)                        # [H_kv, T]

        # --- oracle_step: recompute each step (upper bound for token-budget M) ---
        cand_logits = logits_kvgT[:, :, cand_start:cand_end].amax(dim=1)  # [H_kv, T_c]
        oracle_step_idx = cand_logits.topk(self.M, dim=-1).indices + cand_start

        # --- Coverage per detector (mass mean over kv_heads) ---
        sets = dict(self._sets_by_layer[layer_idx])
        sets["oracle_step"] = oracle_step_idx

        # --- cluster_dynamic: recompute top-K clusters each step using current Q ---
        if layer_idx in self._clusters:
            cdat = self._clusters[layer_idx]
            centroids = cdat["centroids"]            # [H_kv, N, d]
            cluster_ids = cdat["cluster_ids"]        # [H_kv, T_c]
            cand_start_l = cdat["cand_start"]
            T_c = cdat["T_c"]
            Q_g_now = Q.view(1, H_kv, num_kv_groups, d).mean(dim=2).squeeze(0).float() * scale  # [H_kv, d]
            cluster_scores = torch.einsum("hd, hnd -> hn", Q_g_now, centroids)
            top_c = cluster_scores.topk(self.cluster_top_clusters, dim=-1).indices
            K_cand_dyn = K[:, :, cand_start_l : cand_start_l + T_c, :].squeeze(0).float()  # [H_kv, T_c, d]
            cluster_dyn_idx = self._refine_cluster_members_by_qk(
                K_cand_dyn, Q_g_now, cluster_ids, top_c, cand_start_l,
            )
            sets[f"cluster_dyn_K{self.cluster_top_clusters}"] = cluster_dyn_idx
            self._dyn_cluster_selections[layer_idx].append(top_c.cpu())

        cov = {}
        for name, idx in sets.items():
            mass = attn_kv.gather(1, idx).sum(dim=-1)  # [H_kv]
            cov[name] = float(mass.mean().item())

        # Floor metrics for context
        sink_mass = float(attn_kv[:, : self.sink_tokens].sum(dim=-1).mean().item()) if self.sink_tokens else 0.0
        recent_mass = float(attn_kv[:, T - self.recent_tokens :].sum(dim=-1).mean().item()) if self.recent_tokens else 0.0
        pageable_mass = float(attn_kv[:, self.sink_tokens : T - self.recent_tokens].sum(dim=-1).mean().item())

        self.records.append({
            "layer_idx": layer_idx,
            "step": step,
            "kv_len": T,
            "cov": cov,
            "sink_mass": sink_mass,
            "recent_mass": recent_mass,
            "pageable_mass": pageable_mass,
        })


def generate_with_recorder(model, tokenizer, sample, recorder):
    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    set_recording_hook(recorder)
    try:
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=recorder.num_decode_steps,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        set_recording_hook(None)
    return recorder.records, int(input_ids.shape[1])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--tasks", type=str, default="cwe")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--num_decode_steps", type=int, default=16)
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--num_sink_pages", type=int, default=1)
    p.add_argument("--num_recent_pages", type=int, default=4)
    p.add_argument("--M", type=int, default=64, help="outlier budget per kv_head")
    p.add_argument("--skip_layers", type=str, default="0,1")
    p.add_argument("--prefill_last_ns", type=str, default="1,4,16,64",
                   help="Comma-separated N values: average over last-N prefill Qs.")
    p.add_argument("--cluster_N", type=int, default=256, help="k-means cluster count")
    p.add_argument("--cluster_iters", type=int, default=5, help="k-means Lloyd iterations")
    p.add_argument("--cluster_top_clusters", type=int, default=2,
                   help="Top-K clusters to gather members from at each step.")
    p.add_argument("--data_root", type=Path, default=REPO_ROOT / "benchmark/data/ruler_data")
    p.add_argument("--output_dir", type=Path, default=REPO_ROOT / "results_outlier_diag")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    family = _model_family(args.base_model)
    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefill_last_ns = [int(x) for x in args.prefill_last_ns.split(",") if x.strip()]
    prefill_last_ns_sorted = sorted(set(prefill_last_ns))
    n_max = max(prefill_last_ns_sorted) if prefill_last_ns_sorted else 1
    _install_dual_recording_forward(model, family, prefill_last_n=n_max)
    _, tokenizer_family = infer_model_family(args.base_model)
    task_configs = load_task_configs()
    skip_layers = {int(x) for x in args.skip_layers.split(",") if x.strip()}

    sink_tokens = args.num_sink_pages * args.page_size
    recent_tokens = args.num_recent_pages * args.page_size

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    detectors = ["knorm", "lastq_max", "lastq_mean"]
    for n in prefill_last_ns_sorted:
        detectors.append(f"pref{n}_mean")
        detectors.append(f"pref{n}_max")
    detectors.append("random")
    detectors.append(f"cluster_static_K{args.cluster_top_clusters}")
    detectors.append(f"cluster_dyn_K{args.cluster_top_clusters}")
    detectors.append("oracle_step")

    sample_fp = (run_dir / "per_sample.jsonl").open("w")
    overall = {d: [] for d in detectors}
    overall_floor = {"sink": [], "recent": [], "pageable": []}
    overall_hist = {d: [0] * len(POS_BUCKETS) for d in detectors if d != "oracle_step"}
    per_task = {}

    start = time.time()
    for task in tasks:
        if task not in task_configs:
            print(f"  ! skipping unknown task {task}")
            continue
        path = args.data_root / tokenizer_family / str(args.seq_len) / task / "validation.jsonl"
        if not path.exists():
            print(f"  ! missing data: {path}")
            continue
        with path.open() as f:
            samples = [json.loads(line) for line in f][: args.num_samples]
        print(f"\nTASK {task}: {len(samples)} samples @ {args.seq_len}")

        task_cov = {d: [] for d in detectors}
        task_floor = {"sink": [], "recent": [], "pageable": []}
        task_hist = {d: [0] * len(POS_BUCKETS) for d in detectors if d != "oracle_step"}

        for s_idx, sample in enumerate(samples, 1):
            sample.setdefault("index", s_idx - 1)
            recorder = OutlierDiagRecorder(
                num_decode_steps=args.num_decode_steps,
                M=args.M,
                sink_tokens=sink_tokens,
                recent_tokens=recent_tokens,
                skip_layers=skip_layers,
                prefill_last_ns=prefill_last_ns_sorted,
                cluster_N=args.cluster_N,
                cluster_iters=args.cluster_iters,
                cluster_top_clusters=args.cluster_top_clusters,
            )
            records, input_len = generate_with_recorder(model, tokenizer, sample, recorder)
            if not records:
                print(f"  [{s_idx}/{len(samples)}] no records (KV too short?)")
                continue

            cov_mean = {d: 0.0 for d in detectors}
            cnt = 0
            sink_m = recent_m = pageable_m = 0.0
            for r in records:
                for d in detectors:
                    cov_mean[d] += r["cov"][d]
                sink_m += r["sink_mass"]
                recent_m += r["recent_mass"]
                pageable_m += r["pageable_mass"]
                cnt += 1
            for d in detectors:
                cov_mean[d] /= cnt
                task_cov[d].append(cov_mean[d])
                overall[d].append(cov_mean[d])
            sink_m /= cnt; recent_m /= cnt; pageable_m /= cnt
            task_floor["sink"].append(sink_m)
            task_floor["recent"].append(recent_m)
            task_floor["pageable"].append(pageable_m)
            overall_floor["sink"].append(sink_m)
            overall_floor["recent"].append(recent_m)
            overall_floor["pageable"].append(pageable_m)

            # Aggregate histograms across layers (sum across layers).
            # Skip detectors that have no static histogram (e.g., cluster_dyn).
            for d in task_hist.keys():
                for ly_hist in recorder._pos_hist_by_layer.values():
                    if d not in ly_hist:
                        continue
                    for i, c in enumerate(ly_hist[d]):
                        task_hist[d][i] += c
                        overall_hist[d][i] += c

            sample_fp.write(json.dumps({
                "task": task,
                "sample_index": int(sample["index"]),
                "input_len": input_len,
                "num_records": len(records),
                "cov_mean": cov_mean,
                "sink_mass": sink_m,
                "recent_mass": recent_m,
                "pageable_mass": pageable_m,
            }, ensure_ascii=False) + "\n")
            sample_fp.flush()

            # Cluster selection diversity: across all (layer, kv_head), fraction of unique cluster_ids
            # selected over decode steps. 1.0 = entirely different cluster each step.
            uniques = []
            for ly_idx, sels in recorder._dyn_cluster_selections.items():
                if not sels:
                    continue
                stack = torch.stack(sels, dim=0)  # [steps, H_kv, K_top]
                steps_n, H_kv_n, K_top_n = stack.shape
                for h in range(H_kv_n):
                    flat = stack[:, h, :].reshape(-1).tolist()
                    uniques.append(len(set(flat)) / max(1, steps_n * K_top_n))
            cluster_diversity = sum(uniques) / max(1, len(uniques))

            print(
                f"  [{s_idx}/{len(samples)}] sink={sink_m:.3f} recent={recent_m:.3f} "
                f"pageable={pageable_m:.3f}  "
                + " ".join(f"{d}={cov_mean[d]:.3f}" for d in detectors)
                + f"  cl_div={cluster_diversity:.2f}"
            )

        if not task_cov["knorm"]:
            continue
        task_summary = {
            "cov": {d: sum(v) / len(v) for d, v in task_cov.items()},
            "floor": {k: sum(v) / len(v) for k, v in task_floor.items()},
            "pos_hist": task_hist,
        }
        per_task[task] = task_summary
        print(f"  TASK {task} mean: " + " ".join(
            f"{d}={task_summary['cov'][d]:.3f}" for d in detectors
        ))
        print(f"    floor (sink/recent/pageable) = "
              f"{task_summary['floor']['sink']:.3f} / "
              f"{task_summary['floor']['recent']:.3f} / "
              f"{task_summary['floor']['pageable']:.3f}")

        # Histogram pretty-print
        edges = POS_BUCKETS
        edge_lbl = [f"[{edges[i]},{edges[i+1] if i+1 < len(edges) else 'inf'})" for i in range(len(edges))]
        print("    pos_hist (kv_head-summed across layers):")
        for d in task_hist.keys():
            total = sum(task_hist[d]) or 1
            frac = [c / total for c in task_hist[d]]
            print(f"      {d:13s} " + " ".join(f"{f:.2f}" for f in frac))
        print("      " + "         ".join(edge_lbl[:6]) + "  ...")
    sample_fp.close()

    overall_summary = {
        "config": vars(args),
        "per_task": per_task,
        "overall": {
            "cov": {d: (sum(v) / len(v) if v else None) for d, v in overall.items()},
            "floor": {k: (sum(v) / len(v) if v else None) for k, v in overall_floor.items()},
            "pos_hist": overall_hist,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(overall_summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(f"\nResults: {run_dir}")
    print(f"Total time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
