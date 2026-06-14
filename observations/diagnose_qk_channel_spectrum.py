"""Diagnose per-layer Q-channel weight & K within-page variance vs RoPE freq.

Hypothesis under test (Qwen3-8B / QK-norm models):
  - Late layers learn queries that concentrate |q[c]|² on high-RoPE-freq
    channels (low pair index i, where RoPE pair i rotates at θ_i =
    rope_theta^(-2i/d)).
  - In those high-freq channels, K oscillates within a page → DCT-lowpass
    (small comp_size) discards exactly that information.
  - Joint metric q²·var_K shifts toward high-freq pairs in late layers,
    which is what makes Quest (per-channel min/max) beat DCT-lowpass on
    Qwen3 in late layers.

Outputs (per layer, averaged over q_head / h_kv / page / decode_step / sample):
  1. q_pair_mass[i]        = Σ_h <|q[2i]|² + |q[2i+1]|²> / Σ_c <|q[c]|²>
  2. k_pair_var[i]         = Σ_kv <var_S(K[..,2i])+var_S(K[..,2i+1])> / Σ_c <var_S(K[..,c])>
  3. joint_pair[i]         = normalized Σ <q²[c]·var_S(K[..,c])> at pair i
  4. q_highfreq_mass_frac  = fraction of (1) in the "high-freq" pair bin
                             (pairs whose θ_i · page_size ≥ 2π — at least
                             one full rotation per page)
  5. joint_highfreq_frac   = same fraction applied to (3)
  6. lowpass_preserved     = Q²·var-weighted fraction of within-page DCT
                             energy that survives lowpass (comp_size). 1.0
                             = nothing lost; small = proxy can't see it.

A standalone script — does NOT touch the DCT-Page forward. Reuses the
dense-recording hook from attention_mass_recall_ruler so post-RoPE
post-QK-norm K is captured exactly as the existing diagnostic sees it.

Usage:
  python observations/diagnose_qk_channel_spectrum.py \\
    --base_model Qwen/Qwen3-8B \\
    --tasks niah_multikey_3 \\
    --seq_len 32768 --num_samples 5 \\
    --page_size 32 --comp_size 4 \\
    --output_dir results_qk_channel_spectrum
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from observations.attention_mass_recall_ruler import (
    _install_recording_forward,
    _model_family,
    cleanup_model,
    load_model,
    set_recording_hook,
)
from observations.attention_mass_recall_ruler import (
    ALL_TASKS,
    load_task_configs,
)
from eval_ruler import infer_model_family


# ---------------------------------------------------------------------------
# Per-step diagnostic compute
# ---------------------------------------------------------------------------
def _rope_pair_frequencies(head_dim: int, rope_theta: float, device) -> torch.Tensor:
    """θ_i = rope_theta^(-2i/d) for i ∈ [0, head_dim/2). Returns [d/2]."""
    half = head_dim // 2
    i = torch.arange(half, device=device, dtype=torch.float64)
    return rope_theta ** (-2.0 * i / head_dim)


def _extract_rope_inv_freq(model) -> Optional[torch.Tensor]:
    """Find the model's actual inv_freq buffer (post-YaRN/scaling).

    Searches common module paths used by HF rotary embeddings; returns the
    first 1-D float buffer of length head_dim/2 found, on CPU as float64.
    Returns None if not found (caller falls back to config.rope_theta).
    """
    for name in ("model.rotary_emb", "rotary_emb"):
        m = model
        for part in name.split("."):
            m = getattr(m, part, None)
            if m is None:
                break
        if m is None:
            continue
        for bname in ("inv_freq", "original_inv_freq"):
            buf = getattr(m, bname, None)
            if isinstance(buf, torch.Tensor) and buf.dim() == 1:
                return buf.detach().to("cpu", torch.float64)
    return None


def _high_freq_pair_mask(
    pair_theta: torch.Tensor, page_size: int,
) -> torch.Tensor:
    """Pairs whose RoPE rotates at least once per page: θ_i · S ≥ 2π."""
    return (pair_theta * page_size) >= (2.0 * math.pi)


def compute_step_diagnostics(
    query_states: torch.Tensor,       # [1, H_q, 1, d]
    paged_k: torch.Tensor,            # [1, H_kv, P, S, d]
    sink_k: torch.Tensor,             # [1, H_kv, sink_len, d]
    recent_k: torch.Tensor,           # [1, H_kv, recent_len, d]
    high_pair_mask: torch.Tensor,     # [d/2] bool, pairs counted as high-freq
    comp_size: int,
    num_kv_groups: int,
    top_k_middle: int,
    needle_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return per-(q_head) scalars for this single decode step.

    All tensors are [H_q] floats. Aggregation across pages happens inside
    (ratio-of-sums) so each q_head gives one number per metric.
    """
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert bsz == 1 and q_len == 1
    assert H_q == H_kv * num_kv_groups
    assert d == 2 * high_pair_mask.shape[0]

    device = query_states.device
    q_abs2 = query_states.squeeze(0).squeeze(1).float() ** 2          # [H_q, d]

    # ---- K within-page variance per channel -------------------------------
    k_f = paged_k.float()                                              # [1, H_kv, P, S, d]
    k_mean = k_f.mean(dim=3, keepdim=True)                             # [1, H_kv, P, 1, d]
    k_ac = k_f - k_mean                                                # AC component only
    k_var = (k_ac ** 2).mean(dim=3).squeeze(0)                         # [H_kv, P, d]

    # Broadcast K[h_kv] to query head's view via GQA grouping.
    h_kv_of_q = torch.arange(H_q, device=device) // num_kv_groups       # [H_q]
    k_var_q = k_var[h_kv_of_q]                                          # [H_q, P, d]

    # ---- Channel → pair grouping (every 2 consecutive channels) -----------
    # Channel c belongs to pair i = c // 2. Sum the two channels per pair.
    def to_pair(t: torch.Tensor) -> torch.Tensor:
        # t shape [..., d] → [..., d/2]; sum each consecutive pair.
        new_shape = list(t.shape[:-1]) + [d // 2, 2]
        return t.view(*new_shape).sum(-1)

    q_pair_mass = to_pair(q_abs2)                                       # [H_q, d/2]
    k_pair_var = to_pair(k_var)                                         # [H_kv, P, d/2]
    k_pair_var_q = to_pair(k_var_q)                                     # [H_q, P, d/2]

    # ---- Scalar summaries per q_head --------------------------------------
    eps = 1e-12

    # 1) Fraction of Q mass in high-freq pairs.
    q_total = q_pair_mass.sum(-1).clamp(min=eps)                        # [H_q]
    q_high = q_pair_mass[:, high_pair_mask].sum(-1)                     # [H_q]
    q_high_frac = q_high / q_total                                      # [H_q]

    # 2) Fraction of K within-page variance in high-freq pairs.
    #    Aggregate page → ratio-of-sums across pages.
    k_total_per_kv = k_pair_var.sum(dim=-1).sum(dim=-1).clamp(min=eps)  # [H_kv]
    k_high_per_kv = k_pair_var[:, :, high_pair_mask].sum(dim=-1).sum(dim=-1)
    k_high_frac_kv = k_high_per_kv / k_total_per_kv                     # [H_kv]
    # Broadcast K-only metric to q_head index for uniform downstream agg.
    k_high_frac = k_high_frac_kv[h_kv_of_q]                             # [H_q]

    # 3) Joint q²·var_K weighted high-freq fraction.
    joint = q_pair_mass.unsqueeze(1) * k_pair_var_q                     # [H_q, P, d/2]
    joint_high = joint[:, :, high_pair_mask].sum(dim=-1).sum(dim=-1)    # [H_q]
    joint_total = joint.sum(dim=-1).sum(dim=-1).clamp(min=eps)
    joint_high_frac = joint_high / joint_total

    # Build orthonormal DCT-II matrix [S, S] in float32; reused by 4c and 4d.
    n = torch.arange(S, device=device, dtype=torch.float32)
    k_idx = torch.arange(S, device=device, dtype=torch.float32)
    dct_mat = math.sqrt(2.0 / S) * \
        torch.cos(math.pi * (n + 0.5).unsqueeze(0) * k_idx.unsqueeze(1) / S)
    dct_mat[0, :] *= 1.0 / math.sqrt(2.0)                                # [S, S]

    scale = 1.0 / math.sqrt(d)
    q_f = query_states.float().squeeze(0).squeeze(1)                     # [H_q, d]
    paged_k_f = paged_k.float()                                          # [1, H_kv, P, S, d]

    # ----- Compute Q·K[s] per (h_q, p, s) without expanding H_kv → H_q -----
    G = num_kv_groups
    q_grp = q_f.view(H_kv, G, d)                                         # [H_kv, G, d]
    qk = torch.einsum("hgd,bhpsd->hgps", q_grp, paged_k_f).squeeze(0) * scale  # [H_kv, G, P, S]
    qk = qk.reshape(H_q, P, S).contiguous()                              # [H_q, P, S]

    # 4a) Attention needle-ness per page (within-page conditional concentration).
    p_soft = torch.softmax(qk, dim=-1)                                   # [H_q, P, S]
    page_needle_max = p_soft.amax(dim=-1)                                # [H_q, P]
    page_needle = (page_needle_max - 1.0 / S).clamp(min=0.0)             # [H_q, P]
    page_needle_mean = page_needle.mean(dim=-1)                          # [H_q]
    del p_soft, page_needle_max

    # 4c) Per-page proxy / quest reconstruction of max_s Q·K[p, s].
    true_peak = qk.amax(dim=-1)                                          # [H_q, P]
    del qk

    # Proxy: lowpass K (comp_size DCT coefs along S), then Q · K_lowpass.
    cs_p = max(1, min(comp_size, S))
    M_low = dct_mat[:cs_p]                                               # [comp_size, S]
    k_low = torch.einsum("cs,bhpsd->bhpcd", M_low, paged_k_f)            # [1, H_kv, P, comp_size, d]
    proxy_per_comp = torch.einsum(
        "hgd,bhpcd->hgpc", q_grp, k_low,
    ).squeeze(0).reshape(H_q, P, cs_p) * scale                           # [H_q, P, comp_size]
    proxy_est = proxy_per_comp.amax(dim=-1)                              # [H_q, P]
    del k_low, proxy_per_comp

    # Quest: per-channel max/min on K (no H_q expansion), then GQA index.
    K_max_kv = paged_k_f.amax(dim=3).squeeze(0)                          # [H_kv, P, d]
    K_min_kv = paged_k_f.amin(dim=3).squeeze(0)                          # [H_kv, P, d]
    K_max = K_max_kv[h_kv_of_q]                                          # [H_q, P, d]
    K_min = K_min_kv[h_kv_of_q]
    q_e = q_f.unsqueeze(1)                                               # [H_q, 1, d]
    quest_est = torch.maximum(q_e * K_max, q_e * K_min).sum(-1) * scale  # [H_q, P]
    del K_max_kv, K_min_kv, K_max, K_min

    safe_true = true_peak.abs().clamp(min=1e-6)
    proxy_ratio = (proxy_est / safe_true).mean(dim=-1)
    quest_ratio = (quest_est / safe_true).mean(dim=-1)
    proxy_abs_err = ((proxy_est - true_peak).abs() / safe_true).mean(dim=-1)
    quest_abs_err = ((quest_est - true_peak).abs() / safe_true).mean(dim=-1)
    del true_peak, safe_true

    # 4e) GLOBAL per-page mass (softmax over full KV: sink + paged + recent).
    #     This is the "true" page importance used by mass_recall.
    # Concat all K segments along token axis.
    sink_len = sink_k.shape[2] if sink_k is not None and sink_k.numel() > 0 else 0
    recent_len = recent_k.shape[2] if recent_k is not None and recent_k.numel() > 0 else 0
    k_parts = []
    if sink_len > 0:
        k_parts.append(sink_k.float())
    k_parts.append(paged_k_f.reshape(1, H_kv, P * S, d))
    if recent_len > 0:
        k_parts.append(recent_k.float())
    k_full = torch.cat(k_parts, dim=2)                                   # [1, H_kv, T, d]
    T = k_full.shape[2]
    # Q·K_full / √d → softmax over T. Use GQA-aware einsum (no H_q expansion).
    full_logits = torch.einsum(
        "hgd,bhtd->hgt", q_grp, k_full,
    ).squeeze(0).reshape(H_q, T) * scale                                 # [H_q, T]
    full_probs = torch.softmax(full_logits, dim=-1)                      # [H_q, T]
    del full_logits, k_full

    paged_start = sink_len
    paged_end_off = sink_len + P * S
    paged_probs = full_probs[:, paged_start:paged_end_off].view(H_q, P, S)
    page_mass = paged_probs.sum(dim=-1)                                  # [H_q, P]
    del full_probs, paged_probs

    # 4f) Top-K mass recall (matches the original mass_recall_* convention).
    #     Pick top_k_middle pages by each scorer, measure captured page-mass
    #     normalized by total page-mass (i.e. "fraction of page-region mass
    #     captured by the selector's top-K choice").
    K_pick = min(top_k_middle, P)
    page_total = page_mass.sum(dim=-1).clamp(min=eps)                    # [H_q]
    proxy_idx = torch.topk(proxy_est, K_pick, dim=-1).indices            # [H_q, K]
    quest_idx = torch.topk(quest_est, K_pick, dim=-1).indices
    oracle_idx = torch.topk(page_mass, K_pick, dim=-1).indices

    paged_recall_proxy = torch.gather(page_mass, -1, proxy_idx).sum(-1) / page_total
    paged_recall_quest = torch.gather(page_mass, -1, quest_idx).sum(-1) / page_total
    paged_recall_oracle = torch.gather(page_mass, -1, oracle_idx).sum(-1) / page_total
    del proxy_idx, quest_idx, oracle_idx

    # 4g) Spearman rank correlation between estimator score and true page_mass.
    #     Spearman = Pearson on rank vectors. Use argsort.argsort → ranks.
    def _ranks(t: torch.Tensor) -> torch.Tensor:
        # t shape [H_q, P] → float ranks via double argsort (no tie handling).
        return t.argsort(dim=-1).argsort(dim=-1).float()

    rk_mass = _ranks(page_mass)
    rk_proxy = _ranks(proxy_est)
    rk_quest = _ranks(quest_est)

    def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        am = a - a.mean(dim=-1, keepdim=True)
        bm = b - b.mean(dim=-1, keepdim=True)
        num = (am * bm).sum(dim=-1)
        den = (am.pow(2).sum(-1) * bm.pow(2).sum(-1)).clamp(min=eps).sqrt()
        return num / den

    spearman_proxy = _pearson(rk_proxy, rk_mass)                         # [H_q]
    spearman_quest = _pearson(rk_quest, rk_mass)
    del rk_mass, rk_proxy, rk_quest

    # 4h) Needle-mass-share: of the total middle-page mass, what fraction
    #     lands on pages that are *internally* needle-like? If late layers
    #     concentrate mass on needle pages, this rises with layer.
    needle_mask = page_needle >= needle_threshold                        # [H_q, P]
    mass_on_needle = (page_mass * needle_mask.float()).sum(dim=-1)       # [H_q]
    needle_mass_share = mass_on_needle / page_total                      # [H_q]

    # 4i) Gap-on-needle: difference between proxy and quest recall conditioned
    #     on the needle pages being the ground truth. We compute the same
    #     top-K recall but renormalize mass by only the mass on needle pages.
    mass_needle_only = page_mass * needle_mask.float()                   # [H_q, P]
    needle_total = mass_needle_only.sum(dim=-1).clamp(min=eps)           # [H_q]
    # How much of the *needle-page mass* does each selector capture?
    proxy_topk = torch.topk(proxy_est, K_pick, dim=-1).indices
    quest_topk = torch.topk(quest_est, K_pick, dim=-1).indices
    recall_needle_proxy = torch.gather(mass_needle_only, -1, proxy_topk).sum(-1) / needle_total
    recall_needle_quest = torch.gather(mass_needle_only, -1, quest_topk).sum(-1) / needle_total
    del proxy_topk, quest_topk, mass_needle_only, page_mass, page_needle
    del proxy_est, quest_est

    # 4d) DCT-lowpass preserved fraction of (q²-weighted) within-page variance.
    # Apply DCT along S axis directly on k_ac (no H_q expansion).
    k_ac_perm = k_ac.permute(0, 1, 2, 4, 3)                              # [1, H_kv, P, d, S]
    k_dct = torch.matmul(k_ac_perm, dct_mat.transpose(-1, -2))           # [1, H_kv, P, d, S]
    k_dct_energy = (k_dct ** 2).squeeze(0)                               # [H_kv, P, d, S]
    del k_ac_perm, k_dct

    preserved = k_dct_energy[..., 1:cs_p].sum(-1)                        # [H_kv, P, d]
    total_ac = k_dct_energy[..., 1:].sum(-1).clamp(min=eps)              # [H_kv, P, d]
    preserved_frac = preserved / total_ac
    del k_dct_energy, preserved, total_ac

    weight = q_abs2.unsqueeze(1) * k_var_q                               # [H_q, P, d]
    preserved_frac_q = preserved_frac[h_kv_of_q]                         # [H_q, P, d]
    lowpass_pres_num = (preserved_frac_q * weight).sum(dim=-1).sum(dim=-1)
    lowpass_pres_den = weight.sum(dim=-1).sum(dim=-1).clamp(min=eps)
    lowpass_preserved = lowpass_pres_num / lowpass_pres_den              # [H_q]
    del preserved_frac, preserved_frac_q, weight

    return {
        "q_high_frac": q_high_frac,
        "k_high_frac": k_high_frac,
        "joint_high_frac": joint_high_frac,
        "lowpass_preserved": lowpass_preserved,
        "page_needle": page_needle_mean,
        "proxy_ratio": proxy_ratio,
        "quest_ratio": quest_ratio,
        "proxy_abs_err": proxy_abs_err,
        "quest_abs_err": quest_abs_err,
        # New: global mass-recall, rank-correlation, needle-mass-share, and
        # the proxy/quest recall restricted to mass landing on needle pages.
        "paged_recall_proxy": paged_recall_proxy,
        "paged_recall_quest": paged_recall_quest,
        "paged_recall_oracle": paged_recall_oracle,
        "spearman_proxy": spearman_proxy,
        "spearman_quest": spearman_quest,
        "needle_mass_share": needle_mass_share,
        "recall_needle_proxy": recall_needle_proxy,
        "recall_needle_quest": recall_needle_quest,
        # Per-pair profiles (aggregated across pages, but per q_head).
        # Returned as [H_q, d/2] for q_pair_mass and joint_pair, [H_kv, d/2]
        # for k_pair_var (callers handle broadcast).
        "q_pair_mass": q_pair_mass,                                      # [H_q, d/2]
        "k_pair_var": k_pair_var.sum(dim=1) / max(P, 1),                 # [H_kv, d/2]
        "joint_pair": joint.sum(dim=1) / max(P, 1),                      # [H_q, d/2]
    }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class ChannelSpectrumRecorder:
    """Per-layer accumulator. Holds running sums for ratio-of-means aggregation.

    Skips the first ``num_skip_layers`` layers (their stats can be uninformative
    on QK-norm models — early layers are noisier).
    """

    def __init__(
        self,
        page_size: int,
        num_sink_pages: int,
        num_recent_pages: int,
        comp_size: int,
        num_decode_steps: int,
        high_pair_mask: torch.Tensor,
        top_k_middle: int,
        needle_threshold: float,
        num_skip_layers: int = 0,
    ) -> None:
        self.page_size = page_size
        self.num_sink_pages = num_sink_pages
        self.num_recent_pages = num_recent_pages
        self.comp_size = comp_size
        self.num_decode_steps = num_decode_steps
        self.high_pair_mask = high_pair_mask
        self.top_k_middle = top_k_middle
        self.needle_threshold = needle_threshold
        self.num_skip_layers = num_skip_layers

        self._step_counter: dict[int, int] = {}
        # Per-layer running sums (CPU floats).
        self.per_layer_scalar_sum: dict[int, dict[str, float]] = {}
        self.per_layer_scalar_count: dict[int, int] = {}
        self.per_layer_q_pair_sum: dict[int, torch.Tensor] = {}   # [d/2]
        self.per_layer_k_pair_sum: dict[int, torch.Tensor] = {}   # [d/2]
        self.per_layer_joint_pair_sum: dict[int, torch.Tensor] = {}  # [d/2]
        self.per_layer_pair_count: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        if layer_idx < self.num_skip_layers:
            return

        # Decode-step counter per layer (capture only first num_decode_steps).
        step = self._step_counter.get(layer_idx, 0)
        if step >= self.num_decode_steps:
            return
        self._step_counter[layer_idx] = step + 1

        query_states = payload["query_states"]           # [1, H_q, 1, d]
        key_full = payload["key_states_full"]            # [1, H_kv, kv_len, d]
        num_kv_groups = int(payload["num_kv_groups"])

        bsz, H_kv, kv_len, d = key_full.shape
        if d != 2 * self.high_pair_mask.shape[0]:
            return  # head_dim mismatch; skip silently

        sink_len = self.num_sink_pages * self.page_size
        recent_min = self.num_recent_pages * self.page_size
        if kv_len < sink_len + self.page_size + recent_min:
            return

        num_pages = (kv_len - sink_len - recent_min) // self.page_size
        if num_pages < 1:
            return

        paged_end = sink_len + num_pages * self.page_size
        paged_k = key_full[:, :, sink_len:paged_end, :].view(
            bsz, H_kv, num_pages, self.page_size, d,
        )
        sink_k = key_full[:, :, :sink_len, :]
        recent_k = key_full[:, :, paged_end:, :]

        with torch.no_grad():
            mask_gpu = self.high_pair_mask.to(query_states.device)
            diag = compute_step_diagnostics(
                query_states, paged_k, sink_k, recent_k, mask_gpu,
                comp_size=self.comp_size,
                num_kv_groups=num_kv_groups,
                top_k_middle=self.top_k_middle,
                needle_threshold=self.needle_threshold,
            )

        sums = self.per_layer_scalar_sum.setdefault(
            layer_idx,
            {k: 0.0 for k in (
                "q_high_frac", "k_high_frac", "joint_high_frac",
                "lowpass_preserved", "page_needle",
                "proxy_ratio", "quest_ratio",
                "proxy_abs_err", "quest_abs_err",
                "paged_recall_proxy", "paged_recall_quest", "paged_recall_oracle",
                "spearman_proxy", "spearman_quest",
                "needle_mass_share",
                "recall_needle_proxy", "recall_needle_quest",
            )},
        )
        n_q = int(diag["q_high_frac"].shape[0])
        for key in (
            "q_high_frac", "k_high_frac", "joint_high_frac",
            "lowpass_preserved", "page_needle",
            "proxy_ratio", "quest_ratio", "proxy_abs_err", "quest_abs_err",
            "paged_recall_proxy", "paged_recall_quest", "paged_recall_oracle",
            "spearman_proxy", "spearman_quest",
            "needle_mass_share",
            "recall_needle_proxy", "recall_needle_quest",
        ):
            sums[key] += float(diag[key].sum().cpu())
        self.per_layer_scalar_count[layer_idx] = (
            self.per_layer_scalar_count.get(layer_idx, 0) + n_q
        )

        # Per-pair profiles (averaged across H_q for q-side, H_kv for k-side).
        q_pair_mean = diag["q_pair_mass"].mean(dim=0).cpu()              # [d/2]
        k_pair_mean = diag["k_pair_var"].mean(dim=0).cpu()               # [d/2]
        joint_pair_mean = diag["joint_pair"].mean(dim=0).cpu()           # [d/2]
        prev_q = self.per_layer_q_pair_sum.get(layer_idx)
        if prev_q is None:
            self.per_layer_q_pair_sum[layer_idx] = q_pair_mean
            self.per_layer_k_pair_sum[layer_idx] = k_pair_mean
            self.per_layer_joint_pair_sum[layer_idx] = joint_pair_mean
        else:
            self.per_layer_q_pair_sum[layer_idx] = prev_q + q_pair_mean
            self.per_layer_k_pair_sum[layer_idx] = (
                self.per_layer_k_pair_sum[layer_idx] + k_pair_mean
            )
            self.per_layer_joint_pair_sum[layer_idx] = (
                self.per_layer_joint_pair_sum[layer_idx] + joint_pair_mean
            )
        self.per_layer_pair_count[layer_idx] = (
            self.per_layer_pair_count.get(layer_idx, 0) + 1
        )

    def reset_step_counters(self) -> None:
        self._step_counter = {}

    def summarize(self) -> dict[str, Any]:
        """Per-layer means + overall means."""
        scalar_keys = (
            "q_high_frac", "k_high_frac", "joint_high_frac",
            "lowpass_preserved", "page_needle",
            "proxy_ratio", "quest_ratio", "proxy_abs_err", "quest_abs_err",
            "paged_recall_proxy", "paged_recall_quest", "paged_recall_oracle",
            "spearman_proxy", "spearman_quest",
            "needle_mass_share",
            "recall_needle_proxy", "recall_needle_quest",
        )
        per_layer: dict[str, dict[str, Any]] = {}
        all_scalars: dict[str, list[float]] = {k: [] for k in scalar_keys}
        for layer_idx in sorted(self.per_layer_scalar_sum.keys()):
            n = self.per_layer_scalar_count[layer_idx]
            sums = self.per_layer_scalar_sum[layer_idx]
            entry: dict[str, Any] = {"n_qhead_samples": n}
            for k in scalar_keys:
                v = sums[k] / max(n, 1)
                entry[k] = v
                all_scalars[k].append(v)
            np_pairs = self.per_layer_pair_count[layer_idx]
            entry["q_pair_mean"] = (
                self.per_layer_q_pair_sum[layer_idx] / max(np_pairs, 1)
            ).tolist()
            entry["k_pair_mean"] = (
                self.per_layer_k_pair_sum[layer_idx] / max(np_pairs, 1)
            ).tolist()
            entry["joint_pair_mean"] = (
                self.per_layer_joint_pair_sum[layer_idx] / max(np_pairs, 1)
            ).tolist()
            per_layer[str(layer_idx)] = entry

        overall = {
            f"{k}_mean": (sum(v) / max(len(v), 1)) for k, v in all_scalars.items()
        }
        return {"per_layer": per_layer, "overall": overall}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Per-layer Q-channel-weight vs RoPE-frequency diagnostic. "
            "Verifies the claim that late layers concentrate |q|² on fast-"
            "rotating RoPE channels, where DCT-lowpass loses information."
        )
    )
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--tasks", type=str, nargs="+", default=["niah_multikey_3"])
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--data_root", type=Path,
                   default=Path("benchmark/data/ruler_data"))

    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--num_sink_pages", type=int, default=1)
    p.add_argument("--num_recent_pages", type=int, default=4)
    p.add_argument(
        "--comp_size", type=int, default=4,
        help="DCT-lowpass cutoff for preserved-fraction metric (default 4 = "
             "page_size * 0.125 at page_size=32).",
    )
    p.add_argument("--num_decode_steps", type=int, default=10)
    p.add_argument(
        "--skip_layers", type=int, default=0,
        help="Skip the first N layers (their stats are often noisier).",
    )
    p.add_argument(
        "--top_k", type=int, default=64,
        help="Total page budget (sink + recent + middle); middle_K is "
             "derived as top_k - num_sink_pages - num_recent_pages. Matches "
             "the attention_mass_recall_ruler convention.",
    )
    p.add_argument(
        "--needle_threshold", type=float, default=0.5,
        help="Page-needle threshold (max within-page softmax - 1/S). Pages "
             "with needle ≥ this count as 'needle pages' for the mass-share "
             "and conditional recall metrics.",
    )

    p.add_argument("--output_dir", type=Path,
                   default=Path("results_qk_channel_spectrum"))
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def _generate_with_hook(
    model, tokenizer, sample: dict[str, Any],
    recorder: ChannelSpectrumRecorder,
    num_decode_steps: int,
) -> int:
    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder.reset_step_counters()
    set_recording_hook(recorder)
    try:
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_decode_steps,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        set_recording_hook(None)
    return int(input_ids.shape[1])


def _print_summary(summary: dict[str, Any], pair_theta: torch.Tensor, page_size: int) -> None:
    high_mask = _high_freq_pair_mask(pair_theta, page_size).tolist()
    n_high_pairs = sum(high_mask)
    print()
    print("=" * 78)
    print("OVERALL")
    print("=" * 78)
    for k, v in summary["overall"].items():
        print(f"  {k:30s} = {v:.4f}")
    print(f"\n  high-freq pair count: {n_high_pairs} / {len(high_mask)} "
          f"(θ_i · S ≥ 2π, S={page_size})")
    print()
    print("=" * 110)
    print("PER LAYER")
    print("=" * 110)
    print(
        f"{'L':>3}  {'needle':>6}  {'nMass':>6}  "
        f"{'rec_px':>6}  {'rec_qt':>6}  {'rec_or':>6}  "
        f"{'sp_px':>6}  {'sp_qt':>6}  "
        f"{'nrec_px':>7}  {'nrec_qt':>7}  "
        f"{'q_hi':>5}  {'k_hi':>5}  {'lpres':>5}"
    )
    for k in sorted(summary["per_layer"].keys(), key=int):
        e = summary["per_layer"][k]
        print(
            f"{k:>3}  "
            f"{e['page_needle']:>6.3f}  "
            f"{e['needle_mass_share']:>6.3f}  "
            f"{e['paged_recall_proxy']:>6.3f}  "
            f"{e['paged_recall_quest']:>6.3f}  "
            f"{e['paged_recall_oracle']:>6.3f}  "
            f"{e['spearman_proxy']:>6.3f}  "
            f"{e['spearman_quest']:>6.3f}  "
            f"{e['recall_needle_proxy']:>7.3f}  "
            f"{e['recall_needle_quest']:>7.3f}  "
            f"{e['q_high_frac']:>5.3f}  "
            f"{e['k_high_frac']:>5.3f}  "
            f"{e['lowpass_preserved']:>5.3f}"
        )
    print()
    print("legend: needle=mean within-page concentration; nMass=fraction of "
          "page-region mass on needle pages (needle≥thresh);")
    print("  rec_px/qt/or=top-K page-mass recall via proxy/quest/oracle "
          "ranking; sp_px/qt=Spearman(score, true page mass);")
    print("  nrec_px/qt=top-K recall of needle-page mass only (denominator "
          "= mass on needle pages).")


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    start = time.time()

    run_name = args.run_name or (
        f"qkspec_{Path(args.base_model).name}_ps{args.page_size}"
        f"_cs{args.comp_size}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    family = _model_family(args.base_model)
    print(f"Installing dense recording forward (family={family})...")
    _install_recording_forward(model, family)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    # Prefer the live inv_freq buffer (post-YaRN / post-scaling) over
    # config.rope_theta — when YaRN is applied via load_model, the config
    # attribute can be None or stale while inv_freq carries the actual values.
    inv_freq = _extract_rope_inv_freq(model)
    if inv_freq is not None and inv_freq.shape[0] == head_dim // 2:
        pair_theta = inv_freq
        rope_theta_for_log = "(from inv_freq buffer)"
    else:
        # Fallback: derive from config (or rope_parameters' inner rope_theta).
        rp = getattr(cfg, "rope_parameters", None) or {}
        cfg_theta = getattr(cfg, "rope_theta", None)
        rope_theta = float(cfg_theta if cfg_theta is not None else rp.get("rope_theta", 10000.0))
        pair_theta = _rope_pair_frequencies(head_dim, rope_theta, device="cpu")
        rope_theta_for_log = f"{rope_theta:g} (from config)"
    high_pair_mask = _high_freq_pair_mask(pair_theta, args.page_size)

    print(f"  head_dim={head_dim}, rope_theta={rope_theta_for_log}")
    print(f"  high-freq pairs (θ_i·S ≥ 2π, S={args.page_size}): "
          f"{int(high_pair_mask.sum())}/{head_dim // 2}")

    middle_k = max(1, args.top_k - args.num_sink_pages - args.num_recent_pages)
    print(f"  top_k_total={args.top_k}, middle_K={middle_k}, "
          f"needle_threshold={args.needle_threshold}")

    recorder = ChannelSpectrumRecorder(
        page_size=args.page_size,
        num_sink_pages=args.num_sink_pages,
        num_recent_pages=args.num_recent_pages,
        comp_size=args.comp_size,
        num_decode_steps=args.num_decode_steps,
        high_pair_mask=high_pair_mask,
        top_k_middle=middle_k,
        needle_threshold=args.needle_threshold,
        num_skip_layers=args.skip_layers,
    )

    _, tokenizer_family = infer_model_family(args.base_model)
    task_configs = load_task_configs()

    try:
        for task in args.tasks:
            if task not in task_configs:
                print(f"  WARNING: task {task!r} not in RULER configs, skipping")
                continue
            data_path = (
                args.data_root / tokenizer_family / str(args.seq_len)
                / task / "validation.jsonl"
            )
            if not data_path.exists():
                print(f"  WARNING: data not found at {data_path}, skipping")
                continue
            with data_path.open("r", encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]
            if args.num_samples > 0:
                samples = samples[: args.num_samples]

            for sample in tqdm(samples, desc=f"  {task}"):
                _generate_with_hook(
                    model, tokenizer, sample, recorder,
                    num_decode_steps=args.num_decode_steps,
                )
    finally:
        summary = recorder.summarize()
        summary["config"] = vars(args)
        summary["config"]["head_dim"] = head_dim
        summary["config"]["pair_theta"] = pair_theta.tolist()
        summary["config"]["high_pair_mask"] = high_pair_mask.tolist()
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved: {run_dir / 'summary.json'}")
        _print_summary(summary, pair_theta, args.page_size)
        cleanup_model(model)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
