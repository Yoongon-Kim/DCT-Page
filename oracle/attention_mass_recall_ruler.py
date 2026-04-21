#!/usr/bin/env python3
"""
Attention-mass recall on RULER.

For each decode step, applies a single softmax over the **entire KV sequence**
(sink + paged + recent) to compute a per-token mass distribution. Sink and
recent tokens are always kept, so their mass is a fixed floor added to every
mass-recall metric. The per-page mass of each paged page is the sum of
softmax weights on that page's tokens.

Sum invariant per head:   sink_mass + Σ_p page_mass[p] + recent_mass = 1

Reports per-query-head metrics grouped into three families.

(A) FULL-KV MASS metrics (all include the sink + recent floor):

  mass_recall_sink_recent = sink + recent                                  (floor only — shared by every selector)
  mass_recall_proxy       = sink + recent + Σ m[p] over DCT proxy's top-K
  mass_recall_quest       = sink + recent + Σ m[p] over Quest's top-K
  mass_recall_shadowkv    = sink + recent + Σ m[p] over ShadowKV's top-K
  mass_recall_oracle_max  = sink + recent + Σ m[p] over oracle_max's top-K
  mass_recall_mass_topk   = sink + recent + Σ m[p] over top-K by page mass  (ceiling)
  set_recall              = |DCT ∩ oracle_max| / K                         (page-set baseline)

(B) PAGED-ONLY MASS metrics (no sink/recent floor; denominator = total
    paged attention mass Σ_p m[p] = 1 − (sink_mass + recent_mass)):

  paged_mass_recall_proxy       = Σ_{p∈DCT topK} m[p]       / Σ_p m[p]
  paged_mass_recall_quest       = Σ_{p∈Quest topK} m[p]     / Σ_p m[p]
  paged_mass_recall_shadowkv    = Σ_{p∈ShadowKV topK} m[p]  / Σ_p m[p]
  paged_mass_recall_oracle_max  = Σ_{p∈oracle_max topK} m[p]/ Σ_p m[p]
  paged_mass_recall_mass_topk   = Σ_{p∈mass topK} m[p]      / Σ_p m[p]   (ceiling)
  paged_mass_ratio_proxy        = paged_mass_recall_proxy / paged_mass_recall_mass_topk
  paged_mass_ratio_quest        = paged_mass_recall_quest / paged_mass_recall_mass_topk

Paged-only strips the always-kept sink + recent floor from both numerator
and denominator, so values are the fraction of **paged** attention mass
(not total) captured by the paged selection. This rescales each head's
paged mass so the ceiling = 1 exactly when K ≥ P, and separates
selection quality from the always-kept floor. By construction:

  mass_recall_X = paged_mass_recall_X · (1 − mass_recall_sink_recent)
                  + mass_recall_sink_recent
  paged_mass_recall_mass_topk ≤ 1        (= 1 when K ≥ P)

FIDELITY metrics (per-head cosine similarity between full and drop-mode
attention outputs — V-aware, the actual downstream signal):

  output_fidelity_proxy      = cos(full_output, DCT drop output)
  output_fidelity_quest      = cos(full_output, Quest drop output)
  output_fidelity_shadowkv   = cos(full_output, ShadowKV drop output)
  output_fidelity_oracle_max = cos(full_output, oracle_max drop output)

Mass recall can be high while output fidelity diverges — two selections can
carry the same softmax mass but use different V vectors, leading to different
downstream outputs. Fidelity is directly diagnostic of task-level quality.

Quest scoring (Tang et al., MLSys 2024) uses per-channel K min/max within each
page to upper-bound max(Q·K):
    score[p] = (1/√d) · Σ_d max(q[d]·K_max[p, d], q[d]·K_min[p, d])

ShadowKV scoring (Sun et al., NeurIPS 2024) represents each page by a single
landmark vector = per-channel mean of its keys:
    landmark[p] = (1/S) · Σ_s K[p, s, :]
    raw[h, p]   = (q[h] · landmark[p]) / √d
    attn[h, p]  = softmax_p(raw[h, :])     # per query head
    score[kv, p]= reduce_{h in group}(attn[h, p])
ShadowKV's outlier-page bypass and SVD V-reconstruction are omitted (they are
orthogonal to page ranking and would change the effective K budget).

All selectors share the same sink/recent configuration as DCT; only the
page-ranking rule differs.

Each mass metric directly measures the fraction of the full-attention softmax
mass preserved by the corresponding selection (sink + recent are always kept,
so their mass contributes to every selector). Sources of loss:

  1.0 − ceiling        : unavoidable mass loss from budget-K sparsity
  ceiling − oracle_max : max(Q·K)'s own blind spot vs true mass
  oracle_max − proxy   : DCT proxy's approximation gap vs max(Q·K)
  proxy vs quest vs shadowkv : different proxy families against each other

Reuses CLI, model setup, and task-loading utilities from
``compare_proxy_oracle_slices.py`` — no modifications to core code (the debug
hook payload already carries query_states, paged_k, selected_indices,
num_kv_groups, oracle_page_scores).

Usage:
    python oracle/attention_mass_recall_ruler.py \\
        --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --tasks niah_single_1 --num_samples 2 --seq_len 32768 \\
        --page_size 16 --top_k 128 --num_decode_steps 2 \\
        --output_dir results_attention_mass_recall --run_name smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from eval_ruler import infer_model_family
from oracle.compare_proxy_oracle_slices import (
    ALL_TASKS,
    _indices_to_mask,
    apply_monkey_patch,
    cleanup_model,
    load_model,
    load_task_configs,
)


MASS_METRIC_KEYS = [
    # Mass of (sink + recent) alone / full KV — always-kept floor shared by every selector.
    "mass_recall_sink_recent",
    # Mass of (sink + selected pages + recent) / full KV — includes always-kept floor.
    "mass_recall_proxy",
    "mass_recall_quest",
    "mass_recall_shadowkv",
    "mass_recall_oracle_max",
    "mass_recall_mass_topk",
    "set_recall",
    # Mass of (selected pages) / (total paged attention mass Σ_p m[p]).
    # No sink/recent floor in either numerator or denominator; rescales
    # per-head paged mass so the ceiling = 1 exactly when K ≥ P.
    # mass_recall_X = paged_mass_recall_X · (1 − sink_recent) + sink_recent.
    "paged_mass_recall_proxy",
    "paged_mass_recall_quest",
    "paged_mass_recall_shadowkv",
    "paged_mass_recall_oracle_max",
    "paged_mass_recall_mass_topk",
    "paged_mass_ratio_proxy",
    "paged_mass_ratio_quest",
]

FIDELITY_METRIC_KEYS = [
    "output_fidelity_proxy",
    "output_fidelity_quest",
    "output_fidelity_shadowkv",
    "output_fidelity_oracle_max",
]

METRIC_KEYS = MASS_METRIC_KEYS + FIDELITY_METRIC_KEYS


# ---------------------------------------------------------------------------
# Mass computation and metrics
# ---------------------------------------------------------------------------
def compute_per_page_mass(
    query_states: torch.Tensor,
    sink_k: torch.Tensor,
    paged_k: torch.Tensor,
    recent_k: torch.Tensor,
    num_kv_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query-head softmax mass over the full KV sequence.

    Softmax denominator spans sink + paged + recent tokens jointly, so each
    head's mass values describe the true fraction of the attention output
    that each region contributes.

    Args:
        query_states: [bsz=1, H_q, 1, d] — post-RoPE / post-QK-norm.
        sink_k:       [bsz=1, H_kv, sink_len, d] — post-RoPE.
        paged_k:      [bsz=1, H_kv, P, S, d] — post-RoPE, baked in from cache.
        recent_k:     [bsz=1, H_kv, recent_len, d] — post-RoPE.
        num_kv_groups: H_q // H_kv.

    Returns:
        page_mass:  [H_q, P]  — softmax weights summed within each paged page.
        extra_mass: [H_q]     — softmax weights on sink + recent tokens
                                 (always-kept floor). Invariant:
                                 page_mass.sum(-1) + extra_mass = 1.
    """
    bsz, H_q, q_len, d = query_states.shape
    assert bsz == 1 and q_len == 1, f"decode-step only, got shape {query_states.shape}"
    _, H_kv, P, S, _ = paged_k.shape
    assert H_q == H_kv * num_kv_groups
    sink_len = sink_k.shape[2] if sink_k is not None else 0
    recent_len = recent_k.shape[2] if recent_k is not None else 0
    scale = 1.0 / math.sqrt(d)

    # Concat [sink | paged-flattened | recent] along the token axis.
    parts = []
    if sink_len > 0:
        parts.append(sink_k)
    parts.append(paged_k.reshape(bsz, H_kv, P * S, d))
    if recent_len > 0:
        parts.append(recent_k)
    k_full = torch.cat(parts, dim=2)                             # [1, H_kv, T, d]
    k_expanded = k_full.repeat_interleave(num_kv_groups, dim=1)  # [1, H_q, T, d]

    # One softmax over the whole KV sequence per head.
    scores = torch.matmul(query_states, k_expanded.transpose(-1, -2)) * scale
    weights = torch.softmax(scores.float(), dim=-1).squeeze(2)   # [1, H_q, T]

    # Slice the three regions in the order they were concatenated.
    sink_mass = (
        weights[..., :sink_len].sum(-1) if sink_len > 0
        else weights.new_zeros(bsz, H_q)
    )
    paged_weights = weights[..., sink_len:sink_len + P * S]      # [1, H_q, P*S]
    page_mass = paged_weights.view(bsz, H_q, P, S).sum(-1)       # [1, H_q, P]
    recent_mass = (
        weights[..., sink_len + P * S:].sum(-1) if recent_len > 0
        else weights.new_zeros(bsz, H_q)
    )

    extra_mass = (sink_mass + recent_mass).squeeze(0)            # [H_q]
    return page_mass.squeeze(0), extra_mass                      # [H_q, P], [H_q]


def compute_quest_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    num_kv_groups: int,
    group_agg_method: str,
) -> torch.Tensor:
    """Canonical Quest per-page upper-bound scores (Tang et al., MLSys 2024).

    For each page p, compute channel-wise K_max and K_min across its tokens,
    then score page p for each query head h as

        score[h, p] = (1/√d) · Σ_d max(q[h, d]·K_max[p, d], q[h, d]·K_min[p, d])

    This is an exact upper bound on max_{s ∈ page p} q[h] · K[p, s]. Score is
    computed per query head, then reduced across GQA groups via the same
    ``group_agg_method`` DCT uses, so the selection happens at kv-head level
    (matching ``selected_indices`` and DCT's proxy scoring).

    Args:
        query_states: [bsz=1, H_q, 1, d] — post-RoPE / post-QK-norm.
        paged_k:      [bsz=1, H_kv, P, S, d] — post-RoPE, baked in from cache.
        num_kv_groups: H_q // H_kv.
        group_agg_method: "mean" | "max" | "topp" (topp falls back to mean).

    Returns:
        scores: [H_kv, P] — one Quest score per (kv-head, page).
    """
    bsz, H_q, q_len, d = query_states.shape
    assert bsz == 1 and q_len == 1, f"decode-step only, got shape {query_states.shape}"
    _, H_kv, P, S, _ = paged_k.shape
    assert H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)

    # Per-channel min/max across the page_size axis (shared across GQA group).
    K_max = paged_k.max(dim=3).values                             # [1, H_kv, P, d]
    K_min = paged_k.min(dim=3).values                             # [1, H_kv, P, d]

    # Expand to query-head space so the sign-aware max is per query head.
    K_max_q = K_max.repeat_interleave(num_kv_groups, dim=1).float()  # [1, H_q, P, d]
    K_min_q = K_min.repeat_interleave(num_kv_groups, dim=1).float()

    q = query_states.float()                                      # [1, H_q, 1, d]
    prod_max = q * K_max_q                                        # [1, H_q, P, d]
    prod_min = q * K_min_q
    channel_best = torch.maximum(prod_max, prod_min)              # sign-aware
    score_q = channel_best.sum(-1) * scale                        # [1, H_q, P]

    # Reduce query-group dim to kv-head level, mirroring DCT's group_agg_method.
    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.max(dim=2).values
    else:
        # "mean" (default) and "topp" — topp is density-based and not
        # meaningful for Quest's upper-bound scores, so fall back to mean.
        scores = score_g.mean(dim=2)
    return scores.squeeze(0)                                      # [H_kv, P]


def compute_shadowkv_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    num_kv_groups: int,
    group_agg_method: str,
) -> torch.Tensor:
    """ShadowKV landmark-based page scoring (Sun et al., NeurIPS 2024).

    Each page is represented by a single "landmark" vector equal to the mean
    of its keys across the page-size axis. Pages are ranked by

        raw[h, p]  = (q[h] · landmark[p]) / √d
        attn[h, p] = softmax_p(raw[h, :])                     # per query head
        score[kv, p] = reduce_{h in group}(attn[h, p])        # group_agg_method

    The softmax is ShadowKV's native step (normalizes each query head's page
    distribution before the group reduction) — important for GQA since it
    puts heads on a common scale before we reduce across the group. For a
    single head the softmax is a monotonic transform and doesn't change the
    top-K ranking.

    We omit ShadowKV's outlier-page bypass and SVD-based V reconstruction;
    those are orthogonal to page ranking and would change the effective K
    budget, breaking apples-to-apples comparison.

    Args:
        query_states: [bsz=1, H_q, 1, d] — post-RoPE / post-QK-norm.
        paged_k:      [bsz=1, H_kv, P, S, d] — post-RoPE, baked in from cache.
        num_kv_groups: H_q // H_kv.
        group_agg_method: "mean" | "max" | "topp" (topp falls back to mean).

    Returns:
        scores: [H_kv, P] — one ShadowKV score per (kv-head, page).
    """
    bsz, H_q, q_len, d = query_states.shape
    assert bsz == 1 and q_len == 1, f"decode-step only, got shape {query_states.shape}"
    _, H_kv, P, S, _ = paged_k.shape
    assert H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)

    # Landmarks: per-page mean of keys (shared across GQA group).
    landmark = paged_k.mean(dim=3)                                # [1, H_kv, P, d]
    landmark_q = landmark.repeat_interleave(num_kv_groups, dim=1).float()  # [1, H_q, P, d]

    q = query_states.squeeze(2).float()                           # [1, H_q, d]
    raw = torch.einsum("bhd,bhpd->bhp", q, landmark_q) * scale    # [1, H_q, P]

    # ShadowKV softmaxes over pages per query head before any group reduction.
    attn = torch.softmax(raw, dim=-1)                             # [1, H_q, P]

    # Reduce query-group dim to kv-head level (apples-to-apples with DCT/Quest).
    attn_g = attn.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = attn_g.max(dim=2).values
    else:
        # "mean" (default) and "topp" fall back to mean. ShadowKV's own code
        # uses max; --group_agg_method max reproduces paper behaviour.
        scores = attn_g.mean(dim=2)
    return scores.squeeze(0)                                      # [H_kv, P]


# ---------------------------------------------------------------------------
# Attention-output fidelity (V-aware downstream quality)
# ---------------------------------------------------------------------------
def _attention_output(
    query_states: torch.Tensor,   # [1, H_q, 1, d]
    K: torch.Tensor,              # [1, H_kv, T, d]
    V: torch.Tensor,              # [1, H_kv, T, d]
    num_kv_groups: int,
) -> torch.Tensor:
    """softmax(Q·K/√d) · V over the provided (K, V). Returns [H_q, d]."""
    bsz, H_q, _, d = query_states.shape
    _, H_kv, T, _ = K.shape
    assert H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)

    K_exp = K.repeat_interleave(num_kv_groups, dim=1).float()     # [1, H_q, T, d]
    V_exp = V.repeat_interleave(num_kv_groups, dim=1).float()
    q = query_states.float()                                      # [1, H_q, 1, d]

    scores = torch.matmul(q, K_exp.transpose(-1, -2)) * scale     # [1, H_q, 1, T]
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V_exp).squeeze(2)              # [1, H_q, d]
    return output.squeeze(0)                                      # [H_q, d]


def _gather_selected_pages(
    paged_tensor: torch.Tensor,   # [1, H_kv, P, S, d]
    selected_indices: torch.Tensor,   # [H_kv, K]
) -> torch.Tensor:
    """Gather paged tensor along the P axis per kv-head.

    Returns: [1, H_kv, K, S, d] — the selected pages per head.
    """
    bsz, H_kv, P, S, d = paged_tensor.shape
    K = selected_indices.shape[-1]
    sel = selected_indices.view(1, H_kv, K, 1, 1).expand(bsz, H_kv, K, S, d)
    return torch.gather(paged_tensor, 2, sel)


def compute_output_fidelity(
    query_states: torch.Tensor,        # [1, H_q, 1, d]
    sink_k: torch.Tensor, sink_v: torch.Tensor,             # [1, H_kv, sink, d]
    paged_k: torch.Tensor, paged_v: torch.Tensor,           # [1, H_kv, P, S, d]
    recent_k: torch.Tensor, recent_v: torch.Tensor,         # [1, H_kv, recent, d]
    selections: dict[str, torch.Tensor],  # name -> [H_kv, K] GPU indices
    num_kv_groups: int,
) -> dict[str, torch.Tensor]:
    """Per-head cosine similarity between full attention output and each
    selector's drop-mode attention output.

    Full output and drop output are both computed with sink/recent always
    present. The ONLY difference is which paged pages are kept. Matches
    DCT-Page's actual drop-mode attention exactly.

    Returns dict of selector_name -> [H_q] cosine-similarity values in
    [-1, 1]. Typical range is near 1 when selection preserves output.
    """
    bsz, H_kv, P, S, d = paged_k.shape

    # (a) Full attention output reference: use every paged token.
    paged_k_flat = paged_k.reshape(bsz, H_kv, P * S, d)
    paged_v_flat = paged_v.reshape(bsz, H_kv, P * S, d)
    K_full = torch.cat([sink_k, paged_k_flat, recent_k], dim=2)   # [1, H_kv, T_full, d]
    V_full = torch.cat([sink_v, paged_v_flat, recent_v], dim=2)
    full_out = _attention_output(query_states, K_full, V_full, num_kv_groups)

    # (b) Drop-mode output per selector: gather selected pages, re-softmax.
    results: dict[str, torch.Tensor] = {}
    for name, sel_idx in selections.items():
        sel_idx = sel_idx.long()
        sel_k = _gather_selected_pages(paged_k, sel_idx)          # [1, H_kv, K, S, d]
        sel_v = _gather_selected_pages(paged_v, sel_idx)
        K_sel_flat = sel_k.reshape(bsz, H_kv, -1, d)              # [1, H_kv, K*S, d]
        V_sel_flat = sel_v.reshape(bsz, H_kv, -1, d)
        K_drop = torch.cat([sink_k, K_sel_flat, recent_k], dim=2)
        V_drop = torch.cat([sink_v, V_sel_flat, recent_v], dim=2)
        drop_out = _attention_output(query_states, K_drop, V_drop, num_kv_groups)

        # Cosine similarity per head, clamped to [-1, 1] for numerical safety.
        cos = F.cosine_similarity(full_out, drop_out, dim=-1, eps=1e-8)
        results[name] = cos.clamp(-1.0, 1.0)
    return results


def compute_all_metrics(
    page_mass: torch.Tensor,          # [H_q, P]
    extra_mass: torch.Tensor,         # [H_q] — sink + recent mass (always-kept floor)
    selected_indices: torch.Tensor,   # [H_kv, K]
    oracle_page_scores: torch.Tensor, # [H_kv, P]
    quest_scores: torch.Tensor,       # [H_kv, P]
    shadowkv_scores: torch.Tensor,    # [H_kv, P]
    num_kv_groups: int,
) -> dict[str, torch.Tensor]:
    """Compute six recall metrics. Returns dict of [H_q] float32 tensors.

    The five mass metrics include ``extra_mass`` (sink + recent) because
    those regions are always kept regardless of page selection.
    """
    H_q, P = page_mass.shape
    H_kv, K = selected_indices.shape
    assert H_q == H_kv * num_kv_groups, (
        f"H_q={H_q} != H_kv={H_kv} * num_kv_groups={num_kv_groups}"
    )
    assert extra_mass.shape == (H_q,), f"extra_mass shape {extra_mass.shape} != ({H_q},)"
    assert quest_scores.shape == (H_kv, P), (
        f"quest_scores shape {quest_scores.shape} != ({H_kv}, {P})"
    )
    assert shadowkv_scores.shape == (H_kv, P), (
        f"shadowkv_scores shape {shadowkv_scores.shape} != ({H_kv}, {P})"
    )

    page_mass = page_mass.float()
    extra_mass = extra_mass.float()
    oracle_page_scores = oracle_page_scores.float()
    quest_scores = quest_scores.float()
    shadowkv_scores = shadowkv_scores.float()
    selected_indices = selected_indices.long()

    # Same kv-head selection is consumed by every query in the group; expand
    # to H_q so gather indexes page_mass[q_head, :] correctly.
    sel_q = selected_indices.repeat_interleave(num_kv_groups, dim=0)           # [H_q, K]

    # (1) mass_recall_proxy: sink + recent + DCT's selected pages.
    mass_recall_proxy = torch.gather(page_mass, -1, sel_q).sum(-1) + extra_mass

    # (2) mass_recall_quest: sink + recent + Quest's top-K.
    quest_topk = torch.topk(quest_scores, K, dim=-1).indices                   # [H_kv, K]
    quest_topk_q = quest_topk.repeat_interleave(num_kv_groups, dim=0)          # [H_q, K]
    mass_recall_quest = (
        torch.gather(page_mass, -1, quest_topk_q).sum(-1) + extra_mass
    )

    # (3) mass_recall_shadowkv: sink + recent + ShadowKV's top-K.
    shadowkv_topk = torch.topk(shadowkv_scores, K, dim=-1).indices             # [H_kv, K]
    shadowkv_topk_q = shadowkv_topk.repeat_interleave(num_kv_groups, dim=0)    # [H_q, K]
    mass_recall_shadowkv = (
        torch.gather(page_mass, -1, shadowkv_topk_q).sum(-1) + extra_mass
    )

    # (4) mass_recall_oracle_max: sink + recent + oracle_max's top-K.
    oracle_topk = torch.topk(oracle_page_scores, K, dim=-1).indices            # [H_kv, K]
    oracle_topk_q = oracle_topk.repeat_interleave(num_kv_groups, dim=0)        # [H_q, K]
    mass_recall_oracle_max = (
        torch.gather(page_mass, -1, oracle_topk_q).sum(-1) + extra_mass
    )

    # (5) mass_recall_mass_topk: sink + recent + best-K pages by mass (ceiling).
    mass_topk_idx = torch.topk(page_mass, K, dim=-1).indices                   # [H_q, K]
    mass_recall_mass_topk = (
        torch.gather(page_mass, -1, mass_topk_idx).sum(-1) + extra_mass
    )

    # Ceiling must dominate all four selector metrics.
    tol = 1e-5
    if not (mass_recall_mass_topk + tol >= mass_recall_proxy).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_proxy — ceiling violated")
    if not (mass_recall_mass_topk + tol >= mass_recall_quest).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_quest — ceiling violated")
    if not (mass_recall_mass_topk + tol >= mass_recall_shadowkv).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_shadowkv — ceiling violated")
    if not (mass_recall_mass_topk + tol >= mass_recall_oracle_max).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_oracle_max — ceiling violated")

    # (6) set_recall vs oracle_max (existing per-kv-head metric), repeated to
    # query-head space so every row has a uniform schema.
    mO = _indices_to_mask(oracle_topk, P)                                      # [H_kv, P]
    mP = _indices_to_mask(selected_indices, P)                                 # [H_kv, P]
    set_recall_kv = (mP & mO).sum(-1).float() / mO.sum(-1).float().clamp(min=1)
    set_recall = set_recall_kv.repeat_interleave(num_kv_groups, dim=0)         # [H_q]

    # ----- Paged-only mass captured by each selector --------------------------
    # Numerator: softmax mass on the selector's chosen pages (no sink/recent
    # floor). Denominator: total paged mass per head (sink + recent excluded),
    # clamped to 1e-8 to guard against heads whose mass lands entirely on
    # sink/recent. Values are the fraction of paged attention mass captured,
    # with ceiling = 1 exactly when K ≥ P.
    paged_total = page_mass.sum(-1).clamp(min=1e-8)                            # [H_q]
    paged_mass_recall_proxy = (
        torch.gather(page_mass, -1, sel_q).sum(-1) / paged_total
    )
    paged_mass_recall_quest = (
        torch.gather(page_mass, -1, quest_topk_q).sum(-1) / paged_total
    )
    paged_mass_recall_shadowkv = (
        torch.gather(page_mass, -1, shadowkv_topk_q).sum(-1) / paged_total
    )
    paged_mass_recall_oracle_max = (
        torch.gather(page_mass, -1, oracle_topk_q).sum(-1) / paged_total
    )
    paged_mass_recall_mass_topk = (
        torch.gather(page_mass, -1, mass_topk_idx).sum(-1) / paged_total
    )

    # Paged ceiling must dominate all four paged-selector metrics.
    if not (paged_mass_recall_mass_topk + tol >= paged_mass_recall_proxy).all():
        raise AssertionError("paged_mass_recall_mass_topk < paged_mass_recall_proxy — ceiling violated")
    if not (paged_mass_recall_mass_topk + tol >= paged_mass_recall_quest).all():
        raise AssertionError("paged_mass_recall_mass_topk < paged_mass_recall_quest — ceiling violated")
    if not (paged_mass_recall_mass_topk + tol >= paged_mass_recall_shadowkv).all():
        raise AssertionError("paged_mass_recall_mass_topk < paged_mass_recall_shadowkv — ceiling violated")
    if not (paged_mass_recall_mass_topk + tol >= paged_mass_recall_oracle_max).all():
        raise AssertionError("paged_mass_recall_mass_topk < paged_mass_recall_oracle_max — ceiling violated")

    # Ratios against the paged-only ceiling make selector quality easier to
    # interpret when absolute paged-only recalls are small.
    paged_ceiling = paged_mass_recall_mass_topk.clamp(min=1e-8)
    paged_mass_ratio_proxy = paged_mass_recall_proxy / paged_ceiling
    paged_mass_ratio_quest = paged_mass_recall_quest / paged_ceiling

    return {
        "mass_recall_sink_recent": extra_mass,
        "mass_recall_proxy": mass_recall_proxy,
        "mass_recall_quest": mass_recall_quest,
        "mass_recall_shadowkv": mass_recall_shadowkv,
        "mass_recall_oracle_max": mass_recall_oracle_max,
        "mass_recall_mass_topk": mass_recall_mass_topk,
        "set_recall": set_recall,
        "paged_mass_recall_proxy": paged_mass_recall_proxy,
        "paged_mass_recall_quest": paged_mass_recall_quest,
        "paged_mass_recall_shadowkv": paged_mass_recall_shadowkv,
        "paged_mass_recall_oracle_max": paged_mass_recall_oracle_max,
        "paged_mass_recall_mass_topk": paged_mass_recall_mass_topk,
        "paged_mass_ratio_proxy": paged_mass_ratio_proxy,
        "paged_mass_ratio_quest": paged_mass_ratio_quest,
    }


# ---------------------------------------------------------------------------
# Recorder: computes mass inline, discards large tensors before returning
# ---------------------------------------------------------------------------
class MassRecallRecorder:
    """Per-decode-step recorder that computes mass metrics inside the hook.

    paged_k / query_states arrive on GPU and are NOT retained across calls —
    only the small per-record metric tensors are kept.
    """

    def __init__(self, num_decode_steps: int, group_agg_method: str):
        self.num_decode_steps = num_decode_steps
        self.group_agg_method = group_agg_method
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step >= self.num_decode_steps:
            return

        num_kv_groups = int(payload["num_kv_groups"])
        num_pages = int(payload["num_pages"])
        actual_top_k = int(payload["actual_top_k"])

        query_states = payload["query_states"]
        paged_k = payload["paged_k"]
        paged_v = payload["paged_v"]
        sink_k = payload["sink_k"]
        sink_v = payload["sink_v"]
        recent_k = payload["recent_k"]
        recent_v = payload["recent_v"]
        device = paged_k.device

        with torch.no_grad():
            page_mass_gpu, extra_mass_gpu = compute_per_page_mass(
                query_states, sink_k, paged_k, recent_k, num_kv_groups,
            )
            page_mass = page_mass_gpu.float().cpu()                            # [H_q, P]
            extra_mass = extra_mass_gpu.float().cpu()                          # [H_q]

            quest_scores_gpu = compute_quest_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )
            quest_scores = quest_scores_gpu.float().cpu()                      # [H_kv, P]

            shadowkv_scores_gpu = compute_shadowkv_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )
            shadowkv_scores = shadowkv_scores_gpu.float().cpu()                # [H_kv, P]

            # Compute per-selector top-K indices on GPU for output fidelity.
            # oracle_page_scores arrives CPU — move to GPU for topk+gather.
            oracle_scores_gpu = (
                payload["oracle_page_scores"][0].to(device, non_blocking=True)
            )
            proxy_topk_gpu = payload["selected_indices"][0].to(
                device, non_blocking=True
            ).long()
            quest_topk_gpu = torch.topk(quest_scores_gpu, actual_top_k, dim=-1).indices
            shadowkv_topk_gpu = torch.topk(
                shadowkv_scores_gpu, actual_top_k, dim=-1,
            ).indices
            oracle_topk_gpu = torch.topk(oracle_scores_gpu, actual_top_k, dim=-1).indices

            fidelity_gpu = compute_output_fidelity(
                query_states, sink_k, sink_v, paged_k, paged_v, recent_k, recent_v,
                {
                    "output_fidelity_proxy": proxy_topk_gpu,
                    "output_fidelity_quest": quest_topk_gpu,
                    "output_fidelity_shadowkv": shadowkv_topk_gpu,
                    "output_fidelity_oracle_max": oracle_topk_gpu,
                },
                num_kv_groups,
            )
            fidelity = {k: v.float().cpu() for k, v in fidelity_gpu.items()}

        # oracle_page_scores and selected_indices arrive already detached/CPU.
        selected_indices = payload["selected_indices"][0]                      # [H_kv, K]
        oracle_page_scores = payload["oracle_page_scores"][0]                  # [H_kv, P]

        mass_metrics = compute_all_metrics(
            page_mass, extra_mass, selected_indices, oracle_page_scores,
            quest_scores, shadowkv_scores, num_kv_groups,
        )
        metrics = {**mass_metrics, **fidelity}

        # Invariants: mass_* ∈ [0, 1]; fidelity_* ∈ [-1, 1] (cos sim).
        for key, tensor in metrics.items():
            lo, hi = float(tensor.min()), float(tensor.max())
            if key in FIDELITY_METRIC_KEYS:
                low_bound, high_bound = -1.0, 1.0
            else:
                low_bound, high_bound = 0.0, 1.0
            if lo < low_bound - 1e-5 or hi > high_bound + 1e-5:
                raise AssertionError(
                    f"{key} out of [{low_bound}, {high_bound}]: "
                    f"min={lo:.6f}, max={hi:.6f} "
                    f"(layer={layer_idx}, step={decode_step})"
                )

        self.records.append({
            "layer_idx": layer_idx,
            "decode_step": decode_step,
            "num_pages": num_pages,
            "actual_top_k": actual_top_k,
            "num_kv_groups": num_kv_groups,
            "H_q": int(page_mass.shape[0]),
            **{k: metrics[k].tolist() for k in METRIC_KEYS},
        })


def generate_with_mass_traces(
    model, tokenizer, sample: dict[str, Any], num_decode_steps: int,
    group_agg_method: str,
) -> tuple[list[dict[str, Any]], int]:
    """Run generate() with the mass-recall hook installed."""
    from dct_page_attention import set_dct_page_debug_hook

    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder = MassRecallRecorder(num_decode_steps, group_agg_method)
    set_dct_page_debug_hook(recorder)
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
        set_dct_page_debug_hook(None)

    return recorder.records, int(input_ids.shape[1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attention-mass recall vs full-attention softmax on RULER"
    )
    # Model
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    # RULER
    p.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--data_root", type=Path,
                   default=Path("benchmark/data/ruler_data"))

    # DCT page config
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--scoring_method", type=str, default="max",
                   choices=["mean", "max", "sum"])
    p.add_argument("--group_agg_method", type=str, default="max",
                   choices=["mean", "max", "topp"])
    p.add_argument("--unselected_mode", type=str, default="drop",
                   choices=["drop", "compressed"])
    p.add_argument("--compression_method", type=str, default="dct",
                   choices=["haar", "dct"])
    p.add_argument("--compressed_token_rope", type=str, default="mixed",
                   choices=["mixed", "block_center"])
    p.add_argument("--proxy_frequency_layout", type=str, default="low")

    # comp_kv_quant
    p.add_argument("--comp_kv_quant", type=str, default="fp8_e4m3",
                   choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    p.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                   choices=["per_page", "per_comp_token"])

    # Analysis
    p.add_argument("--num_decode_steps", type=int, default=20,
                   help="Number of decode steps per sample to record.")

    # Output
    p.add_argument("--output_dir", type=Path,
                   default=Path("results_attention_mass_recall"))
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _aggregate_metric_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {k: 0.0 for k in METRIC_KEYS}
    return {k: _mean([d[k] for d in dicts if k in d]) for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    start_time = time.time()
    torch.manual_seed(42)

    run_name = args.run_name or (
        f"mass_{args.compression_method}_ps{args.page_size}"
        f"_topk{args.top_k}_cr{args.compress_ratio}_quant_{args.comp_kv_quant}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = run_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("Applying DCT page attention patch...")
    apply_monkey_patch(args)
    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _, tokenizer_family = infer_model_family(args.base_model)
    task_configs = load_task_configs()

    per_task_results: dict[str, Any] = {}

    try:
        for task in args.tasks:
            if task not in task_configs:
                print(f"  WARNING: task {task!r} not in RULER configs, skipping")
                continue
            print(f"\n{'=' * 60}\nTASK: {task}\n{'=' * 60}")

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

            task_overall_records: list[dict[str, float]] = []
            task_per_layer: dict[int, list[dict[str, float]]] = {}

            sample_fp = (per_sample_dir / f"{task}.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )

            for sample_idx, sample in enumerate(
                tqdm(samples, desc=f"  {task}"), start=1
            ):
                records, input_len = generate_with_mass_traces(
                    model, tokenizer, sample, args.num_decode_steps,
                    args.group_agg_method,
                )
                if not records:
                    print(f"  WARNING: no traces for sample {sample['index']} "
                          f"(input_len={input_len}); skipping")
                    continue

                per_head_rows: list[dict[str, Any]] = []
                per_layer_buckets: dict[int, list[dict[str, float]]] = {}

                for rec in records:
                    layer_idx = rec["layer_idx"]
                    decode_step = rec["decode_step"]
                    num_kv_groups = rec["num_kv_groups"]
                    H_q = rec["H_q"]

                    for q in range(H_q):
                        flat = {k: rec[k][q] for k in METRIC_KEYS}
                        row = {
                            "layer_idx": layer_idx,
                            "decode_step": decode_step,
                            "q_head": q,
                            "kv_head": q // num_kv_groups,
                            "actual_top_k": rec["actual_top_k"],
                            **flat,
                        }
                        per_head_rows.append(row)
                        per_layer_buckets.setdefault(layer_idx, []).append(flat)
                        task_overall_records.append(flat)
                        task_per_layer.setdefault(layer_idx, []).append(flat)

                per_layer_mean = {
                    str(lyr): _aggregate_metric_dicts(bucket)
                    for lyr, bucket in sorted(per_layer_buckets.items())
                }

                sample_record = {
                    "sample_index": int(sample["index"]),
                    "input_len": input_len,
                    "num_records": len(records),
                    "per_layer_mean": per_layer_mean,
                    "per_head": per_head_rows,
                }
                sample_fp.write(
                    json.dumps(sample_record, ensure_ascii=False) + "\n"
                )

                if sample_idx % 5 == 0 or sample_idx == len(samples):
                    o = _aggregate_metric_dicts(task_overall_records)
                    print(
                        f"  [{sample_idx}/{len(samples)}] "
                        f"sr={o['mass_recall_sink_recent']:.3f}  "
                        f"mass[p/q/s/o/c] = "
                        f"{o['mass_recall_proxy']:.3f}/"
                        f"{o['mass_recall_quest']:.3f}/"
                        f"{o['mass_recall_shadowkv']:.3f}/"
                        f"{o['mass_recall_oracle_max']:.3f}/"
                        f"{o['mass_recall_mass_topk']:.3f}  "
                        f"paged[p/q/s/o/c] = "
                        f"{o['paged_mass_recall_proxy']:.3f}/"
                        f"{o['paged_mass_recall_quest']:.3f}/"
                        f"{o['paged_mass_recall_shadowkv']:.3f}/"
                        f"{o['paged_mass_recall_oracle_max']:.3f}/"
                        f"{o['paged_mass_recall_mass_topk']:.3f}  "
                        f"ratio[p/q] = "
                        f"{o['paged_mass_ratio_proxy']:.3f}/"
                        f"{o['paged_mass_ratio_quest']:.3f}  "
                        f"fid[p/q/s/o] = "
                        f"{o['output_fidelity_proxy']:.3f}/"
                        f"{o['output_fidelity_quest']:.3f}/"
                        f"{o['output_fidelity_shadowkv']:.3f}/"
                        f"{o['output_fidelity_oracle_max']:.3f}"
                    )

            sample_fp.close()

            per_task_results[task] = {
                "num_samples": len(samples),
                "overall": _aggregate_metric_dicts(task_overall_records),
                "per_layer": {
                    str(lyr): _aggregate_metric_dicts(bucket)
                    for lyr, bucket in sorted(task_per_layer.items())
                },
            }
            o = per_task_results[task]["overall"]
            print(
                f"  TASK SUMMARY\n"
                f"    sink+recent (floor)                     = "
                f"{o['mass_recall_sink_recent']:.3f}\n"
                f"    mass   [proxy/quest/shadow/oracle/ceil] = "
                f"{o['mass_recall_proxy']:.3f} / {o['mass_recall_quest']:.3f} / "
                f"{o['mass_recall_shadowkv']:.3f} / {o['mass_recall_oracle_max']:.3f} / "
                f"{o['mass_recall_mass_topk']:.3f}\n"
                f"    paged  [proxy/quest/shadow/oracle/ceil] = "
                f"{o['paged_mass_recall_proxy']:.3f} / {o['paged_mass_recall_quest']:.3f} / "
                f"{o['paged_mass_recall_shadowkv']:.3f} / {o['paged_mass_recall_oracle_max']:.3f} / "
                f"{o['paged_mass_recall_mass_topk']:.3f}\n"
                f"    ratio  [proxy/quest vs ceil]            = "
                f"{o['paged_mass_ratio_proxy']:.3f} / {o['paged_mass_ratio_quest']:.3f}\n"
                f"    fidelity[proxy/quest/shadow/oracle]     = "
                f"{o['output_fidelity_proxy']:.3f} / {o['output_fidelity_quest']:.3f} / "
                f"{o['output_fidelity_shadowkv']:.3f} / {o['output_fidelity_oracle_max']:.3f}\n"
                f"    set_recall = {o['set_recall']:.3f}"
            )

        overall_task_means = [r["overall"] for r in per_task_results.values()]
        overall = _aggregate_metric_dicts(overall_task_means)

        summary = {
            "config": {
                "base_model": args.base_model,
                "seq_len": args.seq_len,
                "num_samples": args.num_samples,
                "num_decode_steps": args.num_decode_steps,
                "page_size": args.page_size,
                "top_k": args.top_k,
                "sink_size": args.sink_size,
                "recent_size": args.recent_size,
                "compress_ratio": args.compress_ratio,
                "compression_method": args.compression_method,
                "compressed_token_rope": args.compressed_token_rope,
                "proxy_frequency_layout": args.proxy_frequency_layout,
                "scoring_method": args.scoring_method,
                "group_agg_method": args.group_agg_method,
                "unselected_mode": args.unselected_mode,
                "comp_kv_quant": args.comp_kv_quant,
                "comp_kv_quant_granularity": args.comp_kv_quant_granularity,
            },
            "per_task": per_task_results,
            "overall": overall,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        elapsed = (time.time() - start_time) / 60
        print(f"\n{'=' * 60}\nOVERALL RESULTS\n{'=' * 60}")
        for k in METRIC_KEYS:
            print(f"  {k:25s} = {overall[k]:.3f}")
        print(f"\n  Results: {run_dir}")
        print(f"  Total time: {elapsed:.1f} min")

    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
