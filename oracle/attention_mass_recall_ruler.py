#!/usr/bin/env python3
"""
Attention-mass recall on RULER — dense-trajectory reference.

Generation runs under the **unmodified full-KV forward** (no DCT patch, no
selector drives decoding). A recording hook mirrors HF's own attention
forward 1:1 and only observes (Q, K, V) post-RoPE / post-cache-update, so
every selector — DCT Haar proxy, Quest, ShadowKV, oracle_max — is evaluated
against the same neutral Q at each decode step. This removes the
"home-field" bias of scoring Quest/oracle on a Q already shaped by DCT's
earlier page choices.

For each decode step, applies a single softmax over the **entire KV sequence**
(sink + paged + recent) to compute a per-token mass distribution. Sink and
recent tokens are always kept, so their mass is a fixed floor added to every
mass-recall metric. The per-page mass of each paged page is the sum of
softmax weights on that page's tokens.

Sum invariant per head:   sink_mass + Σ_p page_mass[p] + recent_mass = 1

Reports per-query-head metrics grouped into three families.

(A) FULL-KV MASS metrics (all include the sink + recent floor):

  mass_recall_sink        = sink                                           (floor component)
  mass_recall_recent      = recent                                         (floor component)
  mass_recall_proxy       = sink + recent + Σ m[p] over DCT proxy's top-K
  mass_recall_quest       = sink + recent + Σ m[p] over Quest's top-K
  mass_recall_shadowkv    = sink + recent + Σ m[p] over ShadowKV's top-K
  mass_recall_oracle_max  = sink + recent + Σ m[p] over oracle_max's top-K
  mass_recall_mass_topk   = sink + recent + Σ m[p] over top-K by page mass  (ceiling)
  set_recall              = |DCT ∩ oracle_max| / K                         (page-set baseline)

(B) SELECTED-PAGE MASS metrics (fraction of total softmax mass that lands
    on the selector's chosen pages; denominator = 1):

  selected_mass_proxy       = Σ_{p∈DCT topK} m[p]        = 1 − sink − recent − Σ_{unselected} m[p]
  selected_mass_quest       = Σ_{p∈Quest topK} m[p]
  selected_mass_shadowkv    = Σ_{p∈ShadowKV topK} m[p]
  selected_mass_oracle_max  = Σ_{p∈oracle_max topK} m[p]
  selected_mass_mass_topk   = Σ_{p∈mass topK} m[p]                         (ceiling)

By construction:
  mass_recall_X   = selected_mass_X + mass_recall_sink + mass_recall_recent
  selected_mass_X + Σ_{p∉ topK} m[p] = 1 − sink − recent

(C) PAGED-ONLY MASS metrics (no sink/recent floor; denominator = total
    paged attention mass Σ_p m[p] = 1 − (sink_mass + recent_mass)):

  paged_mass_recall_proxy       = Σ_{p∈DCT topK} m[p]       / Σ_p m[p]
  paged_mass_recall_quest       = Σ_{p∈Quest topK} m[p]     / Σ_p m[p]
  paged_mass_recall_shadowkv    = Σ_{p∈ShadowKV topK} m[p]  / Σ_p m[p]
  paged_mass_recall_oracle_max  = Σ_{p∈oracle_max topK} m[p]/ Σ_p m[p]
  paged_mass_recall_mass_topk   = Σ_{p∈mass topK} m[p]      / Σ_p m[p]   (ceiling)
  paged_mass_ratio_proxy        = paged_mass_recall_proxy / paged_mass_recall_mass_topk
  paged_mass_ratio_quest        = paged_mass_recall_quest / paged_mass_recall_mass_topk
  paged_mass_ratio_shadowkv     = paged_mass_recall_shadowkv / paged_mass_recall_mass_topk

Paged-only strips the always-kept sink + recent floor from both numerator
and denominator, so values are the fraction of **paged** attention mass
(not total) captured by the paged selection. This rescales each head's
paged mass so the ceiling = 1 exactly when K ≥ P, and separates
selection quality from the always-kept floor. By construction:

  floor = mass_recall_sink + mass_recall_recent
  mass_recall_X       = paged_mass_recall_X · (1 − floor) + floor
  selected_mass_X     = paged_mass_recall_X · (1 − floor)
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

Reuses the dense recording-forward plumbing from
``attention_mass_recall_ruler_quest.py`` (``_install_recording_forward``,
``set_recording_hook``, ``load_model``). DCT proxy / oracle scores are
reproduced inline from (Q, paged_k) — no dependency on the DCT forward
itself.

Usage:
    python oracle/attention_mass_recall_ruler.py \\
        --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --tasks niah_single_1 --num_samples 2 --seq_len 32768 \\
        --page_size 16 --top_k 128 --num_decode_steps 2 \\
        --output_dir results_attention_mass_recall --run_name smoke
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from eval_ruler import infer_model_family


ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]


def _indices_to_mask(indices: torch.Tensor, num_pages: int) -> torch.Tensor:
    """indices: [..., M] → bool mask [..., num_pages]."""
    shape = list(indices.shape[:-1]) + [num_pages]
    mask = torch.zeros(shape, dtype=torch.bool, device=indices.device)
    mask.scatter_(-1, indices.long(), True)
    return mask


def load_task_configs() -> dict[str, dict]:
    ruler_dir = str(_REPO_ROOT / "benchmark" / "eval_ruler")
    sys.path.insert(0, os.path.join(ruler_dir, "data"))
    data_constants = importlib.import_module("synthetic.constants")
    data_tasks = data_constants.TASKS
    if "synthetic.constants" in sys.modules:
        del sys.modules["synthetic.constants"]
    sys.path.insert(0, os.path.join(ruler_dir, "eval"))
    eval_constants = importlib.import_module("synthetic.constants")
    eval_tasks = eval_constants.TASKS
    with open(os.path.join(ruler_dir, "synthetic.yaml"), "r") as f:
        yaml_tasks = yaml.safe_load(f)
    configs = {}
    for task_name, yaml_cfg in yaml_tasks.items():
        base_task = yaml_cfg["task"]
        cfg = dict(yaml_cfg)
        cfg.update(data_tasks[base_task])
        cfg.update(eval_tasks[base_task])
        configs[task_name] = cfg
    return configs

# Dense recording forward + model helpers live in the Quest sibling script.
# Imported lazily at call sites to avoid a circular import (that module
# imports compute_per_page_mass / compute_quest_scores / compute_output_fidelity
# from this one).


MASS_METRIC_KEYS = [
    # Always-kept floor components (each is a fraction of total softmax mass).
    "mass_recall_sink",
    "mass_recall_recent",
    # Total paged-region mass per head: Σ_p m[p] = 1 − sink − recent.
    # Carried through aggregation as the denominator for paged_mass_recall_*.
    "pages_mass",
    # Mass of (sink + selected pages + recent) / full KV — includes always-kept floor.
    "mass_recall_proxy",
    "mass_recall_quest",
    "mass_recall_shadowkv",
    "mass_recall_oracle_max",
    "mass_recall_mass_topk",
    "set_recall",
    # Mass of (selected pages) / (full KV) — absolute fraction of total
    # attention mass that lands on the selector's chosen pages.
    # Equivalently: 1 − sink − recent − Σ_{unselected} m[p].
    # mass_recall_X = selected_mass_X + sink + recent.
    # paged_mass_recall_* and paged_mass_ratio_* are derived post-hoc from
    # these aggregates (see DERIVED_PAGED_KEYS / _derive_paged_metrics).
    "selected_mass_proxy",
    "selected_mass_quest",
    "selected_mass_shadowkv",
    "selected_mass_oracle_max",
    "selected_mass_mass_topk",
]

# Paged-only metrics derived from aggregated MASS_METRIC_KEYS. Computed at
# print/summary time (ratio-of-means) instead of per decode step:
#   paged_mass_recall_X = mean(selected_mass_X) / mean(pages_mass)
#   paged_mass_ratio_X  = mean(selected_mass_X) / mean(selected_mass_mass_topk)
DERIVED_PAGED_KEYS = [
    "paged_mass_recall_proxy",
    "paged_mass_recall_quest",
    "paged_mass_recall_shadowkv",
    "paged_mass_recall_oracle_max",
    "paged_mass_recall_mass_topk",
    "paged_mass_ratio_proxy",
    "paged_mass_ratio_quest",
    "paged_mass_ratio_shadowkv",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        page_mass:   [H_q, P] — softmax weights summed within each paged page.
        sink_mass:   [H_q]    — softmax weights on sink tokens (always-kept).
        recent_mass: [H_q]    — softmax weights on recent tokens (always-kept).
        Invariant: page_mass.sum(-1) + sink_mass + recent_mass = 1.
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

    return (
        page_mass.squeeze(0),                                    # [H_q, P]
        sink_mass.squeeze(0),                                    # [H_q]
        recent_mass.squeeze(0),                                  # [H_q]
    )


_dct_proj_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_dct_lowpass_projection_matrix(
    page_size: int, comp_size: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Return the [comp_size, page_size] DCT → lowpass truncate → IDCT →
    energy-correction projection matrix, matching DCT-Page's default pipeline.

    Built by ``_build_dct_projection_matrix`` in ``dct_page_attention.py``
    (imported lazily). Cached per (shape, device, dtype).
    """
    key = (page_size, comp_size, device, dtype)
    M = _dct_proj_cache.get(key)
    if M is None:
        from dct_page_attention import _build_dct_projection_matrix
        M = _build_dct_projection_matrix(page_size, comp_size, device, dtype)
        _dct_proj_cache[key] = M
    return M


def compute_dct_lowpass_proxy_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    comp_size: int,
    num_kv_groups: int,
    group_agg_method: str,
    scoring_method: str,
    comp_kv_quant: str = "none",
    comp_kv_quant_granularity: str = "per_page",
) -> torch.Tensor:
    """DCT → lowpass truncate → IDCT → energy-correction proxy page scores,
    matching the default ``eval_ruler.py`` pipeline.

    For each page p of S tokens, apply the DCT-Page projection matrix
    ``M ∈ R^{comp_size × S}`` that bakes in the full DCT-lowpass-IDCT with
    ``√(comp_size/S)`` energy correction:
        comp_k[p, c, :] = Σ_s M[c, s] · paged_k[p, s, :]
    Optionally quantize→dequantize comp_k to simulate low-precision compressed-KV
    storage (``_quantize_for_storage`` + ``_dequantize_comp`` from
    ``dct_page_attention.py``), then score = reduce_c (q[h] · comp_k[p, c]) / √d,
    GQA group-aggregated.

    Args:
        query_states: [bsz=1, H_q, 1, d] — post-RoPE / post-QK-norm.
        paged_k:      [bsz=1, H_kv, P, S, d] — post-RoPE.
        comp_size:    Number of comp tokens per page.
        num_kv_groups: H_q // H_kv.
        group_agg_method: "mean" | "max".
        scoring_method: "max" | "mean" | "sum" over the comp_size axis.
        comp_kv_quant: "none" | "fp8_e4m3" | "fp8_e5m2" | "int8" | "int4".
        comp_kv_quant_granularity: "per_page" | "per_comp_token".

    Returns:
        scores: [H_kv, P] — one proxy score per (kv-head, page).
    """
    bsz, H_q, q_len, d = query_states.shape
    assert bsz == 1 and q_len == 1, f"decode-step only, got shape {query_states.shape}"
    _, H_kv, P, S, _ = paged_k.shape
    assert H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)

    M = _get_dct_lowpass_projection_matrix(
        S, comp_size, paged_k.device, paged_k.dtype,
    )                                                                         # [C, S]
    # Project paged_k along the page-size axis: [..., P, S, d] @ M.T -> [..., P, C, d]
    comp_k = torch.einsum("bhpsd,cs->bhpcd", paged_k, M)                      # [1, H_kv, P, C, d]

    if comp_kv_quant != "none":
        from dct_page_attention import _quantize_for_storage, _dequantize_comp
        x_q, scale_q = _quantize_for_storage(comp_k, comp_kv_quant, comp_kv_quant_granularity)
        comp_k = _dequantize_comp(
            x_q, scale_q, comp_kv_quant, comp_kv_quant_granularity,
            comp_k.shape[-1], out_dtype=comp_k.dtype,
        )

    comp_k_q = comp_k.repeat_interleave(num_kv_groups, dim=1).float()         # [1, H_q, P, C, d]

    q = query_states.float()                                                  # [1, H_q, 1, d]
    scores_per_comp = torch.einsum(
        "bhqd,bhpcd->bhpc", q, comp_k_q,
    ) * scale                                                                 # [1, H_q, P, C]

    if scoring_method == "max":
        score_q = scores_per_comp.amax(dim=-1)
    elif scoring_method == "mean":
        score_q = scores_per_comp.mean(dim=-1)
    elif scoring_method == "sum":
        score_q = scores_per_comp.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported scoring_method: {scoring_method!r}")

    # GQA group reduction to kv-head level.
    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.max(dim=2).values
    else:
        scores = score_g.mean(dim=2)
    return scores.squeeze(0)                                                  # [H_kv, P]


def compute_oracle_max_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    num_kv_groups: int,
    group_agg_method: str,
) -> torch.Tensor:
    """Oracle per-page upper bound: max_{s in page} q · K[p,s] / √d.

    Group-reduced to kv-head level the same way the proxies are.
    """
    bsz, H_q, q_len, d = query_states.shape
    assert bsz == 1 and q_len == 1, f"decode-step only, got shape {query_states.shape}"
    _, H_kv, P, S, _ = paged_k.shape
    assert H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)

    k_exp = paged_k.repeat_interleave(num_kv_groups, dim=1).float()           # [1, H_q, P, S, d]
    q = query_states.float()                                                  # [1, H_q, 1, d]
    qk = torch.einsum("bhqd,bhpsd->bhps", q, k_exp) * scale                   # [1, H_q, P, S]
    score_q = qk.amax(dim=-1)                                                 # [1, H_q, P]

    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.max(dim=2).values
    else:
        scores = score_g.mean(dim=2)
    return scores.squeeze(0)                                                  # [H_kv, P]


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
    sink_mass: torch.Tensor,          # [H_q] — softmax mass on sink tokens
    recent_mass: torch.Tensor,        # [H_q] — softmax mass on recent tokens
    selected_indices: torch.Tensor,   # [H_kv, K]
    oracle_page_scores: torch.Tensor, # [H_kv, P]
    quest_scores: torch.Tensor,       # [H_kv, P]
    shadowkv_scores: torch.Tensor,    # [H_kv, P]
    num_kv_groups: int,
) -> dict[str, torch.Tensor]:
    """Compute mass-recall metrics. Returns dict of [H_q] float32 tensors.

    Full-KV mass metrics include ``sink_mass + recent_mass`` because those
    regions are always kept regardless of page selection.
    """
    H_q, P = page_mass.shape
    H_kv, K = selected_indices.shape
    assert H_q == H_kv * num_kv_groups, (
        f"H_q={H_q} != H_kv={H_kv} * num_kv_groups={num_kv_groups}"
    )
    assert sink_mass.shape == (H_q,), f"sink_mass shape {sink_mass.shape} != ({H_q},)"
    assert recent_mass.shape == (H_q,), f"recent_mass shape {recent_mass.shape} != ({H_q},)"
    assert quest_scores.shape == (H_kv, P), (
        f"quest_scores shape {quest_scores.shape} != ({H_kv}, {P})"
    )
    assert shadowkv_scores.shape == (H_kv, P), (
        f"shadowkv_scores shape {shadowkv_scores.shape} != ({H_kv}, {P})"
    )

    page_mass = page_mass.float()
    sink_mass = sink_mass.float()
    recent_mass = recent_mass.float()
    extra_mass = sink_mass + recent_mass                                       # [H_q]
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

    # ----- Selected-page mass (absolute, fraction of total softmax mass) ------
    # Equivalent to 1 − sink − recent − Σ_{unselected} m[p]. Derived from the
    # full-KV mass metrics by subtracting the always-kept floor.
    selected_mass_proxy = mass_recall_proxy - extra_mass
    selected_mass_quest = mass_recall_quest - extra_mass
    selected_mass_shadowkv = mass_recall_shadowkv - extra_mass
    selected_mass_oracle_max = mass_recall_oracle_max - extra_mass
    selected_mass_mass_topk = mass_recall_mass_topk - extra_mass

    # Total paged-region mass per head (= 1 − sink − recent). Carried through
    # aggregation as the denominator for paged_mass_recall_*; the ratios are
    # derived post-hoc as ratio-of-aggregated-means by ``_derive_paged_metrics``
    # rather than averaged per-step.
    pages_mass = page_mass.sum(-1)                                             # [H_q]

    return {
        "mass_recall_sink": sink_mass,
        "mass_recall_recent": recent_mass,
        "pages_mass": pages_mass,
        "mass_recall_proxy": mass_recall_proxy,
        "mass_recall_quest": mass_recall_quest,
        "mass_recall_shadowkv": mass_recall_shadowkv,
        "mass_recall_oracle_max": mass_recall_oracle_max,
        "mass_recall_mass_topk": mass_recall_mass_topk,
        "set_recall": set_recall,
        "selected_mass_proxy": selected_mass_proxy,
        "selected_mass_quest": selected_mass_quest,
        "selected_mass_shadowkv": selected_mass_shadowkv,
        "selected_mass_oracle_max": selected_mass_oracle_max,
        "selected_mass_mass_topk": selected_mass_mass_topk,
    }


def _derive_paged_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Derive paged_mass_recall_* and paged_mass_ratio_* from aggregated masses.

    Per-step:
        pages_mass            = Σ_p page_mass[p] = 1 − sink − recent
        selected_mass_X       = Σ_{p ∈ topK_X} page_mass[p]
        paged_mass_recall_X   = selected_mass_X / pages_mass
        paged_mass_ratio_X    = selected_mass_X / selected_mass_mass_topk

    Aggregated (across head, step, sample, task, layer) — ratio of means:
        paged_mass_recall_X = mean(selected_mass_X) / mean(pages_mass)
        paged_mass_ratio_X  = mean(selected_mass_X) / mean(selected_mass_mass_topk)

    A ratio-of-means aggregation rather than mean-of-per-step-ratios; more
    stable when per-head paged_mass is tiny.
    """
    pm = float(metrics.get("pages_mass", 0.0))
    smm = float(metrics.get("selected_mass_mass_topk", 0.0))
    out: dict[str, float] = {}
    for sel in ("proxy", "quest", "shadowkv", "oracle_max", "mass_topk"):
        sm = float(metrics.get(f"selected_mass_{sel}", 0.0))
        out[f"paged_mass_recall_{sel}"] = (sm / pm) if pm > 1e-12 else 0.0
    for sel in ("proxy", "quest", "shadowkv"):
        sm = float(metrics.get(f"selected_mass_{sel}", 0.0))
        out[f"paged_mass_ratio_{sel}"] = (sm / smm) if smm > 1e-12 else 0.0
    return out


# ---------------------------------------------------------------------------
# comp_size sweep: per-step selected_mass_* (ratios derived post-hoc)
# ---------------------------------------------------------------------------
def compute_selected_mass_sweep(
    query_states: torch.Tensor,        # [1, H_q, 1, d]
    paged_k: torch.Tensor,             # [1, H_kv, P, S, d]
    page_mass: torch.Tensor,           # [H_q, P]
    top_k: int,
    comp_sizes: list[int],
    num_kv_groups: int,
    group_agg_method: str,
    scoring_method: str,
) -> dict[int, torch.Tensor]:
    """Per-step selected_mass_proxy(c) for each comp_size.

    For each c in ``comp_sizes`` build the proxy with c lowpass tokens per page,
    pick its top-K pages, and return the mass landed on the chosen pages:

        selected_mass_proxy(c) = Σ_{p∈proxy_topK(c)} m[p]

    Aggregation derives paged_mass_recall_proxy(c) and paged_mass_ratio_proxy(c)
    post-hoc as ratio-of-means using selected_mass_mass_topk and pages_mass.
    Quantization is intentionally disabled (full-precision compressed K).
    """
    H_q, P = page_mass.shape
    actual_top_k = min(top_k, P)

    out: dict[int, torch.Tensor] = {}
    for c in comp_sizes:
        proxy_scores = compute_dct_lowpass_proxy_scores(
            query_states, paged_k, c, num_kv_groups,
            group_agg_method, scoring_method,
            comp_kv_quant="none",
        )                                                                      # [H_kv, P]
        proxy_topk = torch.topk(proxy_scores, actual_top_k, dim=-1).indices    # [H_kv, K]
        proxy_topk_q = proxy_topk.repeat_interleave(num_kv_groups, dim=0)      # [H_q, K]
        out[c] = torch.gather(page_mass, -1, proxy_topk_q).sum(-1)             # [H_q]
    return out


def compute_quest_selected_mass(
    query_states: torch.Tensor,        # [1, H_q, 1, d]
    paged_k: torch.Tensor,             # [1, H_kv, P, S, d]
    page_mass: torch.Tensor,           # [H_q, P]
    top_k: int,
    num_kv_groups: int,
    group_agg_method: str,
) -> torch.Tensor:
    """Quest's per-step selected_mass_quest. No comp_size knob."""
    H_q, P = page_mass.shape
    actual_top_k = min(top_k, P)

    quest_scores = compute_quest_scores(
        query_states, paged_k, num_kv_groups, group_agg_method,
    )                                                                          # [H_kv, P]
    quest_topk = torch.topk(quest_scores, actual_top_k, dim=-1).indices        # [H_kv, K]
    quest_topk_q = quest_topk.repeat_interleave(num_kv_groups, dim=0)          # [H_q, K]
    return torch.gather(page_mass, -1, quest_topk_q).sum(-1)                   # [H_q]


def compute_mass_topk_selected_mass(
    page_mass: torch.Tensor,           # [H_q, P]
    top_k: int,
) -> torch.Tensor:
    """Per-step selected_mass_mass_topk: best-K pages by mass (ceiling)."""
    H_q, P = page_mass.shape
    actual_top_k = min(top_k, P)
    mass_topk_idx = torch.topk(page_mass, actual_top_k, dim=-1).indices        # [H_q, K]
    return torch.gather(page_mass, -1, mass_topk_idx).sum(-1)                  # [H_q]


# ---------------------------------------------------------------------------
# Recorder: computes mass inline, discards large tensors before returning
# ---------------------------------------------------------------------------
class MassRecallRecorder:
    """Per-decode-step recorder that computes mass metrics inline.

    Dense-trajectory design: the recording forward (installed via
    ``_install_recording_forward`` from the Quest sibling script) runs
    standard full attention and emits post-RoPE / post-cache-update
    ``query_states`` and ``key_states_full`` / ``value_states_full``. We
    slice the KV into DCT's ``[sink | paged | recent]`` layout here, score
    every selector against the same neutral Q, and compute the usual
    metrics. No selector alters the decode path, so Q is identical across
    all selectors — a fair comparison.

    Large tensors are NOT retained across calls; only per-record [H_q]
    metric tensors are kept.
    """

    def __init__(
        self,
        num_decode_steps: int,
        page_size: int,
        top_k: int,
        sink_size: int,
        recent_size: int,
        comp_size: int,
        scoring_method: str,
        group_agg_method: str,
        comp_kv_quant: str = "none",
        comp_kv_quant_granularity: str = "per_page",
    ):
        self.num_decode_steps = num_decode_steps
        self.page_size = page_size
        self.top_k = top_k
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.comp_size = comp_size
        self.scoring_method = scoring_method
        self.group_agg_method = group_agg_method
        self.comp_kv_quant = comp_kv_quant
        self.comp_kv_quant_granularity = comp_kv_quant_granularity
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step >= self.num_decode_steps:
            return

        query_states = payload["query_states"]    # [1, H_q, 1, d]
        key_full = payload["key_states_full"]     # [1, H_kv, kv_len, d]
        value_full = payload["value_states_full"] # [1, H_kv, kv_len, d]
        num_kv_groups = int(payload["num_kv_groups"])

        bsz, H_kv, kv_len, d = key_full.shape
        _, H_q, q_len, _ = query_states.shape
        assert bsz == 1 and q_len == 1, f"expected decode step, got {query_states.shape}"
        assert H_q == H_kv * num_kv_groups

        # Segment [sink | paged | recent] exactly like DCT's segment_kv:
        # fixed-length sink, fixed-length recent absorbing the alignment
        # remainder, and the middle carved into whole pages of page_size.
        sink_len = self.sink_size
        if kv_len < sink_len + self.page_size + self.recent_size:
            return  # nothing meaningful to page
        after_sink = kv_len - sink_len
        num_pages = (after_sink - self.recent_size) // self.page_size
        if num_pages < 1:
            return
        actual_top_k = min(self.top_k, num_pages)
        if num_pages <= actual_top_k:
            return  # no sparsification happens when top_k covers every page
        actual_recent = kv_len - sink_len - num_pages * self.page_size

        P = num_pages
        S = self.page_size
        paged_end = sink_len + P * S
        sink_k = key_full[:, :, :sink_len, :]
        sink_v = value_full[:, :, :sink_len, :]
        paged_k = key_full[:, :, sink_len:paged_end, :].view(bsz, H_kv, P, S, d)
        paged_v = value_full[:, :, sink_len:paged_end, :].view(bsz, H_kv, P, S, d)
        recent_k = key_full[:, :, paged_end:, :]
        recent_v = value_full[:, :, paged_end:, :]
        assert recent_k.shape[2] == actual_recent

        with torch.no_grad():
            page_mass_gpu, sink_mass_gpu, recent_mass_gpu = compute_per_page_mass(
                query_states, sink_k, paged_k, recent_k, num_kv_groups,
            )
            page_mass = page_mass_gpu.float().cpu()                            # [H_q, P]
            sink_mass = sink_mass_gpu.float().cpu()                            # [H_q]
            recent_mass = recent_mass_gpu.float().cpu()                        # [H_q]

            proxy_scores_gpu = compute_dct_lowpass_proxy_scores(
                query_states, paged_k, self.comp_size, num_kv_groups,
                self.group_agg_method, self.scoring_method,
                comp_kv_quant=self.comp_kv_quant,
                comp_kv_quant_granularity=self.comp_kv_quant_granularity,
            )
            proxy_scores = proxy_scores_gpu.float().cpu()                      # [H_kv, P]

            quest_scores_gpu = compute_quest_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )
            quest_scores = quest_scores_gpu.float().cpu()

            shadowkv_scores_gpu = compute_shadowkv_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )
            shadowkv_scores = shadowkv_scores_gpu.float().cpu()

            oracle_scores_gpu = compute_oracle_max_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )
            oracle_scores = oracle_scores_gpu.float().cpu()

            # Top-K per selector (all at kv-head granularity).
            proxy_topk_gpu = torch.topk(proxy_scores_gpu, actual_top_k, dim=-1).indices
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

            selected_indices = proxy_topk_gpu.cpu()                            # [H_kv, K]

        mass_metrics = compute_all_metrics(
            page_mass, sink_mass, recent_mass, selected_indices,
            oracle_scores, quest_scores, shadowkv_scores, num_kv_groups,
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
    model,
    tokenizer,
    sample: dict[str, Any],
    *,
    num_decode_steps: int,
    page_size: int,
    top_k: int,
    sink_size: int,
    recent_size: int,
    comp_size: int,
    scoring_method: str,
    group_agg_method: str,
    comp_kv_quant: str,
    comp_kv_quant_granularity: str,
) -> tuple[list[dict[str, Any]], int]:
    """Run generate() with a fresh dense-trajectory recording hook installed."""
    from oracle.attention_mass_recall_ruler_quest import set_recording_hook

    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder = MassRecallRecorder(
        num_decode_steps=num_decode_steps,
        page_size=page_size,
        top_k=top_k,
        sink_size=sink_size,
        recent_size=recent_size,
        comp_size=comp_size,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
    )
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

    return recorder.records, int(input_ids.shape[1])


# ---------------------------------------------------------------------------
# comp_size sweep recorder + generator
# ---------------------------------------------------------------------------
class PagedMassRatioSweepRecorder:
    """Per-decode-step recorder for the comp_size sweep.

    Skips shadowkv / oracle / fidelity to keep per-step cost low. Records
    per-step selected_mass values (no per-step ratios); paged_mass_ratio and
    paged_mass_recall are derived post-hoc as ratio-of-aggregated-means by
    ``_run_comp_size_sweep``:

        paged_mass_ratio_X(c) = mean(selected_mass_X(c)) / mean(selected_mass_mass_topk)
        paged_mass_recall_X(c) = mean(selected_mass_X(c)) / mean(pages_mass)
    """

    def __init__(
        self,
        num_decode_steps: int,
        page_size: int,
        top_k: int,
        sink_size: int,
        recent_size: int,
        comp_sizes: list[int],
        scoring_method: str,
        group_agg_method: str,
    ):
        self.num_decode_steps = num_decode_steps
        self.page_size = page_size
        self.top_k = top_k
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.comp_sizes = list(comp_sizes)
        self.scoring_method = scoring_method
        self.group_agg_method = group_agg_method
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        decode_step = self._step_by_layer.get(layer_idx, 0)
        self._step_by_layer[layer_idx] = decode_step + 1
        if decode_step >= self.num_decode_steps:
            return

        query_states = payload["query_states"]
        key_full = payload["key_states_full"]
        value_full = payload["value_states_full"]
        num_kv_groups = int(payload["num_kv_groups"])

        bsz, H_kv, kv_len, d = key_full.shape
        _, H_q, q_len, _ = query_states.shape
        assert bsz == 1 and q_len == 1
        assert H_q == H_kv * num_kv_groups

        sink_len = self.sink_size
        if kv_len < sink_len + self.page_size + self.recent_size:
            return
        after_sink = kv_len - sink_len
        num_pages = (after_sink - self.recent_size) // self.page_size
        if num_pages < 1:
            return
        actual_top_k = min(self.top_k, num_pages)
        if num_pages <= actual_top_k:
            return

        P = num_pages
        S = self.page_size
        paged_end = sink_len + P * S
        sink_k = key_full[:, :, :sink_len, :]
        paged_k = key_full[:, :, sink_len:paged_end, :].view(bsz, H_kv, P, S, d)
        recent_k = key_full[:, :, paged_end:, :]

        with torch.no_grad():
            page_mass_gpu, _, _ = compute_per_page_mass(
                query_states, sink_k, paged_k, recent_k, num_kv_groups,
            )                                                                  # [H_q, P]

            proxy_sel_gpu = compute_selected_mass_sweep(
                query_states, paged_k, page_mass_gpu,
                top_k=actual_top_k,
                comp_sizes=self.comp_sizes,
                num_kv_groups=num_kv_groups,
                group_agg_method=self.group_agg_method,
                scoring_method=self.scoring_method,
            )
            proxy_selected = {
                c: t.float().cpu().tolist() for c, t in proxy_sel_gpu.items()
            }

            quest_sel_gpu = compute_quest_selected_mass(
                query_states, paged_k, page_mass_gpu,
                top_k=actual_top_k,
                num_kv_groups=num_kv_groups,
                group_agg_method=self.group_agg_method,
            )
            quest_selected = quest_sel_gpu.float().cpu().tolist()

            mass_topk_sel_gpu = compute_mass_topk_selected_mass(
                page_mass_gpu, top_k=actual_top_k,
            )
            mass_topk_selected = mass_topk_sel_gpu.float().cpu().tolist()

            pages_mass = page_mass_gpu.sum(-1).float().cpu().tolist()         # [H_q]

        self.records.append({
            "layer_idx": layer_idx,
            "decode_step": decode_step,
            "num_pages": num_pages,
            "actual_top_k": actual_top_k,
            "num_kv_groups": num_kv_groups,
            "H_q": int(page_mass_gpu.shape[0]),
            "proxy_selected": proxy_selected,    # {comp_size: [H_q] list}
            "quest_selected": quest_selected,    # [H_q] list
            "mass_topk_selected": mass_topk_selected,  # [H_q] list
            "pages_mass": pages_mass,            # [H_q] list
        })


def generate_with_paged_mass_ratio_sweep(
    model,
    tokenizer,
    sample: dict[str, Any],
    *,
    num_decode_steps: int,
    page_size: int,
    top_k: int,
    sink_size: int,
    recent_size: int,
    comp_sizes: list[int],
    scoring_method: str,
    group_agg_method: str,
) -> tuple[list[dict[str, Any]], int]:
    """generate() wrapper that installs a fresh PagedMassRatioSweepRecorder."""
    from oracle.attention_mass_recall_ruler_quest import set_recording_hook

    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder = PagedMassRatioSweepRecorder(
        num_decode_steps=num_decode_steps,
        page_size=page_size,
        top_k=top_k,
        sink_size=sink_size,
        recent_size=recent_size,
        comp_sizes=comp_sizes,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
    )
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

    return recorder.records, int(input_ids.shape[1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Attention-mass recall vs full-attention softmax on RULER. "
            "Dense baseline drives decoding; every selector is scored on "
            "the same neutral Q at each decode step."
        )
    )
    # Model
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    # RULER
    p.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--data_root", type=Path,
                   default=Path("benchmark/data/ruler_data"))

    # Page layout + proxy scoring config (no DCT output path involved).
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125,
                   help="Haar proxy compression ratio; comp_size = "
                        "max(1, int(page_size * compress_ratio)).")
    p.add_argument("--scoring_method", type=str, default="max",
                   choices=["mean", "max", "sum"])
    p.add_argument("--group_agg_method", type=str, default="max",
                   choices=["mean", "max"])

    # Fake-quantize the compressed K proxy (simulates low-precision comp-KV
    # storage). Applied AFTER the DCT projection, BEFORE scoring.
    p.add_argument("--comp_kv_quant", type=str, default="fp8_e4m3",
                   choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    p.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                   choices=["per_page", "per_comp_token"])

    # Analysis
    p.add_argument("--num_decode_steps", type=int, default=20,
                   help="Number of decode steps per sample to record.")

    # comp_size sweep mode (paged_mass_ratio_proxy vs lowpass cutoff,
    # mirroring observations/dct_page_energy.py). When set, the script
    # bypasses the quest/shadowkv/oracle/fidelity selectors and forces
    # comp_kv_quant='none'; output goes to <output_dir>/<run_name>/ with
    # a paged_mass_ratio_curve.png plot.
    p.add_argument("--comp_size_sweep", type=str, nargs="?", const="all",
                   default=None,
                   help="Trigger comp_size sweep mode (measures only "
                        "paged_mass_ratio_proxy and emits a plot). Pass with "
                        "no value (or 'all') to auto-sweep every comp_size "
                        "in 1..page_size; otherwise pass a comma-separated "
                        "list (e.g. '1,2,4,8,16,32').")

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
# comp_size sweep: plot + main
# ---------------------------------------------------------------------------
def _render_comp_size_sweep_plot(
    run_dir: Path,
    per_layer_mean: dict[int, dict[int, float]],
    overall_mean: dict[int, float],
    comp_sizes: list[int],
    page_size: int,
    title: str,
    quest_per_layer_mean: dict[int, float] | None = None,
    quest_overall_mean: float | None = None,
) -> None:
    """One-panel plot: x = comp_size, y = paged_mass_ratio_proxy.

    Bold mean line + ±1σ shaded band across layers for each selector. The
    band shows how consistent layers are without the visual clutter of
    plotting all per-layer curves.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.array(comp_sizes, dtype=np.float64)

    # DCT proxy: ±1σ band across layers per comp_size.
    proxy_layers = np.array(
        [[by_c[c] for c in comp_sizes]
         for _, by_c in sorted(per_layer_mean.items())],
        dtype=np.float64,
    )                                              # [num_layers, num_comp_sizes]
    proxy_std = proxy_layers.std(axis=0)
    ys_mean = np.array([overall_mean[c] for c in comp_sizes], dtype=np.float64)
    ax.fill_between(
        xs,
        np.clip(ys_mean - proxy_std, 0.0, 1.0),
        np.clip(ys_mean + proxy_std, 0.0, 1.0),
        color="C0", alpha=0.18,
        label="DCT proxy ±1σ across layers",
    )
    ax.plot(xs, ys_mean, color="C0", linewidth=2.2, marker="o",
            label="DCT proxy (mean over layers)")

    if quest_per_layer_mean and quest_overall_mean is not None:
        # Quest has no comp_size knob; band is a horizontal stripe at
        # quest_overall_mean ±1σ across layers, drawn full-width so it reads
        # as a reference baseline rather than a function of comp_size.
        quest_arr = np.array(list(quest_per_layer_mean.values()), dtype=np.float64)
        quest_std = float(quest_arr.std())
        ax.axhspan(
            max(quest_overall_mean - quest_std, 0.0),
            min(quest_overall_mean + quest_std, 1.0),
            color="C1", alpha=0.12,
            label=f"Quest ±1σ across layers (σ={quest_std:.3f})",
        )
        ax.axhline(quest_overall_mean, color="C1", linewidth=2.2,
                   linestyle="-",
                   label=f"Quest (mean over layers) = {quest_overall_mean:.3f}")

    ax.axhline(1.0, color="red", alpha=0.3, linestyle="--",
               label="paged-mass ceiling")

    ax.set_xlabel(f"comp_size (lowpass cutoff; page_size={page_size})")
    ax.set_ylabel("paged_mass_ratio_proxy")
    ax.set_title(f"{title}\npaged_mass_ratio_proxy vs comp_size")
    ax.set_ylim(0.0, 1.05)
    # Linear x-axis: comp_size is a token-budget count, so equal x-distance
    # should mean equal added bins. Log scale would compress the high-c jump
    # where most of the gain happens.
    is_dense = len(comp_sizes) == max(comp_sizes) - min(comp_sizes) + 1
    if is_dense and len(comp_sizes) > 12:
        # Many consecutive ticks would crowd the axis; show endpoints +
        # powers of 2 in between.
        tick_set = {comp_sizes[0], comp_sizes[-1]}
        p2 = 1
        while p2 <= comp_sizes[-1]:
            if p2 >= comp_sizes[0]:
                tick_set.add(p2)
            p2 *= 2
        ticks = sorted(tick_set)
    else:
        ticks = comp_sizes
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = run_dir / "paged_mass_ratio_curve.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] {out}")


def _run_comp_size_sweep(args: argparse.Namespace) -> None:
    """Sweep comp_size and emit per-layer / overall paged_mass_ratio + plot."""
    from oracle.attention_mass_recall_ruler_quest import (
        _install_recording_forward,
        _model_family,
        cleanup_model,
        load_model,
    )

    spec = args.comp_size_sweep.strip().lower()
    if spec in ("all", "auto", ""):
        comp_sizes = list(range(1, args.page_size + 1))
        sweep_tag = "all"
    else:
        comp_sizes = sorted({int(c) for c in spec.split(",") if c.strip()})
        sweep_tag = "c" + "-".join(str(c) for c in comp_sizes)
    if not comp_sizes:
        raise ValueError("--comp_size_sweep must expand to at least one value")
    for c in comp_sizes:
        if c < 1 or c > args.page_size:
            raise ValueError(
                f"comp_size {c} out of valid range [1, page_size={args.page_size}]"
            )

    start_time = time.time()
    torch.manual_seed(42)

    run_name = args.run_name or (
        f"mass_ratio_sweep_ps{args.page_size}_topk{args.top_k}_{sweep_tag}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = run_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    config = {**vars(args), "comp_sizes": comp_sizes, "mode": "comp_size_sweep"}
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    print("Installing dense recording forward (no DCT patch)...")
    _install_recording_forward(model, _model_family(args.base_model))
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _, tokenizer_family = infer_model_family(args.base_model)
    task_configs = load_task_configs()

    # Accumulate per-step selected_mass values as flat float lists per
    # (layer, comp_size) and per comp_size overall. Each list element is one
    # (head, decode_step, sample, task) observation. paged_mass_ratio and
    # paged_mass_recall are derived post-hoc as ratio-of-means at print/plot
    # time, NOT averaged from per-step ratios.
    overall_proxy: dict[int, list[float]] = {c: [] for c in comp_sizes}
    overall_quest: list[float] = []
    overall_mass_topk: list[float] = []
    overall_pages_mass: list[float] = []
    per_layer_proxy: dict[int, dict[int, list[float]]] = {}
    per_layer_quest: dict[int, list[float]] = {}
    per_layer_mass_topk: dict[int, list[float]] = {}
    per_layer_pages_mass: dict[int, list[float]] = {}
    per_task_proxy: dict[str, dict[int, list[float]]] = {}
    per_task_quest: dict[str, list[float]] = {}
    per_task_mass_topk: dict[str, list[float]] = {}
    per_task_pages_mass: dict[str, list[float]] = {}

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

            task_proxy: dict[int, list[float]] = {c: [] for c in comp_sizes}
            task_quest: list[float] = []
            task_mass_topk: list[float] = []
            task_pages_mass: list[float] = []
            sample_fp = (per_sample_dir / f"{task}.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )

            for sample_idx, sample in enumerate(
                tqdm(samples, desc=f"  {task}"), start=1
            ):
                records, input_len = generate_with_paged_mass_ratio_sweep(
                    model, tokenizer, sample,
                    num_decode_steps=args.num_decode_steps,
                    page_size=args.page_size,
                    top_k=args.top_k,
                    sink_size=args.sink_size,
                    recent_size=args.recent_size,
                    comp_sizes=comp_sizes,
                    scoring_method=args.scoring_method,
                    group_agg_method=args.group_agg_method,
                )
                if not records:
                    print(f"  WARNING: no traces for sample {sample['index']} "
                          f"(input_len={input_len}); skipping")
                    continue

                sample_proxy: dict[int, dict[int, list[float]]] = {}
                sample_quest_by_layer: dict[int, list[float]] = {}
                sample_mass_topk_by_layer: dict[int, list[float]] = {}
                sample_pages_mass_by_layer: dict[int, list[float]] = {}
                for rec in records:
                    layer_idx = rec["layer_idx"]
                    sample_proxy.setdefault(layer_idx, {c: [] for c in comp_sizes})
                    per_layer_proxy.setdefault(layer_idx, {c: [] for c in comp_sizes})
                    for c in comp_sizes:
                        sm = rec["proxy_selected"][c]
                        sample_proxy[layer_idx][c].extend(sm)
                        per_layer_proxy[layer_idx][c].extend(sm)
                        task_proxy[c].extend(sm)
                        overall_proxy[c].extend(sm)

                    quest_per_head = rec["quest_selected"]
                    sample_quest_by_layer.setdefault(layer_idx, []).extend(quest_per_head)
                    per_layer_quest.setdefault(layer_idx, []).extend(quest_per_head)
                    task_quest.extend(quest_per_head)
                    overall_quest.extend(quest_per_head)

                    mass_topk_per_head = rec["mass_topk_selected"]
                    sample_mass_topk_by_layer.setdefault(layer_idx, []).extend(mass_topk_per_head)
                    per_layer_mass_topk.setdefault(layer_idx, []).extend(mass_topk_per_head)
                    task_mass_topk.extend(mass_topk_per_head)
                    overall_mass_topk.extend(mass_topk_per_head)

                    pages_per_head = rec["pages_mass"]
                    sample_pages_mass_by_layer.setdefault(layer_idx, []).extend(pages_per_head)
                    per_layer_pages_mass.setdefault(layer_idx, []).extend(pages_per_head)
                    task_pages_mass.extend(pages_per_head)
                    overall_pages_mass.extend(pages_per_head)

                # Per-sample summary stores the aggregated selected-mass means
                # plus the derived ratio-of-means at each layer, so the JSONL
                # is self-describing without a recomputation step downstream.
                sample_layer_summary: dict[str, dict[str, Any]] = {}
                for lyr in sorted(sample_proxy.keys()):
                    smm = _mean(sample_mass_topk_by_layer.get(lyr, []))
                    pm = _mean(sample_pages_mass_by_layer.get(lyr, []))
                    qm = _mean(sample_quest_by_layer.get(lyr, []))
                    sample_layer_summary[str(lyr)] = {
                        "selected_mass_proxy": {
                            str(c): _mean(vs) for c, vs in sample_proxy[lyr].items()
                        },
                        "selected_mass_quest": qm,
                        "selected_mass_mass_topk": smm,
                        "pages_mass": pm,
                        "paged_mass_ratio_proxy": {
                            str(c): (_mean(vs) / smm) if smm > 1e-12 else 0.0
                            for c, vs in sample_proxy[lyr].items()
                        },
                        "paged_mass_ratio_quest": (qm / smm) if smm > 1e-12 else 0.0,
                        "paged_mass_recall_proxy": {
                            str(c): (_mean(vs) / pm) if pm > 1e-12 else 0.0
                            for c, vs in sample_proxy[lyr].items()
                        },
                        "paged_mass_recall_quest": (qm / pm) if pm > 1e-12 else 0.0,
                    }
                sample_record = {
                    "sample_index": int(sample["index"]),
                    "input_len": input_len,
                    "num_records": len(records),
                    "comp_sizes": comp_sizes,
                    "per_layer": sample_layer_summary,
                }
                sample_fp.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

                if sample_idx % 5 == 0 or sample_idx == len(samples):
                    smm = _mean(task_mass_topk)
                    bits = " ".join(
                        f"c{c}={(_mean(task_proxy[c]) / smm if smm > 1e-12 else 0):.3f}"
                        for c in comp_sizes
                    )
                    qratio = _mean(task_quest) / smm if smm > 1e-12 else 0.0
                    print(f"  [{sample_idx}/{len(samples)}] {bits} "
                          f"quest={qratio:.3f}")

            sample_fp.close()
            per_task_proxy[task] = task_proxy
            per_task_quest[task] = task_quest
            per_task_mass_topk[task] = task_mass_topk
            per_task_pages_mass[task] = task_pages_mass

            smm_task = _mean(task_mass_topk)
            pm_task = _mean(task_pages_mass)
            print("  TASK SUMMARY")
            for c in comp_sizes:
                mp = _mean(task_proxy[c])
                ratio = mp / smm_task if smm_task > 1e-12 else 0.0
                recall = mp / pm_task if pm_task > 1e-12 else 0.0
                print(f"    comp_size={c:3d}  paged_mass_ratio_proxy = "
                      f"{ratio:.4f}  paged_mass_recall_proxy = {recall:.4f}")
            mq = _mean(task_quest)
            qratio = mq / smm_task if smm_task > 1e-12 else 0.0
            qrecall = mq / pm_task if pm_task > 1e-12 else 0.0
            print(f"    quest        paged_mass_ratio_quest = {qratio:.4f}"
                  f"  paged_mass_recall_quest = {qrecall:.4f}")

        # ----- Derive ratios post-hoc from aggregated selected_mass means ---
        # paged_mass_ratio_X(c) = mean(selected_mass_X(c)) / mean(selected_mass_mass_topk)
        # paged_mass_recall_X(c) = mean(selected_mass_X(c)) / mean(pages_mass)
        smm_overall = _mean(overall_mass_topk)
        pm_overall = _mean(overall_pages_mass)

        def _safe_div(num: float, den: float) -> float:
            return num / den if den > 1e-12 else 0.0

        ratio_per_layer: dict[int, dict[int, float]] = {}
        recall_per_layer: dict[int, dict[int, float]] = {}
        for lyr, by_c in sorted(per_layer_proxy.items()):
            smm = _mean(per_layer_mass_topk.get(lyr, []))
            pm = _mean(per_layer_pages_mass.get(lyr, []))
            ratio_per_layer[lyr] = {
                c: _safe_div(_mean(vs), smm) for c, vs in by_c.items()
            }
            recall_per_layer[lyr] = {
                c: _safe_div(_mean(vs), pm) for c, vs in by_c.items()
            }
        ratio_overall: dict[int, float] = {
            c: _safe_div(_mean(vs), smm_overall) for c, vs in overall_proxy.items()
        }
        recall_overall: dict[int, float] = {
            c: _safe_div(_mean(vs), pm_overall) for c, vs in overall_proxy.items()
        }

        quest_ratio_per_layer: dict[int, float] = {
            lyr: _safe_div(_mean(vs), _mean(per_layer_mass_topk.get(lyr, [])))
            for lyr, vs in sorted(per_layer_quest.items())
        }
        quest_recall_per_layer: dict[int, float] = {
            lyr: _safe_div(_mean(vs), _mean(per_layer_pages_mass.get(lyr, [])))
            for lyr, vs in sorted(per_layer_quest.items())
        }
        quest_ratio_overall = _safe_div(_mean(overall_quest), smm_overall)
        quest_recall_overall = _safe_div(_mean(overall_quest), pm_overall)

        per_task_summary: dict[str, Any] = {}
        for task, by_c in per_task_proxy.items():
            smm_task = _mean(per_task_mass_topk[task])
            pm_task = _mean(per_task_pages_mass[task])
            mq_task = _mean(per_task_quest[task])
            per_task_summary[task] = {
                "num_samples": min(args.num_samples, len(samples)) if args.num_samples > 0 else len(samples),
                "selected_mass_proxy": {str(c): _mean(vs) for c, vs in by_c.items()},
                "selected_mass_quest": mq_task,
                "selected_mass_mass_topk": smm_task,
                "pages_mass": pm_task,
                "paged_mass_ratio_proxy": {
                    str(c): _safe_div(_mean(vs), smm_task) for c, vs in by_c.items()
                },
                "paged_mass_ratio_quest": _safe_div(mq_task, smm_task),
                "paged_mass_recall_proxy": {
                    str(c): _safe_div(_mean(vs), pm_task) for c, vs in by_c.items()
                },
                "paged_mass_recall_quest": _safe_div(mq_task, pm_task),
            }

        summary = {
            "config": config,
            "comp_sizes": comp_sizes,
            "per_task": per_task_summary,
            "overall": {
                "selected_mass_proxy": {
                    str(c): _mean(vs) for c, vs in overall_proxy.items()
                },
                "selected_mass_quest": _mean(overall_quest),
                "selected_mass_mass_topk": smm_overall,
                "pages_mass": pm_overall,
                "paged_mass_ratio_proxy": {str(c): m for c, m in ratio_overall.items()},
                "paged_mass_ratio_quest": quest_ratio_overall,
                "paged_mass_recall_proxy": {str(c): m for c, m in recall_overall.items()},
                "paged_mass_recall_quest": quest_recall_overall,
            },
            "per_layer": {
                str(lyr): {
                    "paged_mass_ratio_proxy": {str(c): m for c, m in ratio_per_layer[lyr].items()},
                    "paged_mass_recall_proxy": {str(c): m for c, m in recall_per_layer[lyr].items()},
                    "paged_mass_ratio_quest": quest_ratio_per_layer.get(lyr, 0.0),
                    "paged_mass_recall_quest": quest_recall_per_layer.get(lyr, 0.0),
                }
                for lyr in sorted(per_layer_proxy.keys())
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        title = f"{_model_family(args.base_model)} @ {args.seq_len} (page_size={args.page_size})"
        _render_comp_size_sweep_plot(
            run_dir, ratio_per_layer, ratio_overall,
            comp_sizes, args.page_size, title,
            quest_per_layer_mean=quest_ratio_per_layer,
            quest_overall_mean=quest_ratio_overall,
        )

        elapsed = (time.time() - start_time) / 60
        print(f"\n{'=' * 60}\nOVERALL RESULTS\n{'=' * 60}")
        for c in comp_sizes:
            print(f"  comp_size={c:3d}  paged_mass_ratio_proxy = {ratio_overall[c]:.4f}"
                  f"  paged_mass_recall_proxy = {recall_overall[c]:.4f}")
        print(f"  quest          paged_mass_ratio_quest = {quest_ratio_overall:.4f}"
              f"  paged_mass_recall_quest = {quest_recall_overall:.4f}")
        print(f"\n  Results: {run_dir}")
        print(f"  Total time: {elapsed:.1f} min")

    finally:
        cleanup_model(model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    from oracle.attention_mass_recall_ruler_quest import (
        _install_recording_forward,
        _model_family,
        cleanup_model,
        load_model,
    )

    args = parse_args()
    if args.comp_size_sweep:
        _run_comp_size_sweep(args)
        return
    start_time = time.time()
    torch.manual_seed(42)

    run_name = args.run_name or (
        f"mass_dense_ps{args.page_size}_topk{args.top_k}"
        f"_cr{args.compress_ratio}"
        f"_{args.comp_kv_quant}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = run_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    print("Installing dense recording forward (no DCT patch)...")
    _install_recording_forward(model, _model_family(args.base_model))
    comp_size = max(1, int(args.page_size * args.compress_ratio))
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
                    model, tokenizer, sample,
                    num_decode_steps=args.num_decode_steps,
                    page_size=args.page_size,
                    top_k=args.top_k,
                    sink_size=args.sink_size,
                    recent_size=args.recent_size,
                    comp_size=comp_size,
                    scoring_method=args.scoring_method,
                    group_agg_method=args.group_agg_method,
                    comp_kv_quant=args.comp_kv_quant,
                    comp_kv_quant_granularity=args.comp_kv_quant_granularity,
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
                    o.update(_derive_paged_metrics(o))
                    print(
                        f"  [{sample_idx}/{len(samples)}] "
                        f"sink={o['mass_recall_sink']:.3f} "
                        f"recent={o['mass_recall_recent']:.3f}  "
                        f"mass[p/q/s/o/c] = "
                        f"{o['mass_recall_proxy']:.3f}/"
                        f"{o['mass_recall_quest']:.3f}/"
                        f"{o['mass_recall_shadowkv']:.3f}/"
                        f"{o['mass_recall_oracle_max']:.3f}/"
                        f"{o['mass_recall_mass_topk']:.3f}  "
                        f"sel[p/q/s/o/c] = "
                        f"{o['selected_mass_proxy']:.3f}/"
                        f"{o['selected_mass_quest']:.3f}/"
                        f"{o['selected_mass_shadowkv']:.3f}/"
                        f"{o['selected_mass_oracle_max']:.3f}/"
                        f"{o['selected_mass_mass_topk']:.3f}  "
                        f"paged[p/q/s/o/c] = "
                        f"{o['paged_mass_recall_proxy']:.3f}/"
                        f"{o['paged_mass_recall_quest']:.3f}/"
                        f"{o['paged_mass_recall_shadowkv']:.3f}/"
                        f"{o['paged_mass_recall_oracle_max']:.3f}/"
                        f"{o['paged_mass_recall_mass_topk']:.3f}  "
                        f"ratio[p/q/s] = "
                        f"{o['paged_mass_ratio_proxy']:.3f}/"
                        f"{o['paged_mass_ratio_quest']:.3f}/"
                        f"{o['paged_mass_ratio_shadowkv']:.3f}  "
                        f"fid[p/q/s/o] = "
                        f"{o['output_fidelity_proxy']:.3f}/"
                        f"{o['output_fidelity_quest']:.3f}/"
                        f"{o['output_fidelity_shadowkv']:.3f}/"
                        f"{o['output_fidelity_oracle_max']:.3f}"
                    )

            sample_fp.close()

            task_overall_agg = _aggregate_metric_dicts(task_overall_records)
            task_overall_agg.update(_derive_paged_metrics(task_overall_agg))
            task_per_layer_agg = {}
            for lyr, bucket in sorted(task_per_layer.items()):
                lm = _aggregate_metric_dicts(bucket)
                lm.update(_derive_paged_metrics(lm))
                task_per_layer_agg[str(lyr)] = lm
            per_task_results[task] = {
                "num_samples": len(samples),
                "overall": task_overall_agg,
                "per_layer": task_per_layer_agg,
            }
            o = per_task_results[task]["overall"]
            print(
                f"  TASK SUMMARY\n"
                f"    sink / recent (floor)                   = "
                f"{o['mass_recall_sink']:.3f} / {o['mass_recall_recent']:.3f}\n"
                f"    mass   [proxy/quest/shadow/oracle/ceil] = "
                f"{o['mass_recall_proxy']:.3f} / {o['mass_recall_quest']:.3f} / "
                f"{o['mass_recall_shadowkv']:.3f} / {o['mass_recall_oracle_max']:.3f} / "
                f"{o['mass_recall_mass_topk']:.3f}\n"
                f"    select [proxy/quest/shadow/oracle/ceil] = "
                f"{o['selected_mass_proxy']:.3f} / {o['selected_mass_quest']:.3f} / "
                f"{o['selected_mass_shadowkv']:.3f} / {o['selected_mass_oracle_max']:.3f} / "
                f"{o['selected_mass_mass_topk']:.3f}\n"
                f"    paged  [proxy/quest/shadow/oracle/ceil] = "
                f"{o['paged_mass_recall_proxy']:.3f} / {o['paged_mass_recall_quest']:.3f} / "
                f"{o['paged_mass_recall_shadowkv']:.3f} / {o['paged_mass_recall_oracle_max']:.3f} / "
                f"{o['paged_mass_recall_mass_topk']:.3f}\n"
                f"    ratio  [proxy/quest/shadow vs ceil]     = "
                f"{o['paged_mass_ratio_proxy']:.3f} / {o['paged_mass_ratio_quest']:.3f} / "
                f"{o['paged_mass_ratio_shadowkv']:.3f}\n"
                f"    fidelity[proxy/quest/shadow/oracle]     = "
                f"{o['output_fidelity_proxy']:.3f} / {o['output_fidelity_quest']:.3f} / "
                f"{o['output_fidelity_shadowkv']:.3f} / {o['output_fidelity_oracle_max']:.3f}\n"
                f"    set_recall = {o['set_recall']:.3f}"
            )

        overall_task_means = [r["overall"] for r in per_task_results.values()]
        overall = _aggregate_metric_dicts(overall_task_means)
        overall.update(_derive_paged_metrics(overall))

        summary = {
            "config": {
                "base_model": args.base_model,
                "trajectory": "dense",
                "seq_len": args.seq_len,
                "num_samples": args.num_samples,
                "num_decode_steps": args.num_decode_steps,
                "page_size": args.page_size,
                "top_k": args.top_k,
                "sink_size": args.sink_size,
                "recent_size": args.recent_size,
                "compress_ratio": args.compress_ratio,
                "comp_size": comp_size,
                "scoring_method": args.scoring_method,
                "group_agg_method": args.group_agg_method,
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
        for k in MASS_METRIC_KEYS + DERIVED_PAGED_KEYS + FIDELITY_METRIC_KEYS:
            print(f"  {k:25s} = {overall[k]:.3f}")
        print(f"\n  Results: {run_dir}")
        print(f"  Total time: {elapsed:.1f} min")

    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
