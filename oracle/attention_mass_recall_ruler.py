#!/usr/bin/env python3
"""
Attention-mass recall on RULER — dense-trajectory reference.

Generation runs under the **unmodified full-KV forward** (no DCT patch, no
selector drives decoding). A recording hook mirrors HF's own attention
forward 1:1 and only observes (Q, K, V) post-RoPE / post-cache-update, so
every selector — DCT Haar proxy, Quest, ShadowKV, InfLLM — is evaluated
against the same neutral Q at each decode step. This removes the
"home-field" bias of scoring Quest on a Q already shaped by DCT's earlier
page choices.

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
  mass_recall_infllm      = sink + recent + Σ m[p] over InfLLM's top-K
  mass_recall_mass_topk   = sink + recent + Σ m[p] over top-K by page mass  (ceiling)
  set_recall              = |DCT ∩ mass_topk| / K                          (page-set vs ceiling)

(B) SELECTED-PAGE MASS metrics (fraction of total softmax mass that lands
    on the selector's chosen pages; denominator = 1):

  selected_mass_proxy       = Σ_{p∈DCT topK} m[p]        = 1 − sink − recent − Σ_{unselected} m[p]
  selected_mass_quest       = Σ_{p∈Quest topK} m[p]
  selected_mass_shadowkv    = Σ_{p∈ShadowKV topK} m[p]
  selected_mass_infllm      = Σ_{p∈InfLLM topK} m[p]
  selected_mass_mass_topk   = Σ_{p∈mass topK} m[p]                         (ceiling)

By construction:
  mass_recall_X   = selected_mass_X + mass_recall_sink + mass_recall_recent
  selected_mass_X + Σ_{p∉ topK} m[p] = 1 − sink − recent

(C) PAGED-ONLY MASS metrics (no sink/recent floor; denominator = total
    paged attention mass Σ_p m[p] = 1 − (sink_mass + recent_mass)):

  paged_mass_recall_proxy       = Σ_{p∈DCT topK} m[p]       / Σ_p m[p]
  paged_mass_recall_quest       = Σ_{p∈Quest topK} m[p]     / Σ_p m[p]
  paged_mass_recall_shadowkv    = Σ_{p∈ShadowKV topK} m[p]  / Σ_p m[p]
  paged_mass_recall_infllm      = Σ_{p∈InfLLM topK} m[p]    / Σ_p m[p]
  paged_mass_recall_mass_topk   = Σ_{p∈mass topK} m[p]      / Σ_p m[p]   (ceiling)
  paged_mass_ratio_proxy        = paged_mass_recall_proxy / paged_mass_recall_mass_topk
  paged_mass_ratio_quest        = paged_mass_recall_quest / paged_mass_recall_mass_topk
  paged_mass_ratio_shadowkv     = paged_mass_recall_shadowkv / paged_mass_recall_mass_topk
  paged_mass_ratio_infllm       = paged_mass_recall_infllm / paged_mass_recall_mass_topk

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
  output_fidelity_infllm     = cos(full_output, InfLLM drop output)

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

InfLLM scoring (Xiao et al., 2024) represents each block by the mean of its
``repr_topk`` representative tokens, scored at decode against current Q:
    qk[h, p, s]    = (q[h] · K[p, s, :]) / √d
    repr_idx       = top-repr_topk(qk[h, p, :])
    block_repr[h,p]= mean_{s ∈ repr_idx} K[p, s, :]
    score[h, p]    = (q[h] · block_repr[h, p]) / √d
Upstream InfLLM picks the representative tokens by an accumulated local-
attention score that this dense-trajectory diagnostic does not maintain;
we substitute the natural Q-aware analogue (per-block top-``repr_topk``
tokens by current Q·K) so InfLLM's *block-representative* scoring rule is
isolated from its stateful prefill bookkeeping. Block layout is the
script's page grid (n_init↔sink_size, n_local↔recent_size, block_size↔
page_size), so the only InfLLM-specific knob is ``--infllm_repr_topk``.

All selectors share the same sink/recent configuration as DCT; only the
page-ranking rule differs.

Each mass metric directly measures the fraction of the full-attention softmax
mass preserved by the corresponding selection (sink + recent are always kept,
so their mass contributes to every selector). Sources of loss:

  1.0 − ceiling   : unavoidable mass loss from budget-K sparsity
  ceiling − X     : selector X's gap vs the true mass-optimal selection
                    (decomposes into selector paradigm + approximation quality)
  proxy vs quest vs shadowkv vs infllm : different proxy families against each other

Reuses the dense recording-forward plumbing from
``attention_mass_recall_ruler_quest.py`` (``_install_recording_forward``,
``set_recording_hook``, ``load_model``). DCT proxy scores are
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


# Module-level caches for calibrated proxy bases, keyed by base_model.
# Populated in main() from CLI flags --pca_M_path / --fasa_idom_path.
_PCA_M_CACHE: dict[str, dict] = {}
_FASA_IDOM_CACHE: dict[str, dict] = {}


MASS_METRIC_KEYS = [
    # Always-kept floor components (each is a fraction of total softmax mass).
    "mass_recall_sink",
    "mass_recall_recent",
    # Total paged-region mass per head: Σ_p m[p] = 1 − sink − recent.
    # Carried through aggregation as the denominator for paged_mass_recall_*.
    "pages_mass",
    # Mass of (sink + selected pages + recent) / full KV — includes always-kept floor.
    "mass_recall_proxy",
    "mass_recall_segmean",
    "mass_recall_pca",
    "mass_recall_fasa",
    "mass_recall_quest",
    "mass_recall_shadowkv",
    "mass_recall_infllm",
    "mass_recall_mass_topk",
    "set_recall",
    "set_recall_proxy",
    "set_recall_segmean",
    "set_recall_pca",
    "set_recall_fasa",
    "set_recall_quest",
    "set_recall_shadowkv",
    "set_recall_infllm",
    # set_recall vs oracle_max top-K (max(q·K) per page → group_agg → softmax-optional).
    # By construction proxy at comp_size = page_size matches this ceiling at 1.0,
    # so this is the "RULER-consistent" set_recall (matches the page set
    # `select_with_oracle_page_scores=True` would choose).
    "set_recall_proxy_vs_oraclemax",
    "set_recall_quest_vs_oraclemax",
    "set_recall_shadowkv_vs_oraclemax",
    "set_recall_infllm_vs_oraclemax",
    "set_recall_masstopk_vs_oraclemax",
    # oracle_max-based mass ceiling. mass_recall_oraclemax = mass that lands on
    # oracle_max's K pages — the *RULER-consistent* "mass captured" ceiling.
    "mass_recall_oraclemax",
    "selected_mass_oraclemax",
    "set_overlap_quest_dct",
    "set_overlap_shadowkv_dct",
    "set_overlap_infllm_dct",
    "set_overlap_quest_shadowkv",
    "selected_mass_union_qd",
    "mass_recall_union_qd",
    "union_qd_disjoint_frac",
    "selected_mass_halfhalf_qd",
    "mass_recall_halfhalf_qd",
    "halfhalf_size_frac",
    "selected_mass_fill_qd",
    "mass_recall_fill_qd",
    # Mass of (selected pages) / (full KV) — absolute fraction of total
    # attention mass that lands on the selector's chosen pages.
    # Equivalently: 1 − sink − recent − Σ_{unselected} m[p].
    # mass_recall_X = selected_mass_X + sink + recent.
    # paged_mass_recall_* and paged_mass_ratio_* are derived post-hoc from
    # these aggregates (see DERIVED_PAGED_KEYS / _derive_paged_metrics).
    "selected_mass_proxy",
    "selected_mass_segmean",
    "selected_mass_pca",
    "selected_mass_fasa",
    "selected_mass_quest",
    "selected_mass_shadowkv",
    "selected_mass_infllm",
    "selected_mass_mass_topk",
    # CMR_α (Critical Mass Recall): Σ_p [page_mass[p]]^α · 1[p∈S] / Σ_p [page_mass[p]]^α.
    # α=1 ≡ paged_mass_recall; α>1 amplifies penalty for missing high-mass pages.
    # top1_* = oracle top-1 hit rate (= α→∞ limit).
    "cmr2_proxy", "cmr2_quest", "cmr2_shadowkv", "cmr2_infllm",
    "cmr4_proxy", "cmr4_quest", "cmr4_shadowkv", "cmr4_infllm",
    "top1_proxy", "top1_quest", "top1_shadowkv", "top1_infllm",
    # min_K(τ) coverage: smallest K such that selector's top-K captures τ fraction
    # of paged-region softmax mass. Threshold-based dual of attn_recall.
    # Lower is better; mass-oracle (mass_topk) gives the theoretical lower bound.
    *[f"min_k_{sel}_at_{tau}"
      for sel in ("proxy", "segmean", "pca", "fasa", "quest", "shadowkv", "infllm", "oraclemax", "mass_topk")
      for tau in (50, 70, 90, 95, 99)],
]

# Paged-only metrics derived from aggregated MASS_METRIC_KEYS. Computed at
# print/summary time (ratio-of-means) instead of per decode step:
#   paged_mass_recall_X = mean(selected_mass_X) / mean(pages_mass)
#   paged_mass_ratio_X  = mean(selected_mass_X) / mean(selected_mass_mass_topk)
DERIVED_PAGED_KEYS = [
    "paged_mass_recall_proxy",
    "paged_mass_recall_segmean",
    "paged_mass_recall_pca",
    "paged_mass_recall_fasa",
    "paged_mass_recall_quest",
    "paged_mass_recall_shadowkv",
    "paged_mass_recall_infllm",
    "paged_mass_recall_mass_topk",
    "paged_mass_recall_oraclemax",
    # Legacy ratio vs mass_topk ceiling.
    "paged_mass_ratio_proxy",
    "paged_mass_ratio_segmean",
    "paged_mass_ratio_pca",
    "paged_mass_ratio_fasa",
    "paged_mass_ratio_quest",
    "paged_mass_ratio_shadowkv",
    "paged_mass_ratio_infllm",
    # Canonical "attention recall" — softmax mass / oracle_max's K-page mass.
    "attention_recall_proxy",
    "attention_recall_segmean",
    "attention_recall_pca",
    "attention_recall_fasa",
    "attention_recall_quest",
    "attention_recall_shadowkv",
    "attention_recall_infllm",
]

FIDELITY_METRIC_KEYS = [
    "output_fidelity_proxy",
    "output_fidelity_quest",
    "output_fidelity_shadowkv",
    "output_fidelity_infllm",
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


_dct_proj_cache: dict[tuple, torch.Tensor] = {}


def _get_dct_lowpass_projection_matrix(
    page_size: int, comp_size: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Return the [comp_size, page_size] DCT → lowpass truncate → IDCT →
    energy-correction projection matrix, matching DCT-Page's default pipeline.

    Built by ``_build_dct_projection_matrix`` in ``dct_page_attention.py``
    (imported lazily). Cached per (shape, device, dtype).
    """
    key = ("lowpass", page_size, comp_size, device, dtype)
    M = _dct_proj_cache.get(key)
    if M is None:
        from dct_page_attention import _build_dct_projection_matrix
        M = _build_dct_projection_matrix(page_size, comp_size, device, dtype)
        _dct_proj_cache[key] = M
    return M


def _get_full_dct_matrix(
    page_size: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Return the [page_size, page_size] orthonormal DCT-II matrix D such that
    D @ x produces the full DCT spectrum of vector x. Cached per (S, device, dtype)."""
    key = ("dct_full", page_size, device, dtype)
    M = _dct_proj_cache.get(key)
    if M is not None:
        return M
    import numpy as np
    from scipy.fft import dct as _scipy_dct
    I_S = np.eye(page_size, dtype=np.float64)
    # scipy_dct treats axis=0 as the signal axis. dct(I, axis=0) yields D where
    # D[b, t] = (DCT_b of e_t) — that's the basis matrix we want for D @ x = DCT(x).
    D = _scipy_dct(I_S, axis=0, norm='ortho')                   # [S, S]
    M = torch.from_numpy(D).to(device=device, dtype=dtype).contiguous()
    _dct_proj_cache[key] = M
    return M


def compute_dct_qaware_adaptive_scores(
    query_states: torch.Tensor,      # [bsz=1, H_q, 1, d]
    paged_k: torch.Tensor,           # [bsz=1, H_kv, P, S, d]
    top_bins: int,                   # # of DCT bins to pick per (page, qo_head)
    num_kv_groups: int,
    group_agg_method: str,
    mode: str = "sum_abs",
    selection_query: torch.Tensor | None = None,  # for "lastq_recon_max"
) -> torch.Tensor:
    """Per-page, per-qo-head adaptive top-N DCT bins by |Q·DCT(K_p)[b]|.

    Two scoring rules:
      - mode="sum_abs" (original): score = Σ top-N |Q·DCT(K_p)[b]|. This is a
        frequency-domain L1-on-top-N statistic — does NOT approximate max(Q·K).
        Needle spikes have spread DCT energy → underrated.
      - mode="recon_max": pick top-N bins per (page, qo_head), keep their SIGNED
        Q·DCT(K_p) values, zero the rest, IDCT to time domain, then take max_t
        of the reconstructed v[t]. This DOES approximate max(Q·K) via a partial
        spectral reconstruction. Identical to oracle_max when N = page_size.

    Returns: [H_kv, P] page scores.

    Note: this is a diagnostic upper bound. It requires the full DCT spectrum
    (cs=page_size storage). A production analog would pre-select bins on K
    alone (without Q), trading scoring fidelity for cs<page_size storage.
    """
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert bsz == 1 and q_len == 1 and H_q == H_kv * num_kv_groups
    G = num_kv_groups
    scale = 1.0 / math.sqrt(d)

    D = _get_full_dct_matrix(S, paged_k.device, paged_k.dtype)           # [S, S]
    # K_dct[..., p, k, :] = Σ_s D[k, s] · K[..., p, s, :]
    K_dct = torch.einsum("ks,nhpsd->nhpkd", D, paged_k)                   # [1, H_kv, P, K_bins, d]
    K_dct_q = K_dct.repeat_interleave(G, dim=1).float()                   # [1, H_q, P, K_bins, d]

    Q = query_states.float()                                              # [1, H_q, 1, d]
    scores_per_bin = torch.einsum(
        "nhqd,nhpkd->nhpk", Q, K_dct_q,
    ) * scale                                                             # [1, H_q, P, K_bins]

    N = max(1, min(int(top_bins), S))
    if mode == "sum_abs":
        topN_abs, _ = scores_per_bin.abs().topk(N, dim=-1)                # [1, H_q, P, N]
        score_q = topN_abs.sum(dim=-1)                                    # [1, H_q, P]
    elif mode == "recon_max":
        # Pick top-N bins by |Q·DCT(K)|, keep signed values, zero the rest,
        # then IDCT back to time-domain. score = max_t reconstructed v[t].
        abs_scores = scores_per_bin.abs()
        _, topN_idx = abs_scores.topk(N, dim=-1)                          # [1, H_q, P, N]
        masked = torch.zeros_like(scores_per_bin)
        masked.scatter_(
            -1, topN_idx, scores_per_bin.gather(-1, topN_idx),
        )                                                                 # [1, H_q, P, K_bins]
        v_recon = torch.einsum("kt,nhpk->nhpt", D.float(), masked)        # [1, H_q, P, S]
        score_q = v_recon.amax(dim=-1)                                    # signed max
    elif mode == "lastq_recon_max":
        # Production-feasible Q-aware approximation: pick top-N bins per page using
        # a FIXED query (first-decode-step Q), score with the CURRENT Q. Bin set
        # committed once at first decode step; subsequent steps use the same bins.
        # Storage stays at cs=N (the cached bin indices are static per (sample, layer)).
        assert selection_query is not None, "lastq_recon_max requires selection_query"
        Q_sel = selection_query.float()                                   # [1, H_q, 1, d]
        bin_scores_sel = torch.einsum(
            "nhqd,nhpkd->nhpk", Q_sel, K_dct_q,
        ) * scale                                                         # [1, H_q, P, K_bins]
        _, topN_idx = bin_scores_sel.abs().topk(N, dim=-1)
        masked = torch.zeros_like(scores_per_bin)
        masked.scatter_(
            -1, topN_idx, scores_per_bin.gather(-1, topN_idx),
        )
        v_recon = torch.einsum("kt,nhpk->nhpt", D.float(), masked)
        score_q = v_recon.amax(dim=-1)
    elif mode == "kaware_recon_max":
        # K-aware: per page, pick top-N bins by ||DCT(K_p)[b, :]||_2 (Q-INDEPENDENT).
        # This is production-feasible: bin selection committed at cache time,
        # bin scores cached per page (cs=N storage), Q queried at decode time.
        # Then: reconstruct v_recon[t] from selected-bin scores, take max_t.
        kdct_l2 = K_dct.float().pow(2).sum(dim=-1).sqrt()                  # [1, H_kv, P, K_bins]
        # Same selection for all qo-heads in the group (per-kv-head decision).
        _, topN_idx_kv = kdct_l2.topk(N, dim=-1)                          # [1, H_kv, P, N]
        topN_idx = topN_idx_kv.repeat_interleave(G, dim=1)                # [1, H_q, P, N]
        masked = torch.zeros_like(scores_per_bin)
        masked.scatter_(
            -1, topN_idx, scores_per_bin.gather(-1, topN_idx),
        )
        v_recon = torch.einsum("kt,nhpk->nhpt", D.float(), masked)
        score_q = v_recon.amax(dim=-1)
    else:
        raise ValueError(f"qaware adaptive: unknown mode={mode!r}")

    score_g = score_q.view(bsz, H_kv, G, P)
    if group_agg_method == "max":
        scores = score_g.amax(dim=2)
    elif group_agg_method == "mean":
        scores = score_g.mean(dim=2)
    elif group_agg_method == "sum":
        scores = score_g.sum(dim=2)
    else:
        raise ValueError(f"qaware adaptive: unsupported group_agg={group_agg_method!r}")
    return scores.squeeze(0)                                              # [H_kv, P]


def _get_dct_bandpass_projection_matrix(
    page_size: int, bins: tuple[int, ...],
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """[len(bins), page_size] DCT → keep specified bin indices → IDCT-in-cs-bin-space
    → energy-correction projection matrix. Generalizes lowpass to arbitrary bin
    sets (e.g., spread bands [0, 8, 16, 24] instead of [0, 1, 2, 3]).

    Math: M = IDCT_cs · select_bins · DCT_S · sqrt(cs / S), where DCT_S is the
    page_size-point DCT matrix, select_bins picks the cs specified DCT rows, and
    IDCT_cs is the cs-point IDCT applied to that compressed spectrum. This treats
    the kept bins as if they were the cs lowest bins of a cs-point DCT.

    bins must contain unique values in [0, page_size); order doesn't affect math
    but is preserved for downstream slot mapping.
    """
    bins = tuple(int(b) for b in bins)
    assert len(set(bins)) == len(bins), f"bandpass bins must be unique: {bins}"
    assert all(0 <= b < page_size for b in bins), f"bin out of range: {bins} vs page_size={page_size}"
    cs = len(bins)
    key = ("bandpass", page_size, bins, device, dtype)
    M = _dct_proj_cache.get(key)
    if M is not None:
        return M

    import numpy as np
    from scipy.fft import dct as _scipy_dct, idct as _scipy_idct
    I_S = np.eye(page_size, dtype=np.float64)
    D_full = _scipy_dct(I_S, axis=0, norm='ortho')                  # [S, S]
    D_band = D_full[list(bins), :]                                   # [cs, S]
    I_cs = np.eye(cs, dtype=np.float64)
    M_idct_cs = _scipy_idct(I_cs, axis=0, norm='ortho')              # [cs, cs]
    M_np = (M_idct_cs @ D_band) * math.sqrt(cs / page_size)          # [cs, S]
    M = torch.from_numpy(M_np).to(device=device, dtype=dtype).contiguous()
    _dct_proj_cache[key] = M
    return M


def _get_segmean_projection_matrix(
    page_size: int, comp_size: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Segment-mean projection: split page_size into comp_size contiguous segments,
    take per-segment mean. M[c, t] = 1/seg_size if t in segment c else 0.

    Equivalent to Haar lowpass without detail rows (cf. dct_page_attention.py's
    `_build_haar_projection_matrix(n_detail_per_block=0)`). Serves as the
    simplest cs-budget baseline against DCT lowpass-IDCT.
    """
    assert page_size % comp_size == 0, f"page_size={page_size} not divisible by comp_size={comp_size}"
    seg_size = page_size // comp_size
    key = ("segmean", page_size, comp_size, device, dtype)
    M = _dct_proj_cache.get(key)
    if M is not None:
        return M
    M_np = torch.zeros(comp_size, page_size, dtype=torch.float64)
    for c in range(comp_size):
        M_np[c, c * seg_size : (c + 1) * seg_size] = 1.0 / seg_size
    M = M_np.to(device=device, dtype=dtype).contiguous()
    _dct_proj_cache[key] = M
    return M


def compute_segmean_proxy_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    comp_size: int,
    num_kv_groups: int,
    group_agg_method: str,
    scoring_method: str,
) -> torch.Tensor:
    """Per-page proxy via segment-mean compression of K. Mirrors
    compute_dct_lowpass_proxy_scores but uses block-mean projection (Haar lowpass
    without detail). Returns [H_kv, P] page scores."""
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert bsz == 1 and q_len == 1 and H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)
    M = _get_segmean_projection_matrix(S, comp_size, paged_k.device, paged_k.dtype)
    comp_k = torch.einsum("bhpsd,cs->bhpcd", paged_k, M)
    comp_k_q = comp_k.repeat_interleave(num_kv_groups, dim=1).float()
    q = query_states.float()
    scores_per_comp = torch.einsum("bhqd,bhpcd->bhpc", q, comp_k_q) * scale
    if scoring_method == "max":
        score_q = scores_per_comp.amax(dim=-1)
    elif scoring_method == "mean":
        score_q = scores_per_comp.mean(dim=-1)
    elif scoring_method == "sum":
        score_q = scores_per_comp.sum(dim=-1)
    elif scoring_method == "lse":
        score_q = torch.logsumexp(scores_per_comp, dim=-1)
    else:
        raise ValueError(f"segmean: unsupported scoring={scoring_method!r}")
    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.amax(dim=2)
    elif group_agg_method == "mean":
        scores = score_g.mean(dim=2)
    else:
        raise ValueError(f"segmean: unsupported group_agg={group_agg_method!r}")
    return scores.squeeze(0)


def compute_pca_proxy_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    cs_h: int,
    num_kv_groups: int,
    group_agg_method: str,
    scoring_method: str,
    pca_M_layer: torch.Tensor,
) -> torch.Tensor:
    """Per-page proxy via dense PCA projection along the head_dim axis.

    Given per-(layer, kv_head) PCA basis M [H_kv, cs_h_max, head_dim], project
    K to a cs_h-dim subspace and score in projection space. PRESERVES per-token
    resolution (max over page_size axis) — only the head_dim is compressed.

        comp_K[h, p, t, c] = Σ_d M[h, c, d] · K[h, p, t, d]
        proj_q[h, c]       = Σ_d M[h, c, d] · q[h, d]
        score(h, p, t)     = Σ_c proj_q[h, c] · comp_K[h, p, t, c]
        page_score(h, p)   = max_t score(h, p, t)  (or scoring_method)
        kv_score(h_kv, p)  = group_agg over q-heads

    Returns [H_kv, P] page scores.
    """
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert bsz == 1 and q_len == 1
    assert H_q == H_kv * num_kv_groups
    H_basis = pca_M_layer.shape[0]
    assert H_basis in (H_kv, H_q), (
        f"PCA basis H={H_basis} matches neither H_kv={H_kv} nor H_q={H_q}"
    )
    assert pca_M_layer.shape[-1] == d
    scale = 1.0 / math.sqrt(d)

    # Slice top cs_h rows of stored basis.
    M = pca_M_layer[:, :cs_h, :].to(paged_k.device).to(paged_k.dtype)        # [H_basis, cs_h, d]

    if H_basis == H_kv:
        # Per-kv-head basis (query-blind PCA).
        comp_K = torch.einsum("bhpsd,hcd->bhpsc", paged_k.float(), M.float())
        M_q = M.repeat_interleave(num_kv_groups, dim=0).float()              # [H_q, cs_h, d]
        proj_q = torch.einsum("bhqd,hcd->bhqc", query_states.float(), M_q)   # [1, H_q, 1, cs_h]
        comp_K_q = comp_K.repeat_interleave(num_kv_groups, dim=1)            # [1, H_q, P, S, cs_h]
    else:
        # Per-q-head basis (query-aware D1). K stored per-kv-head; we need
        # to project K through per-q-head bases.
        kv_idx = torch.arange(H_q, device=paged_k.device) // num_kv_groups
        # Expand K to per-q-head view by indexing rather than repeat (cheaper).
        # paged_k [1, H_kv, P, S, d] -> [1, H_q, P, S, d] via index_select on H_kv axis.
        K_q_view = paged_k.index_select(1, kv_idx)                           # [1, H_q, P, S, d]
        M_q = M.float()                                                      # [H_q, cs_h, d]
        comp_K_q = torch.einsum("bhpsd,hcd->bhpsc", K_q_view.float(), M_q)   # [1, H_q, P, S, cs_h]
        proj_q = torch.einsum("bhqd,hcd->bhqc", query_states.float(), M_q)   # [1, H_q, 1, cs_h]

    scores_per_token = torch.einsum(
        "bhqc,bhpsc->bhps", proj_q, comp_K_q,
    ) * scale                                                                # [1, H_q, P, S]

    if scoring_method == "max":
        score_q = scores_per_token.amax(dim=-1)
    elif scoring_method == "mean":
        score_q = scores_per_token.mean(dim=-1)
    elif scoring_method == "lse":
        score_q = torch.logsumexp(scores_per_token, dim=-1)
    else:
        raise ValueError(f"pca: unsupported scoring={scoring_method!r}")

    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.amax(dim=2)
    elif group_agg_method == "mean":
        scores = score_g.mean(dim=2)
    else:
        raise ValueError(f"pca: unsupported group_agg={group_agg_method!r}")
    return scores.squeeze(0)                                                 # [H_kv, P]


def compute_fasa_fc_proxy_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    n_tip: int,
    num_kv_groups: int,
    group_agg_method: str,
    scoring_method: str,
    fasa_idom_layer: torch.Tensor,
) -> torch.Tensor:
    """Per-page proxy via FASA dominant-FC channel subset.

    Given per-(layer, q_head) dominant FC indices I_dom [H_q, n_tip_max], gather
    channels (2i, 2i+1) for each i ∈ I_dom[h] and score:

        score(h, p, t) = Σ_{i ∈ I_dom[h]} q[h, 2i:2i+2] · K[kv(h), p, t, 2i:2i+2]
        page_score(h, p) = max_t score(h, p, t)

    Returns [H_kv, P] page scores.
    """
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert bsz == 1 and q_len == 1
    assert H_q == H_kv * num_kv_groups
    nFC = d // 2
    assert fasa_idom_layer.shape[0] == H_q
    scale = 1.0 / math.sqrt(d)

    # Build [H_q, n_tip*2] channel index tensor.
    idom = fasa_idom_layer[:, :n_tip].to(paged_k.device).long()              # [H_q, n_tip]
    channels = torch.stack([2 * idom, 2 * idom + 1], dim=-1).view(H_q, n_tip * 2)
    # channels[h, :] are the dominant channel indices for q-head h.

    # Gather q over channels: q[1, H_q, 1, d] → q_sel[1, H_q, 1, n_tip*2]
    ch_q = channels.view(1, H_q, 1, n_tip * 2)
    q_sel = torch.gather(query_states, dim=-1, index=ch_q)                   # [1, H_q, 1, n_tip*2]

    # Expand K to per-q-head view, then gather channels.
    K_q = paged_k.repeat_interleave(num_kv_groups, dim=1)                    # [1, H_q, P, S, d]
    ch_K = channels.view(1, H_q, 1, 1, n_tip * 2).expand(bsz, H_q, P, S, n_tip * 2)
    K_sel = torch.gather(K_q, dim=-1, index=ch_K)                            # [1, H_q, P, S, n_tip*2]

    # Score per token: sum over selected channels of q*K. Convert to fp32.
    scores_per_token = torch.einsum(
        "bhqc,bhpsc->bhps", q_sel.float(), K_sel.float(),
    ) * scale                                                                # [1, H_q, P, S]

    if scoring_method == "max":
        score_q = scores_per_token.amax(dim=-1)
    elif scoring_method == "mean":
        score_q = scores_per_token.mean(dim=-1)
    elif scoring_method == "lse":
        score_q = torch.logsumexp(scores_per_token, dim=-1)
    else:
        raise ValueError(f"fasa: unsupported scoring={scoring_method!r}")

    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.amax(dim=2)
    elif group_agg_method == "mean":
        scores = score_g.mean(dim=2)
    else:
        raise ValueError(f"fasa: unsupported group_agg={group_agg_method!r}")
    return scores.squeeze(0)                                                 # [H_kv, P]


def compute_dct_lowpass_proxy_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    comp_size: int,
    num_kv_groups: int,
    group_agg_method: str,
    scoring_method: str,
    comp_kv_quant: str = "none",
    comp_kv_quant_granularity: str = "per_page",
    softmax_before_group: bool = False,
    dct_bins: tuple[int, ...] | None = None,
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

    if dct_bins is not None:
        assert len(dct_bins) == comp_size, (
            f"dct_bins length {len(dct_bins)} != comp_size {comp_size}"
        )
        M = _get_dct_bandpass_projection_matrix(
            S, tuple(dct_bins), paged_k.device, paged_k.dtype,
        )                                                                     # [C, S]
    else:
        M = _get_dct_lowpass_projection_matrix(
            S, comp_size, paged_k.device, paged_k.dtype,
        )                                                                     # [C, S]
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
    elif scoring_method == "lse":
        # log-sum-exp over compressed positions — approximates LSE(q·K) per
        # page = mass-oracle ranking, while preserving spike sensitivity via
        # the max-dominant term. mk3 ceiling ≤ mass_topk RULER (=76 at 32K).
        score_q = torch.logsumexp(scores_per_comp, dim=-1)
    else:
        raise ValueError(f"Unsupported scoring_method: {scoring_method!r}")

    if softmax_before_group:
        # ShadowKV-style: per qo-head, softmax over pages to normalize each head's
        # page distribution onto a common scale before the GQA group reduction.
        # This re-weights so max-over-group picks the page that is *relatively*
        # peaked for any qo-head, instead of the page where the highest-magnitude
        # qo-head dominates.
        score_q = torch.softmax(score_q, dim=-1)                               # [1, H_q, P]
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


def compute_infllm_scores(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    num_kv_groups: int,
    repr_topk: int,
    group_agg_method: str,
) -> torch.Tensor:
    """InfLLM block-representative page scoring (Xiao et al., 2024).

    InfLLM splits KV into fixed-size blocks; each block is represented by
    the mean of its ``repr_topk`` "representative" tokens. Upstream picks
    representatives by an accumulated local-attention score that this
    dense-trajectory diagnostic does not maintain, so we substitute the
    natural Q-aware analogue: per (query head, page), pick the top-
    ``repr_topk`` tokens by current Q·K (apples-to-apples with Quest's own
    use of current Q for its upper bound).

        for each (h, p):
            qk[h, p, s]    = (q[h] · K[p, s, :]) / √d
            repr_idx       = top-repr_topk(qk[h, p, :])
            block_repr[h,p]= mean_{s ∈ repr_idx} K[p, s, :]
            score[h, p]    = (q[h] · block_repr[h, p]) / √d

    Reduce across the GQA group via ``group_agg_method``; ``mean`` matches
    InfLLM's ``.mean(dim=1)`` over unit_size in
    ``ContextManager.get_batched_topk`` (context_manager.py L597-602).

    Args:
        query_states: [bsz=1, H_q, 1, d] — post-RoPE / post-QK-norm.
        paged_k:      [bsz=1, H_kv, P, S, d] — post-RoPE, baked in from cache.
        num_kv_groups: H_q // H_kv.
        repr_topk:    Representative tokens per page (clamped to S).
        group_agg_method: "mean" | "max" | "topp" (topp falls back to mean).

    Returns:
        scores: [H_kv, P] — one InfLLM score per (kv-head, page).
    """
    bsz, H_q, q_len, d = query_states.shape
    assert bsz == 1 and q_len == 1, f"decode-step only, got shape {query_states.shape}"
    _, H_kv, P, S, _ = paged_k.shape
    assert H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)
    actual_repr = min(repr_topk, S)

    k_q = paged_k.repeat_interleave(num_kv_groups, dim=1).float()    # [1, H_q, P, S, d]
    q = query_states.float()                                         # [1, H_q, 1, d]

    qk = torch.einsum("bhqd,bhpsd->bhps", q, k_q) * scale            # [1, H_q, P, S]
    repr_idx = qk.topk(actual_repr, dim=-1).indices                  # [1, H_q, P, R]
    repr_idx_exp = repr_idx[..., None].expand(*repr_idx.shape, d)
    repr_k = torch.gather(k_q, 3, repr_idx_exp)                      # [1, H_q, P, R, d]
    block_repr = repr_k.mean(dim=-2)                                 # [1, H_q, P, d]
    score_q = torch.einsum("bhqd,bhpd->bhp", q, block_repr) * scale  # [1, H_q, P]

    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.max(dim=2).values
    else:
        scores = score_g.mean(dim=2)
    return scores.squeeze(0)                                         # [H_kv, P]


# ---------------------------------------------------------------------------
# Attention-output fidelity (V-aware downstream quality)
# ---------------------------------------------------------------------------
def _compute_prefill_rm(
    q_prefill: torch.Tensor,   # [1, H_q, seq_len, d] post-RoPE / post-q_norm
    k_prefill: torch.Tensor,   # [1, H_kv, seq_len, d] post-RoPE / post-k_norm
    num_kv_groups: int,
    l_L: int,
) -> torch.Tensor:
    """Paper-faithful InfLLM representative-token significance r_m (Xiao 2024, Eq. 1).

        r_m = (1/l_L) Σ_{j=1..l_L} q_{m+j} · k_m

    The score for token m is the average attention RECEIVED from the local
    window of the next l_L tokens. Computed once at prefill — fixed across
    the entire decode trajectory.

    GQA: r_m is computed per qo-head, then reduced via mean across the GQA
    group (matches InfLLM's ContextManager.get_batched_topk .mean(dim=1)).

    Returns:
        r_m_kv: [H_kv, seq_len] — significance score per (kv-head, token).
    """
    bsz, H_q, seq_len, d = q_prefill.shape
    H_kv = k_prefill.shape[1]
    assert H_q == H_kv * num_kv_groups
    pad_dtype = torch.float32  # accumulate in fp32 to avoid bf16 cumsum drift

    # Q_avg[m] = (1/l_L) Σ_{j=1..l_L} q[m+j] via cumsum on a right-zero-padded Q.
    # Tokens m near seq_len - l_L have shorter valid windows; padding makes their
    # r_m slightly under-estimated, but those tokens are typically in the recent
    # region (excluded from pageable selection) so the effect is moot.
    pad = torch.zeros((bsz, H_q, l_L, d), device=q_prefill.device, dtype=pad_dtype)
    q_padded = torch.cat([q_prefill.to(pad_dtype), pad], dim=2)         # [1, H_q, seq+l_L, d]
    cumsum_q = q_padded.cumsum(dim=2)
    q_avg = (cumsum_q[:, :, l_L : l_L + seq_len, :]
             - cumsum_q[:, :, :seq_len, :]) / float(l_L)                # [1, H_q, seq, d]

    k_q = k_prefill.to(pad_dtype).repeat_interleave(num_kv_groups, dim=1)  # [1, H_q, seq, d]
    r_m_qo = (q_avg * k_q).sum(-1)                                      # [1, H_q, seq]
    r_m_kv = r_m_qo.view(bsz, H_kv, num_kv_groups, seq_len).mean(dim=2).squeeze(0)
    return r_m_kv                                                       # [H_kv, seq_len]


def compute_infllm_paper_scores(
    query_states: torch.Tensor,     # [1, H_q, 1, d]
    paged_k: torch.Tensor,          # [1, H_kv, P, S, d]
    repr_indices: torch.Tensor,     # [H_kv, P, R] long, values in [0, S)
    num_kv_groups: int,
    group_agg_method: str,
) -> torch.Tensor:
    """Paper-faithful InfLLM block scoring (Xiao 2024, Eq. 2).

        sim(X, B) = Σ_{j=1..r_k} q · k_{b_j}^B

    Representatives `repr_indices` are FIXED across the decode trajectory
    (selected once at prefill via local-window-attention significance, see
    `_compute_prefill_rm`). Decode just dots current Q against these fixed
    representative K vectors and sums.

    Returns:
        scores: [H_kv, P]
    """
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert bsz == 1 and q_len == 1, f"decode-step only, got shape {query_states.shape}"
    assert H_q == H_kv * num_kv_groups
    R = repr_indices.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # Gather K[h_kv, p, repr_indices[h_kv, p, r]] for each (h_kv, p, r).
    idx_exp = repr_indices.unsqueeze(0).unsqueeze(-1).expand(bsz, H_kv, P, R, d).long()
    K_repr = paged_k.gather(-2, idx_exp)                                    # [1, H_kv, P, R, d]

    K_repr_q = K_repr.repeat_interleave(num_kv_groups, dim=1).float()       # [1, H_q, P, R, d]
    q = query_states.float()
    qk = torch.einsum("bhqd, bhprd -> bhpr", q, K_repr_q) * scale           # [1, H_q, P, R]
    block_score = qk.sum(-1)                                                # [1, H_q, P]

    score_g = block_score.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        scores = score_g.max(dim=2).values
    else:
        scores = score_g.mean(dim=2)
    return scores.squeeze(0)                                                # [H_kv, P]


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
    quest_scores: torch.Tensor,       # [H_kv, P]
    shadowkv_scores: torch.Tensor,    # [H_kv, P]
    infllm_scores: torch.Tensor,      # [H_kv, P]
    num_kv_groups: int,
    oracle_max_scores: torch.Tensor | None = None,  # [H_kv, P] — same pipeline as DCT proxy with comp_size=page_size (identity proxy = oracle_max).
    proxy_scores: torch.Tensor | None = None,  # [H_kv, P] — DCT lowpass scores, needed for min_K coverage metric.
    segmean_scores: torch.Tensor | None = None,  # [H_kv, P] — segment-mean baseline scores.
    pca_scores: torch.Tensor | None = None,  # [H_kv, P] — head-dim PCA dense-projection proxy.
    fasa_scores: torch.Tensor | None = None,  # [H_kv, P] — FASA dominant-FC channel-subset proxy.
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
    assert infllm_scores.shape == (H_kv, P), (
        f"infllm_scores shape {infllm_scores.shape} != ({H_kv}, {P})"
    )

    page_mass = page_mass.float()
    sink_mass = sink_mass.float()
    recent_mass = recent_mass.float()
    extra_mass = sink_mass + recent_mass                                       # [H_q]
    quest_scores = quest_scores.float()
    shadowkv_scores = shadowkv_scores.float()
    infllm_scores = infllm_scores.float()
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

    # (3b) mass_recall_infllm: sink + recent + InfLLM's top-K.
    infllm_topk = torch.topk(infllm_scores, K, dim=-1).indices                 # [H_kv, K]
    infllm_topk_q = infllm_topk.repeat_interleave(num_kv_groups, dim=0)        # [H_q, K]
    mass_recall_infllm = (
        torch.gather(page_mass, -1, infllm_topk_q).sum(-1) + extra_mass
    )

    # (3c) mass_recall_segmean: sink + recent + SegmentMean's top-K (baseline).
    if segmean_scores is not None:
        segmean_topk = torch.topk(segmean_scores.float(), K, dim=-1).indices    # [H_kv, K]
        segmean_topk_q = segmean_topk.repeat_interleave(num_kv_groups, dim=0)   # [H_q, K]
        mass_recall_segmean = (
            torch.gather(page_mass, -1, segmean_topk_q).sum(-1) + extra_mass
        )
        selected_mass_segmean = mass_recall_segmean - extra_mass
    else:
        segmean_topk_q = quest_topk_q.clone()  # dummy
        mass_recall_segmean = torch.zeros_like(mass_recall_quest)
        selected_mass_segmean = torch.zeros_like(mass_recall_quest)

    # (3d) mass_recall_pca / mass_recall_fasa: dense head-dim PCA & FASA-FC.
    if pca_scores is not None:
        pca_topk = torch.topk(pca_scores.float(), K, dim=-1).indices
        pca_topk_q = pca_topk.repeat_interleave(num_kv_groups, dim=0)
        mass_recall_pca = torch.gather(page_mass, -1, pca_topk_q).sum(-1) + extra_mass
        selected_mass_pca = mass_recall_pca - extra_mass
    else:
        pca_topk_q = quest_topk_q.clone()
        mass_recall_pca = torch.zeros_like(mass_recall_quest)
        selected_mass_pca = torch.zeros_like(mass_recall_quest)
    if fasa_scores is not None:
        fasa_topk = torch.topk(fasa_scores.float(), K, dim=-1).indices
        fasa_topk_q = fasa_topk.repeat_interleave(num_kv_groups, dim=0)
        mass_recall_fasa = torch.gather(page_mass, -1, fasa_topk_q).sum(-1) + extra_mass
        selected_mass_fasa = mass_recall_fasa - extra_mass
    else:
        fasa_topk_q = quest_topk_q.clone()
        mass_recall_fasa = torch.zeros_like(mass_recall_quest)
        selected_mass_fasa = torch.zeros_like(mass_recall_quest)

    # (4) mass_recall_mass_topk: sink + recent + best-K pages by mass (ceiling).
    mass_topk_idx = torch.topk(page_mass, K, dim=-1).indices                   # [H_q, K]
    mass_recall_mass_topk = (
        torch.gather(page_mass, -1, mass_topk_idx).sum(-1) + extra_mass
    )

    # Ceiling must dominate all selector metrics.
    tol = 1e-5
    if not (mass_recall_mass_topk + tol >= mass_recall_proxy).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_proxy — ceiling violated")
    if not (mass_recall_mass_topk + tol >= mass_recall_quest).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_quest — ceiling violated")
    if not (mass_recall_mass_topk + tol >= mass_recall_shadowkv).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_shadowkv — ceiling violated")
    if not (mass_recall_mass_topk + tol >= mass_recall_infllm).all():
        raise AssertionError("mass_recall_mass_topk < mass_recall_infllm — ceiling violated")

    # (5) set_recall vs ceiling (mass_topk): per-query-head |selector ∩ mass_topk| / K.
    # mass_topk_idx is already at [H_q, K]; expand each selector's [H_kv, K] to
    # query-head space so every row compares its K pages against that row's true
    # mass-optimal set.
    mass_topk_mask = _indices_to_mask(mass_topk_idx, P)                        # [H_q, P]
    sel_q_mask = _indices_to_mask(sel_q, P)                                    # [H_q, P]
    quest_topk_q_mask = _indices_to_mask(quest_topk_q, P)                      # [H_q, P]
    shadowkv_topk_q_mask = _indices_to_mask(shadowkv_topk_q, P)                # [H_q, P]
    infllm_topk_q_mask = _indices_to_mask(infllm_topk_q, P)                    # [H_q, P]
    set_recall_proxy = (sel_q_mask & mass_topk_mask).sum(-1).float() / float(K)
    set_recall_quest = (quest_topk_q_mask & mass_topk_mask).sum(-1).float() / float(K)
    set_recall_shadowkv = (shadowkv_topk_q_mask & mass_topk_mask).sum(-1).float() / float(K)
    set_recall_infllm = (infllm_topk_q_mask & mass_topk_mask).sum(-1).float() / float(K)
    if segmean_scores is not None:
        segmean_topk_q_mask = _indices_to_mask(segmean_topk_q, P)              # [H_q, P]
        set_recall_segmean = (segmean_topk_q_mask & mass_topk_mask).sum(-1).float() / float(K)
    else:
        segmean_topk_q_mask = torch.zeros_like(quest_topk_q_mask)
        set_recall_segmean = torch.zeros_like(set_recall_quest)
    if pca_scores is not None:
        pca_topk_q_mask = _indices_to_mask(pca_topk_q, P)
        set_recall_pca = (pca_topk_q_mask & mass_topk_mask).sum(-1).float() / float(K)
    else:
        pca_topk_q_mask = torch.zeros_like(quest_topk_q_mask)
        set_recall_pca = torch.zeros_like(set_recall_quest)
    if fasa_scores is not None:
        fasa_topk_q_mask = _indices_to_mask(fasa_topk_q, P)
        set_recall_fasa = (fasa_topk_q_mask & mass_topk_mask).sum(-1).float() / float(K)
    else:
        fasa_topk_q_mask = torch.zeros_like(quest_topk_q_mask)
        set_recall_fasa = torch.zeros_like(set_recall_quest)

    # (5b) set_recall vs oracle_max top-K. oracle_max is the page ordering used by
    # `--select_with_oracle_page_scores`: same pipeline as the DCT proxy but with
    # the identity projection (comp_size = page_size). This is the "RULER-consistent"
    # ceiling for any proxy selector.
    if oracle_max_scores is not None:
        oraclemax_topk = torch.topk(oracle_max_scores.float(), K, dim=-1).indices    # [H_kv, K]
        oraclemax_topk_q = oraclemax_topk.repeat_interleave(num_kv_groups, dim=0)    # [H_q, K]
        oraclemax_mask = _indices_to_mask(oraclemax_topk_q, P)                       # [H_q, P]
        set_recall_proxy_vs_oraclemax = (sel_q_mask & oraclemax_mask).sum(-1).float() / float(K)
        set_recall_quest_vs_oraclemax = (quest_topk_q_mask & oraclemax_mask).sum(-1).float() / float(K)
        set_recall_shadowkv_vs_oraclemax = (shadowkv_topk_q_mask & oraclemax_mask).sum(-1).float() / float(K)
        set_recall_infllm_vs_oraclemax = (infllm_topk_q_mask & oraclemax_mask).sum(-1).float() / float(K)
        set_recall_masstopk_vs_oraclemax = (mass_topk_mask & oraclemax_mask).sum(-1).float() / float(K)

        # ----- Attention recall vs oracle_max ceiling --------------------------
        # `selected_mass_oraclemax` = softmax mass that lands on oracle_max's K pages.
        # This is the new canonical ceiling for "attention mass captured", replacing
        # mass_topk (which selects pages that maximize captured mass, a different
        # optimization target than RULER actually rewards — see oracle_max vs
        # mass-oracle RULER comparison: mk3 80 vs 76).
        selected_mass_oraclemax = torch.gather(page_mass, -1, oraclemax_topk_q).sum(-1)  # [H_q]
        mass_recall_oraclemax = selected_mass_oraclemax + extra_mass                     # [H_q]

        # ----- Rank-resolved hit profile vs oracle_max rank --------------------
        # Like rank_hit_*_per_r but ordering by oracle_max rank (max(q·K) descending)
        # instead of softmax-mass rank. RULER-consistent — these reveal which oracle
        # ranks each selector recovers, against the oracle that actually predicts
        # RULER performance.
        rank_hit_proxy_per_r_vs_oraclemax = sel_q_mask.gather(-1, oraclemax_topk_q).float()      # [H_q, K]
        rank_hit_quest_per_r_vs_oraclemax = quest_topk_q_mask.gather(-1, oraclemax_topk_q).float()
        rank_hit_shadowkv_per_r_vs_oraclemax = shadowkv_topk_q_mask.gather(-1, oraclemax_topk_q).float()
        rank_hit_infllm_per_r_vs_oraclemax = infllm_topk_q_mask.gather(-1, oraclemax_topk_q).float()
    else:
        zeros = torch.zeros(H_q, dtype=torch.float32, device=page_mass.device)
        set_recall_proxy_vs_oraclemax = zeros
        set_recall_quest_vs_oraclemax = zeros
        set_recall_shadowkv_vs_oraclemax = zeros
        set_recall_infllm_vs_oraclemax = zeros
        set_recall_masstopk_vs_oraclemax = zeros
        selected_mass_oraclemax = zeros
        mass_recall_oraclemax = zeros
        zeros_2d = torch.zeros(H_q, K, dtype=torch.float32, device=page_mass.device)
        rank_hit_proxy_per_r_vs_oraclemax = zeros_2d
        rank_hit_quest_per_r_vs_oraclemax = zeros_2d
        rank_hit_shadowkv_per_r_vs_oraclemax = zeros_2d
        rank_hit_infllm_per_r_vs_oraclemax = zeros_2d

    # Canonical alias: mass_topk-based (attention output fidelity ceiling). The
    # oracle_max-based set_recall is still accessible via `set_recall_proxy_vs_oraclemax`
    # for RULER-mk3-aligned analysis, but mass_topk is the task-agnostic default
    # (Bayes-optimal selector for preserving attention mass).
    set_recall = set_recall_proxy

    # Pairwise set overlaps (vs DCT proxy) and union mass with Quest.
    # union mass is the upper bound for any K-budget combination of two selectors
    # (uses |Q ∪ DCT| pages, which varies from K to 2K depending on overlap).
    set_overlap_quest_dct = (sel_q_mask & quest_topk_q_mask).sum(-1).float() / float(K)
    set_overlap_shadowkv_dct = (sel_q_mask & shadowkv_topk_q_mask).sum(-1).float() / float(K)
    set_overlap_infllm_dct = (sel_q_mask & infllm_topk_q_mask).sum(-1).float() / float(K)
    set_overlap_quest_shadowkv = (quest_topk_q_mask & shadowkv_topk_q_mask).sum(-1).float() / float(K)
    union_qd_mask = sel_q_mask | quest_topk_q_mask                              # [H_q, P]
    selected_mass_union_qd = (page_mass * union_qd_mask.float()).sum(-1)        # [H_q]
    mass_recall_union_qd = selected_mass_union_qd + extra_mass
    # Disjoint fraction: (|union| - K) / K ∈ [0, 1]. 0 = identical sets, 1 = fully disjoint.
    union_qd_count = union_qd_mask.sum(-1).float()                              # [H_q] in [K, 2K]
    union_qd_disjoint_frac = (union_qd_count - float(K)) / float(K)             # [H_q] in [0, 1]

    # Half-half combo: Quest top-(K/2) ∪ DCT top-(K/2). Page count varies between K/2
    # and K (smaller when overlap, larger when disjoint). Uses budget ≤ K.
    half = max(1, K // 2)
    quest_half_topk = torch.topk(quest_scores, half, dim=-1).indices            # [H_kv, K/2]
    dct_half_topk = selected_indices[:, :half]                                  # [H_kv, K/2]
    quest_half_q = quest_half_topk.repeat_interleave(num_kv_groups, dim=0)      # [H_q, K/2]
    dct_half_q = dct_half_topk.repeat_interleave(num_kv_groups, dim=0)          # [H_q, K/2]
    quest_half_mask = _indices_to_mask(quest_half_q, P)                         # [H_q, P]
    dct_half_mask = _indices_to_mask(dct_half_q, P)                             # [H_q, P]
    halfhalf_mask = quest_half_mask | dct_half_mask                             # [H_q, P]
    selected_mass_halfhalf_qd = (page_mass * halfhalf_mask.float()).sum(-1)     # [H_q]
    mass_recall_halfhalf_qd = selected_mass_halfhalf_qd + extra_mass
    halfhalf_count = halfhalf_mask.sum(-1).float()                              # [H_q] in [K/2, K]
    # Fraction of K used: 0.5 (full overlap) to 1.0 (fully disjoint).
    halfhalf_size_frac = halfhalf_count / float(K)                              # [H_q] in [0.5, 1]

    # Best-rank fusion: take top-K by min(rank_quest, rank_dct). Effectively starts at
    # rank 0 in both lists and expands the union until exactly K pages are selected.
    # Pages outside either top-K get sentinel rank = K (excluded as long as |union| ≥ K).
    arange_K = torch.arange(K, device=page_mass.device, dtype=torch.long).unsqueeze(0).expand(H_kv, K)
    quest_topk_full = quest_topk                                                # [H_kv, K] from earlier
    dct_topk_full = selected_indices.long()                                     # [H_kv, K]
    quest_rank = torch.full((H_kv, P), K, dtype=torch.long, device=page_mass.device)
    quest_rank.scatter_(-1, quest_topk_full.long(), arange_K)
    dct_rank = torch.full((H_kv, P), K, dtype=torch.long, device=page_mass.device)
    dct_rank.scatter_(-1, dct_topk_full, arange_K)
    min_rank = torch.minimum(quest_rank, dct_rank).float()                      # [H_kv, P]
    # Top-K by smallest min_rank (i.e., -min_rank descending).
    fill_topk = torch.topk(-min_rank, K, dim=-1).indices                        # [H_kv, K]
    fill_topk_q = fill_topk.repeat_interleave(num_kv_groups, dim=0)             # [H_q, K]
    fill_mask = _indices_to_mask(fill_topk_q, P)                                # [H_q, P]
    selected_mass_fill_qd = (page_mass * fill_mask.float()).sum(-1)             # [H_q]
    mass_recall_fill_qd = selected_mass_fill_qd + extra_mass

    # ----- Critical Mass Recall (CMR_α): peak-weighted mass recall -------------
    # CMR_α(S) = Σ_p [page_mass[p]]^α · 1[p ∈ S] / Σ_p [page_mass[p]]^α
    # α=1 reduces to paged_mass_recall (selected_mass / pages_mass).
    # α→∞ collapses to "did we pick the oracle top-1 page?" (binary).
    # We compute α ∈ {2, 4} explicitly; α=∞ is the top-1 oracle-page hit rate.
    def _cmr_alpha(mask_q: torch.Tensor, alpha: float) -> torch.Tensor:
        w = page_mass.clamp(min=0).pow(alpha)                     # [H_q, P]
        denom = w.sum(-1).clamp(min=1e-12)
        return (w * mask_q.float()).sum(-1) / denom               # [H_q]

    def _cmr_top1(mask_q: torch.Tensor) -> torch.Tensor:
        top1_idx = page_mass.argmax(dim=-1, keepdim=True)         # [H_q, 1]
        return mask_q.gather(-1, top1_idx).squeeze(-1).float()    # [H_q] ∈ {0,1}

    # ----- Rank-resolved hit rate -------------------------------------------
    # For each oracle rank r in 0..K-1 (sorted by page_mass desc), record whether
    # the selector's chosen set contains the r-th oracle page. Aggregated this
    # gives a "selection profile vs. oracle rank" — the metric that distinguishes
    # methods that grab the mass-peak vs. methods that grab informative middle
    # ranks (where the answer often sits).
    # mass_topk_idx already holds oracle's top-K page IDs in DESCENDING mass order
    # (torch.topk returns sorted by default).
    rank_hit_proxy_per_r    = sel_q_mask.gather(-1, mass_topk_idx).float()        # [H_q, K]
    rank_hit_quest_per_r    = quest_topk_q_mask.gather(-1, mass_topk_idx).float()
    rank_hit_shadowkv_per_r = shadowkv_topk_q_mask.gather(-1, mass_topk_idx).float()
    rank_hit_infllm_per_r   = infllm_topk_q_mask.gather(-1, mass_topk_idx).float()

    cmr2_proxy   = _cmr_alpha(sel_q_mask, 2.0)
    cmr2_quest   = _cmr_alpha(quest_topk_q_mask, 2.0)
    cmr2_shadowkv = _cmr_alpha(shadowkv_topk_q_mask, 2.0)
    cmr2_infllm  = _cmr_alpha(infllm_topk_q_mask, 2.0)
    cmr4_proxy   = _cmr_alpha(sel_q_mask, 4.0)
    cmr4_quest   = _cmr_alpha(quest_topk_q_mask, 4.0)
    cmr4_shadowkv = _cmr_alpha(shadowkv_topk_q_mask, 4.0)
    cmr4_infllm  = _cmr_alpha(infllm_topk_q_mask, 4.0)
    top1_proxy   = _cmr_top1(sel_q_mask)
    top1_quest   = _cmr_top1(quest_topk_q_mask)
    top1_shadowkv = _cmr_top1(shadowkv_topk_q_mask)
    top1_infllm  = _cmr_top1(infllm_topk_q_mask)

    # ----- Selected-page mass (absolute, fraction of total softmax mass) ------
    # Equivalent to 1 − sink − recent − Σ_{unselected} m[p]. Derived from the
    # full-KV mass metrics by subtracting the always-kept floor.
    selected_mass_proxy = mass_recall_proxy - extra_mass
    selected_mass_quest = mass_recall_quest - extra_mass
    selected_mass_shadowkv = mass_recall_shadowkv - extra_mass
    selected_mass_infllm = mass_recall_infllm - extra_mass
    selected_mass_mass_topk = mass_recall_mass_topk - extra_mass

    # Total paged-region mass per head (= 1 − sink − recent). Carried through
    # aggregation as the denominator for paged_mass_recall_*; the ratios are
    # derived post-hoc as ratio-of-aggregated-means by ``_derive_paged_metrics``
    # rather than averaged per-step.
    pages_mass = page_mass.sum(-1)                                             # [H_q]

    # ----- min_K(τ) coverage metric -------------------------------------------
    # For each selector, find the minimum K such that the top-K pages (ranked
    # by selector scores) capture at least τ × total paged-region mass per
    # qo-head. Threshold-based dual to set_recall/attn_recall; measures the
    # selector's ranking efficiency at the top.
    THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 0.99)
    def _min_k_for_thresholds(scores_qh: torch.Tensor) -> dict[float, torch.Tensor]:
        # scores_qh: [H_q, P], page_mass: [H_q, P] in closure.
        sort_idx = scores_qh.argsort(dim=-1, descending=True)                  # [H_q, P]
        sorted_mass = page_mass.gather(-1, sort_idx)                           # [H_q, P]
        cum = sorted_mass.cumsum(-1)                                           # [H_q, P]
        total = pages_mass.unsqueeze(-1).clamp(min=1e-12)                      # [H_q, 1]
        cum_frac = cum / total                                                 # [H_q, P]
        out: dict[float, torch.Tensor] = {}
        for tau in THRESHOLDS:
            k = (cum_frac < tau).sum(-1) + 1                                   # [H_q]
            k = torch.clamp(k, max=cum_frac.shape[-1]).float()
            out[tau] = k
        return out
    # Expand kv-head scores to qo-head granularity.
    def _to_qh(s_kv: torch.Tensor) -> torch.Tensor:
        return s_kv.repeat_interleave(num_kv_groups, dim=0).float()
    selector_scores_qh = {
        "proxy":     _to_qh(proxy_scores) if proxy_scores is not None else None,
        "segmean":   _to_qh(segmean_scores) if segmean_scores is not None else None,
        "pca":       _to_qh(pca_scores) if pca_scores is not None else None,
        "fasa":      _to_qh(fasa_scores) if fasa_scores is not None else None,
        "quest":     _to_qh(quest_scores),
        "shadowkv":  _to_qh(shadowkv_scores),
        "infllm":    _to_qh(infllm_scores),
        "oraclemax": _to_qh(oracle_max_scores) if oracle_max_scores is not None else None,
        "mass_topk": page_mass.float(),  # mass-oracle: rank by page_mass directly per qo-head
    }
    min_k_metrics: dict[str, torch.Tensor] = {}
    for sel, scores in selector_scores_qh.items():
        if scores is None:
            for tau in THRESHOLDS:
                min_k_metrics[f"min_k_{sel}_at_{int(tau*100)}"] = torch.zeros_like(pages_mass)
            continue
        per_tau = _min_k_for_thresholds(scores)
        for tau, k in per_tau.items():
            min_k_metrics[f"min_k_{sel}_at_{int(tau*100)}"] = k

    return {
        "mass_recall_sink": sink_mass,
        "mass_recall_recent": recent_mass,
        "pages_mass": pages_mass,
        "mass_recall_proxy": mass_recall_proxy,
        "mass_recall_segmean": mass_recall_segmean,
        "mass_recall_pca": mass_recall_pca,
        "mass_recall_fasa": mass_recall_fasa,
        "mass_recall_quest": mass_recall_quest,
        "mass_recall_shadowkv": mass_recall_shadowkv,
        "mass_recall_infllm": mass_recall_infllm,
        "mass_recall_mass_topk": mass_recall_mass_topk,
        "set_recall": set_recall,
        "set_recall_proxy": set_recall_proxy,
        "set_recall_segmean": set_recall_segmean,
        "set_recall_pca": set_recall_pca,
        "set_recall_fasa": set_recall_fasa,
        "set_recall_quest": set_recall_quest,
        "set_recall_shadowkv": set_recall_shadowkv,
        "set_recall_infllm": set_recall_infllm,
        "set_recall_proxy_vs_oraclemax": set_recall_proxy_vs_oraclemax,
        "set_recall_quest_vs_oraclemax": set_recall_quest_vs_oraclemax,
        "set_recall_shadowkv_vs_oraclemax": set_recall_shadowkv_vs_oraclemax,
        "set_recall_infllm_vs_oraclemax": set_recall_infllm_vs_oraclemax,
        "set_recall_masstopk_vs_oraclemax": set_recall_masstopk_vs_oraclemax,
        "mass_recall_oraclemax": mass_recall_oraclemax,
        "selected_mass_oraclemax": selected_mass_oraclemax,
        "set_overlap_quest_dct": set_overlap_quest_dct,
        "set_overlap_shadowkv_dct": set_overlap_shadowkv_dct,
        "set_overlap_infllm_dct": set_overlap_infllm_dct,
        "set_overlap_quest_shadowkv": set_overlap_quest_shadowkv,
        "selected_mass_union_qd": selected_mass_union_qd,
        "mass_recall_union_qd": mass_recall_union_qd,
        "union_qd_disjoint_frac": union_qd_disjoint_frac,
        "selected_mass_halfhalf_qd": selected_mass_halfhalf_qd,
        "mass_recall_halfhalf_qd": mass_recall_halfhalf_qd,
        "halfhalf_size_frac": halfhalf_size_frac,
        "selected_mass_fill_qd": selected_mass_fill_qd,
        "mass_recall_fill_qd": mass_recall_fill_qd,
        "selected_mass_proxy": selected_mass_proxy,
        "selected_mass_segmean": selected_mass_segmean,
        "selected_mass_pca": selected_mass_pca,
        "selected_mass_fasa": selected_mass_fasa,
        "selected_mass_quest": selected_mass_quest,
        "selected_mass_shadowkv": selected_mass_shadowkv,
        "selected_mass_infllm": selected_mass_infllm,
        "selected_mass_mass_topk": selected_mass_mass_topk,
        "cmr2_proxy": cmr2_proxy,
        "cmr2_quest": cmr2_quest,
        "cmr2_shadowkv": cmr2_shadowkv,
        "cmr2_infllm": cmr2_infllm,
        "cmr4_proxy": cmr4_proxy,
        "cmr4_quest": cmr4_quest,
        "cmr4_shadowkv": cmr4_shadowkv,
        "cmr4_infllm": cmr4_infllm,
        "top1_proxy": top1_proxy,
        "top1_quest": top1_quest,
        "top1_shadowkv": top1_shadowkv,
        "top1_infllm": top1_infllm,
        # [H_q, K] rank-resolved hit arrays — NOT in METRIC_KEYS, handled separately.
        "rank_hit_proxy_per_r": rank_hit_proxy_per_r,
        "rank_hit_quest_per_r": rank_hit_quest_per_r,
        "rank_hit_shadowkv_per_r": rank_hit_shadowkv_per_r,
        "rank_hit_infllm_per_r": rank_hit_infllm_per_r,
        # vs oracle_max ranks (RULER-consistent).
        "rank_hit_proxy_per_r_vs_oraclemax": rank_hit_proxy_per_r_vs_oraclemax,
        "rank_hit_quest_per_r_vs_oraclemax": rank_hit_quest_per_r_vs_oraclemax,
        "rank_hit_shadowkv_per_r_vs_oraclemax": rank_hit_shadowkv_per_r_vs_oraclemax,
        "rank_hit_infllm_per_r_vs_oraclemax": rank_hit_infllm_per_r_vs_oraclemax,
        **min_k_metrics,
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
    sm_om = float(metrics.get("selected_mass_oraclemax", 0.0))
    out: dict[str, float] = {}
    for sel in ("proxy", "segmean", "pca", "fasa", "quest", "shadowkv", "infllm", "mass_topk", "oraclemax"):
        sm = float(metrics.get(f"selected_mass_{sel}", 0.0))
        out[f"paged_mass_recall_{sel}"] = (sm / pm) if pm > 1e-12 else 0.0
    for sel in ("proxy", "segmean", "pca", "fasa", "quest", "shadowkv", "infllm"):
        sm = float(metrics.get(f"selected_mass_{sel}", 0.0))
        # Legacy ratio vs mass_topk ceiling (kept for backward compat / comparison).
        out[f"paged_mass_ratio_{sel}"] = (sm / smm) if smm > 1e-12 else 0.0
        # Canonical "attention recall" — softmax mass captured by selector relative
        # to oracle_max's K pages. RULER-consistent ceiling (oracle_max RULER mk3=80
        # vs mass-oracle mk3=76). Can exceed 1.0 when a selector lucks into more
        # mass than oracle_max's set carries, since oracle_max doesn't optimize mass.
        out[f"attention_recall_{sel}"] = (sm / sm_om) if sm_om > 1e-12 else 0.0
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
        num_sink_pages: int,
        num_recent_pages: int,
        comp_size: int,
        scoring_method: str,
        group_agg_method: str,
        infllm_repr_topk: int,
        comp_kv_quant: str = "none",
        comp_kv_quant_granularity: str = "per_page",
        infllm_local_window: int = 128,
        dct_proxy_softmax: bool = False,
        dct_bins: tuple[int, ...] | None = None,
        dct_qaware_topbins: int = 0,
        dct_qaware_mode: str = "sum_abs",
        lastq_window: int = 1,
        pca_M_loaded: dict | None = None,
        pca_cs_h: int = 0,
        fasa_idom_loaded: dict | None = None,
        fasa_n_tip: int = 0,
    ):
        self.num_decode_steps = num_decode_steps
        self.page_size = page_size
        self.top_k = top_k
        self.num_sink_pages = num_sink_pages
        self.num_recent_pages = num_recent_pages
        self.comp_size = comp_size
        self.scoring_method = scoring_method
        self.group_agg_method = group_agg_method
        self.infllm_repr_topk = infllm_repr_topk
        self.infllm_local_window = infllm_local_window
        self.comp_kv_quant = comp_kv_quant
        self.comp_kv_quant_granularity = comp_kv_quant_granularity
        self.dct_proxy_softmax = dct_proxy_softmax
        self.dct_bins = dct_bins
        self.dct_qaware_topbins = int(dct_qaware_topbins or 0)
        self.dct_qaware_mode = dct_qaware_mode
        # layer_idx → frozen "summary Q" used by lastq_recon_max mode. Built from
        # either decode-step-0 Q (window=1) or mean of last N prefill queries
        # (window>1, captured at prefill phase).
        self._lastq_per_layer: dict[int, torch.Tensor] = {}
        # If >1, lastq_recon_max uses mean of last `lastq_window` prefill queries
        # for bin selection (InfLLM r_m–style local window). =1 falls back to
        # using decode-step-0 Q.
        self._lastq_window: int = int(max(1, lastq_window))
        self.km_quest_split: int = 0  # set externally before generation if non-zero
        # Calibration-based proxy bases (PCA + FASA-FC).
        self.pca_M_loaded = pca_M_loaded
        self.pca_cs_h = int(pca_cs_h)
        self.fasa_idom_loaded = fasa_idom_loaded
        self.fasa_n_tip = int(fasa_n_tip)
        self.records: list[dict[str, Any]] = []
        self._step_by_layer: dict[int, int] = {}
        # Paper-faithful InfLLM: r_m computed at prefill, repr indices derived once
        # at first decode step per layer (when pageable region geometry is known).
        self._prefill_r_m: dict[int, torch.Tensor] = {}      # layer_idx → [H_kv, seq_len]
        self._paper_repr_idx: dict[int, torch.Tensor] = {}   # layer_idx → [H_kv, P, R] local idx

    def __call__(self, payload: dict[str, Any]) -> None:
        # Prefill phase: compute r_m for paper-faithful InfLLM, then return.
        if payload.get("phase") == "prefill":
            layer_idx = int(payload["layer_idx"])
            if layer_idx in self._prefill_r_m:
                return  # already processed for this layer
            with torch.no_grad():
                r_m = _compute_prefill_rm(
                    payload["query_states_prefill"],
                    payload["key_states_prefill"],
                    int(payload["num_kv_groups"]),
                    self.infllm_local_window,
                )
            self._prefill_r_m[layer_idx] = r_m.cpu()
            # Cache last-N prefill queries' mean for lastq_window > 1.
            if self._lastq_window > 1:
                q_prefill = payload["query_states_prefill"]                # [1, H_q, seq_len, d]
                N = min(self._lastq_window, q_prefill.shape[2])
                q_window_mean = q_prefill[:, :, -N:, :].mean(dim=2, keepdim=True)   # [1, H_q, 1, d]
                self._lastq_per_layer[layer_idx] = q_window_mean.detach().clone()
            return

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
        # whole-page sink, whole-page recent (last page may be partial absorbing
        # the alignment remainder), and the middle carved into whole pages.
        sink_len = self.num_sink_pages * self.page_size
        recent_min = self.num_recent_pages * self.page_size
        if kv_len < sink_len + self.page_size + recent_min:
            return  # nothing meaningful to page
        num_pages = (kv_len - sink_len - recent_min) // self.page_size
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

            if self.dct_qaware_topbins > 0:
                # Q-aware adaptive: per (page, qo-head) pick top-N DCT bins by |Q·DCT(K)|.
                selection_query = None
                if self.dct_qaware_mode == "lastq_recon_max":
                    # Selection Q: lastq_window>1 → mean of last-N prefill queries
                    # (captured at prefill phase). Window=1 → first-decode-step Q
                    # cached on first encounter. Either way frozen across steps.
                    cached = self._lastq_per_layer.get(layer_idx)
                    if cached is None:
                        cached = query_states.detach().clone()
                        self._lastq_per_layer[layer_idx] = cached
                    selection_query = cached
                proxy_scores_gpu = compute_dct_qaware_adaptive_scores(
                    query_states, paged_k,
                    top_bins=self.dct_qaware_topbins,
                    num_kv_groups=num_kv_groups,
                    group_agg_method=self.group_agg_method,
                    mode=self.dct_qaware_mode,
                    selection_query=selection_query,
                )
            else:
                proxy_scores_gpu = compute_dct_lowpass_proxy_scores(
                    query_states, paged_k, self.comp_size, num_kv_groups,
                    self.group_agg_method, self.scoring_method,
                    comp_kv_quant=self.comp_kv_quant,
                    comp_kv_quant_granularity=self.comp_kv_quant_granularity,
                    softmax_before_group=self.dct_proxy_softmax,
                    dct_bins=self.dct_bins,
                )
            proxy_scores = proxy_scores_gpu.float().cpu()                      # [H_kv, P]

            # oracle_max scores: same pipeline as the proxy but with the identity
            # projection (comp_size = page_size). This is what
            # `--select_with_oracle_page_scores` would select.
            oracle_max_scores_gpu = compute_dct_lowpass_proxy_scores(
                query_states, paged_k, self.page_size, num_kv_groups,
                self.group_agg_method, self.scoring_method,
                comp_kv_quant="none",                       # never quantize oracle reference
                comp_kv_quant_granularity=self.comp_kv_quant_granularity,
                softmax_before_group=self.dct_proxy_softmax,
            )                                                                  # [H_kv, P]

            # SegmentMean baseline: per-page split into comp_size segments, take mean.
            segmean_scores_gpu = compute_segmean_proxy_scores(
                query_states, paged_k, self.comp_size, num_kv_groups,
                self.group_agg_method, self.scoring_method,
            )                                                                  # [H_kv, P]

            # PCA dense head-dim projection (calibrated, per-(layer, kv-head)).
            if (self.pca_M_loaded is not None) and (self.pca_cs_h > 0):
                M_layer = self.pca_M_loaded.get(layer_idx)
                if M_layer is not None:
                    pca_scores_gpu = compute_pca_proxy_scores(
                        query_states, paged_k, self.pca_cs_h, num_kv_groups,
                        self.group_agg_method, self.scoring_method, M_layer,
                    )
                else:
                    pca_scores_gpu = None
            else:
                pca_scores_gpu = None
            # FASA dominant-FC channel subset (calibrated, per-(layer, q-head)).
            if (self.fasa_idom_loaded is not None) and (self.fasa_n_tip > 0):
                idom_layer = self.fasa_idom_loaded.get(layer_idx)
                if idom_layer is not None:
                    fasa_scores_gpu = compute_fasa_fc_proxy_scores(
                        query_states, paged_k, self.fasa_n_tip, num_kv_groups,
                        self.group_agg_method, self.scoring_method, idom_layer,
                    )
                else:
                    fasa_scores_gpu = None
            else:
                fasa_scores_gpu = None

            quest_scores_gpu = compute_quest_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )
            quest_scores = quest_scores_gpu.float().cpu()

            shadowkv_scores_gpu = compute_shadowkv_scores(
                query_states, paged_k, num_kv_groups, self.group_agg_method,
            )
            shadowkv_scores = shadowkv_scores_gpu.float().cpu()

            # Paper-faithful InfLLM: representatives FIXED by local-window-attention r_m
            # (computed at prefill). Slice r_m to the current pageable region each
            # step (P may shift by ±1 as the decode cache grows across page boundaries).
            r_m_full = self._prefill_r_m.get(layer_idx)
            if r_m_full is None:
                # Prefill payload missing — fall back to Q-aware variant.
                infllm_scores_gpu = compute_infllm_scores(
                    query_states, paged_k, num_kv_groups,
                    self.infllm_repr_topk, self.group_agg_method,
                )
            else:
                # r_m_full has length = prefill seq_len; decode-added tokens beyond it
                # are in the recent region (not paged) so the pageable slice fits.
                r_m_seq = r_m_full.shape[-1]
                slice_end = min(paged_end, r_m_seq)
                r_m_paged_flat = r_m_full[:, sink_len:slice_end]
                tokens_avail = r_m_paged_flat.shape[-1]
                pages_avail = tokens_avail // S
                if pages_avail < P:
                    # Last few pages exceed prefill — pad r_m with -inf so they
                    # contribute nothing (their repr ends up at index 0 by topk
                    # convention; the score will be low anyway).
                    pad_amount = (P * S) - tokens_avail
                    r_m_paged_flat = torch.cat([
                        r_m_paged_flat,
                        torch.full((H_kv, pad_amount), float("-inf"),
                                   dtype=r_m_paged_flat.dtype),
                    ], dim=-1)
                r_m_paged = r_m_paged_flat.view(H_kv, P, S).to(paged_k.device)
                R = min(self.infllm_repr_topk, S)
                repr_idx_local = r_m_paged.topk(R, dim=-1).indices.long()  # [H_kv, P, R]
                infllm_scores_gpu = compute_infllm_paper_scores(
                    query_states, paged_k, repr_idx_local,
                    num_kv_groups, self.group_agg_method,
                )
            infllm_scores = infllm_scores_gpu.float().cpu()

            # Top-K per selector (all at kv-head granularity).
            proxy_topk_gpu = torch.topk(proxy_scores_gpu, actual_top_k, dim=-1).indices
            quest_topk_gpu = torch.topk(quest_scores_gpu, actual_top_k, dim=-1).indices

            # K-M union strategy: replace proxy_topk with (DCT top-(K-M)) ∪ (Quest top-M not-in-DCT).
            # Used to measure recall of "DCT + Quest mass-heavy supplement" budget split.
            if self.km_quest_split > 0:
                M = min(int(self.km_quest_split), actual_top_k - 1)
                K_dct = actual_top_k - M
                dct_topkm = torch.topk(proxy_scores_gpu, K_dct, dim=-1).indices    # [H_kv, K-M]
                # Mask DCT-selected pages in quest_scores → Quest picks novel pages
                P_pages = proxy_scores_gpu.shape[-1]
                dct_mask = torch.zeros_like(proxy_scores_gpu, dtype=torch.bool)
                dct_mask.scatter_(-1, dct_topkm, True)
                quest_scores_masked = quest_scores_gpu.masked_fill(dct_mask, float('-inf'))
                quest_topM = torch.topk(quest_scores_masked, M, dim=-1).indices    # [H_kv, M]
                proxy_topk_gpu = torch.cat([dct_topkm, quest_topM], dim=-1)        # [H_kv, K]
            shadowkv_topk_gpu = torch.topk(
                shadowkv_scores_gpu, actual_top_k, dim=-1,
            ).indices
            infllm_topk_gpu = torch.topk(
                infllm_scores_gpu, actual_top_k, dim=-1,
            ).indices

            fidelity_gpu = compute_output_fidelity(
                query_states, sink_k, sink_v, paged_k, paged_v, recent_k, recent_v,
                {
                    "output_fidelity_proxy": proxy_topk_gpu,
                    "output_fidelity_quest": quest_topk_gpu,
                    "output_fidelity_shadowkv": shadowkv_topk_gpu,
                    "output_fidelity_infllm": infllm_topk_gpu,
                },
                num_kv_groups,
            )
            fidelity = {k: v.float().cpu() for k, v in fidelity_gpu.items()}

            selected_indices = proxy_topk_gpu.cpu()                            # [H_kv, K]

        mass_metrics = compute_all_metrics(
            page_mass, sink_mass, recent_mass, selected_indices,
            quest_scores, shadowkv_scores, infllm_scores,
            num_kv_groups,
            oracle_max_scores=oracle_max_scores_gpu.float().cpu(),
            proxy_scores=proxy_scores,
            segmean_scores=segmean_scores_gpu.float().cpu(),
            pca_scores=(pca_scores_gpu.float().cpu() if pca_scores_gpu is not None else None),
            fasa_scores=(fasa_scores_gpu.float().cpu() if fasa_scores_gpu is not None else None),
        )
        metrics = {**mass_metrics, **fidelity}

        # Rank-resolved hit arrays are [H_q, K] tensors stored under non-METRIC_KEY
        # field names; pop them out of `metrics` before scalar-validation.
        rank_keys = ("rank_hit_proxy_per_r", "rank_hit_quest_per_r",
                     "rank_hit_shadowkv_per_r", "rank_hit_infllm_per_r",
                     "rank_hit_proxy_per_r_vs_oraclemax",
                     "rank_hit_quest_per_r_vs_oraclemax",
                     "rank_hit_shadowkv_per_r_vs_oraclemax",
                     "rank_hit_infllm_per_r_vs_oraclemax")
        rank_arrays = {k: metrics.pop(k) for k in rank_keys if k in metrics}

        # Invariants: mass_* ∈ [0, 1]; fidelity_* ∈ [-1, 1] (cos sim);
        # min_k_* metrics are integer page counts in [1, num_pages], skipped.
        for key, tensor in metrics.items():
            if key.startswith("min_k_"):
                continue
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
            **{k: rank_arrays[k].tolist() for k in rank_arrays},
        })


def generate_with_mass_traces(
    model,
    tokenizer,
    sample: dict[str, Any],
    *,
    num_decode_steps: int,
    page_size: int,
    top_k: int,
    num_sink_pages: int,
    num_recent_pages: int,
    comp_size: int,
    scoring_method: str,
    group_agg_method: str,
    infllm_repr_topk: int,
    infllm_local_window: int,
    comp_kv_quant: str,
    comp_kv_quant_granularity: str,
    dct_proxy_softmax: bool = False,
    dct_bins: tuple[int, ...] | None = None,
    dct_qaware_topbins: int = 0,
    dct_qaware_mode: str = "sum_abs",
    lastq_window: int = 1,
    km_quest_split: int = 0,
    pca_M_loaded: dict | None = None,
    pca_cs_h: int = 0,
    fasa_idom_loaded: dict | None = None,
    fasa_n_tip: int = 0,
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
        num_sink_pages=num_sink_pages,
        num_recent_pages=num_recent_pages,
        comp_size=comp_size,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        infllm_repr_topk=infllm_repr_topk,
        infllm_local_window=infllm_local_window,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
        dct_proxy_softmax=dct_proxy_softmax,
        dct_bins=dct_bins,
        dct_qaware_topbins=dct_qaware_topbins,
        dct_qaware_mode=dct_qaware_mode,
        lastq_window=lastq_window,
        pca_M_loaded=pca_M_loaded,
        pca_cs_h=pca_cs_h,
        fasa_idom_loaded=fasa_idom_loaded,
        fasa_n_tip=fasa_n_tip,
    )
    recorder.km_quest_split = int(km_quest_split)
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

    Skips shadowkv / infllm / fidelity to keep per-step cost low. Records
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
        num_sink_pages: int,
        num_recent_pages: int,
        comp_sizes: list[int],
        scoring_method: str,
        group_agg_method: str,
    ):
        self.num_decode_steps = num_decode_steps
        self.page_size = page_size
        self.top_k = top_k
        self.num_sink_pages = num_sink_pages
        self.num_recent_pages = num_recent_pages
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

        sink_len = self.num_sink_pages * self.page_size
        recent_min = self.num_recent_pages * self.page_size
        if kv_len < sink_len + self.page_size + recent_min:
            return
        num_pages = (kv_len - sink_len - recent_min) // self.page_size
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
    num_sink_pages: int,
    num_recent_pages: int,
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
        num_sink_pages=num_sink_pages,
        num_recent_pages=num_recent_pages,
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
    p.add_argument("--model_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16"])
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
    # TOTAL selected page budget (sink + middle + recent). Middle = top_k - sink - recent.
    # Matches eval_ruler / longbench / aime25 / gpqa semantics.
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--num_sink_pages", type=int, default=0)
    p.add_argument("--num_recent_pages", type=int, default=0)
    p.add_argument("--compress_ratio", type=float, default=0.125,
                   help="Haar proxy compression ratio; comp_size = "
                        "max(1, int(page_size * compress_ratio)).")
    p.add_argument("--scoring_method", type=str, default="max",
                   choices=["mean", "max", "sum", "lse"])
    p.add_argument("--group_agg_method", type=str, default="max",
                   choices=["mean", "max"])

    # InfLLM block-representative scoring (current Q·K substitutes for the
    # stateful local-score history that the diagnostic does not maintain).
    p.add_argument("--dct_proxy_softmax", action="store_true",
                   help="DCT proxy: apply softmax over pages per qo-head before GQA group reduction "
                        "(ShadowKV-style head-wise normalization).")
    p.add_argument("--dct_bins", type=int, nargs="+", default=None,
                   help="DCT band-pass bin indices to keep (length must equal comp_size). "
                        "Default: lowpass = [0, 1, ..., comp_size-1]. "
                        "Example: --dct_bins 0 8 16 24 spreads 4 bins across the spectrum.")
    p.add_argument("--dct_qaware_topbins", type=int, default=0,
                   help="If >0, switch the DCT proxy to Q-aware adaptive top-N bin selection. "
                        "Per (page, qo_head), pick the N DCT bins with largest |Q·DCT(K)|.")
    p.add_argument("--dct_qaware_mode", type=str, default="sum_abs",
                   choices=["sum_abs", "recon_max", "kaware_recon_max", "lastq_recon_max"],
                   help="Adaptive scoring rule. 'sum_abs': freq-domain Σ top-N |Q·DCT(K)|. "
                        "'recon_max': Q-aware (per-step), zero non-top-N signed bins, IDCT, max_t. "
                        "'kaware_recon_max': K-aware (Q-independent) bin selection by ||DCT(K)||_2, then recon_max. "
                        "'lastq_recon_max': production-feasible Q-aware approximation — pick bins ONCE "
                        "using last-N prefill queries' mean per layer (or decode-step-0 Q if --lastq_window 1), "
                        "freeze, score subsequent steps with current Q. Same storage budget as kaware/lowpass.")
    p.add_argument("--km_quest_split", type=int, default=0,
                   help="If >0, replace proxy selector with K-M union strategy: "
                        "DCT top-(K-M) + Quest top-M (Quest picks restricted to pages NOT in DCT's set). "
                        "Measures recall of 'DCT + Quest mass-heavy supplement' budget allocation.")
    p.add_argument("--lastq_window", type=int, default=1,
                   help="Window size for lastq_recon_max. =1: use decode-step-0 Q. "
                        ">1: use mean of last-N prefill queries per layer (InfLLM r_m–style "
                        "local-window summary). Common values: 8, 16.")
    p.add_argument("--infllm_local_window", type=int, default=2048,
                   help="Paper-faithful InfLLM local-window size l_L for computing r_m "
                        "(Xiao 2024 uses 2K for Vicuna, 4K for Mistral).")
    p.add_argument("--infllm_repr_topk", type=int, default=4,
                   help="InfLLM: representative tokens per page used to build "
                        "the block representative. The diagnostic substitutes "
                        "upstream InfLLM's stateful local-score with current Q·K.")

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
    # bypasses the quest/shadowkv/infllm/fidelity selectors and forces
    # comp_kv_quant='none'; output goes to <output_dir>/<run_name>/ with
    # a paged_mass_ratio_curve.png plot.
    p.add_argument("--comp_size_sweep", type=str, nargs="?", const="all",
                   default=None,
                   help="Trigger comp_size sweep mode (measures only "
                        "paged_mass_ratio_proxy and emits a plot). Pass with "
                        "no value (or 'all') to auto-sweep every comp_size "
                        "in 1..page_size; otherwise pass a comma-separated "
                        "list (e.g. '1,2,4,8,16,32').")

    # Calibrated proxy bases (PCA + FASA-FC). When set, the recorder
    # additionally computes pca_/fasa_ metrics alongside DCT and SegMean.
    p.add_argument("--pca_M_path", type=str, default=None,
                   help="Path to pca_M_<model>.pt from oracle/calibrate_proxy_bases.py.")
    p.add_argument("--pca_cs_h", type=int, default=0,
                   help="head_dim PCA projection rank (0 disables).")
    p.add_argument("--fasa_idom_path", type=str, default=None,
                   help="Path to fasa_idom_<model>.pt from oracle/calibrate_proxy_bases.py.")
    p.add_argument("--fasa_n_tip", type=int, default=0,
                   help="FASA dominant FC count per (layer, q-head) (0 disables).")

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
                    top_k=max(1, args.top_k - args.num_sink_pages - args.num_recent_pages),
                    num_sink_pages=args.num_sink_pages,
                    num_recent_pages=args.num_recent_pages,
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

    # Optionally load calibrated proxy bases.
    if args.pca_M_path is not None and args.pca_cs_h > 0:
        d = torch.load(args.pca_M_path, weights_only=False, map_location="cpu")
        _PCA_M_CACHE[args.base_model] = d["M"]
        print(f"Loaded PCA bases: {args.pca_M_path}  cs_h_max={d.get('cs_h_max')}")
    if args.fasa_idom_path is not None and args.fasa_n_tip > 0:
        d = torch.load(args.fasa_idom_path, weights_only=False, map_location="cpu")
        _FASA_IDOM_CACHE[args.base_model] = d["idom"]
        print(f"Loaded FASA bases: {args.fasa_idom_path}  n_tip_max={d.get('n_tip_max')}")

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
            task_per_step: dict[int, list[dict[str, float]]] = {}
            # Rank-resolved hit profiles (one [K]-length numpy array per selector,
            # accumulated across all records by mean over H_q).
            import numpy as _np_local
            task_rank_acc: dict[str, list] = {
                "proxy": [], "quest": [], "shadowkv": [], "infllm": [],
                # vs oracle_max ranks (RULER-consistent).
                "proxy_vs_oraclemax": [], "quest_vs_oraclemax": [],
                "shadowkv_vs_oraclemax": [], "infllm_vs_oraclemax": [],
            }

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
                    top_k=max(1, args.top_k - args.num_sink_pages - args.num_recent_pages),
                    num_sink_pages=args.num_sink_pages,
                    num_recent_pages=args.num_recent_pages,
                    comp_size=comp_size,
                    scoring_method=args.scoring_method,
                    group_agg_method=args.group_agg_method,
                    infllm_repr_topk=args.infllm_repr_topk,
                    infllm_local_window=args.infllm_local_window,
                    dct_proxy_softmax=args.dct_proxy_softmax,
                    dct_bins=(tuple(args.dct_bins) if args.dct_bins else None),
                    dct_qaware_topbins=args.dct_qaware_topbins,
                    dct_qaware_mode=args.dct_qaware_mode,
                    lastq_window=args.lastq_window,
                    km_quest_split=args.km_quest_split,
                    comp_kv_quant=args.comp_kv_quant,
                    comp_kv_quant_granularity=args.comp_kv_quant_granularity,
                    pca_M_loaded=_PCA_M_CACHE.get(args.base_model),
                    pca_cs_h=int(getattr(args, "pca_cs_h", 0) or 0),
                    fasa_idom_loaded=_FASA_IDOM_CACHE.get(args.base_model),
                    fasa_n_tip=int(getattr(args, "fasa_n_tip", 0) or 0),
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
                    # Accumulate rank-hit arrays: [H_q, K] → average over H_q to [K].
                    # Two rank orderings: mass-topk (legacy) and oracle_max (canonical).
                    for sel in ("proxy", "quest", "shadowkv", "infllm"):
                        for suffix, acc_key in (
                            ("", sel),
                            ("_vs_oraclemax", f"{sel}_vs_oraclemax"),
                        ):
                            arr = rec.get(f"rank_hit_{sel}_per_r{suffix}")
                            if arr is not None:
                                arr_np = _np_local.asarray(arr, dtype=_np_local.float32)
                                if arr_np.ndim == 2:
                                    arr_np = arr_np.mean(axis=0)
                                task_rank_acc[acc_key].append(arr_np)

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
                        task_per_step.setdefault(decode_step, []).append(flat)

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
                        f"mass[p/q/s/i/c] = "
                        f"{o['mass_recall_proxy']:.3f}/"
                        f"{o['mass_recall_quest']:.3f}/"
                        f"{o['mass_recall_shadowkv']:.3f}/"
                        f"{o['mass_recall_infllm']:.3f}/"
                        f"{o['mass_recall_mass_topk']:.3f}  "
                        f"sel[p/q/s/i/c] = "
                        f"{o['selected_mass_proxy']:.3f}/"
                        f"{o['selected_mass_quest']:.3f}/"
                        f"{o['selected_mass_shadowkv']:.3f}/"
                        f"{o['selected_mass_infllm']:.3f}/"
                        f"{o['selected_mass_mass_topk']:.3f}  "
                        f"paged[p/q/s/i/c] = "
                        f"{o['paged_mass_recall_proxy']:.3f}/"
                        f"{o['paged_mass_recall_quest']:.3f}/"
                        f"{o['paged_mass_recall_shadowkv']:.3f}/"
                        f"{o['paged_mass_recall_infllm']:.3f}/"
                        f"{o['paged_mass_recall_mass_topk']:.3f}  "
                        f"ratio[p/q/s/i] = "
                        f"{o['paged_mass_ratio_proxy']:.3f}/"
                        f"{o['paged_mass_ratio_quest']:.3f}/"
                        f"{o['paged_mass_ratio_shadowkv']:.3f}/"
                        f"{o['paged_mass_ratio_infllm']:.3f}  "
                        f"fid[p/q/s/i] = "
                        f"{o['output_fidelity_proxy']:.3f}/"
                        f"{o['output_fidelity_quest']:.3f}/"
                        f"{o['output_fidelity_shadowkv']:.3f}/"
                        f"{o['output_fidelity_infllm']:.3f}"
                    )

            sample_fp.close()

            task_overall_agg = _aggregate_metric_dicts(task_overall_records)
            task_overall_agg.update(_derive_paged_metrics(task_overall_agg))
            task_per_layer_agg = {}
            for lyr, bucket in sorted(task_per_layer.items()):
                lm = _aggregate_metric_dicts(bucket)
                lm.update(_derive_paged_metrics(lm))
                task_per_layer_agg[str(lyr)] = lm
            task_per_step_agg = {}
            for step, bucket in sorted(task_per_step.items()):
                sm = _aggregate_metric_dicts(bucket)
                sm.update(_derive_paged_metrics(sm))
                task_per_step_agg[str(step)] = sm
            per_task_results[task] = {
                "num_samples": len(samples),
                "overall": task_overall_agg,
                "per_layer": task_per_layer_agg,
                "per_step": task_per_step_agg,
            }
            o = per_task_results[task]["overall"]
            print(
                f"  TASK SUMMARY\n"
                f"    sink / recent (floor)                       = "
                f"{o['mass_recall_sink']:.3f} / {o['mass_recall_recent']:.3f}\n"
                f"    mass   [proxy/quest/shadow/infllm/ceil]     = "
                f"{o['mass_recall_proxy']:.3f} / {o['mass_recall_quest']:.3f} / "
                f"{o['mass_recall_shadowkv']:.3f} / {o['mass_recall_infllm']:.3f} / "
                f"{o['mass_recall_mass_topk']:.3f}\n"
                f"    select [proxy/quest/shadow/infllm/ceil]     = "
                f"{o['selected_mass_proxy']:.3f} / {o['selected_mass_quest']:.3f} / "
                f"{o['selected_mass_shadowkv']:.3f} / {o['selected_mass_infllm']:.3f} / "
                f"{o['selected_mass_mass_topk']:.3f}\n"
                f"    paged  [proxy/quest/shadow/infllm/ceil]     = "
                f"{o['paged_mass_recall_proxy']:.3f} / {o['paged_mass_recall_quest']:.3f} / "
                f"{o['paged_mass_recall_shadowkv']:.3f} / {o['paged_mass_recall_infllm']:.3f} / "
                f"{o['paged_mass_recall_mass_topk']:.3f}\n"
                f"    ratio[mass-topk]  [proxy/quest/shadow/infllm]  = "
                f"{o['paged_mass_ratio_proxy']:.3f} / {o['paged_mass_ratio_quest']:.3f} / "
                f"{o['paged_mass_ratio_shadowkv']:.3f} / {o['paged_mass_ratio_infllm']:.3f}\n"
                f"    attention_recall[oraclemax ceil]    = "
                f"{o['attention_recall_proxy']:.3f} / {o['attention_recall_quest']:.3f} / "
                f"{o['attention_recall_shadowkv']:.3f} / {o['attention_recall_infllm']:.3f}\n"
                f"    fidelity[proxy/quest/shadow/infllm]         = "
                f"{o['output_fidelity_proxy']:.3f} / {o['output_fidelity_quest']:.3f} / "
                f"{o['output_fidelity_shadowkv']:.3f} / {o['output_fidelity_infllm']:.3f}\n"
                f"    set_recall[oraclemax]              = "
                f"{o['set_recall_proxy_vs_oraclemax']:.3f} / {o['set_recall_quest_vs_oraclemax']:.3f} / "
                f"{o['set_recall_shadowkv_vs_oraclemax']:.3f} / {o['set_recall_infllm_vs_oraclemax']:.3f}\n"
                f"    set_recall[mass-topk legacy]        = "
                f"{o['set_recall_proxy']:.3f} / {o['set_recall_quest']:.3f} / "
                f"{o['set_recall_shadowkv']:.3f} / {o['set_recall_infllm']:.3f}"
            )
            # ---- Rank-resolved hit profile (mean over records) -----
            rank_profile = {}
            for sel, arr_list in task_rank_acc.items():
                if arr_list:
                    rank_profile[sel] = _np_local.stack(arr_list, axis=0).mean(axis=0).tolist()
            per_task_results[task]["rank_profile"] = rank_profile
            if rank_profile:
                K = len(rank_profile.get("proxy_vs_oraclemax") or rank_profile.get("proxy") or [])
                if K > 0:
                    sample_ranks = [r for r in [0, 1, 2, 4, 8, 16, 30, K-1] if r < K]
                    head = "      rank " + "  ".join(f"r{r:>2d}" for r in sample_ranks)
                    print("    rank-resolved hit by oracle_max rank (RULER-consistent):")
                    print(head)
                    for sel in ("proxy", "quest", "shadowkv", "infllm"):
                        vals = rank_profile.get(f"{sel}_vs_oraclemax") or []
                        if vals:
                            row = "  ".join(f"{vals[r]:.2f}" for r in sample_ranks)
                            print(f"      {sel:>8s}  {row}")
                    print("    rank-resolved hit by mass-topk rank (legacy):")
                    print(head)
                    for sel in ("proxy", "quest", "shadowkv", "infllm"):
                        vals = rank_profile.get(sel) or []
                        if vals:
                            row = "  ".join(f"{vals[r]:.2f}" for r in sample_ranks)
                            print(f"      {sel:>8s}  {row}")
            print("    per-step mass_recall [proxy/q/s/i / oraclemax] and set_recall vs oracle_max [p/q/s/i]:")
            for step_str, sm in task_per_step_agg.items():
                print(
                    f"      step {int(step_str):2d}: mass="
                    f"{sm['mass_recall_proxy']:.3f}/{sm['mass_recall_quest']:.3f}/"
                    f"{sm['mass_recall_shadowkv']:.3f}/{sm['mass_recall_infllm']:.3f}/"
                    f"{sm.get('mass_recall_oraclemax', 0.0):.3f}  "
                    f"set="
                    f"{sm.get('set_recall_proxy_vs_oraclemax', 0.0):.3f}/"
                    f"{sm.get('set_recall_quest_vs_oraclemax', 0.0):.3f}/"
                    f"{sm.get('set_recall_shadowkv_vs_oraclemax', 0.0):.3f}/"
                    f"{sm.get('set_recall_infllm_vs_oraclemax', 0.0):.3f}"
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
                "num_sink_pages": args.num_sink_pages,
                "num_recent_pages": args.num_recent_pages,
                "compress_ratio": args.compress_ratio,
                "comp_size": comp_size,
                "scoring_method": args.scoring_method,
                "group_agg_method": args.group_agg_method,
                "infllm_repr_topk": args.infllm_repr_topk,
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
