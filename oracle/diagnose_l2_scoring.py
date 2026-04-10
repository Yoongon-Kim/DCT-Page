#!/usr/bin/env python3
"""
Diagnose page scoring methods by comparing them against a configurable ground truth.

Generates tokens with **full KV attention** (no page attention patch), then at
each decode step reshapes the KV cache into pages and evaluates how each scoring
method would rank them.  Recall is averaged across decode steps, layers, and heads
for a precise measurement.

  Ground truths (--ground_truths):
    oracle_max:          full tokens, scoring=max             max_i <q, k_i>
    output_contribution: per-page contribution to attention output
                         || sum_{i in page} softmax(s_i) * v_i ||

  Methods compared against ground truth:
    oracle_max:    full tokens, scoring=max
    oracle_mean:   full tokens, scoring=mean         mean_i <q, k_i>
    proxy_max:     DCT compressed tokens, scoring=max   max_c <q, comp_k_c>
    proxy_mean:    DCT compressed tokens, scoring=mean  mean_c <q, comp_k_c>
    l2_energy:     full tokens, L2 scoring           sqrt(sum_i <q, k_i>^2)

For each method reports (averaged across decode steps, layers, heads):
  - recall: fraction of GT top-k pages also selected by this method
  - false positives / false negatives
  - neg_gt_in_topk: pages in method's top-k with negative GT score
  - fn_rank: average rank (in this method) of missed GT pages
  - fn_gt_score: average GT score of missed pages
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

GROUND_TRUTHS = ["oracle_max", "output_contribution"]


# ---------------------------------------------------------------------------
# KV cache segmentation & query recomputation (full attention, no patch)
# ---------------------------------------------------------------------------
def segment_kv_from_cache(
    key_states: torch.Tensor,    # [bsz, kv_heads, kv_len, head_dim]
    value_states: torch.Tensor,  # [bsz, kv_heads, kv_len, head_dim]
    page_size: int,
    sink_size: int,
    recent_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, int]:
    """Split flat KV cache into sink / paged / recent segments.

    Returns:
        sink_k, sink_v:     [bsz, kv_heads, sink_size, head_dim]
        paged_k, paged_v:   [bsz, kv_heads, num_pages, page_size, head_dim]
        recent_k, recent_v: [bsz, kv_heads, actual_recent, head_dim]
        num_pages
    """
    bsz, kv_heads, kv_len, head_dim = key_states.shape
    pageable_len = kv_len - sink_size - recent_size
    num_pages = pageable_len // page_size
    pages_end = sink_size + num_pages * page_size

    sink_k = key_states[:, :, :sink_size]
    sink_v = value_states[:, :, :sink_size]
    paged_k = key_states[:, :, sink_size:pages_end].reshape(
        bsz, kv_heads, num_pages, page_size, head_dim)
    paged_v = value_states[:, :, sink_size:pages_end].reshape(
        bsz, kv_heads, num_pages, page_size, head_dim)
    recent_k = key_states[:, :, pages_end:]
    recent_v = value_states[:, :, pages_end:]
    return sink_k, sink_v, paged_k, paged_v, recent_k, recent_v, num_pages


def recompute_query(
    attn_module,
    hidden_states: torch.Tensor,  # [bsz, 1, hidden_dim] (post-layernorm)
    kv_len: int,
    config,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Recompute post-RoPE query_states from hidden_states for a single decode token.

    Handles QK-norm (Qwen3) and all RoPE types (default, yarn, llama3).

    Returns: [bsz, num_heads, 1, head_dim]
    """
    from dct_page_attention import _apply_rope, _compute_rope_cos_sin

    bsz = hidden_states.shape[0]
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    # q_proj → [bsz, 1, num_heads * head_dim]
    q = attn_module.q_proj(hidden_states)
    # Reshape → [bsz, 1, num_heads, head_dim]
    q = q.view(bsz, 1, num_heads, head_dim)
    # QK-norm (Qwen3)
    if hasattr(attn_module, "q_norm"):
        q = attn_module.q_norm(q)
    # Transpose → [bsz, num_heads, 1, head_dim]
    q = q.transpose(1, 2)

    # RoPE: position = kv_len - 1 (the position of this decode token in the cache)
    positions = torch.tensor([kv_len - 1], device=device)
    cos, sin = _compute_rope_cos_sin(positions, config, device, dtype)
    q = _apply_rope(q, cos, sin)
    return q


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def compute_per_token_scores(
    query_states: torch.Tensor,   # [bsz, num_heads, 1, head_dim]
    paged_k: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    num_kv_groups: int,
) -> torch.Tensor:
    """Compute per-token Q·K scores with group_agg=max.

    Returns: [bsz, kv_heads, num_pages, page_size] float32
    """
    bsz, num_heads, _, head_dim = query_states.shape
    _, kv_heads, num_pages, page_size, _ = paged_k.shape
    scaling = head_dim ** -0.5

    q = query_states.squeeze(2).reshape(bsz, kv_heads, num_kv_groups, head_dim).float()
    k = paged_k.float()

    # dots: [bsz, kv_heads, num_kv_groups, num_pages, page_size]
    dots = torch.einsum('bhgd,bhnsd->bhgns', q * scaling, k)

    # group_agg=max
    return dots.max(dim=2).values  # [bsz, kv_heads, num_pages, page_size]


def compute_output_contribution(
    query_states: torch.Tensor,   # [bsz, num_heads, 1, head_dim]
    paged_k: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    paged_v: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    num_kv_groups: int,
    sink_k: torch.Tensor | None = None,    # [bsz, kv_heads, sink_size, head_dim]
    sink_v: torch.Tensor | None = None,
    recent_k: torch.Tensor | None = None,  # [bsz, kv_heads, recent_size, head_dim]
    recent_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-page output contribution: || sum_{i in page} softmax(s_i) * v_i ||.

    Softmax is computed over ALL tokens (sink + paged + recent), so the weights
    reflect the real competition. Then contributions are summed per page and the
    L2 norm is taken. Sink/recent tokens participate in softmax but we only
    measure page contributions.

    For GQA, we compute per query head and then take max across the group
    (group_agg=max) to match the other methods.

    Returns: [bsz, kv_heads, num_pages] float32
    """
    bsz, num_heads, _, head_dim = query_states.shape
    _, kv_heads, num_pages, page_size, _ = paged_k.shape
    scaling = head_dim ** -0.5

    q = query_states.squeeze(2).reshape(bsz, kv_heads, num_kv_groups, head_dim).float()

    # Flatten paged keys/values: [bsz, kv_heads, num_pages * page_size, head_dim]
    paged_k_flat = paged_k.float().reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    paged_v_flat = paged_v.float().reshape(bsz, kv_heads, num_pages * page_size, head_dim)

    # Concatenate all keys for global softmax: sink + paged + recent
    all_k_parts = []
    all_v_parts = []
    if sink_k is not None and sink_k.shape[2] > 0:
        all_k_parts.append(sink_k.float())
        all_v_parts.append(sink_v.float())
    all_k_parts.append(paged_k_flat)
    all_v_parts.append(paged_v_flat)
    if recent_k is not None and recent_k.shape[2] > 0:
        all_k_parts.append(recent_k.float())
        all_v_parts.append(recent_v.float())

    # all_k: [bsz, kv_heads, total_len, head_dim]
    all_k = torch.cat(all_k_parts, dim=2)

    # Q·K for all tokens: [bsz, kv_heads, num_kv_groups, total_len]
    all_dots = torch.einsum('bhgd,bhtd->bhgt', q * scaling, all_k)

    # Global softmax over total_len
    all_attn = F.softmax(all_dots, dim=-1)

    # Extract only the paged portion of attention weights
    sink_len = sink_k.shape[2] if sink_k is not None else 0
    paged_start = sink_len
    paged_end = paged_start + num_pages * page_size
    # [bsz, kv_heads, num_kv_groups, num_pages * page_size]
    paged_attn = all_attn[:, :, :, paged_start:paged_end]
    # Reshape to [bsz, kv_heads, num_kv_groups, num_pages, page_size]
    paged_attn = paged_attn.reshape(bsz, kv_heads, num_kv_groups, num_pages, page_size)

    # Weighted values: [bsz, kv_heads, num_kv_groups, num_pages, page_size, head_dim]
    paged_v_5d = paged_v.float()  # [bsz, kv_heads, num_pages, page_size, head_dim]
    weighted_v = paged_attn.unsqueeze(-1) * paged_v_5d.unsqueeze(2)

    # Sum within each page: [bsz, kv_heads, num_kv_groups, num_pages, head_dim]
    page_output = weighted_v.sum(dim=4)

    # L2 norm: [bsz, kv_heads, num_kv_groups, num_pages]
    page_contrib = page_output.norm(dim=-1)

    # group_agg=max
    return page_contrib.max(dim=2).values  # [bsz, kv_heads, num_pages]


def compute_proxy_scores(
    query_states: torch.Tensor,   # [bsz, num_heads, 1, head_dim]
    paged_k: torch.Tensor,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    num_kv_groups: int,
    comp_size: int,
    lambdas: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Simulate proxy scoring with DCT compression, group_agg=max.

    Returns dict with proxy_max, proxy_mean, and spread proxy DC+AC variants.
    All values are [bsz, kv_heads, num_pages] float32.
    """
    from dct_page_attention import (
        _build_dct_projection_matrix,
        _build_dct_spectral_projection_matrix,
        _dct_page_cfg,
    )

    lambdas = lambdas or [0.5, 1.0, 2.0]
    bsz, num_heads, _, head_dim = query_states.shape
    _, kv_heads, num_pages, page_size, _ = paged_k.shape
    scaling = head_dim ** -0.5

    layout = _dct_page_cfg.proxy_frequency_layout if _dct_page_cfg is not None else "low"
    M = _build_dct_projection_matrix(page_size, comp_size, paged_k.device, paged_k.dtype, layout)

    comp_k = torch.einsum('cs,bhnsd->bhncd', M.float(), paged_k.float())

    q = query_states.squeeze(2).reshape(bsz, kv_heads, num_kv_groups, head_dim).float()

    # dots: [bsz, kv_heads, num_kv_groups, num_pages, comp_size]
    dots = torch.einsum('bhgd,bhncd->bhgnc', q * scaling, comp_k)

    # group_agg=max
    dots_gagg = dots.max(dim=2).values  # [bsz, kv_heads, num_pages, comp_size]

    proxy_max = dots_gagg.max(dim=-1).values
    proxy_mean = dots_gagg.mean(dim=-1)

    result = {"proxy_max": proxy_max, "proxy_mean": proxy_mean}

    # --- Exp F: Spread Layout Proxy ---
    # Build spectral projection with spread layout, compute DC+AC from those coefficients
    if comp_size > 1:
        M_spread = _build_dct_spectral_projection_matrix(
            page_size, comp_size, paged_k.device, paged_k.dtype, "spread",
        )
        comp_k_spread = torch.einsum('cs,bhnsd->bhncd', M_spread.float(), paged_k.float())
        dots_spread = torch.einsum('bhgd,bhncd->bhgnc', q * scaling, comp_k_spread)
        dots_spread_gagg = dots_spread.max(dim=2).values  # [bsz, kv_heads, num_pages, comp_size]

        spread_dc = dots_spread_gagg[..., 0]
        spread_ac = dots_spread_gagg[..., 1:].pow(2).sum(-1).sqrt()
        for lam in lambdas:
            result[f"spread_proxy_dc_ac_{lam}"] = spread_dc + lam * spread_ac

        # Also add spread proxy max/mean for reference
        result["spread_proxy_max"] = dots_spread_gagg.max(dim=-1).values
        result["spread_proxy_mean"] = dots_spread_gagg.mean(dim=-1)

    return result


def _log_spread_ac(page_size: int, n_ac: int) -> list[int]:
    """Doubling-gap AC indices from low end: [1, 3, 7, 15, ...] = 2^(k+1)-1.

    Falls back to scaled indices when doubling overshoots page_size.
    """
    if n_ac == 0:
        return []
    max_idx = page_size - 1
    raw = [2 ** (k + 1) - 1 for k in range(n_ac)]
    if raw[-1] <= max_idx:
        return raw
    # Scale to fit, push collisions forward to keep uniqueness
    scale = max_idx / raw[-1]
    scaled = [max(1, round(x * scale)) for x in raw]
    result: list[int] = []
    for s in scaled:
        if result and s <= result[-1]:
            s = result[-1] + 1
        if s <= max_idx:
            result.append(s)
    return result


def _reverse_log_spread_ac(page_size: int, n_ac: int) -> list[int]:
    """Doubling-gap AC indices from high end: [8, 12, 14] style (mirror of log)."""
    if n_ac == 0:
        return []
    max_idx = page_size - 1
    # Offsets from max_idx: 1, 3, 7, 15, ...
    offsets = [2 ** (k + 1) - 1 for k in range(n_ac)]
    if offsets[-1] > max_idx - 1:  # must keep idx >= 1
        scale = (max_idx - 1) / offsets[-1]
        offsets = [max(1, round(o * scale)) for o in offsets]
        deduped: list[int] = []
        for o in offsets:
            if deduped and o <= deduped[-1]:
                o = deduped[-1] + 1
            deduped.append(o)
        offsets = deduped
    return sorted(max_idx - o for o in offsets)


def compute_all_scores(
    per_token_scores: torch.Tensor,  # [bsz, kv_heads, num_pages, page_size]
    proxy_scores: dict[str, torch.Tensor],
    output_contrib: torch.Tensor | None,  # [bsz, kv_heads, num_pages] or None
    lambdas: list[float] | None = None,
    comp_size: int | None = None,
    paged_k: torch.Tensor | None = None,        # [bsz, kv_heads, num_pages, page_size, head_dim]
    query_states: torch.Tensor | None = None,    # [bsz, num_heads, 1, head_dim]
    num_kv_groups: int | None = None,
    betas: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all page scoring variants. All use group_agg=max.

    Returns dict of [bsz, kv_heads, num_pages] tensors.
    """
    import numpy as np
    from dct_page_attention import dct, _resolve_frequency_keep_indices

    lambdas = lambdas or [0.5, 1.0, 2.0]
    betas = betas or [0.25, 0.5, 1.0]
    page_size = per_token_scores.shape[-1]

    scores = {
        "oracle_max": per_token_scores.max(dim=-1).values,
        "oracle_mean": per_token_scores.mean(dim=-1),
        "proxy_max": proxy_scores["proxy_max"],
        "proxy_mean": proxy_scores["proxy_mean"],
        "l2_energy": per_token_scores.pow(2).sum(dim=-1).sqrt(),
    }

    # Oracle DC+AC: apply DCT to per-token scores along page_size dim.
    # By linearity: DCT(⟨q, K⟩)[k] = ⟨q, K̃[k]⟩
    dct_scores = dct(per_token_scores.float())  # [bsz, kv_heads, num_pages, page_size]
    dc = dct_scores[..., 0]                          # signed
    ac = dct_scores[..., 1:].pow(2).sum(-1).sqrt()   # unsigned
    for lam in lambdas:
        scores[f"dc_ac_{lam}"] = dc + lam * ac

    # sign_dc_ac: flip AC direction based on sign of DC
    dc_sign = dc.sign()
    for lam in lambdas:
        scores[f"sign_dc_ac_{lam}"] = dc + lam * ac * dc_sign

    # weighted_dc_ac: scale AC by DC magnitude and direction
    for lam in lambdas:
        scores[f"weighted_dc_ac_{lam}"] = dc + lam * ac * dc

    # Proxy DC+AC: use only coefficients 0..comp_size-1 (what the proxy sees)
    if comp_size is not None and comp_size > 1:
        proxy_ac_sq = dct_scores[..., 1:comp_size].pow(2).sum(-1)  # keep squared for reuse
        proxy_ac = proxy_ac_sq.sqrt()
        for lam in lambdas:
            scores[f"proxy_dc_ac_{lam}"] = dc + lam * proxy_ac

        # sign_proxy_dc_ac: proxy AC flipped by sign of DC
        for lam in lambdas:
            scores[f"sign_proxy_dc_ac_{lam}"] = dc + lam * proxy_ac * dc_sign

        # weighted_proxy_dc_ac: proxy AC scaled by DC
        for lam in lambdas:
            scores[f"weighted_proxy_dc_ac_{lam}"] = dc + lam * proxy_ac * dc

        # --- Exp A: Spread Layout Oracle ---
        # Use the same frequency indices as the spread proxy (Exp F) for a fair oracle
        spread_freq_idx = _resolve_frequency_keep_indices(page_size, comp_size, "spread")
        spread_ac_idx = [i for i in spread_freq_idx if i != 0]  # AC only (exclude DC)
        spread_ac = dct_scores[..., spread_ac_idx].pow(2).sum(-1).sqrt()
        for lam in lambdas:
            scores[f"spread_dc_ac_{lam}"] = dc + lam * spread_ac

        # --- Exp F (from proxy): Spread Layout Proxy ---
        for lam in lambdas:
            scores[f"spread_proxy_dc_ac_{lam}"] = proxy_scores[f"spread_proxy_dc_ac_{lam}"]
        scores["spread_proxy_max"] = proxy_scores["spread_proxy_max"]
        scores["spread_proxy_mean"] = proxy_scores["spread_proxy_mean"]

        # --- Exp G: Log-Spread Oracle (dense near low freqs) ---
        log_ac_idx = _log_spread_ac(page_size, comp_size - 1)
        log_ac = dct_scores[..., log_ac_idx].pow(2).sum(-1).sqrt()
        for lam in lambdas:
            scores[f"log_spread_dc_ac_{lam}"] = dc + lam * log_ac

        # --- Exp H: Reverse-Log-Spread Oracle (dense near high freqs) ---
        rlog_ac_idx = _reverse_log_spread_ac(page_size, comp_size - 1)
        rlog_ac = dct_scores[..., rlog_ac_idx].pow(2).sum(-1).sqrt()
        for lam in lambdas:
            scores[f"reverse_log_spread_dc_ac_{lam}"] = dc + lam * rlog_ac

        # --- Exp B: Oracle Score-Space Parseval Correction ---
        # By Parseval: sum_k dct_scores[k]^2 = sum_i per_token_scores[i]^2
        # So exact missing AC energy = l2_energy^2 - sum_{k<c} dct_scores[k]^2
        l2_sq = per_token_scores.pow(2).sum(dim=-1)  # [bsz, kv_heads, num_pages]
        known_sq = dct_scores[..., :comp_size].pow(2).sum(dim=-1)
        residual_score = (l2_sq - known_sq).clamp(min=0)
        for lam in lambdas:
            for beta in betas:
                scores[f"parseval_{lam}_b{beta}"] = dc + lam * (proxy_ac_sq + beta * residual_score).sqrt()

        # --- Exp C: Key-Space Cauchy-Schwarz Correction ---
        # ||q||^2 * (||K||_F^2 - ||K_tilde[:c]||_F^2) bounds missing score energy
        if paged_k is not None and query_states is not None and num_kv_groups is not None:
            bsz, kv_heads = paged_k.shape[:2]
            head_dim = paged_k.shape[-1]
            scaling = head_dim ** -0.5

            # Per-page key Frobenius norm squared: [bsz, kv_heads, num_pages]
            page_fro_sq = paged_k.float().pow(2).sum(dim=(-2, -1))

            # Retained spectral coefficient energy via DCT of keys
            # K_tilde[k] shape: [head_dim] per page — ||K_tilde[k]||^2 summed over k<c
            # DCT along page_size dim: transpose so page_size is last, apply dct, transpose back
            k_float = paged_k.float()  # [bsz, kv_heads, num_pages, page_size, head_dim]
            k_for_dct = k_float.transpose(-2, -1)  # [..., head_dim, page_size]
            k_dct = dct(k_for_dct).transpose(-2, -1)  # [..., page_size, head_dim]
            retained_energy = k_dct[..., :comp_size, :].pow(2).sum(dim=(-2, -1))  # [bsz, kv, pages]
            residual_key = (page_fro_sq - retained_energy).clamp(min=0)

            # q norm squared per kv_head (max across group): [bsz, kv_heads]
            q = query_states.squeeze(2).float()  # [bsz, num_heads, head_dim]
            q_scaled = q * scaling
            q_norm_sq = q_scaled.pow(2).sum(-1)  # [bsz, num_heads]
            q_norm_sq = q_norm_sq.reshape(bsz, kv_heads, num_kv_groups).max(dim=-1).values  # [bsz, kv_heads]

            cs_correction = q_norm_sq.unsqueeze(-1) * residual_key  # [bsz, kv_heads, num_pages]
            for lam in lambdas:
                for beta in betas:
                    scores[f"cs_key_{lam}_b{beta}"] = dc + lam * (proxy_ac_sq + beta * cs_correction).sqrt()

            # --- Exp D: Diagonal Gram Proxy ---
            # diag(K^T K) per page, then q^T diag q estimates l2_energy^2
            diag_gram = k_float.pow(2).sum(dim=3)  # [bsz, kv_heads, num_pages, head_dim]
            q_sq = q_scaled.pow(2).reshape(bsz, kv_heads, num_kv_groups, head_dim)

            # l2_sq estimate: [bsz, kv_heads, num_kv_groups, num_pages]
            l2_sq_est = torch.einsum('bhgd,bhnd->bhgn', q_sq, diag_gram)
            l2_sq_est = l2_sq_est.max(dim=2).values  # group_agg=max: [bsz, kv_heads, num_pages]

            # known_sq is already group-agg'd (from dct_scores which used group_agg'd per_token_scores)
            residual_diag = (l2_sq_est - known_sq).clamp(min=0)
            for lam in lambdas:
                for beta in betas:
                    scores[f"diag_gram_{lam}_b{beta}"] = dc + lam * (proxy_ac_sq + beta * residual_diag).sqrt()

        # --- Exp E: Per-Frequency Discrimination Analysis ---
        # Rank correlation of each DCT coefficient with oracle_max (stored as metadata)
        gt_scores = per_token_scores.max(dim=-1).values  # oracle_max: [bsz, kv_heads, num_pages]
        freq_corrs = []
        for k in range(page_size):
            coeff_k = dct_scores[..., k].abs()  # [bsz, kv_heads, num_pages]
            # Spearman = Pearson on ranks; compute per head then average
            corrs = []
            for b in range(coeff_k.shape[0]):
                for h in range(coeff_k.shape[1]):
                    x = coeff_k[b, h]
                    y = gt_scores[b, h]
                    rx = x.argsort(descending=True).argsort().float()
                    ry = y.argsort(descending=True).argsort().float()
                    corr = torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()
                    corrs.append(corr)
            freq_corrs.append(sum(corrs) / len(corrs))
        # Store as a special entry (not a scoring method)
        scores["_freq_discrimination"] = freq_corrs

    if output_contrib is not None:
        scores["output_contribution"] = output_contrib
    return scores


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyze_method_vs_gt(
    method_scores: torch.Tensor,   # [num_pages]
    gt_scores: torch.Tensor,       # [num_pages]
    gt_topk_idx: set[int],
    actual_top_k: int,
) -> dict[str, Any]:
    """Analyze one method against ground truth for one head."""
    method_topk_idx = set(method_scores.topk(actual_top_k).indices.tolist())
    overlap = len(method_topk_idx & gt_topk_idx)
    false_positives = method_topk_idx - gt_topk_idx
    false_negatives = gt_topk_idx - method_topk_idx

    method_ranks = method_scores.argsort(descending=True).argsort()

    neg_gt = sum(1 for idx in method_topk_idx if gt_scores[idx].item() < 0)
    fp_gt_scores = [gt_scores[idx].item() for idx in false_positives]
    fn_ranks = [method_ranks[idx].item() for idx in false_negatives]
    fn_gt_scores = [gt_scores[idx].item() for idx in false_negatives]

    return {
        "recall": overlap / actual_top_k,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "neg_gt_in_topk": neg_gt,
        "fp_gt_score_mean": (
            sum(fp_gt_scores) / len(fp_gt_scores) if fp_gt_scores else 0.0
        ),
        "fn_rank_mean": (
            sum(fn_ranks) / len(fn_ranks) if fn_ranks else 0.0
        ),
        "fn_gt_score_mean": (
            sum(fn_gt_scores) / len(fn_gt_scores) if fn_gt_scores else 0.0
        ),
    }


def analyze_one_sample(
    all_scores: dict[str, torch.Tensor],  # each [bsz, kv_heads, num_pages]
    ground_truth: str,
    methods: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Analyze all methods vs ground truth for a single sample across all heads."""
    gt = all_scores[ground_truth][0]  # [kv_heads, num_pages]

    kv_heads, num_pages = gt.shape
    actual_top_k = min(top_k, num_pages)

    head_results = []
    for h in range(kv_heads):
        gt_h = gt[h]
        gt_topk_idx = set(gt_h.topk(actual_top_k).indices.tolist())

        head_row = {}
        for method in methods:
            result = _analyze_method_vs_gt(
                all_scores[method][0, h], gt_h, gt_topk_idx, actual_top_k,
            )
            for k, v in result.items():
                head_row[f"{method}_{k}"] = v
        head_results.append(head_row)

    n = len(head_results)
    agg = {
        "num_heads": n,
        "actual_top_k": actual_top_k,
        "num_pages": num_pages,
        "ground_truth": ground_truth,
    }
    metric_keys = [k for k in head_results[0] if isinstance(head_results[0][k], (int, float))]
    for k in metric_keys:
        agg[f"{k}_avg"] = sum(r[k] for r in head_results) / n
    agg["head_results"] = head_results
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose page scoring methods")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--data_root", type=Path, default=Path("benchmark/data/ruler_data"))
    p.add_argument("--output_dir", type=Path, default=Path("results/results_ruler/l2_diagnostic"))
    p.add_argument("--tasks", default="cwe", help="Comma-separated task names or 'all'")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--num_decode_steps", type=int, default=10,
                   help="Number of decode steps to evaluate. Recall is averaged across steps.")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--ground_truths",
        default="oracle_max,output_contribution",
        help="Comma-separated ground truths to loop over (choices: "
             + ", ".join(GROUND_TRUTHS) + ")",
    )

    # DCT page config
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--lambdas", default="0.25,0.5,1.0,1.5,2.0",
                   help="Comma-separated lambda values for DC+AC scoring")
    p.add_argument("--betas", default="0.25,0.5,1.0",
                   help="Comma-separated beta values for residual correction experiments")
    return p.parse_args()


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def load_model(args: argparse.Namespace):
    yarn_kwargs = {}
    if "qwen3" in args.model_name_or_path.lower():
        yarn_kwargs = {
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            "max_position_embeddings": 131072,
        }
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.cuda_device},
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
        **yarn_kwargs,
    ).eval()


def resolve_model_family(model_name_or_path: str) -> str:
    name = model_name_or_path.lower().split("/")[-1]
    if "qwen3" in name:
        return "qwen3"
    elif "qwen2" in name:
        return "qwen2"
    elif "llama-3" in name or "llama3" in name:
        return "llama3"
    elif "llama" in name:
        return "llama"
    else:
        return name


def cleanup_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_ground_truths(value: str) -> list[str]:
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in GROUND_TRUTHS]
    if unknown:
        raise ValueError(f"Unknown ground truths: {unknown}. Choose from {GROUND_TRUTHS}")
    return requested


def _score_one_layer(
    attn_module,
    hidden_states: torch.Tensor,  # [bsz, 1, hidden_dim] (post-layernorm)
    key_states: torch.Tensor,     # [bsz, kv_heads, kv_len, head_dim]
    value_states: torch.Tensor,   # [bsz, kv_heads, kv_len, head_dim]
    config,
    num_kv_groups: int,
    comp_size: int,
    page_size: int,
    sink_size: int,
    recent_size: int,
    top_k: int,
    lambdas: list[float],
    betas: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    """Compute all scoring methods for one layer at one decode step.

    Returns all_scores dict (values are [bsz, kv_heads, num_pages] tensors),
    or None if there are not enough tokens for at least one page.
    """
    kv_len = key_states.shape[2]
    pageable_len = kv_len - sink_size - recent_size
    if pageable_len < page_size:
        return None

    sink_k, sink_v, paged_k, paged_v, recent_k, recent_v, num_pages = \
        segment_kv_from_cache(key_states, value_states, page_size, sink_size, recent_size)

    query_states = recompute_query(attn_module, hidden_states, kv_len, config, device, dtype)

    per_token_scores = compute_per_token_scores(query_states, paged_k, num_kv_groups)
    proxy_scores = compute_proxy_scores(
        query_states, paged_k, num_kv_groups, comp_size, lambdas=lambdas,
    )
    output_contrib = compute_output_contribution(
        query_states, paged_k, paged_v, num_kv_groups,
        sink_k=sink_k, sink_v=sink_v, recent_k=recent_k, recent_v=recent_v,
    )
    all_scores = compute_all_scores(
        per_token_scores, proxy_scores, output_contrib, lambdas,
        comp_size=comp_size, paged_k=paged_k,
        query_states=query_states, num_kv_groups=num_kv_groups, betas=betas,
    )
    return all_scores


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    gt_names = resolve_ground_truths(args.ground_truths)
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]

    all_methods = ["oracle_max", "oracle_mean", "proxy_max", "proxy_mean", "l2_energy",
                   "output_contribution"]
    all_methods += [f"dc_ac_{lam}" for lam in lambdas]
    all_methods += [f"sign_dc_ac_{lam}" for lam in lambdas]
    all_methods += [f"weighted_dc_ac_{lam}" for lam in lambdas]
    comp_size = max(1, int(args.page_size * args.compress_ratio))
    if comp_size > 1:
        all_methods += [f"proxy_dc_ac_{lam}" for lam in lambdas]
        all_methods += [f"sign_proxy_dc_ac_{lam}" for lam in lambdas]
        all_methods += [f"weighted_proxy_dc_ac_{lam}" for lam in lambdas]
        # Exp A: spread layout oracle
        all_methods += [f"spread_dc_ac_{lam}" for lam in lambdas]
        # Exp B: Parseval correction
        all_methods += [f"parseval_{lam}_b{beta}" for lam in lambdas for beta in betas]
        # Exp C: key-space CS correction
        all_methods += [f"cs_key_{lam}_b{beta}" for lam in lambdas for beta in betas]
        # Exp D: diagonal Gram proxy
        all_methods += [f"diag_gram_{lam}_b{beta}" for lam in lambdas for beta in betas]
        # Exp F: spread proxy DC+AC
        all_methods += [f"spread_proxy_dc_ac_{lam}" for lam in lambdas]
        all_methods += ["spread_proxy_max", "spread_proxy_mean"]
        # Exp G: log-spread oracle (dense near low freqs)
        all_methods += [f"log_spread_dc_ac_{lam}" for lam in lambdas]
        # Exp H: reverse-log-spread oracle (dense near high freqs)
        all_methods += [f"reverse_log_spread_dc_ac_{lam}" for lam in lambdas]

    if comp_size > 1:
        from dct_page_attention import _resolve_frequency_keep_indices
        print(f"Frequency indices (comp_size={comp_size}, page_size={args.page_size}):")
        print(f"  low (proxy_dc_ac):              [0] + {list(range(1, comp_size))}")
        print(f"  spread (spread_dc_ac):          {_resolve_frequency_keep_indices(args.page_size, comp_size, 'spread')}")
        print(f"  log_spread (log_spread_dc_ac):  [0] + {_log_spread_ac(args.page_size, comp_size - 1)}")
        print(f"  reverse_log (rev_log_dc_ac):    [0] + {_reverse_log_spread_ac(args.page_size, comp_size - 1)}")

    print(f"Full attention generation, {args.num_decode_steps} decode steps, scoring all layers")
    print(f"Ground truths: {gt_names}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name_or_path}")
    model = load_model(args)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_layers = model.config.num_hidden_layers
    num_kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads
    print(f"  num_layers={num_layers}, num_kv_groups={num_kv_groups}")

    # Create output dirs for each ground truth
    run_dirs = {}
    for gt_name in gt_names:
        run_dir = args.output_dir / f"ps{args.page_size}_topk{args.top_k}_cr{args.compress_ratio}_gt_{gt_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        run_dirs[gt_name] = run_dir

    try:
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"TASK: {task}")
            print(f"{'='*60}")

            model_family = resolve_model_family(args.model_name_or_path)
            data_path = args.data_root / model_family / str(args.context_len) / task / "validation.jsonl"
            if not data_path.exists():
                print(f"  WARNING: data not found at {data_path}, skipping")
                continue

            with data_path.open("r", encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]
            samples = samples[:args.num_samples]

            # scored_samples: list of (sample, mean_analysis_per_gt)
            # mean_analysis_per_gt: dict[gt_name -> analysis_dict]
            scored_samples: list[tuple[dict, dict[str, dict]]] = []
            freq_data_all: list[list[float]] = []

            for idx, sample in enumerate(tqdm(samples, desc=f"  {task}"), start=1):
                encoded = tokenizer(sample["input"], return_tensors="pt")
                input_ids = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)
                prefill_len = input_ids.shape[1]

                # --- Prefill (full attention, no patch) ---
                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

                # Accumulate per-(step, layer) analyses, keyed by gt_name
                # gt_name -> list of analysis dicts (one per step×layer observation)
                observations: dict[str, list[dict]] = {gt: [] for gt in gt_names}
                step_freq_data: list[list[float]] = []

                # --- Decode steps ---
                for step in range(args.num_decode_steps):
                    cache_position = torch.tensor([prefill_len + step], device=device)
                    with torch.no_grad():
                        outputs = model(
                            next_token,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_position,
                            output_hidden_states=True,
                        )
                    past_key_values = outputs.past_key_values
                    next_token = outputs.logits[:, -1:].argmax(dim=-1)

                    # Score every layer at this decode step
                    for layer_idx in range(num_layers):
                        # hidden_states[layer_idx] is input to layer layer_idx (pre-layernorm)
                        h = outputs.hidden_states[layer_idx][:, -1:, :]
                        h = model.model.layers[layer_idx].input_layernorm(h)

                        attn = model.model.layers[layer_idx].self_attn
                        key_cache = past_key_values.key_cache[layer_idx]
                        value_cache = past_key_values.value_cache[layer_idx]

                        all_scores = _score_one_layer(
                            attn, h, key_cache, value_cache,
                            model.config, num_kv_groups, comp_size,
                            args.page_size, args.sink_size, args.recent_size,
                            args.top_k, lambdas, betas, device, dtype,
                        )
                        if all_scores is None:
                            continue

                        # Collect freq discrimination data
                        fd = all_scores.get("_freq_discrimination")
                        if fd is not None:
                            step_freq_data.append(fd)

                        num_pages = all_scores["oracle_max"].shape[-1]
                        actual_top_k = min(args.top_k, num_pages)

                        for gt_name in gt_names:
                            methods = [m for m in all_methods if m != gt_name]
                            analysis = analyze_one_sample(
                                all_scores, gt_name, methods, actual_top_k,
                            )
                            observations[gt_name].append(analysis)

                # Free KV cache for this sample
                del past_key_values, outputs
                torch.cuda.empty_cache()

                if step_freq_data:
                    freq_data_all.extend(step_freq_data)

                # Average across steps×layers for each GT
                mean_per_gt: dict[str, dict] = {}
                for gt_name in gt_names:
                    obs = observations[gt_name]
                    if not obs:
                        continue
                    n_obs = len(obs)
                    avg_keys = [k for k in obs[0] if k.endswith("_avg")]
                    agg: dict[str, Any] = {
                        "sample_index": sample["index"],
                        "num_observations": n_obs,
                        "num_decode_steps": args.num_decode_steps,
                        "num_layers": num_layers,
                        "num_heads": obs[0]["num_heads"],
                        "num_pages": obs[0]["num_pages"],
                        "actual_top_k": obs[0]["actual_top_k"],
                        "ground_truth": gt_name,
                    }
                    for k in avg_keys:
                        agg[k] = sum(o[k] for o in obs) / n_obs
                    mean_per_gt[gt_name] = agg

                if mean_per_gt:
                    scored_samples.append((sample, mean_per_gt))

            if not scored_samples:
                print(f"  No results for {task}")
                continue

            # --- Exp E: Print per-frequency discrimination ---
            if freq_data_all:
                import numpy as _np
                avg_corr = _np.mean(freq_data_all, axis=0)
                print(f"\n  Per-frequency Spearman correlation with oracle_max (task={task}):")
                print(f"    {'freq':>4s}  {'corr':>6s}")
                for k, c in enumerate(avg_corr):
                    marker = " *" if k < comp_size else ""
                    print(f"    {k:4d}  {c:+.4f}{marker}")
                print(f"    (* = retained by proxy with comp_size={comp_size})")

            # Analyze against each ground truth
            for gt_name in gt_names:
                methods = [m for m in all_methods if m != gt_name]
                sample_results = []

                for i, (sample, mean_per_gt) in enumerate(scored_samples, start=1):
                    if gt_name not in mean_per_gt:
                        continue
                    analysis = mean_per_gt[gt_name]
                    sample_results.append(analysis)

                    if i % 5 == 0 or i == len(scored_samples):
                        recalls = " ".join(
                            f"{m}={analysis[f'{m}_recall_avg']:.3f}" for m in methods
                        )
                        print(f"  [{i}/{len(scored_samples)}] gt={gt_name} recall: {recalls}")

                if not sample_results:
                    continue

                # Aggregate across samples
                n = len(sample_results)
                task_summary: dict[str, Any] = {
                    "task": task,
                    "ground_truth": gt_name,
                    "num_samples": n,
                    "num_decode_steps": args.num_decode_steps,
                    "num_layers": num_layers,
                    "num_pages": sample_results[0]["num_pages"],
                    "actual_top_k": sample_results[0]["actual_top_k"],
                }
                avg_keys = [k for k in sample_results[0] if k.endswith("_avg")]
                for k in avg_keys:
                    task_summary[k] = sum(r[k] for r in sample_results) / n

                print(f"\n  === {task} Summary (vs {gt_name}, group_agg=max, "
                      f"{args.num_decode_steps} steps x {num_layers} layers) ===")
                print(f"  Pages: {task_summary['num_pages']}, Top-k: {task_summary['actual_top_k']}")
                for m in methods:
                    print(
                        f"  {m:30s}  "
                        f"recall={task_summary[f'{m}_recall_avg']:.3f}  "
                        f"FP={task_summary[f'{m}_false_positive_count_avg']:.1f}  "
                        f"FN={task_summary[f'{m}_false_negative_count_avg']:.1f}  "
                        f"neg_gt={task_summary[f'{m}_neg_gt_in_topk_avg']:.2f}  "
                        f"FN_rank={task_summary[f'{m}_fn_rank_mean_avg']:.1f}  "
                        f"FN_gt_score={task_summary[f'{m}_fn_gt_score_mean_avg']:.4f}"
                    )

                run_dir = run_dirs[gt_name]
                output = {
                    "summary": task_summary,
                    "samples": [
                        {k: v for k, v in r.items() if k != "head_results"}
                        for r in sample_results
                    ],
                }
                (run_dir / f"{task}.json").write_text(
                    json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"  Results saved to: {run_dir / f'{task}.json'}")

    finally:
        cleanup_model(model)

    print(f"\nAll results in: {[str(d) for d in run_dirs.values()]}")


if __name__ == "__main__":
    main()
