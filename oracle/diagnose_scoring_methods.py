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

  Base methods compared against ground truth:
    oracle_max:    full tokens, scoring=max
    oracle_mean:   full tokens, scoring=mean         mean_i <q, k_i>
    proxy_max:     DCT compressed tokens, scoring=max   max_c <q, comp_k_c>
    proxy_mean:    DCT compressed tokens, scoring=mean  mean_c <q, comp_k_c>
    l2_energy:     full tokens, L2 scoring           sqrt(sum_i <q, k_i>^2)
    dc_ac_{lam}, proxy_dc_ac_{lam}, spread_dc_ac_{lam}, ... (see compute_all_scores)

  Codex spectral reconstruction variants:
    spectral_recon_max
      Reconstruct the low-pass score curve on the original P token positions
      from the c retained DCT coefficients, take the max over those positions.
          shat[t] = sum_{m=1..c} a_m * phi_{j_m}(t)
          score(page) = max_t shat[t]

    spectral_recon_ucb{1,2,3}_b{beta}
      Add a position-dependent UCB bonus from omitted-frequency energy.
        level 1: u[t] = sigma_res         (constant per page; pure scalar shift)
        level 2: u[t]^2 = sum_band E_band * g_band(t)^2
        level 3: u[t]^2 = sum_{j not kept} Var(b_j) * phi_j(t)^2

    continuous_cosine_max
      Same retained coefficients, but maximize the continuous cosine sum over
      a dense (sub-integer) grid of X = continuous_multiplier * P samples.

    hybrid_*
      Budgeted scoring: (c - M) DCT coefficients + M highlight tokens, where
      highlights are chosen by K-norm, AC-energy, or query-aware residual oracle.

For each method reports (averaged across decode steps, layers, heads):
  - recall: fraction of GT top-k pages also selected by this method
  - false positives / false negatives
  - neg_gt_in_topk: pages in method's top-k with negative GT score
  - fn_rank: average rank (in this method) of missed GT pages
  - fn_gt_score: average GT score of missed pages

Run --self_test for numerical sanity checks (no model required).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import torch
import torch.nn.functional as F

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

GROUND_TRUTHS = ["oracle_max", "output_contribution"]

_BASIS_CACHE: dict = {}
_CIDCT_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Local helpers (used by codex spectral methods; avoid heavy imports for --self_test)
# ---------------------------------------------------------------------------
def _dct_local(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II along the last dim, identical to dct_page_attention.dct.

    Inlined so this file imports without pulling in dct_page_attention (which
    imports transformers at module top). At runtime inside main(), the
    dct_page_attention module is also loaded for other reasons; both
    implementations agree.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v.to(torch.float32), dim=1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i

    V[:, 0] /= np.sqrt(N) * 2
    V[:, 1:] /= np.sqrt(N / 2) * 2
    return (2 * V).view(*x_shape)


def _normalize_frequency_indices_local(indices, seq_len, comp_size):
    """Dedupe + clamp + pad to length comp_size, mirroring dct_page_attention's helper."""
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        idx = int(max(0, min(seq_len - 1, idx)))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    fill = 0
    while len(out) < comp_size and fill < seq_len:
        if fill not in seen:
            out.append(fill)
            seen.add(fill)
        fill += 1
    return out[:comp_size]


def _resolve_frequency_keep_indices_local(seq_len, comp_size, layout):
    """Local copy of dct_page_attention._resolve_frequency_keep_indices."""
    if comp_size >= seq_len or layout == "low":
        return list(range(min(comp_size, seq_len)))
    if layout == "low_high":
        low_count = (comp_size + 1) // 2
        high_count = comp_size - low_count
        indices = list(range(low_count)) + list(range(seq_len - high_count, seq_len))
        return _normalize_frequency_indices_local(indices, seq_len, comp_size)
    if layout == "low_mid_high":
        low_count = min(2, comp_size)
        remaining = comp_size - low_count
        indices = list(range(low_count))
        if remaining > 0:
            tail = np.linspace(seq_len // 2, seq_len - 1, num=remaining, dtype=int).tolist()
            indices.extend(tail)
        return _normalize_frequency_indices_local(indices, seq_len, comp_size)
    if layout == "spread":
        indices = np.linspace(0, seq_len - 1, num=comp_size, dtype=int).tolist()
        return _normalize_frequency_indices_local(indices, seq_len, comp_size)
    raise ValueError(f"Unsupported proxy_frequency_layout: {layout}")


def _get_cidct_matrix(c: int, device: torch.device) -> torch.Tensor:
    """Return cached c×c IDCT matrix (orthonormal DCT-III).

    Treats c coefficient scores as a c-length frequency-domain signal and
    transforms them to c time-domain samples via inverse DCT.  This allows
    constructive interference between coefficients without expanding to
    page_size.
    """
    key = (c, str(device))
    if key not in _CIDCT_CACHE:
        I_c = torch.eye(c, device=device, dtype=torch.float32)
        _CIDCT_CACHE[key] = _dct_local(I_c)  # [c, c]: IDCT basis
    return _CIDCT_CACHE[key]


def _get_codex_basis(page_size, comp_size, layout, X, device, dtype):
    """Build / cache synthesis matrices and frequency index lists for one (P,c,layout)."""
    P = int(page_size)
    c = max(1, min(int(comp_size), P))
    X = max(1, int(X))
    key = (P, c, layout, X, str(device), str(dtype))
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]

    J = list(_resolve_frequency_keep_indices_local(P, c, layout))
    not_J = sorted(set(range(P)) - set(J))

    I = torch.eye(P, device=device, dtype=torch.float32)
    D_mat = _dct_local(I)  # [P, P]; D_mat[t, j] = phi_j(t)

    J_tensor = torch.tensor(J, device=device, dtype=torch.long)
    notJ_tensor = (
        torch.tensor(not_J, device=device, dtype=torch.long) if not_J else None
    )

    Phi_kept_T = D_mat.index_select(1, J_tensor).contiguous()  # [P, c]
    Phi_notJ_T = (
        D_mat.index_select(1, notJ_tensor).contiguous() if not_J else None
    )

    xs = torch.linspace(0.0, P - 1.0, X, device=device, dtype=torch.float32)
    Phi_dense_kept = torch.empty(X, len(J), device=device, dtype=torch.float32)
    for m, j in enumerate(J):
        if j == 0:
            Phi_dense_kept[:, m] = 1.0 / math.sqrt(P)
        else:
            Phi_dense_kept[:, m] = math.sqrt(2.0 / P) * torch.cos(
                math.pi * (2.0 * xs + 1.0) * float(j) / (2.0 * P)
            )

    result = (J, not_J, J_tensor, notJ_tensor, Phi_kept_T, Phi_notJ_T, Phi_dense_kept)
    _BASIS_CACHE[key] = result
    return result


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

    Returns dict with proxy_max and proxy_mean.
    All values are [bsz, kv_heads, num_pages] float32.
    """
    from dct_page_attention import _build_dct_projection_matrix

    lambdas = lambdas or [0.5, 1.0, 2.0]
    bsz, num_heads, _, head_dim = query_states.shape
    _, kv_heads, num_pages, page_size, _ = paged_k.shape
    scaling = head_dim ** -0.5

    M = _build_dct_projection_matrix(page_size, comp_size, paged_k.device, paged_k.dtype)

    comp_k = torch.einsum('cs,bhnsd->bhncd', M.float(), paged_k.float())

    q = query_states.squeeze(2).reshape(bsz, kv_heads, num_kv_groups, head_dim).float()

    # dots: [bsz, kv_heads, num_kv_groups, num_pages, comp_size]
    dots = torch.einsum('bhgd,bhncd->bhgnc', q * scaling, comp_k)

    # group_agg=max
    dots_gagg = dots.max(dim=2).values  # [bsz, kv_heads, num_pages, comp_size]

    proxy_max = dots_gagg.max(dim=-1).values
    proxy_mean = dots_gagg.mean(dim=-1)

    return {"proxy_max": proxy_max, "proxy_mean": proxy_mean}


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
    from dct_page_attention import dct

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

    # --- L∞ aggregation of AC coefficients (oracle: all AC frequencies) ---
    # Tests max single-frequency projection vs L2 energy. Both unscaled and
    # spike-recovery scaled (√(M)) variants — if AC is Gaussian-like the
    # unscaled wins, if time-sparse the scaled wins.
    ac_all = dct_scores[..., 1:]  # [bsz, kv, num_pages, page_size - 1]
    signed_max_ac = ac_all.max(dim=-1).values          # signed max
    abs_max_ac = ac_all.abs().max(dim=-1).values       # magnitude max
    relu_max_ac = signed_max_ac.clamp(min=0)           # positive-only max
    oracle_scale = float(page_size - 1) ** 0.5         # √(N-1) spike-recovery factor
    for lam in lambdas:
        scores[f"signed_max_ac_{lam}"] = dc + lam * signed_max_ac
        scores[f"relu_max_ac_{lam}"] = dc + lam * relu_max_ac
        scores[f"abs_max_ac_{lam}"] = dc + lam * abs_max_ac
        scores[f"signed_max_ac_scaled_{lam}"] = dc + lam * oracle_scale * signed_max_ac
        scores[f"relu_max_ac_scaled_{lam}"] = dc + lam * oracle_scale * relu_max_ac
        scores[f"abs_max_ac_scaled_{lam}"] = dc + lam * oracle_scale * abs_max_ac

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

        # --- L∞ aggregation of proxy AC coefficients (comp_size-1 freqs only) ---
        proxy_ac_all = dct_scores[..., 1:comp_size]  # [bsz, kv, N, comp_size - 1]
        proxy_signed_max = proxy_ac_all.max(dim=-1).values
        proxy_abs_max = proxy_ac_all.abs().max(dim=-1).values
        proxy_relu_max = proxy_signed_max.clamp(min=0)
        proxy_scale = float(max(comp_size - 1, 1)) ** 0.5  # √(comp_size-1)
        for lam in lambdas:
            scores[f"proxy_signed_max_ac_{lam}"] = dc + lam * proxy_signed_max
            scores[f"proxy_relu_max_ac_{lam}"] = dc + lam * proxy_relu_max
            scores[f"proxy_abs_max_ac_{lam}"] = dc + lam * proxy_abs_max
            scores[f"proxy_signed_max_ac_scaled_{lam}"] = dc + lam * proxy_scale * proxy_signed_max
            scores[f"proxy_relu_max_ac_scaled_{lam}"] = dc + lam * proxy_scale * proxy_relu_max
            scores[f"proxy_abs_max_ac_scaled_{lam}"] = dc + lam * proxy_scale * proxy_abs_max

        # --- Exp A: Spread Layout Oracle ---
        # Use the same frequency indices as the spread proxy (Exp F) for a fair oracle
        spread_freq_idx = _resolve_frequency_keep_indices_local(page_size, comp_size, "spread")
        spread_ac_idx = [i for i in spread_freq_idx if i != 0]  # AC only (exclude DC)
        spread_ac = dct_scores[..., spread_ac_idx].pow(2).sum(-1).sqrt()
        for lam in lambdas:
            scores[f"spread_dc_ac_{lam}"] = dc + lam * spread_ac

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

            # known_sq: energy of retained DCT score coefficients (group-agg'd via per_token_scores)
            known_sq = dct_scores[..., :comp_size].pow(2).sum(dim=-1)  # [bsz, kv_heads, num_pages]
            residual_diag = (l2_sq_est - known_sq).clamp(min=0)
            for lam in lambdas:
                for beta in betas:
                    scores[f"diag_gram_{lam}_b{beta}"] = dc + lam * (proxy_ac_sq + beta * residual_diag).sqrt()

        # --- Exp E: Per-Frequency Discrimination Analysis ---
        # Vectorized Spearman across all (b, h, freq): Pearson on ranks reduces to
        # cov / var_pop since rank vectors of length N have constant variance (N^2-1)/12.
        gt_scores = per_token_scores.max(dim=-1).values  # [bsz, kv_heads, num_pages]
        coeff_abs = dct_scores.abs().permute(0, 1, 3, 2)  # [bsz, kv_heads, page_size, num_pages]
        N = coeff_abs.shape[-1]
        rx = coeff_abs.argsort(dim=-1, descending=True).argsort(dim=-1).float()  # [bsz, kv, F, N]
        ry = gt_scores.argsort(dim=-1, descending=True).argsort(dim=-1).float()  # [bsz, kv, N]
        center = (N - 1) / 2
        var_pop = (N * N - 1) / 12
        corrs_bhf = ((rx - center) * (ry - center).unsqueeze(-2)).mean(dim=-1) / var_pop  # [bsz, kv, F]
        scores["_freq_discrimination"] = corrs_bhf.mean(dim=(0, 1)).cpu().tolist()

        # --- Approach 3: DCT Residual Energy Profile (Query-Adaptive) ---
        # Instead of isotropic correction (Exp C: ||q||^2 * scalar), use
        # per-dimension missing energy: diag(C_missing)[d] = sum_{k>=c} K_tilde[k][d]^2
        # Then: missing_AC^2 = sum_d q_d^2 * diag(C_missing)[d]  (diagonal approx of q^T C_missing q)
        if paged_k is not None and query_states is not None and num_kv_groups is not None:
            # k_dct already computed in Exp C: [bsz, kv_heads, num_pages, page_size, head_dim]
            # Per-dimension residual energy from missing DCT coefficients
            residual_diag_profile = k_dct[..., comp_size:, :].pow(2).sum(dim=-2)  # [bsz, kv, N, head_dim]

            # DC scores per group head (not yet group-agg'd)
            q_r = q_scaled.reshape(bsz, kv_heads, num_kv_groups, head_dim)
            dc_per_group = torch.einsum('bhgd,bhnd->bhgn', q_r, k_dct[..., 0, :])  # [bsz, kv, G, N]

            # Query-adaptive missing AC: sqrt(sum_d q_d^2 * residual_diag[d])
            q_r_sq = q_r.pow(2)  # [bsz, kv, G, head_dim]
            missing_ac_sq = torch.einsum('bhgd,bhnd->bhgn', q_r_sq, residual_diag_profile)  # [bsz, kv, G, N]
            missing_ac = missing_ac_sq.clamp(min=0).sqrt()

            for lam in lambdas:
                scores[f"residual_profile_{lam}"] = (dc_per_group + lam * missing_ac).max(dim=2).values

        # --- Approach 5: Adaptive DCT Frequency Selection Per Page ---
        # Keep DC + the AC coefficient with the most key-space energy per page
        if paged_k is not None and query_states is not None and num_kv_groups is not None:
            # Per-frequency AC key energy: ||K_tilde[k]||^2 for k >= 1
            ac_key_energies = k_dct[..., 1:, :].pow(2).sum(dim=-1)  # [bsz, kv, N, page_size-1]
            best_ac_rel = ac_key_energies.argmax(dim=-1)  # [bsz, kv, N]

            # Gather best AC key vector per page
            # k_dct shape: [bsz, kv, N, page_size, head_dim]
            best_ac_abs = best_ac_rel + 1  # shift to absolute frequency index
            best_ac_idx = best_ac_abs.unsqueeze(-1).unsqueeze(-1).expand(
                *best_ac_abs.shape, 1, head_dim
            )  # [bsz, kv, N, 1, head_dim]
            k_best_ac = k_dct.gather(-2, best_ac_idx).squeeze(-2)  # [bsz, kv, N, head_dim]

            # q_r already computed above: [bsz, kv, G, head_dim]
            ac_best_score = torch.einsum('bhgd,bhnd->bhgn', q_r, k_best_ac).abs()  # [bsz, kv, G, N]
            for lam in lambdas:
                scores[f"adaptive_freq_{lam}"] = (dc_per_group + lam * ac_best_score).max(dim=2).values

        # --- Approach 6: Refined Cauchy-Schwarz Beta ---
        # Exp C uses beta in [0.25, 0.5, 1.0], but E[(q·K_tilde[k])^2] = ||q||^2 * ||K_tilde[k]||^2 / head_dim
        # => beta_optimal ~ 1/head_dim ~ 0.008 for head_dim=128. Current betas overestimate by 30-125x.
        if paged_k is not None and query_states is not None and num_kv_groups is not None:
            refined_betas = [0.004, 0.008, 0.012, 0.016, 0.023]
            for lam in lambdas:
                for rbeta in refined_betas:
                    scores[f"cs_refined_{lam}_b{rbeta}"] = dc + lam * (proxy_ac_sq + rbeta * cs_correction).sqrt()

    if output_contrib is not None:
        scores["output_contribution"] = output_contrib
    return scores


# ---------------------------------------------------------------------------
# Codex spectral reconstruction methods
# ---------------------------------------------------------------------------
def _compute_codex_spectral_scores(
    paged_k: torch.Tensor,        # [B, H, N, P, D]
    query_states: torch.Tensor,   # [B, Hq, 1, D]
    num_kv_groups: int,
    comp_size: int,
    layout: str,
    betas: list[float],
    ucb_levels: list[int],
    ucb_num_bands: int,
    continuous_multiplier: int,
    name_suffix: str = "",
) -> dict[str, torch.Tensor]:
    """Compute spectral_recon_max, spectral_recon_ucb{1,2,3}_b{beta}, continuous_cosine_max.

    `name_suffix` is appended after the method base name (e.g. "_spread") so that
    multi-layout runs can register layout-specific variants side-by-side.
    UCB names get the suffix between the level and the beta tag, e.g.
    `spectral_recon_ucb1_spread_b0.5`.
    """
    out: dict[str, torch.Tensor] = {}
    if comp_size is None or comp_size <= 0 or paged_k is None or query_states is None:
        return out

    B, H, N, P, D = paged_k.shape
    G = int(num_kv_groups)
    c = max(1, min(int(comp_size), P))
    X = max(1, int(continuous_multiplier)) * P

    layout = layout or "low"
    device = paged_k.device

    (
        J,
        not_J,
        J_tensor,
        notJ_tensor,
        Phi_kept_T,
        Phi_notJ_T,
        Phi_dense_kept,
    ) = _get_codex_basis(P, c, layout, X, device, torch.float32)

    k_float = paged_k.float()
    k_for_dct = k_float.transpose(-2, -1)               # [B, H, N, D, P]
    k_dct = _dct_local(k_for_dct).transpose(-2, -1)     # [B, H, N, P, D]

    scaling = D ** -0.5
    q = query_states.squeeze(2).reshape(B, H, G, D).float()
    q_scaled = q * scaling                              # [B, H, G, D]

    k_dct_kept = k_dct.index_select(-2, J_tensor)       # [B, H, N, c, D]
    a = torch.einsum('bhgd,bhncd->bhgnc', q_scaled, k_dct_kept)  # [B, H, G, N, c]

    sfx = name_suffix

    # --- Method 1: spectral_recon_max ---
    shat = torch.einsum('tc,bhgnc->bhgnt', Phi_kept_T, a)        # [B, H, G, N, P]
    shat_max_per_group = shat.max(dim=-1).values                  # [B, H, G, N]
    out[f'spectral_recon_max{sfx}'] = shat_max_per_group.max(dim=2).values  # [B, H, N]

    # --- Method 3: continuous_cosine_max ---
    shat_cont = torch.einsum('xc,bhgnc->bhgnx', Phi_dense_kept, a)  # [B, H, G, N, X]
    out[f'continuous_cosine_max{sfx}'] = (
        shat_cont.max(dim=-1).values.max(dim=2).values
    )

    # --- Method 2: spectral_recon_ucb (3 levels) ---
    if not_J and ucb_levels:
        k_dct_notJ = k_dct.index_select(-2, notJ_tensor)          # [B, H, N, P-c, D]
        q_sq = q_scaled.pow(2)                                     # [B, H, G, D]
        var_b = torch.einsum('bhgd,bhnjd->bhgnj', q_sq, k_dct_notJ.pow(2))
        # var_b: [B, H, G, N, P-c]

        betas_t = torch.tensor(betas, device=device, dtype=torch.float32)  # [Bet]

        if 1 in ucb_levels:
            sigma_res = var_b.sum(dim=-1).clamp(min=0).sqrt()      # [B, H, G, N]
            combined = (
                shat_max_per_group.unsqueeze(-1)
                + sigma_res.unsqueeze(-1) * betas_t
            )  # [B, H, G, N, Bet]
            v1 = combined.max(dim=2).values                        # [B, H, N, Bet]
            for bi, beta in enumerate(betas):
                out[f"spectral_recon_ucb1{sfx}_b{beta}"] = v1[..., bi]

        if 3 in ucb_levels:
            Phi_notJ_T_sq = Phi_notJ_T.pow(2)                      # [P, P-c]
            u_t_sq = torch.einsum(
                'bhgnj,tj->bhgnt', var_b, Phi_notJ_T_sq
            ).clamp(min=0)
            u_t = u_t_sq.sqrt()                                    # [B, H, G, N, P]
            score_curves = shat.unsqueeze(-1) + u_t.unsqueeze(-1) * betas_t
            v3 = score_curves.max(dim=-2).values.max(dim=2).values  # [B, H, N, Bet]
            for bi, beta in enumerate(betas):
                out[f"spectral_recon_ucb3{sfx}_b{beta}"] = v3[..., bi]

        if 2 in ucb_levels:
            num_dropped = len(not_J)
            num_bands = max(1, min(int(ucb_num_bands), num_dropped))
            edges = np.linspace(0, num_dropped, num_bands + 1, dtype=int)
            bands = [
                (int(edges[i]), int(edges[i + 1]))
                for i in range(num_bands)
                if int(edges[i + 1]) > int(edges[i])
            ]
            E_band = torch.stack(
                [var_b[..., lo:hi].sum(dim=-1) for (lo, hi) in bands], dim=-1
            )  # [B, H, G, N, num_bands]
            g_band_sq = torch.zeros(P, len(bands), device=device, dtype=torch.float32)
            for bi_, (lo, hi) in enumerate(bands):
                g_band_sq[:, bi_] = Phi_notJ_T[:, lo:hi].pow(2).mean(dim=-1)
            u_t_sq_l2 = torch.einsum(
                'bhgne,te->bhgnt', E_band, g_band_sq
            ).clamp(min=0)
            u_t_l2 = u_t_sq_l2.sqrt()
            score_curves_l2 = shat.unsqueeze(-1) + u_t_l2.unsqueeze(-1) * betas_t
            v2 = score_curves_l2.max(dim=-2).values.max(dim=2).values
            for bi, beta in enumerate(betas):
                out[f"spectral_recon_ucb2{sfx}_b{beta}"] = v2[..., bi]
    else:
        # Empty notJ: UCB variants degenerate to spectral_recon_max
        for lvl in ucb_levels or []:
            for beta in betas:
                out[f"spectral_recon_ucb{lvl}{sfx}_b{beta}"] = out[f'spectral_recon_max{sfx}']

    return out


def _compute_codex_hybrid_scores(
    paged_k: torch.Tensor | None,
    query_states: torch.Tensor | None,
    num_kv_groups: int | None,
    comp_size: int | None,
    layout: str,
    alphas: list[float],
    continuous_multiplier: int,
    multi_highlights: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Hybrid storage budget: (c-1) DCT coefficients + highlight token(s).

    Two query-independent highlight heuristics (single token):
      highlight  — K-norm argmax: argmax_t(||k_t||²)
      specresid  — spectral residual norm: argmax_t(||k_t - k_recon||²)

    Multi-highlight (M>1 tokens, K-norm top-M):
      hybrid_multi{M}  — (c-M) DCT + M tokens

    Oracle (query-aware residual argmax, heuristic-independent):
      hybrid_highlight_{max,add}_oracle

    All swept over alpha (spectral scaling factor). Highlight is the anchor;
    alpha controls how much to trust the smooth reconstruction.
    """
    out: dict[str, torch.Tensor] = {}
    if (
        paged_k is None or query_states is None
        or num_kv_groups is None or comp_size is None or comp_size <= 0
        or not alphas
    ):
        return out

    B, H, N, P, D = paged_k.shape
    G = int(num_kv_groups)
    c = max(1, min(int(comp_size), P))
    c_lo = max(1, c - 1)
    device = paged_k.device
    scaling = D ** -0.5
    X = max(1, int(continuous_multiplier)) * P

    k_f = paged_k.float()
    q = query_states.squeeze(2).reshape(B, H, G, D).float() * scaling

    k_for_dct = k_f.transpose(-2, -1)
    k_dct = _dct_local(k_for_dct).transpose(-2, -1)

    alphas_t = torch.tensor(alphas, device=device, dtype=torch.float32)

    def _spectral_curve(c_used):
        _, _, J_tensor, _, Phi_kept_T, _, _ = _get_codex_basis(
            P, c_used, layout, X, device, torch.float32
        )
        k_dct_kept = k_dct.index_select(-2, J_tensor)
        a = torch.einsum('bhgd,bhncd->bhgnc', q, k_dct_kept)
        shat = torch.einsum('tc,bhgnc->bhgnt', Phi_kept_T, a)
        return shat

    def _spectral_cidct_max(c_used):
        """Coefficient scores through c×c IDCT, then max over c outputs.

        Unlike _spectral_curve which expands to page_size via [P, c] basis,
        this stays in the compressed dimension using a [c, c] IDCT matrix.
        """
        _, _, J_tensor, _, _, _, _ = _get_codex_basis(
            P, c_used, layout, X, device, torch.float32
        )
        k_dct_kept = k_dct.index_select(-2, J_tensor)
        a_coeff = torch.einsum('bhgd,bhncd->bhgnc', q, k_dct_kept)
        cidct_mat = _get_cidct_matrix(c_used, device)             # [c, c]
        shat_c = torch.einsum('rc,bhgnc->bhgnr', cidct_mat, a_coeff)
        return shat_c.max(dim=-1).values                          # [B,H,G,N]

    # --- Shared spectral curves (computed once) ---
    shat_lo = _spectral_curve(c_lo)
    spectral_lo_max = shat_lo.max(dim=-1).values           # [B,H,G,N]

    # --- Inner helper: emit strict variants for one highlight score ---
    def _emit_heuristic(name: str, hi_score: torch.Tensor):
        """hi_score: [B,H,G,N]"""
        # Strict budget: (c-1) DCT + 1 highlight
        scaled_lo = spectral_lo_max.unsqueeze(-1) * alphas_t       # [B,H,G,N,A]
        hi_exp = hi_score.unsqueeze(-1).expand_as(scaled_lo)
        v_max = torch.maximum(scaled_lo, hi_exp).max(dim=2).values  # [B,H,N,A]
        v_add = (scaled_lo + hi_exp).max(dim=2).values
        for ai, a in enumerate(alphas):
            out[f'hybrid_{name}_max_a{a}'] = v_max[..., ai]
            out[f'hybrid_{name}_add_a{a}'] = v_add[..., ai]

    # --- Helper: gather single highlight token and compute score ---
    def _highlight_score_from_idx(idx: torch.Tensor) -> torch.Tensor:
        """idx: [B,H,N] → highlight_score: [B,H,G,N]"""
        ie = idx.unsqueeze(-1).unsqueeze(-1).expand(B, H, N, 1, D)
        K_hi = k_f.gather(dim=-2, index=ie).squeeze(-2)     # [B,H,N,D]
        return torch.einsum('bhgd,bhnd->bhgn', q, K_hi)

    # === Heuristic 1: highlight (K-norm argmax) ===
    k_norm_tok = k_f.pow(2).sum(dim=-1)                     # [B,H,N,P]
    highlight_idx = k_norm_tok.argmax(dim=-1)                # [B,H,N]
    _emit_heuristic('highlight', _highlight_score_from_idx(highlight_idx))

    # === AC energy (used by multi-highlight acnorm selection below) ===
    k_mean = k_f.mean(dim=-2, keepdim=True)                 # [B,H,N,1,D]
    ac_energy = (k_f - k_mean).pow(2).sum(dim=-1)           # [B,H,N,P]

    # === Heuristic 2: specresid (spectral residual norm argmax) ===
    # Token with most energy in omitted DCT frequencies.
    # By Cauchy-Schwarz: |q.(k_t - k_recon)| <= ||q|| * ||k_t - k_recon||
    _, _, J_lo_tensor, _, Phi_kept_T_lo, _, _ = _get_codex_basis(
        P, c_lo, layout, X, device, torch.float32
    )
    k_dct_kept_lo = k_dct.index_select(-2, J_lo_tensor)     # [B,H,N,c_lo,D]
    k_recon = torch.einsum(
        'tc,bhncd->bhntd', Phi_kept_T_lo, k_dct_kept_lo
    )                                                        # [B,H,N,P,D]
    specresid_energy = (k_f - k_recon).pow(2).sum(dim=-1)   # [B,H,N,P]
    specresid_idx = specresid_energy.argmax(dim=-1)          # [B,H,N]
    _emit_heuristic('specresid', _highlight_score_from_idx(specresid_idx))

    # === Multi-highlight: M>1 K-norm top-M tokens ===
    for M in (multi_highlights or []):
        M = int(M)
        if M < 1 or M >= P:
            continue
        c_multi = max(0, c - M)

        _, topM_idx = k_norm_tok.topk(M, dim=-1)            # [B,H,N,M]
        idx_M = topM_idx.unsqueeze(-1).expand(B, H, N, M, D)
        K_multi = k_f.gather(dim=-2, index=idx_M)           # [B,H,N,M,D]
        multi_scores = torch.einsum(
            'bhgd,bhnmd->bhgnm', q, K_multi
        )                                                    # [B,H,G,N,M]
        multi_hi = multi_scores.max(dim=-1).values           # [B,H,G,N]

        if c_multi > 0:
            shat_multi = _spectral_curve(c_multi)
            spectral_multi_max = shat_multi.max(dim=-1).values

            scaled_m = spectral_multi_max.unsqueeze(-1) * alphas_t
            hi_m = multi_hi.unsqueeze(-1).expand_as(scaled_m)
            vm_max = torch.maximum(scaled_m, hi_m).max(dim=2).values
            vm_add = (scaled_m + hi_m).max(dim=2).values
            for ai, a in enumerate(alphas):
                out[f'hybrid_multi{M}_max_a{a}'] = vm_max[..., ai]
                out[f'hybrid_multi{M}_add_a{a}'] = vm_add[..., ai]
        else:
            # c_multi == 0: no DCT budget, score = max_m(q . K_m)
            out[f'hybrid_multi{M}_only'] = multi_hi.max(dim=2).values

    # === Multi-highlight with acnorm selection + optional residual decode ===
    for M in (multi_highlights or []):
        M = int(M)
        if M < 1 or M >= P:
            continue
        c_multi = max(0, c - M)

        # Acnorm top-M selection
        _, topM_idx_ac = ac_energy.topk(M, dim=-1)              # [B,H,N,M]
        idx_M_ac = topM_idx_ac.unsqueeze(-1).expand(B, H, N, M, D)
        K_multi_ac = k_f.gather(dim=-2, index=idx_M_ac)         # [B,H,N,M,D]
        multi_scores_ac = torch.einsum(
            'bhgd,bhnmd->bhgnm', q, K_multi_ac
        )                                                        # [B,H,G,N,M]
        multi_hi_ac = multi_scores_ac.max(dim=-1).values         # [B,H,G,N]

        if c_multi > 0:
            shat_multi_ac = _spectral_curve(c_multi)
            spectral_multi_ac_max = shat_multi_ac.max(dim=-1).values

            # --- ac: max-score decode ---
            scaled_m = spectral_multi_ac_max.unsqueeze(-1) * alphas_t
            hi_m = multi_hi_ac.unsqueeze(-1).expand_as(scaled_m)
            vm_max = torch.maximum(scaled_m, hi_m).max(dim=2).values
            vm_add = (scaled_m + hi_m).max(dim=2).values
            for ai, a in enumerate(alphas):
                out[f'hybrid_multi{M}_ac_max_a{a}'] = vm_max[..., ai]
                out[f'hybrid_multi{M}_ac_add_a{a}'] = vm_add[..., ai]

            # --- ac: cidct (c×c IDCT, no expansion to page_size) ---
            spectral_cidct = _spectral_cidct_max(c_multi)
            scaled_cidct = spectral_cidct.unsqueeze(-1) * alphas_t
            hi_cidct = multi_hi_ac.unsqueeze(-1).expand_as(scaled_cidct)
            vm_cidct = torch.maximum(scaled_cidct, hi_cidct).max(dim=2).values
            for ai, a in enumerate(alphas):
                out[f'hybrid_multi{M}_ac_cidct_max_a{a}'] = vm_cidct[..., ai]

            # --- acresid: residual decode (query-aware selection) ---
            topM_idx_g = topM_idx_ac.unsqueeze(2).expand(B, H, G, N, M)
            shat_at_cands = shat_multi_ac.gather(dim=-1, index=topM_idx_g)
            residuals = multi_scores_ac - shat_at_cands
            best_m_idx = residuals.argmax(dim=-1)
            best_score = multi_scores_ac.gather(
                dim=-1, index=best_m_idx.unsqueeze(-1)
            ).squeeze(-1)

            scaled_m_r = spectral_multi_ac_max.unsqueeze(-1) * alphas_t
            hi_r = best_score.unsqueeze(-1).expand_as(scaled_m_r)
            vr_max = torch.maximum(scaled_m_r, hi_r).max(dim=2).values
            vr_add = (scaled_m_r + hi_r).max(dim=2).values
            for ai, a in enumerate(alphas):
                out[f'hybrid_multi{M}_acresid_max_a{a}'] = vr_max[..., ai]
                out[f'hybrid_multi{M}_acresid_add_a{a}'] = vr_add[..., ai]
        else:
            # c_multi == 0: both ac and acresid degenerate to max_m(q·K_m)
            out[f'hybrid_multi{M}_ac_only'] = multi_hi_ac.max(dim=2).values

    # === Oracle: query-aware residual-argmax highlight (heuristic-independent) ===
    per_tok = torch.einsum('bhgd,bhnsd->bhgns', q, k_f)     # [B,H,G,N,P]
    residual = per_tok - shat_lo                              # signed, not absolute
    oracle_idx = residual.argmax(dim=-1)                      # [B,H,G,N]
    oracle_score = per_tok.gather(
        dim=-1, index=oracle_idx.unsqueeze(-1)
    ).squeeze(-1)                                             # [B,H,G,N]

    scaled_spec_lo_o = spectral_lo_max.unsqueeze(-1) * alphas_t
    or_exp = oracle_score.unsqueeze(-1).expand_as(scaled_spec_lo_o)

    v_max_oracle = torch.maximum(scaled_spec_lo_o, or_exp).max(dim=2).values
    for ai, alpha in enumerate(alphas):
        out[f'hybrid_highlight_max_oracle_a{alpha}'] = v_max_oracle[..., ai]

    v_add_oracle = (scaled_spec_lo_o + or_exp).max(dim=2).values
    for ai, alpha in enumerate(alphas):
        out[f'hybrid_highlight_add_oracle_a{alpha}'] = v_add_oracle[..., ai]

    return out


def compute_all_scores_with_codex(
    per_token_scores: torch.Tensor,
    proxy_scores: dict[str, torch.Tensor],
    output_contrib: torch.Tensor | None,
    lambdas: list[float] | None = None,
    comp_size: int | None = None,
    paged_k: torch.Tensor | None = None,
    query_states: torch.Tensor | None = None,
    num_kv_groups: int | None = None,
    betas: list[float] | None = None,
    *,
    layouts: list[str] | None = None,
    ucb_levels: list[int] | None = None,
    ucb_num_bands: int = 4,
    continuous_multiplier: int = 4,
    alphas: list[float] | None = None,
    multi_highlights: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Wrap compute_all_scores and merge codex spectral methods.

    `layouts` is a list of frequency-keep layouts to evaluate. For a single
    layout, codex method names match the original (no suffix). For >1 layouts,
    each method name is suffixed with `_{layout}` so the variants can be
    analyzed side-by-side in one diagnostic run.
    """
    scores = compute_all_scores(
        per_token_scores,
        proxy_scores,
        output_contrib,
        lambdas,
        comp_size=comp_size,
        paged_k=paged_k,
        query_states=query_states,
        num_kv_groups=num_kv_groups,
        betas=betas,
    )

    layouts_eff = list(layouts) if layouts else ["low"]

    if (
        comp_size is not None
        and comp_size > 0
        and paged_k is not None
        and query_states is not None
        and num_kv_groups is not None
    ):
        # -- spectral_recon / continuous_cosine disabled for now --
        # multi = len(layouts_eff) > 1
        # for layout in layouts_eff:
        #     suffix = f"_{layout}" if multi else ""
        #     codex_scores = _compute_codex_spectral_scores(
        #         paged_k=paged_k,
        #         query_states=query_states,
        #         num_kv_groups=num_kv_groups,
        #         comp_size=comp_size,
        #         layout=layout,
        #         betas=betas or [0.25, 0.5, 1.0],
        #         ucb_levels=ucb_levels if ucb_levels is not None else [1, 2, 3],
        #         ucb_num_bands=ucb_num_bands,
        #         continuous_multiplier=continuous_multiplier,
        #         name_suffix=suffix,
        #     )
        #     scores.update(codex_scores)

        # Hybrid methods (layout-independent; emitted once)
        scores.update(_compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states,
            num_kv_groups=num_kv_groups, comp_size=comp_size,
            layout=layouts_eff[0] if layouts_eff else "low",
            alphas=alphas or [1.0, 2.0, 4.0, 8.0],
            continuous_multiplier=continuous_multiplier,
            multi_highlights=multi_highlights,
        ))

    return scores


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_one_sample(
    all_scores: dict[str, torch.Tensor],  # each [bsz, kv_heads, num_pages]
    ground_truth: str,
    methods: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Analyze all methods vs ground truth for a single sample across all heads.

    Vectorized: stacks all methods into one tensor and computes every metric in
    batch, with a single GPU→CPU sync at the end.
    """
    gt = all_scores[ground_truth][0]  # [kv_heads, num_pages]
    kv_heads, num_pages = gt.shape
    actual_top_k = min(top_k, num_pages)
    device = gt.device

    # Stack methods: [M, kv_heads, num_pages]
    method_stack = torch.stack([all_scores[m][0] for m in methods], dim=0)
    num_methods = method_stack.shape[0]

    # Top-k membership masks
    gt_topk_idx = gt.topk(actual_top_k, dim=-1).indices  # [kv, k]
    gt_mask = torch.zeros(kv_heads, num_pages, dtype=torch.bool, device=device)
    gt_mask.scatter_(1, gt_topk_idx, True)

    method_topk_idx = method_stack.topk(actual_top_k, dim=-1).indices  # [M, kv, k]
    method_mask = torch.zeros(num_methods, kv_heads, num_pages, dtype=torch.bool, device=device)
    method_mask.scatter_(2, method_topk_idx, True)

    gt_mask_b = gt_mask.unsqueeze(0)  # [1, kv, N]
    gt_b = gt.unsqueeze(0).float()    # [1, kv, N]

    overlap = (method_mask & gt_mask_b).sum(dim=-1).float()  # [M, kv]
    recall = overlap / actual_top_k
    # |method_topk| == |gt_topk| == actual_top_k, so FP == FN == k - overlap
    fp_fn_count = float(actual_top_k) - overlap

    neg_gt = (method_mask & (gt_b < 0)).sum(dim=-1).float()  # [M, kv]

    # Ranks (argsort-argsort): [M, kv, N]
    method_ranks = method_stack.argsort(dim=-1, descending=True).argsort(dim=-1).float()

    fp_mask = method_mask & ~gt_mask_b
    fn_mask = gt_mask_b & ~method_mask
    fp_mask_f = fp_mask.float()
    fn_mask_f = fn_mask.float()

    fp_count_f = fp_mask_f.sum(dim=-1).clamp(min=1)
    fn_count_f = fn_mask_f.sum(dim=-1).clamp(min=1)
    has_fp = fp_mask.any(dim=-1).float()
    has_fn = fn_mask.any(dim=-1).float()

    fp_gt_mean = (gt_b * fp_mask_f).sum(dim=-1) / fp_count_f * has_fp
    fn_gt_mean = (gt_b * fn_mask_f).sum(dim=-1) / fn_count_f * has_fn
    fn_rank_mean = (method_ranks * fn_mask_f).sum(dim=-1) / fn_count_f * has_fn

    # Average across kv_heads, then stack metrics for a single .cpu() sync
    metrics_stack = torch.stack([
        recall.mean(dim=1),
        fp_fn_count.mean(dim=1),    # false_positive_count
        fp_fn_count.mean(dim=1),    # false_negative_count
        neg_gt.mean(dim=1),
        fp_gt_mean.mean(dim=1),
        fn_rank_mean.mean(dim=1),
        fn_gt_mean.mean(dim=1),
    ], dim=0).cpu().tolist()  # [7, M]

    agg: dict[str, Any] = {
        "num_heads": kv_heads,
        "actual_top_k": actual_top_k,
        "num_pages": num_pages,
        "ground_truth": ground_truth,
    }
    metric_names = [
        "recall", "false_positive_count", "false_negative_count",
        "neg_gt_in_topk", "fp_gt_score_mean", "fn_rank_mean", "fn_gt_score_mean",
    ]
    for mi, mname in enumerate(metric_names):
        row = metrics_stack[mi]
        for j, method in enumerate(methods):
            agg[f"{method}_{mname}_avg"] = row[j]
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose page scoring methods (base + codex spectral reconstruction)"
    )
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--data_root", type=Path, default=Path("benchmark/data/ruler_data"))
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/results_ruler/scoring_methods_diagnostic"),
    )
    p.add_argument("--tasks", default="cwe", help="Comma-separated task names or 'all'")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--num_decode_steps", type=int, default=10,
                   help="Number of decode steps to evaluate. Recall is averaged across steps.")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--device_map", type=str, default=None,
                   help="HF device_map string (e.g. 'auto'). Overrides --cuda_device.")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--ground_truths",
        default="oracle_max,output_contribution",
        help="Comma-separated ground truths to loop over (choices: "
             + ", ".join(GROUND_TRUTHS) + ")",
    )

    # DCT page config
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--lambdas", default="0.25,0.5,1.0,1.5,2.0",
                   help="Comma-separated lambda values for DC+AC scoring")
    p.add_argument("--betas", default="0.25,0.5,1.0",
                   help="Comma-separated beta values for residual correction experiments")

    # Codex spectral / hybrid config
    p.add_argument(
        "--layout",
        default=None,
        help="Single-layout shortcut. Equivalent to --layouts <layout>. "
             "Ignored if --layouts is also set.",
    )
    p.add_argument(
        "--layouts",
        default=None,
        help="Comma-separated list of frequency layouts to register side-by-side "
             "for codex methods (low|spread|low_high|low_mid_high). When more than "
             "one is given, codex method names are suffixed with _{layout}. "
             "Defaults to 'low'.",
    )
    p.add_argument(
        "--ucb_levels",
        default="1,2,3",
        help="Comma-separated UCB uncertainty levels to enable (subset of {1,2,3}).",
    )
    p.add_argument(
        "--ucb_num_bands",
        type=int,
        default=4,
        help="Number of bands for UCB level 2 (bandwise).",
    )
    p.add_argument(
        "--continuous_multiplier",
        type=int,
        default=4,
        help="continuous_cosine_max dense samples = multiplier * page_size.",
    )
    p.add_argument(
        "--alphas",
        default="1.0,2.0,4.0,8.0",
        help="Comma-separated alpha values for hybrid highlight spectral scaling. "
             "Balance point ≈ sqrt(page_size / (comp_size - 1)).",
    )
    p.add_argument(
        "--multi_highlights",
        default="2,3",
        help="Comma-separated M values for multi-highlight variants. "
             "Each M uses K-norm top-M and (c-M) DCT coefficients.",
    )
    p.add_argument(
        "--self_test",
        action="store_true",
        help="Run numerical self-tests and exit.",
    )
    return p.parse_args()


def resolve_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return requested


def resolve_ground_truths(value: str) -> list[str]:
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in GROUND_TRUTHS]
    if unknown:
        raise ValueError(f"Unknown ground truths: {unknown}. Choose from {GROUND_TRUTHS}")
    return requested


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def load_model(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

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
    device_map = {"": args.cuda_device} if args.device_map is None else args.device_map
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map=device_map,
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
# Per-layer scoring
# ---------------------------------------------------------------------------
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
    *,
    layouts: list[str] | None = None,
    ucb_levels: list[int] | None = None,
    ucb_num_bands: int = 4,
    continuous_multiplier: int = 4,
    alphas: list[float] | None = None,
    multi_highlights: list[int] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Compute all scoring methods (base + codex) for one layer at one decode step.

    Returns all_scores dict (values are [bsz, kv_heads, num_pages] tensors),
    or None if there are not enough tokens for at least one page.
    """
    kv_len = key_states.shape[2]
    pageable_len = kv_len - sink_size - recent_size
    if pageable_len < page_size:
        return None

    # When using device_map="auto", tensors may be on different GPUs.
    # Use the attention module's device (where q_proj lives) as the scoring device,
    # and move all tensors there.
    score_device = next(attn_module.parameters()).device
    key_states = key_states.to(score_device)
    value_states = value_states.to(score_device)
    hidden_states = hidden_states.to(score_device)

    sink_k, sink_v, paged_k, paged_v, recent_k, recent_v, num_pages = \
        segment_kv_from_cache(key_states, value_states, page_size, sink_size, recent_size)

    query_states = recompute_query(attn_module, hidden_states, kv_len, config, score_device, dtype)

    per_token_scores = compute_per_token_scores(query_states, paged_k, num_kv_groups)
    proxy_scores = compute_proxy_scores(
        query_states, paged_k, num_kv_groups, comp_size, lambdas=lambdas,
    )
    output_contrib = compute_output_contribution(
        query_states, paged_k, paged_v, num_kv_groups,
        sink_k=sink_k, sink_v=sink_v, recent_k=recent_k, recent_v=recent_v,
    )
    all_scores = compute_all_scores_with_codex(
        per_token_scores, proxy_scores, output_contrib, lambdas,
        comp_size=comp_size, paged_k=paged_k,
        query_states=query_states, num_kv_groups=num_kv_groups, betas=betas,
        layouts=layouts, ucb_levels=ucb_levels,
        ucb_num_bands=ucb_num_bands, continuous_multiplier=continuous_multiplier,
        alphas=alphas, multi_highlights=multi_highlights,
    )
    return all_scores


def _build_method_registry(lambdas, betas, comp_size, ucb_levels, layouts, alphas,
                           multi_highlights=None):
    """Slim whitelist of methods that get analyzed/printed.

    For >1 layouts, codex spectral method names are suffixed with _{layout}.
    Hybrid methods are layout-independent (use first layout).
    """
    methods = [
        "oracle_max", "oracle_mean",
        "proxy_max", "proxy_mean",
        "l2_energy",
        "output_contribution",
    ]
    methods += [f"dc_ac_{lam}" for lam in lambdas]
    if comp_size > 1:
        methods += [f"proxy_dc_ac_{lam}" for lam in lambdas]
        methods += [f"spread_dc_ac_{lam}" for lam in lambdas]
    if comp_size > 0:
        # -- spectral_recon / continuous_cosine disabled for now --
        # multi = len(layouts) > 1
        # for layout in layouts:
        #     sfx = f"_{layout}" if multi else ""
        #     methods.append(f"spectral_recon_max{sfx}")
        #     for lvl in ucb_levels:
        #         methods += [f"spectral_recon_ucb{lvl}{sfx}_b{b}" for b in betas]
        #     methods.append(f"continuous_cosine_max{sfx}")
        # Hybrid heuristic methods — each heuristic × {max,add} × alpha
        for hname in ('highlight', 'specresid'):
            methods += [f"hybrid_{hname}_max_a{a}" for a in alphas]
            methods += [f"hybrid_{hname}_add_a{a}" for a in alphas]
        # Oracle (heuristic-independent)
        methods += [f"hybrid_highlight_max_oracle_a{a}" for a in alphas]
        methods += [f"hybrid_highlight_add_oracle_a{a}" for a in alphas]
        # Multi-highlight methods (K-norm selection)
        for M in (multi_highlights or []):
            c_multi = max(0, comp_size - M)
            if c_multi > 0:
                methods += [f"hybrid_multi{M}_max_a{a}" for a in alphas]
                methods += [f"hybrid_multi{M}_add_a{a}" for a in alphas]
            else:
                methods.append(f"hybrid_multi{M}_only")
        # Multi-highlight with acnorm selection + residual decode
        for M in (multi_highlights or []):
            c_multi = max(0, comp_size - M)
            if c_multi > 0:
                methods += [f"hybrid_multi{M}_ac_max_a{a}" for a in alphas]
                methods += [f"hybrid_multi{M}_ac_add_a{a}" for a in alphas]
                methods += [f"hybrid_multi{M}_ac_cidct_max_a{a}" for a in alphas]
                methods += [f"hybrid_multi{M}_acresid_max_a{a}" for a in alphas]
                methods += [f"hybrid_multi{M}_acresid_add_a{a}" for a in alphas]
            else:
                methods.append(f"hybrid_multi{M}_ac_only")
    return methods


def _self_test() -> None:
    """Numerical sanity tests for codex hybrid methods."""
    print("Running self-test...")
    torch.manual_seed(42)

    B, H, N, P, D, G = 1, 2, 4, 16, 64, 2
    Hq = H * G
    paged_k = torch.randn(B, H, N, P, D, dtype=torch.float32) * 0.1
    query_states = torch.randn(B, Hq, 1, D, dtype=torch.float32) * 0.1

    k_for_dct = paged_k.transpose(-2, -1)
    k_dct_full = _dct_local(k_for_dct).transpose(-2, -1)

    scaling = D ** -0.5
    q = query_states.squeeze(2).reshape(B, H, G, D)
    q_scaled = q * scaling

    per_tok = torch.einsum('bhgd,bhntd->bhgnt', q_scaled, paged_k)
    oracle_max_ref = per_tok.max(dim=-1).values.max(dim=2).values  # [B, H, N]

    test_betas = [0.5]
    test_levels = [1, 2, 3]

    # -- Tests 1-6 (spectral_recon / continuous_cosine) disabled --

    # ---- Test 7: hybrid_highlight_max >= spectral_recon_max(c-1) ----
    test_alphas = [1.0, 4.0]
    for c in [2, 4, 8, 16]:
        hy = _compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c, layout="low", alphas=test_alphas, continuous_multiplier=4,
        )
        sp_lo_ref = _compute_codex_spectral_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=max(1, c - 1), layout="low", betas=[], ucb_levels=[],
            ucb_num_bands=4, continuous_multiplier=4,
        )['spectral_recon_max']
        for alpha in test_alphas:
            diff7 = (hy[f'hybrid_highlight_max_a{alpha}'] - sp_lo_ref).min().item()
            assert diff7 >= -1e-5, (
                f"Test 7 (c={c}, a={alpha}): hybrid_max < spectral_lo by {-diff7}"
            )

    # ---- Test 9: hybrid_highlight_max_oracle >= spectral_recon_max(c-1) ----
    for c in [2, 4, 8]:
        hy_o = _compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c, layout="low", alphas=[1.0], continuous_multiplier=4,
        )
        sp_lo = _compute_codex_spectral_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=max(1, c - 1), layout="low", betas=[], ucb_levels=[],
            ucb_num_bands=4, continuous_multiplier=4,
        )['spectral_recon_max']
        diff9 = (hy_o['hybrid_highlight_max_oracle_a1.0'] - sp_lo).min().item()
        assert diff9 >= -1e-5, (
            f"Test 9 (c={c}): hybrid_oracle < spectral_lo by {-diff9}"
        )

    # ---- Test 12: hybrid_specresid_max >= spectral_recon_max(c-1) ----
    for c in [2, 4, 8, 16]:
        hy_sr = _compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c, layout="low", alphas=[1.0], continuous_multiplier=4,
        )
        sp_lo_12 = _compute_codex_spectral_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=max(1, c - 1), layout="low", betas=[], ucb_levels=[],
            ucb_num_bands=4, continuous_multiplier=4,
        )['spectral_recon_max']
        diff12 = (hy_sr['hybrid_specresid_max_a1.0'] - sp_lo_12).min().item()
        assert diff12 >= -1e-5, (
            f"Test 12 (c={c}): hybrid_specresid_max < spectral_lo by {-diff12}"
        )

    # ---- Test 13: hybrid_multi2_max >= spectral_recon_max(c-2) for c >= 4 ----
    for c in [4, 8, 16]:
        c_multi = c - 2
        hy_m2 = _compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c, layout="low", alphas=[1.0], continuous_multiplier=4,
            multi_highlights=[2],
        )
        sp_lo_13 = _compute_codex_spectral_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c_multi, layout="low", betas=[], ucb_levels=[],
            ucb_num_bands=4, continuous_multiplier=4,
        )['spectral_recon_max']
        diff13 = (hy_m2['hybrid_multi2_max_a1.0'] - sp_lo_13).min().item()
        assert diff13 >= -1e-5, (
            f"Test 13 (c={c}): hybrid_multi2_max < spectral_lo(c-2) by {-diff13}"
        )

    # ---- Test 14: hybrid_multi2_ac_max >= spectral_recon_max(c-2) for c >= 4 ----
    for c in [4, 8, 16]:
        c_multi = c - 2
        hy_ac2 = _compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c, layout="low", alphas=[1.0], continuous_multiplier=4,
            multi_highlights=[2],
        )
        sp_lo_14 = _compute_codex_spectral_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c_multi, layout="low", betas=[], ucb_levels=[],
            ucb_num_bands=4, continuous_multiplier=4,
        )['spectral_recon_max']
        diff14 = (hy_ac2['hybrid_multi2_ac_max_a1.0'] - sp_lo_14).min().item()
        assert diff14 >= -1e-5, (
            f"Test 14 (c={c}): hybrid_multi2_ac_max < spectral_lo(c-2) by {-diff14}"
        )

    # ---- Test 15: hybrid_multi2_acresid_max >= spectral_recon_max(c-2) for c >= 4 ----
    for c in [4, 8, 16]:
        c_multi = c - 2
        hy_ar2 = _compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c, layout="low", alphas=[1.0], continuous_multiplier=4,
            multi_highlights=[2],
        )
        sp_lo_15 = _compute_codex_spectral_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c_multi, layout="low", betas=[], ucb_levels=[],
            ucb_num_bands=4, continuous_multiplier=4,
        )['spectral_recon_max']
        diff15 = (hy_ar2['hybrid_multi2_acresid_max_a1.0'] - sp_lo_15).min().item()
        assert diff15 >= -1e-5, (
            f"Test 15 (c={c}): hybrid_multi2_acresid_max < spectral_lo(c-2) by {-diff15}"
        )

    # ---- Test 16: hybrid_multi2_ac_cidct_max is finite and present for c >= 4 ----
    for c in [4, 8, 16]:
        hy_cidct = _compute_codex_hybrid_scores(
            paged_k=paged_k, query_states=query_states, num_kv_groups=G,
            comp_size=c, layout="low", alphas=[1.0, 4.0], continuous_multiplier=4,
            multi_highlights=[2],
        )
        for alpha in [1.0, 4.0]:
            key = f'hybrid_multi2_ac_cidct_max_a{alpha}'
            assert key in hy_cidct, f"Test 16 (c={c}): {key} not in output"
            assert hy_cidct[key].isfinite().all(), (
                f"Test 16 (c={c}): {key} has non-finite values"
            )
            # cidct score should equal ac_max when c×c IDCT happens to
            # recover the same peak as the P-expansion (not guaranteed in
            # general, but values should be in a similar ballpark)
            ac_max_key = f'hybrid_multi2_ac_max_a{alpha}'
            ratio = (
                hy_cidct[key].abs().mean()
                / hy_cidct[ac_max_key].abs().mean().clamp(min=1e-12)
            )
            assert 0.01 < ratio < 100, (
                f"Test 16 (c={c}, a={alpha}): cidct/ac_max ratio {ratio:.4f} "
                f"outside [0.01, 100]"
            )

    # ---- Test 17: c×c IDCT matrix is orthonormal ----
    for c in [2, 4, 8]:
        cidct = _get_cidct_matrix(c, torch.device('cpu'))
        eye_check = cidct @ cidct.T
        diff17 = (eye_check - torch.eye(c)).abs().max().item()
        assert diff17 < 1e-5, (
            f"Test 17 (c={c}): IDCT not orthonormal, max err {diff17}"
        )

    print("All self-tests passed:")
    print("  7. hybrid_highlight_max >= spectral_recon_max(c-1)")
    print("  9. hybrid_highlight_max_oracle >= spectral_recon_max(c-1)")
    print(" 12. hybrid_specresid_max >= spectral_recon_max(c-1)")
    print(" 13. hybrid_multi2_max >= spectral_recon_max(c-2) for c >= 4")
    print(" 14. hybrid_multi2_ac_max >= spectral_recon_max(c-2) for c >= 4")
    print(" 15. hybrid_multi2_acresid_max >= spectral_recon_max(c-2) for c >= 4")
    print(" 16. hybrid_multi2_ac_cidct_max is finite and well-scaled")
    print(" 17. c×c IDCT matrix is orthonormal")


def main() -> None:
    args = parse_args()

    if args.self_test:
        _self_test()
        return

    from tqdm import tqdm
    from transformers import AutoTokenizer

    tasks = resolve_tasks(args.tasks)
    gt_names = resolve_ground_truths(args.ground_truths)
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    ucb_levels = sorted({int(x.strip()) for x in args.ucb_levels.split(",") if x.strip()})
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    multi_highlights = [int(x.strip()) for x in args.multi_highlights.split(",") if x.strip()]

    if args.layouts:
        raw_layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    elif args.layout:
        raw_layouts = [args.layout]
    else:
        raw_layouts = ["low"]
    # Dedupe while preserving order
    layouts: list[str] = []
    for lay in raw_layouts:
        if lay not in layouts:
            layouts.append(lay)

    comp_size = max(1, int(args.page_size * args.compress_ratio))
    all_methods = _build_method_registry(
        lambdas, betas, comp_size, ucb_levels, layouts, alphas, multi_highlights,
    )

    if comp_size > 1:
        print(f"Frequency indices (comp_size={comp_size}, page_size={args.page_size}):")
        for lay in layouts:
            print(
                f"  layout={lay!r}: "
                f"{_resolve_frequency_keep_indices_local(args.page_size, comp_size, lay)}"
            )

    print(f"Full attention generation, {args.num_decode_steps} decode steps, scoring all layers")
    print(f"Ground truths: {gt_names}")
    print(f"UCB levels: {ucb_levels}, betas: {betas}, alphas: {alphas}, layouts: {layouts}")
    print(f"Multi-highlight M values: {multi_highlights}")
    print(f"continuous_multiplier: {args.continuous_multiplier} -> dense samples = {args.continuous_multiplier * args.page_size}")
    print(f"Methods printed: {len(all_methods)}")

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

    layouts_tag = "+".join(layouts)
    run_dirs = {}
    for gt_name in gt_names:
        run_dir = (
            args.output_dir
            / f"ps{args.page_size}_topk{args.top_k}_cr{args.compress_ratio}_lay{layouts_tag}_gt_{gt_name}"
        )
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
            data_path = (
                args.data_root / model_family / str(args.context_len) / task / "validation.jsonl"
            )
            if not data_path.exists():
                print(f"  WARNING: data not found at {data_path}, skipping")
                continue

            with data_path.open("r", encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]
            samples = samples[: args.num_samples]

            scored_samples: list[tuple[dict, dict[str, dict]]] = []

            for idx, sample in enumerate(tqdm(samples, desc=f"  {task}"), start=1):
                encoded = tokenizer(sample["input"], return_tensors="pt")
                input_ids = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)
                prefill_len = input_ids.shape[1]

                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

                observations: dict[str, list[dict]] = {gt: [] for gt in gt_names}

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

                    for layer_idx in range(num_layers):
                        h = outputs.hidden_states[layer_idx][:, -1:, :]
                        h = model.model.layers[layer_idx].input_layernorm(h)

                        attn = model.model.layers[layer_idx].self_attn
                        if hasattr(past_key_values, 'key_cache'):
                            key_cache = past_key_values.key_cache[layer_idx]
                            value_cache = past_key_values.value_cache[layer_idx]
                        else:
                            key_cache = past_key_values.layers[layer_idx].keys
                            value_cache = past_key_values.layers[layer_idx].values

                        all_scores = _score_one_layer(
                            attn, h, key_cache, value_cache,
                            model.config, num_kv_groups, comp_size,
                            args.page_size, args.sink_size, args.recent_size,
                            args.top_k, lambdas, betas, device, dtype,
                            layouts=layouts,
                            ucb_levels=ucb_levels,
                            ucb_num_bands=args.ucb_num_bands,
                            continuous_multiplier=args.continuous_multiplier,
                            alphas=alphas,
                            multi_highlights=multi_highlights,
                        )
                        if all_scores is None:
                            continue

                        num_pages = all_scores["oracle_max"].shape[-1]
                        actual_top_k = min(args.top_k, num_pages)

                        for gt_name in gt_names:
                            methods = [m for m in all_methods if m != gt_name]
                            analysis = analyze_one_sample(
                                all_scores, gt_name, methods, actual_top_k,
                            )
                            observations[gt_name].append(analysis)

                del past_key_values, outputs
                torch.cuda.empty_cache()

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

            gt_summaries: dict[str, dict[str, Any]] = {}

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

                gt_summaries[gt_name] = task_summary

                print(
                    f"\n  === {task} Summary (vs {gt_name}, group_agg=max, "
                    f"{args.num_decode_steps} steps x {num_layers} layers) ==="
                )
                print(
                    f"  Pages: {task_summary['num_pages']}, "
                    f"Top-k: {task_summary['actual_top_k']}"
                )
                for m in methods:
                    print(
                        f"  {m:35s}  "
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

            # --- Combined cross-GT recall comparison ---
            if len(gt_summaries) >= 2:
                gt_order = [gt for gt in gt_names if gt in gt_summaries]
                showable = [
                    m for m in all_methods
                    if all(f"{m}_recall_avg" in gt_summaries[gt] for gt in gt_order)
                ]

                def _hdr(gt: str) -> str:
                    return f"recall_vs_{gt}"

                col_w = {gt: max(len(_hdr(gt)), 7) for gt in gt_order}
                header_cells = "  ".join(f"{_hdr(gt):<{col_w[gt]}}" for gt in gt_order)
                print(
                    f"\n  === {task} Combined recall (vs {', '.join(gt_order)}) ==="
                )
                print(f"  {'method':35s}  {header_cells}")
                for m in showable:
                    cells = "  ".join(
                        f"{gt_summaries[gt][f'{m}_recall_avg']:.3f}".ljust(col_w[gt])
                        for gt in gt_order
                    )
                    print(f"  {m:35s}  {cells}")

    finally:
        cleanup_model(model)

    print(f"\nAll results in: {[str(d) for d in run_dirs.values()]}")


if __name__ == "__main__":
    main()
