#!/usr/bin/env python3
"""
Diagnose page scoring methods — codex spectral reconstruction variants.

Sibling to diagnose_l2_scoring.py. Imports the base diagnostic helpers and
adds three new scoring methods that model the within-page score-curve peak
directly, rather than estimating page energy:

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

The printed/JSON output is filtered to a short whitelist:
  oracle_max, oracle_mean, proxy_max, proxy_mean, l2_energy,
  output_contribution, dc_ac_{lam}, proxy_dc_ac_{lam}, spread_dc_ac_{lam},
  spectral_recon_max, spectral_recon_ucb{1,2,3}_b{beta}, continuous_cosine_max.

Run --self_test for numerical sanity checks (no model required).
"""

from __future__ import annotations

import argparse
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

# NOTE: heavy imports (transformers, diagnose_l2_scoring, dct_page_attention) are
# deferred to the functions that need them so that --self_test runs without
# loading transformers. These constants are duplicated from diagnose_l2_scoring.py
# (kept in sync).
GROUND_TRUTHS = ["oracle_max", "output_contribution"]

_BASIS_CACHE: dict = {}


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
            import numpy as _np
            edges = _np.linspace(0, num_dropped, num_bands + 1, dtype=int)
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


def compute_all_scores_codex(
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
) -> dict[str, torch.Tensor]:
    """Wrap base.compute_all_scores and merge codex spectral methods.

    `layouts` is a list of frequency-keep layouts to evaluate. For a single
    layout, codex method names match the original (no suffix). For >1 layouts,
    each method name is suffixed with `_{layout}` so the variants can be
    analyzed side-by-side in one diagnostic run.
    """
    import diagnose_l2_scoring as base

    scores = base.compute_all_scores(
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
        multi = len(layouts_eff) > 1
        for layout in layouts_eff:
            suffix = f"_{layout}" if multi else ""
            codex_scores = _compute_codex_spectral_scores(
                paged_k=paged_k,
                query_states=query_states,
                num_kv_groups=num_kv_groups,
                comp_size=comp_size,
                layout=layout,
                betas=betas or [0.25, 0.5, 1.0],
                ucb_levels=ucb_levels if ucb_levels is not None else [1, 2, 3],
                ucb_num_bands=ucb_num_bands,
                continuous_multiplier=continuous_multiplier,
                name_suffix=suffix,
            )
            scores.update(codex_scores)

    return scores


def _score_one_layer_codex(
    attn_module,
    hidden_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
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
) -> dict[str, torch.Tensor] | None:
    """Same body as base._score_one_layer but routes to compute_all_scores_codex."""
    from diagnose_l2_scoring import (
        segment_kv_from_cache,
        recompute_query,
        compute_per_token_scores,
        compute_proxy_scores,
        compute_output_contribution,
    )

    kv_len = key_states.shape[2]
    pageable_len = kv_len - sink_size - recent_size
    if pageable_len < page_size:
        return None

    score_device = next(attn_module.parameters()).device
    key_states = key_states.to(score_device)
    value_states = value_states.to(score_device)
    hidden_states = hidden_states.to(score_device)

    sink_k, sink_v, paged_k, paged_v, recent_k, recent_v, num_pages = \
        segment_kv_from_cache(key_states, value_states, page_size, sink_size, recent_size)

    query_states = recompute_query(
        attn_module, hidden_states, kv_len, config, score_device, dtype
    )

    per_token_scores = compute_per_token_scores(query_states, paged_k, num_kv_groups)
    proxy_scores = compute_proxy_scores(
        query_states, paged_k, num_kv_groups, comp_size, lambdas=lambdas,
    )
    output_contrib = compute_output_contribution(
        query_states, paged_k, paged_v, num_kv_groups,
        sink_k=sink_k, sink_v=sink_v, recent_k=recent_k, recent_v=recent_v,
    )
    all_scores = compute_all_scores_codex(
        per_token_scores, proxy_scores, output_contrib, lambdas,
        comp_size=comp_size, paged_k=paged_k,
        query_states=query_states, num_kv_groups=num_kv_groups, betas=betas,
        layouts=layouts, ucb_levels=ucb_levels,
        ucb_num_bands=ucb_num_bands, continuous_multiplier=continuous_multiplier,
    )
    return all_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose page scoring methods (codex spectral reconstruction)"
    )
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--data_root", type=Path, default=Path("benchmark/data/ruler_data"))
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/results_ruler/l2_diagnostic_codex"),
    )
    p.add_argument("--tasks", default="cwe", help="Comma-separated task names or 'all'")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--num_decode_steps", type=int, default=10)
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--ground_truths",
        default="oracle_max,output_contribution",
        help="Comma-separated ground truths to loop over (choices: "
             + ", ".join(GROUND_TRUTHS) + ")",
    )

    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument(
        "--lambdas",
        default="0.25,0.5,1.0,1.5,2.0",
        help="Comma-separated lambda values for DC+AC scoring",
    )
    p.add_argument(
        "--betas",
        default="0.25,0.5,1.0",
        help="Comma-separated beta values for spectral_recon_ucb",
    )

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
        "--self_test",
        action="store_true",
        help="Run numerical self-tests and exit.",
    )
    return p.parse_args()


def _build_method_registry(lambdas, betas, comp_size, ucb_levels, layouts):
    """Slim whitelist of methods that get analyzed/printed.

    For >1 layouts, codex method names are suffixed with _{layout}.
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
        multi = len(layouts) > 1
        for layout in layouts:
            sfx = f"_{layout}" if multi else ""
            methods.append(f"spectral_recon_max{sfx}")
            for lvl in ucb_levels:
                methods += [f"spectral_recon_ucb{lvl}{sfx}_b{b}" for b in betas]
            methods.append(f"continuous_cosine_max{sfx}")
    return methods


def _self_test() -> None:
    """Numerical sanity tests for the three new spectral methods."""
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

    for c in [1, 2, 4, 8, 16]:
        out = _compute_codex_spectral_scores(
            paged_k=paged_k,
            query_states=query_states,
            num_kv_groups=G,
            comp_size=c,
            layout="low",
            betas=test_betas,
            ucb_levels=test_levels,
            ucb_num_bands=4,
            continuous_multiplier=4,
        )

        I = torch.eye(P)
        Phi = _dct_local(I)  # [P, P]; Phi[t, j] = phi_j(t)
        J = list(_resolve_frequency_keep_indices_local(P, c, "low"))
        J_tensor = torch.tensor(J, dtype=torch.long)
        Phi_kept_T_ref = Phi.index_select(1, J_tensor)
        k_dct_kept_ref = k_dct_full.index_select(-2, J_tensor)
        a_ref = torch.einsum('bhgd,bhncd->bhgnc', q_scaled, k_dct_kept_ref)
        shat_ref = torch.einsum('tc,bhgnc->bhgnt', Phi_kept_T_ref, a_ref)
        ref_score = shat_ref.max(dim=-1).values.max(dim=2).values

        diff1 = (out['spectral_recon_max'] - ref_score).abs().max().item()
        assert diff1 < 1e-4, f"Test 1 (c={c}): spectral_recon_max diff {diff1}"

        if c == P:
            diff_oracle = (out['spectral_recon_max'] - oracle_max_ref).abs().max().item()
            assert diff_oracle < 1e-4, f"Test 2 (c=P): vs oracle_max diff {diff_oracle}"
            for lvl in test_levels:
                key = f"spectral_recon_ucb{lvl}_b{test_betas[0]}"
                diff_ucb = (out[key] - out['spectral_recon_max']).abs().max().item()
                assert diff_ucb < 1e-6, f"Test 5 (c=P, lvl={lvl}): UCB!=base, diff {diff_ucb}"

        diff_cont = out['continuous_cosine_max'] - out['spectral_recon_max']
        assert diff_cont.min().item() >= -1e-5, (
            f"Test 3 (c={c}): continuous_cosine_max < spectral_recon_max by "
            f"{-diff_cont.min().item():.2e}"
        )

        if c < P:
            not_J = sorted(set(range(P)) - set(J))
            notJ_tensor = torch.tensor(not_J, dtype=torch.long)
            k_dct_notJ_ref = k_dct_full.index_select(-2, notJ_tensor)
            q_sq = q_scaled.pow(2)
            var_b = torch.einsum('bhgd,bhnjd->bhgnj', q_sq, k_dct_notJ_ref.pow(2))
            sigma_res_sq = var_b.sum(dim=-1)

            Phi_notJ_T_ref = Phi.index_select(1, notJ_tensor)
            u_t_sq = torch.einsum('bhgnj,tj->bhgnt', var_b, Phi_notJ_T_ref.pow(2))
            u_t_total = u_t_sq.sum(dim=-1)
            rel_err = (
                (u_t_total - sigma_res_sq).abs()
                / sigma_res_sq.clamp(min=1e-12)
            ).max().item()
            assert rel_err < 1e-4, f"Test 4 (c={c}): Parseval rel err {rel_err}"

    # ---- Test 6: multi-layout naming via name_suffix ----
    out_low = _compute_codex_spectral_scores(
        paged_k=paged_k, query_states=query_states, num_kv_groups=G,
        comp_size=4, layout="low", betas=test_betas, ucb_levels=test_levels,
        ucb_num_bands=4, continuous_multiplier=4, name_suffix="_low",
    )
    out_spread = _compute_codex_spectral_scores(
        paged_k=paged_k, query_states=query_states, num_kv_groups=G,
        comp_size=4, layout="spread", betas=test_betas, ucb_levels=test_levels,
        ucb_num_bands=4, continuous_multiplier=4, name_suffix="_spread",
    )
    expected_low = {
        "spectral_recon_max_low",
        "continuous_cosine_max_low",
        "spectral_recon_ucb1_low_b0.5",
        "spectral_recon_ucb2_low_b0.5",
        "spectral_recon_ucb3_low_b0.5",
    }
    expected_spread = {k.replace("_low", "_spread") for k in expected_low}
    assert expected_low.issubset(out_low.keys()), \
        f"Test 6 (low): missing keys {expected_low - set(out_low.keys())}"
    assert expected_spread.issubset(out_spread.keys()), \
        f"Test 6 (spread): missing keys {expected_spread - set(out_spread.keys())}"
    # The low-layout suffixed value must equal the unsuffixed run
    out_unsuffixed = _compute_codex_spectral_scores(
        paged_k=paged_k, query_states=query_states, num_kv_groups=G,
        comp_size=4, layout="low", betas=test_betas, ucb_levels=test_levels,
        ucb_num_bands=4, continuous_multiplier=4,
    )
    diff_suffix = (
        out_low["spectral_recon_max_low"] - out_unsuffixed["spectral_recon_max"]
    ).abs().max().item()
    assert diff_suffix < 1e-7, f"Test 6: suffix vs unsuffix diff {diff_suffix}"
    # Spread should differ from low (different reconstructed signal)
    if not torch.allclose(
        out_low["spectral_recon_max_low"],
        out_spread["spectral_recon_max_spread"],
    ):
        pass  # expected: distinct values
    else:
        raise AssertionError("Test 6: spread layout produced identical scores to low (unexpected)")

    print("All self-tests passed:")
    print("  1. spectral_recon_max matches naive reference (c in {1,2,4,8,16})")
    print("  2. c=P: spectral_recon_max == oracle_max")
    print("  3. continuous_cosine_max >= spectral_recon_max element-wise")
    print("  4. UCB level 3 Parseval: sum_t u_t^2 == sigma_res^2")
    print("  5. c=P (empty notJ): all UCB variants == spectral_recon_max")
    print("  6. multi-layout name_suffix produces distinct keys, low-suffix == unsuffixed")


def main() -> None:
    args = parse_args()

    if args.self_test:
        _self_test()
        return

    from tqdm import tqdm
    from transformers import AutoTokenizer
    from diagnose_l2_scoring import (
        analyze_one_sample,
        resolve_tasks,
        resolve_ground_truths,
        resolve_model_family,
        load_model,
        cleanup_model,
    )

    tasks = resolve_tasks(args.tasks)
    gt_names = resolve_ground_truths(args.ground_truths)
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    ucb_levels = sorted({int(x.strip()) for x in args.ucb_levels.split(",") if x.strip()})

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
    all_methods = _build_method_registry(lambdas, betas, comp_size, ucb_levels, layouts)

    if comp_size > 1:
        print(f"Frequency indices (comp_size={comp_size}, page_size={args.page_size}):")
        for lay in layouts:
            print(
                f"  layout={lay!r}: "
                f"{_resolve_frequency_keep_indices_local(args.page_size, comp_size, lay)}"
            )

    print(f"Full attention generation, {args.num_decode_steps} decode steps, scoring all layers")
    print(f"Ground truths: {gt_names}")
    print(f"UCB levels: {ucb_levels}, betas: {betas}, layouts: {layouts}")
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

                        all_scores = _score_one_layer_codex(
                            attn, h, key_cache, value_cache,
                            model.config, num_kv_groups, comp_size,
                            args.page_size, args.sink_size, args.recent_size,
                            args.top_k, lambdas, betas, device, dtype,
                            layouts=layouts,
                            ucb_levels=ucb_levels,
                            ucb_num_bands=args.ucb_num_bands,
                            continuous_multiplier=args.continuous_multiplier,
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
