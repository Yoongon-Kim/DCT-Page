"""Haar wavelet utilities for HARP (Haar-based Adaptive Resolution Page attention).

Forward decomposition, top-K detail selection, and adaptive 2-token
representative reconstruction (L_3 ± β · H_selected). Pure PyTorch — no
Triton fallback at this stage. Used by oracle/diagnose_haar_mass_correlation.py
and (later) by the Haar selector in oracle/attention_mass_recall_ruler.py.

Shape convention:
  All operations work along the page-internal token axis dim=-2 of a
  tensor shaped [..., S, d] (or [..., P, S, d] when pages are stacked).
  S must be divisible by 2**levels.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Forward Haar decomposition
# ---------------------------------------------------------------------------
def _haar_step(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One orthonormal Haar step along dim=-2.

    Input:  x [..., S, d], S even.
    Output: (L, H) each [..., S/2, d].
        L[k] = (x[2k] + x[2k+1]) / sqrt(2)
        H[k] = (x[2k] - x[2k+1]) / sqrt(2)
    """
    S = x.shape[-2]
    assert S % 2 == 0, f"Haar step needs even length on dim=-2, got {S}"
    even = x[..., 0::2, :]
    odd = x[..., 1::2, :]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return (even + odd) * inv_sqrt2, (even - odd) * inv_sqrt2


def haar_decompose(
    x: torch.Tensor, levels: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Multi-level orthonormal Haar decomposition along dim=-2.

    Returns:
        L_final: [..., S/2**levels, d]  — coarsest low-pass approximation.
        H_list:  [H_1, H_2, ..., H_levels], where H_l has [..., S/2**l, d].
    """
    S = x.shape[-2]
    assert S % (1 << levels) == 0, (
        f"S={S} not divisible by 2**levels={1 << levels}"
    )
    H_list: list[torch.Tensor] = []
    cur = x
    for _ in range(levels):
        cur, h = _haar_step(cur)
        H_list.append(h)
    return cur, H_list


# ---------------------------------------------------------------------------
# Per-page features (query-agnostic)
# ---------------------------------------------------------------------------
def haar_page_features(
    paged_k: torch.Tensor, levels: int = 3, top_k: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Query-agnostic per-page features derived from Haar coefficients.

    Args:
        paged_k: [..., P, S, d] keys grouped into pages.
        levels:  Haar decomposition depth (default 3 → HARP plan).
        top_k:   Number of H coefficients to keep per page when computing
                 ``s_top`` (default ``S // 8`` matching HARP's N/8 selection).

    Returns dict of [..., P] tensors:
        s_max      — max ||H||_2 over all H coefficients in the page.
        s_sum      — sum  ||H||_2 over all H coefficients in the page.
        s_top      — sum of top-K  ||H||_2 (default K = S/8).
        s_l3_norm  — Frobenius norm of L_3 per page (low-pass energy).
        s_total_var — sum_k ||K[k] - K_block_mean||^2 (raw page variance,
                      cross-check for "K varies a lot ⇒ high mass" premise).
    """
    S = paged_k.shape[-2]
    if top_k is None:
        top_k = max(1, S // 8)
    L_final, H_list = haar_decompose(paged_k, levels)
    # ||H||_2 over hidden_dim d for every H coefficient.
    norms_per_level = [(h * h).sum(dim=-1).sqrt() for h in H_list]
    # Concatenate along the within-page coefficient axis.
    norms_cat = torch.cat(norms_per_level, dim=-1)  # [..., P, sum_n]
    k = min(top_k, norms_cat.shape[-1])
    top_vals, _ = torch.topk(norms_cat, k, dim=-1)

    l3_flat = L_final.reshape(*L_final.shape[:-2], -1)            # [..., P, blk*d]
    s_l3_norm = (l3_flat * l3_flat).sum(dim=-1).sqrt()

    # Raw within-page variance (Parseval check; should ≈ s_sum² for orthonormal Haar).
    block_mean = paged_k.mean(dim=-2, keepdim=True)
    centered = paged_k - block_mean
    s_total_var = (centered * centered).sum(dim=(-1, -2))

    return {
        "s_max":       norms_cat.amax(dim=-1),
        "s_sum":       norms_cat.sum(dim=-1),
        "s_top":       top_vals.sum(dim=-1),
        "s_l3_norm":   s_l3_norm,
        "s_total_var": s_total_var,
    }


# ---------------------------------------------------------------------------
# Top-K H selection + adaptive 2-token reconstruction
# ---------------------------------------------------------------------------
def haar_topk_per_page(
    H_list: list[torch.Tensor], top_k: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-page top-K selection of H coefficients across all levels.

    Args:
        H_list: list of [..., n_l, d] (one per level).
        top_k:  number of H vectors kept per page (across all levels combined).

    Returns:
        masks_per_level: list of bool tensors [..., n_l], True at kept positions.
        norms_per_level: list of [..., n_l] L2 norms (computed once, reused).
    """
    norms = [(h * h).sum(dim=-1).sqrt() for h in H_list]
    norms_cat = torch.cat(norms, dim=-1)
    k = min(top_k, norms_cat.shape[-1])
    _, idx = torch.topk(norms_cat, k, dim=-1)
    mask_cat = torch.zeros_like(norms_cat, dtype=torch.bool)
    mask_cat.scatter_(-1, idx, True)
    masks: list[torch.Tensor] = []
    offset = 0
    for h in H_list:
        n_l = h.shape[-2]
        masks.append(mask_cat[..., offset:offset + n_l])
        offset += n_l
    return masks, norms


def haar_representative_tokens(
    paged_k: torch.Tensor,
    levels: int = 3,
    top_k: Optional[int] = None,
    beta: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build HARP Phase-2 representative tokens via L_3 ± β · H_best per block.

    For each L_3 block, find the highest-L2-norm H coefficient (across all
    levels) whose support covers that block AND was kept by per-page top-K.
    If at least one such H exists, output two reps (L_3 + β H, L_3 - β H);
    otherwise output two duplicates of L_3 (keeps tensor shape constant).
    Stage-0 scoring uses ``score = max_r q · rep_r``, so duplicates do not
    inflate scores.

    Args:
        paged_k: [..., P, S, d].
        levels:  Haar depth (default 3 — one L_3 block covers 8 tokens).
        top_k:   N/8 by default (matches HARP plan).
        beta:    1/√2 by default (pure inverse-Haar weight; equivalent to
                 fully restoring the H pair's energy on the two L_2 children).

    Returns:
        rep:      [..., P, 2 * (S / 2**levels), d] representative tokens.
        has_any:  [..., P, S / 2**levels] bool — True where injection happened
                  (useful for adaptive scoring / budget metrics).
    """
    if top_k is None:
        top_k = max(1, paged_k.shape[-2] // 8)
    if beta is None:
        beta = 1.0 / math.sqrt(2.0)

    L_final, H_list = haar_decompose(paged_k, levels)
    # L_final: [..., P, num_blocks, d], num_blocks = S / 2**levels
    num_blocks = L_final.shape[-2]
    d = L_final.shape[-1]
    leading_shape = L_final.shape[:-2]

    masks, norms = haar_topk_per_page(H_list, top_k)

    inject = torch.zeros_like(L_final)                               # [..., num_blocks, d]
    best_norm = torch.full(
        L_final.shape[:-1], float("-inf"),
        device=paged_k.device, dtype=paged_k.dtype,
    )                                                                # [..., num_blocks]
    has_any = torch.zeros_like(best_norm, dtype=torch.bool)

    NEG_INF = float("-inf")
    for l_idx, (h, m, n) in enumerate(zip(H_list, masks, norms)):
        level = l_idx + 1
        cells_per_block = 1 << (levels - level)            # H_l indices per L_3 block
        n_l = h.shape[-2]
        assert n_l == num_blocks * cells_per_block

        # Reshape H_l, mask, norm to [..., num_blocks, cells_per_block, ...].
        h_b = h.view(*leading_shape, num_blocks, cells_per_block, d)
        m_b = m.view(*leading_shape, num_blocks, cells_per_block)
        n_b = n.view(*leading_shape, num_blocks, cells_per_block)

        # Suppress non-selected positions to -inf so they cannot win argmax.
        masked = torch.where(
            m_b, n_b,
            torch.full_like(n_b, NEG_INF),
        )
        block_max, block_argmax = masked.max(dim=-1)        # [..., num_blocks]
        valid = block_max > NEG_INF                         # [..., num_blocks]

        # Gather the chosen H vector for each block.
        gather_idx = block_argmax.unsqueeze(-1).unsqueeze(-1).expand(
            *block_argmax.shape, 1, d,
        )
        gathered = torch.gather(h_b, dim=-2, index=gather_idx).squeeze(-2)
        # gathered: [..., num_blocks, d]

        # Update if this level provides a new per-block best.
        new_best = valid & (block_max > best_norm)
        update_v = new_best.unsqueeze(-1).expand_as(gathered)
        inject = torch.where(update_v, gathered, inject)
        best_norm = torch.where(new_best, block_max, best_norm)
        has_any = has_any | valid

    rep_a = L_final + beta * inject
    rep_b = L_final - beta * inject
    rep = torch.stack([rep_a, rep_b], dim=-2)
    rep = rep.view(*leading_shape, num_blocks * 2, d)
    return rep, has_any


# ---------------------------------------------------------------------------
# Query-aware page score using Haar representatives
# ---------------------------------------------------------------------------
def haar_score_per_page(
    query_states: torch.Tensor,   # [1, H_q, 1, d]
    paged_k: torch.Tensor,        # [1, H_kv, P, S, d]
    num_kv_groups: int,
    levels: int = 3,
    top_k: Optional[int] = None,
    beta: Optional[float] = None,
    use_injection: bool = True,
    group_agg: str = "mean",
) -> torch.Tensor:
    """Per-(query head, page) Haar proxy score.

    score[h, p] = (1/√d) · max_r ⟨q[h], rep[p, r]⟩
    rep is either pure L_3 (use_injection=False) or HARP's adaptive ±β·H reps.

    Returns: [H_q, P] (per-query-head) and then reduces to [H_kv, P] via
    ``group_agg`` ("mean" or "max"). Mirrors the convention used by other
    selectors in attention_mass_recall_ruler.py.
    """
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert bsz == 1 and q_len == 1
    assert H_q == H_kv * num_kv_groups
    scale = 1.0 / math.sqrt(d)

    if use_injection:
        rep, _ = haar_representative_tokens(
            paged_k, levels=levels, top_k=top_k, beta=beta,
        )
    else:
        rep, _ = haar_decompose(paged_k, levels)

    rep_q = rep.repeat_interleave(num_kv_groups, dim=1).float()       # [1, H_q, P, R, d]
    q = query_states.float()                                          # [1, H_q, 1, d]
    scores_per_rep = torch.einsum("bhqd,bhprd->bhpr", q, rep_q) * scale
    score_q = scores_per_rep.amax(dim=-1).squeeze(0)                  # [H_q, P]

    score_g = score_q.view(H_kv, num_kv_groups, P)
    if group_agg == "max":
        return score_g.max(dim=1).values                              # [H_kv, P]
    return score_g.mean(dim=1)                                        # [H_kv, P]


# ---------------------------------------------------------------------------
# Self-test (numeric sanity, no model required)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    B, H_kv, P, S, d = 1, 2, 3, 32, 64
    x = torch.randn(B, H_kv, P, S, d, dtype=torch.float64)

    # Energy preservation (Parseval): sum of squared coeffs equals sum of x².
    L, H_list = haar_decompose(x, levels=3)
    coeff_energy = (L * L).sum() + sum((h * h).sum() for h in H_list)
    x_energy = (x * x).sum()
    rel = float((coeff_energy - x_energy).abs() / x_energy.clamp(min=1e-12))
    assert rel < 1e-10, f"Parseval violated: rel={rel}"

    # Top-K + reconstruct: with top_k = number of H coeffs, reps should
    # reconstruct the original two children of each L_3 block exactly when
    # beta = 1/√2 and we pick one H per pair at every level. We don't test
    # full reconstruction here; instead verify shapes + has_any logic.
    rep, has_any = haar_representative_tokens(x, levels=3, top_k=4, beta=None)
    assert rep.shape == (B, H_kv, P, 2 * (S // 8), d), rep.shape
    assert has_any.shape == (B, H_kv, P, S // 8), has_any.shape
    # With top_k=4 over (16+8+4)=28 coeffs, at least one block must be hit.
    assert bool(has_any.any()), "top-K=4 should hit at least one block"

    # Features sanity: s_sum >= s_top >= s_max for any non-degenerate page.
    feats = haar_page_features(x, levels=3)
    s_sum, s_top, s_max = feats["s_sum"], feats["s_top"], feats["s_max"]
    assert torch.all(s_sum >= s_top - 1e-6), "s_sum < s_top"
    assert torch.all(s_top >= s_max - 1e-6), "s_top < s_max"

    # Score function: shapes + finiteness.
    q = torch.randn(1, H_kv * 2, 1, d, dtype=torch.float64)
    scores = haar_score_per_page(
        q, x.float(), num_kv_groups=2, levels=3, top_k=4,
    )
    assert scores.shape == (H_kv, P), scores.shape
    assert torch.isfinite(scores).all(), "non-finite scores"

    print("haar_kernels self-test OK")


if __name__ == "__main__":
    _self_test()
