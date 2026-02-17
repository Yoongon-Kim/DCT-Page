"""
DCT Page Attention: Sparse attention via DCT-compressed page representatives.

Decode-only page selection: prefill uses standard full attention, page attention
is applied only during decode (q_len=1) to select top-k relevant pages from
the KV cache.
"""

import math
import warnings
from copy import copy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache

from config import DCTPageConfig


# ---------------------------------------------------------------------------
# Global config — set by replace_qwen2_attn()
# ---------------------------------------------------------------------------
_dct_page_cfg: Optional[DCTPageConfig] = None


# ---------------------------------------------------------------------------
# DCT / IDCT (copied from FreqKV)
# ---------------------------------------------------------------------------
def dct(x, norm='ortho'):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    
    => Instead of computing DCT directly (which is O(N^2)), this uses the mathematical equivalence: 
        reorder elements -> FFT(O(NlogN)) -> multiply by twiddle factors -> DCT result
    """
    x_shape = x.shape
    N = x_shape[-1] # N=squence length
    x = x.contiguous().view(-1, N) # x.shape=(B,N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1) # [x[0], x[2], x[4], ..., x[5], x[3], x[1]] => This specific reordering makes the FFT of v directly related to the DCT of x.
    Vc = torch.fft.fft(v.to(torch.float32), dim=1) # Standard FFT on the reordered signal. Cast to float32 for numerical precision. Vc is complex-valued: has .real and .imag parts.

    # We are trying to twiddle Vc so that we can make FFT of v into DCT of x. 
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k) # Real part of exp(-j*pi*k' / 2N) where k=-pi*k'/2N
    W_i = torch.sin(k) # Imag part of exp(-j*pi*k' / 2N) where k=-pi*k'/2N

    # The real part of Vc*exp(j*k) is the DCT of x.
    V = Vc.real * W_r - Vc.imag * W_i

    # The 'ortho' normalization makes the DCT matrix orthonormal - meaning 'idct(dct(x))==x' without any extra scaling.
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2       # DC component (index 0) scaled by 1/(2*sqrt(N))
        V[:, 1:] /= np.sqrt(N / 2) * 2  # All other components scaled by 1/(2*sqrt(N/2))

    V = 2 * V.view(*x_shape) # part of the DCT-II formula, then reshape back to the original dim
    return V


def idct(X, norm='ortho'):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III.
    Our definition of idct is that idct(dct(x)) == x
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    
    => The exact reverse pipeline of dct:
        undo normalization -> conjugate twiddle factors -> complex multiply -> IFFT -> undo reordering -> original signal
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2 # Flatten to [batch, N] and undo the 2 * V at the end of dct
    if norm == 'ortho': # Undo orthonormal normalization
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    # Twiddle factors (conjugate of dct's => twiddle in the reverse direction)
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N) # k is positive here, because the inverse requires the complex conjugate twiddle factor
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # Build a complex signal from real DCT coefficients
    # The DCT coefficients are the real part of Vc*exp(-j*k), but IFFT needs complex input.
    # This constructs the imaginary part using the Hermitian symmetry property.
    # If you do FFT on real value signal, the result always satisfies the Hermitian symmetry property.
    V_t_r = X_v # Imaginary part of index 0 is zero
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1) # Remaining imaginary parts are the reversed, negated DCT coefficients. However, this is not the exact definition of the Hermitian symmetry property. This code is true only if X_v is the FFT of an even-extended real signal, which uniquely determines the imaginary spectrum from the real one.

    V_r = V_t_r * W_r - V_t_i * W_i # Real part of V_t*exp(j*k)
    V_i = V_t_r * W_i + V_t_i * W_r # Imag part of V_t*exp(j*k)

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2) # V_r.shape == (B, N), V_r.unsqueeze(2).shape == (B, N, 1), V.shape == (B, N, 2)
    V = torch.view_as_complex(V) # V.shape == (B, N), V.dtype == complex64(may not be exact but complex datatype anyway)

    v = torch.fft.ifft(V, dim=1).real # v should be close to real, but because of error in calculation or dropping high-freq components, we need to drop any tiny floating-point imaginary residue.
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)] # First half of v -> even indices
    x[:, 1::2] += v.flip([1])[:, :N // 2] # Reversed second half of v -> odd indices

    return x.view(*x_shape)


# ---------------------------------------------------------------------------
# DCT page compression
# ---------------------------------------------------------------------------
def dct_compress_page(x, compressed_len):
    """
    Compress a KV tensor along the sequence dimension using DCT.

    Args:
        x: [bsz, num_heads, seq_len, head_dim], seq_len: sequence length per page
        compressed_len: target sequence length after compression

    Returns:
        [bsz, num_heads, compressed_len, head_dim]
    """
    if compressed_len >= x.shape[2]:
        return x

    bsz, num_heads, seq_len, head_dim = x.shape
    # Merge heads: [bsz, seq_len, num_heads * head_dim]
    x_merged = x.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim) # We compress along the seq dim, so all feature dimensions are processed together.

    # DCT along sequence dim: transpose so seq is last dim
    # [bsz, num_heads * head_dim, seq_len]
    x_dct = dct(x_merged.transpose(1, 2), norm='ortho')
    x_dct = x_dct[:, :, :compressed_len] # Keep only the low-frequency coefficients. x_dct.shape=(bsz, num_heads * head_dim, compressed_len)
    x_idct = idct(x_dct, norm='ortho').transpose(1, 2) * math.sqrt(compressed_len / seq_len) # '* math.sqrt(compressed_len / seq_len)': energy correction factor. This is added because the Parseval's theorem energy is reduced, as we dropped coefficients.

    compressed = x_idct.to(x.dtype)
    return compressed.reshape(bsz, compressed_len, num_heads, head_dim).transpose(1, 2)


# ---------------------------------------------------------------------------
# RoPE helpers (for continuous RoPE in continuous_rope mode)
# ---------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """Apply rotary position embedding to a single tensor.

    Args:
        x:   [bsz, num_heads, seq_len, head_dim]
        cos: [1, 1, seq_len, head_dim]
        sin: [1, 1, seq_len, head_dim]
    """
    return (x * cos) + (_rotate_half(x) * sin)


def _compute_rope_cos_sin(positions, head_dim, rope_theta, device, dtype):
    """Compute cos/sin for arbitrary positions (matches Qwen2RotaryEmbedding).

    Args:
        positions: [seq_len] integer tensor
        head_dim:  int
        rope_theta: float (from model config)

    Returns:
        cos, sin: each [1, 1, seq_len, head_dim]
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions.float(), inv_freq)   # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)             # [seq_len, head_dim]
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)  # [1, 1, seq_len, head_dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
    return cos, sin


# ---------------------------------------------------------------------------
# Page construction and compression
# ---------------------------------------------------------------------------
def build_pages_and_compress(key_states, value_states, cfg):
    """
    Divide KV cache into sink / pages / recent and compress pages via DCT.

    Layout: [sink | page_0 | page_1 | ... | page_{N-1} | recent + leftover]

    All pages are exactly `page_size` tokens. The last partial page (leftover)
    is merged into the recent window.

    Args:
        key_states:   [bsz, num_kv_heads, kv_len, head_dim]
        value_states: [bsz, num_kv_heads, kv_len, head_dim]
        cfg: DCTPageConfig

    Returns:
        sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
        recent_k, recent_v, num_pages, actual_recent
    """
    bsz, num_kv_heads, kv_len, head_dim = key_states.shape

    # Segmentation
    pageable_len = kv_len - cfg.sink_size - cfg.recent_size
    num_pages = pageable_len // cfg.page_size
    leftover = pageable_len % cfg.page_size
    actual_recent = cfg.recent_size + leftover

    # Sink
    sink_k = key_states[:, :, :cfg.sink_size]
    sink_v = value_states[:, :, :cfg.sink_size]

    # Pages: [sink_size, sink_size + num_pages * page_size)
    pages_end = cfg.sink_size + num_pages * cfg.page_size
    pages_k = key_states[:, :, cfg.sink_size:pages_end]
    pages_v = value_states[:, :, cfg.sink_size:pages_end]

    # Reshape to [bsz, num_kv_heads, num_pages, page_size, head_dim]
    paged_k = pages_k.view(bsz, num_kv_heads, num_pages, cfg.page_size, head_dim)
    paged_v = pages_v.view(bsz, num_kv_heads, num_pages, cfg.page_size, head_dim)

    # Recent (includes leftover)
    recent_k = key_states[:, :, pages_end:]
    recent_v = value_states[:, :, pages_end:]

    # Batched DCT compression of all pages
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))

    # Flatten pages into batch dim: [bsz * num_kv_heads * num_pages, 1, page_size, head_dim]
    flat_k = paged_k.reshape(bsz * num_kv_heads * num_pages, 1, cfg.page_size, head_dim)
    flat_v = paged_v.reshape(bsz * num_kv_heads * num_pages, 1, cfg.page_size, head_dim)

    comp_k = dct_compress_page(flat_k, comp_size)
    comp_v = dct_compress_page(flat_v, comp_size)

    # Reshape back: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
    comp_k = comp_k.view(bsz, num_kv_heads, num_pages, comp_size, head_dim)
    comp_v = comp_v.view(bsz, num_kv_heads, num_pages, comp_size, head_dim)

    return (sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
            recent_k, recent_v, num_pages, actual_recent)


# ---------------------------------------------------------------------------
# Page scoring
# ---------------------------------------------------------------------------
def score_pages(query_states, compressed_keys, cfg, num_kv_groups):
    """
    Score each page per KV-head group (group-aware selection).
    Each GQA group independently selects its own top-k pages.

    Scoring uses all query heads without expanding keys: queries are reshaped
    into groups, and einsum broadcasts the KV head over query heads in the
    group. Per-head max scores are averaged within each group to produce one
    score per page per group.

    Args:
        query_states:    [bsz, num_heads, q_len, head_dim]  (q_len=1 during decode)
        compressed_keys: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
        cfg: DCTPageConfig
        num_kv_groups: int (num_heads // num_kv_heads, for GQA)

    Returns:
        selected_indices: [bsz, num_kv_heads, actual_top_k] — sorted page indices per group
    """
    bsz, num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, num_pages, comp_size, _ = compressed_keys.shape
    scaling = head_dim ** -0.5

    # Reshape queries into GQA groups: [bsz, num_kv_heads, num_kv_groups, q_len, head_dim]
    query_grouped = query_states.view(bsz, num_kv_heads, num_kv_groups, q_len, head_dim)

    # Score each query head against its group's KV head (no key expansion needed).
    # einsum broadcasts compressed_keys over the num_kv_groups dim.
    # -> [bsz, num_kv_heads, num_kv_groups, q_len, num_pages, comp_size]
    scores = torch.einsum(
        'bgiqd,bgncd->bgiqnc',
        query_grouped * scaling, compressed_keys
    )

    # Reduce over comp_size (compressed tokens within each page)
    if cfg.scoring_method == "max":
        page_scores = scores.max(dim=-1).values   # [bsz, num_kv_heads, num_kv_groups, q_len, num_pages]
    elif cfg.scoring_method == "mean":
        page_scores = scores.mean(dim=-1)
    elif cfg.scoring_method == "sum":
        page_scores = scores.sum(dim=-1)
    else:
        raise ValueError(f"Unknown scoring method: {cfg.scoring_method}")

    # Reduce over query positions -> [bsz, num_kv_heads, num_kv_groups, num_pages]
    page_scores = page_scores.mean(dim=3)

    # Aggregate per-head scores within each GQA group -> [bsz, num_kv_heads, num_pages]
    if cfg.group_agg_method == "mean":
        page_scores = page_scores.mean(dim=2)
    elif cfg.group_agg_method == "max":
        page_scores = page_scores.max(dim=2).values
    elif cfg.group_agg_method == "topp":
        # Each head votes for its top-p pages, then pages are ranked by vote count.
        # p = top_k so each head nominates as many pages as will be selected.
        p = min(cfg.top_k, num_pages)
        _, head_topk = torch.topk(page_scores, p, dim=-1)  # [bsz, num_kv_heads, num_kv_groups, p]
        votes = torch.zeros(
            bsz, num_kv_heads, num_pages,
            dtype=page_scores.dtype, device=page_scores.device
        )
        votes.scatter_add_(
            2,
            head_topk.reshape(bsz, num_kv_heads, num_kv_groups * p),
            torch.ones(bsz, num_kv_heads, num_kv_groups * p,
                        dtype=page_scores.dtype, device=page_scores.device)
        )
        # Use vote count as primary score, mean attention score as tiebreaker
        mean_scores = page_scores.mean(dim=2)
        # Normalize tiebreaker to [0, 1) so it never outweighs a single vote
        score_range = mean_scores.max(dim=-1, keepdim=True).values - mean_scores.min(dim=-1, keepdim=True).values
        tiebreaker = (mean_scores - mean_scores.min(dim=-1, keepdim=True).values) / (score_range + 1e-8)
        page_scores = votes + tiebreaker
    else:
        raise ValueError(f"Unknown group_agg_method: {cfg.group_agg_method}")

    # Select top-k per KV head group
    actual_top_k = min(cfg.top_k, num_pages)
    _, selected_indices = torch.topk(page_scores, actual_top_k, dim=-1)

    # Sort for positional ordering within each group
    selected_indices, _ = selected_indices.sort(dim=-1)

    return selected_indices, page_scores


def rescore_with_full_keys(query_states, paged_k, candidate_indices, cfg, num_kv_groups):
    """
    Re-score candidate pages using full (uncompressed) keys to select final top-k.

    Args:
        query_states:      [bsz, num_heads, q_len, head_dim]
        paged_k:           [bsz, num_kv_heads, num_pages, page_size, head_dim]
        candidate_indices: [bsz, num_kv_heads, num_candidates] (sorted, from coarse scoring)
        cfg: DCTPageConfig
        num_kv_groups: int

    Returns:
        selected_indices: [bsz, num_kv_heads, top_k] — sorted global page indices
    """
    bsz, num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, _, page_size, _ = paged_k.shape
    num_candidates = candidate_indices.shape[2]
    scaling = head_dim ** -0.5

    # Gather candidate pages: [bsz, num_kv_heads, num_candidates, page_size, head_dim]
    idx = candidate_indices[:, :, :, None, None].expand(
        bsz, num_kv_heads, num_candidates, page_size, head_dim
    )
    candidate_keys = torch.gather(paged_k, 2, idx)

    # Reshape queries into GQA groups: [bsz, num_kv_heads, num_kv_groups, q_len, head_dim]
    query_grouped = query_states.view(bsz, num_kv_heads, num_kv_groups, q_len, head_dim)

    # Score: [bsz, num_kv_heads, num_kv_groups, q_len, num_candidates, page_size]
    scores = torch.einsum(
        'bgiqd,bgncd->bgiqnc',
        query_grouped * scaling, candidate_keys
    )

    # Reduce over page_size (tokens within each page)
    if cfg.scoring_method == "max":
        page_scores = scores.max(dim=-1).values
    elif cfg.scoring_method == "mean":
        page_scores = scores.mean(dim=-1)
    elif cfg.scoring_method == "sum":
        page_scores = scores.sum(dim=-1)
    else:
        raise ValueError(f"Unknown scoring method: {cfg.scoring_method}")

    # Reduce over query positions -> [bsz, num_kv_heads, num_kv_groups, num_candidates]
    page_scores = page_scores.mean(dim=3)

    # Aggregate within GQA group -> [bsz, num_kv_heads, num_candidates]
    if cfg.group_agg_method == "mean":
        page_scores = page_scores.mean(dim=2)
    elif cfg.group_agg_method == "max":
        page_scores = page_scores.max(dim=2).values
    elif cfg.group_agg_method == "topp":
        actual_top_k = min(cfg.top_k, num_candidates)
        _, head_topk = torch.topk(page_scores, actual_top_k, dim=-1)
        votes = torch.zeros(
            bsz, num_kv_heads, num_candidates,
            dtype=page_scores.dtype, device=page_scores.device
        )
        votes.scatter_add_(
            2,
            head_topk.reshape(bsz, num_kv_heads, num_kv_groups * actual_top_k),
            torch.ones(bsz, num_kv_heads, num_kv_groups * actual_top_k,
                        dtype=page_scores.dtype, device=page_scores.device)
        )
        mean_scores = page_scores.mean(dim=2)
        score_range = mean_scores.max(dim=-1, keepdim=True).values - mean_scores.min(dim=-1, keepdim=True).values
        tiebreaker = (mean_scores - mean_scores.min(dim=-1, keepdim=True).values) / (score_range + 1e-8)
        page_scores = votes + tiebreaker
    else:
        raise ValueError(f"Unknown group_agg_method: {cfg.group_agg_method}")

    # Select top-k from candidates
    actual_top_k = min(cfg.top_k, num_candidates)
    _, local_indices = torch.topk(page_scores, actual_top_k, dim=-1)

    # Map local indices back to global page indices
    selected_indices = torch.gather(candidate_indices, 2, local_indices)

    # Sort for positional ordering
    selected_indices, _ = selected_indices.sort(dim=-1)

    return selected_indices


# ---------------------------------------------------------------------------
# KV assembly
# ---------------------------------------------------------------------------
def assemble_kv(sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
                recent_k, recent_v, selected_indices, cfg, num_pages):
    """
    Assemble the final K/V tensors from sink + selected pages + recent.
    Each KV head independently gathers its own selected pages (group-aware).

    Args:
        sink_k/v:     [bsz, num_kv_heads, sink_size, head_dim]
        paged_k/v:    [bsz, num_kv_heads, num_pages, page_size, head_dim]
        comp_k/v:     [bsz, num_kv_heads, num_pages, comp_size, head_dim]
        recent_k/v:   [bsz, num_kv_heads, actual_recent, head_dim]
        selected_indices: [bsz, num_kv_heads, top_k]
        cfg: DCTPageConfig
        num_pages: int

    Returns:
        final_k, final_v: [bsz, num_kv_heads, assembled_len, head_dim]
    """
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape
    top_k = selected_indices.shape[2]

    if cfg.unselected_mode == "drop":
        # Gather selected full pages per KV head
        idx = selected_indices[:, :, :, None, None].expand(
            bsz, num_kv_heads, top_k, page_size, head_dim
        )
        sel_k = torch.gather(paged_k, 2, idx)  # [bsz, num_kv_heads, top_k, page_size, head_dim]
        sel_v = torch.gather(paged_v, 2, idx)

        # Flatten: [bsz, num_kv_heads, top_k * page_size, head_dim]
        sel_k = sel_k.reshape(bsz, num_kv_heads, top_k * page_size, head_dim)
        sel_v = sel_v.reshape(bsz, num_kv_heads, top_k * page_size, head_dim)

        final_k = torch.cat([sink_k, sel_k, recent_k], dim=2)
        final_v = torch.cat([sink_v, sel_v, recent_v], dim=2)

    elif cfg.unselected_mode == "compressed":
        comp_size = comp_k.shape[3]
        num_unselected = num_pages - top_k
        middle_len = top_k * page_size + num_unselected * comp_size

        # Build selection mask and compute per-page write offsets
        selected_mask = torch.zeros(
            bsz, num_kv_heads, num_pages, dtype=torch.bool,
            device=selected_indices.device
        )
        selected_mask.scatter_(2, selected_indices, True)

        # Each page contributes page_size (if selected) or comp_size (if not) tokens
        token_counts = torch.where(selected_mask, page_size, comp_size)
        # Exclusive cumsum: page_offsets[p] = start position of page p in middle section
        page_offsets = torch.zeros(
            bsz, num_kv_heads, num_pages, dtype=torch.long,
            device=selected_indices.device
        )
        page_offsets[:, :, 1:] = token_counts[:, :, :-1].cumsum(dim=-1) # cumsum is cumulative sum

        # Gather selected pages (full resolution) per KV head
        idx_sel = selected_indices[:, :, :, None, None].expand(
            bsz, num_kv_heads, top_k, page_size, head_dim
        )
        sel_k = torch.gather(paged_k, 2, idx_sel).reshape(
            bsz, num_kv_heads, top_k * page_size, head_dim
        )
        sel_v = torch.gather(paged_v, 2, idx_sel).reshape(
            bsz, num_kv_heads, top_k * page_size, head_dim
        )

        # Derive unselected indices (positional order via stable argsort)
        sort_perm = torch.argsort(selected_mask.int(), dim=-1, stable=True) # argsort returns a list of indexs
        unselected_indices = sort_perm[:, :, :num_unselected]

        # Gather unselected pages (compressed) per KV head
        idx_unsel = unselected_indices[:, :, :, None, None].expand(
            bsz, num_kv_heads, num_unselected, comp_size, head_dim
        )
        unsel_k = torch.gather(comp_k, 2, idx_unsel).reshape(
            bsz, num_kv_heads, num_unselected * comp_size, head_dim
        )
        unsel_v = torch.gather(comp_v, 2, idx_unsel).reshape(
            bsz, num_kv_heads, num_unselected * comp_size, head_dim
        )

        # Compute per-token write offsets to interleave in original page order
        # Selected pages: each contributes page_size tokens at page_offsets[selected_page]
        sel_offsets = torch.gather(page_offsets, 2, selected_indices)  # sel_offsets: [bsz, num_kv_heads, top_k] - positional offsets of selected pages. selected_indices: [bsz, num_kv_heads, top_k] - the page indices each KV head selected. page_offsets: [bsz, num_kv_heads, num_pages] - the starting write positions of each page in the middle section.
        sel_dst = (
            sel_offsets[:, :, :, None] # Adds a dimension for broadcasting (broadcasting: saving individual tokens' positions)
            + torch.arange(page_size, device=sel_offsets.device) # Broadcasts [0, 1, 2, ... , page_size-1]
        ).reshape(bsz, num_kv_heads, top_k * page_size) # The result is the token positions for all selected pages' tokens.

        # Unselected pages: each contributes comp_size tokens
        unsel_offsets = torch.gather(page_offsets, 2, unselected_indices)  # unsel_offsets: [bsz, num_kv_heads, num_unselected] - positional offsets of unselected pages. unselected_indices: [bsz, num_kv_heads, num_unselected] - the page indices each KV head not selected.
        unsel_dst = (
            unsel_offsets[:, :, :, None]
            + torch.arange(comp_size, device=unsel_offsets.device)
        ).reshape(bsz, num_kv_heads, num_unselected * comp_size) # The result is the token positions for all unselected pages' tokens.

        # Scatter into middle section preserving original page order
        middle_k = torch.empty(
            bsz, num_kv_heads, middle_len, head_dim,
            dtype=sink_k.dtype, device=sink_k.device
        )
        middle_v = torch.empty_like(middle_k)

        middle_k.scatter_(2, sel_dst[:, :, :, None].expand_as(sel_k), sel_k) # 'sel_dst[:, :, :, None]' adds a head_dim dimension ([bsz, num_kv_heads, top_k*page_size]->[bsz, num_kv_heads, top_k*page_size, 1]). '.expand_as(sel_k)' broadcasts to [bsz, num_kv_heads, top_k*page_size, head_dim]. 'scatter_' writes values from sel_k into middle_k at positions specified by the index tensor, along dim 2 (the sequence dimension).
        middle_k.scatter_(2, unsel_dst[:, :, :, None].expand_as(unsel_k), unsel_k)
        middle_v.scatter_(2, sel_dst[:, :, :, None].expand_as(sel_v), sel_v)
        middle_v.scatter_(2, unsel_dst[:, :, :, None].expand_as(unsel_v), unsel_v)

        final_k = torch.cat([sink_k, middle_k, recent_k], dim=2)
        final_v = torch.cat([sink_v, middle_v, recent_v], dim=2)

    else:
        raise ValueError(f"Unknown unselected_mode: {cfg.unselected_mode}")

    return final_k, final_v


# ---------------------------------------------------------------------------
# Main attention forward (replaces Qwen2Attention.forward)
# ---------------------------------------------------------------------------
def dct_page_attention_forward(
    self, # the Qwen2Attention instance (we access its projections like self.q_proj, config like self.config, etc.)
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None, # The type can be torch.Tensor or None, and the default value is None
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """
    Replacement for Qwen2Attention.forward.

    - Prefill (q_len > 1): standard full causal attention.
    - Decode (q_len == 1, long KV cache): DCT page attention.
    """
    cfg = _dct_page_cfg
    input_shape = hidden_states.shape[:-1] # (bsz, q_len)
    hidden_shape = (*input_shape, -1, self.head_dim) # (bsz, q_len, num_heads, head_dim)
    bsz, q_len = input_shape

    # Step 1: Q/K/V projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_heads, q_len, head_dim)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa

    # Step 2 & 3: RoPE and KV cache
    cos, sin = position_embeddings

    if cfg.continuous_rope:
        # Compute RoPE'd Q/K for prefill/short-seq attention only
        query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # Cache stores pre-RoPE K/V (continuous RoPE applied after assembly during decode)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_cached, value_cached = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            key_cached, value_cached = key_states, value_states
        kv_len = key_cached.shape[2]
    else:
        # Standard: apply RoPE then cache
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        kv_len = key_states.shape[2]

    # Step 4: Branch — prefill vs decode
    min_len_for_paging = cfg.sink_size + cfg.page_size*(cfg.top_k+1) + cfg.recent_size # If the number of formed pages is smaller than top-k config, we do full attention
    if q_len > 1 or kv_len < min_len_for_paging:
        # PREFILL or SHORT SEQUENCE
        if cfg.continuous_rope:
            if q_len > 1:
                # Prefill: use locally RoPE'd Q/K (assumes single prefill call)
                attn_q, attn_k, attn_v = query_rope, key_rope, value_states
            else:
                # Short decode: apply standard RoPE to all cached pre-RoPE KV
                all_pos = torch.arange(kv_len, device=key_cached.device)
                cos_all, sin_all = _compute_rope_cos_sin(
                    all_pos, self.head_dim, self.config.rope_parameters['rope_theta'],
                    key_cached.device, key_cached.dtype
                )
                attn_q = query_rope
                attn_k = _apply_rope(key_cached, cos_all, sin_all)
                attn_v = value_cached
        else:
            attn_q, attn_k, attn_v = query_states, key_states, value_states

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            self,
            attn_q,
            attn_k,
            attn_v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ---- DECODE PATH (q_len == 1, long KV cache) ----

    if cfg.continuous_rope:
        # Use pre-RoPE KV from cache for page building and scoring
        key_states = key_cached
        value_states = value_cached

    # Pre-compute num_pages to check if page selection is needed
    pageable_len = kv_len - cfg.sink_size - cfg.recent_size
    num_pages = pageable_len // cfg.page_size

    # Step 5: Build pages and compress
    (sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
        recent_k, recent_v, num_pages, actual_recent) = build_pages_and_compress(
        key_states, value_states, cfg
    )

    # Step 6: Score pages and select top-k
    if cfg.selection_mode == "hierarchical":
        # Stage 1: coarse selection of 2*top_k candidates using compressed keys
        coarse_cfg = copy(cfg)
        coarse_cfg.top_k = min(2 * cfg.top_k, num_pages)
        candidate_indices, _ = score_pages(
            query_states, comp_k, coarse_cfg, self.num_key_value_groups
        )
        # Stage 2: re-score candidates with full keys, select final top_k
        selected_indices = rescore_with_full_keys(
            query_states, paged_k, candidate_indices, cfg,
            self.num_key_value_groups
        )
    else:
        selected_indices, _page_scores = score_pages(
            query_states, comp_k, cfg, self.num_key_value_groups
        )

    # Step 7: Assemble selected K/V
    final_k, final_v = assemble_kv(
        sink_k, sink_v, paged_k, paged_v, comp_k, comp_v,
        recent_k, recent_v, selected_indices, cfg, num_pages
    )

    # Step 7.5: Apply continuous RoPE (continuous_rope mode)
    if cfg.continuous_rope:
        assembled_len = final_k.shape[2]
        k_pos = torch.arange(assembled_len, device=final_k.device)
        q_pos = torch.tensor([assembled_len - 1], device=final_k.device)
        cos_k, sin_k = _compute_rope_cos_sin(
            k_pos, self.head_dim, self.config.rope_parameters['rope_theta'],
            final_k.device, final_k.dtype
        )
        cos_q, sin_q = _compute_rope_cos_sin(
            q_pos, self.head_dim, self.config.rope_parameters['rope_theta'],
            final_k.device, final_k.dtype
        )
        final_k = _apply_rope(final_k, cos_k, sin_k)
        query_states = _apply_rope(query_states, cos_q, sin_q)

    # Step 8: Compute attention (no causal mask needed for q_len=1)
    final_k_expanded = repeat_kv(final_k, self.num_key_value_groups)
    final_v_expanded = repeat_kv(final_v, self.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states, final_k_expanded.transpose(2, 3)
    ) * self.scaling
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, final_v_expanded)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # Step 9: Output projection
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Monkey-patch entry point
# ---------------------------------------------------------------------------
def replace_qwen2_attn(
    page_size=128,
    top_k=8,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.25,
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    selection_mode="standard",
    continuous_rope=False,
):
    """
    Replace Qwen2Attention.forward with DCT Page Attention.

    Must be called BEFORE loading the model.
    """
    global _dct_page_cfg
    _dct_page_cfg = DCTPageConfig(
        page_size=page_size,
        top_k=top_k,
        sink_size=sink_size,
        recent_size=recent_size,
        compress_ratio=compress_ratio,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        selection_mode=selection_mode,
        continuous_rope=continuous_rope,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config:")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, selection_mode={selection_mode}")
    print(f"  continuous_rope={continuous_rope}")
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = dct_page_attention_forward


def replace_llama_attn(
    page_size=128,
    top_k=8,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.25,
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    selection_mode="standard",
    continuous_rope=False,
):
    """
    Replace LlamaAttention.forward with DCT Page Attention.

    Must be called BEFORE loading the model.
    LlamaAttention has the same forward signature and attributes as Qwen2Attention,
    so we reuse dct_page_attention_forward directly.
    """
    global _dct_page_cfg
    _dct_page_cfg = DCTPageConfig(
        page_size=page_size,
        top_k=top_k,
        sink_size=sink_size,
        recent_size=recent_size,
        compress_ratio=compress_ratio,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        selection_mode=selection_mode,
        continuous_rope=continuous_rope,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config (Llama):")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, selection_mode={selection_mode}")
    print(f"  continuous_rope={continuous_rope}")
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.llama.modeling_llama.LlamaAttention.forward = dct_page_attention_forward
