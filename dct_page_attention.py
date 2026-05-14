"""
DCT Page Attention: Sparse attention via DCT-compressed page representatives.

Decode-only page selection: prefill uses standard full attention, page attention
is applied only during decode (q_len=1) to select top-k relevant pages from
the KV cache.
"""

import math
import warnings

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicLayer

from config import DCTPageConfig
from triton_kernels import (
    assemble_kv_split_triton,
    build_assemble_stride_cache,
    topk_sort,
    assemble_kv_drop_triton,
    score_pages_triton,
    apply_rope_q_direct,
    topk_sort_and_pack_triton,
)

# Module-level FlashInfer cache handle. The profiling driver / test harness
# builds the cache post-prefill via speed.flashinfer_backend.build_flashinfer_paged_cache
# and assigns it here BEFORE the first decode step. The FlashInfer forward
# reads this on every layer; layer 0 advances the cache counters.
_flashinfer_cache_ref = [None]

# Module-level upstream-FlashInfer cache handle. The eval forward lazy-builds
# this on the first decode step (layer 0); cleared by reset_upstream_fi_cache_state
# between every model.generate() call.
_upstream_fi_cache_ref = [None]

# ---------------------------------------------------------------------------
# Pre-allocated KV cache (avoids torch.cat during decode, fixes strides)
# ---------------------------------------------------------------------------
class PreAllocatedLayer(DynamicLayer):
    """Drop-in replacement for DynamicLayer that uses pre-allocated buffers.

    Instead of torch.cat (O(seq_len) alloc+copy per step), uses index
    assignment into a pre-allocated buffer (O(1) write per step).
    Strides remain fixed across all decode steps.

    FI mode (`_fi_mode=True`): the FlashInfer paged buf is the single source
    of truth at decode time. `keys`/`values` are freed after FI build, and
    `update()` becomes a counter-only shim — it advances `_seen` and returns
    `None, None`. Decode forwards must read KV from `cache.buf` directly.
    """

    _fi_mode = False  # class default; flipped to True on each layer at FI build

    @classmethod
    def from_dynamic_layer(cls, layer, extra_tokens):
        """Convert a populated DynamicLayer into a pre-allocated version."""
        new_layer = cls()
        k, v = layer.keys, layer.values
        bsz, heads, seq_len, dim = k.shape

        alloc_len = seq_len + extra_tokens
        new_layer.keys = torch.empty(bsz, heads, alloc_len, dim,
                                     dtype=k.dtype, device=k.device)
        new_layer.values = torch.empty(bsz, heads, alloc_len, dim,
                                       dtype=v.dtype, device=v.device)
        new_layer.keys[:, :, :seq_len, :] = k
        new_layer.values[:, :, :seq_len, :] = v

        new_layer._seen = seq_len
        new_layer._alloc_len = alloc_len
        new_layer.is_initialized = True
        new_layer.dtype = k.dtype
        new_layer.device = k.device
        return new_layer

    def update(self, key_states, value_states, cache_kwargs=None):
        seq_len = key_states.shape[-2]
        if self._fi_mode:
            # Counter-only shim: flat keys/values were freed at FI build.
            self._seen += seq_len
            return None, None
        # When _fi_mode=True, the grow path below at :97-105 is unreachable for upstream-FI generates (early return above).

        start = self._seen
        end = start + seq_len

        if end > self._alloc_len:
            # Grow the buffer by 4*page_size
            new_alloc = max(end, self._alloc_len + _dct_page_cfg.page_size*4)
            new_k = torch.empty(*self.keys.shape[:2], new_alloc, self.keys.shape[-1],
                                dtype=self.dtype, device=self.device)
            new_v = torch.empty_like(new_k)
            new_k[:, :, :start, :] = self.keys[:, :, :start, :]
            new_v[:, :, :start, :] = self.values[:, :, :start, :]
            self.keys = new_k
            self.values = new_v
            self._alloc_len = new_alloc

        self.keys[:, :, start:end, :] = key_states
        self.values[:, :, start:end, :] = value_states
        self._seen = end

        # Return view of valid portion (zero-copy, strides unchanged)
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_seq_length(self, cache_position=None):
        return self._seen


def pre_allocate_cache(cache, extra_tokens=256):
    """Convert a DynamicCache (after prefill) to use pre-allocated layers."""
    for i, layer in enumerate(cache.layers):
        cache.layers[i] = PreAllocatedLayer.from_dynamic_layer(layer, extra_tokens)
    return cache


# ---------------------------------------------------------------------------
# Global config / debug hook
# ---------------------------------------------------------------------------
_dct_page_cfg: Optional[DCTPageConfig] = None
_dct_page_debug_hook: Optional[Callable[[dict], None]] = None


def set_dct_page_debug_hook(hook: Optional[Callable[[dict], None]]) -> None:
    """Install an optional callback for decode-time page selection debugging."""
    global _dct_page_debug_hook
    _dct_page_debug_hook = hook


def _get_attention_interface(attn_module: nn.Module) -> Callable:
    """Mirror the upstream attention backend dispatch."""
    if attn_module.config._attn_implementation == "eager":
        return eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS[attn_module.config._attn_implementation]


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
# DCT Projection Matrix (replaces FFT with a single matmul)
# ---------------------------------------------------------------------------
def _build_dct_projection_matrix(page_size, comp_size, device, dtype):
    """Precompute the [comp_size, page_size] DCT-lowpass-IDCT projection matrix.

    The full DCT compression pipeline (DCT → keep leading comp_size
    coefficients → IDCT → energy correction) is a linear transform. We compute
    it by running dct_compress_page on an identity matrix.
    """
    I = torch.eye(page_size, device=device, dtype=torch.float32)
    I = I.unsqueeze(0).unsqueeze(0)  # [1, 1, page_size, page_size]
    M = dct_compress_page(I, comp_size)  # [1, 1, comp_size, page_size]
    # dct_compress_page ends with a transpose, so M is non-contiguous
    # with strides (1, COMP_SIZE). compress_pages_triton hardcodes
    # row-major access, so force a contiguous copy here (built once, cached).
    return M.squeeze(0).squeeze(0).contiguous().to(dtype)  # [comp_size, page_size]


def _build_block_haar_basis(block_size: int, dtype) -> torch.Tensor:
    """Build orthonormal Haar basis [block_size, block_size] (block_size must be 2^k).
    Coarse-to-fine ordering: row 0 = DC, row 1 = top-level wavelet, ..., last rows = finest wavelets.
    """
    assert block_size > 0 and (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    H = torch.zeros(block_size, block_size, dtype=torch.float32)
    H[0] = 1.0 / math.sqrt(block_size)
    row = 1
    scale = 1  # number of wavelet groups at this level
    while scale < block_size:
        span = block_size // scale          # tokens per wavelet at this level
        half = span // 2
        for grp in range(scale):
            base = grp * span
            v = 1.0 / math.sqrt(span)
            H[row, base : base + half] = v
            H[row, base + half : base + span] = -v
            row += 1
        scale *= 2
    return H.to(dtype)


def _build_haar_projection_matrix(page_size, comp_size, n_detail_per_block, device, dtype,
                                  detail_with_negation: bool = False):
    """Build Haar projection: comp_size lowpass + detail rows (optionally with ± negation pair).

    With detail_with_negation=True, every detail row is duplicated with its negation so that
    max(Q·row, Q·−row) = |Q·row| (max-aggregated scoring effectively uses |·|).
    Output shape: [comp_size * (1 + n_detail_per_block * (2 if negation else 1)), page_size].
    """
    assert page_size % comp_size == 0, "page_size must be divisible by comp_size"
    block_size = page_size // comp_size
    detail_mult = 2 if detail_with_negation else 1
    detail_rows_per_block = max(0, n_detail_per_block) * detail_mult
    out_rows = comp_size * (1 + detail_rows_per_block)
    H = torch.zeros(out_rows, page_size, device=device, dtype=dtype)
    lp_norm = 1.0 / math.sqrt(block_size)
    for i in range(comp_size):
        H[i, i * block_size : (i + 1) * block_size] = lp_norm
    if n_detail_per_block > 0:
        assert (block_size & (block_size - 1)) == 0, "n_detail_per_block > 0 requires power-of-2 block_size"
        per_block = _build_block_haar_basis(block_size, dtype=dtype).to(device)
        N = min(n_detail_per_block, block_size - 1)
        for i in range(comp_size):
            for j in range(N):
                row_off = comp_size + i * detail_rows_per_block + j * detail_mult
                H[row_off, i * block_size : (i + 1) * block_size] = per_block[j + 1]
                if detail_with_negation:
                    H[row_off + 1, i * block_size : (i + 1) * block_size] = -per_block[j + 1]
    return H.contiguous()


def _build_haar_c2f_projection_matrix(page_size, comp_size, device, dtype):
    """Haar coarse-to-fine truncation: first comp_size rows of the page-wide ortho Haar basis.

    Row 0 = DC (page mean), row 1 = top-level wavelet (first half − second half),
    rows 2-3 = next-level wavelets, etc. Spans the same K-dim subspace as scaling
    at depth log2(N/K) (= block means), but uses signed wavelet rows instead of
    positive block-mean indicators — different scoring behavior under Q·comp_K max.
    """
    assert page_size > 0 and (page_size & (page_size - 1)) == 0, "page_size must be power of 2"
    assert 0 < comp_size <= page_size
    H = _build_block_haar_basis(page_size, dtype=dtype).to(device)
    return H[:comp_size].contiguous()


def _build_dct_haar_projection_matrix(page_size, comp_size, detail_per_block, device, dtype,
                                       detail_with_negation: bool = False):
    """DCT lowpass + Haar detail. Optionally duplicates each detail row with its negation.

    Structure (rows in order):
      [0..comp_size-1]                        DCT lowpass rows.
      [comp_size..]                           Detail rows per block: detail_per_block rows per block,
                                              each optionally followed by its negation.
    Shape: [comp_size * (1 + detail_per_block * (2 if negation else 1)), page_size].
    """
    dct_rows = _build_dct_projection_matrix(page_size, comp_size, device, dtype)
    if detail_per_block <= 0:
        return dct_rows
    assert page_size % comp_size == 0, "page_size must be divisible by comp_size"
    block_size = page_size // comp_size
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2 for detail"
    per_block = _build_block_haar_basis(block_size, dtype=dtype).to(device)
    detail_mult = 2 if detail_with_negation else 1
    detail_rows_per_block = detail_per_block * detail_mult
    total = comp_size * (1 + detail_rows_per_block)
    M = torch.zeros(total, page_size, device=device, dtype=dtype)
    M[:comp_size] = dct_rows
    N = min(detail_per_block, block_size - 1)
    for i in range(comp_size):
        for j in range(N):
            row_off = comp_size + i * detail_rows_per_block + j * detail_mult
            M[row_off, i * block_size : (i + 1) * block_size] = per_block[j + 1]
            if detail_with_negation:
                M[row_off + 1, i * block_size : (i + 1) * block_size] = -per_block[j + 1]
    return M.contiguous()


def _proxy_effective_comp_size(cfg, base_comp_size: int) -> int:
    """Effective comp_size after Haar detail expansion (used for cache shape).
    'dct': base_comp_size unchanged.
    'haar' / 'dct_haar': base * (1 + detail_per_block * (2 if negation else 1)).
    'harp': base * 2 (L_3 + H_3 per block; expansion is decided at scoring time, not storage)."""
    proxy = getattr(cfg, "proxy_method", "dct")
    if proxy == "harp":
        return base_comp_size * 2
    detail_n = max(0, int(getattr(cfg, "haar_detail_per_block", 0)))
    if proxy in ("haar", "dct_haar") and detail_n > 0:
        mult = 2 if getattr(cfg, "haar_detail_with_negation", False) else 1
        return base_comp_size * (1 + detail_n * mult)
    return base_comp_size


def _get_or_build_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached projection matrix, building it on first call.
    Dispatches on cfg.proxy_method ('dct' | 'haar' | 'dct_haar'). The returned matrix has
    shape [_proxy_effective_comp_size(cfg, comp_size), page_size]."""
    cfg = _dct_page_cfg
    proxy = getattr(cfg, "proxy_method", "dct")
    detail_n = max(0, int(getattr(cfg, "haar_detail_per_block", 0)))
    detail_neg = bool(getattr(cfg, "haar_detail_with_negation", False))
    effective_rows = _proxy_effective_comp_size(cfg, comp_size)
    M = getattr(attn_module, '_dct_proj_matrix', None)
    cached_proxy = getattr(attn_module, '_dct_proj_method', None)
    cached_detail = getattr(attn_module, '_dct_proj_detail_n', None)
    cached_neg = getattr(attn_module, '_dct_proj_detail_neg', None)
    if (
        M is None
        or M.shape != (effective_rows, page_size)
        or M.device != device
        or cached_proxy != proxy
        or cached_detail != detail_n
        or cached_neg != detail_neg
    ):
        if proxy == "haar":
            M = _build_haar_projection_matrix(page_size, comp_size, detail_n, device, dtype,
                                              detail_with_negation=detail_neg)
        elif proxy == "haar_c2f":
            M = _build_haar_c2f_projection_matrix(page_size, comp_size, device, dtype)
        elif proxy == "dct_haar":
            M = _build_dct_haar_projection_matrix(page_size, comp_size, detail_n, device, dtype,
                                                   detail_with_negation=detail_neg)
        elif proxy == "harp":
            # HARP stores L_3 (block-mean) + H_3 (top-level wavelet per block) per page.
            # Adaptive expansion happens in scoring (`_score_pages_harp`) using H_3 L2 norms.
            M = _build_haar_projection_matrix(
                page_size, comp_size, n_detail_per_block=1,
                device=device, dtype=dtype, detail_with_negation=False,
            )
        else:  # "dct"
            M = _build_dct_projection_matrix(page_size, comp_size, device, dtype)
        attn_module._dct_proj_matrix = M
        attn_module._dct_proj_method = proxy
        attn_module._dct_proj_detail_n = detail_n
        attn_module._dct_proj_detail_neg = detail_neg
    return M


def _next_page_capacity(required_pages, current_capacity):
    """Grow page caches geometrically to avoid repeated realloc/copy."""
    if current_capacity >= required_pages:
        return current_capacity
    new_capacity = max(8, current_capacity or 0)
    while new_capacity < required_pages:
        new_capacity *= 2
    return new_capacity



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


def _get_rope_inv_freq_and_scaling(config, device):
    """Resolve inv_freq and attention_scaling from the model config.

    Handles both modern rope_parameters (transformers 5.x) and legacy
    rope_scaling.  Falls back to standard RoPE when rope_type == "default"
    (which is NOT in ROPE_INIT_FUNCTIONS).
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rope_type = "default"
    rope_config = None
    if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
        rope_config = config.rope_parameters
    elif hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        rope_config = config.rope_scaling
    if rope_config is not None:
        rope_type = rope_config.get("rope_type", rope_config.get("type", "default"))

    if rope_type == "default":
        # Standard RoPE — not in ROPE_INIT_FUNCTIONS, compute directly.
        rope_theta = None
        if rope_config is not None:
            rope_theta = rope_config.get("rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(config, "rope_theta", 10000.0)
        dim = getattr(config, "head_dim", None)
        if dim is None:
            dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        attention_scaling = 1.0
    else:
        inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device)

    return inv_freq, attention_scaling


def _compute_rope_cos_sin(positions, config, device, dtype):
    """Compute cos/sin for arbitrary positions, using the model's rope config.

    Supports all rope types: default (Qwen3), llama3 (Llama 3.1), yarn, etc.

    Args:
        positions: [seq_len] integer tensor
        config:    model config object

    Returns:
        cos, sin: each [1, 1, seq_len, head_dim]
    """
    inv_freq, attention_scaling = _get_rope_inv_freq_and_scaling(config, device)

    freqs = torch.outer(positions.float(), inv_freq)   # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)             # [seq_len, head_dim]
    cos = (emb.cos() * attention_scaling).unsqueeze(0).unsqueeze(0).to(dtype)
    sin = (emb.sin() * attention_scaling).unsqueeze(0).unsqueeze(0).to(dtype)
    return cos, sin


def _apply_decode_query_rope(attn_module, query_states, cos, sin, cfg):
    """Apply RoPE to a single-token decode query, using Triton when safe."""
    if (
        cfg.use_triton
        and query_states.shape[0] == 1
        and query_states.shape[2] == 1
        and cos.ndim == 3
        and sin.ndim == 3
        and cos.shape[0] == 1
        and sin.shape[0] == 1
        and cos.shape[1] == 1
        and sin.shape[1] == 1
    ):
        q_rope_buf = getattr(attn_module, "_q_rope_buf", None)
        if q_rope_buf is None or q_rope_buf.shape != query_states.shape:
            attn_module._q_rope_buf = torch.empty_like(query_states)
            q_rope_buf = attn_module._q_rope_buf
        return apply_rope_q_direct(query_states, cos[0, 0].contiguous(), sin[0, 0].contiguous(), q_rope_buf)
    query_states_rope, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
    return query_states_rope


def _get_or_build_original_position_rope_tables(attn_module, required_len, config, device, dtype):
    """Cache 2D RoPE tables for contiguous original token positions [0, required_len)."""
    cached_len = getattr(attn_module, "_orig_pos_rope_cache_len", 0)
    cached_cos = getattr(attn_module, "_orig_pos_rope_cos_2d", None)
    cached_sin = getattr(attn_module, "_orig_pos_rope_sin_2d", None)

    need_rebuild = (
        cached_cos is None
        or cached_sin is None
        or cached_len < required_len
        or cached_cos.device != device
        or cached_cos.dtype != dtype
    )
    if need_rebuild:
        cache_len = _next_page_capacity(required_len, cached_len)
        positions = torch.arange(cache_len, device=device)
        cos, sin = _compute_rope_cos_sin(positions, config, device, dtype)
        attn_module._orig_pos_rope_cos_2d = cos[0, 0]
        attn_module._orig_pos_rope_sin_2d = sin[0, 0]
        attn_module._orig_pos_rope_cache_len = cache_len

    return (
        attn_module._orig_pos_rope_cos_2d[:required_len],
        attn_module._orig_pos_rope_sin_2d[:required_len],
    )


def _compute_rope_cos_sin_for_position_ids(position_ids, config, device, dtype):
    """Compute cos/sin for arbitrary per-head position ids.

    Args:
        position_ids: integer tensor shaped [..., seq_len]

    Returns:
        cos, sin: tensors shaped [..., seq_len, head_dim]
    """
    inv_freq, attention_scaling = _get_rope_inv_freq_and_scaling(config, device)
    freqs = position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq.view(
        *([1] * position_ids.dim()), -1
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).to(dtype)
    sin = (emb.sin() * attention_scaling).to(dtype)
    return cos, sin


# ---------------------------------------------------------------------------
# KV segmentation without DCT (for incremental compression)
# ---------------------------------------------------------------------------
def segment_kv(key_states, value_states, cfg):
    """
    Divide KV cache into sink / pages / recent WITHOUT running DCT.

    Identical layout to build_pages_and_compress but skips compression.
    Used together with _update_comp_cache so DCT is only run once per page.

    Returns:
        sink_k, sink_v, paged_k, paged_v, recent_k, recent_v, num_pages, actual_recent
    """
    bsz, num_kv_heads, kv_len, head_dim = key_states.shape

    sink_tokens = cfg.num_sink_pages * cfg.page_size
    recent_tokens_min = cfg.num_recent_pages * cfg.page_size
    pageable_len = kv_len - sink_tokens - recent_tokens_min
    num_pages = pageable_len // cfg.page_size
    actual_recent = kv_len - sink_tokens - num_pages * cfg.page_size

    sink_k = key_states[:, :, :sink_tokens]
    sink_v = value_states[:, :, :sink_tokens]

    pages_end = sink_tokens + num_pages * cfg.page_size
    paged_k = key_states[:, :, sink_tokens:pages_end].view(
        bsz, num_kv_heads, num_pages, cfg.page_size, head_dim
    )
    paged_v = value_states[:, :, sink_tokens:pages_end].view(
        bsz, num_kv_heads, num_pages, cfg.page_size, head_dim
    )

    recent_k = key_states[:, :, pages_end:]
    recent_v = value_states[:, :, pages_end:]

    return sink_k, sink_v, paged_k, paged_v, recent_k, recent_v, num_pages, actual_recent


def paged_views_from_buf(buf_layer, num_sink_pages, num_pages, bsz=1, pages_per_batch=None):
    """Zero-copy view of the middle-page range of a FlashInfer paged buffer
    in the layout `_update_comp_cache` expects.

    `buf_layer` is `cache.buf[layer_idx]` with shape
        (capacity_pages, 2, page_size, num_kv_heads, head_dim).

    bsz=1 fast path: returns `(paged_k, paged_v)` each shaped
        (1, num_kv_heads, num_pages, page_size, head_dim)
    via stride-only permute+unsqueeze. `stride(-1) == 1` is preserved, so
    `_compress_pages` keeps its triton fast path.

    bsz>1: gathers the per-batch middle slice
    `buf_layer[b*pages_per_batch + num_sink_pages : b*pages_per_batch +
    num_sink_pages + num_pages]` for each b and stacks into
        (bsz, num_kv_heads, num_pages, page_size, head_dim).
    Each batch's middle-page block is contiguous in `buf_layer` so the per-batch
    slice is still a stride-only view; `torch.stack` of these views materializes
    a fresh contiguous tensor only on the leading bsz dim.
    """
    if bsz == 1:
        middle = buf_layer[num_sink_pages : num_sink_pages + num_pages]   # (num_pages, 2, ps, nkv, d)
        paged_k = middle[:, 0].permute(2, 0, 1, 3).unsqueeze(0)            # (1, nkv, num_pages, ps, d)
        paged_v = middle[:, 1].permute(2, 0, 1, 3).unsqueeze(0)
        return paged_k, paged_v
    assert pages_per_batch is not None, "pages_per_batch must be provided for bsz>1"
    paged_ks = []
    paged_vs = []
    for b in range(bsz):
        base = b * pages_per_batch
        middle = buf_layer[base + num_sink_pages : base + num_sink_pages + num_pages]
        paged_ks.append(middle[:, 0].permute(2, 0, 1, 3))   # (nkv, num_pages, ps, d)
        paged_vs.append(middle[:, 1].permute(2, 0, 1, 3))
    paged_k = torch.stack(paged_ks, dim=0)   # (bsz, nkv, num_pages, ps, d)
    paged_v = torch.stack(paged_vs, dim=0)
    return paged_k, paged_v


# ---------------------------------------------------------------------------
# Incremental compressed page cache
# ---------------------------------------------------------------------------
def _update_comp_cache(attn_module, paged_k, paged_v, num_pages, comp_size, cfg):
    # When Haar detail is enabled, the projection matrix returns more rows than
    # `comp_size`; the persistent cache must be sized to the effective row count.
    comp_size_eff = _proxy_effective_comp_size(cfg, comp_size)
    """
    Incrementally maintain compressed page representations using DCT-IDCT
    projection (via _compress_pages).

    K is always built (used for page scoring). V is built only when
    cfg.unselected_mode != "drop"; drop mode returns comp_v=None and skips
    V compression, quant, allocation, and storage. Flipping unselected_mode
    mid-run invalidates the whole cache.

    Storage: when cfg.comp_kv_quant != "none", the persistent cache holds
    low-precision quantized values (int8 / fp8_e4m3 / fp8_e5m2 / int4-packed)
    plus an fp32 scale tensor for dequant-on-read. Values returned to callers
    are always bf16, so downstream kernels are unchanged.

    RoPE handling for compressed K (cfg.compressed_token_rope):
      - "mixed":        compress post-RoPE keys directly.
      - "block_center": invert RoPE on raw page → compress → re-rotate at block-center positions.
                        Under real quantization the re-rotation applies on bf16 BEFORE quantize,
                        so the stored low-precision values are already in their final orientation.

    Values are unaffected by RoPE — always compressed as-is.
    """
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    # V is only consumed in compressed mode; skip compress/store entirely for drop.
    store_v = cfg.unselected_mode != "drop"

    cur_strategy = cfg.compressed_token_rope
    cur_quant = cfg.comp_kv_quant
    cur_quant_granularity = cfg.comp_kv_quant_granularity

    # Fast-path: cache hit — no new pages and all config keys match. On decode,
    # n_new == 0 for (page_size - 1)/page_size of steps, so this skips the slow
    # Python dispatch (getattrs, slice creation, dequant) on most steps.
    n_cached = getattr(attn_module, '_comp_n_pages_cached', 0)
    if (n_cached == num_pages and n_cached > 0
            and getattr(attn_module, '_comp_cache_strategy', None) == cur_strategy
            and getattr(attn_module, '_comp_cache_quant', None) == cur_quant
            and getattr(attn_module, '_comp_cache_quant_granularity', None) == cur_quant_granularity
            and getattr(attn_module, '_comp_cache_store_v', None) == store_v):
        last = getattr(attn_module, '_last_comp_kv', None)
        # Also verify shape (protects against page_size/compress_ratio/bsz changes
        # between decode loops that would otherwise be caught by the slow path).
        if (last is not None
                and last[0].shape[0] == bsz
                and last[0].shape[3] == comp_size_eff):
            return last

    cached_k = getattr(attn_module, '_comp_k_cache', None)
    cached_v = getattr(attn_module, '_comp_v_cache', None)
    capacity = getattr(attn_module, '_comp_cache_capacity', 0)
    cached_strategy = getattr(attn_module, '_comp_cache_strategy', None)
    cached_quant = getattr(attn_module, '_comp_cache_quant', None)
    cached_quant_granularity = getattr(attn_module, '_comp_cache_quant_granularity', None)
    cached_store_v = getattr(attn_module, '_comp_cache_store_v', None)

    # Invalidate cache when the sequence restarts, shape changes, RoPE strategy changes,
    # quant config changes (requires different storage dtype / scale shape), or the
    # unselected_mode switch flips whether V is stored.
    if (cached_k is None
            or (store_v and cached_v is None)
            or num_pages < n_cached
            or cached_k.shape[0] != bsz
            or cached_k.shape[3] != comp_size_eff
            or cached_strategy != cur_strategy
            or cached_quant != cur_quant
            or cached_quant_granularity != cur_quant_granularity
            or cached_store_v != store_v):
        attn_module._comp_k_cache = None
        attn_module._comp_v_cache = None
        attn_module._comp_k_scale_cache = None
        attn_module._comp_v_scale_cache = None
        attn_module._comp_n_pages_cached = 0
        attn_module._comp_cache_capacity = 0
        attn_module._last_comp_kv = None
        n_cached = 0
        capacity = 0
    attn_module._comp_cache_strategy = cur_strategy
    attn_module._comp_cache_quant = cur_quant
    attn_module._comp_cache_quant_granularity = cur_quant_granularity
    attn_module._comp_cache_store_v = store_v

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        new_v = paged_v[:, :, n_cached:num_pages] if store_v else None

        # Step A: optionally invert RoPE on new_k to recover raw (un-roped) keys.
        # When continuous_rope=False (current default), the cache stores post-RoPE keys.
        #
        # The forward RoPE applies (alpha * R_theta) to k, where alpha = attention_scaling
        # (alpha != 1 for YaRN and similar scaled RoPE types). To invert (alpha * R_theta * k)
        # and recover k, we apply (1/alpha) * R_{-theta}, which means dividing the rotation
        # matrix entries by alpha. Since `_compute_rope_cos_sin` already returns cos/sin
        # pre-multiplied by alpha, we must divide the returned values by alpha**2 (one alpha
        # to remove the forward scaling, another alpha to apply the 1/alpha inverse scaling).
        if cur_strategy == "block_center":
            sink_tokens = cfg.num_sink_pages * cfg.page_size
            start_pos = sink_tokens + n_cached * page_size
            end_pos = sink_tokens + num_pages * page_size
            positions = torch.arange(start_pos, end_pos, device=new_k.device)
            cos, sin = _compute_rope_cos_sin(
                positions, attn_module.config, new_k.device, new_k.dtype
            )
            _, attention_scaling = _get_rope_inv_freq_and_scaling(attn_module.config, new_k.device)
            inv_factor = 1.0 / (attention_scaling * attention_scaling)
            cos_inv = cos * inv_factor
            sin_inv = sin * inv_factor
            flat_k = new_k.reshape(bsz, num_kv_heads, n_new * page_size, head_dim)
            flat_raw_k = _apply_rope(flat_k, cos_inv, -sin_inv)
            new_k_for_compress = flat_raw_k.reshape(bsz, num_kv_heads, n_new, page_size, head_dim)
        else:
            new_k_for_compress = new_k

        # Step B: compress K (always — needed for scoring) and V (only when stored).
        new_comp_k = _compress_pages(attn_module, new_k_for_compress, comp_size)
        new_comp_v = _compress_pages(attn_module, new_v, comp_size) if store_v else None

        # Step C: re-apply RoPE to compressed K at block-center positions (still bf16).
        if cur_strategy == "block_center":
            assert comp_size_eff == comp_size, (
                "block_center RoPE strategy is incompatible with Haar detail expansion; "
                "use compressed_token_rope='mixed' when haar_detail_per_block > 0."
            )
            new_positions = _block_center_positions(
                n_cached, n_new, cfg.page_size, comp_size, cfg.num_sink_pages, new_comp_k.device,
            ).reshape(-1)
            cos_new, sin_new = _compute_rope_cos_sin(
                new_positions, attn_module.config, new_comp_k.device, new_comp_k.dtype
            )
            flat_comp_k = new_comp_k.reshape(bsz, num_kv_heads, n_new * comp_size, head_dim)
            flat_comp_k = _apply_rope(flat_comp_k, cos_new, sin_new)
            new_comp_k = flat_comp_k.reshape(bsz, num_kv_heads, n_new, comp_size, head_dim)

        # Step D: quantize for persistent storage (no-op when cur_quant == "none").
        new_v_store = new_v_scale = None
        if cur_quant == "none":
            new_k_store, new_k_scale = new_comp_k, None
            if store_v:
                new_v_store = new_comp_v
        else:
            new_k_store, new_k_scale = _quantize_for_storage(
                new_comp_k, cur_quant, cur_quant_granularity,
            )
            if store_v:
                new_v_store, new_v_scale = _quantize_for_storage(
                    new_comp_v, cur_quant, cur_quant_granularity,
                )

        # Allocate (or grow) the persistent cache + optional scale cache.
        if num_pages > capacity:
            new_capacity = _next_page_capacity(num_pages, capacity)
            storage_dtype, storage_d = _comp_cache_spec(cur_quant, head_dim)
            new_k_cache = torch.empty(
                bsz, num_kv_heads, new_capacity, comp_size_eff, storage_d,
                dtype=storage_dtype, device=paged_k.device,
            )
            if n_cached > 0 and attn_module._comp_k_cache is not None:
                new_k_cache[:, :, :n_cached].copy_(attn_module._comp_k_cache[:, :, :n_cached])
            attn_module._comp_k_cache = new_k_cache
            attn_module._comp_cache_capacity = new_capacity

            if store_v:
                new_v_cache = torch.empty_like(new_k_cache)
                if n_cached > 0 and attn_module._comp_v_cache is not None:
                    new_v_cache[:, :, :n_cached].copy_(attn_module._comp_v_cache[:, :, :n_cached])
                attn_module._comp_v_cache = new_v_cache
            else:
                attn_module._comp_v_cache = None

            if cur_quant != "none":
                scale_shape = _comp_scale_shape(
                    cur_quant_granularity, bsz, num_kv_heads, new_capacity, comp_size_eff,
                )
                new_k_scale_cache = torch.empty(
                    scale_shape, dtype=torch.float32, device=paged_k.device,
                )
                if n_cached > 0 and attn_module._comp_k_scale_cache is not None:
                    new_k_scale_cache[:, :, :n_cached].copy_(
                        attn_module._comp_k_scale_cache[:, :, :n_cached]
                    )
                attn_module._comp_k_scale_cache = new_k_scale_cache

                if store_v:
                    new_v_scale_cache = torch.empty_like(new_k_scale_cache)
                    if n_cached > 0 and attn_module._comp_v_scale_cache is not None:
                        new_v_scale_cache[:, :, :n_cached].copy_(
                            attn_module._comp_v_scale_cache[:, :, :n_cached]
                        )
                    attn_module._comp_v_scale_cache = new_v_scale_cache
                else:
                    attn_module._comp_v_scale_cache = None
            else:
                attn_module._comp_k_scale_cache = None
                attn_module._comp_v_scale_cache = None

        attn_module._comp_k_cache[:, :, n_cached:num_pages].copy_(new_k_store)
        if store_v:
            attn_module._comp_v_cache[:, :, n_cached:num_pages].copy_(new_v_store)
        if cur_quant != "none":
            attn_module._comp_k_scale_cache[:, :, n_cached:num_pages].copy_(new_k_scale)
            if store_v:
                attn_module._comp_v_scale_cache[:, :, n_cached:num_pages].copy_(new_v_scale)
        attn_module._comp_n_pages_cached = num_pages

    if attn_module._comp_k_cache is None:
        attn_module._last_comp_kv = None
        return None, None

    k_slice = attn_module._comp_k_cache[:, :, :num_pages]
    if cur_quant == "none":
        comp_k = k_slice
    else:
        comp_k = _dequantize_comp(
            k_slice,
            attn_module._comp_k_scale_cache[:, :, :num_pages],
            cur_quant, cur_quant_granularity, head_dim,
        )

    if not store_v or attn_module._comp_v_cache is None:
        result = (comp_k, None)
        attn_module._last_comp_kv = result
        return result

    v_slice = attn_module._comp_v_cache[:, :, :num_pages]
    if cur_quant == "none":
        comp_v = v_slice
    else:
        comp_v = _dequantize_comp(
            v_slice,
            attn_module._comp_v_scale_cache[:, :, :num_pages],
            cur_quant, cur_quant_granularity, head_dim,
        )
    result = (comp_k, comp_v)
    attn_module._last_comp_kv = result
    return result


_FP8_MAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0}
_FP8_DTYPE = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}


def _comp_cache_spec(quant_type: str, head_dim: int):
    """Persistent storage dtype + head_dim for the compressed KV cache.

    int4 packs two nibbles per uint8 byte, so the storage head_dim is halved.
    """
    if quant_type == "none":     return torch.bfloat16, head_dim
    if quant_type == "int8":     return torch.int8, head_dim
    if quant_type == "fp8_e4m3": return torch.float8_e4m3fn, head_dim
    if quant_type == "fp8_e5m2": return torch.float8_e5m2, head_dim
    if quant_type == "int4":
        assert head_dim % 2 == 0, f"int4 storage requires even head_dim, got {head_dim}"
        return torch.uint8, head_dim // 2
    raise ValueError(f"Unsupported comp_kv_quant: {quant_type}")


def _comp_scale_shape(granularity: str, bsz: int, num_kv_heads: int,
                      capacity: int, comp_size: int):
    """Fp32 scale cache shape, broadcastable against the comp K/V cache."""
    if granularity == "per_page":       return (bsz, num_kv_heads, capacity, 1, 1)
    if granularity == "per_comp_token": return (bsz, num_kv_heads, capacity, comp_size, 1)
    raise ValueError(f"Unsupported comp_kv_quant_granularity: {granularity}")


def _quant_reduce_dims(granularity: str):
    if granularity == "per_page":       return (-2, -1)
    if granularity == "per_comp_token": return (-1,)
    raise ValueError(f"Unsupported comp_kv_quant_granularity: {granularity}")


def _pack_int4(x_q_i8: torch.Tensor) -> torch.Tensor:
    """Pack signed-int4 values (int8 [-8, 7]) into uint8 [..., D//2] (2 nibbles per byte)."""
    assert x_q_i8.shape[-1] % 2 == 0, f"int4 packing requires even head_dim, got {x_q_i8.shape[-1]}"
    x_u = (x_q_i8.to(torch.int8) & 0x0F).to(torch.uint8)
    low = x_u[..., 0::2]
    high = x_u[..., 1::2]
    return (low | (high << 4)).contiguous()


def _unpack_int4(x_packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack uint8 [..., D//2] into signed int8 [..., D] with sign extension."""
    low = (x_packed & 0x0F).to(torch.int16)
    high = ((x_packed >> 4) & 0x0F).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)
    out = torch.empty(
        x_packed.shape[:-1] + (head_dim,), dtype=torch.int8, device=x_packed.device,
    )
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _quantize_for_storage(x: torch.Tensor, quant_type: str, granularity: str):
    """Quantize bf16 tensor for persistent low-precision storage.

    Returns (x_q, scale_fp32):
      x_q:   storage-dtype tensor. Shape matches x except int4 packs last dim to D//2.
      scale: fp32, shape broadcastable against x per granularity (per_page or per_comp_token).

    Callers must not pass quant_type="none" here (use the raw tensor directly).
    """
    if quant_type == "none":
        raise ValueError("_quantize_for_storage called with quant_type='none'")

    reduce_dims = _quant_reduce_dims(granularity)
    x_fp = x.to(torch.float32)
    abs_max = x_fp.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)

    if quant_type in ("fp8_e4m3", "fp8_e5m2"):
        fp8_max = _FP8_MAX[quant_type]
        scale = abs_max / fp8_max
        x_q = (x_fp / scale).to(_FP8_DTYPE[quant_type])
        return x_q, scale

    if quant_type == "int8":
        scale = abs_max / 127.0
        x_q = torch.round(x_fp / scale).clamp(-128.0, 127.0).to(torch.int8)
        return x_q, scale

    if quant_type == "int4":
        scale = abs_max / 7.0
        x_q_i8 = torch.round(x_fp / scale).clamp(-8.0, 7.0).to(torch.int8)
        return _pack_int4(x_q_i8), scale

    raise ValueError(f"Unsupported comp_kv_quant: {quant_type}")


def _dequantize_comp(x_q: torch.Tensor, scale: torch.Tensor,
                     quant_type: str, granularity: str, head_dim: int,
                     out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize stored quantized tensor back to out_dtype (default bf16).

    head_dim is only consulted for int4 (to know the unpacked last-dim size).
    """
    if quant_type == "none":
        return x_q.to(out_dtype) if x_q.dtype != out_dtype else x_q

    if quant_type in ("fp8_e4m3", "fp8_e5m2", "int8"):
        return (x_q.to(torch.float32) * scale).to(out_dtype)

    if quant_type == "int4":
        x_i8 = _unpack_int4(x_q, head_dim)
        return (x_i8.to(torch.float32) * scale).to(out_dtype)

    raise ValueError(f"Unsupported comp_kv_quant: {quant_type}")


def _update_quest_metadata(attn_module, paged_k, num_pages):
    """Incrementally maintain per-page per-channel min/max key metadata for QUEST scoring.

    Returns (min_k, max_k) each of shape [bsz, num_kv_heads, num_pages, head_dim].
    """
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    cached_min = getattr(attn_module, '_quest_min_k_cache', None)
    cached_max = getattr(attn_module, '_quest_max_k_cache', None)
    n_cached = getattr(attn_module, '_quest_n_pages_cached', 0)

    # Invalidate if sequence restarted
    if cached_min is None or num_pages < n_cached or cached_min.shape[0] != bsz:
        cached_min = None
        cached_max = None
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]  # [B, H, n_new, page_size, D]
        # Per-channel min/max across page_size tokens
        new_min = new_k.amin(dim=3)  # [B, H, n_new, D]
        new_max = new_k.amax(dim=3)  # [B, H, n_new, D]

        if cached_min is None:
            cached_min = new_min
            cached_max = new_max
        else:
            cached_min = torch.cat([cached_min, new_min], dim=2)
            cached_max = torch.cat([cached_max, new_max], dim=2)

    attn_module._quest_min_k_cache = cached_min
    attn_module._quest_max_k_cache = cached_max
    attn_module._quest_n_pages_cached = num_pages

    return cached_min, cached_max


def _score_pages_quest(query, min_k, max_k, group_agg_method, num_kv_groups, out=None, top_k=None):
    """QUEST-style page scoring: score = sum_d max(q_d * max_d, q_d * min_d).

    Args:
        query: [bsz, num_heads, 1, head_dim]
        min_k: [bsz, num_kv_heads, num_pages, head_dim]
        max_k: [bsz, num_kv_heads, num_pages, head_dim]
        group_agg_method: "mean" | "max" | "topp" | "per_head_union"
            - per_head_union: each query head in a GQA group picks its own
              top-(top_k // num_kv_groups) pages; per-kv-head selection is the
              union across the G heads. Returns scores where union pages have
              their max-over-G score and non-union pages have -inf (so a
              downstream top-K respects the union). Requires top_k arg.
        num_kv_groups: number of GQA groups (num_heads // num_kv_heads)
        out: optional pre-allocated [bsz, num_kv_heads, num_pages] buffer
        top_k: required when group_agg_method == "per_head_union"; size of the
            per-(kv_head) page budget the union must fit in.

    Returns:
        page_scores: [bsz, num_kv_heads, num_pages]
    """
    bsz, num_heads, _, head_dim = query.shape
    num_kv_heads = min_k.shape[1]
    num_pages = min_k.shape[2]

    # Reshape query for GQA: [bsz, num_kv_heads, G, head_dim]
    q = query.squeeze(2).float()
    q = q.reshape(bsz, num_kv_heads, num_kv_groups, head_dim)

    # QUEST scoring formula (Tang et al., MLSys 2024):
    #   score[p] = Σ_d max(q[d]*K_max[p,d], q[d]*K_min[p,d])
    # Per-channel: pick K_max if q[d]≥0 else K_min, then sum. Equivalent via:
    #   score = einsum(q⁺, K_max) + einsum(q⁻, K_min)
    # NOT max(Σ q·K_max, Σ q·K_min) — that's a different formula and was a bug.
    q_pos = q.clamp(min=0)
    q_neg = q.clamp(max=0)
    max_k_f = max_k.float()
    min_k_f = min_k.float()
    page_scores = (
        torch.einsum('bhgd,bhpd->bhgp', q_pos, max_k_f) +
        torch.einsum('bhgd,bhpd->bhgp', q_neg, min_k_f)
    )                                                             # [B, kv_heads, G, P]

    # Group aggregation (matches existing score_pages_triton logic)
    if group_agg_method == "mean":
        page_scores = page_scores.mean(dim=2)                     # [B, kv_heads, P]
    elif group_agg_method == "max":
        page_scores = page_scores.max(dim=2).values               # [B, kv_heads, P]
    elif group_agg_method == "topp":
        k_top = min(2, num_kv_groups)
        page_scores = page_scores.topk(k_top, dim=2).values.mean(dim=2)
    elif group_agg_method == "per_head_union":
        # Per-query-head selection: each of G heads picks top-(top_k/G) pages.
        # Per-kv-head selected set = UNION across G. Returned score masks non-union
        # pages with -inf so the downstream top-(top_k) call selects only union
        # members (and tiebreaks by max-over-heads score within the union).
        if top_k is None:
            raise ValueError("per_head_union requires top_k argument")
        K_per_head = max(1, top_k // num_kv_groups)
        per_head_topk_idx = page_scores.topk(K_per_head, dim=-1).indices       # [B, kv, G, K/G]
        flat_topk = per_head_topk_idx.reshape(bsz, num_kv_heads, num_kv_groups * K_per_head)
        union_mask = torch.zeros(bsz, num_kv_heads, num_pages,
                                 dtype=torch.bool, device=page_scores.device)
        union_mask.scatter_(-1, flat_topk, True)
        max_score = page_scores.max(dim=2).values                              # [B, kv, P]
        neg_inf = torch.full_like(max_score, float('-inf'))
        page_scores = torch.where(union_mask, max_score, neg_inf)
    else:
        raise ValueError(f"Unsupported group_agg_method: {group_agg_method}")

    if out is not None:
        out[:, :, :num_pages].copy_(page_scores)
        return out[:, :, :num_pages]
    return page_scores


def _score_pages_dct_perheadunion(query, comp_k, scoring_method,
                                  num_kv_groups, top_k, out=None):
    """DCT scoring with per_head_union group aggregation.

    Each qo-head in a GQA group picks its own top-(top_k // num_kv_groups) pages;
    per-kv-head selection = UNION across G heads. Returns scores where non-union
    pages are -inf so the downstream top-(top_k) call selects only union members
    (tiebreaks by max-over-heads score). Mirrors `_score_pages_quest`'s
    per_head_union branch.
    """
    bsz, num_q_heads, q_len, head_dim = query.shape
    _, num_kv_heads, num_pages, _comp_size, _ = comp_k.shape
    assert q_len == 1, "per_head_union DCT scoring is decode-only"
    if top_k is None:
        raise ValueError("per_head_union requires top_k argument")

    q = query.float().squeeze(2).view(bsz, num_kv_heads, num_kv_groups, head_dim)
    k = comp_k.float()
    scale = head_dim ** -0.5
    # [B, kv, G, P, C]
    group_token_scores = torch.einsum("bhgd,bhpcd->bhgpc", q, k) * scale
    if scoring_method == "max":
        group_page_scores = group_token_scores.amax(dim=-1)               # [B, kv, G, P]
    elif scoring_method == "mean":
        group_page_scores = group_token_scores.mean(dim=-1)
    elif scoring_method == "sum":
        group_page_scores = group_token_scores.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported scoring_method: {scoring_method!r}")

    K_per_head = max(1, top_k // num_kv_groups)
    per_head_topk_idx = group_page_scores.topk(K_per_head, dim=-1).indices  # [B, kv, G, K/G]
    flat_topk = per_head_topk_idx.reshape(bsz, num_kv_heads, num_kv_groups * K_per_head)
    union_mask = torch.zeros(
        bsz, num_kv_heads, num_pages,
        dtype=torch.bool, device=query.device,
    )
    union_mask.scatter_(-1, flat_topk, True)
    max_score = group_page_scores.amax(dim=2)                              # [B, kv, P]
    neg_inf = torch.full_like(max_score, float('-inf'))
    page_scores = torch.where(union_mask, max_score, neg_inf)

    if out is not None:
        out[:, :, :num_pages].copy_(page_scores)
        return out[:, :, :num_pages]
    return page_scores


_full_dct_matrix_cache: dict[tuple, torch.Tensor] = {}


# ---- Head-dim selection proxy basis caches (PCA D1/D2 and FASA-FC) --------
# Loaded by `load_proxy_basis()` once at startup; consumed in the scoring
# dispatch via cfg.proxy_method == "pca_qaware" / "fasa_fc".
_PROXY_BASIS_PCA: dict[int, torch.Tensor] = {}   # layer_idx -> [H_basis, cs_h_max, head_dim]
_PROXY_BASIS_FASA: dict[int, torch.Tensor] = {}  # layer_idx -> [H_q, n_tip_max]
_PROXY_BASIS_META: dict = {}                     # last-loaded metadata


def load_proxy_basis(path: str) -> None:
    """Load a calibrated proxy basis file produced by
    ``oracle/calibrate_proxy_bases.py``.

    Two formats supported, distinguished by which key is present:
      - PCA-style: {"M": {layer_idx: [H_basis, cs_h_max, d]}, "cs_h_max": int,
                    "granularity": "q_head" | "kv_head", ...}
      - FASA-style: {"idom": {layer_idx: [H_q, n_tip_max]}, "n_tip_max": int, ...}
    """
    d = torch.load(path, weights_only=False, map_location="cpu")
    global _PROXY_BASIS_PCA, _PROXY_BASIS_FASA, _PROXY_BASIS_META
    if "M" in d:
        _PROXY_BASIS_PCA = d["M"]
        _PROXY_BASIS_META = {k: v for k, v in d.items() if k != "M"}
    elif "idom" in d:
        _PROXY_BASIS_FASA = d["idom"]
        _PROXY_BASIS_META = {k: v for k, v in d.items() if k != "idom"}
    else:
        raise ValueError(f"Unrecognized basis file format: keys={list(d.keys())}")


def _score_pages_pca_qaware(query, paged_k, M_layer, cs_h, scoring_method,
                            group_agg_method, num_kv_groups, out=None):
    """Dense head-dim PCA scoring: project K via per-(layer, head) basis M of
    rank cs_h, project q via the same M, score = max_t (proj_q · proj_K).

    M_layer can be either:
      - [H_kv, cs_h_max, d]   — per-(layer, kv-head)
      - [H_q,  cs_h_max, d]   — per-(layer, q-head)
    Returns [bsz, num_kv_heads, num_pages] page scores."""
    bsz, H_q, q_len, d = query.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert q_len == 1 and H_q == H_kv * num_kv_groups
    H_basis = M_layer.shape[0]
    assert H_basis in (H_kv, H_q)
    cs_h = max(1, min(int(cs_h), M_layer.shape[1]))
    scale = d ** -0.5

    M = M_layer[:, :cs_h, :].to(paged_k.device).to(paged_k.dtype)

    if H_basis == H_kv:
        # Per-kv-head basis. comp_K shared across GQA group.
        comp_K = torch.einsum("bhpsd,hcd->bhpsc", paged_k.float(), M.float())  # [1, H_kv, P, S, cs_h]
        M_q = M.repeat_interleave(num_kv_groups, dim=0).float()                # [H_q, cs_h, d]
        proj_q = torch.einsum("bhqd,hcd->bhqc", query.float(), M_q)            # [1, H_q, 1, cs_h]
        comp_K_q = comp_K.repeat_interleave(num_kv_groups, dim=1)              # [1, H_q, P, S, cs_h]
    else:
        # Per-q-head basis. K stored per kv-head; expand via index_select.
        kv_idx = torch.arange(H_q, device=paged_k.device) // num_kv_groups
        K_q_view = paged_k.index_select(1, kv_idx)                             # [1, H_q, P, S, d]
        M_q = M.float()                                                        # [H_q, cs_h, d]
        comp_K_q = torch.einsum("bhpsd,hcd->bhpsc", K_q_view.float(), M_q)     # [1, H_q, P, S, cs_h]
        proj_q = torch.einsum("bhqd,hcd->bhqc", query.float(), M_q)            # [1, H_q, 1, cs_h]

    scores_per_token = torch.einsum("bhqc,bhpsc->bhps", proj_q, comp_K_q) * scale  # [1, H_q, P, S]

    if scoring_method == "max":
        score_q = scores_per_token.amax(dim=-1)
    elif scoring_method == "mean":
        score_q = scores_per_token.mean(dim=-1)
    elif scoring_method == "lse":
        score_q = torch.logsumexp(scores_per_token, dim=-1)
    else:
        raise ValueError(f"pca_qaware: unsupported scoring={scoring_method!r}")

    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        page_scores = score_g.amax(dim=2)
    elif group_agg_method == "mean":
        page_scores = score_g.mean(dim=2)
    else:
        raise ValueError(f"pca_qaware: unsupported group_agg={group_agg_method!r}")

    if out is not None:
        out[:, :, :P].copy_(page_scores)
        return out[:, :, :P]
    return page_scores


def _score_pages_fasa_fc(query, paged_k, idom_layer, n_tip, scoring_method,
                         group_agg_method, num_kv_groups, out=None):
    """FASA dominant-FC channel-subset scoring: gather n_tip RoPE-pair channel
    pairs per q-head, score = max_t (sum_{i ∈ I_dom[h]} q[2i:2i+2] · K[..., 2i:2i+2]).
    idom_layer: [H_q, n_tip_max] integer FC indices in [0, head_dim/2).
    Returns [bsz, num_kv_heads, num_pages] page scores."""
    bsz, H_q, q_len, d = query.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert q_len == 1 and H_q == H_kv * num_kv_groups
    nFC = d // 2
    assert idom_layer.shape[0] == H_q
    n_tip = max(1, min(int(n_tip), idom_layer.shape[1]))
    scale = d ** -0.5

    idom = idom_layer[:, :n_tip].to(paged_k.device).long()                    # [H_q, n_tip]
    channels = torch.stack([2 * idom, 2 * idom + 1], dim=-1).view(H_q, n_tip * 2)  # [H_q, 2*n_tip]

    # Gather q on selected channels.
    ch_q = channels.view(1, H_q, 1, n_tip * 2)
    q_sel = torch.gather(query, dim=-1, index=ch_q)                           # [1, H_q, 1, 2*n_tip]

    # Expand K to per-q-head view (via index_select on kv-head axis), gather channels.
    kv_idx = torch.arange(H_q, device=paged_k.device) // num_kv_groups
    K_q_view = paged_k.index_select(1, kv_idx)                                # [1, H_q, P, S, d]
    ch_K = channels.view(1, H_q, 1, 1, n_tip * 2).expand(bsz, H_q, P, S, n_tip * 2)
    K_sel = torch.gather(K_q_view, dim=-1, index=ch_K)                        # [1, H_q, P, S, 2*n_tip]

    scores_per_token = torch.einsum(
        "bhqc,bhpsc->bhps", q_sel.float(), K_sel.float(),
    ) * scale                                                                 # [1, H_q, P, S]

    if scoring_method == "max":
        score_q = scores_per_token.amax(dim=-1)
    elif scoring_method == "mean":
        score_q = scores_per_token.mean(dim=-1)
    elif scoring_method == "lse":
        score_q = torch.logsumexp(scores_per_token, dim=-1)
    else:
        raise ValueError(f"fasa_fc: unsupported scoring={scoring_method!r}")

    score_g = score_q.view(bsz, H_kv, num_kv_groups, P)
    if group_agg_method == "max":
        page_scores = score_g.amax(dim=2)
    elif group_agg_method == "mean":
        page_scores = score_g.mean(dim=2)
    else:
        raise ValueError(f"fasa_fc: unsupported group_agg={group_agg_method!r}")

    if out is not None:
        out[:, :, :P].copy_(page_scores)
        return out[:, :, :P]
    return page_scores


def _get_full_dct_matrix_prod(page_size: int, device, dtype):
    key = ("dct_full_prod", page_size, device, dtype)
    M = _full_dct_matrix_cache.get(key)
    if M is None:
        import numpy as np
        from scipy.fft import dct as _scipy_dct
        I_S = np.eye(page_size, dtype=np.float64)
        D = _scipy_dct(I_S, axis=0, norm="ortho")               # [S, S]
        M = torch.from_numpy(D).to(device=device, dtype=dtype).contiguous()
        _full_dct_matrix_cache[key] = M
    return M


def _score_pages_dct_qaware_lastq(attn_module, query, paged_k, top_bins,
                                  num_kv_groups, lastq_window=1, out=None):
    """Q-aware lastq adaptive bin selection (production).

    Bin selection frozen from Q_lastq (first-decode Q if window=1, else mean of
    last-`lastq_window` prefill queries cached on `attn_module._qaware_lastq_q`).
    At each decode step:
      1. Full DCT of paged_k along page-token axis → K_dct [B, kv, P, S, d]
      2. selection scores = |Q_lastq · K_dct| → top-N bins per (page, qo-head)
      3. masked signed bin scores from current Q, IDCT back to time domain
      4. score = max_t v_recon[t], group-aggregated via max

    Note: this uses paged_k (full uncompressed K of all paged pages), so the
    DCT cost is O(P·S^2·d) per step — heavier than lowpass cs=N scoring.
    Quality-only measurement for now.
    """
    bsz, num_q_heads, q_len, d = query.shape
    _, num_kv_heads, P, S, _ = paged_k.shape
    assert q_len == 1
    G = num_kv_groups
    scale = d ** -0.5

    D = _get_full_dct_matrix_prod(S, paged_k.device, paged_k.dtype).float()        # [S, S]
    K_dct = torch.einsum("ks,nhpsd->nhpkd", D, paged_k.float())                    # [B, kv, P, S, d]
    K_dct_q = K_dct.repeat_interleave(G, dim=1)                                    # [B, H_q, P, S, d]

    # Lastq Q for bin selection — cached on attn_module per (sample, layer).
    Q_sel = getattr(attn_module, "_qaware_lastq_q", None)
    if Q_sel is None:
        # First decode step: capture current Q (the user-requested lastq_window=1
        # falls back here; window>1 is captured at prefill end in a hook we don't
        # have in production — treated as window=1 here unless populated elsewhere).
        Q_sel = query.detach().clone()
        attn_module._qaware_lastq_q = Q_sel
    Q_sel = Q_sel.float()

    sel_scores = torch.einsum("nhqd,nhpkd->nhpk", Q_sel, K_dct_q) * scale          # [B, H_q, P, S_bins]
    N = max(1, min(int(top_bins), S))
    _, topN_idx = sel_scores.abs().topk(N, dim=-1)                                 # [B, H_q, P, N]

    Q = query.float()
    bin_scores = torch.einsum("nhqd,nhpkd->nhpk", Q, K_dct_q) * scale              # [B, H_q, P, S_bins]
    masked = torch.zeros_like(bin_scores)
    masked.scatter_(-1, topN_idx, bin_scores.gather(-1, topN_idx))
    v_recon = torch.einsum("kt,nhpk->nhpt", D, masked)                             # [B, H_q, P, S]
    score_q = v_recon.amax(dim=-1)                                                 # [B, H_q, P]
    score_g = score_q.view(bsz, num_kv_heads, G, P)
    page_scores = score_g.amax(dim=2)                                              # [B, kv, P]

    if out is not None:
        out[:, :, :P].copy_(page_scores)
        return out[:, :, :P]
    return page_scores


def _score_pages_dct_softmax(query, comp_k, scoring_method, group_agg_method,
                             num_kv_groups, out=None):
    """DCT scoring with per-qo-head softmax over pages BEFORE group aggregation.

    Matches `oracle/attention_mass_recall_ruler.compute_dct_lowpass_proxy_scores`
    with softmax_before_group=True. ShadowKV-style head-magnitude normalization.

    Args:
        query: [bsz, num_q_heads, 1, head_dim]
        comp_k: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
        scoring_method: "max" | "mean" | "sum" over comp axis.
        group_agg_method: "mean" | "max" over GQA group axis.
        num_kv_groups: H_q // H_kv.
        out: optional [bsz, num_kv_heads, capacity] float32 buffer.

    Returns:
        page_scores: [bsz, num_kv_heads, num_pages] float32.
    """
    bsz, num_q_heads, q_len, head_dim = query.shape
    _, num_kv_heads, num_pages, _comp_size, _ = comp_k.shape
    assert q_len == 1, "softmax DCT scoring is decode-only"
    scale = head_dim ** -0.5

    q = query.float().squeeze(2).view(bsz, num_kv_heads, num_kv_groups, head_dim)
    comp_k_f = comp_k.float()                                                 # [B, kv, P, C, d]
    # [B, kv, G, P, C] = einsum(q[B,kv,G,d], comp_k[B,kv,P,C,d])
    scores_per_comp = torch.einsum("bhgd,bhpcd->bhgpc", q, comp_k_f) * scale

    if scoring_method == "max":
        score_q = scores_per_comp.amax(dim=-1)
    elif scoring_method == "mean":
        score_q = scores_per_comp.mean(dim=-1)
    elif scoring_method == "sum":
        score_q = scores_per_comp.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported scoring_method: {scoring_method!r}")

    # Softmax over pages per qo-head before group aggregation.
    score_q = torch.softmax(score_q, dim=-1)                                  # [B, kv, G, P]

    if group_agg_method == "max":
        page_scores = score_q.amax(dim=2)
    elif group_agg_method == "mean":
        page_scores = score_q.mean(dim=2)
    else:
        raise ValueError(
            f"score_softmax + group_agg={group_agg_method!r} not supported"
        )

    if out is not None:
        out[:, :, :num_pages].copy_(page_scores)
        return out[:, :, :num_pages]
    return page_scores


def _detect_outliers_knorm(paged_k, paged_v, M):
    """Pick top-M outlier tokens per (batch, kv_head) by L2 norm of K.

    Args:
        paged_k: [bsz, kv_heads, num_pages, page_size, head_dim]
        paged_v: same shape
        M: number of outlier tokens per kv_head

    Returns:
        outlier_K, outlier_V: [bsz, kv_heads, M_eff, head_dim]
    """
    bsz, kv_heads, num_pages, page_size, head_dim = paged_k.shape
    flat_k = paged_k.reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    flat_v = paged_v.reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    knorm = flat_k.float().norm(dim=-1)  # [bsz, kv_heads, T]
    M_eff = min(M, knorm.shape[-1])
    top_idx = knorm.topk(M_eff, dim=-1).indices  # [bsz, kv_heads, M_eff]
    idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    outlier_K = flat_k.gather(2, idx_exp)
    outlier_V = flat_v.gather(2, idx_exp)
    return outlier_K.contiguous(), outlier_V.contiguous()


def _detect_outliers_lastq_mean(query_states, paged_k, paged_v, M, num_kv_groups):
    """Pick top-M tokens per (batch, kv_head) by group-mean(Q · K).

    Q is the first decode-step query (post-RoPE, post-q_norm), mean-aggregated
    across each GQA group; scoring against K is then per-kv-head dot product.

    Args:
        query_states: [bsz, num_q_heads, 1, head_dim]
        paged_k, paged_v: [bsz, kv_heads, num_pages, page_size, head_dim]
        M: outlier budget per kv_head
        num_kv_groups: num_q_heads // kv_heads
    """
    bsz, kv_heads, num_pages, page_size, head_dim = paged_k.shape
    flat_k = paged_k.reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    flat_v = paged_v.reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    Q = query_states.float()
    Q_g = Q.view(bsz, kv_heads, num_kv_groups, 1, head_dim).mean(dim=2).squeeze(2)  # [bsz, kv_heads, d]
    scores = torch.einsum("bhd, bhtd -> bht", Q_g, flat_k.float())                    # [bsz, kv_heads, T]
    M_eff = min(M, scores.shape[-1])
    top_idx = scores.topk(M_eff, dim=-1).indices                                      # [bsz, kv_heads, M_eff]
    idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    outlier_K = flat_k.gather(2, idx_exp)
    outlier_V = flat_v.gather(2, idx_exp)
    return outlier_K.contiguous(), outlier_V.contiguous()


def _kmeans_outlier(K: torch.Tensor, N: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-kv-head k-means (single-level, hard assignment).

    Args:
        K: [H_kv, T, d] candidate K vectors (float)
        N: cluster count
        iters: Lloyd iterations
    Returns:
        centroids: [H_kv, N, d]
        cluster_ids: [H_kv, T] long
    """
    H_kv, T, d = K.shape
    g = torch.Generator(device=K.device).manual_seed(seed)
    init_idx = torch.stack([
        torch.randperm(T, generator=g, device=K.device)[:N]
        for _ in range(H_kv)
    ])  # [H_kv, N]
    centroids = K.gather(1, init_idx.unsqueeze(-1).expand(-1, -1, d)).clone()
    cluster_ids = torch.zeros(H_kv, T, dtype=torch.long, device=K.device)
    # Cap peak memory of the [H_kv, T, N] dots tensor by chunking the token axis.
    # At T=128k, N=4096, fp32, the full tensor would be 16 GB and OOM.
    bytes_per_elem = K.element_size()
    max_dots_bytes = 2 * 1024**3
    T_chunk = max(1, min(T, max_dots_bytes // max(1, H_kv * N * bytes_per_elem)))
    for _ in range(iters):
        c_norm_sq = (centroids * centroids).sum(-1)                # [H_kv, N]
        for t0 in range(0, T, T_chunk):
            t1 = min(t0 + T_chunk, T)
            dots_c = torch.einsum("htd, hnd -> htn", K[:, t0:t1, :], centroids)
            d_sq_c = c_norm_sq.unsqueeze(1) - 2.0 * dots_c
            cluster_ids[:, t0:t1] = d_sq_c.argmin(dim=-1)
        sums = torch.zeros(H_kv, N, d, device=K.device, dtype=K.dtype)
        sums.scatter_add_(1, cluster_ids.unsqueeze(-1).expand(-1, -1, d), K)
        counts = torch.zeros(H_kv, N, device=K.device, dtype=K.dtype)
        counts.scatter_add_(1, cluster_ids, torch.ones_like(cluster_ids, dtype=K.dtype))
        keep = (counts > 0).unsqueeze(-1)
        centroids = torch.where(keep, sums / counts.unsqueeze(-1).clamp(min=1), centroids)
    return centroids, cluster_ids


def _build_cluster_state(paged_k, N, iters, seed, scoring: str = "centroid"):
    """Run k-means on the flattened pageable region (one-shot, post-prefill).
    Returns (centroids, cluster_ids, T_init, K_max, K_min) — K_max/K_min are None
    when scoring='centroid', and per-(kv_h, cluster, dim) min/max when 'minmax'.
    paged_k may grow during decode; selection at later steps must clip to T_init."""
    bsz, kv_heads, num_pages, page_size, head_dim = paged_k.shape
    assert bsz == 1, "cluster_dyn outlier currently supports bsz=1 only"
    T_init = num_pages * page_size
    flat_k = paged_k.reshape(bsz, kv_heads, T_init, head_dim).squeeze(0).float()
    centroids, cluster_ids = _kmeans_outlier(flat_k, N, iters, seed)

    K_max = K_min = None
    if scoring == "minmax":
        idx = cluster_ids.unsqueeze(-1).expand(-1, -1, head_dim)
        K_max = torch.full(
            (kv_heads, N, head_dim), float("-inf"),
            device=flat_k.device, dtype=flat_k.dtype,
        )
        K_max.scatter_reduce_(1, idx, flat_k, reduce="amax", include_self=False)
        K_min = torch.full(
            (kv_heads, N, head_dim), float("inf"),
            device=flat_k.device, dtype=flat_k.dtype,
        )
        K_min.scatter_reduce_(1, idx, flat_k, reduce="amin", include_self=False)
        # Empty clusters: leave K_max=-inf, K_min=inf — scoring will return -inf for them.
        # Replace inf/-inf with 0 so the sum doesn't NaN out.
        empty_mask = ~K_max.isfinite()  # both K_max=-inf and K_min=+inf for empty clusters
        K_max = torch.where(empty_mask, torch.zeros_like(K_max), K_max)
        K_min = torch.where(empty_mask, torch.zeros_like(K_min), K_min)
    return centroids, cluster_ids, T_init, K_max, K_min


def _select_cluster_outliers_dynamic(
    query_states, paged_k, paged_v,
    centroids, cluster_ids, T_init,
    M, K_top, num_kv_groups,
    q_agg: str = "mean",
    scoring: str = "centroid",
    K_max=None, K_min=None,
):
    """Per-step outlier selection: top-K clusters by Q·centroid, refine within by Q·K, take top-M.

    Args:
        query_states: [bsz, num_q_heads, 1, head_dim]
        paged_k, paged_v: [bsz, kv_heads, num_pages, page_size, head_dim] (may grow)
        centroids: [kv_heads, N, head_dim] (float)
        cluster_ids: [kv_heads, T_init] long
        T_init: token count used at cluster build time; later positions are ignored.
        M: outlier budget per kv_head
        K_top: # of top clusters
        num_kv_groups: GQA group factor
        q_agg: "mean" | "max" — aggregate scores across qo-heads in each GQA group.
            "mean" averages Q first (1 dot product per kv_head).
            "max" computes per-qo-head dot products, max-reduces over the group
            (G× more compute on cluster scoring AND refinement).
    """
    bsz, kv_heads, num_pages, page_size, head_dim = paged_k.shape
    flat_k_all = paged_k.reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    flat_v_all = paged_v.reshape(bsz, kv_heads, num_pages * page_size, head_dim)
    flat_k = flat_k_all[:, :, :T_init, :]
    flat_v = flat_v_all[:, :, :T_init, :]
    Q_grouped = query_states.view(bsz, kv_heads, num_kv_groups, head_dim).squeeze(0).float()  # [kv_h, G, d]
    if scoring == "minmax":
        assert K_max is not None and K_min is not None, "minmax scoring requires K_max/K_min"
        # Quest-style upper bound: Σ_d max(q_d · K_max[c,d], q_d · K_min[c,d])
        # Decomposes into q_pos · K_max + q_neg · K_min (since q>0 picks K_max, q<0 picks K_min).
        if q_agg == "max":
            q_pos = Q_grouped.clamp(min=0)  # [kv_h, G, d]
            q_neg = Q_grouped.clamp(max=0)
            scores_hg = (
                torch.einsum("hgd, hnd -> hgn", q_pos, K_max)
                + torch.einsum("hgd, hnd -> hgn", q_neg, K_min)
            )
            cluster_scores = scores_hg.amax(dim=1)
        else:
            Q_g = Q_grouped.mean(dim=1)
            q_pos = Q_g.clamp(min=0)
            q_neg = Q_g.clamp(max=0)
            cluster_scores = (
                torch.einsum("hd, hnd -> hn", q_pos, K_max)
                + torch.einsum("hd, hnd -> hn", q_neg, K_min)
            )
    elif q_agg == "max":
        # Per-qo-head scoring, max over GQA group.
        cluster_scores_hg = torch.einsum("hgd, hnd -> hgn", Q_grouped, centroids)             # [kv_h, G, N]
        cluster_scores = cluster_scores_hg.amax(dim=1)                                         # [kv_h, N]
    else:
        Q_g = Q_grouped.mean(dim=1)                                                            # [kv_h, d]
        cluster_scores = torch.einsum("hd, hnd -> hn", Q_g, centroids)                         # [kv_h, N]
    top_c = cluster_scores.topk(min(K_top, cluster_scores.shape[-1]), dim=-1).indices          # [kv_h, K_top]
    member_mask = (cluster_ids.unsqueeze(-1) == top_c.unsqueeze(1)).any(dim=-1)                # [kv_h, T_init]
    if q_agg == "max":
        qk_hg = torch.einsum("hgd, htd -> hgt", Q_grouped, flat_k.squeeze(0).float())          # [kv_h, G, T]
        qk = qk_hg.amax(dim=1)                                                                 # [kv_h, T]
    else:
        Q_g = Q_grouped.mean(dim=1)
        qk = torch.einsum("hd, htd -> ht", Q_g, flat_k.squeeze(0).float())                     # [kv_h, T]
    qk = qk.masked_fill(~member_mask, float("-inf"))
    M_eff = min(M, qk.shape[-1])
    top_idx = qk.topk(M_eff, dim=-1).indices.unsqueeze(0)                                      # [1, kv_h, M_eff]
    idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    outlier_K = flat_k.gather(2, idx_exp)
    outlier_V = flat_v.gather(2, idx_exp)
    return outlier_K.contiguous(), outlier_V.contiguous()


def _score_pages_harp(query, comp_k, num_kv_groups, scoring_method, group_agg_method,
                      harp_detail_topk, out=None):
    """HARP scoring: L_3 (block-mean) + H_3 (top-level wavelet) with adaptive expansion.

    comp_k layout: [bsz, kv_heads, num_pages, 2*comp_size_lp, head_dim]
      slots 0..cs_lp-1   = L_3 (block-mean rows, all positive)
      slots cs_lp..2cs_lp-1 = H_3 (top-level wavelet rows, signed +/-)

    Per page, the top-`harp_detail_topk` blocks (by per-block |H_3| L2 norm) are
    'expanded' — their score adds |Q·H_3| to Q·L_3 (= max of L_2 half-block scores).
    The remaining blocks score with Q·L_3 only (no expansion).

    Returns: [bsz, kv_heads, num_pages] page scores (float).
    """
    bsz, _num_q_heads, q_len, d = query.shape
    _, kv_heads, num_pages, total_cs, _ = comp_k.shape
    assert q_len == 1, "HARP scoring is decode-only"
    assert total_cs % 2 == 0, f"HARP expects even comp_size_total, got {total_cs}"
    cs_lp = total_cs // 2

    L_3 = comp_k[..., :cs_lp, :].float()
    H_3 = comp_k[..., cs_lp:, :].float()

    h3_norm = H_3.norm(dim=-1)  # [B, kv_h, P, cs_lp]

    k_d = max(0, min(int(harp_detail_topk), cs_lp))
    if k_d == 0:
        expanded_mask = torch.zeros_like(h3_norm, dtype=torch.bool)
    elif k_d >= cs_lp:
        expanded_mask = torch.ones_like(h3_norm, dtype=torch.bool)
    else:
        threshold = h3_norm.topk(k_d, dim=-1).values[..., -1:]
        expanded_mask = h3_norm >= threshold

    Q = query.float()  # [B, H_q, 1, d]
    Q_grouped = Q.view(bsz, kv_heads, num_kv_groups, d)

    if group_agg_method == "mean":
        Q_g = Q_grouped.mean(dim=2)  # [B, kv_h, d]
        QL3 = torch.einsum("bhd, bhpcd -> bhpc", Q_g, L_3)
        QH3 = torch.einsum("bhd, bhpcd -> bhpc", Q_g, H_3)
        block_scores = QL3 + expanded_mask.float() * QH3.abs()
        page_scores = block_scores.amax(dim=-1)
    else:  # "max"
        QL3 = torch.einsum("bhgd, bhpcd -> bhgpc", Q_grouped, L_3)
        QH3 = torch.einsum("bhgd, bhpcd -> bhgpc", Q_grouped, H_3)
        block_scores = QL3 + expanded_mask.unsqueeze(2).float() * QH3.abs()
        page_scores = block_scores.amax(dim=-1).amax(dim=2)

    # Scoring across blocks within page: HARP uses max (the formula already does amax(-1)).
    # 'mean'/'sum' alternatives would require different per-block aggregation; not supported here.
    if out is not None:
        out[..., :num_pages].copy_(page_scores.to(out.dtype))
        return out[..., :num_pages]
    return page_scores.to(comp_k.dtype if comp_k.is_floating_point() else torch.float32)


def _compress_pages(attn_module, paged_x, comp_size):
    """Project [bsz, kv_heads, num_pages, page_size, head_dim] pages to comp_size via DCT."""
    page_size = paged_x.shape[3]
    M = _get_or_build_projection_matrix(
        attn_module, page_size, comp_size, paged_x.device, paged_x.dtype
    )
    # Triton compress kernel uses tl.arange(0, COMP_SIZE) which requires power-of-2.
    # Haar detail with ± negation pairs can produce non-power-of-2 effective comp_size;
    # fall back to einsum in that case.
    comp_rows = M.shape[0]
    is_pow2 = (comp_rows > 0) and ((comp_rows & (comp_rows - 1)) == 0)
    if _dct_page_cfg.use_triton and paged_x.stride(-1) == 1 and is_pow2:
        from triton_kernels import compress_pages_triton
        return compress_pages_triton(paged_x, M)
    return torch.einsum('cs,bhnsd->bhncd', M, paged_x)


def _block_center_positions(start_page_idx, n_pages, page_size, comp_size, num_sink_pages, device):
    """Compute the center position of each compressed block within a page.

    For comp_size=4 and page_size=128, each page is split into 4 blocks of 32.
    The anchor position for each block is its center token's original position.
    Returns: [n_pages, comp_size] integer tensor of absolute positions.
    """
    page_ids = torch.arange(start_page_idx, start_page_idx + n_pages, device=device, dtype=torch.long)
    page_bases = num_sink_pages * page_size + page_ids[:, None] * page_size
    starts = torch.tensor(
        [(idx * page_size) // comp_size for idx in range(comp_size)],
        device=device, dtype=torch.long,
    )
    ends = torch.tensor(
        [max(((idx + 1) * page_size) // comp_size, (idx * page_size) // comp_size + 1) for idx in range(comp_size)],
        device=device, dtype=torch.long,
    )
    proxy_offsets = ((starts + ends - 1) // 2).clamp_max(page_size - 1)
    return page_bases + proxy_offsets[None, :]


def _apply_original_position_rope_to_paged_k(paged_k, num_sink_pages, page_size_cfg, model_config):
    """Apply original-position RoPE to full paged keys for debug/oracle scoring."""
    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    flat_k = paged_k.reshape(bsz, num_kv_heads, num_pages * page_size, head_dim)
    sink_tokens = num_sink_pages * page_size_cfg
    positions = torch.arange(sink_tokens + num_pages * page_size, device=paged_k.device)
    cos_pages, sin_pages = _compute_rope_cos_sin(
        positions, model_config, paged_k.device, paged_k.dtype
    )
    cos_pages = cos_pages[:, :, sink_tokens:]
    sin_pages = sin_pages[:, :, sink_tokens:]
    flat_k_rope = _apply_rope(flat_k, cos_pages, sin_pages)
    return flat_k_rope.reshape(bsz, num_kv_heads, num_pages, page_size, head_dim)


def _compute_debug_oracle_page_scores(attn_module, query_states, paged_k, cfg, cos, sin):
    """Compute full-page oracle scores for debug comparisons against proxies.

    Two modes (selected by cfg.oracle_score_mode):
      - "max":  oracle_max — score_per_page = max(q·K) over the page's tokens,
                then group-aggregated. Same selection as cs=page_size proxy.
      - "mass": mass oracle — score_per_page = Σ_t exp(q·K[p, t]) (proportional
                to the page's softmax mass since the denominator is shared),
                then group-aggregated.
    """
    oracle_query_states = query_states
    oracle_paged_k = paged_k
    if cfg.continuous_rope:
        oracle_query_states, _ = apply_rotary_pos_emb(
            query_states, query_states, cos, sin
        )
        oracle_paged_k = _apply_original_position_rope_to_paged_k(
            paged_k, cfg.num_sink_pages, cfg.page_size, attn_module.config
        )
    if cfg.oracle_score_mode == "mass":
        return _compute_mass_oracle_page_scores(
            oracle_query_states, oracle_paged_k,
            attn_module.num_key_value_groups,
            cfg.group_agg_method,
        )
    return score_pages_triton(
        oracle_query_states,
        oracle_paged_k,
        cfg.scoring_method,
        cfg.group_agg_method,
        attn_module.num_key_value_groups,
    )


def _compute_mass_oracle_page_scores(query_states, paged_k, num_kv_groups,
                                     group_agg_method):
    """Per-page softmax mass over paged region, group-aggregated.

    score[b, kv_h, p] = group_agg_g( Σ_t exp(q[b,h,t']·K[b,kv_h,p,t]) / Z )
    where Z normalizes over all paged tokens. Since Z is shared across pages
    within a qo-head, ranking by this mass = ranking by unnormalized
    Σ_t exp(q·K[p,t]) — sink/recent regions are not needed for ordering.
    """
    bsz, H_q, q_len, d = query_states.shape
    _, H_kv, P, S, _ = paged_k.shape
    assert q_len == 1, "mass oracle is decode-only"
    G = num_kv_groups
    scale = d ** -0.5

    q = query_states.float()                                            # [B, H_q, 1, d]
    k_expanded = paged_k.float().repeat_interleave(G, dim=1)            # [B, H_q, P, S, d]
    flat_k = k_expanded.reshape(bsz, H_q, P * S, d)
    logits = torch.matmul(q, flat_k.transpose(-1, -2)).squeeze(2) * scale  # [B, H_q, P*S]
    weights = torch.softmax(logits, dim=-1)                             # [B, H_q, P*S]
    page_mass = weights.view(bsz, H_q, P, S).sum(-1)                    # [B, H_q, P]
    page_mass_g = page_mass.view(bsz, H_kv, G, P)
    if group_agg_method == "max":
        return page_mass_g.amax(dim=2)
    elif group_agg_method == "mean":
        return page_mass_g.mean(dim=2)
    elif group_agg_method == "sum":
        return page_mass_g.sum(dim=2)
    else:
        raise ValueError(f"mass oracle: unsupported group_agg={group_agg_method!r}")


def _apply_original_position_rope_to_final_k(
    attn_module,
    final_k,
    selected_indices,
    num_pages,
    actual_recent,
    cfg,
    model_config,
):
    """Apply RoPE to assembled drop-mode KV using the tokens' original positions."""
    bsz, num_kv_heads, _, head_dim = final_k.shape
    actual_top_k = selected_indices.shape[2]
    selected_indices_long = selected_indices.to(torch.long)
    sink_tokens = cfg.num_sink_pages * cfg.page_size
    cos_table, sin_table = _get_or_build_original_position_rope_tables(
        attn_module,
        num_pages * cfg.page_size + sink_tokens + actual_recent,
        model_config,
        final_k.device,
        final_k.dtype,
    )

    cos_parts = []
    sin_parts = []

    if sink_tokens > 0:
        sink_cos = cos_table[:sink_tokens].view(1, 1, sink_tokens, head_dim)
        sink_sin = sin_table[:sink_tokens].view(1, 1, sink_tokens, head_dim)
        cos_parts.append(sink_cos.expand(bsz, num_kv_heads, -1, -1))
        sin_parts.append(sink_sin.expand(bsz, num_kv_heads, -1, -1))

    middle_start = sink_tokens
    middle_end = middle_start + num_pages * cfg.page_size
    if actual_top_k > 0:
        page_cos_table = cos_table[middle_start:middle_end].view(num_pages, cfg.page_size, head_dim)
        page_sin_table = sin_table[middle_start:middle_end].view(num_pages, cfg.page_size, head_dim)
        selected_cos = page_cos_table[selected_indices_long].reshape(
            bsz, num_kv_heads, actual_top_k * cfg.page_size, head_dim
        )
        selected_sin = page_sin_table[selected_indices_long].reshape(
            bsz, num_kv_heads, actual_top_k * cfg.page_size, head_dim
        )
        cos_parts.append(selected_cos)
        sin_parts.append(selected_sin)

    if actual_recent > 0:
        recent_start = middle_end
        recent_cos = cos_table[recent_start:recent_start + actual_recent].view(
            1, 1, actual_recent, head_dim
        )
        recent_sin = sin_table[recent_start:recent_start + actual_recent].view(
            1, 1, actual_recent, head_dim
        )
        cos_parts.append(recent_cos.expand(bsz, num_kv_heads, -1, -1))
        sin_parts.append(recent_sin.expand(bsz, num_kv_heads, -1, -1))

    if len(cos_parts) == 1:
        cos = cos_parts[0]
        sin = sin_parts[0]
    else:
        cos = torch.cat(cos_parts, dim=2)
        sin = torch.cat(sin_parts, dim=2)
    return _apply_rope(final_k, cos, sin)



_DCT_RUNTIME_STATE_ATTRS = (
    "_comp_k_cache",
    "_comp_v_cache",
    "_comp_k_scale_cache",
    "_comp_v_scale_cache",
    "_comp_n_pages_cached",
    "_comp_cache_capacity",
    "_comp_cache_strategy",
    "_comp_cache_quant",
    "_comp_cache_quant_granularity",
    "_comp_cache_store_v",
    "_last_comp_kv",
    "_page_scores_buf",
    "_page_scores_np",
    "_page_scores_capacity",
    "_topk_out_buf",
    "_topk_scratch_buf",
    "_assemble_stride_cache",
    "_final_k_buf",
    "_final_v_buf",
    "_final_bias_buf",
    "_sel_idx_buf",
    "_assemble_buf_len",
    "_orig_pos_rope_cos_2d",
    "_orig_pos_rope_sin_2d",
    "_orig_pos_rope_cache_len",
    "_q_rope_buf",
    # Quest min/max metadata (used by score_use_quest_minmax / score_combine_quest_dct).
    # Must be reset between samples or stale min/max from prior sample's K leaks in.
    "_quest_min_k_cache",
    "_quest_max_k_cache",
    "_quest_n_pages_cached",
    # Outlier bank: detected once post-prefill, then always included in decode attention.
    "_outlier_K",
    "_outlier_V",
    "_outlier_indices",
    # Cluster outlier (cluster_dyn): centroids + cluster_ids built post-prefill, reused per step.
    "_cluster_centroids",
    "_cluster_ids",
    "_cluster_T_init",
    # Q-aware lastq adaptive bin selection.
    "_qaware_lastq_q",
    "_cluster_K_max",
    "_cluster_K_min",
)


def _maybe_reset_dct_runtime_state(attn_module, past_key_values):
    """Clear per-generation runtime caches when the HF cache object changes."""
    cached_ref = getattr(attn_module, "_dct_runtime_cache_ref", None)
    if cached_ref is past_key_values:
        return

    for attr in _DCT_RUNTIME_STATE_ATTRS:
        if hasattr(attn_module, attr):
            delattr(attn_module, attr)
    attn_module._dct_runtime_cache_ref = past_key_values


# ---------------------------------------------------------------------------
# Main attention forward
# ---------------------------------------------------------------------------
def dct_page_attention_forward(
    self, # the Qwen2Attention/LlamaAttention instance (we access its projections like self.q_proj, config like self.config, etc.)
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None, # The type can be torch.Tensor or None, and the default value is None
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """
    Replacement for Qwen2Attention.forward or LlamaAttention.forward

    - Prefill (q_len > 1): standard full causal attention.
    - Decode (q_len == 1, long KV cache): DCT page attention.
    """
    cfg = _dct_page_cfg
    if cfg.continuous_rope:
        raise NotImplementedError(
            "continuous_rope=True is temporarily disabled. "
            "Use continuous_rope=False (default) instead."
        )
    input_shape = hidden_states.shape[:-1] # (bsz, q_len)
    hidden_shape = (*input_shape, -1, self.head_dim) # (bsz, q_len, num_heads, head_dim)
    bsz, q_len = input_shape     
    _maybe_reset_dct_runtime_state(self, past_key_values)
    min_len_for_paging = max(
        (cfg.num_sink_pages + cfg.top_k + 1 + cfg.num_recent_pages) * cfg.page_size,
        getattr(cfg, "min_decode_kv_len_for_paging", 0),
    )

    # Qwen3 uses QK-norm (RMSNorm on q/k after projection, before RoPE).
    # Qwen2 and Llama do not have q_norm/k_norm, so we check for their presence.
    _has_qk_norm = hasattr(self, "q_norm") and hasattr(self, "k_norm")

    if q_len>1:
        # Step 1: Q/K/V projection
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        if _has_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2) # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
        
        # Step 2 & 3: RoPE and KV cache
        cos, sin = position_embeddings
        query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_q, attn_k, attn_v = query_rope, key_rope, value_states
        if past_key_values is not None: # unless we call the model directly with use_cache=False, past_key_values is not None.
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            attn_k, attn_v = past_key_values.update(
                key_rope, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = _get_attention_interface(self)
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

        # Reset compressed page cache from prior generate() call
        # TODO: Verify if this is needed or not
        self._comp_k_cache = None
        self._comp_v_cache = None
        self._comp_k_scale_cache = None
        self._comp_v_scale_cache = None
        self._comp_n_pages_cached = 0
        self._last_comp_kv = None

        extra_tokens = cfg.page_size * 2

        # Convert DynamicCache → PreAllocatedLayer at end of prefill (last layer only).
        # All layers are converted at once, so by the first decode step every
        # layer's cache.update() already uses PreAllocatedLayer (fixed strides).
        if (past_key_values is not None
                and self.layer_idx == self.config.num_hidden_layers - 1
                and not getattr(past_key_values, '_preallocated', False)):
            pre_allocate_cache(past_key_values, extra_tokens=extra_tokens)
            past_key_values._preallocated = True

        return attn_output, attn_weights

    # ---- DECODE PATH (q_len == 1, long KV cache) ----
    # Step 1: QKV projection (with QK-norm for Qwen3)
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if _has_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # Step 2: RoPE + KV cache update (post-RoPE stored in cache)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    kv_len = key_states.shape[2]

    # Fallback to standard attention when KV cache is too short for paging
    # OR when this layer is in the dense-first-N-layers range (Quest-style skip).
    dense_layer_skip = (
        getattr(cfg, "dense_first_n_layers", 0) > 0
        and self.layer_idx < cfg.dense_first_n_layers
    )
    if kv_len < min_len_for_paging or dense_layer_skip:
        attention_interface = _get_attention_interface(self)
        attn_output, _ = attention_interface(
            self,
            query_states, key_states, value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    # Step 3: Segment KV cache and update the incremental compressed page cache.
    # DCT is computed only for pages that are newly finalized since the last
    # decode step; all previously cached compressed representations are reused.
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    (sink_k, sink_v, paged_k, paged_v,
        recent_k, recent_v, num_pages, actual_recent) = segment_kv(
        key_states, value_states, cfg
    )

    # Outlier bank: detect once on the first decode step (per generation/sample)
    # using the post-prefill pageable region only (sink/recent are already always
    # attended). Outliers are concatenated to the gathered KV before SDPA below.
    if cfg.outlier_budget > 0:
        if cfg.outlier_detector == "knorm":
            if getattr(self, "_outlier_K", None) is None:
                self._outlier_K, self._outlier_V = _detect_outliers_knorm(
                    paged_k, paged_v, cfg.outlier_budget
                )
        elif cfg.outlier_detector == "lastq_mean":
            if getattr(self, "_outlier_K", None) is None:
                self._outlier_K, self._outlier_V = _detect_outliers_lastq_mean(
                    query_states, paged_k, paged_v, cfg.outlier_budget, self.num_key_value_groups,
                )
        elif cfg.outlier_detector == "cluster_dyn":
            # Build cluster state once post-prefill, then refresh outlier K/V per step.
            if getattr(self, "_cluster_centroids", None) is None:
                (self._cluster_centroids, self._cluster_ids, self._cluster_T_init,
                 self._cluster_K_max, self._cluster_K_min) = _build_cluster_state(
                    paged_k, cfg.cluster_outlier_N, cfg.cluster_outlier_iters,
                    seed=self.layer_idx + 12345, scoring=cfg.cluster_outlier_scoring,
                )
            self._outlier_K, self._outlier_V = _select_cluster_outliers_dynamic(
                query_states, paged_k, paged_v,
                self._cluster_centroids, self._cluster_ids, self._cluster_T_init,
                cfg.outlier_budget, cfg.cluster_outlier_top_k, self.num_key_value_groups,
                q_agg=cfg.cluster_outlier_q_agg,
                scoring=cfg.cluster_outlier_scoring,
                K_max=self._cluster_K_max, K_min=self._cluster_K_min,
            )
        else:
            raise ValueError(
                f"Unsupported outlier_detector: {cfg.outlier_detector!r}"
            )
    # Step 4: Compressed cache maintenance.
    # comp_k is always built (used for scoring). comp_v is built only in
    # compressed mode (used for assembly); drop mode returns comp_v=None.
    comp_k, comp_v = _update_comp_cache(
        self, paged_k, paged_v, num_pages, comp_size, cfg,
    )

    # Step 5: Score pages (Triton kernel 1 — returns page_scores only)
    _num_kv_heads = self.config.num_key_value_heads # 8 for Llama-3.1-8B
    page_scores_buf = getattr(self, '_page_scores_buf', None)
    if (
        page_scores_buf is None
        or page_scores_buf.shape[0] != bsz
        or page_scores_buf.shape[1] != _num_kv_heads
        or page_scores_buf.shape[2] < num_pages
    ):
        self._page_scores_buf = torch.empty(
            bsz, _num_kv_heads, num_pages,
            dtype=torch.float32, device=paged_k.device,
        )

    score_query_states = query_states
    if getattr(cfg, "km_quest_split", 0) > 0:
        # K-M union strategy: DCT top-(K-M) + Quest top-M (Quest restricted to
        # pages NOT in DCT's set). Total = K pages. Boosts attn_recall via mass-
        # heavy Quest contributions at the margin while preserving DCT's needle
        # picks at the core.
        dct_scores = score_pages_triton(
            score_query_states, comp_k, cfg.scoring_method, cfg.group_agg_method, self.num_key_value_groups,
        )
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        quest_scores = _score_pages_quest(
            score_query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
        )
        K_topk = min(cfg.top_k, num_pages)
        M = min(int(cfg.km_quest_split), K_topk - 1)
        K_dct = K_topk - M
        dct_topkm = torch.topk(dct_scores, K_dct, dim=-1).indices                   # [bsz, kv, K-M]
        dct_mask = torch.zeros_like(dct_scores, dtype=torch.bool)
        dct_mask.scatter_(-1, dct_topkm, True)
        quest_scores_masked = quest_scores.masked_fill(dct_mask, float('-inf'))
        quest_topM = torch.topk(quest_scores_masked, M, dim=-1).indices             # [bsz, kv, M]
        # Build fused scores: DCT-selected get rank 0..K-M-1 (highest), Quest-selected
        # get rank K-M..K-1, rest get -K_topk (so top-K downstream picks exactly the union).
        fused = torch.full_like(
            self._page_scores_buf[:, :, :num_pages], float('-inf'),
        )
        # Rank values: highest score for DCT-selected, slightly lower for Quest-supplement,
        # -inf for others. Within each subset, give monotonic ranks for deterministic order.
        dct_ranks = torch.arange(K_dct, 0, -1, device=paged_k.device, dtype=torch.float32)  # K-M..1
        dct_ranks = dct_ranks.view(1, 1, K_dct).expand(bsz, _num_kv_heads, K_dct)
        fused.scatter_(-1, dct_topkm.long(), dct_ranks + M)                          # K_dct..K
        quest_ranks = torch.arange(M, 0, -1, device=paged_k.device, dtype=torch.float32)
        quest_ranks = quest_ranks.view(1, 1, M).expand(bsz, _num_kv_heads, M)
        fused.scatter_(-1, quest_topM.long(), quest_ranks)                           # 1..M
        self._page_scores_buf[:, :, :num_pages].copy_(fused)
        page_scores = self._page_scores_buf[:, :, :num_pages]
    elif cfg.score_combine_quest_dct:
        # Best-rank fusion: compute both DCT lowpass + Quest min/max scores, derive a
        # per-page rank in each, set page_scores = -min(rank_dct, rank_quest). Downstream
        # topk on these fused scores selects the union of top-K from each in min-rank order.
        dct_scores = score_pages_triton(
            score_query_states, comp_k, cfg.scoring_method, cfg.group_agg_method, self.num_key_value_groups,
        )                                                                       # [bsz, num_kv_heads, num_pages]
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        quest_scores = _score_pages_quest(
            score_query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
        )                                                                       # [bsz, num_kv_heads, num_pages]
        K_topk = min(cfg.top_k, num_pages)
        dct_topk_idx = dct_scores.topk(K_topk, dim=-1).indices                  # [bsz, num_kv_heads, K]
        quest_topk_idx = quest_scores.topk(K_topk, dim=-1).indices              # [bsz, num_kv_heads, K]
        arange_K = torch.arange(K_topk, device=paged_k.device, dtype=torch.long)
        arange_K = arange_K.view(1, 1, K_topk).expand(bsz, _num_kv_heads, K_topk)
        sentinel = K_topk  # pages outside top-K get rank = K (excluded)
        dct_rank = torch.full(
            (bsz, _num_kv_heads, num_pages), sentinel,
            dtype=torch.long, device=paged_k.device,
        )
        dct_rank.scatter_(-1, dct_topk_idx.long(), arange_K)
        quest_rank = torch.full_like(dct_rank, sentinel)
        quest_rank.scatter_(-1, quest_topk_idx.long(), arange_K)
        min_rank = torch.minimum(dct_rank, quest_rank).float()                  # [bsz, num_kv_heads, num_pages]
        fused = (-min_rank).to(self._page_scores_buf.dtype)
        self._page_scores_buf[:, :, :num_pages].copy_(fused)
        page_scores = self._page_scores_buf[:, :, :num_pages]
    elif cfg.score_use_quest_minmax:
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        page_scores = _score_pages_quest(
            score_query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
            out=self._page_scores_buf, top_k=cfg.top_k,
        )
    elif cfg.proxy_method == "pca_qaware":
        M_layer = _PROXY_BASIS_PCA.get(int(self.layer_idx))
        if M_layer is None:
            raise RuntimeError(
                f"pca_qaware: no PCA basis loaded for layer {self.layer_idx}; "
                f"call load_proxy_basis() with a calibrated qpca_*.pt file."
            )
        cs_h_used = cfg.pca_cs_h if cfg.pca_cs_h > 0 else M_layer.shape[1]
        page_scores = _score_pages_pca_qaware(
            score_query_states, paged_k, M_layer, cs_h_used,
            cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups, out=self._page_scores_buf,
        )
    elif cfg.proxy_method == "fasa_fc":
        idom_layer = _PROXY_BASIS_FASA.get(int(self.layer_idx))
        if idom_layer is None:
            raise RuntimeError(
                f"fasa_fc: no FASA I_dom loaded for layer {self.layer_idx}; "
                f"call load_proxy_basis() with a calibrated fasa_idom_*.pt file."
            )
        n_tip_used = cfg.fasa_n_tip if cfg.fasa_n_tip > 0 else idom_layer.shape[1]
        page_scores = _score_pages_fasa_fc(
            score_query_states, paged_k, idom_layer, n_tip_used,
            cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups, out=self._page_scores_buf,
        )
    elif cfg.proxy_method == "harp":
        page_scores = _score_pages_harp(
            score_query_states, comp_k, self.num_key_value_groups,
            cfg.scoring_method, cfg.group_agg_method,
            cfg.harp_detail_topk, out=self._page_scores_buf,
        )
    elif cfg.score_softmax:
        page_scores = _score_pages_dct_softmax(
            score_query_states, comp_k, cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups, out=self._page_scores_buf,
        )
    elif cfg.group_agg_method == "per_head_union":
        page_scores = _score_pages_dct_perheadunion(
            score_query_states, comp_k, cfg.scoring_method,
            self.num_key_value_groups, top_k=cfg.top_k,
            out=self._page_scores_buf,
        )
    elif cfg.qaware_lastq_topbins > 0:
        page_scores = _score_pages_dct_qaware_lastq(
            self, score_query_states, paged_k, cfg.qaware_lastq_topbins,
            self.num_key_value_groups,
            lastq_window=cfg.qaware_lastq_window,
            out=self._page_scores_buf,
        )
    else:
        page_scores = score_pages_triton(
            score_query_states, comp_k, cfg.scoring_method, cfg.group_agg_method, self.num_key_value_groups,
            out=self._page_scores_buf,
        )
    debug_hook = _dct_page_debug_hook
    oracle_page_scores = None
    if debug_hook is not None or cfg.select_with_oracle_page_scores:
        oracle_page_scores = _compute_debug_oracle_page_scores(
            self, query_states, paged_k, cfg, cos, sin
        )
    selection_page_scores = oracle_page_scores if cfg.select_with_oracle_page_scores else page_scores
    
    actual_top_k = min(cfg.top_k, num_pages)
    
    # Pre-allocate selected_indices buffer (constant shape across all decode steps)
    topk_buf = getattr(self, '_topk_out_buf', None)
    if (
        topk_buf is None
        or topk_buf.shape[0] != bsz
        or topk_buf.shape[1] != _num_kv_heads
        or topk_buf.shape[2] < actual_top_k
    ):
        self._topk_out_buf = torch.empty(
            bsz, _num_kv_heads, actual_top_k, dtype=torch.int32, device=paged_k.device
        )

    # Pre-allocate two-stage topk scratch (only used when num_pages > 1024).
    # Holds num_chunks * top_k packed int64 entries; sized generously once.
    _topk_scratch_n = 8 * actual_top_k
    scratch_buf = getattr(self, '_topk_scratch_buf', None)
    if (
        scratch_buf is None
        or scratch_buf.shape[0] != bsz
        or scratch_buf.shape[1] != _num_kv_heads
        or scratch_buf.shape[2] < _topk_scratch_n
    ):
        self._topk_scratch_buf = torch.empty(
            bsz, _num_kv_heads, _topk_scratch_n, dtype=torch.int64, device=paged_k.device
        )

    selected_indices = topk_sort(
        selection_page_scores,
        actual_top_k,
        out=self._topk_out_buf[:, :, :actual_top_k],
        scratch=self._topk_scratch_buf[:, :, :_topk_scratch_n],
        sort_ascending=(cfg.unselected_mode == "compressed"),
    )

    if debug_hook is not None:
        debug_hook(
            {
                "layer_idx": int(self.layer_idx),
                "kv_len": int(kv_len),
                "num_pages": int(num_pages),
                "actual_top_k": int(actual_top_k),
                "page_size": int(cfg.page_size),
                "num_sink_pages": int(cfg.num_sink_pages),
                "num_recent_pages": int(cfg.num_recent_pages),

                "cache_position": None
                if cache_position is None
                else cache_position.detach().cpu(),
                "page_scores": page_scores.detach().float().cpu(),
                "oracle_page_scores": oracle_page_scores.detach().float().cpu(),
                "selection_used_oracle_page_scores": bool(cfg.select_with_oracle_page_scores),
                "selected_indices": selected_indices.detach().cpu(),
                "sink_k": sink_k,
                "sink_v": sink_v,
                "paged_k": paged_k,
                "paged_v": paged_v,
                "recent_k": recent_k,
                "recent_v": recent_v,
                "query_states": query_states,
                "num_kv_groups": int(self.num_key_value_groups),
            }
        )

    # Step 6b: Assemble KV for attention.
    # With continuous_rope=False, all keys already have RoPE baked in.
    # No additional RoPE needed during assembly.
    bias_out_arg = None  # Set in compressed branch when weight_compressed_by_population is on

    if cfg.unselected_mode == "drop":
        assembled_len = cfg.num_sink_pages * cfg.page_size + actual_top_k * cfg.page_size + actual_recent

        # Pre-allocate or expand output buffers
        _buf_len = getattr(self, '_assemble_buf_len', 0)
        if assembled_len > _buf_len:
            _max_len = assembled_len + cfg.page_size
            _nkv = _num_kv_heads
            self._final_k_buf = torch.empty(
                bsz, _nkv, _max_len, self.head_dim, dtype=paged_k.dtype, device=paged_k.device
            )
            self._final_v_buf = torch.empty_like(self._final_k_buf)
            self._sel_idx_buf = torch.empty(
                bsz, _nkv, actual_top_k, dtype=torch.int32, device=paged_k.device
            )
            self._assemble_buf_len = _max_len

        final_k, final_v = assemble_kv_drop_triton(
            paged_k, paged_v,
            sink_k, sink_v, recent_k, recent_v,
            selected_indices,
            None, None,  # no RoPE in Triton (already baked in cache)
            out_k=self._final_k_buf,
            out_v=self._final_v_buf,
            out_sel_idx=self._sel_idx_buf,
            original_position_rope=False,
        )

    elif cfg.unselected_mode == "compressed":
        num_unselected = num_pages - actual_top_k

        # Determine effective number of unselected pages to keep as compressed.
        if cfg.max_unselected_compressed >= 0:
            effective_num_comp = min(cfg.max_unselected_compressed, num_unselected)
        else:
            effective_num_comp = num_unselected  # -1 means unlimited

        if effective_num_comp == 0:
            # ---- Path C: no compressed pages — equivalent to drop mode ----
            assembled_len = cfg.num_sink_pages * cfg.page_size + actual_top_k * cfg.page_size + actual_recent

            _buf_len = getattr(self, '_assemble_buf_len', 0)
            if assembled_len > _buf_len:
                _max_len = assembled_len + cfg.page_size
                _nkv = _num_kv_heads
                self._final_k_buf = torch.empty(
                    bsz, _nkv, _max_len, self.head_dim, dtype=paged_k.dtype, device=paged_k.device
                )
                self._final_v_buf = torch.empty_like(self._final_k_buf)
                self._sel_idx_buf = torch.empty(
                    bsz, _nkv, actual_top_k, dtype=torch.int32, device=paged_k.device
                )
                self._assemble_buf_len = _max_len

            final_k, final_v = assemble_kv_drop_triton(
                paged_k, paged_v,
                sink_k, sink_v, recent_k, recent_v,
                selected_indices,
                None, None,
                out_k=self._final_k_buf,
                out_v=self._final_v_buf,
                out_sel_idx=self._sel_idx_buf,
                original_position_rope=False,
            )

        elif effective_num_comp < num_unselected:
            # ---- Path B: limited compressed pages — PyTorch gather+scatter ----
            # Select which unselected pages to keep as compressed (top-N by score).
            _masked_scores = selection_page_scores.clone()
            _masked_scores.scatter_(2, selected_indices.long(), float('-inf'))
            _, compressed_indices = torch.topk(_masked_scores, effective_num_comp, dim=-1)
            compressed_indices = compressed_indices.sort(dim=-1).values  # [bsz, kv_heads, N]

            # Compute interleaved write offsets using searchsorted on sorted indices.
            _sel_long = selected_indices.long()
            _comp_long = compressed_indices.long()
            # For each selected page: how many compressed pages come before it?
            count_comp_before_sel = torch.searchsorted(_comp_long, _sel_long)
            _ranks_sel = torch.arange(actual_top_k, device=paged_k.device).view(1, 1, -1)
            selected_write_offsets = (
                cfg.num_sink_pages * cfg.page_size + _ranks_sel * cfg.page_size + count_comp_before_sel * comp_size
            )
            # For each compressed page: how many selected pages come before it?
            count_full_before_comp = torch.searchsorted(_sel_long, _comp_long)
            _ranks_comp = torch.arange(effective_num_comp, device=paged_k.device).view(1, 1, -1)
            compressed_write_offsets = (
                cfg.num_sink_pages * cfg.page_size + count_full_before_comp * cfg.page_size + _ranks_comp * comp_size
            )

            middle_len = actual_top_k * cfg.page_size + effective_num_comp * comp_size
            assembled_len = cfg.num_sink_pages * cfg.page_size + middle_len + actual_recent

            # Population weighting
            weight_pop = (
                cfg.weight_compressed_by_population
                and effective_num_comp > 0
                and comp_size < cfg.page_size
            )
            log_pop_weight = math.log(cfg.page_size / comp_size) if weight_pop else 0.0

            # Pre-allocate or expand output buffers
            _buf_len = getattr(self, '_assemble_buf_len', 0)
            if assembled_len > _buf_len:
                _max_len = assembled_len + cfg.page_size
                _nkv = _num_kv_heads
                self._final_k_buf = torch.empty(
                    bsz, _nkv, _max_len, self.head_dim, dtype=paged_k.dtype, device=paged_k.device
                )
                self._final_v_buf = torch.empty_like(self._final_k_buf)
                self._final_bias_buf = None
                self._assemble_buf_len = _max_len

            if weight_pop and (
                getattr(self, '_final_bias_buf', None) is None
                or self._final_bias_buf.shape[2] < self._assemble_buf_len
            ):
                self._final_bias_buf = torch.zeros(
                    bsz, _num_kv_heads, self._assemble_buf_len,
                    dtype=torch.float32, device=paged_k.device,
                )

            final_k = self._final_k_buf[:, :, :assembled_len, :]
            final_v = self._final_v_buf[:, :, :assembled_len, :]

            # --- Sink ---
            _sink_tokens = cfg.num_sink_pages * cfg.page_size
            final_k[:, :, :_sink_tokens, :] = sink_k
            final_v[:, :, :_sink_tokens, :] = sink_v

            # --- Scatter selected pages (full KV) ---
            _sel_idx_exp = _sel_long.unsqueeze(-1).unsqueeze(-1).expand(
                bsz, _num_kv_heads, actual_top_k, cfg.page_size, self.head_dim
            )
            _sel_k = paged_k.gather(2, _sel_idx_exp)
            _sel_v = paged_v.gather(2, _sel_idx_exp)
            _t_off_sel = torch.arange(cfg.page_size, device=paged_k.device).view(1, 1, 1, -1)
            _dest_sel = (selected_write_offsets.unsqueeze(-1) + _t_off_sel).reshape(
                bsz, _num_kv_heads, actual_top_k * cfg.page_size
            ).unsqueeze(-1).expand(bsz, _num_kv_heads, actual_top_k * cfg.page_size, self.head_dim)
            final_k.scatter_(2, _dest_sel, _sel_k.reshape(bsz, _num_kv_heads, -1, self.head_dim))
            final_v.scatter_(2, _dest_sel, _sel_v.reshape(bsz, _num_kv_heads, -1, self.head_dim))

            # --- Scatter compressed pages (compressed KV) ---
            _comp_idx_exp = _comp_long.unsqueeze(-1).unsqueeze(-1).expand(
                bsz, _num_kv_heads, effective_num_comp, comp_size, self.head_dim
            )
            _comp_k = comp_k.gather(2, _comp_idx_exp)
            _comp_v = comp_v.gather(2, _comp_idx_exp)
            _t_off_comp = torch.arange(comp_size, device=paged_k.device).view(1, 1, 1, -1)
            _dest_comp = (compressed_write_offsets.unsqueeze(-1) + _t_off_comp).reshape(
                bsz, _num_kv_heads, effective_num_comp * comp_size
            ).unsqueeze(-1).expand(bsz, _num_kv_heads, effective_num_comp * comp_size, self.head_dim)
            final_k.scatter_(2, _dest_comp, _comp_k.reshape(bsz, _num_kv_heads, -1, self.head_dim))
            final_v.scatter_(2, _dest_comp, _comp_v.reshape(bsz, _num_kv_heads, -1, self.head_dim))

            # --- Recent ---
            final_k[:, :, assembled_len - actual_recent:assembled_len, :] = recent_k
            final_v[:, :, assembled_len - actual_recent:assembled_len, :] = recent_v

            # --- Population bias ---
            if weight_pop:
                self._final_bias_buf[:, :, :assembled_len].zero_()
                _dest_comp_flat = (compressed_write_offsets.unsqueeze(-1) + _t_off_comp).reshape(
                    bsz, _num_kv_heads, effective_num_comp * comp_size
                )
                _bias_vals = torch.full_like(_dest_comp_flat, log_pop_weight, dtype=torch.float32)
                self._final_bias_buf.scatter_(2, _dest_comp_flat, _bias_vals)
                bias_out_arg = self._final_bias_buf

        else:
            # ---- Path A: all unselected pages compressed — existing Triton assembly ----
            middle_len = actual_top_k * cfg.page_size + num_unselected * comp_size
            assembled_len = cfg.num_sink_pages * cfg.page_size + middle_len + actual_recent

            # Population weighting
            weight_pop = (
                cfg.weight_compressed_by_population
                and num_unselected > 0
                and comp_size < cfg.page_size
            )
            if weight_pop:
                log_pop_weight = math.log(cfg.page_size / comp_size)
            else:
                log_pop_weight = 0.0

            # Pre-allocate or expand output buffers
            _buf_len = getattr(self, '_assemble_buf_len', 0)
            _buf_grew = False
            if assembled_len > _buf_len:
                _max_len = assembled_len + cfg.page_size
                _nkv = _num_kv_heads
                self._final_k_buf = torch.empty(
                    bsz, _nkv, _max_len, self.head_dim, dtype=paged_k.dtype, device=paged_k.device
                )
                self._final_v_buf = torch.empty_like(self._final_k_buf)
                self._sel_idx_buf = torch.empty(
                    bsz, _nkv, actual_top_k, dtype=torch.int32, device=paged_k.device
                )
                self._final_bias_buf = None
                self._assemble_buf_len = _max_len
                _buf_grew = True

            if weight_pop and (
                getattr(self, '_final_bias_buf', None) is None
                or self._final_bias_buf.shape[2] < self._assemble_buf_len
            ):
                self._final_bias_buf = torch.zeros(
                    bsz, _num_kv_heads, self._assemble_buf_len,
                    dtype=torch.float32, device=paged_k.device,
                )

            if weight_pop:
                self._final_bias_buf[:, :, :assembled_len].zero_()
                bias_out_arg = self._final_bias_buf
            else:
                bias_out_arg = None

            _cur_paged_strides = (paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3))
            _cur_bias_strides = (
                (self._final_bias_buf.stride(0), self._final_bias_buf.stride(1), self._final_bias_buf.stride(2))
                if weight_pop else (0, 0, 0)
            )
            _cached = getattr(self, '_assemble_stride_cache', None)
            if (_buf_grew
                    or _cached is None
                    or _cached['paged_strides'] != _cur_paged_strides
                    or _cached.get('bias_strides', (0, 0, 0)) != _cur_bias_strides):
                self._assemble_stride_cache = build_assemble_stride_cache(
                    paged_k, comp_k, sink_k, recent_k, selected_indices,
                    None, self._final_k_buf,
                    bias_out=self._final_bias_buf if weight_pop else None,
                )

            final_k, final_v = assemble_kv_split_triton(
                paged_k, paged_v, comp_k, comp_v,
                sink_k, sink_v, recent_k, recent_v,
                selected_indices,
                None, None,  # no RoPE in Triton (already baked in cache)
                out_k=self._final_k_buf,
                out_v=self._final_v_buf,
                stride_cache=self._assemble_stride_cache,
                bias_out=bias_out_arg,
                log_pop_weight=log_pop_weight,
            )

    else:
        raise ValueError(f"Unsupported unselected_mode: {cfg.unselected_mode}")

    # Step 7a: Compute attention (no causal mask needed for q_len=1).
    # In compressed mode with population weighting, pass the per-position bias
    # built by the assembly kernel as an additive attn_mask. SDPA adds it to
    # the QK logits before softmax, which (via the log(n) bias trick) makes
    # each unselected-page rep contribute as if it were page_size/comp_size
    # real tokens — analogous to multipole_attn's `p * nkeys` weighting.
    if cfg.unselected_mode == "compressed" and bias_out_arg is not None:
        # bias_out_arg: [bsz, num_kv_heads, max_len]. SDPA with enable_gqa
        # internally repeats K/V to num_q_heads, so attn_mask must also be at
        # the query-head granularity (or broadcastable to it). The bias layout
        # is per-kv-head because selected_indices differs per kv-head, so we
        # repeat_interleave by num_key_value_groups along the head dim.
        kv_bias = bias_out_arg[:, :, :final_k.shape[2]]
        attn_bias = kv_bias.repeat_interleave(self.num_key_value_groups, dim=1).unsqueeze(2)
    else:
        attn_bias = None

    # Outlier bank: append always-attended outlier K/V (detected once post-prefill)
    # to the assembled KV. Outliers get zero bias so they participate with their
    # natural QK logits alongside sink + selected + recent.
    if cfg.outlier_budget > 0 and getattr(self, "_outlier_K", None) is not None:
        final_k = torch.cat([final_k, self._outlier_K], dim=2)
        final_v = torch.cat([final_v, self._outlier_V], dim=2)
        if attn_bias is not None:
            M_out = self._outlier_K.shape[2]
            zero_pad = attn_bias.new_zeros(
                attn_bias.shape[0], attn_bias.shape[1], attn_bias.shape[2], M_out
            )
            attn_bias = torch.cat([attn_bias, zero_pad], dim=3)

    attn_output = F.scaled_dot_product_attention(
        query_states, final_k, final_v,
        attn_mask=attn_bias,
        is_causal=False,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

    # Step 7b: Output projection
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# FlashInfer forward — Phase 2b Stage 7
# ---------------------------------------------------------------------------
def dct_page_attention_forward_flashinfer(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """FlashInfer-backed decode forward. Drop-mode only.

    Replaces `assemble_kv_drop_triton` + SDPA with
    `topk_sort_and_pack_triton` + `flashinfer_decode_attention`, eliminating
    the gather pass. Prefill and short-KV fallback delegate to
    `dct_page_attention_forward` (no duplication).

    Caller contract: build `FlashInferPagedKVCache` post-prefill and assign
    to `_flashinfer_cache_ref[0]` before the first decode step. `cfg.top_k`
    must match the `top_k` passed to `build_flashinfer_paged_cache`.

    Verify path: when `self._verify_flashinfer` is True, gather the same
    selected pages from `cache.buf` via `cache.indices_buf` and run SDPA;
    append `(layer_idx, step_max_abs_diff)` to `self._verify_diffs`.
    """
    cfg = _dct_page_cfg
    if cfg.unselected_mode != "drop":
        raise NotImplementedError(
            "dct_page_attention_forward_flashinfer supports drop mode only; "
            "use dct_page_attention_forward for compressed mode."
        )
    if cfg.continuous_rope:
        raise NotImplementedError(
            "continuous_rope=True is temporarily disabled."
        )

    input_shape = hidden_states.shape[:-1]
    bsz, q_len = input_shape
    _maybe_reset_dct_runtime_state(self, past_key_values)

    # Prefill delegates to the SDPA forward (cache alloc + prefill attention).
    if q_len > 1:
        return dct_page_attention_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    # Short-KV fallback: delegate to the SDPA forward. This peeks at the
    # cached length pre-projection so we don't double-update the cache.
    min_len_for_paging = max(
        (cfg.num_sink_pages + cfg.top_k + 1 + cfg.num_recent_pages) * cfg.page_size,
        getattr(cfg, "min_decode_kv_len_for_paging", 0),
    )
    if past_key_values is not None:
        prev_len = int(past_key_values.layers[self.layer_idx].get_seq_length())
    else:
        prev_len = 0
    if prev_len + q_len < min_len_for_paging:
        return dct_page_attention_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    # ---- DECODE PATH with FlashInfer ----
    hidden_shape = (*input_shape, -1, self.head_dim)
    _has_qk_norm = hasattr(self, "q_norm") and hasattr(self, "k_norm")

    # Step 1: QKV projection (with optional QK-norm for Qwen3).
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if _has_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Step 2: RoPE + bookkeeping. The FI paged buf is the single source of
    # truth at decode time; `past_key_values.update()` is a counter-only shim
    # that advances `_seen` (flat keys/values were freed at FI build).
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Acquire FlashInfer cache (built once by the driver post-prefill).
    cache = _flashinfer_cache_ref[0]
    if cache is None:
        raise RuntimeError(
            "FlashInfer cache is not set. Build via "
            "speed.flashinfer_backend.build_flashinfer_paged_cache(...) "
            "post-prefill and assign to "
            "dct_page_attention._flashinfer_cache_ref[0] before decode."
        )
    if cache.top_k != cfg.top_k:
        raise RuntimeError(
            f"cfg.top_k ({cfg.top_k}) != FlashInfer cache.top_k "
            f"({cache.top_k}); cache top_k is fixed at build time."
        )

    # Lazy import: flashinfer_backend lives under speed/, not the repo root.
    # Caller must have the project root on sys.path (standard for entry points).
    import sys as _sys, os as _os
    _speed_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "speed")
    if _speed_dir not in _sys.path:
        _sys.path.insert(0, _speed_dir)
    from flashinfer_backend import (
        append_flashinfer_cache,
        flashinfer_decode_attention,
    )

    # Step 3: Append new K/V into FlashInfer's paged cache. Only layer 0
    # advances the shared counters (last_page_idx_py / last_page_len_py).
    append_flashinfer_cache(
        cache,
        key_states[:, :, -1:, :],
        value_states[:, :, -1:, :],
        self.layer_idx,
    )

    # Step 4: Build paged views over cache.buf and update the compressed
    # proxy for scoring. `num_pages` matches FI's eligible middle range so
    # comp_cache and FI selection see the same page set. `last_page_idx_py`
    # is the per-batch LOGICAL page index (lockstep across batch).
    num_pages = (
        cache.last_page_idx_py - cache.num_sink_pages
        - cache.num_recent_pages_fixed
    )
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    paged_k, paged_v = paged_views_from_buf(
        cache.buf[self.layer_idx], cache.num_sink_pages, num_pages,
        bsz=cache.bsz, pages_per_batch=cache.pages_per_batch,
    )
    comp_k, comp_v = _update_comp_cache(
        self, paged_k, paged_v, num_pages, comp_size, cfg,
    )

    # Step 5: Score pages. Buffer sized for num_pages; we pass a slice to
    # the fused kernel so its writes match FI's eligible middle range.
    _num_kv_heads = self.config.num_key_value_heads
    page_scores_buf = getattr(self, '_page_scores_buf', None)
    if (
        page_scores_buf is None
        or page_scores_buf.shape[0] != bsz
        or page_scores_buf.shape[1] != _num_kv_heads
        or page_scores_buf.shape[2] < num_pages
    ):
        self._page_scores_buf = torch.empty(
            bsz, _num_kv_heads, num_pages,
            dtype=torch.float32, device=paged_k.device,
        )
    if cfg.score_use_quest_minmax:
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        page_scores = _score_pages_quest(
            query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )
    elif cfg.score_softmax:
        page_scores = _score_pages_dct_softmax(
            query_states, comp_k, cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )
    else:
        page_scores = score_pages_triton(
            query_states, comp_k, cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )

    # Step 6: num_pages already matches FI's eligible middle range (count of
    # pages between sink and recent, excluding the open page). Score the full
    # range — no truncation needed in the unified paged-buf layout.
    if num_pages < cache.top_k:
        raise RuntimeError(
            f"num_pages ({num_pages}) < cache.top_k ({cache.top_k}). "
            f"Configure min_decode_kv_len_for_paging to keep decode in the "
            f"steady-state regime before enabling FlashInfer."
        )
    eff_scores = page_scores[:, :, :num_pages]

    # Step 7: Fused topk + pack. Writes indices_buf[:, :, num_sink:num_sink+top_k]
    # with (topk_indices + num_sink_pages + b*pages_per_batch) and the recent
    # slice with (last_page_idx[b] + recent_offsets) where last_page_idx is
    # already batch-biased. Sink slice was filled at cache init (also biased).
    topk_sort_and_pack_triton(
        eff_scores,
        cache.indices_buf,
        num_sink_pages=cache.num_sink_pages,
        top_k=cache.top_k,
        last_page_idx=cache.last_page_idx,
        recent_offsets=cache.recent_offsets,
        sort_ascending=False,  # drop mode — order-invariant middle
        pages_per_batch=(cache.pages_per_batch if cache.bsz > 1 else 0),
    )

    # Step 8: FlashInfer paged decode attention. Native bf16 end-to-end.
    attn_output_fi = flashinfer_decode_attention(
        query_states, cache, self.layer_idx,
    )  # (1, num_qo_heads, 1, head_dim) bf16

    # Optional verify: gather the SAME pages FI used and run SDPA, then
    # compare. Identical K/V coverage on both sides, so the diff measures
    # kernel numerics only (bf16 floor + split_kv drift). Per-(b, h) gather
    # — `cache.indices_buf[b, h]` holds batch-biased physical IDs that index
    # `cache.buf[layer_idx]` directly.
    if getattr(self, "_verify_flashinfer", False):
        buf_l = cache.buf[self.layer_idx]       # (cap, 2, ps, nkv, d)
        page_budget = cache.page_budget
        last_page_len = cache.last_page_len_py
        full_len = (page_budget - 1) * cache.page_size + last_page_len
        cache_bsz = cache.bsz
        batch_kv = []
        for b in range(cache_bsz):
            k_pages = []
            v_pages = []
            for h in range(_num_kv_heads):
                sel_h = cache.indices_buf[b, h].long()
                kv_h = buf_l[sel_h][:, :, :, h, :]    # (page_budget, 2, ps, d)
                k_h = kv_h[:, 0].reshape(page_budget * cache.page_size, self.head_dim)
                v_h = kv_h[:, 1].reshape(page_budget * cache.page_size, self.head_dim)
                k_pages.append(k_h[:full_len])
                v_pages.append(v_h[:full_len])
            k_b = torch.stack(k_pages, dim=0)      # (nkv, full_len, d)
            v_b = torch.stack(v_pages, dim=0)
            batch_kv.append((k_b, v_b))
        k_flat = torch.stack([kv[0] for kv in batch_kv], dim=0)   # (bsz, nkv, full_len, d)
        v_flat = torch.stack([kv[1] for kv in batch_kv], dim=0)
        sdpa_out = F.scaled_dot_product_attention(
            query_states, k_flat, v_flat,
            is_causal=False, enable_gqa=True,
        )
        max_diff = (attn_output_fi.float() - sdpa_out.float()).abs().max().item()
        if not hasattr(self, "_verify_diffs"):
            self._verify_diffs = []
        self._verify_diffs.append(max_diff)

    # Step 9: Output projection. attn_output_fi has the same (bsz, num_qo_heads,
    # 1, head_dim) layout as SDPA's pre-transpose output — reuse the tail.
    attn_output = attn_output_fi.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Upstream-FlashInfer forward — eval-side canonical
# ---------------------------------------------------------------------------
def dct_page_attention_forward_upstream_flashinfer(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """Upstream-FlashInfer-backed decode forward. Drop-mode only.

    Mirrors `profile_decode_upstream_flash_infer.profiled_dct_upstream_flashinfer_forward`
    minus the profiler instrumentation, with the additional Quest-minmax
    branch from the fork-FI variant. Lazy-builds the upstream-FI cache on
    the first decode step (layer 0) so eval scripts keep using
    `model.generate()` unchanged.

    Cache lifecycle: built by this forward on first decode step; torn down
    between every `model.generate()` call by `_generate_with_upstream_fi`.

    Verify path: when `self._verify_upstream` is True, gather the same pages
    upstream-FI used (per-(b, h) head-local indices in `cache.indices_buf_3d`)
    and run SDPA, appending the per-step max-abs-diff to `self._verify_diffs`.
    """
    cfg = _dct_page_cfg
    if cfg.unselected_mode != "drop":
        raise NotImplementedError(
            "dct_page_attention_forward_upstream_flashinfer supports drop mode only; "
            "use dct_page_attention_forward (--attention_backend sdpa) for compressed mode."
        )
    if cfg.continuous_rope:
        raise NotImplementedError(
            "continuous_rope=True is temporarily disabled."
        )

    input_shape = hidden_states.shape[:-1]
    bsz, q_len = input_shape
    _maybe_reset_dct_runtime_state(self, past_key_values)

    # Prefill delegates to the SDPA forward (cache alloc + prefill attention).
    if q_len > 1:
        return dct_page_attention_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    # Short-KV fallback: delegate to the SDPA forward. Peek at the cached
    # length pre-projection so we don't double-update the cache.
    min_len_for_paging = max(
        (cfg.num_sink_pages + cfg.top_k + 1 + cfg.num_recent_pages) * cfg.page_size,
        getattr(cfg, "min_decode_kv_len_for_paging", 0),
    )
    if past_key_values is not None:
        prev_len = int(past_key_values.layers[self.layer_idx].get_seq_length())
    else:
        prev_len = 0
    if prev_len + q_len < min_len_for_paging:
        return dct_page_attention_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    # ---- DECODE PATH with upstream FlashInfer ----
    hidden_shape = (*input_shape, -1, self.head_dim)
    _has_qk_norm = hasattr(self, "q_norm") and hasattr(self, "k_norm")

    # Step 1: QKV projection (with optional QK-norm for Qwen3).
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if _has_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Lazy-init the upstream-FI cache on the first decode step (layer 0 only).
    # MUST happen BEFORE `past_key_values.update()` — otherwise layer 0's
    # update bumps `_seen` to `prefill_len + 1` and writes its first decode K
    # to the flat KV buffer, while layers 1..N-1 have not yet had update
    # called (we are still inside layer 0's forward). The cache build then
    # packs `layer.keys[:, :, :prefill_len + 1, :]` for ALL layers, but the
    # +1 slot is uninitialized garbage for every layer except layer 0. The
    # subsequent explicit per-layer K/V write also lands at slot 1 instead
    # of slot 0 of the open page (because `last_page_len_py` is 1 at build
    # and gets bumped to 2 in the layer-0 counter-advance below). Building
    # before update — with `prefill_len = layer 0._seen` (pre-update) —
    # mirrors the profiler driver and keeps the open-page slot accounting
    # consistent with the explicit decode-time write.
    # Prefill already ran `pre_allocate_cache` — DO NOT call it again here;
    # the second call would null-deref on `_fi_mode=True` layers.
    if self.layer_idx == 0 and _upstream_fi_cache_ref[0] is None:
        # Lazy import: upstream_flashinfer_backend lives under speed/, not the
        # repo root. Keeps flashinfer import out of the cold path.
        import sys as _sys, os as _os
        _speed_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "speed")
        if _speed_dir not in _sys.path:
            _sys.path.insert(0, _speed_dir)
        from upstream_flashinfer_backend import (
            build_upstream_flashinfer_paged_cache,
        )

        prefill_len = int(past_key_values.layers[0]._seen)
        max_decode_steps = int(
            getattr(self, "_upstream_fi_build_kwargs", {}).get("max_decode_steps", 0)
        )
        _upstream_fi_cache_ref[0] = build_upstream_flashinfer_paged_cache(
            preallocated_layers=past_key_values.layers,
            prefill_len=prefill_len,
            page_size=cfg.page_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.head_dim,
            num_qo_heads=self.config.num_attention_heads,
            num_layers=self.config.num_hidden_layers,
            max_decode_steps=max_decode_steps,
            dtype=past_key_values.layers[0].keys.dtype,
            device=past_key_values.layers[0].keys.device,
            num_sink_pages=cfg.num_sink_pages,
            top_k=cfg.top_k,
            num_recent_pages_fixed=cfg.num_recent_pages,
            bsz=hidden_states.shape[0],
        )

    # Step 2: RoPE + bookkeeping. The FI paged buf is the source of truth at
    # decode time; `past_key_values.update()` is a counter-only shim (in
    # _fi_mode the flat KV was freed at FI build).
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    cache = _upstream_fi_cache_ref[0]
    if cache is None:
        raise RuntimeError(
            "upstream FI cache is not set. The lazy-init on layer 0 should "
            "have populated it. Call `_generate_with_upstream_fi(...)` rather "
            "than `model.generate(...)` directly."
        )
    if cache.top_k != cfg.top_k:
        raise RuntimeError(
            f"cfg.top_k ({cfg.top_k}) != cache.top_k ({cache.top_k}); "
            f"cache top_k is fixed at build time."
        )

    # Lazy import the per-step entry points.
    import sys as _sys, os as _os
    _speed_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "speed")
    if _speed_dir not in _sys.path:
        _sys.path.insert(0, _speed_dir)
    from upstream_flashinfer_backend import (
        refresh_upstream_indices_flat,
        upstream_flashinfer_decode_attention,
    )

    # Layer-0 advances the shared per-batch counters (lockstep across batch).
    if self.layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1
        cache.last_page_idx.fill_(cache.last_page_idx_py)
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    # Step 3: paged views from FI buf. cache.buf_views[l] shape
    # (B, H, P, 2, ps, 1, d) sliced to (B, H, num_pages, ps, d).
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    num_pages = (
        cache.last_page_idx_py - cache.num_sink_pages
        - cache.num_recent_pages_fixed
    )
    buf_l = cache.buf_views[self.layer_idx]   # (B, H, P, 2, ps, 1, d)
    middle = buf_l[:, :, cache.num_sink_pages:cache.num_sink_pages + num_pages]
    paged_k = middle[:, :, :, 0, :, 0, :]   # (B, H, num_pages, ps, d)
    paged_v = middle[:, :, :, 1, :, 0, :]

    # Step 4: compressed page cache (DCT proxy for scoring).
    comp_k, comp_v = _update_comp_cache(
        self, paged_k, paged_v, num_pages, comp_size, cfg,
    )

    # Step 5: score pages.
    _num_kv_heads = self.config.num_key_value_heads
    page_scores_buf = getattr(self, '_page_scores_buf', None)
    if (
        page_scores_buf is None
        or page_scores_buf.shape[0] != bsz
        or page_scores_buf.shape[1] != _num_kv_heads
        or page_scores_buf.shape[2] < num_pages
    ):
        self._page_scores_buf = torch.empty(
            bsz, _num_kv_heads, num_pages,
            dtype=torch.float32, device=paged_k.device,
        )
    if cfg.score_use_quest_minmax:
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        page_scores = _score_pages_quest(
            query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )
    elif cfg.score_softmax:
        page_scores = _score_pages_dct_softmax(
            query_states, comp_k, cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )
    else:
        page_scores = score_pages_triton(
            query_states, comp_k, cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )

    # Step 6: fused topk + pack — head-local IDs (upstream contract).
    if num_pages < cache.top_k:
        raise RuntimeError(
            f"num_pages ({num_pages}) < cache.top_k ({cache.top_k}). "
            f"Configure min_decode_kv_len_for_paging."
        )
    eff_scores = page_scores[:, :, :num_pages]
    topk_sort_and_pack_triton(
        eff_scores,
        cache.indices_buf_3d,
        num_sink_pages=cache.num_sink_pages,
        top_k=cache.top_k,
        last_page_idx=cache.last_page_idx,
        recent_offsets=cache.recent_offsets,
        sort_ascending=False,
        pages_per_batch=0,
        allow_head_local_multibatch=True,
    )

    # Step 7: cache K/V write + indices bias + FI run.
    page_idx = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    if page_idx >= cache.pages_per_head:
        raise RuntimeError(
            f"upstream-FI cache overflow: page_idx={page_idx} >= "
            f"pages_per_head={cache.pages_per_head}"
        )
    k_flat = key_states[:, :, -1:, :].reshape(bsz, cache.num_kv_heads, cache.head_dim)
    v_flat = value_states[:, :, -1:, :].reshape(bsz, cache.num_kv_heads, cache.head_dim)
    cache.buf_views[self.layer_idx][:, :, page_idx, 0, slot, 0, :].copy_(k_flat)
    cache.buf_views[self.layer_idx][:, :, page_idx, 1, slot, 0, :].copy_(v_flat)

    refresh_upstream_indices_flat(cache)
    attn_output_fi = upstream_flashinfer_decode_attention(
        query_states, cache, self.layer_idx,
    )

    # Verify path — gather the SAME pages FI saw and run SDPA.
    if getattr(self, "_verify_upstream", False):
        buf_l_8d = cache.buf_views[self.layer_idx]  # (B, H, P, 2, ps, 1, d)
        page_budget = cache.page_budget
        last_page_len = cache.last_page_len_py
        full_len = (page_budget - 1) * cache.page_size + last_page_len
        batch_kv = []
        for b in range(bsz):
            k_pages = []
            v_pages = []
            for h in range(_num_kv_heads):
                sel_bh = cache.indices_buf_3d[b, h].long()    # head-local IDs
                kv_bh = buf_l_8d[b, h][sel_bh]                # (page_budget, 2, ps, 1, d)
                k_bh = kv_bh[:, 0, :, 0, :].reshape(
                    page_budget * cache.page_size, self.head_dim
                )
                v_bh = kv_bh[:, 1, :, 0, :].reshape(
                    page_budget * cache.page_size, self.head_dim
                )
                k_pages.append(k_bh[:full_len])
                v_pages.append(v_bh[:full_len])
            batch_kv.append((torch.stack(k_pages, dim=0), torch.stack(v_pages, dim=0)))
        k_ref = torch.stack([kv[0] for kv in batch_kv], dim=0)  # (B, H, full_len, d)
        v_ref = torch.stack([kv[1] for kv in batch_kv], dim=0)
        sdpa_out = F.scaled_dot_product_attention(
            query_states, k_ref, v_ref,
            is_causal=False, enable_gqa=True,
        )
        max_diff = (attn_output_fi.float() - sdpa_out.float()).abs().max().item()
        if not hasattr(self, "_verify_diffs"):
            self._verify_diffs = []
        self._verify_diffs.append(max_diff)

    # Step 8: Output projection.
    attn_output = attn_output_fi.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Upstream-FI lifecycle helpers
# ---------------------------------------------------------------------------
def _init_upstream_fi_build_kwargs(model):
    """Walk model.modules() and seed `_upstream_fi_build_kwargs = {}` on each
    attention module. Eval harness must call this AFTER `from_pretrained`
    returns; `replace_*_attn` runs BEFORE model load by project convention.
    """
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            module._upstream_fi_build_kwargs = {}


def _set_upstream_fi_max_decode_steps(model, max_decode_steps):
    """Set per-instance `_upstream_fi_build_kwargs['max_decode_steps']` on
    every attention module. The +16 padding mirrors the profile driver
    (profile_decode_upstream_flash_infer.py:425) — covers EOS-token-dodging
    overshoots without crashing on `cache overflow`.
    """
    padded = int(max_decode_steps) + 16
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            kwargs = getattr(module, "_upstream_fi_build_kwargs", None)
            if kwargs is None:
                module._upstream_fi_build_kwargs = {}
                kwargs = module._upstream_fi_build_kwargs
            kwargs["max_decode_steps"] = padded


def reset_upstream_fi_cache_state(model):
    """Tear down upstream-FI per-generate state. Called between every HF
    `model.generate()` call by `_generate_with_upstream_fi`.

    Clears module-level `_upstream_fi_cache_ref`, plus per-attention-module
    `_dct_runtime_cache_ref`, `_verify_diffs`, `_page_scores_buf`,
    `_upstream_fi_build_kwargs`. `_dct_runtime_cache_ref` and `_verify_diffs`
    are cleared EXPLICITLY here — they are deliberately NOT in
    `_DCT_RUNTIME_STATE_ATTRS` because `_dct_runtime_cache_ref` is the guard
    variable that triggers `_maybe_reset_dct_runtime_state` (lines 1119-1121);
    adding it to the auto-reset tuple would defeat that guard.
    """
    _upstream_fi_cache_ref[0] = None
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            for attr in ("_dct_runtime_cache_ref", "_verify_diffs", "_page_scores_buf"):
                if hasattr(module, attr):
                    delattr(module, attr)
            module._upstream_fi_build_kwargs = {}


def _generate_with_upstream_fi(model, input_ids, max_new_tokens, on_post_generate=None, **gen_kwargs):
    """Centralized teardown contract for upstream-FI generates.

    Defensive double-clear at entry covers crash-mid-build cases (where
    `_upstream_fi_cache_ref[0]` was populated but per-layer state was partial).
    `try/finally` guarantees teardown on EOS-early-stop, KeyboardInterrupt,
    and OOM mid-generate.

    `on_post_generate`: optional callable(model, output_ids) -> None, runs
    inside the try block AFTER `model.generate()` returns successfully but
    BEFORE the finally-block teardown. Use this to harvest per-layer
    `_verify_diffs` before `reset_upstream_fi_cache_state()` deletes them.
    Callback exceptions propagate (fatal).
    """
    reset_upstream_fi_cache_state(model)
    _set_upstream_fi_max_decode_steps(model, max_new_tokens)
    try:
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
        if on_post_generate is not None:
            on_post_generate(model, output_ids)
        return output_ids
    finally:
        reset_upstream_fi_cache_state(model)
        import torch as _torch
        _torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Monkey-patch entry point
# ---------------------------------------------------------------------------
def _select_dct_forward(attention_backend):
    """Pick the forward to monkey-patch based on `attention_backend`.

    `"sdpa"` — original DCT forward (assemble_kv_drop_triton + SDPA).
    `"flashinfer"` — Phase 2b Stage 7 fork-FlashInfer forward (drop mode only,
    reads FI cache from `_flashinfer_cache_ref[0]` which the driver/harness
    populates post-prefill).
    `"upstream_flashinfer"` — eval-side upstream-FlashInfer forward (drop
    mode only); lazy-builds the cache on the first decode step. Eval scripts
    must wrap `model.generate()` calls in `_generate_with_upstream_fi(...)`
    so the per-generate teardown contract holds.
    """
    if attention_backend == "sdpa":
        return dct_page_attention_forward
    if attention_backend == "flashinfer":
        return dct_page_attention_forward_flashinfer
    if attention_backend == "upstream_flashinfer":
        return dct_page_attention_forward_upstream_flashinfer
    raise ValueError(
        f"Unsupported attention_backend={attention_backend!r}; "
        f"expected 'sdpa', 'flashinfer', or 'upstream_flashinfer'."
    )


def replace_qwen2_attn(
    page_size=32,
    top_k=64,
    num_sink_pages=0,
    num_recent_pages=0,
    compress_ratio=0.03125,
    proxy_method="dct",
    haar_detail_per_block=0,
    haar_detail_with_negation=False,
    harp_detail_topk=4,
    min_decode_kv_len_for_paging=8192,
    dense_first_n_layers=0,
    scoring_method="max",
    group_agg_method="mean",
    score_softmax=False,
    qaware_lastq_topbins=0,
    qaware_lastq_window=1,
    unselected_mode="drop",
    compressed_token_rope="mixed",
    continuous_rope=False,
    score_use_quest_minmax=False,
    score_combine_quest_dct=False,
    km_quest_split=0,
    select_with_oracle_page_scores=False,
    oracle_score_mode="max",
    use_triton=True,
    weight_compressed_by_population=False,
    max_unselected_compressed=-1,
    comp_kv_quant="none",
    comp_kv_quant_granularity="per_page",
    outlier_budget=0,
    outlier_detector="lastq_mean",
    cluster_outlier_N=256,
    cluster_outlier_iters=5,
    cluster_outlier_top_k=8,
    cluster_outlier_q_agg="mean",
    cluster_outlier_scoring="centroid",
    attention_backend="sdpa",
    proxy_basis_path: str = "",
    pca_cs_h: int = 0,
    fasa_n_tip: int = 0,
):
    """
    Replace Qwen2Attention.forward with DCT Page Attention.

    Must be called BEFORE loading the model.
    """
    global _dct_page_cfg
    _dct_page_cfg = DCTPageConfig(
        page_size=page_size,
        top_k=top_k,
        num_sink_pages=num_sink_pages,
        num_recent_pages=num_recent_pages,
        compress_ratio=compress_ratio,
        proxy_method=proxy_method,
        haar_detail_per_block=haar_detail_per_block,
        haar_detail_with_negation=haar_detail_with_negation,
        harp_detail_topk=harp_detail_topk,
        min_decode_kv_len_for_paging=min_decode_kv_len_for_paging,
        dense_first_n_layers=dense_first_n_layers,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        score_softmax=score_softmax,
        qaware_lastq_topbins=qaware_lastq_topbins,
        qaware_lastq_window=qaware_lastq_window,
        unselected_mode=unselected_mode,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
        score_use_quest_minmax=score_use_quest_minmax,
        score_combine_quest_dct=score_combine_quest_dct,
        km_quest_split=km_quest_split,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        oracle_score_mode=oracle_score_mode,
        use_triton=use_triton,
        weight_compressed_by_population=weight_compressed_by_population,
        max_unselected_compressed=max_unselected_compressed,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
        outlier_budget=outlier_budget,
        outlier_detector=outlier_detector,
        cluster_outlier_N=cluster_outlier_N,
        cluster_outlier_iters=cluster_outlier_iters,
        cluster_outlier_top_k=cluster_outlier_top_k,
        cluster_outlier_q_agg=cluster_outlier_q_agg,
        cluster_outlier_scoring=cluster_outlier_scoring,
        proxy_basis_path=proxy_basis_path,
        pca_cs_h=pca_cs_h,
        fasa_n_tip=fasa_n_tip,
    )
    if proxy_basis_path:
        load_proxy_basis(proxy_basis_path)
        print(f"  Loaded proxy basis from {proxy_basis_path}")

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config:")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  num_sink_pages={num_sink_pages}, num_recent_pages={num_recent_pages}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}, score_softmax={score_softmax}")
    print(f"  unselected_mode={unselected_mode}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}, "
        f"weight_compressed_by_population={weight_compressed_by_population}, "
        f"max_unselected_compressed={max_unselected_compressed}, "
        f"comp_kv_quant={comp_kv_quant}, "
        f"comp_kv_quant_granularity={comp_kv_quant_granularity}, "
        f"attention_backend={attention_backend}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = _select_dct_forward(attention_backend)


def replace_qwen3_attn(
    page_size=32,
    top_k=64,
    num_sink_pages=0,
    num_recent_pages=0,
    compress_ratio=0.03125,
    proxy_method="dct",
    haar_detail_per_block=0,
    haar_detail_with_negation=False,
    harp_detail_topk=4,
    min_decode_kv_len_for_paging=8192,
    dense_first_n_layers=0,
    scoring_method="max",
    group_agg_method="mean",
    score_softmax=False,
    qaware_lastq_topbins=0,
    qaware_lastq_window=1,
    unselected_mode="drop",
    compressed_token_rope="mixed",
    continuous_rope=False,
    score_use_quest_minmax=False,
    score_combine_quest_dct=False,
    km_quest_split=0,
    select_with_oracle_page_scores=False,
    oracle_score_mode="max",
    use_triton=True,
    weight_compressed_by_population=False,
    max_unselected_compressed=-1,
    comp_kv_quant="none",
    comp_kv_quant_granularity="per_page",
    outlier_budget=0,
    outlier_detector="lastq_mean",
    cluster_outlier_N=256,
    cluster_outlier_iters=5,
    cluster_outlier_top_k=8,
    cluster_outlier_q_agg="mean",
    cluster_outlier_scoring="centroid",
    attention_backend="sdpa",
    proxy_basis_path: str = "",
    pca_cs_h: int = 0,
    fasa_n_tip: int = 0,
):
    """
    Replace Qwen3Attention.forward with DCT Page Attention.

    Must be called BEFORE loading the model.
    Qwen3Attention uses QK-norm (q_norm/k_norm) which is handled inside
    dct_page_attention_forward via hasattr checks.
    """
    global _dct_page_cfg
    _dct_page_cfg = DCTPageConfig(
        page_size=page_size,
        top_k=top_k,
        num_sink_pages=num_sink_pages,
        num_recent_pages=num_recent_pages,
        compress_ratio=compress_ratio,
        proxy_method=proxy_method,
        haar_detail_per_block=haar_detail_per_block,
        haar_detail_with_negation=haar_detail_with_negation,
        harp_detail_topk=harp_detail_topk,
        min_decode_kv_len_for_paging=min_decode_kv_len_for_paging,
        dense_first_n_layers=dense_first_n_layers,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        score_softmax=score_softmax,
        qaware_lastq_topbins=qaware_lastq_topbins,
        qaware_lastq_window=qaware_lastq_window,
        unselected_mode=unselected_mode,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
        score_use_quest_minmax=score_use_quest_minmax,
        score_combine_quest_dct=score_combine_quest_dct,
        km_quest_split=km_quest_split,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        oracle_score_mode=oracle_score_mode,
        use_triton=use_triton,
        weight_compressed_by_population=weight_compressed_by_population,
        max_unselected_compressed=max_unselected_compressed,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
        outlier_budget=outlier_budget,
        outlier_detector=outlier_detector,
        cluster_outlier_N=cluster_outlier_N,
        cluster_outlier_iters=cluster_outlier_iters,
        cluster_outlier_top_k=cluster_outlier_top_k,
        cluster_outlier_q_agg=cluster_outlier_q_agg,
        cluster_outlier_scoring=cluster_outlier_scoring,
        proxy_basis_path=proxy_basis_path,
        pca_cs_h=pca_cs_h,
        fasa_n_tip=fasa_n_tip,
    )
    if proxy_basis_path:
        load_proxy_basis(proxy_basis_path)
        print(f"  Loaded proxy basis from {proxy_basis_path}")

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config (Qwen3):")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  num_sink_pages={num_sink_pages}, num_recent_pages={num_recent_pages}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}, score_softmax={score_softmax}")
    print(f"  unselected_mode={unselected_mode}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}, "
        f"weight_compressed_by_population={weight_compressed_by_population}, "
        f"max_unselected_compressed={max_unselected_compressed}, "
        f"comp_kv_quant={comp_kv_quant}, "
        f"comp_kv_quant_granularity={comp_kv_quant_granularity}, "
        f"attention_backend={attention_backend}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = _select_dct_forward(attention_backend)


def replace_llama_attn(
    page_size=32,
    top_k=64,
    num_sink_pages=0,
    num_recent_pages=0,
    compress_ratio=0.03125,
    proxy_method="dct",
    haar_detail_per_block=0,
    haar_detail_with_negation=False,
    harp_detail_topk=4,
    min_decode_kv_len_for_paging=8192,
    dense_first_n_layers=0,
    scoring_method="max",
    group_agg_method="mean",
    score_softmax=False,
    qaware_lastq_topbins=0,
    qaware_lastq_window=1,
    unselected_mode="drop",
    compressed_token_rope="mixed",
    continuous_rope=False,
    score_use_quest_minmax=False,
    score_combine_quest_dct=False,
    km_quest_split=0,
    select_with_oracle_page_scores=False,
    oracle_score_mode="max",
    use_triton=True,
    weight_compressed_by_population=False,
    max_unselected_compressed=-1,
    comp_kv_quant="none",
    comp_kv_quant_granularity="per_page",
    outlier_budget=0,
    outlier_detector="lastq_mean",
    cluster_outlier_N=256,
    cluster_outlier_iters=5,
    cluster_outlier_top_k=8,
    cluster_outlier_q_agg="mean",
    cluster_outlier_scoring="centroid",
    attention_backend="sdpa",
    proxy_basis_path: str = "",
    pca_cs_h: int = 0,
    fasa_n_tip: int = 0,
):
    """
    Replace LlamaAttention.forward with DCT Page Attention.

    Must be called BEFORE loading the model.
    LlamaAttention has the same forward signature and attributes as Qwen2Attention,
    so we reuse dct_page_attention_forward directly.

    `attention_backend="flashinfer"` patches `dct_page_attention_forward_flashinfer`
    instead. Requires an FI cache to be assigned to
    `_flashinfer_cache_ref[0]` post-prefill (driver/harness responsibility).
    """
    global _dct_page_cfg
    _dct_page_cfg = DCTPageConfig(
        page_size=page_size,
        top_k=top_k,
        num_sink_pages=num_sink_pages,
        num_recent_pages=num_recent_pages,
        compress_ratio=compress_ratio,
        proxy_method=proxy_method,
        haar_detail_per_block=haar_detail_per_block,
        haar_detail_with_negation=haar_detail_with_negation,
        harp_detail_topk=harp_detail_topk,
        min_decode_kv_len_for_paging=min_decode_kv_len_for_paging,
        dense_first_n_layers=dense_first_n_layers,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        score_softmax=score_softmax,
        qaware_lastq_topbins=qaware_lastq_topbins,
        qaware_lastq_window=qaware_lastq_window,
        unselected_mode=unselected_mode,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
        score_use_quest_minmax=score_use_quest_minmax,
        score_combine_quest_dct=score_combine_quest_dct,
        km_quest_split=km_quest_split,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        oracle_score_mode=oracle_score_mode,
        use_triton=use_triton,
        weight_compressed_by_population=weight_compressed_by_population,
        max_unselected_compressed=max_unselected_compressed,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
        outlier_budget=outlier_budget,
        outlier_detector=outlier_detector,
        cluster_outlier_N=cluster_outlier_N,
        cluster_outlier_iters=cluster_outlier_iters,
        cluster_outlier_top_k=cluster_outlier_top_k,
        cluster_outlier_q_agg=cluster_outlier_q_agg,
        cluster_outlier_scoring=cluster_outlier_scoring,
        proxy_basis_path=proxy_basis_path,
        pca_cs_h=pca_cs_h,
        fasa_n_tip=fasa_n_tip,
    )
    if proxy_basis_path:
        load_proxy_basis(proxy_basis_path)
        print(f"  Loaded proxy basis from {proxy_basis_path}")

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config (Llama):")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  num_sink_pages={num_sink_pages}, num_recent_pages={num_recent_pages}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}, score_softmax={score_softmax}")
    print(f"  unselected_mode={unselected_mode}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}, "
        f"weight_compressed_by_population={weight_compressed_by_population}, "
        f"max_unselected_compressed={max_unselected_compressed}, "
        f"comp_kv_quant={comp_kv_quant}, "
        f"comp_kv_quant_granularity={comp_kv_quant_granularity}, "
        f"attention_backend={attention_backend}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.llama.modeling_llama.LlamaAttention.forward = _select_dct_forward(attention_backend)
