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
    build_assemble_drop_stride_cache,
    topk_sort_triton, 
    assemble_kv_drop_triton, 
    score_pages_triton,
    apply_rope_q_direct,
)

# ---------------------------------------------------------------------------
# Pre-allocated KV cache (avoids torch.cat during decode, fixes strides)
# ---------------------------------------------------------------------------
class PreAllocatedLayer(DynamicLayer):
    """Drop-in replacement for DynamicLayer that uses pre-allocated buffers.

    Instead of torch.cat (O(seq_len) alloc+copy per step), uses index
    assignment into a pre-allocated buffer (O(1) write per step).
    Strides remain fixed across all decode steps.
    """

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


def dct_project_page(x, compressed_len):
    """
    Project a KV tensor to leading DCT coefficients along the sequence dimension.

    Args:
        x: [bsz, num_heads, seq_len, head_dim], seq_len: sequence length per page
        compressed_len: number of low-frequency coefficients to keep

    Returns:
        [bsz, num_heads, min(compressed_len, seq_len), head_dim]
    """
    bsz, num_heads, seq_len, head_dim = x.shape
    keep_len = min(compressed_len, seq_len)

    x_merged = x.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    x_dct = dct(x_merged.transpose(1, 2), norm='ortho')
    x_dct = x_dct[:, :, :keep_len].transpose(1, 2)
    spectral = x_dct.to(x.dtype)
    return spectral.reshape(bsz, keep_len, num_heads, head_dim).transpose(1, 2)


def _normalize_frequency_indices(indices, seq_len, comp_size):
    """Deduplicate/clamp indices while preserving order and filling missing slots."""
    seen = set()
    normalized = []
    for idx in indices:
        idx = max(0, min(int(idx), seq_len - 1))
        if idx in seen:
            continue
        normalized.append(idx)
        seen.add(idx)
        if len(normalized) == comp_size:
            return normalized

    for idx in range(seq_len):
        if idx in seen:
            continue
        normalized.append(idx)
        if len(normalized) == comp_size:
            break
    return normalized


def _resolve_frequency_keep_indices(seq_len, comp_size, layout):
    """Choose which DCT frequencies to retain for proxy construction."""
    if comp_size >= seq_len or layout == "low":
        return list(range(min(comp_size, seq_len)))

    if layout == "low_high":
        low_count = (comp_size + 1) // 2
        high_count = comp_size - low_count
        indices = list(range(low_count)) + list(range(seq_len - high_count, seq_len))
        return _normalize_frequency_indices(indices, seq_len, comp_size)

    if layout == "low_mid_high":
        low_count = min(2, comp_size)
        remaining = comp_size - low_count
        indices = list(range(low_count))
        if remaining > 0:
            tail = np.linspace(seq_len // 2, seq_len - 1, num=remaining, dtype=int).tolist()
            indices.extend(tail)
        return _normalize_frequency_indices(indices, seq_len, comp_size)

    if layout == "spread":
        indices = np.linspace(0, seq_len - 1, num=comp_size, dtype=int).tolist()
        return _normalize_frequency_indices(indices, seq_len, comp_size)

    raise ValueError(f"Unsupported proxy_frequency_layout: {layout}")


def _parse_dc_ac_scoring_method(scoring_method: str):
    """Parse DC+AC scoring method string.

    Returns (layout, lambda_val, full_spectrum) or None if not a DC+AC method.
      "proxy_dc_ac_0.5"   -> ("low", 0.5, False)
      "spread_dc_ac_1.0"  -> ("spread", 1.0, False)
      "dc_ac_1.0"         -> ("low", 1.0, True)   # full page_size DCT coefficients
    """
    for prefix, layout, full in (
        ("proxy_dc_ac_", "low", False),
        ("spread_dc_ac_", "spread", False),
        ("dc_ac_", "low", True),
    ):
        if scoring_method.startswith(prefix):
            return layout, float(scoring_method[len(prefix):]), full
    return None


def _parse_hybrid_multi_scoring_method(scoring_method: str):
    """Parse hybrid_multi scoring method string.

    Returns (M, alpha) or None if not a hybrid_multi method.
      "hybrid_multi2_ac_max_a0.5" -> (2, 0.5)
      "hybrid_multi4_ac_max_a1.0" -> (4, 1.0)
    """
    import re
    m = re.match(r'^hybrid_multi(\d+)_ac_max_a([\d.]+)$', scoring_method)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def dct_compress_page_with_indices(x, freq_indices):
    """Compress a page by selecting specific DCT frequencies before IDCT."""
    comp_size = len(freq_indices)
    if comp_size >= x.shape[2] and list(freq_indices) == list(range(x.shape[2])):
        return x

    bsz, num_heads, seq_len, head_dim = x.shape
    x_merged = x.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    x_dct = dct(x_merged.transpose(1, 2), norm='ortho')
    index_tensor = torch.tensor(freq_indices, device=x.device, dtype=torch.long)
    x_dct = x_dct.index_select(2, index_tensor)
    x_idct = idct(x_dct, norm='ortho').transpose(1, 2) * math.sqrt(comp_size / seq_len)
    compressed = x_idct.to(x.dtype)
    return compressed.reshape(bsz, comp_size, num_heads, head_dim).transpose(1, 2)


# ---------------------------------------------------------------------------
# DCT Projection Matrix (replaces FFT with a single matmul)
# ---------------------------------------------------------------------------
def _build_dct_projection_matrix(page_size, comp_size, device, dtype, layout):
    """Precompute the [comp_size, page_size] projection matrix.

    The full DCT compression pipeline (DCT → truncate → IDCT → energy
    correction) is a linear transform.  We compute it by running the
    existing dct_compress_page on an identity matrix.
    """
    I = torch.eye(page_size, device=device, dtype=torch.float32)
    I = I.unsqueeze(0).unsqueeze(0)  # [1, 1, page_size, page_size]
    freq_indices = _resolve_frequency_keep_indices(page_size, comp_size, layout)
    M = dct_compress_page_with_indices(I, freq_indices)  # [1, 1, comp_size, page_size]
    return M.squeeze(0).squeeze(0).to(dtype)  # [comp_size, page_size]


def _build_dct_spectral_projection_matrix(page_size, comp_size, device, dtype, layout):
    """Precompute the [comp_size, page_size] DCT basis for direct spectral proxies."""
    I = torch.eye(page_size, device=device, dtype=torch.float32)
    I = I.unsqueeze(0).unsqueeze(0)  # [1, 1, page_size, page_size]
    freq_indices = _resolve_frequency_keep_indices(page_size, comp_size, layout)
    M = dct_project_page(I, page_size)
    index_tensor = torch.tensor(freq_indices, device=device, dtype=torch.long)
    M = M.index_select(2, index_tensor)  # [1, 1, comp_size, page_size]
    return M.squeeze(0).squeeze(0).to(dtype)  # [comp_size, page_size]


def _build_haar_lowpass_projection_matrix(page_size, comp_size, device, dtype):
    """Precompute a coarse Haar-style lowpass projection matrix.

    Each proxy is the mean over an evenly partitioned contiguous block. For
    `page_size=128, comp_size=4`, this yields four 32-token lowpass proxies.
    """
    M = torch.zeros(comp_size, page_size, device=device, dtype=torch.float32)
    for idx in range(comp_size):
        start = (idx * page_size) // comp_size
        end = ((idx + 1) * page_size) // comp_size
        if end <= start:
            end = min(page_size, start + 1)
        M[idx, start:end] = 1.0 / float(end - start)
    return M.to(dtype)


def _build_haar_mixed_supports(page_size, comp_size):
    """Return support intervals for global-plus-detail Haar proxies."""
    supports = [(0, page_size)]
    if comp_size <= 1:
        return supports

    intervals = [(0, page_size)]
    queue_idx = 0
    while len(supports) < comp_size and queue_idx < len(intervals):
        start, end = intervals[queue_idx]
        queue_idx += 1
        if end - start < 2:
            continue

        mid = (start + end) // 2
        if mid <= start or mid >= end:
            continue

        supports.append((start, end))
        intervals.append((start, mid))
        intervals.append((mid, end))

    fill_idx = 0
    while len(supports) < comp_size:
        start = (fill_idx * page_size) // comp_size
        end = max(((fill_idx + 1) * page_size) // comp_size, start + 1)
        supports.append((start, min(page_size, end)))
        fill_idx += 1

    return supports


def _build_haar_mixed_projection_matrix(page_size, comp_size, device, dtype):
    """Precompute global-mean plus coarse-detail Haar proxies.

    Row 0 is a full-page mean. Remaining rows follow a breadth-first Haar
    detail decomposition, using half-differences over each interval.
    """
    supports = _build_haar_mixed_supports(page_size, comp_size)
    M = torch.zeros(comp_size, page_size, device=device, dtype=torch.float32)

    for idx, (start, end) in enumerate(supports):
        if idx == 0:
            M[idx, start:end] = 1.0 / float(max(end - start, 1))
            continue

        mid = (start + end) // 2
        if mid <= start or mid >= end:
            M[idx, start:end] = 1.0 / float(max(end - start, 1))
            continue

        left_len = mid - start
        right_len = end - mid
        # Keep the proxy scale comparable to block means.
        M[idx, start:mid] = 0.5 / float(left_len)
        M[idx, mid:end] = -0.5 / float(right_len)

    return M.to(dtype)


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _build_walsh_hadamard_matrix(size, device, dtype):
    """Build an orthonormal Walsh-Hadamard matrix in sequency order."""
    if not _is_power_of_two(size):
        raise ValueError(f"Hadamard proxy requires power-of-two size, got {size}")

    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < size:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    H = H / math.sqrt(size)
    sign_changes = (H[:, 1:] * H[:, :-1] < 0).sum(dim=1)
    perm = torch.argsort(sign_changes, stable=True)
    return H.index_select(0, perm).to(dtype)


def _build_hadamard_projection_matrix(page_size, comp_size, device, dtype):
    """Precompute the [comp_size, page_size] Walsh-Hadamard projection matrix."""
    if not _is_power_of_two(page_size) or not _is_power_of_two(comp_size):
        raise ValueError(
            f"Hadamard proxy requires power-of-two page_size/comp_size, got "
            f"{page_size}/{comp_size}"
        )
    H_page = _build_walsh_hadamard_matrix(page_size, device, torch.float32)
    H_comp = _build_walsh_hadamard_matrix(comp_size, device, torch.float32)
    M = math.sqrt(comp_size / page_size) * torch.matmul(H_comp.transpose(0, 1), H_page[:comp_size, :])
    return M.to(dtype)


def _get_or_build_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached projection matrix, building it on first call."""
    M = getattr(attn_module, '_dct_proj_matrix', None)
    layout = _dct_page_cfg.proxy_frequency_layout
    cached_layout = getattr(attn_module, '_dct_proj_matrix_layout', None)
    if (
        M is None
        or M.shape != (comp_size, page_size)
        or M.device != device
        or cached_layout != layout
    ):
        M = _build_dct_projection_matrix(page_size, comp_size, device, dtype, layout)
        attn_module._dct_proj_matrix = M
        attn_module._dct_proj_matrix_layout = layout
    return M


def _get_or_build_spectral_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached DCT basis matrix for direct spectral score proxies."""
    M = getattr(attn_module, '_dct_spectral_proj_matrix', None)
    layout = _dct_page_cfg.proxy_frequency_layout
    cached_layout = getattr(attn_module, '_dct_spectral_proj_matrix_layout', None)
    if (
        M is None
        or M.shape != (comp_size, page_size)
        or M.device != device
        or cached_layout != layout
    ):
        M = _build_dct_spectral_projection_matrix(page_size, comp_size, device, dtype, layout)
        attn_module._dct_spectral_proj_matrix = M
        attn_module._dct_spectral_proj_matrix_layout = layout
    return M


def _get_or_build_haar_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached Haar lowpass projection matrix."""
    M = getattr(attn_module, '_dct_haar_proj_matrix', None)
    if M is None or M.shape != (comp_size, page_size) or M.device != device:
        M = _build_haar_lowpass_projection_matrix(page_size, comp_size, device, dtype)
        attn_module._dct_haar_proj_matrix = M
    return M


def _get_or_build_haar_mixed_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached mixed Haar projection matrix."""
    M = getattr(attn_module, '_dct_haar_mixed_proj_matrix', None)
    if M is None or M.shape != (comp_size, page_size) or M.device != device:
        M = _build_haar_mixed_projection_matrix(page_size, comp_size, device, dtype)
        attn_module._dct_haar_mixed_proj_matrix = M
    return M


def _get_or_build_hadamard_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached Hadamard projection matrix."""
    M = getattr(attn_module, '_dct_hadamard_proj_matrix', None)
    if M is None or M.shape != (comp_size, page_size) or M.device != device:
        M = _build_hadamard_projection_matrix(page_size, comp_size, device, dtype)
        attn_module._dct_hadamard_proj_matrix = M
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

    pageable_len = kv_len - cfg.sink_size - cfg.recent_size
    num_pages = pageable_len // cfg.page_size
    leftover = pageable_len % cfg.page_size
    actual_recent = cfg.recent_size + leftover

    sink_k = key_states[:, :, :cfg.sink_size]
    sink_v = value_states[:, :, :cfg.sink_size]

    pages_end = cfg.sink_size + num_pages * cfg.page_size
    paged_k = key_states[:, :, cfg.sink_size:pages_end].view(
        bsz, num_kv_heads, num_pages, cfg.page_size, head_dim
    )
    paged_v = value_states[:, :, cfg.sink_size:pages_end].view(
        bsz, num_kv_heads, num_pages, cfg.page_size, head_dim
    )

    recent_k = key_states[:, :, pages_end:]
    recent_v = value_states[:, :, pages_end:]

    return sink_k, sink_v, paged_k, paged_v, recent_k, recent_v, num_pages, actual_recent


# ---------------------------------------------------------------------------
# Incremental compressed page cache
# ---------------------------------------------------------------------------
def _update_comp_cache(attn_module, paged_k, paged_v, num_pages, comp_size, cfg):
    """
    Incrementally maintain compressed page representations (both K and V).

    Dispatches to the compression method specified in cfg.compression_method:
      - "dct": DCT-IDCT projection (via _compress_pages)
      - "haar": Haar lowpass block means (via _project_pages_to_haar_lowpass)

    RoPE handling for compressed K (cfg.compressed_token_rope):
      - "mixed":        compress post-RoPE keys directly (current behavior — mixed RoPE phases)
      - "block_center": invert RoPE on raw page → compress → re-rotate at block-center positions
                        (mirrors multipole's invert/cluster/wRoPE flow but with block-center positions)

    Values are unaffected by RoPE — always compressed as-is.
    """
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_comp_k_cache', None)
    cached_v = getattr(attn_module, '_comp_v_cache', None)
    n_cached = getattr(attn_module, '_comp_n_pages_cached', 0)
    capacity = getattr(attn_module, '_comp_cache_capacity', 0)
    cached_strategy = getattr(attn_module, '_comp_cache_strategy', None)
    cur_strategy = (cfg.compression_method, cfg.compressed_token_rope)

    # Invalidate cache when the sequence restarts, shape changes, or RoPE/compression strategy changes
    if (cached_k is None
            or cached_v is None
            or num_pages < n_cached
            or cached_k.shape[0] != bsz
            or cached_k.shape[3] != comp_size
            or cached_strategy != cur_strategy):
        attn_module._comp_k_cache = None
        attn_module._comp_v_cache = None
        attn_module._comp_n_pages_cached = 0
        attn_module._comp_cache_capacity = 0
        n_cached = 0
        capacity = 0
    attn_module._comp_cache_strategy = cur_strategy

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        new_v = paged_v[:, :, n_cached:num_pages]

        # Step A: optionally invert RoPE on new_k to recover raw (un-roped) keys.
        # When continuous_rope=False (current default), the cache stores post-RoPE keys.
        #
        # The forward RoPE applies (alpha * R_theta) to k, where alpha = attention_scaling
        # (alpha != 1 for YaRN and similar scaled RoPE types). To invert (alpha * R_theta * k)
        # and recover k, we apply (1/alpha) * R_{-theta}, which means dividing the rotation
        # matrix entries by alpha. Since `_compute_rope_cos_sin` already returns cos/sin
        # pre-multiplied by alpha, we must divide the returned values by alpha**2 (one alpha
        # to remove the forward scaling, another alpha to apply the 1/alpha inverse scaling).
        if cfg.compressed_token_rope == "block_center":
            start_pos = cfg.sink_size + n_cached * page_size
            end_pos = cfg.sink_size + num_pages * page_size
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
            # "mixed": compress post-RoPE keys directly (no inversion)
            new_k_for_compress = new_k

        # Step B: compress K and V with the configured method
        if cfg.compression_method == "haar":
            new_comp_k = _project_pages_to_haar_lowpass(attn_module, new_k_for_compress, comp_size)
            new_comp_v = _project_pages_to_haar_lowpass(attn_module, new_v, comp_size)
        else:  # "dct"
            new_comp_k = _compress_pages(attn_module, new_k_for_compress, comp_size)
            new_comp_v = _compress_pages(attn_module, new_v, comp_size)

        # Step B': fake-quantize compressed K and V to simulate low-precision storage.
        # No-op when cfg.comp_kv_quant == "none". Runs before block_center RoPE so the
        # re-rotation applies to the quantized values (matches real fp8-store semantics).
        if cfg.comp_kv_quant != "none":
            new_comp_k = _fake_quantize_comp(
                new_comp_k, cfg.comp_kv_quant, cfg.comp_kv_quant_granularity
            )
            new_comp_v = _fake_quantize_comp(
                new_comp_v, cfg.comp_kv_quant, cfg.comp_kv_quant_granularity
            )

        # Step C: re-apply RoPE to compressed K at block-center positions
        if cfg.compressed_token_rope == "block_center":
            new_positions = _block_center_positions(
                n_cached, n_new, cfg.page_size, comp_size, cfg.sink_size, new_comp_k.device,
            ).reshape(-1)
            cos_new, sin_new = _compute_rope_cos_sin(
                new_positions, attn_module.config, new_comp_k.device, new_comp_k.dtype
            )
            flat_comp_k = new_comp_k.reshape(bsz, num_kv_heads, n_new * comp_size, head_dim)
            flat_comp_k = _apply_rope(flat_comp_k, cos_new, sin_new)
            new_comp_k = flat_comp_k.reshape(bsz, num_kv_heads, n_new, comp_size, head_dim)
        # else "mixed": leave new_comp_k as-is

        if num_pages > capacity:
            new_capacity = _next_page_capacity(num_pages, capacity)
            new_k_cache = torch.empty(
                bsz, num_kv_heads, new_capacity, comp_size, head_dim,
                dtype=new_comp_k.dtype, device=new_comp_k.device,
            )
            new_v_cache = torch.empty_like(new_k_cache)
            if n_cached > 0 and attn_module._comp_k_cache is not None:
                new_k_cache[:, :, :n_cached].copy_(attn_module._comp_k_cache[:, :, :n_cached])
                new_v_cache[:, :, :n_cached].copy_(attn_module._comp_v_cache[:, :, :n_cached])
            attn_module._comp_k_cache = new_k_cache
            attn_module._comp_v_cache = new_v_cache
            attn_module._comp_cache_capacity = new_capacity

        attn_module._comp_k_cache[:, :, n_cached:num_pages].copy_(new_comp_k)
        attn_module._comp_v_cache[:, :, n_cached:num_pages].copy_(new_comp_v)
        attn_module._comp_n_pages_cached = num_pages

    comp_k = attn_module._comp_k_cache
    comp_v = attn_module._comp_v_cache
    if comp_k is None:
        return None, None
    return comp_k[:, :, :num_pages], comp_v[:, :, :num_pages]


_FP8_MAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0}
_FP8_DTYPE = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}


def _fake_quantize_comp(x: torch.Tensor, quant_type: str, granularity: str) -> torch.Tensor:
    """Quantize→dequantize round-trip for compressed K/V.

    Returns a tensor with the same shape and dtype as the input, but with
    precision loss baked in. Used to study whether storing compressed KV in
    low precision would degrade page selection quality.

    x shape: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
    granularity:
      per_page:       one absmax scale per (bsz, kv_head, page)
      per_comp_token: one absmax scale per (bsz, kv_head, page, comp_idx)
    """
    if quant_type == "none":
        return x

    if granularity == "per_page":
        reduce_dims = (-2, -1)
    elif granularity == "per_comp_token":
        reduce_dims = (-1,)
    else:
        raise ValueError(f"Unsupported comp_kv_quant_granularity: {granularity}")

    orig_dtype = x.dtype
    x_fp = x.to(torch.float32)
    abs_max = x_fp.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)

    if quant_type in ("fp8_e4m3", "fp8_e5m2"):
        fp8_max = _FP8_MAX[quant_type]
        fp8_dtype = _FP8_DTYPE[quant_type]
        scale = abs_max / fp8_max
        x_q = (x_fp / scale).to(fp8_dtype)
        return (x_q.to(torch.float32) * scale).to(orig_dtype)

    if quant_type == "int8":
        scale = abs_max / 127.0
        x_q = torch.round(x_fp / scale).clamp(-128.0, 127.0)
        return (x_q * scale).to(orig_dtype)

    if quant_type == "int4":
        scale = abs_max / 7.0
        x_q = torch.round(x_fp / scale).clamp(-8.0, 7.0)
        return (x_q * scale).to(orig_dtype)

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


def _score_pages_quest(query, min_k, max_k, group_agg_method, num_kv_groups, out=None):
    """QUEST-style page scoring: score = sum_d max(q_d * max_d, q_d * min_d).

    Args:
        query: [bsz, num_heads, 1, head_dim]
        min_k: [bsz, num_kv_heads, num_pages, head_dim]
        max_k: [bsz, num_kv_heads, num_pages, head_dim]
        group_agg_method: "mean" | "max" | "topp"
        num_kv_groups: number of GQA groups (num_heads // num_kv_heads)
        out: optional pre-allocated [bsz, num_kv_heads, num_pages] buffer

    Returns:
        page_scores: [bsz, num_kv_heads, num_pages]
    """
    bsz, num_heads, _, head_dim = query.shape
    num_kv_heads = min_k.shape[1]
    num_pages = min_k.shape[2]

    # Reshape query for GQA: [bsz, num_kv_heads, G, head_dim]
    q = query.squeeze(2).float()
    q = q.reshape(bsz, num_kv_heads, num_kv_groups, head_dim)

    # QUEST scoring per kv-head, per GQA group
    q_max = torch.einsum('bhgd,bhpd->bhgp', q, max_k.float())   # [B, kv_heads, G, P]
    q_min = torch.einsum('bhgd,bhpd->bhgp', q, min_k.float())   # [B, kv_heads, G, P]
    page_scores = torch.maximum(q_max, q_min)                     # [B, kv_heads, G, P]

    # Group aggregation (matches existing score_pages_triton logic)
    if group_agg_method == "mean":
        page_scores = page_scores.mean(dim=2)                     # [B, kv_heads, P]
    elif group_agg_method == "max":
        page_scores = page_scores.max(dim=2).values               # [B, kv_heads, P]
    elif group_agg_method == "topp":
        k_top = min(2, num_kv_groups)
        page_scores = page_scores.topk(k_top, dim=2).values.mean(dim=2)
    else:
        raise ValueError(f"Unsupported group_agg_method: {group_agg_method}")

    if out is not None:
        out[:, :, :num_pages].copy_(page_scores)
        return out[:, :, :num_pages]
    return page_scores


def _compress_pages(attn_module, paged_x, comp_size):
    """Project [bsz, kv_heads, num_pages, page_size, head_dim] pages to comp_size via DCT."""
    page_size = paged_x.shape[3]
    M = _get_or_build_projection_matrix(
        attn_module, page_size, comp_size, paged_x.device, paged_x.dtype
    )
    return torch.einsum('cs,bhnsd->bhncd', M, paged_x)


def _project_pages_to_spectral(attn_module, paged_x, comp_size):
    """Project pages to their leading DCT coefficients without IDCT reconstruction."""
    page_size = paged_x.shape[3]
    M = _get_or_build_spectral_projection_matrix(
        attn_module, page_size, comp_size, paged_x.device, paged_x.dtype
    )
    return torch.einsum('cs,bhnsd->bhncd', M, paged_x)


def _project_pages_to_haar_lowpass(attn_module, paged_x, comp_size):
    """Project pages to coarse Haar lowpass block means."""
    page_size = paged_x.shape[3]
    if comp_size == 1:
        return paged_x.mean(dim=3, keepdim=True)
    # The Haar lowpass projection is exactly a contiguous block mean. When the
    # page splits evenly into `comp_size` blocks (the default fast paths like
    # 32->4 and 128->4), avoid the projection-matrix einsum entirely.
    if page_size % comp_size == 0:
        block_size = page_size // comp_size
        return paged_x.reshape(*paged_x.shape[:3], comp_size, block_size, paged_x.shape[4]).mean(dim=4)
    M = _get_or_build_haar_projection_matrix(
        attn_module, page_size, comp_size, paged_x.device, paged_x.dtype
    )
    return torch.einsum('cs,bhnsd->bhncd', M, paged_x)

def _project_pages_to_haar_mixed(attn_module, paged_x, comp_size):
    """Project pages to mixed global/detail Haar proxies."""
    page_size = paged_x.shape[3]
    M = _get_or_build_haar_mixed_projection_matrix(
        attn_module, page_size, comp_size, paged_x.device, paged_x.dtype
    )
    return torch.einsum('cs,bhnsd->bhncd', M, paged_x)


def _project_pages_to_hadamard(attn_module, paged_x, comp_size):
    """Project pages to Walsh-Hadamard compressed proxies."""
    page_size = paged_x.shape[3]
    M = _get_or_build_hadamard_projection_matrix(
        attn_module, page_size, comp_size, paged_x.device, paged_x.dtype
    )
    return torch.einsum('cs,bhnsd->bhncd', M, paged_x)


def _get_or_build_dc_ac_spectral_matrix(attn_module, page_size, comp_size, device, dtype, layout):
    """Return cached DCT basis matrix for DC+AC scoring with explicit layout."""
    M = getattr(attn_module, '_dc_ac_spectral_proj_matrix', None)
    cached_layout = getattr(attn_module, '_dc_ac_spectral_proj_layout', None)
    if (
        M is None
        or M.shape != (comp_size, page_size)
        or M.device != device
        or cached_layout != layout
    ):
        M = _build_dct_spectral_projection_matrix(page_size, comp_size, device, dtype, layout)
        attn_module._dc_ac_spectral_proj_matrix = M
        attn_module._dc_ac_spectral_proj_layout = layout
    return M


def _update_spectral_score_cache(attn_module, paged_k, num_pages, comp_size, cfg, layout):
    """Incrementally maintain spectral (raw DCT) compressed keys for DC+AC scoring.

    Like _update_comp_cache but only builds K (no V needed for scoring) and uses
    spectral projection (raw DCT coefficients, no IDCT) with an explicit layout.
    """
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_spectral_score_k_cache', None)
    n_cached = getattr(attn_module, '_spectral_score_n_cached', 0)
    capacity = getattr(attn_module, '_spectral_score_capacity', 0)
    cached_layout = getattr(attn_module, '_spectral_score_layout', None)

    if (cached_k is None
            or num_pages < n_cached
            or cached_k.shape[0] != bsz
            or cached_k.shape[3] != comp_size
            or cached_layout != (layout, cfg.compressed_token_rope)):
        attn_module._spectral_score_k_cache = None
        attn_module._spectral_score_n_cached = 0
        attn_module._spectral_score_capacity = 0
        n_cached = 0
        capacity = 0
    attn_module._spectral_score_layout = (layout, cfg.compressed_token_rope)

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]

        # Step A: optionally invert RoPE (same as _update_comp_cache)
        if cfg.compressed_token_rope == "block_center":
            start_pos = cfg.sink_size + n_cached * page_size
            end_pos = cfg.sink_size + num_pages * page_size
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

        # Step B: spectral projection with explicit layout
        M = _get_or_build_dc_ac_spectral_matrix(
            attn_module, page_size, comp_size, new_k_for_compress.device,
            new_k_for_compress.dtype, layout,
        )
        new_comp_k = torch.einsum('cs,bhnsd->bhncd', M, new_k_for_compress)

        # Step C: re-apply RoPE at block-center positions
        if cfg.compressed_token_rope == "block_center":
            new_positions = _block_center_positions(
                n_cached, n_new, cfg.page_size, comp_size, cfg.sink_size, new_comp_k.device,
            ).reshape(-1)
            cos_new, sin_new = _compute_rope_cos_sin(
                new_positions, attn_module.config, new_comp_k.device, new_comp_k.dtype
            )
            flat_comp_k = new_comp_k.reshape(bsz, num_kv_heads, n_new * comp_size, head_dim)
            flat_comp_k = _apply_rope(flat_comp_k, cos_new, sin_new)
            new_comp_k = flat_comp_k.reshape(bsz, num_kv_heads, n_new, comp_size, head_dim)

        if num_pages > capacity:
            new_capacity = _next_page_capacity(num_pages, capacity)
            new_k_cache = torch.empty(
                bsz, num_kv_heads, new_capacity, comp_size, head_dim,
                dtype=new_comp_k.dtype, device=new_comp_k.device,
            )
            if n_cached > 0 and attn_module._spectral_score_k_cache is not None:
                new_k_cache[:, :, :n_cached].copy_(attn_module._spectral_score_k_cache[:, :, :n_cached])
            attn_module._spectral_score_k_cache = new_k_cache
            attn_module._spectral_score_capacity = new_capacity

        attn_module._spectral_score_k_cache[:, :, n_cached:num_pages].copy_(new_comp_k)
        attn_module._spectral_score_n_cached = num_pages

    cache = attn_module._spectral_score_k_cache
    if cache is None:
        return None
    return cache[:, :, :num_pages]


def _score_pages_hybrid_multi(
    query_states,
    paged_k,
    spectral_comp_k,
    M_proj,
    M_highlight,
    alpha,
    num_kv_groups,
    group_agg_method,
    out=None,
):
    """Score pages with hybrid multi-highlight: M exact tokens + spectral reconstruction.

    Combines top-M AC-energy tokens (exact Q.K) with spectral curve reconstruction
    from the remaining DCT coefficients.

    Args:
        query_states:    [bsz, num_heads, 1, head_dim]
        paged_k:         [bsz, num_kv_heads, num_pages, page_size, head_dim]
        spectral_comp_k: [bsz, num_kv_heads, num_pages, c_multi, head_dim] or None
        M_proj:          [c_multi, page_size] DCT spectral projection matrix, or None
        M_highlight:     number of AC-energy-selected tokens per page
        alpha:           spectral scaling factor
        num_kv_groups:   GQA group count
        group_agg_method: "mean" | "max" | "topp"
        out:             optional pre-allocated [bsz, num_kv_heads, capacity] float32 buffer

    Returns:
        page_scores: [bsz, num_kv_heads, num_pages] (float32)
    """
    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    scaling = head_dim ** -0.5

    # Reshape query for GQA: [bsz, num_kv_heads, G, head_dim]
    q = query_states.squeeze(2).float()
    q = q.reshape(bsz, num_kv_heads, num_kv_groups, head_dim) * scaling

    k_f = paged_k.float()

    # --- AC energy selection: pick top-M tokens per page ---
    k_mean = k_f.mean(dim=-2, keepdim=True)                     # [B,H,N,1,D]
    ac_energy = (k_f - k_mean).pow(2).sum(dim=-1)               # [B,H,N,P]
    _, topM_idx = ac_energy.topk(M_highlight, dim=-1)            # [B,H,N,M]

    idx_exp = topM_idx.unsqueeze(-1).expand(
        bsz, num_kv_heads, num_pages, M_highlight, head_dim
    )
    K_multi = k_f.gather(dim=-2, index=idx_exp)                  # [B,H,N,M,D]

    # Exact Q.K for selected tokens
    multi_scores = torch.einsum(
        'bhgd,bhnmd->bhgnm', q, K_multi
    )                                                             # [B,H,G,N,M]
    multi_hi = multi_scores.max(dim=-1).values                   # [B,H,G,N]

    if spectral_comp_k is not None and M_proj is not None:
        # --- Spectral reconstruction from c_multi DCT coefficients ---
        a = torch.einsum(
            'bhgd,bhncd->bhgnc', q, spectral_comp_k.float()
        )                                                         # [B,H,G,N,c_multi]

        Phi_T = M_proj.T.float().contiguous()                    # [P, c_multi]
        shat = torch.einsum(
            'tc,bhgnc->bhgnt', Phi_T, a
        )                                                         # [B,H,G,N,P]
        spectral_max = shat.max(dim=-1).values                   # [B,H,G,N]

        # Combine: max(alpha * spectral_max, multi_hi)
        combined = torch.maximum(alpha * spectral_max, multi_hi)  # [B,H,G,N]
    else:
        # No DCT budget: degenerate to multi-highlight only
        combined = multi_hi                                       # [B,H,G,N]

    # Group aggregation
    if group_agg_method == "mean":
        page_scores = combined.mean(dim=2)
    elif group_agg_method == "max":
        page_scores = combined.max(dim=2).values
    elif group_agg_method == "topp":
        k_top = min(2, num_kv_groups)
        page_scores = combined.topk(k_top, dim=2).values.mean(dim=2)
    else:
        raise ValueError(f"Unsupported group_agg_method: {group_agg_method}")

    if out is not None:
        out_view = out[:, :, :num_pages]
        out_view.copy_(page_scores)
        return out_view
    return page_scores.contiguous()


def _score_pages_spectral_recon_max(
    query_states,
    spectral_comp_k,
    M_proj,
    num_kv_groups,
    group_agg_method,
    out=None,
):
    """Score pages via spectral reconstruction of per-token Q·K scores.

    Dot-products the scaled query with the (already DCT'd) compressed keys to get
    frequency-domain score coefficients, then reconstructs them back to `page_size`
    per-token scores using the IDCT-style synthesis matrix. Takes the max across
    tokens in each page, then aggregates across GQA groups.

    Args:
        query_states:    [bsz, num_heads, 1, head_dim]
        spectral_comp_k: [bsz, num_kv_heads, num_pages, comp_size, head_dim]
                         (raw DCT coefficients of keys; from _update_spectral_score_cache)
        M_proj:          [comp_size, page_size] DCT spectral projection matrix
        num_kv_groups:   GQA group count
        group_agg_method: "mean" | "max" | "topp"
        out:             optional pre-allocated [bsz, num_kv_heads, capacity] float32 buffer

    Returns:
        page_scores: [bsz, num_kv_heads, num_pages] (float32)
    """
    bsz, num_kv_heads, num_pages, comp_size, head_dim = spectral_comp_k.shape
    scaling = head_dim ** -0.5
    q = query_states.squeeze(2).float()
    q = q.reshape(bsz, num_kv_heads, num_kv_groups, head_dim) * scaling

    a = torch.einsum(
        'bhgd,bhncd->bhgnc', q, spectral_comp_k.float()
    )                                                             # [B,H,G,N,c]

    Phi_T = M_proj.T.float().contiguous()                        # [P, c]
    shat = torch.einsum('tc,bhgnc->bhgnt', Phi_T, a)             # [B,H,G,N,P]
    per_group = shat.max(dim=-1).values                          # [B,H,G,N]

    if group_agg_method == "mean":
        page_scores = per_group.mean(dim=2)
    elif group_agg_method == "max":
        page_scores = per_group.max(dim=2).values
    elif group_agg_method == "topp":
        k_top = min(2, num_kv_groups)
        page_scores = per_group.topk(k_top, dim=2).values.mean(dim=2)
    else:
        raise ValueError(f"Unsupported group_agg_method: {group_agg_method}")

    if out is not None:
        out_view = out[:, :, :num_pages]
        out_view.copy_(page_scores)
        return out_view
    return page_scores.contiguous()


def _block_center_positions(start_page_idx, n_pages, page_size, comp_size, sink_size, device):
    """Compute the center position of each compressed block within a page.

    For comp_size=4 and page_size=128, each page is split into 4 blocks of 32.
    The anchor position for each block is its center token's original position.
    Returns: [n_pages, comp_size] integer tensor of absolute positions.
    """
    page_ids = torch.arange(start_page_idx, start_page_idx + n_pages, device=device, dtype=torch.long)
    page_bases = sink_size + page_ids[:, None] * page_size
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


def _apply_original_position_rope_to_paged_k(paged_k, sink_size, model_config):
    """Apply original-position RoPE to full paged keys for debug/oracle scoring."""
    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    flat_k = paged_k.reshape(bsz, num_kv_heads, num_pages * page_size, head_dim)
    positions = torch.arange(sink_size + num_pages * page_size, device=paged_k.device)
    cos_pages, sin_pages = _compute_rope_cos_sin(
        positions, model_config, paged_k.device, paged_k.dtype
    )
    cos_pages = cos_pages[:, :, sink_size:]
    sin_pages = sin_pages[:, :, sink_size:]
    flat_k_rope = _apply_rope(flat_k, cos_pages, sin_pages)
    return flat_k_rope.reshape(bsz, num_kv_heads, num_pages, page_size, head_dim)


def _compute_debug_oracle_page_scores(attn_module, query_states, paged_k, cfg, cos, sin):
    """Compute full-page oracle scores for debug comparisons against proxies."""
    oracle_query_states = query_states
    oracle_paged_k = paged_k
    if cfg.continuous_rope:
        oracle_query_states, _ = apply_rotary_pos_emb(
            query_states, query_states, cos, sin
        )
        oracle_paged_k = _apply_original_position_rope_to_paged_k(
            paged_k, cfg.sink_size, attn_module.config
        )
    return score_pages_triton(
        oracle_query_states,
        oracle_paged_k,
        cfg.scoring_method,
        cfg.group_agg_method,
        attn_module.num_key_value_groups,
    )


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
    cos_table, sin_table = _get_or_build_original_position_rope_tables(
        attn_module,
        num_pages * cfg.page_size + cfg.sink_size + actual_recent,
        model_config,
        final_k.device,
        final_k.dtype,
    )

    cos_parts = []
    sin_parts = []

    if cfg.sink_size > 0:
        sink_cos = cos_table[:cfg.sink_size].view(1, 1, cfg.sink_size, head_dim)
        sink_sin = sin_table[:cfg.sink_size].view(1, 1, cfg.sink_size, head_dim)
        cos_parts.append(sink_cos.expand(bsz, num_kv_heads, -1, -1))
        sin_parts.append(sink_sin.expand(bsz, num_kv_heads, -1, -1))

    middle_start = cfg.sink_size
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
    "_comp_n_pages_cached",
    "_comp_cache_capacity",
    "_page_scores_buf",
    "_page_scores_np",
    "_page_scores_capacity",
    "_topk_out_buf",
    "_assemble_stride_cache",
    "_assemble_drop_stride_cache",
    "_final_k_buf",
    "_final_v_buf",
    "_final_bias_buf",
    "_sel_idx_buf",
    "_assemble_buf_len",
    "_orig_pos_rope_cos_2d",
    "_orig_pos_rope_sin_2d",
    "_orig_pos_rope_cache_len",
    "_q_rope_buf",
    "_spectral_score_k_cache",
    "_spectral_score_n_cached",
    "_spectral_score_capacity",
    "_spectral_score_layout",
    "_dc_ac_spectral_proj_matrix",
    "_dc_ac_spectral_proj_layout",
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
    num_score_proxy_modes = sum(
        int(flag)
        for flag in (
            cfg.score_use_direct_spectral_proxy,
            cfg.score_use_haar_proxy,
            cfg.score_use_haar_mixed_proxy,
            cfg.score_use_hadamard_proxy,
        )
    )
    if num_score_proxy_modes > 1:
        raise ValueError(
            "Only one score-time proxy rope mode may be enabled at once."
    )
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
        cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size,
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
        self._comp_n_pages_cached = 0

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

    # Fallback to standard attention when KV cache is too short for paging.
    if kv_len < min_len_for_paging:
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
    # Step 4: Compressed cache maintenance (K+V).
    # Always built — comp_k used for scoring, comp_v used for assembly in compressed mode.
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
    if cfg.score_use_quest_minmax:
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        page_scores = _score_pages_quest(
            score_query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
            out=self._page_scores_buf,
        )
    else:
        hybrid_multi_parsed = _parse_hybrid_multi_scoring_method(cfg.scoring_method)
        if hybrid_multi_parsed is not None:
            hm_M, hm_alpha = hybrid_multi_parsed
            c_multi = comp_size
            if c_multi > 0:
                spectral_comp_k = _update_spectral_score_cache(
                    self, paged_k, num_pages, c_multi, cfg, "low",
                )
                hm_proj = _get_or_build_dc_ac_spectral_matrix(
                    self, cfg.page_size, c_multi, paged_k.device, paged_k.dtype, "low",
                )
            else:
                spectral_comp_k = None
                hm_proj = None
            page_scores = _score_pages_hybrid_multi(
                score_query_states, paged_k[:, :, :num_pages], spectral_comp_k, hm_proj,
                hm_M, hm_alpha, self.num_key_value_groups, cfg.group_agg_method,
                out=self._page_scores_buf,
        )
        elif cfg.scoring_method == "spectral_recon_max":
            # Lowpass spectral reconstruction scoring (layout hardcoded to "low").
            if comp_size > 0:
                sr_spectral_comp_k = _update_spectral_score_cache(
                    self, paged_k, num_pages, comp_size, cfg, "low",
                )
                sr_proj = _get_or_build_dc_ac_spectral_matrix(
                    self, cfg.page_size, comp_size, paged_k.device, paged_k.dtype, "low",
                )
                page_scores = _score_pages_spectral_recon_max(
                    score_query_states,
                    sr_spectral_comp_k[:, :, :num_pages],
                    sr_proj,
                    self.num_key_value_groups,
                    cfg.group_agg_method,
                    out=self._page_scores_buf,
                    )
            else:
                page_scores = self._page_scores_buf[:, :, :num_pages].zero_()
        else:
            dc_ac_parsed = _parse_dc_ac_scoring_method(cfg.scoring_method)
            if dc_ac_parsed is not None:
                dc_ac_layout, _, dc_ac_full = dc_ac_parsed
                score_cache_size = cfg.page_size if dc_ac_full else comp_size
                score_comp_k = _update_spectral_score_cache(
                    self, paged_k, num_pages, score_cache_size, cfg, dc_ac_layout,
                )
            else:
                score_comp_k = comp_k

            page_scores = score_pages_triton(
                score_query_states, score_comp_k, cfg.scoring_method, cfg.group_agg_method, self.num_key_value_groups,
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

    selected_indices = topk_sort_triton(
        selection_page_scores, actual_top_k, out=self._topk_out_buf[:, :, :actual_top_k]
    )

    if debug_hook is not None:
        debug_hook(
            {
                "layer_idx": int(self.layer_idx),
                "kv_len": int(kv_len),
                "num_pages": int(num_pages),
                "actual_top_k": int(actual_top_k),
                "page_size": int(cfg.page_size),
                "sink_size": int(cfg.sink_size),
                "recent_size": int(cfg.recent_size),
                "proxy_frequency_layout": str(cfg.proxy_frequency_layout),
                "score_use_direct_spectral_proxy": bool(cfg.score_use_direct_spectral_proxy),
                "score_use_haar_proxy": bool(cfg.score_use_haar_proxy),
                "score_use_haar_mixed_proxy": bool(cfg.score_use_haar_mixed_proxy),
                "score_use_hadamard_proxy": bool(cfg.score_use_hadamard_proxy),

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
        assembled_len = cfg.sink_size + actual_top_k * cfg.page_size + actual_recent

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
            assembled_len = cfg.sink_size + actual_top_k * cfg.page_size + actual_recent

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
                cfg.sink_size + _ranks_sel * cfg.page_size + count_comp_before_sel * comp_size
            )
            # For each compressed page: how many selected pages come before it?
            count_full_before_comp = torch.searchsorted(_sel_long, _comp_long)
            _ranks_comp = torch.arange(effective_num_comp, device=paged_k.device).view(1, 1, -1)
            compressed_write_offsets = (
                cfg.sink_size + count_full_before_comp * cfg.page_size + _ranks_comp * comp_size
            )

            middle_len = actual_top_k * cfg.page_size + effective_num_comp * comp_size
            assembled_len = cfg.sink_size + middle_len + actual_recent

            # Population weighting
            weight_pop = (
                cfg.weight_compressed_by_population
                and not cfg.score_use_direct_spectral_proxy
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
            final_k[:, :, :cfg.sink_size, :] = sink_k
            final_v[:, :, :cfg.sink_size, :] = sink_v

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
            assembled_len = cfg.sink_size + middle_len + actual_recent

            # Population weighting
            weight_pop = (
                cfg.weight_compressed_by_population
                and not cfg.score_use_direct_spectral_proxy
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

    attn_output = F.scaled_dot_product_attention(
        query_states, final_k, final_v,
        attn_mask=attn_bias,
        is_causal=False,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    # Step 7b: Output projection
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Monkey-patch entry point
# ---------------------------------------------------------------------------
def replace_qwen2_attn(
    page_size=32,
    top_k=64,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.03125,
    min_decode_kv_len_for_paging=8192,
    proxy_frequency_layout="low",
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    compression_method="haar",
    compressed_token_rope="mixed",
    continuous_rope=False,
    score_use_direct_spectral_proxy=False,
    score_use_haar_proxy=True,
    score_use_haar_mixed_proxy=False,
    score_use_hadamard_proxy=False,
    score_use_quest_minmax=False,
    select_with_oracle_page_scores=False,
    use_triton=True,
    weight_compressed_by_population=False,
    max_unselected_compressed=-1,
    comp_kv_quant="none",
    comp_kv_quant_granularity="per_page",
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
        min_decode_kv_len_for_paging=min_decode_kv_len_for_paging,
        proxy_frequency_layout=proxy_frequency_layout,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        compression_method=compression_method,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
        score_use_direct_spectral_proxy=score_use_direct_spectral_proxy,
        score_use_haar_proxy=score_use_haar_proxy,
        score_use_haar_mixed_proxy=score_use_haar_mixed_proxy,
        score_use_hadamard_proxy=score_use_hadamard_proxy,
        score_use_quest_minmax=score_use_quest_minmax,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        use_triton=use_triton,
        weight_compressed_by_population=weight_compressed_by_population,
        max_unselected_compressed=max_unselected_compressed,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config:")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  proxy_frequency_layout={proxy_frequency_layout}")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, compression_method={compression_method}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"score_use_direct_spectral_proxy={score_use_direct_spectral_proxy}, "
        f"score_use_haar_proxy={score_use_haar_proxy}, "
        f"score_use_haar_mixed_proxy={score_use_haar_mixed_proxy}, "
        f"score_use_hadamard_proxy={score_use_hadamard_proxy}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}, "
        f"weight_compressed_by_population={weight_compressed_by_population}, "
        f"max_unselected_compressed={max_unselected_compressed}, "
        f"comp_kv_quant={comp_kv_quant}, "
        f"comp_kv_quant_granularity={comp_kv_quant_granularity}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = dct_page_attention_forward


def replace_qwen3_attn(
    page_size=32,
    top_k=64,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.03125,
    min_decode_kv_len_for_paging=8192,
    proxy_frequency_layout="low",
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    compression_method="haar",
    compressed_token_rope="mixed",
    continuous_rope=False,
    score_use_direct_spectral_proxy=False,
    score_use_haar_proxy=True,
    score_use_haar_mixed_proxy=False,
    score_use_hadamard_proxy=False,
    score_use_quest_minmax=False,
    select_with_oracle_page_scores=False,
    use_triton=True,
    weight_compressed_by_population=False,
    max_unselected_compressed=-1,
    comp_kv_quant="none",
    comp_kv_quant_granularity="per_page",
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
        sink_size=sink_size,
        recent_size=recent_size,
        compress_ratio=compress_ratio,
        min_decode_kv_len_for_paging=min_decode_kv_len_for_paging,
        proxy_frequency_layout=proxy_frequency_layout,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        compression_method=compression_method,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
        score_use_direct_spectral_proxy=score_use_direct_spectral_proxy,
        score_use_haar_proxy=score_use_haar_proxy,
        score_use_haar_mixed_proxy=score_use_haar_mixed_proxy,
        score_use_hadamard_proxy=score_use_hadamard_proxy,
        score_use_quest_minmax=score_use_quest_minmax,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        use_triton=use_triton,
        weight_compressed_by_population=weight_compressed_by_population,
        max_unselected_compressed=max_unselected_compressed,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config (Qwen3):")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  proxy_frequency_layout={proxy_frequency_layout}")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, compression_method={compression_method}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"score_use_direct_spectral_proxy={score_use_direct_spectral_proxy}, "
        f"score_use_haar_proxy={score_use_haar_proxy}, "
        f"score_use_haar_mixed_proxy={score_use_haar_mixed_proxy}, "
        f"score_use_hadamard_proxy={score_use_hadamard_proxy}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}, "
        f"weight_compressed_by_population={weight_compressed_by_population}, "
        f"max_unselected_compressed={max_unselected_compressed}, "
        f"comp_kv_quant={comp_kv_quant}, "
        f"comp_kv_quant_granularity={comp_kv_quant_granularity}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = dct_page_attention_forward


def replace_llama_attn(
    page_size=32,
    top_k=64,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.03125,
    min_decode_kv_len_for_paging=8192,
    proxy_frequency_layout="low",
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    compression_method="haar",
    compressed_token_rope="mixed",
    continuous_rope=False,
    score_use_direct_spectral_proxy=False,
    score_use_haar_proxy=True,
    score_use_haar_mixed_proxy=False,
    score_use_hadamard_proxy=False,
    score_use_quest_minmax=False,
    select_with_oracle_page_scores=False,
    use_triton=True,
    weight_compressed_by_population=False,
    max_unselected_compressed=-1,
    comp_kv_quant="none",
    comp_kv_quant_granularity="per_page",
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
        min_decode_kv_len_for_paging=min_decode_kv_len_for_paging,
        proxy_frequency_layout=proxy_frequency_layout,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        compression_method=compression_method,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
        score_use_direct_spectral_proxy=score_use_direct_spectral_proxy,
        score_use_haar_proxy=score_use_haar_proxy,
        score_use_haar_mixed_proxy=score_use_haar_mixed_proxy,
        score_use_hadamard_proxy=score_use_hadamard_proxy,
        score_use_quest_minmax=score_use_quest_minmax,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        use_triton=use_triton,
        weight_compressed_by_population=weight_compressed_by_population,
        max_unselected_compressed=max_unselected_compressed,
        comp_kv_quant=comp_kv_quant,
        comp_kv_quant_granularity=comp_kv_quant_granularity,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config (Llama):")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  proxy_frequency_layout={proxy_frequency_layout}")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, compression_method={compression_method}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"score_use_direct_spectral_proxy={score_use_direct_spectral_proxy}, "
        f"score_use_haar_proxy={score_use_haar_proxy}, "
        f"score_use_haar_mixed_proxy={score_use_haar_mixed_proxy}, "
        f"score_use_hadamard_proxy={score_use_hadamard_proxy}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}, "
        f"weight_compressed_by_population={weight_compressed_by_population}, "
        f"max_unselected_compressed={max_unselected_compressed}, "
        f"comp_kv_quant={comp_kv_quant}, "
        f"comp_kv_quant_granularity={comp_kv_quant_granularity}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.llama.modeling_llama.LlamaAttention.forward = dct_page_attention_forward
