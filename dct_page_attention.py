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
    topk_sort_triton, 
    assemble_kv_drop_triton, 
    score_pages_triton,
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
    """Mirror the upstream attention backend dispatch used by HF 4.54.x."""
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


def _compute_rope_cos_sin(positions, config, device, dtype):
    """Compute cos/sin for arbitrary positions, using the model's rope_scaling.

    Uses transformers' ROPE_INIT_FUNCTIONS (the same function that
    LlamaRotaryEmbedding.__init__ calls) so the inv_freq and
    attention_scaling are identical to the model's own RoPE.

    Args:
        positions: [seq_len] integer tensor
        config:    model config object

    Returns:
        cos, sin: each [1, 1, seq_len, head_dim]
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rope_type = "default"
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))

    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device)

    freqs = torch.outer(positions.float(), inv_freq)   # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)             # [seq_len, head_dim]
    cos = (emb.cos() * attention_scaling).unsqueeze(0).unsqueeze(0).to(dtype)
    sin = (emb.sin() * attention_scaling).unsqueeze(0).unsqueeze(0).to(dtype)
    return cos, sin


def _compute_rope_cos_sin_for_position_ids(position_ids, config, device, dtype):
    """Compute cos/sin for arbitrary per-head position ids.

    Args:
        position_ids: integer tensor shaped [..., seq_len]

    Returns:
        cos, sin: tensors shaped [..., seq_len, head_dim]
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rope_type = "default"
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))

    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device)
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
def _update_comp_cache(attn_module, paged_k, paged_v, num_pages, comp_size):
    """
    Incrementally maintain DCT-compressed page representations.

    Uses torch.cat to append new compressed pages. Output is always contiguous,
    which keeps score_pages_triton L2-cache-friendly.  The comp_k strides change
    every step (dim-2 grows), so they must NOT be cached — compute them live.
    """
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_dct_comp_k_cache', None)
    n_cached  = getattr(attn_module, '_dct_n_pages_cached', 0)

    # Invalidate cache when the sequence restarts or config changed
    if (cached_k is None
            or num_pages < n_cached
            or cached_k.shape[0] != bsz
            or cached_k.shape[3] != comp_size):
        attn_module._dct_comp_k_cache  = None
        attn_module._dct_comp_v_cache  = None
        attn_module._dct_n_pages_cached = 0
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        new_v = paged_v[:, :, n_cached:num_pages]

        M = _get_or_build_projection_matrix(
            attn_module, page_size, comp_size, new_k.device, new_k.dtype
        )
        new_comp_k = torch.einsum('cs,bhnsd->bhncd', M, new_k)
        new_comp_v = torch.einsum('cs,bhnsd->bhncd', M, new_v)

        if attn_module._dct_comp_k_cache is None:
            attn_module._dct_comp_k_cache = new_comp_k
            attn_module._dct_comp_v_cache = new_comp_v
        else:
            attn_module._dct_comp_k_cache = torch.cat(
                [attn_module._dct_comp_k_cache, new_comp_k], dim=2
            )
            attn_module._dct_comp_v_cache = torch.cat(
                [attn_module._dct_comp_v_cache, new_comp_v], dim=2
            )

        attn_module._dct_n_pages_cached = num_pages

    return attn_module._dct_comp_k_cache, attn_module._dct_comp_v_cache


def _compress_pages(attn_module, paged_x, comp_size):
    """Project [bsz, kv_heads, num_pages, page_size, head_dim] pages to comp_size."""
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


def _update_score_key_cache(attn_module, paged_k, num_pages, comp_size, cfg):
    """Maintain a separate score-time key cache in original-position RoPE space."""
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_dct_score_comp_k_cache', None)
    n_cached = getattr(attn_module, '_dct_score_n_pages_cached', 0)

    if (
        cached_k is None
        or num_pages < n_cached
        or cached_k.shape[0] != bsz
        or cached_k.shape[3] != comp_size
    ):
        attn_module._dct_score_comp_k_cache = None
        attn_module._dct_score_n_pages_cached = 0
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        flat_k = new_k.reshape(bsz, num_kv_heads, n_new * page_size, head_dim)
        start_pos = cfg.sink_size + n_cached * page_size
        positions = torch.arange(
            start_pos,
            start_pos + n_new * page_size,
            device=new_k.device,
        )
        cos_pages, sin_pages = _compute_rope_cos_sin(
            positions, attn_module.config, new_k.device, new_k.dtype
        )
        flat_k_rope = _apply_rope(flat_k, cos_pages, sin_pages)
        new_k_rope = flat_k_rope.reshape(bsz, num_kv_heads, n_new, page_size, head_dim)
        new_comp_k = _compress_pages(attn_module, new_k_rope, comp_size)

        if attn_module._dct_score_comp_k_cache is None:
            attn_module._dct_score_comp_k_cache = new_comp_k
        else:
            attn_module._dct_score_comp_k_cache = torch.cat(
                [attn_module._dct_score_comp_k_cache, new_comp_k], dim=2
            )

        attn_module._dct_score_n_pages_cached = num_pages

    return attn_module._dct_score_comp_k_cache


def _update_score_spectral_key_cache(attn_module, paged_k, num_pages, comp_size, cfg):
    """Maintain a score-time cache of direct spectral proxies in original-position RoPE space."""
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_dct_score_spectral_k_cache', None)
    n_cached = getattr(attn_module, '_dct_score_spectral_n_pages_cached', 0)

    if (
        cached_k is None
        or num_pages < n_cached
        or cached_k.shape[0] != bsz
        or cached_k.shape[3] != comp_size
    ):
        attn_module._dct_score_spectral_k_cache = None
        attn_module._dct_score_spectral_n_pages_cached = 0
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        flat_k = new_k.reshape(bsz, num_kv_heads, n_new * page_size, head_dim)
        start_pos = cfg.sink_size + n_cached * page_size
        positions = torch.arange(
            start_pos,
            start_pos + n_new * page_size,
            device=new_k.device,
        )
        cos_pages, sin_pages = _compute_rope_cos_sin(
            positions, attn_module.config, new_k.device, new_k.dtype
        )
        flat_k_rope = _apply_rope(flat_k, cos_pages, sin_pages)
        new_k_rope = flat_k_rope.reshape(bsz, num_kv_heads, n_new, page_size, head_dim)
        new_spec_k = _project_pages_to_spectral(attn_module, new_k_rope, comp_size)

        if attn_module._dct_score_spectral_k_cache is None:
            attn_module._dct_score_spectral_k_cache = new_spec_k
        else:
            attn_module._dct_score_spectral_k_cache = torch.cat(
                [attn_module._dct_score_spectral_k_cache, new_spec_k], dim=2
            )

        attn_module._dct_score_spectral_n_pages_cached = num_pages

    return attn_module._dct_score_spectral_k_cache


def _haar_proxy_block_center_positions(start_page_idx, n_pages, page_size, comp_size, sink_size, device):
    """Anchor each Haar lowpass proxy to the center of its source block."""
    page_ids = torch.arange(start_page_idx, start_page_idx + n_pages, device=device, dtype=torch.long)
    page_bases = sink_size + page_ids[:, None] * page_size
    starts = torch.tensor(
        [(idx * page_size) // comp_size for idx in range(comp_size)],
        device=device,
        dtype=torch.long,
    )
    ends = torch.tensor(
        [max(((idx + 1) * page_size) // comp_size, (idx * page_size) // comp_size + 1) for idx in range(comp_size)],
        device=device,
        dtype=torch.long,
    )
    proxy_offsets = ((starts + ends - 1) // 2).clamp_max(page_size - 1)
    return page_bases + proxy_offsets[None, :]


def _haar_mixed_proxy_anchor_positions(start_page_idx, n_pages, page_size, comp_size, sink_size, device):
    """Anchor each mixed Haar proxy to the center of its support interval."""
    page_ids = torch.arange(start_page_idx, start_page_idx + n_pages, device=device, dtype=torch.long)
    page_bases = sink_size + page_ids[:, None] * page_size
    supports = _build_haar_mixed_supports(page_size, comp_size)
    proxy_offsets = torch.tensor(
        [min(page_size - 1, (start + end - 1) // 2) for start, end in supports],
        device=device,
        dtype=torch.long,
    )
    return page_bases + proxy_offsets[None, :]


def _update_score_haar_key_cache(attn_module, paged_k, num_pages, comp_size, cfg):
    """Maintain a separate score-time cache of Haar lowpass proxies."""
    bsz, num_kv_heads, _, _, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_dct_score_haar_k_cache', None)
    n_cached = getattr(attn_module, '_dct_score_haar_n_pages_cached', 0)

    if (
        cached_k is None
        or num_pages < n_cached
        or cached_k.shape[0] != bsz
        or cached_k.shape[3] != comp_size
    ):
        attn_module._dct_score_haar_k_cache = None
        attn_module._dct_score_haar_n_pages_cached = 0
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        new_proxy_k = _project_pages_to_haar_lowpass(attn_module, new_k, comp_size)
        positions = _haar_proxy_block_center_positions(
            n_cached,
            n_new,
            cfg.page_size,
            comp_size,
            cfg.sink_size,
            new_proxy_k.device,
        ).reshape(-1)
        cos_proxy, sin_proxy = _compute_rope_cos_sin(
            positions, attn_module.config, new_proxy_k.device, new_proxy_k.dtype
        )
        flat_proxy_k = new_proxy_k.reshape(bsz, num_kv_heads, n_new * comp_size, head_dim)
        flat_proxy_k = _apply_rope(flat_proxy_k, cos_proxy, sin_proxy)
        new_score_proxy_k = flat_proxy_k.reshape(bsz, num_kv_heads, n_new, comp_size, head_dim)

        if attn_module._dct_score_haar_k_cache is None:
            attn_module._dct_score_haar_k_cache = new_score_proxy_k
        else:
            attn_module._dct_score_haar_k_cache = torch.cat(
                [attn_module._dct_score_haar_k_cache, new_score_proxy_k], dim=2
            )

        attn_module._dct_score_haar_n_pages_cached = num_pages

    return attn_module._dct_score_haar_k_cache


def _update_score_haar_mixed_key_cache(attn_module, paged_k, num_pages, comp_size, cfg):
    """Maintain a separate score-time cache of mixed Haar proxies."""
    bsz, num_kv_heads, _, _, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_dct_score_haar_mixed_k_cache', None)
    n_cached = getattr(attn_module, '_dct_score_haar_mixed_n_pages_cached', 0)

    if (
        cached_k is None
        or num_pages < n_cached
        or cached_k.shape[0] != bsz
        or cached_k.shape[3] != comp_size
    ):
        attn_module._dct_score_haar_mixed_k_cache = None
        attn_module._dct_score_haar_mixed_n_pages_cached = 0
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        new_proxy_k = _project_pages_to_haar_mixed(attn_module, new_k, comp_size)
        positions = _haar_mixed_proxy_anchor_positions(
            n_cached,
            n_new,
            cfg.page_size,
            comp_size,
            cfg.sink_size,
            new_proxy_k.device,
        ).reshape(-1)
        cos_proxy, sin_proxy = _compute_rope_cos_sin(
            positions, attn_module.config, new_proxy_k.device, new_proxy_k.dtype
        )
        flat_proxy_k = new_proxy_k.reshape(bsz, num_kv_heads, n_new * comp_size, head_dim)
        flat_proxy_k = _apply_rope(flat_proxy_k, cos_proxy, sin_proxy)
        new_score_proxy_k = flat_proxy_k.reshape(bsz, num_kv_heads, n_new, comp_size, head_dim)

        if attn_module._dct_score_haar_mixed_k_cache is None:
            attn_module._dct_score_haar_mixed_k_cache = new_score_proxy_k
        else:
            attn_module._dct_score_haar_mixed_k_cache = torch.cat(
                [attn_module._dct_score_haar_mixed_k_cache, new_score_proxy_k], dim=2
            )

        attn_module._dct_score_haar_mixed_n_pages_cached = num_pages

    return attn_module._dct_score_haar_mixed_k_cache


def _update_score_hadamard_key_cache(attn_module, paged_k, num_pages, comp_size, cfg):
    """Maintain a score-time cache of Hadamard proxies in original-position RoPE space."""
    bsz, num_kv_heads, _, page_size, head_dim = paged_k.shape

    cached_k = getattr(attn_module, '_dct_score_hadamard_k_cache', None)
    n_cached = getattr(attn_module, '_dct_score_hadamard_n_pages_cached', 0)

    if (
        cached_k is None
        or num_pages < n_cached
        or cached_k.shape[0] != bsz
        or cached_k.shape[3] != comp_size
    ):
        attn_module._dct_score_hadamard_k_cache = None
        attn_module._dct_score_hadamard_n_pages_cached = 0
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_k = paged_k[:, :, n_cached:num_pages]
        flat_k = new_k.reshape(bsz, num_kv_heads, n_new * page_size, head_dim)
        start_pos = cfg.sink_size + n_cached * page_size
        positions = torch.arange(
            start_pos,
            start_pos + n_new * page_size,
            device=new_k.device,
        )
        cos_pages, sin_pages = _compute_rope_cos_sin(
            positions, attn_module.config, new_k.device, new_k.dtype
        )
        flat_k_rope = _apply_rope(flat_k, cos_pages, sin_pages)
        new_k_rope = flat_k_rope.reshape(bsz, num_kv_heads, n_new, page_size, head_dim)
        new_proxy_k = _project_pages_to_hadamard(attn_module, new_k_rope, comp_size)

        if attn_module._dct_score_hadamard_k_cache is None:
            attn_module._dct_score_hadamard_k_cache = new_proxy_k
        else:
            attn_module._dct_score_hadamard_k_cache = torch.cat(
                [attn_module._dct_score_hadamard_k_cache, new_proxy_k], dim=2
            )

        attn_module._dct_score_hadamard_n_pages_cached = num_pages

    return attn_module._dct_score_hadamard_k_cache


def _apply_original_position_rope_to_paged_k(paged_k, sink_size, model_config):
    """Apply original-position RoPE to full paged keys for debug/oracle scoring."""
    bsz, num_kv_heads, num_pages, page_size, head_dim = paged_k.shape
    flat_k = paged_k.reshape(bsz, num_kv_heads, num_pages * page_size, head_dim)
    positions = torch.arange(
        sink_size,
        sink_size + num_pages * page_size,
        device=paged_k.device,
    )
    cos_pages, sin_pages = _compute_rope_cos_sin(
        positions, model_config, paged_k.device, paged_k.dtype
    )
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


def _proxy_block_center_positions(start_page_idx, n_pages, page_size, comp_size, sink_size, device):
    """Anchor each compressed proxy to the center of its source block range."""
    page_ids = torch.arange(start_page_idx, start_page_idx + n_pages, device=device, dtype=torch.long)
    page_bases = sink_size + page_ids[:, None] * page_size
    proxy_ids = torch.arange(comp_size, device=device, dtype=torch.long)
    proxy_offsets = ((2 * proxy_ids + 1) * page_size) // (2 * comp_size)
    proxy_offsets = proxy_offsets.clamp_max(page_size - 1)
    return page_bases + proxy_offsets[None, :]


def _shared_block_anchor_positions(start_page_idx, n_pages, page_size, comp_size, sink_size, device, anchor_mode):
    """Assign every proxy in a page the same shared anchor position."""
    page_ids = torch.arange(start_page_idx, start_page_idx + n_pages, device=device, dtype=torch.long)
    page_bases = sink_size + page_ids * page_size
    if anchor_mode == "center":
        anchor_offset = page_size // 2
    elif anchor_mode == "start":
        anchor_offset = 0
    else:
        raise ValueError(f"Unsupported proxy block anchor mode: {anchor_mode}")
    anchors = page_bases + anchor_offset
    return anchors[:, None].expand(n_pages, comp_size)


def _update_score_proxy_key_cache(attn_module, comp_k, num_pages, comp_size, cfg):
    """Apply block-aware proxy-position RoPE to already-compressed page proxies."""
    bsz, num_kv_heads, _, _, head_dim = comp_k.shape

    cached_k = getattr(attn_module, '_dct_score_proxy_comp_k_cache', None)
    n_cached = getattr(attn_module, '_dct_score_proxy_n_pages_cached', 0)

    if (
        cached_k is None
        or num_pages < n_cached
        or cached_k.shape[0] != bsz
        or cached_k.shape[3] != comp_size
    ):
        attn_module._dct_score_proxy_comp_k_cache = None
        attn_module._dct_score_proxy_n_pages_cached = 0
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_comp_k = comp_k[:, :, n_cached:num_pages]
        positions = _proxy_block_center_positions(
            n_cached,
            n_new,
            cfg.page_size,
            comp_size,
            cfg.sink_size,
            new_comp_k.device,
        ).reshape(-1)
        cos_proxy, sin_proxy = _compute_rope_cos_sin(
            positions, attn_module.config, new_comp_k.device, new_comp_k.dtype
        )
        flat_comp_k = new_comp_k.reshape(bsz, num_kv_heads, n_new * comp_size, head_dim)
        flat_comp_k = _apply_rope(flat_comp_k, cos_proxy, sin_proxy)
        new_score_comp_k = flat_comp_k.reshape(bsz, num_kv_heads, n_new, comp_size, head_dim)

        if attn_module._dct_score_proxy_comp_k_cache is None:
            attn_module._dct_score_proxy_comp_k_cache = new_score_comp_k
        else:
            attn_module._dct_score_proxy_comp_k_cache = torch.cat(
                [attn_module._dct_score_proxy_comp_k_cache, new_score_comp_k], dim=2
            )

        attn_module._dct_score_proxy_n_pages_cached = num_pages

    return attn_module._dct_score_proxy_comp_k_cache


def _update_score_proxy_shared_anchor_key_cache(attn_module, comp_k, num_pages, comp_size, cfg, anchor_mode):
    """Apply a shared block anchor position to all proxies inside each page."""
    bsz, num_kv_heads, _, _, head_dim = comp_k.shape

    cache_attr = f"_dct_score_proxy_shared_{anchor_mode}_comp_k_cache"
    n_cache_attr = f"_dct_score_proxy_shared_{anchor_mode}_n_pages_cached"
    cached_k = getattr(attn_module, cache_attr, None)
    n_cached = getattr(attn_module, n_cache_attr, 0)

    if (
        cached_k is None
        or num_pages < n_cached
        or cached_k.shape[0] != bsz
        or cached_k.shape[3] != comp_size
    ):
        setattr(attn_module, cache_attr, None)
        setattr(attn_module, n_cache_attr, 0)
        n_cached = 0

    n_new = num_pages - n_cached
    if n_new > 0:
        new_comp_k = comp_k[:, :, n_cached:num_pages]
        positions = _shared_block_anchor_positions(
            n_cached,
            n_new,
            cfg.page_size,
            comp_size,
            cfg.sink_size,
            new_comp_k.device,
            anchor_mode,
        ).reshape(-1)
        cos_proxy, sin_proxy = _compute_rope_cos_sin(
            positions, attn_module.config, new_comp_k.device, new_comp_k.dtype
        )
        flat_comp_k = new_comp_k.reshape(bsz, num_kv_heads, n_new * comp_size, head_dim)
        flat_comp_k = _apply_rope(flat_comp_k, cos_proxy, sin_proxy)
        new_score_comp_k = flat_comp_k.reshape(bsz, num_kv_heads, n_new, comp_size, head_dim)

        cached_k = getattr(attn_module, cache_attr, None)
        if cached_k is None:
            setattr(attn_module, cache_attr, new_score_comp_k)
        else:
            setattr(
                attn_module,
                cache_attr,
                torch.cat([cached_k, new_score_comp_k], dim=2),
            )

        setattr(attn_module, n_cache_attr, num_pages)

    return getattr(attn_module, cache_attr)


def _build_drop_mode_position_ids(selected_indices, cfg, num_pages, actual_recent):
    """Build original token positions for assembled drop-mode KV."""
    bsz, num_kv_heads, actual_top_k = selected_indices.shape
    device = selected_indices.device

    sink_positions = torch.arange(cfg.sink_size, device=device, dtype=torch.long)
    sink_positions = sink_positions.view(1, 1, -1).expand(bsz, num_kv_heads, -1)

    page_offsets = torch.arange(cfg.page_size, device=device, dtype=torch.long)
    selected_positions = cfg.sink_size + selected_indices.to(torch.long).unsqueeze(-1) * cfg.page_size
    selected_positions = selected_positions + page_offsets.view(1, 1, 1, -1)
    selected_positions = selected_positions.reshape(bsz, num_kv_heads, actual_top_k * cfg.page_size)

    recent_start = cfg.sink_size + num_pages * cfg.page_size
    recent_positions = recent_start + torch.arange(actual_recent, device=device, dtype=torch.long)
    recent_positions = recent_positions.view(1, 1, -1).expand(bsz, num_kv_heads, -1)

    return torch.cat((sink_positions, selected_positions, recent_positions), dim=-1)


def _apply_original_position_rope_to_final_k(final_k, selected_indices, num_pages, actual_recent, cfg, model_config):
    """Apply RoPE to assembled drop-mode KV using the tokens' original positions."""
    position_ids = _build_drop_mode_position_ids(selected_indices, cfg, num_pages, actual_recent)
    cos, sin = _compute_rope_cos_sin_for_position_ids(position_ids, model_config, final_k.device, final_k.dtype)
    return _apply_rope(final_k, cos, sin)


_DCT_RUNTIME_STATE_ATTRS = (
    "_dct_comp_k_cache",
    "_dct_comp_v_cache",
    "_dct_n_pages_cached",
    "_dct_score_comp_k_cache",
    "_dct_score_n_pages_cached",
    "_dct_score_spectral_k_cache",
    "_dct_score_spectral_n_pages_cached",
    "_dct_score_haar_k_cache",
    "_dct_score_haar_n_pages_cached",
    "_dct_score_haar_mixed_k_cache",
    "_dct_score_haar_mixed_n_pages_cached",
    "_dct_score_hadamard_k_cache",
    "_dct_score_hadamard_n_pages_cached",
    "_dct_score_proxy_comp_k_cache",
    "_dct_score_proxy_n_pages_cached",
    "_dct_score_proxy_shared_center_comp_k_cache",
    "_dct_score_proxy_shared_center_n_pages_cached",
    "_dct_score_proxy_shared_start_comp_k_cache",
    "_dct_score_proxy_shared_start_n_pages_cached",
    "_page_scores_buf",
    "_page_scores_np",
    "_topk_out_buf",
    "_assemble_stride_cache",
    "_final_k_buf",
    "_final_v_buf",
    "_sel_idx_buf",
    "_assemble_buf_len",
)


def _maybe_reset_dct_runtime_state(attn_module, past_key_value):
    """Clear per-generation runtime caches when the HF cache object changes."""
    cached_ref = getattr(attn_module, "_dct_runtime_cache_ref", None)
    if cached_ref is past_key_value:
        return

    for attr in _DCT_RUNTIME_STATE_ATTRS:
        if hasattr(attn_module, attr):
            delattr(attn_module, attr)
    attn_module._dct_runtime_cache_ref = past_key_value


# ---------------------------------------------------------------------------
# Main attention forward
# ---------------------------------------------------------------------------
def dct_page_attention_forward(
    self, # the Qwen2Attention/LlamaAttention instance (we access its projections like self.q_proj, config like self.config, etc.)
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None, # The type can be torch.Tensor or None, and the default value is None
    past_key_value: Optional[Cache] = None,
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
            cfg.score_with_original_rope,
            cfg.score_use_direct_spectral_proxy,
            cfg.score_use_haar_proxy,
            cfg.score_use_haar_mixed_proxy,
            cfg.score_use_hadamard_proxy,
            cfg.score_proxy_with_block_position_rope,
            cfg.score_proxy_with_shared_block_center_rope,
            cfg.score_proxy_with_shared_block_start_rope,
        )
    )
    if num_score_proxy_modes > 1:
        raise ValueError(
            "Only one score-time proxy rope mode may be enabled at once."
        )
    input_shape = hidden_states.shape[:-1] # (bsz, q_len)
    hidden_shape = (*input_shape, -1, self.head_dim) # (bsz, q_len, num_heads, head_dim)
    bsz, q_len = input_shape     
    _maybe_reset_dct_runtime_state(self, past_key_value)

    if q_len>1:
        # Step 1: Q/K/V projection
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_heads, q_len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
        
        # Step 2 & 3: RoPE and KV cache
        cos, sin = position_embeddings
        query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_q, attn_k, attn_v = query_rope, key_rope, value_states
        if past_key_value is not None: # unless we call the model directly with use_cache=False, past_key_value is not None.
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if cfg.continuous_rope:
                # Keep the cache in pre-RoPE space for decode-time page assembly, but
                # still attend over the full cached context during chunked prefill.
                key_cached, value_cached = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
                all_pos = torch.arange(key_cached.shape[2], device=key_cached.device)
                cos_all, sin_all = _compute_rope_cos_sin(
                    all_pos, self.config, key_cached.device, key_cached.dtype
                )
                attn_k = _apply_rope(key_cached, cos_all, sin_all)
                attn_v = value_cached
            else:
                attn_k, attn_v = past_key_value.update(
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

        extra_tokens = cfg.page_size * 2

        # Convert DynamicCache → PreAllocatedLayer at end of prefill (last layer only).
        # All layers are converted at once, so by the first decode step every
        # layer's cache.update() already uses PreAllocatedLayer (fixed strides).
        if (past_key_value is not None
                and self.layer_idx == self.config.num_hidden_layers - 1
                and not getattr(past_key_value, '_preallocated', False)):
            pre_allocate_cache(past_key_value, extra_tokens=extra_tokens)
            past_key_value._preallocated = True

        return attn_output, attn_weights

    # ---- DECODE PATH (q_len == 1, long KV cache) ----
    # Step 1: QKV projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    # Step 2: RoPE + KV cache update
    cos, sin = position_embeddings
    if cfg.continuous_rope:
        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_cached, value_cached = past_key_value.update(
                key_states, value_states, self.layer_idx, # cache_kwargs # commented out because we will compute rope table later.
            )
        else:
            key_cached, value_cached = key_states, value_states
        kv_len = key_cached.shape[2]
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        kv_len = key_states.shape[2]
    
    # Check if DCT path is active
    min_len_for_paging = cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size
    
    # No need to do dct paging since there isn't enough tokens.
    if kv_len < min_len_for_paging:
        if cfg.continuous_rope:
            all_pos = torch.arange(kv_len, device=key_cached.device)
            cos_all, sin_all = _compute_rope_cos_sin(
                all_pos, self.config, key_cached.device, key_cached.dtype
            )
            attn_q, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
            attn_k = _apply_rope(key_cached, cos_all, sin_all)
            attn_v = value_cached
        else:
            attn_q, attn_k, attn_v = query_states, key_states, value_states

        attention_interface = _get_attention_interface(self)
        attn_output, attn_weights = attention_interface(
            self, attn_q, attn_k, attn_v, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    # Use pre-RoPE KV for page building in continuous_rope mode
    if cfg.continuous_rope:
        key_states = key_cached
        value_states = value_cached

    if cfg.continuous_rope and cfg.unselected_mode != "drop":
        raise NotImplementedError(
            "continuous_rope currently supports unselected_mode='drop' only."
        )
    
    # Step 3: Segment KV cache and update the incremental compressed page cache.
    # DCT is computed only for pages that are newly finalized since the last
    # decode step; all previously cached compressed representations are reused.
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    (sink_k, sink_v, paged_k, paged_v,
        recent_k, recent_v, num_pages, actual_recent) = segment_kv(
        key_states, value_states, cfg
    )
    
    # Step 4: DCT compression (torch.cat - always contiguous, L2-warm)
    comp_k, comp_v = _update_comp_cache(self, paged_k, paged_v, num_pages, comp_size)

    # Step 5: Score pages (Triton kernel 1 — returns page_scores only)
    # Pre-allocate page_scores buffer (reallocate only when num_pages changes)
    _num_kv_heads = self.config.num_key_value_heads # 8 for Llama-3.1-8B
    if getattr(self, '_page_scores_np', 0) != num_pages:
        self._page_scores_buf = torch.empty(
            bsz, _num_kv_heads, num_pages,
            dtype=torch.float32, device=comp_k.device,
        )
        self._page_scores_np = num_pages

    score_query_states = query_states
    score_comp_k = comp_k
    if cfg.continuous_rope and num_score_proxy_modes:
        score_query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        if cfg.score_use_direct_spectral_proxy:
            score_comp_k = _update_score_spectral_key_cache(
                self, paged_k, num_pages, comp_size, cfg
            )
        elif cfg.score_use_haar_proxy:
            score_comp_k = _update_score_haar_key_cache(
                self, paged_k, num_pages, comp_size, cfg
            )
        elif cfg.score_use_haar_mixed_proxy:
            score_comp_k = _update_score_haar_mixed_key_cache(
                self, paged_k, num_pages, comp_size, cfg
            )
        elif cfg.score_use_hadamard_proxy:
            score_comp_k = _update_score_hadamard_key_cache(
                self, paged_k, num_pages, comp_size, cfg
            )
        elif cfg.score_with_original_rope:
            score_comp_k = _update_score_key_cache(self, paged_k, num_pages, comp_size, cfg)
        elif cfg.score_proxy_with_shared_block_center_rope:
            score_comp_k = _update_score_proxy_shared_anchor_key_cache(
                self, comp_k, num_pages, comp_size, cfg, anchor_mode="center"
            )
        elif cfg.score_proxy_with_shared_block_start_rope:
            score_comp_k = _update_score_proxy_shared_anchor_key_cache(
                self, comp_k, num_pages, comp_size, cfg, anchor_mode="start"
            )
        else:
            score_comp_k = _update_score_proxy_key_cache(self, comp_k, num_pages, comp_size, cfg)

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
    if not hasattr(self, '_topk_out_buf'):
        self._topk_out_buf = torch.empty(bsz, _num_kv_heads, actual_top_k, dtype=torch.int32, device=comp_k.device)

    selected_indices = topk_sort_triton(selection_page_scores, actual_top_k, out=self._topk_out_buf)

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
                "score_with_original_rope": bool(cfg.score_with_original_rope),
                "score_use_direct_spectral_proxy": bool(cfg.score_use_direct_spectral_proxy),
                "score_use_haar_proxy": bool(cfg.score_use_haar_proxy),
                "score_use_haar_mixed_proxy": bool(cfg.score_use_haar_mixed_proxy),
                "score_use_hadamard_proxy": bool(cfg.score_use_hadamard_proxy),
                "score_proxy_with_block_position_rope": bool(cfg.score_proxy_with_block_position_rope),
                "score_proxy_with_shared_block_center_rope": bool(cfg.score_proxy_with_shared_block_center_rope),
                "score_proxy_with_shared_block_start_rope": bool(cfg.score_proxy_with_shared_block_start_rope),
                "cache_position": None
                if cache_position is None
                else cache_position.detach().cpu(),
                "page_scores": page_scores.detach().float().cpu(),
                "oracle_page_scores": oracle_page_scores.detach().float().cpu(),
                "selection_used_oracle_page_scores": bool(cfg.select_with_oracle_page_scores),
                "selected_indices": selected_indices.detach().cpu(),
            }
        )

    if cfg.unselected_mode == "drop":
        assembled_len = cfg.sink_size + actual_top_k * cfg.page_size + actual_recent
    else:
        num_unselected = num_pages - actual_top_k
        middle_len = actual_top_k * cfg.page_size + num_unselected * comp_size
        assembled_len = cfg.sink_size + middle_len + actual_recent

    cos_table = None
    sin_table = None

    # Pre-allocate or expand output buffers (avoids torch.empty per step)
    _buf_len = getattr(self, '_assemble_buf_len', 0)
    if assembled_len > _buf_len:
        _max_len = assembled_len + cfg.page_size
        _nkv = _num_kv_heads
        self._final_k_buf = torch.empty(bsz, _nkv, _max_len, self.head_dim, dtype=comp_k.dtype, device=comp_k.device)
        self._final_v_buf = torch.empty_like(self._final_k_buf)
        self._sel_idx_buf = torch.empty(bsz, _nkv, actual_top_k, dtype=torch.int32, device=comp_k.device)
        self._assemble_buf_len = _max_len

    # Step 6b: Assemble KV for attention.
    if cfg.unselected_mode == "drop":
        final_k, final_v = assemble_kv_drop_triton(
            paged_k, paged_v,
            sink_k, sink_v, recent_k, recent_v,
            selected_indices,
            cos_table, sin_table,
            out_k=self._final_k_buf,
            out_v=self._final_v_buf,
            out_sel_idx=self._sel_idx_buf,
        )
    
    else:
        # Build stride cache on first call; rebuild when strides change
        # (strides are fixed within one sample's decode, but change across
        # samples because the pre-allocated KV cache is re-created)
        _cur_paged_strides = (paged_k.stride(0), paged_k.stride(1), paged_k.stride(2), paged_k.stride(3))
        if (not hasattr(self, '_assemble_stride_cache')
                or self._assemble_stride_cache['paged_strides'] != _cur_paged_strides):
            self._assemble_stride_cache = build_assemble_stride_cache(
                paged_k, comp_k, sink_k, recent_k, selected_indices,
                cos_table, self._final_k_buf,
            )

        final_k, final_v = assemble_kv_split_triton(
            paged_k, paged_v, comp_k, comp_v,
            sink_k, sink_v, recent_k, recent_v,
            selected_indices,
            cos_table, sin_table,
            out_k=self._final_k_buf,
            out_v=self._final_v_buf,
            stride_cache=self._assemble_stride_cache,
        )

    if cfg.continuous_rope:
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        final_k = _apply_original_position_rope_to_final_k(
            final_k,
            selected_indices,
            num_pages,
            actual_recent,
            cfg,
            self.config,
        )

    # Step 7a: Compute attention (no causal mask needed for q_len=1)
    attn_output = F.scaled_dot_product_attention(
        query_states, final_k, final_v,
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
    page_size=128,
    top_k=8,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.25,
    proxy_frequency_layout="low",
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    continuous_rope=True,
    score_with_original_rope=False,
    score_use_direct_spectral_proxy=False,
    score_use_haar_proxy=False,
    score_use_haar_mixed_proxy=False,
    score_use_hadamard_proxy=False,
    score_proxy_with_block_position_rope=False,
    score_proxy_with_shared_block_center_rope=False,
    score_proxy_with_shared_block_start_rope=False,
    select_with_oracle_page_scores=False,
    use_triton=True,
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
        proxy_frequency_layout=proxy_frequency_layout,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        continuous_rope=continuous_rope,
        score_with_original_rope=score_with_original_rope,
        score_use_direct_spectral_proxy=score_use_direct_spectral_proxy,
        score_use_haar_proxy=score_use_haar_proxy,
        score_use_haar_mixed_proxy=score_use_haar_mixed_proxy,
        score_use_hadamard_proxy=score_use_hadamard_proxy,
        score_proxy_with_block_position_rope=score_proxy_with_block_position_rope,
        score_proxy_with_shared_block_center_rope=score_proxy_with_shared_block_center_rope,
        score_proxy_with_shared_block_start_rope=score_proxy_with_shared_block_start_rope,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        use_triton=use_triton,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config:")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  proxy_frequency_layout={proxy_frequency_layout}")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"score_with_original_rope={score_with_original_rope}, "
        f"score_use_direct_spectral_proxy={score_use_direct_spectral_proxy}, "
        f"score_use_haar_proxy={score_use_haar_proxy}, "
        f"score_use_haar_mixed_proxy={score_use_haar_mixed_proxy}, "
        f"score_use_hadamard_proxy={score_use_hadamard_proxy}, "
        f"score_proxy_with_block_position_rope={score_proxy_with_block_position_rope}, "
        f"score_proxy_with_shared_block_center_rope={score_proxy_with_shared_block_center_rope}, "
        f"score_proxy_with_shared_block_start_rope={score_proxy_with_shared_block_start_rope}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = dct_page_attention_forward


def replace_llama_attn(
    page_size=128,
    top_k=8,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.25,
    proxy_frequency_layout="low",
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    continuous_rope=True,
    score_with_original_rope=False,
    score_use_direct_spectral_proxy=False,
    score_use_haar_proxy=False,
    score_use_haar_mixed_proxy=False,
    score_use_hadamard_proxy=False,
    score_proxy_with_block_position_rope=False,
    score_proxy_with_shared_block_center_rope=False,
    score_proxy_with_shared_block_start_rope=False,
    select_with_oracle_page_scores=False,
    use_triton=True,
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
        proxy_frequency_layout=proxy_frequency_layout,
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        continuous_rope=continuous_rope,
        score_with_original_rope=score_with_original_rope,
        score_use_direct_spectral_proxy=score_use_direct_spectral_proxy,
        score_use_haar_proxy=score_use_haar_proxy,
        score_use_haar_mixed_proxy=score_use_haar_mixed_proxy,
        score_use_hadamard_proxy=score_use_hadamard_proxy,
        score_proxy_with_block_position_rope=score_proxy_with_block_position_rope,
        score_proxy_with_shared_block_center_rope=score_proxy_with_shared_block_center_rope,
        score_proxy_with_shared_block_start_rope=score_proxy_with_shared_block_start_rope,
        select_with_oracle_page_scores=select_with_oracle_page_scores,
        use_triton=use_triton,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config (Llama):")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  proxy_frequency_layout={proxy_frequency_layout}")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"score_with_original_rope={score_with_original_rope}, "
        f"score_use_direct_spectral_proxy={score_use_direct_spectral_proxy}, "
        f"score_use_haar_proxy={score_use_haar_proxy}, "
        f"score_use_haar_mixed_proxy={score_use_haar_mixed_proxy}, "
        f"score_use_hadamard_proxy={score_use_hadamard_proxy}, "
        f"score_proxy_with_block_position_rope={score_proxy_with_block_position_rope}, "
        f"score_proxy_with_shared_block_center_rope={score_proxy_with_shared_block_center_rope}, "
        f"score_proxy_with_shared_block_start_rope={score_proxy_with_shared_block_start_rope}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.llama.modeling_llama.LlamaAttention.forward = dct_page_attention_forward
