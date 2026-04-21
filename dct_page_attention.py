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
    return M.squeeze(0).squeeze(0).to(dtype)  # [comp_size, page_size]


def _get_or_build_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached projection matrix, building it on first call."""
    M = getattr(attn_module, '_dct_proj_matrix', None)
    if (
        M is None
        or M.shape != (comp_size, page_size)
        or M.device != device
    ):
        M = _build_dct_projection_matrix(page_size, comp_size, device, dtype)
        attn_module._dct_proj_matrix = M
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

    cached_k = getattr(attn_module, '_comp_k_cache', None)
    cached_v = getattr(attn_module, '_comp_v_cache', None)
    n_cached = getattr(attn_module, '_comp_n_pages_cached', 0)
    capacity = getattr(attn_module, '_comp_cache_capacity', 0)
    cached_strategy = getattr(attn_module, '_comp_cache_strategy', None)
    cached_quant = getattr(attn_module, '_comp_cache_quant', None)
    cached_quant_granularity = getattr(attn_module, '_comp_cache_quant_granularity', None)
    cached_store_v = getattr(attn_module, '_comp_cache_store_v', None)
    cur_strategy = cfg.compressed_token_rope
    cur_quant = cfg.comp_kv_quant
    cur_quant_granularity = cfg.comp_kv_quant_granularity

    # Invalidate cache when the sequence restarts, shape changes, RoPE strategy changes,
    # quant config changes (requires different storage dtype / scale shape), or the
    # unselected_mode switch flips whether V is stored.
    if (cached_k is None
            or (store_v and cached_v is None)
            or num_pages < n_cached
            or cached_k.shape[0] != bsz
            or cached_k.shape[3] != comp_size
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

        # Step B: compress K (always — needed for scoring) and V (only when stored).
        new_comp_k = _compress_pages(attn_module, new_k_for_compress, comp_size)
        new_comp_v = _compress_pages(attn_module, new_v, comp_size) if store_v else None

        # Step C: re-apply RoPE to compressed K at block-center positions (still bf16).
        if cur_strategy == "block_center":
            new_positions = _block_center_positions(
                n_cached, n_new, cfg.page_size, comp_size, cfg.sink_size, new_comp_k.device,
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
                bsz, num_kv_heads, new_capacity, comp_size, storage_d,
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
                    cur_quant_granularity, bsz, num_kv_heads, new_capacity, comp_size,
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
        return comp_k, None

    v_slice = attn_module._comp_v_cache[:, :, :num_pages]
    if cur_quant == "none":
        comp_v = v_slice
    else:
        comp_v = _dequantize_comp(
            v_slice,
            attn_module._comp_v_scale_cache[:, :, :num_pages],
            cur_quant, cur_quant_granularity, head_dim,
        )
    return comp_k, comp_v


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
    "_comp_k_scale_cache",
    "_comp_v_scale_cache",
    "_comp_n_pages_cached",
    "_comp_cache_capacity",
    "_comp_cache_strategy",
    "_comp_cache_quant",
    "_comp_cache_quant_granularity",
    "_comp_cache_store_v",
    "_page_scores_buf",
    "_page_scores_np",
    "_page_scores_capacity",
    "_topk_out_buf",
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
        self._comp_k_scale_cache = None
        self._comp_v_scale_cache = None
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
    if cfg.score_use_quest_minmax:
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        page_scores = _score_pages_quest(
            score_query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
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
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    compressed_token_rope="mixed",
    continuous_rope=False,
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
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
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
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
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
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    compressed_token_rope="mixed",
    continuous_rope=False,
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
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
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
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
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
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    compressed_token_rope="mixed",
    continuous_rope=False,
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
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        compressed_token_rope=compressed_token_rope,
        continuous_rope=continuous_rope,
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
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}, compressed_token_rope={compressed_token_rope}")
    print(
        f"  continuous_rope={continuous_rope}, "
        f"select_with_oracle_page_scores={select_with_oracle_page_scores}, "
        f"use_triton={use_triton}, "
        f"weight_compressed_by_population={weight_compressed_by_population}, "
        f"max_unselected_compressed={max_unselected_compressed}, "
        f"comp_kv_quant={comp_kv_quant}, "
        f"comp_kv_quant_granularity={comp_kv_quant_granularity}"
    )
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.llama.modeling_llama.LlamaAttention.forward = dct_page_attention_forward
