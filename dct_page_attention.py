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
    apply_rope_q_direct, 
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

    def get_seq_length(self):
        return self._seen


def pre_allocate_cache(cache, extra_tokens=256):
    """Convert a DynamicCache (after prefill) to use pre-allocated layers."""
    for i, layer in enumerate(cache.layers):
        cache.layers[i] = PreAllocatedLayer.from_dynamic_layer(layer, extra_tokens)
    return cache


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
# DCT Projection Matrix (replaces FFT with a single matmul)
# ---------------------------------------------------------------------------
def _build_dct_projection_matrix(page_size, comp_size, device, dtype):
    """Precompute the [comp_size, page_size] projection matrix.

    The full DCT compression pipeline (DCT → truncate → IDCT → energy
    correction) is a linear transform.  We compute it by running the
    existing dct_compress_page on an identity matrix.
    """
    I = torch.eye(page_size, device=device, dtype=torch.float32)
    I = I.unsqueeze(0).unsqueeze(0)  # [1, 1, page_size, page_size]
    M = dct_compress_page(I, comp_size)  # [1, 1, comp_size, page_size]
    return M.squeeze(0).squeeze(0).to(dtype)  # [comp_size, page_size]


def _get_or_build_projection_matrix(attn_module, page_size, comp_size, device, dtype):
    """Return cached projection matrix, building it on first call."""
    M = getattr(attn_module, '_dct_proj_matrix', None)
    if M is None or M.shape != (comp_size, page_size) or M.device != device:
        M = _build_dct_projection_matrix(page_size, comp_size, device, dtype)
        attn_module._dct_proj_matrix = M
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
    input_shape = hidden_states.shape[:-1] # (bsz, q_len)
    hidden_shape = (*input_shape, -1, self.head_dim) # (bsz, q_len, num_heads, head_dim)
    bsz, q_len = input_shape     

    if q_len>1:
        # Step 1: Q/K/V projection
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_heads, q_len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bsz, num_kv_heads, q_len, head_dim): num_kv_heads for gqa
        
        # Step 2 & 3: RoPE and KV cache
        cos, sin = position_embeddings
        query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None: # unless we call the model directly with use_cache=False, past_key_values is not None.
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        attn_q, attn_k, attn_v = query_rope, key_rope, value_states

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

        # Cache the model's RoPE table (correctly handles rope_scaling).
        # Pre-allocate with extra slots so decode extends are O(1) writes.
        # cos/sin shape from position_embeddings: [bsz, seq_len, head_dim]
        # _rope_cos_2d / _rope_sin_2d shape: [alloc_len, head_dim]
        extra_tokens = cfg.page_size * 2
        alloc_len = q_len + extra_tokens
        self._rope_cos_2d = torch.empty(alloc_len, self.head_dim, dtype=cos.dtype, device=cos.device)
        self._rope_sin_2d = torch.empty(alloc_len, self.head_dim, dtype=sin.dtype, device=sin.device)
        self._rope_cos_2d[:q_len] = cos[0]  # [seq_len, head_dim]
        self._rope_sin_2d[:q_len] = sin[0]
        self._rope_cache_len = q_len

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
    # Step 1: QKV projection
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    # Step 2: RoPE + KV cache update
    cos, sin = position_embeddings
    if past_key_values is not None:
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_cached, value_cached = past_key_values.update(
            key_states, value_states, self.layer_idx, # cache_kwargs # commented out because we will compute rope table later.
        )
    else:
        key_cached, value_cached = key_states, value_states
    kv_len = key_cached.shape[2]
    
    # Check if DCT path is active
    min_len_for_paging = cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size
    
    # No need to do dct paging since there isn't enough tokens.
    if kv_len < min_len_for_paging:
        print("FallBackk!!!!!!!!!!!")
        all_pos = torch.arange(kv_len, device=key_cached.device)
        cos_all, sin_all = _compute_rope_cos_sin(
            all_pos, self.config, key_cached.device, key_cached.dtype
        )
        attn_q, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        attn_k = _apply_rope(key_cached, cos_all, sin_all)
        attn_v = value_cached

        attention_interface = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, attn_q, attn_k, attn_v, attention_mask,
            dropout=0.0, scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    # Use pre-RoPE KV for page building in continuous_rope mode
    key_states = key_cached
    value_states = value_cached
    
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

    page_scores = score_pages_triton(
        query_states, comp_k, cfg.scoring_method, cfg.group_agg_method, self.num_key_value_groups,
        out=self._page_scores_buf,
    )
    
    actual_top_k = min(cfg.top_k, num_pages)
    
    # Pre-allocate selected_indices buffer (constant shape across all decode steps)
    if not hasattr(self, '_topk_out_buf'):
        self._topk_out_buf = torch.empty(bsz, _num_kv_heads, actual_top_k, dtype=torch.int32, device=comp_k.device)

    selected_indices = topk_sort_triton(page_scores, actual_top_k, out=self._topk_out_buf)
    
    if cfg.unselected_mode == "drop":
        assembled_len = cfg.sink_size + actual_top_k * cfg.page_size + actual_recent
    else:
        num_unselected = num_pages - actual_top_k
        middle_len = actual_top_k * cfg.page_size + num_unselected * comp_size
        assembled_len = cfg.sink_size + middle_len + actual_recent

    cos_table = None
    sin_table = None
    cached_len = getattr(self, '_rope_cache_len', 0)
    if assembled_len > cached_len:
        max_len = assembled_len + cfg.page_size
        positions = torch.arange(max_len, device=comp_k.device)
        cos_cached, sin_cached = _compute_rope_cos_sin(
            positions, self.config, comp_k.device, comp_k.dtype,
        )
        # Store pre-squeezed 2D [max_len, head_dim] — eliminates [0,0,:N] indexing
        self._rope_cos_2d = cos_cached[0, 0] # [max_len, head_dim]
        self._rope_sin_2d = sin_cached[0, 0] # [max_len, head_dim]
        self._rope_cache_len = max_len
    # Pass full 2D table — kernel only reads positions 0..total_len-1
    cos_table = self._rope_cos_2d
    sin_table = self._rope_sin_2d

    # Pre-allocate or expand output buffers (avoids torch.empty per step)
    _buf_len = getattr(self, '_assemble_buf_len', 0)
    if assembled_len > _buf_len:
        _max_len = assembled_len + cfg.page_size
        _nkv = _num_kv_heads
        self._final_k_buf = torch.empty(bsz, _nkv, _max_len, self.head_dim, dtype=comp_k.dtype, device=comp_k.device)
        self._final_v_buf = torch.empty_like(self._final_k_buf)
        self._sel_idx_buf = torch.empty(bsz, _nkv, actual_top_k, dtype=torch.int32, device=comp_k.device)
        self._assemble_buf_len = _max_len

    # Pre-allocate Q-RoPE buffer
    if not hasattr(self, '_q_rope_buf'):
        self._q_rope_buf = torch.empty_like(query_states)
    q_rope_cos = self._rope_cos_2d[assembled_len - 1] # [head_dim] — 1D index, fast
    q_rope_sin = self._rope_sin_2d[assembled_len - 1] # [head_dim]

    # Step 6b: Assemble + K-RoPE (+ fused Q-RoPE in Kernel A for split mode)
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
        query_states = apply_rope_q_direct(query_states, q_rope_cos, q_rope_sin, self._q_rope_buf)
    
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
                query_states=query_states if cfg.continuous_rope else None,
            )

        # Fused assemble + Q-RoPE with stride cache
        final_k, final_v, q_rope_out = assemble_kv_split_triton(
            paged_k, paged_v, comp_k, comp_v,
            sink_k, sink_v, recent_k, recent_v,
            selected_indices,
            cos_table, sin_table,
            out_k=self._final_k_buf,
            out_v=self._final_v_buf,
            query_states=query_states,
            q_rope_cos=q_rope_cos,
            q_rope_sin=q_rope_sin,
            q_rope_buf=self._q_rope_buf,
            stride_cache=self._assemble_stride_cache,
        )
        query_states = q_rope_out

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
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",
    continuous_rope=True,
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
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        continuous_rope=continuous_rope,
        use_triton=use_triton,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config:")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}")
    print(f"  continuous_rope={continuous_rope}, use_triton={use_triton}")
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
    continuous_rope=True,
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
        scoring_method=scoring_method,
        group_agg_method=group_agg_method,
        unselected_mode=unselected_mode,
        continuous_rope=continuous_rope,
        use_triton=use_triton,
    )

    comp_size = max(1, int(page_size * compress_ratio))
    print(f"DCT Page Attention config (Llama):")
    print(f"  page_size={page_size}, top_k={top_k}")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")
    print(f"  compress_ratio={compress_ratio} ({page_size} -> {comp_size} tokens)")
    print(f"  scoring_method={scoring_method}, group_agg_method={group_agg_method}")
    print(f"  unselected_mode={unselected_mode}")
    print(f"  continuous_rope={continuous_rope}, use_triton={use_triton}")
    print(f"  Page attention active during decode only (prefill uses full attention)")

    transformers.models.llama.modeling_llama.LlamaAttention.forward = dct_page_attention_forward
