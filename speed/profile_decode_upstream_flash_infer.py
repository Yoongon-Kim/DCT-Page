"""
Profile DCT + upstream-FlashInfer decode (virtual-batch-per-(b, h) layout).

Sibling of `speed/profile_decode_flash_infer.py`. That driver uses the
DCT-Page fork of FlashInfer at `/home/yoongonkim/flashinfer-dct` with a
per-head `indices` patch (plan() `page_budget` kwarg). This driver tests
whether we can drop that patch entirely by reshaping the KV cache so each
physical page holds one (batch, KV head)'s slice, then treating each
(batch, KV head) pair as a virtual batch entry for stock FlashInfer's
2-D indices API.

Correctness argument: multi-head attention is separable over KV heads
AND batches — softmax is per Q head per batch. Packing "(batch b, KV head
h)'s selected pages" as virtual batch v = b*H + h, with the `group_size`
Q heads that attend to it as that batch's query heads, computes the exact
same attention output (up to FI kernel numerics).

Modes (via `--mode`, default `dct_upstream_flashinfer`):
  - baseline                   : full-KV FlashInfer (shared with the fork
                                  profiler — no per-head selection needed).
  - dct_sdpa                   : DCT + SDPA (pure reference).
  - dct_upstream_flashinfer    : DCT + upstream FI via virtual batching.
  - all                        : run all three back-to-back with comparison.

Usage:
    CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_upstream_flash_infer.py \\
        --context_length 32768 --page_size 32 --top_k 64 \\
        --num_sink_pages 1 --num_recent_pages 5 \\
        --num_decode_steps 128 --warmup_steps 8 \\
        --batch_size 2 --mode all --verify_upstream

Multibatch (Phase 1, v2): `--batch_size B` is now supported (B >= 1).
Memory cost scales linearly with B — see the v2 plan
(`.omc/plans/upstream-fi-multibatch-v2.md`) for the per-(B, ctx) ceiling
on A6000 48 GiB. Hard gate: bsz=4/16K verify PASS. Best-effort: bsz=4/32K.

Rollback path (if bsz>1 verify fails):
  - Single-line: pass `allow_head_local_multibatch=False` at the upstream
    call site in `upstream_flashinfer_backend.py`'s topk wrapper call —
    multibatch is un-armed, fork-side guardrail intact.
  - Full per-file revert: revert each phase commit independently;
    `triton_kernels.py` (Phase 3, kernel flag), then this driver
    (Phase 2), then `upstream_flashinfer_backend.py` (Phase 1).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn.functional as F
import transformers

import profile_decode as _pd
from profile_decode import (
    print_profile,
    profiled_dct_page_attention_forward,
)
# `pre_allocate_cache` MUST come from dct_page_attention (not profile_decode)
# because the dct_page_attention.PreAllocatedLayer.update has the _fi_mode
# counter-only shim that lets us free flat KV after FI build. The
# profile_decode.PreAllocatedLayer.update unconditionally writes to
# self.keys, which crashes with NoneType once flat KV is freed.

# Full-KV FI baseline lives locally in this module (see
# `build_baseline_upstream_fi_cache` and
# `profiled_baseline_upstream_flashinfer_forward` below) so that the upstream
# profiler has zero dependency on the fork profiler / fork backend's per-head
# `indices` patch. Only stock `BatchDecodeWithPagedKVCacheWrapper` is used.
from dataclasses import dataclass
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from speed_test_dummy import (
    load_model_and_tokenizer,
    get_original_forward,
    restore_forward,
    chunked_prefill,
)

import dct_page_attention as _dpa
from dct_page_attention import (
    apply_rotary_pos_emb,
    dct_page_attention_forward,
    pre_allocate_cache,
    replace_llama_attn,
    _maybe_reset_dct_runtime_state,
    _update_comp_cache,
)
from triton_kernels import (
    score_pages_triton,
    topk_sort_and_pack_triton,
)
from upstream_flashinfer_backend import (
    UpstreamFlashInferPagedKVCache,
    append_upstream_flashinfer_cache,
    build_upstream_flashinfer_paged_cache,
    refresh_upstream_indices_flat,
    upstream_flashinfer_decode_attention,
)


# ---------------------------------------------------------------------------
# Module-level ref so the forward can reach the cache without plumbing it
# through transformers' forward signature.
# ---------------------------------------------------------------------------
_upstream_fi_cache_ref = [None]


# ---------------------------------------------------------------------------
# Instrumented DCT + upstream-FI forward
# ---------------------------------------------------------------------------
def profiled_dct_upstream_flashinfer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Instrumented DCT + upstream-FlashInfer decode forward.

    Emits 8 chained CUDA events:
        1_qkv_proj
        2_rope_and_cache_append     (RoPE + past_key_values.update + FI
                                     counter advance on layer 0)
        3_segment                   (segment_kv)
        4_compress                  (_update_comp_cache)
        5_score_pages_kernel
        6_topk_and_pack             (fused Stage 5 kernel, head-local)
        7_upstream_fi_run           (indices bias + FI K/V write + wrapper.run)
        8_o_proj

    Verify path (`self._verify_upstream == True`): gather the SAME pages
    FI used (post-bias, per virtual batch) and run SDPA. Max-abs-diff is
    appended to `self._verify_diffs`. Gathered OUTSIDE the event window so
    it doesn't bias 7_upstream_fi_run.
    """
    cfg = _dpa._dct_page_cfg
    if cfg.unselected_mode != "drop":
        raise NotImplementedError("upstream-FI forward supports drop mode only")
    if cfg.continuous_rope:
        raise NotImplementedError("continuous_rope=True is temporarily disabled")

    input_shape = hidden_states.shape[:-1]
    bsz, q_len = input_shape
    _maybe_reset_dct_runtime_state(self, past_key_values)

    if q_len > 1:
        return dct_page_attention_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

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

    hidden_shape = (*input_shape, -1, self.head_dim)
    _has_qk_norm = hasattr(self, "q_norm") and hasattr(self, "k_norm")

    if _pd._enabled:
        _dev = hidden_states.device
        _stream = torch.cuda.current_stream(_dev)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
        _cpu_ts = []

        def _rec(i):
            if _pd._sync_mode:
                torch.cuda.synchronize(_dev)
            ev[i].record(_stream)
            if _pd._sync_mode:
                _cpu_ts.append(time.perf_counter())

        _rec(0)

    # Step 1: QKV projection.
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if _has_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _pd._enabled:
        _rec(1)

    # Step 2: RoPE + DCT cache update + FI counter advance on layer 0.
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # Counter-only update (in _fi_mode the shim returns (None, None) and
        # skips the flat-KV write — flat keys/values were freed at FI build).
        # The upstream forward never reads from past_key_values.layers[l].keys
        # in steady state; cache.buf_views[l] is the source of truth for K/V.
        past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    cache = _upstream_fi_cache_ref[0]
    if cache is None:
        raise RuntimeError(
            "upstream FI cache is not set. Build via "
            "build_upstream_flashinfer_paged_cache(...) and assign to "
            "_upstream_fi_cache_ref[0] before decode."
        )
    if cache.top_k != cfg.top_k:
        raise RuntimeError(
            f"cfg.top_k ({cfg.top_k}) != cache.top_k ({cache.top_k})"
        )

    if self.layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1
        cache.last_page_idx.fill_(cache.last_page_idx_py)
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    if _pd._enabled:
        _rec(2)

    # Step 3: paged views from FI buf (cache.buf_views[l] is the SoT — flat
    # KV was freed at FI build). Mirrors the fork's `paged_views_from_buf`
    # but for upstream's per-(b, h) layout: `cache.buf_views[l]` shape
    # `(B, H, P, 2, ps, 1, d)` slices to `(B, H, num_pages, ps, d)` with no
    # copy via stride-only indexing on the page dim.
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    num_pages = (
        cache.last_page_idx_py - cache.num_sink_pages
        - cache.num_recent_pages_fixed
    )
    buf_l = cache.buf_views[self.layer_idx]   # (B, H, P, 2, ps, 1, d)
    middle = buf_l[:, :, cache.num_sink_pages:cache.num_sink_pages + num_pages]
    paged_k = middle[:, :, :, 0, :, 0, :]   # (B, H, num_pages, ps, d)
    paged_v = middle[:, :, :, 1, :, 0, :]

    if _pd._enabled:
        _rec(3)

    # Step 4: compressed page cache (DCT proxy for scoring).
    comp_k, comp_v = _update_comp_cache(
        self, paged_k, paged_v, num_pages, comp_size, cfg,
    )

    if _pd._enabled:
        _rec(4)

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
        from dct_page_attention import _update_quest_metadata, _score_pages_quest
        quest_min_k, quest_max_k = _update_quest_metadata(self, paged_k, num_pages)
        page_scores = _score_pages_quest(
            query_states, quest_min_k, quest_max_k,
            cfg.group_agg_method, self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )
    else:
        page_scores = score_pages_triton(
            query_states, comp_k,
            cfg.scoring_method, cfg.group_agg_method,
            self.num_key_value_groups,
            out=self._page_scores_buf[:, :, :num_pages],
        )

    if _pd._enabled:
        _rec(5)

    # Step 6: fused topk + pack. Writes head-local indices into
    # `indices_buf_3d`; sink slice was filled once at cache init.
    # `num_pages` already equals the eligible middle range (cache.buf_views
    # is the source of truth, no off-by-one between DCT and FI counts).
    if num_pages < cache.top_k:
        raise RuntimeError(
            f"num_pages ({num_pages}) < cache.top_k "
            f"({cache.top_k}). Configure min_decode_kv_len_for_paging."
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
        # upstream uses per-(b, h) pools — bias is applied later by
        # refresh_upstream_indices_flat, so the kernel writes head-local IDs.
        # `pages_per_batch=0` + this flag is the upstream contract.
        pages_per_batch=0,
        allow_head_local_multibatch=True,
    )

    if _pd._enabled:
        _rec(6)

    # Step 7: bias indices to physical pages + FI K/V write + wrapper.run.
    page_idx = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    if page_idx >= cache.pages_per_head:
        raise RuntimeError(
            f"upstream-FI cache overflow: page_idx={page_idx} >= "
            f"pages_per_head={cache.pages_per_head}"
        )
    k_flat = key_states[:, :, -1:, :].reshape(bsz, cache.num_kv_heads, cache.head_dim)
    v_flat = value_states[:, :, -1:, :].reshape(bsz, cache.num_kv_heads, cache.head_dim)
    # buf_views[l]: (B, H, P, 2, ps, 1, d). Touch all (b, h) at (page_idx, slot).
    cache.buf_views[self.layer_idx][:, :, page_idx, 0, slot, 0, :].copy_(k_flat)
    cache.buf_views[self.layer_idx][:, :, page_idx, 1, slot, 0, :].copy_(v_flat)

    refresh_upstream_indices_flat(cache)
    attn_output_fi = upstream_flashinfer_decode_attention(
        query_states, cache, self.layer_idx,
    )

    if _pd._enabled:
        _rec(7)

    # Verify path — outside the event window. Recreates the same K/V set that
    # FI saw from the cache and runs SDPA on it. Each (b, h) virtual batch
    # has its own page pool in buf_views, so gather is per-(b, h) directly.
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

    # Step 8: output projection.
    attn_output = attn_output_fi.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if _pd._enabled:
        _rec(8)
        step_names = [
            "1_qkv_proj",
            "2_rope_and_cache_append",
            "3_segment",
            "4_compress",
            "5_score_pages_kernel",
            "6_topk_and_pack",
            "7_upstream_fi_run",
            "8_o_proj",
        ]
        for i, name in enumerate(step_names):
            _pd._pending_events.append((name, ev[i], ev[i + 1]))
        if _pd._sync_mode:
            for i, name in enumerate(step_names):
                cpu_ms = (_cpu_ts[i + 1] - _cpu_ts[i]) * 1000
                _pd._cpu_timings[name].append(cpu_ms)

    _pd._current_layer += 1
    return attn_output, None


# ---------------------------------------------------------------------------
# Build / teardown
# ---------------------------------------------------------------------------
def _build_upstream_fi_cache(model, past_key_values, prefill_len, args):
    cfg_model = model.config
    num_kv_heads = cfg_model.num_key_value_heads
    num_qo_heads = cfg_model.num_attention_heads
    head_dim = cfg_model.hidden_size // num_qo_heads
    num_layers = cfg_model.num_hidden_layers
    num_sink_pages = args.num_sink_pages
    num_recent_pages_fixed = args.num_recent_pages
    max_decode_steps = args.warmup_steps + args.num_decode_steps + 16
    if args.cudagraph:
        max_decode_steps += 64
    page_budget = num_sink_pages + args.top_k + num_recent_pages_fixed + 1
    bsz = args.batch_size
    vbsz = bsz * num_kv_heads
    print(
        f"  Building upstream-FI cache: layers={num_layers}, bsz={bsz}, "
        f"num_sink_pages={num_sink_pages}, top_k={args.top_k}, "
        f"num_recent_pages_fixed={num_recent_pages_fixed}, "
        f"page_budget={page_budget}, vbsz={vbsz} (=B*H), "
        f"group_size={num_qo_heads // num_kv_heads}..."
    )
    device = next(model.parameters()).device
    cache = build_upstream_flashinfer_paged_cache(
        preallocated_layers=past_key_values.layers,
        prefill_len=prefill_len,
        page_size=args.page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_layers=num_layers,
        max_decode_steps=max_decode_steps,
        dtype=past_key_values.layers[0].keys.dtype,
        device=device,
        num_sink_pages=num_sink_pages,
        top_k=args.top_k,
        num_recent_pages_fixed=num_recent_pages_fixed,
        bsz=bsz,
    )
    _upstream_fi_cache_ref[0] = cache
    print(
        f"  upstream-FI cache ready: pages_per_head={cache.pages_per_head}, "
        f"total_pages={cache.total_pages}, cur_seqlen={cache.cur_seqlen}, "
        f"last_page_idx={cache.last_page_idx_py}, "
        f"last_page_len={cache.last_page_len_py}"
    )
    return cache


# ---------------------------------------------------------------------------
# Full-KV upstream-FlashInfer baseline (self-contained: no fork dep)
# ---------------------------------------------------------------------------
# Mirrors the layout of the fork profiler's full-KV baseline but lives entirely
# inside this module so the upstream profiler is decoupled. Only stock
# `BatchDecodeWithPagedKVCacheWrapper` APIs (plan/run + 2-D indices) are used —
# no per-head `indices` / `page_budget` kwargs from the fork.
#
# Layout per layer:
#   buf[l]: (capacity_pages, 2, page_size, num_kv_heads, head_dim)
#   capacity_pages = bsz * pages_per_batch.
#   Batch b owns physical pages [b*pages_per_batch, (b+1)*pages_per_batch).
#
# plan() is re-called inside the forward each time a new physical page opens
# (every page_size decode steps), since `indptr_buf` stride changes with
# `num_active_pages`. Between page boundaries only `last_page_len_buf` mutates.
# `use_cuda_graph=True` pins indptr / indices / last_page_len buffers so replan
# is an in-place scheduler refresh.
# ---------------------------------------------------------------------------
@dataclass
class FullKVUpstreamFIBaselineCache:
    buf: list  # list[Tensor], one (capacity_pages, 2, ps, nkv, d) per layer
    wrapper: BatchDecodeWithPagedKVCacheWrapper

    bsz: int
    page_size: int
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int
    num_layers: int
    capacity_pages: int
    pages_per_batch: int

    dtype: torch.dtype
    device: torch.device

    indices_buf: torch.Tensor          # (bsz * pages_per_batch,) int32
    indptr_buf: torch.Tensor           # (bsz+1,) int32
    last_page_len_buf: torch.Tensor    # (bsz,) int32

    cur_seqlen: int = 0
    last_page_idx_py: int = 0   # per-batch LOGICAL page index (lockstep)
    last_page_len_py: int = 0
    num_active_pages: int = 0   # per-batch active page count (lockstep)


def _pack_preallocated_to_paged_per_layer(
    preallocated_layers,
    prefill_len: int,
    page_size: int,
    capacity_pages: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    bsz: int,
    pages_per_batch: int,
):
    """Pack PreAllocatedLayer → paged buf one layer at a time, freeing each
    layer's flat keys/values immediately after pack. Pure-torch (no FI ops).

    Mirrors `flashinfer_backend._build_paged_buf_per_layer` — copied locally to
    sever the upstream profiler's dependency on the fork backend. `_fi_mode`
    flips on each layer so subsequent `PreAllocatedLayer.update` calls in
    counter-only mode skip writes to the freed flat KV.
    """
    prefill_pages = (prefill_len + page_size - 1) // page_size
    pad = prefill_pages * page_size - prefill_len
    bufs = []
    for layer in preallocated_layers:
        layer_buf = torch.zeros(
            capacity_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        for b in range(bsz):
            k = layer.keys[b, :, :prefill_len, :]
            v = layer.values[b, :, :prefill_len, :]
            if pad:
                k = torch.nn.functional.pad(k, (0, 0, 0, pad))
                v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            k = k.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(dtype)
            v = v.view(num_kv_heads, prefill_pages, page_size, head_dim).permute(1, 2, 0, 3).to(dtype)
            base = b * pages_per_batch
            layer_buf[base:base + prefill_pages, 0].copy_(k)
            layer_buf[base:base + prefill_pages, 1].copy_(v)
        bufs.append(layer_buf)
        layer._fi_mode = True
        layer.keys = None
        layer.values = None
    return bufs


def build_baseline_upstream_fi_cache(
    preallocated_layers,
    prefill_len: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    num_qo_heads: int,
    num_layers: int,
    max_decode_steps: int,
    dtype: torch.dtype,
    device: torch.device,
    bsz: int = 1,
    workspace_bytes: int = 128 * 1024 * 1024,
) -> FullKVUpstreamFIBaselineCache:
    """Pack prefill into an FI paged buffer and plan a full-KV decode wrapper
    against stock FlashInfer. Per-layer pack frees flat KV one layer at a time
    so peak transient stays ~one layer's worth above steady-state paged size.
    """
    prefill_pages = (prefill_len + page_size - 1) // page_size
    decode_pages = (max_decode_steps + page_size - 1) // page_size
    pages_per_batch = prefill_pages + decode_pages + 4
    capacity_pages = bsz * pages_per_batch

    buf = _pack_preallocated_to_paged_per_layer(
        preallocated_layers, prefill_len, page_size, capacity_pages,
        num_kv_heads, head_dim, dtype, device,
        bsz=bsz, pages_per_batch=pages_per_batch,
    )
    torch.cuda.empty_cache()

    base_arange = torch.arange(pages_per_batch, dtype=torch.int32, device=device)
    batch_offsets = (
        torch.arange(bsz, dtype=torch.int32, device=device) * pages_per_batch
    )
    indices_buf = (
        base_arange[None, :] + batch_offsets[:, None]
    ).contiguous().view(-1)

    indptr_buf = (
        torch.arange(bsz + 1, dtype=torch.int32, device=device) * prefill_pages
    ).contiguous()

    last_open_page = (prefill_len - 1) // page_size
    last_open_len = prefill_len - last_open_page * page_size
    last_page_len_buf = torch.full(
        (bsz,), last_open_len, dtype=torch.int32, device=device,
    )

    float_workspace_buffer = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device,
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_page_len_buf,
    )
    active_slab = (
        base_arange[:prefill_pages][None, :] + batch_offsets[:, None]
    ).contiguous().view(-1)
    wrapper.plan(
        indptr_buf, active_slab, last_page_len_buf,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_data_type=dtype, kv_data_type=dtype,
    )

    return FullKVUpstreamFIBaselineCache(
        buf=buf, wrapper=wrapper,
        bsz=bsz,
        page_size=page_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        num_qo_heads=num_qo_heads, num_layers=num_layers,
        capacity_pages=capacity_pages,
        pages_per_batch=pages_per_batch,
        dtype=dtype, device=device,
        indices_buf=indices_buf, indptr_buf=indptr_buf,
        last_page_len_buf=last_page_len_buf,
        cur_seqlen=prefill_len,
        last_page_idx_py=last_open_page,
        last_page_len_py=last_open_len,
        num_active_pages=prefill_pages,
    )


_fi_baseline_cache_ref = [None]


def profiled_baseline_upstream_flashinfer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Full-KV (no page selection) decode via stock upstream FlashInfer paged
    decode. Mirrors `profiled_baseline_flashinfer_forward` from the fork
    profiler but uses only stock `BatchDecodeWithPagedKVCacheWrapper` APIs.

    4 chained events (same labels as the fork-profiler baseline so the
    summary table aligns):
        1_qkv_proj
        2_rope_and_cache_append
        8_flashinfer_run
        9_o_proj
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    if q_len > 1:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = _pd._get_attention_interface(self)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None), **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ---- DECODE PATH (q_len == 1) ----
    cache = _fi_baseline_cache_ref[0]
    if cache is None:
        raise RuntimeError(
            "Full-KV upstream-FI baseline cache is not set. Build via "
            "build_baseline_upstream_fi_cache(...) post-prefill and assign to "
            "_fi_baseline_cache_ref[0] before decode."
        )

    if _pd._enabled:
        _dev = hidden_states.device
        _stream = torch.cuda.current_stream(_dev)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        ev[0].record(_stream)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if _pd._enabled:
        ev[1].record(_stream)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_bsz = cache.bsz
    if self.layer_idx == 0:
        if cache.last_page_len_py == cache.page_size:
            cache.last_page_idx_py += 1
            cache.last_page_len_py = 0
            cache.num_active_pages += 1
            if cache.num_active_pages > cache.pages_per_batch:
                raise RuntimeError(
                    f"FullKVUpstreamFIBaselineCache overflow: "
                    f"num_active_pages={cache.num_active_pages} > "
                    f"pages_per_batch={cache.pages_per_batch}. Increase "
                    f"max_decode_steps at build time."
                )
            new_indptr = (
                torch.arange(
                    cache_bsz + 1, dtype=torch.int32, device=cache.device,
                ) * cache.num_active_pages
            )
            cache.indptr_buf.copy_(new_indptr)
            indices_view = cache.indices_buf.view(cache_bsz, cache.pages_per_batch)
            active_slab = indices_view[:, :cache.num_active_pages].contiguous().view(-1)
            cache.wrapper.plan(
                cache.indptr_buf,
                active_slab,
                cache.last_page_len_buf,
                cache.num_qo_heads, cache.num_kv_heads, cache.head_dim,
                cache.page_size,
                q_data_type=cache.dtype, kv_data_type=cache.dtype,
            )
        cache.last_page_len_py += 1
        cache.cur_seqlen += 1
        cache.last_page_len_buf.fill_(cache.last_page_len_py)

    logical_page = cache.last_page_idx_py
    slot = cache.last_page_len_py - 1
    layer_buf = cache.buf[self.layer_idx]
    if cache_bsz == 1:
        k_flat = key_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
        v_flat = value_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
        layer_buf[logical_page, 0, slot].copy_(k_flat)
        layer_buf[logical_page, 1, slot].copy_(v_flat)
    else:
        phys = (
            torch.arange(cache_bsz, dtype=torch.long, device=cache.device)
            * cache.pages_per_batch + logical_page
        )
        k_flat = key_states[:, :, -1:, :].reshape(cache_bsz, cache.num_kv_heads, cache.head_dim)
        v_flat = value_states[:, :, -1:, :].reshape(cache_bsz, cache.num_kv_heads, cache.head_dim)
        layer_buf[phys, 0, slot] = k_flat
        layer_buf[phys, 1, slot] = v_flat

    if _pd._enabled:
        ev[2].record(_stream)

    q_flat = query_states.reshape(cache_bsz, cache.num_qo_heads, cache.head_dim).contiguous()
    attn_output = cache.wrapper.run(q_flat, cache.buf[self.layer_idx])
    attn_output = attn_output.view(cache_bsz, cache.num_qo_heads, 1, cache.head_dim)

    if _pd._enabled:
        ev[3].record(_stream)

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if _pd._enabled:
        ev[4].record(_stream)
        step_names = [
            "1_qkv_proj",
            "2_rope_and_cache_append",
            "8_flashinfer_run",
            "9_o_proj",
        ]
        for i, name in enumerate(step_names):
            _pd._pending_events.append((name, ev[i], ev[i + 1]))

    return attn_output, None


def _reset_mode_state():
    _upstream_fi_cache_ref[0] = None
    _fi_baseline_cache_ref[0] = None
    _pd._quest_cache_ref[0] = None
    _pd._step_timings.clear()
    _pd._cpu_timings.clear()
    _pd._pending_events.clear()
    _pd._enabled = False
    _pd._current_layer = 0


def _probe_event_record_in_graph(device) -> bool:
    """Returns True iff cudaEventRecord inside torch.cuda.graph capture
    is supported on this torch+CUDA+driver combo. Probe is FI-state-free
    (two events + a no-op graph on an explicit non-default stream).

    Dual-purpose: gate for Option A (events) AND a documented capability
    stub. See feedback_event_in_graph_unsupported.md."""
    try:
        probe_g = torch.cuda.CUDAGraph()
        probe_s = torch.cuda.Event(enable_timing=True)
        probe_e = torch.cuda.Event(enable_timing=True)
        s_probe = torch.cuda.Stream(device=device)
        s_probe.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s_probe), torch.cuda.graph(probe_g):
            probe_s.record()
            probe_e.record()
        torch.cuda.current_stream(device).wait_stream(s_probe)
        probe_g.replay()
        torch.cuda.synchronize(device)
        # Back-to-back records can legitimately yield 0.0 ms; gate on < 0.
        ok = probe_s.elapsed_time(probe_e) >= 0
        # C3: explicit cleanup of probe locals so the empty captured graph
        # and its events drop their refs immediately rather than at GC.
        del probe_g, probe_s, probe_e, s_probe
        return ok
    except Exception:
        return False


class _CudagraphBreakdownToggle:
    """Enable `_pd._enabled` (with `_pd._sync_mode` forced off) for the
    duration of a CUDA-graph priming + capture block, so the chained-event
    chain in `profiled_dct_upstream_flashinfer_forward` records into the
    captured graph.

    Do NOT call `_pd._flush_events()` inside the cudagraph block — the
    read-after-final-replay pattern requires `_pending_events` to retain
    its capture-time entries until the explicit walk. `_flush_events()`
    clears the list and would break the readout.
    """

    def __enter__(self):
        self._old_enabled = _pd._enabled
        self._old_sync_mode = _pd._sync_mode
        _pd._enabled = True
        _pd._sync_mode = False
        _pd._pending_events.clear()  # discard stale events from previous mode
        return self

    def __exit__(self, exc_type, exc, tb):
        _pd._enabled = self._old_enabled
        _pd._sync_mode = self._old_sync_mode
        return False


def _fi_cache_for_mode(mode):
    """Return the active FI cache for the given mode, or None.

    Used by the CUDA-graph block to align `last_page_len_py` away from a page
    boundary before priming + capture. Both FI-based modes
    (`dct_upstream_flashinfer` and `baseline`) need the alignment hook
    although only `baseline` actually replans inside the forward; for the
    upstream-FI mode plan() is called once at build, so alignment is a no-op.
    """
    if mode == "dct_upstream_flashinfer":
        return _upstream_fi_cache_ref[0]
    if mode == "baseline":
        return _fi_baseline_cache_ref[0]
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Profile DCT + upstream-FlashInfer decode "
                    "(virtual-batch-per-head layout, no custom fork patch)."
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--context_length", type=int, default=32768)
    p.add_argument("--num_decode_steps", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=8)
    p.add_argument(
        "--mode",
        choices=["baseline", "dct_sdpa", "dct_upstream_flashinfer", "all"],
        default="dct_upstream_flashinfer",
    )

    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--num_sink_pages", type=int, default=1)
    p.add_argument("--num_recent_pages", type=int, default=4)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--scoring_method", default="max", choices=["mean", "max"])
    p.add_argument("--group_agg_method", default="max", choices=["mean", "max"])
    p.add_argument("--unselected_mode", default="drop", choices=["drop"])
    p.add_argument("--compressed_token_rope", default="mixed",
                   choices=["mixed", "block_center"])
    p.add_argument("--comp_kv_quant", default="fp8_e5m2",
                   choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"])
    p.add_argument("--comp_kv_quant_granularity", default="per_page",
                   choices=["per_page", "per_comp_token"])
    p.add_argument("--no_triton", action="store_true")
    p.add_argument("--chunk_size", type=int, default=0)
    p.add_argument("--sync", action="store_true")
    p.add_argument("--topk_impl",
                   choices=["auto", "fused", "twostage", "torch"],
                   default="auto")

    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size (B). vbsz = B * num_kv_heads — each (b, h) "
                        "becomes a virtual FlashInfer batch entry with its own "
                        "page pool. Memory cost scales linearly with B; on a "
                        "48 GiB A6000, bsz=4/16K is the safe gate, bsz=4/32K "
                        "is best-effort (per memory math in the v2 plan).")
    p.add_argument("--verify_upstream", action="store_true",
                   help="Per-layer upstream-FI vs SDPA shadow verification. "
                        "At bsz=1 uses --verify_threshold (default 0.05 — the "
                        "empirical bf16 LSB floor at typical attention output "
                        "magnitudes ~ 2^-5); at bsz>1 a ladder threshold = "
                        "max(--verify_threshold, 3 * wd1) is used, where "
                        "wd1 = bsz=1 empirical worst diff (logged at run time).")
    p.add_argument("--verify_threshold", type=float, default=0.05,
                   help="Floor for the verify max-abs-diff at bsz=1. At "
                        "bsz>1 the threshold is max(this, 3 * wd1). 0.05 "
                        "matches the empirical bf16 noise floor; tighten to "
                        "0.02 only if you've also tightened FI/SDPA reduction "
                        "order to match exactly.")
    p.add_argument("--bsz1_wd1", type=float, default=0.0,
                   help="Empirical worst max-abs-diff observed at bsz=1, used "
                        "as the bsz>1 threshold ladder floor. Run bsz=1 verify "
                        "first to record this; pass it on bsz>1 runs to set "
                        "threshold = max(--verify_threshold, 3 * wd1). "
                        "0.0 (default) → ladder degenerates to --verify_threshold.")
    p.add_argument("--cudagraph", action="store_true",
                   help="Capture one decode step into a CUDA graph and "
                        "benchmark replay. plan() is called once at build "
                        "time so graph capture is safe by construction.")
    p.add_argument("--cudagraph_replays", type=int, default=0)
    p.add_argument("--torch_profiler_trace", default=None)
    p.add_argument("--cudagraph_breakdown", action="store_true",
                   help="Capture per-attention-substep CUDA events inside "
                        "the captured graph and report them after replay. "
                        "Implies `--cudagraph`. Forces `--sync` off inside "
                        "capture (sync is not graph-capturable). Sibling "
                        "driver `profile_decode_flash_infer.py` lacks the "
                        "same flag — fork-FI variant is a follow-up PR.")
    p.add_argument("--cudagraph_breakdown_method",
                   choices=["profiler", "events"], default="profiler",
                   help="Breakdown mechanism. `profiler` (default): wrap "
                        "timed g.replay() loop in torch.profiler, bucket "
                        "kernel-level CUPTI activity into attention substeps "
                        "by kernel name. `events`: in-graph CUDA events "
                        "(Option A — dormant on this build per "
                        "feedback_event_in_graph_unsupported.md; auto-rearms "
                        "if a future torch/CUDA bump re-enables event-record "
                        "during graph capture).")
    p.add_argument("--cudagraph_breakdown_disambig",
                   choices=["merge", "ordering"], default="merge",
                   help="cublasLt gemm bucket strategy (profiler method only). "
                        "`merge` (default): collapse ALL cublasLt gemms (qkv + "
                        "o_proj + MLP gate/up/down + lm_head) into one "
                        "`gemm_total` bucket — fully kernel-name grounded but "
                        "not attention-scoped. `ordering`: split into `1_qkv_proj` and "
                        "`8_o_proj` via per-launch ordering within each layer "
                        "window (boundary marker = `_score_pages_*_kernel`). "
                        "8 rows matching eager column shape. Heuristic — "
                        "see Step 4 comment block for the off-by-one rule. "
                        "Auto-falls back to `merge` if cublasLt-count canary "
                        "fails.")
    p.add_argument("--cudagraph_breakdown_dump_kernels",
                   action="store_true", default=False,
                   help="Diagnostic: after replay, print all CUPTI kernel names "
                        "+ counts from prof.events() for one replay window. "
                        "Use when the residual line warns the bucket dict is "
                        "stale (kernel renames across torch/triton/flashinfer "
                        "versions).")
    p.add_argument("--cudagraph_breakdown_seal_microbench",
                   action="store_true", default=False,
                   help="[EAGER MODE] After graph capture, run a same-process "
                        "eager forced-seal microbench of _update_comp_cache "
                        "(no CUDA graph) to produce a truth row for the "
                        "4_compress reconciliation block. See Plan v3 "
                        "(cudagraph-substep-breakdown.md). No-op when flag is absent "
                        "(byte-identical to pre-edit default output).")
    p.add_argument("--cudagraph_breakdown_seal_microbench_iters",
                   type=int, default=100,
                   help="Number of timed forced-seal _update_comp_cache calls "
                        "(default 100). Higher = lower variance.")
    p.add_argument("--cudagraph_breakdown_seal_microbench_warmup",
                   type=int, default=5,
                   help="Discarded warmup forced-seal calls before timing begins "
                        "(default 5). Absorbs slow-path realloc + Triton JIT.")
    p.add_argument("--cudagraph_capture_with_seal", action="store_true",
                   default=False,
                   help="[IN-GRAPH MODE] Capture a SLOW-PATH (sealing) decode "
                        "step instead of the default fast-path step. Just "
                        "before capture, force `_comp_n_pages_cached -= 1` on "
                        "every DCT layer so n_new=1 fires and the captured "
                        "graph contains the compress GEMM + page-copy "
                        "kernels. Per-replay `4_compress` then reports the "
                        "in-graph cost (per replay = AS IF every step "
                        "sealed); the printer also shows the amortized value "
                        "(graph_4c / page_size). Replays write to a fixed "
                        "cache slot every time, so this is for TIMING "
                        "measurement only — never use for correctness runs. "
                        "Implies --cudagraph.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------
def _rebind_instance_forward(model, attn_cls, forward_fn):
    for module in model.modules():
        if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
            module._old_forward = types.MethodType(forward_fn, module)


def _patch_baseline(model, args, original_forward):
    """Full-KV FI baseline using stock upstream FlashInfer (defined locally
    in this module)."""
    restore_forward(args.model, original_forward, model)
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    attn_cls.forward = profiled_baseline_upstream_flashinfer_forward
    _rebind_instance_forward(model, attn_cls, profiled_baseline_upstream_flashinfer_forward)


def _patch_dct_sdpa(model, args, original_forward):
    restore_forward(args.model, original_forward, model)
    replace_llama_attn(
        page_size=args.page_size, top_k=args.top_k,
        num_sink_pages=args.num_sink_pages, num_recent_pages=args.num_recent_pages,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        compressed_token_rope=args.compressed_token_rope,
        use_triton=not args.no_triton,
        weight_compressed_by_population=True,
        comp_kv_quant=args.comp_kv_quant,
        comp_kv_quant_granularity=args.comp_kv_quant_granularity,
        attention_backend="sdpa",
    )
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    attn_cls.forward = profiled_dct_page_attention_forward
    _rebind_instance_forward(model, attn_cls, profiled_dct_page_attention_forward)


def _patch_dct_upstream_flashinfer(model, args, original_forward):
    """Patch forward to the upstream-FI profiled path. `replace_llama_attn`
    is called with `attention_backend='sdpa'` — the DCT monkey-patch only
    needs the config object to be set up; our custom forward overrides the
    attention entirely so the backend selection inside `replace_llama_attn`
    is moot for this mode.
    """
    restore_forward(args.model, original_forward, model)
    replace_llama_attn(
        page_size=args.page_size, top_k=args.top_k,
        num_sink_pages=args.num_sink_pages, num_recent_pages=args.num_recent_pages,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        compressed_token_rope=args.compressed_token_rope,
        use_triton=not args.no_triton,
        weight_compressed_by_population=True,
        comp_kv_quant=args.comp_kv_quant,
        comp_kv_quant_granularity=args.comp_kv_quant_granularity,
        attention_backend="sdpa",
    )
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    attn_cls.forward = profiled_dct_upstream_flashinfer_forward
    _rebind_instance_forward(model, attn_cls, profiled_dct_upstream_flashinfer_forward)


# ---------------------------------------------------------------------------
# Eager forced-seal microbench for 4_compress reconciliation (Plan v3).
#
# Why needed: the captured graph always replays non-sealing steps (alignment
# ensures last_page_len <= page_size - 4), so bucket 4_compress in the graph
# row is structurally ~0.  Without a direct measurement the reconciliation
# table would show "graph ≈0, eager-avg ≈ t_seal/ps" and imply the graph
# eliminated compression cost — misleading.  This helper measures the actual
# per-seal GPU time directly by forcing n_new=1 each iteration.
#
# Runs AFTER graph capture+replay, BEFORE the printer call.
# Eager mode means it uses the standard caching allocator — no graph
# stream-private pool hazard.
# ---------------------------------------------------------------------------
def _run_seal_microbench(model, fi_cache, attn_modules, args, num_layers):
    # _dct_page_cfg is a module-level global, NOT an attn_module attribute.
    from dct_page_attention import _dct_page_cfg as cfg
    if cfg is None:
        print("[ERROR] seal microbench: DCT patch not active (_dct_page_cfg is None)")
        return None
    if not attn_modules:
        print("[ERROR] seal microbench: no DCT attention modules found; mode is likely baseline")
        return None

    # Use layer-0 attn_module for determinism across torch versions / refactors.
    attn_module = attn_modules[0]

    # Derive arguments the forward uses at decode time (drop mode only here;
    # argparse restricts choices=["drop"] so compressed path is unreachable).
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    page_size = cfg.page_size

    # Build paged_k / paged_v views from the live FI cache buffer.
    # fi_cache.buf_views[0] is 7-D: (B, H, P, 2, ps, 1, d); see
    # speed/upstream_flashinfer_backend.py:93. Axis 3: 0=K, 1=V. Axis 5 is a
    # singleton retained for the FI 7-D layout — index with 0 to drop it so
    # _update_comp_cache (which expects 5-D [B, H, num_pages, ps, d]) is happy.
    buf = fi_cache.buf_views[0]
    sink = cfg.num_sink_pages
    recent = cfg.num_recent_pages
    last_idx = fi_cache.last_page_idx_py  # index of the currently-open page
    # num_pages = pages EXCLUDING sink and recent (the "paged" region scored by DCT).
    num_pages = last_idx - sink - recent + 1
    if num_pages <= 0:
        print(f"[INFO] seal microbench: num_pages={num_pages} <= 0; context too short "
              f"for seal microbench (need at least sink+recent+1 pages). Skipping.")
        return None

    paged_k = buf[:, :, sink:sink + num_pages, 0, :, 0, :]  # [B, H, num_pages, ps, d]
    paged_v = buf[:, :, sink:sink + num_pages, 1, :, 0, :]  # [B, H, num_pages, ps, d]

    # Stash original cache state so we can restore it after the microbench.
    original_n_cached = getattr(attn_module, '_comp_n_pages_cached', 0)

    iters = args.cudagraph_breakdown_seal_microbench_iters
    warmup = args.cudagraph_breakdown_seal_microbench_warmup

    # Warmup: absorbs slow-path realloc (first fire) + Triton JIT compilation.
    # Discard timing.
    for _ in range(warmup):
        attn_module._comp_n_pages_cached = num_pages - 1  # force n_new=1
        _update_comp_cache(attn_module, paged_k, paged_v, num_pages, comp_size, cfg)
    torch.cuda.synchronize()

    # Timed loop: single event pair around all `iters` calls, divide at end.
    # Slow-path auto-writes _comp_n_pages_cached = num_pages at L722 of
    # dct_page_attention.py; the manual reset to num_pages-1 inside the loop
    # is what forces n_new=1 each iteration.
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        attn_module._comp_n_pages_cached = num_pages - 1
        _update_comp_cache(attn_module, paged_k, paged_v, num_pages, comp_size, cfg)
    end_ev.record()
    torch.cuda.synchronize()

    t_seal_per_layer_ms = start_ev.elapsed_time(end_ev) / iters

    # Defensive teardown: restore both _comp_n_pages_cached AND _last_comp_kv.
    # The slow-path writes _last_comp_kv = result (dct_page_attention.py:740/753);
    # the live-forward fast-path (L563) returns it without recomputing.  Without
    # clearing _last_comp_kv, the next real decode step receives microbench-mutated
    # cache identity.
    attn_module._comp_n_pages_cached = original_n_cached
    attn_module._last_comp_kv = None

    amortized_per_layer_ms = t_seal_per_layer_ms / page_size
    amortized_total_ms = amortized_per_layer_ms * num_layers

    return {
        "t_seal_per_layer_ms": t_seal_per_layer_ms,
        "iters": iters,
        "page_size": page_size,
        "num_layers": num_layers,
        "amortized_per_step_per_layer_ms": amortized_per_layer_ms,
        "amortized_per_step_total_ms": amortized_total_ms,
    }


# ---------------------------------------------------------------------------
# CUDA-graph per-substep breakdown printer (adjacent to the --cudagraph block)
# ---------------------------------------------------------------------------
def _print_graph_breakdown(
    per_replay_ms,
    substep_per_token,
    num_layers,
    bsz,
    mode="dct_upstream_flashinfer",
    all_kernel_sum_ms_per_step=None,
    eager_per_token=None,
    seal_microbench=None,
    seal_capture_active=False,
    page_size=None,
):
    print(f"\n{'=' * 70}")
    print(f"PROFILE: {mode.upper()} (CUDA GRAPH)")
    print(f"{'=' * 70}")
    # C2 legend: distinguishes the two reconciliation lines that follow.
    # `non_attn_residual` ≈ MLP + layernorm + lm_head (real, non-attention work
    #     captured by per_replay but not by the attention-bucket sum).
    # `residual` (All-kernel reconciliation) ≈ unbucketed CUPTI activity time
    #     vs per_replay; this is a *bucketing integrity check* — large values
    #     mean the kernel-name → bucket dict missed kernels (likely renamed),
    #     not real un-instrumented work.
    if bsz == 1:
        print(f"  Attention total (graph): "
              f"{sum(substep_per_token.values()):.3f} ms/step")
        print(f"  Per-replay (graph):      {per_replay_ms:.3f} ms/step")
    else:
        agg = bsz / per_replay_ms * 1000.0
        print(f"  Attention total (graph): "
              f"{sum(substep_per_token.values()):.3f} ms/step")
        print(f"  Per-replay (graph):      {per_replay_ms:.3f} ms/step  "
              f"({agg:.2f} agg tok/s @ bsz={bsz})")
    print()
    attn_total = sum(substep_per_token.values())
    print(f"  {'Step':<25} {'Per-token (ms)':>15} {'% of attn':>12}")
    print(f"  {'-' * 25} {'-' * 15} {'-' * 12}")
    for name in sorted(substep_per_token.keys()):
        per_token = substep_per_token[name]
        pct = per_token / attn_total * 100 if attn_total > 0 else 0.0
        print(f"  {name:<25} {per_token:>15.3f} {pct:>11.1f}%")
    print(f"  {'-' * 25} {'-' * 15} {'-' * 12}")
    print(f"  {'TOTAL':<25} {attn_total:>15.3f} {'100.0':>11}%")
    residual = max(0.0, per_replay_ms - attn_total)
    pct = (residual / per_replay_ms * 100) if per_replay_ms > 0 else 0.0
    # mod 8 — rename: this is the attention-only reconciliation (the residual
    # here is real non-attention work like MLP/layernorm/lm_head).
    print(
        f"  Graph-mode reconciliation (attn-only): "
        f"substep_sum={attn_total:.3f} ms/step, "
        f"per_replay={per_replay_ms:.3f} ms/step, "
        f"non_attn_residual={residual:.3f} ms ({pct:.1f}%)"
    )
    # NEW: All-kernel reconciliation = bucketing integrity check.
    if all_kernel_sum_ms_per_step is not None:
        residual_full = max(0.0, per_replay_ms - all_kernel_sum_ms_per_step)
        pct_full = (residual_full / per_replay_ms * 100) if per_replay_ms > 0 else 0.0
        print(
            f"  All-kernel reconciliation: "
            f"all_kernel_sum={all_kernel_sum_ms_per_step:.3f} ms/step, "
            f"per_replay={per_replay_ms:.3f} ms/step, "
            f"residual={residual_full:.3f} ms ({pct_full:.1f}%)  "
            f"# integrity check; should match within ~5%"
        )
        # mod 5 — residual-based escalation.
        if per_replay_ms > 0 and (residual_full / per_replay_ms) > 0.25:
            print(
                "  [INFO] kernel-name dict likely stale; rerun with "
                "--cudagraph_breakdown_dump_kernels to discover renames."
            )
    # NEW: per-substep eager_ms / graph_ms ratio table — disambig sanity check.
    #
    # Unit reconciliation: `eager_per_token[name]` is per-LAYER-per-step
    # (each of the N decoder layers records its own substep events into
    # `_pd._step_timings`, and the eager-stash averages those without summing
    # across layers). `substep_per_token[name]` is per-STEP TOTAL (sum over
    # all layers, divided by num_replays). To compare on equal footing, we
    # scale the eager value up to per-step by multiplying by `num_layers`.
    if eager_per_token:
        print()
        print(f"  {'Step':<25} {'eager (ms)':>12} {'graph (ms)':>12} "
              f"{'ratio':>8}")
        print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 8}")
        # Critical substeps for disambig sanity (1 launch / layer / step).
        # `7_upstream_fi_run` / `8_flashinfer_run` is INTENTIONALLY EXCLUDED:
        # it groups multiple FI kernels (BatchDecode + MergeStates) plus
        # per-call setup. CUDA graphs eliminate the per-FI-call CPU dispatch
        # overhead between those sub-kernels, so the graph/eager ratio is
        # structurally ~0.3-0.4 (the speedup is real, not a bucketing defect).
        # Baseline has no DCT-side critical buckets; the only attention-side
        # bucket besides FI run is `2_rope_and_cache_append` (index_kernel),
        # which is left unconstrained.
        if mode == "baseline":
            critical = set()
        else:
            critical = {"5_score_pages_kernel", "6_topk_and_pack"}
        if substep_per_token.get("1_qkv_proj") is not None:
            # Ordering-mode disambig produces these two; merge-mode uses
            # `gemm_total` instead and they're absent.
            critical |= {"1_qkv_proj", "8_o_proj"}
        out_of_band = False
        for name in sorted(set(substep_per_token.keys()) | set(eager_per_token.keys())):
            e_ms_per_layer = eager_per_token.get(name)
            g_ms = substep_per_token.get(name)
            if e_ms_per_layer is None or g_ms is None:
                # Skip substeps with no counterpart (e.g. merged `gemm_total`
                # has no eager peer; eager `1_qkv_proj`/`8_o_proj` have no
                # graph peer in merge mode). Mentioned in plan Step 5.
                continue
            # Scale eager up to per-step (sum across layers) to match graph.
            e_ms_per_step = e_ms_per_layer * num_layers
            ratio = g_ms / e_ms_per_step if e_ms_per_step > 0 else float('inf')
            print(f"  {name:<25} {e_ms_per_step:>12.3f} {g_ms:>12.3f} "
                  f"{ratio:>8.2f}")
            if name in critical and (ratio < 0.5 or ratio > 2.0):
                out_of_band = True
        if out_of_band:
            print(
                "  [INFO] disambig may be wrong, rerun with "
                "--cudagraph_breakdown_dump_kernels"
            )

    # 4_compress reconciliation block.
    # Triggered by EITHER --cudagraph_breakdown_seal_microbench (eager truth)
    # OR --cudagraph_capture_with_seal (in-graph truth via slow-path capture).
    # With both flags off, the block is skipped entirely (byte-identity with
    # pre-edit default output preserved).
    if seal_microbench is not None or seal_capture_active:
        graph_4c = substep_per_token.get("4_compress", 0.0)
        eager_4c_per_step = (
            eager_per_token.get("4_compress", 0.0) * num_layers
            if eager_per_token else 0.0
        )
        print()
        print("  Compression reconciliation (4_compress):")
        if seal_capture_active:
            # In seal-capture mode the graph captured the slow-path, so every
            # replay seals all layers. graph_4c is the per-replay sum (large);
            # divide by page_size to get the per-real-decode-step amortized
            # value comparable to eager-avg / microbench truth.
            print(f"    graph (captured, slow-path) per replay:       "
                  f"{graph_4c:.3f} ms/step"
                  f"    # in-graph cost AS IF every step sealed")
            if page_size:
                print(f"    graph (captured, slow-path) / page_size:      "
                      f"{graph_4c / page_size:.3f} ms/step"
                      f"    # amortized in-graph 4_compress per real step")
        else:
            print(f"    graph (captured, fast-path):                  "
                  f"{graph_4c:.3f} ms/step"
                  f"    # ~0 by design (non-sealing steps captured)")
        print(f"    eager average over all steps x layers:        "
              f"{eager_4c_per_step:.3f} ms/step"
              f"    # = t_seal/ps x layers (in expectation)")
        if seal_microbench is not None:
            forced_truth = seal_microbench["amortized_per_step_total_ms"]
            mb_iters = seal_microbench["iters"]
            mb_ps = seal_microbench["page_size"]
            print(f"    forced-seal microbench / page_size x layers:  "
                  f"{forced_truth:.3f} ms/step"
                  f"    # eager truth (direct measurement, n={mb_iters})")
            print(f"    note: page_size={mb_ps}, microbench_iters={mb_iters}, "
                  f"num_layers={num_layers}")
            # Plausibility advisory: runtime-derived window from eager average.
            # Fallback [0.05, 5.0] ms/step only when eager average is unavailable.
            if eager_4c_per_step > 0:
                lo, hi = eager_4c_per_step / 2.0, eager_4c_per_step * 2.0
            else:
                lo, hi = 0.05, 5.0
            if not (lo <= forced_truth <= hi):
                print(
                    f"  [INFO] forced-seal/ps ({forced_truth:.3f} ms/step) outside "
                    f"+-2x eager avg [{lo:.3f}, {hi:.3f}]; consider verifying with nsys."
                )

    # Non-shared attention substeps — the work that DIFFERS between
    # baseline (full-KV FI) and DCT (page-selected FI + DCT preprocessing).
    # Model-shared substeps {1_qkv_proj, 2_rope_and_cache_append, o_proj}
    # are excluded from this sum; they fire in any attention method.
    #
    # DCT (3+4+5+6+7): segmenting, compression, scoring, top-k, paged FI run.
    # Baseline (8): full-KV FI run only.
    #
    # Subtracting the two gives the marginal cost of DCT page selection vs
    # full attention.
    if mode == "baseline":
        attn_substeps = ("8_flashinfer_run",)
        attn_label = "Baseline non-shared attention (8) — full-KV FI run"
        compare_hint = (
            "    Compare against DCT mode's 'DCT-only attention substeps' "
            "to see the cost of page selection vs full attention."
        )
    else:
        attn_substeps = (
            "3_segment", "4_compress", "5_score_pages_kernel",
            "6_topk_and_pack", "7_upstream_fi_run",
        )
        attn_label = (
            "DCT-only attention substeps (3+4+5+6+7) — REPLACES SDPA full "
            "attention"
        )
        compare_hint = (
            "    Compare against SDPA full-attention latency (same model + "
            "context, attention_backend=sdpa) to evaluate DCT replacement cost."
        )
    # When seal-capture is active, 4_compress in substep_per_token is the
    # per-replay slow-path cost (every replay seals all layers). To get the
    # apples-to-apples per-real-decode-step value, divide by page_size.
    if seal_capture_active and page_size:
        graph_attn_sum = sum(
            (substep_per_token.get(name, 0.0) / page_size
             if name == "4_compress"
             else substep_per_token.get(name, 0.0))
            for name in attn_substeps
        )
    else:
        graph_attn_sum = sum(
            substep_per_token.get(name, 0.0) for name in attn_substeps
        )
    eager_attn_sum_per_step = None
    if eager_per_token:
        eager_attn_sum_per_step = sum(
            eager_per_token.get(name, 0.0) * num_layers
            for name in attn_substeps
        )
    print()
    print(f"  {attn_label}:")
    if eager_attn_sum_per_step is not None and eager_attn_sum_per_step > 0:
        speedup = eager_attn_sum_per_step / graph_attn_sum if graph_attn_sum > 0 else float("inf")
        print(f"    eager total: {eager_attn_sum_per_step:>7.3f} ms/step  "
              f"(per-layer × {num_layers} layers)")
        print(f"    graph total: {graph_attn_sum:>7.3f} ms/step  "
              f"(graph speedup: {speedup:.2f}x)")
    else:
        print(f"    graph total: {graph_attn_sum:>7.3f} ms/step")
    print(compare_hint)


# ---------------------------------------------------------------------------
# CUDA-graph torch.profiler bucketing (Option B path).
#
# Maps CUPTI kernel names emitted by the captured DCT-upstream-FlashInfer
# forward into the 8 attention substep buckets defined by
# `profiled_dct_upstream_flashinfer_forward`'s event chain:
#
#   1_qkv_proj                  : cublasLt gemm (FIRST per layer; see disambig)
#   2_rope_and_cache_append     : RoPE element-wise kernels + index_copy
#   3_segment                   : stride-only view (no kernel — empty bucket)
#   4_compress                  : Triton _update_comp_cache kernel(s)
#   5_score_pages_kernel        : Triton `_score_pages_<scoring>_<group_agg>_kernel`
#                                 (e.g. `_score_pages_max_max_kernel`)
#   6_topk_and_pack             : Triton `_topk_sort_and_pack_kernel`
#                                 (single fused kernel — topk + pack)
#   7_upstream_fi_run           : flashinfer batch_prefill / batch_decode
#                                 kernel + index_copy_ for K/V write
#   8_o_proj                    : cublasLt gemm (LAST per layer; see disambig)
#
# Kernel-name signatures observed (this box, torch 2.10 + flashinfer 0.2 +
# triton 3.6, captured via prof.events() on a 32K Llama-3.1-8B step):
#
#   - cublasLt gemm:   "ampere_h16816gemm_..." or "ampere_bf16_s16816gemm_..."
#                      or "void at::native::elementwise_kernel<...>" (epilogue)
#   - RoPE:            "void at::native::vectorized_elementwise_kernel<..."
#                      — DELIBERATELY UNMAPPED. This generic elementwise kernel
#                      fires for SiLU, residuals, RMSNorm internals, RoPE
#                      complex-multiplies, AND MLP activations. Mapping it to
#                      `2_rope_and_cache_append` would inflate that bucket
#                      with MLP work and corrupt the {5,6,7} ratio checks.
#                      RoPE elementwise multiplies currently land in
#                      `non_attn_residual` (i.e. they show up in
#                      all_kernel_sum but not in any attn-bucket). Positional
#                      disambiguation (assign elementwise-kernel events to
#                      their layer-window position) is a future enhancement —
#                      see Follow-up: positional-window bucketing.
#   - index_copy:      "void at::native::index_kernel<...>" or
#                      "void at::native::indexing_backward_kernel<...>"
#   - _update_comp_cache:    "_update_comp_cache_kernel" (Triton, exact name)
#   - _score_pages_*:        "_score_pages_<scoring>_<group_agg>_kernel"
#                            (Triton; specialized per `scoring_method` x
#                            `group_agg_method`, e.g. `_score_pages_max_max_kernel`,
#                            `_score_pages_max_mean_kernel`, plus `*_c4_g4` /
#                            `*_c1_g4` constexpr specializations). We match by
#                            the `_score_pages_` prefix to cover all variants.
#   - _topk_sort_and_pack:   "_topk_sort_and_pack_kernel" (Triton, single
#                            fused kernel — topk + pack are not separate
#                            kernels in the current codebase).
#   - flashinfer:            "flashinfer::BatchDecodeWithPagedKVCacheKernel"
#                            (also an internal "_apply_pos_encoding_inplace"
#                            but our cache path bypasses it — unused).
#                            Plus "PersistentVariableLengthMergeStatesKernel":
#                            on the upstream-FI virtual-batch-per-(b, h) layout
#                            FI runs a split-kv decode whose partial outputs
#                            are merged by a separate merge-states kernel —
#                            observed empirically (32 layers × N replays).
#                            Both kernels MUST be bucketed into 7_upstream_fi_run
#                            or the ratio falls under the [0.5, 2.0] band.
#
# IMPORTANT: kernel names are subject to change across torch/triton/flashinfer
# versions. The bucketing function (Step 4) hard-asserts two canaries:
#   - `5_score_pages_kernel` count == num_layers * num_replays (boundary anchor)
#   - cublasLt count per layer window matches the expected layout
#     (3 for window 0, 4 for windows 1..N-1) — only enforced when
#     --cudagraph_breakdown_disambig=ordering is set.
# Either canary failing forces fallback to merged `gemm_total` and prints an
# INFO line. The diagnostic flag --cudagraph_breakdown_dump_kernels prints
# all kernel names + counts after one replay window so the user can update
# this dict when names drift.

_SUBSTEP_NAME_PATTERNS = {
    # Triton kernels (substring match against the demangled kernel name).
    # Compression Triton kernel is `_compress_pages_kernel` (triton_kernels.py:1934).
    # The cublasLt GEMM portion of the seal (DCT projection matmul) lands in
    # gemm_total — not bucketed here.
    "_compress_pages_kernel":      "4_compress",
    # Score-pages: substring covers all `_score_pages_<scoring>_<group_agg>_kernel`
    # specializations (e.g. `_score_pages_max_max_kernel`, plus `*_c4_g4` /
    # `*_c1_g4` constexpr variants).
    "_score_pages_":               "5_score_pages_kernel",
    # Topk + pack: single-stage kernel fires for num_pages < 1025; two-stage
    # path (_topk_local_kernel + _topk_merge_and_pack_kernel) fires above that
    # threshold (triton_kernels.py:_TOPK_TWOSTAGE_MIN_PAGES). Map all three so
    # `6_topk_and_pack` populates regardless of context length.
    "_topk_sort_and_pack_kernel":  "6_topk_and_pack",
    "_topk_local_kernel":          "6_topk_and_pack",
    "_topk_merge_and_pack_kernel": "6_topk_and_pack",
    # FlashInfer — both the decode kernel AND the merge-states kernel
    # (split-kv path on the virtual-batch-per-(b, h) layout) must land here.
    "BatchDecodeWithPagedKVCacheKernel": "7_upstream_fi_run",
    "PersistentVariableLengthMergeStatesKernel": "7_upstream_fi_run",
    # Cache append (index_copy_). RoPE's `vectorized_elementwise_kernel` is
    # DELIBERATELY UNMAPPED — see the comment block above.
    "index_kernel":                "2_rope_and_cache_append",
}

# Baseline (full-KV FI) only fires {1_qkv_proj, 2_rope_and_cache_append,
# 8_flashinfer_run, 9_o_proj} per the eager-stash in
# `profiled_baseline_upstream_flashinfer_forward` above. The DCT-only Triton
# patterns can't fire in baseline, so we only need to remap the FI bucket to
# match the baseline eager-stash name (`8_flashinfer_run` vs DCT's
# `7_upstream_fi_run`).
_SUBSTEP_NAME_PATTERNS_BASELINE = {
    "BatchDecodeWithPagedKVCacheKernel": "8_flashinfer_run",
    "PersistentVariableLengthMergeStatesKernel": "8_flashinfer_run",
    "index_kernel":                "2_rope_and_cache_append",
}


def _substep_patterns_for(mode):
    """Per-mode kernel-name → bucket dict. Baseline uses 8_flashinfer_run
    (matches its eager forward); DCT modes use 7_upstream_fi_run."""
    if mode == "baseline":
        return _SUBSTEP_NAME_PATTERNS_BASELINE
    return _SUBSTEP_NAME_PATTERNS
# cublasLt gemms: by default merged into a single `gemm_total` bucket. This
# bucket includes ALL cublasLt matmuls in the captured forward — attention
# (qkv + o_proj), MLP (gate + up + down), AND lm_head — because they share the
# same kernel name and cannot be distinguished by name alone. The bucket is
# therefore NOT attention-scoped; expect it to dwarf the attention substeps
# because MLP is ~3 gemms per layer (vs attention's 2). The eager profile's
# `1_qkv_proj` + `8_o_proj` rows give the attention-only gemm cost for cross-
# checking.
#
# If --cudagraph_breakdown_disambig=ordering is set, the bucket is split into
# 1_qkv_proj / 8_o_proj via per-launch ordering — see Step 4 layer-window
# comment. In ordering mode, MLP and lm_head gemms are explicitly skipped.
_CUBLASLT_PATTERNS = ("gemm", "cutlass", "ampere_")  # case-insensitive


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_one_mode(model, tokenizer, args, mode, original_forward):
    _reset_mode_state()
    _pd._sync_mode = args.sync
    _pd._profile_topk_impl.value = args.topk_impl

    # Per-mode breakdown overrides (locals, not args mutation — args is reused
    # across modes when --mode all). Baseline can't run DCT-only breakdown
    # features: ordering disambig anchors on `_score_pages_*` (DCT-only),
    # seal microbench probes `_comp_n_pages_cached` (DCT-only), and
    # capture-with-seal forces the DCT slow path. Downgrade these silently
    # with an info message so --mode all + --cudagraph_breakdown still works.
    _breakdown_disambig = args.cudagraph_breakdown_disambig
    _seal_microbench = args.cudagraph_breakdown_seal_microbench
    _capture_with_seal = args.cudagraph_capture_with_seal
    if mode == "baseline":
        if _breakdown_disambig == "ordering":
            print(
                "[INFO] baseline mode: --cudagraph_breakdown_disambig=ordering "
                "is DCT-only (anchor = _score_pages_*); falling back to merge."
            )
            _breakdown_disambig = "merge"
        if _seal_microbench:
            print(
                "[INFO] baseline mode: --cudagraph_breakdown_seal_microbench "
                "is DCT-only (probes _comp_n_pages_cached); disabling."
            )
            _seal_microbench = False
        if _capture_with_seal:
            print(
                "[INFO] baseline mode: --cudagraph_capture_with_seal is "
                "DCT-only (forces DCT slow path); disabling."
            )
            _capture_with_seal = False

    if mode == "baseline":
        _pd._profile_attn_backend.value = "sdpa"
        _pd._profile_attn_backend.verify = False
        _patch_baseline(model, args, original_forward)
    elif mode == "dct_sdpa":
        _pd._profile_attn_backend.value = "sdpa"
        _pd._profile_attn_backend.verify = False
        _patch_dct_sdpa(model, args, original_forward)
    elif mode == "dct_upstream_flashinfer":
        _pd._profile_attn_backend.value = "sdpa"  # unused by our forward
        _pd._profile_attn_backend.verify = False
        _patch_dct_upstream_flashinfer(model, args, original_forward)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size
    bsz = args.batch_size

    torch.manual_seed(0)
    input_ids = torch.randint(
        0, vocab_size, (bsz, args.context_length), dtype=torch.long, device=device,
    )

    chunk_size = args.chunk_size
    print(f"  Prefilling ({args.context_length} tokens, bsz={bsz}, chunk={chunk_size})...")
    _pd._enabled = False
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = chunked_prefill(model, input_ids, chunk_size)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prefill done: {prefill_ms:.0f}ms")

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)
    prefill_len = args.context_length

    extra = args.warmup_steps + args.num_decode_steps + 16
    if args.cudagraph:
        extra += 64
    # FI-based modes (dct_upstream_flashinfer + baseline) free the per-layer
    # flat keys/values right after the prefill→paged pack, so the per-layer
    # `extra` slack would be allocated and immediately freed — skip it. This
    # mirrors `profile_decode_flash_infer.py:1318`. Without this, at long
    # context × large bsz the slack alloc adds gigabytes of transient memory
    # that drives OOM at buf alloc time.
    pa_extra = 0 if mode in ("dct_upstream_flashinfer", "baseline") else extra
    past_key_values = pre_allocate_cache(past_key_values, extra_tokens=pa_extra)
    print(f"  Converted to pre-allocated cache (+{pa_extra} tokens)")

    if mode == "dct_upstream_flashinfer":
        _build_upstream_fi_cache(model, past_key_values, prefill_len, args)
    elif mode == "baseline":
        cfg_model = model.config
        num_kv_heads = cfg_model.num_key_value_heads
        num_qo_heads = cfg_model.num_attention_heads
        head_dim = cfg_model.hidden_size // num_qo_heads
        num_layers = cfg_model.num_hidden_layers
        max_decode_steps = extra
        print(
            f"  Building full-KV FI baseline cache (layers={num_layers}, "
            f"bsz={args.batch_size}, page_size={args.page_size})..."
        )
        _fi_baseline_cache_ref[0] = build_baseline_upstream_fi_cache(
            preallocated_layers=past_key_values.layers,
            prefill_len=prefill_len,
            page_size=args.page_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_layers=num_layers,
            max_decode_steps=max_decode_steps,
            dtype=past_key_values.layers[0].keys.dtype,
            device=device,
            bsz=args.batch_size,
        )
        _bc = _fi_baseline_cache_ref[0]
        print(
            f"  FI baseline cache ready: capacity_pages={_bc.capacity_pages} "
            f"(pages_per_batch={_bc.pages_per_batch} × bsz={_bc.bsz}), "
            f"num_active_pages={_bc.num_active_pages}"
        )

    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    for m in model.modules():
        if isinstance(m, attn_cls):
            m._verify_upstream = (
                mode == "dct_upstream_flashinfer" and args.verify_upstream
            )
            if hasattr(m, "_verify_diffs"):
                del m._verify_diffs

    # Disable verify during warmup.
    saved_verify = {}
    if args.verify_upstream and mode == "dct_upstream_flashinfer":
        for m in model.modules():
            if isinstance(m, attn_cls):
                saved_verify[id(m)] = m._verify_upstream
                m._verify_upstream = False

    total_times = []

    def _do_one_decode_step(step_idx, profiled):
        nonlocal next_token, past_key_values
        if profiled:
            _pd._current_layer = 0
        cache_position = torch.tensor([prefill_len + step_idx], device=device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter() if profiled else None
        with torch.no_grad():
            out = model(
                next_token, past_key_values=past_key_values,
                use_cache=True, cache_position=cache_position,
            )
        if profiled:
            _pd._flush_events()
            total_times.append((time.perf_counter() - t0) * 1000)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)

    print(f"  Warming up ({args.warmup_steps} steps)...")
    _pd._enabled = False
    for step in range(args.warmup_steps):
        _do_one_decode_step(step, profiled=False)
    torch.cuda.synchronize(device)

    if args.verify_upstream and mode == "dct_upstream_flashinfer":
        for m in model.modules():
            if isinstance(m, attn_cls):
                m._verify_upstream = saved_verify.get(id(m), True)
                m._verify_diffs = []

    print(f"  Profiling ({args.num_decode_steps} steps)...")
    _pd._step_timings.clear()
    _pd._cpu_timings.clear()
    _pd._pending_events.clear()
    _pd._enabled = True
    for step in range(args.num_decode_steps):
        _do_one_decode_step(args.warmup_steps + step, profiled=True)
    _pd._enabled = False
    # B1: stash per-substep eager averages BEFORE the cudagraph block so the
    # graph-mode bucketer can compute eager_ms / graph_ms ratios as a disambig
    # sanity check. Reads from `_pd._step_timings` populated by the eager
    # forward's `_pd._flush_events()` call (deque of per-step ms, one entry
    # per substep per step). MUST live here in `_run_one_mode` (not in
    # `main()`); placing it near `print_profile` would be too late — bucketer
    # below reads it during this same `_run_one_mode` invocation.
    _pd._last_eager_per_token = {
        name: sum(t) / len(t)
        for name, t in _pd._step_timings.items() if t
    }
    torch.cuda.synchronize(device)

    avg_total = sum(total_times) / len(total_times)
    tok_s = 1000.0 / avg_total

    verify_ok = None
    if args.verify_upstream and mode == "dct_upstream_flashinfer":
        per_layer_diffs = {}
        for m in model.modules():
            if isinstance(m, attn_cls) and getattr(m, "_verify_diffs", None):
                lid = getattr(m, "layer_idx", None)
                per_layer_diffs[lid] = list(m._verify_diffs)
        if per_layer_diffs:
            all_steps = max(len(v) for v in per_layer_diffs.values())
            per_step_worst = [0.0] * all_steps
            per_step_layer = [-1] * all_steps
            for lid, diffs in per_layer_diffs.items():
                for s, d in enumerate(diffs):
                    if d > per_step_worst[s]:
                        per_step_worst[s] = d
                        per_step_layer[s] = lid
            worst = max(max(v) for v in per_layer_diffs.values())
            # Threshold ladder (Critic A): at bsz>1 use max(floor, 3 * wd1)
            # where wd1 = bsz=1 empirical worst, supplied via --bsz1_wd1.
            # At bsz=1 the floor (--verify_threshold) is the gate; we also
            # log the observed worst as wd1 so the user can plug it back in
            # for subsequent bsz>1 runs.
            if args.batch_size == 1:
                threshold = args.verify_threshold
                ladder_note = "bsz=1 floor"
            else:
                threshold = max(args.verify_threshold, 3.0 * args.bsz1_wd1)
                ladder_note = (
                    f"max({args.verify_threshold:.0e}, 3*{args.bsz1_wd1:.3e})"
                )
            print(
                f"  [VERIFY] wd1={args.bsz1_wd1:.3e} threshold={threshold:.3e} "
                f"({ladder_note}) observed_worst={worst:.3e}"
            )
            print(
                f"  [verify_upstream] worst max-abs-diff across "
                f"{len(per_layer_diffs)} layers x {all_steps} steps = "
                f"{worst:.3e} (threshold = {threshold:.3e})"
            )
            head = min(8, all_steps)
            for s in range(head):
                ok = per_step_worst[s] < threshold
                print(
                    f"    step {s}: {per_step_worst[s]:.3e}  "
                    f"worst layer={per_step_layer[s]:>2}  "
                    f"{'OK' if ok else 'FAIL'}"
                )
            if all_steps > head:
                print(f"    ... ({all_steps - head} more steps)")
            verify_ok = worst < threshold
            print(f"  [verify_upstream] overall: {'PASS' if verify_ok else 'FAIL'}")
            if args.batch_size == 1:
                print(
                    f"  [VERIFY] bsz=1 wd1 (record for bsz>1 ladder): {worst:.3e} "
                    f"→ pass `--bsz1_wd1={worst:.3e}` on subsequent bsz>1 runs"
                )

    # Optional CUDA graph benchmark.
    graph_stats = None
    if args.cudagraph:
        # FI-mode page alignment.
        # The full-KV baseline forward (`profiled_baseline_upstream_flashinfer_forward`)
        # calls `wrapper.plan()` whenever a new physical page opens (every
        # page_size decode steps). plan() is NOT graph-capturable. Priming
        # runs the forward 3 extra times before the captured iteration, so
        # unless `last_page_len_py + 4 <= page_size` plan() will land
        # somewhere in the priming/capture window.
        #
        # Mitigation: run extra eager decode steps (no profiling) until
        # `last_page_len_py <= page_size - 4`, advancing the open-page counter
        # safely past the boundary if needed. Upstream-FI mode never replans,
        # so this loop is a no-op / cheap there.
        fi_cache = _fi_cache_for_mode(mode)
        align_steps = 0
        if fi_cache is not None:
            ps = fi_cache.page_size
            while fi_cache.last_page_len_py > ps - 4:
                _do_one_decode_step(
                    args.warmup_steps + args.num_decode_steps + align_steps,
                    profiled=False,
                )
                align_steps += 1
                if align_steps > ps + 1:  # safety: never loop forever
                    break
            print(
                f"  Graph alignment: {align_steps} eager step(s) "
                f"(last_page_len={fi_cache.last_page_len_py}/{ps})"
            )

        # The probe is the original Option A capability probe (event-record-
        # in-graph). It stays in tree because Option A is correct on hardware/torch
        # versions where event-record-in-graph works (NOT this box per
        # feedback_event_in_graph_unsupported.md, but plausibly future versions).
        # When --cudagraph_breakdown_method == "events" we call the probe and gate
        # the feature on it; when == "profiler" (default) we skip the probe and
        # use torch.profiler instead (Step 2).
        if args.cudagraph_breakdown and args.cudagraph_breakdown_method == "events":
            if not _probe_event_record_in_graph(device):
                print(
                    "[INFO] CUDA graph breakdown unsupported on this build "
                    "(cudaEventRecord-in-graph silently dropped); disabling "
                    "--cudagraph_breakdown"
                )
                args.cudagraph_breakdown = False

        torch.cuda.synchronize(device)
        num_replays = args.cudagraph_replays or args.num_decode_steps
        current_pos = (
            prefill_len + args.warmup_steps + args.num_decode_steps + align_steps
        )

        static_input = next_token.clone()
        static_pos = torch.tensor([current_pos], device=device, dtype=torch.long)

        if args.cudagraph_breakdown:
            _breakdown_toggle = _CudagraphBreakdownToggle()
            _breakdown_toggle.__enter__()
        try:
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        model(static_input, past_key_values=past_key_values,
                              use_cache=True, cache_position=static_pos)
            torch.cuda.current_stream(device).wait_stream(s)
            torch.cuda.synchronize(device)

            try:
                if args.cudagraph_breakdown:
                    # discard priming-loop events; only capture-time events should be read
                    _pd._pending_events.clear()

                # Seal-capture: force n_new=1 on every DCT layer just before
                # capture so the captured graph contains the slow-path
                # (compress GEMM + page-copies). Replays then time the
                # in-graph 4_compress cost. Per-layer state is mutated and
                # restored AFTER the replay loop completes (before bucketing
                # / microbench, both of which depend on a clean state).
                seal_capture_state = []
                if _capture_with_seal:
                    dct_modules_all = [
                        m for m in model.modules()
                        if hasattr(m, '_comp_n_pages_cached')
                    ]
                    if not dct_modules_all:
                        print(
                            "[ERROR] --cudagraph_capture_with_seal: no DCT "
                            "attention modules found; mode is likely baseline"
                        )
                    else:
                        for m in dct_modules_all:
                            seal_capture_state.append(
                                (m, m._comp_n_pages_cached)
                            )
                            if m._comp_n_pages_cached > 0:
                                m._comp_n_pages_cached -= 1
                        print(
                            f"  Seal-capture: forced n_new=1 on "
                            f"{len(dct_modules_all)} DCT layers "
                            f"(captured graph will run slow-path; replays "
                            f"overwrite a fixed cache slot — TIMING ONLY)"
                        )

                g = torch.cuda.CUDAGraph()
                print(f"  Capturing CUDA graph...")
                with torch.cuda.graph(g):
                    with torch.no_grad():
                        model(static_input, past_key_values=past_key_values,
                              use_cache=True, cache_position=static_pos)
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize(device)
                print(f"  Replaying graph ({num_replays} steps) for throughput...")
                # The torch.profiler import is conditional and unreachable when
                # --cudagraph_breakdown is OFF (preserves the byte-identical
                # no-flag invariant — Principle 1).
                use_profiler = (
                    args.cudagraph_breakdown
                    and args.cudagraph_breakdown_method == "profiler"
                )
                if use_profiler:
                    from torch.profiler import profile, ProfilerActivity
                    prof_ctx = profile(activities=[ProfilerActivity.CUDA])
                else:
                    from contextlib import nullcontext
                    prof_ctx = nullcontext()
                with prof_ctx as prof:
                    t0 = time.perf_counter()
                    for _ in range(num_replays):
                        g.replay()
                    # cuda.synchronize stays inside the profiler scope so
                    # headline measurement is taken under identical conditions
                    # whether profiling or not.
                    torch.cuda.synchronize(device)
                    per_replay_ms = (time.perf_counter() - t0) * 1000 / num_replays
                graph_tok_s = 1000.0 / per_replay_ms
                graph_stats = (per_replay_ms, graph_tok_s)
                if args.batch_size == 1:
                    print(f"  CUDA graph: {per_replay_ms:.3f} ms/step  ({graph_tok_s:.2f} tok/s)")
                else:
                    print(
                        f"  CUDA graph: {per_replay_ms:.3f} ms/step  "
                        f"({graph_tok_s:.2f} step/s, {graph_tok_s * args.batch_size:.2f} agg tok/s @ bsz={args.batch_size})"
                    )

                # Seal-capture restore: replays don't mutate Python state
                # (only GPU kernels replay), so by here the slow-path during
                # capture already wrote _comp_n_pages_cached back to num_pages
                # at dct_page_attention.py:722. Defensive restore + invalidate
                # _last_comp_kv so the seal microbench (if also enabled) sees
                # a clean state and any future eager forward isn't poisoned
                # by stale tuple identity.
                for m, orig in seal_capture_state:
                    m._comp_n_pages_cached = orig
                    m._last_comp_kv = None

                if args.cudagraph_breakdown:
                    # Layer windowing — required reading before touching the disambig logic.
                    #
                    # The eager forward at speed/profile_decode_upstream_flash_infer.py:204-228
                    # emits the following GPU event stream PER LAYER (linear, no branches at
                    # steady state for unfused-qkv Llama-3.1-8B):
                    #
                    #   ... [PREV LAYER] [o_proj_{li-1}]                                   <-- LAST cublasLt of previous layer's window
                    #   [layernorm_li] [q_proj_li] [k_proj_li] [v_proj_li]                 <-- FIRST 3 cublasLt of current layer's window
                    #   [q_norm?] [k_norm?] [transpose×2] [apply_rotary_pos_emb (multi-kernel)]
                    #   [past_key_values.update / index_copy] [2× .fill_]
                    #   [_update_comp_cache_kernel] [_score_pages_*_kernel(li)]            <-- BOUNDARY MARKER
                    #   ... [topk] [pack] [flashinfer_kernel] [o_proj_li]                  <-- LAST cublasLt of current layer's window
                    #   [NEXT LAYER] ...
                    #
                    # Define layer window `li` as the half-open event-index range
                    #   [score_kernel[li-1] + 1, score_kernel[li] + 1)
                    # i.e. starts AFTER the previous score kernel and includes the current one.
                    #
                    # Within window `li`:
                    #   - The FIRST 3 cublasLt events  = q_proj_li, k_proj_li, v_proj_li
                    #                                                            -> bucket 1_qkv_proj
                    #   - The LAST cublasLt event      = o_proj_{li-1}           -> bucket 8_o_proj
                    #
                    # Yes — within window `li` the o_proj belongs to the PREVIOUS layer. This is
                    # intentional: the score kernel is a clean boundary marker but it falls in
                    # the MIDDLE of layer `li`'s execution, not at its edge. The bookkeeping
                    # handles this off-by-one explicitly.
                    #
                    # Edge cases:
                    #   - FIRST window (li=0): there is no score_kernel[-1], so window 0 starts
                    #     from event 0 and contains no o_proj (only the model's initial qkv
                    #     before the first score kernel). Expected cublasLt count in window 0:
                    #     3 (qkv only).
                    #   - LAST window (li = num_layers - 1): final-layer o_proj has no anchor
                    #     after it (no next score kernel). We accept misclassification of the
                    #     final-layer o_proj as a 1/N error; acceptable for an opt-in advanced
                    #     disambig path. C4 prints an INFO line about this in ordering mode.
                    #
                    # This windowing is ONLY used when --cudagraph_breakdown_disambig=ordering.
                    # Default `merge` mode skips all of this and assigns every cublasLt to a
                    # single `gemm_total` bucket.
                    num_layers_local = model.config.num_hidden_layers
                    substep_total = {}
                    all_kernel_sum_ms = 0.0
                    bucketing_ok = True

                    if args.cudagraph_breakdown_method == "profiler":
                        # `prof` was created above (use_profiler == True here).
                        events_list = list(prof.events()) if prof is not None else []
                        if len(events_list) == 0:
                            print(
                                "[INFO] torch.profiler returned no CUDA events; "
                                "skipping breakdown (headline still printed)."
                            )
                            bucketing_ok = False
                        else:
                            # Primary filter: device_time_total > 0 — most robust
                            # across torch versions. (device_type-based filter
                            # would be a fallback but is not stable on this
                            # torch build.)
                            cuda_events = [
                                ev for ev in events_list
                                if (getattr(ev, "device_time_total", None)
                                    or getattr(ev, "cuda_time_total", 0)) > 0
                            ]
                            # Sort by timeline start. Events without time_range
                            # sort to END (float("inf")) so they don't scramble
                            # the layer-boundary scan.
                            cuda_events.sort(
                                key=lambda e: (
                                    e.time_range.start
                                    if hasattr(e, "time_range") else float("inf")
                                )
                            )

                            # Optional diagnostic dump of all kernel names.
                            if args.cudagraph_breakdown_dump_kernels:
                                from collections import Counter
                                ctr = Counter(ev.key for ev in cuda_events)
                                print("[DIAG] CUPTI kernel name counts (full trace):")
                                for k, v in sorted(ctr.items(), key=lambda kv: -kv[1]):
                                    print(f"  {v:6d}  {k}")

                            # Find score-kernel anchors (one per layer per replay).
                            # Match by `_score_pages_` prefix to cover all
                            # `_score_pages_<scoring>_<group_agg>_kernel`
                            # specializations.
                            score_idx = [
                                i for i, ev in enumerate(cuda_events)
                                if "_score_pages_" in ev.key
                            ]
                            expected_score_hits = num_layers_local * num_replays
                            # C1: gate score-kernel-count INFO on ordering mode
                            # only — in merge mode the canary doesn't gate
                            # disambig, so the message would mislead.
                            if (
                                len(score_idx) != expected_score_hits
                                and _breakdown_disambig == "ordering"
                            ):
                                print(
                                    f"[INFO] score-kernel hit count mismatch: "
                                    f"got {len(score_idx)}, expected "
                                    f"{expected_score_hits}. Forcing merged gemm bucket."
                                )
                            use_ordering = (
                                _breakdown_disambig == "ordering"
                                and len(score_idx) == expected_score_hits
                            )

                            # Build per-layer cublasLt index lists when ordering active.
                            gemm_qkv_idx = set()
                            gemm_oproj_idx = set()
                            if use_ordering:
                                cublaslt_count_anomaly = False
                                for replay_i in range(num_replays):
                                    replay_start = replay_i * num_layers_local
                                    replay_anchors = score_idx[
                                        replay_start: replay_start + num_layers_local
                                    ]
                                    for li, anchor in enumerate(replay_anchors):
                                        # Window li = [prev_anchor + 1, anchor + 1).
                                        if li > 0:
                                            prev_anchor = replay_anchors[li - 1]
                                        else:
                                            # Window 0 of this replay: back up far
                                            # enough to capture the initial qkv
                                            # gemms before the first score kernel.
                                            prev_anchor = replay_anchors[0] - 100
                                        win_start = max(0, prev_anchor + 1)
                                        win_end = anchor + 1
                                        gemms = [
                                            j for j in range(win_start, win_end)
                                            if any(
                                                p in cuda_events[j].key.lower()
                                                for p in _CUBLASLT_PATTERNS
                                            )
                                        ]
                                        # cublasLt-count canary: window 0 of replay 0
                                        # has 3 (qkv only); other windows have 4
                                        # (qkv + previous-layer o_proj).
                                        expected = 3 if (replay_i == 0 and li == 0) else 4
                                        if len(gemms) != expected:
                                            cublaslt_count_anomaly = True
                                            print(
                                                f"[INFO] cublasLt-per-layer-window "
                                                f"count mismatch (replay={replay_i}, "
                                                f"layer={li}): got {len(gemms)}, "
                                                f"expected {expected}; merging gemms."
                                            )
                                            break
                                        # qkv = first 3, o_proj_{li-1} = last (li>=1).
                                        gemm_qkv_idx.update(gemms[:3])
                                        if li > 0:
                                            gemm_oproj_idx.add(gemms[-1])
                                    if cublaslt_count_anomaly:
                                        break
                                if cublaslt_count_anomaly:
                                    use_ordering = False
                                    gemm_qkv_idx.clear()
                                    gemm_oproj_idx.clear()

                            for j, ev in enumerate(cuda_events):
                                bucket = None
                                key_lower = ev.key.lower()
                                is_gemm = any(
                                    p in key_lower for p in _CUBLASLT_PATTERNS
                                )
                                if is_gemm:
                                    if not use_ordering:
                                        bucket = "gemm_total"  # merged path
                                    elif j in gemm_qkv_idx:
                                        bucket = "1_qkv_proj"
                                    elif j in gemm_oproj_idx:
                                        bucket = "8_o_proj"
                                    # else: gemm in non-attention region
                                    # (MLP / lm_head) — skip
                                else:
                                    for pat, b in _substep_patterns_for(mode).items():
                                        if pat in ev.key:
                                            bucket = b
                                            break
                                # torch ≥ 2.10 renamed cuda_time_total → device_time_total
                                ev_us = (getattr(ev, "device_time_total", None)
                                         or ev.cuda_time_total)
                                ev_ms = ev_us / 1000.0  # us → ms
                                all_kernel_sum_ms += ev_ms
                                if bucket:
                                    substep_total[bucket] = (
                                        substep_total.get(bucket, 0.0) + ev_ms
                                    )

                            # Per-step normalization: profiler ran over `num_replays`
                            # replays; divide each bucket by num_replays.
                            substep_per_token = {
                                name: total / num_replays
                                for name, total in substep_total.items()
                            }
                    else:  # method == "events" — Option A path (dormant on this box)
                        substep_count = {}
                        for name, ev_s, ev_e in _pd._pending_events:
                            substep_total[name] = (
                                substep_total.get(name, 0.0)
                                + ev_s.elapsed_time(ev_e)
                            )
                            substep_count[name] = substep_count.get(name, 0) + 1
                        substep_per_token = {
                            name: substep_total[name] / substep_count[name]
                            for name in substep_total
                        }
                        # all_kernel_sum unavailable via Option A.
                        all_kernel_sum_ms = None

                    if bucketing_ok and substep_per_token:
                        # Run seal microbench BEFORE the printer (uses live
                        # attn_module state; must run while KV cache is intact).
                        # layer_idx==0 pin: model.modules() registration order is
                        # not spec-guaranteed; pinning makes the choice deterministic.
                        seal_mb = None
                        if _seal_microbench:
                            attn_modules = [
                                m for m in model.modules()
                                if hasattr(m, '_comp_n_pages_cached')
                                and getattr(m, 'layer_idx', -1) == 0
                            ]
                            if not attn_modules:
                                print(
                                    "[ERROR] seal microbench: no DCT attention "
                                    "modules found; mode is likely baseline"
                                )
                            else:
                                seal_mb = _run_seal_microbench(
                                    model, fi_cache, attn_modules, args,
                                    num_layers_local,
                                )
                        _print_graph_breakdown(
                            per_replay_ms, substep_per_token,
                            num_layers_local, args.batch_size,
                            mode=mode,
                            all_kernel_sum_ms_per_step=(
                                all_kernel_sum_ms / num_replays
                                if all_kernel_sum_ms is not None else None
                            ),
                            eager_per_token=getattr(
                                _pd, "_last_eager_per_token", None
                            ),
                            seal_microbench=seal_mb,
                            seal_capture_active=_capture_with_seal,
                            page_size=args.page_size,
                        )
                        # C4: ordering-mode-only INFO about excluded final o_proj.
                        if (
                            args.cudagraph_breakdown_method == "profiler"
                            and _breakdown_disambig == "ordering"
                        ):
                            print(
                                "  [INFO] 8_o_proj excludes the final-layer "
                                "o_proj (1/N-error, acknowledged in plan)."
                            )
            except Exception as e:
                print(f"  CUDA graph benchmark failed: {type(e).__name__}: {e}")
                graph_stats = None
        finally:
            if args.cudagraph_breakdown:
                _breakdown_toggle.__exit__(None, None, None)

    return (
        avg_total, tok_s,
        dict(_pd._step_timings), dict(_pd._cpu_timings),
        verify_ok, graph_stats,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    original_forward = get_original_forward(args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)
    model._original_attn_forward = original_forward

    num_layers = model.config.num_hidden_layers
    print(f"Model layers: {num_layers}")
    print(f"Context length: {args.context_length}  batch_size: {args.batch_size}")

    modes_order = ["baseline", "dct_sdpa", "dct_upstream_flashinfer"]
    if args.mode == "all":
        modes_to_run = modes_order
    else:
        modes_to_run = [args.mode]

    results = {}
    verify_state = {}
    for mode in modes_to_run:
        print(f"\n{'=' * 70}")
        print(f"MODE: {mode}")
        print(f"{'=' * 70}")
        (avg_total, tok_s, timings, cpu_timings,
         verify_ok, graph_stats) = _run_one_mode(
            model, tokenizer, args, mode, original_forward,
        )
        print_profile(mode, avg_total, tok_s, timings, num_layers, cpu_timings, bsz=args.batch_size)
        if graph_stats is not None:
            gp, gts = graph_stats
            print(
                f"  Speedup (graph vs profiled): {avg_total / gp:.2f}x  "
                f"(saved {avg_total - gp:.2f} ms/step)"
            )
        results[mode] = (avg_total, tok_s, timings, graph_stats)
        if verify_ok is not None:
            verify_state[mode] = verify_ok

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if len(results) >= 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        # At bsz>1 split tok/s into step/s + agg tok/s. The "tok/s" reported
        # by tok_s = 1000/avg_total is steps/sec; one step generates `bsz`
        # tokens, so aggregate = bsz * step/s.
        bsz = args.batch_size
        if bsz == 1:
            print(f"  {'Mode':<28} {'ms/step':>10} {'tok/s':>10} {'vs baseline':>14}")
            print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 14}")
        else:
            print(
                f"  {'Mode':<28} {'ms/step':>10} {'step/s':>10} "
                f"{'agg tok/s':>11} {'vs baseline':>14}"
            )
            print(
                f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 11} {'-' * 14}"
            )
        base = results.get("baseline")
        for mode in modes_order:
            if mode not in results:
                continue
            avg, tok, _, _graph = results[mode]
            if base is not None and mode != "baseline":
                delta_pct = (tok - base[1]) / base[1] * 100
                vs_str = f"{delta_pct:+.1f}%"
            elif mode == "baseline":
                vs_str = "(ref)"
            else:
                vs_str = "—"
            if bsz == 1:
                print(f"  {mode:<28} {avg:>10.2f} {tok:>10.2f} {vs_str:>14}")
            else:
                print(
                    f"  {mode:<28} {avg:>10.2f} {tok:>10.2f} "
                    f"{tok * bsz:>11.2f} {vs_str:>14}"
                )

        any_graph = any(r[3] is not None for r in results.values())
        if any_graph:
            if bsz == 1:
                print(f"\n  {'Mode (graph)':<28} {'ms/step':>10} {'tok/s':>10} {'vs baseline':>14}")
                print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 14}")
            else:
                print(
                    f"\n  {'Mode (graph)':<28} {'ms/step':>10} {'step/s':>10} "
                    f"{'agg tok/s':>11} {'vs baseline':>14}"
                )
                print(
                    f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 11} {'-' * 14}"
                )
            base_graph = results.get("baseline", (None,) * 4)[3]
            for mode in modes_order:
                if mode not in results or results[mode][3] is None:
                    continue
                gp, gts = results[mode][3]
                if base_graph is not None and mode != "baseline":
                    delta_pct = (gts - base_graph[1]) / base_graph[1] * 100
                    vs_str = f"{delta_pct:+.1f}%"
                elif mode == "baseline":
                    vs_str = "(ref)"
                else:
                    vs_str = "—"
                if bsz == 1:
                    print(f"  {mode:<28} {gp:>10.2f} {gts:>10.2f} {vs_str:>14}")
                else:
                    print(
                        f"  {mode:<28} {gp:>10.2f} {gts:>10.2f} "
                        f"{gts * bsz:>11.2f} {vs_str:>14}"
                    )

    if verify_state:
        print()
        for mode, ok in verify_state.items():
            tag = "PASS" if ok else "FAIL"
            print(f"  [verify_upstream] {mode}: {tag}")


if __name__ == "__main__":
    main()
