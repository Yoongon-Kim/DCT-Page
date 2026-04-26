"""
Profile DCT + upstream-FlashInfer decode (virtual-batch-per-head layout).

Sibling of `speed/profile_decode_flash_infer.py`. That driver uses the
DCT-Page fork of FlashInfer at `/home/yoongonkim/flashinfer-dct` with a
per-head `indices` patch (plan() `page_budget` kwarg). This driver tests
whether we can drop that patch entirely by reshaping the KV cache so each
physical page holds one KV head's slice, then treating each KV head as a
virtual batch entry for stock FlashInfer's 2-D indices API.

Correctness argument: multi-head attention is separable over KV heads —
softmax is per Q head. Packing "KV head h's selected pages" as virtual
batch h, with the `group_size` Q heads that attend to it as that batch's
query heads, computes the exact same attention output (up to FI kernel
numerics).

Modes (via `--mode`, default `dct_upstream_flashinfer`):
  - baseline                   : full-KV FlashInfer (shared with the fork
                                  profiler — no per-head selection needed).
  - dct_sdpa                   : DCT + SDPA (pure reference).
  - dct_upstream_flashinfer    : DCT + upstream FI via virtual batching.
  - all                        : run all three back-to-back with comparison.

Usage:
    CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_upstream_flash_infer.py \\
        --context_length 32768 --page_size 32 --top_k 64 \\
        --sink_size 32 --recent_size 128 \\
        --num_decode_steps 128 --warmup_steps 8 \\
        --mode all --verify_upstream
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
    pre_allocate_cache,
    print_profile,
    profiled_dct_page_attention_forward,
)

# Share the full-KV FlashInfer baseline from the fork profiler — neither
# baseline nor dct_sdpa need per-head selection, so no reason to duplicate.
import profile_decode_flash_infer as _pdfi
from profile_decode_flash_infer import (
    build_fi_baseline_cache,
    profiled_baseline_flashinfer_forward,
)

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
    replace_llama_attn,
    segment_kv,
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
        cfg.sink_size + cfg.page_size * (cfg.top_k + 1) + cfg.recent_size,
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
        key_states, value_states = past_key_values.update(
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

    # Step 3: segment DCT cache.
    comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))
    (sink_k, sink_v, paged_k, paged_v,
     recent_k, recent_v, num_pages, actual_recent) = segment_kv(
        key_states, value_states, cfg,
    )

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
    num_middle_pages = (
        cache.last_page_idx_py - cache.num_sink_pages
        - cache.num_recent_pages_fixed + 1
    )
    if num_middle_pages < cache.top_k:
        raise RuntimeError(
            f"num_middle_pages ({num_middle_pages}) < cache.top_k "
            f"({cache.top_k}). Configure min_decode_kv_len_for_paging."
        )
    effective_num_pages = min(num_pages, num_middle_pages)
    eff_scores = page_scores[:, :, :effective_num_pages]
    topk_sort_and_pack_triton(
        eff_scores,
        cache.indices_buf_3d,
        num_sink_pages=cache.num_sink_pages,
        top_k=cache.top_k,
        last_page_idx=cache.last_page_idx,
        recent_offsets=cache.recent_offsets,
        sort_ascending=False,
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
    k_flat = key_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
    v_flat = value_states[:, :, -1:, :].reshape(cache.num_kv_heads, cache.head_dim)
    cache.buf_7d[self.layer_idx, :, page_idx, 0, slot, 0, :].copy_(k_flat)
    cache.buf_7d[self.layer_idx, :, page_idx, 1, slot, 0, :].copy_(v_flat)

    refresh_upstream_indices_flat(cache)
    attn_output_fi = upstream_flashinfer_decode_attention(
        query_states, cache, self.layer_idx,
    )

    if _pd._enabled:
        _rec(7)

    # Verify path — outside the event window. Recreates the same K/V set that
    # FI saw from the cache and runs SDPA on it. Because each head's data
    # now lives in its own page pool, gather uses buf_7d directly (no need
    # to slice `num_kv_heads` out of each page).
    if getattr(self, "_verify_upstream", False):
        buf_l_7d = cache.buf_7d[self.layer_idx]  # (H, P, 2, ps, 1, d)
        page_budget = cache.page_budget
        last_page_len = cache.last_page_len_py
        full_len = (page_budget - 1) * cache.page_size + last_page_len
        k_pages = []
        v_pages = []
        for h in range(_num_kv_heads):
            sel_h = cache.indices_buf_3d[0, h].long()    # head-local IDs
            kv_h = buf_l_7d[h][sel_h]                    # (page_budget, 2, ps, 1, d)
            k_h = kv_h[:, 0, :, 0, :].reshape(
                page_budget * cache.page_size, self.head_dim
            )
            v_h = kv_h[:, 1, :, 0, :].reshape(
                page_budget * cache.page_size, self.head_dim
            )
            k_pages.append(k_h[:full_len])
            v_pages.append(v_h[:full_len])
        k_ref = torch.stack(k_pages, dim=0).unsqueeze(0)
        v_ref = torch.stack(v_pages, dim=0).unsqueeze(0)
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
    num_sink_pages = (args.sink_size + args.page_size - 1) // args.page_size
    num_recent_pages_fixed = (
        (args.recent_size + args.page_size - 1) // args.page_size + 1
    )
    max_decode_steps = args.warmup_steps + args.num_decode_steps + 16
    if args.cudagraph:
        max_decode_steps += 64
    page_budget = num_sink_pages + args.top_k + num_recent_pages_fixed
    print(
        f"  Building upstream-FI cache: layers={num_layers}, "
        f"num_sink_pages={num_sink_pages}, top_k={args.top_k}, "
        f"num_recent_pages_fixed={num_recent_pages_fixed}, "
        f"page_budget={page_budget}, vbsz={num_kv_heads}, "
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
    )
    _upstream_fi_cache_ref[0] = cache
    print(
        f"  upstream-FI cache ready: pages_per_head={cache.pages_per_head}, "
        f"total_pages={cache.total_pages}, cur_seqlen={cache.cur_seqlen}, "
        f"last_page_idx={cache.last_page_idx_py}, "
        f"last_page_len={cache.last_page_len_py}"
    )
    return cache


def _reset_mode_state():
    _upstream_fi_cache_ref[0] = None
    _pdfi._fi_baseline_cache_ref[0] = None
    _pd._quest_cache_ref[0] = None
    _pd._step_timings.clear()
    _pd._cpu_timings.clear()
    _pd._pending_events.clear()
    _pd._enabled = False
    _pd._current_layer = 0


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
    p.add_argument("--sink_size", type=int, default=32)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--scoring_method", default="max", choices=["mean", "max"])
    p.add_argument("--group_agg_method", default="max", choices=["mean", "max"])
    p.add_argument("--unselected_mode", default="drop", choices=["drop"])
    p.add_argument("--compressed_token_rope", default="mixed",
                   choices=["mixed", "block_center"])
    p.add_argument("--comp_kv_quant", default="none",
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
                   help="Batch size. Only 1 is supported by this adapter — "
                        "wired for parity with future multi-batch work.")
    p.add_argument("--verify_upstream", action="store_true",
                   help="Per-layer upstream-FI vs SDPA shadow verification. "
                        "Threshold 0.02.")
    p.add_argument("--verify_threshold", type=float, default=0.02)
    p.add_argument("--cudagraph", action="store_true",
                   help="Capture one decode step into a CUDA graph and "
                        "benchmark replay. plan() is called once at build "
                        "time so graph capture is safe by construction.")
    p.add_argument("--cudagraph_replays", type=int, default=0)
    p.add_argument("--torch_profiler_trace", default=None)

    args = p.parse_args()
    if args.batch_size != 1 and args.mode in ("dct_upstream_flashinfer", "all"):
        print(
            f"WARN: --batch_size={args.batch_size} > 1 is not supported by "
            f"the upstream-FI adapter; dct_upstream_flashinfer will fail."
        )
    return args


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------
def _rebind_instance_forward(model, attn_cls, forward_fn):
    for module in model.modules():
        if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
            module._old_forward = types.MethodType(forward_fn, module)


def _patch_baseline(model, args, original_forward):
    """Shared full-KV FI baseline (imported from profile_decode_flash_infer)."""
    restore_forward(args.model, original_forward, model)
    attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
    attn_cls.forward = profiled_baseline_flashinfer_forward
    _rebind_instance_forward(model, attn_cls, profiled_baseline_flashinfer_forward)


def _patch_dct_sdpa(model, args, original_forward):
    restore_forward(args.model, original_forward, model)
    replace_llama_attn(
        page_size=args.page_size, top_k=args.top_k,
        sink_size=args.sink_size, recent_size=args.recent_size,
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
        sink_size=args.sink_size, recent_size=args.recent_size,
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
# Runner
# ---------------------------------------------------------------------------
def _run_one_mode(model, tokenizer, args, mode, original_forward):
    _reset_mode_state()
    _pd._sync_mode = args.sync
    _pd._profile_topk_impl.value = args.topk_impl

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
    past_key_values = pre_allocate_cache(past_key_values, extra_tokens=extra)
    print(f"  Converted to pre-allocated cache (+{extra} tokens)")

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
            f"page_size={args.page_size})..."
        )
        _pdfi._fi_baseline_cache_ref[0] = build_fi_baseline_cache(
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
        )
        _bc = _pdfi._fi_baseline_cache_ref[0]
        print(
            f"  FI baseline cache ready: capacity_pages={_bc.capacity_pages}, "
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
            print(
                f"  [verify_upstream] worst max-abs-diff across "
                f"{len(per_layer_diffs)} layers x {all_steps} steps = "
                f"{worst:.3e} (threshold = {args.verify_threshold:.0e})"
            )
            head = min(8, all_steps)
            for s in range(head):
                ok = per_step_worst[s] < args.verify_threshold
                print(
                    f"    step {s}: {per_step_worst[s]:.3e}  "
                    f"worst layer={per_step_layer[s]:>2}  "
                    f"{'OK' if ok else 'FAIL'}"
                )
            if all_steps > head:
                print(f"    ... ({all_steps - head} more steps)")
            verify_ok = worst < args.verify_threshold
            print(f"  [verify_upstream] overall: {'PASS' if verify_ok else 'FAIL'}")

    # Optional CUDA graph benchmark.
    graph_stats = None
    if args.cudagraph:
        torch.cuda.synchronize(device)
        num_replays = args.cudagraph_replays or args.num_decode_steps
        current_pos = prefill_len + args.warmup_steps + args.num_decode_steps

        static_input = next_token.clone()
        static_pos = torch.tensor([current_pos], device=device, dtype=torch.long)

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
            t0 = time.perf_counter()
            for _ in range(num_replays):
                g.replay()
            torch.cuda.synchronize(device)
            per_replay_ms = (time.perf_counter() - t0) * 1000 / num_replays
            graph_tok_s = 1000.0 / per_replay_ms
            graph_stats = (per_replay_ms, graph_tok_s)
            print(f"  CUDA graph: {per_replay_ms:.3f} ms/step  ({graph_tok_s:.1f} tok/s)")
        except Exception as e:
            print(f"  CUDA graph benchmark failed: {type(e).__name__}: {e}")
            graph_stats = None

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
        print_profile(mode, avg_total, tok_s, timings, num_layers, cpu_timings)
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
        print(f"  {'Mode':<28} {'ms/tok':>10} {'tok/s':>10} {'vs baseline':>14}")
        print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 14}")
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
            print(f"  {mode:<28} {avg:>10.2f} {tok:>10.1f} {vs_str:>14}")

        any_graph = any(r[3] is not None for r in results.values())
        if any_graph:
            print(f"\n  {'Mode (graph)':<28} {'ms/tok':>10} {'tok/s':>10} {'vs baseline':>14}")
            print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 14}")
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
                print(f"  {mode:<28} {gp:>10.2f} {gts:>10.1f} {vs_str:>14}")

    if verify_state:
        print()
        for mode, ok in verify_state.items():
            tag = "PASS" if ok else "FAIL"
            print(f"  [verify_upstream] {mode}: {tag}")


if __name__ == "__main__":
    main()
