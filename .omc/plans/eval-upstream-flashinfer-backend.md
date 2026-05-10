# Add `upstream_flashinfer` attention backend to all production eval scripts

**Plan only — no code edits.** Initial implementation plan for Architect review and Critic evaluation.

---

## RALPLAN-DR Summary

### Principles (load-bearing)
1. **SDPA bit-identity preserved.** The default `--attention_backend sdpa` path must execute the same code that runs today. No behavior or perf delta when the flag is unset, modulo the new dispatch case + `_upstream_fi_cache_ref` import-time module state. Bit-identity claim narrowed: scoring outputs (`page_scores`, `attn_output`) are byte-identical when `--attention_backend` defaults to `sdpa` AND `--verify_upstream_fi` is off.
2. **Drop-mode-only is a hard gate, not a silent downgrade.** Upstream-FI does not implement `compressed`; the script must error at startup with a clear message rather than silently fall back to SDPA, because the user's intent (asking for upstream-FI) is incompatible with what `compressed` requires.
3. **Lazy cache init keeps `model.generate()` untouched.** The eval scripts already rely on HF `generate()` for sampling, EOS handling, attention masks, etc. Building the FI cache on the *first* `q_len==1` step (after prefill is complete inside the same generate call) avoids splitting prefill / decode at the script layer.
4. **Reuse, don't re-implement.** The new forward delegates to `upstream_flashinfer_backend.py` (already validated end-to-end by the profile driver) — no duplicated cache/build/append/refresh code.
5. **Per-generate teardown is mandatory.** Eval scripts loop over hundreds of samples; the cache and PreAllocatedLayer flat-KV state must be torn down between **every HF `model.generate()` call** (note: some benchmarks may issue multiple generates per sample on retries — the contract is per-generate, not per-sample). Otherwise memory will leak and the next generate's prefill will crash on the `_fi_mode` shim.
6. **FI runs are score-reproducible but not byte-reproducible.** SDPA with deterministic kernels reproduces bit-for-bit across runs. FlashInfer's split-kv reduction order depends on per-batch tile counts (which depend on `pages_per_head`, which depends on per-sample `prefill_len`), so logits may drift on bf16. Same seed + same upstream-FI config must produce final scores within ±0.5 pp across two runs; if not, FI numerics drift exceeds the noise envelope and must be escalated.

### Decision Drivers (top 4)
1. **Minimum disruption to eval CLIs and call sites.** One new flag per script, one pass-through kwarg into the existing `replace_*_attn` calls.
2. **Clean fallback / error semantics.** SDPA short-KV fallback already exists in the forward. `compressed` mode is hard-rejected at argparse. Non-`page_attention` modes are unaffected (they don't even call `replace_*_attn`).
3. **Reuse the validated upstream backend without re-implementing it.** The `UpstreamFlashInferPagedKVCache` + `build_*` + `append_*` + `refresh_upstream_indices_flat` + `upstream_flashinfer_decode_attention` API in `speed/upstream_flashinfer_backend.py` is already the single source of truth; the forward is the only new code.
4. **Eager-mode tradeoff explicitly accepted.** Per memory `project_phase2b_stage9_fail.md`, fork-FI in eager-mode at 32K bsz=1 *lost* baseline by ~5% — bottleneck is CPU dispatch, not the attention kernel. The user has verbally accepted this tradeoff for the eval path (CUDA-graph integration is a separate plan). Plan must surface a startup `print()` showing the expected ±X% delta vs SDPA at the user's config (sourced from profile driver), and a `--profile_decode_first` recipe in §8 lets users validate before committing to a full sweep.

### Viable Options

| | A — Lazy-init cache inside the forward (RECOMMENDED) | B — Manual prefill + cache build + manual decode loop | C — Pre-generate hook via `LogitsProcessor` / `StoppingCriteria` |
|---|---|---|---|
| **How** | Forward checks `_upstream_fi_cache_ref[0]`; if `None` and we're at the first decode step (`q_len==1` and `prev_len >= min_decode_kv_len_for_paging`), read `prefill_len` from the already-allocated `past_key_values.layers[0]._seen` (prefill ran `pre_allocate_cache` already), then call `build_upstream_flashinfer_paged_cache(...)` and store the cache. From step 2 onward the cache is hot. | Replace `model.generate()` with a custom loop: prefill via `model(input_ids, use_cache=True)`, then `pre_allocate_cache`, then `build_upstream_flashinfer_paged_cache`, then per-step decode. | Register an HF callback that fires between prefill and the first decode step and builds the cache. |
| **Pros** | Zero changes to eval script generate-call surface (just flag plumbing + cache cleanup hook). Cache lifecycle is co-located with the only place that knows the layer/dim metadata. Cleanly reuses HF's EOS / attention-mask / sampling behavior. | Most explicit lifecycle. Easy to instrument timing. | Theoretically least invasive to forward. |
| **Cons** | Forward gains lazy-init branch (cheap; runs once per generate). Need an explicit `reset_upstream_fi_cache_state` between generates (centralized via `_generate_with_upstream_fi` helper, §3). | 3 of 5 eval scripts have non-trivial `model.generate()` keyword surfaces (LongBench v1 has `do_sample=False`, EOS lists, custom token ids). Re-implementing them invites drift. AIME/GPQA pass through `eval_ruler` helpers — duplication multiplies. | HF `LogitsProcessor` runs *after* every forward, not between prefill and decode. `StoppingCriteria` runs on the cumulative ids and also doesn't fire between prefill and decode. There is no native HF hook between prefill and the first decode step — would need to monkey-patch `GenerationMixin._sample` or similar, which is more invasive than Option A and brittle across transformers versions. |

**Recommended: Option A (lazy-init in the forward).** Option B's churn against five eval scripts (each with its own generate keyword surface) outweighs the localized branch cost in the forward; Option C is invalidated by the absence of the right HF hook at the prefill-to-decode boundary (verified: `LogitsProcessor` and `StoppingCriteria` do not fire there in transformers 5.x).

> **Invalidation rationale for Option C:** transformers 5.x `GenerationMixin._sample` runs the prefill forward and the first decode forward inside the same call without firing any registered processor / stopping-criteria callback at the prefill→decode transition. Confirmed by the absence of any such hook in the public extension API. The only "hook" that fires at that boundary is the forward itself — which is exactly Option A.

### Mode
**DELIBERATE.** Promotion triggered by Architect items 1 + 4 (cache-lifecycle interaction with HF `generate()` is more subtle than initially estimated: the `pre_allocate_cache` idempotency hole would null-deref on second call against `_fi_mode=True` layers, and the per-sample teardown obligation must be centralized through one helper rather than scattered across 5 eval scripts). DELIBERATE mode adds a pre-mortem (§11), expanded test plan (§9 T14–T19), and a per-instance state model (§3) instead of a cross-module global.

---

## 1. Source-of-truth forward

**Location:** new function `dct_page_attention_forward_upstream_flashinfer` in `dct_page_attention.py`, placed immediately after `dct_page_attention_forward_flashinfer` (`dct_page_attention.py:1661–1905`). Same module so it shares `_dct_page_cfg`, `_maybe_reset_dct_runtime_state`, `apply_rotary_pos_emb`, and the SDPA / short-KV fallback delegate without circular imports.

**Module-level cache ref:** add `_upstream_fi_cache_ref = [None]` next to `_flashinfer_cache_ref = [None]` at `dct_page_attention.py:43`. Mirrors the existing fork-FI pattern exactly.

**Lazy-import policy** for the upstream backend (mirrors lines 1769–1778 in the fork variant):
```
import sys as _sys, os as _os
_speed_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "speed")
if _speed_dir not in _sys.path:
    _sys.path.insert(0, _speed_dir)
from upstream_flashinfer_backend import (
    build_upstream_flashinfer_paged_cache,
    append_upstream_flashinfer_cache,
    refresh_upstream_indices_flat,
    upstream_flashinfer_decode_attention,
)
```
This keeps `flashinfer` import out of the cold path for users who never enable the backend.

**8-step structure (mirrors `profile_decode_upstream_flash_infer.py:122–411` minus the profiler instrumentation):**
1. `cfg.unselected_mode != "drop"` → `raise NotImplementedError` with a message pointing to the SDPA backend (same wording the fork variant uses at line 1686). `cfg.continuous_rope` → same NotImplementedError.
2. `_maybe_reset_dct_runtime_state(self, past_key_values)` (existing hook at `dct_page_attention.py:1117`).
3. `q_len > 1` → delegate to `dct_page_attention_forward(...)` (prefill path; identical to fork variant lines 1700–1708).
4. Short-KV fallback: compute `min_len_for_paging = max((cfg.num_sink_pages + cfg.top_k + 1 + cfg.num_recent_pages) * cfg.page_size, getattr(cfg, "min_decode_kv_len_for_paging", 0))`, peek `prev_len = int(past_key_values.layers[self.layer_idx].get_seq_length())`, and if `prev_len + q_len < min_len_for_paging` delegate to `dct_page_attention_forward(...)`. Identical to fork lines 1710–1727.
5. **Lazy-init the upstream-FI cache (NEW vs fork variant).** First time we get past the short-KV gate, `_upstream_fi_cache_ref[0]` is `None`. We must:
   - **Do NOT call `pre_allocate_cache` here.** Prefill already ran it at `dct_page_attention.py:1216-1220` (with `extra_tokens=64`), which converted HF `DynamicLayer`s into `PreAllocatedLayer` and set `_preallocated=True` on the cache object at `:1218-1220`. **`pre_allocate_cache` itself has no idempotency guard inside its body** — calling it again against `_fi_mode=True` layers (where `_build_paged_bufs_per_layer_upstream` already nulled `layer.keys` at `speed/upstream_flashinfer_backend.py:197-203`) would null-deref on `bsz, heads, seq_len, dim = k.shape` at `dct_page_attention.py:68` (`PreAllocatedLayer.from_dynamic_layer`). The build path only reads `layer.keys[:prefill_len]` then frees flat KV via the `_fi_mode` shim, so re-extending flat KV is unnecessary.
   - Compute `prefill_len = int(past_key_values.layers[0]._seen)` directly from the already-allocated layer's seen-counter.
   - Read `max_decode_steps` from `getattr(self, "_upstream_fi_build_kwargs", {}).get("max_decode_steps", 0)` (per-instance attribute set by the eval harness via the `_generate_with_upstream_fi` helper; see §3).
   - Call `build_upstream_flashinfer_paged_cache(preallocated_layers=past_key_values.layers, prefill_len=prefill_len, page_size=cfg.page_size, num_kv_heads=self.config.num_key_value_heads, head_dim=self.head_dim, num_qo_heads=self.config.num_attention_heads, num_layers=self.config.num_hidden_layers, max_decode_steps=max_decode_steps, dtype=past_key_values.layers[0].keys.dtype, device=past_key_values.layers[0].keys.device, num_sink_pages=cfg.num_sink_pages, top_k=cfg.top_k, num_recent_pages_fixed=cfg.num_recent_pages, bsz=hidden_states.shape[0])`.
   - Assign to `_upstream_fi_cache_ref[0]`.
   - **This must run on layer 0 only** — guard with `if self.layer_idx == 0 and _upstream_fi_cache_ref[0] is None`. Subsequent layers in the same step see the populated ref.
   - **Defensive double-clear:** if `_upstream_fi_cache_ref[0]` is non-None at *entry* to a generate call (e.g. previous generate crashed mid-build leaving `_upstream_fi_cache_ref[0]` populated but per-layer state partial), `_generate_with_upstream_fi` clears it before generate runs. See §3.
6. Steps 1–8 from the profile driver lines 203–390, removing all `_pd._enabled` event recording: QKV proj + optional q_norm/k_norm → RoPE → counter-only `past_key_values.update` → layer-0 counter advance on the cache → `paged_k/v` slice from `cache.buf_views[self.layer_idx]` → `_update_comp_cache` → `score_pages_triton` (with `score_use_quest_minmax` branch — see note below) → `topk_sort_and_pack_triton(..., pages_per_batch=0, allow_head_local_multibatch=True)` → cache K/V write → `refresh_upstream_indices_flat(cache)` → `upstream_flashinfer_decode_attention(query_states, cache, self.layer_idx)` → output proj.

**Differences from fork variant (`dct_page_attention_forward_flashinfer`):**
- Cache layout: `cache.buf_views[l]` is `(B, H, P, 2, ps, 1, d)`; the fork uses a flat `cache.buf[l]` plus `paged_views_from_buf(...)`. The upstream forward slices `buf_views[l][:, :, num_sink_pages:num_sink_pages+num_pages, 0/1, :, 0, :]` directly — **no `paged_views_from_buf` call**.
- Step 5 kernel call passes `pages_per_batch=0, allow_head_local_multibatch=True` (head-local IDs). The fork uses `pages_per_batch=cache.pages_per_batch if cache.bsz>1 else 0` (batch-biased IDs).
- New step 7 sub-call `refresh_upstream_indices_flat(cache)` which biases head-local IDs into the flat physical buffer FI's wrapper reads. Fork has no equivalent because its kernel writes batch-biased IDs directly into `indices_buf`.
- Lazy cache build (§1 step 5 above) — fork relies on the driver to build pre-decode.

**Quest-minmax branch (simultaneous closure):** the profile driver at `profile_decode_upstream_flash_infer.py:280-297` calls `score_pages_triton` unconditionally (no quest path). For eval parity with fork-FI / SDPA, the new forward must mirror the fork variant's branch at `dct_page_attention.py:1820–1832`: if `cfg.score_use_quest_minmax` then `_update_quest_metadata` + `_score_pages_quest`, else `score_pages_triton`. **To prevent creating new drift in the opposite direction, this same change also adds the Quest branch to the profile driver at `profile_decode_upstream_flash_infer.py:280-297`** — ~10 lines mirroring the fork variant logic. Both surfaces gain the branch in this single change.

**Verify path:** include the per-layer SDPA shadow comparison from profile driver lines 355–385 (gathers per-(b,h) using `cache.indices_buf_3d[b, h]` head-local IDs, runs `F.scaled_dot_product_attention`, appends `max_diff` to `self._verify_diffs`). Gated by `getattr(self, "_verify_upstream", False)`. Off by default in eval; turned on by `--verify_upstream_fi` (see Section 6).

**Recommendation re: profile driver refactor.** Keep `profile_decode_upstream_flash_infer.py` as-is for now and treat the new `dct_page_attention_forward_upstream_flashinfer` as the canonical forward. *Follow-up* (post-merge): refactor the profile driver to wrap the canonical forward + emit events around its sub-steps via a thin `with _ProfileEvents()` context manager. Don't bundle the refactor with this change because (a) it would touch a working profiler that has its own test surface, and (b) the canonical forward needs to land + soak in eval first.

## 2. Backend dispatch

**File:** `dct_page_attention.py:1911–1926`.

**Change:** extend `_select_dct_forward` from a 2-way to a 3-way dispatch:

```
def _select_dct_forward(attention_backend):
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
```

**`replace_*_attn` signatures already accept the new value.** Verified at `dct_page_attention.py:1929`, `:1999`, `:2071` — all three (`replace_qwen2_attn`, `replace_qwen3_attn`, `replace_llama_attn`) take `attention_backend="sdpa"` as a kwarg and pass it through to `_select_dct_forward` (called inside the patched-forward closure). **No signature change needed.**

The fork-FI path (`"flashinfer"`) remains live for the existing `speed/profile_decode_flash_infer.py` driver. It is not exposed in eval scripts because no eval consumer has built the fork cache.

## 3. Cache lifecycle in eval scripts (Option A — lazy init)

**Granularity contract:** the cleanup contract is **per HF `model.generate()` call**, not strictly per-sample. Some benchmarks may issue multiple generates per sample on retries (e.g. EOS dodging, format-retry loops in LongBench v2). The `_generate_with_upstream_fi` helper enforces teardown around *every* generate call, so the contract is robust to whatever the per-sample loop does.

**Build contract (refined from §1 step 5):**
- Built on the first `q_len==1` step that crosses the short-KV gate, on layer 0 only, **after** prefill's `pre_allocate_cache` has already run (no extra `pre_allocate_cache` here — see §1 step 5).
- `max_decode_steps` read from per-attention-module attribute `self._upstream_fi_build_kwargs["max_decode_steps"]`. Set by `_generate_with_upstream_fi` immediately before `model.generate()` runs. Source: `tokens_to_generate` (RULER) / `task_max_gen_tokens` (LongBench) / `args.max_new_tokens` (AIME/GPQA), padded by `+16` for safety (matches profile driver `profile_decode_upstream_flash_infer.py:425`).
- **Memory cost:** flat KV was already allocated by prefill's `pre_allocate_cache` to `(prefill_len + 64)` per layer per head — ~268 MiB at bsz=1/32K/Llama-3.1-8B. **The flat KV is then freed by `_build_paged_bufs_per_layer_upstream(..., free_flat_kv=True)` immediately after copy** (`speed/upstream_flashinfer_backend.py:197-203`), so peak transient mem is one layer's `{flat, paged}` pair, not all layers'.

**Module-level state added to `dct_page_attention.py`:**
```
_upstream_fi_cache_ref = [None]   # cache singleton, cleared by reset_upstream_fi_cache_state
_DCT_RUNTIME_STATE_ATTRS         # existing tuple at :1087-1114; do NOT add _dct_runtime_cache_ref or _verify_diffs
```

**Per-instance state set on each attention module (NEW pattern — matches `_dct_runtime_cache_ref`, `_page_scores_buf`, `_verify_diffs`):**
```
module._upstream_fi_build_kwargs = {"max_decode_steps": 0}
module._verify_upstream = False     # only when --verify_upstream_fi
```

**Patch-time wiring (inside `replace_*_attn`):** walk all attention modules on the model after monkey-patching the forward, and stash an empty `_upstream_fi_build_kwargs = {}` dict on each. Eval harness mutates this per-instance attribute via `_set_upstream_fi_build_kwargs(model, **kwargs)` (a small helper that walks modules and updates the dict). This matches the existing per-attention-module state pattern (`_dct_runtime_cache_ref`, `_page_scores_buf`, `_verify_diffs`) and avoids cross-module global mutation.

**Centralized helper `_generate_with_upstream_fi(model, input_ids, max_new_tokens, **gen_kwargs)`** — lives in `dct_page_attention.py` (or a small new helper module imported by all eval scripts):
```
def _generate_with_upstream_fi(model, input_ids, max_new_tokens, **gen_kwargs):
    # Defensive double-clear: if a previous generate crashed mid-build,
    # _upstream_fi_cache_ref[0] may be non-None with partial per-layer state.
    if _upstream_fi_cache_ref[0] is not None:
        _upstream_fi_cache_ref[0] = None
    _set_upstream_fi_build_kwargs(model, max_decode_steps=max_new_tokens + 16)
    try:
        return model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
    finally:
        reset_upstream_fi_cache_state(model)
        torch.cuda.empty_cache()
```
This handles three failure modes uniformly:
- EOS-early-stop (normal return): `finally` runs teardown.
- `KeyboardInterrupt`: propagates, `finally` still runs.
- OOM mid-generate (`RuntimeError`): teardown runs **before** the exception bubbles, freeing the previous sample's cache so the next sample can allocate.

**`reset_upstream_fi_cache_state(model)` semantics:**
- Set `_upstream_fi_cache_ref[0] = None`.
- For each attention module on the model: clear `_dct_runtime_cache_ref` AND `_verify_diffs` explicitly (these are deliberately NOT in `_DCT_RUNTIME_STATE_ATTRS` because `_dct_runtime_cache_ref` is the *guard variable* that triggers `_maybe_reset_dct_runtime_state` at `dct_page_attention.py:1119-1121` — adding it to the auto-reset tuple would defeat the guard). Then run the existing `_DCT_RUNTIME_STATE_ATTRS` cleanup (`_page_scores_buf`, `_dct_proj_matrix`, etc. from `:1087-1114`).
- Choice (a) **`_verify_diffs` is cleared per generate**, not accumulated for whole-run histograms — this gives cleaner per-sample stats. The eval harness harvests `_verify_diffs` *before* calling `reset_upstream_fi_cache_state` if it wants to log per-sample percentile stats.
- Caller is responsible for any `torch.cuda.empty_cache()` (the helper above does it as part of the `finally` block).

**Eval-script integration:** all five eval scripts replace their `model.generate(...)` call with `_generate_with_upstream_fi(model, input_ids, max_new_tokens, ...)` **only when `args.attention_backend == "upstream_flashinfer"`**. SDPA path keeps direct `model.generate` to preserve bit-identity. Locations:
- `eval_ruler.py:649` — gate the call at the dispatch into the helper.
- `eval_longbench_v1.py:574` — same.
- `eval_longbench_v2.py:294` — same.
- `eval_aime25.py` / `eval_gpqa.py` — they call into `eval_ruler` helpers; the wrapping happens once in the shared helper and AIME/GPQA inherit it for free.

**Why the cache must die between generate calls:** `cache.buf_views` carry a (potentially long) prefill plus all decode steps from the previous generate. Reusing the cache for the next call would silently re-use stale physical pages or OOM trying to grow them.

**Why flat KV is fine across generates:** HF creates a fresh `DynamicCache` per `model.generate()` call by default, so the freed-layer state from generate N never reaches generate N+1 — the only module-level state we need to clean up is `_upstream_fi_cache_ref` plus the per-attention-module attributes.

## 4. Compressed-mode handling

**Hard error at startup, not silent downgrade.** Add a guard at the top of each eval script's `main()` (or in a shared helper):

```
if (
    args.mode == "page_attention"
    and getattr(args, "attention_backend", "sdpa") == "upstream_flashinfer"
    and args.unselected_mode != "drop"
):
    raise SystemExit(
        f"--attention_backend upstream_flashinfer requires "
        f"--unselected_mode drop (got {args.unselected_mode!r}). "
        f"The upstream-FlashInfer backend does not implement compressed mode. "
        f"Use --attention_backend sdpa for compressed mode."
    )
```

Place this:
- `eval_ruler.py`: before `apply_monkey_patch(args)` is called (around line 322 / `apply_monkey_patch` entry).
- `eval_longbench_v1.py`, `eval_longbench_v2.py`: same — before the corresponding `replace_*_attn` block.
- `eval_aime25.py`, `eval_gpqa.py`: in `parse_args()` post-parse, since they reuse `eval_ruler.apply_monkey_patch` and `eval_ruler.load_model_and_tokenizer`.

**Why hard error rather than silent SDPA fallback:** users typing `--attention_backend upstream_flashinfer` are explicitly comparing the two backends; silently running SDPA produces invalid comparison numbers and is the worse failure mode. The forward already raises `NotImplementedError` for compressed mode (Section 1, step 1) — front-loading it to argparse turns a confusing run-time crash into a clean preflight error.

## 5. Eval-script surface changes

### 5.1 Argparse additions (one per script)

```
parser.add_argument(
    "--attention_backend",
    type=str,
    default="sdpa",
    choices=["sdpa", "upstream_flashinfer"],
    help=(
        "Attention backend for page_attention mode. "
        "'sdpa' (default): assemble + torch.scaled_dot_product_attention "
        "(unchanged production path). "
        "'upstream_flashinfer': stock FlashInfer paged-decode kernel via "
        "virtual-batch-per-(batch, KV head) layout (drop mode only). "
        "Ignored for non-page_attention modes."
    ),
)
parser.add_argument(
    "--verify_upstream_fi",
    action="store_true",
    help=(
        "When --attention_backend upstream_flashinfer, run a per-layer SDPA "
        "shadow comparison and log the per-step max-abs-diff distribution. "
        "bf16 noise floor on this hardware is 0.05 — see project memory "
        "project_upstream_fi_multibatch.md."
    ),
)
```

Locations:
- `eval_ruler.py`: after the existing `--use_triton`/`--no_triton` flag (somewhere near the DCT-Page args block).
- `eval_longbench_v1.py`, `eval_longbench_v2.py`: same place.
- `eval_aime25.py`, `eval_gpqa.py`: add to parser; they already mirror the DCT arg surface (verified — `--page_size`, `--top_k`, `--unselected_mode`, etc. all exist in both at lines 224, 225, 235 / 186, 187, 197).

### 5.2 Pass-through into `replace_*_attn`

**`eval_ruler.py:330–391`** — `apply_monkey_patch`. Append `attention_backend=args.attention_backend` to all three blocks (`replace_llama_attn` at 331, `replace_qwen3_attn` at 350, `replace_qwen2_attn` at 369).

**`eval_longbench_v1.py:749–785`** — same three blocks.

**`eval_longbench_v2.py:507–544`** — same three blocks.

**`eval_aime25.py` / `eval_gpqa.py`** — these do `from eval_ruler import apply_monkey_patch, load_model_and_tokenizer` (lines 46 of each). The `args` namespace already carries through; once `--attention_backend` is added to their argparse, `apply_monkey_patch(args)` will see it. **No additional plumbing needed in those two files beyond argparse.**

### 5.3 `_generate_with_upstream_fi` integration

In each eval script's per-sample loop body, replace the bare `model.generate(...)` call with a backend-gated dispatch:
```
if args.mode == "page_attention" and args.attention_backend == "upstream_flashinfer":
    from dct_page_attention import _generate_with_upstream_fi
    output_ids = _generate_with_upstream_fi(
        model, input_ids, max_new_tokens=tokens_to_generate,  # or task_max_gen_tokens / args.max_new_tokens
        **other_gen_kwargs,
    )
else:
    output_ids = model.generate(input_ids, max_new_tokens=tokens_to_generate, **other_gen_kwargs)
```

The helper (defined in §3) handles `_set_upstream_fi_build_kwargs` (per-instance attribute on each attention module), `try/finally` teardown via `reset_upstream_fi_cache_state`, defensive double-clear at entry, and `torch.cuda.empty_cache()`. **Eval scripts do not touch any module-level globals or per-module attributes themselves.**

Locations:
- `eval_ruler.py:649` (`output_ids = model.generate(...)`, `tokens_to_generate` already in scope at line 591).
- `eval_longbench_v1.py:574` and `eval_longbench_v2.py:294` (their own per-task token budget).
- AIME25/GPQA: inherit through `eval_ruler` helper reuse — no per-script change beyond argparse.

### 5.4 Verify-flag plumbing

When `args.verify_upstream_fi` is set, walk every attention module on the model **once at model-load time** (after `replace_*_attn`) and set `module._verify_upstream = True`. The flag is sticky for the whole eval run:
```
if args.attention_backend == "upstream_flashinfer" and args.verify_upstream_fi:
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            module._verify_upstream = True
```
After each generate, harvest `module._verify_diffs` and log percentile stats (p50 / p99 / max) **before** `_generate_with_upstream_fi`'s `finally` block clears `_verify_diffs`. To do this cleanly without leaking implementation details, expose a small helper `harvest_verify_diffs(model) -> dict[layer_idx, list[float]]` in `dct_page_attention.py` that the eval script calls between `model.generate()` and the implicit teardown — practically, the harvest is wired into a callback the helper takes.

Recommended pattern: extend `_generate_with_upstream_fi` with an optional `on_post_generate=None` callback parameter:
```
output_ids = _generate_with_upstream_fi(
    model, input_ids, max_new_tokens=tokens_to_generate,
    on_post_generate=lambda m: log_verify_stats(harvest_verify_diffs(m), sample_idx),
    **other_gen_kwargs,
)
```
The callback runs *inside* the `try` block, after `generate` returns successfully, before the `finally` teardown clears state. Optional: write per-sample histograms to a JSON sidecar (see §6 acceptance + T19).

### 5.5 Confirm non-`page_attention` modes are unaffected

Verified in code: every script's model-load path branches on `args.mode == "page_attention"` for the `replace_*_attn` call. Non-`page_attention` modes (`baseline`, `seer_attention`, `multipole_attention`, `quest_attention`, `duo_attention`, `inf_llm`, `shadowkv`, `seer_prefill`, `rope_gap`) never reach `_select_dct_forward`. The new `--attention_backend` flag is silently ignored for those modes — the help text says so. **No changes needed to those branches.**

## 6. Verify / accuracy parity

**Per-layer shadow check** (`--verify_upstream_fi`): the new forward includes the same SDPA gather/compare loop as the profile driver (lines 355–385). Per memory `project_upstream_fi_multibatch.md`, expected per-step max-abs-diff is **<= 0.05 in bf16** (the noise floor); higher means a real bug.

**Canary task choice:** **`niah_multikey_2`** (multi-key needle-in-a-haystack), NOT `niah_single_3`. Rationale: `niah_single_*` is exact under top_k coverage by construction (the answer always lives in the recent window); a multi-key task is far more sensitive to top-k score-rank drift under FI's split-kv reduction order, so it's a stronger gate.

**Acceptance criterion (single config):** Qwen3-8B, ctx=32768, page_size=32, top_k=64, unselected_mode=drop, run RULER `niah_multikey_2` under both `--attention_backend sdpa` and `--attention_backend upstream_flashinfer`. Pass criteria switch from "score-delta in pp" to "per-sample agreement" because at small `n` (n=25) one mismatched sample = 4 pp swing, making any pp-threshold brittle:

- **Per-sample agreement (preferred):** ≥ 24/25 samples agree on per-sample correctness between SDPA and upstream-FI runs. Mismatches must be on samples where both runs agree on a wrong answer (model uncertainty), not where one is right and the other is wrong (numerics drift).
- **Fallback (if per-sample correctness is hard to extract):** bump `num_samples` to 50 and use score-delta thresholds — at n=50, Wilson 95% CI shrinks to ~14 pp and bf16 noise floor `0.05` per `project_upstream_fi_multibatch.md` translates to a tighter envelope:
  - `niah_single_*`: within ±1.0 pp.
  - `niah_multikey_*` / `niah_multivalue`: within ±2.0 pp.
  - Variable-tracking / common-words: within ±3.0 pp.
- **Verify path:** at least 99% of recorded `_verify_diffs` entries are < 0.05; max < 0.10.
- **Reproducibility (NEW, Principle 6):** two runs of the same upstream-FI config with the same seed must produce final scores within ±0.5 pp. If not, FI numerics drift exceeds the expected noise envelope — escalate (likely a split-kv reduction-order non-determinism that needs investigation before continuing the sweep).

**Run as sanity gate before merging.** Capture stdout + summary JSON to `.omc/research/upstream_fi_eval_parity_<date>.md`. Include per-sample peak-GPU-memory (T18) and verify-shadow histogram (T19).

## 7. Sweep-script updates

**Out of scope by user direction.** No new `run_*.sh` shell files will be created in this change. Users who want to sweep `--attention_backend upstream_flashinfer` either:
- Run the eval CLI manually with the new flag (the most common use case during validation), or
- Edit existing `run_*.sh` scripts in place to append `--attention_backend upstream_flashinfer` to the python invocation (one-line change per script; no new file).

**Implication for Quest-minmax × upstream-FI:** the new forward closes the Quest-minmax functional gap (§1, "Quest-minmax branch") so `score_use_quest_minmax=True` remains a valid cell when invoked via the CLI. No sweep-script wiring is added; users compose the cells themselves.

**Implication for `--skip_existing` resume semantics:** unchanged — still honored by the eval CLIs themselves regardless of how they're invoked.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Eager-mode net throughput tradeoff (hoisted to Decision Driver 4).** Per memory `project_phase2b_stage9_fail.md`, fork-FI in eager-mode at 32K bsz=1 *lost* baseline by ~5%; the bottleneck is CPU dispatch, not the attention kernel. The user has explicitly accepted this tradeoff; the plan must surface it loudly. | (1) Add a startup `print()` at eval-script entry showing "expected eager-mode delta vs SDPA at this config (per profile driver) ≈ ±X%" — sourced from a small lookup table seeded from the profile driver's measured numbers, keyed on (model, ctx_len). (2) Document in `--help` that net throughput depends on context length and model size. (3) Provide a `--profile_decode_first` recipe in the script docstring that runs `speed/profile_decode_upstream_flash_infer.py` at the user's target config first so they can validate the delta before committing to a full sweep. (4) Reference profile driver numbers in `.omc/research/upstream_fi_eval_parity_*.md`. |
| **Memory pressure at >32K (hard preflight, NOT a soft warning).** A6000 48 GiB is "right at the OOM cliff at 64K" per memory. A soft warning is too easy to ignore and downstream OOM at sample N corrupts the entire sweep run. | Add a **hard preflight check** at startup that computes projected paged-KV memory: `proj_bytes = bsz * num_kv_heads * pages_per_head * 2 * page_size * head_dim * 2 * num_layers` where `pages_per_head = ceil((max(args.seq_lengths) + max_decode_steps) / page_size) - num_sink_pages - num_recent_pages`. **Refuse to start** if `proj_bytes > 0.9 * torch.cuda.get_device_properties(0).total_memory`. Print computed projection and threshold on refusal so the user can adjust `--seq_lengths` or run on a larger GPU. |
| **Cache-cleanup hygiene between generates.** Failure to clean up leaks one cache per generate → OOM after ~3–5 generates on 32K Llama. | Centralized `_generate_with_upstream_fi` helper (§3) wraps every generate in `try / finally` and calls `reset_upstream_fi_cache_state(model)` + `torch.cuda.empty_cache()`. Defensive double-clear at entry covers crash-mid-build cases (where `_upstream_fi_cache_ref[0]` is non-None but per-layer state is partial). |
| **`_fi_mode` PreAllocatedLayer shim coverage.** Verified at `dct_page_attention.py:85-112` — the shim exists and skips flat-KV writes when `_fi_mode=True`. After teardown, the next generate creates a fresh `DynamicCache` from HF, so the next `pre_allocate_cache` call walks fresh `DynamicLayer`s — `_fi_mode` re-flips as the new build runs. **No coverage gap.** |
| **`_alloc_len` dead-code in `_fi_mode`.** In `_fi_mode=True` the `PreAllocatedLayer.update` early-return at `:90` means the grow path at `:97-105` is dead code for FI generates. Not a bug, but easy for the next maintainer to hit a confusing breakpoint. | Spec a code comment for the maintainer at `dct_page_attention.py:90` that clearly says "early return when `_fi_mode=True`; the grow path below at `:97-105` is unreachable for upstream-FI generates" so a reader doesn't try to add logic to the grow path expecting it to fire. |
| **Greedy-only assumption.** Beam search, contrastive decoding, sampling break the lazy-init contract because they may run multiple parallel forward passes per logical "step". Eval scripts use greedy (`do_sample=False`) only. | Hard assert at script startup, not just in `--help`: explicitly forbid `num_beams > 1`, `num_return_sequences > 1`, `do_sample=True` whenever `args.attention_backend == "upstream_flashinfer"`. Inspect `model.generation_config` and any per-script generate kwargs; raise `SystemExit` with a clear message before the first generate. |
| **`max_decode_steps` underestimate causes cache overflow.** `cache.pages_per_head` is sized at build time; if the model generates more than `max_decode_steps` tokens (e.g. EOS-token dodging), the next `append_upstream_flashinfer_cache` raises `RuntimeError: cache overflow` (`upstream_flashinfer_backend.py:441`). | Pad `max_decode_steps` by `+16` like the profile driver does (`profile_decode_upstream_flash_infer.py:425`). For LongBench v2 / AIME25 / GPQA where output budgets can be large, pass `tokens_to_generate * 1.1` minimum. |
| **Compressed-mode silent-fallback bait.** Users may `--unselected_mode compressed --attention_backend upstream_flashinfer` and expect an answer. | Hard error at startup (§4). |
| **Profile driver and eval-side forward will drift.** Two implementations of the same forward = inevitable divergence. | Listed as a follow-up (§10): refactor profile driver to wrap the canonical forward + thin event context manager. Not in scope for this plan. |
| **Quest-minmax simultaneous closure.** The profile driver omits `score_use_quest_minmax`; the eval forward includes it. Asymmetric mirror = future drift. | Same change adds the Quest branch to `profile_decode_upstream_flash_infer.py:280-297` (~10 lines) — both surfaces gain it together. See §1 Quest-minmax branch. |
| **Defensive teardown re-entry.** If the model crashes after layer-0 build but before all layers complete, `_upstream_fi_cache_ref[0]` is non-None but per-layer state is partial. Next generate would see a populated ref and skip rebuild, accessing partial state. | `_generate_with_upstream_fi` always nukes `_upstream_fi_cache_ref[0]` on entry if non-None (defensive double-clear, see §3). |
| **FI numerics non-reproducibility (Principle 6).** Split-kv reduction order depends on per-batch tile counts; bf16 logits may drift across runs of the same config. | Reproducibility test (T14) runs the canary config twice with same seed; gate is ±0.5 pp on final scores. Higher drift = escalation, not acceptance. |

## 9. Test plan

Naming convention: argparse flag is `--attention_backend upstream_flashinfer` (full name; matches the `_select_dct_forward` value at §2 line 178 in argparse spec). The plan also uses the shorthand `upstream-FI` in prose; the CLI value is always `upstream_flashinfer`.

| # | Test | Command (sketch) | Pass condition |
|---|---|---|---|
| T1 | Short-KV fallback | `python eval_ruler.py --mode page_attention --attention_backend upstream_flashinfer --seq_lengths 4096 --num_samples 1 --tasks niah_multikey_2 --base_model Qwen/Qwen3-8B --page_size 32 --top_k 64 --unselected_mode drop` | Runs cleanly using SDPA fallback (KV < `min_decode_kv_len_for_paging`); cache is never built; `_upstream_fi_cache_ref[0]` stays `None`. |
| T2 | Sanity smoke at 32K | Same as T1 but `--seq_lengths 32768`, `--num_samples 1`, `--tasks niah_multikey_2` | Forward fires; cache builds; output non-NaN; pred_text non-empty. |
| T3 | SDPA scoring stability | `python eval_ruler.py --mode page_attention --attention_backend sdpa --tasks niah_multikey_2 ...` (default config, two runs) | Scoring outputs (`page_scores`, `attn_output`) byte-identical when `--attention_backend` defaults to `sdpa` AND `--verify_upstream_fi` is off. **Note:** full-JSON byte-identity is NOT claimed because the new dispatch case + `_upstream_fi_cache_ref` import-time module state changes the import surface. |
| T4 | Per-task parity (canary) | Two runs at `niah_multikey_2` 32K, 25 samples, one each with `--attention_backend sdpa` and `--attention_backend upstream_flashinfer` | Per-sample agreement on ≥ 24/25 samples by per-sample correctness (preferred), or per §6 fallback at n=50 with score-delta thresholds. |
| T5 | Verify shadow | T4 upstream-FI run with `--verify_upstream_fi` added; canary `niah_multikey_2` | ≥99% of `_verify_diffs` entries < 0.05; max < 0.10. |
| T6 | Per-sample memory | T4 plus `nvidia-smi dmon -s u` parallel; record peak mem per sample | Peak mem stable across samples (no monotonic growth → no leak). Within ~1–2 GiB of SDPA peak. |
| T7 | Compressed mode hard-error | `--mode page_attention --attention_backend upstream_flashinfer --unselected_mode compressed ...` | Exits with the explicit error message, no model load attempted. |
| T8 | Qwen3 q_norm/k_norm path | T2 with `--base_model Qwen/Qwen3-8B` (already the default in T2) | Forward runs; QK-norm branch in step 6 fires (verifiable via assert + verify-shadow numerical match). |
| T9 | Llama path | T2 with `--base_model meta-llama/Llama-3.1-8B-Instruct` | Same as T8 but exercising the no-q_norm branch. |
| T10 | Regression on SDPA sweep | Re-run one cell of `run_ruler.sh` (e.g. `qwen_ps32_topk64_cr0.125_drop_*`) post-merge | Output JSON matches pre-change run within hash. |
| T11 | Cache cleanup on exception | Wrap `_generate_with_upstream_fi`'s inner `model.generate` call with a monkey-patched `MagicMock(side_effect=RuntimeError("synthetic OOM"))` for sample 1 of a 3-sample test harness; restore mock after. | Sample 2 builds a fresh cache (`_upstream_fi_cache_ref[0]` is `None` at sample-2 entry) and runs cleanly. Asserts: `finally` block ran (verify via spy on `reset_upstream_fi_cache_state`), `torch.cuda.empty_cache()` was called, no leaked allocations from sample 1. |
| T12 | LongBench v1 + v2 minimal | Run one task of each (`narrativeqa` / one v2 task) at default config under upstream-FI | Score within sweep noise of SDPA; no crash. |
| T13 | AIME25/GPQA minimal | Run 1 problem of AIME25 + 1 question of GPQA-diamond under upstream-FI | Generation completes; output non-empty; cache builds. |
| T14 | Reproducibility (NEW — Principle 6) | Run T4 upstream-FI configuration twice with the same seed | Final scores within ±0.5 pp across the two runs. Higher drift = escalation. |
| T15 | Unit test: `reset_upstream_fi_cache_state` (NEW) | pytest unit on a synthetic 1-layer module with all `_DCT_RUNTIME_STATE_ATTRS` (`_page_scores_buf`, `_dct_proj_matrix`, etc.) set, plus `_dct_runtime_cache_ref` and `_verify_diffs` populated | After call, all listed attrs are cleared on every module; `_upstream_fi_cache_ref[0]` is `None`; `_dct_runtime_cache_ref` and `_verify_diffs` cleared explicitly (not via the auto-reset tuple). |
| T16 | Integration: prefill→decode→teardown lifecycle (NEW) | Build a 1-layer synthetic model, run `_generate_with_upstream_fi` for 8 decode steps, assert teardown completes cleanly | `_upstream_fi_cache_ref[0]` is `None` after the helper returns; module attrs cleared; second `_generate_with_upstream_fi` call on same model rebuilds cache and produces matching output to first call. |
| T17 | E2E RULER 13-task subset (NEW) | Run all 13 RULER tasks at 32K, 10 samples each, under upstream-FI; compare to SDPA baseline | Per-task agreement on ≥ 9/10 samples for each task; no crashes; per-sample peak mem stable across the full 13×10 run. |
| T18 | Per-sample peak GPU memory observability (NEW) | T17 with `torch.cuda.max_memory_allocated()` reset + harvest per sample, written to per-sample JSON sidecar | Histogram (p50/p95/p99/max) emitted in summary; no monotonic growth across samples within a task; peak < hard preflight threshold from §8. |
| T19 | Verify-shadow distribution histogram (NEW) | T17 with `--verify_upstream_fi`; emit per-percentile (`p10, p50, p90, p99, p99.9, max`) of `_verify_diffs` per layer per sample to JSON sidecar | Distribution histogram (not just max) recorded for every (sample, layer); aggregate histogram across all layers stays within bf16 noise envelope (`p99 < 0.05`, `max < 0.10`). |

## 10. ADR

**Decision.** Add `"upstream_flashinfer"` as a third value in the `attention_backend` dispatch in `dct_page_attention.py`, implemented by a new `dct_page_attention_forward_upstream_flashinfer` that lazy-builds the upstream-FI cache on the first decode forward of each generate. Eval scripts (`eval_ruler.py`, `eval_longbench_v1.py`, `eval_longbench_v2.py`, `eval_aime25.py`, `eval_gpqa.py`) gain one new flag (`--attention_backend`) plus an optional `--verify_upstream_fi`, and route their generate calls through a new centralized `_generate_with_upstream_fi(...)` helper that handles per-instance build-kwargs, defensive double-clear, and `try/finally` teardown of cache + per-module state. **Quest-minmax simultaneous closure:** the same change adds the Quest-minmax branch to both the new eval forward AND the existing `profile_decode_upstream_flash_infer.py` (~10 lines) so the two surfaces stay in lockstep. Compressed mode is a hard error; greedy-only is asserted at startup; 64K+ seqlen runs through a hard preflight memory check, not a soft warning.

**Drivers.**
1. Minimum eval-script churn (one flag, one pass-through, one helper call).
2. Reuse the validated `upstream_flashinfer_backend.py` API end-to-end.
3. Preserve the SDPA path bit-identically (scoring outputs only) when the new flag is absent.
4. **Eager-mode tradeoff explicitly accepted.** The user has verbally accepted the eager-mode net-throughput delta vs SDPA at 32K (see memory `project_phase2b_stage9_fail.md`); the plan surfaces it via startup `print()`, `--profile_decode_first` recipe, and parity artifact rather than hiding it in a risk row.

**Alternatives considered.**
- **Option B (split generate):** more invasive, would re-implement HF generate-keyword surfaces five times.
- **Option C (HF callback):** invalidated — there is no HF hook between prefill and the first decode step in transformers 5.x.
- **In-place rewrite of fork-FI variant:** would break the existing `speed/profile_decode_flash_infer.py` driver and regress the validated fork path with no upside.
- **Sweep flag as a new dimension in `run_ruler.sh`:** inflates an already-large matrix and most cells are invalid.
- **Build cache eagerly inside `apply_monkey_patch` / `load_model_and_tokenizer`:** can't — at that point the model has no `past_key_values` yet (prefill hasn't happened).
- **Re-use the same `_flashinfer_cache_ref` slot for both fork and upstream:** type-conflict risk; cleaner to have separate refs.
- **Module-global `_upstream_fi_max_decode_steps`:** rejected in favor of per-instance `module._upstream_fi_build_kwargs` to match the existing per-attention-module state pattern (`_dct_runtime_cache_ref`, `_page_scores_buf`, `_verify_diffs`) and avoid cross-module global mutation.
- **Soft 64K warning:** rejected in favor of a hard preflight memory check that refuses to start when projected paged-KV exceeds 90% of GPU memory; OOM mid-sweep is too costly.

**Why chosen.** Option A is the only path that meets all four drivers simultaneously: zero generate-call surface change, full reuse of the upstream backend, preserved SDPA default for scoring, and explicit eager-mode tradeoff visibility. The lazy-init branch in the forward adds a single conditional that runs once per generate; everything else is mechanical plumbing routed through `_generate_with_upstream_fi`.

**Consequences.**
- One new module-level state in `dct_page_attention.py`: `_upstream_fi_cache_ref` (singleton list).
- One new module-level forward: `dct_page_attention_forward_upstream_flashinfer`.
- One new module function: `reset_upstream_fi_cache_state(model)` for per-generate teardown.
- One new module function: `_generate_with_upstream_fi(model, input_ids, max_new_tokens, **gen_kwargs)` for centralized teardown contract.
- One new module function: `_set_upstream_fi_build_kwargs(model, **kwargs)` for per-instance build-kwargs propagation.
- One new per-attention-module attribute: `module._upstream_fi_build_kwargs` (dict).
- One new flag (`--attention_backend`) plus one optional flag (`--verify_upstream_fi`) in five eval scripts.
- Eager-mode net throughput depends on context length / model — may be flat or slightly negative vs SDPA at 32K (Driver 4). Surfaced via startup print and `--profile_decode_first` recipe.
- **Reproducibility envelope (Principle 6):** FI runs are score-reproducible to ±0.5 pp but not byte-reproducible on bf16 logits; this is acceptable for eval but not for byte-level regression testing.
- Profile driver and eval forward become two implementations of the same logic — drift risk acknowledged, refactor tracked as follow-up.
- Profile driver gains the Quest-minmax branch in this same change (simultaneous closure).

**Follow-ups.**
1. Refactor `speed/profile_decode_upstream_flash_infer.py` to wrap the canonical eval forward with a thin event context manager, so there's only one forward to maintain.
2. CUDA-graph-friendly eval loop (separate plan) — current eager-mode tradeoff is the main net-throughput friction. Per memory `project_cudagraph_decode.md`, graph capture flips the baseline-vs-DCT ordering.
3. Dedicated Qwen3 parity sweep on a larger sample set than T4 (full RULER 13-task × 32K under both backends — covered by T17 baseline; deeper sweep is a follow-up).
4. Optional: extend `--attention_backend upstream_flashinfer` to the oracle diagnostics scripts (`oracle/oracle_ruler.py`, etc.) once the eval path is soaked.
5. (Removed.) Sweep-script work is out of scope per user direction; users invoke the CLI flag directly.
6. Wire `OMC_FI_PLAN_DEBUG=1` (already supported by `build_upstream_flashinfer_paged_cache`) into a CI-style smoke check.

---

## Open questions (to persist into `.omc/plans/open-questions.md`)

1. Should the upstream-FI cache be opt-in via a separate sub-flag like `--upstream_fi_max_decode_steps_padding` for users who want to override the `+16` slack? — *Why it matters:* AIME / LongBench-v2 max output budgets can vary widely; insufficient slack causes mid-generation `cache overflow` errors.

   *Resolved-in-plan items removed:*
   - ~~Verify-shadow distribution as histogram vs stdout~~ — RESOLVED by T19 (per-percentile per-(sample, layer) JSON sidecar).
   - ~~Quest-minmax scoring gap in profile driver~~ — RESOLVED by simultaneous closure (§1, §8, §10): same change adds Quest branch to both eval forward and profile driver.

---

## 11. Pre-mortem (DELIBERATE mode)

Three failure scenarios with detection signals + mitigations:

### Scenario A — Cache leak across samples → OOM by sample 5
**Hypothesis:** an exception in `model.generate` for a single sample never triggers teardown because the eval-script author forgot to wrap in `try/finally`. Cache from sample N stays resident; sample N+1's prefill allocates a second one; OOM by sample 4-5.

- **Detection signals:** monotonic peak-mem growth across samples in T6 / T18 sidecar. `nvidia-smi dmon` shows allocated memory ratcheting up, not oscillating around a baseline. `torch.cuda.max_memory_allocated()` per-sample histogram from T18 has positive trend.
- **Mitigation:** centralized `_generate_with_upstream_fi` helper (§3) is the **only** path to generate under upstream-FI; eval scripts cannot bypass its `try/finally`. Defensive double-clear at entry handles any partial state from a previous crash. T11 actively injects an exception to verify the teardown contract.

### Scenario B — OOM at 64K because soft warning was ignored
**Hypothesis:** user runs `--seq_lengths 65536 --attention_backend upstream_flashinfer` on A6000 48 GiB; the soft warning (original plan) is invisible in a long startup log; cache build OOMs at 64K, corrupting any partial sweep state.

- **Detection signals:** `RuntimeError: CUDA out of memory` during the first decode forward; sweep run dies at sample 1 of the 64K cell with no JSON output written.
- **Mitigation:** hard preflight check in §8 — projects paged-KV memory before model load, refuses to start if `> 0.9 * total_memory`. Prints the projection and threshold so the user knows what to adjust. Cannot be silently overridden.

### Scenario C — FI numerics drift on `niah_multikey_2` exceeds ±2 pp
**Hypothesis:** split-kv reduction order in FI's split-kv path produces bf16 logit drift that compounds across decode steps. By the end of generation, top-1 prediction differs from SDPA on enough samples that the 24/25 agreement threshold fails.

- **Detection signals:** T4 fails per-sample agreement; T5 verify-shadow shows `_verify_diffs` p99 > 0.05 (above bf16 noise floor); T14 reproducibility test fails (two same-seed runs differ by > 0.5 pp), confirming non-deterministic reduction.
- **Mitigation:** if T14 passes (drift is consistent across runs but exceeds ±2 pp): document the drift and consider it acceptable for the "approximate" backend. If T14 fails (drift is non-deterministic): escalate — inspect FlashInfer split-kv reduction; consider pinning `pages_per_head` to a value that forces a single split; if no fix, narrow the supported configs (e.g. require ctx ≤ 16K, where pages_per_head fits in a single split).

---

## 12. Implementation Order

Land changes in this dependency order so each step builds on a verifiable green state.

### Step 1: Module-level state and dispatch (foundational)
- Add `_upstream_fi_cache_ref = [None]` to `dct_page_attention.py:43`.
- Extend `_select_dct_forward` at `:1911-1926` to handle `"upstream_flashinfer"`.
- Add new function stub `dct_page_attention_forward_upstream_flashinfer` (raises `NotImplementedError` initially) so dispatch compiles cleanly.
- **Verify:** SDPA path still passes T3 (scoring stability).

### Step 2: New eval forward + Quest-minmax simultaneous closure
- Implement `dct_page_attention_forward_upstream_flashinfer` per §1 (steps 1-8, with the corrected step 5 — no `pre_allocate_cache` call).
- Mirror the Quest-minmax branch into `profile_decode_upstream_flash_infer.py:280-297`.
- **Verify:** profile driver still runs end-to-end with Quest-minmax both off and on.

### Step 3: Lifecycle helpers
- Add `_set_upstream_fi_build_kwargs(model, **kwargs)`.
- Add `reset_upstream_fi_cache_state(model)` (with explicit `_dct_runtime_cache_ref` + `_verify_diffs` clearing, NOT via `_DCT_RUNTIME_STATE_ATTRS`).
- Add `_generate_with_upstream_fi(model, input_ids, max_new_tokens, on_post_generate=None, **gen_kwargs)`.
- Wire patch-time per-instance attribute init inside `replace_*_attn` (empty `_upstream_fi_build_kwargs = {}` on each attention module).
- Add `harvest_verify_diffs(model)` helper.
- Add `_alloc_len` dead-code maintainer comment at `dct_page_attention.py:90` per §8.
- **Verify:** T15 (unit test for `reset_upstream_fi_cache_state`), T16 (integration prefill→decode→teardown).

### Step 4: Argparse + hard preflights
- Add `--attention_backend` and `--verify_upstream_fi` to all 5 eval scripts (§5.1).
- Add hard preflights at startup (§8): compressed-mode reject (§4), greedy-only assert, 64K memory preflight.
- Add `replace_*_attn` pass-through for `attention_backend=` kwarg (§5.2).
- **Verify:** T7 (compressed hard-error), preflight refuses 64K on small GPU.

### Step 5: Eval-script integration
- Replace per-sample `model.generate` call with `_generate_with_upstream_fi` in 3 scripts (§5.3) gated by backend flag.
- Wire `--verify_upstream_fi` callback (§5.4) using `on_post_generate` parameter.
- Add startup `print()` showing expected eager-mode delta vs SDPA (§8 row 1).
- **Verify:** T1, T2 (smoke), T8, T9 (Qwen3 + Llama paths).

### Step 6: Sweep scripts (skipped per §7)
- No new shell files. Users invoke the CLI flag directly or edit existing `run_*.sh` in place.
- **Verify:** N/A.

### Step 7: Validation gate (DELIBERATE mode)
- Run T4 (canary parity at `niah_multikey_2`), T5 (verify shadow), T6 (memory), T11 (exception injection), T14 (reproducibility), T17 (E2E 13-task), T18 (memory observability), T19 (verify histogram).
- Capture results to `.omc/research/upstream_fi_eval_parity_<date>.md`.
- **Gate:** all of T1-T19 must pass before merging. T14 failure or T18 monotonic-growth detection → rollback and re-investigate.
