# Plan: Extend upstream-FlashInfer DCT decode to bsz>=1

## Context

The upstream-FlashInfer adapter at `speed/upstream_flashinfer_backend.py` and its profiler driver `speed/profile_decode_upstream_flash_infer.py` use a "virtual batch = KV head" trick (vbsz = H) that avoids the per-head `page.cuh` patch in the DCT-Page FlashInfer fork. Today the path is hard-coded to `bsz=1`. The clean extension is **vbsz = B*H** with each `(b, h)` pair owning its own per-head page pool of `pages_per_head` physical pages.

Per memory `feedback_fork_per_head_bsz1_only.md`: the fork's per-head mode at bsz>1 needs a `page.cuh` patch. The upstream extension achieves the same with no FlashInfer-side patch — only Python/Triton edits in this repo.

## Work Objectives

1. Generalize `UpstreamFlashInferPagedKVCache` and its build/append/refresh/run helpers to support `bsz >= 1` with vbsz = `B*H` ordering `v = b*H + h` (h-contiguous within each batch).
2. Relax the multibatch assert in `triton_kernels.py::topk_sort_and_pack_triton` to allow `bsz>1 ∧ pages_per_batch=0` (head-local IDs; per-(b,h) bias applied later by `refresh_upstream_indices_flat`).
3. Update the profiler driver to plumb `bsz` through cache construction, K/V append, and the verify path; drop the `bsz>1` WARN; and validate `--cudagraph` capture/replay at multibatch.

## Guardrails

### Must Have
- Bsz=1 numeric path bit-equivalent to the current code (no regression in verify or perf within noise).
- No FlashInfer-side patch — must stay on stock `BatchDecodeWithPagedKVCacheWrapper`.
- `--verify_upstream` PASS at bsz∈{1, 2, 4} for context_length=32768 (worst max-abs-diff < 0.02).
- Lockstep batch (every batch advances together) — no ragged batch.
- CUDA graph (`--cudagraph`) captures and replays at bsz=2 without error.

### Must NOT Have
- New FlashInfer fork dependencies.
- Ragged-batch / variable seqlen support (out of scope; lockstep only).
- Per-layer pack-and-free OOM mitigation (v1; flat KV stays alive — bsz∈{1,2,4} at 32K is the safe envelope on A6000 48 GiB).
- Changes to `flashinfer_backend.py` (the fork-based sibling); only `upstream_flashinfer_backend.py` is in scope.

## Task Flow

```
Step 1 (kernel) ──► Step 2 (backend) ──► Step 3 (driver) ──► Step 4 (validate)
   |                    |                     |                    |
   relax assert         dataclass + build     forward + verify     bsz=1, 2, 4
   pages_per_batch=0    + append + refresh    + cudagraph path     verify_upstream
```

## Detailed TODOs

### Step 1 — Relax topk assert (kernel)

**File:** `triton_kernels.py` (~line 999-1003)

Current:
```python
if bsz > 1:
    assert pages_per_batch > 0, (
        f"bsz={bsz} requires pages_per_batch > 0 to bias topk physical "
        f"IDs into per-batch page pools (got pages_per_batch={pages_per_batch})"
    )
```

Replace with:
```python
if bsz > 1:
    assert pages_per_batch >= 0, (
        f"bsz={bsz} requires pages_per_batch >= 0 (got pages_per_batch={pages_per_batch}). "
        f"pages_per_batch=0 is valid for head-local-IDs callers (e.g. upstream-FI "
        f"per-(b,h) page pools) where the per-batch bias is applied externally."
    )
```

Rationale: `pages_per_batch=0` means no batch bias — already correct behavior for a per-(b,h) pool design where bias is applied later by `refresh_upstream_indices_flat`. Kernel-side, `PAGES_PER_BATCH=0` is already an accepted constexpr (the existing `pages_per_batch=0` path is exercised by the bsz=1 case). The assert was a guardrail for the fork-style backend; it must not block the upstream-style backend.

**Acceptance:**
- Existing fork-backend tests at `bsz>1 ∧ pages_per_batch>0` still pass (assert still enforces non-negative).
- New upstream-backend path at `bsz>1 ∧ pages_per_batch=0` no longer hits the assert.
- Bsz=1 numerics unchanged.

### Step 2 — Backend generalization (`speed/upstream_flashinfer_backend.py`)

**2a. Dataclass — `UpstreamFlashInferPagedKVCache` (lines 59-111)**

Add `bsz: int` field. Field semantics post-change:

| Field | Old shape | New shape (B = bsz) |
|---|---|---|
| `buf` (FI-facing) | `(L, H*P, 2, ps, 1, d)` | `(L, B*H*P, 2, ps, 1, d)` |
| `buf_8d` (rename `buf_7d`) | `(L, H, P, 2, ps, 1, d)` | `(L, B, H, P, 2, ps, 1, d)` |
| `indices_buf_3d` | `(1, H, page_budget)` | `(B, H, page_budget)` |
| `indices_flat_buf` | `(H * pb,)` | `(B * H * pb,)` |
| `indptr_buf` | `(H+1,) * pb` | `(B*H+1,) * pb` |
| `last_page_len_buf` | `(H,)` | `(B*H,)` |
| `last_page_idx` | `(1,)` | `(B,)` (broadcast head-local logical) |
| `head_offset` | `(H, 1)`, value `h*pages_per_head` | `(B, H, 1)`, value `(b*H + h)*pages_per_head` |

**Backward-compat alias:** keep `buf_7d` as a Python `@property` returning `buf_8d[:, 0]` when `bsz==1`, OR rename across the codebase. Prefer rename (cleaner; only 2 callsites in the driver, both touched here).

**2b. `_pack_preallocated_to_paged_upstream` (lines 114-143)**

Rename signature parameter `buf_7d → buf_8d`. Add `bsz` param. Loop over batch:
```python
for l, layer in enumerate(preallocated_layers):
    for b in range(bsz):
        k = layer.keys[b, :, :prefill_len, :]    # (H, T, d)
        v = layer.values[b, :, :prefill_len, :]
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        k = k.view(num_kv_heads, prefill_pages, page_size, head_dim).to(dtype)
        v = v.view(num_kv_heads, prefill_pages, page_size, head_dim).to(dtype)
        buf_8d[l, b, :, :prefill_pages, 0, :, 0, :].copy_(k)
        buf_8d[l, b, :, :prefill_pages, 1, :, 0, :].copy_(v)
```

(Vectorize the inner B loop later if it shows up in profiles; for v1 it's once at build time so a Python loop is fine.)

**2c. `build_upstream_flashinfer_paged_cache` (lines 146-300)**

- Add `bsz: int = 1` kwarg.
- `total_pages = bsz * num_kv_heads * pages_per_head`.
- Allocate `buf_8d = torch.zeros(num_layers, bsz, num_kv_heads, pages_per_head, 2, page_size, 1, head_dim, ...)`.
- `buf = buf_8d.view(num_layers, total_pages, 2, page_size, 1, head_dim)`.
- `indices_buf_3d = torch.zeros(bsz, num_kv_heads, page_budget, ...)`.
- Sink seeding `indices_buf_3d[:, :, :num_sink_pages] = arange(num_sink_pages)` — broadcasts across both bsz and H.
- `indices_flat_buf = torch.zeros(bsz * num_kv_heads * page_budget, ...)`.
- `head_offset = (torch.arange(bsz * num_kv_heads, ...) * pages_per_head).view(bsz, num_kv_heads, 1)`.
- Initial `torch.add(indices_buf_3d, head_offset, out=indices_flat_buf.view(bsz, num_kv_heads, page_budget))`.
- `indptr_buf = torch.arange(bsz * num_kv_heads + 1, ...) * page_budget`.
- `last_page_len_buf = torch.full((bsz * num_kv_heads,), last_open_len, ...)`.
- `last_page_idx = torch.full((bsz,), last_open_page, ...)`.
- `wrapper.plan(...)` args (`group_size`, 1, `head_dim`, `page_size`) **unchanged** — vbsz comes implicitly from `len(indptr_buf)-1 = B*H`.
- Pass `bsz` to `_pack_preallocated_to_paged_upstream`.
- Return populated dataclass with `bsz=bsz`.

**2d. `append_upstream_flashinfer_cache` (lines 303-337)**

- `cache.last_page_idx.fill_(...)` and `cache.last_page_len_buf.fill_(...)` already broadcast — no changes needed (`fill_` works on any shape).
- `k_flat = new_k.reshape(cache.bsz, cache.num_kv_heads, cache.head_dim)` (was `(num_kv_heads, head_dim)`).
- `cache.buf_8d[layer_idx, :, :, page_idx, 0, slot, 0, :].copy_(k_flat)` — slices over both batch and head.
- `cache.buf_8d[layer_idx, :, :, page_idx, 1, slot, 0, :].copy_(v_flat)`.

**2e. `refresh_upstream_indices_flat` (lines 340-355)**

```python
torch.add(
    cache.indices_buf_3d,                 # (B, H, pb)
    cache.head_offset,                    # (B, H, 1) broadcasts
    out=cache.indices_flat_buf.view(cache.bsz, cache.num_kv_heads, cache.page_budget),
)
```
(Drop the `[0]` indexing on `indices_buf_3d`.)

**2f. `upstream_flashinfer_decode_attention` (lines 358-384)**

- `q_flat = query_states.reshape(cache.bsz, cache.num_kv_heads, cache.group_size, cache.head_dim).reshape(cache.bsz * cache.num_kv_heads, cache.group_size, cache.head_dim)`.
- `out = cache.wrapper.run(q_flat, cache.buf[layer_idx])` returns `(B*H, gs, d)`.
- Return `out.view(cache.bsz, cache.num_kv_heads * cache.group_size, 1, cache.head_dim)`.

**Acceptance for Step 2:**
- `build_upstream_flashinfer_paged_cache(..., bsz=1)` produces a cache that is element-wise identical to the old one (excluding the new `bsz` field and the renamed `buf_8d`). Spot-check: `cache.indices_flat_buf.shape == (1*H*pb,)`, `cache.indptr_buf.shape == (1*H+1,)`, etc.
- `cache.head_offset[b, h, 0] == (b*H + h) * pages_per_head` for B>1.
- `wrapper.plan` is still called once at build time with the same `(group_size, 1, head_dim, page_size)` 4-tuple — only `len(indptr_buf)` grows.

### Step 3 — Driver (`speed/profile_decode_upstream_flash_infer.py`)

**3a. CLI (lines 477-497)**

Drop the bsz>1 WARN block. Replace help text on `--batch_size`:
```python
p.add_argument(
    "--batch_size", type=int, default=1,
    help="Lockstep batch size. KV cache memory scales linearly: "
         "per-layer KV bytes = bsz * num_kv_heads * pages_per_head * 2 * page_size * head_dim * dtype_bytes. "
         "On A6000 (48 GiB), bsz∈{1,2,4} at 32K is the validated envelope.",
)
```
Remove the bsz!=1 conditional WARN (lines 492-496).

**3b. `_build_upstream_fi_cache` (lines 382-424)**

Pass `bsz=args.batch_size` to `build_upstream_flashinfer_paged_cache`. Update the print line to include `bsz`:
```python
print(
    f"  Building upstream-FI cache: layers={num_layers}, bsz={args.batch_size}, "
    f"num_sink_pages={num_sink_pages}, top_k={args.top_k}, "
    f"num_recent_pages_fixed={num_recent_pages_fixed}, "
    f"page_budget={page_budget}, vbsz={args.batch_size * num_kv_heads}, "
    f"group_size={num_qo_heads // num_kv_heads}..."
)
```

**3c. `profiled_dct_upstream_flashinfer_forward` (lines 104-376)**

- Step 2 counter advance (lines 219-227): unchanged — `last_page_idx.fill_(...)` and `last_page_len_buf.fill_(...)` broadcast across any shape.
- Step 6 topk: pass `pages_per_batch=0` explicitly (or let it default — explicit is clearer). The Step 1 assert change permits this combo.
- Step 7 K/V copy (lines 306-309): rewrite from `buf_7d` to `buf_8d` with batch axis:
  ```python
  k_flat = key_states[:, :, -1:, :].reshape(cache.bsz, cache.num_kv_heads, cache.head_dim)
  v_flat = value_states[:, :, -1:, :].reshape(cache.bsz, cache.num_kv_heads, cache.head_dim)
  cache.buf_8d[self.layer_idx, :, :, page_idx, 0, slot, 0, :].copy_(k_flat)
  cache.buf_8d[self.layer_idx, :, :, page_idx, 1, slot, 0, :].copy_(v_flat)
  ```
- Verify path (lines 322-350): generalize per-h loop to per-(b,h):
  ```python
  buf_l_8d = cache.buf_8d[self.layer_idx]   # (B, H, P, 2, ps, 1, d)
  page_budget = cache.page_budget
  last_page_len = cache.last_page_len_py
  full_len = (page_budget - 1) * cache.page_size + last_page_len
  k_pages = []
  v_pages = []
  for b in range(cache.bsz):
      kb = []
      vb = []
      for h in range(_num_kv_heads):
          sel_bh = cache.indices_buf_3d[b, h].long()    # head-local IDs
          kv_bh = buf_l_8d[b, h][sel_bh]                # (page_budget, 2, ps, 1, d)
          k_h = kv_bh[:, 0, :, 0, :].reshape(page_budget * cache.page_size, self.head_dim)
          v_h = kv_bh[:, 1, :, 0, :].reshape(page_budget * cache.page_size, self.head_dim)
          kb.append(k_h[:full_len])
          vb.append(v_h[:full_len])
      k_pages.append(torch.stack(kb, dim=0))
      v_pages.append(torch.stack(vb, dim=0))
  k_ref = torch.stack(k_pages, dim=0)        # (B, H, full_len, d)
  v_ref = torch.stack(v_pages, dim=0)
  sdpa_out = F.scaled_dot_product_attention(
      query_states, k_ref, v_ref,
      is_causal=False, enable_gqa=True,
  )
  max_diff = (attn_output_fi.float() - sdpa_out.float()).abs().max().item()
  ```

**3d. CUDA graph path (lines 748-787)**

No code changes required — the path uses `static_input` and `static_pos` which `model(...)` will reshape to (B, 1) automatically when `next_token` was generated at bsz=B. The pinned FI buffers (`indptr_buf`, `indices_flat_buf`, `last_page_len_buf`) all already use `use_cuda_graph=True` and broadcast-fill (`fill_`) for counters. Validate experimentally; no expected change.

**Acceptance for Step 3:**
- bsz=1 invocation produces identical printouts (modulo the new `bsz=1` token) and identical numeric verify diffs vs. pre-change baseline.
- bsz=2 invocation runs without shape/index errors all the way through verify and (optionally) cudagraph replay.
- Verify diffs at bsz>1 are <0.02 worst-case across all (layer, step, batch) triples.

### Step 4 — Validation

Run from repo root with `CUDA_VISIBLE_DEVICES=1` (or whichever A6000 is free).

```bash
# 4a. bsz=1 regression — must match pre-change verify output.
CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_upstream_flash_infer.py \
  --mode dct_upstream_flashinfer --verify_upstream \
  --batch_size 1 --context_length 32768 \
  --num_decode_steps 32 --warmup_steps 4

# 4b. bsz=2.
CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_upstream_flash_infer.py \
  --mode dct_upstream_flashinfer --verify_upstream \
  --batch_size 2 --context_length 32768 \
  --num_decode_steps 32 --warmup_steps 4

# 4c. bsz=4 (memory ceiling check). If OOM at 32K, fall back to 16K for this row.
CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_upstream_flash_infer.py \
  --mode dct_upstream_flashinfer --verify_upstream \
  --batch_size 4 --context_length 32768 \
  --num_decode_steps 32 --warmup_steps 4

# 4d. all-mode comparison at bsz=2 (baseline + dct_sdpa + dct_upstream_fi).
CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_upstream_flash_infer.py \
  --mode all --batch_size 2 --context_length 32768 \
  --num_decode_steps 64 --warmup_steps 8

# 4e. CUDA graph at bsz=2.
CUDA_VISIBLE_DEVICES=1 python speed/profile_decode_upstream_flash_infer.py \
  --mode dct_upstream_flashinfer --batch_size 2 --context_length 32768 \
  --num_decode_steps 64 --warmup_steps 8 --cudagraph
```

**Pass criteria:**
- 4a–4c print `[verify_upstream] overall: PASS` with worst max-abs-diff < 0.02.
- 4a verify diffs are within numerical noise (≤ 1e-9 absolute change vs. pre-change run, ideally identical) — protects bit-equivalence guardrail.
- 4d shows DCT+upstream-FI ≈ +10–15% vs baseline (per memory `project_upstream_flashinfer_works.md`).
- 4e captures and replays without exception; per-replay ms/step is within ~1.1× of the bsz=1 cudagraph baseline scaled by B (i.e., ~B× the per-step work).

## Success Criteria

1. `--batch_size 1` verify PASS (bit-equivalent to pre-change baseline).
2. `--batch_size 2` verify PASS (worst max-abs-diff < 0.02).
3. `--batch_size 4` verify PASS at 32K (or documented OOM fallback to 16K).
4. `--mode all --batch_size 2`: COMPARISON table prints; DCT+upstream-FI shows expected speedup vs baseline.
5. `--cudagraph --batch_size 2`: graph captures and replays without error.
6. Bsz=1 numeric path remains bit-equivalent (verify diffs unchanged from pre-change run).
7. No FlashInfer-side patch required.

## Risk Surface

| Risk | Symptom | Mitigation |
|---|---|---|
| **Vbatch ordering mismatch** (FI expects v = b*H+h vs h*B+b) | verify FAIL; first b is correct, others wrong | Anchor on `indptr_buf = arange(B*H+1)*page_budget` — vbatch v owns `indptr[v]:indptr[v+1]`. With `head_offset.view(B, H, 1)` setting `(b, h, 0) → (b*H + h)*pages_per_head`, `indices_flat_buf.view(B, H, pb)[b, h]` lands at flat slot `(b*H+h)*pb` which matches indptr. Bsz=1 is the trivial special case; check at bsz=2 with a per-(b,h) sentinel before launching kernel. |
| **Index aliasing across batches** (b=0's indices physically overlap b=1's pool) | verify FAIL; off-by-`pages_per_head` slot data | `head_offset[b, h] = (b*H + h)*pages_per_head` — verify with `assert (head_offset.flatten() == torch.arange(B*H, ...) * pages_per_head).all()` at build time. |
| **OOM at bsz=4, 32K** | CUDA OOM during build | Per-layer KV bytes ≈ `B * H * pages_per_head * 2 * ps * d * dtype_bytes`. For Llama-3.1-8B (H=8, d=128, bf16=2B), ps=32, prefill_pages≈1024+slack, decode_pages≈4: `4 * 8 * 1028 * 2 * 32 * 128 * 2 ≈ 539 MiB/layer × 32 layers ≈ 17 GiB`. Fits. If not, fall back to context=16K. |
| **plan-time scheduler** sized for `B*H` virtual batches at build time | First decode step segfaults if vbsz changes | Vbsz is fixed at build time and never changes. The driver computes it from `args.batch_size * num_kv_heads`. Since `wrapper.plan(...)` is called once at build with the post-bsz `indptr_buf`, scheduler memory is sized correctly. The contract — vbsz fixed, indices_flat refreshed per step — extends naturally. |
| **`fill_` broadcast on `last_page_idx` shape change** | Off-by-one when last_page_idx is read by Stage 5 kernel | Step 5 (`topk_sort_and_pack_triton`) requires `last_page_idx.shape == (bsz,)`. Old shape was `(1,)`, which matched `bsz=1`. New shape `(bsz,)` matches the new bsz directly. Counter advance writes a single scalar via `fill_`, which broadcasts to `(B,)` correctly. |
| **DCT runtime state at bsz>1** | DCT compress/segment paths may have hidden bsz=1 assumptions | The `_update_comp_cache`, `segment_kv`, `score_pages_triton` paths all already handle `bsz` from `query_states.shape[0]` per CLAUDE.md / config. The driver already passes `bsz=args.batch_size` to `chunked_prefill`. Spot-validate by running `--mode dct_sdpa --batch_size 2` (no FI involvement) to confirm the DCT path itself is bsz-clean before adding the upstream-FI extension. |
| **`--mode dct_sdpa` already at bsz>1** | If `dct_sdpa` fails at bsz=2, the bug isn't in this work | Out of scope — but flag as a dependency check before running 4d. If `dct_sdpa` at bsz=2 fails, file a separate bug; the upstream-FI work can still be validated against the SDPA path **at bsz=1** for verify equivalence. |

## Rollback Plan

If validation fails:

1. **Step 1 (kernel) regression:** revert the assert relaxation. Kernel is unchanged otherwise.
2. **Step 2 (backend) regression:** the dataclass and helpers are isolated to one file. Revert `speed/upstream_flashinfer_backend.py` to the prior commit. The only external dependency is the `bsz=1` default kwarg, so callers that don't pass `bsz` see the original behavior.
3. **Step 3 (driver) regression:** revert `speed/profile_decode_upstream_flash_infer.py`. The bsz=1 path is restored.
4. **Partial fix:** if bsz=2 verify passes but bsz=4 OOMs, document the safe envelope in CLAUDE.md and ship bsz∈{1, 2} only.

The change set is three files; a single `git revert` returns to bsz=1-only behavior with no cross-file orphans.

## Open Questions

- Should we vectorize the `_pack_preallocated_to_paged_upstream` batch loop? (v1: no — once at build time. v2: maybe, if profiles show it.)
- Should the `head_offset` shape be `(B, H, 1)` or flat `(B*H, 1)`? Both work mathematically; `(B, H, 1)` broadcasts cleanly against `indices_buf_3d` of shape `(B, H, pb)`. Picked `(B, H, 1)`.
- Should we expose `pages_per_batch` as a kwarg in `topk_sort_and_pack_triton` even for the upstream backend (for future ragged-batch extension)? v1 leaves it at default `0`.
