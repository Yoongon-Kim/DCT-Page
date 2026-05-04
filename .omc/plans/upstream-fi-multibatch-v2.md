# Upstream FlashInfer Multibatch — v2 (Architect+Critic-revised)

**Status:** READY FOR EXECUTION (iteration 1 of 5)
**Supersedes:** `upstream-fi-multibatch.md` (v1)
**Mode:** SHORT (consensus)
**Date:** 2026-05-04

---

## RALPLAN-DR

### Principles
1. Preserve guardrails — kernel asserts encode invariants; relaxing them globally is a regression vector. Prefer opt-in flags.
2. Smallest revertable surface first — every edit must have a one-file revert path; risky edits land last.
3. Verify before claim — each phase has an empirically-grounded gate, not a "should work" gate.
4. Single source of layout truth — vbatch ordering `v = b*H + h` is asserted at build, not assumed at use.
5. Match upstream contract — head-local logical IDs into FI; physical IDs only inside the assemble-into-buf path.

### Decision Drivers
1. Correctness at bsz>1 without re-architecting the kernel chain.
2. Memory ceiling on A6000 48 GiB — bsz=4/32K at the OOM cliff; quote the math.
3. Revertability — staging must allow rolling back the riskiest edit independently of the others.

### Options
| Opt | Description | Verdict |
|---|---|---|
| **A** | Kernel flag `allow_head_local_multibatch=False` default; upstream sets True | **CHOSEN** |
| B | Kernel relax (global assert removal) | REJECTED — guardrail loss |
| C | Per-batch independent FI calls | REJECTED — perf regression |

---

## Approach

vbsz = `B*H`, per-(b,h) page pools, head-local topk IDs.

### Shape changes (unchanged from v1)
- `buf_7d → buf_8d`: `(L, B, H, P, 2, ps, 1, d)` C-contiguous; flat view `(L, B*H*P, 2, ps, 1, d)`
- `indices_buf_3d`: `(B, H, page_budget)`
- `indices_flat_buf`: `(B*H*pb,)`
- `indptr_buf`: `(B*H+1,)`
- `last_page_len_buf`: `(B*H,)`
- `last_page_idx`: `(B,)` int32 broadcast head-local logical
- `head_offset`: `(B*H, 1)` viewed as `(B, H, 1)` with values `(b*H + h) * pages_per_head`
- Vbatch ordering: `v = b*H + h`
- Q reshape: `(B, num_qo_heads, 1, d) → (B, H, gs, d) → (B*H, gs, d)`

---

## Memory Math (A6000 48 GiB ceiling)

**Per-layer KV bytes** = `2 * num_kv_heads * page_size * head_dim * dtype_bytes * num_pages`
For Llama-3.1-8B: `H=8`, `ps=32`, `d=128`, `bf16=2 B`, `L=32`.

| Config | num_pages (per b) | Per-layer KV (B) | All-layer KV (B) | Weights | Workspace+scratch | Total | Headroom |
|---|---|---|---|---|---|---|---|
| bsz=1, 16K | 512 | 2·8·32·128·2·512 = 33.6 MiB | 1.05 GiB | 16 GiB | ~3 GiB | ~20 GiB | OK |
| bsz=2, 16K | 512×2 | 67.1 MiB | 2.10 GiB | 16 GiB | ~3 GiB | ~21 GiB | OK |
| bsz=4, 16K | 512×4 | 134.2 MiB | 4.19 GiB | 16 GiB | ~3 GiB | ~23 GiB | **OK (gate)** |
| bsz=1, 32K | 1024 | 67.1 MiB | 2.10 GiB | 16 GiB | ~3 GiB | ~21 GiB | OK |
| bsz=2, 32K | 1024×2 | 134.2 MiB | 4.19 GiB | 16 GiB | ~3 GiB | ~23 GiB | OK |
| bsz=4, 32K | 1024×4 | 268.4 MiB | 8.39 GiB | 16 GiB | ~3 GiB | ~27 GiB | **best-effort** |

**Note**: Pre-allocated paged buf is `(L, B, H, P, 2, ps, 1, d)` — same KV footprint, just re-shaped. The "best-effort" 32K bsz=4 risk is workspace + activation scratch (CUDA-graph capture roughly doubles transient pools); empirical OOM observed historically near 27 GiB on A6000.

---

## Edit Sequence (re-ordered: backend → driver → kernel-flag)

### Phase 1 — Backend (`speed/upstream_flashinfer_backend.py`)

**Goal**: Multibatch shapes wired, per-(b,h) FI plan/run with head-local IDs, invariant asserts at build, no kernel changes yet.

#### 1.1 Build-time edits (`__init__` / build helper)
- Allocate `buf_8d` with shape `(L, B, H, P, 2, ps, 1, d)`, C-contiguous; expose flat view `buf_flat` of `(L, B*H*P, 2, ps, 1, d)`.
- Allocate `indices_buf_3d` `(B, H, page_budget)`, `indices_flat_buf` `(B*H*pb,)`, `indptr_buf` `(B*H+1,)`, `last_page_len_buf` `(B*H,)`, `last_page_idx` `(B,)` int32.
- Allocate `head_offset` `(B*H, 1)` filled with `(b*H + h) * pages_per_head`.
- **Invariant asserts (Architect rec #2)**:
  ```python
  assert buf_8d.is_contiguous()
  assert indices_buf_3d.is_contiguous()
  expected = (torch.arange(B*H, dtype=torch.int32, device=dev) * pages_per_head).view(B*H, 1)
  assert head_offset.view(B*H, 1).eq(expected).all()
  ```
- **CUDA graph sanity (Architect rec #5)**: at build, after first `wrapper.plan(...)`, print `wrapper._plan_info` (or scheduler partition) once at vbsz∈{32, 64} to confirm scheduler isn't degenerate. Print-once gated by env `OMC_FI_PLAN_DEBUG=1`.

#### 1.2 `_pack_preallocated_to_paged_upstream` (Critic rec C)
- Add `bsz: int = 1` kwarg.
- Loop `for b in range(bsz): buf_8d[l, b, ...]  ←  layer.keys[b, ...]`-equivalent packing.
- Zero new copies for bsz=1 (default path identical to today).

#### 1.3 Q reshape + parity gate (Critic rec B + G)
- Forward path: `q_full = query_states.reshape(B, H, gs, d).reshape(B*H, gs, d).contiguous()`.
- **Parity gate (dev-only, dropped after Phase 1 PASS)**: at bsz=1, also compute `q_old = query_states.reshape(nkv, gs, d)` and assert `(q_full - q_old).abs().max() == 0`. Gated by env `OMC_FI_PARITY_GATE=1`. Removed after one clean run.

#### 1.4 FI plan/run wiring
- `wrapper.plan(indptr_buf, indices_flat_buf, last_page_len_buf, num_qo_heads=B*H*gs, num_kv_heads=B*H, ...)` — virtual-batch-per-head.
- Output reshape `(B*H, gs, d) → (B, num_qo_heads, 1, d)` matches today's bsz=1 contract.

#### 1.5 Acceptance gate — Phase 1
- Existing bsz=1 verify (`--mode dct_upstream_fi --batch_size 1 --verify_against_baseline`) PASSES with same threshold (0.02).
- Parity gate PASSES with bit-equal Q.
- No kernel edits yet; topk path still asserts bsz=1, so bsz>1 will *not* run yet — that's expected.

---

### Phase 2 — Driver (`speed/profile_decode_upstream_flash_infer.py`)

**Goal**: Driver supports `--batch_size B`, surfaces dct_sdpa pre-flight, threshold ladder, and rollback flag-default plumbing. Still no kernel edits.

#### 2.1 CLI surface
- `--batch_size B` (default 1).
- `--verify_threshold_override` (optional, float; if set, skips ladder).
- `--allow_head_local_multibatch` flag plumbing — passes through to backend, which passes to kernel call site (default True at upstream call site, the kernel still has bsz=1 assert so this is dormant until Phase 3).

#### 2.2 dct_sdpa pre-flight at bsz=2 (Critic rec F)
- Before running any FI bsz=2 path, run `--mode dct_sdpa --batch_size 2 --verify_against_baseline` and require PASS.
- If FAIL → halt; the bug is upstream of FI (in DCT or SDPA assembly), not in this work.
- Print clear log line: `[PREFLIGHT] dct_sdpa bsz=2 verify: PASS|FAIL`.

#### 2.3 Verify threshold ladder (Critic rec A)
1. Run `--mode dct_upstream_fi --batch_size 1 --verify_against_baseline` → record `wd1 = empirical max-abs-diff`.
2. For bsz>1 verify: `threshold = max(0.02, 3 * wd1)`.
3. Log both `wd1` and `threshold` in run output: `[VERIFY] wd1=<x> threshold=<y> (ladder)`.

#### 2.4 Rollback path documented in driver header
- Module docstring lists: "to disarm multibatch: set `--allow_head_local_multibatch=False` (or revert call site default)".

#### 2.5 Acceptance gate — Phase 2
- bsz=1 verify still PASS (regression check).
- dct_sdpa bsz=2 PASS (pre-flight).
- bsz>1 path will fail at the kernel assert until Phase 3 — expected; the driver should log this clearly: `[BLOCKED] kernel assert: relax pending Phase 3`.

---

### Phase 3 — Kernel flag (`triton_kernels.py` `topk_sort_and_pack_triton`)

**Goal**: Smallest, riskiest edit. Lands last so revert is mechanical.

#### 3.1 Signature change (Architect rec #1, Option A)
- Add kwarg `allow_head_local_multibatch: bool = False`.
- Replace `assert pages_per_batch == 0 → bsz == 1` with:
  ```python
  if pages_per_batch == 0 and bsz != 1:
      assert allow_head_local_multibatch, (
          "pages_per_batch=0 with bsz>1 requires allow_head_local_multibatch=True "
          "(caller must use head-local logical page IDs; vbsz layout is caller's contract)"
      )
  ```
- Recent-region write `last_page_idx[b] + recent_offsets[r]` continues to produce head-local logical IDs (Critic rec D — confirmed: kernel just adds offsets to whatever `last_page_idx[b]` contains, which the caller sets head-local).

#### 3.2 Upstream call site (in `upstream_flashinfer_backend.py`)
- Pass `allow_head_local_multibatch=True` at the upstream-FI call only.
- Fork/Quest/SDPA call sites unchanged → default False → guardrail intact for them.

#### 3.3 Acceptance gate — Phase 3 (the real gates)
1. **bsz=1 verify**: PASS at threshold 0.02 (regression).
2. **bsz=2/16K verify**: PASS at ladder threshold.
3. **bsz=4/16K verify**: PASS at ladder threshold (HARD GATE — Architect rec #4).
4. **bsz=2/32K verify**: PASS at ladder threshold.
5. **bsz=4/32K verify**: BEST-EFFORT (Architect rec #4). If OOM, log `[OOM] bsz=4/32K: documented fallback`. Not a blocker.
6. **CUDA graph sanity**: `_plan_info` print at vbsz=32 (bsz=4, H=8) shows non-degenerate scheduler partition.

---

## Risk Surface (revised)

| # | Risk | Mitigation | Detection |
|---|---|---|---|
| R1 | **Q reshape ambiguity** (Critic B) | Parity gate at bsz=1 in Phase 1; contiguity force after reshape (Critic G) | Bit-equality assert in dev mode |
| R2 | **vbatch ordering aliasing** (silent corruption if `head_offset` mis-laid) | Build-time invariant asserts (Architect #2) | Assert fails at build, never reaches kernel |
| R3 | **Kernel guardrail loss** for non-upstream callers | Option A flag default False (Architect #1) | Other callers can't accidentally opt in |
| R4 | **FI scheduler degenerate at high vbsz** | `_plan_info` print at vbsz=32, 64 (Architect #5) | Manual inspection; non-degenerate = many partitions |
| R5 | **OOM at bsz=4/32K** | Math quoted above (~27 GiB); marked best-effort; bsz=4/16K is the gate | torch.cuda.OutOfMemoryError trapped, logged |
| R6 | **DCT path bsz>1 bug masking FI bug** | dct_sdpa bsz=2 pre-flight (Critic F) | Pre-flight FAIL → halt before Phase 3 |
| R7 | **Verify threshold too tight or too loose at bsz>1** | Ladder: `max(0.02, 3*wd1)` (Critic A) | wd1 logged; threshold logged |
| R8 | **`q_flat` non-contiguous after reshape** (Critic G) | Force `.contiguous()` after reshape | Eager catches stride mismatch |

---

## Rollback Paths

| Phase | Revert action | Effect |
|---|---|---|
| Phase 3 (kernel) | Set upstream call site's `allow_head_local_multibatch=False` (one-line revert in `upstream_flashinfer_backend.py`) | Multibatch un-armed; kernel unchanged |
| Phase 3 (kernel, full) | `git revert` the `triton_kernels.py` commit | Flag removed; default behavior restored |
| Phase 2 (driver) | `git revert` the `profile_decode_upstream_flash_infer.py` commit | CLI back to bsz=1 only |
| Phase 1 (backend) | `git revert` the `upstream_flashinfer_backend.py` commit | All shapes back to 7-D / B=1 |

Each phase is one file's diff. No phase depends on a later phase's existence at runtime (Phase 3 dormant until call site flips True).

---

## Acceptance Criteria (consolidated)

### Hard gates (must PASS to ship)
- [ ] bsz=1 verify PASS at 0.02 (regression).
- [ ] dct_sdpa bsz=2 verify PASS (pre-flight, Critic F).
- [ ] bsz=1 parity gate PASS (Q reshape, Critic B).
- [ ] bsz=2/16K verify PASS at ladder threshold.
- [ ] **bsz=4/16K verify PASS at ladder threshold** (replaces former bsz=4/32K hard gate).
- [ ] bsz=2/32K verify PASS at ladder threshold.
- [ ] CUDA graph `_plan_info` non-degenerate at vbsz=32.
- [ ] All build-time invariant asserts pass at every (B, ctx) tested.

### Best-effort gates (log + document, not block)
- [ ] bsz=4/32K verify (OOM-permitting; document fallback if OOM).
- [ ] Throughput at bsz>1 logged for future regression detection.

### Documentation gates
- [ ] Verify threshold ladder logged (`wd1` and `threshold`).
- [ ] Memory math table reproduced in driver run log.
- [ ] Rollback paths in driver module docstring.

---

## Files Touched

| File | Phase | Approx LoC | Purpose |
|---|---|---|---|
| `speed/upstream_flashinfer_backend.py` | 1, 3 (call site flip) | ~80 | 8-D bufs, per-(b,h) plan/run, invariant asserts, flag pass-through |
| `speed/profile_decode_upstream_flash_infer.py` | 2 | ~40 | `--batch_size`, threshold ladder, dct_sdpa pre-flight, rollback docstring |
| `triton_kernels.py` | 3 | ~10 | `allow_head_local_multibatch` kwarg + conditional assert |

Total: ~130 LoC across 3 files, 3 commits (one per phase).

---

## Follow-ups (out of v1/v2 scope)

- **Per-layer free pattern** mirroring `flashinfer_backend.py:181-204` `_build_paged_buf_per_layer` — defers paged-buf alloc to per-layer scope, freeing memory between layers. Tracked separately; would relax bsz=4/32K from best-effort to gate.
- **vbsz=64+ scheduler tuning** if `_plan_info` shows degeneracy.
- **Ladder threshold formula** (currently `3 * wd1`) — if Phase 3 reveals systematic drift, swap for relative-error-aware formula.

---

## Open Questions Resolved in v2

| Q | Resolution |
|---|---|
| FI scheduler at vbsz=64? | Built-time `_plan_info` print at vbsz∈{32, 64} (Architect #5). |
| Memory ceiling math? | Quoted in Memory Math table; bsz=4/32K best-effort, bsz=4/16K gate. |
| Kernel relax vs flag? | Flag (Option A); guardrail preserved for other callers. |
| bsz>1 verify threshold? | Ladder `max(0.02, 3*wd1)` (Critic A). |
| Q reshape correctness? | Parity gate at bsz=1, force contiguous (Critic B + G). |
| Hidden bsz=1 assumption in DCT? | dct_sdpa bsz=2 pre-flight (Critic F). |

---

## ADR (lightweight, SHORT mode)

- **Decision**: Add `allow_head_local_multibatch` flag to `topk_sort_and_pack_triton`; upstream call site sets True. Build 8-D paged buf with explicit vbatch ordering. Stage edits backend → driver → kernel-flag. Ship at bsz=4/16K hard gate; bsz=4/32K best-effort.
- **Drivers**: Correctness at bsz>1 without kernel-chain rewrite; A6000 48 GiB ceiling; revertability per-file.
- **Alternatives considered**: (B) global kernel relax — rejected for guardrail loss; (C) per-batch FI calls — rejected for perf regression (defeats vbsz batching).
- **Why chosen**: (A) preserves fork-side guardrail (default False), one-call-site rollback un-arms multibatch, ~10 LoC kernel diff is the smallest revertable surface that satisfies the upstream contract.
- **Consequences**: A flag now exists in the kernel signature (minor cognitive load); upstream backend grows to 8-D shapes (one-way migration on that file); 32K bsz=4 stays best-effort until per-layer free pattern lands.
- **Follow-ups**: Per-layer paged-buf alloc to relax 32K bsz=4 to gate; vbsz=64 scheduler tuning if `_plan_info` shows degeneracy.
