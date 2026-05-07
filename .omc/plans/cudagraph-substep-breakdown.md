# Plan v2: Same-Process Eager Forced-Seal Microbench for `4_compress` Reconciliation

Iteration 2 of RALPLAN-DR. Architect rejected the v1 two-process / `align_target=compress` design (would crash on `wrapper.plan()`) and the byte-identity-violating label edit. This plan implements the recommended pivot: a flag-gated, same-process eager microbench that measures the average `_update_comp_cache` GPU time per fire and reports it as the truth row in the existing reconciliation block.

Target file: `/home/yoongonkim/DCT-Page/speed/profile_decode_upstream_flash_infer.py` (sole edit surface).
Out of scope: `speed/profile_decode_flash_infer.py` and `dct_page_attention.py` — read-only references for verifying behavior.

## Step 0: Verification of load-bearing facts (already done; encode as inline comments)

Verified by reading source before drafting this plan. Each fact MUST be encoded as a one-line comment adjacent to the corresponding new code site so a future reader doesn't have to re-derive it:

- **`_pd._record('4_compress', ...)` semantics** — `profile_decode_upstream_flash_infer.py:269-277`. `_rec(3)` records at L269 right before `_update_comp_cache(...)`; `_rec(4)` at L277 right after. The event pair is appended to `_pd._pending_events` for EVERY decode step (L391 `if _pd._enabled`). On fast-path (`n_new == 0`, `dct_page_attention.py:558-569`), the call returns immediately with no GPU work, so the bucket records ~0 ms. On seal steps, the kernel runs. Therefore `_pd._step_timings['4_compress']` is **already an average over all steps** = `t_seal_per_layer × (1 / page_size)` (in expectation). The existing printer at L853-854 multiplying `eager_per_token['4_compress'] * num_layers` is **per-step total in expectation** — the Architect's "almost correct" claim is correct.
- **Why we still need a microbench**: graph capture targets `last_page_len_py <= ps - 4` (L1215), so the captured replay sequence is 4 non-sealing steps in a row. Bucket `4_compress` in the graph row is structurally near-zero (no `_update_comp_cache_kernel` ever fires inside the captured iteration). Without the microbench, the reconciliation table compares "graph ≈0" against "eager average ≈ t_seal/ps" and an unaware reader concludes the graph eliminated compression cost — false. The microbench produces a third row "forced-seal-amortized eager truth = t_seal/ps measured directly with high precision".
- **Slow-path allocation hazard**: `dct_page_attention.py:582-605` reallocates `_comp_k_cache` etc. on first fire or shape change. The microbench must run a warmup forced-seal call (timing discarded) before measurement. Eager mode means it uses the standard caching allocator — no graph stream-private pool hazard.
- **FI `wrapper.plan()` is NOT graph-capturable**: confirmed at L1199-1210. We MUST NOT advance past a page boundary during alignment; original `align_target=compress` design (targeting `ps-1`) is dead.
- **Pattern dict** (`_SUBSTEP_NAME_PATTERNS`, L942-958): `_update_comp_cache_kernel` → `4_compress` is the only mapping; `index_kernel` → `2_rope_and_cache_append`; cublasLt gemms either go to `gemm_total` (merged) or split into `1_qkv_proj`/`8_o_proj` (ordering mode). Confirms the existing v1-plan claim that "slice + cublasLt + memcpy" goes under `4_compress" was wrong.

### RALPLAN-DR Summary

**Principles:**
1. Byte-identity for default invocation — no flag, no behavior change. Verifiable by `diff` of stdout/stderr against pre-change run.
2. Truth via direct measurement — produce a number that IS `t_seal_per_layer / page_size × num_layers`, not a derivation from suspect averages.
3. No graph-side surgery — `wrapper.plan()` is non-capturable, slow-path allocations are non-capturable, and eager forced-seal saves microseconds vs graph forced-seal anyway. Stay eager.
4. Same `attn_module`, same projection cache, same KV layout — the microbench number is directly comparable to a real forward pass.
5. Fail loudly on inapplicable modes — baseline mode has no compression; flag must error early.

**Decision Drivers (top 3):**
1. **Correctness over capture**: capturing a sealing step would either crash on `wrapper.plan()` (page-crossing) or silently use the cudagraph stream-private allocator pool for `_comp_k_cache` realloc (silent corruption risk). Eager forced-seal sidesteps both.
2. **Minimum surface area**: edits ONLY in `speed/profile_decode_upstream_flash_infer.py`. No bash wrapper, no JSON sidecar, no `dct_page_attention.py` mods.
3. **Diagnostic value per LOC**: ~80-100 LOC of new code produces a single load-bearing number (`t_seal_amortized_per_step_ms`) that fixes the reconciliation table's worst defect (`4_compress` ≈0 in graph, ≈0 in eager-avg, both misleading without a third row).

**Viable Options (>=2):**

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A. Same-process eager forced-seal microbench (CHOSEN)** | New flag `--cudagraph_breakdown_seal_microbench`. After main capture, run K (~100) forced-seal `_update_comp_cache` calls in eager mode using the live `attn_module`. Print 3-row reconciliation: graph (~0), eager-avg×layers (existing), forced-seal/ps×layers (new truth). | Hits all gates. ~80-100 LOC. No graph surgery. Same allocator/cache as forward. Failure mode is local. | The number is per-LAYER × num_layers, not a true cross-layer measurement; if seal-cost varies by layer this hides it. Mitigation: optional all-layers loop in follow-up. |
| **B. Capture-time event-record around `_update_comp_cache`** | Wrap the call site in graph-recordable events that fire inside the captured replay AND on a forced-seal warm-up replay. | Direct measurement of in-graph cost. | Box-killer per `feedback_event_in_graph_unsupported.md`: cudaEventRecord-in-graph silently dropped on torch 2.10/A6000. Output is empty. Dead until torch/CUDA bump. |
| **C. Second graph capture targeting `last_page_len_py == ps-1`** | Capture a second graph aligned so the captured iteration IS a sealing step; bucket via existing kernel-name → bucket dict. | Reuses existing CUPTI bucketing infra. | `wrapper.plan()` would fire on the page boundary inside the captured replay → graph capture fails OR produces a graph that crashes on replay. Slow-path `_comp_k_cache` realloc inside capture uses stream-private allocator pool (data-corruption risk per Architect finding 4). REJECTED. |

(Only Option A and B are technically viable; C is invalidated by the FI `plan()` and allocator hazards. B is invalidated by the box-specific torch/CUDA bug. A is the chosen option by elimination AND on its merits.)

### Detailed Steps

All edits in `/home/yoongonkim/DCT-Page/speed/profile_decode_upstream_flash_infer.py`. Each step lists its anchor line range and a `grep` query to re-verify before the edit (line numbers may drift between iterations).

#### Step 1: Add CLI flags (~10 LOC)

**Anchor**: argparse block. Verify with `grep -n "cudagraph_breakdown" speed/profile_decode_upstream_flash_infer.py | head` to find the existing `--cudagraph_breakdown` family. Insert immediately after the existing flags.

**Add three flags:**
- `--cudagraph_breakdown_seal_microbench` (bool, `store_true`, default False)
- `--cudagraph_breakdown_seal_microbench_iters` (int, default 100) — number of timed forced-seal calls
- `--cudagraph_breakdown_seal_microbench_warmup` (int, default 5) — discarded warmup calls (covers slow-path realloc + JIT)

**Acceptance**: `python speed/profile_decode_upstream_flash_infer.py --help | grep seal_microbench` prints all three flags. Default invocation (no flag set) leaves the existing arg surface unchanged.

#### Step 2: Implement `_run_seal_microbench(...)` helper (~50-60 LOC)

**Anchor**: new helper, place just above `_print_graph_breakdown` (currently at L726). Verify with `grep -n "def _print_graph_breakdown" speed/profile_decode_upstream_flash_infer.py`.

**Logic** (eager-only; runs AFTER graph capture+replay completes, BEFORE the printer call):

1. **Mode guard**: assert DCT patch is active (a layer-0 attn module with `_comp_n_pages_cached` exists). The driver's argparse at `profile_decode_upstream_flash_infer.py:572` restricts `--unselected_mode` to `'drop'` (`choices=["drop"]`), so we do not branch on `compressed` here — the `compressed` codepath is unreachable in this driver. On no-DCT (baseline) mode, the wiring at Step 4 detects the empty `attn_modules` list and prints `[ERROR] seal microbench: no DCT attention modules found; mode is likely baseline`; the helper itself returns `None` early.
2. **Pick representative `attn_module`**: take `attn_modules[0]` (first decoder layer's attention module). Document in comment that this is per-LAYER and assumes layer parity; expose all-layers in follow-up.
3. **Reuse live tensors**: re-derive the same arguments the forward synthesizes at L258-274:
   - `comp_size = max(1, int(cfg.page_size * cfg.compress_ratio))`
   - `num_pages = fi_cache.last_page_idx_py - fi_cache.num_sink_pages - fi_cache.num_recent_pages_fixed + 1`
   - `paged_k`, `paged_v` from `cache.buf_views[0][:, :, sink:sink+num_pages, 0/1, :, 0, :]`
   - `from dct_page_attention import _dct_page_cfg as cfg; assert cfg is not None, "DCT patch not active"` (DCTPageConfig is set as a module-level `_dct_page_cfg` per CLAUDE.md "Monkey-patch pattern"; import directly).
4. **Force seal mechanic**: stash `original_n_cached = attn_module._comp_n_pages_cached`. Each timed iteration:
   - Set `attn_module._comp_n_pages_cached = num_pages - 1` (so `n_new = 1`).
   - Call `_update_comp_cache(attn_module, paged_k, paged_v, num_pages, comp_size, cfg)`.
   - Slow-path auto-writes `_comp_n_pages_cached = num_pages` per iteration (verified at `dct_page_attention.py:722`); the manual `_comp_n_pages_cached = num_pages - 1` setter inside the loop is what forces `n_new=1` each iter. **Defensive teardown** after the microbench loop:
     ```python
     attn_module._comp_n_pages_cached = original_n_cached
     attn_module._last_comp_kv = None  # invalidate fast-path cache so next live forward redoes one slow path
     ```
     Reason: `_update_comp_cache` slow-path writes `attn_module._last_comp_kv = result` at `dct_page_attention.py:740/753`, and the live forward fast-path at L563 returns this without recomputing. Without invalidation, the next real decode step receives microbench-mutated cache identity.
5. **Warmup**: `args.cudagraph_breakdown_seal_microbench_warmup` (default 5) forced-seal calls — discard timing. Absorbs slow-path realloc on first fire if needed plus JIT.
6. **Measure**: chained `torch.cuda.Event(enable_timing=True)` pair around `iters` calls in a tight loop, single `torch.cuda.synchronize()` after the loop, `start.elapsed_time(end) / iters` → `t_seal_per_layer_ms` (ms for a single layer's single call).
7. **Return** dict:
   ```
   {"t_seal_per_layer_ms": <float>,
    "iters": <int>,
    "page_size": <int>,
    "amortized_per_step_per_layer_ms": t_seal / page_size,
    "amortized_per_step_total_ms": (t_seal / page_size) * num_layers}
   ```

**Acceptance:**
- Returns dict with `t_seal_per_layer_ms > 0` when `unselected_mode in {drop, compressed}`.
- Returns `None` and prints clear `[ERROR]` line for baseline mode.
- Microbench does NOT durably mutate `_comp_n_pages_cached` (assert pre/post equal in a debug print on first run).

#### Step 3: Plumb microbench output into `_print_graph_breakdown` (~25 LOC)

**Anchor**: modify the helper at L726-733 (signature) and the eager-table block ending around L835.

**Changes:**

1. Add new keyword-only argument `seal_microbench=None` to `_print_graph_breakdown`'s signature (default None preserves all existing call signatures → byte-identity safe).
2. Inside the existing `if eager_per_token:` block (currently L799), AFTER the per-step ratio table, add a new section gated on `seal_microbench is not None`:

   ```
     Compression reconciliation (4_compress):
       graph (captured, fast-path):                  X.XXX ms/step    # ~0 by design (4 non-sealing steps captured)
       eager average over all steps × layers:        X.XXX ms/step    # = t_seal/ps × layers (in expectation)
       forced-seal microbench / page_size × layers:  X.XXX ms/step    # truth (direct measurement, n=<iters>)
       note: page_size=<ps>, microbench_iters=<iters>, num_layers=<layers>
   ```

   Numbers come from:
   - graph: `substep_per_token.get('4_compress', 0.0)`
   - eager-avg×layers: `eager_per_token.get('4_compress', 0.0) * num_layers`
   - forced-truth: `seal_microbench['amortized_per_step_total_ms']`

3. **Plausibility advisory** (Acceptance gate 3): derive the window at runtime from the existing eager average:
   ```python
   eager_4c_total_per_step = eager_per_token.get('4_compress', 0.0) * num_layers
   if eager_4c_total_per_step > 0:
       lo, hi = eager_4c_total_per_step / 2.0, eager_4c_total_per_step * 2.0
   else:
       lo, hi = 0.05, 5.0  # fallback only when eager average is unavailable
   if not (lo <= forced_truth_per_step <= hi):
       print(f"[INFO] forced-seal/ps ({forced_truth_per_step:.3f} ms/step) outside ±2× eager avg [{lo:.3f}, {hi:.3f}]; consider verifying with nsys.")
   ```
   The run does NOT fail.

**Acceptance:**
- Default invocation (no flag): printer signature accepts `seal_microbench=None`, NO new lines printed → byte-identity preserved.
- With flag: 4-5 new lines added under the existing eager-vs-graph table.

#### Step 4: Wire microbench into the cudagraph branch (~15 LOC)

**Anchor**: cudagraph block, immediately before the `_print_graph_breakdown(...)` call. Verify with `grep -n "_print_graph_breakdown(" speed/profile_decode_upstream_flash_infer.py`.

**Changes:**

1. Just before the printer call, conditional block:
   ```python
   seal_mb = None
   if args.cudagraph_breakdown_seal_microbench:
       attn_modules = [m for m in model.modules()
                       if hasattr(m, '_comp_n_pages_cached')
                       and getattr(m, 'layer_idx', -1) == 0]
       if not attn_modules:
           print("[ERROR] seal microbench: no DCT attention modules found; mode is likely baseline")
       else:
           seal_mb = _run_seal_microbench(model, fi_cache, attn_modules, args, num_layers)
   ```
   Why pin `layer_idx == 0`: `model.modules()` registration order is not spec-guaranteed; pinning to layer 0 makes the choice deterministic across torch versions and refactors.
2. Pass `seal_microbench=seal_mb` to `_print_graph_breakdown(...)`.
3. **Critical: byte-identity guard** — if `args.cudagraph_breakdown_seal_microbench` is False, `seal_mb` stays `None`, `_run_seal_microbench` is never called (no stdout from it), and printer's new section skips cleanly. With flag off, full stdout MUST be byte-identical to pre-edit.

**Acceptance:** with flag off, `diff` of full stdout vs pre-change baseline is empty.

#### Step 5: Smoke test + acceptance verification

Run on Llama-3.1-8B at 32K context. Use `dct_upstream_fi` as the parent mode (single-driver scope per CLAUDE.md note about this PR being upstream-FI only).

```bash
# Gate 1: byte-identity baseline (no new flag)
python speed/profile_decode_upstream_flash_infer.py --cudagraph --cudagraph_breakdown \
  --context_length 32768 --num_decode_steps 64 --batch_size 1 \
  2>&1 | tee /tmp/run_post.log
# Compare to identical pre-edit run /tmp/run_pre.log → diff must be empty.

# Gate 2 + 3: microbench produces non-zero plausible number (drop mode)
python speed/profile_decode_upstream_flash_infer.py --cudagraph --cudagraph_breakdown \
  --cudagraph_breakdown_seal_microbench \
  --context_length 32768 --num_decode_steps 64 --batch_size 1 \
  --unselected_mode drop 2>&1 | tee /tmp/run_seal.log
# Expect line: "forced-seal microbench / page_size × layers: <X> ms/step" with X > 0 and ideally in [0.1, 1.0].

# Gate 4: 32K Llama smoke — no OOM, capture succeeds, throughput unchanged with flag off
nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 500 > /tmp/mem.log &
# (run gate-2 command, then kill nvidia-smi tail; max(used) should be the same as pre-edit ±<200 MiB)

# Gate 5: error path on baseline mode (no DCT patch active in this driver, so we
# need to verify the fail-fast path even if the upstream-FI driver is DCT-only by design.
# Check via inspecting the no-DCT-modules early-print path or by temporarily forcing
# `attn_modules = []` in a debug invocation.)
```

**Acceptance gates (final mapping):**

| Gate | Verification |
|---|---|
| 1. Default invocation byte-identical | `diff /tmp/run_pre.log /tmp/run_post.log` is empty. |
| 2. Flag produces non-zero forced-seal/ps number | `grep "forced-seal microbench" /tmp/run_seal.log` returns a line with a numeric value > 0. |
| 3. Number is plausible (within ±2× eager avg) | `forced_truth ∈ [eager_avg/2, eager_avg×2]` where `eager_avg = eager_per_token['4_compress'] × num_layers`. (Outside this range triggers `[INFO]` advisory but does not fail. Fallback `[0.05, 5.0]` ms/step only when eager average is unavailable.) |
| 4. 32K Llama smoke: no OOM, no graph regression | Process exits 0; `Per-replay (graph)` line within ±5% of pre-edit. |
| 5. Baseline / no-DCT mode error path | `[ERROR]` line printed; rest of breakdown completes. |

### Test Plan (per gate)

| Gate | Test | Expected | Risk if fails |
|---|---|---|---|
| 1 | `diff` of stdout/stderr default invocation, pre vs post edit | empty diff | Default users see new noise → reject. |
| 2 | Run with flag on `--unselected_mode drop` | line "forced-seal microbench / page_size × layers: X ms/step" with X > 0 | Microbench plumbing broken or wrong attribute name on `attn_module`. |
| 3 | Inspect X from gate-2 vs eager avg | X ∈ [eager_avg/2, eager_avg×2] (fallback [0.05, 5.0] when eager avg unavailable) | Either a real perf surprise (worth investigating) or a counting bug; advisory line tells the user. |
| 4 | Same as gate-2 plus `nvidia-smi` tail | no OOM; `Per-replay (graph)` line within ±5% of pre-edit | Microbench leaked state into the cache or the printer change disturbed graph timing. |
| 5 | Run baseline mode invocation OR force `attn_modules = []` | clear `[ERROR]` line; rest of run completes | Crash on baseline mode → poor UX. |

### ADR

**Decision**: Implement same-process eager forced-seal microbench for `4_compress` reconciliation, gated on a new flag `--cudagraph_breakdown_seal_microbench`. All edits stay in `speed/profile_decode_upstream_flash_infer.py` (~80-100 LOC across 4 surgical sites: argparse, helper, printer, wiring).

**Drivers** (priority order):
1. Correctness: graph-side capture would crash on `wrapper.plan()` page-boundary or use stream-private allocator pool for `_comp_k_cache` realloc.
2. Byte-identity for default invocation (Gate #1) — non-negotiable.
3. Diagnostic value: produces the truth row that resolves the reconciliation defect.

**Alternatives considered:**
- Option B (capture-time event-record): rejected — `feedback_event_in_graph_unsupported.md` confirms torch 2.10/A6000 silently drops it.
- Option C (second capture targeting `ps-1`): rejected — `wrapper.plan()` non-capturable AND slow-path realloc allocator-pool hazard.
- Original v1 plan with two-process design + JSON sidecar: rejected — scope creep; same-process microbench needs no IPC.
- Original v1 `align_target=compress` mechanic: rejected — would advance past page boundary, triggering uncapturable `plan()`.

**Why chosen:** Option A is the only technically viable path that produces a directly measurable truth number, costs <100 LOC, and preserves byte-identity. The Architect's "graph mode would save ≈0.1 µs/step amortized" estimate confirms the eager forced-seal answer is the right answer; capturing the seal step would buy nothing but cost stability.

**Consequences:**
- New per-LAYER × num_layers approximation; if seal cost varies across layers this is hidden. Acceptable given Llama-3.1 decoder layers share shape and projection-matrix cache. Mitigation deferred to follow-up.
- One new flag plus two tuning knobs (iters, warmup). Maintenance cost is minimal — purely additive surface.
- Reconciliation block grows by 4-5 lines when flag is on. Default output unchanged.
- Microbench transiently mutates `_comp_n_pages_cached`; defensive restore in place. If a future caller adds work after the microbench within the same process, state must remain clean.

**Follow-ups:**
1. (Optional) `--cudagraph_breakdown_seal_microbench_all_layers` to loop over all `attn_modules`, report min/median/max `t_seal_per_layer_ms` for layer-skew diagnosis. Cheap (~10 LOC); deferred to keep this PR small.
2. (Optional) Write microbench result to existing `state.json` artifact for downstream comparison tooling. Deferred — JSON sidecar was a v1 scope-creep concern.
3. (Optional) Verify on Qwen3-8B once Llama smoke passes. `_update_comp_cache` is model-agnostic; pattern-identical run expected.
4. Document the 3-row reconciliation pattern in `CLAUDE.md` "Speed/profiling notes" once landed and verified.

### Open Questions

(Append to `.omc/plans/open-questions.md` after this plan ships.)

- [ ] Should the microbench loop over all decoder layers by default, or stay on layer 0? — Layer-skew is unmeasured; current default may understate variance. Decision deferred to follow-up #1.
- [ ] Should the plausibility window be tunable via flag or stay derived from the eager average (with `[0.05, 5.0]` ms/step fallback)? — Currently runtime-derived; revisit if very small models or unusual `page_size`/`compress_ratio` combos push outside.
- Note: slow-path auto-writes `_comp_n_pages_cached = num_pages` per iteration (verified at `dct_page_attention.py:722`); the manual `_comp_n_pages_cached = num_pages - 1` setter inside the loop is what forces `n_new=1` each iter. The auto-restore in slow-path means manual reset after each iter is NOT required, but the defensive teardown reset (Step 2.4) is still required to restore the entry-snapshot value (warmup may leave state at `num_pages` while we want the original entry value).
