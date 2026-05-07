## Handoff: team-plan → team-exec (dct-seal-microbench team)

- **Decided**: RALPLAN-DR consensus iteration 3 APPROVE. Single-file edit per `/home/yoongonkim/DCT-Page/.omc/plans/cudagraph-substep-breakdown.md` (Plan v3 — Same-Process Eager Forced-Seal Microbench). 4 surgical sites in `speed/profile_decode_upstream_flash_infer.py`: argparse (~10 LOC), `_run_seal_microbench` helper (~50-60 LOC), `_print_graph_breakdown` 3-row reconciliation block (~25 LOC), cudagraph-branch wiring (~15 LOC). Total ~80-100 LOC.
- **Rejected** during planning:
  - v1 two-process / `--cudagraph_breakdown_align_target=compress` design — `wrapper.plan()` is not graph-capturable; capture would crash on page-boundary or use stream-private allocator pool for `_comp_k_cache` realloc.
  - cudaEventRecord-in-graph (Option B) — silently dropped on torch 2.10/A6000 per `feedback_event_in_graph_unsupported.md`.
  - JSON sidecar + bash wrapper — v1 scope creep; same-process microbench needs no IPC.
- **Risks**:
  - Forced-seal does NOT exercise the realloc branch (`_next_page_capacity` doubles only on first warmup) — measures steady-state seal cost, NOT amortized realloc cost. Acceptable; production realloc is sub-µs over 1000+ steps. Document in code comment.
  - Microbench transiently mutates `_comp_n_pages_cached`. Defensive teardown MUST clear BOTH `_comp_n_pages_cached = original_n_cached` AND `_last_comp_kv = None` (the latter prevents poisoning the next live forward via fast-path at `dct_page_attention.py:563`).
  - Per-LAYER × num_layers extrapolation hides layer-skew if seal-cost varies across layers. Llama-3.1 decoder layers share shape and projection-matrix cache; acceptable for v1. Flag follow-up #1 in plan.
  - Plausibility window MUST be runtime-derived from `eager_per_token['4_compress'] × num_layers ± 2×`. Hardcoded fallback `[0.05, 5.0]` ms/step only when eager average is unavailable.
- **5 Load-bearing details** (Architect+Critic verified; do NOT deviate):
  1. `from dct_page_attention import _dct_page_cfg as cfg; assert cfg is not None, "DCT patch not active"` — `_dct_page_cfg` is a module-level global, NOT an `attn_module` attribute.
  2. `attn_modules = [m for m in model.modules() if hasattr(m, '_comp_n_pages_cached') and getattr(m, 'layer_idx', -1) == 0]` — pinned to layer 0 for determinism.
  3. Defensive teardown clears BOTH `_comp_n_pages_cached` and `_last_comp_kv`.
  4. Plausibility window derived at runtime.
  5. Mode guard simplified to "DCT patch active" only — `unselected_mode == "compressed"` is unreachable in this driver (argparse `choices=["drop"]` at L572).
- **Files**:
  - **MODIFY**: `/home/yoongonkim/DCT-Page/speed/profile_decode_upstream_flash_infer.py` (single file; ~80-100 LOC additive).
  - **READ-ONLY donor refs**: `/home/yoongonkim/DCT-Page/dct_page_attention.py:128, 522-754, 740, 753, 563` (`_update_comp_cache` body + fast-path read site).
  - **OUT OF SCOPE — DO NOT TOUCH**: `dct_page_attention.py`, sibling drivers, baselines/, oracle/. NO bash wrappers, NO JSON sidecars.
- **GPU pinning convention**: `CUDA_VISIBLE_DEVICES=2` for all eval/profile invocations (project convention; check `nvidia-smi` first; 4× A6000 box).
- **Conda env**: Default DCT_Page (transformers 5.2.0, torch 2.10.0, triton 3.6.0).
- **Acceptance gates** (5 total):
  1. Default invocation byte-identical to pre-edit (`diff` empty).
  2. `--cudagraph_breakdown_seal_microbench` produces non-zero `forced-seal microbench / page_size × layers` line.
  3. Number is plausible: within ±2× of eager-avg × num_layers (runtime-derived window).
  4. 32K Llama-3.1-8B smoke: no OOM, `Per-replay (graph)` within ±5% of pre-edit baseline.
  5. No-DCT path: helper exits with `[ERROR]` line and returns None.
- **Pipeline**: planner=DONE → executor (worker-1) → verifier (worker-2). Spawn debugger only on real defects.
- **Remaining**:
  - **team-exec**: implement plan steps 1-4 in `speed/profile_decode_upstream_flash_infer.py`. Single executor (worker-1).
  - **team-verify**: 5 acceptance gates (worker-2). Sequential GPU usage on device 2.
  - **team-fix**: only on real defect.
