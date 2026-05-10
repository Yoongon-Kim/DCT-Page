## Handoff: team-plan → team-exec (eval-upstream-fi team)

- **Decided**: Decompose plan `/home/yoongonkim/DCT-Page/.omc/plans/eval-upstream-flashinfer-backend.md` into 8 tasks. 3 workers:
  - **worker-1** (foundation): #1 dct_page_attention.py (new forward + dispatch + 4 helpers + per-instance attr + maintainer comment), #8 CLAUDE.md doc fix.
  - **worker-2** (parallelizable): #2 Quest-minmax mirror in profile_decode_upstream_flash_infer.py, #4 eval_longbench_v1.py, #5 eval_longbench_v2.py.
  - **worker-3** (sequential pipeline): #3 eval_ruler.py, #6 eval_aime25.py, #7 eval_gpqa.py (#6/#7 reuse eval_ruler helpers, hence sequential).
- **Dependency graph**: #1 blocks #3, #4, #5, #6, #7. #3 additionally blocks #6, #7. #2 and #8 are independent.
- **Rejected**:
  - One-task-per-worker (overhead with no parallel speedup; #1 is critical path).
  - Merging #1 + #3 into one worker (would block #4 and #5 unnecessarily).
  - New shell sweep scripts (user explicitly removed from scope; §7 of plan).
- **Risks**:
  - **#1 is large** (new forward ~250 LOC + 4 helpers + dispatch + per-instance attr + maintainer comment). If it lands buggy, #3-#7 verify will all fail. Worker-1 must verify imports + 3-way dispatch immediately after edit.
  - **Compressed-mode preflight is duplicated** across #3, #4, #5; drift risk. Consider extracting to a shared helper in dct_page_attention.py if the planner had specced it (didn't); workers replicate verbatim instead.
  - **`_init_upstream_fi_build_kwargs` post-load pattern is new** — must pick up Llama AND Qwen3 attention modules (different classes; use `hasattr(q_proj, k_proj)` not isinstance check).
  - **Lazy-init layer-0 race** (architect-flagged; out-of-scope refinement) — assumes vanilla HF generate fires layer 0 first. True for transformers 5.x sequential decode; not asserted.
  - **Profile driver Quest closure (#2) introduces a lazy import from dct_page_attention** — verify it doesn't break the cold path when score_use_quest_minmax=False.
- **Load-bearing details (DO NOT deviate)**:
  1. NEW forward MUST NOT call `pre_allocate_cache` — prefill already did at line 1216-1220; second call would null-deref on `_fi_mode=True` layers (line 68). Read `prefill_len = past_key_values.layers[0]._seen` directly.
  2. `_upstream_fi_max_decode_steps` is a PER-INSTANCE attribute (`module._upstream_fi_build_kwargs["max_decode_steps"]`), NOT a module global.
  3. `reset_upstream_fi_cache_state(model)` clears `_dct_runtime_cache_ref` and `_verify_diffs` EXPLICITLY (do NOT add either to `_DCT_RUNTIME_STATE_ATTRS` — that defeats the guard at lines 1119-1121).
  4. `_generate_with_upstream_fi` does defensive double-clear at entry, sets max_decode_steps, then `try: model.generate(...) finally: reset + empty_cache()`.
  5. Compressed mode is a HARD ERROR at argparse (not silent SDPA fallback). Hard 64K preflight (NOT soft warning). Greedy-only assert.
  6. SDPA path bit-identical when `--attention_backend` defaults (no behavior change).
  7. Quest-minmax simultaneous closure: BOTH new eval forward AND profile driver line 280-297 get the branch. Lazy import in profile driver to keep cold path clean.
- **Files modified**:
  - `dct_page_attention.py` (#1)
  - `speed/profile_decode_upstream_flash_infer.py` (#2 only — Quest mirror)
  - `eval_ruler.py` (#3)
  - `eval_longbench_v1.py` (#4)
  - `eval_longbench_v2.py` (#5)
  - `eval_aime25.py` (#6)
  - `eval_gpqa.py` (#7)
  - `CLAUDE.md` (#8 — doc fix)
- **Files DO NOT TOUCH**: `speed/upstream_flashinfer_backend.py`, `speed/flashinfer_backend.py`, `triton_kernels.py`, `config.py`, `baselines/**`, all `run_*.sh`.
- **GPU pinning**: `CUDA_VISIBLE_DEVICES=2` for any eval/profile invocations during verify (project convention; 4× A6000 box).
- **Conda env**: Default DCT_Page (transformers 5.2.0, torch 2.10.0, triton 3.6.0, flashinfer present).
- **Remaining**:
  - **team-exec**: workers run #1 + #2 + #8 in parallel; #3, #4, #5 unlock when #1 done; #6, #7 unlock when #1+#3 done.
  - **team-verify**: import check + AST parse on all 7 edited files; help-text grep for new flags; 3-way dispatch test; lazy-import smoke test on upstream FI helpers.
  - **team-fix**: only on real defect.
