# Add InfLLM support to `eval_aime25.py`

**Mode:** RALPLAN-DR consensus, SHORT (medium-risk; brief pre-mortem included).
**Owner:** executor.
**Target file:** `/home/yoongonkim/DCT-Page/eval_aime25.py` (single file edit; no other source files modified).

---

## RALPLAN-DR Summary

### Principles (5)

1. **Reuse the donor wiring; don't fork it.** `apply_monkey_patch` and `load_model_and_tokenizer` in `eval_ruler.py` already implement the InfLLM init path correctly (eager attn, `torch_dtype`, `load_llama_config_stripped_rope`, `init_inf_llm`, `build_inf_llm_generator`). The aime25 change must pull through it, not redo it.
2. **Keep Qwen3 the default; let Llama in only for `inf_llm`.** Existing reasoning evals (baseline / page_attention / seer / multipole on Qwen3-8B) must remain regression-free.
3. **One generator dispatch per mode.** Mirror the v2 fork shape (`eval_longbench_v2.py:277-283`) — InfLLM uses `args._inf_llm_generator.generate(...)` and clears state per-sample.
4. **Honest acceptance criteria.** Llama-3.1-8B-Instruct is not reasoning-tuned; the bar is "runs end-to-end with InfLLM exercised", not "matches Qwen3 accuracy".
5. **Fail fast and clearly.** A user who passes `--mode inf_llm --base_model Qwen/Qwen3-8B` must see a Llama-only error from aime25 *before* model load, not from deep inside `assert_llama_only` after eager-attn config building.

### Decision Drivers (top 3)

1. **Minimum surface area.** This is one eval script change; the InfLLM shim, generator, and donor loader are already correct.
2. **Compatibility with existing Qwen3 guards.** Aime25's `_assert_qwen3_compatible` is wired into argparse's epilogue (`eval_aime25.py:246`); we have to relax it without making it permissive for everything.
3. **Mirroring established convention.** `eval_ruler.py` and `eval_longbench_v2.py` already define the canonical InfLLM hook points (CLI args, run-name, summary block, generator dispatch, `clear()` reset). Drift = future bugs.

### Viable Options

**Option A (chosen) — Per-mode model-family guard.**
Add `LLAMA_ONLY_MODES = {"inf_llm"}`. Replace `_assert_qwen3_compatible(mode)` (a mode-only guard) with `_assert_mode_model_compatible(mode, base_model)` that routes by mode → required family.

- Pros: Generalizes — when (e.g.) DuoAttention support lands later, drop it into the same set. Centralizes the cross-axis (mode × model) check. Catches the bad combo before model load.
- Cons: Slight refactor of one helper (still ~12 lines). Renames the call site at `eval_aime25.py:246`.

**Option B — Special-case skip when `mode==inf_llm` and base_model is Llama-3.x.**
Leave `_assert_qwen3_compatible` alone; just early-return when those two conditions hold.

- Pros: Tiniest possible diff (3 lines).
- Cons: Doesn't generalize; the next baseline that goes Llama-only (DuoAttention, ShadowKV) needs another bespoke escape hatch. Harder to reason about — the function name lies. Encourages copy-paste.

**Option C — Drop the guard entirely; rely on `assert_llama_only` deep in `load_model_and_tokenizer`.**

- Pros: Even smaller diff; one less helper.
- Cons: Error surfaces *after* eager attn impl resolution and config-rope-stripping run, which is confusing when the real problem is wrong model + wrong mode. Worse UX. Also leaves the `--base_model choices=['Qwen/Qwen3-8B']` argparse hard-block in place, so this option can't actually let Llama in.

**Decision: Option A.** It is the only choice that (a) lets Llama-3.x in for `inf_llm`, (b) keeps the existing guard semantics for all other modes, and (c) generalizes for the next Llama-only baseline without further surgery.

### Pre-mortem (3 risks)

1. **Risk: argparse `--base_model choices=['Qwen/Qwen3-8B']` blocks Llama before our guard runs.**
   *Mitigation:* Drop `choices=` on `--base_model` entirely (mirrors `eval_ruler.py:62` and `eval_longbench_v2.py:62`, which both accept `--base_model` as a free string and validate downstream). Validation flows through the new `_assert_mode_model_compatible`, which carries the typo-catching role on the (mode × family) cross-axis.

2. **Risk: tokenizer chat-template branch mishandles Llama.**
   `tokenize_prompt` (eval_aime25.py:90-106) toggles `enable_thinking=True` only if the literal string appears in the chat template. Llama-3.1-Instruct's chat template is *expected* not to contain `enable_thinking` (unverified on this box; tokenizer not cached locally), and the branch's `(tokenizer.chat_template or '')` short-circuit is defensive against missing templates.
   *Mitigation:* Add a one-line family guard at the `enable_thinking` site: `if "qwen3" in args.base_model.lower() and "enable_thinking" in (tokenizer.chat_template or ""): chat_kwargs["enable_thinking"] = True`. Cheap, eliminates risk regardless of upstream template revisions, and codified as **Step 3.0** below.

3. **Risk: AIME's `max_new_tokens=16384` × InfLLM eviction-heavy decode is not exercised by ruler — A6000 OOM cliff.**
   Per `reference_gpu_a6000.md`, this box is 4×RTX A6000 (48 GiB), not H100. AIME's `max_new_tokens=16384` × InfLLM eviction-heavy decode is not exercised by ruler.
   *Mitigation:* Smoke test must run at least one full-default invocation (`--max_new_tokens 16384 --num_samples 1`) and report `torch.cuda.max_memory_allocated()` to stdout. If peak > 44 GiB, the executor lowers default to 8192 in a follow-up. Code-side knob: **Step 3.3** prints `torch.cuda.max_memory_allocated()/1e9` once at end of `evaluate(...)` (a single line, easy to grep in stdout).

---

## Detailed Plan

All edits are in **`/home/yoongonkim/DCT-Page/eval_aime25.py`** only. No new files.

### Step 1 — Module docstring + compatibility tables (lines 1-67)

**Change 1.1.** Update docstring (lines 1-27) to note InfLLM is the one Llama-only mode in this script. Replace the line "Modes that do not support Qwen3 (seer_prefill, quest_attention, duo_attention, shadowkv) are still listed in `--mode` for parity, but the script raises ValueError before model load if one is selected." with a description that also calls out the new `inf_llm` Llama-only path.

**Change 1.2.** Replace the `QWEN3_SUPPORTED_MODES` / `QWEN3_UNSUPPORTED_MODES` block (lines 52-67) with three sets and one cross-axis guard. The helper is intentionally slim (~12 lines):

```python
# Modes that run on Qwen3-8B (current default reasoning model).
QWEN3_SUPPORTED_MODES = {
    "baseline", "page_attention", "seer_attention", "multipole_attention",
}
# Modes that are Llama-only in this harness (e.g. InfLLM's transformers==4.37
# shim and Llama-only RoPE replacement). When --mode is in this set, --base_model
# must be a Llama-family ID.
LLAMA_ONLY_MODES = {"inf_llm"}
# Modes intentionally listed in --mode for argparse parity but unsupported here.
UNSUPPORTED_MODES = {
    "seer_prefill", "quest_attention", "duo_attention", "shadowkv",
}


def _assert_mode_model_compatible(mode: str, base_model: str) -> None:
    # Intentionally duplicates the family check in baselines/infllm/__init__.py:33-46
    # (`assert_llama_only`) — we want early failure *before* model load and config
    # rope stripping. The (mode × family) cross-axis lives only here.
    if mode in UNSUPPORTED_MODES:
        raise ValueError(f"--mode {mode!r} is not supported by eval_aime25.")
    name = base_model.lower()
    if mode in LLAMA_ONLY_MODES and "llama" not in name:
        raise ValueError(f"--mode {mode!r} is Llama-only; got {base_model!r}.")
    if mode in QWEN3_SUPPORTED_MODES and "qwen3" not in name:
        raise ValueError(f"--mode {mode!r} requires Qwen3; got {base_model!r}.")
    if mode not in (UNSUPPORTED_MODES | LLAMA_ONLY_MODES | QWEN3_SUPPORTED_MODES):
        raise ValueError(f"Unknown --mode {mode!r}.")
```

**Why:** drops the mode-only guard for a (mode, base_model) guard. Generalizes for any future Llama-only baseline without another special case.

Note on duplication: this still duplicates `assert_llama_only`'s job for the Llama path, but the early-failure UX (catch before `from_pretrained` and config-rope-stripping) is the justification, and the helper centralizes the (mode × family) cross-axis in one place.

### Step 2 — CLI: add `inf_llm` mode + free-string base_model + InfLLM args (lines 177-244)

**Change 2.1.** Extend `--mode` choices (lines 181-186) by appending `"inf_llm"`:

```python
parser.add_argument("--mode", type=str, required=True,
                    choices=["baseline", "page_attention", "seer_attention",
                             "seer_prefill",
                             "multipole_attention", "quest_attention",
                             "duo_attention",
                             "shadowkv",
                             "inf_llm"])
```

**Change 2.2.** Drop `choices=` on `--base_model` (lines 189-191) and accept any HF ID:

```python
parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
```

Rationale: `eval_ruler.py:62` and `eval_longbench_v2.py:62` both accept `--base_model` as a free string and validate downstream. The new `_assert_mode_model_compatible` helper carries the typo-catching role on the (mode × family) cross-axis. A curated allowlist would be easy to break the moment a Llama-3-Math variant lands; the guard remains accurate regardless.

**Change 2.3.** Add the four `--inf_llm_*` args, mirroring `eval_ruler.py:166-176` (with one *intentional* deviation — see note). Insert immediately after the ShadowKV parity block (after line 240):

```python
# InfLLM baseline params (only used when --mode inf_llm). Llama 3.x only.
parser.add_argument("--inf_llm_n_init", type=int, default=128,
                    help="InfLLM: sink token count.")
parser.add_argument("--inf_llm_repr_topk", type=int, default=4,
                    help="InfLLM: representative tokens per block.")
parser.add_argument("--inf_llm_max_cached_block", type=int, default=128,
                    help="InfLLM: GPU block cache size. MUST be >= "
                         "INF_LLM_CONFIG['topk'] (default 64); upstream "
                         "MemoryCache.alloc hard-asserts this.")
parser.add_argument("--inf_llm_chunk_size", type=int, default=8192,
                    help="InfLLM: prefill chunk size for GreedySearch.")
```

Note: Default differs from `eval_ruler.py:172` (which has its own latent bug — see follow-up #2). Shipping the broken default to preserve cross-script consistency would knowingly fail the smoke test. `INF_LLM_CONFIG['topk']=64` and the upstream `MemoryCache` block manager hard-asserts `max_cached_block >= topk` (`feedback_infllm_max_cached_block.md`), so default 32 is broken at model init and would make Acceptance Criterion #1 expected-to-fail. We pick 128 here.

**Change 2.4.** Replace the guard call at line 246:

```python
_assert_mode_model_compatible(args.mode, args.base_model)
```

**Change 2.5.** Extend the auto run-name `if/elif` chain (lines 251-264) by appending an `inf_llm` branch. Mirror `eval_ruler.py:203-205` and add the `_aime25` suffix:

```python
elif args.mode == "inf_llm":
    args.run_name = (f"{tag}_inf_llm_nini{args.inf_llm_n_init}"
                     f"_repr{args.inf_llm_repr_topk}_{suffix}")
```

**Change 2.6 (diagnostic short-circuit for `--num_samples 0`).** Insert immediately after the existing `if args.skip_existing:` block (current `eval_aime25.py:265-269`) inside `parse_args()`. Both short-circuits live next to each other:

```python
# Diagnostic short-circuit: --num_samples 0 prints the auto-generated
# run_name to stdout and exits 0 without loading the model. Mirrors the
# --skip_existing short-circuit at eval_aime25.py:265-269.
if args.num_samples == 0:
    print(args.run_name)
    sys.exit(0)
```

Why: the existing decode-time guard `if args.num_samples > 0:` (`eval_aime25.py:293`) does *not* short-circuit on `0` — it falls through and runs all 30 samples. Acceptance #5 below probes the auto-generated run name and must not load the model; this gate makes that probe cheap (<2 s, no GPU touch).

### Step 3 — Generator dispatch + tokenizer guard + memory probe in `evaluate(...)` (lines 277-357)

**Change 3.0 (tokenizer chat-template family guard).** At `eval_aime25.py:98`, replace the unconditional `enable_thinking` branch with a Qwen3-only guard so a future Llama chat-template revision can never silently flip the flag:

```python
if "qwen3" in args.base_model.lower() and "enable_thinking" in (tokenizer.chat_template or ""):
    chat_kwargs["enable_thinking"] = True
```

Why: Pre-mortem #2. Defensive against missing templates *and* upstream chat-template revisions on either family.

**Change 3.1.** Extend the dispatch fork at lines 310-324 with an `inf_llm` branch. Place it between the `seer_attention` and the trailing `else` so the order matches `eval_longbench_v2.py:270-292`:

```python
with torch.no_grad():
    if args.mode == "seer_attention":
        output_ids, _ = model.batch_exist_generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_length=input_len + args.max_new_tokens,
            do_sample=False,
        )
    elif args.mode == "inf_llm":
        # InfLLM uses a stateful ContextManager KV cache that HF generate()
        # cannot round-trip. Use the GreedySearch adapter built in
        # load_model_and_tokenizer. AIME extracts numeric \boxed{...}, so
        # we omit extra_end_token_ids (mirrors eval_longbench_v2.py:280-283).
        output_ids = args._inf_llm_generator.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
```

**Change 3.2.** Add the per-sample state reset. Currently lines 326-328 do `del input_ids, output_ids; torch.cuda.empty_cache()`. Insert the InfLLM `clear()` *before* the `del`:

```python
generated_ids = output_ids[0, input_len:]
if args.mode == "inf_llm":
    # ContextManager persists past_kv across samples; reset it.
    args._inf_llm_generator.clear()
del input_ids, output_ids
torch.cuda.empty_cache()
```

Order rationale: `clear()` drops the past_kv reference (sets `self.past_kv = None` per `baselines/infllm/__init__.py:106`); the subsequent `empty_cache()` returns those freed allocations to CUDA. Mirrors v2's order. Either ordering of `clear()` vs `del` is correct — we keep `clear → del → empty_cache` for repo consistency with `eval_longbench_v2.py:294-302`.

**Change 3.3 (memory probe).** At end of `evaluate(...)` (insertion anchor `eval_aime25.py:357`, after the final results-write but before return), add a one-liner so the smoke test can grep peak GPU memory:

```python
print(f"[mem] peak_alloc_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}",
      flush=True)
```

Why: Pre-mortem #3 observability hook. AIME `max_new_tokens=16384` × InfLLM eviction-heavy decode on A6000 (48 GiB) is the OOM-risk path; printing peak alloc lets us assert < 44 GiB without instrumentation overhead.

**Change 3.4 (sample-0 decode snippet).** Inside `evaluate(...)`, immediately after the existing `out_f.write(json.dumps(result) + "\n")` and `out_f.flush()` (around `eval_aime25.py:343-344`) and *before* the `if is_correct: correct += 1` line (line 346), write the first 200 decoded chars of the first completed sample to a sidecar file:

```python
# Step 3.4 — write a sample-0 response snippet for smoke-test inspection.
if total == 0:  # i.e. this is the first completed sample of this run
    sample_path = os.path.join(args.output_dir, f"{args.run_name}_sample0.txt")
    with open(sample_path, "w", encoding="utf-8") as sf:
        sf.write(response[:200])
```

Why: Acceptance #1's "decode quality probe" needs an explicit code-side hook so the smoke test can grep a deterministic file rather than parse the JSONL. The `total == 0` check fires on the first completed sample (the per-sample `total += 1` happens later at the loop tail).

### Step 4 — Summary: include `inf_llm_config` (lines 379-415)

**Change 4.1.** Extend the `if args.mode == ... elif ...` chain in `build_summary` (lines 400-413) with one more elif, mirroring `eval_longbench_v2.py:443-445`:

```python
elif args.mode == "inf_llm":
    from infllm.config import INF_LLM_CONFIG
    summary["inf_llm_config"] = INF_LLM_CONFIG
```

The config dict is mutated by `load_model_and_tokenizer` *before* this branch fires (Python module-level dicts are shared), so the snapshot reflects the actual run-time values for `n_init`, `repr_topk`, `max_cached_block`, `chunk_size`. (Same pattern as ruler's `inf_llm_config` block.)

### Step 5 — No changes needed to `apply_monkey_patch` / `load_model_and_tokenizer`

`eval_aime25.py:46` already imports both donors from `eval_ruler.py`. The `inf_llm` branches in `apply_monkey_patch` (`eval_ruler.py:369-370`, no-op) and `load_model_and_tokenizer` (`eval_ruler.py:493-538`, eager attn + `torch_dtype` + stripped-rope LlamaConfig + `init_inf_llm` + `build_inf_llm_generator`) are correct as-is. **Do not duplicate any of this logic into eval_aime25.**

The donor sets `args._inf_llm_generator` on the parsed-args object, which Step 3 then reads. AIME inherits this for free because of the import on line 46.

### Step 6 — `--skip_existing` audit

`eval_aime25.py:265-269` checks `{output_dir}/{run_name}_summary.json` existence. Step 2.5 produces a deterministic `inf_llm` run-name, so resume works without code changes. **Verification:** the smoke test below re-runs the same command twice and confirms the second invocation prints `SKIP (already exists): ...`.

---

## Acceptance Criteria (testable from shell)

All commands run from `/home/yoongonkim/DCT-Page` in the InfLLM env (transformers==4.37.2; per memory `reference_conda_paths.md`: `conda activate infllm`).

1. **Smoke run completes, `inf_llm_config` is recorded, decode is non-degenerate, and peak memory is bounded.**
   ```bash
   conda activate infllm
   python eval_aime25.py --mode inf_llm \
     --base_model meta-llama/Llama-3.1-8B-Instruct \
     --num_samples 1 --max_new_tokens 16384 \
     --output_dir /tmp/aime25_inf_llm_smoke --run_name smoke \
     2>&1 | tee /tmp/aime25_inf_llm_smoke/stdout.log
   ```
   - Exit code 0.
   - `/tmp/aime25_inf_llm_smoke/smoke.jsonl` has 1 line.
   - `/tmp/aime25_inf_llm_smoke/smoke_summary.json` exists; the parsed JSON contains a non-empty `"inf_llm_config"` dict whose `n_init`, `repr_topk`, `max_cached_block`, `chunk_size` match the CLI defaults (`max_cached_block=128`).
   - **Decode quality probe (new):** the first 200 decoded chars of sample 0's `generated_ids` are printed to stdout and saved to `/tmp/aime25_inf_llm_smoke/smoke_sample0.txt` for grep. The text is non-empty and decodable (no UTF-8 errors).
   - **Extraction probe (new):** parse `smoke_summary.json` and assert `extraction_failures < num_samples`, i.e. at least one valid `\boxed{...}` was extracted across the run. If all extractions fail, the run is a fail even with exit code 0. (The harness's existing `extraction_failures` counter at `eval_aime25.py:~340` is sufficient.)
   - **Memory probe (new):** stdout contains a `[mem] peak_alloc_gb=<X>` line (from Step 3.3). Assert `X < 44.0`. If `X >= 44`, raise the issue and lower `--max_new_tokens` default to 8192 in a follow-up.

2. **Wrong-pair guard fires before model load.**
   ```bash
   python eval_aime25.py --mode inf_llm --base_model Qwen/Qwen3-8B --num_samples 1
   ```
   - Exit code != 0.
   - stderr contains the substring `is Llama-only`.
   - No HF download is triggered (you can verify by observing the failure happens within ~2 seconds, before tokenizer fetch).

3. **Existing Qwen3 baseline regression-free.**
   ```bash
   python eval_aime25.py --mode baseline \
     --base_model Qwen/Qwen3-8B --num_samples 1 \
     --output_dir /tmp/aime25_baseline_regress --run_name regress
   ```
   - Exit code 0.
   - `/tmp/aime25_baseline_regress/regress_summary.json` exists with `mode == "baseline"` and exactly 1 record under `per_problem`.

4. **`--skip_existing` skips a completed inf_llm run.**
   ```bash
   # After the smoke run from (1):
   python eval_aime25.py --mode inf_llm \
     --base_model meta-llama/Llama-3.1-8B-Instruct \
     --num_samples 1 --max_new_tokens 16384 \
     --output_dir /tmp/aime25_inf_llm_smoke --run_name smoke \
     --skip_existing
   ```
   - Exit code 0.
   - stdout contains `SKIP (already exists): /tmp/aime25_inf_llm_smoke/smoke_summary.json`.
   - No process touches the GPU (no model load).

5. **Run-name auto-generation produces the expected default (no model load).**
   ```bash
   python eval_aime25.py --mode inf_llm \
     --base_model meta-llama/Llama-3.1-8B-Instruct \
     --num_samples 0
   ```
   - Exit code 0 in < 2 seconds.
   - stdout is exactly the auto-generated run name, e.g. `qwen_inf_llm_nini128_repr4_aime25` for Qwen base or `llama_inf_llm_nini128_repr4_aime25` for Llama base (match `model_name_tag()` output).
   - No HF download, no GPU touch.

**Explicitly NOT in scope:** matching Qwen3 reasoning accuracy on AIME25. Llama-3.1-8B-Instruct is not a reasoning-tuned model. The deliverable here is "InfLLM is exercised end-to-end on AIME25", not "InfLLM scores well on AIME25". `overall_accuracy` in the smoke run may reasonably be 0%.

---

## ADR

**Decision.** Add `inf_llm` as a Llama-only mode in `eval_aime25.py` by (a) generalizing the per-mode model-family guard, (b) dropping `choices=` on `--base_model` (matching ruler/v2), (c) adding the four `--inf_llm_*` CLI args (with `max_cached_block` default raised to 128 to satisfy the upstream `>= topk` assert), (d) adding an `inf_llm` branch in the `evaluate(...)` generator dispatch with per-sample `clear()`, (e) tightening the chat-template `enable_thinking` toggle with a Qwen3 family guard, (f) emitting a one-line peak-memory probe at the end of `evaluate(...)`, and (g) adding the `inf_llm_config` block to `build_summary`. All other plumbing (monkey-patch no-op, model load, generator construction, `assert_llama_only`) is reused unchanged from `eval_ruler.py` via the existing import on `eval_aime25.py:46`.

**Drivers.**
1. Reuse the donor wiring; do not duplicate it.
2. Keep Qwen3 the default and prevent regression on existing modes.
3. Mirror `eval_longbench_v2.py`'s generator/clear/summary pattern for consistency across eval scripts.

**Alternatives considered.**
- *Option B* (special-case skip in the existing Qwen3 guard) — rejected: doesn't generalize for the next Llama-only baseline; helper name lies.
- *Option C* (drop the guard entirely; rely on `assert_llama_only` deep in the donor loader) — rejected: error surfaces too late, after eager-attn config building; also can't lift the `--base_model choices=` argparse hard-block on its own.
- *Duplicating the InfLLM init code into eval_aime25* — rejected by the "Hard constraints" in the brief; the donor is already correct.
- *Curated 2-element allowlist for `--base_model`* — rejected: ruler/v2 take it as a free string, and any future Llama-3-Math variant would require another patch. The new guard already catches typos on the cross-axis.
- *Inheriting `inf_llm_max_cached_block=32` from ruler verbatim* — rejected: that default is a latent bug (`max_cached_block (32) < topk (64)`). Shipping it knowingly would make Acceptance #1 fail. Logged as ruler follow-up #2.

**Why chosen.** Option A is the only path that simultaneously (a) lets Llama-3.x in for `inf_llm`, (b) keeps `_assert_*_compatible` semantically honest for every other mode, (c) catches `--mode inf_llm --base_model Qwen/Qwen3-8B` *before* model load with a clear error, and (d) leaves a clean extension point for the next Llama-only baseline (DuoAttention, ShadowKV) to be added by appending one element to `LLAMA_ONLY_MODES`.

**Consequences.**
- *Positive:* one-file diff (~60 lines), zero baseline duplication, future Llama-only baselines plug in trivially, error UX is improved (the guard now reports the actual problem: mode-vs-model mismatch). The `[mem] peak_alloc_gb=...` probe gives a permanent observability hook for future eviction-heavy decode work.
- *Neutral:* the renamed guard `_assert_qwen3_compatible` → `_assert_mode_model_compatible` is a private helper; no external callers. AIME is the only script with this helper *because* it's the only script that needs the cross-axis (mode × family) check today; ruler/v2 don't use mode×family today and forcing them to adopt the helper would expand scope. We accept the small duplication-of-purpose vs `assert_llama_only` for early-failure UX.
- *Negative:* aime25 now has to be invoked from the InfLLM env (`transformers==4.37.2`) when running `inf_llm` mode. This is intrinsic to the InfLLM baseline's env pinning and is already the operational reality for `eval_ruler.py` / `eval_longbench_v2.py`; we are not making it worse.
- *Risk:* InfLLM `max_new_tokens=16384` × A6000 (48 GiB) is untested at this combo. Mitigated by the Step 3.3 memory probe and Acceptance #1's `< 44 GiB` assertion; if tripped, lower default in a follow-up.

**Follow-ups (NOT in this PR).**
1. Sweep wrapper `run_aime25_infllm.sh` mirroring `run_ruler_infllm.sh` (out of scope: this plan is the script change only).
2. Fix the latent `--inf_llm_max_cached_block=32` default in `eval_ruler.py:172` (it violates `>= topk=64`); aime25 already ships with the correct default of 128.
3. When a reasoning-tuned Llama (e.g. Llama-3.x-Math-Instruct) lands publicly, document it in `--base_model` help text so AIME25 + InfLLM produces a meaningful accuracy number, not just a "runs end-to-end" number.

---

## Step Count and Complexity

- **Steps:** 6 numbered steps + 5 sub-steps (2.6 num_samples=0 short-circuit, 3.0 tokenizer guard, 3.1+3.2 dispatch+clear, 3.3 memory probe, 3.4 sample-0 snippet).
- **Files touched:** 1 (`eval_aime25.py`).
- **Estimated diff:** ~65 lines added, ~10 lines modified, ~6 lines deleted (the old `_assert_qwen3_compatible` body and the unconditional `enable_thinking` branch are replaced).
- **Estimated complexity:** LOW.
