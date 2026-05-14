# InfLLM: Vendor Upstream + Decouple `n_recent` from `n_local`

**Status:** Drafted (RALPLAN-DR, SHORT mode)
**Risk:** Low/Medium — vendoring + additive parameter, fully reversible, no public API breakage
**Plan name:** `infllm_vendor_and_n_recent`

---

## RALPLAN-DR Summary

### Principles (1–5)

1. **Vendor with minimal divergence from upstream.** Patches to `inf_llm/` are surgical, optional, and clearly marked. Default behavior is byte-identical to today.
2. **Additive parameter, safe default.** `n_recent=None` (and `output_window=None` at the kernel boundary) preserves current semantics exactly. The new knob is opt-in.
3. **Preserve eval reproducibility.** Existing `INF_LLM_CONFIG` defaults, run-name encoding, and CLI surfaces remain backward compatible. Sweep scripts (`run_ruler_infllm.sh` etc.) keep working without modification.
4. **One verification cycle, not two.** Ship vendoring + `n_recent` together so a single RULER smoke test validates both changes.
5. **Localize the patch surface.** Five upstream files are modified: three patched (`torch_impl.py`, `context_manager.py`, `inf_llm.py`) and two trimmed re-exports (`__init__.py`, `utils/__init__.py`). All other vendored files are byte-identical to upstream.

### Decision Drivers (top 3)

1. **Dependency removal** — Eliminate the editable `pip install -e /home/yoongonkim/InfLLM/` requirement so `baselines/infllm/` is self-contained.
2. **Ablation flexibility** — Enable experiments where InfLLM's output window (sliding window for the local-attention output sum) is smaller than the scoring window (block promotion horizon).
3. **Reproducibility** — Record the exact upstream commit, prevent silent upstream drift, and keep the wrapper trivially clonable.

### Viable Options

#### Option A (CHOSEN): Vendor + `n_recent` in a single change

**Pros:**
- One verification cycle (one RULER smoke run validates both vendoring and the new param).
- Vendoring alone has no functional payoff for the user's ablation roadmap — bundling avoids a no-op intermediate state.
- Patch surface is small enough (3 files) that mixing it with the copy step does not obscure review.
- Backward-compat default (`n_recent=None`) means the vendored copy is byte-equivalent to upstream when the flag is unset, so vendoring correctness can still be isolated via a "no flag" smoke run.

**Cons:**
- Slightly larger diff than a pure-vendor PR.
- If a regression appears, two hypotheses to bisect (vendoring vs `n_recent` patch). Mitigated by the "no-flag run matches a clean upstream run" smoke test.

#### Option B (REJECTED): Vendor first, then add `n_recent` in a follow-up

**Invalidation rationale:**
- A pure-vendor PR has zero functional payoff — it produces a dead artifact that just shuffles code around without unlocking any user-facing work.
- Doubles the verification cost: two RULER smoke runs, two reviewer passes, two opportunities for upstream drift between PRs.
- The `n_recent` patches are surgical (3 files, ~30 lines total) and don't risk hiding inside the vendoring diff — they're clearly localized to the `output_window` plumbing.
- The user explicitly chose A; this option exists only for completeness.

#### Option C (REJECTED): Keep editable install, only add `n_recent`

**Invalidation rationale:**
- Does not solve the original pain point (fragile cross-repo dependency).
- Patches to `/home/yoongonkim/InfLLM/` are not version-controlled in DCT-Page and silently break for anyone cloning fresh.
- Reproducibility risk: a colleague's upstream checkout could diverge undetectably.

---

## Requirements Summary

### Functional

- `baselines/infllm/` must work as a self-contained baseline with no external editable install requirement.
- `from baselines.infllm.upstream import patch_hf` resolves to the vendored copy.
- A new `n_recent` parameter (Optional[int], default None) propagates from CLI → `INF_LLM_CONFIG` → `patch_hf` → `inf_llm_forward` → `ContextManager` → `torch_impl.append(output_window=...)`.
- When `n_recent` is None or equals `n_local`, attention output is byte-identical to current behavior.
- When `n_recent < n_local`, only the **output sliding window** narrows; scoring horizon, block promotion, RoPE, local_k trim, and global_remainder boundary continue to use `n_local`.
- Assertion `n_recent <= n_local` enforced at construction.
- Triton path (`fattn=True`) is guarded: rejects `n_recent != n_local` with a clear error.

### Non-Functional

- Vendored files (other than the three patched ones) must be unmodified copies of upstream.
- Wrapper API (`patch_hf` kwargs) remains backward compatible — existing sweep scripts keep working without changes.
- Run-name encoding preserves backward compatibility (no `_nrec` suffix when flag is unset).
- The wrapper directory must remain named `infllm/` (no underscore) to avoid colliding with the upstream `inf_llm` package import path.

### Out of Scope

- No refactor of `context_manager.py` beyond the minimal `n_recent` plumbing.
- No changes to `chat.py`, `utils/patch_mc.py`, `utils/greedy_search.py` (not used by the wrapper, do not vendor).
- No expansion to other baselines.
- No new sweep scripts (existing `run_ruler_infllm.sh` etc. unchanged).

---

## Acceptance Criteria

1. **Self-contained import:** From `/home/yoongonkim/DCT-Page/`, `python -c "from baselines.infllm.upstream import patch_hf; print(patch_hf)"` succeeds without `/home/yoongonkim/InfLLM/` being on `sys.path` or pip-installed.
2. **Wrapper import unchanged:** `python -c "from baselines.infllm import init_inf_llm, InfLLMGenerator, build_inf_llm_generator"` succeeds (these are the public symbols re-exported from `baselines/infllm/__init__.py:187-193`).
3. **Back-compat byte-identity:** A 2-sample RULER run with no `--inf_llm_n_recent` flag produces a results JSONL byte-identical to a run with `--inf_llm_n_recent 4096` (where `n_local=4096`). Verified by `diff` or `sha256sum` of the prediction output files.
4. **New behavior works:** A 2-sample RULER run with `--inf_llm_n_recent 512 --inf_llm_n_local 4096` completes without errors, emits the expected `_nrec512` suffix in the run name, and produces a valid results JSONL.
5. **Assertion fires:** Setting `n_recent > n_local` raises an `AssertionError` from `ContextManager.__init__` at attention-replacement time.
6. **Triton-path guard:** With `fattn=True` and `n_recent != n_local`, model load raises an `AssertionError` with a message pointing at the torch-impl requirement.
7. **No upstream drift:** `diff -r /home/yoongonkim/InfLLM/inf_llm/attention/dot_production_attention/triton_impl.py baselines/infllm/upstream/attention/dot_production_attention/triton_impl.py` shows zero diff (unpatched file).
8. **CLAUDE.md updated:** The `pip install -e /home/yoongonkim/InfLLM/` line is removed; a note indicates `baselines/infllm/` is now self-contained.

---

## Implementation Steps

### Step 1 — Vendor upstream

**Action:** Copy `/home/yoongonkim/InfLLM/inf_llm/` into `/home/yoongonkim/DCT-Page/baselines/infllm/upstream/`, excluding files not used by the wrapper.

**Files to copy (keep):**
- `__init__.py`
- `utils/__init__.py`
- `utils/patch.py`
- `attention/__init__.py`
- `attention/inf_llm.py`
- `attention/infinite_lm.py`
- `attention/stream_llm.py`
- `attention/origin.py`
- `attention/rope.py`
- `attention/utils.py`
- `attention/context_manager.py`
- `attention/dot_production_attention/__init__.py`
- `attention/dot_production_attention/base.py`
- `attention/dot_production_attention/torch_impl.py`
- `attention/dot_production_attention/triton_impl.py`

**Files to skip (not imported by `baselines/infllm/__init__.py`):**
- `chat.py`
- `utils/patch_mc.py` (model_center wrapper)
- `utils/greedy_search.py` (we use `InfLLMGenerator` in the wrapper)

**Commands:**
```bash
mkdir -p /home/yoongonkim/DCT-Page/baselines/infllm/upstream
cp -r /home/yoongonkim/InfLLM/inf_llm/. /home/yoongonkim/DCT-Page/baselines/infllm/upstream/
rm -f /home/yoongonkim/DCT-Page/baselines/infllm/upstream/chat.py
rm -f /home/yoongonkim/DCT-Page/baselines/infllm/upstream/utils/patch_mc.py
rm -f /home/yoongonkim/DCT-Page/baselines/infllm/upstream/utils/greedy_search.py
```

**Also:** Trim re-exports in `baselines/infllm/upstream/__init__.py` and `upstream/utils/__init__.py` so they don't import the removed `patch_model_center` / `GreedySearch` symbols. The literal post-trim content is:

```python
# baselines/infllm/upstream/__init__.py — exactly:
from .utils import patch_hf

# baselines/infllm/upstream/utils/__init__.py — exactly:
from .patch import patch_hf
```

Do NOT leave `GreedySearch`/`patch_model_center` placeholders; delete the import lines entirely so missing references surface immediately rather than as `NoneType` later.

**Optional:** Write `baselines/infllm/upstream/UPSTREAM_VERSION.txt` with `git -C /home/yoongonkim/InfLLM/ rev-parse HEAD` output and the copy date.

**Acceptance:**
- `python -c "from baselines.infllm.upstream import patch_hf"` succeeds from DCT-Page root.
- `python -c "from baselines.infllm.upstream.attention.context_manager import ContextManager"` succeeds.
- Directory tree under `baselines/infllm/upstream/` matches the keep-list above.

---

### Step 2 — Switch wrapper import + add `n_recent` to forwarded kwargs

**File:** `baselines/infllm/__init__.py`

**Edits:**
- Line ~69: `from inf_llm import patch_hf` → `from .upstream import patch_hf`
- Lines 17–30 (`_ATTN_KWARGS_KEYS`): add `"n_recent"` to the tuple. Place it adjacent to `"n_local"` for readability.
- Update the diagnostic print line (if it lists kwargs) to include `n_recent` when not None.

**Acceptance:**
- `python -c "from baselines.infllm import *"` succeeds without touching `/home/yoongonkim/InfLLM/`.
- `_ATTN_KWARGS_KEYS` contains `"n_recent"`.

---

### Step 3 — Patch `torch_impl.py` to split score-mask from output-mask

**File:** `baselines/infllm/upstream/attention/dot_production_attention/torch_impl.py`

**Function:** `append()` (lines 39–97 in upstream)

**Change:**

1. Add `output_window: Optional[int] = None` (and import `Optional` if not already imported — verified present at upstream line 2) to `append()`'s signature, placed adjacent to `sliding_window`.

2. **Mirror the `sliding_window` int→tuple normalization for `output_window`.** Immediately after the existing block at upstream lines 67–68 (`if isinstance(sliding_window, int): sliding_window = (len_k - len_q, sliding_window)`), add:
   ```python
   if isinstance(output_window, int):
       output_window = (len_k - len_q, output_window)
   ```
   This is load-bearing: without it, the `dist` comparison anchor differs between the two masks, and `output_window=512` would be measured from position 0 instead of from the start of the local window.

3. After the existing `mask` is built (upstream lines 60–81), conditionally build `output_mask` using the **same** complement-vs-non-complement logic but with `output_window` in place of `sliding_window`:
   - If `output_window is None` or `output_window == sliding_window`: `output_mask = mask` (Python identity — back-compat fast path, no allocation, no FP drift).
   - Else: build `output_mask` with the same `dist = arange(len_q)[:, None] - arange(len_k)[None, :] + output_window[0]` formula and the same `complement_sliding_window` branch, then reshape via the same `m_shape = [1] * (4-output_mask.dim()) + list(output_mask.shape); output_mask = output_mask.view(m_shape)` pattern (upstream lines 80–81). The reshape is required for the `perhead=True` path (`context_manager.py:690-704`) which uses a 4-d query.

4. Store `output_mask` alongside `mask` by adding `self.output_mask_list.append(output_mask)` (initialize `self.output_mask_list = []` in `__init__`).

**Recommended refactor (optional cleanup):** factor the mask-construction block (upstream lines 60–81) into a helper `_build_mask(window, len_q, len_k, complement, device)` and call it twice. Keeps the diff smaller and removes copy/paste.

**Function:** `finalize()`

**Change (pinned prescription — do NOT use bool-multiply):**

Inside the `for kv_len, mask, get_score, v in zip(...)` loop, after the existing `tmp = torch.masked_fill(tmp, mask==False, 0)` line, **chain a second masked_fill** that applies `output_mask` on top of `tmp`, but only into a separate variable used for the output sum:

```python
# INVARIANT: tmp uses the wide score-mask (sliding_window=n_local).
# tmp_output uses the narrow output-mask (output_window=n_recent).
# ContextManager.append_global() slices loc_score by [-exc_length-n_local:]
# (see context_manager.py:624), so collapsing these two would silently
# bias block scoring. Do NOT collapse.
output_mask = self.output_mask_list[i]  # i = current loop index
if output_mask is mask:
    tmp_output = tmp                      # identity fast path, zero alloc
else:
    tmp_output = torch.masked_fill(tmp, output_mask == False, 0)
if get_score:
    self.score_list.append(tmp.sum(dim=-2))       # WIDE mask for scores
else:
    self.score_list.append(None)
self.ret.add_(torch.matmul(tmp_output, v))         # NARROW mask for output
```

**Why `torch.masked_fill` and NOT `p * output_mask`:** `bool * bf16` upcasts via a different kernel than `masked_fill` and is not bitwise-identical for fp16/bf16. Use `masked_fill` exclusively to preserve byte-identity when `output_mask is mask`.

**Why chained on `tmp` and NOT recomputed from `p`:** The `output_mask ⊆ mask` inclusion invariant (enforced by `assert self.n_recent <= self.n_local` in `ContextManager.__init__`) means any position already zeroed by `mask` stays zero. Chaining avoids a second softmax-slice fetch and an extra allocation on the narrow path.

**Default behavior guarantee:** When `output_window is None`, `output_mask is mask` (Python identity), so `tmp_output is tmp` and the matmul/sum path runs unchanged — zero extra allocation, zero FP drift, byte-identical to upstream.

**Acceptance:**
- Python-level unit micro-test (see Step 10.1a): construct a fake `TorchMultiStageDotProductionAttention`, drive it through `append()` with `output_window=None` vs `output_window=sliding_window` on the same input, `torch.equal()` the `.ret` and `.score_list[0]` outputs.
- Manual trace: when `output_window < sliding_window`, `score_list` accumulation uses the full `sliding_window` mass while `self.ret` zeroes contributions from positions outside `output_window`.

---

### Step 4 — Patch `context_manager.py` to plumb `n_recent` to the output mask

**File:** `baselines/infllm/upstream/attention/context_manager.py`

**Edits:**

1. `ContextManager.__init__`:
   - Add `n_recent: Optional[int] = None` to the signature (alongside the existing `n_local`).
   - Body: `self.n_recent = n_local if n_recent is None else n_recent`
   - Assertion: `assert self.n_recent <= self.n_local, f"n_recent ({self.n_recent}) must be <= n_local ({self.n_local})"`

2. `_append()` — the first `attn.append(...)` call at upstream lines 486–489 (the local-context append, the one that consumes the sliding window for output):
   - Add `output_window=self.n_recent` to that single call.
   - Add an invariant comment immediately above the call:
     ```python
     # INVARIANT: sliding_window=self.n_local (wide) sets the scoring horizon
     # consumed by append_global() at line 624 via local_score[-exc_length-n_local:].
     # output_window=self.n_recent (narrow) only affects the output sum in
     # torch_impl.finalize(). Do NOT swap these — see Step 3 invariant.
     ```

3. **Do NOT touch** the second `attn.append(...)` call at upstream lines 526–531 (the global path with `complement_sliding_window=True`). The narrow `output_window` only applies to the local sliding-window output; the global complement path keeps its original semantics. Do not pass `output_window=...` to that call. If desired, assert this by adding a one-line comment: `# global path: output_window intentionally omitted (no narrow window on complement).`

4. **Do NOT touch** the other `self.n_local` references at upstream lines 214, 219, 346, 391, 437, 624, 628, 634, 649, 746, 772, 796–798. These govern scoring horizon, block promotion, RoPE distance, local_k trim, and global_remainder boundary, all of which must continue using `n_local`.

**Verification aid:** After editing, `grep -n "self\.n_local\b" context_manager.py` should still show ~11 hits; `grep -n "self\.n_recent\b" context_manager.py` should show exactly two (init assignment + the single output_window plumb).

**Acceptance:**
- Constructing `ContextManager(n_local=4096, n_recent=512)` succeeds.
- Constructing with `n_recent=8192, n_local=4096` raises `AssertionError`.
- Default `n_recent=None` results in `self.n_recent == self.n_local`.

---

### Step 5 — Patch `inf_llm.py` to thread `n_recent` and guard the triton path

**File:** `baselines/infllm/upstream/attention/inf_llm.py`

**Edits:**

1. `inf_llm_forward(...)` signature: add `n_recent: Optional[int] = None` (place near `n_local`). `Optional` is already imported at upstream line 2 — no new import needed.
2. **Triton guard at construction time** (NOT inside the returned `forward()`) — place this assertion in the body of `inf_llm_forward()` itself, before the inner `def forward(...)` definition, so it fires at model patch time rather than at first inference:
   ```python
   assert (not fattn) or (n_recent is None) or (n_recent == n_local), \
       "n_recent != n_local requires the torch implementation; set fattn=False."
   ```
   This matches Acceptance Criterion 6 ("model load raises AssertionError") and fails fast before the first eval sample touches GPU.
3. Forward `n_recent=n_recent` into the `ContextManager(...)` constructor (insert near the existing `n_local=n_local` keyword in the constructor call).

**Acceptance:**
- `inf_llm_forward(n_local=4096, n_recent=512, fattn=False, ...)` returns a working forward closure.
- `inf_llm_forward(n_local=4096, n_recent=512, fattn=True, ...)` raises `AssertionError` immediately at call time (NOT first-forward).
- Default (no `n_recent` kwarg) behaves identically to today.

---

### Step 6 — Add `n_recent` to wrapper config default

**File:** `baselines/infllm/config.py`

**Edit:** Add `"n_recent": None,` to `INF_LLM_CONFIG` (near `"n_local"`). Comment it inline:
```python
"n_recent": None,  # Output sliding window; None => uses n_local. Must be <= n_local.
```

**Acceptance:**
- `INF_LLM_CONFIG["n_recent"] is None`
- Existing scripts that read `INF_LLM_CONFIG` keep working (key addition is non-breaking).

---

### Step 7 — Wire CLI flag into `eval_ruler.py`

**File:** `eval_ruler.py`

**Edits (three touch points, verified against current file state):**

1. **Argparse, immediately after the existing `--inf_llm_n_local` block at line 196–197:**
   ```python
   parser.add_argument(
       "--inf_llm_n_recent", type=int, default=None,
       help="InfLLM output sliding window (defaults to n_local). Must be <= n_local. "
            "Decouples the local-output window from the block-scoring horizon.",
   )
   ```

2. **Run-name encoding** at lines 265–270 (the `args.mode == "inf_llm"` branch that builds `args.run_name`): append `f"_nrec{args.inf_llm_n_recent}"` **only when `args.inf_llm_n_recent is not None`**, e.g.:
   ```python
   if args.inf_llm_n_recent is not None:
       args.run_name += f"_nrec{args.inf_llm_n_recent}"
   ```
   This preserves backward-compatibility of existing run names. Note: the run name change yields a different output directory, so Acceptance Criterion 3's byte-identity diff compares predictions across two different result paths (`infllm_smoke_default/...` vs `infllm_smoke_nrec4096/...`) — the directory names differ but the prediction file contents must hash-match.

3. **Config injection** immediately after the existing `INF_LLM_CONFIG["chunk_size"] = args.inf_llm_chunk_size` line (currently around line 726):
   ```python
   INF_LLM_CONFIG["n_recent"] = args.inf_llm_n_recent
   ```

**Acceptance:**
- `python eval_ruler.py --mode inf_llm --help` shows the new flag.
- Run name with flag set includes `_nrec{N}`; without flag, no `_nrec` suffix.
- `INF_LLM_CONFIG["n_recent"]` reflects the CLI value at eval time.

---

### Step 8 — (Optional, parity) Wire CLI flag into LongBench scripts

**Files:** `eval_longbench_v1.py`, `eval_longbench_v2.py`

**Edits:** Same three-line pattern as Step 7 (argparse, run-name suffix, config injection). Skip if the user wants to keep this minimal — RULER alone proves the plumbing.

**Acceptance:**
- LongBench eval works with and without `--inf_llm_n_recent`.

---

### Step 9 — Update CLAUDE.md

**File:** `CLAUDE.md` (project file at repo root)

**Edits:**
- Remove the `pip install -e /home/yoongonkim/InfLLM/` instruction from the InfLLM environment notes.
- Add a one-line note: "`baselines/infllm/` is self-contained — the upstream `inf_llm` package is vendored under `baselines/infllm/upstream/`."
- If a section documents InfLLM CLI knobs, add `--inf_llm_n_recent` to the list.

**Acceptance:**
- `grep -n "pip install -e /home/yoongonkim/InfLLM" CLAUDE.md` returns no hits.
- New self-contained note is present.

---

### Step 10 — Smoke verification

**Run from `/home/yoongonkim/DCT-Page/`:**

1. **Import smoke (no env activation needed for this step):**
   ```bash
   python -c "from baselines.infllm.upstream import patch_hf; print(patch_hf)"
   python -c "from baselines.infllm.upstream.attention.context_manager import ContextManager; import inspect; print('n_recent' in inspect.signature(ContextManager.__init__).parameters)"
   ```
   Both must succeed; the second must print `True`.

1a. **Python-level kernel micro-test (fast, isolates kernel correctness from vendoring):**

   Create a temporary script `/tmp/test_mask_split.py`:
   ```python
   import torch
   from baselines.infllm.upstream.attention.dot_production_attention.torch_impl import (
       TorchMultiStageDotProductionAttention as Attn,
   )
   torch.manual_seed(0)
   B, H, Lq, Lk, D = 1, 4, 1, 4096, 128
   q = torch.randn(B, H, Lq, D, device="cuda", dtype=torch.bfloat16)
   k = torch.randn(B, H, Lk, D, device="cuda", dtype=torch.bfloat16)
   v = torch.randn(B, H, Lk, D, device="cuda", dtype=torch.bfloat16)
   def run(output_window):
       a = Attn(q.shape, q.dtype, q.device)
       a.append(q, k, v, sliding_window=4096, get_score=True, end=True,
                output_window=output_window)
       ret, scores = a.get_result()
       return ret.clone(), scores[0].clone()
   r1, s1 = run(None)
   r2, s2 = run(4096)        # explicit equal-to-sliding case
   r3, s3 = run(512)         # narrow case
   assert torch.equal(r1, r2),  "ret diverges between None and ==n_local"
   assert torch.equal(s1, s2),  "score diverges between None and ==n_local"
   assert not torch.equal(r1, r3), "ret should differ when output_window=512"
   assert torch.equal(s1, s3), "score must NOT differ — scoring horizon is n_local-wide"
   print("kernel micro-test: PASS")
   ```
   Run: `python /tmp/test_mask_split.py`. Must print `PASS`. This catches FP-drift bugs in seconds (no model load) and verifies the score-vs-output invariant (s1==s3, r1!=r3) independently of RULER.

2. **Back-compat byte-identity (2 samples, default `n_local=4096`):**
   ```bash
   # Activate the infllm env first (per project MEMORY: `conda activate infllm`, torch 2.5.1+cu124)
   python eval_ruler.py --mode inf_llm --base_model meta-llama/Llama-3.1-8B-Instruct \
       --seq_lengths 32768 --num_samples 2 \
       --output_dir results_ruler --run_name infllm_smoke_default

   python eval_ruler.py --mode inf_llm --base_model meta-llama/Llama-3.1-8B-Instruct \
       --seq_lengths 32768 --num_samples 2 \
       --inf_llm_n_recent 4096 \
       --output_dir results_ruler --run_name infllm_smoke_nrec4096

   # Find each run's predictions.jsonl and hash them
   PRED_DEFAULT=$(find results_ruler -path '*infllm_smoke_default*' -name 'predictions.jsonl' | head -1)
   PRED_NREC=$(find results_ruler -path '*infllm_smoke_nrec4096*' -name 'predictions.jsonl' | head -1)
   sha256sum "$PRED_DEFAULT" "$PRED_NREC"
   ```
   Both hashes must match. Test exercises the `output_mask is mask` identity fast path end-to-end.

3. **New behavior (2 samples, narrowed output window):**
   ```bash
   python eval_ruler.py --mode inf_llm --base_model meta-llama/Llama-3.1-8B-Instruct \
       --seq_lengths 32768 --num_samples 2 \
       --inf_llm_n_recent 512 \
       --output_dir results_ruler --run_name infllm_smoke_nrec512
   ```
   Must complete without error; run name must contain `_nrec512`. Note: this exercises the chunked-decode path because the wrapper's `chunk_size=8192` default applies during prefill — confirms `_use_chunk_topk` interaction at `context_manager.py:749-770` still works under narrowed output window.

4. **Assertion smoke (should error out fast):**
   ```bash
   python eval_ruler.py --mode inf_llm --base_model meta-llama/Llama-3.1-8B-Instruct \
       --seq_lengths 32768 --num_samples 1 \
       --inf_llm_n_local 4096 --inf_llm_n_recent 8192 \
       --output_dir results_ruler --run_name infllm_smoke_assert
   ```
   Must raise `AssertionError` from `ContextManager.__init__` at attention replacement time (before any decode work).

5. **Whole-tree upstream-diff check** (broadened from the original 3-file check):
   ```bash
   diff -rq /home/yoongonkim/InfLLM/inf_llm/ \
            /home/yoongonkim/DCT-Page/baselines/infllm/upstream/ \
       | grep -v -E "(chat\.py|patch_mc\.py|greedy_search\.py|__pycache__)" \
       | tee /tmp/infllm_vendor_diff.txt
   ```
   `/tmp/infllm_vendor_diff.txt` must list EXACTLY the five expected diffs:
   - `__init__.py` (trimmed re-export)
   - `utils/__init__.py` (trimmed re-export)
   - `attention/context_manager.py` (n_recent patch)
   - `attention/inf_llm.py` (n_recent thread + triton guard)
   - `attention/dot_production_attention/torch_impl.py` (mask-split)

   Any extra entry indicates accidental modification or `cp -r` corruption.

**Acceptance:** All six smoke checks pass.

---

## Risks and Mitigations

### R1 — Back-compat regression in `torch_impl.py` (HIGH-IMPACT, low-likelihood)

**Risk:** The mask-split refactor in `finalize()` introduces FP drift even when `output_window=None`, breaking every existing InfLLM result.

**Mitigation:**
- Default code path uses `output_mask is mask` (Python identity), guaranteeing the existing single-mask matmul/sum runs unchanged with no extra allocation.
- **Two-layer verification gate:**
  - **Layer 1 — Python micro-test (Step 10.1a):** GPU-resident kernel test that asserts `torch.equal()` for `output_window=None` vs `output_window=sliding_window` (back-compat) AND `score_list` invariance under narrowing AND `ret` divergence under narrowing. Runs in seconds, isolates kernel correctness from any vendoring/copy step.
  - **Layer 2 — Hash-equality smoke (Step 10.2):** `sha256sum` of 2-sample RULER predictions with no flag vs `--inf_llm_n_recent 4096`. Exercises the full eval pipeline end-to-end.
- The pinned `torch.masked_fill` prescription (NOT `bool * fp`) in Step 3 eliminates the bool-cast FP drift class.
- If Layer 1 fails: bug is in the kernel patch only — revert Step 3 and re-implement.
- If Layer 1 passes but Layer 2 diverges: bug is in plumbing (ContextManager → torch_impl), not the kernel — focus on Step 4.

### R2 — Triton path silent divergence (MEDIUM)

**Risk:** A future user enables `fattn=True` with `n_recent != n_local` and silently gets wrong results because the triton kernel doesn't honor the output window.

**Mitigation:**
- Step 5 adds an explicit `assert` at forward construction time.
- Step 1 leaves `triton_impl.py` byte-identical to upstream, so any future port is a clean delta.

### R3 — Wrapper rename collision with upstream `inf_llm` package (LOW)

**Risk:** Someone renames `baselines/infllm/` to `baselines/inf_llm/` and breaks `from .upstream import patch_hf` because the directory name now shadows the upstream package, replaying the issue documented in `feedback_infllm_shim_naming.md`.

**Mitigation:**
- Keep the directory name `infllm/` (no underscore). Document this in `baselines/infllm/__init__.py` as a module-level comment.
- Mitigation already in place in MEMORY: `feedback_infllm_shim_naming.md`. No new action needed beyond preserving the existing name.

### R4 — Vendored copy drifts from upstream over time (LOW)

**Risk:** Upstream `/home/yoongonkim/InfLLM/` gets new fixes that don't make it into the vendored copy.

**Mitigation:**
- Optional `UPSTREAM_VERSION.txt` (Step 1) records the source SHA.
- Step 10's `diff -r` of unpatched files (triton_impl.py, rope.py, utils.py) serves as a periodic drift check anyone can re-run.
- This is acceptable risk: InfLLM is a research baseline, not a live dependency, and we control the upstream tree.

### R5 — Removing `chat.py` / `patch_mc.py` / `greedy_search.py` breaks an import chain (LOW)

**Risk:** A file we kept transitively imports one of the skipped files.

**Mitigation:**
- Step 1 includes a follow-up edit to `upstream/__init__.py` and `upstream/utils/__init__.py` to drop re-exports of the removed symbols.
- Step 2's wrapper import smoke (`python -c "from baselines.infllm.upstream import patch_hf"`) catches any missed transitive import immediately.
- If a missed import surfaces, the fix is either (a) copy the missing file back, or (b) prune the re-export.

### R6 — Sweep scripts hardcode config dict keys (LOW)

**Risk:** A sweep script (`run_ruler_infllm.sh` or similar) does `cat > config.py <<EOF` and overwrites our additions.

**Mitigation:**
- `n_recent` defaults to `None` in `INF_LLM_CONFIG`, so any sweep that doesn't know about it gets current behavior.
- No sweep script needs editing as part of this plan.

---

## Verification Summary

| Check | Gate | Where |
|---|---|---|
| Vendored import works | `from baselines.infllm.upstream import patch_hf` | Step 1 acceptance |
| Wrapper import works | `from baselines.infllm import init_inf_llm, InfLLMGenerator, build_inf_llm_generator` | Step 2 acceptance |
| **Kernel micro-test** (isolation gate) | `torch.equal()` proves `ret`/`score` invariance under back-compat AND `score` invariance + `ret` divergence under narrowing | Step 10.1a |
| Back-compat byte-identity | `sha256sum` of 2-sample predictions matches with and without `--inf_llm_n_recent 4096` | Step 10.2 |
| New behavior | 2-sample RULER with `--inf_llm_n_recent 512` completes (also exercises chunked decode) | Step 10.3 |
| Assertion enforced | `n_recent > n_local` raises at `ContextManager.__init__` | Step 10.4 |
| Triton guard | `fattn=True` with `n_recent != n_local` raises at `inf_llm_forward()` call time | Step 5 acceptance |
| Whole-tree upstream diff | `diff -rq` lists EXACTLY 5 expected files: 2 trimmed + 3 patched | Step 10.5 |
| CLAUDE.md updated | `grep` shows no `pip install -e /home/yoongonkim/InfLLM` | Step 9 acceptance |

---

## ADR (Architecture Decision Record)

- **Decision:** Vendor InfLLM and add `n_recent` in a single change (Option A).
- **Drivers:** Dependency removal, ablation flexibility, reproducibility.
- **Alternatives considered:** (B) Vendor-then-add — rejected: dead intermediate state, double verification cost. (C) Keep editable install, add `n_recent` only — rejected: doesn't solve the original reproducibility pain point.
- **Why chosen:** A single bundle gives one verification cycle, the patch surface is small and localized (3 files), and the byte-identity smoke test isolates vendoring correctness from the new param.
- **Consequences:**
  - Wrapper becomes self-contained; clones of DCT-Page no longer need a sibling InfLLM checkout.
  - The vendored copy is now a forked artifact — future upstream fixes require manual re-vendoring.
  - Triton path is feature-locked at parity with today (no `n_recent` support); torch path carries the new knob.
  - The `n_recent` knob enables a class of ablations comparing "small output window + large scoring horizon" configurations.
- **Follow-ups:**
  - (Optional) Port `n_recent` to `triton_impl.py` if/when a sweep needs `fattn=True`.
  - (Optional) Extend the CLI knob to `eval_longbench_v1.py` / `eval_longbench_v2.py` (Step 8) for parity if LongBench InfLLM ablations are needed.
  - (Optional) Add `UPSTREAM_VERSION.txt` with the source SHA at vendor time.

---

## Open Questions

None at planning time — the user has verified the codebase facts, chosen Option A, and specified the patch surface precisely.

---

## Changelog (consensus iteration 1 → 2)

Architect and Critic both flagged the same load-bearing defects in the iteration-1 draft. The following revisions were applied in iteration 2:

| Edit | What changed | Why (Critic/Architect finding) |
|---|---|---|
| Step 1 | Spelled out literal post-trim content for both `__init__.py` files (delete imports, don't leave `NoneType` placeholders). | Critic Finding 6: vague trim instructions → two valid implementations, one hides bugs as `NoneType`. |
| Step 3 | Pinned the mask-split prescription to `torch.masked_fill` (NOT `p * output_mask`); chained on `tmp` (Patch Y) exploiting `output_mask ⊆ mask`. | Architect/Critic Finding 1: bool-multiply is not FP-equivalent to `masked_fill` for bf16/fp16. |
| Step 3 | Added explicit `isinstance(output_window, int) → (len_k - len_q, output_window)` normalization, mirroring `sliding_window` at torch_impl.py:67-68. | Architect/Critic Finding 2: without normalization, `dist` anchor is off by `len_k - len_q`. |
| Step 3 | Added INVARIANT comment block tying score-mask vs output-mask to `append_global()` slice at context_manager.py:624. | Critic Finding 4: silent invariant breaks are the worst regression class; no unit tests exist. |
| Step 3 | Mandated `output_mask` reshape via same `m_shape` pattern as `mask` (for `perhead=True` path correctness). | Critic "What's Missing": `perhead=True` reshape was not specified. |
| Step 4 | Added INVARIANT comment above the `attn.append(...)` call at lines 486–489 documenting the score-vs-output split. | Critic Finding 4 (mirror of Step 3). |
| Step 4 | Explicit "do NOT touch" callout for the second (`complement_sliding_window=True`) `attn.append(...)` call. | Critic "What's Missing": global path output_window plumbing was ambiguous. |
| Step 5 | Moved triton guard to `inf_llm_forward()` body (construction time), NOT inside the returned `forward()`. | Critic minor: original placement only fires at first inference, contradicting AC6 "model load raises AssertionError". |
| Step 7 | Replaced "near line ~N" with exact reference points (after line 196 / lines 265-270 / after line 726). Added note that run-name suffix produces a different output dir for the byte-identity test. | Critic M3: line refs could collide; M-run-name interaction was unclear. |
| Step 10 | Added Step 10.1a Python kernel micro-test that isolates kernel correctness in seconds via `torch.equal()`. | Architect/Critic Finding 3: the original byte-identity test does NOT isolate vendoring from the patch (both paths share the patched code). |
| Step 10.5 | Broadened from 3-file diff to full-tree `diff -rq` with grep filter, asserting EXACTLY 5 expected diffs. | Critic Finding 5: narrow diff misses silent `cp -r` corruption of the other 9 kept files. |
| R1 | Updated mitigation to describe the two-layer gate (Layer 1: kernel micro-test, Layer 2: hash equality), and the bug-localization signal from each layer's failure mode. | Critic Finding 3: R1 was overclaiming the original byte-identity test's isolation strength. |
| Verification Summary table | Added kernel-micro-test row; updated diff-check row; updated wrapper-import row with concrete symbols. | Critic M2: AC2 cited nonexistent symbol `setup_infllm`. |
| AC2 | Replaced `setup_infllm` (does not exist) with `init_inf_llm, InfLLMGenerator, build_inf_llm_generator` (verified in `baselines/infllm/__init__.py:187-193`). | Critic M2. |
| Principle 5 | Updated from "three modified files" to "five modified files (three patched, two trimmed)" for accuracy. | Critic M1: numerical inconsistency. |

Informational (no edit, preserved by vendoring):
- Upstream `attention/__init__.py:15` declares `__all__ = ["RotaryEmbeddingESM", "ATTN_FORWARD"]` (typo'd as `FORWARD`) while the actual exported symbol is `ATTN_FORWRAD` (typo'd as `FORWRAD`). Latent upstream bug; byte-identical vendoring preserves it. Affects only `from .attention import *`, which the wrapper does not use. Documented for posterity.
