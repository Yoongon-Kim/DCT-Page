# SnapKV — Vendor as upstream-directory layout

Status: revised iter 3 (planner pass; final iter before APPROVE per architect + critic)
Author: planner agent (ralplan consensus)
Date: 2026-05-11 (iter 1) / 2026-05-11 (iter 2) / 2026-05-11 (iter 3)
Plan name: `snapkv_vendor_upstream_dir`

---

## Iteration 3 changelog (2026-05-11)

Architect verdict on iter 2: APPROVE-WITH-FIXES (one symmetry violation + polish).
Critic verdict on iter 2: APPROVE-WITH-FIXES. Eight fix items applied
(3 blocking, 5 non-blocking). Summary:

- **Fix 1 (blocking)** — Disclosed the SNAPKV_TRACE carry-forward as edit
  #6 in §4.3. The current `_vendor.py:85-86` has a local-only env-gated
  debug print (`if os.environ.get("SNAPKV_TRACE") == "1": print(...)`)
  that upstream `snapkv_utils.py` does NOT have. Iter-2 silently dropped
  it via the verbatim `cp`. Now disclosed symmetrically with the
  `check_version` re-introduction: §4.3 edit #6, §4.4 expanded,
  §6 Step 1 acceptance qualified, §6 Step 4 VENDORING.md template,
  §10 ADR Decision 8 + Consequences.
- **Fix 2 (blocking)** — Added mechanical grep checks for edit #6 landing
  to §6 Step 2 acceptance (`grep -nE '^import os'` and
  `grep -n 'SNAPKV_TRACE'` both expect 1 match).
- **Fix 3 (blocking)** — Qualified §6 Step 1 acceptance line about the
  snapkv_utils.py diff being empty: empty after Step 1's cp, 4-line
  delta after Step 2's edit #6 (1 import + 2-line if/print block + 0
  blank lines vs upstream).
- **Fix 4 (non-blocking, Architect R3)** — Added a standalone
  `import baselines.snap_kv.upstream` snippet between Step 5 Snippets A
  and B (now Snippet B; old B / C renumbered to C / D) to surface
  circular-import regressions with a clear label.
- **Fix 5 (non-blocking, Architect R4 / Critic-refined)** — Simplified
  Step 7 guard: dropped the redundant `importlib.import_module` probe,
  anchored regex to `add_argument("--mode", … choices=[…])` instead of
  the first `choices=[…]` literal. `eval_ruler.py` has 11 separate
  `choices=[…]` literals (verified) so the anchor matters.
- **Fix 6 (non-blocking, Architect R5)** — Toned down "match infllm
  verbatim" framing in §1.1 Principle 2 and §1.3 Option A Pros. The plan
  has 4 disclosed divergences from the infllm precedent (re-export
  surface size, regular vs namespace packages, `check_version`,
  SNAPKV_TRACE); honesty wins over precedent-purity.
- **Fix 7 (non-blocking, Architect R6)** — §9 Success Criteria now
  enumerates `baselines/snap_kv/upstream/LICENSE` as a separate line
  item rather than rolling it into "6 files + LICENSE".
- **Fix 8 (non-blocking, Architect R2)** — §6 rollback paragraph now
  warns "Do not `git commit` between Step 1 and Step 6.5 acceptance"
  (rollback procedure depends on staged/worktree state only).

Length target: ~30-50 lines net delta vs iter-2. Iter 2 was 1121 lines;
iter 3 lands at ~1180.

---

## Iteration 2 changelog (2026-05-11)

Architect verdict on iter 1: NEEDS-REVISION. Critic verdict on iter 1: ITERATE.
Eleven fix items applied (6 blocking, 5 non-blocking). Summary:

- **Fix 1 (blocking)** — Resolved the `upstream/__init__.py` empty-vs-re-export
  contradiction. Adopted **option (a)** from the review: `upstream/__init__.py`
  now re-exports `init_snapkv` and `replace_llama`, and
  `baselines/snap_kv/__init__.py` imports via `from .upstream import …`,
  honestly mirroring infllm's `from .utils import patch_hf` pattern.
  Updated §1.3 Option A description, §6 Step 1 marker-vs-re-export
  language, §6 Step 3 init-shim rewrite, §10 ADR Consequences.
- **Fix 2 (blocking)** — Added ADR Decision 6 + a new §4.4 documenting the
  re-introduction of upstream's `check_version()` + `warnings.warn()` block.
  This is silent in the pinned `snap_kv` env (transformers 4.37.2) and
  stderr-noisy under any other version. Vendoring fidelity wins over the
  iter-1 strip-and-pretend posture.
- **Fix 3 (blocking)** — Added Step 4.5 (pre-verify cache wipe) and Step 6.5
  (post-delete cache wipe) with `find … -name __pycache__ -exec rm -rf {} +`
  to prevent a stale `_vendor.cpython-311.pyc` from masking import-path bugs.
- **Fix 4 (blocking)** — Strengthened Step 5 patch-identity check with a
  `__module__`-suffix assertion so the test snippet does not rely on the
  same import resolution as the production patch.
- **Fix 5 (blocking)** — Added a rollback paragraph at the foot of §6:
  if Step 5 fails after Steps 1–4 land, `git restore` and investigate; do
  NOT proceed to Step 6.
- **Fix 6 (blocking)** — Disambiguated §4.3 item 2's "Lines 4–6: drop the
  Mistral and Mixtral imports". Rewritten with explicit per-line wording
  (line 4 rewritten, lines 5–6 dropped) and a one-liner confirming
  `check_version()` stays.
- **Fix 7 (non-blocking)** — Adopted option (a) for `run_ruler_snapkv.sh`:
  added a top-of-script guard that prints a clear error and exits 2 until
  the follow-up wireup plan lands. Documented as Step 7 (new) + §6.
  Justified in §7 with the explicit "the end-to-end run is still broken"
  call-out.
- **Fix 8 (non-blocking)** — Step 5 first snippet now also exercises
  `from transformers.models.llama.modeling_llama import repeat_kv,
  apply_rotary_pos_emb` so a wrong-transformers-version environment trips
  ImportError before patch-identity testing.
- **Fix 9 (non-blocking)** — Added implicit→regular package divergence as
  edit #5 in VENDORING.md (Step 4 contents). Honestly disclosed that we
  add three `__init__.py` markers upstream does not have.
- **Fix 10 (non-blocking)** — Step 6 grep broadened from
  `grep -rn '_vendor' baselines/snap_kv/ eval_*.py run_*.sh` to
  `git grep -n '_vendor' -- ':!*.md' ':!.omc/'` so `oracle/`, `speed/`,
  and any other surface is caught.
- **Fix 11 (non-blocking)** — Added git-blame-preservation note to §6 Step 1:
  `git mv _vendor.py upstream/snapkv/monkeypatch/llama_hijack_4_37.py` for
  the largest single overlap (~196 LOC), `git add` for the smaller two
  files.

Length target: ~700–850 lines (iter 1 was 681). This iter 2 lands at ~810.

---

## 0. TL;DR

`baselines/snap_kv/` is already vendored, but as a single flat concatenated
`_vendor.py` (300 LOC, 3 upstream files glued together with intra-package
imports stripped). The repo's other "vendored upstream" baseline,
`baselines/infllm/`, instead uses an `upstream/` subdirectory that mirrors the
original package layout AND re-exports a small public surface at
`upstream/__init__.py` (`from .utils import patch_hf`). The user's request —
"vendor snap_kv" — most plausibly means *"bring snap_kv in line with the
infllm pattern"*: replace the flat `_vendor.py` with
`baselines/snap_kv/upstream/snapkv/monkeypatch/*.py`, preserving upstream
file names so future commit bumps are a trivial three-way diff, and have
`upstream/__init__.py` re-export the same two symbols the shim already
needs.

This plan is a refactor with no behavior change *to the shim's public
surface*. Vendoring fidelity does re-introduce upstream's
`check_version()` warning block (§4.4); this is a deliberate disclosed
divergence from the iter-1 design and is silent inside the pinned
`snap_kv` conda env. Mode wiring in `eval_ruler.py` (which currently lacks
`snap_kv` in its argparse `choices` — `run_ruler_snapkv.sh` would fail at
argparse-time today) is **explicitly out-of-scope** for this plan; it
ships as a follow-up. To prevent silently-broken end-to-end runs in the
interim, this plan adds a guard at the top of `run_ruler_snapkv.sh` that
fails fast with a pointer to the follow-up plan (§6 Step 7).

---

## 1. RALPLAN-DR Summary

### 1.1 Principles (4)

1. **Mirror upstream layout when vendoring.** The point of vendoring (vs.
   pip-install) is to keep the upstream file structure intact so `git diff
   upstream-clone baselines/snap_kv/upstream/` is meaningful. A flat
   concatenated `_vendor.py` defeats this — every commit bump becomes a
   manual re-concatenation. The repo already has one canonical example
   (`baselines/infllm/upstream/`) and we should match it.
2. **Extend the conventions of sibling baselines (with disclosed
   divergences).** `baselines/infllm/` is the only other Apache-2.0 /
   non-pip-install / non-Quest baseline; it defines the in-repo standard,
   including the `upstream/__init__.py` re-export pattern. This plan
   adopts that idiom but with four disclosed divergences (two-symbol vs
   infllm's one-symbol re-export; regular packages vs upstream's
   implicit-namespace; `check_version` warning re-introduced from
   upstream; SNAPKV_TRACE carry-forward from prior `_vendor.py`). Reuse
   beats invent — but match the idiom honestly, not verbatim.
3. **Behaviorally identical to current shim public surface.** The current
   `baselines/snap_kv/__init__.py`'s public surface (`init_snap_kv`,
   `assert_llama_only`, `load_llama_config_stripped_rope`) is presumed in
   use by `run_ruler_snapkv.sh` and any future `eval_ruler.py` integration.
   The refactor must not change names, signatures, or semantics for those
   three symbols. Internal-only behavior changes (e.g. re-introducing
   upstream's `check_version()` warning) are allowed if documented in
   §4.4 + VENDORING.md.
4. **License hygiene.** Vendoring requires bundling the upstream LICENSE and
   a `VENDORING.md` recording the upstream URL + commit. The flat
   `_vendor.py` has neither today; the refactor fixes that.

### 1.2 Decision Drivers (top 3)

1. **Diff-against-upstream ergonomics.** When SnapKV bumps a commit, can a
   reviewer eyeball what changed? Flat `_vendor.py` → no, requires re-doing
   the concatenation by hand. Mirrored `upstream/snapkv/monkeypatch/*.py` →
   yes, file-by-file diff.
2. **Consistency cost for future maintainers.** Two vendored baselines with
   two different conventions is a steady tax. Infllm is the older / larger
   shim; aligning snap_kv to it is the cheaper move.
3. **Risk surface of the refactor itself.** Behaviorally identical
   modulo the two disclosed deltas (`check_version` warning,
   SNAPKV_TRACE carry-forward; §4.4) means the risk is mechanical:
   file split + import rewrite + license drop-in. No SnapKV-logic
   edits, no transformers monkey-patch changes. Verification is a
   four-snippet importability + re-export resolution + patch-identity
   + assert-llama-only check.

### 1.3 Viable Options (≥2)

#### Option A — Upstream-directory layout (RECOMMENDED, REVISED)

Split `_vendor.py` back into the original three files and place them under
`baselines/snap_kv/upstream/snapkv/monkeypatch/`, preserving filenames. Add
`__init__.py` markers at `upstream/snapkv/` and
`upstream/snapkv/monkeypatch/` (empty), and a **re-exporting**
`upstream/__init__.py` mirroring `baselines/infllm/upstream/__init__.py`'s
`from .utils import patch_hf` style. Rewrite
`baselines/snap_kv/__init__.py` to import via `from .upstream import
init_snapkv as _init_snapkv, replace_llama as _replace_llama`. Delete
`_vendor.py`. Add `upstream/LICENSE` and `VENDORING.md`.

- Pros
  - Extends the in-repo `baselines/infllm/upstream/` idiom (both layout
    AND the `upstream/__init__.py` re-export indirection) with four
    disclosed divergences enumerated in §4.3 and ADR Consequences. Not
    a verbatim mirror — an honest extension.
  - Trivial `diff -r baselines/snap_kv/upstream/snapkv/
    /home/yoongonkim/SnapKV/snapkv/` for future commit bumps (modulo
    the six whitelisted edits).
  - Easy to add Mistral / Mixtral later without code rewrites — just drop
    the upstream file in and add a re-export line.
  - License attribution lives next to the vendored code, not buried in a
    docstring.
  - Shim's import line is the same shape as infllm
    (`from .upstream import …`); future maintainers learn one pattern.
- Cons
  - One-time refactor work (~7 small file edits, no logic changes).
  - Two extra empty `__init__.py` files in the tree (`upstream/snapkv/`,
    `upstream/snapkv/monkeypatch/`) — upstream uses implicit-namespace
    packages, we use regular. Documented as edit #5 in VENDORING.md.
  - Slightly slower import (one more package boundary to cross — irrelevant
    in practice).

#### Option B — Keep flat `_vendor.py`, just clean up and document

Leave `_vendor.py` as the source-of-truth. Add a top-of-file `VENDORING`
comment block with upstream URL + commit + Apache-2.0 notice. Drop in
`baselines/snap_kv/LICENSE` alongside the file. No structural changes.

- Pros
  - Smaller diff. Zero risk of import-path bugs.
  - Slightly faster import (one file vs. four).
- Cons
  - Diff-against-upstream remains impossible without manual re-concatenation
    every bump → maintenance tax compounds.
  - Diverges from `baselines/infllm/upstream/` convention → two patterns in
    the same directory, future maintainers have to learn both.
  - Adding Mistral / Mixtral later means re-doing the manual concatenation.
  - The current `_vendor.py` header already says "Intra-package imports
    have been deleted" — i.e., the vendoring is destructive and not
    reversible without going back to the upstream clone. This is a smell.
- Invalidation rationale
  - Principle 1 ("mirror upstream layout when vendoring") is directly
    violated. Driver 1 ("diff-against-upstream ergonomics") is the dominant
    long-term cost and Option B optimizes the wrong axis (one-time refactor
    cost vs. recurring bump cost). Option B is only preferable if the team
    commits to never bumping SnapKV, which is unrealistic for a research
    repo that's still actively comparing baselines.

#### Option C — No-op: just clarify the current state to the user

Reply: "SnapKV is already vendored as `_vendor.py`; here's the file tree;
nothing to do." No code change.

- Pros
  - Zero risk.
- Cons
  - Does not satisfy the user's stated request ("I want you to vendor it").
  - Leaves the inconsistency with `baselines/infllm/` indefinitely.
  - Leaves the missing LICENSE / VENDORING.md problem unaddressed.
- Invalidation rationale
  - The user's request is unambiguous in *direction* ("vendor it"), even if
    factually inaccurate about the starting state ("pip install snapkv
    method"). Saying "no-op" misreads the intent: the user knows the file
    layout doesn't match the convention they expect, and is asking for it
    to be normalized. C does not match Principle 2 (sibling-baseline
    consistency) or Principle 4 (license hygiene).

### 1.4 Mode

**SHORT** consensus mode (default). The change is mechanical, single-package,
~7 small file edits, no SnapKV-logic edits, no cross-baseline blast radius.
The two disclosed behavior deltas (`check_version` warning and SNAPKV_TRACE
carry-forward) are covered by §4.4 Delta 1/2 + ADR Decisions 7/8; neither
needs a pre-mortem.

---

## 2. Context

### 2.1 What `baselines/snap_kv/` looks like today (verified 2026-05-11)

```
baselines/snap_kv/
├── __init__.py        71 LOC   — public shim
├── _vendor.py        300 LOC   — concatenation of 3 upstream files
└── config.py          17 LOC   — SNAPKV_CONFIG dict
```

`__init__.py` imports `init_snapkv` and `replace_llama` from `._vendor`.
`_vendor.py`'s docstring states the upstream commit is
`e216ddc84c5bd210378cbdbbba12ba02102aa640` and that intra-package imports
have been stripped. `pip show snapkv` returns nothing in any conda env on
this box. The conda env `snap_kv` exists but has no pip-installed snapkv.
A `_vendor.cpython-311.pyc` (14904 bytes) currently lives in
`baselines/snap_kv/__pycache__/`; cache hygiene is addressed in Step 4.5.

### 2.2 What `baselines/infllm/` (the in-repo precedent) looks like

```
baselines/infllm/
├── __init__.py             6772 B  — substantial shim
├── config.py               1225 B
└── upstream/
    ├── __init__.py         28 B  — `from .utils import patch_hf`   ← re-export, NOT empty
    ├── attention/
    │   ├── __init__.py
    │   ├── context_manager.py
    │   ├── inf_llm.py
    │   ├── infinite_lm.py
    │   ├── origin.py
    │   ├── rope.py
    │   ├── stream_llm.py
    │   ├── utils.py
    │   └── dot_production_attention/{__init__.py, base.py, torch_impl.py, triton_impl.py}
    └── utils/{__init__.py, patch.py}
```

Upstream file names are preserved verbatim under `upstream/`, **and
`upstream/__init__.py` re-exports the single symbol the shim needs
(`patch_hf`)**. The shim `baselines/infllm/__init__.py` imports from
`.upstream` (not deeper) and presents a clean DCT-Page-facing API.
Project memory `feedback_infllm_shim_naming.md` documents the lesson
learned: the wrapper directory must not collide with the upstream package
name. For snap_kv this is not a hazard (snap_kv vs. snapkv differ in
underscore + nesting depth), but it confirms the convention.

### 2.3 Upstream SnapKV (verified)

```
/home/yoongonkim/SnapKV/snapkv/
└── monkeypatch/
    ├── snapkv_utils.py            86 LOC
    ├── llama_hijack_4_37.py      196 LOC
    ├── mistral_hijack_4_37.py    254 LOC
    ├── mixtral_hijack_4_37.py    241 LOC
    └── monkeypatch.py             51 LOC
```

There is **no** top-level `snapkv/__init__.py` upstream — `snapkv` is an
implicit-namespace package (PEP 420). The plan adds three `__init__.py`
markers (`upstream/snapkv/__init__.py` and
`upstream/snapkv/monkeypatch/__init__.py` empty;
`upstream/__init__.py` a re-export). This conversion from
implicit-namespace to regular package is a *real* edit and is documented
as edit #5 in VENDORING.md (§4.3 item 4, Step 4 contents).
Apache-2.0 LICENSE lives at `/home/yoongonkim/SnapKV/LICENSE`. Upstream
HEAD commit: `e216ddc84c5bd210378cbdbbba12ba02102aa640` (matches what
`_vendor.py` already cites).

Intra-package imports (verified):

```python
# snapkv/monkeypatch/llama_hijack_4_37.py:14
from snapkv.monkeypatch.snapkv_utils import init_snapkv

# snapkv/monkeypatch/monkeypatch.py:4-6
from snapkv.monkeypatch.llama_hijack_4_37 import (
    llama_flash_attn2_forward as llama_flash_attn2_forward_4_37,
    prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37,
)
from snapkv.monkeypatch.mistral_hijack_4_37 import ...
from snapkv.monkeypatch.mixtral_hijack_4_37 import ...
```

The Mistral / Mixtral imports in `monkeypatch.py` are unconditional and will
fail at import time if we vendor only the Llama files. This is a real edit
point, not a hypothetical one. See §4.3 item 2 for the chosen handling.

### 2.4 The eval_ruler.py mismatch (verified, deferred)

`eval_ruler.py` argparse `choices` at line 99–104:

```python
choices=["baseline", "page_attention", "seer_attention", "seer_prefill",
         "multipole_attention", "quest_attention", "duo_attention",
         "shadowkv", "inf_llm"]
```

`snap_kv` is absent. `grep -nE "snap_kv|snapkv" eval_ruler.py` returns zero
matches. `run_ruler_snapkv.sh` already references `--mode snap_kv`, so it
would die at argparse-time today. **This plan does not fix the wireup**,
but it *does* add a top-of-script guard to `run_ruler_snapkv.sh` so
running it produces a clear error pointing at the follow-up plan instead
of a confusing argparse stack trace (§6 Step 7).

---

## 3. Work Objectives

1. Replace the flat `baselines/snap_kv/_vendor.py` with an `upstream/`
   subdirectory that mirrors the original SnapKV package layout, preserving
   filenames, and add an `upstream/__init__.py` that re-exports the same
   two symbols the shim consumes today (`init_snapkv`, `replace_llama`).
2. Rewrite `baselines/snap_kv/__init__.py` so its single import line reads
   `from .upstream import init_snapkv as _init_snapkv, replace_llama as
   _replace_llama` — matching the infllm pattern verbatim — and otherwise
   leaves the public surface (`init_snap_kv`, `assert_llama_only`,
   `load_llama_config_stripped_rope`) untouched.
3. Add `baselines/snap_kv/upstream/LICENSE` (Apache-2.0, copied verbatim
   from `/home/yoongonkim/SnapKV/LICENSE`) and a
   `baselines/snap_kv/VENDORING.md` recording upstream URL + commit hash +
   included-files list + the 6 documented edits applied to vendored
   files + the two disclosed behavior deltas (`check_version`,
   SNAPKV_TRACE).
4. Guard `run_ruler_snapkv.sh` against confusing argparse failures by
   adding a top-of-script fail-fast that points at the follow-up wireup
   plan.
5. Verify behavioral equivalence (importability, patch identity, module
   provenance) and clear `__pycache__` to keep stale `.pyc` files from
   masking import-path bugs.

---

## 4. Guardrails

### 4.1 Must have

- Public symbols on `baselines.snap_kv` unchanged: `init_snap_kv`,
  `assert_llama_only`, `load_llama_config_stripped_rope`.
- `SNAPKV_CONFIG` (in `config.py`) unchanged. Do not edit `config.py`.
- The vendored Llama hijack function `llama_flash_attn2_forward` must be
  byte-identical to upstream `snapkv/monkeypatch/llama_hijack_4_37.py`
  except for the single intra-package import (see §4.3 edit #1).
- The vendored `replace_llama()` must be byte-identical to upstream,
  including the `check_version()` definition and the `for version in
  version_list / warnings.warn(...)` warning block. See §4.4 Delta 1.
- The vendored `SnapKVCluster.update_kv()` must be byte-identical to
  upstream **except for the SNAPKV_TRACE carry-forward** (edit #6: one
  `import os` line + a 2-line env-gated `print()` block). The
  SNAPKV_TRACE block must remain inside the else-branch body,
  immediately before `return key_states, value_states`. See §4.4
  Delta 2.
- Apache-2.0 LICENSE attribution present under `baselines/snap_kv/upstream/`.
- `VENDORING.md` records upstream URL + commit hash + files vendored +
  edits-applied list (6 edits, see §4.3).
- `baselines/snap_kv/__pycache__/` is empty (or absent) after the plan
  completes, so re-imports rebuild from the new layout.

### 4.2 Must NOT have

- **No SnapKV-logic edits.** Function bodies of `llama_flash_attn2_forward`,
  `prepare_inputs_for_generation_llama`, `init_snapkv`, `SnapKVCluster`,
  `replace_llama`, and `check_version` must remain byte-identical to
  upstream, **except for the SNAPKV_TRACE diagnostic block inside
  `SnapKVCluster.update_kv()` carried forward as edit #6** (and its
  matching `import os` at the top of `snapkv_utils.py`). No "while I'm
  here" cleanups, no docstring reformatting, no whitespace normalization
  runs. The only allowed edits are the 6 whitelisted ones in §4.3.
- **No changes to `eval_ruler.py`** in this plan. The argparse `choices`
  fix and mode dispatch is a separate follow-up plan (§7).
- **No vendoring of Mistral / Mixtral files.** YAGNI. The shim is Llama-only
  by the existing `assert_llama_only` guard. See §4.3 for how
  `monkeypatch.py` handles their absence.
- **No `_vendor.py` left in the tree** after the refactor. (A compatibility
  re-export shim was considered and rejected — see §8 Decision 1.)
- **No edits to `run_ruler_snapkv.sh`'s body** — only the top-of-script
  guard added per §6 Step 7. The conda activation, env-var configuration,
  and sweep loop remain byte-identical.

### 4.3 Edits applied to vendored upstream files (whitelist)

Only these six mechanical edits are allowed; all other content is
byte-identical to upstream.

1. `upstream/snapkv/monkeypatch/llama_hijack_4_37.py`
   - Line 14 change:
     `from snapkv.monkeypatch.snapkv_utils import init_snapkv`
     → `from .snapkv_utils import init_snapkv`
   - Rationale: with `upstream/snapkv/monkeypatch/__init__.py` present, the
     monkeypatch directory is a package and relative imports resolve
     without needing `snapkv` on `sys.path`.

2. `upstream/snapkv/monkeypatch/monkeypatch.py` (per-line, anchored to
   `/home/yoongonkim/SnapKV/snapkv/monkeypatch/monkeypatch.py`)
   - **Rewrite** line 4 (the Llama hijack import) from absolute to relative:
     `from snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37`
     → `from .llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37`
   - **Drop** lines 5 and 6 entirely (the Mistral and Mixtral hijack
     imports). The Mistral / Mixtral upstream files are not vendored, so
     leaving these lines would cause an unconditional ImportError.
   - **Keep** `check_version()` (lines 8–13 upstream) byte-identical. It is
     called by `replace_llama()` (lines 15–26 upstream).
   - **Keep** `replace_llama()` (lines 15–26 upstream) byte-identical
     INCLUDING its `version_list = ['4.37']` / `for version in version_list:
     ... warnings.warn(...)` block. See §4.4 for the disclosed
     consequence.
   - **Delete** `replace_mistral()` and `replace_mixtral()` function bodies
     in full (Llama-only scope; no callers exist in the shim and their
     definitions would reference the dropped Mistral / Mixtral imports).

3. `upstream/snapkv/monkeypatch/snapkv_utils.py`
   - **One edit (edit #6 below)** preserves a local-only debug trace
     carried forward from prior `_vendor.py:85-86`. No upstream-symbol
     changes; no intra-package import changes (the file has none).

4. `upstream/snapkv/__init__.py` and
   `upstream/snapkv/monkeypatch/__init__.py`
   - **Add** as empty marker files (zero bytes). Upstream uses
     implicit-namespace packages (PEP 420); we use regular packages so the
     `.snapkv_utils` and `.llama_hijack_4_37` relative imports inside
     `llama_hijack_4_37.py` and `monkeypatch.py` resolve unambiguously
     even if some future `pip install -e /home/yoongonkim/SnapKV` shadows
     the `snapkv` name on `sys.path`.

5. `upstream/__init__.py`
   - **Add** as a re-export file mirroring
     `baselines/infllm/upstream/__init__.py`'s
     `from .utils import patch_hf` pattern:
     ```python
     from .snapkv.monkeypatch.snapkv_utils import init_snapkv
     from .snapkv.monkeypatch.monkeypatch import replace_llama

     __all__ = ["init_snapkv", "replace_llama"]
     ```
   - Rationale: this is the only file that does not exist upstream at all
     (the top-of-vendored-tree marker). It is the indirection layer that
     lets `baselines/snap_kv/__init__.py` say `from .upstream import …`,
     in the same shape as infllm's `from .upstream import patch_hf`
     (infllm re-exports one symbol; we re-export two — disclosed
     divergence #1).

6. `upstream/snapkv/monkeypatch/snapkv_utils.py` — SNAPKV_TRACE
   carry-forward
   - **Add** `import os` at top of file. (Upstream `snapkv_utils.py` does
     not import os; this is required for the env-gated debug print
     below.) Place it on a new line immediately after the existing
     `import math` (line 6 upstream), so the import block remains
     contiguous.
   - **Insert** the env-gated debug print inside
     `SnapKVCluster.update_kv()`, immediately BEFORE the final
     `return key_states, value_states` line of the else-branch body
     (line 70 upstream). The exact text to insert (indented to match the
     else-branch body — 12 spaces):
     ```python
             if os.environ.get("SNAPKV_TRACE") == "1":
                 print(f"[snapkv] update_kv fired: q_len={q_len}, cap={self.max_capacity_prompt}", flush=True)
     ```
     Match the indentation level of the surrounding
     `key_states = torch.cat(...)` and `value_states = torch.cat(...)`
     lines, NOT the `return` line. (Upstream uses 12-space indent for
     the else-branch body content.)
   - Rationale: the prior `_vendor.py:85-86` had this block as a
     local-only addition; iter-1 and iter-2 silently dropped it by doing
     a verbatim upstream `cp`. Without this carry-forward,
     `SNAPKV_TRACE=1 bash run_ruler_snapkv.sh` produces no diagnostics
     after the refactor. Disclosed symmetrically with the `check_version`
     re-introduction (§4.4) as one of the two acknowledged behavior
     deltas. See VENDORING.md "Disclosed behavior deltas" and ADR
     Decision 8.

All 6 edits are listed verbatim in `VENDORING.md` (Step 4) so a future
commit-bump diff is unambiguous.

### 4.4 Disclosed behavior changes (two deltas)

Two acknowledged behavior deltas exist between the prior `_vendor.py`,
the new vendored tree, and upstream. They point in opposite directions:
one is a current→upstream-restored change; the other is a
current→preserved-forward change. Both are disclosed in VENDORING.md
"Disclosed behavior deltas" and ADR Consequences.

#### Delta 1: `check_version` warning re-introduction

The current `_vendor.py:297-301` has a stripped 4-line `replace_llama()`
with a docstring noting the version-check loop was removed. This plan
re-introduces upstream's full 12-line `replace_llama()` plus the helper
`check_version()`.

Behavior delta:
- Inside the pinned `snap_kv` conda env (`transformers==4.37.2`): silent.
  The substring `'4.37'` is in `'4.37.2'`, so `warning_flag` stays
  `False` and no `warnings.warn(...)` fires.
- Outside that env (transformers != 4.37.x): one `UserWarning` printed to
  stderr per `replace_llama()` call:
  `Transformers version <X> might not be compatible with SnapKV. SnapKV
  is tested with Transformers version ['4.37'].`

Justification: vendoring fidelity. Future commit bumps become a clean
file-diff iff we keep upstream verbatim. Suppressing the warning is a
behavior change that future maintainers would have to re-apply on every
bump. Documented in ADR Decision 6 + VENDORING.md "Disclosed behavior
deltas" section.

#### Delta 2: SNAPKV_TRACE debug-print carry-forward

The prior `_vendor.py:85-86` had this block inside
`SnapKVCluster.update_kv()`:

```python
if os.environ.get("SNAPKV_TRACE") == "1":
    print(f"[snapkv] update_kv fired: q_len={q_len}, cap={self.max_capacity_prompt}", flush=True)
```

Upstream `snapkv_utils.py` does NOT contain this block and does NOT
`import os`. Iter-1 and iter-2 of this plan would have silently dropped
this block by doing a verbatim `cp` from upstream. Iter 3 makes this
explicit via edit #6 (§4.3 item 6) — the block is carried forward into
the vendored `snapkv_utils.py`, along with the matching `import os` at
the top.

Behavior delta vs upstream (the direction matters):
- With `SNAPKV_TRACE` unset or != "1" (the default): silent. No-op.
- With `SNAPKV_TRACE=1` in the environment: one print per
  `update_kv` call (i.e., per attention layer × per prefill step) to
  stdout. Useful for spot-checking whether SnapKV's compression
  branch fires.

Behavior delta vs prior `_vendor.py`: zero. The point of edit #6 is
parity with the prior local state so that diagnostic scripts and
documentation that rely on `SNAPKV_TRACE=1` continue to work after the
refactor.

Justification: the divergence from upstream is small, local, env-gated
(off by default), and useful enough that previous DCT-Page work
deliberately added it. Dropping it during a structural refactor would
be a silent regression. Documented in ADR Decision 8 + VENDORING.md
"Disclosed behavior deltas" section.

---

## 5. Task Flow

```
Step 1 (file moves + git-mv)         →
Step 2 (5 import / structural edits) →
Step 3 (init shim update)            →
Step 4 (LICENSE + VENDORING.md)      →
Step 4.5 (__pycache__ wipe pre-verify) →
Step 5 (smoke verify, 3 snippets)    →
Step 6 (delete _vendor.py)           →
Step 6.5 (__pycache__ wipe post-delete) →
Step 7 (run_ruler_snapkv.sh guard)
```

No parallelism opportunities here — every step depends on the previous one
landing first. Total estimated diff size: ~620 lines added (5 vendored
files + 2 doc files + 1 sh guard + 1 re-export `__init__.py`), ~300 lines
removed (`_vendor.py`), ~3 lines modified in the shim `__init__.py`, ~5
lines added at the top of `run_ruler_snapkv.sh`.

---

## 6. Detailed TODOs

### Step 1 — Create the `upstream/` package skeleton and copy files

Create the directory tree:

```
baselines/snap_kv/upstream/
├── __init__.py                                  (re-export, 4 lines)
├── LICENSE                                      (copied from /home/yoongonkim/SnapKV/LICENSE)
└── snapkv/
    ├── __init__.py                              (empty marker)
    └── monkeypatch/
        ├── __init__.py                          (empty marker)
        ├── snapkv_utils.py                      (copied verbatim from upstream)
        ├── llama_hijack_4_37.py                 (copied via git mv where applicable, see below)
        └── monkeypatch.py                       (copied verbatim from upstream)
```

**Git-blame preservation note (fix #11):** the existing
`_vendor.py` is a concatenation of three upstream files; ~196 of its 300
LOC are byte-for-byte from `llama_hijack_4_37.py`. To preserve
`git blame` continuity for the largest chunk, use `git mv` on the
upstream-clone path-substituted path **for `llama_hijack_4_37.py` only**:

```bash
SRC=/home/yoongonkim/SnapKV
DST=/home/yoongonkim/DCT-Page/baselines/snap_kv/upstream
mkdir -p "$DST/snapkv/monkeypatch"

# Largest single overlap (~196 LOC): preserve blame.
git mv baselines/snap_kv/_vendor.py "$DST/snapkv/monkeypatch/llama_hijack_4_37.py"
# Then overwrite the moved file with the upstream copy. Git tracks this
# as a rename + modify; blame survives for matching lines.
cp "$SRC/snapkv/monkeypatch/llama_hijack_4_37.py" "$DST/snapkv/monkeypatch/llama_hijack_4_37.py"

# Smaller files: plain copy + git add. Blame would not survive a rename
# anyway since the overlap is partial (~86 LOC for snapkv_utils, ~26 LOC
# for the unique parts of monkeypatch.py).
cp "$SRC/snapkv/monkeypatch/snapkv_utils.py"     "$DST/snapkv/monkeypatch/snapkv_utils.py"
cp "$SRC/snapkv/monkeypatch/monkeypatch.py"      "$DST/snapkv/monkeypatch/monkeypatch.py"
cp "$SRC/LICENSE"                                "$DST/LICENSE"

# Marker / re-export package files (filled in Step 2 for upstream/__init__.py).
: > "$DST/snapkv/__init__.py"
: > "$DST/snapkv/monkeypatch/__init__.py"
: > "$DST/__init__.py"   # placeholder; Step 2 writes the re-export body
```

After this Step, `_vendor.py` is gone from `baselines/snap_kv/` (it was
`git mv`'d), and the three upstream files have been dropped on top of the
moved path / new locations. The executor can sanity-check with
`git status` — it should show a `R` (rename) for `_vendor.py →
upstream/snapkv/monkeypatch/llama_hijack_4_37.py` plus added files.

Acceptance:
- `find baselines/snap_kv/upstream -name '*.py' | sort` lists exactly 6
  paths (3 marker / re-export `__init__.py` + 3 vendored modules).
- **After Step 1's `cp` but BEFORE Step 2's edit #6** is applied:
  `diff /home/yoongonkim/SnapKV/snapkv/monkeypatch/snapkv_utils.py
  baselines/snap_kv/upstream/snapkv/monkeypatch/snapkv_utils.py` is
  empty. **After Step 2's edit #6** the diff is the SNAPKV_TRACE block
  (one `import os` line + two-line `if os.environ.get(...) / print(...)`
  block); the diff is non-empty by design and is the only divergence
  from upstream `snapkv_utils.py`.
- `diff` of `llama_hijack_4_37.py` and `monkeypatch.py` is empty at this
  step (edits #1–4 applied in Step 2).
- `git status -s` shows `R  baselines/snap_kv/_vendor.py →
  baselines/snap_kv/upstream/snapkv/monkeypatch/llama_hijack_4_37.py`.

### Step 2 — Apply the five whitelisted edits

In `baselines/snap_kv/upstream/snapkv/monkeypatch/llama_hijack_4_37.py`:
- **Edit #1:** Replace line 14 (only this single line):
  `from snapkv.monkeypatch.snapkv_utils import init_snapkv`
  → `from .snapkv_utils import init_snapkv`

In `baselines/snap_kv/upstream/snapkv/monkeypatch/monkeypatch.py`:
- **Edit #2 (line 4 — rewrite, NOT drop):** Replace the upstream Llama
  hijack import line:
  `from snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37`
  → `from .llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37`
- **Edit #3 (lines 5 and 6 — drop):** Delete the two upstream import
  lines that pull from `snapkv.monkeypatch.mistral_hijack_4_37` and
  `snapkv.monkeypatch.mixtral_hijack_4_37` (the files are not vendored).
- **Edit #4 (`replace_mistral` / `replace_mixtral` bodies — delete):**
  Delete the `replace_mistral()` and `replace_mixtral()` function
  definitions in full (Llama-only scope; no shim callers exist).
- **Keep `check_version()` (upstream lines 8–13)** byte-identical; it is
  called by `replace_llama()` (lines 15–26).
- **Keep `replace_llama()`** byte-identical, including its
  `version_list = ['4.37']` / `warnings.warn(...)` block. (See §4.4.)

In `baselines/snap_kv/upstream/__init__.py`:
- **Edit #5:** Write the re-export body (4 content lines + blank line +
  `__all__`):
  ```python
  from .snapkv.monkeypatch.snapkv_utils import init_snapkv
  from .snapkv.monkeypatch.monkeypatch import replace_llama

  __all__ = ["init_snapkv", "replace_llama"]
  ```

In `baselines/snap_kv/upstream/snapkv/monkeypatch/snapkv_utils.py`:
- **Edit #6:** Carry forward the SNAPKV_TRACE debug print from the prior
  `_vendor.py:85-86`. Two sub-edits:
  1. Add `import os` as a new line immediately after the existing
     `import math` (line 6 upstream). The import block becomes:
     ```python
     import torch
     import time
     import torch.nn.functional as F
     import torch.nn as nn
     import math
     import os
     ```
  2. Inside `SnapKVCluster.update_kv()`, find the line
     `value_states = torch.cat([v_past_compress, v_cur], dim = 2)`
     (line 69 upstream — the second-to-last line of the else-branch
     body). Immediately after it, before `return key_states,
     value_states`, insert (matching the else-branch's 12-space indent):
     ```python
             if os.environ.get("SNAPKV_TRACE") == "1":
                 print(f"[snapkv] update_kv fired: q_len={q_len}, cap={self.max_capacity_prompt}", flush=True)
     ```
  Result: upstream's 86-line `snapkv_utils.py` becomes a ~89-line file
  with exactly the 3-line delta enumerated above. No other content
  changes.

`upstream/snapkv/__init__.py` and `upstream/snapkv/monkeypatch/__init__.py`
remain empty (zero-byte marker files).

Acceptance:
- `python -c "import ast,
  pathlib;
  ast.parse(pathlib.Path('baselines/snap_kv/upstream/snapkv/monkeypatch/llama_hijack_4_37.py').read_text())"`
  exits 0 (file parses).
- Same `ast.parse` check on `monkeypatch.py` exits 0.
- Same on `upstream/__init__.py` exits 0.
- `grep -nE 'from snapkv\.' baselines/snap_kv/upstream/snapkv/monkeypatch/`
  returns zero matches (every absolute intra-package import is gone).
- `grep -nE '^def replace_(mistral|mixtral)' baselines/snap_kv/upstream/snapkv/monkeypatch/monkeypatch.py`
  returns zero matches.
- `grep -nE '^def replace_llama' baselines/snap_kv/upstream/snapkv/monkeypatch/monkeypatch.py`
  returns exactly one match.
- `grep -nE '^def check_version' baselines/snap_kv/upstream/snapkv/monkeypatch/monkeypatch.py`
  returns exactly one match (vendoring fidelity, see §4.4 Delta 1).
- **Edit #6 landing checks (Fix #2):**
  - `grep -cE '^import os$' baselines/snap_kv/upstream/snapkv/monkeypatch/snapkv_utils.py`
    returns exactly `1` (the new `import os` added by edit #6).
  - `grep -c 'SNAPKV_TRACE' baselines/snap_kv/upstream/snapkv/monkeypatch/snapkv_utils.py`
    returns exactly `1` (the env-gated debug print body).
  - `python -c "import ast,
    pathlib;
    ast.parse(pathlib.Path('baselines/snap_kv/upstream/snapkv/monkeypatch/snapkv_utils.py').read_text())"`
    exits 0 (file parses after the insertion — indentation correct).

### Step 3 — Rewrite `baselines/snap_kv/__init__.py` import surface

Replace exactly one import line at the top of `__init__.py`:

- Old:
  `from ._vendor import init_snapkv as _init_snapkv, replace_llama as _replace_llama`
- New (infllm-style indirection through `upstream/__init__.py`):
  ```python
  from .upstream import init_snapkv as _init_snapkv, replace_llama as _replace_llama
  ```

Do **not** touch any other line in `__init__.py`. The functions
`assert_llama_only`, `load_llama_config_stripped_rope`, `init_snap_kv`, and
the `__all__` list remain byte-identical.

Acceptance:
- `git diff baselines/snap_kv/__init__.py` shows exactly the one-line
  removal + one-line addition above and nothing else.
- `python -c "import ast,
  pathlib;
  ast.parse(pathlib.Path('baselines/snap_kv/__init__.py').read_text())"` exits 0.

### Step 4 — Add `VENDORING.md`

Create `baselines/snap_kv/VENDORING.md` with the following content
(`%%upstream-commit%%` is a literal in this plan; the executor writes the
real hash):

```
# SnapKV — vendoring record

Upstream:        https://github.com/FasterDecoding/SnapKV
Upstream commit: e216ddc84c5bd210378cbdbbba12ba02102aa640
License:         Apache 2.0  (see upstream/LICENSE)
Vendored on:     2026-05-11

## Files vendored

- upstream/snapkv/monkeypatch/snapkv_utils.py        (1 import + 2-line SNAPKV_TRACE block carried forward; see edit #6 below)
- upstream/snapkv/monkeypatch/llama_hijack_4_37.py   (1-line import edit; see below)
- upstream/snapkv/monkeypatch/monkeypatch.py         (Mistral / Mixtral removed, imports made relative; see below)

## Files NOT vendored (intentionally)

- snapkv/monkeypatch/mistral_hijack_4_37.py
- snapkv/monkeypatch/mixtral_hijack_4_37.py
  Rationale: the wrapper is Llama-only (see baselines/snap_kv/__init__.py:
  assert_llama_only). If SnapKV-on-Mistral or SnapKV-on-Mixtral
  comparisons are ever wanted, drop the upstream files in place under
  upstream/snapkv/monkeypatch/, re-add the matching imports to
  monkeypatch.py, and add a re-export line to upstream/__init__.py.

## Edits applied to vendored files (6 total)

1. llama_hijack_4_37.py:14
   - from snapkv.monkeypatch.snapkv_utils import init_snapkv
   + from .snapkv_utils import init_snapkv

2. monkeypatch.py:4
   - from snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37
   + from .llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37

3. monkeypatch.py:5-6
   - dropped: `from snapkv.monkeypatch.mistral_hijack_4_37 import ...`
   - dropped: `from snapkv.monkeypatch.mixtral_hijack_4_37 import ...`

4. monkeypatch.py
   - dropped: def replace_mistral(...) and def replace_mixtral(...) bodies
     (Llama-only scope; their bodies reference the dropped imports above)

5. Implicit→regular package conversion (3 new __init__.py files; PEP 420
   conversion):
   - Added: upstream/snapkv/__init__.py             (empty marker)
   - Added: upstream/snapkv/monkeypatch/__init__.py (empty marker)
   - Added: upstream/__init__.py                    (re-export of
                                                     init_snapkv and
                                                     replace_llama)
   Rationale: upstream is an implicit-namespace package (PEP 420). The
   vendoring uses regular packages so the relative imports inside
   llama_hijack_4_37.py and monkeypatch.py are robust against a future
   `pip install -e /home/yoongonkim/SnapKV` shadowing the `snapkv` name
   on sys.path, and so baselines/snap_kv/__init__.py can mirror the
   infllm `from .upstream import …` pattern.

6. snapkv_utils.py — SNAPKV_TRACE carry-forward from prior _vendor.py:85-86
   + `import os`
   - Added: `import os` to the top-of-file import block (after
     `import math`).
   - Inserted inside SnapKVCluster.update_kv() (else-branch body, just
     before the final return):
     ```
     if os.environ.get("SNAPKV_TRACE") == "1":
         print(f"[snapkv] update_kv fired: q_len={q_len}, cap={self.max_capacity_prompt}", flush=True)
     ```
   Rationale: SNAPKV_TRACE=1 is an env-gated debug print previously
   present in _vendor.py; preserving it avoids a silent regression
   for diagnostic scripts and existing documentation that rely on it.

## Disclosed behavior deltas (two)

Two acknowledged deltas exist in opposite directions:

- **Delta 1 (current→upstream-restored):** replace_llama() now re-runs
  upstream's `check_version()` + `warnings.warn()` block. Inside the
  pinned snap_kv env (transformers==4.37.2) this is silent; outside it
  emits one UserWarning per replace_llama() call. The prior _vendor.py
  had this stripped. Rationale: vendoring fidelity makes future
  commit-bumps clean diffs.

- **Delta 2 (current→preserved-forward):** snapkv_utils.py keeps the
  SNAPKV_TRACE env-gated debug print from prior _vendor.py:85-86
  (NOT present upstream). With SNAPKV_TRACE=1, one print per
  update_kv() call is emitted to stdout. Default is silent. Rationale:
  diagnostic scripts and documentation that rely on SNAPKV_TRACE=1
  continue to work after the refactor.

## Re-vendoring procedure

    BUMP=<new commit>
    rsync -a --delete /path/to/SnapKV/snapkv/monkeypatch/{snapkv_utils,llama_hijack_4_37,monkeypatch}.py \
        baselines/snap_kv/upstream/snapkv/monkeypatch/
    # then re-apply the 6 edits above (or merge them via patch).
    # In particular: edit #6 (SNAPKV_TRACE) MUST be reapplied to
    # snapkv_utils.py — the rsync overwrite drops it by default.
    # bump the "Upstream commit" line above
    # re-run Step 5 smoke verifications (in this plan).
```

Acceptance:
- File exists at `baselines/snap_kv/VENDORING.md`.
- Contains literal string `e216ddc84c5bd210378cbdbbba12ba02102aa640`.
- Lists exactly the 3 vendored Python files.
- "Edits applied to vendored files (6 total)" section enumerates 1–6.
- "Disclosed behavior deltas (two)" section names BOTH `check_version` /
  `warnings.warn` (Delta 1) AND `SNAPKV_TRACE` (Delta 2).

### Step 4.5 — `__pycache__` wipe (pre-verify)

Stale `.pyc` files (notably `baselines/snap_kv/__pycache__/_vendor.cpython-311.pyc`,
14904 bytes — verified to exist) can mask import-path bugs in Step 5 by
satisfying imports from the cached bytecode of a file that no longer
exists. Wipe before verifying:

```bash
find /home/yoongonkim/DCT-Page/baselines/snap_kv -name __pycache__ -type d -exec rm -rf {} +
```

Acceptance:
- `find /home/yoongonkim/DCT-Page/baselines/snap_kv -name __pycache__ -type
  d` returns no output.

### Step 5 — Smoke verify (three snippets)

#### Snippet A — public-surface import smoke + transformers version check

```bash
cd /home/yoongonkim/DCT-Page
conda run -n snap_kv python -c "
from baselines.snap_kv import init_snap_kv, assert_llama_only, load_llama_config_stripped_rope
from baselines.snap_kv.config import SNAPKV_CONFIG
# Fix #8: prove transformers 4.37.x is on path before the patch-identity step.
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
print('imports OK')
assert_llama_only('meta-llama/Llama-3.1-8B-Instruct')
print('assert_llama_only OK')
"
```

Expected: prints `imports OK` then `assert_llama_only OK` and exits 0.
Failure mode (transformers != 4.37.x): the
`from transformers.models.llama.modeling_llama import repeat_kv,
apply_rotary_pos_emb` line raises a clear `ImportError` before any
patch-identity testing — i.e. wrong env detected up front, not mid-test.

#### Snippet B — `upstream/` re-export resolves cleanly (Fix #4)

Surface circular-import or re-export regressions before testing the
monkey-patch path, so the failure label points at `upstream/__init__.py`
rather than masquerading as a public-surface import error in Snippet C.

```bash
cd /home/yoongonkim/DCT-Page
conda run -n snap_kv python -c "
import baselines.snap_kv.upstream as u
print('upstream re-export resolves:', getattr(u, '__all__', '(no __all__)'))
# Edit #5 wrote __all__ = ['init_snapkv', 'replace_llama']; assert that
# both names are bound on the upstream package module object.
assert hasattr(u, 'init_snapkv'), 'upstream.init_snapkv missing'
assert hasattr(u, 'replace_llama'), 'upstream.replace_llama missing'
print('upstream symbols OK')
"
```

Expected: prints
`upstream re-export resolves: ['init_snapkv', 'replace_llama']`
then `upstream symbols OK` and exits 0.

#### Snippet C — patch identity + module provenance

```bash
conda run -n snap_kv python -c "
import transformers
from transformers.models.llama.modeling_llama import LlamaFlashAttention2
from baselines.snap_kv.upstream.snapkv.monkeypatch.monkeypatch import replace_llama
from baselines.snap_kv.upstream.snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward
before = LlamaFlashAttention2.forward
replace_llama()
after  = LlamaFlashAttention2.forward
assert before is not after, 'replace_llama did not patch'
assert after is llama_flash_attn2_forward, 'patched function is not the vendored one'
# Fix #4: decouple the test snippet's import from the production patch
# resolution. The patched function MUST come from inside our vendored tree.
assert llama_flash_attn2_forward.__module__.endswith(
    'upstream.snapkv.monkeypatch.llama_hijack_4_37'
), llama_flash_attn2_forward.__module__
print('patch identity OK')
"
```

Expected: prints `patch identity OK` and exits 0. Inside the pinned
`snap_kv` env, no UserWarning fires; outside it, one UserWarning fires
during `replace_llama()` per §4.4 Delta 1 — non-fatal.

#### Snippet D — non-Llama assert fails fast

```bash
conda run -n snap_kv python -c "
from baselines.snap_kv import assert_llama_only
try:
    assert_llama_only('Qwen/Qwen3-8B')
except ValueError as e:
    print('non-Llama refused:', e); raise SystemExit(0)
raise SystemExit('assert_llama_only failed to refuse Qwen3')
"
```

Expected: prints `non-Llama refused: SnapKV baseline wrapper only supports
Llama models …` and exits 0.

Acceptance:
- Snippet A exits 0 with both expected prints (`imports OK`,
  `assert_llama_only OK`).
- Snippet B exits 0 with `upstream re-export resolves: …` and
  `upstream symbols OK`.
- Snippet C exits 0 with `patch identity OK`.
- Snippet D exits 0 with `non-Llama refused:` prefix.

### Step 6 — Delete `_vendor.py`

Note: Step 1's `git mv _vendor.py upstream/snapkv/monkeypatch/llama_hijack_4_37.py`
already removes `_vendor.py` from `baselines/snap_kv/`. This Step is a
**defensive check + grep sweep** rather than an additional rm, to catch
the case where an executor took the `git add` path on iter-1 thinking
instead of the `git mv` path.

```bash
cd /home/yoongonkim/DCT-Page

# Defensive: if _vendor.py somehow still exists (e.g. executor used `git
# add` instead of `git mv` in Step 1), remove it now.
[ -f baselines/snap_kv/_vendor.py ] && git rm baselines/snap_kv/_vendor.py

# Fix #10: broaden grep scope to the whole tree, excluding markdown and
# .omc/ planning docs (which legitimately reference the old name).
git grep -n '_vendor' -- ':!*.md' ':!.omc/' || echo 'no _vendor references in code'
```

Acceptance:
- `ls baselines/snap_kv/_vendor.py 2>&1` returns "No such file or
  directory".
- `git grep -n '_vendor' -- ':!*.md' ':!.omc/'` returns zero matches
  (i.e. no code path imports `_vendor` anywhere in the tree).
- Step 5 snippets A, B, C, D all still pass after the delete + cache
  wipe in Step 6.5.

**Rollback (fix #5 + fix #8):** if any of Step 5's snippets B, C, or D
have failed and the executor has reached this Step, **DO NOT proceed**.
Run `git restore --source=HEAD --staged --worktree baselines/snap_kv/`
to revert all of Steps 1–4. Investigate the verify failure before
re-trying the plan from Step 1. Step 6 is irreversible without a
`git reflog` walk because it removes the only on-disk copy of the
concatenated `_vendor.py`.

**IMPORTANT (fix #8):** do NOT `git commit` between Step 1 and the
completion of Step 6.5 acceptance. The rollback procedure depends on
the changes being staged/worktree-only — once committed, `git restore
--source=HEAD …` is a no-op and recovery requires a `git reset --hard`
to the pre-plan commit (which is destructive and out of policy without
explicit user confirmation). A single squashed commit at the end of
Step 7 is the correct posture.

### Step 6.5 — `__pycache__` wipe (post-delete)

Wipe again after `_vendor.py` is removed, so that any subsequent imports
in the same Python process / pytest run cannot resurrect bytecode from
the deleted source:

```bash
find /home/yoongonkim/DCT-Page/baselines/snap_kv -name __pycache__ -type d -exec rm -rf {} +
```

Then re-run Snippet C from Step 5 once more (post-renumber: Snippet C
is the patch-identity + module-provenance check). Expected: still passes.

Acceptance:
- `find /home/yoongonkim/DCT-Page/baselines/snap_kv -name __pycache__ -type
  d` returns no output.
- Snippet C (post-cache-wipe re-run) exits 0 with `patch identity OK`.

### Step 7 — Guard `run_ruler_snapkv.sh` against argparse failure

`eval_ruler.py` still does not accept `--mode snap_kv` after this plan
lands (deferred per §7). Without a guard, running
`bash run_ruler_snapkv.sh` produces a confusing argparse stack trace
that does not point at the vendoring plan. Add a guard right after the
conda activation block (immediately before the `BASE_MODEL=` line so
env vars set on the command line still pass through to the eventual
real script):

Insert at the appropriate location in `run_ruler_snapkv.sh` (after
`conda activate "$SNAPKV_ENV_NAME"` and before `BASE_MODEL=...`):

```bash
# --- snap_kv mode not yet wired into eval_ruler.py ---
# This vendoring refactor (.omc/plans/snapkv_vendor_upstream_dir.md) only
# refactored baselines/snap_kv/. The eval_ruler.py argparse `choices`
# for `--mode` does NOT yet include "snap_kv". The wireup ships
# separately as .omc/plans/snapkv_eval_ruler_wireup.md.
#
# The regex below is anchored specifically to `add_argument("--mode", …
# choices=[…])` rather than the first `choices=[…]` literal in the file,
# because eval_ruler.py has 11 separate `choices=[…]` literals (one per
# CLI flag) and matching the first would produce false positives the
# moment any earlier flag's choices list happened to contain the string
# 'snap_kv'.
if ! python -c "
import re, pathlib
src = pathlib.Path('eval_ruler.py').read_text()
m = re.search(r'add_argument\(\s*[\"\\']--mode[\"\\'][^)]*choices\s*=\s*\[([^]]*)\]', src, re.S)
import sys
sys.exit(0 if (m and 'snap_kv' in m.group(1)) else 1)
"; then
    echo 'ERROR: --mode snap_kv is not yet wired into eval_ruler.py.'
    echo 'See .omc/plans/snapkv_eval_ruler_wireup.md for follow-up.'
    exit 2
fi
```

Rationale: fail fast with an actionable message. Once the wireup plan
lands and adds `snap_kv` to the argparse `choices`, this guard
transparently becomes a no-op and can be removed in that plan's diff.

Acceptance:
- `bash run_ruler_snapkv.sh` (with no `eval_ruler.py` edits in place)
  prints `ERROR: --mode snap_kv is not yet wired into eval_ruler.py.`
  followed by the wireup-plan pointer, then exits 2.
- `git diff run_ruler_snapkv.sh` shows ONLY the inserted guard block —
  the conda activation, env-var configuration, sweep loop, and the
  `python eval_ruler.py …` invocation are byte-identical.

---

## 7. Out of scope (deferred to follow-up plan)

- **Wiring `--mode snap_kv` into `eval_ruler.py`.** This is broken today
  (`run_ruler_snapkv.sh` will fail at argparse), and **after this plan
  lands it remains broken — by design**. The vendoring refactor and the
  argparse wireup are separate reviewable diffs. Fixing the wireup
  requires decisions outside the vendoring refactor: argparse `choices`
  extension, argparse flag wiring for `window_size` /
  `max_capacity_prompt` / `kernel_size` / `pooling`, dispatch branch that
  calls `assert_llama_only` + `load_llama_config_stripped_rope` +
  `init_snap_kv`, and parity with the existing `inf_llm` mode's call
  pattern. Follow-up plan: `snapkv_eval_ruler_wireup.md`. **The end-to-end
  run remains broken until that ships; Step 7's guard makes that explicit.**
- **Mistral / Mixtral support.** YAGNI per Principle 2; if added later, drop
  the upstream files into `upstream/snapkv/monkeypatch/`, add their
  imports back to `monkeypatch.py`, add the relevant re-exports to
  `upstream/__init__.py`, and update the shim's assert. The vendoring
  layout chosen here makes this a one-step change.
- **LongBench v1 / v2 `--mode snap_kv` wiring.** Same family of changes as
  the RULER wireup; bundle with that follow-up.
- **Config refactor.** `SNAPKV_CONFIG` is a flat dict today. Leave it.

---

## 8. Decisions (resolutions to the planner's open questions)

1. **Drop `_vendor.py` outright (no compatibility re-export).**
   Rationale: it has exactly one import site
   (`baselines/snap_kv/__init__.py`) which Step 3 rewrites. A
   compatibility shim would carry dead code indefinitely. Verified with
   `git grep -n '_vendor' -- ':!*.md' ':!.omc/'` — only the one __init__
   site exists today. Acceptance check in Step 6 makes this explicit.
2. **Preserve upstream filenames inside `upstream/snapkv/monkeypatch/`.**
   The whole point of Option A is bumpability; renaming defeats it.
3. **Use relative imports inside upstream files, not `sys.path` hacks.**
   The infllm precedent doesn't do `sys.path`; we don't either. Relative
   imports require empty `__init__.py` markers at two levels — small
   price.
4. **Llama-only.** Match the existing `assert_llama_only` guard. Mistral /
   Mixtral upstream files are not copied. `replace_mistral()` /
   `replace_mixtral()` are deleted from the vendored `monkeypatch.py`
   (their import lines would otherwise fail, since the supporting files
   are absent).
5. **Defer `eval_ruler.py` wiring; guard the run script instead.** The
   user's verbatim request is about vendoring, not mode wiring. Mixing
   them risks scope creep and increases the diff's review surface. Step
   7 adds a top-of-script guard to `run_ruler_snapkv.sh` so executors
   running it see a clear pointer to the follow-up plan rather than an
   argparse stack trace. The follow-up plan is named in §7.
6. **Mirror infllm's `upstream/__init__.py` re-export pattern (fix #1
   option a).** `upstream/__init__.py` re-exports `init_snapkv` and
   `replace_llama` so `baselines/snap_kv/__init__.py` can say
   `from .upstream import …` verbatim, exactly like infllm's
   `from .upstream import patch_hf`. This is the honest precedent-match.
7. **Re-introduce upstream's `check_version` warning (iter-2 fix #2).**
   Vendoring fidelity wins over the iter-1 strip-and-pretend posture.
   Silent in the pinned `snap_kv` env; one UserWarning per
   `replace_llama()` call elsewhere. Documented in §4.4 Delta 1 +
   VENDORING.md.
8. **Carry forward the SNAPKV_TRACE debug print (iter-3 fix #1).** The
   prior `_vendor.py:85-86` had a local-only env-gated
   `if os.environ.get("SNAPKV_TRACE") == "1": print(...)` block inside
   `SnapKVCluster.update_kv()`. Upstream `snapkv_utils.py` does NOT.
   Iter-1 and iter-2 would have silently dropped it via verbatim `cp`.
   Iter 3 makes this edit #6 explicit and disclosed: the block is
   preserved, the matching `import os` is added at the top of
   `snapkv_utils.py`, and both are recorded in VENDORING.md "Edits
   applied" + "Disclosed behavior deltas" sections. This is the
   inverse-direction counterpart of Decision 7: there we restored
   upstream behavior that prior `_vendor.py` had stripped; here we
   preserve `_vendor.py` behavior that upstream lacks. Documented in
   §4.4 Delta 2 + VENDORING.md.

---

## 9. Success Criteria (whole-plan)

- `git status` after the plan shows changes to:
  - `baselines/snap_kv/__init__.py`  (1 line removed, 1 added)
  - `baselines/snap_kv/_vendor.py`   (renamed via git mv → upstream/snapkv/monkeypatch/llama_hijack_4_37.py, then overwritten)
  - `baselines/snap_kv/upstream/__init__.py`                       (new, 4 content lines: re-export of init_snapkv + replace_llama)
  - `baselines/snap_kv/upstream/LICENSE`                           (new, verbatim Apache-2.0 from /home/yoongonkim/SnapKV/LICENSE)
  - `baselines/snap_kv/upstream/snapkv/__init__.py`                (new, empty marker)
  - `baselines/snap_kv/upstream/snapkv/monkeypatch/__init__.py`    (new, empty marker)
  - `baselines/snap_kv/upstream/snapkv/monkeypatch/snapkv_utils.py`     (new, upstream + edit #6 SNAPKV_TRACE)
  - `baselines/snap_kv/upstream/snapkv/monkeypatch/llama_hijack_4_37.py` (rename target of _vendor.py, then upstream + edit #1)
  - `baselines/snap_kv/upstream/snapkv/monkeypatch/monkeypatch.py`      (new, upstream + edits #2/#3/#4)
  - `baselines/snap_kv/VENDORING.md` (new file, 6-edit disclosure + 2-delta disclosure)
  - `run_ruler_snapkv.sh`            (top-of-script guard added; rest byte-identical)
- Step 5 snippets A, B, C, D all pass under `conda run -n snap_kv`.
- Step 6.5 post-cache-wipe re-run of Snippet C passes.
- Step 7 acceptance: `bash run_ruler_snapkv.sh` exits 2 with the ERROR
  block.
- No edits to `config.py`, `eval_ruler.py`, `eval_longbench_v{1,2}.py`,
  `oracle/*`, `speed/*`, or any other baseline.
- `diff -r baselines/snap_kv/upstream/snapkv/monkeypatch/
  /home/yoongonkim/SnapKV/snapkv/monkeypatch/` shows only:
  - the documented in-file edits:
    - edit #1 — 1 line in llama_hijack_4_37.py
    - edits #2, #3, #4 — 3 changes in monkeypatch.py
    - edit #6 — 1 `import os` line + 2-line SNAPKV_TRACE block in
      snapkv_utils.py (previously listed as "verbatim" — corrected
      in iter 3)
  - `mistral_hijack_4_37.py only in /home/yoongonkim/SnapKV/...`
  - `mixtral_hijack_4_37.py only in /home/yoongonkim/SnapKV/...`
  No other differences. (Edit #5, the regular-package conversion, lives
  in the parent `upstream/` not under `upstream/snapkv/monkeypatch/`.)

---

## 10. ADR

### Decision
Refactor `baselines/snap_kv/` from a flat concatenated `_vendor.py` to an
`upstream/snapkv/monkeypatch/`-style layout that mirrors the SnapKV
upstream package, matching the existing `baselines/infllm/upstream/`
precedent end-to-end (both layout and the `upstream/__init__.py`
re-export indirection). Delete `_vendor.py`. Add LICENSE + VENDORING.md.
Guard `run_ruler_snapkv.sh` until the follow-up wireup plan lands.

### Drivers
1. Diff-against-upstream ergonomics for future commit bumps.
2. Consistency with the in-repo `baselines/infllm/upstream/` convention.
3. Low refactor risk (mechanical, no SnapKV-logic edits, easy
   verification with three smoke snippets).

### Alternatives considered
- **Option B (keep flat `_vendor.py` + cleanup):** invalidated because it
  optimizes the one-time refactor cost over the recurring commit-bump
  cost, and diverges from the in-repo convention.
- **Option C (no-op, just explain):** invalidated because it does not
  satisfy the user's stated request and leaves the license-attribution
  gap in place.

### Why chosen
Option A is the only choice that aligns with all four principles. The cost
is a single mechanical refactor; the benefit compounds every time SnapKV
is bumped or a sibling baseline is added.

### Consequences
- Future SnapKV commit bumps become a `rsync + 5-edit patch` operation
  instead of a manual re-concatenation.
- Two `baselines/*/upstream/`-style trees now exist (infllm + snap_kv),
  establishing the convention. Both use the same
  `upstream/__init__.py`-re-export indirection — future vendored
  baselines should follow suit.
- Three additional `__init__.py` files in the tree: two empty markers
  and one re-export. Net effect: regular packages instead of
  implicit-namespace, documented as edit #5 in VENDORING.md.
- The `_vendor.py` filename is gone forever; if any external script we
  haven't accounted for imports it, that script breaks. Step 6 grep
  (broadened to `git grep -n '_vendor' -- ':!*.md' ':!.omc/'`) verifies
  no such importer exists in code.
- **Two disclosed behavior deltas, in opposite directions** (iter-3
  symmetry fix). See §4.4 and VENDORING.md "Disclosed behavior deltas"
  for full rationale.
  - **Delta 1 (current→upstream-restored, ADR Decision 7):**
    re-introduces upstream's transformers-version warning
    (`check_version()` + `warnings.warn(...)` block inside
    `replace_llama()`). Silent in the pinned `snap_kv` env
    (`transformers==4.37.2`); informative under any other version.
  - **Delta 2 (current→preserved-forward, ADR Decision 8):** preserves
    the `_vendor.py:85-86` SNAPKV_TRACE env-gated debug print inside
    `SnapKVCluster.update_kv()` (and its `import os`). Silent unless
    `SNAPKV_TRACE=1`. Not present upstream; without this carry-forward,
    diagnostic scripts that rely on `SNAPKV_TRACE=1` silently regress.

### Follow-ups
- `snapkv_eval_ruler_wireup.md` — add `--mode snap_kv` to `eval_ruler.py`,
  wire argparse flags, and add the dispatch branch. **Required** for
  `run_ruler_snapkv.sh` to work end-to-end. When that plan lands, the
  top-of-script guard added by Step 7 becomes a no-op and should be
  removed by that plan's diff.
- `snapkv_longbench_wireup.md` — same wireup for
  `eval_longbench_v{1,2}.py` (only if SnapKV is actually planned for
  LongBench comparisons; otherwise skip).
- Optional future: vendor `mistral_hijack_4_37.py` /
  `mixtral_hijack_4_37.py` if SnapKV-on-Mistral or SnapKV-on-Mixtral
  comparisons are wanted. Layout already supports this trivially: drop
  the upstream files in, add their imports back to `monkeypatch.py`, add
  their re-exports to `upstream/__init__.py`, and loosen the shim's
  `assert_llama_only`.

---

## 11. Notes for Architect / Critic re-review (iter 2)

- **All 11 fix items applied.** See "Iteration 2 changelog" at top.
- **Verification was actually done before drafting iter 1.** `pip show
  snapkv` was run (returns nothing); `baselines/snap_kv/__init__.py` was
  read end-to-end (`from ._vendor import …` confirmed at line 14);
  upstream layout and imports were grepped (`from
  snapkv.monkeypatch.snapkv_utils import init_snapkv` at
  `llama_hijack_4_37.py:14`); eval_ruler.py argparse `choices` was read
  (line 99–104, no `snap_kv`); commit hash was cross-checked against
  `_vendor.py`'s docstring (both say `e216ddc…`).
- **Iter 2 re-verifications done after the architect / critic verdicts:**
  upstream `monkeypatch.py` was re-read end-to-end so §4.3 item 2 anchors
  precisely to lines 4 (rewrite), 5–6 (drop), 8–13 (`check_version`,
  keep), and 15–26 (`replace_llama`, keep including the warning block).
  `baselines/infllm/upstream/__init__.py` was re-read (28 bytes,
  `from .utils import patch_hf` — confirmed re-export, NOT empty).
- **The user's statement "we are using pip install snapkv method" is
  factually wrong about the current state.** Whether to call this out to
  the user in the planner-confirmation message is a UX call, not a plan
  call. Recommendation: yes — one sentence in the confirmation summary
  ("note: snap_kv is already vendored as `_vendor.py`, not pip-installed;
  this plan reorganizes that vendoring") so the user can short-circuit
  the work if their actual concern was the integration breakage and not
  the vendoring layout. The added Step 7 (run-script guard) partially
  hedges that risk.
- **Critic should probe (still open after iter 2):** Step 7's guard
  uses a regex on `eval_ruler.py` source to detect `snap_kv` in the
  argparse `choices` literal. The regex is conservative (matches the
  first `choices=[...]` literal), which is correct for the current file
  (line 99) but is structurally brittle. Alternative: import
  `eval_ruler` and inspect the argparse `parser._actions` — but that
  requires running the script as a module with no side effects, which
  the current `eval_ruler.py` does not support (it has imports at module
  scope that fail without the right env). The regex is the simpler
  pragmatic choice; if Critic disagrees, the alternative is
  `python -c "import eval_ruler" && python -c "from eval_ruler import
  build_parser; …"` — but `eval_ruler.py` does not currently expose a
  `build_parser()` function. Defer.
- **Architect should probe (still open after iter 2):** the
  `check_version` re-introduction is justified on fidelity grounds, but
  if the snap_kv env's transformers ever drifts off 4.37.x (e.g. to
  4.37.3 — still matches the substring; or 4.38.x — does NOT match), the
  warning will start firing in production. Recommendation if this is a
  concern: bump the snap_kv conda env's transformers pin to an exact
  version in `requirements_snap_kv.txt` (out of scope for this plan;
  belongs to the env-management track).
