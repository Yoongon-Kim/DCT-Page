# Plan: Redefine `--top_k` as TOTAL Selected Page Budget in Eval Scripts (v2)

**Mode:** RALPLAN consensus (SHORT). Eval-script-only, semantics-preserving for `dct_page_attention.py`. Plus a one-sentence in-scope edit to `CLAUDE.md` (Conventions section) for cross-script semantic asymmetry.
**Scope:** `eval_ruler.py`, `eval_longbench_v1.py`, `eval_longbench_v2.py`, `eval_aime25.py`, `eval_gpqa.py`, `CLAUDE.md` (one-line addition under Conventions). Read-only planning — no source edits in this PR.

---

## RALPLAN-DR Summary (v2)

### Principles (P1–P5)

- **P1 — Eval-surface change only.** `DCTPageConfig.top_k` semantics ("middle pages selected by score+topk") stays untouched. Only what the *eval scripts* pass into it changes. This keeps `dct_page_attention.py`, the kernels, and the oracle scripts free of churn.
- **P2 — Total-budget = sink + middle + recent (not + open page).** New `--top_k` is the count of *full pages* user-visible in the attended KV: `num_sink_pages + middle_top_k + num_recent_pages`. The currently-open partial page is **not** counted (it lives inside `num_recent_pages` per `dct_page_attention.py:1155`).
- **P3 — Single derivation point per script, mode-gated.** Compute `middle_top_k = args.top_k - num_sink_pages - num_recent_pages` exactly once, *only when `args.mode == "page_attention"`*. Quest, Multipole, Seer, Duo, ShadowKV, InfLLM keep their existing `args.top_k` interpretation untouched.
- **P4 — Run-name + summary fields preserve user intent (total) with a `T` marker on the page-attention path.** For page_attention, run_name encodes the new total budget as `topk{N}T` so old (`topk{N}`) and new directories never collide. `summary["top_k"]` continues to report the user-typed total; `summary["middle_top_k"]` is added for traceability.
- **P5 — Cross-script semantic asymmetry must be documented inline.** The plan adds one explicit sentence to `CLAUDE.md` Conventions calling out that `--top_k` in `eval_*.py` (page_attention) means TOTAL, while `--dct_top_k` in `oracle/*` keeps meaning MIDDLE. Without this note, the asymmetry is a future-debugger trap.

### Decision Drivers (top 3)

1. **D1 — Backward-compat for sweep scripts.** `run_ruler.sh` / `run_longbench_v*.sh` / `run_ruler_quest.sh` pass `--top_k 64` etc. assuming the OLD semantics. Any change here silently shrinks the actual page budget for every reproducible sweep we have on disk. Specifically `run_longbench_v1.sh:60` and `run_longbench_v2.sh:52` hardcode `topk${TOP_K}` in their `--run_name`, so the plan must change run_name format AND flag those bash scripts in F1. → **Must be detectable, opt-in to old behavior, and run_name must be DISJOINT from old.**
2. **D2 — Quest reuses `args.top_k` as `page_budget`.** `eval_ruler.py:433` and `:448`/`:772-773` use `args.top_k` *as-is* as Quest's per-decode page budget. Quest has no sink/recent. Subtracting would silently change Quest's effective budget by `+sink+recent` (at default = 6 pages → 192 tokens at page_size=32). → **Must NOT subtract for `quest_attention` mode.**
3. **D3 — Result reproducibility & file-name continuity.** `summary["top_k"]` and `run_name` (e.g. `topk64`) are the join key for plots, the dedup key for `--skip_existing`, and the per-task JSONL stream key for v1/v2 resume. The page_attention run_name must change to a disjoint form (`topk{N}T`) so old and new directories never collide and the v1/v2 per-task `*.jsonl` resume files cannot be accidentally appended to under new semantics. → **Run-name uses `topk{N}T` for page_attention; non-page modes keep `topk{N}`; `summary` records both `top_k` (total) and `middle_top_k`.**

### Viable Options (≥2)

#### Option A — Local subtraction in each eval script's `apply_monkey_patch`/config-call site

Compute `middle_top_k = args.top_k - args.num_sink_pages - args.num_recent_pages` inside each script (or, for AIME/GPQA, inside the shared `eval_ruler.apply_monkey_patch`), pass `middle_top_k` only to the `DCTPageConfig`/`replace_*_attn` calls. Leave `args.top_k` itself unchanged so all other consumers (Quest, run_name, summary, `compute_effective_len`) see the total.

- **Pros**
  - Smallest blast radius; one helper + 1-line replacement at three call-sites in each of three scripts (RULER, LongBench v1, v2). AIME/GPQA inherit automatically because they call `apply_monkey_patch(args)`.
  - Quest path at `eval_ruler.py:433` is automatically unaffected — it reads `args.top_k` (total), not the derived value.
  - `run_name`/`summary["top_k"]` automatically report total; no schema change needed for downstream tooling.
  - `compute_effective_len` in v1/v2 already computes a total-budget formula (`num_sink_pages + top_k + 1 + num_recent_pages`) that becomes correct *as-is* if `top_k` means total minus sink minus recent — see "Semantics drift in compute_effective_len" below.
- **Cons**
  - Subtraction logic is duplicated five times (or once in a helper imported by all five). Need a tiny helper to keep DRY.
  - `compute_effective_len` formulas in v1/v2 (lines 403, 412, 199, 208) need either rewriting OR a renamed local var, because they currently assume `args.top_k` means *middle*. With the new total semantics they would over-count by `sink+recent`. → must update.

#### Option B — Mutate `args.top_k` post-parse to the derived middle, and add a new `args.total_top_k` field

In `parse_args`, do `args.total_top_k = args.top_k; args.top_k = max(1, args.top_k - args.num_sink_pages - args.num_recent_pages)`. Then the *rest of the eval script is unchanged* (config calls, `compute_effective_len`, etc. all keep working).

- **Pros**
  - Zero changes to monkey-patch call sites and to the assembled-len formulas in v1/v2.
  - Conceptually clean: "`args.top_k` = what the kernel uses; `args.total_top_k` = what the user typed."
- **Cons**
  - **Inverts the user's mental model**: after `parse_args` returns, `args.top_k` no longer matches `--top_k`. Anyone reading the script later (or pulling debug prints) will be confused.
  - Quest path at `eval_ruler.py:433`/`:448`/`:772` would silently change semantics — would need `args.total_top_k` substituted at every Quest site, which is *more* edits than Option A, not fewer.
  - `summary["top_k"]` (line 756 etc.) would silently start emitting middle, breaking historical comparability unless explicitly switched to `total_top_k`. We end up touching every site we tried to avoid touching.
  - `run_name` for `page_attention` mode would emit middle (e.g. `topk58` for default) — silent breakage of file paths/dedup keys.
  - Brittle to future code added below `parse_args` that reaches for `args.top_k` expecting "what the user typed."

#### Option C — Add a new flag `--top_k_total` (default unset), keep `--top_k` meaning middle

User opt-in: when `--top_k_total N` is passed, the script computes `args.top_k = N - sink - recent` and ignores `--top_k`. Otherwise old behavior.

- **Pros**
  - Pure additive change. Old sweep scripts keep working unchanged.
- **Cons**
  - **Doesn't deliver the requested redefinition** — user explicitly asked for `--top_k` itself to mean total. C punts the actual semantic flip.
  - Two flags doing similar things → confusion + bug surface.

### Decision

**Adopt Option A, plus the page_attention run_name marker `topk{N}T` (Architect Patch A) and a one-line CLAUDE.md Conventions edit (Architect Patch D).** Option A directly satisfies the request, has the smallest cross-cutting footprint, leaves Quest/Multipole/Seer/Duo/ShadowKV/InfLLM untouched by construction (they read `args.top_k` directly). The `T`-suffix marker on the page_attention run_name makes old/new disjoint at the directory level so v1/v2 per-task JSONL streams cannot be silently contaminated under new semantics. The CLAUDE.md sentence captures the eval-vs-oracle asymmetry inline. To address D1 (sweep-script backward-compat), we add an explicit error when `args.top_k <= num_sink_pages + num_recent_pages` (so old sweeps that used very small `--top_k` fail loud rather than silently).

**Patch B (move helper to `DCTPageConfig.from_total_budget` factory):** *DEFERRED*. Architecturally cleaner and would consolidate the v1/v2 `compute_effective_len` rewrites onto the factory, but it expands the user's stated scope ("only the five eval scripts") to also touch `config.py`. Filed as F6 for a follow-up if the asymmetry becomes annoying. See ADR for justification.

**Invalidation rationale for B (post-parse mutation):** B's "zero call-site change" pitch evaporates because Quest, run_name, and summary fields *all* need to keep reading the user-typed total — so we'd touch the same number of sites as A while introducing a hidden-mutation footgun.

**Invalidation rationale for C (new flag):** Doesn't fulfill the user's stated requirement (redefine `--top_k`). User can layer C on later as backward-compat if needed; not in scope here.

---

## Concrete Answers to the Five Questions

### Q1: Does the redefinition apply only to `page_attention` mode?

**Answer: YES — only `page_attention` mode.**

Rationale:
- Quest (`eval_ruler.py:433` `token_budget = args.page_size * args.top_k`) treats `args.top_k` as Quest's per-decode page budget. Quest has no sink/recent concept. Subtracting would silently shrink Quest's budget by 6 pages (=192 tokens at default `page_size=32`), invalidating every Quest sweep on disk and making Quest unfairly less competitive.
- Multipole, Seer, Duo, ShadowKV, InfLLM never read `args.top_k` (they have their own budget knobs: `--sparse_budget`, `--inf_llm_topk`, etc.).
- Implementation simply guards on `args.mode == "page_attention"` before subtracting (it's already the only branch where `--top_k` flows into `DCTPageConfig`).

### Q2: Where does the subtraction happen?

**Answer: A small helper `_resolve_middle_top_k(args)` at the top of each eval script (or a single helper in a new tiny module `eval_topk_helper.py`), called immediately before each `replace_*_attn(...)` invocation.** Subtraction does NOT happen in argparse post-processing.

Rationale (rules out other locations):
- **NOT argparse post-processing** — that would mutate `args.top_k` and break run_name/summary/Quest (Option B).
- **NOT in `config.py`** — the user explicitly excluded `DCTPageConfig` from the change. Putting it there would make `cfg.top_k` ambiguous across the codebase (kernels, oracle scripts, dct_page_attention.py).
- **At the patch call site** is the unique location where (a) we know the mode is `page_attention`, (b) the value is consumed exactly once, (c) the helper has access to all three relevant args.

Concrete shape:

```python
def _resolve_middle_top_k(args):
    """Translate user-facing --top_k (TOTAL page budget incl. sink+recent)
    into the middle-only top_k that DCTPageConfig expects."""
    middle = args.top_k - args.num_sink_pages - args.num_recent_pages
    if middle < 1:
        raise ValueError(
            f"--top_k={args.top_k} is too small: must be > "
            f"--num_sink_pages ({args.num_sink_pages}) + "
            f"--num_recent_pages ({args.num_recent_pages}) = "
            f"{args.num_sink_pages + args.num_recent_pages}. "
            f"Note: as of this version --top_k is the TOTAL page budget "
            f"(sink + middle + recent), not just middle."
        )
    return middle
```

For `eval_aime25.py` and `eval_gpqa.py`, since they import `apply_monkey_patch` from `eval_ruler.py` (lines 46 in both), the helper lives in `eval_ruler.py` and gets reused for free — they need no edits to their patch flow, only to their run_name/summary output (see Q4).

### Q3: Validation if `args.top_k <= num_sink_pages + num_recent_pages`?

**Answer: HARD ERROR (raise `ValueError` in the helper).**

Rationale:
- Clamping to 1 silently shrinks user budgets — exactly the trust-eroding behavior we want to avoid given D1 (sweep-script backward-compat).
- Warning is too soft for headless sweeps that pipe through hundreds of runs; users won't see it.
- Hard error fails fast with a message that explicitly explains the new semantics, which is the most useful debugging signal for someone hitting it via an old `run_*.sh`.

Edge case: at the **defaults** (`top_k=64`, `sink=1`, `recent=5` → middle=58), this passes trivially. The error only fires for `--top_k <= 6` at default sink/recent, or `--top_k <= sink+recent` in general.

### Q4: Run-name strings, summary JSON fields, and `min_len_for_paging` formulas — total or middle?

**Answer: TOTAL (= what the user typed = `args.top_k`), PLUS a `T` marker on the page_attention run_name so old/new directories are disjoint, PLUS a `middle_top_k` schema addition for traceability.**

Sub-decisions per surface:

| Surface | Value | Rationale |
|---|---|---|
| `run_name` for `page_attention` (`topk{args.top_k}` → `topk{args.top_k}T`) | `args.top_k` (total) **with `T` suffix** | Disjoint from old `topk{N}` run_names. Prevents `--skip_existing` and v1/v2 per-task JSONL resume from silently contaminating new-semantics runs with old-semantics state. Architect Patch A. |
| `run_name` for non-page modes (baseline, seer, multipole, quest, duo, shadowkv, inf_llm) | unchanged | Quest's `topk{N}_pb{args.top_k}` etc. keep their existing format — those modes are not affected by the redefinition. |
| `summary["top_k"]` | `args.top_k` (total) | Downstream plotting/joining keys keep their meaning (now disambiguated by the run_name `T` marker). |
| `summary["middle_top_k"]` *(new field, page_attention only)* | `args.top_k - sink - recent` | Anyone slicing summaries can recover the kernel-level value. Doubles as the schema marker AC9 uses. |
| `compute_effective_len` `min_len_for_paging` (v1:403, v2:199) | Currently `(sink + top_k + 1 + recent) * page_size`. **Update to `(args.top_k + 1) * page_size`** because new `args.top_k` already includes sink+recent. Identity check: `(sink + (top_k - sink - recent) + 1 + recent) = top_k + 1`. ✓ | Semantics-preserving rewrite. |
| `compute_effective_len` `top_k = min(args.top_k, num_pages)` (v1:412, v2:208) | **Update to `top_k = min(args.top_k - args.num_sink_pages - args.num_recent_pages, num_pages)`** | This local `top_k` represents *middle pages selected*, so it must use the derived middle value. |
| Quest summary (`page_budget=args.top_k`, `token_budget=page_size*args.top_k`) | `args.top_k` (total, unchanged) | Per Q1, Quest is not in scope of the redefinition. |
| `dct_page_attention.py:1155, 1710` paging threshold `(num_sink_pages + cfg.top_k + 1 + num_recent_pages) * page_size` | unchanged source | `cfg.top_k` still means middle; threshold formula already correct. After the eval-side change it equates to `(total + 1) * page_size` — semantics-preserving. |

### Q5: Sweep scripts `run_*.sh` — implicitly assume old semantics? Backward-compat flag?

**Answer: YES, all `run_*.sh` for `page_attention` mode implicitly assume OLD semantics. NO backward-compat flag in this PR — handled by (1) the hard error in Q3, (2) the disjoint `topk{N}T` run_name marker for page_attention, and (3) a follow-up sweep-script audit.**

Concretely:
- `run_ruler.sh` / `run_ruler_llama.sh`: pass `--top_k 64` (typical) → with new semantics this becomes middle=58 instead of middle=64. Quality numbers will shift slightly. Their hardcoded `--run_name` (if any) using the old `topk${TOP_K}` format will produce *non-T* directory names that won't collide with new auto-generated `topk{N}T` names. **Action:** flag in F1; let the user decide whether to bump `--top_k 70` (=64+1+5) to preserve OLD middle, or accept new total semantics.
- `run_longbench_v1.sh:60` and `run_longbench_v2.sh:52`: hardcode `--run_name "qwen3_page_attn_ps${PAGE_SIZE}_topk${TOP_K}_..."` (verified at this iteration). These bash run_names will NOT auto-pick up the `T` marker the Python code adds, because they pass `--run_name` explicitly. **Action:** F1 must update these two lines to `topk${TOP_K}T` *or* drop `--run_name` and rely on the Python auto-generated name. Recommended: append the literal `T` in bash to keep semantics explicit.
- `run_ruler_quest.sh`: passes `--top_k` as Quest's page_budget. Per Q1, Quest is unaffected → no action needed.
- `run_ruler_seer.sh`, `run_ruler_multipole.sh`, `run_ruler_duo.sh`: don't use `args.top_k` for DCT path → no action.

Safety nets:
- Hard error at `args.top_k <= sink + recent` (defaults never trip it).
- Disjoint `topk{N}T` run_name marker prevents collisions with old result trees.
- AC9 verifies that v1/v2's per-task JSONL resume cannot be silently contaminated, since the new run_name lands in a different directory.

---

## File-by-File Diff Plan

### Shared helper (introduced in `eval_ruler.py`, imported by AIME/GPQA via the existing import pattern)

**`eval_ruler.py`** — new helper near the top (after argparse, before `apply_monkey_patch`):

```python
def _resolve_middle_top_k(args):
    middle = args.top_k - args.num_sink_pages - args.num_recent_pages
    if middle < 1:
        raise ValueError(
            f"--top_k={args.top_k} is too small: must exceed "
            f"--num_sink_pages ({args.num_sink_pages}) + "
            f"--num_recent_pages ({args.num_recent_pages}) "
            f"= {args.num_sink_pages + args.num_recent_pages}. "
            f"Note: --top_k now denotes the TOTAL selected page budget "
            f"(sink + middle + recent), not just middle pages."
        )
    return middle
```

Also add help text to `--top_k` argparse calls in all 5 scripts (Architect Patch C — verbatim):

```
help="Total selected page budget (sink + middle + recent). "
     "DCTPageConfig receives total - sink - recent as its internal top_k."
```

### `eval_ruler.py`

| Line | Current | New | Notes |
|---|---|---|---|
| 126 | `parser.add_argument("--top_k", type=int, default=64)` | + help text | Document new semantics |
| 314 | `top_k=args.top_k,` (Llama branch) | `top_k=_resolve_middle_top_k(args),` | Llama replace_llama_attn call |
| 333 | `top_k=args.top_k,` (Qwen3 branch) | `top_k=_resolve_middle_top_k(args),` | Qwen3 replace_qwen3_attn call |
| 352 | `top_k=args.top_k,` (Qwen2 branch) | `top_k=_resolve_middle_top_k(args),` | Qwen2 replace_qwen2_attn call |
| 195 | `args.run_name = f"{tag}_page_attn_topk{args.top_k}_..."` | `args.run_name = f"{tag}_page_attn_topk{args.top_k}T_..."` | Append `T` marker — Architect Patch A. Disjoint from old run_names; protects v1/v2 JSONL resume + `--skip_existing` from semantic contamination (AC9). |
| 204 | `args.run_name = f"{tag}_quest_ps{args.page_size}_pb{args.top_k}"` | unchanged | Quest path, total semantics (Q1). NO `T` marker — Quest is not affected by the redefinition. |
| 433 | `token_budget = args.page_size * args.top_k` | unchanged | Quest, total semantics |
| 448 | `print(f"Loading Quest model: ... page_budget={args.top_k} ...")` | unchanged | Quest |
| 756 | `summary["top_k"] = args.top_k` | unchanged + add `summary["middle_top_k"] = _resolve_middle_top_k(args)` | Schema addition |
| 772 | `"page_budget": args.top_k,` | unchanged | Quest summary |
| 773 | `"token_budget": args.page_size * args.top_k,` | unchanged | Quest summary |

Net edits: 3 call-site changes + 1 helper insertion + 1 summary key addition + 1 help-text edit = **6 hunks**.

### `eval_longbench_v1.py`

| Line | Current | New | Notes |
|---|---|---|---|
| 455 | `parser.add_argument("--top_k", type=int, default=64)` | + help text | Document new semantics |
| 403 | `min_len_for_paging = (args.num_sink_pages + args.top_k + 1 + args.num_recent_pages) * args.page_size` | `min_len_for_paging = (args.top_k + 1) * args.page_size` | Identity check: equivalent under new semantics (Q4 table) |
| 412 | `top_k = min(args.top_k, num_pages)` | `top_k = min(args.top_k - args.num_sink_pages - args.num_recent_pages, num_pages)` | Local var `top_k` here means *middle*, so use derived middle |
| 512 | `args.run_name = f"{tag}_page_attn_topk{args.top_k}_{args.comp_kv_quant}"` | `args.run_name = f"{tag}_page_attn_topk{args.top_k}T_{args.comp_kv_quant}"` | Append `T` marker — Architect Patch A. Critical for v1: JSONL resume at lines 531-537 keys on the run_name directory; the `T` marker forces new-semantics runs into a disjoint directory so old per-task `*.jsonl` files can never be appended to (AC9). |
| 698 | `summary_json["top_k"] = args.top_k` | unchanged + add `summary_json["middle_top_k"] = ...` | Schema addition |
| 749 | `top_k=args.top_k,` (Llama branch) | `top_k=_resolve_middle_top_k(args),` | Import helper from `eval_ruler` (or duplicate locally) |
| 767 | `top_k=args.top_k,` (Qwen3 branch) | `top_k=_resolve_middle_top_k(args),` | |
| 785 | `top_k=args.top_k,` (Qwen2 branch) | `top_k=_resolve_middle_top_k(args),` | |
| 972 | `summary["top_k"] = args.top_k` | unchanged + add `summary["middle_top_k"] = ...` | Second summary point (likely a CSV/different summary path) |

Net edits: **8 hunks** (3 patch call-sites + 2 compute_effective_len lines + 2 summary additions + 1 help text).

Decision on helper import vs duplicate: prefer **import** (`from eval_ruler import _resolve_middle_top_k` at top of v1) to keep the validation message consistent.

### `eval_longbench_v2.py`

| Line | Current | New | Notes |
|---|---|---|---|
| 85 | `parser.add_argument("--top_k", type=int, default=64)` | + help text | |
| 199 | `min_len_for_paging = (args.num_sink_pages + args.top_k + 1 + args.num_recent_pages) * args.page_size` | `min_len_for_paging = (args.top_k + 1) * args.page_size` | Same identity as v1 |
| 208 | `top_k = min(args.top_k, num_pages)` | `top_k = min(args.top_k - args.num_sink_pages - args.num_recent_pages, num_pages)` | Local var = middle |
| 145 | `args.run_name = f"{tag}_page_attn_topk{args.top_k}_{args.comp_kv_quant}"` | `args.run_name = f"{tag}_page_attn_topk{args.top_k}T_{args.comp_kv_quant}"` | Append `T` marker — Architect Patch A. Critical for v2: JSONL resume at lines 240-246 keys on `f"{run_name}.jsonl"`; the `T` marker forces new-semantics runs into a disjoint output filename so old `.jsonl` files can never be appended to (AC9). |
| 426 | `summary["top_k"] = args.top_k` | unchanged + add `summary["middle_top_k"] = ...` | |
| 507 | `top_k=args.top_k,` (Llama) | `top_k=_resolve_middle_top_k(args),` | Import from `eval_ruler` |
| 525 | `top_k=args.top_k,` (Qwen3) | `top_k=_resolve_middle_top_k(args),` | |
| 543 | `top_k=args.top_k,` (Qwen2) | `top_k=_resolve_middle_top_k(args),` | |

Net edits: **8 hunks**.

### `eval_aime25.py`

AIME shells out to `eval_ruler.apply_monkey_patch(args)` (line 514). Once `eval_ruler.apply_monkey_patch` is updated (covered above), AIME's patch path is correct **for free**. Only the AIME-local emit sites need touching.

| Line | Current | New | Notes |
|---|---|---|---|
| 225 | `parser.add_argument("--top_k", type=int, default=64)` | + help text | |
| 281 | `args.run_name = (f"{tag}_page_attn_topk{args.top_k}_cr{args.compress_ratio}_..."` | `args.run_name = (f"{tag}_page_attn_topk{args.top_k}T_cr{args.compress_ratio}_..."` | Append `T` marker — Architect Patch A. Disjoint from old run_names. AIME's `--skip_existing` keys on `<run_name>_summary.json` (line 293), so disjoint marker also gates skip-on-rerun (AC9). |
| 454 | `summary["top_k"] = args.top_k` | unchanged + add `summary["middle_top_k"] = ...` | Schema addition |

Net edits: **3 hunks** (1 help text + 1 run_name `T` marker + 1 summary addition).

### `eval_gpqa.py`

GPQA shells out to `eval_ruler.apply_monkey_patch(args)` (line 458). Same story as AIME.

| Line | Current | New | Notes |
|---|---|---|---|
| 187 | `parser.add_argument("--top_k", type=int, default=64)` | + help text | |
| 231 | `args.run_name = (f"{tag}_page_attn_topk{args.top_k}_cr{args.compress_ratio}_..."` | `args.run_name = (f"{tag}_page_attn_topk{args.top_k}T_cr{args.compress_ratio}_..."` | Append `T` marker — Architect Patch A. Same skip-existing rationale as AIME (line 240). |
| 400 | `summary["top_k"] = args.top_k` | unchanged + add `summary["middle_top_k"] = ...` | |

Net edits: **3 hunks** (1 help text + 1 run_name `T` marker + 1 summary addition).

### CLAUDE.md (Conventions section) — Architect Patch D

Add **one sentence** to the "Conventions" bullet list near the top of `## Conventions`:

```
- **`--top_k` semantics**: in `eval_*.py` (page_attention mode) `--top_k` means TOTAL pages
  (sink + middle + recent); in `oracle/*` scripts `--dct_top_k` still means MIDDLE pages.
```

Net edits: **1 hunk**.

### Total diff footprint (v2)

- **6 files touched** (5 eval scripts + `CLAUDE.md` one-line note). ~30 hunks total. ~55 lines net added (help text, `T` markers, `middle_top_k` schema additions, CLAUDE.md sentence).
- **0 files in `dct_page_attention.py`, `config.py`, `triton_kernels.py`, `oracle/`, `baselines/`, `run_*.sh`.**
- AC10 enforces this: `git diff --stat -- ':!eval_*.py' ':!CLAUDE.md'` returns empty.

---

## Edge Cases

1. **`args.top_k <= sink + recent`** → hard error in `_resolve_middle_top_k` with explanatory message (Q3). At defaults (64, 1, 5) this never fires; at extreme small budgets (`--top_k 6 --num_sink_pages 1 --num_recent_pages 5`) it would.
2. **Quest mode** (`eval_ruler.py:433, 448, 772-773`) — `args.top_k` is read directly; **no** subtraction. Verified by Q1; no edit at those lines.
3. **AIME/GPQA shell-out to `apply_monkey_patch`** — covered by the `eval_ruler.py` edits to that function. AIME's `apply_monkey_patch(args)` call (`eval_aime25.py:514`) and GPQA's (`eval_gpqa.py:458`) require **zero** changes.
4. **`compute_effective_len` (v1, v2)** — formulas at lines 403/412 (v1) and 199/208 (v2) currently treat `args.top_k` as middle; under new semantics `args.top_k` is total. Both functions need rewriting per the Q4 table — this is the *only* place in the eval scripts (outside the patch call sites) where `args.top_k` is used as if it were middle.
5. **`min_decode_kv_len_for_paging` floor (CLAUDE.md)** — `dct_page_attention.py` enforces `min_decode_kv_len_for_paging=8192`. With new semantics + defaults, eval-side `min_len_for_paging = (64 + 1) * 32 = 2080` < 8192, so the kernel's floor still gates paging. No behavior change at defaults.
6. **`replace_llama_attn` / `replace_qwen3_attn` / `replace_qwen2_attn` signatures** — these accept `top_k` as a positional/keyword arg already. No upstream signature change. Verified via grep: 3 call sites in `eval_ruler.py`, 3 in `eval_longbench_v1.py`, 3 in `eval_longbench_v2.py` — all in the `apply_monkey_patch`/`main` patch-application flow.
7. **`--skip_existing` and v1/v2 per-task JSONL contamination — RESOLVED via the `topk{N}T` run_name marker (Architect Patch A).**

   **Old hazard (now closed):** at default `--top_k=64`, the old run_name (`topk64`) would have matched a hypothetical new run_name (`topk64`) but with different effective config. v1/v2 also use per-task `*.jsonl` resume (`eval_longbench_v1.py:531-537`, `eval_longbench_v2.py:240-246`) — under the v1 plan, those would have silently appended new-semantics samples onto old-semantics JSONL streams.

   **Resolution:** appending `T` to the page_attention run_name (Patch A) makes the new directory disjoint from any old result tree:
   - `eval_ruler.py:195` → new dir `<tag>_page_attn_topk64T_…/` is disjoint from old `<tag>_page_attn_topk64_…/`
   - `eval_longbench_v1.py:512` → new run_dir contains per-task `*.jsonl` files in a fresh directory; old `narrativeqa.jsonl` etc. are in the old dir.
   - `eval_longbench_v2.py:145` → new output filename `<run_name>T.jsonl` cannot append to old `<run_name>.jsonl`.
   - `eval_aime25.py:281` and `eval_gpqa.py:231` → analogous; `--skip_existing` keys on `<run_name>_summary.json` and the `T` suffix gates reruns into a fresh path.

   AC9 verifies this: synthesize an old-style `narrativeqa.jsonl` *without* `middle_top_k` schema marker; run new code with `--top_k 64`; assert the new run lands in a disjoint output dir (no append to the old JSONL).

   No staleness-detect-and-truncate code path is needed — disjoint paths are simpler and have less surface area to test.

---

## Acceptance Criteria (testable without invoking the LLM)

These are static / dry-run-style checks that pass/fail without GPU or model load.

1. **AC1 — Helper math.** `python -c "from eval_ruler import _resolve_middle_top_k; import argparse; ns = argparse.Namespace(top_k=64, num_sink_pages=1, num_recent_pages=5); assert _resolve_middle_top_k(ns) == 58"` returns 0.
2. **AC2 — Hard error on too-small budget.** `python -c "from eval_ruler import _resolve_middle_top_k; import argparse; ns = argparse.Namespace(top_k=6, num_sink_pages=1, num_recent_pages=5); _resolve_middle_top_k(ns)"` exits non-zero with a message containing `"TOTAL selected page budget"`.
3. **AC3 — Quest path unchanged.** Diff `eval_ruler.py:433` and `:448` and `:772-773` show no `_resolve_middle_top_k` reference.
4. **AC4 — Run-name uses total + `T` marker.** `python eval_ruler.py --mode page_attention --base_model Qwen/Qwen3-8B --top_k 64 --num_samples 0 --skip_existing` prints a run_name containing **`topk64T`** (NOT `topk58`, NOT `topk64`).
5. **AC5 — `compute_effective_len` identity at default.** Add a one-shot script (or scratch test) that calls `eval_longbench_v1.compute_effective_len(input_len=32768, args=...)` with new-semantics `top_k=64, sink=1, recent=5` and confirms the returned value equals the old-semantics call with `top_k=58, sink=1, recent=5` (i.e. the rewrite is identity-preserving at defaults).
6. **AC6 — Patch call site receives middle.** Monkey-patch `dct_page_attention.replace_qwen3_attn` to a stub that records its `top_k` kwarg; invoke `eval_ruler.apply_monkey_patch(Namespace(mode="page_attention", base_model="Qwen/Qwen3-8B", top_k=64, num_sink_pages=1, num_recent_pages=5, ...))`; assert recorded `top_k == 58`.
7. **AC7 — Summary schema additive.** After a dry run with `--num_samples 1` (or via direct call to the summary-build helper), `summary["top_k"] == 64` AND `summary["middle_top_k"] == 58`.
8. **AC8 — Threshold semantics preserved.** With new semantics `top_k=64, sink=1, recent=5, page_size=32`, `min_len_for_paging = (64 + 1) * 32 = 2080`, identical to old-semantics `(1 + 58 + 1 + 5) * 32 = 2080`. Add a unit-style assertion.
9. **AC9 — v1/v2 JSONL staleness via disjoint run_name (Critic mandatory).** Reproducer:
   ```bash
   # Pre-stage an "old-semantics" v1 result tree
   mkdir -p /tmp/v1_test/qwen3_page_attn_topk64_none/
   echo '{"_id":"old","middle_top_k":null}' > /tmp/v1_test/qwen3_page_attn_topk64_none/narrativeqa.jsonl
   # Run new code with --top_k 64; new run_name should be ...topk64T_none, NOT ...topk64_none
   python eval_longbench_v1.py --mode page_attention --base_model Qwen/Qwen3-8B \
       --top_k 64 --num_samples 0 --output_dir /tmp/v1_test --tasks narrativeqa 2>&1 | tee /tmp/v1_dryrun.log
   # Assertion: the auto-generated run_name in the log ends with `topk64T_none`,
   # and the OLD jsonl file is untouched (no append).
   grep -q 'topk64T' /tmp/v1_dryrun.log
   test "$(wc -l < /tmp/v1_test/qwen3_page_attn_topk64_none/narrativeqa.jsonl)" = "1"
   ```
   Same assertion variant for v2: pre-stage `<run_name>.jsonl` (without `T`) and confirm new run writes to `<run_name>T.jsonl` instead.
10. **AC10 — No source touched outside the eval scripts + CLAUDE.md (Critic mandatory).**
    ```bash
    git diff --stat -- ':!eval_*.py' ':!CLAUDE.md'
    # Expected output: empty (zero files changed).
    ```
    This guards against accidental edits to `dct_page_attention.py`, `config.py`, kernels, oracle scripts, baselines, or sweep `.sh` files.

---

## Verification Steps

One-liners or scratch checks per script (no model load needed):

```bash
# 1. Helper math + hard-error sanity
python -c "
from eval_ruler import _resolve_middle_top_k
import argparse
assert _resolve_middle_top_k(argparse.Namespace(top_k=64, num_sink_pages=1, num_recent_pages=5)) == 58
try:
    _resolve_middle_top_k(argparse.Namespace(top_k=6, num_sink_pages=1, num_recent_pages=5))
    raise SystemExit('expected ValueError')
except ValueError as e:
    assert 'TOTAL' in str(e)
print('OK')
"

# 2. Run-name uses total + `T` marker
python eval_ruler.py --mode page_attention --base_model Qwen/Qwen3-8B \
    --top_k 64 --num_samples 0 --output_dir /tmp/t --skip_existing 2>&1 | grep -q 'topk64T' && echo OK

# 3. Patch dispatch sees middle (mocked).
#    Mock Namespace mirrors EVERY kwarg apply_monkey_patch reads from args.
#    Cross-checked against eval_ruler.py:308-366 (Llama/Qwen3/Qwen2 branches all
#    consume the same set; helper-internal kwargs like weight_compressed_by_population=True
#    are passed as literals inside apply_monkey_patch and don't need to be on args).
python -c "
import sys, argparse
import dct_page_attention as dpa
recorded = {}
def fake_qwen3(**kw): recorded.update(kw)
dpa.replace_qwen3_attn = fake_qwen3
from eval_ruler import apply_monkey_patch
ns = argparse.Namespace(
    mode='page_attention',
    base_model='Qwen/Qwen3-8B',
    page_size=32,
    top_k=64,
    num_sink_pages=1,
    num_recent_pages=5,
    compress_ratio=0.125,
    scoring_method='max',
    group_agg_method='max',
    unselected_mode='drop',
    compressed_token_rope='mixed',
    continuous_rope=False,
    no_triton=True,
    comp_kv_quant='none',
    comp_kv_quant_granularity='per_page',
    score_use_quest_minmax=False,
)
apply_monkey_patch(ns)
# weight_compressed_by_population is hardcoded to True inside apply_monkey_patch
# (eval_ruler.py:324, :343, :362), so it appears in `recorded` but isn't on `ns`.
assert recorded['top_k'] == 58, recorded
assert recorded['weight_compressed_by_population'] is True, recorded
print('OK')
"

# 4. compute_effective_len identity check (LongBench v1)
python -c "
import argparse
from eval_longbench_v1 import compute_effective_len
ns_new = argparse.Namespace(mode='page_attention', top_k=64, num_sink_pages=1, num_recent_pages=5,
                             page_size=32, unselected_mode='drop', compress_ratio=0.03125)
new = compute_effective_len(32768, ns_new)
# Hand-compute the old-semantics expected value
sink_tokens = 1 * 32; recent_min = 4 * 32
pageable = 32769 - sink_tokens - recent_min
num_pages = pageable // 32
actual_recent = 32769 - sink_tokens - num_pages * 32
top_k_middle = min(58, num_pages)
expected = sink_tokens + top_k_middle * 32 + actual_recent
assert new == expected, (new, expected)
print('OK')
"

# 5. End-to-end smoke (small): drop run with --num_samples 1, confirm summary.json has both keys
python eval_ruler.py --mode page_attention --base_model Qwen/Qwen3-8B \
    --seq_lengths 32768 --num_samples 1 --top_k 64 --output_dir /tmp/topk_test \
    --run_name smoke
python -c "import json; s = json.load(open('/tmp/topk_test/smoke/summary.json'));
print(s['top_k'], s['middle_top_k']); assert s['top_k'] == 64 and s['middle_top_k'] == 58"
```

---

## Follow-ups (out of scope for this PR)

- **F1 — Sweep-script audit (page_attention mode only).**
  - `run_longbench_v1.sh:60` and `run_longbench_v2.sh:52` hardcode `--run_name "..._topk${TOP_K}_..."`. Update these two lines to `..._topk${TOP_K}T_..."` to mirror the Python auto-generated `T` marker. Otherwise their result trees won't carry the disjoint marker and may collide with old result trees if anyone reused those scripts.
  - `run_ruler.sh`, `run_ruler_llama.sh`: do NOT hardcode `--run_name`, so they inherit the Python-side `T` marker for free. Only the `--top_k` value semantics shift; user must decide whether to bump to `--top_k 70` to preserve old middle=64 OR accept new total=64 / middle=58.
  - **Quest+DCT alignment note (Architect Patch D rationale):** sweep scripts that compare DCT vs Quest at "the same `--top_k`" are now comparing *different* things — DCT's middle drops by `sink+recent`, while Quest's page_budget stays at `--top_k`. To preserve the apples-to-apples Quest-vs-DCT page-budget comparison documented in past results, sweep authors should pass DCT `--top_k = (Quest_page_budget) + sink + recent` going forward. Document this in the run-script README or in inline comments at `run_ruler_quest.sh`.
- **F2 — Oracle scripts.** `oracle/oracle_ruler.py`, `oracle/run_ruler_oracle_selection.py`, `oracle/dc_ac_ruler.py`, `oracle/hybridmulti_ruler.py`, `oracle/attention_mass_recall_ruler.py` — currently use their own `--top_k` / `--dct_top_k` flags. Decision codified in this PR via Patch D / CLAUDE.md edit: **do not** change oracle semantics; `--dct_top_k` keeps meaning MIDDLE. Mark this section as resolved (no PR work needed) — the asymmetry is now documented inline.
- **F3 — `summary["middle_top_k"]` schema rollout.** Update any external plot scripts that aggregate eval summaries to optionally read `middle_top_k`. Most plotters key on `top_k` (still total), so this is additive.
- **F4 — `--skip_existing` staleness guard.** Closed by Patch A (`topk{N}T` marker makes new dirs disjoint from old ones). No staleness-detection code path needed.
- **F5 — Optional backward-compat flag.** If sweep regressions are painful, add `--top_k_semantics={total,middle}` (defaulting to `total`) as a one-PR backward-compat shim.
- **F6 — `DCTPageConfig.from_total_budget` factory (Architect Patch B, deferred).** Move `_resolve_middle_top_k` into `config.py` as a classmethod factory `DCTPageConfig.from_total_budget(total_top_k, num_sink_pages, num_recent_pages, ...)`. Removes the `from eval_ruler import _resolve_middle_top_k` smell across v1/v2/AIME/GPQA, lets `compute_effective_len` reuse the same factory, and makes the total-vs-middle conversion a first-class API. Deferred because this PR is scoped to "the five eval scripts" and Patch B expands the surface to `config.py`. Adopt as a 1-hunk follow-up if/when oracle scripts also want a `from_total_budget` entry point.

---

## ADR (v2)

- **Decision:** Redefine `--top_k` in the five eval scripts (`eval_ruler.py`, `eval_longbench_v1.py`, `eval_longbench_v2.py`, `eval_aime25.py`, `eval_gpqa.py`) as the TOTAL selected page budget = sink + middle + recent. Implement via a local `_resolve_middle_top_k(args)` helper invoked exclusively at `replace_*_attn` call sites under `args.mode == "page_attention"`. Append a `T` marker to all page_attention auto-generated run_names (`topk{N}` → `topk{N}T`) so old/new result trees are disjoint. Add one sentence to `CLAUDE.md` Conventions documenting the eval-vs-oracle semantic asymmetry.
- **Drivers:** D1 sweep-script backward-compat, D2 Quest's reuse of `args.top_k` as page_budget, D3 result reproducibility via run_name + summary keys + v1/v2 per-task JSONL streams.
- **Alternatives considered:**
  - *Architect Patch B — `DCTPageConfig.from_total_budget` factory in `config.py`:* **DEFERRED to F6**. Architecturally cleaner; would consolidate `compute_effective_len` and remove the `from eval_ruler import _resolve_middle_top_k` cross-script smell. Rejected for THIS PR because user explicitly scoped the change to "only the five eval scripts" and Patch B touches `config.py`. The asymmetry it introduces (helper imported from a sibling eval script) is acceptable for a 5-file change. Re-evaluate when oracle scripts also want a `from_total_budget` entry point.
  - *Option B (mutate `args.top_k` post-parse, add `args.total_top_k`):* rejected — inverts user mental model, silently changes Quest semantics, breaks run_name.
  - *Option C (add new `--top_k_total` flag, keep `--top_k` meaning middle):* rejected — does not deliver the requested redefinition; punts the semantic flip to a future flag and adds a redundant knob.
  - *Critic option (b) — per-task JSONL staleness check via schema marker:* rejected in favor of Critic option (a) `topk{N}T` run_name marker. The marker makes new dirs disjoint at the filesystem level, eliminating the need for a runtime staleness check (less code, fewer failure modes).
- **Why chosen:** Option A + Patch A + Patch C + Patch D delivers the requested redefinition with the smallest blast radius (6 files including a one-line CLAUDE.md edit; ~30 hunks; ~55 net lines). Quest is unaffected by construction. Run_name carries an explicit semantic marker (`T`) that prevents `--skip_existing` and v1/v2 per-task JSONL resume from contaminating new-semantics runs with old-semantics state. Hard error on `top_k <= sink + recent` ensures any silent backward-compat trap fails loud. CLAUDE.md sentence captures the eval-vs-oracle asymmetry inline.
- **Consequences:**
  - At defaults (`--top_k 64 --num_sink_pages 1 --num_recent_pages 5`), kernel-side middle drops from 64 → 58. Quality numbers will shift slightly compared to past runs.
  - All page_attention result directories under the new code carry a `T` marker (`topk64T` etc.); old directories (`topk64`) remain readable side-by-side without collision.
  - `summary["middle_top_k"]` is added to disambiguate going forward.
  - Quest, Multipole, Seer, Duo, ShadowKV, InfLLM modes are unaffected.
  - `run_longbench_v1.sh:60` and `run_longbench_v2.sh:52` will not auto-pick up the `T` marker (they hardcode `--run_name`); F1 mandates a one-line bash edit there. Other sweep scripts inherit the marker for free.
  - `--skip_existing` collisions are eliminated by the disjoint marker; no staleness-detection code path needed.
- **Follow-ups:** F1 sweep-script audit (with explicit Quest+DCT alignment note), F2 oracle alignment **resolved via CLAUDE.md edit**, F3 schema rollout, F4 skip-existing **closed by Patch A**, F5 optional `--top_k_semantics` flag, F6 `DCTPageConfig.from_total_budget` factory (deferred Architect Patch B).

---

## Revision history

### v2 (iteration 1, 2026-05-08) — Architect STRENGTHEN + Critic ITERATE feedback applied

**Adopted:**
- **Critic mandatory #1 — `topk{N}T` run_name marker (option a, equivalent to Architect Patch A).** Page_attention auto-generated run_names now carry a `T` suffix at `eval_ruler.py:195`, `eval_longbench_v1.py:512`, `eval_longbench_v2.py:145`, `eval_aime25.py:281`, `eval_gpqa.py:231`. Disjoint paths protect v1/v2 per-task JSONL resume + `--skip_existing` from semantic contamination.
- **Critic mandatory #2 — AC9 added.** Verifies v1/v2 JSONL staleness via disjoint run_name; reproducer scripted in the Acceptance Criteria section.
- **Critic mandatory #3 — AC10 added.** `git diff --stat -- ':!eval_*.py' ':!CLAUDE.md'` must return empty.
- **Critic mandatory #4 — CLAUDE.md Conventions edit (Architect Patch D).** One-sentence in-scope addition documenting that `--top_k` in `eval_*.py` (page_attention) means TOTAL while `--dct_top_k` in `oracle/*` keeps meaning MIDDLE. F2 marked as resolved (no oracle code changes needed).
- **Critic mandatory #5 — Verification step 3 mock Namespace cross-checked against `eval_ruler.py:308-366` and padded.** Added comment noting `weight_compressed_by_population=True` is hardcoded inside `apply_monkey_patch` (eval_ruler.py:324, :343, :362), not on `args` — so the mock Namespace correctly omits it but the assertion checks `recorded['weight_compressed_by_population'] is True`. All other args kwargs (`page_size`, `top_k`, `num_sink_pages`, `num_recent_pages`, `compress_ratio`, `scoring_method`, `group_agg_method`, `unselected_mode`, `compressed_token_rope`, `continuous_rope`, `no_triton`, `comp_kv_quant`, `comp_kv_quant_granularity`, `score_use_quest_minmax`) are present.
- **Architect Patch C — verbatim help text.** "Total selected page budget (sink + middle + recent). DCTPageConfig receives total - sink - recent as its internal top_k." Applied to all 5 `--top_k` argparse calls.
- **Architect Patch D — F1 Quest+DCT alignment doc note.** F1 now explicitly calls out that DCT-vs-Quest comparisons at "the same `--top_k`" are no longer apples-to-apples; sweep authors must adjust DCT `--top_k` by `+sink+recent` to preserve the old comparison.

**Deferred:**
- **Architect Patch B — `DCTPageConfig.from_total_budget` factory.** Architecturally cleaner but expands scope to `config.py`. Recorded as F6 follow-up. ADR explicitly justifies deferral.

**Not adopted:**
- **Critic option (b) — runtime JSONL staleness-detect-and-truncate.** Replaced by the cleaner option (a) `T` marker, which moves the disambiguation to the filesystem level (no runtime check needed).

**Document changes:**
- Title bumped to `(v2)`.
- Principles expanded P1–P4 → P1–P5 (added P5 for cross-script asymmetry doc).
- Decision Drivers D1/D3 reworded to explicitly include v1/v2 per-task JSONL resume and the `topk{N}T` disjoint-path requirement.
- "Decision" block split: explicitly enumerates Patch A / Patch C / Patch D as adopted, Patch B as deferred.
- File-by-file diff plan: 5 run_name table rows updated from "unchanged" to `T`-marker edits, with cross-references to AC9.
- New "CLAUDE.md (Conventions section)" sub-section added under Diff Plan with the exact one-line edit.
- Total diff footprint updated: 5 → 6 files, ~26 → ~30 hunks, ~50 → ~55 lines.
- Edge case #7 rewritten: "RESOLVED via the `topk{N}T` run_name marker" with per-script disjoint-path proof.
- Acceptance Criteria: AC4 updated (`topk64T` not `topk64`); AC9 (JSONL staleness reproducer) and AC10 (`git diff --stat` scope guard) added.
- Verification block: step 2's grep target updated to `topk64T`; step 3's mock Namespace padded with comment cross-referencing `eval_ruler.py:308-366`.
- Follow-ups: F2 marked resolved; F4 marked closed; F6 (deferred Patch B) added.
- ADR: title bumped to (v2); "Alternatives considered" expanded to record Patch B deferral and Critic option (b) rejection; "Consequences" updated to reflect disjoint paths.

### v1 (initial, 2026-05-08)

Initial plan. Option A adopted. Five-question matrix completed. ~26 hunks across 5 files. AC1–AC8. Follow-ups F1–F5.
