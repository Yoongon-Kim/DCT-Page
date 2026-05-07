# SnapKV Baseline Integration Plan

Integrate SnapKV (FasterDecoding) as a new baseline in DCT-Page, mirroring the
`infllm` shim pattern (closest analog: same `transformers==4.37` pin,
Llama-only constraint, custom `prepare_inputs_for_generation`).

Upstream: `/home/yoongonkim/SnapKV` (https://github.com/FasterDecoding/SnapKV).
Target repo: `/home/yoongonkim/DCT-Page`.

---

## Iteration 3 changelog (2026-05-07)

Targeted revisions in response to Critic iter-2 ITERATE verdict (2 blocking + 2 recommended):

- **§13 (NEW intra-package import block):** Concatenating the three upstream files into one `_vendor.py` requires deleting `from snapkv.monkeypatch.* import ...` lines (e.g. `llama_hijack_4_37.py:14`); the symbols become local definitions, so the imports become no-ops. — Critic iter-2 BLOCKING #1 (Architect concern A).
- **§11 Step B firing-count gate relaxed from `== 160` to `>= 32`:** at-least-one-full-sample threshold tolerates samples below `max_capacity_prompt`; strict-equality runs require precondition-filtering admitted samples by `input_ids` length. Acceptance summary updated to match. — Critic iter-2 BLOCKING #2 (Architect concern C).
- **§13 re-export rename `init_snapkv` → `_init_snapkv`:** private leading-underscore on the vendored helper avoids confusion with the public `init_snap_kv` (two-word) shim. — Critic iter-2 RECOMMENDED #1.
- **§13 firing-print + §11 Step B smoke command + ADR Consequences:** firing-print gated behind `SNAPKV_TRACE=1` env var; production sweeps run unset (no stdout flood); §11 Step B sets `SNAPKV_TRACE=1` explicitly. — Critic iter-2 RECOMMENDED #2.

---

## Iteration 2 changelog (2026-05-07)

Targeted revisions in response to Critic ITERATE / Architect NEEDS-REVISION verdicts:

- **§4 Edit 5 + §5 longbench equivalents:** Llama-only guard moved to the `attn_impl`
  switch (pre-load `SystemExit`), replacing the old post-load-only assert. — Critic blocking #4.
- **§4 Edit 2 + §5 longbench equivalents:** Argparse-time invariant
  `--snapkv_max_capacity_prompt > --snapkv_window_size` added via `parser.error(...)`
  (matches the `assert` at `snapkv_utils.py:27`). — Critic blocking #3.
- **§4 Edit 11 + §5 longbench equivalents:** New `summary["rope_caveat"]` block
  written into result JSON for every snap_kv run, marking
  `comparable_to_tx5_baseline=False`. — Critic blocking #1.
- **§11 Step B:** Acceptance gate now requires positive `[snapkv] update_kv fired:`
  stdout line at `num_layers x num_samples` count, sourced from a one-line vendored
  print in `_vendor.py` (see §13). — Critic blocking #2.
- **§4 / §5:** All "line N" references replaced with pattern-anchored locations
  (verified against current HEAD `b311efd`); a small set of confirmed numeric
  anchors retained for orientation. — Critic blocking #5.
- **§11 Step C:** Tightened acceptance from 5pp to 2pp on niah_single_1 with
  documented escalation path. — Critic non-blocking #1.
- **§7:** Pinned `transformers==4.37.2` (matching `infllm` env) instead of 4.37.0;
  both pass the `'4.37'` substring check. — Critic non-blocking #2.
- **§4 Edit 7 + §5 longbench equivalents:** `inf_llm_config_override` renamed to
  `legacy_config_override` (now has 2+ consumers). Promoted from optional to required.
  — Critic non-blocking #3.
- **RALPLAN-DR pre-mortem:** 4th scenario added (`kv_seq_len` carry-over across
  `model.generate()` calls). — Critic non-blocking #4.
- **§11 Step E:** Cross-baseline gate replaced with same-baseline-deltas gate;
  added explicit budget-translation note (`infllm.topk * block_size ~= snapkv.max_capacity_prompt`).
  — Critic non-blocking #5.
- **§13 (NEW) + ADR Decision/Consequences/Follow-ups:** Adopted Architect's
  vendoring proposal — `baselines/snap_kv/_vendor.py` carries `llama_hijack_4_37.py`
  + `SnapKVCluster` + `init_snapkv` (~280 LOC) instead of `pip install -e /home/yoongonkim/SnapKV`.
  Strict reduction in moving parts; upstream is dormant. — Architect steelman.

---

## RALPLAN-DR Summary

### Principles (5)
1. **Mirror the `infllm` shim shape exactly.** Same file layout, same
   `assert_llama_only` / `load_llama_config_stripped_rope` / `init_*` API
   surface. Reviewers and the eval harness should see one pattern.
2. **No collisions with upstream package names.** `baselines/<dir>` must NOT
   shadow `snapkv` (the upstream package the shim imports). See lesson
   `feedback_infllm_shim_naming.md` and CLAUDE.md.
3. **Fail closed when env / model is wrong.** Hard-error if `--base_model`
   isn't Llama (Qwen3 unsupported upstream); warn loudly if `transformers`
   isn't `4.37.x`.
4. **Minimize churn in `eval_*.py`.** All wiring follows the existing inf_llm
   pattern: identical CLI block, identical `attn_impl` switch, identical
   summary-dump shape. No restructuring of the shared model-load path.
5. **No custom generator unless the upstream forces one.** SnapKV patches
   `prepare_inputs_for_generation` AND `LlamaFlashAttention2.forward`, so HF
   `model.generate()` works out of the box - no `SnapKVGenerator` class
   needed. (This is the one place we diverge from `infllm`'s shape.)

### Decision drivers (top 3)
1. **`transformers==4.37.0` pin** — needs a dedicated env; cannot share the
   main `DCT_Page` env (transformers 5.2.0).
2. **No Qwen3 upstream** — only Llama / Mistral / Mixtral hijacks exist; we
   scope v1 to Llama-3.x and document Qwen3 as out of scope. AIME25/GPQA
   (Qwen3-only) cannot use SnapKV in v1.
3. **`flash_attention_2` is mandatory** — SnapKV patches
   `LlamaFlashAttention2.forward`. We must pass
   `attn_implementation="flash_attention_2"`, not `"eager"` or `"sdpa"`.
   This is the load-bearing difference vs `infllm` (which uses `eager`) and
   `duo_attention` (which uses `eager`).

### Viable options

**Option A — Standalone shim mirroring infllm (RECOMMENDED, baseline of
this plan).**
- Pros: 1-to-1 review against `baselines/infllm/`. Reuses
  `load_llama_config_stripped_rope` (same problem). Fresh conda env keeps
  `flash-attn==2.4.0` requirement contained.
- Cons: Duplicated `assert_llama_only` / `load_llama_config_stripped_rope`
  helpers. Could be factored into `baselines/_shared.py` later (out of scope).

**Option B — Reuse the existing `infllm` conda env.**
- Pros: Saves disk and one env's worth of install time.
- Cons: `infllm` env was provisioned for `transformers==4.37.2`. SnapKV
  hijack is keyed off the exact 4.37.x string match (works), BUT `infllm`
  installs `flash-attn==2.6.3` from `feedback_flashinfer_torch_upgrade.md`
  era and SnapKV README pins `flash-attn==2.4.0`. Risk of subtle
  numeric/API drift in `_flash_attention_forward`. Verdict: defer; only
  pick this if Option A's env build fails. **Document as a fallback.**

**Option C — Re-implement SnapKV's prefill compression on top of eager
attention (no flash-attn dep).**
- Pros: Could share the `DCT_Page` env (transformers 5.2.0). No
  flash-attn 2.4.0 chase.
- Cons: Forks upstream code; needs porting `llama_flash_attn2_forward` to
  the eager path, mirroring transformers 5.x's
  `LlamaAttention.forward(eager-mode)` signature, then maintaining it. ~1
  week of work + ongoing maintenance vs Option A's day-one shim. **Reject
  for v1**; revisit only if flash-attn 2.4.0 wheels are unavailable for
  cu12.x torch.

### Pre-mortem (4 scenarios, deliberate-mode)

1. **`transformers==4.37.0` cannot load Llama-3.1's `rope_scaling`**
   (`rope_type='llama3'`).  
   *Likelihood:* certain — `infllm` already hits this at line 49-64 of
   `baselines/infllm/__init__.py`.  
   *Mitigation:* port `load_llama_config_stripped_rope(base_model)` byte-for-byte
   into `baselines/snap_kv/__init__.py`, pass via `config=` kwarg in
   `from_pretrained`. Same code path as `inf_llm_config_override` at
   `eval_ruler.py:496-499`. Verified safe because SnapKV does NOT replace
   RoPE — but Llama-3.1's `rope_type='llama3'` scaling is also not the
   default `linear`/`dynamic` SnapKV is tested with. We accept that
   long-context positions will use the unscaled base RoPE (same trade-off
   as transformers 4.37 era papers).
2. **`flash-attn==2.4.0` wheel unavailable for current torch/cu12.x.**  
   *Likelihood:* low — `pip index versions flash-attn` shows 2.4.0.post1
   available; cu12.x compatible builds exist for torch 2.1+.  
   *Mitigation:* try `flash-attn==2.4.0.post1` first; if that fails,
   relax to `flash-attn>=2.4.0,<2.6` and explicitly verify SnapKV's
   `_flash_attention_forward` signature still binds. If 2.4.x is entirely
   unavailable, fall back to Option B (reuse `infllm` env's 2.6.3 — risk
   accepted with smoke-test verification).
3. **SnapKV's `prepare_inputs_for_generation_llama` is incompatible with
   HF `generate()` in transformers 4.37 because of `DynamicCache`
   semantics changing across patch versions.**  
   *Likelihood:* low — the hijack is copied from transformers 4.37 source
   verbatim; both `Cache` and tuple paths are handled.  
   *Mitigation:* run the smoke test (RULER `niah_single_1` @ 8K, 5
   samples) BEFORE wiring the full sweep. If `generate()` fails, fall back
   to a `duo_attention`-style manual greedy loop in `eval_ruler.py`
   (`duo_generate_greedy` is the template). Cost: +30 LOC, no design
   change.
4. **`kv_seq_len` carry-over across `model.generate()` calls when an
   internal cache is reused.**  
   *Likelihood:* low — SnapKV's patched `prepare_inputs_for_generation_llama`
   resets `kv_seq_len = 0` on the `past_key_values is None` branch
   (`llama_hijack_4_37.py:141-143`). Each `model.generate()` call in the
   eval loop creates a fresh `past_key_values` and never reuses it.  
   *Mitigation:* the eval loop already calls `model.generate()` once per
   sample with no shared cache — make this **explicit** in the plan
   (covered in §11 Step B with a 5-sample smoke). If a future refactor
   introduces cache reuse, add a `model.snap_kv_clear()`-style hook that
   walks `model.model.layers` and calls
   `attn.kv_cluster.reset(...)` (the SnapKVCluster `reset()` already
   exists at `snapkv_utils.py:31`).

---

## 1. Directory layout: `baselines/snap_kv/`

**Folder name choice: `snap_kv` (NOT `snapkv`).** Justification:
- Vendored upstream module path is `snap_kv._vendor.snapkv_utils` /
  `snap_kv._vendor.llama_hijack`. We do NOT `pip install -e /home/yoongonkim/SnapKV`
  (see §13 vendoring decision).
- If we kept the upstream pip-install path, the local dir name `snapkv`
  would shadow the upstream package via `sys.path.insert(0, "baselines")`
  (done in each eval script's preamble) — same bug that drove
  `inf_llm` -> `infllm` (CLAUDE.md, `feedback_infllm_shim_naming.md`).
- `snap_kv` (with underscore) is a strict-collision-free name regardless
  of vendoring vs pip-install path. We keep it for forward compatibility
  with v2 (e.g. if a fork is needed).

Files:

```
baselines/snap_kv/
├── __init__.py        # Public API: assert_llama_only,
│                      # load_llama_config_stripped_rope, init_snap_kv
├── config.py          # SNAPKV_CONFIG dict (mirrors INF_LLM_CONFIG)
└── _vendor.py         # Vendored: SnapKVCluster, init_snapkv,
                       # llama_hijack_4_37 (patched forward +
                       # prepare_inputs_for_generation_llama),
                       # replace_llama. ~280 LOC. See §13.
```

Concretely:

### `baselines/snap_kv/__init__.py`
Public surface:
```python
def assert_llama_only(base_model: str) -> None: ...
def load_llama_config_stripped_rope(base_model: str): ...
def init_snap_kv(model, cfg: dict) -> None: ...
__all__ = ["assert_llama_only", "load_llama_config_stripped_rope", "init_snap_kv"]
```

`init_snap_kv(model, cfg)` does **two** things in this exact order:

1. Call `replace_llama()` from upstream (patches the `LlamaForCausalLM`
   class methods). NOTE: this patches the class, so it can run before OR
   after `from_pretrained`; we run it post-load to keep the eval harness
   shape symmetric with `init_inf_llm`.
2. Walk `for layer in model.model.layers: snapkv_attrs(layer.self_attn,
   cfg)` to set `layer.self_attn.config.window_size /
   max_capacity_prompt / kernel_size / pooling`. SnapKV's
   `init_snapkv(self)` (in `snapkv_utils.py:72`) reads these off
   `self.config` lazily on the first prefill.

   **Important detail:** `init_snapkv` in upstream is invoked from
   inside the patched `llama_flash_attn2_forward` on every forward (idempotent
   `if not hasattr(self, "kv_cluster")` guard). We MUST set the four
   `config.*` attributes on each `self_attn` BEFORE the first forward
   call, otherwise SnapKV's defaults (window_size=32, max_capacity_prompt=2048)
   take effect silently. Our `init_snap_kv` shim is what guarantees this.

Pseudo-code body:

```python
def init_snap_kv(model, cfg):
    from snap_kv._vendor import replace_llama   # vendored, NOT pip-install
    replace_llama()
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.config.window_size = cfg["window_size"]
        attn.config.max_capacity_prompt = cfg["max_capacity_prompt"]
        attn.config.kernel_size = cfg["kernel_size"]
        attn.config.pooling = cfg["pooling"]
    print(f"[snap_kv] window_size={cfg['window_size']} "
          f"max_capacity_prompt={cfg['max_capacity_prompt']} "
          f"kernel_size={cfg['kernel_size']} pooling={cfg['pooling']}")
```

Note: the vendored `_vendor.py` ships with one **load-bearing one-line
diff** vs upstream — a `print(f"[snapkv] update_kv fired: q_len={q_len}, "
f"cap={self.max_capacity_prompt}")` inside the `else:` branch of
`SnapKVCluster.update_kv` (snapkv_utils.py:43, the line right after the
`if q_len < self.max_capacity_prompt: return ...` short-circuit). This
print is the basis of the §11 Step B firing-count acceptance gate.

`assert_llama_only` and `load_llama_config_stripped_rope` are copied
from `baselines/infllm/__init__.py:33-64` with the `InfLLM` strings
swapped to `SnapKV`.

### `baselines/snap_kv/config.py`

```python
"""SnapKV evaluation configuration (auto-generated by run_ruler_snapkv.sh).

Requires transformers==4.37.0 + flash-attn==2.4.0; see
.omc/plans/snapkv-baseline-integration.md for env setup. Llama 3.x only.
Qwen3 is NOT supported upstream (no Qwen3 hijack file in
/home/yoongonkim/SnapKV/snapkv/monkeypatch/).
"""

SNAPKV_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",

    # Sliding window of "current" tokens that trigger compression scoring.
    # SnapKV uses these as the query window; default 32 in upstream
    # init_snapkv (snapkv_utils.py:75).
    "window_size": 32,

    # Total tokens kept after prefill compression
    # (= top-(max_capacity_prompt - window_size) compressed + window_size verbatim).
    # Default 2048 upstream; sweep target.
    "max_capacity_prompt": 2048,

    # Pool kernel applied to per-key attention scores before topk.
    # Smooths spiky scores; default 5 upstream.
    "kernel_size": 5,

    # 'avgpool' or 'maxpool'.
    "pooling": "avgpool",
}
```

Per-layer lists are NOT supported in v1 (upstream's `init_snapkv` reads
scalar attrs from `self.config`). Out of scope.

---

## 2. Shim API surface — decisions

| Function / class | Status | Justification |
|---|---|---|
| `assert_llama_only(base_model)` | **Yes** | Qwen3 / Mistral hijacks not ported. Hard-fail. |
| `load_llama_config_stripped_rope(base_model)` | **Yes** | Same `transformers 4.37` cannot parse Llama-3.1 `rope_type='llama3'` issue; reuse infllm's strip-and-load. |
| `init_snap_kv(model, cfg)` | **Yes** | Patches class methods + sets per-attn config attrs. |
| `SnapKVGenerator` class | **NO** | SnapKV hijacks `prepare_inputs_for_generation_llama` AND uses standard `DynamicCache.update`. HF `model.generate()` works as-is. (`infllm` needed a generator only because its `ContextManager` KV cache can't round-trip through HF's `Cache` interface.) |
| `build_snap_kv_generator()` | **NO** | Same. |
| `snap_kv_clear()` | **NO** | SnapKV's `kv_seq_len` is reset by `prepare_inputs_for_generation_llama` itself when `past_key_values is None` (snapkv/monkeypatch/llama_hijack_4_37.py:141-143). Each fresh `model.generate()` triggers this. No per-sample teardown needed. |

---

## 3. Config schema (`SNAPKV_CONFIG`)

Fields (4 total, all CLI-overridable):

| Key | Type | Default | CLI flag | Notes |
|---|---|---|---|---|
| `base_model` | str | (required, set from `args.base_model`) | n/a | Llama 3.x only. |
| `window_size` | int | 32 | `--snapkv_window_size` | Query window for scoring. |
| `max_capacity_prompt` | int | 2048 | `--snapkv_max_capacity_prompt` | Total tokens kept after prefill. Main quality knob; sweep target (analog of `inf_llm.topk`). |
| `kernel_size` | int | 5 | `--snapkv_kernel_size` | Pool kernel size. |
| `pooling` | str | "avgpool" | `--snapkv_pooling` | `avgpool` or `maxpool`. |

Constraint (asserted upstream): `max_capacity_prompt - window_size > 0`.
Mirror the assert in our config doc.

---

## 4. `eval_ruler.py` wiring (pattern-anchored edits)

Edits use **pattern anchors** (the surrounding code construct) instead of
brittle line numbers. A handful of confirmed line numbers from current
HEAD `b311efd` are kept inline for reader orientation; treat them as
informational, not authoritative — the patterns are the source of truth.

### Edit 1 — argparse `--mode` choices

**Anchor:** `parser.add_argument("--mode", ..., choices=[...])` — verified at
~line 98 of `eval_ruler.py`.

Add `"snap_kv"` to the `choices=[...]` list:

```python
parser.add_argument("--mode", type=str, required=True,
                    choices=["baseline", "page_attention", "seer_attention",
                             "seer_prefill",
                             "multipole_attention", "quest_attention",
                             "duo_attention",
                             "shadowkv",
                             "inf_llm",
                             "snap_kv"])    # NEW
```

### Edit 2 — SnapKV CLI args + argparse invariant

**Anchor:** insert the new `add_argument(...)` block immediately after the
last `--inf_llm_*` add_argument call. Then immediately after `args = parser.parse_args()`,
add the `parser.error(...)` invariant.

```python
# SnapKV baseline params (only used when --mode snap_kv).
parser.add_argument("--snapkv_window_size", type=int, default=32,
                    help="SnapKV: query window size for prefill scoring.")
parser.add_argument("--snapkv_max_capacity_prompt", type=int, default=2048,
                    help="SnapKV: total tokens kept after prefill compression.")
parser.add_argument("--snapkv_kernel_size", type=int, default=5,
                    help="SnapKV: pool kernel size on attention scores.")
parser.add_argument("--snapkv_pooling", type=str, default="avgpool",
                    choices=["avgpool", "maxpool"],
                    help="SnapKV: pooling op on attention scores.")
```

Argparse-time invariant (mirrors the `assert` at `snapkv_utils.py:27`,
which would otherwise crash 30s into model load):

```python
# Right after `args = parser.parse_args()`:
if args.mode == "snap_kv" and args.snapkv_max_capacity_prompt <= args.snapkv_window_size:
    parser.error(
        f"--snapkv_max_capacity_prompt ({args.snapkv_max_capacity_prompt}) "
        f"must be > --snapkv_window_size ({args.snapkv_window_size}) "
        f"(SnapKVCluster asserts this; see snapkv_utils.py:27)"
    )
```

### Edit 3 — Default `run_name`

**Anchor:** the `if args.run_name is None:` block, after the existing
`elif args.mode == "inf_llm":` branch.

```python
elif args.mode == "snap_kv":
    args.run_name = (f"{tag}_snap_kv_w{args.snapkv_window_size}"
                     f"_cap{args.snapkv_max_capacity_prompt}"
                     f"_k{args.snapkv_kernel_size}"
                     f"_{args.snapkv_pooling}")
```

### Edit 4 — `apply_monkey_patch`

**Anchor:** the `apply_monkey_patch(args, ...)` function, after the existing
`elif args.mode == "inf_llm":` branch (which is also a no-op).

```python
elif args.mode == "snap_kv":
    pass  # SnapKV patches LlamaForCausalLM class methods post-load (see init_snap_kv).
```

(The actual class patching happens inside `init_snap_kv` so the call
order matches `inf_llm` and `duo_attention`: load model first, then
patch.)

### Edit 5 — `attn_impl` switch (Llama-only guard moved here)

**Anchor:** the single line `attn_impl = "eager" if args.mode in {...} else "sdpa"`.
Verified at line 470 of `eval_ruler.py`.

**This is the load-bearing change vs inf_llm.** SnapKV patches
`LlamaFlashAttention2.forward`, so we MUST request `flash_attention_2`,
not `eager`. The Llama-only guard is enforced **here** (pre-load) instead
of post-load — so a `--mode snap_kv --base_model Qwen/...` invocation
fails fast without spending 30s loading the model:

```python
# DuoAttention's and InfLLM's replacement forwards assume eager-style Q/K/V
# signatures. SnapKV patches LlamaFlashAttention2.forward, so it requires
# flash_attention_2.
if args.mode == "snap_kv":
    if "llama" not in args.base_model.lower():
        raise SystemExit(
            f"snap_kv baseline only supports Llama models "
            f"(got base_model={args.base_model!r}); Qwen3 has no upstream "
            f"SnapKV hijack. See snapkv-baseline-integration.md §6."
        )
    attn_impl = "flash_attention_2"
elif args.mode in {"duo_attention", "inf_llm"}:
    attn_impl = "eager"
else:
    attn_impl = "sdpa"
```

### Edit 6 — `dtype_kwarg`

**Anchor:** the `dtype_kwarg = (...)` ternary, immediately above the
`AutoModelForCausalLM.from_pretrained(...)` call.

`snap_kv` runs in transformers 4.37, which only accepts `torch_dtype=`.
Add to the old-transformers set:

```python
dtype_kwarg = (
    {"torch_dtype": torch.bfloat16}
    if args.mode in {"duo_attention", "inf_llm", "snap_kv"}
    else {"dtype": torch.bfloat16}
)
```

### Edit 7 — `config=` override (rename to `legacy_config_override`)

**Anchor:** the dict initialization `inf_llm_config_override = {}`
immediately above the `AutoModelForCausalLM.from_pretrained(...)` call.

Required: rename `inf_llm_config_override` -> `legacy_config_override`
throughout this section (now has 2+ consumers — infllm and snap_kv).
Same `rope_scaling` strip needed for SnapKV's transformers 4.37:

```python
legacy_config_override = {}
if args.mode == "inf_llm":
    from infllm import load_llama_config_stripped_rope
    legacy_config_override["config"] = load_llama_config_stripped_rope(args.base_model)
elif args.mode == "snap_kv":
    from snap_kv import load_llama_config_stripped_rope
    legacy_config_override["config"] = load_llama_config_stripped_rope(args.base_model)
```

And update the `from_pretrained(... **inf_llm_config_override)` call site
to `**legacy_config_override`. Single-callsite rename; mechanical.

### Edit 8 — Post-load `init_snap_kv` call

**Anchor:** the `if args.mode == "inf_llm":` block that calls
`init_inf_llm(model, INF_LLM_CONFIG)`. Verified the `init_inf_llm` import
is at line 527, the call at line 536. Insert the snap_kv branch
immediately after.

```python
elif args.mode == "snap_kv":
    from snap_kv import assert_llama_only, init_snap_kv
    from snap_kv.config import SNAPKV_CONFIG
    assert_llama_only(args.base_model)  # defense-in-depth (also gated in Edit 5)
    SNAPKV_CONFIG["base_model"] = args.base_model
    SNAPKV_CONFIG["window_size"] = args.snapkv_window_size
    SNAPKV_CONFIG["max_capacity_prompt"] = args.snapkv_max_capacity_prompt
    SNAPKV_CONFIG["kernel_size"] = args.snapkv_kernel_size
    SNAPKV_CONFIG["pooling"] = args.snapkv_pooling
    init_snap_kv(model, SNAPKV_CONFIG)
```

### Edit 9 — Generation loop: NO CHANGE

SnapKV uses HF `model.generate()` directly. The `else: model.generate(...)`
branch covers it. **Do not add a snap_kv elif.**

### Edit 10 — Per-sample reset: NO CHANGE

SnapKV's `prepare_inputs_for_generation_llama` resets `kv_seq_len` per
generate() call (`if past_key_values is None`). No `model.snap_kv_clear()`
needed.

### Edit 11 — Summary dump (config + RoPE caveat)

**Anchor:** the result-summary block where each mode contributes its config
to `summary[...]`. Insert after the existing
`summary["inf_llm_config"] = INF_LLM_CONFIG` line.

```python
elif args.mode == "snap_kv":
    from snap_kv.config import SNAPKV_CONFIG
    summary["snap_kv_config"] = SNAPKV_CONFIG
    summary["rope_caveat"] = {
        "transformers_version": "4.37.x",
        "rope_scaling_stripped": True,
        "comparable_to_tx5_baseline": False,
        "note": (
            "Llama-3.1 rope_type='llama3' is NOT applied; long-context "
            "(>8K) numbers are NOT directly comparable to tx 5.x baselines. "
            "SnapKV-vs-SnapKV deltas are valid; SnapKV-vs-baseline at 32K "
            "is not."
        ),
    }
```

This caveat field is **load-bearing** for downstream analysis: the result
JSON is the source of truth for plot scripts and reviewer notes, so
the unscaled-RoPE issue must travel with the data, not just live in this
plan doc.

---

## 5. `eval_longbench_v1.py` and `eval_longbench_v2.py` wiring

**Same shape as eval_ruler.py.** The eleven edits above translate 1-to-1
to both files at the analogous **pattern anchors** (numeric line ranges
below are advisory; verified at HEAD `b311efd`).

### Mandatory translations (all 11 edits applied, mirroring §4)

For each of `eval_longbench_v1.py` and `eval_longbench_v2.py`:
1. `--mode` choices: add `"snap_kv"` to the choices list.
2. SnapKV CLI args block + argparse invariant (`parser.error` on
   `max_capacity_prompt <= window_size`).
3. `run_name` default in the `if args.run_name is None:` chain.
4. `apply_monkey_patch`: add no-op `elif args.mode == "snap_kv": pass`.
5. `attn_impl` switch: **insert the Llama-only guard +
   `flash_attention_2` branch BEFORE the existing `eager`/`sdpa` ternary**
   (same 3-way block as §4 Edit 5; pre-load `SystemExit`).
6. `dtype_kwarg`: add `"snap_kv"` to the old-transformers set.
7. `config=` override: rename `inf_llm_config_override` ->
   `legacy_config_override`; add the snap_kv branch.
8. Post-load `init_snap_kv` call (mirror §4 Edit 8 verbatim).
9. Generation loop: NO CHANGE (SnapKV falls through to the standard
   `model.generate()` branch).
10. Per-sample teardown: NO CHANGE.
11. Summary dump: write **both** `summary["snap_kv_config"]` and the
    `summary["rope_caveat"]` block (verbatim from §4 Edit 11).

### `eval_longbench_v1.py` — informational line numbers (HEAD `b311efd`)
- `attn_impl` switch: line 867
- `legacy_config_override` (currently `inf_llm_config_override`): line 885
- `from_pretrained(... attn_implementation=attn_impl ...)`: line 900
- `init_inf_llm(model, INF_LLM_CONFIG)`: line 932 — insert snap_kv branch
  immediately after.
- `summary["inf_llm_config"] = INF_LLM_CONFIG`: line 991 — insert snap_kv
  branch (config + rope_caveat) immediately after.

### `eval_longbench_v2.py` — informational line numbers (HEAD `b311efd`)
- `attn_impl` switch: line 631
- `legacy_config_override` (currently `inf_llm_config_override`): line 653
- `from_pretrained(... attn_implementation=attn_impl ...)`: line 668
- `init_inf_llm(model, INF_LLM_CONFIG)`: line 700 — insert snap_kv branch
  immediately after.
- `summary["inf_llm_config"] = INF_LLM_CONFIG`: line 445 — insert snap_kv
  branch (config + rope_caveat) immediately after.

If any of these line numbers drift before the executor runs the edits,
fall back to the pattern anchors above — they are robust to line shifts.

### LongBench-specific divergences to flag:
- `tokenize_and_truncate(prompt, tokenizer, args.max_input_len, task)`
  is called in v1 (line 552) — SnapKV's prefill compression triggers
  only when `q_len >= max_capacity_prompt`. Long inputs (>2K tokens) will
  exercise the compression path; short inputs (e.g. `lcc`, `repobench-p`
  <2K tokens) will fall through unchanged. **This is correct behavior**
  — document it.
- LongBench v1 no-chat tasks (`trec`, `triviaqa`, `samsum`, `lcc`,
  `repobench-p`) bypass `tokenizer.apply_chat_template`. SnapKV doesn't
  care; same path.
- LongBench v2's MC eval reads only the first generated token; SnapKV's
  decode path (`else` branch in the patched forward) is the standard
  `past_key_value.update` flow. Compatible.

---

## 6. AIME25 / GPQA decision

**Out of scope for v1.**

Reason: `eval_aime25.py` and `eval_gpqa.py` are Qwen3-only (per
CLAUDE.md). SnapKV upstream has no Qwen3 hijack file
(`/home/yoongonkim/SnapKV/snapkv/monkeypatch/` contains only
`llama_hijack_4_37.py`, `mistral_hijack_4_37.py`, `mixtral_hijack_4_37.py`).

Out-of-scope work to enable later (NOT in this plan, ~3-5 days):
1. Port `llama_hijack_4_37.py` to a `qwen3_hijack_4_37.py` accounting
   for `q_norm` / `k_norm` (same gap that blocks ShadowKV from Qwen3).
2. Add a `replace_qwen3()` function calling
   `transformers.models.qwen3.modeling_qwen3.Qwen3FlashAttention2.forward = ...`.
3. Decide whether transformers 4.37 even ships a `Qwen3FlashAttention2`
   (it does NOT — Qwen3 was added in 4.50+). So Qwen3 + SnapKV requires
   a NEWER transformers env, which means re-validating the hijack
   against the post-4.45 forward signature change. Real work.

Document in the plan and CLAUDE.md baselines table that SnapKV is
"Llama 3.x only".

---

## 7. Conda env setup

**Env name: `snap_kv`**, created fresh. Do NOT reuse `infllm` (Option B
is fallback only).

```bash
# Driver / hardware reference: 4x RTX A6000, CUDA 12.9 driver, cu12.x torch.
# Lessons: feedback_flashinfer_torch_upgrade.md (--no-deps to keep torch on cu12.x),
#          project_infllm_env_broken.md (torch must NOT be cu130).

conda create -n snap_kv python=3.10 -y
source /home/tools/anaconda3/etc/profile.d/conda.sh
conda activate snap_kv

# torch first, pinned to cu12.x. Pick the version that has flash-attn 2.4.0
# wheels available. torch 2.1.2+cu121 was the original SnapKV target.
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# transformers MUST match the '4.37' substring at snapkv_utils.py:17.
# Pin 4.37.2 (matches the existing infllm env per project-memory; both
# 4.37.0 and 4.37.2 pass the substring check; 4.37.2 has minor bug fixes).
pip install transformers==4.37.2 accelerate sentencepiece protobuf

# flash-attn — try the README pin first.
pip install flash-attn==2.4.0.post1 --no-build-isolation

# NO upstream SnapKV install — vendored into baselines/snap_kv/_vendor.py.
# See §13 vendoring decision. (Skipping `pip install -e /home/yoongonkim/SnapKV`.)

# DCT-Page eval-time deps (subset; we don't need triton/flashinfer here).
pip install pyyaml tqdm pandas datasets
```

**`--no-deps` discipline:** `pip install -e /home/yoongonkim/SnapKV` on
its own would resolve `transformers>=4.34, torch, accelerate, ...` and may
upgrade torch off cu12.x. Use `--no-deps` after the explicit installs
above. (Same lesson as `feedback_flashinfer_torch_upgrade.md`.)

**Smoke test of env:**
```bash
cd /home/yoongonkim/DCT-Page  # so `from snap_kv._vendor import ...` resolves
python -c "
import sys; sys.path.insert(0, 'baselines')
import torch, transformers
from snap_kv._vendor import replace_llama   # vendored, NOT pip-install
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('transformers', transformers.__version__)
import flash_attn
print('flash_attn', flash_attn.__version__)
replace_llama()
print('replace_llama() OK')
"
```
Expected: `torch 2.1.2+cu121 cuda 12.1`, `transformers 4.37.2`,
`flash_attn 2.4.0.post1`, `replace_llama() OK`.

If `flash-attn==2.4.0.post1` build fails: try `flash-attn==2.5.x` (still
within SnapKV's 4.37-era flash interface), then `2.6.3` last (the version
in the existing `infllm` env). Each fallback requires re-running the
smoke test plus the prefix smoke test in §11.

---

## 8. Run script: `run_ruler_snapkv.sh`

Modeled exactly after `run_ruler_infllm.sh`. Activates the `snap_kv`
env, sweeps `max_capacity_prompt` (the main quality knob), and rewrites
`baselines/snap_kv/config.py` per iteration so the JSON summary captures
the exact config used.

Skeleton (full content goes in the actual script):

```bash
#!/bin/bash
# RULER Evaluation — SnapKV
# Sweeps max_capacity_prompt and pooling. Llama 3.x only.
set -e

SNAPKV_ENV_NAME="${SNAPKV_ENV_NAME:-snap_kv}"
CONDA_SETUP="${CONDA_SETUP:-/home/tools/anaconda3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SETUP" ]]; then
    source "$CONDA_SETUP"
    conda activate "$SNAPKV_ENV_NAME"
fi

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
PREPARE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model)   BASE_MODEL="$2"; shift 2 ;;
        --num_samples)  NUM_SAMPLES="$2"; shift 2 ;;
        --prepare)      PREPARE_FLAG="--prepare"; shift ;;
        *)              echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Llama only.
case "$(echo "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')" in
    *llama*)  MODEL_TAG="llama" ;;
    *) echo "SnapKV supports Llama only (got: $BASE_MODEL)"; exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-results_ruler/snap_kv/${MODEL_TAG}}"
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}"

WINDOW_SIZE=32
KERNEL_SIZE=5

for MAX_CAP in 1024 2048 4096; do
    for POOLING in avgpool maxpool; do
        RUN_NAME="${MODEL_TAG}_snap_kv_w${WINDOW_SIZE}_cap${MAX_CAP}_k${KERNEL_SIZE}_${POOLING}"
        echo ""
        echo "===================================================="
        echo "SNAPKV: max_capacity=${MAX_CAP} pooling=${POOLING}"
        echo "===================================================="
        python eval_ruler.py \
            --mode snap_kv \
            --base_model "$BASE_MODEL" \
            --skip_existing \
            $PREPARE_FLAG \
            --seq_lengths $SEQ_LENGTHS \
            --num_samples "$NUM_SAMPLES" \
            --snapkv_window_size "$WINDOW_SIZE" \
            --snapkv_max_capacity_prompt "$MAX_CAP" \
            --snapkv_kernel_size "$KERNEL_SIZE" \
            --snapkv_pooling "$POOLING" \
            --output_dir "$OUTPUT_DIR" \
            --run_name "$RUN_NAME"
    done
done
```

Path: `/home/yoongonkim/DCT-Page/run_ruler_snapkv.sh`. `chmod +x` it.

---

## 9. `init_snap_kv` per-layer iteration — verification

Walk: `for layer in model.model.layers: ... layer.self_attn`.

This is correct for Llama-family in transformers 4.37. Verified by:
- `transformers.models.llama.modeling_llama.LlamaModel.layers` is the
  `nn.ModuleList` of `LlamaDecoderLayer`.
- `LlamaDecoderLayer.self_attn` is the attention module that gets
  patched by `replace_llama()` (its class becomes the SnapKV-patched
  `LlamaFlashAttention2`).
- SnapKV's `init_snapkv(self)` (snapkv_utils.py:72-87) reads
  `self.config.{window_size,max_capacity_prompt,kernel_size,pooling}`,
  not the model-level config. We MUST set them on `layer.self_attn.config`,
  not `model.config` — except in transformers 4.37 the per-layer
  attention shares a reference to `model.config` (`self.config = config`
  in `LlamaAttention.__init__`). Setting on `layer.self_attn.config` is
  equivalent to setting on `model.config`, but doing the loop is
  defensive (matches upstream notebook examples) and survives any future
  per-layer config refactor.

Pre-flight sanity in the shim:
```python
assert hasattr(model, "model"), "expected LlamaForCausalLM (model.model.layers)"
assert len(model.model.layers) > 0, "no layers found"
```

---

## 10. CLAUDE.md update

Add a row to the baselines table (currently between `quest_attn` and
`shadow_kv`) at `/home/yoongonkim/DCT-Page/CLAUDE.md`:

```
| `snap_kv/` | SnapKV (prefill-time KV compression by attention-score topk) | Llama 3.x only | Requires `transformers==4.37.0`, `flash-attn==2.4.0`, upstream `SnapKV` installed editable. Config: `snap_kv/config.py` (`window_size`, `max_capacity_prompt`, `kernel_size`, `pooling`). Compresses prompt KV at the end of prefill; decode is standard. **Note:** directory must NOT be `snapkv` (collides with upstream package). |
```

Also append to the eval-script supported-modes column (RULER /
LongBench v1 / LongBench v2 lines): `snap_kv`. Keep AIME / GPQA columns
unchanged (Qwen3-only).

---

## 11. Verification plan

### Step A — Env smoke test (must pass before any eval)
```bash
conda activate snap_kv
cd /home/yoongonkim/DCT-Page
python -c "
import sys; sys.path.insert(0, 'baselines')
import torch, transformers, flash_attn
from snap_kv._vendor import replace_llama
from snap_kv import init_snap_kv, assert_llama_only, load_llama_config_stripped_rope
print('all imports OK')
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('flash_attn', flash_attn.__version__)
"
```

### Step B — Prefix smoke test (RULER niah_single_1, 8K, 5 samples) + firing-count gate

The vendored `_vendor.py` ships with a single load-bearing print line
inside `SnapKVCluster.update_kv` (the compression branch):
```
[snapkv] update_kv fired: q_len=<int>, cap=<int>
```
This print is the **acceptance gate** — it proves compression actually
ran (otherwise SnapKV silently behaves as a no-op when `q_len <
max_capacity_prompt`).

```bash
cd /home/yoongonkim/DCT-Page
# SNAPKV_TRACE=1 enables the gated firing-print (see §13).
# Production sweeps run WITHOUT this env var to keep tqdm bars clean.
SNAPKV_TRACE=1 python eval_ruler.py \
    --mode snap_kv \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --seq_lengths 8192 \
    --tasks niah_single_1 \
    --num_samples 5 \
    --snapkv_window_size 32 \
    --snapkv_max_capacity_prompt 2048 \
    --snapkv_pooling avgpool \
    --output_dir results_ruler/snap_kv_smoke \
    --run_name smoke_w32_cap2048 \
    2>&1 | tee results_ruler/snap_kv_smoke/run.log
```

Acceptance (all four must hold; firing count is a positive pass/fail,
not just an accuracy floor):
1. Process completes without traceback.
2. `results_ruler/snap_kv_smoke/smoke_w32_cap2048/summary.json` contains
   `"mode": "snap_kv"`, a non-zero score, AND
   `summary["rope_caveat"]["comparable_to_tx5_baseline"] == False`.
3. **Firing-count gate (positive, sample-tolerant):**
   ```bash
   FIRED=$(grep -c "^\[snapkv\] update_kv fired:" \
       results_ruler/snap_kv_smoke/run.log)
   # Llama-3.1-8B-Instruct has 32 layers. We require at least one full
   # sample to have crossed `max_capacity_prompt` (FIRED >= 32). The
   # short-circuit at snapkv_utils.py:42 is per-sample, so any sample
   # tokenizing below the cap contributes 0 firings — relaxing from
   # `== 160` to `>= 32` avoids spurious failure on borderline-length
   # samples while still proving compression actually executed.
   test "$FIRED" -ge 32 || { echo "FAIL: firing count $FIRED < 32 (no sample triggered compression)"; exit 1; }
   ```
   If `FIRED == 0`: the prefill never crossed `max_capacity_prompt` (lower
   the cap or raise context). For a strict-equality run (`FIRED == 32 *
   num_samples`), precondition the test by filtering to samples whose
   tokenized `input_ids` length is `>= max_capacity_prompt` first (e.g.
   add an explicit length-filter step to `--num_samples` selection so all
   admitted samples cross the threshold).
4. niah_single_1 score >= 80 (task is easy; SnapKV defaults should
   preserve it).

### Step C — Comparison test (baseline vs snap_kv, 5 samples)
```bash
python eval_ruler.py --mode baseline \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --seq_lengths 8192 --tasks niah_single_1 --num_samples 5 \
    --output_dir results_ruler/snap_kv_smoke --run_name baseline_smoke

# (snap_kv run from Step B already logged.)
```

Acceptance:
- |baseline_score - snap_kv_score| <= **2 pp** on niah_single_1 at
  `max_capacity_prompt=2048` and 8K context. Justification: SnapKV is
  keeping top-2K most-relevant tokens of 8K, the task has a single
  needle, and 8K is below the rope_scaling threshold so the unscaled-RoPE
  caveat does not yet bite. This is a tight gate by design.
- **Escalation path (if 2pp does not hold):**
  1. Confirm Step B firing count == 160 (rules out "compression never
     ran").
  2. Compare against `max_capacity_prompt={4096, 6144}` — if the gap
     closes monotonically, the issue is purely budget, not RoPE.
  3. If gap persists at `max_capacity_prompt=6144` (75 percent of 8K),
     escalate to a RoPE investigation: re-run baseline in tx 4.37 (with
     stripped RoPE) to isolate the RoPE-scaling effect from the SnapKV
     compression effect.
  4. Document the resolution in `.omc/plans/open-questions.md`.

### Step D — Wider sweep (after smoke passes)
Run the full `run_ruler_snapkv.sh` sweep at 32K. Compare against
existing `results_ruler/inf_llm/...` and `results_ruler/quest_attn/...`
runs at the same context length, **with the budget-translation note in
mind** (see Step E acceptance).

### Step E — LongBench v1 spot-check
```bash
python eval_longbench_v1.py --mode snap_kv \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --tasks narrativeqa qasper \
    --snapkv_max_capacity_prompt 2048 \
    --output_dir results/longbench_v1/snap_kv \
    --run_name llama_snap_kv_cap2048
```

Acceptance (revised — same-baseline-deltas, not cross-baseline parity):
- The run completes without traceback and produces `summary.json` with
  `summary["rope_caveat"]["comparable_to_tx5_baseline"] == False`.
- For internal SnapKV deltas (e.g. cap=1024 vs cap=2048 vs cap=4096),
  scores should be **monotone non-decreasing in capacity** (within +/- 2
  pp noise) — that is the meaningful pass/fail.
- **Cross-baseline (vs `inf_llm`) is NOT a gate.** InfLLM's `topk` is a
  per-block budget (selected blocks per query) while SnapKV's
  `max_capacity_prompt` is a total-tokens budget. They are not directly
  comparable. As a translation reference for plot scripts:
  ```
  approximate_token_budget(inf_llm) = topk * block_size
  # e.g. infllm topk=8, block_size=128 -> ~1024 tokens
  # roughly comparable to snap_kv max_capacity_prompt=1024
  ```
  Use this only for plot annotation; do NOT make pass/fail decisions on
  cross-baseline parity.

---

## 12. Risks / unknowns

1. **flash-attn 2.4.0 vs current toolchain.** torch 2.1.2+cu121 wheels
   for flash-attn 2.4.0.post1 exist on PyPI; building from source on
   the A6000 (CUDA 12.9 driver, cu12.x torch) has worked for the
   `infllm` env's flash-attn 2.6.3. Likelihood low; mitigation = fall
   back to 2.6.3 wheel from the existing `infllm` env recipe.

2. **Llama-3.1 `rope_scaling` parsing in transformers 4.37.** Already
   solved by `load_llama_config_stripped_rope`. Risk for SnapKV
   specifically: SnapKV does NOT replace RoPE (unlike InfLLM which uses
   its own `RotaryEmbeddingESM`), so the model uses transformers 4.37's
   default rotary embedding without llama3 scaling. At seq_len > 8K
   this MAY cause mild quality drop on Llama-3.1 vs the baseline run
   (which uses transformers 5.x with proper llama3 scaling).
   **Mitigation is now load-bearing in the output, not just documented:**
   §4 Edit 11 / §5 longbench equivalents write a `summary["rope_caveat"]`
   block into every SnapKV result JSON marking
   `comparable_to_tx5_baseline=False`. Plot scripts and reviewers must
   read this field before drawing SnapKV-vs-baseline comparisons at long
   context. Same caveat is accepted for `inf_llm` and `duo_attention`,
   but for those baselines the caveat lives only in this plan and
   CLAUDE.md — SnapKV is the first to surface it in result JSON.

3. **DynamicCache vs tuple KV.** transformers 4.37 returns
   `DynamicCache` from `model.generate()` by default. SnapKV's
   `prepare_inputs_for_generation_llama` handles both `Cache` and tuple
   paths (lines 145-153 of llama_hijack_4_37.py). Should work; verify
   in Step B.

4. **Qwen3 unsupported.** Documented in §6 and CLAUDE.md update.

5. **`prepare_inputs_for_generation_llama` × HF generate() in tx 4.37.**
   Resets `kv_seq_len = 0` on each fresh `past_key_values is None`
   call. The eval loop creates fresh `past_key_values` per sample (HF
   `generate()` does not pass `past_key_values` in). Should be correct.
   If repeated `generate()` calls share state, the patched
   `kv_seq_len` carry-over could corrupt the second sample. Smoke test
   covers this (5 samples in a row).

6. **`max_capacity_prompt < q_len` is required for compression to
   trigger** (`snapkv_utils.py:42-43`). At seq_len=8K with
   `max_capacity_prompt=2048`, it triggers. At seq_len=2K with
   `max_capacity_prompt=2048` it does NOT (returns full key/value). For
   short LongBench tasks this means SnapKV silently degrades to
   baseline — correct behavior, but document.

7. **Pooling kernel padding edge case.** `F.avg_pool1d(..., padding=2,
   stride=1)` with `kernel_size=5` produces same-length output. Topk
   gather indices reference `[..., :-window_size, :]`. Verified safe by
   reading `snapkv_utils.py:54-69`. No bug; just noting it for code
   review.

---

## Open questions (write to .omc/plans/open-questions.md)

- Should `assert_llama_only` and `load_llama_config_stripped_rope` be
  factored into `baselines/_shared.py` after this lands? (Out of scope
  v1; tracking.)
- Should we add a `--snap_kv_per_layer_capacity` JSON arg later for
  per-layer sweep papers (SnapKV authors note headroom there)?
- AIME25 / GPQA (Qwen3) — when do we want a Qwen3 hijack? Need
  separate scoping.

---

## 13. Vendoring spec — `baselines/snap_kv/_vendor.py`

We vendor (copy with attribution) the load-bearing upstream files
instead of `pip install -e /home/yoongonkim/SnapKV`. Adopted in
iteration 2 in response to the Architect's steelman.

**Sources (from `/home/yoongonkim/SnapKV/snapkv/monkeypatch/`):**
- `snapkv_utils.py` — `SnapKVCluster` class (~70 LOC) and
  `init_snapkv(self)` helper (~15 LOC).
- `llama_hijack_4_37.py` — patched
  `llama_flash_attn2_forward` (~140 LOC) and
  `prepare_inputs_for_generation_llama` (~30 LOC).
- `monkeypatch.py` — `replace_llama()` thunk (~10 LOC) that swaps
  both methods on `LlamaForCausalLM` and `LlamaFlashAttention2`.

**Total:** ~280 LOC, single file. Mistral and Mixtral hijacks NOT
vendored (out of scope per §6).

**Intra-package import rewrite (mandatory at concat time).** When the
three upstream files are concatenated into one `_vendor.py`, every
`from snapkv.monkeypatch.* import ...` line must be **deleted** (not
rewritten) — the symbols become local module-level definitions in the
same file, so the imports become no-ops and would otherwise fail at
module load (no `snapkv` package on `sys.path`). Concretely, delete:
- `from snapkv.monkeypatch.snapkv_utils import init_snapkv` (in
  upstream `llama_hijack_4_37.py:14`).
- Any `from snapkv.monkeypatch.snapkv_utils import ...` and
  `from snapkv.monkeypatch.llama_hijack_4_37 import ...` lines in
  upstream `monkeypatch.py`.
External imports (`torch`, `transformers.*`, `torch.nn.functional`,
etc.) are kept verbatim. Order the concatenated definitions so
`SnapKVCluster` / `init_snapkv` (from `snapkv_utils.py`) appear
**before** `llama_flash_attn2_forward` (from `llama_hijack_4_37.py`),
which appears before `replace_llama` (from `monkeypatch.py`).

**Required diff vs upstream — one line, load-bearing:**

Inside `SnapKVCluster.update_kv` (corresponds to
`snapkv_utils.py:38-70` upstream), in the **else branch** (the actual
compression path; entered when `q_len >= self.max_capacity_prompt`),
add a single print at the **top** of the branch:

```python
# At top of _vendor.py: `import os` (vendor module already imports os
# upstream, but ensure it survives the concat step).
#
# In SnapKVCluster.update_kv, immediately after the
# `if q_len < self.max_capacity_prompt: return key_states, value_states` line:
else:
    if os.environ.get("SNAPKV_TRACE") == "1":
        print(f"[snapkv] update_kv fired: q_len={q_len}, "
              f"cap={self.max_capacity_prompt}", flush=True)
    # ... rest of the upstream compression path ...
```

This print is the basis of the Step B firing-count acceptance gate
(`grep -c "^\[snapkv\] update_kv fired:" run.log >= num_layers`). It is
the ONLY diff vs upstream; everything else is verbatim.

**Why gated by `SNAPKV_TRACE=1`.** The print fires `num_layers *
num_samples * num_seq_lengths` times during a sweep (e.g. 32 * 25 * 1 =
800 lines per RULER task at 32K), garbling the `tqdm` progress bars in
`eval_ruler.py` / `eval_longbench_v{1,2}.py`. Production sweeps run
**without** `SNAPKV_TRACE` → no stdout flood. The Step B smoke command
sets `SNAPKV_TRACE=1` explicitly to enable the firing-count gate.

**Header / attribution:** `_vendor.py` opens with
```python
"""Vendored from FasterDecoding/SnapKV @ <commit-sha>.

Source files combined into one module:
- snapkv/monkeypatch/snapkv_utils.py
- snapkv/monkeypatch/llama_hijack_4_37.py
- snapkv/monkeypatch/monkeypatch.py

Diffs vs upstream:
- Added one `print('[snapkv] update_kv fired: ...')` line at the top
  of the compression branch in SnapKVCluster.update_kv (load-bearing
  for the Step B firing-count acceptance gate; see
  .omc/plans/snapkv-baseline-integration.md §13).

License: Apache-2.0 (upstream LICENSE preserved at
  baselines/snap_kv/UPSTREAM_LICENSE.txt).
"""
```

**Pin to a commit:** capture the upstream commit SHA at vendor time
(`cd /home/yoongonkim/SnapKV && git rev-parse HEAD`) and write it into
the docstring above. This is the reproducibility anchor.

**Public re-export from `__init__.py`:**
```python
# Note: upstream's `init_snapkv` (one word, the per-attn-module helper)
# is renamed to `_init_snapkv` on re-export to avoid being mistaken for
# our public shim `init_snap_kv` (two words). Only `replace_llama` and
# `SnapKVCluster` need to surface for debugging; `_init_snapkv` is
# kept private (leading underscore) and is invoked internally by the
# patched forward via `if not hasattr(self, "kv_cluster")` guard.
from snap_kv._vendor import (
    replace_llama,
    SnapKVCluster,
    init_snapkv as _init_snapkv,
)
```
Keep these out of `__all__` so callers prefer the public
`init_snap_kv(model, cfg)` shim — but make them available for
debugging.

---

## ADR

**Decision.** Add `baselines/snap_kv/` shim mirroring
`baselines/infllm/` shape, with **vendored upstream code** at
`baselines/snap_kv/_vendor.py` (~280 LOC: `SnapKVCluster`,
`init_snapkv`, `llama_hijack_4_37`, `replace_llama`). Wire SnapKV as a
new `--mode snap_kv` in RULER and LongBench v1/v2 evals. Run in a
dedicated `snap_kv` conda env pinned to `transformers==4.37.2` +
`flash-attn==2.4.0.post1`. Llama 3.x only in v1; AIME / GPQA out of
scope.

**Drivers.**
1. `transformers==4.37.x` pin (load-bearing for the hijack).
2. No Qwen3 upstream support.
3. Mandatory `attn_implementation="flash_attention_2"` (the hijack
   patches `LlamaFlashAttention2.forward`).
4. Strict reduction in moving parts (vendoring vs `pip install -e`):
   the upstream is dormant (last commit > 18 months old), so a frozen
   vendored copy is more reproducible than a live editable install.

**Alternatives considered.**
- **Pip-install upstream (`pip install -e /home/yoongonkim/SnapKV`).**
  Rejected as v1 default (now a v2 fallback only). Pros: zero copy.
  Cons: relies on a dormant upstream, an extra editable-install path,
  and the local clone could drift; also forces the local dir name
  collision concern (`baselines/snapkv/` vs upstream `snapkv` package).
- **Reuse `infllm` env.** Rejected: flash-attn version drift (2.4.0
  vs 2.6.3); kept as a fallback if `flash-attn==2.4.0.post1` build
  fails on the box.
- **Re-implement on top of eager Llama attention in the main `DCT_Page`
  env.** Rejected: ~1 week port + ongoing maintenance vs day-one shim.

**Why chosen.** Vendored Option A is a strict superset of pip-install
Option A's review surface — it adds one explicit `_vendor.py` file
whose contents the executor reads at vendor time, and lets us ship
the load-bearing one-line print diff (firing-count gate) without a
patched fork. It keeps a 1-to-1 review surface against the existing
`infllm` baseline (same shim API), isolates flash-attn 2.4.0 to its
own env, surfaces the unscaled-RoPE caveat in result JSON, and lands
the baseline in one focused change. Pre-mortem mitigations are all
single-line fallbacks (`config=` strip, `attn_implementation` switch,
env fallback to infllm's flash-attn 2.6.3, optional fresh-cache
walk for `kv_seq_len`).

**Consequences.**
- +3 files in `baselines/snap_kv/` (`__init__.py`, `config.py`,
  `_vendor.py`), +1 file `run_ruler_snapkv.sh`.
- ~30 LOC across each of `eval_ruler.py`, `eval_longbench_v1.py`,
  `eval_longbench_v2.py` (the +5 LOC over iter-1 are the argparse
  invariant + Llama-only guard at the attn_impl switch + rope_caveat
  summary entry).
- One new conda env (`snap_kv`, ~6 GiB). NO `pip install -e
  /home/yoongonkim/SnapKV` — vendored.
- ~280 LOC vendored code with a one-line firing-print diff that is
  load-bearing for the §11 Step B acceptance gate. Diff vs upstream is
  documented in §13.
- New runtime opt-in env var `SNAPKV_TRACE=1` enables the firing-print
  (off by default to keep `tqdm` bars clean in production sweeps; the
  smoke command in §11 Step B sets it explicitly). Sweeps run with
  `SNAPKV_TRACE` unset see no SnapKV stdout pollution.
- All SnapKV result JSONs carry a `summary["rope_caveat"]` block
  marking `comparable_to_tx5_baseline=False`. Plot/analysis scripts
  must respect it.
- CLAUDE.md baselines table grows by one row.
- Future port to Qwen3 / AIME / GPQA is non-trivial (separate ADR).

**Follow-ups.**
- Factor `_shared.py` for `assert_llama_only` /
  `load_llama_config_stripped_rope` once a third baseline needs the
  same code.
- Speed/profile pass: SnapKV's prefill-only compression should make
  decode tok/s very close to baseline; worth a `speed/` measurement
  for the comparison table.
- Optional Qwen3 hijack port (separate plan).
- Backport `summary["rope_caveat"]` to the existing `inf_llm` and
  `duo_attention` modes for consistency (small, mechanical).
- If upstream SnapKV ever revives, revisit pip-install Option A —
  vendoring buys reproducibility today, not lock-in.

---

## Acceptance criteria for "plan executed"

1. `baselines/snap_kv/{__init__.py,config.py,_vendor.py}` exist and are
   importable from the `snap_kv` env. `_vendor.py` carries the upstream
   commit SHA in its docstring and the one-line firing-print diff
   (§13).
2. `run_ruler_snapkv.sh` exists, executable, activates the env,
   sweeps `max_capacity_prompt` and `pooling`.
3. Smoke test (§11 Step B) **all four sub-acceptances pass**:
   no traceback; `summary["mode"] == "snap_kv"`; firing-count gate
   `grep -c "^\[snapkv\] update_kv fired:" run.log >= 32` (at least one
   full sample crossed `max_capacity_prompt`); niah_single_1 score >= 80.
4. Comparison test (§11 Step C) holds the tightened 2pp gate, OR the
   escalation path is documented in `.omc/plans/open-questions.md`.
5. `summary["rope_caveat"]["comparable_to_tx5_baseline"] == False` is
   present in every SnapKV result JSON (RULER, LongBench v1, LongBench
   v2).
6. `python eval_ruler.py --mode snap_kv --help` shows the four
   `--snapkv_*` flags. The argparse invariant
   `max_capacity_prompt > window_size` triggers a clean
   `parser.error(...)` (no traceback) when violated.
7. The Llama-only guard at the `attn_impl` switch fires pre-load on
   `--mode snap_kv --base_model Qwen/...` (no model load attempted).
8. CLAUDE.md baselines table includes the SnapKV row.
9. No regression in any existing `--mode` choice (run an existing
   `--mode baseline` smoke check after the eval-script edits, AND run
   one `--mode inf_llm` smoke to verify the
   `inf_llm_config_override` -> `legacy_config_override` rename did
   not break that baseline).
