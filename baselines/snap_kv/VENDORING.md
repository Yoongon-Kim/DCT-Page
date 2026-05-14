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

## Live path vs reference path

Two parallel entry points exist into the SnapKV algorithm. Both share ONLY
`SnapKVCluster` from `upstream/snapkv/monkeypatch/snapkv_utils.py`.

```
                        baselines/snap_kv/
                              |
        +---------------------+---------------------+
        |                                           |
   LIVE PATH                                  REFERENCE PATH
   (transformers==5.2.0,                      (transformers==4.37.2,
    DCT_Page conda env;                        snap_kv conda env;
    Llama 3.x + Qwen3)                         Llama 3.x only)
        |                                           |
        v                                           v
   __init__.py                              upstream/__init__.py
   init_snap_kv(model, cfg)                 init_snapkv / replace_llama
        |                                           |
        v                                           v
   patch_v5.py                              upstream/snapkv/monkeypatch/
   _snapkv_attention_forward                 monkeypatch.replace_llama()
   replace_llama_v5(model)                          |
   replace_qwen3_v5(model)                          v
        |                                  upstream/snapkv/monkeypatch/
        |                                  llama_hijack_4_37.py
        |                                          |
        +-----------------+------------------------+
                          |
                          v
              upstream/snapkv/monkeypatch/snapkv_utils.py
              SnapKVCluster (algorithm only — DO NOT MODIFY)
```

The LIVE PATH (`patch_v5.py`) is hand-authored, non-upstream glue that hooks
into modern `LlamaAttention.forward` and `Qwen3Attention.forward`. It reuses
`SnapKVCluster` verbatim (head-count-agnostic; only T-axis assertion at
snapkv_utils.py:41). GQA reduction is strategy (c) — group-mean Q only;
K and V stay un-repeated at `num_kv_heads`.

The REFERENCE PATH is the historical 4.37.2 hijack code. It is kept as a
museum for diffability against the upstream repo. Do NOT modify anything
under `upstream/`.

## Re-vendoring procedure

    BUMP=<new commit>
    rsync -a --delete /path/to/SnapKV/snapkv/monkeypatch/{snapkv_utils,llama_hijack_4_37,monkeypatch}.py \
        baselines/snap_kv/upstream/snapkv/monkeypatch/
    # then re-apply the 6 edits above (or merge them via patch).
    # In particular: edit #6 (SNAPKV_TRACE) MUST be reapplied to
    # snapkv_utils.py — the rsync overwrite drops it by default.
    # bump the "Upstream commit" line above
    # re-run Step 5 smoke verifications (in this plan).
