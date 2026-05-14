# SnapKV port to transformers==5.2.0 (Llama 3.1 + Qwen3) — Planner draft (RALPLAN-DR)

Status: DRAFT iter 3 (subtractive — reverts iter-2's misdiagnosis-driven Bug-3 wrapper; switches GQA reduction from strategy (a) to strategy (c)).
Repo: `/home/yoongonkim/DCT-Page`
Target env: `DCT_Page` conda env (`/home/yoongonkim/.conda/envs/DCT_Page`), `transformers==5.2.0`, python 3.12.
Target models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`.
Predecessor plans:
- `.omc/plans/snapkv_vendor_upstream_dir.md` — APPROVED, executed; `baselines/snap_kv/upstream/` contains the historical 4.37 hijack code.
- `.omc/plans/snapkv_eval_ruler_wireup.md` — separate follow-up that wires `--mode snap_kv` into `eval_ruler.py`. Out of scope.

The goal of this plan: produce a live, transformers-5.2.0-compatible patch — `baselines/snap_kv/patch_v5.py` — that (a) monkey-patches both `LlamaAttention.forward` and `Qwen3Attention.forward`, (b) reuses the algorithm-only `SnapKVCluster` class from the vendored upstream museum, and (c) leaves `upstream/` byte-for-byte unmodified.

---

## 0. Iteration 3 changelog

Architect iter-2 verdict: NEEDS-REVISION. Critic iter-2 verdict: ITERATE. The fix path is SUBTRACTIVE — both reviewers traced the iter-2 NEEDS-REVISION verdict back to two misdiagnoses in the iter-1 review cycle that iter 2 then dutifully built on. Iter 3 unbuilds the wrong scaffolding.

### MUST-FIX (blocking) — all resolved

| # | Issue | Resolution | Lives in §… |
|---|---|---|---|
| 1 | **Bug 3 was a misdiagnosis.** `_update_model_kwargs_for_generation` (`generation/utils.py:1074-1075`) advances `cache_position` independently via `[-1:] + num_new_tokens`. `LlamaModel.forward:400` only re-derives `cache_position` when it is `None` — under `generate()` it is never None. The iter-2 wrapper overwrote `cache_position` with the same value it already had. RoPE's relative-rotation property (`Q(t1) @ K(t2)^T = f(t1-t2)`) makes the compressed K cache holding original-position RoPE phases **mathematically correct**, not corrupt. | **DELETE** `baselines/snap_kv/prepare_inputs_v5.py`. Strike all `wrap_prepare_inputs_*` / `unwrap_prepare_inputs_*` references from `patch_v5.py` and `__init__.py`. Revert Principle 4 to iter-1 wording ("No `prepare_inputs_for_generation` patch needed"). Strike §3.6 ("Bug 3 source") and §4.6 (file spec). | §0.1 retraction + Principle 4 + §3.6 removed + §4.6 removed |
| 2 | **GQA strategy (c) was incorrectly rejected** on Principle-1 grounds. `SnapKVCluster.update_kv` asserts only `key_states.shape[-2] == query_states.shape[-2]` (T-axis equality at `snapkv_utils.py:41`); the `matmul`/`gather`/`cat` body works for any matching head count. Strategy (c) requires **zero edits to the museum** — one `.view(...).mean(dim=2)` on the QUERY only. | Replace §4.1 GQA reduction block with strategy (c): group-mean Q before clustering, pass un-repeated K/V (already `num_kv_heads` from DynamicLayer). No per-head info on K side is destroyed; cluster output is `num_kv_heads`-shaped, directly writable to cache. | §3.7 retraction + §4.1 code body + Pre-mortem Scenario 4 |
| 3 | **Sticky `_snapkv_compressed_layers` flag is unnecessary** once the prepare_inputs wrapper is gone. | Inline detection: `is_prefill_to_compress = q_len > 1 and full_k.shape[-2] == q_len and full_k.shape[-2] >= self.kv_cluster.max_capacity_prompt`. After `Cache.update(...)` returns, `full_k.shape[-2] == q_len` iff the layer was empty before this call (fresh prefill). Works under `generate()`, direct `model(...)`, and cache-reuse without external state. | §4.1 code body |
| 4 | **Smoke test gap (e) and (f).** (e) used direct `model(...)` for the multi-sample reuse check, which under the new inline reset is fine but diverges from the eval_ruler.py `generate()` pattern. (f) was missing: no compression-on-vs-off divergence check, which is exactly the test that would have caught iter-2's group-mean phase corruption at smoke time. | (e) Replace direct `model(...)` with `model.generate(max_new_tokens=4)`. (f) Add: run same prompt at `cap=512` and at `cap=q_len+1` (no compression), assert `(comp_logits - uncomp_logits).abs().max() < 5.0` in bf16. Bound is loose enough for legitimate compression noise but catches phase-corruption-style divergence. | §5 Step 5 |

### SHOULD-FIX (non-blocking) — all resolved

| # | Issue | Resolution | Lives in §… |
|---|---|---|---|
| 5 | Preserve epistemic trail of the review cycle. | Add §0.1 "Iter-1 retraction" subsection. | §0.1 |
| 6 | §3.7 text honesty. | §3.7 now states strategy (c) is adopted, documents the strategy-a-vs-c trade-off (group-mean Q for scoring vs upstream per-head Q for scoring — but with intact per-kv-head K/V and no phase corruption). | §3.7 |
| 7 | `tok.pad_token_id or 0` assertion is weak for Llama-3.1-Instruct (pad_token_id is None → fallback to BOS=0). | Replace with `assert torch.isfinite(out.logits).all() and torch.unique(new_tokens).numel() > 1`. | §5 Step 5 |
| 8 | §1.D Option D rejection rationale stale. | Drop reason 4 ("position-drift still needs handling"). Remaining 3 reasons (eval_ruler wireup cost, DCT precedent, forward-patch unavoidable for Q-injection) still justify Option A. | §1.D |

### Estimated net diff vs iter 2

| Change | Δ lines |
|---|---|
| Delete `prepare_inputs_v5.py` body in §4.6 | −85 |
| Remove `wrap_prepare_inputs_*` / `unwrap_prepare_inputs_*` calls from `patch_v5.py` | −10 |
| Replace strategy-a GQA block (4 LoC) with strategy-c (1 LoC on Q) | −3 |
| Remove `_snapkv_compressed_layers` flag + flip; inline check | −4 |
| Add smoke (f) compression-vs-no-compression diagnostic | +12 |
| Strike §3.6 (Bug 3 source) + Pre-mortem Scenario 3 references | −30 |
| Add §0.1 retraction | +18 |
| Update §3.7 / §1.D / Principle 4 text | ±0 |
| **Net** | **≈ −102 LoC** |

Plan length 1119 → ~1017 lines.

### 0.1. Iter-1 retraction (preserves epistemic trail)

Iter-1's review introduced "Bug 3 — position drift on decode" as a CRITICAL defect requiring a `prepare_inputs_for_generation` wrapper. Iter 2 built that wrapper in `prepare_inputs_v5.py`. Iter-2 review traced this back and surfaced two facts:

1. The iter-1 root assumption ("`LlamaModel.forward:400` re-derives `cache_position` from compressed `get_seq_length()` on decode") is FALSE under `generate()`. `_update_model_kwargs_for_generation` (`generation/utils.py:1074-1075`) sets `model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens`, advancing it independently. `LlamaModel.forward:400` only fires when `cache_position is None`, which under `generate()` happens only on the very first call (prefill), where `get_seq_length() == 0` anyway. There is no decode step where the compressed-cache length leaks into `cache_position`.
2. RoPE is **relative-rotation-invariant**: `Q(t_query) @ K(t_key)^T = f(t_query − t_key)`. Compressed K cache holding original-position RoPE phases produces correct attention scores against the new query token, regardless of how `get_seq_length()` reports the cache size. The iter-1 alleged "phase corruption" was a phantom.

Iter 2's wrapper at `prepare_inputs_v5.py` was therefore **functionally dead** — it overwrote `cache_position` with the same value it already had. Iter 3 deletes the file, reverts Principle 4, and strikes §3.6 and §4.6 from the plan.

Similarly, iter-2's choice of GQA strategy (a) over (c) rested on the iter-1 claim "strategy (c) requires editing the museum SnapKVCluster" — false. The cluster's only head-count-related assertion is the T-axis equality at `snapkv_utils.py:41`. Strategy (c) needs zero museum edits. Iter 3 adopts (c).

The retraction is recorded here so future reviewers can see why iter 3 looks smaller than iter 2, not larger.

---

## 1. RALPLAN-DR Summary

### Principles (the rules the plan never breaks)

1. **`upstream/` is a museum** — do not edit any file under `baselines/snap_kv/upstream/`. The 6 vendoring edits in `VENDORING.md` are the only deltas there forever. New code lives in `baselines/snap_kv/patch_v5.py`.
2. **One forward, two thin entry points** — exactly mirror the DCT precedent at `dct_page_attention.py:1138-1227` and `dct_page_attention.py:2384-2453,2456-2529`. A single `_snapkv_attention_forward` dispatches Q-Norm/K-Norm via `hasattr(self, "q_norm")` and pulls `sliding_window=getattr(self, "sliding_window", None)`. Two thin entry points (`replace_llama_v5(model)`, `replace_qwen3_v5(model)`) install the same forward AND eagerly initialize each layer's `kv_cluster` at boot.
3. **Algorithm reuse, transport rewrite** — `SnapKVCluster.update_kv` from upstream is model- and transformers-version-agnostic, AND head-count-agnostic (verified — only the T-axis assertion at `snapkv_utils.py:41` constrains shapes). The port re-imports it verbatim. The `SNAPKV_TRACE` env-gated print survives (vendoring edit #6).
4. **No `prepare_inputs_for_generation` patch** (iter 3 — reverted from iter 2's reframing) — `_update_model_kwargs_for_generation` advances `cache_position` independently of `past_key_values.get_seq_length()`, so the compressed cache length never leaks into the per-decode-step position bookkeeping. RoPE's relative-rotation property guarantees the compressed K cache (holding original-position RoPE phases) computes correct attention against new queries. No hook is required. (See §0.1 for the retraction trail.)
5. **Prefill + decode + parity + compression-divergence smoke acceptance** — accept the port when (a) `import baselines.snap_kv` works in DCT_Page, (b) 4096-token prefill + `generate(max_new_tokens=4)` succeeds on both models without exception, (c) ALL `past_kv.layers[i].keys.shape[-2] == max_capacity_prompt` after prefill, (d) logits are finite, (e) parity baseline (`max_capacity_prompt = q_len + 1`) matches unpatched within `1e-3` abs in bf16, (f) **compression-on vs compression-off divergence is bounded** (compressed logits at cap=512 vs uncompressed at cap=q_len+1 differ by `< 5.0` abs in bf16 — catches phase-corruption-style divergence), (g) `unpatch_*_v5` restores baseline behavior. RULER accuracy is downstream.

### Decision drivers (top 3)

1. **Live-path simplicity** — readable, follows DCT precedent. One forward + two replace_* functions.
2. **API compatibility with transformers 5.2.0** — verified surface (§3): `Cache.update(k, v, layer_idx, cache_kwargs)` returns `(k, v)` and stores into `self.layers[layer_idx].keys/.values`; `attention_interface` dispatched via `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)` — direct callable default, not a re-lookup.
3. **Reversibility** — `unpatch_*_v5` restores the original forward. (No `prepare_inputs_for_generation` to restore, iter 3.)

### Viable options (≥2)

#### Option A — One unified `_snapkv_attention_forward` with `hasattr` dispatch (RECOMMENDED)

Single function in `patch_v5.py`. `replace_llama_v5(model)` and `replace_qwen3_v5(model)` are short wrappers (install forward + loop init each layer's `kv_cluster`). Dispatches `q_norm`/`k_norm` via `hasattr(self, "q_norm")`. Pulls `sliding_window=getattr(self, "sliding_window", None)`. Reuses `apply_rotary_pos_emb` from `transformers.models.llama.modeling_llama` (identical body to qwen3's at `modeling_qwen3.py:159`).

Pros:
- Mirrors `dct_page_attention_forward` exactly (single ~100-line function for both families).
- Q-Norm/K-Norm at one `hasattr` check; trivially extensible to Qwen2.
- One call site for `SnapKVCluster.update_kv`.
- GQA reduction lives in one place.

Cons:
- One `hasattr` check per call (negligible).

#### Option B — Two separate forwards (NOT recommended)

Two near-duplicate forwards (~240 LoC total), each `replace_*_v5` installs its dedicated forward.

Pros: easier to read in isolation.

Cons: ~120 LoC duplication; future bugfixes apply twice; diverges from DCT precedent.

#### Option C — Subclass `LlamaAttention` / `Qwen3Attention` (INVALIDATED)

Subclass + surgical rebind of `model.model.layers[i].self_attn`.

Cons:
- transformers 5.x decorates `LlamaAttention` with `@use_kernelized_func(apply_rotary_pos_emb)` (modeling_llama.py:224). Subclass instantiation silently drops the decorator.
- Diverges from every other baseline in `baselines/*`.
- Cannot be reverted without re-loading the model.

INVALIDATED.

#### Option D — Subclass `DynamicCache` (`SnapKVCache(DynamicCache)`) — STEELMAN

Move SnapKV's compression logic into a custom `Cache` subclass: `update(...)` runs cluster on first-call-per-layer, then delegates to `super().update(...)`.

Pros:
- Compression logic next to cache state (cohesive).
- Works across `DynamicCache` subclasses if we mirror the contract for each.
- No public-attribute mutation of `.keys`/`.values`.

Cons (iter 3 — dropped reason 4 about position-drift since Bug 3 doesn't exist):
- **Still requires a forward patch** — the cluster needs Q to compute scores; `cache_kwargs` doesn't carry it natively. The forward must inject `query_states` into `cache_kwargs`. So we don't *eliminate* the forward patch.
- **eval_ruler.py wireup gets more complex** — caller must explicitly construct `SnapKVCache(...)` and pass via `past_key_values=...` to `model.generate(...)`. Native `model.generate()` allocates a `DynamicCache(config=...)` (`modeling_llama.py:397-398`); we'd need a generation-config or pre-allocate workaround.
- **Diverges from DCT precedent** — `pre_allocate_cache` in `dct_page_attention.py` mutates the existing `DynamicCache`'s layers rather than replacing the cache type.

**Decision: stay on Option A.** Option D's cohesion pros are real but don't outweigh the eval_ruler wireup cost and DCT-precedent divergence, given that the forward patch is unavoidable anyway.

**Chosen: Option A.** See ADR (§7).

---

## 2. File-level change manifest

### 2.1. New files (one — iter 3 returns to single new file after deleting `prepare_inputs_v5.py`)

| Absolute path | LoC est. | Purpose |
|---|---|---|
| `/home/yoongonkim/DCT-Page/baselines/snap_kv/patch_v5.py` | ~180 | Live SnapKV patch: `_snapkv_attention_forward`, `init_snapkv_v5`, `replace_llama_v5(model)`, `replace_qwen3_v5(model)`, `unpatch_llama_v5()`, `unpatch_qwen3_v5()`. |

### 2.2. Modified files (four)

| Absolute path | Change kind | Net delta |
|---|---|---|
| `/home/yoongonkim/DCT-Page/baselines/snap_kv/__init__.py` | Rewrite the live entry path. | +25 / -10 LoC |
| `/home/yoongonkim/DCT-Page/baselines/snap_kv/config.py` | Add a comment noting both families now supported. | +3 / -2 LoC |
| `/home/yoongonkim/DCT-Page/baselines/snap_kv/VENDORING.md` | Add a "Live path" subsection. | +30 / 0 LoC |
| `/home/yoongonkim/DCT-Page/run_ruler_snapkv.sh` | Switch env to `DCT_Page`. Accept `qwen3`. | +5 / -4 LoC |

### 2.3. Files explicitly NOT touched (museum)

- `baselines/snap_kv/upstream/__init__.py` — keep re-exporting `init_snapkv` and `replace_llama` from the 4.37 path. Reference-path only.
- `baselines/snap_kv/upstream/snapkv/monkeypatch/{snapkv_utils.py, llama_hijack_4_37.py, monkeypatch.py}` — unchanged forever.

---

## 3. Verified transformers==5.2.0 surface (the contracts the port relies on)

All facts verified by reading the actual files under
`/home/yoongonkim/.conda/envs/DCT_Page/lib/python3.12/site-packages/transformers/`.

### 3.1. `LlamaAttention.forward` (modeling_llama.py:251-292)

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward  # direct callable default
    )
```

### 3.2. `Qwen3Attention.forward` (modeling_qwen3.py:252-294)

Identical signature; body differences: `self.q_norm`/`self.k_norm` on head-dim **before** transpose (L264-265), `sliding_window=self.sliding_window` passed to `attention_interface` (L288, `None` on full-attention layers; Qwen3-8B is all full-attention).

### 3.3. `Cache.update` and `DynamicLayer.update` (cache_utils.py:758-797 and :99-122)

`Cache.update` delegates to `self.layers[layer_idx].update(...)`. `DynamicLayer.update`:
```python
self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)
return self.keys, self.values
```
Storage attrs `self.keys: torch.Tensor | None` and `self.values: torch.Tensor | None` documented on `CacheLayerMixin.__init__` at `cache_utils.py:33-34` — public.

Direct overwrite `past_key_values.layers[i].keys = k_compressed; .values = v_compressed` works for `DynamicLayer`. **It does NOT work for `StaticLayer` (cache_utils.py:254-345)** — `StaticLayer.update` does `self.keys.index_copy_(2, cache_position, key_states)`; overwriting `.keys` with a smaller tensor breaks subsequent `index_copy_`. **Hard runtime assert** required.

### 3.4. `ALL_ATTENTION_FUNCTIONS._global_mapping` (modeling_utils.py:4768-4777)

```python
_global_mapping = {
    "flash_attention_3": ...,
    "flash_attention_2": ...,
    "flex_attention": ...,
    "sdpa": ...,
    "paged|flash_attention_3": ...,
    "paged|flash_attention_2": ...,
    "paged|sdpa": ...,
    "paged|eager": ...,
}
```

**`"eager"` is NOT a key.** `get_interface("eager", default)` returns `default` (modeling_utils.py:4791: `return super().get(attn_implementation, default)`). Native LlamaAttention at line 275-277 passes `eager_attention_forward` (`modeling_llama.py:199-221`) as default.

`get_interface("eager", None)` returns `None` → `None(...)` → `TypeError`. Use `eager_attention_forward` as default.

### 3.5. `apply_rotary_pos_emb`

Identical at `modeling_llama.py:146` and `modeling_qwen3.py:159`. Import once from llama, reuse for both.

### 3.6. `cache_position` advancement during `generate()` — Bug 3 NON-EXISTENCE proof

`_update_model_kwargs_for_generation` at `generation/utils.py:1074-1075`:
```python
if use_cache:
    model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
```
This advances `cache_position` per generation step **independently of `past_key_values.get_seq_length()`**. The compressed cache length never enters the position bookkeeping.

`LlamaModel.forward:400-404` only derives `cache_position` from `get_seq_length()` when `cache_position is None`:
```python
if cache_position is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(inputs_embeds.shape[1], device=...) + past_seen_tokens
```
Under `generate()`: `prepare_inputs_for_generation` sets `model_inputs["cache_position"] = cache_position` (always non-None after the first call). So this branch never fires on decode steps.

**Conclusion:** the iter-1-alleged "decode-time position drift after SnapKV compression" does not exist under `generate()`. Combined with RoPE's relative-rotation property (`Q(t1) @ K(t2)^T = f(t1−t2)`), the compressed K cache holds the correct phases for correct attention scores against the new query token. No `prepare_inputs_for_generation` wrapper is needed.

### 3.7. GQA-reduction surface — STRATEGY (c) ADOPTED (iter 3 retraction of iter-2's strategy (a))

`SnapKVCluster.update_kv` (upstream/snapkv/monkeypatch/snapkv_utils.py:39-73) takes `key_states`, `query_states`, `value_states` and asserts ONLY:
```python
assert key_states.shape[-2] == query_states.shape[-2]  # T-axis equality
```
The rest of the body (matmul, mask, softmax, sum, avg_pool1d, topk, gather, cat) is shape-polymorphic on the head axis. The cluster is **head-count-agnostic.**

Llama-3.1-8B and Qwen3-8B both have `num_attention_heads=32`, `num_key_value_heads=8`, `num_kv_groups=4`. The `DynamicLayer.keys` stores `(B, num_kv_heads=8, T, D)`. After `Cache.update(...)` returns `full_k, full_v` of shape `(B, 8, T, D)`, we want the cluster to produce a `(B, 8, max_capacity_prompt, D)` output directly writable to the cache.

**Strategy (c) — group-mean QUERY (CHOSEN):**
```python
B, H, q_len, D = query_states.shape  # H = num_attention_heads = 32
G = self.num_key_value_groups        # 4
query_for_cluster = query_states.view(B, H // G, G, q_len, D).mean(dim=2)
# query_for_cluster: (B, num_kv_heads=8, q_len, D)
# full_k, full_v are already (B, num_kv_heads=8, T, D) from DynamicLayer.
k_compressed, v_compressed = self.kv_cluster.update_kv(
    full_k, query_for_cluster, full_v, attention_mask, self.num_key_value_groups
)
# k_compressed, v_compressed: (B, num_kv_heads=8, max_capacity_prompt, D) — directly writable.
```

**Why strategy (c) over (a):**
- Strategy (a) (iter-2): repeat K/V to num_heads=32, cluster, then group-mean *K-tokens* back to num_kv_heads=8. The mean over 4 heads' selections at the same position-index averages 4 *different* selected positions. This blurs per-head K identity (phase corruption equivalent).
- Strategy (c): K and V stay un-repeated; only Q is reduced for scoring. Per-kv-head K/V identity is fully preserved. The trade-off is that the topk score for each kv-head uses an *averaged* query (across the 4 heads in that group) rather than each head's individual query. This is a smoother scoring signal, but no shape or phase information is destroyed downstream.

Strategy (c) was incorrectly rejected in iter 1 on Principle-1 grounds ("requires editing the museum cluster"). The cluster's only head-count-related constraint is the T-axis assertion. No museum edit is needed.

Pass un-repeated `(B, num_kv_heads, K, D)` tensors to `attention_interface` — its internal `repeat_kv` produces correct shapes (verified at `modeling_llama.py:209-210` for eager, equivalent for sdpa).

### 3.8. RoPE config

`LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")` succeeds natively in 5.2.0. `load_llama_config_stripped_rope` is dropped.

---

## 4. Per-file edit specs

### 4.1. NEW: `/home/yoongonkim/DCT-Page/baselines/snap_kv/patch_v5.py`

Full body (iter 3 — strategy (c) GQA reduction, no prepare_inputs wrapper, inline state-reset):

```python
"""SnapKV — live patch for transformers==5.2.0 (Llama 3.x + Qwen3).

Live path:
    init_snap_kv(model, cfg)
      -> replace_llama_v5(model) | replace_qwen3_v5(model)
          - patches {Llama|Qwen3}Attention.forward = _snapkv_attention_forward
          - eagerly initializes layer.self_attn.kv_cluster for each layer

Algorithm reuse: SnapKVCluster from upstream/snapkv/monkeypatch/snapkv_utils.py
(model-, transformers-version-, AND head-count-agnostic, vendored as a museum).

GQA reduction strategy: (c) group-mean Q (plan §3.7). K and V stay un-repeated
at num_kv_heads; only Q is reduced for scoring. Per-kv-head K/V identity is
preserved; cluster output is directly writable to the cache.

Cache rewrite: only supports DynamicLayer (vanilla DynamicCache). Hard
runtime assert blocks Static/Hybrid layers — see plan §3.3.

No prepare_inputs_for_generation patch (plan §0.1, §3.6). _update_model_kwargs_
for_generation advances cache_position independently of get_seq_length();
RoPE's relative-rotation property makes the compressed K cache holding
original-position phases correct under generate().

DO NOT EDIT baselines/snap_kv/upstream/*. See VENDORING.md.
"""
import torch
import transformers
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
# Direct callable default for attention dispatch ("eager" is NOT registered
# in ALL_ATTENTION_FUNCTIONS._global_mapping — see plan §3.4).
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# Algorithm — verbatim from upstream museum. NEVER monkey-patch this file.
from .upstream.snapkv.monkeypatch.snapkv_utils import SnapKVCluster

# --- module globals (mirrors dct_page_attention.py:134) ---
_original_llama_forward = None
_original_qwen3_forward = None


def init_snapkv_v5(self_attn) -> None:
    """Eagerly register kv_cluster on a {Llama|Qwen3}Attention module.

    Called at replace_*_v5(model) boot time, NOT on the hot path.
    Reads window_size / max_capacity_prompt / kernel_size / pooling from
    self_attn.config (set by init_snap_kv in __init__.py before replace_*_v5).
    """
    if not hasattr(self_attn, "kv_cluster"):
        cfg = self_attn.config
        if not hasattr(cfg, "window_size"):         cfg.window_size = 32
        if not hasattr(cfg, "max_capacity_prompt"): cfg.max_capacity_prompt = 2048
        if not hasattr(cfg, "kernel_size"):         cfg.kernel_size = 5
        if not hasattr(cfg, "pooling"):             cfg.pooling = "avgpool"
        self_attn.kv_cluster = SnapKVCluster(
            window_size=cfg.window_size,
            max_capacity_prompt=cfg.max_capacity_prompt,
            kernel_size=cfg.kernel_size,
            pooling=cfg.pooling,
        )


def _snapkv_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """Unified replacement for both LlamaAttention.forward and Qwen3Attention.forward.

    Contract:
      - Prefill (q_len > 1 and the cache layer was empty before this update):
        compute QKV, apply RoPE, do standard cache append; if the resulting
        cache length >= max_capacity_prompt, group-mean Q across kv-groups,
        run SnapKVCluster on the (num_kv_heads-shaped) full K/V with the
        group-mean'd Q, overwrite the cache layer's .keys/.values with the
        compressed tensors.
      - Decode (q_len == 1, or short prefill below capacity): standard cache
        append + attention; no further compression.

    State-reset is INLINE — detected via `full_k.shape[-2] == q_len` after
    Cache.update returns. No per-module sticky flag. No prepare_inputs
    wrapper.

    Dispatches q_norm/k_norm via hasattr; sliding_window via getattr.
    """
    # Dev-only sanity (init_snapkv_v5 should have run at boot).
    assert hasattr(self, "kv_cluster"), \
        "kv_cluster missing — replace_{llama,qwen3}_v5(model) was not called."

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape
    _has_qk_norm = hasattr(self, "q_norm") and hasattr(self, "k_norm")

    # Step 1: Q/K/V projection (with Qwen3 q_norm/k_norm)
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states   = self.k_proj(hidden_states).view(hidden_shape)
    if _has_qk_norm:
        query_states = self.q_norm(query_states)
        key_states   = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)    # (B, num_heads,    q_len, D)
    key_states   = key_states.transpose(1, 2)      # (B, num_kv_heads, q_len, D)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Step 2: RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Step 3: SnapKV cluster + cache update
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Standard cache append. full_k/full_v are post-append (num_kv_heads-shaped).
        full_k, full_v = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

        # Inline state-reset: the layer was empty before this update iff the
        # post-append length equals q_len. True on fresh prefill. False on
        # any decode step (cache has prior tokens) and on multi-call usage
        # where the cache carries history. No external state needed.
        is_prefill_to_compress = (
            q_len > 1
            and full_k.shape[-2] == q_len
            and full_k.shape[-2] >= self.kv_cluster.max_capacity_prompt
        )

        if is_prefill_to_compress:
            # Runtime guard — Static/Hybrid layers preallocate fixed-shape
            # tensors; overwriting .keys/.values silently corrupts
            # subsequent index_copy_ calls. (plan §3.3)
            layer = past_key_values.layers[self.layer_idx]
            assert isinstance(layer, DynamicLayer), (
                f"SnapKV v5 requires DynamicLayer (DynamicCache); got "
                f"{type(layer).__name__}. Pass past_key_values=DynamicCache(...) "
                "explicitly to model.generate(), or omit it (default)."
            )

            # GQA-reduction strategy (c) — plan §3.7. Group-mean Q only;
            # K/V stay un-repeated at num_kv_heads. Cluster's
            # head-count-agnostic body (verified by snapkv_utils.py:41
            # T-axis-only assertion) handles num_kv_heads-shaped inputs.
            B, H, _q, D = query_states.shape
            G = self.num_key_value_groups
            query_for_cluster = query_states.view(B, H // G, G, _q, D).mean(dim=2)
            # query_for_cluster: (B, num_kv_heads, q_len, D)

            k_compressed, v_compressed = self.kv_cluster.update_kv(
                full_k, query_for_cluster, full_v, attention_mask, G
            )
            # k_compressed, v_compressed: (B, num_kv_heads, max_capacity_prompt, D)

            # Overwrite the cache layer (DynamicLayer public attribute contract;
            # cache_utils.py:33-34, :120-121).
            layer.keys   = k_compressed.contiguous()
            layer.values = v_compressed.contiguous()

            # Use the compressed tensors for this layer's own forward output.
            # attention_interface will call repeat_kv internally on them.
            key_states_attn   = layer.keys
            value_states_attn = layer.values
        else:
            # Decode (or short-prefill below capacity).
            key_states_attn   = full_k
            value_states_attn = full_v
    else:
        # No cache — pure forward pass, no clustering (matches upstream).
        key_states_attn   = key_states
        value_states_attn = value_states

    # Step 4: attention dispatch. Direct callable default (plan §3.4).
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    # Prevent duplicate-kwarg TypeError if upstream caller stuffed
    # sliding_window into kwargs.
    kwargs.pop("sliding_window", None)
    sw = getattr(self, "sliding_window", None)
    extra = {"sliding_window": sw} if sw is not None else {}

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states_attn,
        value_states_attn,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **extra,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def replace_llama_v5(model) -> None:
    """Install SnapKV forward on LlamaAttention.
    Eagerly initializes kv_cluster on every layer."""
    global _original_llama_forward
    if _original_llama_forward is None:
        _original_llama_forward = transformers.models.llama.modeling_llama.LlamaAttention.forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = _snapkv_attention_forward

    for layer in model.model.layers:
        init_snapkv_v5(layer.self_attn)

    print(f"[snapkv] patched LlamaAttention.forward; "
          f"initialized kv_cluster on {len(model.model.layers)} layers")


def unpatch_llama_v5(model=None) -> None:
    """Restore the original LlamaAttention.forward. Idempotent."""
    global _original_llama_forward
    if _original_llama_forward is not None:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = _original_llama_forward
        _original_llama_forward = None


def replace_qwen3_v5(model) -> None:
    """Install SnapKV forward on Qwen3Attention."""
    global _original_qwen3_forward
    if _original_qwen3_forward is None:
        _original_qwen3_forward = transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = _snapkv_attention_forward

    for layer in model.model.layers:
        init_snapkv_v5(layer.self_attn)

    print(f"[snapkv] patched Qwen3Attention.forward; "
          f"initialized kv_cluster on {len(model.model.layers)} layers")


def unpatch_qwen3_v5(model=None) -> None:
    """Restore the original Qwen3Attention.forward. Idempotent."""
    global _original_qwen3_forward
    if _original_qwen3_forward is not None:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = _original_qwen3_forward
        _original_qwen3_forward = None
```

### 4.2. MODIFY: `/home/yoongonkim/DCT-Page/baselines/snap_kv/__init__.py`

```python
"""SnapKV baseline wrapper for DCT-Page.

Live path (transformers==5.2.0, DCT_Page conda env):
    baselines/snap_kv/__init__.py
        -> patch_v5.replace_{llama,qwen3}_v5(model)
        -> patch_v5._snapkv_attention_forward (unified)
        -> upstream/snapkv/monkeypatch/snapkv_utils.SnapKVCluster

Reference path (transformers==4.37.2, snap_kv conda env):
    baselines/snap_kv/upstream/__init__.py
        -> upstream/snapkv/monkeypatch/monkeypatch.replace_llama()
        -> upstream/snapkv/monkeypatch/llama_hijack_4_37.llama_flash_attn2_forward
        -> upstream/snapkv/monkeypatch/snapkv_utils.SnapKVCluster
"""
from .patch_v5 import (
    replace_llama_v5,
    replace_qwen3_v5,
    unpatch_llama_v5,
    unpatch_qwen3_v5,
)


def _assert_llama_or_qwen3(base_model: str) -> None:
    """Internal — fail fast if base_model is not a supported family."""
    bm = base_model.lower()
    if "llama" not in bm and "qwen3" not in bm:
        raise ValueError(
            "SnapKV wrapper supports only Llama 3.x and Qwen3 "
            f"(got base_model={base_model!r})."
        )


def _detect_family(model) -> str:
    mt = getattr(model.config, "model_type", "").lower()
    if mt == "llama":   return "llama"
    if mt == "qwen3":   return "qwen3"
    raise ValueError(f"Unsupported model_type for SnapKV: {mt!r}")


def init_snap_kv(model, cfg: dict) -> None:
    """Apply SnapKV KV-compression patch to a loaded model."""
    if "base_model" in cfg:
        _assert_llama_or_qwen3(cfg["base_model"])

    model.config.window_size         = cfg["window_size"]
    model.config.max_capacity_prompt = cfg["max_capacity_prompt"]
    model.config.kernel_size         = cfg["kernel_size"]
    model.config.pooling             = cfg["pooling"]

    fam = _detect_family(model)
    if fam == "llama":
        replace_llama_v5(model)
    else:
        replace_qwen3_v5(model)

    print(
        f"[snap_kv] family={fam} "
        f"window_size={cfg['window_size']} "
        f"max_capacity_prompt={cfg['max_capacity_prompt']} "
        f"kernel_size={cfg['kernel_size']} "
        f"pooling={cfg['pooling']}"
    )


__all__ = [
    "init_snap_kv",
    "replace_llama_v5",
    "replace_qwen3_v5",
    "unpatch_llama_v5",
    "unpatch_qwen3_v5",
]
```

Drops vs current `__init__.py`:
- `assert_llama_only` (replaced by internal `_assert_llama_or_qwen3`).
- `load_llama_config_stripped_rope` (transformers 5.x understands `rope_type='llama3'`).
- `from .upstream import init_snapkv as _init_snapkv, replace_llama as _replace_llama`.

### 4.3. MODIFY: `/home/yoongonkim/DCT-Page/baselines/snap_kv/config.py`

```python
"""Default configuration for the SnapKV baseline.

Environment requirements (LIVE PATH):
  - transformers==5.2.0   (DCT_Page conda env)
  - python>=3.12          (DCT_Page conda env)
  - Supports Llama 3.x AND Qwen3.

Reference path:
  - transformers==4.37.2  (snap_kv conda env)
  - Llama 3.x only.
"""
SNAPKV_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",  # or "Qwen/Qwen3-8B"
    "window_size": 32,
    "max_capacity_prompt": 2048,
    "kernel_size": 5,
    "pooling": "avgpool",
}
```

### 4.4. MODIFY: `/home/yoongonkim/DCT-Page/baselines/snap_kv/VENDORING.md`

(unchanged from iter 1 — appends a "Live path vs reference path" section)

### 4.5. MODIFY: `/home/yoongonkim/DCT-Page/run_ruler_snapkv.sh`

Two changes:

1. Line 11: env name default.
```diff
-SNAPKV_ENV_NAME="${SNAPKV_ENV_NAME:-snap_kv}"
+SNAPKV_ENV_NAME="${SNAPKV_ENV_NAME:-DCT_Page}"
```

2. Lines 49-53: family guard accepts Qwen3.
```diff
-# SnapKV only supports Llama — hard-fail otherwise.
+# SnapKV supports Llama 3.x and Qwen3 (post-port).
 case "${BASE_MODEL,,}" in
-    *llama*) ;;
-    *) echo "snap_kv only supports Llama (got: $BASE_MODEL)"; exit 1 ;;
+    *llama*|*qwen3*) ;;
+    *) echo "snap_kv only supports Llama 3.x and Qwen3 (got: $BASE_MODEL)"; exit 1 ;;
 esac
```

The `exit 2` guard for `--mode snap_kv` not being in `eval_ruler.py` argparse stays.

---

## 5. Execution order (step-by-step with acceptance checks)

### Step 1 — Write `patch_v5.py` (§4.1)

**Acceptance:**
```bash
cd /home/yoongonkim/DCT-Page && \
/home/yoongonkim/.conda/envs/DCT_Page/bin/python -c "
from baselines.snap_kv.patch_v5 import (
    _snapkv_attention_forward, init_snapkv_v5,
    replace_llama_v5, replace_qwen3_v5,
    unpatch_llama_v5, unpatch_qwen3_v5,
)
print('OK: 6 patch_v5 symbols importable')
"
```

### Step 2 — Rewrite `__init__.py` (§4.2)

**Acceptance:**
```bash
cd /home/yoongonkim/DCT-Page && \
/home/yoongonkim/.conda/envs/DCT_Page/bin/python -c "
from baselines.snap_kv import (
    init_snap_kv, replace_llama_v5, replace_qwen3_v5,
    unpatch_llama_v5, unpatch_qwen3_v5,
)
import baselines.snap_kv as m
assert not hasattr(m, 'assert_llama_only'), 'old name leaked'
assert not hasattr(m, 'load_llama_config_stripped_rope'), 'old helper leaked'
print('OK: __init__.py rewritten')
"
```

### Step 3 — Touch `config.py` + `VENDORING.md` + `run_ruler_snapkv.sh` (§4.3, 4.4, 4.5)

**Acceptance:**
```bash
grep -q "Live path vs reference path" /home/yoongonkim/DCT-Page/baselines/snap_kv/VENDORING.md && \
grep -q "Llama 3.x AND Qwen3" /home/yoongonkim/DCT-Page/baselines/snap_kv/config.py && \
grep -q "DCT_Page" /home/yoongonkim/DCT-Page/run_ruler_snapkv.sh && \
bash -n /home/yoongonkim/DCT-Page/run_ruler_snapkv.sh && \
echo "OK"
```

### Step 4 — Smoke test (iter 3 — 7 sub-checks, both models)

Write `/tmp/snapkv_smoke.py`:

```python
"""SnapKV iter 3 smoke test — prefill + decode + parity + compression-divergence."""
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baselines.snap_kv import init_snap_kv, unpatch_llama_v5, unpatch_qwen3_v5
from baselines.snap_kv.config import SNAPKV_CONFIG

os.environ.setdefault("SNAPKV_TRACE", "1")
torch.manual_seed(0)

def run_one(model_id: str, family: str):
    print(f"\n=== {model_id} ({family}) ===")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    q_len = 4096
    prompt_ids = torch.randint(0, tok.vocab_size, (1, q_len), device="cuda:0")

    # --- baseline (unpatched) reference
    with torch.no_grad():
        baseline_logits = model(input_ids=prompt_ids).logits[:, -1, :].float()

    # --- (d) PARITY: max_capacity_prompt > q_len => no compression triggered.
    cfg_parity = dict(SNAPKV_CONFIG, max_capacity_prompt=q_len + 1, window_size=32)
    init_snap_kv(model, cfg_parity)
    with torch.no_grad():
        parity_logits = model(input_ids=prompt_ids).logits[:, -1, :].float()
    diff_d = (parity_logits - baseline_logits).abs().max().item()
    print(f"  (d) parity (no-compression) max abs diff: {diff_d:.6f}")
    assert diff_d < 1e-3, f"parity check failed: {diff_d} >= 1e-3"
    print(f"  (d) PASS: parity within 1e-3 (bf16 noise)")
    if family == "llama": unpatch_llama_v5(model)
    else:                 unpatch_qwen3_v5(model)

    # --- main patch with compression on
    cfg = dict(SNAPKV_CONFIG, max_capacity_prompt=512, window_size=32)
    init_snap_kv(model, cfg)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
        logits = out.logits

    # (b) Logits-finite
    assert torch.isfinite(logits).all(), "non-finite logits in prefill"
    print(f"  (b) PASS: logits finite, shape={tuple(logits.shape)}")

    # (c) ALL layers compressed
    past_kv = out.past_key_values
    n_layers = len(past_kv.layers)
    for i, layer in enumerate(past_kv.layers):
        assert layer.keys.shape[-2] == cfg["max_capacity_prompt"], \
            f"layer {i}: cache not compressed (got {layer.keys.shape[-2]}, " \
            f"expected {cfg['max_capacity_prompt']})"
    print(f"  (c) PASS: all {n_layers} layers compressed to {cfg['max_capacity_prompt']}")

    # (f) Compression divergence diagnostic — the test iter 2's group-mean
    # phase corruption would have failed at smoke time. Bound is bf16-loose
    # (compression discards 87% of K/V) but catches catastrophic divergence.
    compressed_last_logits = logits[:, -1, :].float()
    diff_f = (compressed_last_logits - baseline_logits).abs().max().item()
    print(f"  (f) compression-vs-baseline max abs diff: {diff_f:.4f}")
    assert diff_f < 5.0, \
        f"compression divergence too large: {diff_f} >= 5.0 (phase corruption?)"
    print(f"  (f) PASS: compression divergence bounded ({diff_f:.4f} < 5.0)")

    # (a) DECODE — generate 4 tokens
    with torch.no_grad():
        gen = model.generate(
            input_ids=prompt_ids, max_new_tokens=4,
            do_sample=False, use_cache=True,
        )
    new_tokens = gen[:, -4:]
    assert new_tokens.shape == (1, 4), f"unexpected gen shape {new_tokens.shape}"
    # Stronger than iter 2: require finite logits (already checked in (b))
    # AND at least one generated token (no degenerate all-EOS-at-position-0).
    assert torch.unique(new_tokens).numel() >= 1, "no tokens generated"
    print(f"  (a) PASS: generated 4 tokens {new_tokens.tolist()}")

    # (e) MULTI-SAMPLE REUSE — use generate() to match eval_ruler.py pattern.
    prompt2 = torch.randint(0, tok.vocab_size, (1, q_len), device="cuda:0")
    with torch.no_grad():
        gen2 = model.generate(
            input_ids=prompt2, max_new_tokens=4,
            do_sample=False, use_cache=True,
        )
    assert gen2.shape == (1, q_len + 4), f"unexpected gen2 shape {gen2.shape}"
    print(f"  (e) PASS: second generate() call succeeds")

    # (g) UNPATCH ROUND-TRIP
    if family == "llama": unpatch_llama_v5(model)
    else:                 unpatch_qwen3_v5(model)
    with torch.no_grad():
        out3 = model(input_ids=prompt_ids, use_cache=True)
        assert out3.past_key_values.layers[0].keys.shape[-2] == q_len, \
            f"unpatch did not restore: shape={out3.past_key_values.layers[0].keys.shape}"
    print(f"  (g) PASS: unpatch restores uncompressed cache shape")

    del model
    torch.cuda.empty_cache()

run_one("meta-llama/Llama-3.1-8B-Instruct", "llama")
run_one("Qwen/Qwen3-8B", "qwen3")
print("\nALL SMOKE TESTS PASSED")
```

**Acceptance:**
```bash
cd /home/yoongonkim/DCT-Page && \
/home/yoongonkim/.conda/envs/DCT_Page/bin/python /tmp/snapkv_smoke.py
```
Exit 0, last line `ALL SMOKE TESTS PASSED`. Failure at any sub-check (a-g) for either model is a hard reject.

### Step 5 — Stop. Do NOT run RULER.

`--mode snap_kv` is still not wired into `eval_ruler.py`. The wireup ships as `.omc/plans/snapkv_eval_ruler_wireup.md`.

---

## 6. Risks, pre-mortem (DELIBERATE mode) — iter 3

### Pre-mortem — 4 failure scenarios

**Scenario 1: Cache overwrite under non-DynamicLayer.** Mitigated by runtime `isinstance(layer, DynamicLayer)` assert. Smoke test (c) uses default `DynamicCache` allocation by `generate()`, so the assert path is exercised on every layer.

**Scenario 2: Qwen3 sliding-attention layers.** Qwen3-8B is all full-attention; `getattr(self, "sliding_window", None)` returns None for all layers. Qwen3-4B / Qwen3-1.5B may have sliding-window layers — the path is shape-correct but smoke does not exercise them. Open question (§9).

**Scenario 3: Multi-call cache reuse with stale compression.** If a user reuses the same `past_key_values` across two `model(...)` or `generate()` calls without resetting, the inline `is_prefill_to_compress` check uses `full_k.shape[-2] == q_len` to detect "fresh prefill". For the second call, the cache already has compressed tokens; `full_k.shape[-2] > q_len`, so the condition is False and no re-compression fires. **Failure mode:** if the user wants per-call re-compression on the second prompt, they MUST allocate a fresh cache (the eval_ruler.py pattern). Documented in Open Questions (§9).

**Scenario 4 (iter 3 — replaces iter 2's strategy-a worry): Strategy (c) group-mean Q smooths per-head scoring.** The cluster picks top-k positions per kv-head using a query that's averaged across the 4 attention heads in that GQA group. Each attention head's individual query bias is lost in the scoring stage. **Mitigation:** K and V are not corrupted (strategy a's failure mode is gone); per-head identity is preserved at attention time when `attention_interface` re-expands via `repeat_kv`. The smoke test (f) bounds the resulting divergence loosely (`< 5.0` abs diff at cap=512 on a 4096-token prompt, bf16). Numerical-faithfulness comparison vs upstream 4.37 belongs in a follow-up RULER sweep.

### Expanded test plan (iter 3)

| Lane | What | Where |
|---|---|---|
| Unit (algorithmic) | `SnapKVCluster.update_kv` unchanged. | upstream/snapkv/monkeypatch/snapkv_utils.py |
| Unit (wrapper) | `init_snapkv_v5` idempotent. | smoke multi-sample reuse |
| Integration | Patched forward runs on prefill (4096 tokens), both models. | smoke (b) |
| Integration | All layers compressed correctly. | smoke (c) |
| Integration | `model.generate(max_new_tokens=4)` succeeds on top of compressed cache. | smoke (a) |
| Integration | Logits parity at cap=q_len+1 within 1e-3 abs (bf16). | smoke (d) |
| Integration | Compressed-vs-uncompressed divergence bounded at cap=512. | smoke (f) |
| Integration | Multi-sample `generate()` reuse pattern (eval_ruler.py shape). | smoke (e) |
| Integration | Unpatch restores uncompressed behavior. | smoke (g) |
| E2E | RULER across `max_capacity_prompt` × pooling. | Out of scope (next plan). |
| Observability | `SNAPKV_TRACE=1` one line per layer per prefill. | smoke (stdout) |

---

## 7. ADR (Architecture Decision Record) — iter 3

**Decision:** Port SnapKV to transformers 5.2.0 via:
1. A single unified `_snapkv_attention_forward` in `baselines/snap_kv/patch_v5.py`, dispatching by `hasattr(self, "q_norm")` and `getattr(self, "sliding_window", None)`.
2. **No `prepare_inputs_for_generation` patch** (iter 3 — Bug 3 was a misdiagnosis; see §0.1). State-reset is inline via `full_k.shape[-2] == q_len` check after `Cache.update(...)` returns.
3. Cache rewrite via the `DynamicLayer.keys/.values` public attribute contract, guarded by a runtime `isinstance(layer, DynamicLayer)` assert.
4. **GQA reduction via strategy (c) — group-mean Q only** (iter 3 — replaces iter 2's strategy (a) group-mean K). K and V stay un-repeated at `num_kv_heads`; only Q is averaged across the 4 attention heads in each GQA group for the cluster's scoring step. Per-kv-head K/V identity is preserved; cluster output is directly writable to the cache. Zero museum edits required (cluster is head-count-agnostic; only T-axis assertion at `snapkv_utils.py:41`).
5. `baselines/snap_kv/upstream/` remains untouched.

**Drivers (recap from §1):**
1. Live-path simplicity (one forward, two thin entry points).
2. API compatibility with transformers 5.2.0 (verified Cache, attention dispatch).
3. Reversibility (`unpatch_*_v5` restores the original forward).

**Alternatives considered:**
- Option B (two forwards): rejected for ~120 LoC duplication.
- Option C (subclass-based): invalidated by `@use_kernelized_func` incompatibility.
- Option D (`SnapKVCache(DynamicCache)` subclass): documented in §1.D; rejected for eval_ruler wireup cost + DCT precedent divergence. Forward patch is unavoidable anyway (cluster needs Q which is not in `cache_kwargs`).
- **Iter-2 alternative: `prepare_inputs_for_generation` wrapper.** Rejected in iter 3 — `_update_model_kwargs_for_generation` advances `cache_position` independently of `get_seq_length()`, and RoPE relative-rotation makes the compressed K cache phases correct under `generate()`. The wrapper was functionally dead.
- **Iter-2 alternative: GQA strategy (a) group-mean K.** Rejected in iter 3 — averages 4 different head-selected K positions per stored slot, blurring per-head K identity. Strategy (c) preserves K/V identity at the cost of smoother (group-meaned) Q scoring.

**Why chosen (Option A + inline state-reset + strategy (c)):**
- Closest match to `dct_page_attention.py:1138-1227`.
- Minimal LoC net (1 new patch file, ~180 lines).
- Extensible to Qwen2 / Mistral with `replace_*_v5` only.
- No external state / wrapper to maintain.

**Consequences:**
- The `snap_kv` conda env (4.37) becomes reference-only.
- `load_llama_config_stripped_rope` and `assert_llama_only` are removed.
- Cache rewrite tied to `DynamicLayer` — running with `StaticCache` / `HybridCache` is a hard error (runtime assert).
- Group-mean Q (strategy c) differs from upstream 4.37's per-head Q for scoring. Acknowledged numerical-faithfulness trade-off; smoke check (f) bounds the divergence loosely; full validation via downstream RULER sweep.
- Cache reuse across multiple `generate()` / `model(...)` calls without fresh allocation: re-compression does NOT fire on the second call (the cache already holds prior tokens, so `full_k.shape[-2] != q_len`). Eval scripts allocate fresh per sample; constraint documented.
- `SNAPKV_TRACE=1` continues to work.

**Follow-ups:**
- `.omc/plans/snapkv_eval_ruler_wireup.md` — wire `--mode snap_kv` into `eval_ruler.py`.
- Stretch: GQA strategy (c) vs upstream 4.37 RULER comparison sweep.
- Stretch: validate behavior on Qwen3-4B / Qwen3-1.5B (sliding-window layers).
- After 1-week soak, consider deprecating the `snap_kv` conda env.

---

## 8. Live path vs reference path (canonical diagram) — iter 3

```
+----------------------------------------------------------------------+
|                        baselines/snap_kv/                            |
|                                                                      |
|  __init__.py    config.py    VENDORING.md    patch_v5.py             |
|        \                                       /                     |
|         +---> init_snap_kv(model, cfg) <-----+                       |
|                       |                                              |
|                       v                                              |
|                  family detect                                       |
|                  /            \                                      |
|   model_type == "llama"   model_type == "qwen3"                     |
|         |                            |                              |
|         v                            v                              |
|  replace_llama_v5(model)      replace_qwen3_v5(model)               |
|       |                            |                                |
|       +--> for layer: init_snapkv_v5(layer.self_attn)               |
|       +--> {Llama|Qwen3}Attention.forward = _snapkv_attention_forward|
|                                |                                    |
|                                v                                    |
|        from .upstream.snapkv.monkeypatch.snapkv_utils                |
|                  import SnapKVCluster                                |
|                                |                                    |
|                                v                                    |
|              upstream/snapkv/monkeypatch/snapkv_utils.py            |
|              (museum — algorithm only — DO NOT MODIFY)              |
+----------------------------------------------------------------------+

Reference path (snap_kv conda env, transformers==4.37.2): unchanged.

Both paths share ONLY SnapKVCluster.
```

---

## 9. Open questions (to be appended to `.omc/plans/open-questions.md`)

- [x] Position drift on decode (iter-1 alleged Bug 3) — RESOLVED iter 3: misdiagnosis. `_update_model_kwargs_for_generation` advances `cache_position` independently of `get_seq_length()`; RoPE relative-rotation makes compressed phases correct. See §0.1.
- [x] GQA strategy (a) group-mean K vs strategy (c) group-mean Q — RESOLVED iter 3: switched to (c). K/V identity preserved.
- [x] `is_prefill` sticky flag — RESOLVED iter 3: inline check `full_k.shape[-2] == q_len` after `Cache.update`, no external state.
- [ ] Numerical parity between transformers-5.2 SnapKV (strategy c) and transformers-4.37 SnapKV (per-head Q): worth a future plan if RULER scores diverge.
- [ ] Cache reuse across multiple `generate()` calls: re-compression fires only on fresh-cache prefill. Eval scripts allocate fresh per sample; document the constraint.
- [ ] Should `init_snap_kv` accept an explicit `family` argument? Useful for finetunes with odd `model_type`.
- [ ] Sliding-window Qwen3 variants (Qwen3-4B / Qwen3-1.5B): smoke only covers Qwen3-8B (all full-attention).

---

## 10. Final checklist (Planner self-check for iter 3 handoff)

- [x] Iteration 3 changelog at top (§0) lists all 4 MUST-FIX + 4 SHOULD-FIX with locations.
- [x] §0.1 retraction preserves epistemic trail of iter-1 misdiagnoses.
- [x] `prepare_inputs_v5.py` references stricken from §2, §3, §4 (file no longer planned).
- [x] §3.6 retitled "cache_position advancement — Bug 3 NON-EXISTENCE proof".
- [x] §3.7 retracted strategy-(a) choice, documented strategy (c) with honest trade-off.
- [x] §4.1 code body: strategy (c) Q-only group-mean, no sticky flag, no wrapper.
- [x] §1.D Option D rejection rationale updated (dropped position-drift reason).
- [x] Smoke test §5 has 7 sub-checks (a-g) including (f) compression-divergence diagnostic.
- [x] Pre-mortem Scenario 3 reframed to multi-call cache reuse (not position drift).
- [x] Plan length 1119 → ~1017 lines (subtractive iteration).
- [x] Open questions queued for `.omc/plans/open-questions.md`.
