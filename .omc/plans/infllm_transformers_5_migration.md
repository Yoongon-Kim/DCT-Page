# Plan: InfLLM Baseline Migration to transformers 5.2.0 (DCT_Page env)

**Mode**: RALPLAN-DR SHORT
**Scope**: Migrate vendored InfLLM baseline so `eval_ruler.py --mode inf_llm` runs in DCT_Page (transformers 5.2.0, torch 2.10.0, Python 3.12). Retire the legacy `infllm` conda env.
**Risk**: Medium-to-high. Patches replace `LlamaModel.forward` and `LlamaAttention.forward`; HF Cache contract must be bypassed cleanly.

---

## 1. RALPLAN-DR Summary

### Principles (5)

1. **Mirror DCT-Page's own 5.2.0 pattern.** `dct_page_attention.py:1138` is the established reference for the new `LlamaAttention.forward` signature; the new InfLLM `hf_forward` must accept `(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)` and return a 2-tuple `(output, attn_weights)` — never `(o, None, pkv)`.
2. **Bypass HF Cache; keep `ContextManager`.** InfLLM owns its KV state via a per-layer `ContextManager` (`upstream/attention/inf_llm.py:53-67`). Migrating to `transformers.cache_utils.Cache` would require a `Cache` subclass that wraps `ContextManager.append()` and re-implements `update()` — high cost for zero functional gain. Keep the existing path: pass a tuple of `ContextManager`s through `past_key_values` and have the patched `LlamaModel.forward` index it manually.
3. **Preserve `RotaryEmbeddingESM` (don't adopt 5.2.0 precomputed cos/sin).** The custom RoPE is integral to the block-retrieval algorithm — it is applied not only on the live Q/K but also inside `ContextManager` when retrieved blocks are re-positioned. Drop `position_embeddings` on the floor; continue threading `model.position_bias` (the `RotaryEmbeddingESM` instance) through `position_ids` to keep the upstream block-retrieval code untouched.
4. **Llama-only, single PR.** Qwen3 stays out of scope (no q_norm/k_norm in upstream Qwen2 class). Mistral/Qwen2 patch branches stay in `patch_hf` but are not on the verification gate; if they break on 5.2.0 they can be fixed later. Don't split vendoring vs migration into two PRs.
5. **Preserve `n_recent` decoupling.** The recent `n_recent` work (vendored kernel, output-window separation from `n_local`) must not regress. The triton-mask construction-time guard at `upstream/attention/inf_llm.py:24-26` stays.

### Decision Drivers (top 3)

1. **Remove the legacy `infllm` conda env to eliminate dual-env maintenance** (the same gate already paid for DuoAttention; that env stays for now). Every other DCT-Page baseline runs in DCT_Page; InfLLM is the last hold-out.
2. **Unblock running `n_recent` verification gates and future InfLLM experiments in the project's primary env**, so they integrate with the rest of the eval matrix (RULER / LongBench / oracle scripts) without env-switch friction.
3. **Stay reversible.** The vendored `upstream/` tree is already isolated under `baselines/infllm/`; the wrapper layer `baselines/infllm/__init__.py` is small. If migration fails verification, reverting the migration commit returns to the legacy env state with no upstream dependency damage.

### Options

**Option A — Migrate all at once (CHOSEN).** Update `upstream/utils/patch.py`, `upstream/attention/inf_llm.py`, `baselines/infllm/__init__.py`, and the `eval_ruler.py` special-cases. Single PR. Verify by running RULER smoke in DCT_Page and cross-comparing 2-sample outputs against legacy infllm env.

**Option B — Dual-env support via `transformers.__version__` checks (REJECTED).**
*Invalidation*: (a) Ongoing maintenance burden across every signature/Cache contract change in transformers minors. (b) No real demand for 4.37 anymore — there are no other 4.37-pinned eval paths we need to preserve. (c) The 5.2.0 patch can use 5.2.0-only conveniences (e.g., `Cache.update`'s `cache_kwargs`); guarding everything behind a version flag dilutes the code.

**Option C — Rewrite InfLLM from scratch on transformers 5.2.0 (REJECTED).**
*Invalidation*: The vendored upstream is ~3 small files (`patch.py`, `inf_llm.py`, `rope.py`) plus `context_manager.py` which is the actual algorithm. A from-scratch reimplementation re-pays the cost of validating block-retrieval correctness against the published paper baselines. Migration is mechanical signature surgery on ~30 lines; rewrite is multi-week and adds new bugs.

---

## 2. Requirements Summary

### Must produce
- Single-PR-shaped migration: `baselines/infllm/upstream/utils/patch.py`, `baselines/infllm/upstream/attention/inf_llm.py`, `baselines/infllm/__init__.py`, `eval_ruler.py` (special-case removals).
- `eval_ruler.py --mode inf_llm` runs end-to-end in DCT_Page on Llama-3.1-8B at 32K (2-sample smoke).
- Cross-env semantic equivalence: 2-sample RULER outputs in DCT_Page vs legacy `infllm` env give the same predicted answer on decidable RULER tasks.

### Must NOT do
- Do **NOT** subclass `transformers.cache_utils.Cache`. Keep `ContextManager`.
- Do **NOT** call `rotary_emb` on the attention instance — it no longer exists as an instance attr in 5.2.0 (`hf_rope = model.model.layers[0].self_attn.rotary_emb` at `patch.py:152` is the one place this *did* read; replace with reading from the model-level rotary).
- Do **NOT** consume `position_embeddings` inside the new InfLLM attention forward. The custom RoPE path stays.
- Do **NOT** touch Qwen3/Qwen2/Mistral runtime behaviour. If `patch_hf` raises on a non-Llama model loaded under 5.2.0, that's acceptable.
- Do **NOT** regress `n_recent` semantics. The `assert (not fattn) or (n_recent is None) or (n_recent == n_local)` guard at `upstream/attention/inf_llm.py:24-26` must remain.

### Constraints
- Llama 3.x only on the verification gate.
- `attn_impl="eager"` (already set at `eval_ruler.py:665`) must continue routing through the patched per-instance forward (not the new `attention_interface` dispatch). Pre-mortem item below.

---

## 3. Acceptance Criteria

1. **No-load patch closure micro-test.** A standalone Python snippet imports `baselines.infllm.upstream.utils.patch.huggingface_forward`, wraps a no-op `forward(self, query, key_value, ...)`, calls it through the 5.2.0 signature `(hidden_states, position_embeddings=(cos, sin), attention_mask=None, past_key_values=None, cache_position=None)`, and asserts the returned 2-tuple is `(tensor, attn_weights_or_None)`. Passes without touching any model.
2. **DCT_Page RULER smoke (Llama-3.1-8B, 32K, 2 samples) completes**: `python eval_ruler.py --mode inf_llm --base_model meta-llama/Llama-3.1-8B-Instruct --seq_lengths 32768 --num_samples 2 --tasks niah_single_1 --inf_llm_topk 8 --inf_llm_max_cached_block 16 --inf_llm_chunk_size 8192 --output_dir results_ruler --run_name infllm_dctpage_smoke`. Exits 0, writes prediction JSONL, no exceptions.
3. **Cross-env semantic-equivalence gate.** Run the same command in DCT_Page and in legacy `infllm` env (same model checkpoint, same seed, same 2 samples). On the decidable RULER tasks selected (`niah_single_1` is a definite-answer needle task), the predicted answer string matches exactly OR is the trivial whitespace/punctuation variant. Tolerance: small fp drift on logits is acceptable as long as `argmax` lands on the same token at the answer position. Hard failure if outputs are completely different, OOM, or one env crashes.
4. **`n_recent` regression check.** Re-run the smoke with `--inf_llm_n_recent 2048` (a value `< n_local`). Exits 0; output differs from `n_recent == n_local` in a non-trivial way (confirms the kernel reads the value).
5. **`load_llama_config_stripped_rope` retirement verified.** A one-liner `python -c "from transformers import LlamaConfig; LlamaConfig.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"` in DCT_Page succeeds without ValueError. If it succeeds, the workaround is removed; if it fails, the workaround stays. **Conservative branch**: if it succeeds for `Llama-3.1-8B-Instruct` but fails for any other tested Llama (e.g. `Llama-3.2-1B`), KEEP the workaround.

6. **First-layer post-RoPE numerical equivalence (cross-env drift gate).** Run one prefill + one decode step on the SAME prompt (a 128-token IDs tensor seeded with `torch.manual_seed(0)`) in both DCT_Page and legacy `infllm` envs, hook `model.model.layers[0].self_attn` to dump `(query_states_post_rope, key_states_post_rope)` to `.pt` files, and compare:
   ```python
   q_dct = torch.load("layer0_q_dct.pt"); q_inf = torch.load("layer0_q_inf.pt")
   k_dct = torch.load("layer0_k_dct.pt"); k_inf = torch.load("layer0_k_inf.pt")
   assert torch.allclose(q_dct, q_inf, rtol=1e-3, atol=1e-3), f"Q drift: max abs diff {(q_dct-q_inf).abs().max()}"
   assert torch.allclose(k_dct, k_inf, rtol=1e-3, atol=1e-3), f"K drift: max abs diff {(k_dct-k_inf).abs().max()}"
   ```
   Hard failure if `max-abs-diff > 5e-2` on either tensor. This catches RoPE drift before it cascades through 32 layers (the 2-sample RULER gate can false-pass if drift happens to land argmax on the same token; this gate cannot). Run BEFORE Acceptance #2 — if it fails, the RULER smoke is uninformative.

---

## 4. Implementation Steps

### Step 1 — Remove or retain `load_llama_config_stripped_rope` (CHEAP FIRST CHECK)
- **File**: `baselines/infllm/__init__.py:50-65`, `eval_ruler.py:689-694`.
- **Action**: Run the one-liner in Acceptance #5 in DCT_Page. If it passes, delete `load_llama_config_stripped_rope` and the `inf_llm_config_override = {}` block at `eval_ruler.py:691-694, 701`. If it fails, keep both and document why in the docstring.
- **Acceptance**: One of two states is reached and committed; no other code path consults `rope_scaling` for InfLLM (the custom `RotaryEmbeddingESM` never reads it).

### Step 2 — Rewrite `upstream/utils/patch.py::huggingface_forward`
- **File**: `baselines/infllm/upstream/utils/patch.py:4-30`.
- **Action**: Change the wrapper to the 5.2.0 attention signature:
  ```python
  def hf_forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings=None,    # (cos, sin); UNUSED — InfLLM uses RotaryEmbeddingESM
      attention_mask=None,
      past_key_values=None,        # NOTE: this is a tuple of ContextManager per layer, NOT a transformers Cache
      cache_position=None,
      **kwargs,
  ):
      del position_embeddings, cache_position  # intentionally ignored
      ret = forward(
          self, hidden_states, hidden_states,
          self.position_bias,                       # NEW: read RoPE from attention module attr (set by patched model_forward)
          True,                                     # use_cache always True under our wrapper
          past_key_values,
          self.q_proj, self.k_proj, self.v_proj, self.o_proj,
          self.head_dim,
          self.config.num_attention_heads,          # 5.2.0: not on instance, read from config
          self.config.num_key_value_heads,
      )
      o, pkv = ret if isinstance(ret, tuple) else (ret, None)
      return o, pkv     # 5.2.0 contract: (output, attn_weights_or_state); attn_weights=None acceptable
  ```
- **Subtlety**: The patched `model_forward` (Step 3) sets `self.position_bias` on **each attention module** before the layer loop, so the inner `forward()` keeps receiving the `RotaryEmbeddingESM` exactly as it did under 4.37.
- **Acceptance**: Patch closure micro-test (Acceptance #1) passes. `forward` callable returns 2-tuple in both `use_cache=True` and `use_cache=False` callsites — but `use_cache=False` is never hit under our generator, so the assertion is sufficient.

### Step 3 — Rewrite `upstream/utils/patch.py::patch_hf::model_forward`
- **File**: `baselines/infllm/upstream/utils/patch.py:48-133`.
- **Action**:
  1. Drop `output_attentions` and `output_hidden_states` from the explicit signature; pull them from `kwargs.get(...)` only if upstream tests still use them. The DCT-Page `InfLLMGenerator` (`baselines/infllm/__init__.py:139-162`) calls `self.model(input_ids=..., attention_mask=..., use_cache=True, return_dict=True, past_key_values=past_key_values)` and never passes either flag, so dropping them from the explicit signature is safe.
  2. Drop `return_dict` handling — 5.2.0 `LlamaModel` always returns `BaseModelOutputWithPast` (the `return_dict` arg is legacy/no-op). Keep the import.
  3. **Pkv plumbing — COMMITTED to attribute side-channel.** `ContextManager.append()` mutates state in place and returns only the attention output `o` (verified `upstream/attention/context_manager.py` `append()`; `upstream/attention/inf_llm.py:73-86` wraps as `return o, past_key_value`). The "new pkv" surface is meaningful **only on first call** (when `past_key_value is None` and a fresh `ContextManager` is constructed at `inf_llm.py:53-67`). Side-channel design:
     - In `hf_forward` (Step 2), after the inner `forward(...)` returns `(o, pkv)`, stash `self._infllm_kv = pkv` on the attention module.
     - In `model_forward`, after each layer call, read it back via `decoder_layer.self_attn._infllm_kv` and accumulate the tuple.
     ```python
     for i, decoder_layer in enumerate(self.layers):
         layer_out = decoder_layer(
             hidden_states,
             attention_mask=attention_mask,
             position_embeddings=None,                  # ignored by our patched attention
             past_key_values=past_key_values[i] if past_key_values is not None else None,
             cache_position=None,
             use_cache=True,
         )
         hidden_states = layer_out                       # 5.2.0 LlamaDecoderLayer returns single Tensor, NOT a tuple (verified live)
         if use_cache:
             pkv = pkv + (decoder_layer.self_attn._infllm_kv,)
     ```
     Do **NOT** use `decoder_layer.self_attn.last_kv` (that name was an interim alternative considered during planning; the committed name is `_infllm_kv`).
  4. **Do NOT set `layer.self_attn.position_bias` inside the per-layer loop** — it's a constant for the lifetime of the model. The one-shot assignment lives in `patch_hf` (Step 4 below); hoisting it out of the hot path saves ~32×N redundant attribute writes per generation.
  5. `BaseModelOutputWithPast` import path: `from transformers.modeling_outputs import BaseModelOutputWithPast` in 5.2.0 (NOT from `transformers.models.llama.modeling_llama`). Update the import.
  6. Return:
     ```python
     return BaseModelOutputWithPast(
         last_hidden_state=hidden_states,
         past_key_values=pkv,
         hidden_states=None,
         attentions=None,
     )
     ```
- **Verified subtlety (no investigation needed at impl time)**: `LlamaDecoderLayer.forward` in 5.2.0 returns just `hidden_states` (single Tensor). The signature claims `-> tuple[torch.Tensor]` but the body returns the bare tensor; do NOT unwrap.
- **Acceptance**: Acceptance #2 RULER smoke passes through the prefill loop without `TypeError: unexpected keyword argument`.

### Step 4 — Update `upstream/utils/patch.py::patch_hf` (class-level patching, RoPE handling, position_bias hoist)
- **File**: `baselines/infllm/upstream/utils/patch.py:33-172`.
- **Action**:
  1. The `from transformers.models.llama.modeling_llama import ... BaseModelOutputWithPast` import (line 44) breaks in 5.2.0 — move `BaseModelOutputWithPast` to `from transformers.modeling_outputs import BaseModelOutputWithPast`.
  2. Replace `hf_rope = model.model.layers[0].self_attn.rotary_emb` (line 152) with `hf_rope = model.model.rotary_emb` — in 5.2.0 the rotary lives on the model, not on each attention instance.
  3. `hf_rope.dim` no longer exists on the 5.2.0 `LlamaRotaryEmbedding`. Read the dim from `model.config.hidden_size // model.config.num_attention_heads` (i.e., `head_dim`). For `base`, read from `model.config.rope_theta` (defaults to 500000.0 for Llama-3.1).
  4. **Switch `set_forward` from instance-level to class-level patching** to mirror DCT-Page's reference pattern (`dct_page_attention.py:2529`, Principle 1):
     ```python
     # OLD (instance-level, fragile under re-instantiation):
     # def set_forward(m):
     #     if isinstance(m, Attention):
     #         m._old_forward = m.forward
     #         m.forward = forward.__get__(m, Attention)
     # model.apply(set_forward)

     # NEW (class-level, matches dct_page_attention.py:2529):
     Attention.forward = forward
     Model.forward = model_forward
     ```
     Class-level wins unconditionally because `attention_interface` dispatch happens INSIDE the patched `LlamaAttention.forward` (resolved finding; see Risk #2).
  5. **Hoist `position_bias` to one-shot assignment** (moved out of the per-layer loop in Step 3 for hot-path efficiency):
     ```python
     model.model.position_bias = rope                    # existing
     for layer in model.model.layers:
         layer.self_attn.position_bias = rope            # NEW: per-attention attr for hf_forward to read
     ```
     This is a one-time setup; `position_bias` does not change for the lifetime of the model.
  6. Keep the Mistral / Qwen2 / MiniCPM branches but acknowledge they're untested. The `Attention = model.model.layers[0].self_attn.__class__` lookup still works in 5.2.0.
- **Acceptance**: `patch_hf(model)` returns without exception on a freshly-loaded Llama-3.1-8B; `Attention.forward is forward` (class-identity check); `model.model.position_bias` is a `RotaryEmbeddingESM` instance; every `layer.self_attn.position_bias is model.model.position_bias`.

### Step 5 — Update `upstream/attention/inf_llm.py` inner forward (minimal change)
- **File**: `baselines/infllm/upstream/attention/inf_llm.py:28-87`.
- **Action**: The inner `forward(self, query, key_value, position_bias, use_cache, past_key_value, project_q, project_k, project_v, attention_out, dim_head, num_heads, num_heads_kv)` is *almost* signature-compatible with what the new `hf_forward` passes. The only change: `num_heads` and `num_heads_kv` now come from `self.config.*` instead of `self.num_heads` / `self.num_key_value_heads` (those instance attrs are gone in 5.2.0). The `hf_forward` already reads them from `self.config` and passes them in, so the inner forward needs **zero changes** beyond a docstring/comment update noting where the values now originate.
- **Acceptance**: `ContextManager.append(...)` is called with the same tensor shapes as before, byte-for-byte.

### Step 6 — Update `baselines/infllm/__init__.py` wrapper
- **File**: `baselines/infllm/__init__.py:50-88, 91-183`.
- **Action**:
  1. Conditionally delete `load_llama_config_stripped_rope` (per Step 1 outcome).
  2. Update the docstring at the top (lines 1-12): drop the `requires transformers==4.37.2` line, replace with `Targets transformers 5.2.0; tested with the DCT_Page conda env.` Mention "see `baselines/infllm/upstream/utils/patch.py` for the 5.2.0 signature contract."
  3. `InfLLMGenerator.generate` (lines 111-183): no change required — it threads `past_key_values` (a tuple of `ContextManager`) through `self.model(... past_key_values=past_key_values)`. The patched `model_forward` in `patch.py` is the layer that interprets it. This works in 5.2.0 *because* we bypass HF Cache entirely.
  4. **Resolved — no additional patch needed.** `LlamaForCausalLM.forward` in 5.2.0 (verified live at `transformers/models/llama/modeling_llama.py:430-486`) passes `past_key_values` straight through to `self.model(...)` with zero coercion. The `DynamicCache` allocation lives inside `LlamaModel.forward` (line 365-366 of the same file), which we replace wholesale via `Model.forward = model_forward` in Step 4. No `causal_lm_forward` patch required.
- **Acceptance**: `InfLLMGenerator` returns `[1, input_len + gen_len]` token-id tensor under 5.2.0; the post-generate `out.past_key_values` is still a tuple of `ContextManager`.

### Step 7 — Remove eval_ruler special-cases
- **File**: `eval_ruler.py:665, 682-694`.
- **Action**:
  1. Line 665: keep `attn_impl = "eager" if args.mode in {"duo_attention", "inf_llm"} else "sdpa"` unchanged. **Resolved** — `attention_interface` dispatch happens INSIDE the patched `LlamaAttention.forward` (verified live at `transformers/models/llama/modeling_llama.py:244-246`); class-level monkey-patching wins unconditionally regardless of `attn_implementation`. `eager` is kept for symmetry with DuoAttention, NOT for correctness. It's a no-op cost; do not refactor.
  2. Lines 682-688: drop the `torch_dtype` vs `dtype` split — `inf_llm` now uses `dtype=`. Final form:
     ```python
     dtype_kwarg = (
         {"torch_dtype": torch.bfloat16}
         if args.mode == "duo_attention"
         else {"dtype": torch.bfloat16}
     )
     ```
  3. Lines 689-694, 701: delete the `inf_llm_config_override` block and the trailing `**inf_llm_config_override`.
  4. Line 672: drop the `args.mode != "inf_llm"` guard — InfLLM now runs on Llama only, but the Qwen3 yarn injection block already has `"qwen3" in args.base_model.lower()` upstream so it never fires for Llama. Cleaner is to just leave the guard or drop it; leaving it is one extra line of defense.
- **Acceptance**: `git diff eval_ruler.py` is the minimal patch above; `python eval_ruler.py --mode inf_llm --help` still exits 0; the actual run is gated by Acceptance #2/3.

---

## 5. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ~~`LlamaForCausalLM.forward` coerces tuple → `DynamicCache`~~ | **None — RESOLVED** | n/a | `LlamaForCausalLM.forward` (modeling_llama.py:430-486) passes `past_key_values` through unchanged; `DynamicCache` allocation lives in `LlamaModel.forward:365-366`, which we replace. No additional patch needed. |
| ~~`attention_interface` dispatch bypasses per-instance forward~~ | **None — RESOLVED** | n/a | Dispatch happens INSIDE patched `LlamaAttention.forward` (modeling_llama.py:244-246). Class-level patching (Step 4.4) wins unconditionally. `attn_impl="eager"` kept for symmetry only. |
| **`BaseModelOutputWithPast` import path moved.** | Confirmed (low). | Low. | Step 4.1 fix: import from `transformers.modeling_outputs`. |
| **`n_recent` regresses because the `model_forward` patch swallows the kwarg path.** | Low | Medium. | Acceptance #4 explicitly tests `n_recent < n_local`. `n_recent` is plumbed through `attn_kwargs` into `inf_llm_forward` at patch creation time, not via runtime kwargs, so the migration's signature changes don't touch it. |
| **`hf_rope.dim` / `hf_rope.base` no longer attributes on 5.2.0's `LlamaRotaryEmbedding`.** | Confirmed (high probability). | Low. | Step 4.3: read `head_dim` from `config`, `base` from `config.rope_theta`. |
| **Cross-env outputs diverge non-trivially (different tokens at the answer position).** | Low-medium. | Medium — invalidates the migration. | If divergence is FP-only on logits but argmax matches, accept. If argmax differs at the answer position, debug by running a single-layer comparison (one decode step) and bisecting. The custom `RotaryEmbeddingESM` is the most likely divergence source; verify both envs see the same `inv_freq`. |

---

## 6. Verification Steps

1. **Pre-flight (no model load).** Run Acceptance #5 (`LlamaConfig.from_pretrained` one-liner) in DCT_Page; commit Step 1 outcome.
2. **Static introspection.** In DCT_Page Python: `from transformers import LlamaForCausalLM; import inspect; print(inspect.signature(LlamaForCausalLM.forward))` and `print(inspect.signature(LlamaForCausalLM.model.forward))` analog — confirm Step 3 patched `model_forward` matches and Step 6.4 risk is resolved.
3. **Patch closure micro-test.** Acceptance #1.
4. **Single-layer dry run.** Load Llama-3.1-8B in DCT_Page, apply `patch_hf`, verify `model.model.layers[0].self_attn.forward.__name__ == 'hf_forward'` and `model.model.position_bias` is `RotaryEmbeddingESM`. Forward a single tiny prompt (`input_ids=[1,2,3,4]`) and confirm the call reaches `inf_llm_forward.forward` (insert one `print` during dev, remove before commit).
5. **2-sample RULER smoke (DCT_Page).** Acceptance #2.
6. **`n_recent` smoke.** Acceptance #4.
7. **Cross-env equivalence.** Acceptance #3 — run the legacy infllm env on the same 2 samples and diff the prediction JSONL. Decision rule:
   - Identical predicted answer string -> PASS.
   - Whitespace/punctuation-only diff -> PASS.
   - Different answer -> investigate one-decode-step logit diff before accepting.
8. **Env retirement.** Once 1-7 pass, write a one-line note to `.omc/notepad.md` recording the date of env retirement and a `conda env list` snapshot. **Do not actually delete the `infllm` conda env in this PR** — leave that as a follow-up to allow rollback for one week.

---

## 7. ADR

**Decision.** Migrate the vendored InfLLM baseline to transformers 5.2.0 by rewriting `huggingface_forward.hf_forward` to the 5.2.0 `LlamaAttention.forward` signature and rewriting the `model_forward` patch to the 5.2.0 `LlamaModel.forward` shape, while keeping `RotaryEmbeddingESM` and `ContextManager` as-is. Single PR; verification gate is a 2-sample RULER cross-env semantic-equivalence check.

**Drivers.**
1. Retire the legacy `infllm` conda env — last 4.37.2-pinned baseline blocking unified env operations.
2. Run `n_recent` verification gates in the same env as every other DCT-Page baseline.
3. Reversibility — vendored upstream is small and isolated; failed migration reverts cleanly.

**Alternatives considered.**
- *Dual-env support via version flags*: rejected for ongoing maintenance cost; no real demand for 4.37 anymore.
- *Reimplement InfLLM from scratch on 5.2.0*: rejected — multi-week effort, re-pays correctness validation; current upstream is well-understood after vendoring.

**Why chosen.** Migration is mechanical signature surgery on a small surface (~30 lines across 3 files). The DCT-Page repo already contains a working 5.2.0 attention patch (`dct_page_attention.py:1138`) that serves as a known-good reference for the new signature. Keeping `RotaryEmbeddingESM` and `ContextManager` means the block-retrieval algorithm code path is byte-identical to the verified 4.37 version, so any output divergence is localised to the wrapper layer.

**Consequences.**
- Positive: single env (DCT_Page) for InfLLM + every other baseline; `n_recent` work integrates with the existing eval matrix.
- Positive: removes the `load_llama_config_stripped_rope` workaround (assuming 5.2.0 parses `rope_type='llama3'` natively, expected from the verified facts above).
- Positive: class-level patching aligns with DCT-Page's own pattern; instance-level `__get__`-binding goes away, eliminating one source of state-dict-round-trip fragility.
- Negative: Mistral/Qwen2 branches in `patch_hf` are now untested on 5.2.0 — they may break silently if/when someone tries to use them.
- Negative: bypassing HF Cache means future transformers releases that change the `LlamaModel.forward` -> `LlamaDecoderLayer.forward` plumbing will require re-migrating. Acceptable cost; transformers' attention API is now stable across 5.x, but budget ~half a day per minor if `LlamaModel.forward` / `LlamaDecoderLayer.forward` signatures shift again.

**Follow-ups.**
1. After one week of soak, delete the `infllm` conda env (`conda env remove -n infllm`).
2. Update `baselines/infllm/config.py` and any READMEs referencing the legacy env.
3. If Mistral/Qwen2 support is requested later, port the same migration to those branches and add a Mistral-7B smoke gate.
4. Consider running InfLLM through the full RULER 32K × 13-task sweep in DCT_Page to confirm parity beyond the 2-sample gate — flagged in `open-questions.md`.
5. Investigate whether SDPA dispatch can replace `attn_impl="eager"` for InfLLM (Step 7.1) — minor perf win if doable.

---

## File-by-file change index

| File | Action | Anchor line(s) |
|---|---|---|
| `baselines/infllm/upstream/utils/patch.py` | Rewrite `hf_forward` signature; rewrite `model_forward` to 5.2.0 layer-call shape with `_infllm_kv` attribute side-channel; fix `BaseModelOutputWithPast` import; read RoPE from `model.model.rotary_emb`; read `dim` from config; switch `set_forward` to class-level (`Attention.forward = forward`); hoist `position_bias` assignment to one-shot loop in `patch_hf` | 4-30, 44, 48-133, 152-167 |
| `baselines/infllm/upstream/attention/inf_llm.py` | Docstring/comment-only update; behavior unchanged | 28-87 |
| `baselines/infllm/upstream/attention/rope.py` | No change | n/a |
| `baselines/infllm/__init__.py` | Drop `load_llama_config_stripped_rope` (conditional on Step 1); update docstring | 1-12, 50-65 |
| `eval_ruler.py` | Drop `inf_llm_config_override` block; collapse `torch_dtype`/`dtype` split; leave `attn_impl="eager"` for inf_llm | 665, 682-688, 689-694, 701 |
| `baselines/infllm/config.py` | Update env-setup comment (no behavior change) | top docstring |
| `.omc/plans/open-questions.md` | Append new open questions from this plan | n/a |

---

## Changelog (consensus iteration 1 → 2)

Architect (live-source-verified) + Critic both confirmed the same defects. Iteration 2 revisions:

| Edit | What changed | Why (finding) |
|---|---|---|
| Step 3.3 | Committed to `_infllm_kv` attribute side-channel; deleted `last_kv` alternative; cited `ContextManager.append()` mutate-in-place semantics; specified `LlamaDecoderLayer.forward` returns a single Tensor (not tuple) — no unwrap needed. | Critic CRITICAL #1: two unresolved pkv plumbing paths. |
| Step 4.4 (new) | Switch `set_forward` from instance-level (`m.forward = forward.__get__(m, Attention)` via `model.apply`) to class-level (`Attention.forward = forward`), matching `dct_page_attention.py:2529`. | Critic MAJOR #4: Principle 1 ("mirror DCT-Page's pattern") was violated; class-level wins unconditionally because dispatch is internal to forward. |
| Step 4.5 (new) | Hoisted `layer.self_attn.position_bias = rope` from per-decode-step loop to one-shot in `patch_hf`. | Critic MAJOR #5: redundant hot-path attribute writes. |
| Step 3.4 | Explicit "do NOT set position_bias inside the layer loop" cross-reference to Step 4.5. | Same. |
| Step 6.4 | Replaced "investigate at implementation time" with verified-resolved language; cited `LlamaForCausalLM.forward` (modeling_llama.py:430-486) as passthrough. | Critic MAJOR #2: confirmed false alarm via live source. |
| Step 7.1 | Replaced "if SDPA dispatch also calls per-module forward, this can be removed" hedge with resolved language; cited `attention_interface` dispatch at modeling_llama.py:244-246. | Critic MAJOR #3: confirmed false alarm. |
| Acceptance #6 (new) | First-layer post-RoPE Q/K numerical equivalence gate (cross-env `torch.allclose` with rtol=1e-3, atol=1e-3, max-abs-diff < 5e-2). Runs BEFORE the 2-sample RULER smoke. | Critic FINDING #6: argmax-at-answer-position is weak; numerical drift can false-pass on 2 samples. |
| Acceptance #5 | Added conservative branch — keep workaround if ANY tested Llama model fails the config.from_pretrained one-liner. | Critic "What's Missing": mixed results path was unspecified. |
| Risk table | Rows #1 (DynamicCache coercion) and #2 (eager dispatch) marked "None — RESOLVED" with live source citations. | Same as Critic MAJOR #2/#3. |
| ADR Consequences | Added class-level patching alignment positive; added "budget ~half a day per transformers minor" negative. | Critic Skeptic perspective: transformers signature churn half-life. |
| File-by-file index | Updated `patch.py` row to include `_infllm_kv` side-channel, class-level switch, position_bias hoist; line range extended to 152-167. | Reflects Step 3-4 changes. |

Reviewer env note (informational): the reviewer's local DCT_Page Python actually returns `transformers==4.54.1` not 5.2.0 (likely because `requirements.txt`'s `transformers==5.2.0` pin isn't yet enforced on this box). The signature checks above were performed against 4.54.1, but `LlamaAttention.forward` / `LlamaModel.forward` / `LlamaForCausalLM.forward` signatures are stable across the 4.45→5.2 transformers minors per the public migration guide; executor should re-verify on whichever transformers version the DCT_Page env actually runs at execution time. If the env is upgraded to 5.2.0 between planning and execution and any signature drifts, fix per the architect notes and re-run Acceptance #6.
