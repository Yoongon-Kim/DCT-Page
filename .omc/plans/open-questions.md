## upstream-fi-multibatch - 2026-05-04
- [ ] Should `_pack_preallocated_to_paged_upstream` vectorize the batch loop? — Currently a Python `for b in range(bsz)` at build time. Once-only call so v1 keeps it simple. Revisit if profiles show build-time regression.
- [ ] Confirm `dct_sdpa` mode already works at `--batch_size 2` before validating the all-mode comparison (4d). — If dct_sdpa is itself bsz=1-only, the comparison row needs adjustment or a separate fix.
- [ ] Should we expose `pages_per_batch` for the upstream backend to support a future ragged-batch extension? — v1 leaves it at default 0; ragged-batch is explicitly out of scope.
- [ ] At `--batch_size 4 --context_length 32768`, is the A6000 KV-cache budget definitely safe? — Estimate: ~17 GiB for Llama-3.1-8B; fits 48 GiB but with model weights + activations the headroom may be tight. Document fallback to 16K if OOM.

## eval-upstream-flashinfer-backend - 2026-05-10
- [ ] Should the upstream-FI cache be opt-in via a separate sub-flag like `--upstream_fi_max_decode_steps_padding` for users who want to override the `+16` slack? — AIME / LongBench-v2 max output budgets vary widely; insufficient slack causes mid-generation `cache overflow` errors.
- [ ] Is there an appetite to expose the verify-shadow's per-step diff distribution as a histogram in the eval summary JSON, or is "log to stdout" sufficient? — Shapes how observability scales when running a 13-task × 32K parity sweep.
- [ ] Should the new `dct_page_attention_forward_upstream_flashinfer` close the `score_use_quest_minmax` gap that the profile driver currently has, or stay symmetric with the profile driver and refuse `score_use_quest_minmax=True` until the profile driver is updated? — Sweep scripts that enable Quest-minmax via `--score_use_quest_minmax` would silently degrade if we don't close it; but closing it adds a code path the profile driver doesn't exercise.
- [ ] Should we also wire the upstream-FI backend into `oracle/oracle_ruler.py` and the diagnostic scripts under `oracle/`? — Listed as a follow-up in the ADR but not in the initial scope; revisit once eval is soaked.
- [ ] Should a soft `--seq_lengths > 32768` warning escalate to a hard refusal under `--attention_backend upstream_flashinfer` on a 48 GiB GPU? — Prevents OOM-mid-eval but blocks legitimate experimentation; current plan is soft warning only.

## infllm_transformers_5_migration - 2026-05-11
- [ ] Does 5.2.0's `LlamaForCausalLM.forward` coerce `past_key_values` into a `DynamicCache` before delegating to `LlamaModel.forward`? — If yes, we need an additional `causal_lm_forward` patch alongside `model_forward` to preserve our tuple-of-`ContextManager`. Verify by reading the live 5.2.0 source in DCT_Page during Step 6.4 before writing the patch.
- [ ] Does 5.2.0's `attention_interface` dispatch bypass per-instance `forward` when `attn_implementation="eager"`? — Acceptance #2 will catch this implicitly (outputs would match the unpatched baseline). If it bypasses, our patched `hf_forward` never runs and we need a different attach point (likely `attn_implementation_dispatch` shim).
- [ ] Is `LlamaDecoderLayer.forward` returning just `hidden_states` or still a tuple in 5.2.0? — Step 3's per-layer unwrap depends on the live return shape; resolve via `inspect.getsource` at implementation time rather than guessing.
- [ ] After 1-week soak in DCT_Page, is it safe to delete the `infllm` conda env? — Listed as a follow-up in the ADR; revisit only once the 2-sample cross-env gate has been passed and at least one full RULER 32K × 13-task sweep has run cleanly under DCT_Page.
- [ ] Should we keep the Mistral / Qwen2 / MiniCPM branches in `patch_hf` despite no verification gate on 5.2.0? — Keeping them adds untested code that may break silently. Dropping them narrows surface to verified Llama-only. Default is keep + add a `# untested on transformers 5.x` comment; revisit on first failure report.
- [ ] Does Acceptance #5's `LlamaConfig.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')` actually succeed natively in DCT_Page (5.2.0)? — If yes, `load_llama_config_stripped_rope` is deleted in Step 1; if no, it stays. This gates whether the migration is purely code-shape or also needs to keep a config workaround.
- [ ] Should `attn_impl="eager"` for InfLLM be relaxed to `"sdpa"` post-migration as a perf follow-up? — Listed as Follow-up #5 in the ADR. Depends on whether 5.2.0's SDPA dispatch path still calls per-instance `forward` — open until verified.

## snapkv_port_transformers_5 - 2026-05-11
- [x] Is `past_key_values.layers[i].keys = ...` safe across all `Cache` subclasses we exercise (especially `HybridCache`, `StaticCache`)? — Iter 2: hard runtime `isinstance(layer, DynamicLayer)` assert added in `_snapkv_attention_forward`. Closed.
- [ ] Should `init_snap_kv` accept an explicit `family` argument that overrides `model.config.model_type` detection? — Useful for finetunes that report odd `model_type`; defer to follow-up unless Critic flags.
- [ ] Should the live patch mirror DCT's `_dct_page_cfg` module-global config pattern, or keep config attached to `model.config`? — Current plan keeps it on `model.config` (matches upstream SnapKV); could switch later if eval_ruler.py wireup wants a single-source-of-truth dict.
- [ ] Numerical parity between transformers-5.2 SnapKV (v5 port) and transformers-4.37 SnapKV (reference path): out of scope here, but worth a future plan if RULER scores diverge across envs.
- [x] `is_prefill` heuristic `past_key_values.get_seq_length(self.layer_idx) == 0` — Iter 2: replaced with per-layer one-shot flag `self._snapkv_compressed_layers`, reset by `prepare_inputs_for_generation` wrapper on fresh generation. Closed.

## snapkv_port_transformers_5 (iter 2 additions) - 2026-05-11
- [x] Group-mean GQA reduction (iter 2: strategy a) vs upstream 4.37's per-head storage — RESOLVED iter 3: switched to strategy (c) (group-mean Q). K/V identity preserved at num_kv_heads; only Q is averaged for scoring. Strategy (c) requires zero museum edits (cluster is head-count-agnostic per `snapkv_utils.py:41`).
- [x] `prepare_inputs_for_generation` wrapper is `generate()`-only — RESOLVED iter 3: wrapper DELETED. Bug 3 was a misdiagnosis (`_update_model_kwargs_for_generation` advances `cache_position` independently of `get_seq_length()`; RoPE relative-rotation makes compressed phases correct). See iter-3 §0.1 retraction.
- [ ] Sliding-window Qwen3 layers (Qwen3-4B / Qwen3-1.5B variants): `getattr(self, "sliding_window", None)` should be correct but smoke only exercises Qwen3-8B (all full-attention). Validate when smaller Qwen3 variants enter scope.
- [x] Should `_make_wrapper` in `prepare_inputs_v5.py` reuse a single global `_real_seen_tokens` attribute or per-cache dict — RESOLVED iter 3: wrapper DELETED. No external state needed.
- [x] When unpatching, `_snapkv_seen_tokens_override` / `_snapkv_compressed_layers` attrs left on modules — RESOLVED iter 3: those attributes are gone (no wrapper, no sticky flag). The remaining `kv_cluster` attribute is left on modules after unpatch — cosmetic; revisit if Critic flags.

## snapkv_port_transformers_5 (iter 3 additions) - 2026-05-11
- [ ] Numerical-faithfulness comparison: strategy (c) group-mean Q (iter 3 chosen) vs upstream 4.37's per-head Q. Worth a small RULER comparison sweep after wireup if scores diverge.
- [ ] Cache reuse across multiple `generate()` / `model(...)` calls without fresh-cache allocation: the inline `full_k.shape[-2] == q_len` check returns False on the 2nd call, so re-compression does NOT fire. Eval scripts allocate fresh per sample; document the constraint in `__init__.py` docstring.
- [ ] Smoke check (f)'s `< 5.0` abs diff bound at cap=512 (87% compression) is bf16-loose. Tighten after first RULER run if the observed envelope is consistently smaller.

