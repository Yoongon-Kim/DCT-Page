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
