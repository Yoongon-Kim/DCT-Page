## upstream-fi-multibatch - 2026-05-04
- [ ] Should `_pack_preallocated_to_paged_upstream` vectorize the batch loop? — Currently a Python `for b in range(bsz)` at build time. Once-only call so v1 keeps it simple. Revisit if profiles show build-time regression.
- [ ] Confirm `dct_sdpa` mode already works at `--batch_size 2` before validating the all-mode comparison (4d). — If dct_sdpa is itself bsz=1-only, the comparison row needs adjustment or a separate fix.
- [ ] Should we expose `pages_per_batch` for the upstream backend to support a future ragged-batch extension? — v1 leaves it at default 0; ragged-batch is explicitly out of scope.
- [ ] At `--batch_size 4 --context_length 32768`, is the A6000 KV-cache budget definitely safe? — Estimate: ~17 GiB for Llama-3.1-8B; fits 48 GiB but with model weights + activations the headroom may be tight. Document fallback to 16K if OOM.
