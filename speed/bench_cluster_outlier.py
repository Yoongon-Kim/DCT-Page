"""Measure prefill + decode latency for cluster_dyn outlier configurations.

Three configurations are timed back-to-back on the same loaded model:
  - dct_no_outlier   : baseline DCT-Page, outlier_budget=0
  - dct_cluster_N256 : cluster_dyn N=256, K_top=8, q=max, scoring=centroid
  - dct_cluster_N1024: same with N=1024
For each: random 32k-token input, warmup, then num_repeats × (prefill + 64 decode).

Reports per-config mean prefill latency and mean per-step decode latency, plus
the relative cost vs the no-outlier baseline.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def patch_qwen3(args, outlier_budget: int, cluster_outlier_N: int):
    from dct_page_attention import replace_qwen3_attn
    replace_qwen3_attn(
        page_size=args.page_size,
        top_k=args.top_k,
        num_sink_pages=args.num_sink_pages,
        num_recent_pages=args.num_recent_pages,
        compress_ratio=args.compress_ratio,
        unselected_mode="drop",
        use_triton=True,
        outlier_budget=outlier_budget,
        outlier_detector="cluster_dyn" if outlier_budget > 0 else "lastq_mean",
        cluster_outlier_N=cluster_outlier_N,
        cluster_outlier_iters=args.cluster_outlier_iters,
        cluster_outlier_top_k=args.cluster_outlier_top_k,
        cluster_outlier_q_agg="max",
        cluster_outlier_scoring="centroid",
        attention_backend=args.attention_backend,
    )


def _maybe_reset_fi(model, attention_backend, decode_steps):
    """Tear down upstream-FI per-generate state and re-seed build kwargs."""
    if attention_backend != "upstream_flashinfer":
        return
    from dct_page_attention import (
        reset_upstream_fi_cache_state, _set_upstream_fi_max_decode_steps,
    )
    reset_upstream_fi_cache_state(model)
    _set_upstream_fi_max_decode_steps(model, decode_steps)


def time_prefill_and_decode(model, tokenizer, ctx_len: int, decode_steps: int, attention_backend: str):
    """Returns (prefill_ms, decode_step_ms_avg)."""
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(
        low=10, high=vocab_size - 10, size=(1, ctx_len),
        device=device, dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    # 1-token call (= prefill + 1 decode token)
    _maybe_reset_fi(model, attention_backend, decode_steps=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=1, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    t_prefill_plus_1 = time.perf_counter() - t0

    # prefill + decode_steps tokens
    _maybe_reset_fi(model, attention_backend, decode_steps=decode_steps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=decode_steps, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    decode_step_ms_avg = ((t_total - t_prefill_plus_1) / max(1, decode_steps - 1)) * 1000
    prefill_ms = t_prefill_plus_1 * 1000 - decode_step_ms_avg
    return prefill_ms, decode_step_ms_avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--ctx_len", type=int, default=32768)
    p.add_argument("--decode_steps", type=int, default=64)
    p.add_argument("--num_repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--num_sink_pages", type=int, default=1)
    p.add_argument("--num_recent_pages", type=int, default=4)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--cluster_outlier_iters", type=int, default=5)
    p.add_argument("--cluster_outlier_top_k", type=int, default=8)
    p.add_argument("--outlier_budget", type=int, default=64)
    p.add_argument("--cluster_N", type=int, default=1024,
                   help="Number of clusters for k-means (scale with context length).")
    p.add_argument("--attention_backend", type=str, default="sdpa",
                   choices=["sdpa", "flashinfer", "upstream_flashinfer"],
                   help="Attention backend for the DCT decode forward.")
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    yarn_kwargs = {}
    if "qwen3" in args.model.lower():
        yarn_kwargs = {
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            "max_position_embeddings": 131072,
        }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    configs = [
        ("no_outlier",                    0, args.cluster_N),
        (f"cluster_N{args.cluster_N}",    args.outlier_budget, args.cluster_N),
    ]

    results = {}
    for label, ob, N in configs:
        print(f"\n=== {label}: outlier_budget={ob}, cluster_N={N} ===")
        # Re-patch and reload the model so each config starts fresh (avoids
        # PreAllocatedLayer carry-over and ensures clean kv cache state).
        patch_qwen3(args, outlier_budget=ob, cluster_outlier_N=N)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", **yarn_kwargs,
        ).eval()

        # upstream-FlashInfer needs per-attention-module kwargs seeded post-load
        # so the lazy cache build sizes correctly for our decode budget.
        if args.attention_backend == "upstream_flashinfer":
            from dct_page_attention import (
                _init_upstream_fi_build_kwargs, _set_upstream_fi_max_decode_steps,
                reset_upstream_fi_cache_state,
            )
            _init_upstream_fi_build_kwargs(model)
            _set_upstream_fi_max_decode_steps(model, args.decode_steps)

        # Warmup
        for _ in range(args.warmup):
            _ = time_prefill_and_decode(model, tokenizer, args.ctx_len, decode_steps=8, attention_backend=args.attention_backend)

        prefill_samples = []
        decode_samples = []
        for r in range(args.num_repeats):
            pre, dec = time_prefill_and_decode(model, tokenizer, args.ctx_len, args.decode_steps, attention_backend=args.attention_backend)
            prefill_samples.append(pre)
            decode_samples.append(dec)
            print(f"  repeat {r+1}: prefill={pre:.1f}ms  decode/step={dec:.3f}ms")
        results[label] = {
            "prefill_ms": sum(prefill_samples) / len(prefill_samples),
            "decode_step_ms": sum(decode_samples) / len(decode_samples),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Report
    print("\n" + "=" * 80)
    print(f"Results at ctx_len={args.ctx_len}, decode_steps={args.decode_steps}, repeats={args.num_repeats}")
    print("=" * 80)
    print(f"{'config':<20s}  {'prefill (ms)':>14s}  {'decode/step (ms)':>17s}")
    print("-" * 60)
    for label, _, _ in configs:
        r = results[label]
        print(f"{label:<20s}  {r['prefill_ms']:>14.1f}  {r['decode_step_ms']:>17.3f}")
    base = results["no_outlier"]
    print("\nOverhead vs no_outlier:")
    for label, _, _ in configs:
        if label == "no_outlier":
            continue
        r = results[label]
        pre_overhead = (r["prefill_ms"] - base["prefill_ms"]) / base["prefill_ms"] * 100
        dec_overhead = (r["decode_step_ms"] - base["decode_step_ms"]) / base["decode_step_ms"] * 100
        print(f"  {label:<18s}  prefill +{pre_overhead:5.1f}%  |  decode/step +{dec_overhead:5.1f}%")


if __name__ == "__main__":
    main()
