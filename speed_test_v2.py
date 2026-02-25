"""
Decode speed benchmark on LongBench v2: Baseline vs DCT Page Attention.

Selects --num_samples from the full 503-sample LongBench v2 dataset
(filtered by --min_context_len after tokenization) and times prefill
and decode separately using a manual decode loop.

Results are saved under:
    results_speed_test_v2/<run_name>/
        samples.jsonl   — per-sample timing records
        summary.json    — aggregated stats

Usage:
    python speed_test_v2.py                        # both modes
    python speed_test_v2.py --mode baseline
    python speed_test_v2.py --mode dct --top_k 8
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_longbench import build_prompt, tokenize_and_truncate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LongBench v2 decode speed benchmark")

    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--mode", choices=["baseline", "dct", "both"], default="both")
    p.add_argument("--num_samples", type=int, default=20,
                   help="Samples to benchmark from the full dataset")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_input_len", type=int, default=120000)
    p.add_argument("--min_context_len", type=int, default=128000,
                   help="Skip samples shorter than this after tokenization")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--output_dir", default="results_speed_test_v2")
    p.add_argument("--run_name", default=None)

    dct = p.add_argument_group("DCT Page Attention config")
    dct.add_argument("--page_size", type=int, default=128)
    dct.add_argument("--top_k", type=int, default=8)
    dct.add_argument("--sink_size", type=int, default=4)
    dct.add_argument("--recent_size", type=int, default=128)
    dct.add_argument("--compress_ratio", type=float, default=0.03)
    dct.add_argument("--scoring_method", default="max", choices=["mean", "max"])
    dct.add_argument("--group_agg_method", default="mean",
                     choices=["mean", "max", "topp"])
    dct.add_argument("--unselected_mode", default="drop",
                     choices=["drop", "compressed"])
    dct.add_argument("--selection_mode", default="standard",
                     choices=["standard", "hierarchical"])
    dct.add_argument("--continuous_rope", action="store_true")
    dct.add_argument("--no_triton", action="store_true",
                     help="Disable Triton kernels (use pure PyTorch for comparison)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_family(model_name):
    name = model_name.lower()
    if "llama" in name:
        return "llama"
    elif "qwen" in name:
        return "qwen2"
    return model_name.split("/")[-1].lower()


def make_run_name(label, args):
    family = model_family(args.model)
    if label == "baseline":
        return f"{family}_baseline"
    parts = [
        family, "page_attn",
        str(args.compress_ratio),
        f"topk{args.top_k}",
        args.scoring_method,
        args.group_agg_method,
        args.unselected_mode,
    ]
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded: {model_name} ({n_params:.2f}B params)")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Attention patch helpers
# ---------------------------------------------------------------------------
def get_original_forward(model_name):
    import transformers
    if "llama" in model_name.lower():
        return transformers.models.llama.modeling_llama.LlamaAttention.forward
    return transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward


def restore_forward(model_name, original_forward, model=None):
    import types
    import transformers
    if "llama" in model_name.lower():
        transformers.models.llama.modeling_llama.LlamaAttention.forward = original_forward
        # When device_map="auto" is used, accelerate stores module.forward as an
        # instance attribute (_old_forward) that shadows class-level patches.
        # Restore _old_forward on each instance so the hook calls the right function.
        if model is not None:
            attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(original_forward, module)
    else:
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = original_forward
        if model is not None:
            attn_cls = transformers.models.qwen2.modeling_qwen2.Qwen2Attention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(original_forward, module)


def apply_dct_patch(args, model=None):
    patch_kwargs = dict(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
        selection_mode=args.selection_mode,
        continuous_rope=args.continuous_rope,
        use_triton=not getattr(args, 'no_triton', False),
    )
    if "llama" in args.model.lower():
        import types
        import transformers
        from dct_page_attention import replace_llama_attn, dct_page_attention_forward
        replace_llama_attn(**patch_kwargs)
        # Patch instance-level _old_forward on each attention module so that
        # accelerate's device hooks call dct_page_attention_forward instead of
        # the original forward captured at hook-install time.
        if model is not None:
            attn_cls = transformers.models.llama.modeling_llama.LlamaAttention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(dct_page_attention_forward, module)
    else:
        import types
        import transformers
        from dct_page_attention import replace_qwen2_attn, dct_page_attention_forward
        replace_qwen2_attn(**patch_kwargs)
        if model is not None:
            attn_cls = transformers.models.qwen2.modeling_qwen2.Qwen2Attention
            for module in model.modules():
                if isinstance(module, attn_cls) and hasattr(module, "_old_forward"):
                    module._old_forward = types.MethodType(dct_page_attention_forward, module)


# ---------------------------------------------------------------------------
# Per-sample timing
# ---------------------------------------------------------------------------
def time_sample(model, tokenizer, input_ids, max_new_tokens, warmup_steps):
    device = input_ids.device
    prefill_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)

    step_times = []
    step = 0
    for step in range(max_new_tokens):
        cache_position = torch.tensor([prefill_len + step], device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)

        if step >= warmup_steps:
            step_times.append(elapsed)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return prefill_time, step_times, step + 1


# ---------------------------------------------------------------------------
# Pre-selection (once, shared across all modes)
# ---------------------------------------------------------------------------
def preselect_samples(dataset, tokenizer, args):
    """
    Select sample indices using a character-length proxy instead of
    tokenization.

    Tokenizing full documents (some are 200K+ tokens) just to check their
    length is prohibitively slow.  A character count is O(1) and sufficient
    for filtering: ~3 chars/token is a conservative lower-bound estimate that
    avoids false negatives.

    Note: min_context_len filters by the ORIGINAL document length.  The
    actual benchmark will truncate inputs to max_input_len, so it is fine
    (and intentional) for min_context_len to exceed max_input_len — it simply
    ensures the document was genuinely long before truncation.
    """
    # Conservative lower-bound: 3 chars ≈ 1 token, so a document with
    # min_context_len tokens has at least min_context_len * 3 characters.
    char_threshold = args.min_context_len * 3

    print(f"\nPre-selecting {args.num_samples} samples "
          f"(min_context_len={args.min_context_len} tokens ≈ "
          f"{char_threshold:,} chars) from {len(dataset)} total...")

    selected = []
    for i, item in enumerate(dataset):
        if len(item.get("context", "")) >= char_threshold:
            selected.append(i)
        if len(selected) >= args.num_samples:
            break

    if len(selected) < args.num_samples:
        print(f"  WARNING: only {len(selected)}/{args.num_samples} samples "
              f"meet min_context_len={args.min_context_len}")
    else:
        print(f"  {len(selected)} samples selected")
    return selected


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(model, tokenizer, dataset, args, label, selected_indices):
    device = next(model.parameters()).device

    prefill_times = []
    all_step_times = []
    per_sample = []

    for i in selected_indices:
        item = dataset[i]
        prompt_text = build_prompt(item)
        input_ids = tokenize_and_truncate(prompt_text, tokenizer, args.max_input_len)
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]

        prefill_time, step_times, n_generated = time_sample(
            model, tokenizer, input_ids, args.max_new_tokens, args.warmup_steps
        )

        prefill_times.append(prefill_time)
        all_step_times.extend(step_times)

        if step_times:
            avg_ms = sum(step_times) / len(step_times) * 1000
            tok_s  = 1.0 / (sum(step_times) / len(step_times))
        else:
            avg_ms = tok_s = float("nan")

        per_sample.append({
            "sample_id":            item.get("_id", i),
            "length":               item.get("length", ""),
            "difficulty":           item.get("difficulty", ""),
            "ctx_len":              seq_len,
            "prefill_ms":           round(prefill_time * 1000, 2),
            "avg_decode_ms_per_tok": round(avg_ms, 3),
            "decode_tok_per_s":     round(tok_s, 2),
            "n_decode_steps":       len(step_times),
        })

        print(f"  [{label}] {len(per_sample)}/{len(selected_indices)}: "
              f"ctx={seq_len} ({item.get('length','?')}), "
              f"prefill={prefill_time*1000:.0f}ms, "
              f"decode={avg_ms:.1f}ms/tok ({tok_s:.1f} tok/s), "
              f"steps={n_generated}")

    n = len(prefill_times)
    stats = {
        "label":                  label,
        "n_samples":              n,
        "avg_prefill_ms":         round(sum(prefill_times) / n * 1000, 2) if n else None,
        "avg_decode_ms_per_tok":  (
            round(sum(all_step_times) / len(all_step_times) * 1000, 3)
            if all_step_times else None
        ),
        "decode_tok_per_s":       (
            round(len(all_step_times) / sum(all_step_times), 2)
            if all_step_times else None
        ),
        "total_decode_steps":     len(all_step_times),
    }
    return stats, per_sample


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_results(per_sample, run_dir):
    path = Path(run_dir) / "samples.jsonl"
    with open(path, "w") as f:
        for record in per_sample:
            f.write(json.dumps(record) + "\n")


def save_summary(stats, run_dir, args, label):
    summary = {
        "label":                       label,
        "model":                       args.model,
        "overall_decode_tok_per_s":    stats["decode_tok_per_s"],
        "overall_avg_decode_ms_per_tok": stats["avg_decode_ms_per_tok"],
        "n_samples":                   stats["n_samples"],
        "avg_prefill_ms":              stats["avg_prefill_ms"],
        "total_decode_steps":          stats["total_decode_steps"],
    }
    if label != "baseline":
        summary.update({
            "page_size":      args.page_size,
            "top_k":          args.top_k,
            "sink_size":      args.sink_size,
            "recent_size":    args.recent_size,
            "compress_ratio": args.compress_ratio,
            "scoring_method": args.scoring_method,
            "unselected_mode": args.unselected_mode,
            "selection_mode": args.selection_mode,
        })
    path = Path(run_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to: {path}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(results):
    print("\n" + "=" * 65)
    print("DECODE SPEED SUMMARY  (LongBench v2)")
    print("=" * 65)
    for label, stats in results.items():
        tok_s = stats["decode_tok_per_s"]
        ms    = stats["avg_decode_ms_per_tok"]
        if tok_s is None:
            continue
        print(f"\n  {label.upper()}")
        print(f"    {tok_s:.1f} tok/s  |  {ms:.2f} ms/tok")
        print(f"    {stats['n_samples']} samples, "
              f"{stats['total_decode_steps']} decode steps, "
              f"avg prefill={stats['avg_prefill_ms']:.0f}ms")

    if "baseline" in results and "dct" in results:
        b = results["baseline"]["decode_tok_per_s"]
        d = results["dct"]["decode_tok_per_s"]
        if b and d:
            print(f"\n  Decode speedup (DCT / baseline): {d/b:.2f}x")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    original_forward = get_original_forward(args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)

    print("\nLoading LongBench v2 dataset...")
    dataset = list(load_dataset("THUDM/LongBench-v2", split="train"))
    print(f"  {len(dataset)} total samples")

    selected_indices = preselect_samples(dataset, tokenizer, args)

    results = {}

    def run_mode(label, patch=False):
        run_name = args.run_name or make_run_name(label, args)
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        restore_forward(args.model, original_forward, model)
        if patch:
            apply_dct_patch(args, model)

        stats, per_sample = benchmark(
            model, tokenizer, dataset, args, label, selected_indices
        )
        save_results(per_sample, run_dir)
        save_summary(stats, run_dir, args, label)
        results[label] = stats
        print(f"\nResults written to: {run_dir}/")

    if args.mode in ("baseline", "both"):
        print("\n" + "=" * 65)
        print("BASELINE (full attention)")
        print("=" * 65)
        run_mode("baseline", patch=False)

    if args.mode in ("dct", "both"):
        print("\n" + "=" * 65)
        print("DCT PAGE ATTENTION")
        print("=" * 65)
        run_mode("dct", patch=True)

    print_summary(results)


if __name__ == "__main__":
    main()