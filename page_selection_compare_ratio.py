"""
Compare page selection across different DCT compression ratios.

For each of the first N LongBench v2 samples, runs inference with different
compression ratios and measures how much the selected pages overlap pairwise.

Usage:
    python page_selection_compare_ratio.py
    python page_selection_compare_ratio.py --num_samples 5 --ratios 1.0 0.5 0.25 0.03
"""

import os
import sys
import json
import argparse
import random
from itertools import combinations

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_longbench import build_prompt, tokenize_and_truncate


# ---------------------------------------------------------------------------
# Capture state for page selections
# ---------------------------------------------------------------------------
class CaptureState:
    enabled = False
    selections = []  # list of [bsz, num_kv_heads, top_k] tensors
    scores = []      # list of [bsz, num_kv_heads, num_pages] tensors


def install_capture_hook():
    """Monkey-patch score_pages to capture selected indices during decode."""
    import dct_page_attention as dpa
    orig_score_pages = dpa.score_pages

    def capturing_score_pages(*args, **kwargs):
        indices, page_scores = orig_score_pages(*args, **kwargs)
        if CaptureState.enabled:
            CaptureState.selections.append(indices.detach().cpu())
            CaptureState.scores.append(page_scores.detach().cpu())
        return indices, page_scores

    dpa.score_pages = capturing_score_pages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare page selection across compression ratios"
    )
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_input_len", type=int, default=120000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=10)

    # DCT Page Attention params
    parser.add_argument("--page_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--scoring_method", type=str, default="mean",
                        choices=["mean", "max", "sum"])
    parser.add_argument("--group_agg_method", type=str, default="max",
                        choices=["mean", "max", "topp"])
    parser.add_argument("--unselected_mode", type=str, default="compressed",
                        choices=["drop", "compressed"])

    # Compression ratios to compare
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[1.0, 0.5, 0.25, 0.03])

    # JSON output
    parser.add_argument("--output_json", type=str,
                        default="page_selections.json",
                        help="Path to write JSON with per-head page selections "
                             "ordered by score rank")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------
def compute_pairwise_overlap(caps1, caps2, top_k):
    """
    Compute average page overlap between two capture lists.

    Each list contains tensors of shape [1, num_kv_heads, top_k], one per
    (decode_step, layer) invocation.

    Returns: (avg_common_pages, avg_pct, num_comparisons)
    """
    num_compare = min(len(caps1), len(caps2))
    if num_compare == 0:
        return 0.0, 0.0, 0

    total_overlap = 0
    total_count = 0

    for t in range(num_compare):
        sel1 = caps1[t]  # [1, num_kv_heads, top_k]
        sel2 = caps2[t]
        num_kv_heads = sel1.shape[1]

        for h in range(num_kv_heads):
            set1 = set(sel1[0, h].tolist())
            set2 = set(sel2[0, h].tolist())
            total_overlap += len(set1 & set2)
            total_count += 1

    avg_overlap = total_overlap / total_count
    avg_pct = avg_overlap / top_k * 100
    return avg_overlap, avg_pct, total_count


def compute_first_step_overlap(caps1, caps2, num_layers, top_k):
    """
    Same as above but only for the first decode step (first num_layers entries).
    This is the cleanest comparison since the query token is identical across
    compression ratios (same prefill).
    """
    n1 = min(len(caps1), num_layers)
    n2 = min(len(caps2), num_layers)
    num_compare = min(n1, n2)
    if num_compare == 0:
        return 0.0, 0.0, 0

    total_overlap = 0
    total_count = 0

    for t in range(num_compare):
        sel1 = caps1[t]
        sel2 = caps2[t]
        num_kv_heads = sel1.shape[1]

        for h in range(num_kv_heads):
            set1 = set(sel1[0, h].tolist())
            set2 = set(sel2[0, h].tolist())
            total_overlap += len(set1 & set2)
            total_count += 1

    avg_overlap = total_overlap / total_count
    avg_pct = avg_overlap / top_k * 100
    return avg_overlap, avg_pct, total_count


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------
def _spearman_rho(a, b):
    """Compute Spearman rank correlation between two 1-D numpy arrays."""
    n = len(a)
    rank_a = a.argsort().argsort().astype(np.float64)
    rank_b = b.argsort().argsort().astype(np.float64)
    d = rank_a - rank_b
    return 1.0 - 6.0 * np.sum(d * d) / (n * (n * n - 1))


def compute_spearman(scores_ref, scores_comp, limit=None):
    """
    Compute Spearman rank correlation between reference (r=1.0) and compressed
    page scores over ALL pages, per (invocation, head).

    Args:
        scores_ref:  list of [1, num_kv_heads, num_pages] score tensors
        scores_comp: list of [1, num_kv_heads, num_pages] score tensors
        limit: if set, only use the first `limit` invocations

    Returns: list of per-(invocation, head) rho values
    """
    n = min(len(scores_ref), len(scores_comp))
    if limit is not None:
        n = min(n, limit)
    rhos = []
    for t in range(n):
        sc_ref = scores_ref[t]   # [1, num_kv_heads, num_pages]
        sc_cmp = scores_comp[t]
        num_kv_heads = sc_ref.shape[1]
        for h in range(num_kv_heads):
            rho = _spearman_rho(sc_ref[0, h].float().numpy(), sc_cmp[0, h].float().numpy())
            rhos.append(rho)
    return rhos


def compute_spearman_per_layer(scores_ref, scores_comp, num_layers, limit=None):
    """
    Same as compute_spearman but returns a dict {layer_idx: [rho_values]}.

    Args:
        limit: if set, only use the first `limit` invocations
    """
    n = min(len(scores_ref), len(scores_comp))
    if limit is not None:
        n = min(n, limit)
    per_layer = {l: [] for l in range(num_layers)}
    for t in range(n):
        layer = t % num_layers
        sc_ref = scores_ref[t]
        sc_cmp = scores_comp[t]
        num_kv_heads = sc_ref.shape[1]
        for h in range(num_kv_heads):
            rho = _spearman_rho(sc_ref[0, h].float().numpy(), sc_cmp[0, h].float().numpy())
            per_layer[layer].append(rho)
    return per_layer


# ---------------------------------------------------------------------------
# Top-k recall
# ---------------------------------------------------------------------------
def compute_topk_recall(scores_ref, scores_comp, k_values, limit=None):
    """
    For each k in k_values, compute the fraction of the reference top-k pages
    that also appear in the compressed ratio's top-k.

    Args:
        scores_ref:  list of [1, num_kv_heads, num_pages] score tensors
        scores_comp: list of [1, num_kv_heads, num_pages] score tensors
        k_values: list of int, e.g. [5, 10, 15, 20, 25, 32]
        limit: if set, only use the first `limit` invocations

    Returns: dict {k: avg_recall_pct}
    """
    n = min(len(scores_ref), len(scores_comp))
    if limit is not None:
        n = min(n, limit)
    if n == 0:
        return {k: 0.0 for k in k_values}

    totals = {k: 0.0 for k in k_values}
    count = 0

    for t in range(n):
        sc_ref = scores_ref[t]   # [1, num_kv_heads, num_pages]
        sc_cmp = scores_comp[t]
        num_kv_heads = sc_ref.shape[1]
        for h in range(num_kv_heads):
            ref_ranked = sc_ref[0, h].argsort(descending=True)
            cmp_ranked = sc_cmp[0, h].argsort(descending=True)
            for k in k_values:
                ref_topk = set(ref_ranked[:k].tolist())
                cmp_topk = set(cmp_ranked[:k].tolist())
                totals[k] += len(ref_topk & cmp_topk) / k * 100
            count += 1

    return {k: totals[k] / count for k in k_values}


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def print_matrix(ratios, get_value, top_k, title):
    """Print a symmetric matrix of pairwise values."""
    print(f"\n{title}")
    print("-" * (14 + 10 * len(ratios)))

    # Header
    print(f"{'':>12}", end="")
    for r in ratios:
        print(f"{'r=' + str(r):>10}", end="")
    print()

    # Rows
    for r1 in ratios:
        print(f"  r={r1:<8}", end="")
        for r2 in ratios:
            if r1 == r2:
                val_str = f"{top_k:.1f}"
            else:
                val = get_value(r1, r2)
                val_str = f"{val:.1f}"
            print(f"{val_str:>10}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    ratios = sorted(args.ratios, reverse=True)

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load dataset
    print("Loading LongBench v2 dataset...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    samples = list(dataset)[:args.num_samples]
    print(f"Using first {len(samples)} samples")

    # Apply page attention monkey-patch (initial ratio; updated per run)
    from dct_page_attention import replace_qwen2_attn
    import dct_page_attention as dpa

    replace_qwen2_attn(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=ratios[0],
        scoring_method=args.scoring_method,
        group_agg_method=args.group_agg_method,
        unselected_mode=args.unselected_mode,
    )
    install_capture_hook()

    # Load model
    print(f"\nLoading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded ({num_layers} layers, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    # -----------------------------------------------------------------------
    # Run inference and capture page selections
    # -----------------------------------------------------------------------
    # all_captures[sample_idx][ratio] = list of selected_indices tensors
    # all_scores[sample_idx][ratio]   = list of page_scores tensors
    all_captures = []
    all_scores = []

    for sample_idx, item in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}/{len(samples)} (id={item['_id']})")
        print(f"{'='*60}")

        prompt_text = build_prompt(item)
        input_ids = tokenize_and_truncate(
            prompt_text, tokenizer, args.max_input_len
        )
        input_ids = input_ids.to(model.device)
        input_len = input_ids.shape[1]

        pageable_len = input_len - args.sink_size - args.recent_size
        num_pages = pageable_len // args.page_size
        print(f"  Input: {input_len} tokens, {num_pages} pages")

        sample_captures = {}
        sample_scores = {}

        for ratio in ratios:
            # Update compression ratio in the global config
            dpa._dct_page_cfg.compress_ratio = ratio
            comp_size = max(1, int(args.page_size * ratio))

            # Reset capture buffer
            CaptureState.selections.clear()
            CaptureState.scores.clear()
            CaptureState.enabled = True

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            CaptureState.enabled = False

            num_generated = output_ids.shape[1] - input_len
            num_decode_calls = len(CaptureState.selections) // num_layers

            print(f"  ratio={ratio:>4} (comp_size={comp_size:>3}): "
                  f"generated {num_generated} tokens, "
                  f"{num_decode_calls} decode steps with page attn")

            sample_captures[ratio] = [t.clone() for t in CaptureState.selections]
            sample_scores[ratio] = [t.clone() for t in CaptureState.scores]

        all_captures.append(sample_captures)
        all_scores.append(sample_scores)

    # -----------------------------------------------------------------------
    # Compute pairwise overlap
    # -----------------------------------------------------------------------
    ratio_pairs = list(combinations(ratios, 2))
    top_k = args.top_k

    # -- All decode steps --
    all_step_overlaps = {}
    for r1, r2 in ratio_pairs:
        overlaps = []
        for sample_caps in all_captures:
            avg, pct, _ = compute_pairwise_overlap(
                sample_caps[r1], sample_caps[r2], top_k
            )
            overlaps.append(avg)
        all_step_overlaps[(r1, r2)] = np.mean(overlaps)
        all_step_overlaps[(r2, r1)] = np.mean(overlaps)

    # -- First decode step only --
    first_step_overlaps = {}
    for r1, r2 in ratio_pairs:
        overlaps = []
        for sample_caps in all_captures:
            avg, pct, _ = compute_first_step_overlap(
                sample_caps[r1], sample_caps[r2], num_layers, top_k
            )
            overlaps.append(avg)
        first_step_overlaps[(r1, r2)] = np.mean(overlaps)
        first_step_overlaps[(r2, r1)] = np.mean(overlaps)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PAGE SELECTION OVERLAP (top_k={top_k}, {len(samples)} samples)")
    print("=" * 70)

    print_matrix(
        ratios,
        lambda r1, r2: all_step_overlaps[(r1, r2)],
        top_k,
        f"Average common pages — ALL decode steps (out of {top_k}):"
    )

    print_matrix(
        ratios,
        lambda r1, r2: all_step_overlaps[(r1, r2)] / top_k * 100,
        100.0,
        "As percentage (%):"
    )

    print_matrix(
        ratios,
        lambda r1, r2: first_step_overlaps[(r1, r2)],
        top_k,
        f"Average common pages — FIRST decode step only (out of {top_k}):"
    )

    print_matrix(
        ratios,
        lambda r1, r2: first_step_overlaps[(r1, r2)] / top_k * 100,
        100.0,
        "As percentage (%):"
    )

    # -- Per-sample breakdown (all steps) --
    print(f"\nPer-sample breakdown (avg common pages, all decode steps):")
    print("-" * (10 + 14 * len(ratio_pairs)))

    # Header
    print(f"{'Sample':>8}", end="")
    for r1, r2 in ratio_pairs:
        label = f"{r1}v{r2}"
        print(f"{label:>14}", end="")
    print()

    for sample_idx, sample_caps in enumerate(all_captures):
        print(f"  {sample_idx:>6}", end="")
        for r1, r2 in ratio_pairs:
            avg, _, _ = compute_pairwise_overlap(
                sample_caps[r1], sample_caps[r2], top_k
            )
            print(f"{avg:>14.1f}", end="")
        print()

    # Averages row
    print(f"  {'avg':>6}", end="")
    for r1, r2 in ratio_pairs:
        print(f"{all_step_overlaps[(r1, r2)]:>14.1f}", end="")
    print()

    print(f"\nConfig: page_size={args.page_size}, scoring={args.scoring_method}, "
          f"group_agg={args.group_agg_method}, unselected={args.unselected_mode}")

    # -----------------------------------------------------------------------
    # Spearman rank correlation vs r=1.0
    # -----------------------------------------------------------------------
    assert 1.0 in ratios, "ratios must include 1.0 as the reference"
    comp_ratios = [r for r in ratios if r != 1.0]

    def _collect_spearman(limit_fn):
        """Collect Spearman rhos for each compressed ratio."""
        result = {}
        for cr in comp_ratios:
            all_rhos = []
            for sample_idx in range(len(all_scores)):
                sc = all_scores[sample_idx]
                lim = limit_fn(sc)
                rhos = compute_spearman(sc[1.0], sc[cr], limit=lim)
                all_rhos.extend(rhos)
            result[cr] = np.array(all_rhos)
        return result

    spearman_all = _collect_spearman(lambda sc: None)
    spearman_first = _collect_spearman(lambda sc: num_layers)

    print("\n" + "=" * 70)
    print("SPEARMAN RANK CORRELATION vs r=1.0 (over all pages)")
    print("=" * 70)

    for label, sp_data in [("All decode steps", spearman_all),
                           ("First decode step only", spearman_first)]:
        print(f"\n{label}:")
        print(f"  {'Ratio':>8} {'Avg rho':>10} {'Std rho':>10} "
              f"{'Min rho':>10} {'Max rho':>10}")
        print(f"  {'-'*48}")
        for cr in comp_ratios:
            rhos = sp_data[cr]
            print(f"  r={cr:<5} {np.mean(rhos):>10.4f} {np.std(rhos):>10.4f} "
                  f"{np.min(rhos):>10.4f} {np.max(rhos):>10.4f}")

    # Per-layer Spearman breakdown (first decode step, avg over samples & heads)
    print(f"\nPer-layer Spearman (first decode step, avg over samples & heads):")
    header = f"  {'Layer':>6}"
    for cr in comp_ratios:
        header += f"  {'r=' + str(cr):>8}"
    print(header)
    print(f"  {'-' * (6 + 10 * len(comp_ratios))}")

    per_layer_data = {}
    for cr in comp_ratios:
        merged = {l: [] for l in range(num_layers)}
        for sample_idx in range(len(all_scores)):
            sc = all_scores[sample_idx]
            pl = compute_spearman_per_layer(sc[1.0], sc[cr], num_layers,
                                            limit=num_layers)
            for l in range(num_layers):
                merged[l].extend(pl[l])
        per_layer_data[cr] = merged

    for l in range(num_layers):
        row = f"  {l:>6}"
        for cr in comp_ratios:
            avg_rho = np.mean(per_layer_data[cr][l])
            row += f"  {avg_rho:>8.4f}"
        print(row)

    # -----------------------------------------------------------------------
    # Top-k recall vs r=1.0
    # -----------------------------------------------------------------------
    k_values = [5, 10, 15, 20, 25, top_k]

    print("\n" + "=" * 70)
    print("TOP-k RECALL vs r=1.0 (%)")
    print("=" * 70)

    for label, limit_fn in [("All decode steps", lambda sc: None),
                            ("First decode step only", lambda sc: num_layers)]:
        print(f"\n{label}:")
        header = f"  {'Ratio':>8}"
        for k in k_values:
            header += f"  {'k=' + str(k):>8}"
        print(header)
        print(f"  {'-' * (8 + 10 * len(k_values))}")

        for cr in comp_ratios:
            per_k_accum = {k: [] for k in k_values}
            for sample_idx in range(len(all_scores)):
                sc = all_scores[sample_idx]
                lim = limit_fn(sc)
                recalls = compute_topk_recall(sc[1.0], sc[cr], k_values,
                                              limit=lim)
                for k in k_values:
                    per_k_accum[k].append(recalls[k])
            row = f"  r={cr:<5}"
            for k in k_values:
                row += f"  {np.mean(per_k_accum[k]):>8.1f}"
            print(row)

    # -----------------------------------------------------------------------
    # Write JSON with selected pages ordered by score rank
    # -----------------------------------------------------------------------
    json_data = {
        "config": {
            "page_size": args.page_size,
            "top_k": args.top_k,
            "sink_size": args.sink_size,
            "recent_size": args.recent_size,
            "scoring_method": args.scoring_method,
            "group_agg_method": args.group_agg_method,
            "ratios": ratios,
            "num_layers": num_layers,
        },
        "samples": [],
    }

    for sample_idx in range(len(all_captures)):
        sample_caps = all_captures[sample_idx]
        sample_sc = all_scores[sample_idx]
        # Different ratios may produce different numbers of decode steps;
        # use the minimum across all ratios so every entry has all ratios.
        num_invocations = min(len(sample_caps[r]) for r in ratios)

        sample_entry = {
            "sample_idx": sample_idx,
            "sample_id": samples[sample_idx]["_id"],
            "num_pages": sample_sc[ratios[0]][0].shape[-1],
            "selections": [],
        }

        for t in range(num_invocations):
            decode_step = t // num_layers
            layer = t % num_layers

            invocation_entry = {
                "decode_step": decode_step,
                "layer": layer,
            }

            # For each ratio, sort selected pages by score (highest first)
            num_kv_heads = sample_caps[ratios[0]][t].shape[1]
            for ratio in ratios:
                sel = sample_caps[ratio][t]   # [1, num_kv_heads, top_k]
                sc = sample_sc[ratio][t]      # [1, num_kv_heads, num_pages]

                heads_list = []
                for h in range(num_kv_heads):
                    page_indices = sel[0, h]              # [top_k]
                    page_scores = sc[0, h][page_indices]  # [top_k]
                    rank_order = page_scores.argsort(descending=True)
                    pages_by_rank = page_indices[rank_order].tolist()
                    heads_list.append(pages_by_rank)

                invocation_entry[str(ratio)] = heads_list

            sample_entry["selections"].append(invocation_entry)

        json_data["samples"].append(sample_entry)

    output_path = os.path.join(os.path.dirname(__file__) or ".", args.output_json)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON selections written to: {output_path}")


if __name__ == "__main__":
    main()