"""Needle-position diagnostic: measure each selector's needle-page hit rate
directly, not via set_recall (which uses oracle_max set as proxy ceiling).

For RULER niah_multikey_3: the answer is a UUID in `outputs[0]`. Find its
token position in `input`, convert to paged-region page index, then per
(layer, decode_step) check whether that page is in each selector's top-K.

Compares: DCT lowpass cs=4, Quest, ShadowKV, InfLLM (paper-faithful), and
oracle_max ceiling (cs=32 identity proxy).
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from oracle.attention_mass_recall_ruler import (
    _get_dct_lowpass_projection_matrix,
    _get_full_dct_matrix,
    compute_per_page_mass,
    compute_dct_lowpass_proxy_scores,
    compute_quest_scores,
    compute_shadowkv_scores,
    compute_infllm_paper_scores,
    _compute_prefill_rm,
)
from oracle.attention_mass_recall_ruler_quest import (
    set_recording_hook, _install_recording_forward,
)


def find_needle_pages(
    input_text: str, answer: str, tokenizer,
    page_size: int, num_sink_pages: int,
    chat_template: bool = True,
) -> tuple[set[int], int]:
    """Find paged-region page indices containing the answer string.

    Returns (set of page indices in paged region, num_paged_pages).
    A negative index means the answer falls in sink (-1) or recent (-2) regions.
    """
    char_idx = input_text.find(answer)
    if char_idx < 0:
        return set(), -1
    char_end = char_idx + len(answer)

    full_ids = tokenizer(input_text, return_tensors="pt").input_ids[0].tolist()
    prefix_ids = tokenizer(input_text[:char_idx], return_tensors="pt").input_ids[0].tolist()
    prefix_with_answer_ids = tokenizer(input_text[:char_end], return_tensors="pt").input_ids[0].tolist()

    token_start = len(prefix_ids)
    token_end = len(prefix_with_answer_ids)

    sink_len = num_sink_pages * page_size
    paged_token_start = token_start - sink_len
    paged_token_end = token_end - sink_len

    if paged_token_end < 0:
        return {-1}, len(full_ids)
    page_start = max(0, paged_token_start) // page_size
    page_end = max(0, paged_token_end - 1) // page_size
    return set(range(page_start, page_end + 1)), len(full_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--task", type=str, default="niah_multikey_3")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=64,
                   help="TOTAL pages selected (sink + middle + recent). Middle = top_k - sink - recent.")
    p.add_argument("--num_sink_pages", type=int, default=1)
    p.add_argument("--num_recent_pages", type=int, default=4)
    p.add_argument("--compress_ratio", type=float, default=0.125)
    p.add_argument("--num_decode_steps", type=int, default=5)
    p.add_argument("--infllm_local_window", type=int, default=4096)
    p.add_argument("--infllm_repr_topk", type=int, default=4)
    args = p.parse_args()

    middle_top_k = max(1, args.top_k - args.num_sink_pages - args.num_recent_pages)
    comp_size = max(1, int(args.page_size * args.compress_ratio))
    data_root = "benchmark/data/ruler_data/qwen3"
    data_file = f"{data_root}/{args.seq_len}/{args.task}/validation.jsonl"
    print(f"Loading samples from {data_file}")
    with open(data_file) as fp:
        samples = [json.loads(line) for line in fp][:args.num_samples]

    print(f"Loading {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    family = "qwen3"
    _install_recording_forward(None, family)

    # Pre-compute needle pages per sample.
    needle_info: list[tuple[set[int], int]] = []
    for s in samples:
        ans = s["outputs"][0]
        pages, total_tokens = find_needle_pages(
            s["input"], ans, tok, args.page_size, args.num_sink_pages,
        )
        needle_info.append((pages, total_tokens))
    valid_idx = [i for i, (p, _) in enumerate(needle_info) if p and -1 not in p and -2 not in p]
    print(f"Samples with identifiable paged needle: {len(valid_idx)}/{len(samples)}")
    for i, (p, t) in enumerate(needle_info):
        if not p:
            print(f"  sample {i}: needle NOT FOUND (skip)")

    selectors = ("proxy", "quest", "shadowkv", "infllm", "oraclemax")
    # Aggregate hit rate: per (layer, decode_step) average over samples.
    # Then average over layers/steps. Single scalar per selector.
    hit_buckets: dict[str, list[float]] = {sel: [] for sel in selectors}
    # Per-(layer, step) breakdown: hit_per_layer_step[(L, S)][sel] = list of mean-over-kv-head hit rates
    hit_per_layer_step: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {sel: [] for sel in selectors}
    )

    # InfLLM paper-faithful: capture prefill r_m per layer
    prefill_rm_cache: dict[int, torch.Tensor] = {}
    step_by_layer: dict[int, int] = {}

    sample_idx_iter = iter(valid_idx)
    current_needle_pages: set[int] = set()
    current_num_paged_pages: int = -1

    def hook(payload):
        nonlocal current_needle_pages, current_num_paged_pages, prefill_rm_cache, step_by_layer
        layer_idx = int(payload["layer_idx"])
        if payload.get("phase") == "prefill":
            # Paper-faithful InfLLM r_m: cache once per (sample, layer)
            q_prefill = payload["query_states_prefill"]
            k_prefill = payload["key_states_prefill"]
            G = int(payload["num_kv_groups"])
            r_m = _compute_prefill_rm(
                q_prefill, k_prefill, G, l_L=args.infllm_local_window,
            )
            prefill_rm_cache[layer_idx] = r_m
            return

        # decode payload
        if not current_needle_pages:
            return
        decode_step = step_by_layer.get(layer_idx, 0)
        step_by_layer[layer_idx] = decode_step + 1
        kf = payload["key_states_full"]
        qf = payload["query_states"]
        G = int(payload["num_kv_groups"])
        kv_len = kf.shape[2]
        ps = args.page_size
        sink_len = args.num_sink_pages * ps
        recent_min = args.num_recent_pages * ps
        if kv_len < sink_len + ps + recent_min:
            return
        num_pages = (kv_len - sink_len - recent_min) // ps
        if num_pages != current_num_paged_pages:
            # Page count changed since needle computation — skip (extremely rare,
            # happens at page boundaries).
            return
        paged_end = sink_len + num_pages * ps
        paged_k = kf[:, :, sink_len:paged_end, :].view(1, kf.shape[1], num_pages, ps, kf.shape[-1])

        # Compute selector scores
        K = min(middle_top_k, num_pages)
        # DCT lowpass cs=4
        proxy_scores = compute_dct_lowpass_proxy_scores(
            qf, paged_k, comp_size, G, "max", "max",
            comp_kv_quant="none",
        )                                                        # [H_kv, P]
        # oracle_max = identity proxy at cs=page_size
        oracle_scores = compute_dct_lowpass_proxy_scores(
            qf, paged_k, ps, G, "max", "max",
            comp_kv_quant="none",
        )
        # Quest
        quest_scores = compute_quest_scores(qf, paged_k, G, "max")
        # ShadowKV
        shadow_scores = compute_shadowkv_scores(qf, paged_k, G, "max")
        # InfLLM paper-faithful
        r_m_full = prefill_rm_cache.get(layer_idx)
        if r_m_full is not None:
            r_m_seq = r_m_full.shape[-1]
            slice_end = min(paged_end, r_m_seq)
            r_m_paged_flat = r_m_full[:, sink_len:slice_end]
            tokens_avail = r_m_paged_flat.shape[-1]
            pages_avail = tokens_avail // ps
            if pages_avail < num_pages:
                pad_amount = (num_pages * ps) - tokens_avail
                r_m_paged_flat = torch.cat([
                    r_m_paged_flat,
                    torch.full((kf.shape[1], pad_amount), float("-inf"),
                               dtype=r_m_paged_flat.dtype),
                ], dim=-1)
            r_m_paged = r_m_paged_flat.view(kf.shape[1], num_pages, ps).to(paged_k.device)
            R = min(args.infllm_repr_topk, ps)
            repr_idx_local = r_m_paged.topk(R, dim=-1).indices.long()
            infllm_scores = compute_infllm_paper_scores(
                qf, paged_k, repr_idx_local, G, "max",
            )
        else:
            infllm_scores = quest_scores  # fallback (won't fire if prefill captured)

        # Get top-K page sets per selector — per-kv-head
        all_scores = {
            "proxy": proxy_scores, "oraclemax": oracle_scores,
            "quest": quest_scores, "shadowkv": shadow_scores, "infllm": infllm_scores,
        }
        # needle_hit per selector = fraction of kv-heads whose top-K includes ANY needle page
        for sel, scores in all_scores.items():
            topk = torch.topk(scores, K, dim=-1).indices            # [H_kv, K]
            # check if any needle page in each kv-head's set
            np_tensor = torch.tensor(list(current_needle_pages), dtype=torch.long, device=topk.device)
            hits = (topk.unsqueeze(-1) == np_tensor.view(1, 1, -1)).any(dim=-1).any(dim=-1)  # [H_kv]
            mean_hit = hits.float().mean().item()
            hit_buckets[sel].append(mean_hit)
            hit_per_layer_step[(layer_idx, decode_step)][sel].append(mean_hit)

    set_recording_hook(hook)

    for vi in valid_idx:
        s = samples[vi]
        current_needle_pages, total_tokens = needle_info[vi]
        # paged page count at start of decode (no decode tokens yet, kv_len = total_tokens)
        kv_len = total_tokens
        sink_len = args.num_sink_pages * args.page_size
        recent_min = args.num_recent_pages * args.page_size
        current_num_paged_pages = (kv_len - sink_len - recent_min) // args.page_size
        prefill_rm_cache = {}
        step_by_layer = {}
        input_ids = tok(s["input"], return_tensors="pt").input_ids.to("cuda:0")
        with torch.no_grad():
            model.generate(
                input_ids, max_new_tokens=args.num_decode_steps,
                do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id,
            )

    set_recording_hook(None)

    print()
    print(f"{'selector':<10} {'needle_page_hit_rate':>22}")
    print("-" * 35)
    for sel in selectors:
        vals = hit_buckets[sel]
        if vals:
            print(f"{sel:<10} {sum(vals)/len(vals):>22.4f}  (n={len(vals)} records)")
        else:
            print(f"{sel:<10} {'no records':>22}")

    # Per-layer breakdown (averaged over decode steps and samples)
    print("\n=== Per-layer needle_hit_rate (averaged over decode steps × samples) ===")
    print(f"{'layer':>5} | " + " | ".join(f"{sel:>9}" for sel in selectors))
    layers = sorted({L for (L, _) in hit_per_layer_step.keys()})
    per_layer_avg: dict[str, list[float]] = {sel: [] for sel in selectors}
    for L in layers:
        row = {sel: [] for sel in selectors}
        for s in range(args.num_decode_steps):
            entry = hit_per_layer_step.get((L, s))
            if entry:
                for sel in selectors:
                    row[sel].extend(entry[sel])
        cell = []
        for sel in selectors:
            if row[sel]:
                avg = sum(row[sel]) / len(row[sel])
                per_layer_avg[sel].append(avg)
                cell.append(f"{avg:>9.3f}")
            else:
                cell.append(f"{'—':>9}")
        print(f"{L:>5} | " + " | ".join(cell))

    # Per-decode-step breakdown (averaged over layers × samples)
    print("\n=== Per-decode-step needle_hit_rate (averaged over layers × samples) ===")
    print(f"{'step':>5} | " + " | ".join(f"{sel:>9}" for sel in selectors))
    for s in range(args.num_decode_steps):
        row = {sel: [] for sel in selectors}
        for L in layers:
            entry = hit_per_layer_step.get((L, s))
            if entry:
                for sel in selectors:
                    row[sel].extend(entry[sel])
        cell = []
        for sel in selectors:
            if row[sel]:
                cell.append(f"{sum(row[sel])/len(row[sel]):>9.3f}")
            else:
                cell.append(f"{'—':>9}")
        print(f"{s:>5} | " + " | ".join(cell))

    # Late-layer summary (last 1/3 of layers)
    if layers:
        late_layers = layers[len(layers)*2//3:]
        print(f"\n=== Late layers only (layers {late_layers[0]}-{late_layers[-1]}) ===")
        for sel in selectors:
            vals = [per_layer_avg[sel][i] for i, L in enumerate(layers) if L in late_layers]
            if vals:
                print(f"  {sel:<10}: {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
