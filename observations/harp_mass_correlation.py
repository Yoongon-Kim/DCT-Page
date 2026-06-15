#!/usr/bin/env python3
"""Stage-0 HARP premise validation — Spearman correlation between Haar
detail-coefficient features and true per-page attention mass.

Hypothesis under test (HARP premise):
    "If K varies a lot across hidden dimensions inside a page, that page
     receives high attention mass." Operationally:
        Haar H L2-norm summary  ↔  true softmax mass per page.

The recording-forward hook from attention_mass_recall_ruler.py
runs the **unmodified full-KV forward** and captures (Q, K) per decode
step. We then, per layer and decode step:
  1. Reshape K into [pages × page_size] (paged_k), preserving the
     sink/recent floors HARP keeps anyway.
  2. Compute per-page features:
       (Q-agnostic, "literal premise")  s_max, s_sum, s_top, s_l3_norm,
                                        s_total_var
       (Q-aware Haar, "HARP actual")    haar_lowpass, haar_inject
       (Q-aware baselines)              quest, dct_lowpass, oracle_max
  3. Compute true per-page softmax mass (sink + paged + recent share
     denominator).
  4. Per query head, Spearman ρ across pages between each feature and
     mass. Average across heads and decode steps inside a layer.

Outputs per-layer ρ values to ``summary.json``. Stage-0 gate (see
``.claude/plans/harp-experiment-plan.md``):
    late-third layers average ρ(haar_inject) ≥ 0.30
                          and  ρ(haar_inject) ≥ ρ(dct_lowpass) + 0.05.

Usage:
  python observations/harp_mass_correlation.py \\
    --base_model Qwen/Qwen3-8B \\
    --tasks niah_multikey_3 niah_multivalue qa_2 \\
    --seq_len 32768 --num_samples 5 \\
    --page_size 32 --num_decode_steps 8 \\
    --output_dir results_harp_premise --run_name qwen3_premise
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from observations.attention_mass_recall_ruler import (
    _install_recording_forward,
    _model_family,
    cleanup_model,
    load_model,
    set_recording_hook,
)
from observations.attention_mass_recall_ruler import (
    ALL_TASKS,
    compute_dct_lowpass_proxy_scores,
    compute_oracle_max_scores,
    compute_per_page_mass,
    compute_quest_scores,
    load_task_configs,
)
from observations.harp_kernels import (
    haar_page_features,
    haar_score_per_page,
)
from eval_ruler import infer_model_family


# ---------------------------------------------------------------------------
# Spearman correlation (along last dim, treats ties via argsort)
# ---------------------------------------------------------------------------
def _ranks_along_last(x: torch.Tensor) -> torch.Tensor:
    """Dense ranks (1..N) along the last axis. Ties broken by argsort order
    (good enough for continuous features; mass / norms rarely tie)."""
    order = x.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    n = x.shape[-1]
    arange = torch.arange(
        n, device=x.device, dtype=torch.float32,
    ).expand_as(order)
    ranks.scatter_(-1, order, arange)
    return ranks


def spearman_corr_last(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12,
) -> torch.Tensor:
    """Spearman ρ along dim=-1. Returns x.shape[:-1]."""
    rx = _ranks_along_last(x.float())
    ry = _ranks_along_last(y.float())
    rx = rx - rx.mean(dim=-1, keepdim=True)
    ry = ry - ry.mean(dim=-1, keepdim=True)
    num = (rx * ry).sum(dim=-1)
    den = (rx.pow(2).sum(dim=-1) * ry.pow(2).sum(dim=-1)).sqrt().clamp(min=eps)
    return num / den


# ---------------------------------------------------------------------------
# Features computed per decode step
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Q-agnostic ("literal premise"): per-(kv-head, page).
    "s_max",
    "s_sum",
    "s_top",
    "s_l3_norm",
    "s_total_var",
    # Q-aware Haar (HARP's actual mechanism): per-(kv-head, page).
    "haar_lowpass",
    "haar_inject",
    # Q-aware baselines.
    "quest",
    "dct_lowpass",
    "oracle_max",
]

# Selectors evaluated for top-K mass recall (apples-to-apples with
# attention_mass_recall_ruler.py's mass_recall_* family).
SELECTORS_FOR_RECALL = [
    "haar_inject",
    "haar_lowpass",
    "dct_lowpass",
    "quest",
    "oracle_max",
]
DEFAULT_K_CANDIDATES = [8, 16, 32, 64, 128]


def compute_all_features(
    query_states: torch.Tensor,
    paged_k: torch.Tensor,
    num_kv_groups: int,
    levels: int,
    top_k: int,
    beta: float,
    dct_comp_size: int,
) -> dict[str, torch.Tensor]:
    """Returns dict feature_name -> [H_kv, P] tensor."""
    feats: dict[str, torch.Tensor] = {}

    qa = haar_page_features(paged_k, levels=levels, top_k=top_k)
    for k in ("s_max", "s_sum", "s_top", "s_l3_norm", "s_total_var"):
        feats[k] = qa[k].squeeze(0)        # [H_kv, P]

    feats["haar_lowpass"] = haar_score_per_page(
        query_states, paged_k, num_kv_groups,
        levels=levels, top_k=top_k, beta=beta,
        use_injection=False, group_agg="mean",
    )
    feats["haar_inject"] = haar_score_per_page(
        query_states, paged_k, num_kv_groups,
        levels=levels, top_k=top_k, beta=beta,
        use_injection=True, group_agg="mean",
    )
    feats["quest"] = compute_quest_scores(
        query_states, paged_k, num_kv_groups, group_agg_method="mean",
    )
    feats["dct_lowpass"] = compute_dct_lowpass_proxy_scores(
        query_states, paged_k,
        comp_size=dct_comp_size, num_kv_groups=num_kv_groups,
        group_agg_method="mean", scoring_method="max",
    )
    feats["oracle_max"] = compute_oracle_max_scores(
        query_states, paged_k, num_kv_groups, group_agg_method="mean",
    )
    return feats


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
def _topk_selected_mass(
    scores_kv: torch.Tensor,         # [H_kv, P]
    page_mass: torch.Tensor,         # [H_q, P]
    h_kv_of_q: torch.Tensor,         # [H_q]
    K: int,
) -> torch.Tensor:
    """Σ m[p] over selector's top-K pages, per query head.

    Selection happens at kv-head granularity (all q-heads in a GQA group
    share the same selected pages), matching how attention_mass_recall_ruler
    selectors operate. Returns [H_q].
    """
    P = scores_kv.shape[-1]
    K_eff = min(K, P)
    if K_eff <= 0:
        return page_mass.new_zeros(page_mass.shape[0])
    topk_idx = torch.topk(scores_kv, K_eff, dim=-1).indices         # [H_kv, K]
    topk_idx_q = topk_idx[h_kv_of_q]                                # [H_q, K]
    return torch.gather(page_mass, -1, topk_idx_q).sum(-1)          # [H_q]


def _topk_ceiling_mass(page_mass: torch.Tensor, K: int) -> torch.Tensor:
    """True per-q-head top-K mass — upper bound any K-budget selector can hit."""
    P = page_mass.shape[-1]
    K_eff = min(K, P)
    if K_eff <= 0:
        return page_mass.new_zeros(page_mass.shape[0])
    return torch.topk(page_mass, K_eff, dim=-1).values.sum(-1)      # [H_q]


class HaarMassRecorder:
    """Per-layer accumulator. ρ aggregated as mean-of-means across
    (decode step × query head). Also tracks top-K mass recall per
    (selector, K) for apples-to-apples comparison with
    attention_mass_recall_ruler.py."""

    def __init__(
        self,
        page_size: int,
        num_sink_pages: int,
        num_recent_pages: int,
        levels: int,
        haar_top_k: int,
        haar_beta: float,
        dct_comp_size: int,
        num_decode_steps: int,
        k_candidates: list[int],
        num_skip_layers: int = 0,
    ) -> None:
        self.page_size = page_size
        self.num_sink_pages = num_sink_pages
        self.num_recent_pages = num_recent_pages
        self.levels = levels
        self.haar_top_k = haar_top_k
        self.haar_beta = haar_beta
        self.dct_comp_size = dct_comp_size
        self.num_decode_steps = num_decode_steps
        self.k_candidates = sorted(set(int(k) for k in k_candidates))
        self.num_skip_layers = num_skip_layers

        self._step_counter: dict[int, int] = {}
        # Per (layer, feature) running sum + count of ρ values.
        self.sum_rho: dict[int, dict[str, float]] = {}
        self.cnt_rho: dict[int, dict[str, int]] = {}
        # Per (layer, K, selector) — total mass_recall = sink + recent + selected.
        # Stored as floats; mean = sum / count over (head × step).
        self.sum_recall: dict[int, dict[tuple[int, str], float]] = {}
        self.sum_selected: dict[int, dict[tuple[int, str], float]] = {}
        self.cnt_recall: dict[int, dict[tuple[int, str], int]] = {}
        # Per (layer, K) — ceiling (true top-K mass) and floor (sink+recent).
        self.sum_ceiling: dict[int, dict[int, float]] = {}
        self.sum_floor: dict[int, float] = {}
        self.cnt_kfloor: dict[int, int] = {}
        # Per layer: count of mass-tied / degenerate steps (rho == nan).
        self.bad_steps: dict[int, int] = {}

    def __call__(self, payload: dict[str, Any]) -> None:
        layer_idx = int(payload["layer_idx"])
        if layer_idx < self.num_skip_layers:
            return
        step = self._step_counter.get(layer_idx, 0)
        if step >= self.num_decode_steps:
            return
        self._step_counter[layer_idx] = step + 1

        query_states = payload["query_states"]            # [1, H_q, 1, d]
        key_full = payload["key_states_full"]             # [1, H_kv, kv_len, d]
        num_kv_groups = int(payload["num_kv_groups"])

        bsz, H_kv, kv_len, d = key_full.shape
        sink_len = self.num_sink_pages * self.page_size
        recent_min = self.num_recent_pages * self.page_size
        if kv_len < sink_len + self.page_size + recent_min + 1:
            return
        num_pages = (kv_len - sink_len - recent_min) // self.page_size
        if num_pages < 4:               # need ≥ 4 pages for ρ to be meaningful
            return
        paged_end = sink_len + num_pages * self.page_size

        sink_k = key_full[:, :, :sink_len, :].contiguous()
        paged_k = key_full[:, :, sink_len:paged_end, :].view(
            bsz, H_kv, num_pages, self.page_size, d,
        )
        recent_k = key_full[:, :, paged_end:, :].contiguous()

        with torch.no_grad():
            page_mass, sink_mass, recent_mass = compute_per_page_mass(
                query_states, sink_k, paged_k, recent_k, num_kv_groups,
            )                                          # mass: [H_q, P], floor: [H_q]

            feats = compute_all_features(
                query_states, paged_k, num_kv_groups,
                levels=self.levels, top_k=self.haar_top_k,
                beta=self.haar_beta, dct_comp_size=self.dct_comp_size,
            )

        # Spearman ρ per query head (broadcast kv-head features to H_q).
        H_q = page_mass.shape[0]
        sums = self.sum_rho.setdefault(layer_idx, {k: 0.0 for k in FEATURE_NAMES})
        cnts = self.cnt_rho.setdefault(layer_idx, {k: 0 for k in FEATURE_NAMES})

        # Mass variance check — if mass is essentially constant (e.g. cold
        # prefill), ranks collapse and ρ is meaningless. Skip in that case.
        mass_std = page_mass.std(dim=-1)               # [H_q]
        live = mass_std > 1e-9                         # [H_q] bool
        if not bool(live.any()):
            self.bad_steps[layer_idx] = self.bad_steps.get(layer_idx, 0) + 1
            return

        # Expand kv-head features to q-head index.
        h_kv_of_q = torch.arange(
            H_q, device=page_mass.device,
        ) // num_kv_groups                              # [H_q]

        for name in FEATURE_NAMES:
            f = feats[name]                             # [H_kv, P]
            f_q = f[h_kv_of_q]                          # [H_q, P]
            rho = spearman_corr_last(f_q, page_mass)    # [H_q]
            # Only count heads where mass has variance AND ρ is finite.
            valid = live & torch.isfinite(rho)
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            sums[name] += float(rho[valid].sum().item())
            cnts[name] += n_valid

        # ----- Top-K mass recall (apples-to-apples with attention_mass_recall_ruler) -----
        floor = sink_mass + recent_mass                 # [H_q]
        rec_sum = self.sum_recall.setdefault(layer_idx, {})
        sel_sum = self.sum_selected.setdefault(layer_idx, {})
        rec_cnt = self.cnt_recall.setdefault(layer_idx, {})
        ceil_sum = self.sum_ceiling.setdefault(layer_idx, {})
        n_q = H_q
        with torch.no_grad():
            for K in self.k_candidates:
                ceil = _topk_ceiling_mass(page_mass, K)             # [H_q]
                ceil_sum[K] = ceil_sum.get(K, 0.0) + float(ceil.sum().item())
                for sel in SELECTORS_FOR_RECALL:
                    sel_mass = _topk_selected_mass(
                        feats[sel], page_mass, h_kv_of_q, K,
                    )                                                # [H_q]
                    recall = sel_mass + floor
                    key = (K, sel)
                    rec_sum[key] = rec_sum.get(key, 0.0) + float(recall.sum().item())
                    sel_sum[key] = sel_sum.get(key, 0.0) + float(sel_mass.sum().item())
                    rec_cnt[key] = rec_cnt.get(key, 0) + n_q
            self.sum_floor[layer_idx] = self.sum_floor.get(layer_idx, 0.0) + float(floor.sum().item())
            self.cnt_kfloor[layer_idx] = self.cnt_kfloor.get(layer_idx, 0) + n_q

    def reset_step_counters(self) -> None:
        self._step_counter = {}

    def summarize(self) -> dict[str, Any]:
        per_layer: dict[str, dict[str, Any]] = {}
        all_layers_sums = {k: 0.0 for k in FEATURE_NAMES}
        all_layers_cnts = {k: 0 for k in FEATURE_NAMES}

        layer_ids = sorted(self.sum_rho.keys())
        for layer_idx in layer_ids:
            s = self.sum_rho[layer_idx]
            c = self.cnt_rho[layer_idx]
            entry: dict[str, Any] = {
                "bad_steps": self.bad_steps.get(layer_idx, 0),
                "rho_count": dict(c),
            }
            for k in FEATURE_NAMES:
                entry[f"rho_{k}"] = s[k] / max(c[k], 1)
                all_layers_sums[k] += s[k]
                all_layers_cnts[k] += c[k]
            # Mass recall aggregates for this layer.
            rec_cnt = self.cnt_recall.get(layer_idx, {})
            rec_sum = self.sum_recall.get(layer_idx, {})
            sel_sum = self.sum_selected.get(layer_idx, {})
            ceil_sum = self.sum_ceiling.get(layer_idx, {})
            floor_n = max(self.cnt_kfloor.get(layer_idx, 0), 1)
            floor_mean = self.sum_floor.get(layer_idx, 0.0) / floor_n
            entry["floor"] = floor_mean
            entry["recall"] = {}
            entry["selected"] = {}
            entry["ceiling"] = {}
            for K in self.k_candidates:
                entry["ceiling"][str(K)] = ceil_sum.get(K, 0.0) / floor_n + floor_mean
                for sel in SELECTORS_FOR_RECALL:
                    key = (K, sel)
                    n = max(rec_cnt.get(key, 0), 1)
                    entry["recall"].setdefault(str(K), {})[sel] = rec_sum.get(key, 0.0) / n
                    entry["selected"].setdefault(str(K), {})[sel] = sel_sum.get(key, 0.0) / n
            per_layer[str(layer_idx)] = entry

        overall = {
            f"rho_{k}_mean": all_layers_sums[k] / max(all_layers_cnts[k], 1)
            for k in FEATURE_NAMES
        }

        # Late-third aggregation (the Stage-0 gate horizon).
        if layer_ids:
            late_start = layer_ids[-(len(layer_ids) // 3)] if len(layer_ids) >= 3 else layer_ids[0]
            late_layers = [li for li in layer_ids if li >= late_start]
        else:
            late_layers = []
        late = {k: (0.0, 0) for k in FEATURE_NAMES}
        for layer_idx in late_layers:
            s = self.sum_rho[layer_idx]
            c = self.cnt_rho[layer_idx]
            for k in FEATURE_NAMES:
                ss, cc = late[k]
                late[k] = (ss + s[k], cc + c[k])
        late_summary = {
            f"rho_{k}_late_third": (v[0] / max(v[1], 1)) for k, v in late.items()
        }
        late_summary["late_layer_ids"] = late_layers

        # Per-K mass recall aggregation (overall and late-third).
        def _agg_recall(layer_set: list[int]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            floor_num = sum(self.sum_floor.get(li, 0.0) for li in layer_set)
            floor_den = max(sum(self.cnt_kfloor.get(li, 0) for li in layer_set), 1)
            out["floor"] = floor_num / floor_den
            for K in self.k_candidates:
                ceil_num = sum(self.sum_ceiling.get(li, {}).get(K, 0.0) for li in layer_set)
                out_k = {"ceiling": ceil_num / floor_den + out["floor"]}
                for sel in SELECTORS_FOR_RECALL:
                    key = (K, sel)
                    num = sum(self.sum_recall.get(li, {}).get(key, 0.0) for li in layer_set)
                    den = max(sum(self.cnt_recall.get(li, {}).get(key, 0) for li in layer_set), 1)
                    out_k[sel] = num / den
                    s_num = sum(self.sum_selected.get(li, {}).get(key, 0.0) for li in layer_set)
                    out_k[f"selected_{sel}"] = s_num / den
                out[str(K)] = out_k
            return out

        mass_recall_overall = _agg_recall(layer_ids)
        mass_recall_late = _agg_recall(late_layers)

        return {
            "per_layer": per_layer,
            "overall": overall,
            "late_third": late_summary,
            "mass_recall_overall": mass_recall_overall,
            "mass_recall_late_third": mass_recall_late,
            "k_candidates": self.k_candidates,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage-0 HARP premise validation: Spearman ρ between Haar "
            "detail-coefficient features and true per-page attention mass."
        )
    )
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--tasks", type=str, nargs="+",
                   default=["niah_multikey_3", "niah_multivalue", "qa_2"])
    p.add_argument("--seq_len", type=int, default=32768)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--data_root", type=Path,
                   default=Path("benchmark/data/ruler_data"))

    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--num_sink_pages", type=int, default=1)
    p.add_argument("--num_recent_pages", type=int, default=4)
    p.add_argument("--num_decode_steps", type=int, default=8)
    p.add_argument("--skip_layers", type=int, default=0)

    p.add_argument("--haar_levels", type=int, default=3)
    p.add_argument(
        "--haar_top_k", type=int, default=-1,
        help="Top-K H coefficients per page. -1 = page_size // 8 (HARP plan).",
    )
    p.add_argument("--haar_beta", type=float, default=1.0 / (2 ** 0.5),
                   help="L_3 ± β·H weight. Default 1/√2 (pure inverse-Haar).")
    p.add_argument(
        "--dct_comp_size", type=int, default=-1,
        help="DCT lowpass cutoff for baseline. -1 = page_size // 8.",
    )

    p.add_argument("--output_dir", type=Path,
                   default=Path("results_harp_premise"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--k_candidates", type=int, nargs="+", default=DEFAULT_K_CANDIDATES,
        help="Top-K page budgets for mass-recall evaluation (default: 8 16 32 64 128).",
    )
    p.add_argument("--self_test", action="store_true",
                   help="Run numerical sanity check (no model) and exit.")
    return p.parse_args()


def _self_test() -> None:
    """No-model sanity: known correlation between mass and feature."""
    from observations.harp_kernels import _self_test as ker_test
    ker_test()
    # Inject a synthetic relationship: bigger Haar H → bigger mass.
    torch.manual_seed(0)
    B, H_kv, P, S, d = 1, 4, 16, 32, 64
    base = torch.randn(B, H_kv, P, S, d)
    # Amplify a different "outlier token" inside each page by a varying factor.
    scale = torch.linspace(0.1, 4.0, P).view(1, 1, P, 1, 1)
    base[..., 0:1, :] = base[..., 0:1, :] * scale
    # Hand-built "mass" = page-internal variance — should correlate with s_sum.
    var = (base - base.mean(dim=-2, keepdim=True)).pow(2).sum(dim=(-1, -2))
    feats = haar_page_features(base, levels=3)
    rho = spearman_corr_last(feats["s_sum"], var)
    print("synthetic ρ(s_sum, var) per kv-head:", rho.squeeze(0).tolist())
    assert (rho > 0.5).all(), "synthetic premise should give ρ ≫ 0"
    print("self-test OK")


def _generate_with_hook(
    model, tokenizer, sample: dict[str, Any],
    recorder: HaarMassRecorder,
    num_decode_steps: int,
) -> int:
    device = next(model.parameters()).device
    encoded = tokenizer(sample["input"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    recorder.reset_step_counters()
    set_recording_hook(recorder)
    try:
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_decode_steps,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        set_recording_hook(None)
    return int(input_ids.shape[1])


def _print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 92)
    print("OVERALL  (mean ρ across all layers × heads × decode steps)")
    print("=" * 92)
    for k in FEATURE_NAMES:
        v = summary["overall"][f"rho_{k}_mean"]
        print(f"  {k:>14s}  ρ = {v:+.4f}")

    print()
    print("=" * 92)
    print(f"LATE THIRD  (layers {summary['late_third'].get('late_layer_ids', [])})")
    print("=" * 92)
    for k in FEATURE_NAMES:
        v = summary["late_third"][f"rho_{k}_late_third"]
        print(f"  {k:>14s}  ρ = {v:+.4f}")

    # Gate verdict.
    rho_inject = summary["late_third"]["rho_haar_inject_late_third"]
    rho_dct = summary["late_third"]["rho_dct_lowpass_late_third"]
    print()
    print("=" * 92)
    print("STAGE-0 GATE")
    print("=" * 92)
    print(f"  late-third ρ(haar_inject) = {rho_inject:+.4f}    (threshold ≥ 0.30)")
    print(f"  late-third ρ(dct_lowpass) = {rho_dct:+.4f}")
    print(f"  uplift = {rho_inject - rho_dct:+.4f}    (threshold ≥ +0.05)")
    if rho_inject >= 0.30 and (rho_inject - rho_dct) >= 0.05:
        verdict = "PASS — proceed to Stage 1"
    elif rho_inject >= 0.15:
        verdict = "CONDITIONAL — proceed to Stage 1 with q̄ augmentation mandatory"
    else:
        verdict = "FAIL — premise rejected, write postmortem"
    print(f"  verdict: {verdict}")

    # Mass recall table — apples-to-apples with attention_mass_recall_ruler.py.
    if "mass_recall_late_third" in summary:
        print()
        print("=" * 92)
        print("TOP-K MASS RECALL  (sink + recent + Σ m[p] over selector's top-K pages)")
        print("=" * 92)
        for scope_name, scope_key in (
            ("OVERALL (all layers)", "mass_recall_overall"),
            ("LATE THIRD", "mass_recall_late_third"),
        ):
            sec = summary[scope_key]
            print(f"\n  {scope_name}   (floor = sink + recent = {sec['floor']:.4f})")
            header = (f"    {'K':>4s}  {'haar_inj':>9s}  {'haar_low':>9s}  "
                      f"{'dct_low':>9s}  {'quest':>9s}  {'oracle':>9s}  {'ceiling':>9s}")
            print(header)
            for K in summary["k_candidates"]:
                e = sec[str(K)]
                print(
                    f"    {K:>4d}  "
                    f"{e['haar_inject']:>9.4f}  "
                    f"{e['haar_lowpass']:>9.4f}  "
                    f"{e['dct_lowpass']:>9.4f}  "
                    f"{e['quest']:>9.4f}  "
                    f"{e['oracle_max']:>9.4f}  "
                    f"{e['ceiling']:>9.4f}"
                )

    print()
    print("=" * 92)
    print("PER LAYER  (ρ_haar_inject, ρ_haar_lowpass, ρ_dct_lowpass, ρ_quest, ρ_oracle_max)")
    print("=" * 92)
    header = f"{'L':>4}  {'h_inj':>7}  {'h_low':>7}  {'dct':>7}  {'quest':>7}  {'oracle':>7}  {'s_sum':>7}  {'bad':>4}"
    print(header)
    for layer_str in sorted(summary["per_layer"].keys(), key=int):
        e = summary["per_layer"][layer_str]
        print(
            f"{layer_str:>4}  "
            f"{e['rho_haar_inject']:>+7.4f}  "
            f"{e['rho_haar_lowpass']:>+7.4f}  "
            f"{e['rho_dct_lowpass']:>+7.4f}  "
            f"{e['rho_quest']:>+7.4f}  "
            f"{e['rho_oracle_max']:>+7.4f}  "
            f"{e['rho_s_sum']:>+7.4f}  "
            f"{e['bad_steps']:>4d}"
        )


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return

    torch.manual_seed(42)
    start = time.time()

    if args.haar_top_k <= 0:
        args.haar_top_k = max(1, args.page_size // 8)
    if args.dct_comp_size <= 0:
        args.dct_comp_size = max(1, args.page_size // 8)

    run_name = args.run_name or (
        f"haarprem_{Path(args.base_model).name}"
        f"_ps{args.page_size}_lv{args.haar_levels}"
        f"_tk{args.haar_top_k}_b{args.haar_beta:.3f}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Loading model: {args.base_model}")
    model = load_model(args)
    family = _model_family(args.base_model)
    print(f"Installing dense recording forward (family={family})...")
    _install_recording_forward(model, family)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    recorder = HaarMassRecorder(
        page_size=args.page_size,
        num_sink_pages=args.num_sink_pages,
        num_recent_pages=args.num_recent_pages,
        levels=args.haar_levels,
        haar_top_k=args.haar_top_k,
        haar_beta=args.haar_beta,
        dct_comp_size=args.dct_comp_size,
        num_decode_steps=args.num_decode_steps,
        k_candidates=args.k_candidates,
        num_skip_layers=args.skip_layers,
    )

    _, tokenizer_family = infer_model_family(args.base_model)
    task_configs = load_task_configs()

    try:
        for task in args.tasks:
            if task not in task_configs:
                print(f"  WARNING: task {task!r} not in RULER configs, skipping")
                continue
            data_path = (
                args.data_root / tokenizer_family / str(args.seq_len)
                / task / "validation.jsonl"
            )
            if not data_path.exists():
                print(f"  WARNING: data not found at {data_path}, skipping")
                continue
            with data_path.open("r", encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]
            if args.num_samples > 0:
                samples = samples[: args.num_samples]
            for sample in tqdm(samples, desc=f"  {task}"):
                _generate_with_hook(
                    model, tokenizer, sample, recorder,
                    num_decode_steps=args.num_decode_steps,
                )
    finally:
        summary = recorder.summarize()
        summary["config"] = vars(args)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved: {run_dir / 'summary.json'}")
        _print_summary(summary)
        cleanup_model(model)

    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
