"""Measure DCT energy distribution across KV-cache pages.

Runs a baseline (unmodified) prefill on a RULER sample, reshapes each layer's
cached K into pages of `page_size`, takes a full DCT-II along the page axis,
and reports:

  * per-bin energy fraction          (sums to 1 per layer)
  * cumulative energy kept by a lowpass cutoff at k ∈ {1..page_size} bins

The point is to empirically show that K pages are spectrally concentrated — a
lowpass cutoff keeps most of the per-page energy, which is exactly what the
DCT-lowpass-IDCT score proxy and compressed-mode representative rely on.

Usage
-----

  # Single run on Qwen3-8B
  python observations/dct_page_energy.py --model_name_or_path Qwen/Qwen3-8B \
      --context_len 32768 --task niah_single_1 --page_size 32 \
      --run_name qwen3_8b_32k_ps32

  # Compare two prior runs in one plot
  python observations/dct_page_energy.py --compare_runs \
      result/energy/qwen3_8b_32k_ps32,result/energy/llama31_8b_32k_ps32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DCTPageConfig  # noqa: E402
from dct_page_attention import dct, segment_kv  # noqa: E402


HEADLINE_CUTOFFS = [1, 2, 4, 8, 16]


def resolve_model_family(model_name_or_path: str) -> str:
    name = model_name_or_path.lower().split("/")[-1]
    if "qwen3" in name:
        return "qwen3"
    if "qwen2" in name:
        return "qwen2"
    if "llama" in name:
        return "llama"
    return name.split("-")[0]


def default_run_name(args: argparse.Namespace) -> str:
    short = args.model_name_or_path.split("/")[-1].lower().replace(".", "")
    base = f"{short}_{args.context_len}_ps{args.page_size}_{args.task}"
    if getattr(args, "granularity", "layer") == "head":
        base += "_ghead"
    return base


_MODEL_FAMILY_PRETTY = {"qwen3": "Qwen3", "qwen2": "Qwen2", "llama": "Llama"}
_TASK_PRETTY = {
    "niah_single_1": "NIAH Single 1",
    "niah_single_2": "NIAH Single 2",
    "niah_single_3": "NIAH Single 3",
    "niah_multikey_1": "NIAH MultiKey 1",
    "niah_multikey_2": "NIAH MultiKey 2",
    "niah_multikey_3": "NIAH MultiKey 3",
    "niah_multivalue": "NIAH MultiValue",
    "niah_multiquery": "NIAH MultiQuery",
    "vt": "VT",
    "cwe": "CWE",
    "fwe": "FWE",
    "qa_1": "QA 1",
    "qa_2": "QA 2",
}


def _pretty_model_family(name: str) -> str:
    return _MODEL_FAMILY_PRETTY.get(name.lower(), name.capitalize())


def _pretty_task(name: str) -> str:
    return _TASK_PRETTY.get(name, name.replace("_", " ").title())


def _format_plot_title(model_family: str, context_len, task: str) -> str:
    return f"{_pretty_model_family(model_family)} @ {context_len} tokens — {_pretty_task(task)}"


def load_samples(path: Path, num_samples: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as fp:
        rows = [json.loads(line) for line in fp if line.strip()]
    return rows[:num_samples]


def load_model(model_name_or_path: str, cuda_device: int):
    yarn_kwargs = {}
    if "qwen3" in model_name_or_path.lower():
        yarn_kwargs = {
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            "max_position_embeddings": 131072,
        }
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": cuda_device},
        attn_implementation="sdpa",
        **yarn_kwargs,
    ).eval()


def extract_layer_kv(past_key_values):
    """Return (list_of_K, list_of_V) across layers, handling Cache API variants."""
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(past_key_values.key_cache), list(past_key_values.value_cache)
    if hasattr(past_key_values, "layers"):
        ks, vs = [], []
        for layer in past_key_values.layers:
            k = getattr(layer, "keys", None)
            if k is None:
                k = getattr(layer, "key_cache", None)
            v = getattr(layer, "values", None)
            if v is None:
                v = getattr(layer, "value_cache", None)
            ks.append(k)
            vs.append(v)
        return ks, vs
    ks, vs = [], []
    for i in range(len(past_key_values)):
        k, v = past_key_values[i]
        ks.append(k)
        vs.append(v)
    return ks, vs


def per_bin_k_energy(paged_k: torch.Tensor, *, per_head: bool = False) -> np.ndarray:
    """paged_k: [B, H, P, page_size, D] → normalized energy.

    per_head=False (default): reduce over (B, H, P, D) → shape [page_size].
    per_head=True: reduce over (B, P, D) only → shape [H, page_size]; rows are
    individually normalized; rows whose sum < 1e-12 are filled with NaN.
    """
    x = paged_k.to(torch.float32)
    x = x.permute(0, 1, 2, 4, 3).contiguous()  # page_size to last dim
    X = dct(x, norm="ortho")
    if not per_head:
        energy = X.pow(2).mean(dim=(0, 1, 2, 3))  # (page_size,)
        arr = energy.detach().cpu().numpy()
        total = float(arr.sum())
        if total <= 0:
            return arr
        return arr / total
    # per_head=True: keep H axis (=axis 1).
    energy = X.pow(2).mean(dim=(0, 2, 3))  # (H, page_size)
    arr = energy.detach().cpu().numpy().astype(np.float64)
    row_sums = arr.sum(axis=1)
    out = np.empty_like(arr)
    bad = row_sums < 1e-12
    if bad.any():
        out[bad, :] = np.nan
    good = ~bad
    if good.any():
        out[good, :] = arr[good, :] / row_sums[good, None]
    return out


def compute_layer_energies(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    cfg: DCTPageConfig,
    *,
    granularity: str = "layer",
) -> tuple[list[dict], list[dict] | None]:
    """Run one sample through segment_kv + per_bin_k_energy across all layers.

    granularity="layer": returns (per_layer_rows, None). Identical to legacy.
    granularity="head":  returns (per_layer_rows, per_head_rows) where the
    per_layer_rows path uses the same fully-reduced call as legacy (bitwise
    parity required) and per_head_rows holds one row per (layer_idx, kv_head).
    """
    per_layer_rows: list[dict] = []
    per_head_rows: list[dict] | None = [] if granularity == "head" else None
    for layer_idx, (k, v) in enumerate(zip(k_caches, v_caches)):
        if k is None:
            continue
        _, _, paged_k, _, _, _, num_pages, _ = segment_kv(k, v, cfg)
        if num_pages == 0:
            continue
        # BITWISE-LOAD-BEARING: do not refactor
        frac = per_bin_k_energy(paged_k, per_head=False)  # site (a)
        cum = np.cumsum(frac)
        per_layer_rows.append(
            {
                "layer_idx": layer_idx,
                "num_pages": int(num_pages),
                "k_energy_fraction": frac.tolist(),
                "k_cumulative": cum.tolist(),
            }
        )
        if per_head_rows is not None:
            head_frac = per_bin_k_energy(paged_k, per_head=True)  # [H, page_size]
            head_cum = np.cumsum(head_frac, axis=1)
            for h_idx in range(head_frac.shape[0]):
                per_head_rows.append(
                    {
                        "layer_idx": layer_idx,
                        "kv_head_idx": int(h_idx),
                        "num_pages": int(num_pages),
                        "k_energy_fraction": head_frac[h_idx].tolist(),
                        "k_cumulative": head_cum[h_idx].tolist(),
                    }
                )
    return per_layer_rows, per_head_rows


def aggregate_layers(per_layer: list[dict], page_size: int) -> dict:
    fracs = np.array([r["k_energy_fraction"] for r in per_layer], dtype=np.float64)
    mean_frac = fracs.mean(axis=0)
    mean_cum = np.cumsum(mean_frac)
    headline = {
        str(c): float(mean_cum[c - 1])
        for c in HEADLINE_CUTOFFS
        if c <= page_size
    }
    return {
        "k_energy_fraction": mean_frac.tolist(),
        "k_cumulative": mean_cum.tolist(),
        "headline": headline,
    }


def aggregate_layers_per_head(
    layer_head_accum: list,
    num_hidden_layers: int,
    num_kv_heads: int,
    page_size: int,
) -> dict:
    """Build the [L, H, page_size] head-resolved summary block.

    layer_head_accum: list of length num_hidden_layers; each entry is either
    None / [] (skipped layer) or a list of np.ndarrays of shape [H, page_size]
    (one per sample). Skipped layers become NaN rows so the heatmap renders
    them as full-gray strips.
    """
    fracs = np.full((num_hidden_layers, num_kv_heads, page_size), np.nan, dtype=np.float64)
    cums = np.full((num_hidden_layers, num_kv_heads, page_size), np.nan, dtype=np.float64)
    for li in range(num_hidden_layers):
        runs = layer_head_accum[li] if li < len(layer_head_accum) else None
        if not runs:
            continue
        stacked = np.stack(runs, axis=0)  # [num_samples, H, page_size]
        with np.errstate(all="ignore"):
            mean_frac = np.nanmean(stacked, axis=0)  # [H, page_size]
            row_sums = np.nansum(mean_frac, axis=1)  # [H]
            good = row_sums > 1e-12
            renorm = np.full_like(mean_frac, np.nan)
            if good.any():
                renorm[good, :] = mean_frac[good, :] / row_sums[good, None]
            cum = np.full_like(renorm, np.nan)
            if good.any():
                cum[good, :] = np.cumsum(renorm[good, :], axis=1)
        fracs[li] = renorm
        cums[li] = cum

    headline: dict[str, list] = {}
    head_cv: dict[str, float] = {}
    with np.errstate(all="ignore"):
        for c in HEADLINE_CUTOFFS:
            if c > page_size:
                continue
            cut = cums[:, :, c - 1]  # [L, H]
            headline[str(c)] = cut.tolist()
            std_h = np.nanstd(cut, axis=1, ddof=0)  # per-layer
            mean_h = np.nanmean(cut, axis=1)
            ratio = np.where(mean_h > 0, std_h / np.where(mean_h == 0, np.nan, mean_h), np.nan)
            cv = float(np.nanmean(ratio))
            head_cv[str(c)] = cv

    return {
        "k_energy_fraction": fracs.tolist(),
        "k_cumulative": cums.tolist(),
        "headline": headline,
        "head_cv": head_cv,
    }


def write_outputs(
    run_dir: Path,
    config_dict: dict,
    per_layer: list[dict],
    summary: dict,
    *,
    per_head_rows: list[dict] | None = None,
    summary_per_head: dict | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as fp:
        json.dump(config_dict, fp, indent=2)
    with (run_dir / "per_layer.jsonl").open("w") as fp:
        for row in per_layer:
            fp.write(json.dumps(row) + "\n")
    # Layer-mode summary.json invariant: legacy keys only. Conditional add.
    if per_head_rows is not None:
        summary["per_head"] = summary_per_head
        with (run_dir / "per_head.jsonl").open("w") as fp:
            for row in per_head_rows:
                fp.write(json.dumps(row) + "\n")
    with (run_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)


def render_plot(run_dir: Path, per_layer: list[dict], summary: dict, page_size: int, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    bin_x = np.arange(page_size)                 # 0-indexed DCT bin for the energy-fraction plot
    kept_x = np.arange(1, page_size + 1)          # 1-indexed "number of bins kept" for the cumulative plot
    for r in per_layer:
        ax1.plot(bin_x, r["k_energy_fraction"], color="lightgray", linewidth=0.5)
    ax1.plot(bin_x, summary["k_energy_fraction"], color="C0", linewidth=2, label="Mean over layers")
    ax1.set_yscale("log")
    ax1.set_xlabel("DCT bin (low → high frequency)")
    ax1.set_ylabel("Energy fraction")
    ax1.set_title(f"{title}\nPer-bin K energy (page size = {page_size})")
    ax1.legend()

    for r in per_layer:
        ax2.plot(kept_x, r["k_cumulative"], color="lightgray", linewidth=0.5)
    ax2.plot(kept_x, summary["k_cumulative"], color="C0", linewidth=2, label="Mean over layers")
    for c in HEADLINE_CUTOFFS:
        if c <= page_size:
            ax2.axvline(c, color="red", alpha=0.25, linestyle="--")
    ax2.axhline(0.9, color="black", alpha=0.3, linestyle=":")
    ax2.axhline(0.99, color="black", alpha=0.3, linestyle=":")
    ax2.set_xlabel("Number of low-frequency bins kept")
    ax2.set_ylabel("Cumulative energy retained")
    ax2.set_title("Cumulative K energy vs. lowpass cutoff")
    ax2.set_ylim(0.0, 1.02)
    ax2.legend()

    plt.tight_layout()
    out = run_dir / "energy_curve.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] {out}")


def render_per_layer_grid(
    run_dir: Path,
    per_layer: list[dict],
    page_size: int,
    title: str,
    layer_cols: int = 4,
) -> None:
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not per_layer:
        print("[per-layer plot] no layers to render")
        return

    bin_x = np.arange(page_size)
    kept_x = np.arange(1, page_size + 1)

    num_layers = len(per_layer)
    nrows = math.ceil(num_layers / layer_cols)
    ncols = 2 * layer_cols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.0 * nrows),
        squeeze=False,
    )

    for idx, r in enumerate(per_layer):
        row = idx // layer_cols
        col_pair = idx % layer_cols
        ax_frac = axes[row][2 * col_pair]
        ax_cum = axes[row][2 * col_pair + 1]

        ax_frac.plot(bin_x, r["k_energy_fraction"], color="C0", linewidth=1.2)
        ax_frac.set_yscale("log")
        ax_frac.set_title(f"Layer {r['layer_idx']} — fraction", fontsize=9)
        if row == nrows - 1:
            ax_frac.set_xlabel("DCT bin", fontsize=8)
        if col_pair == 0:
            ax_frac.set_ylabel("Energy fraction", fontsize=8)
        ax_frac.tick_params(labelsize=7)

        ax_cum.plot(kept_x, r["k_cumulative"], color="C0", linewidth=1.2)
        for c in HEADLINE_CUTOFFS:
            if c <= page_size:
                ax_cum.axvline(c, color="red", alpha=0.25, linestyle="--")
        ax_cum.axhline(0.9, color="black", alpha=0.3, linestyle=":")
        ax_cum.axhline(0.99, color="black", alpha=0.3, linestyle=":")
        ax_cum.set_ylim(0.0, 1.02)
        ax_cum.set_title(f"Layer {r['layer_idx']} — cumulative", fontsize=9)
        if row == nrows - 1:
            ax_cum.set_xlabel("Bins kept", fontsize=8)
        ax_cum.tick_params(labelsize=7)

    total_cells = nrows * layer_cols
    for idx in range(num_layers, total_cells):
        row = idx // layer_cols
        col_pair = idx % layer_cols
        axes[row][2 * col_pair].set_visible(False)
        axes[row][2 * col_pair + 1].set_visible(False)

    fig.suptitle(f"{title}\nPer-layer K energy (page size = {page_size})", fontsize=11)
    plt.tight_layout()
    fig.subplots_adjust(top=1.0 - 0.6 / max(nrows, 1))
    out = run_dir / "energy_curve_per_layer.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[per-layer plot] {out}")


def render_per_head_grid(
    run_dir: Path,
    summary_per_head: dict,
    page_size: int,
    title: str,
    head_cols: int = 4,
) -> None:
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = np.array(summary_per_head["k_energy_fraction"], dtype=np.float64)  # [L, H, P]
    cums = np.array(summary_per_head["k_cumulative"], dtype=np.float64)        # [L, H, P]
    L, H, _ = fracs.shape

    bin_x = np.arange(page_size)
    kept_x = np.arange(1, page_size + 1)

    nrows = math.ceil(H / head_cols)
    ncols = 2 * head_cols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.0 * nrows),
        squeeze=False,
    )

    with np.errstate(all="ignore"):
        head_mean_frac = np.nanmean(fracs, axis=0)  # [H, P]
        head_mean_cum = np.nanmean(cums, axis=0)    # [H, P]

    for h in range(H):
        row = h // head_cols
        col_pair = h % head_cols
        ax_frac = axes[row][2 * col_pair]
        ax_cum = axes[row][2 * col_pair + 1]

        for li in range(L):
            ax_frac.plot(bin_x, fracs[li, h, :], color="lightgray", linewidth=0.5)
            ax_cum.plot(kept_x, cums[li, h, :], color="lightgray", linewidth=0.5)

        ax_frac.plot(bin_x, head_mean_frac[h], color="C0", linewidth=1.4)
        ax_frac.set_yscale("log")
        ax_frac.set_title(f"Head {h} — fraction", fontsize=9)
        if row == nrows - 1:
            ax_frac.set_xlabel("DCT bin", fontsize=8)
        if col_pair == 0:
            ax_frac.set_ylabel("Energy fraction", fontsize=8)
        ax_frac.tick_params(labelsize=7)

        ax_cum.plot(kept_x, head_mean_cum[h], color="C0", linewidth=1.4)
        for c in HEADLINE_CUTOFFS:
            if c <= page_size:
                ax_cum.axvline(c, color="red", alpha=0.25, linestyle="--")
        ax_cum.axhline(0.9, color="black", alpha=0.3, linestyle=":")
        ax_cum.axhline(0.99, color="black", alpha=0.3, linestyle=":")
        ax_cum.set_ylim(0.0, 1.02)
        ax_cum.set_title(f"Head {h} — cumulative", fontsize=9)
        if row == nrows - 1:
            ax_cum.set_xlabel("Bins kept", fontsize=8)
        ax_cum.tick_params(labelsize=7)

    total_cells = nrows * head_cols
    for idx in range(H, total_cells):
        row = idx // head_cols
        col_pair = idx % head_cols
        axes[row][2 * col_pair].set_visible(False)
        axes[row][2 * col_pair + 1].set_visible(False)

    fig.suptitle(f"{title}\nPer-head K energy (page size = {page_size})", fontsize=11)
    plt.tight_layout()
    fig.subplots_adjust(top=1.0 - 0.6 / max(nrows, 1))
    out = run_dir / "energy_curve_per_head.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[per-head plot] {out}")


def render_per_head_heatmap(
    run_dir: Path,
    summary_per_head: dict,
    page_size: int,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fracs = np.array(summary_per_head["k_energy_fraction"], dtype=np.float64)  # [L, H, P]
    cums = np.array(summary_per_head["k_cumulative"], dtype=np.float64)        # [L, H, P]
    L, H, _ = fracs.shape

    nrows = H
    ncols = 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols + 2.0, max(1.0, 0.35 * L) * nrows),
        squeeze=False,
    )
    cmap_frac = plt.get_cmap("viridis").copy()
    cmap_frac.set_bad("lightgray")
    cmap_cum = plt.get_cmap("viridis").copy()
    cmap_cum.set_bad("lightgray")

    norm_frac = LogNorm(vmin=1e-6, vmax=1.0)

    last_left = None
    last_right = None
    for h in range(H):
        ax_l = axes[h][0]
        ax_r = axes[h][1]

        last_left = ax_l.imshow(
            fracs[:, h, :],
            aspect="auto",
            origin="upper",
            cmap=cmap_frac,
            norm=norm_frac,
            interpolation="nearest",
        )
        ax_l.set_title(f"Head {h} — fraction", fontsize=8)
        ax_l.set_ylabel("Layer", fontsize=8)
        if h == H - 1:
            ax_l.set_xlabel("DCT bin", fontsize=8)
        ax_l.tick_params(labelsize=7)

        last_right = ax_r.imshow(
            cums[:, h, :],
            aspect="auto",
            origin="upper",
            cmap=cmap_cum,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        for c in HEADLINE_CUTOFFS:
            if c <= page_size:
                ax_r.axvline(c - 0.5, color="red", alpha=0.5, linestyle="--", linewidth=0.8)
        ax_r.set_title(f"Head {h} — cumulative", fontsize=8)
        if h == H - 1:
            ax_r.set_xlabel("Bins kept", fontsize=8)
        ax_r.tick_params(labelsize=7)

    if last_left is not None:
        fig.colorbar(last_left, ax=axes[:, 0].tolist(), shrink=0.6, label="Energy fraction")
    if last_right is not None:
        fig.colorbar(last_right, ax=axes[:, 1].tolist(), shrink=0.6, label="Cumulative energy")

    fig.suptitle(f"{title}\nPer-head K energy heatmap (page size = {page_size})", fontsize=11)
    out = run_dir / "energy_curve_per_head_heatmap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[per-head heatmap] {out}")


def render_per_head_heatmap_norm(
    run_dir: Path,
    summary_per_head: dict,
    page_size: int,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = np.array(summary_per_head["k_energy_fraction"], dtype=np.float64)  # [L, H, P]
    L, H, P = fracs.shape
    if P < 2:
        print("[per-head heatmap norm] page_size < 2; nothing to render after dropping DC")
        return

    ac = fracs[:, :, 1:]  # [L, H, P-1]
    with np.errstate(all="ignore"):
        denom = np.nansum(ac, axis=2, keepdims=True)
        denom = np.where(denom > 1e-12, denom, np.nan)
        ac_norm = ac / denom  # [L, H, P-1]

    ncols = H
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(2.4 * ncols + 1.5, max(2.0, 0.35 * L)),
        squeeze=False,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    ac_vmax = float(np.nanmax(ac_norm)) if np.isfinite(np.nanmax(ac_norm)) else 1.0
    if ac_vmax <= 0:
        ac_vmax = 1.0

    last_im = None
    for h in range(H):
        ax = axes[0][h]
        last_im = ax.imshow(
            ac_norm[:, h, :],
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=0.0,
            vmax=ac_vmax,
            interpolation="nearest",
        )
        ax.set_title(f"Head {h}", fontsize=9)
        if h == 0:
            ax.set_ylabel("Layer", fontsize=8)
        ax.set_xlabel(f"DCT bin (1..{page_size - 1})", fontsize=8)
        ax.tick_params(labelsize=7)

    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes[0, :].tolist(),
            shrink=0.7,
            label=f"Energy fraction (DC dropped, renormalized; vmax = {ac_vmax:.3f})",
        )

    fig.suptitle(
        f"{title}\nPer-head AC-only K energy heatmap (page size = {page_size}, DC dropped)",
        fontsize=11,
    )
    out = run_dir / "energy_curve_head_heatmap_norm.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[per-head heatmap norm] {out}")


def render_layer_head_heatmap(
    run_dir: Path,
    summary_per_head: dict,
    page_size: int,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headline = summary_per_head.get("headline", {})
    head_cv = summary_per_head.get("head_cv", {})
    cutoffs = [c for c in HEADLINE_CUTOFFS if c <= page_size and str(c) in headline]
    if not cutoffs:
        print("[layer-head heatmap] no cutoffs to render")
        return

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    ncols = len(cutoffs)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(3.4 * ncols, 5.0),
        squeeze=False,
    )
    last_im = None
    for ci, c in enumerate(cutoffs):
        ax = axes[0][ci]
        mat = np.array(headline[str(c)], dtype=np.float64)  # [L, H]
        last_im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(f"Cumulative @ {c} bins", fontsize=10)
        ax.set_xlabel("KV head", fontsize=9)
        if ci == 0:
            ax.set_ylabel("Layer", fontsize=9)
        ax.tick_params(labelsize=7)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[0].tolist(), shrink=0.85, label="Cumulative energy")

    cv_str = ", ".join(f"{c}: {head_cv.get(str(c), float('nan')):.3f}" for c in cutoffs)
    fig.suptitle(
        f"{title}\nPer-(layer, head) cumulative energy at cutoffs (head CV — {cv_str})",
        fontsize=11,
    )
    out = run_dir / "energy_heatmap_layer_head.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[layer-head heatmap] {out}")


def render_compare(run_dirs: list[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    page_size = None
    for rd in run_dirs:
        with (rd / "summary.json").open() as fp:
            summary = json.load(fp)
        with (rd / "config.json").open() as fp:
            cfg = json.load(fp)
        label = _pretty_model_family(cfg.get("model_family", rd.name))
        cum = summary["k_cumulative"]
        if page_size is None:
            page_size = len(cum)
        ax.plot(np.arange(1, len(cum) + 1), cum, linewidth=2, label=label)

    for c in HEADLINE_CUTOFFS:
        if page_size and c <= page_size:
            ax.axvline(c, color="red", alpha=0.2, linestyle="--")
    ax.axhline(0.9, color="black", alpha=0.3, linestyle=":")
    ax.axhline(0.99, color="black", alpha=0.3, linestyle=":")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Number of low-frequency bins kept")
    ax.set_ylabel("Cumulative K energy retained")
    ax.set_title("K energy concentration across models")
    ax.legend()
    plt.tight_layout()

    out = run_dirs[0] / "energy_curve_compare.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[compare plot] {out}")


def print_headline_table(run_name: str, summary: dict, page_size: int) -> None:
    headline = summary["headline"]
    print()
    print(f"=== {run_name} ===")
    print(f"{'cutoff / page_size':>22}  {'k_cumulative':>14}")
    for c in HEADLINE_CUTOFFS:
        key = str(c)
        if key not in headline:
            continue
        print(f"{c:>4} / {page_size:<12}  {headline[key]:>14.4f}")


def build_config(args: argparse.Namespace) -> DCTPageConfig:
    return DCTPageConfig(
        page_size=args.page_size,
        num_sink_pages=args.num_sink_pages,
        num_recent_pages=args.num_recent_pages,
    )


def run_measurement(args: argparse.Namespace) -> None:
    model_family = resolve_model_family(args.model_name_or_path)
    data_path = args.data_root / model_family / str(args.context_len) / args.task / "validation.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"RULER data missing at {data_path}. "
            f"Prepare it via benchmark/eval_ruler/data/prepare.py."
        )

    run_name = args.run_name or default_run_name(args)
    run_dir = args.output_dir / run_name

    # Pre-write clobber guard: refuse to mix granularities in one run_dir.
    existing_per_layer = run_dir / "per_layer.jsonl"
    existing_cfg_path = run_dir / "config.json"
    if existing_per_layer.exists() and existing_cfg_path.exists():
        try:
            with existing_cfg_path.open() as fp:
                existing_cfg = json.load(fp)
        except Exception:
            existing_cfg = {}
        existing_g = existing_cfg.get("granularity", "layer")
        if existing_g != args.granularity and not args.overwrite:
            raise RuntimeError(
                f"refusing to clobber {run_dir}: existing run was --granularity={existing_g} "
                f"but --granularity={args.granularity} was requested. "
                f"Pass --overwrite or use --run_name <new>."
            )

    samples = load_samples(data_path, args.num_samples)
    print(f"[setup] model={args.model_name_or_path} family={model_family} "
          f"ctx={args.context_len} task={args.task} samples={len(samples)} "
          f"granularity={args.granularity}")
    print(f"[setup] data={data_path}")
    print(f"[setup] run_dir={run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = load_model(args.model_name_or_path, args.cuda_device)
    device = next(model.parameters()).device
    cfg = build_config(args)

    num_hidden_layers = int(model.config.num_hidden_layers)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))

    layer_accum: list[list[np.ndarray]] = []
    layer_num_pages: list[int] = []
    # Head-mode parallel buffer; sized to num_hidden_layers (NOT post-skip).
    layer_head_accum: list[list[np.ndarray]] = (
        [[] for _ in range(num_hidden_layers)] if args.granularity == "head" else []
    )

    for s_idx, sample in enumerate(samples, start=1):
        encoded = tokenizer(sample["input"], return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        print(f"[forward {s_idx}/{len(samples)}] input_len={input_ids.shape[1]}")
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        k_caches, v_caches = extract_layer_kv(out.past_key_values)
        per_layer, per_head_sample = compute_layer_energies(
            k_caches, v_caches, cfg, granularity=args.granularity
        )

        if not layer_accum:
            layer_accum = [[] for _ in per_layer]
            layer_num_pages = [r["num_pages"] for r in per_layer]
        for i, r in enumerate(per_layer):
            # BITWISE-LOAD-BEARING: do not refactor
            layer_accum[i].append(np.array(r["k_energy_fraction"], dtype=np.float64))  # site (b)

        if args.granularity == "head" and per_head_sample is not None:
            # Bucket per-head rows by layer_idx, stack into [H, page_size] per layer.
            by_layer: dict[int, list[np.ndarray]] = {}
            for r in per_head_sample:
                li = int(r["layer_idx"])
                by_layer.setdefault(li, []).append(
                    np.array(r["k_energy_fraction"], dtype=np.float64)
                )
            for li, rows in by_layer.items():
                stacked = np.stack(rows, axis=0)  # [H, page_size]
                layer_head_accum[li].append(stacked)

        del out
        torch.cuda.empty_cache()

    per_layer_rows = []
    for li, runs in enumerate(layer_accum):
        mean_frac = np.mean(np.stack(runs, axis=0), axis=0)
        # BITWISE-LOAD-BEARING: do not refactor
        mean_frac = mean_frac / mean_frac.sum()  # site (c) — re-normalize after averaging
        cum = np.cumsum(mean_frac)
        per_layer_rows.append(
            {
                "layer_idx": li,
                "num_pages": layer_num_pages[li],
                "k_energy_fraction": mean_frac.tolist(),
                "k_cumulative": cum.tolist(),
            }
        )

    summary = aggregate_layers(per_layer_rows, args.page_size)

    per_head_rows: list[dict] | None = None
    summary_per_head: dict | None = None
    if args.granularity == "head":
        # Zero-valid-layers guard: fire BEFORE aggregate_layers_per_head.
        if not any(layer_head_accum[li] for li in range(num_hidden_layers)):
            raise RuntimeError(
                "zero valid layers in layer_head_accum; check context length / page config"
            )
        per_head_rows = []
        for li in range(num_hidden_layers):
            runs = layer_head_accum[li]
            if not runs:
                continue
            stacked = np.stack(runs, axis=0)  # [num_samples, H, page_size]
            with np.errstate(all="ignore"):
                mean_frac = np.nanmean(stacked, axis=0)  # [H, page_size]
                row_sums = np.nansum(mean_frac, axis=1)
            for h_idx in range(mean_frac.shape[0]):
                rs = float(row_sums[h_idx])
                if rs <= 1e-12 or not np.isfinite(rs):
                    frac_row = np.full(args.page_size, np.nan, dtype=np.float64)
                    cum_row = np.full(args.page_size, np.nan, dtype=np.float64)
                else:
                    frac_row = mean_frac[h_idx] / rs
                    cum_row = np.cumsum(frac_row)
                per_head_rows.append(
                    {
                        "layer_idx": li,
                        "kv_head_idx": int(h_idx),
                        "num_pages": layer_num_pages[li] if li < len(layer_num_pages) else 0,
                        "k_energy_fraction": frac_row.tolist(),
                        "k_cumulative": cum_row.tolist(),
                    }
                )
        summary_per_head = aggregate_layers_per_head(
            layer_head_accum, num_hidden_layers, num_kv_heads, args.page_size
        )

    config_dict = {
        "model_name_or_path": args.model_name_or_path,
        "model_family": model_family,
        "context_len": args.context_len,
        "task": args.task,
        "num_samples": args.num_samples,
        "page_size": args.page_size,
        "num_sink_pages": args.num_sink_pages,
        "num_recent_pages": args.num_recent_pages,
        "data_path": str(data_path),
        "run_name": run_name,
        "granularity": args.granularity,
    }

    write_outputs(
        run_dir,
        config_dict,
        per_layer_rows,
        summary,
        per_head_rows=per_head_rows,
        summary_per_head=summary_per_head,
    )

    if args.plot:
        title = _format_plot_title(model_family, args.context_len, args.task)
        render_plot(run_dir, per_layer_rows, summary, args.page_size, title)
        render_per_layer_grid(run_dir, per_layer_rows, args.page_size, title)
        if args.granularity == "head" and summary_per_head is not None:
            render_per_head_grid(run_dir, summary_per_head, args.page_size, title)
            render_per_head_heatmap(run_dir, summary_per_head, args.page_size, title)
            render_per_head_heatmap_norm(run_dir, summary_per_head, args.page_size, title)
            render_layer_head_heatmap(run_dir, summary_per_head, args.page_size, title)

    print_headline_table(run_name, summary, args.page_size)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--task", default="niah_single_1")
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--num_sink_pages", type=int, default=1)
    p.add_argument("--num_recent_pages", type=int, default=4)
    p.add_argument("--data_root", type=Path, default=_REPO_ROOT / "benchmark" / "data" / "ruler_data")
    p.add_argument("--output_dir", type=Path, default=_REPO_ROOT / "result" / "energy")
    p.add_argument("--run_name", default=None)
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--granularity",
        choices=["layer", "head"],
        default="layer",
        help="Reduction axis for the energy measurement. 'layer' (default) "
             "averages over batch+heads+pages+head_dim; output is unchanged. "
             "'head' keeps the kv-head axis separate and writes per_head.jsonl "
             "plus a per-head block inside summary.json.",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow writing into a run_dir whose existing config.json records a "
             "different --granularity. Default refuses to clobber.",
    )
    p.add_argument(
        "--compare_runs",
        default=None,
        help="Comma-separated run directories to overlay into energy_curve_compare.png. "
             "Skips the forward pass.",
    )
    p.add_argument(
        "--replot_from",
        type=Path,
        default=None,
        help="Replot energy_curve.png from an existing run directory without rerunning the forward.",
    )
    return p.parse_args()


def replot(run_dir: Path) -> None:
    with (run_dir / "config.json").open() as fp:
        cfg = json.load(fp)
    with (run_dir / "summary.json").open() as fp:
        summary = json.load(fp)
    per_layer: list[dict] = []
    with (run_dir / "per_layer.jsonl").open() as fp:
        for line in fp:
            if line.strip():
                per_layer.append(json.loads(line))
    title = _format_plot_title(
        str(cfg.get("model_family", run_dir.name)),
        cfg.get("context_len", "?"),
        str(cfg.get("task", "?")),
    )
    render_plot(run_dir, per_layer, summary, cfg["page_size"], title)
    render_per_layer_grid(run_dir, per_layer, cfg["page_size"], title)

    # Head plots are additive: detect by file presence, not config flag.
    per_head_present = (run_dir / "per_head.jsonl").exists()
    if per_head_present and isinstance(summary.get("per_head"), dict):
        summary_per_head = summary["per_head"]
        render_per_head_grid(run_dir, summary_per_head, cfg["page_size"], title)
        render_per_head_heatmap(run_dir, summary_per_head, cfg["page_size"], title)
        render_per_head_heatmap_norm(run_dir, summary_per_head, cfg["page_size"], title)
        render_layer_head_heatmap(run_dir, summary_per_head, cfg["page_size"], title)

    print_headline_table(cfg.get("run_name", run_dir.name), summary, cfg["page_size"])


def main() -> None:
    args = parse_args()
    if args.replot_from:
        replot(args.replot_from)
        return
    if args.compare_runs:
        run_dirs = [Path(p.strip()) for p in args.compare_runs.split(",") if p.strip()]
        for rd in run_dirs:
            if not (rd / "summary.json").exists():
                raise FileNotFoundError(f"missing summary.json in {rd}")
        render_compare(run_dirs)
        return
    run_measurement(args)


if __name__ == "__main__":
    main()
