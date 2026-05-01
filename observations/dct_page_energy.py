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
      observations/results/dct_page_energy/qwen3_8b_32k_ps32,observations/results/dct_page_energy/llama31_8b_32k_ps32
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
    return f"{short}_{args.context_len}_ps{args.page_size}_{args.task}"


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


def per_bin_k_energy(paged_k: torch.Tensor) -> np.ndarray:
    """paged_k: [B, H, P, page_size, D] → normalized energy vector of length page_size."""
    x = paged_k.to(torch.float32)
    x = x.permute(0, 1, 2, 4, 3).contiguous()  # page_size to last dim
    X = dct(x, norm="ortho")
    energy = X.pow(2).mean(dim=(0, 1, 2, 3))  # (page_size,)
    arr = energy.detach().cpu().numpy()
    total = float(arr.sum())
    if total <= 0:
        return arr
    return arr / total


def compute_layer_energies(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    cfg: DCTPageConfig,
) -> list[dict]:
    rows = []
    for layer_idx, (k, v) in enumerate(zip(k_caches, v_caches)):
        if k is None:
            continue
        _, _, paged_k, _, _, _, num_pages, _ = segment_kv(k, v, cfg)
        if num_pages == 0:
            continue
        frac = per_bin_k_energy(paged_k)
        cum = np.cumsum(frac)
        rows.append(
            {
                "layer_idx": layer_idx,
                "num_pages": int(num_pages),
                "k_energy_fraction": frac.tolist(),
                "k_cumulative": cum.tolist(),
            }
        )
    return rows


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


def write_outputs(run_dir: Path, config_dict: dict, per_layer: list[dict], summary: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as fp:
        json.dump(config_dict, fp, indent=2)
    with (run_dir / "per_layer.jsonl").open("w") as fp:
        for row in per_layer:
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
    ax1.plot(bin_x, summary["k_energy_fraction"], color="C0", linewidth=2, label="mean over layers")
    ax1.set_yscale("log")
    ax1.set_xlabel("DCT bin (low → high freq)")
    ax1.set_ylabel("energy fraction")
    ax1.set_title(f"{title}\nper-bin K energy (page_size={page_size})")
    ax1.legend()

    for r in per_layer:
        ax2.plot(kept_x, r["k_cumulative"], color="lightgray", linewidth=0.5)
    ax2.plot(kept_x, summary["k_cumulative"], color="C0", linewidth=2, label="mean over layers")
    for c in HEADLINE_CUTOFFS:
        if c <= page_size:
            ax2.axvline(c, color="red", alpha=0.25, linestyle="--")
    ax2.axhline(0.9, color="black", alpha=0.3, linestyle=":")
    ax2.axhline(0.99, color="black", alpha=0.3, linestyle=":")
    ax2.set_xlabel("number of low-frequency bins kept")
    ax2.set_ylabel("cumulative energy retained")
    ax2.set_title("cumulative K energy vs lowpass cutoff")
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
        ax_frac.set_title(f"L{r['layer_idx']} fraction", fontsize=9)
        if row == nrows - 1:
            ax_frac.set_xlabel("DCT bin", fontsize=8)
        if col_pair == 0:
            ax_frac.set_ylabel("energy frac", fontsize=8)
        ax_frac.tick_params(labelsize=7)

        ax_cum.plot(kept_x, r["k_cumulative"], color="C0", linewidth=1.2)
        for c in HEADLINE_CUTOFFS:
            if c <= page_size:
                ax_cum.axvline(c, color="red", alpha=0.25, linestyle="--")
        ax_cum.axhline(0.9, color="black", alpha=0.3, linestyle=":")
        ax_cum.axhline(0.99, color="black", alpha=0.3, linestyle=":")
        ax_cum.set_ylim(0.0, 1.02)
        ax_cum.set_title(f"L{r['layer_idx']} cumulative", fontsize=9)
        if row == nrows - 1:
            ax_cum.set_xlabel("bins kept", fontsize=8)
        ax_cum.tick_params(labelsize=7)

    total_cells = nrows * layer_cols
    for idx in range(num_layers, total_cells):
        row = idx // layer_cols
        col_pair = idx % layer_cols
        axes[row][2 * col_pair].set_visible(False)
        axes[row][2 * col_pair + 1].set_visible(False)

    fig.suptitle(f"{title}\nper-layer K energy (page_size={page_size})", fontsize=11)
    plt.tight_layout()
    fig.subplots_adjust(top=1.0 - 0.6 / max(nrows, 1))
    out = run_dir / "energy_curve_per_layer.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[per-layer plot] {out}")


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
        label = cfg.get("model_family", rd.name)
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
    ax.set_xlabel("number of low-frequency bins kept")
    ax.set_ylabel("cumulative K energy retained")
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
        sink_size=args.sink_size,
        recent_size=args.recent_size,
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

    samples = load_samples(data_path, args.num_samples)
    print(f"[setup] model={args.model_name_or_path} family={model_family} "
          f"ctx={args.context_len} task={args.task} samples={len(samples)}")
    print(f"[setup] data={data_path}")
    print(f"[setup] run_dir={run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = load_model(args.model_name_or_path, args.cuda_device)
    device = next(model.parameters()).device
    cfg = build_config(args)

    layer_accum: list[list[np.ndarray]] = []
    layer_num_pages: list[int] = []

    for s_idx, sample in enumerate(samples, start=1):
        encoded = tokenizer(sample["input"], return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        print(f"[forward {s_idx}/{len(samples)}] input_len={input_ids.shape[1]}")
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        k_caches, v_caches = extract_layer_kv(out.past_key_values)
        per_layer = compute_layer_energies(k_caches, v_caches, cfg)

        if not layer_accum:
            layer_accum = [[] for _ in per_layer]
            layer_num_pages = [r["num_pages"] for r in per_layer]
        for i, r in enumerate(per_layer):
            layer_accum[i].append(np.array(r["k_energy_fraction"], dtype=np.float64))

        del out
        torch.cuda.empty_cache()

    per_layer_rows = []
    for li, runs in enumerate(layer_accum):
        mean_frac = np.mean(np.stack(runs, axis=0), axis=0)
        mean_frac = mean_frac / mean_frac.sum()  # re-normalize after averaging
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

    config_dict = {
        "model_name_or_path": args.model_name_or_path,
        "model_family": model_family,
        "context_len": args.context_len,
        "task": args.task,
        "num_samples": args.num_samples,
        "page_size": args.page_size,
        "sink_size": args.sink_size,
        "recent_size": args.recent_size,
        "data_path": str(data_path),
        "run_name": run_name,
    }

    write_outputs(run_dir, config_dict, per_layer_rows, summary)

    if args.plot:
        title = f"{model_family} @ {args.context_len} ({args.task})"
        render_plot(run_dir, per_layer_rows, summary, args.page_size, title)
        render_per_layer_grid(run_dir, per_layer_rows, args.page_size, title)

    print_headline_table(run_name, summary, args.page_size)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--task", default="niah_single_1")
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=128)
    p.add_argument("--data_root", type=Path, default=_REPO_ROOT / "benchmark" / "data" / "ruler_data")
    p.add_argument("--output_dir", type=Path, default=_REPO_ROOT / "observations" / "results" / "dct_page_energy")
    p.add_argument("--run_name", default=None)
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
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
    title = f"{cfg.get('model_family', run_dir.name)} @ {cfg.get('context_len', '?')} ({cfg.get('task', '?')})"
    render_plot(run_dir, per_layer, summary, cfg["page_size"], title)
    render_per_layer_grid(run_dir, per_layer, cfg["page_size"], title)
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
