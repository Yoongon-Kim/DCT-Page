"""
Unified RULER benchmark evaluation.

Supports many attention modes (baseline, page_attention, seer_attention,
seer_prefill, multipole_attention, quest_attention, duo_attention, shadowkv,
inf_llm, snap_kv) with optional data preparation.  Mirrors the mode-based
dispatch pattern of eval_longbench_v1.py.

Supported models: Llama 3.1 8B Instruct and Qwen3-8B. Chat template and
data-directory family are derived from --base_model.

Usage examples:
    # Prepare data + run baseline
    python eval_ruler.py --mode baseline \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --prepare \
        --output_dir results_ruler --run_name baseline

    # Run seer attention (data already prepared)
    python eval_ruler.py --mode seer_attention \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results_ruler --run_name seer_budget1024
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure baselines/ packages (seer_attn, multipole_attn, quest_attn) are importable
_BASELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

import torch
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# RULER task list (matches eval_ruler/synthetic.yaml)
# ---------------------------------------------------------------------------
ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

DEFAULT_SEQ_LENGTHS = [32768, 8192, 16384] #[4096, 8192, 16384, 32768, 65536, 131072]

RULER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark", "eval_ruler")
RULER_DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "benchmark" / "data" / "ruler_data"


# ---------------------------------------------------------------------------
# Model family inference
# ---------------------------------------------------------------------------
# Only Llama 3.1 and Qwen3 are supported. The chat template (for prepare.py)
# and the data-directory family are derived from --base_model so the three
# can never drift out of sync.
def infer_model_family(base_model: str) -> tuple[str, str]:
    """Return (model_template_type, tokenizer_family) for the base model."""
    s = base_model.lower()
    if "llama" in s:
        return "llama-3", "llama"
    if "qwen3" in s:
        return "qwen-3", "qwen3"
    raise ValueError(
        f"Unsupported --base_model: {base_model!r}. "
        "Only Llama 3.x and Qwen3 are supported."
    )


def model_name_tag(base_model: str) -> str:
    """Short, human-friendly tag for run names.

    'Qwen/Qwen3-8B' -> 'qwen', 'meta-llama/Llama-3.1-8B-Instruct' -> 'llama'.
    """
    s = base_model.lower()
    if "llama" in s:
        return "llama"
    if "qwen" in s:
        return "qwen"
    raise ValueError(f"Unsupported --base_model: {base_model!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified RULER Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "page_attention", "seer_attention",
                                 "seer_prefill",
                                 "multipole_attention", "quest_attention",
                                 "duo_attention",
                                 "shadowkv",
                                 "inf_llm",
                                 "snap_kv"])

    # Model
    parser.add_argument("--base_model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")

    # Data preparation
    parser.add_argument("--prepare", action="store_true",
                        help="Run data preparation before prediction (skips if data exists)")

    # RULER config
    parser.add_argument("--seq_lengths", type=int, nargs="+",
                        default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_samples", type=int, default=25)

    # Output
    parser.add_argument("--output_dir", type=str, default="results/results_ruler")
    parser.add_argument("--run_name", type=str, default=None)

    # DCT Page Attention params
    parser.add_argument("--page_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64,
                        help="Total selected page budget (sink + middle + recent). "
                             "DCTPageConfig receives total - sink - recent as its internal top_k.")
    parser.add_argument("--num_sink_pages", type=int, default=1)
    parser.add_argument("--num_recent_pages", type=int, default=4)
    parser.add_argument("--compress_ratio", type=float, default=0.125)
    parser.add_argument("--scoring_method", type=str, default="max",
                        choices=["mean", "max"])
    parser.add_argument("--group_agg_method", type=str, default="max",
                        choices=["mean", "max"])
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--compressed_token_rope", type=str, default="mixed",
                        choices=["mixed", "block_center"],
                        help="RoPE handling for compressed tokens. "
                             "'mixed': compress post-RoPE keys directly. "
                             "'block_center': invert RoPE, compress raw keys, re-rotate at block-center positions.")
    parser.add_argument("--continuous_rope", action="store_true",
                        help="Temporarily disabled — raises error if used")
    parser.add_argument("--score_use_quest_minmax", action="store_true",
                        help="Use QUEST-style min/max key metadata scoring instead of compressed proxy scoring")
    parser.add_argument("--no_triton", action="store_true")
    parser.add_argument("--attention_backend", type=str, default="upastream_flashinfer",
                        choices=["sdpa", "upstream_flashinfer"],
                        help="Attention backend for page_attention mode. "
                             "'sdpa' (default): assemble + torch.scaled_dot_product_attention "
                             "(unchanged production path). "
                             "'upstream_flashinfer': stock FlashInfer paged-decode kernel via "
                             "virtual-batch-per-(batch, KV head) layout (drop mode only). "
                             "Ignored for non-page_attention modes.")
    parser.add_argument("--verify_upstream_fi", action="store_true",
                        help="When --attention_backend upstream_flashinfer, run a per-layer SDPA "
                             "shadow comparison and log the per-step max-abs-diff distribution. "
                             "bf16 noise floor on this hardware is 0.05.")
    parser.add_argument("--comp_kv_quant", type=str, default="none",
                        choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"],
                        help="Fake-quantization of compressed K/V at write time "
                             "(precision study; no real byte-level storage change)")
    parser.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                        choices=["per_page", "per_comp_token"],
                        help="Scale granularity for comp_kv_quant")
    
    # Quest baseline (--mode quest_attention) — separate from DCT's --page_size/--top_k.
    # token_budget = quest_page_size * quest_top_k.
    parser.add_argument("--quest_page_size", type=int, default=32,
                        help="Quest baseline: tokens per KV page (Quest paper default: 16)")
    parser.add_argument("--quest_top_k", type=int, default=64,
                        help="Quest baseline: page budget (=token_budget/page_size). "
                             "Default 128 → token_budget=2048 with quest_page_size=16")

    # ShadowKV baseline params (only used when --mode shadowkv)
    parser.add_argument("--shadowkv_cache_mode", type=str, default="shadowkv_cpu",
                        choices=["shadowkv", "shadowkv_cpu"],
                        help="ShadowKVCache_CPU offloads V to CPU (production); "
                             "ShadowKVCache is GPU-only and batch=1 (sanity).")
    parser.add_argument("--sparse_budget", type=int, default=2192,
                        help="ShadowKV: tokens attended to per decode step.")
    parser.add_argument("--rank", type=int, default=160,
                        help="ShadowKV: SVD rank for compressed key cache.")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="ShadowKV: tokens per landmark chunk.")

    # InfLLM baseline params (only used when --mode inf_llm). All five
    # accuracy-relevant knobs (topk, block_size, n_local, n_init, repr_topk)
    # plus max_cached_block / chunk_size are exposed as CLI overrides on top
    # of baselines/infllm/config.py.
    parser.add_argument("--inf_llm_topk", type=int, default=64,
                        help="InfLLM: blocks attended per decode step (main sparsity dial).")
    parser.add_argument("--inf_llm_block_size", type=int, default=32,
                        help="InfLLM: tokens per block (retrieval granularity).")
    parser.add_argument("--inf_llm_n_local", type=int, default=4096,
                        help="InfLLM: sliding-window of always-attended recent tokens.")
    parser.add_argument("--inf_llm_n_recent", type=int, default=None,
                        help="InfLLM: output sliding window (defaults to n_local — byte-identical "
                             "to upstream behavior). Must be <= n_local. When set < n_local, "
                             "decouples the local-output window from the block-scoring horizon "
                             "and ATTENUATES the local-attention output (output mass over keys "
                             "in [n_recent, n_local) is zeroed; softmax is NOT renormalized), "
                             "which can significantly degrade retrieval at long context.")
    parser.add_argument("--inf_llm_n_init", type=int, default=32,
                        help="InfLLM: sink token count.")
    parser.add_argument("--inf_llm_repr_topk", type=int, default=4,
                        help="InfLLM: representative tokens per block.")
    parser.add_argument("--inf_llm_max_cached_block", type=int, default=128,
                        help="InfLLM: GPU block cache size (must be >= --inf_llm_topk).")
    parser.add_argument("--inf_llm_chunk_size", type=int, default=8192,
                        help="InfLLM: prefill chunk size for GreedySearch.")
    parser.add_argument("--inf_llm_exc_block_size", type=int, default=None,
                        help="InfLLM: per-iteration global-attention block size during prefill. "
                             "Upstream asserts exc_block_size <= n_local. None => "
                             "min(INF_LLM_CONFIG['exc_block_size'], n_local).")

    # SnapKV baseline params (only used when --mode snap_kv). Overrides on top
    # of baselines/snap_kv/config.py SNAPKV_CONFIG. SnapKV compresses the
    # prefill KV cache to `max_capacity_prompt` tokens per layer.
    parser.add_argument("--snapkv_window_size", type=int, default=32,
                        help="SnapKV: query window for attention-mass scoring (last N queries vote).")
    parser.add_argument("--snapkv_max_capacity_prompt", type=int, default=2048,
                        help="SnapKV: total kept tokens per layer after compression "
                             "(includes the trailing window_size tokens).")
    parser.add_argument("--snapkv_kernel_size", type=int, default=5,
                        help="SnapKV: 1D pooling kernel size for smoothing attention-mass scores.")
    parser.add_argument("--snapkv_pooling", type=str, default="avgpool",
                        choices=["avgpool", "maxpool"],
                        help="SnapKV: pooling op over attention-mass scores before topk.")

    # SeerAttention-R overrides (only used when --mode seer_attention).
    # CLI takes precedence over baselines/seer_attn/config.py (None = fall back).
    parser.add_argument("--seer_model", type=str, default="SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates",
                        help="SeerAttention-R AttnGates checkpoint (HF Hub ID or local path). "
                             "Default None → use SEER_ATTN_CONFIG['seer_model'].")
    parser.add_argument("--seerattn_sparsity_method", type=str, default="token_budget",
                        choices=["token_budget", "threshold"],
                        help="Decode sparsity method. Default None → use config.py value. "
                             "'token_budget' selects top-k blocks per step (k = "
                             "token_budget // gate_block_size). 'threshold' keeps blocks whose "
                             "gate softmax score exceeds SEER_ATTN_CONFIG['threshold']. "
                             "Note: 'nz_ratio' is prefill-only and rejected here.")
    parser.add_argument("--seerattn_token_budget", type=int, default=2048,
                        help="Active tokens per decode step (only used when "
                             "sparsity_method='token_budget'). Default None → use config.py "
                             "value. Internally rounded to block_budget = token_budget // "
                             "gate_block_size; pass a multiple of gate_block_size (default 64) "
                             "to avoid truncation.")
    parser.add_argument("--seerattn_gate_block_size", type=int, default=64,
                        choices=[16, 32, 64],
                        help="SeerAttention-R gate block size (= sparse-decode tile size). "
                             "Default 64 matches the released SeerAttention/SeerAttention-Decode-* "
                             "AttnGates checkpoints. Overriding to 16 or 32 feeds the gate K-pool "
                             "reps at a different granularity than it was trained on, so accuracy "
                             "degrades. Useful only as an ablation/sanity sweep.")

    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip run if summary.json already exists in output dir")

    args = parser.parse_args()

    if args.run_name is None:
        tag = model_name_tag(args.base_model)
        if args.mode == "baseline":
            args.run_name = f"{tag}_baseline"
        elif args.mode == "page_attention":
            args.run_name = (f"{tag}_page_attn_topk{args.top_k}T_cr{args.compress_ratio}"
                             f"_ps{args.page_size}_{args.unselected_mode}_{args.comp_kv_quant}")
        elif args.mode == "seer_attention":
            args.run_name = f"{tag}_seer_attention"
            if args.seerattn_sparsity_method is not None:
                args.run_name += f"_{args.seerattn_sparsity_method}"
            if args.seerattn_token_budget is not None:
                args.run_name += f"_b{args.seerattn_token_budget}"
            if args.seerattn_gate_block_size != 64:
                args.run_name += f"_bs{args.seerattn_gate_block_size}"
        elif args.mode == "seer_prefill":
            args.run_name = f"{tag}_seer_prefill"
        elif args.mode == "multipole_attention":
            args.run_name = f"{tag}_multipole_attention"
        elif args.mode == "quest_attention":
            args.run_name = f"{tag}_quest_ps{args.quest_page_size}_pb{args.quest_top_k}"
        elif args.mode == "duo_attention":
            args.run_name = f"{tag}_duo_attention"
        elif args.mode == "shadowkv":
            args.run_name = (f"{tag}_shadowkv_{args.shadowkv_cache_mode}"
                             f"_sb{args.sparse_budget}_r{args.rank}"
                             f"_cs{args.chunk_size}")
        elif args.mode == "inf_llm":
            args.run_name = (f"{tag}_inf_llm_topk{args.inf_llm_topk}"
                             f"_bs{args.inf_llm_block_size}"
                             f"_nlocal{args.inf_llm_n_local}"
                             f"_nini{args.inf_llm_n_init}"
                             f"_repr{args.inf_llm_repr_topk}")
            if args.inf_llm_n_recent is not None:
                args.run_name += f"_nrec{args.inf_llm_n_recent}"
        elif args.mode == "snap_kv":
            args.run_name = (f"{tag}_snap_kv_cap{args.snapkv_max_capacity_prompt}"
                             f"_{args.snapkv_pooling}"
                             f"_win{args.snapkv_window_size}"
                             f"_ks{args.snapkv_kernel_size}")

    if args.skip_existing:
        summary_path = Path(args.output_dir) / args.run_name / "summary.json"
        if summary_path.exists():
            print(f"SKIP (already exists): {summary_path}")
            sys.exit(0)

    return args


# ---------------------------------------------------------------------------
# Load RULER task configs (tokens_to_generate, metric_fn, etc.)
# ---------------------------------------------------------------------------
def load_task_configs():
    """Load task configurations from eval_ruler YAML and constants."""
    # Data constants (tokens_to_generate, template, etc.)
    sys.path.insert(0, os.path.join(RULER_DIR, "data"))
    data_constants = importlib.import_module("synthetic.constants")
    data_tasks = data_constants.TASKS

    # Eval constants (metric_fn)
    # Remove cached data version so we can load the eval version
    if "synthetic.constants" in sys.modules:
        del sys.modules["synthetic.constants"]
    sys.path.insert(0, os.path.join(RULER_DIR, "eval"))
    eval_constants = importlib.import_module("synthetic.constants")
    eval_tasks = eval_constants.TASKS

    # YAML customization
    with open(os.path.join(RULER_DIR, "synthetic.yaml"), "r") as f:
        yaml_tasks = yaml.safe_load(f)

    configs = {}
    for task_name, yaml_cfg in yaml_tasks.items():
        base_task = yaml_cfg["task"]
        cfg = dict(yaml_cfg)
        cfg.update(data_tasks[base_task])
        cfg.update(eval_tasks[base_task])
        configs[task_name] = cfg

    return configs


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(args):
    """Run eval_ruler/data/prepare.py for tasks that don't have data yet
    or whose cached JSONL has fewer samples than --num_samples. Existing
    files are overwritten by prepare.py (open with mode='w')."""
    model_template_type, model_family = infer_model_family(args.base_model)
    prepare_script = os.path.join(RULER_DIR, "data", "prepare.py")

    for seq_len in args.seq_lengths:
        for task in args.tasks:
            data_file = RULER_DATA_DIR / model_family / str(seq_len) / task / "validation.jsonl"
            if data_file.exists():
                with open(data_file, "r") as f:
                    existing = sum(1 for line in f if line.strip())
                if existing >= args.num_samples:
                    print(f"  Data exists ({existing} >= {args.num_samples} samples), skipping: {data_file}")
                    continue
                print(f"  Data has {existing} < {args.num_samples} samples, regenerating: {data_file}")

            data_file.parent.mkdir(parents=True, exist_ok=True)
            save_dir = str(RULER_DATA_DIR / model_family / str(seq_len))

            cmd = [
                sys.executable, prepare_script,
                "--save_dir", save_dir,
                "--benchmark", "synthetic",
                "--task", task,
                "--tokenizer_path", args.base_model,
                "--tokenizer_type", "hf",
                "--max_seq_length", str(seq_len),
                "--model_template_type", model_template_type,
                "--num_samples", str(args.num_samples),
            ]
            print(f"  Preparing {task} @ seq_len={seq_len} (num_samples={args.num_samples})...")
            result = subprocess.run(cmd, cwd=RULER_DIR, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR preparing {task}: {result.stderr}")
            else:
                print(f"  Done: {data_file}")


# ---------------------------------------------------------------------------
# top_k resolution helper
# ---------------------------------------------------------------------------
def _resolve_middle_top_k(args):
    middle = args.top_k - args.num_sink_pages - args.num_recent_pages
    if middle < 1:
        raise ValueError(
            f"--top_k={args.top_k} is too small: must exceed "
            f"--num_sink_pages ({args.num_sink_pages}) + "
            f"--num_recent_pages ({args.num_recent_pages}) "
            f"= {args.num_sink_pages + args.num_recent_pages}. "
            f"Note: --top_k now denotes the TOTAL selected page budget "
            f"(sink + middle + recent), not just middle pages."
        )
    return middle


# ---------------------------------------------------------------------------
# Upstream-FlashInfer preflight validation
# ---------------------------------------------------------------------------
# NOTE: keep this function name/logic IDENTICAL across eval_ruler.py,
# eval_longbench_v1.py, and eval_longbench_v2.py so future maintainers can grep.
# AIME25 / GPQA reuse it directly via `from eval_ruler import _validate_upstream_fi_args`.
def _validate_upstream_fi_args(args):
    """Hard preflights for --attention_backend upstream_flashinfer.

    Three checks (all hard errors, no silent downgrade):
      1. compressed-mode reject: upstream FI does not implement compressed mode.
      2. 64K memory preflight: refuses to start when projected paged-KV exceeds
         0.9 * total GPU memory (uses approximate model dims for known families).
      3. greedy-only assert: beam search / sampling break the lazy-init contract.
    """
    backend = getattr(args, "attention_backend", "sdpa")
    if backend != "upstream_flashinfer":
        return
    if getattr(args, "mode", None) != "page_attention":
        return

    # 1. Compressed-mode hard error.
    if getattr(args, "unselected_mode", "drop") != "drop":
        raise SystemExit(
            f"--attention_backend upstream_flashinfer requires "
            f"--unselected_mode drop (got {args.unselected_mode!r}). "
            f"Use --attention_backend sdpa for compressed mode. "
            f"To sweep both modes, run two passes."
        )

    # 2. Single-forward-per-decode-step hard assert. The lazy-init paged cache
    # is built once at the first decode step and assumes batch shape is stable
    # across steps. do_sample=True is fine — sampling only changes argmax ->
    # multinomial on the final logits; the forward pass and KV writes are
    # identical to greedy. What actually breaks the contract is multi-forward
    # decoding: beam search (num_beams>1) or parallel sampling
    # (num_return_sequences>1), which would either run N forwards per step or
    # expand the batch shape mid-generate.
    num_beams = getattr(args, "num_beams", 1)
    num_return_sequences = getattr(args, "num_return_sequences", 1)
    if num_beams > 1 or num_return_sequences > 1:
        raise SystemExit(
            f"--attention_backend upstream_flashinfer requires single-sequence "
            f"decoding (num_beams=1, num_return_sequences=1). "
            f"Got num_beams={num_beams}, num_return_sequences={num_return_sequences}. "
            f"The lazy-init cache contract assumes one forward per decode step "
            f"and a stable batch shape. (do_sample=True is supported.)"
        )

    # 3. 64K paged-KV memory preflight. Use approximate dims for known families.
    seq_lengths = list(getattr(args, "seq_lengths", []) or [])
    if not seq_lengths:
        return
    max_seq_len = max(seq_lengths)
    max_decode_steps = max(int(getattr(args, "max_new_tokens", 0)), 256)
    page_size = int(getattr(args, "page_size", 32))
    num_sink_pages = int(getattr(args, "num_sink_pages", 1))
    num_recent_pages = int(getattr(args, "num_recent_pages", 5))

    name = (getattr(args, "base_model", "") or "").lower()
    if "llama" in name:
        num_kv_heads, num_layers, head_dim = 8, 32, 128
    elif "qwen3" in name:
        num_kv_heads, num_layers, head_dim = 8, 36, 128
    else:
        print(f"  [preflight] WARNING: unknown model family for {name!r}; "
              f"skipping upstream-FI memory preflight.")
        return

    import math
    pages_per_head = (
        math.ceil((max_seq_len + max_decode_steps) / page_size)
        + num_sink_pages
        + (num_recent_pages + 1)
    )
    bsz = 1  # eval scripts process one sample at a time
    proj_bytes = (
        bsz * num_kv_heads * pages_per_head * 2 * page_size * head_dim * 2 * num_layers
    )
    try:
        import torch as _torch
        total_bytes = _torch.cuda.get_device_properties(0).total_memory
    except Exception as e:
        print(f"  [preflight] WARNING: could not query GPU memory ({e}); "
              f"skipping upstream-FI memory preflight.")
        return
    threshold = 0.9 * total_bytes
    if proj_bytes > threshold:
        raise SystemExit(
            f"--attention_backend upstream_flashinfer projected paged-KV memory "
            f"({proj_bytes / 1e9:.2f} GiB) exceeds 0.9 * total GPU memory "
            f"({threshold / 1e9:.2f} GiB of {total_bytes / 1e9:.2f} GiB). "
            f"Reduce --seq_lengths (max={max_seq_len}) or run on a larger GPU."
        )
    print(f"  [preflight] upstream-FI projected paged-KV: "
          f"{proj_bytes / 1e9:.2f} GiB (threshold {threshold / 1e9:.2f} GiB) — OK")


# ---------------------------------------------------------------------------
# Monkey-patching
# ---------------------------------------------------------------------------
def apply_monkey_patch(args):
    """Apply attention monkey-patch before model loading."""
    if args.mode == "page_attention":
        model_name_lower = args.base_model.lower()
        if "llama" in model_name_lower:
            from dct_page_attention import replace_llama_attn
            replace_llama_attn(
                page_size=args.page_size,
                top_k=_resolve_middle_top_k(args),
                num_sink_pages=args.num_sink_pages,
                num_recent_pages=args.num_recent_pages,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=True,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
                score_use_quest_minmax=args.score_use_quest_minmax,
                attention_backend=args.attention_backend,
            )
        elif "qwen3" in model_name_lower:
            from dct_page_attention import replace_qwen3_attn
            replace_qwen3_attn(
                page_size=args.page_size,
                top_k=_resolve_middle_top_k(args),
                num_sink_pages=args.num_sink_pages,
                num_recent_pages=args.num_recent_pages,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=True,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
                score_use_quest_minmax=args.score_use_quest_minmax,
                attention_backend=args.attention_backend,
            )
        else:
            from dct_page_attention import replace_qwen2_attn
            replace_qwen2_attn(
                page_size=args.page_size,
                top_k=_resolve_middle_top_k(args),
                num_sink_pages=args.num_sink_pages,
                num_recent_pages=args.num_recent_pages,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=True,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
                score_use_quest_minmax=args.score_use_quest_minmax,
                attention_backend=args.attention_backend,
            )
    elif args.mode == "multipole_attention":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines"))
        from multipole_attn import replace_attn_multipole
        from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
        sys.path.pop(0)
        MULTIPOLE_ATTN_CONFIG["base_model"] = args.base_model
        replace_attn_multipole(MULTIPOLE_ATTN_CONFIG)
    elif args.mode == "quest_attention":
        pass  # Quest uses custom model class, no monkey-patch needed
    elif args.mode == "duo_attention":
        pass  # DuoAttention patches per-instance forwards post-load (see load_model_and_tokenizer)
    elif args.mode == "shadowkv":
        pass  # ShadowKV uses a custom LLM class; no monkey-patch needed.
    elif args.mode == "inf_llm":
        pass  # InfLLM patches per-instance forwards post-load (see load_model_and_tokenizer)
    elif args.mode == "snap_kv":
        pass  # SnapKV patches the attention class post-load (see load_model_and_tokenizer)
    elif args.mode == "baseline":
        print("Baseline mode: full attention (no monkey-patch)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(args):
    """Load model and tokenizer based on mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        from seer_attn import SeerDecodingQwen3ForCausalLM

        seer_model = args.seer_model or SEER_ATTN_CONFIG["seer_model"]
        sparsity_method = args.seerattn_sparsity_method or SEER_ATTN_CONFIG["sparsity_method"]
        token_budget = (args.seerattn_token_budget
                        if args.seerattn_token_budget is not None
                        else SEER_ATTN_CONFIG["token_budget"])
        print(f"Loading SeerAttention-R model: {seer_model}")
        if args.seer_model is not None:
            print(f"  Override: seer_model={args.seer_model}")
        if args.seerattn_sparsity_method is not None:
            print(f"  Override: seerattn_sparsity_method={args.seerattn_sparsity_method}")
        if args.seerattn_token_budget is not None:
            print(f"  Override: seerattn_token_budget={args.seerattn_token_budget}")
        if args.seerattn_gate_block_size != 64:
            print(f"  Override: seerattn_gate_block_size={args.seerattn_gate_block_size} "
                  f"(checkpoint default is 64; ablation only)")
        model = SeerDecodingQwen3ForCausalLM.from_pretrained(
            seer_model,
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method=sparsity_method,
            seerattn_token_budget=token_budget,
            seerattn_threshold=SEER_ATTN_CONFIG["threshold"],
            seerattn_start_layer=SEER_ATTN_CONFIG["start_layer"],
            seerattn_gate_block_size=args.seerattn_gate_block_size,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            max_position_embeddings=131072,
        ).cuda()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    elif args.mode == "seer_prefill":
        from seer_attn.prefill_config import SEER_PREFILL_CONFIG, load_seer_prefill_model

        seer_model = SEER_PREFILL_CONFIG["seer_model"]
        print(f"Loading SeerAttention prefill-sparse: {seer_model}")
        model, hf_config = load_seer_prefill_model(SEER_PREFILL_CONFIG, torch.bfloat16)
        model = model.cuda()
        model.eval()
        # AttnGates repo contains only gate weights; tokenizer lives on base_model.
        tokenizer = AutoTokenizer.from_pretrained(hf_config.base_model, padding_side="left")
        print(f"Model loaded (base={hf_config.base_model}). "
              f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    elif args.mode == "quest_attention":
        base_model = args.base_model
        page_size = args.quest_page_size
        token_budget = args.quest_page_size * args.quest_top_k  # quest_top_k = page_budget
        # Size the preallocated KV pool to the actual eval workload (with headroom for generation).
        max_seq_len = max(args.seq_lengths) + 4096

        model_name_lower = base_model.lower()
        if "qwen3" in model_name_lower:
            from quest_attn import Qwen3ForCausalLM as QuestModel
        elif any(fam in model_name_lower for fam in ["llama", "mistral"]):
            from quest_attn import LlamaForCausalLM as QuestModel
        else:
            raise ValueError(
                f"Quest supports LLaMA-family (Llama-2, Llama-3.x, Mistral) and Qwen3 models, "
                f"got: {base_model}"
            )
        print(f"Loading Quest model: {base_model} (page_size={page_size}, page_budget={args.quest_top_k}, token_budget={token_budget})")
        model = QuestModel.from_pretrained(
            base_model,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        model.quest_init(
            page_size=page_size,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    elif args.mode == "shadowkv":
        from shadow_kv import SHADOWKV_CONFIG, build_shadowkv_llm

        SHADOWKV_CONFIG["base_model"] = args.base_model
        SHADOWKV_CONFIG["cache_mode"] = args.shadowkv_cache_mode
        SHADOWKV_CONFIG["sparse_budget"] = args.sparse_budget
        SHADOWKV_CONFIG["rank"] = args.rank
        SHADOWKV_CONFIG["chunk_size"] = args.chunk_size
        SHADOWKV_CONFIG["max_length"] = max(args.seq_lengths) + 4096
        print(f"Loading ShadowKV model: {args.base_model} "
              f"(cache={args.shadowkv_cache_mode}, sparse_budget={args.sparse_budget}, "
              f"rank={args.rank}, chunk_size={args.chunk_size})")
        model = build_shadowkv_llm(SHADOWKV_CONFIG)
        tokenizer = model.tokenizer
        print("ShadowKV LLM ready.")
    else:
        # DuoAttention's, InfLLM's, and SnapKV's replacement forwards assume eager-style Q/K/V signatures.
        attn_impl = "eager" if args.mode in {"duo_attention", "inf_llm", "snap_kv"} else "sdpa"
        print(f"Loading model: {args.base_model} (attn: {attn_impl})")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        yarn_kwargs = {}
        # InfLLM replaces RoPE with its own RotaryEmbeddingESM and is Llama-only,
        # so the Qwen3-yarn rope_parameters injection is irrelevant for inf_llm.
        if "qwen3" in args.base_model.lower() and args.mode != "inf_llm":
            yarn_kwargs = {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 1000000.0,
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                },
                "max_position_embeddings": 131072,
            }
        # Old transformers envs (duo_attention) only accept torch_dtype=;
        # transformers 5.x (main DCT-Page env, including inf_llm) only accepts dtype=.
        dtype_kwarg = (
            {"torch_dtype": torch.bfloat16}
            if args.mode == "duo_attention"
            else {"dtype": torch.bfloat16}
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            **dtype_kwarg,
            device_map="cuda:0",
            attn_implementation=attn_impl,
            **yarn_kwargs,
        )
        model.eval()
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        if args.mode == "multipole_attention":
            from multipole_attn import init_multipole_layers
            init_multipole_layers(model)
            print("Multipole attention layers initialized.")

        if args.mode == "duo_attention":
            from duo_attn import init_duo_attention, assert_llama
            from duo_attn.config import DUO_ATTN_CONFIG
            assert_llama(args.base_model)
            DUO_ATTN_CONFIG["base_model"] = args.base_model
            init_duo_attention(model, DUO_ATTN_CONFIG)

        if args.mode == "inf_llm":
            from infllm import (
                assert_llama_only,
                build_inf_llm_generator,
                init_inf_llm,
            )
            from infllm.config import INF_LLM_CONFIG
            assert_llama_only(args.base_model)
            INF_LLM_CONFIG["base_model"] = args.base_model
            INF_LLM_CONFIG["topk"] = args.inf_llm_topk
            INF_LLM_CONFIG["block_size"] = args.inf_llm_block_size
            INF_LLM_CONFIG["n_local"] = args.inf_llm_n_local
            INF_LLM_CONFIG["n_init"] = args.inf_llm_n_init
            INF_LLM_CONFIG["repr_topk"] = args.inf_llm_repr_topk
            INF_LLM_CONFIG["max_cached_block"] = args.inf_llm_max_cached_block
            INF_LLM_CONFIG["chunk_size"] = args.inf_llm_chunk_size
            INF_LLM_CONFIG["n_recent"] = args.inf_llm_n_recent
            requested_exc = (args.inf_llm_exc_block_size
                             if args.inf_llm_exc_block_size is not None
                             else INF_LLM_CONFIG["exc_block_size"])
            INF_LLM_CONFIG["exc_block_size"] = min(requested_exc, args.inf_llm_n_local)
            init_inf_llm(model, INF_LLM_CONFIG)
            args._inf_llm_generator = build_inf_llm_generator(model, tokenizer, INF_LLM_CONFIG)

        if args.mode == "snap_kv":
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines"))
            from snap_kv import init_snap_kv
            from snap_kv.config import SNAPKV_CONFIG
            sys.path.pop(0)
            SNAPKV_CONFIG["base_model"] = args.base_model
            SNAPKV_CONFIG["window_size"] = args.snapkv_window_size
            SNAPKV_CONFIG["max_capacity_prompt"] = args.snapkv_max_capacity_prompt
            SNAPKV_CONFIG["kernel_size"] = args.snapkv_kernel_size
            SNAPKV_CONFIG["pooling"] = args.snapkv_pooling
            init_snap_kv(model, SNAPKV_CONFIG)

    # Seed per-attention-module `_upstream_fi_build_kwargs` AFTER from_pretrained.
    # `replace_*_attn` runs BEFORE model load by project convention, so this
    # post-load walk is the only safe time to stash per-instance state.
    if args.mode == "page_attention" and getattr(args, "attention_backend", "sdpa") == "upstream_flashinfer":
        from dct_page_attention import _init_upstream_fi_build_kwargs
        _init_upstream_fi_build_kwargs(model)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction for a single task
# ---------------------------------------------------------------------------
def predict_task(model, tokenizer, task, task_config, data_dir, pred_dir, args):
    """Generate predictions for one RULER task. Returns list of result dicts."""
    data_file = data_dir / task / "validation.jsonl"
    if not data_file.exists():
        print(f"  WARNING: data file not found: {data_file}, skipping")
        return []

    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    if args.num_samples > 0 and len(data) > args.num_samples:
        data = data[:args.num_samples]

    tokens_to_generate = task_config["tokens_to_generate"]

    # Resume support
    pred_file = pred_dir / f"{task}.jsonl"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    completed_indices = set()
    if pred_file.exists():
        with open(pred_file, "r") as f:
            for line in f:
                if line.strip():
                    completed_indices.add(json.loads(line)["index"])
        print(f"  Resuming {task}: {len(completed_indices)} already done")

    remaining = [s for s in data if s["index"] not in completed_indices]
    if not remaining:
        print(f"  {task}: all samples already completed")
        # Return all results for scoring
        with open(pred_file, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(pred_file, "a", encoding="utf-8", buffering=1) as fout:
        for sample in tqdm(remaining, desc=f"  {task}"):
            prompt = sample["input"]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                if args.mode == "seer_attention":
                    output_ids, _ = model.batch_exist_generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_length=input_len + tokens_to_generate,
                        do_sample=False,
                    )
                elif args.mode == "duo_attention":
                    # DuoAttention's v4.34 tuple-cache forward is incompatible
                    # with transformers>=4.37 generate()'s DynamicCache. Use a
                    # manual greedy loop matching duo-attention/eval/LongBench/pred.py.
                    from duo_attn import duo_generate_greedy
                    _eos = model.generation_config.eos_token_id
                    eos_ids = _eos if isinstance(_eos, (list, tuple)) else [_eos]
                    output_ids = duo_generate_greedy(
                        model, input_ids,
                        max_new_tokens=tokens_to_generate,
                        eos_token_ids=eos_ids,
                    )
                elif args.mode == "inf_llm":
                    # InfLLM uses a stateful ContextManager KV cache that HF
                    # generate() cannot round-trip. Use our GreedySearch adapter.
                    _eos = model.generation_config.eos_token_id
                    extra_eos = (list(_eos) if isinstance(_eos, (list, tuple)) else [_eos])
                    extra_eos = [e for e in extra_eos if e is not None and e != tokenizer.eos_token_id]
                    output_ids = args._inf_llm_generator.generate(
                        input_ids,
                        max_new_tokens=tokens_to_generate,
                        extra_end_token_ids=extra_eos,
                    )
                elif args.mode == "page_attention" and args.attention_backend == "upstream_flashinfer":
                    from dct_page_attention import _generate_with_upstream_fi
                    if args.verify_upstream_fi:
                        for _m in model.modules():
                            if hasattr(_m, "q_proj") and hasattr(_m, "k_proj"):
                                _m._verify_upstream = True

                    def _harvest_verify_diffs(_model, _output_ids):
                        if not args.verify_upstream_fi:
                            return
                        all_diffs = []
                        for _m in _model.modules():
                            diffs = getattr(_m, "_verify_diffs", None)
                            if diffs:
                                all_diffs.extend(diffs)
                        if all_diffs:
                            import numpy as _np
                            arr = _np.array(all_diffs)
                            print(f"  [verify] _verify_diffs n={len(all_diffs)} "
                                  f"max={arr.max():.4f} "
                                  f"p99={_np.percentile(arr, 99):.4f} "
                                  f"p50={_np.percentile(arr, 50):.4f}")

                    output_ids = _generate_with_upstream_fi(
                        model,
                        input_ids,
                        max_new_tokens=tokens_to_generate,
                        on_post_generate=_harvest_verify_diffs,
                        do_sample=False,
                        use_cache=True,
                    )
                else:
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=tokens_to_generate,
                        do_sample=False,
                        use_cache=True,
                    )

            generated_ids = output_ids[0, input_len:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if args.mode == "quest_attention":
                model.quest_clear()
            elif args.mode == "shadowkv":
                model.shadowkv_clear()
            elif args.mode == "inf_llm":
                # ContextManager persists past_kv across samples; reset it.
                args._inf_llm_generator.clear()

            result = {
                "index": sample["index"],
                "pred": pred_text,
                "input": prompt,
                "outputs": sample["outputs"],
                "others": sample.get("others", {}),
                "truncation": sample.get("truncation", -1),
                "length": sample.get("length", -1),
            }
            fout.write(json.dumps(result) + "\n")

    # Reload all results for scoring
    with open(pred_file, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Evaluation (scoring)
# ---------------------------------------------------------------------------
def postprocess_pred(predict_str):
    """Clean prediction string (matches eval_ruler/eval/evaluate.py)."""
    import re
    predict_str = predict_str.strip()
    np_pattern = re.compile(r"[\x00-\x1f]")
    predict_str = np_pattern.sub("\n", predict_str).strip()
    return predict_str


def evaluate_task(predictions, task_config):
    """Score predictions for one task. Returns (score, nulls_str)."""
    preds = [postprocess_pred(p["pred"]) for p in predictions]
    refs = [p["outputs"] for p in predictions]

    nulls = f"{sum(len(x) == 0 for x in preds)}/{len(preds)}"

    if len(refs) > 0 and refs[0][0] is not None:
        score = task_config["metric_fn"](preds, refs)
    else:
        score = 0.0

    return score, nulls


def write_summary_csv(eval_results, pred_dir):
    """Write summary.csv to pred_dir (same format as eval_ruler/eval/evaluate.py)."""
    import pandas as pd

    tasks = list(eval_results.keys())
    scores = [eval_results[t]["score"] for t in tasks]
    nulls = [eval_results[t]["nulls"] for t in tasks]

    dfs = [
        ["Tasks"] + tasks,
        ["Score"] + scores,
        ["Nulls"] + nulls,
    ]
    output_file = pred_dir / ("summary.csv" if len(tasks) > 1 else f"summary-{tasks[0]}.csv")
    df = pd.DataFrame(dfs)
    df.to_csv(output_file, index=False)
    print(f"  Summary saved to {output_file}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _save_summary(args, all_seq_results):
    """Build and save summary.csv + summary.json from current results."""
    if not all_seq_results:
        return

    seq_lens = sorted(all_seq_results.keys())

    # Per-seq-len summary.csv
    for sl in seq_lens:
        sl_results = all_seq_results.get(sl, {})
        if not sl_results:
            continue
        pred_dir = Path(args.output_dir) / args.run_name / "synthetic" / str(sl) / "pred"
        eval_results = {t: {"score": s, "nulls": ""} for t, s in sl_results.items()}
        write_summary_csv(eval_results, pred_dir)

    # Consolidated summary.json
    task_avgs = {}
    for task in args.tasks:
        scores = []
        for sl in seq_lens:
            if task in all_seq_results.get(sl, {}):
                scores.append(all_seq_results[sl][task])
        if scores:
            task_avgs[task] = sum(scores) / len(scores)

    overall_scores = []
    for sl in seq_lens:
        sl_scores = list(all_seq_results.get(sl, {}).values())
        if sl_scores:
            overall_scores.append(sum(sl_scores) / len(sl_scores))
    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    summary = {
        "mode": args.mode,
        "base_model": args.base_model,
        "run_name": args.run_name,
        "seq_length_scores": {str(sl): all_seq_results[sl] for sl in seq_lens},
        "task_averages": {t: round(a, 2) for t, a in task_avgs.items()},
        "overall": round(overall, 2),
    }
    if args.mode == "page_attention":
        summary["top_k"] = args.top_k
        summary["middle_top_k"] = _resolve_middle_top_k(args)
        summary["page_size"] = args.page_size
        summary["compress_ratio"] = args.compress_ratio
        summary["scoring_method"] = args.scoring_method
        summary["group_agg_method"] = args.group_agg_method
        summary["unselected_mode"] = args.unselected_mode
    elif args.mode == "seer_attention":
        from seer_attn.config import SEER_ATTN_CONFIG
        summary["seer_attn_config"] = dict(SEER_ATTN_CONFIG)
        if args.seer_model is not None:
            summary["seer_attn_config"]["seer_model"] = args.seer_model
        if args.seerattn_sparsity_method is not None:
            summary["seer_attn_config"]["sparsity_method"] = args.seerattn_sparsity_method
        if args.seerattn_token_budget is not None:
            summary["seer_attn_config"]["token_budget"] = args.seerattn_token_budget
        if args.seerattn_gate_block_size is not None:
            summary["seer_attn_config"]["seerattn_gate_block_size"] = args.seerattn_gate_block_size
    elif args.mode == "multipole_attention":
        from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
        summary["multipole_attn_config"] = MULTIPOLE_ATTN_CONFIG
    elif args.mode == "quest_attention":
        summary["quest_attn_config"] = {
            "base_model": args.base_model,
            "page_size": args.quest_page_size,
            "page_budget": args.quest_top_k,
            "token_budget": args.quest_page_size * args.quest_top_k,
        }
    elif args.mode == "snap_kv":
        summary["snap_kv_config"] = {
            "base_model": args.base_model,
            "window_size": args.snapkv_window_size,
            "max_capacity_prompt": args.snapkv_max_capacity_prompt,
            "kernel_size": args.snapkv_kernel_size,
            "pooling": args.snapkv_pooling,
        }

    summary_path = Path(args.output_dir) / args.run_name / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def _print_results_table(args, all_seq_results):
    """Print the formatted results table."""
    print(f"\n{'=' * 60}")
    print(f"RULER Results — {args.run_name}")
    print("=" * 60)

    if not all_seq_results:
        return

    seq_lens = sorted(all_seq_results.keys())
    header = f"{'Task':24s}" + "".join(f" {sl:>8d}" for sl in seq_lens) + "     Avg"
    print(header)
    print("-" * len(header))

    task_avgs = {}
    for task in args.tasks:
        scores = []
        row = f"{task:24s}"
        for sl in seq_lens:
            if task in all_seq_results.get(sl, {}):
                s = all_seq_results[sl][task]
                scores.append(s)
                row += f" {s:>8.2f}"
            else:
                row += f" {'N/A':>8s}"
        avg = sum(scores) / len(scores) if scores else 0
        task_avgs[task] = avg
        row += f" {avg:>7.2f}"
        print(row)

    print("-" * len(header))
    overall_scores = []
    row = f"{'AVERAGE':24s}"
    for sl in seq_lens:
        sl_scores = list(all_seq_results.get(sl, {}).values())
        if sl_scores:
            sl_avg = sum(sl_scores) / len(sl_scores)
            overall_scores.append(sl_avg)
            row += f" {sl_avg:>8.2f}"
        else:
            row += f" {'N/A':>8s}"
    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    row += f" {overall:>7.2f}"
    print(row)
    print("=" * len(header))


def main():
    args = parse_args()
    start_time = time.time()

    torch.manual_seed(42)

    # Hard preflights for --attention_backend upstream_flashinfer (compressed-mode
    # reject, greedy-only assert, projected paged-KV memory). No-op for sdpa.
    _validate_upstream_fi_args(args)

    task_configs = load_task_configs()

    # Validate requested tasks
    for task in args.tasks:
        if task not in task_configs:
            raise ValueError(f"Unknown task: {task}. Available: {list(task_configs.keys())}")

    # Step 1: Data preparation (optional)
    if args.prepare:
        print("=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        prepare_data(args)

    # Step 2: Monkey-patch + load model
    print("\n" + "=" * 60)
    print(f"LOADING MODEL (mode={args.mode})")
    print("=" * 60)
    apply_monkey_patch(args)
    model, tokenizer = load_model_and_tokenizer(args)

    _, model_family = infer_model_family(args.base_model)

    # Step 3: Predict + evaluate per sequence length
    all_seq_results = {}  # {seq_len: {task: score}}

    for seq_len in args.seq_lengths:
        print(f"\n{'=' * 60}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print("=" * 60)

        data_dir = RULER_DATA_DIR / model_family / str(seq_len)
        pred_dir = Path(args.output_dir) / args.run_name / "synthetic" / str(seq_len) / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)

        for task in args.tasks:
            if task not in task_configs:
                continue

            predictions = predict_task(
                model, tokenizer, task, task_configs[task],
                data_dir, pred_dir, args,
            )

            if predictions:
                score, nulls = evaluate_task(predictions, task_configs[task])
                print(f"  {task}: score={score:.2f}, nulls={nulls}")

                # Update accumulated results and save immediately
                if seq_len not in all_seq_results:
                    all_seq_results[seq_len] = {}
                all_seq_results[seq_len][task] = score
                _save_summary(args, all_seq_results)

    # Step 4: Final results table
    _print_results_table(args, all_seq_results)

    elapsed = (time.time() - start_time) / 60
    print(f"Total time: {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
