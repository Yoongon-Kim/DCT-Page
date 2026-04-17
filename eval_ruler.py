"""
Unified RULER benchmark evaluation.

Supports four attention modes (baseline, page_attention, seer_attention,
multipole_attention) with optional data preparation.  Mirrors the mode-based
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

DEFAULT_SEQ_LENGTHS = [32768] #[4096, 8192, 16384, 32768, 65536, 131072]

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
        return "meta-llama3", "llama"
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
                                 "multipole_attention", "quest_attention"])

    # Model
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen3-8B")

    # Data preparation
    parser.add_argument("--prepare", action="store_true",
                        help="Run data preparation before prediction (skips if data exists)")

    # RULER config
    parser.add_argument("--seq_lengths", type=int, nargs="+",
                        default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--tasks", type=str, nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_samples", type=int, default=25)

    # Output
    parser.add_argument("--output_dir", type=str, default="results_ruler")
    parser.add_argument("--run_name", type=str, default=None)

    # DCT Page Attention params
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--compress_ratio", type=float, default=0.125)
    parser.add_argument("--scoring_method", type=str, default="max",
                        help="'mean'|'max'|'sum'|'proxy_dc_ac_{lam}'|'spread_dc_ac_{lam}'|'spectral_recon_max'")
    parser.add_argument("--group_agg_method", type=str, default="max",
                        choices=["mean", "max", "topp"])
    parser.add_argument("--unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed"])
    parser.add_argument("--compression_method", type=str, default="dct",
                        choices=["haar", "dct"],
                        help="Compression method for unselected pages (used when unselected_mode=compressed)")
    parser.add_argument("--compressed_token_rope", type=str, default="mixed",
                        choices=["mixed", "block_center"],
                        help="RoPE handling for compressed tokens. "
                             "'mixed': compress post-RoPE keys directly. "
                             "'block_center': invert RoPE, compress raw keys, re-rotate at block-center positions.")
    parser.add_argument("--continuous_rope", action="store_true",
                        help="Temporarily disabled — raises error if used")
    parser.add_argument("--weight_compressed_by_population", action="store_true",
                        help="In compressed mode, scale each unselected-page rep's softmax mass "
                             "by page_size/comp_size via a log(n) bias on QK logits "
                             "(multipole-style population weighting). No-op for drop mode.")
    parser.add_argument("--no_triton", action="store_true")
    parser.add_argument("--comp_kv_quant", type=str, default="none",
                        choices=["none", "fp8_e4m3", "fp8_e5m2", "int8", "int4"],
                        help="Fake-quantization of compressed K/V at write time "
                             "(precision study; no real byte-level storage change)")
    parser.add_argument("--comp_kv_quant_granularity", type=str, default="per_page",
                        choices=["per_page", "per_comp_token"],
                        help="Scale granularity for comp_kv_quant")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip run if summary.json already exists in output dir")

    args = parser.parse_args()

    if args.run_name is None:
        tag = model_name_tag(args.base_model)
        if args.mode == "baseline":
            args.run_name = f"{tag}_baseline"
        elif args.mode == "page_attention":
            args.run_name = (f"{tag}_page_attn_topk{args.top_k}_cr{args.compress_ratio}"
                             f"_ps{args.page_size}_{args.unselected_mode}_{args.comp_kv_quant}")
        elif args.mode == "seer_attention":
            args.run_name = f"{tag}_seer_attention"
        elif args.mode == "multipole_attention":
            args.run_name = f"{tag}_multipole_attention"
        elif args.mode == "quest_attention":
            args.run_name = f"{tag}_quest_attention"

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
    """Run eval_ruler/data/prepare.py for tasks that don't have data yet."""
    _, model_family = infer_model_family(args.base_model)
    prepare_script = os.path.join(RULER_DIR, "data", "prepare.py")

    for seq_len in args.seq_lengths:
        for task in args.tasks:
            data_file = RULER_DATA_DIR / model_family / str(seq_len) / task / "validation.jsonl"
            if data_file.exists():
                print(f"  Data exists, skipping: {data_file}")
                continue

            data_file.parent.mkdir(parents=True, exist_ok=True)
            save_dir = str(RULER_DATA_DIR / model_family / str(seq_len))

            # NeMo-Skills-style prep: bare task text, no chat tokens baked in.
            # The chat template is applied at inference time via apply_chat_template.
            cmd = [
                sys.executable, prepare_script,
                "--save_dir", save_dir,
                "--benchmark", "synthetic",
                "--task", task,
                "--tokenizer_path", args.base_model,
                "--tokenizer_type", "hf",
                "--max_seq_length", str(seq_len),
                "--model_template_type", "base",
                "--prepare_for_ns",
                "--num_samples", str(args.num_samples),
            ]
            print(f"  Preparing {task} @ seq_len={seq_len}...")
            result = subprocess.run(cmd, cwd=RULER_DIR, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR preparing {task}: {result.stderr}")
            else:
                print(f"  Done: {data_file}")


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
                top_k=args.top_k,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compression_method=args.compression_method,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=args.weight_compressed_by_population,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
            )
        elif "qwen3" in model_name_lower:
            from dct_page_attention import replace_qwen3_attn
            replace_qwen3_attn(
                page_size=args.page_size,
                top_k=args.top_k,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compression_method=args.compression_method,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=args.weight_compressed_by_population,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
            )
        else:
            from dct_page_attention import replace_qwen2_attn
            replace_qwen2_attn(
                page_size=args.page_size,
                top_k=args.top_k,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                compress_ratio=args.compress_ratio,
                scoring_method=args.scoring_method,
                group_agg_method=args.group_agg_method,
                unselected_mode=args.unselected_mode,
                compression_method=args.compression_method,
                compressed_token_rope=args.compressed_token_rope,
                continuous_rope=args.continuous_rope,
                use_triton=not args.no_triton,
                weight_compressed_by_population=args.weight_compressed_by_population,
                comp_kv_quant=args.comp_kv_quant,
                comp_kv_quant_granularity=args.comp_kv_quant_granularity,
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

        seer_model = SEER_ATTN_CONFIG["seer_model"]
        print(f"Loading SeerAttention-R model: {seer_model}")
        model = SeerDecodingQwen3ForCausalLM.from_pretrained(
            seer_model,
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method=SEER_ATTN_CONFIG["sparsity_method"],
            seerattn_token_budget=SEER_ATTN_CONFIG["token_budget"],
            seerattn_threshold=SEER_ATTN_CONFIG["threshold"],
            seerattn_start_layer=SEER_ATTN_CONFIG["start_layer"],
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
    elif args.mode == "quest_attention":
        from quest_attn.config import QUEST_ATTN_CONFIG
        from quest_attn import LlamaForCausalLM as QuestLlamaForCausalLM

        base_model = QUEST_ATTN_CONFIG["base_model"]
        model_name_lower = base_model.lower()
        if not any(fam in model_name_lower for fam in ["llama", "mistral"]):
            raise ValueError(
                f"Quest only supports LLaMA-family models (Llama-2, Llama-3.x, Mistral), "
                f"got: {base_model}"
            )
        print(f"Loading Quest model: {base_model}")
        model = QuestLlamaForCausalLM.from_pretrained(
            base_model,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        model.quest_init(
            page_size=QUEST_ATTN_CONFIG["page_size"],
            max_seq_len=QUEST_ATTN_CONFIG["max_seq_len"],
            token_budget=QUEST_ATTN_CONFIG["token_budget"],
            dtype=torch.float16,
            device=torch.device("cuda:0"),
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    else:
        attn_impl = "sdpa"
        print(f"Loading model: {args.base_model} (attn: {attn_impl})")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        yarn_kwargs = {}
        if "qwen3" in args.base_model.lower():
            yarn_kwargs = {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 1000000.0,
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                },
                "max_position_embeddings": 131072,
            }
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            **yarn_kwargs,
        )
        model.eval()
        print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        if args.mode == "multipole_attention":
            from multipole_attn import init_multipole_layers
            init_multipole_layers(model)
            print("Multipole attention layers initialized.")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction for a single task
# ---------------------------------------------------------------------------
def _build_chat_messages(sample):
    """Build the chat messages for a RULER sample.

    The cwe/vt task scripts embed a one-shot example inside `input`. If we can
    detect it (by finding the answer_prefix text inside `input`), we split
    the embedded example into proper user/assistant/user turns instead of
    flattening everything into one user message. Chat models are trained on
    alternating turns; this gives the few-shot a clearer structure.
    """
    answer_prefix = sample.get("answer_prefix", "")
    inp = sample["input"]

    if not answer_prefix:
        return [{"role": "user", "content": inp}], False  # no partial assistant

    # Detect embedded few-shot by searching for the answer_prefix text in the
    # input. Try the full stripped prefix first, then a 40-char prefix (to
    # tolerate format-arg substitution, e.g. vt's {num_v}/{query}).
    stripped = answer_prefix.strip()
    idx = inp.find(stripped)
    if idx < 0 and len(stripped) > 40:
        idx = inp.find(stripped[:40])

    if idx < 0:
        # No few-shot detected: single user turn + partial assistant
        return [
            {"role": "user", "content": inp},
            {"role": "assistant", "content": answer_prefix},
        ], True

    # Few-shot present: split into three turns + partial assistant.
    end = inp.find("\n", idx)
    if end < 0:
        # Unexpected: no newline after the embedded answer → fall back
        return [
            {"role": "user", "content": inp},
            {"role": "assistant", "content": answer_prefix},
        ], True

    return [
        {"role": "user", "content": inp[:idx].rstrip()},
        {"role": "assistant", "content": inp[idx:end].strip()},
        {"role": "user", "content": inp[end + 1:].lstrip()},
        {"role": "assistant", "content": answer_prefix},
    ], True


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
            # Data is prepped with --model_template_type base --prepare_for_ns,
            # so `input` is bare task text. Apply the model's own chat template
            # at inference time (NeMo-Skills approach). For few-shot tasks
            # (cwe, vt), _build_chat_messages splits the embedded example into
            # proper alternating user/assistant/user turns; otherwise it's a
            # single user turn. answer_prefix (if any) becomes a partial
            # assistant turn completed by the model via continue_final_message.
            # date_string is pinned for reproducibility (Llama-3.1 would inject
            # today's date otherwise).
            messages, is_partial = _build_chat_messages(sample)
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                continue_final_message=is_partial,
                add_generation_prompt=not is_partial,
                return_tensors="pt",
                date_string="26 Jul 2024",
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
            input_ids = input_ids.to(model.device)
            input_len = input_ids.shape[1]
            prompt_text = sample["input"] + sample.get("answer_prefix", "")

            with torch.no_grad():
                if args.mode == "seer_attention":
                    output_ids, _ = model.batch_exist_generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_length=input_len + tokens_to_generate,
                        do_sample=False,
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

            result = {
                "index": sample["index"],
                "pred": pred_text,
                "input": prompt_text,
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
def main():
    args = parse_args()
    start_time = time.time()

    torch.manual_seed(42)

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

        eval_results = {}
        for task in args.tasks:
            if task not in task_configs:
                continue

            predictions = predict_task(
                model, tokenizer, task, task_configs[task],
                data_dir, pred_dir, args,
            )

            if predictions:
                score, nulls = evaluate_task(predictions, task_configs[task])
                eval_results[task] = {"score": score, "nulls": nulls}
                print(f"  {task}: score={score:.2f}, nulls={nulls}")

        if eval_results:
            write_summary_csv(eval_results, pred_dir)
            all_seq_results[seq_len] = {t: r["score"] for t, r in eval_results.items()}

    # Step 4: Consolidated summary
    print(f"\n{'=' * 60}")
    print(f"RULER Results — {args.run_name}")
    print("=" * 60)

    if all_seq_results:
        # Header
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

        # Overall average
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

        # Save consolidated summary JSON
        summary_path = Path(args.output_dir) / args.run_name / "summary.json"
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
            summary["page_size"] = args.page_size
            summary["compress_ratio"] = args.compress_ratio
            summary["scoring_method"] = args.scoring_method
            summary["group_agg_method"] = args.group_agg_method
            summary["unselected_mode"] = args.unselected_mode
        elif args.mode == "seer_attention":
            from seer_attn.config import SEER_ATTN_CONFIG
            summary["seer_attn_config"] = SEER_ATTN_CONFIG
        elif args.mode == "multipole_attention":
            from multipole_attn.config import MULTIPOLE_ATTN_CONFIG
            summary["multipole_attn_config"] = MULTIPOLE_ATTN_CONFIG
        elif args.mode == "quest_attention":
            from quest_attn.config import QUEST_ATTN_CONFIG
            summary["quest_attn_config"] = QUEST_ATTN_CONFIG
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

    elapsed = (time.time() - start_time) / 60
    print(f"Total time: {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
