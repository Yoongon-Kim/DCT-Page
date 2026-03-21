"""
DCT-Page RULER prediction script.

Follows the same direct model loading pattern as eval_longbench_v1.py:
monkey-patch -> load model -> tokenize -> model.generate().

Reads prepared data from prepare.py and writes prediction JSONL files
compatible with evaluate.py.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


def read_manifest(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="DCT-Page RULER Prediction")

    # Data
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="synthetic")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--subset", type=str, default="validation")

    # Model
    parser.add_argument("--model_name_or_path", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")

    # DCT-Page parameters
    parser.add_argument("--dct_page_size", type=int, default=128)
    parser.add_argument("--dct_top_k", type=int, default=8)
    parser.add_argument("--dct_sink_size", type=int, default=4)
    parser.add_argument("--dct_recent_size", type=int, default=128)
    parser.add_argument("--dct_compress_ratio", type=float, default=0.25)
    parser.add_argument("--dct_scoring_method", type=str, default="max",
                        choices=["mean", "max", "sum"])
    parser.add_argument("--dct_group_agg_method", type=str, default="mean",
                        choices=["mean", "max", "topp"])
    parser.add_argument("--dct_unselected_mode", type=str, default="drop",
                        choices=["drop", "compressed", "hybrid"])
    parser.add_argument("--dct_no_continuous_rope", action="store_true")
    parser.add_argument("--dct_no_triton", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Resolve task config (tokens_to_generate, etc.)
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(curr_folder))
    module = importlib.import_module(f"data.{args.benchmark}.constants")
    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} not found in {args.benchmark}.yaml")

    config = tasks_customized[args.task]
    config.update(tasks_base[config["task"]])
    tokens_to_generate = config["tokens_to_generate"]

    # Add repo root to sys.path for dct_page_attention imports
    repo_root = os.path.abspath(os.path.join(curr_folder, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Monkey-patch attention BEFORE loading the model (same as eval_longbench_v1.py)
    model_name_lower = args.model_name_or_path.lower()
    if "llama" in model_name_lower:
        from dct_page_attention import replace_llama_attn
        replace_llama_attn(
            page_size=args.dct_page_size,
            top_k=args.dct_top_k,
            sink_size=args.dct_sink_size,
            recent_size=args.dct_recent_size,
            compress_ratio=args.dct_compress_ratio,
            scoring_method=args.dct_scoring_method,
            group_agg_method=args.dct_group_agg_method,
            unselected_mode=args.dct_unselected_mode,
            continuous_rope=not args.dct_no_continuous_rope,
            use_triton=not args.dct_no_triton,
        )
    else:
        from dct_page_attention import replace_qwen2_attn
        replace_qwen2_attn(
            page_size=args.dct_page_size,
            top_k=args.dct_top_k,
            sink_size=args.dct_sink_size,
            recent_size=args.dct_recent_size,
            compress_ratio=args.dct_compress_ratio,
            scoring_method=args.dct_scoring_method,
            group_agg_method=args.dct_group_agg_method,
            unselected_mode=args.dct_unselected_mode,
            continuous_rope=not args.dct_no_continuous_rope,
            use_triton=not args.dct_no_triton,
        )

    # Load model directly (same pattern as eval_longbench_v1.py)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Load prepared data
    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"
    print(f"Reading data from {task_file}")
    data = read_manifest(task_file)

    # Resume support
    pred_file = args.save_dir / f"{args.task}.jsonl"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    if pred_file.exists():
        pred_index = {s["index"] for s in read_manifest(pred_file)}
        data = [s for s in data if s["index"] not in pred_index]
        print(f"Resuming: {len(pred_index)} already done, {len(data)} remaining")

    print(f"Generating predictions for {args.task} ({len(data)} samples, "
          f"max_new_tokens={tokens_to_generate})")

    with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
        for sample in tqdm(data, desc=args.task):
            prompt = sample["input"]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=tokens_to_generate,
                    do_sample=False,
                    use_cache=True,
                )

            generated_ids = output_ids[0, input_ids.shape[1]:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

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

    print(f"Predictions saved to {pred_file}")
    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    main()
