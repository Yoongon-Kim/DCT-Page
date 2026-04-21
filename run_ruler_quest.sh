#!/bin/bash
# RULER Evaluation — Quest-style scoring (page_attention + score_use_quest_minmax)
# Runs LLaMA (GPU 2) and Qwen3 (GPU 3) in parallel for 32K and 64K contexts.
# page_size=32, page_budget=64 (top_k=64)

set -e

PYTHON="conda run -n dct python"

OUTPUT_DIR="${OUTPUT_DIR:-results/results_ruler/quest_minmax}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
PAGE_SIZE="${PAGE_SIZE:-32}"
TOP_K="${TOP_K:-64}"
PREPARE_FLAG=""

# ---- Parse CLI flags ----
usage() {
    echo "Usage: $0 [--num_samples N] [--prepare]"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_samples)  NUM_SAMPLES="$2"; shift 2 ;;
        --prepare)      PREPARE_FLAG="--prepare"; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"
QWEN_MODEL="Qwen/Qwen3-8B"

echo "===================================================================="
echo "Quest-minmax RULER: LLaMA (GPU 2) + Qwen3 (GPU 3) | ps=${PAGE_SIZE}, pb=${TOP_K}"
echo "===================================================================="

# LLaMA — GPU 2
CUDA_VISIBLE_DEVICES=2 $PYTHON eval_ruler.py \
    --mode page_attention \
    --base_model "$LLAMA_MODEL" \
    $PREPARE_FLAG \
    --model_template_type llama-3 \
    --tokenizer_family llama \
    --seq_lengths 32768 65536 \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "quest_minmax_llama_ps${PAGE_SIZE}_pb${TOP_K}" \
    --page_size "$PAGE_SIZE" \
    --top_k "$TOP_K" \
    --score_use_quest_minmax \
    --unselected_mode drop \
    --skip_existing \
    &
LLAMA_PID=$!
echo "LLaMA job started (PID=$LLAMA_PID) on GPU 2"

# Qwen3 — GPU 3
CUDA_VISIBLE_DEVICES=3 $PYTHON eval_ruler.py \
    --mode page_attention \
    --base_model "$QWEN_MODEL" \
    $PREPARE_FLAG \
    --model_template_type qwen-3 \
    --tokenizer_family qwen3 \
    --seq_lengths 32768 65536 \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "quest_minmax_qwen3_ps${PAGE_SIZE}_pb${TOP_K}" \
    --page_size "$PAGE_SIZE" \
    --top_k "$TOP_K" \
    --score_use_quest_minmax \
    --unselected_mode drop \
    --skip_existing \
    &
QWEN_PID=$!
echo "Qwen3 job started (PID=$QWEN_PID) on GPU 3"

echo ""
echo "Both jobs running in parallel. Waiting for completion..."
wait $LLAMA_PID && echo "LLaMA done." || echo "LLaMA FAILED (exit $?)."
wait $QWEN_PID  && echo "Qwen3 done." || echo "Qwen3 FAILED (exit $?)."

echo ""
echo "===================================================================="
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "===================================================================="
