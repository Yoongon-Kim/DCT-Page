#!/bin/bash
# RULER Evaluation — DCT Page Attention (DCT-lowpass-IDCT scoring proxy)
# Sweeps (page_size, top_k) pairs with compress_ratio 1/32 and 4/32.
set -e

# ---- Configuration (env defaults, overridable via CLI flags below) ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results/results_ruler/page_attention}"
PREPARE_FLAG=""

# ---- Parse CLI flags ----
usage() {
    echo "Usage: $0 [--base_model MODEL] [--num_samples N] [--prepare]"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model)   BASE_MODEL="$2"; shift 2 ;;
        --num_samples)  NUM_SAMPLES="$2"; shift 2 ;;
        --prepare)      PREPARE_FLAG="--prepare"; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Derive a short model tag from BASE_MODEL (used in run names).
# Only Llama 3.x and Qwen3 are supported — eval_ruler.py enforces this.
case "$(echo "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')" in
    *llama*)  MODEL_TAG="llama" ;;
    *qwen3*)  MODEL_TAG="qwen" ;;
    *) echo "Unsupported BASE_MODEL: $BASE_MODEL (only Llama 3.x / Qwen3)"; exit 1 ;;
esac

# Sequence lengths to evaluate
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}" #"${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072}"

# Tasks to evaluate (space-separated). Leave empty to use eval_ruler.py default (all tasks).
# Example: TASKS="niah_single_1 niah_multikey_1 qa_1"
TASKS="${TASKS:-}"
TASKS_FLAG=""
if [[ -n "$TASKS" ]]; then
    TASKS_FLAG="--tasks $TASKS"
fi

# Fixed params
SINK_SIZE=4
RECENT_SIZE=128
SCORING_METHOD="max"
GROUP_AGG_METHOD="max"
# ---- Sweep (page_size, top_k) x compress_ratio x mode x compressed_token_rope x weight_compressed_by_population ----
for PS_TK in "16,128" "16,64" "32,64" "32,32"; do
    IFS=',' read -r PAGE_SIZE TOP_K <<< "$PS_TK"
    for COMPRESS_RATIO in 0.125; do
      for MODE in drop compressed; do
        for COMP_TOKEN_ROPE in mixed; do
          for WEIGHT_POP in 1; do
            if [[ "$WEIGHT_POP" == "1" ]]; then
                WEIGHT_POP_FLAG="--weight_compressed_by_population"
                WEIGHT_POP_TAG="popw"
            else
                WEIGHT_POP_FLAG=""
                WEIGHT_POP_TAG="nopopw"
            fi
            RUN_NAME="${MODEL_TAG}_ps${PAGE_SIZE}_topk${TOP_K}_cr${COMPRESS_RATIO}_${MODE}_tokenrope${COMP_TOKEN_ROPE}_${WEIGHT_POP_TAG}"

            echo ""
            echo "===================================================================="
            echo "PAGE ATTENTION: ps=${PAGE_SIZE}, top_k=${TOP_K}, cr=${COMPRESS_RATIO}, mode=${MODE}, token_rope=${COMP_TOKEN_ROPE}, weight_pop=${WEIGHT_POP}"
            echo "===================================================================="
            python eval_ruler.py \
                --mode page_attention \
                --base_model "$BASE_MODEL" \
                $PREPARE_FLAG \
                --seq_lengths $SEQ_LENGTHS \
                --num_samples "$NUM_SAMPLES" \
                --output_dir "$OUTPUT_DIR" \
                --run_name "$RUN_NAME" \
                --page_size "$PAGE_SIZE" \
                --top_k "$TOP_K" \
                --sink_size "$SINK_SIZE" \
                --recent_size "$RECENT_SIZE" \
                --compress_ratio "$COMPRESS_RATIO" \
                --scoring_method "$SCORING_METHOD" \
                --group_agg_method "$GROUP_AGG_METHOD" \
                --compressed_token_rope "$COMP_TOKEN_ROPE" \
                $WEIGHT_POP_FLAG \
                $TASKS_FLAG \
                --skip_existing \
                --unselected_mode "$MODE"
          done
        done
      done
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"