#!/bin/bash
# RULER Evaluation — DCT Page Attention (DCT-lowpass-IDCT scoring proxy)
# Sweeps (page_size, top_k) pairs with compress_ratio 1/32 and 4/32.
set -e

# ---- Configuration (env defaults, overridable via CLI flags below) ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results/results_ruler/page_attention/VRAM}"
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
MODE="drop"
COMP_TOKEN_ROPE="mixed"
# ---- Sweep (page_size, top_k) x (compress_ratio, comp_kv_quant) ----
for PS_TK in "16,64" "32,32"; do
    IFS=',' read -r PAGE_SIZE TOP_K <<< "$PS_TK"
    for CR_QUANT in "0.0625,none" "0.125,fp8_e4m3" "0.125,fp8_e5m2" "0.25,int4"; do
        IFS=',' read -r COMPRESS_RATIO COMP_KV_QUANT <<< "$CR_QUANT"
        RUN_NAME="${MODEL_TAG}_${MODE}_ps${PAGE_SIZE}_topk${TOP_K}_cr${COMPRESS_RATIO}_${COMP_KV_QUANT}"

        echo ""
        echo "===================================================================="
        echo "PAGE ATTENTION: ps=${PAGE_SIZE}, top_k=${TOP_K}, cr=${COMPRESS_RATIO}, quant=${COMP_KV_QUANT}, mode=${MODE}, token_rope=${COMP_TOKEN_ROPE}"
        echo "===================================================================="
        python eval_ruler.py \
            --mode page_attention \
            --base_model "$BASE_MODEL" \
            $PREPARE_FLAG \
            --seq_lengths $SEQ_LENGTHS \
            $TASKS_FLAG \
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
            --unselected_mode "$MODE"\
            --comp_kv_quant "$COMP_KV_QUANT" \
            --skip_existing
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"