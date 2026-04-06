#!/bin/bash
# RULER Evaluation — DCT Page Attention (Haar scoring proxy)
# Sweeps (page_size, top_k) pairs with compress_ratio 1/32 and 4/32.
# Haar proxy is the default scoring method — no extra flags needed.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-qwen-3}"
TOKENIZER_FAMILY="${TOKENIZER_FAMILY:-qwen3}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ruler/haar_page_attention}"

# Sequence lengths to evaluate
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}" #"${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072}"

# Pass --prepare to also prepare data (skips if already exists)
PREPARE_FLAG=""
if [[ "$*" == *"--prepare"* ]]; then
    PREPARE_FLAG="--prepare"
fi

# Fixed params
SINK_SIZE=4
RECENT_SIZE=128
SCORING_METHOD="max"
GROUP_AGG_METHOD="mean"
# ---- Sweep (page_size, top_k) x compress_ratio x mode x compression_method ----
for PS_TK in "32,64" "64,32" "32,32" "64,16"; do
    IFS=',' read -r PAGE_SIZE TOP_K <<< "$PS_TK"
    for COMPRESS_RATIO in 0.125; do
      for MODE in compressed drop; do
        for COMP_METHOD in haar dct; do
            RUN_NAME="${TOKENIZER_FAMILY}_ps${PAGE_SIZE}_topk${TOP_K}_cr${COMPRESS_RATIO}_${MODE}_${COMP_METHOD}"

            echo ""
            echo "===================================================================="
            echo "PAGE ATTENTION: ps=${PAGE_SIZE}, top_k=${TOP_K}, cr=${COMPRESS_RATIO}, mode=${MODE}, comp=${COMP_METHOD}"
            echo "===================================================================="
            python eval_ruler.py \
                --mode page_attention \
                --base_model "$BASE_MODEL" \
                $PREPARE_FLAG --model_template_type "$MODEL_TEMPLATE" --tokenizer_family "$TOKENIZER_FAMILY" \
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
                --compression_method "$COMP_METHOD" \
                --skip_existing \
                --unselected_mode "$MODE"
        done
      done
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"