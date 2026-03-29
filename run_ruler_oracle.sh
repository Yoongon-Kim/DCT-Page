#!/bin/bash
# RULER Evaluation — Oracle Page Selection
# Sweeps (page_size, top_k) combinations with oracle page scores.
set -e

MODEL="${MODEL:-Qwen/Qwen3-8B}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"

for PS_TK in "32,64" "64,32" "128,16" "32,32" "64,16" "128,8"; do
    IFS=',' read -r PAGE_SIZE TOP_K <<< "$PS_TK"

    RUN_DIR="results_ruler/oracle/ps${PAGE_SIZE}_topk${TOP_K}"

    echo ""
    echo "===================================================================="
    echo "ORACLE: page_size=${PAGE_SIZE}, top_k=${TOP_K}"
    echo "===================================================================="

    python run_ruler_eval.py \
        --mode page_attention \
        --model_name_or_path "$MODEL" \
        --context_len "$CONTEXT_LEN" \
        --num_samples "$NUM_SAMPLES" \
        --dct_select_with_oracle_page_scores \
        --dct_page_size "$PAGE_SIZE" \
        --dct_top_k "$TOP_K" \
        --run_dir "$RUN_DIR"
done

echo ""
echo "===================================================================="
echo "ALL RUNS COMPLETE. Results in: results_ruler/oracle/"
echo "===================================================================="