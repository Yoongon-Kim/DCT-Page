#!/bin/bash
# LongBench v2 Evaluation — Baseline vs DCT Page Attention
# Runs baseline (full attention) then sweeps top_k values.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-120000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-results_longbench}"

# DCT Page Attention defaults
PAGE_SIZE=128
SINK_SIZE=4
RECENT_SIZE=128
COMPRESS_RATIO=0.03
SCORING_METHOD="mean"
GROUP_AGG_METHOD="mean"
MODE="drop"

# ---- Step 1: Baseline (full attention, no monkey-patch) ----
echo "============================================================"
echo "BASELINE: Full attention (no monkey-patch)"
echo "============================================================"
python eval_longbench.py \
    --mode baseline \
    --base_model "$BASE_MODEL" \
    --max_input_len "$MAX_INPUT_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name baseline

# ---- Step 2: Sweep top_k x group_agg_method (page attention, drop mode) ----
for TOP_K in 8; do # 4 8 16 32; do
    for SCORING_METHOD in mean max; do
        for GAM in max mean topp; do
            for MODE in drop compressed; do
                echo ""
                echo "===================================================================="
                echo "PAGE ATTENTION : top_k=${TOP_K}, scoring_method=${SCORING_METHOD}, group_agg=${GAM}"
                echo "===================================================================="
                python eval_longbench.py \
                    --mode page_attention \
                    --base_model "$BASE_MODEL" \
                    --max_input_len "$MAX_INPUT_LEN" \
                    --max_new_tokens "$MAX_NEW_TOKENS" \
                    --num_samples "$NUM_SAMPLES" \
                    --output_dir "$OUTPUT_DIR" \
                    --run_name "page_attn_${COMPRESS_RATIO}_topk${TOP_K}_${SCORING_METHOD}_${GAM}_${MODE}_continuous_rope" \
                    --page_size "$PAGE_SIZE" \
                    --top_k "$TOP_K" \
                    --sink_size "$SINK_SIZE" \
                    --recent_size "$RECENT_SIZE" \
                    --compress_ratio "$COMPRESS_RATIO" \
                    --scoring_method "$SCORING_METHOD" \
                    --group_agg_method "$GAM" \
                    --unselected_mode "$MODE" \
                    --continuous_rope
            done
        done
    done
done

# ---- Step 3: Comparison table ----
echo ""
echo "============================================================"
echo "COMPARISON SUMMARY"
echo "============================================================"
echo ""
printf "%-26s | %s\n" "Run Name" "Overall Acc"
printf "%-26s-|-%s\n" "--------------------------" "-----------"
for f in "$OUTPUT_DIR"/*_summary.json; do
    if [ -f "$f" ]; then
        name=$(python3 -c "import json; d=json.load(open('$f')); print(d['run_name'])")
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['overall_accuracy']:.1f}%\")")
        printf "%-26s | %s\n" "$name" "$acc"
    fi
done
echo ""
echo "Detailed results in: $OUTPUT_DIR/"
