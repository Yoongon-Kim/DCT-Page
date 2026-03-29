#!/bin/bash
# LongBench v2 Evaluation — Baseline vs DCT Page Attention
# Runs baseline (full attention) then sweeps top_k values.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-127500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/longbench_v2}"

# DCT Page Attention defaults
PAGE_SIZE=128
SINK_SIZE=4
RECENT_SIZE=128
SCORING_METHOD="mean"
GROUP_AGG_METHOD="mean"
MODE="drop"

# ---- Step 1: Baseline (full attention, no monkey-patch) ----
echo "============================================================"
echo "BASELINE: Full attention (no monkey-patch)"
echo "============================================================"
python eval_longbench_v2.py \
    --mode baseline \
    --base_model "$BASE_MODEL" \
    --max_input_len "$MAX_INPUT_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name qwen3_baseline

# ---- Step 2: Sweep compress_ratio x top_k x group_agg_method (page attention) ----
for COMPRESS_RATIO in 0.03125 0.0625 0.125 0.25; do  # 4/128, 8/128, 16/128, 32/128
    for TOP_K in 4 8 16 32; do
        for SCORING_METHOD in mean max; do
            for GAM in max mean; do
                for MODE in drop compressed; do
                    echo ""
                    echo "===================================================================="
                    echo "PAGE ATTENTION : cr=${COMPRESS_RATIO}, top_k=${TOP_K}, scoring_method=${SCORING_METHOD}, group_agg=${GAM}, mode=${MODE}"
                    echo "===================================================================="
                    python eval_longbench_v2.py \
                        --mode page_attention \
                        --base_model "$BASE_MODEL" \
                        --max_input_len "$MAX_INPUT_LEN" \
                        --max_new_tokens "$MAX_NEW_TOKENS" \
                        --num_samples "$NUM_SAMPLES" \
                        --output_dir "$OUTPUT_DIR" \
                        --run_name "qwen3_page_attn_${COMPRESS_RATIO}_topk${TOP_K}_${SCORING_METHOD}_${GAM}_${MODE}" \
                        --page_size "$PAGE_SIZE" \
                        --top_k "$TOP_K" \
                        --sink_size "$SINK_SIZE" \
                        --recent_size "$RECENT_SIZE" \
                        --compress_ratio "$COMPRESS_RATIO" \
                        --scoring_method "$SCORING_METHOD" \
                        --group_agg_method "$GAM" \
                        --unselected_mode "$MODE"
                done
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
