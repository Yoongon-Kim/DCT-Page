#!/bin/bash
# LongBench v1 Evaluation — Baseline vs DCT Page Attention
# Runs baseline (full attention) then sweeps top_k values.
# Results are saved per-task under results_longbench_v1/<run_name>/.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-120000}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-results_longbench_v1}"

# Optional: specify tasks (space-separated), empty = all 16 English tasks
# Example: TASKS="narrativeqa hotpotqa gov_report" ./run_longbench_v1.sh
# TASKS="${TASKS:-}"
TASKS="${TASKS:-narrativeqa qasper gov_report 2wikimqa multifieldqa_en triviaqa}"

# DCT Page Attention defaults
PAGE_SIZE=128
SINK_SIZE=4
RECENT_SIZE=128
COMPRESS_RATIO=0.03
SCORING_METHOD="max"
GROUP_AGG_METHOD="mean"

# Build task args
TASK_ARGS=""
if [ -n "$TASKS" ]; then
    TASK_ARGS="--tasks $TASKS"
fi

# ---- Step 1: Baseline (full attention, no monkey-patch) ----
echo "============================================================"
echo "BASELINE: Full attention (no monkey-patch)"
echo "============================================================"
python eval_longbench_v1.py \
    --mode baseline \
    --base_model "$BASE_MODEL" \
    --max_input_len "$MAX_INPUT_LEN" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name llama_baseline \
    $TASK_ARGS

# ---- Step 2: Sweep top_k x scoring_method x group_agg_method x mode ----
for TOP_K in 32 8; do #4 8 16 32; do
    for SCORING_METHOD in mean max; do
        for GAM in max mean topp; do
            for MODE in drop compressed; do
                echo ""
                echo "===================================================================="
                echo "PAGE ATTENTION: top_k=${TOP_K}, scoring_method=${SCORING_METHOD}, group_agg=${GAM}, mode=${MODE}"
                echo "===================================================================="
                python eval_longbench_v1.py \
                    --mode page_attention \
                    --base_model "$BASE_MODEL" \
                    --max_input_len "$MAX_INPUT_LEN" \
                    --num_samples "$NUM_SAMPLES" \
                    --output_dir "$OUTPUT_DIR" \
                    --run_name "llama_page_attn_${COMPRESS_RATIO}_topk${TOP_K}_${SCORING_METHOD}_${GAM}_${MODE}_continuous_rope" \
                    --page_size "$PAGE_SIZE" \
                    --top_k "$TOP_K" \
                    --sink_size "$SINK_SIZE" \
                    --recent_size "$RECENT_SIZE" \
                    --compress_ratio "$COMPRESS_RATIO" \
                    --scoring_method "$SCORING_METHOD" \
                    --group_agg_method "$GAM" \
                    --unselected_mode "$MODE" \
                    --continuous_rope \
                    $TASK_ARGS
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
printf "%-26s | %s\n" "Run Name" "Overall Score"
printf "%-26s-|-%s\n" "--------------------------" "-------------"
for d in "$OUTPUT_DIR"/*/; do
    summary="${d}summary.json"
    if [ -f "$summary" ]; then
        name=$(python3 -c "import json; d=json.load(open('$summary')); print(d['run_name'])")
        score=$(python3 -c "import json; d=json.load(open('$summary')); print(f\"{d['overall']:.1f}%\")")
        printf "%-26s | %s\n" "$name" "$score"
    fi
done
echo ""
echo "Detailed results in: $OUTPUT_DIR/"