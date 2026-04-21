#!/bin/bash
# LongBench v1 Evaluation — Baseline vs DCT Page Attention
# Runs baseline (full attention) then sweeps top_k values.
# Results are saved per-task under results/longbench_v1/<run_name>/.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-127500}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-results_longbench_v1/page_attention}"

# Optional: specify tasks (space-separated), empty = all 16 English tasks
# Example: TASKS="narrativeqa hotpotqa gov_report" ./run_longbench_v1.sh
# TASKS="${TASKS:-}"
TASKS="${TASKS:-narrativeqa qasper gov_report 2wikimqa multifieldqa_en triviaqa}"

# DCT Page Attention defaults
SINK_SIZE=4
RECENT_SIZE=128
SCORING_METHOD="mean"
GROUP_AGG_METHOD="max"

# Build task args
TASK_ARGS=""
if [ -n "$TASKS" ]; then
    TASK_ARGS="--tasks $TASKS"
fi

# ---- Step 1: Baseline (full attention, no monkey-patch) ----
# echo "============================================================"
# echo "BASELINE: Full attention (no monkey-patch)"
# echo "============================================================"
# python eval_longbench_v1.py \
#     --mode baseline \
#     --base_model "$BASE_MODEL" \
#     --max_input_len "$MAX_INPUT_LEN" \
#     --num_samples "$NUM_SAMPLES" \
#     --output_dir "$OUTPUT_DIR" \
#     --run_name qwen3_baseline \
#     $TASK_ARGS

# ---- Step 2: Sweep compress_ratio x top_k x scoring_method x group_agg_method x mode ----
for PS_TK in "32,64" "32,32" "16,124" "16,64"; do
    IFS=',' read -r PAGE_SIZE TOP_K <<< "$PS_TK"
    for COMPRESS_RATIO in 0.125; do  # 4/128, 8/128, 16/128, 32/128
        for SCORING_METHOD in mean max; do
            for GAM in max mean; do
                for MODE in drop compressed; do
                    echo ""
                    echo "===================================================================="
                    echo "PAGE ATTENTION: ps=${PAGE_SIZE}, top_k=${TOP_K}, cr=${COMPRESS_RATIO}, scoring_method=${SCORING_METHOD}, group_agg=${GAM}, mode=${MODE}"
                    echo "===================================================================="
                    python eval_longbench_v1.py \
                        --mode page_attention \
                        --base_model "$BASE_MODEL" \
                        --max_input_len "$MAX_INPUT_LEN" \
                        --num_samples "$NUM_SAMPLES" \
                        --output_dir "$OUTPUT_DIR" \
                        --run_name "qwen3_page_attn_ps${PAGE_SIZE}_topk${TOP_K}_cr${COMPRESS_RATIO}_${SCORING_METHOD}_${GAM}_${MODE}" \
                        --page_size "$PAGE_SIZE" \
                        --top_k "$TOP_K" \
                        --sink_size "$SINK_SIZE" \
                        --recent_size "$RECENT_SIZE" \
                        --compress_ratio "$COMPRESS_RATIO" \
                        --scoring_method "$SCORING_METHOD" \
                        --group_agg_method "$GAM" \
                        --unselected_mode "$MODE" \
                        $TASK_ARGS
                done
            done
        done
    done
done

# ---- Step 3: Summarize all results ----
echo ""
echo "============================================================"
echo "SUMMARIZING ALL RESULTS"
echo "============================================================"
python3 summarize_longbench_v1.py "$OUTPUT_DIR"
