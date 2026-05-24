#!/bin/bash
# LongBench v1 — Llama 3.1 8B Instruct, page_attention sweep (drop + compressed)
set -e

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MAX_INPUT_LEN=127500
OUTPUT_DIR="results/longbench_v1"
TASKS="narrativeqa qasper gov_report 2wikimqa multifieldqa_en triviaqa"

PAGE_SIZE=16
TOP_K=128
COMPRESS_RATIO=0.0625
NUM_SINK_PAGES=1
NUM_RECENT_PAGES=9
SCORING_METHOD=max
GROUP_AGG_METHOD=max

for MODE in drop compressed; do
    echo ""
    echo "===================================================================="
    echo "PAGE ATTENTION: mode=${MODE} | ps=${PAGE_SIZE} top_k=${TOP_K} cr=${COMPRESS_RATIO}"
    echo "===================================================================="
    python eval_longbench_v1.py \
        --mode page_attention \
        --base_model "$BASE_MODEL" \
        --max_input_len "$MAX_INPUT_LEN" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "llama31_page_attn_ps${PAGE_SIZE}_topk${TOP_K}_cr${COMPRESS_RATIO}_${SCORING_METHOD}_${GROUP_AGG_METHOD}_${MODE}" \
        --page_size "$PAGE_SIZE" \
        --top_k "$TOP_K" \
        --num_sink_pages "$NUM_SINK_PAGES" \
        --num_recent_pages "$NUM_RECENT_PAGES" \
        --compress_ratio "$COMPRESS_RATIO" \
        --scoring_method "$SCORING_METHOD" \
        --group_agg_method "$GROUP_AGG_METHOD" \
        --unselected_mode "$MODE" \
        --weight_compressed_by_population \
        --tasks $TASKS
done

echo ""
echo "ALL PAGE_ATTENTION RUNS COMPLETE"
