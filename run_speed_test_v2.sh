#!/bin/bash
# Decode speed benchmark on LongBench v2 — Baseline vs DCT Page Attention
# Set MIN_CONTEXT_LEN to target the context-length regime you care about:
#   ~32000  -> medium bucket (32K–128K)
#   ~128000 -> long bucket   (128K+)
set -e

# ---- Configuration ----
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-120000}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
MIN_CONTEXT_LEN="${MIN_CONTEXT_LEN:-128000}"
OUTPUT_DIR="${OUTPUT_DIR:-results_speed_test_v2}"

# Fixed DCT parameters
PAGE_SIZE=128
SINK_SIZE=4
RECENT_SIZE=128

COMMON_ARGS="--model $MODEL \
    --max_input_len $MAX_INPUT_LEN \
    --num_samples $NUM_SAMPLES \
    --min_context_len $MIN_CONTEXT_LEN \
    --output_dir $OUTPUT_DIR"

# ---- Step 1: Baseline ----
echo "============================================================"
echo "BASELINE: full attention"
echo "============================================================"
python speed_test_v2.py \
    --mode baseline \
    $COMMON_ARGS

# ---- Step 2: DCT configurations ----
for COMPRESS_RATIO in 0.03; do
    for TOP_K in 8 32; do
        for SCORING in max; do
            for GAM in mean; do
                for UMODE in drop; do
                    echo ""
                    echo "============================================================"
                    echo "DCT: compress=${COMPRESS_RATIO} top_k=${TOP_K} scoring=${SCORING} gam=${GAM} unselected=${UMODE}"
                    echo "============================================================"
                    python speed_test_v2.py \
                        --mode dct \
                        $COMMON_ARGS \
                        --page_size        $PAGE_SIZE \
                        --sink_size        $SINK_SIZE \
                        --recent_size      $RECENT_SIZE \
                        --compress_ratio   $COMPRESS_RATIO \
                        --top_k            $TOP_K \
                        --scoring_method   $SCORING \
                        --group_agg_method $GAM \
                        --unselected_mode  $UMODE
                done
            done
        done
    done
done

# ---- Step 3: Comparison table ----
echo ""
echo "============================================================"
echo "DECODE SPEED COMPARISON  (LongBench v2)"
echo "============================================================"
printf "%-60s | %s\n" "Run Name" "tok/s"
printf "%-60s-|-%s\n" "------------------------------------------------------------" "-----"
for d in "$OUTPUT_DIR"/*/; do
    summary="${d}summary.json"
    if [ -f "$summary" ]; then
        run=$(basename "$d")
        tok_s=$(python3 -c "import json; d=json.load(open('$summary')); v=d.get('overall_decode_tok_per_s'); print(f'{v:.1f}' if v else 'N/A')")
        printf "%-60s | %s tok/s\n" "$run" "$tok_s"
    fi
done
echo ""
echo "Detailed results in: $OUTPUT_DIR/"