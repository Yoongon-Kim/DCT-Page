#!/bin/bash
# Decode speed benchmark with dummy inputs — Baseline vs DCT Page Attention
set -e

# ---- Configuration ----
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-120000}" #4096,8192,16384,32768,65536,120000}"
NUM_REPEATS="${NUM_REPEATS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-results_speed_test_dummy}"

# Fixed DCT parameters
PAGE_SIZE=128
SINK_SIZE=4
RECENT_SIZE=128

COMMON_ARGS="--model $MODEL \
    --context_lengths $CONTEXT_LENGTHS \
    --num_repeats $NUM_REPEATS \
    --output_dir $OUTPUT_DIR"

# ---- Step 1: Baseline ----
echo "============================================================"
echo "BASELINE: full attention (dummy inputs)"
echo "============================================================"
python speed_test_dummy.py \
    --mode baseline \
    $COMMON_ARGS

# ---- Step 2: DCT configurations ----
for COMPRESS_RATIO in 0.03; do
    for TOP_K in 8; do
        for SCORING in mean; do
            for GAM in max; do
                for UMODE in compressed; do
                    for CROPE in continuous; do
                        ROPE_FLAG=""
                        if [ "$CROPE" = "continuous" ]; then
                            ROPE_FLAG="--continuous_rope"
                        fi
                        for TRITON in triton notriton; do
                            TRITON_FLAG=""
                            if [ "$TRITON" = "notriton" ]; then
                                TRITON_FLAG="--no_triton"
                            fi
                            echo ""
                            echo "============================================================"
                            echo "DCT: compress=${COMPRESS_RATIO} top_k=${TOP_K} scoring=${SCORING} gam=${GAM} unselected=${UMODE} rope=${CROPE} ${TRITON}"
                            echo "============================================================"
                            python speed_test_dummy.py \
                                --mode dct \
                                $COMMON_ARGS \
                                --page_size        $PAGE_SIZE \
                                --sink_size        $SINK_SIZE \
                                --recent_size      $RECENT_SIZE \
                                --compress_ratio   $COMPRESS_RATIO \
                                --top_k            $TOP_K \
                                --scoring_method   $SCORING \
                                --group_agg_method $GAM \
                                --unselected_mode  $UMODE \
                                $ROPE_FLAG \
                                $TRITON_FLAG
                        done
                    done
                done
            done
        done
    done
done

# ---- Step 3: Comparison table ----
echo ""
echo "============================================================"
echo "DECODE SPEED COMPARISON  (dummy inputs)"
echo "============================================================"
printf "%-60s | %s\n" "Run Name" "tok/s"
printf "%-60s-|-%s\n" "------------------------------------------------------------" "-----"
for d in "$OUTPUT_DIR"/*/; do
    summary="${d}summary.json"
    if [ -f "$summary" ]; then
        run=$(basename "$d")
        tok_s=$(python3 -c "import json; d=json.load(open('$summary')); v=d.get('decode_tok_per_s'); print(f'{v:.1f}' if v else 'N/A')")
        printf "%-60s | %s tok/s\n" "$run" "$tok_s"
    fi
done
echo ""
echo "Detailed results in: $OUTPUT_DIR/"
