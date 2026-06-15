#!/bin/bash
# Compare decode speed (dummy inputs): runs Baseline + DCT Page Attention,
# then prints a tok/s table across every run under $OUTPUT_DIR.
set -e

# ---- Configuration ----
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-65536}" #4096,8192,16384,32768,65536,120000}"
NUM_REPEATS="${NUM_REPEATS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-results/speed}"

# Fixed DCT parameters
PAGE_SIZE=128
NUM_SINK_PAGES=1
NUM_RECENT_PAGES=2

COMMON_ARGS="--model $MODEL \
    --context_lengths $CONTEXT_LENGTHS \
    --num_repeats $NUM_REPEATS \
    --output_dir $OUTPUT_DIR"

# ---- Step 1: Baseline ----
echo "============================================================"
echo "BASELINE: full attention (dummy inputs)"
echo "============================================================"
python speed_dummy.py \
    --mode baseline \
    $COMMON_ARGS

# ---- Step 2: DCT configurations ----
for COMPRESS_RATIO in 0.032; do
    for TOP_K in 8; do
        for SCORING in mean; do
            for GAM in max; do
                for UMODE in compressed; do
                    echo ""
                    echo "============================================================"
                    echo "DCT: compress=${COMPRESS_RATIO} top_k=${TOP_K} scoring=${SCORING} gam=${GAM} unselected=${UMODE}"
                    echo "============================================================"
                    python speed_dummy.py \
                        --mode dct \
                        $COMMON_ARGS \
                        --page_size        $PAGE_SIZE \
                        --num_sink_pages   $NUM_SINK_PAGES \
                        --num_recent_pages $NUM_RECENT_PAGES \
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
