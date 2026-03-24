#!/bin/bash
# RULER Evaluation — Baseline vs DCT Page Attention
# Runs baseline (full attention) then sweeps page attention configs.
# Results are saved under results_ruler/<run_name>/.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-llama-3}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ruler}"

# Sequence lengths to evaluate
SEQ_LENGTHS="${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072}"

# DCT Page Attention defaults
PAGE_SIZE=128
SINK_SIZE=4
RECENT_SIZE=128
SCORING_METHOD="mean"
GROUP_AGG_METHOD="max"

# ---- Step 1: Baseline (full attention, no monkey-patch) ----
echo "============================================================"
echo "BASELINE: Full attention (no monkey-patch)"
echo "============================================================"
python eval_ruler.py \
    --mode baseline \
    --base_model "$BASE_MODEL" \
    --prepare --model_template_type "$MODEL_TEMPLATE" \
    --seq_lengths $SEQ_LENGTHS \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name llama_baseline

# ---- Step 2: Sweep compress_ratio x top_k x scoring_method x group_agg_method x mode ----
for COMPRESS_RATIO in 0.03125 0.0625 0.125 0.25; do  # 4/128, 8/128, 16/128, 32/128
    for TOP_K in 4 8 16 32; do
        for SCORING_METHOD in mean max; do
            for GAM in max mean; do
                for MODE in drop compressed; do
                    echo ""
                    echo "===================================================================="
                    echo "PAGE ATTENTION: cr=${COMPRESS_RATIO}, top_k=${TOP_K}, scoring_method=${SCORING_METHOD}, group_agg=${GAM}, mode=${MODE}"
                    echo "===================================================================="
                    python eval_ruler.py \
                        --mode page_attention \
                        --base_model "$BASE_MODEL" \
                        --model_template_type "$MODEL_TEMPLATE" \
                        --seq_lengths $SEQ_LENGTHS \
                        --num_samples "$NUM_SAMPLES" \
                        --output_dir "$OUTPUT_DIR" \
                        --run_name "llama_page_attn_${COMPRESS_RATIO}_topk${TOP_K}_${SCORING_METHOD}_${GAM}_${MODE}" \
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

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"
