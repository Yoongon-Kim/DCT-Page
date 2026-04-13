#!/bin/bash
# RULER Evaluation — DCT Page Attention (Haar scoring proxy)
# Sweeps (page_size, top_k) pairs with compress_ratio 1/32 and 4/32.
# Haar proxy is the default scoring method — no extra flags needed.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-llama-3}"
TOKENIZER_FAMILY="${TOKENIZER_FAMILY:-llama}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results/results_ruler/proxy_max}"

# Sequence lengths to evaluate
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}" #"${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072}"

# Tasks to evaluate (space-separated). Leave empty to use eval_ruler.py default (all tasks).
# Example: TASKS="niah_single_1 niah_multikey_1 qa_1"
TASKS="${TASKS:-}"
TASKS_FLAG=""
if [[ -n "$TASKS" ]]; then
    TASKS_FLAG="--tasks $TASKS"
fi

# Pass --prepare to also prepare data (skips if already exists)
PREPARE_FLAG=""
if [[ "$*" == *"--prepare"* ]]; then
    PREPARE_FLAG="--prepare"
fi

# Fixed params
SINK_SIZE=4
RECENT_SIZE=128
SCORING_METHOD="max"
GROUP_AGG_METHOD="max"
# ---- Sweep (page_size, top_k) x compress_ratio x mode x compression_method x compressed_token_rope x weight_compressed_by_population ----
for PS_TK in "16,188" "16,248" "32,93" "32,124"; do
    IFS=',' read -r PAGE_SIZE TOP_K <<< "$PS_TK"
    for COMPRESS_RATIO in 0.0625; do
      for MODE in drop; do
        for COMP_METHOD in dct; do
          for COMP_TOKEN_ROPE in mixed; do
            for WEIGHT_POP in 1; do
              if [[ "$WEIGHT_POP" == "1" ]]; then
                  WEIGHT_POP_FLAG="--weight_compressed_by_population"
                  WEIGHT_POP_TAG="popw"
              else
                  WEIGHT_POP_FLAG=""
                  WEIGHT_POP_TAG="nopopw"
              fi
              RUN_NAME="${TOKENIZER_FAMILY}_ps${PAGE_SIZE}_topk${TOP_K}_cr${COMPRESS_RATIO}_${MODE}_${COMP_METHOD}_tokenrope${COMP_TOKEN_ROPE}_${WEIGHT_POP_TAG}"

              echo ""
              echo "===================================================================="
              echo "PAGE ATTENTION: ps=${PAGE_SIZE}, top_k=${TOP_K}, cr=${COMPRESS_RATIO}, mode=${MODE}, comp=${COMP_METHOD}, token_rope=${COMP_TOKEN_ROPE}, weight_pop=${WEIGHT_POP}"
              echo "===================================================================="
              python eval_ruler.py \
                  --mode page_attention \
                  --base_model "$BASE_MODEL" \
                  $PREPARE_FLAG --model_template_type "$MODEL_TEMPLATE" --tokenizer_family "$TOKENIZER_FAMILY" \
                  --seq_lengths $SEQ_LENGTHS \
                  --num_samples "$NUM_SAMPLES" \
                  --output_dir "$OUTPUT_DIR/$COMP_METHOD" \
                  --run_name "$RUN_NAME" \
                  --page_size "$PAGE_SIZE" \
                  --top_k "$TOP_K" \
                  --sink_size "$SINK_SIZE" \
                  --recent_size "$RECENT_SIZE" \
                  --compress_ratio "$COMPRESS_RATIO" \
                  --scoring_method "$SCORING_METHOD" \
                  --group_agg_method "$GROUP_AGG_METHOD" \
                  --compression_method "$COMP_METHOD" \
                  --compressed_token_rope "$COMP_TOKEN_ROPE" \
                  $WEIGHT_POP_FLAG \
                  $TASKS_FLAG \
                  --skip_existing \
                  --unselected_mode "$MODE"
            done
          done
        done
      done
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"