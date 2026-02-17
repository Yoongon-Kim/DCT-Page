#!/bin/bash
# DCT Page Attention - Evaluation Script
# Runs streaming perplexity evaluation with different configurations.

set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B}"
DATA_PATH="${DATA_PATH:-/path/to/tokenized/data.bin}"
SEQ_LEN="${SEQ_LEN:-8192}"
PREFILL_LEN="${PREFILL_LEN:-512}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SLIDING_WINDOW="${SLIDING_WINDOW:-256}"

# DCT Page Attention defaults
PAGE_SIZE=128
SINK_SIZE=4
RECENT_SIZE=128
COMPRESS_RATIO=0.25
SCORING_METHOD="max"

# ---- Baseline: Full attention (no page selection) ----
echo "============================================================"
echo "Baseline: Full attention (top_k large enough to select all)"
echo "============================================================"
python eval.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --seq_len "$SEQ_LEN" \
    --prefill_len "$PREFILL_LEN" \
    --num_samples "$NUM_SAMPLES" \
    --sliding_window "$SLIDING_WINDOW" \
    --page_size "$PAGE_SIZE" \
    --top_k 9999 \
    --sink_size "$SINK_SIZE" \
    --recent_size "$RECENT_SIZE" \
    --compress_ratio "$COMPRESS_RATIO" \
    --scoring_method "$SCORING_METHOD" \
    --unselected_mode drop

# ---- Sweep top_k values (drop mode) ----
for TOP_K in 4 8 16 32; do
    echo ""
    echo "============================================================"
    echo "Drop mode: top_k=${TOP_K}"
    echo "============================================================"
    python eval.py \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --seq_len "$SEQ_LEN" \
        --prefill_len "$PREFILL_LEN" \
        --num_samples "$NUM_SAMPLES" \
        --sliding_window "$SLIDING_WINDOW" \
        --page_size "$PAGE_SIZE" \
        --top_k "$TOP_K" \
        --sink_size "$SINK_SIZE" \
        --recent_size "$RECENT_SIZE" \
        --compress_ratio "$COMPRESS_RATIO" \
        --scoring_method "$SCORING_METHOD" \
        --unselected_mode drop
done

# ---- Sweep top_k values (compressed mode) ----
for TOP_K in 4 8 16; do
    echo ""
    echo "============================================================"
    echo "Compressed mode: top_k=${TOP_K}"
    echo "============================================================"
    python eval.py \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --seq_len "$SEQ_LEN" \
        --prefill_len "$PREFILL_LEN" \
        --num_samples "$NUM_SAMPLES" \
        --sliding_window "$SLIDING_WINDOW" \
        --page_size "$PAGE_SIZE" \
        --top_k "$TOP_K" \
        --sink_size "$SINK_SIZE" \
        --recent_size "$RECENT_SIZE" \
        --compress_ratio "$COMPRESS_RATIO" \
        --scoring_method "$SCORING_METHOD" \
        --unselected_mode compressed
done
