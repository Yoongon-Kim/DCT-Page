#!/bin/bash
# RoPE Gap Experiment — Sweep gap_size to measure positional discontinuity effect
# Full attention (no compression), only position_ids have artificial gaps.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-120000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-results_longbench}"

NUM_GAPS=17

# ---- Step 1: Baseline (if not already run) ----
if [ ! -f "$OUTPUT_DIR/baseline.jsonl" ]; then
    echo "============================================================"
    echo "BASELINE: Full attention, continuous positions"
    echo "============================================================"
    python eval_longbench.py \
        --mode baseline \
        --base_model "$BASE_MODEL" \
        --max_input_len "$MAX_INPUT_LEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --run_name baseline
else
    echo "Baseline already exists, skipping."
fi

# ---- Step 2: Sweep gap_size ----
for GAP_SIZE in 2400; do
    RUN_NAME="rope_gap_${NUM_GAPS}x${GAP_SIZE}"
    echo ""
    echo "============================================================"
    echo "ROPE GAP: num_gaps=${NUM_GAPS}, gap_size=${GAP_SIZE}"
    echo "  Total max offset: $((NUM_GAPS * GAP_SIZE)) positions"
    echo "============================================================"
    python eval_longbench.py \
        --mode rope_gap \
        --base_model "$BASE_MODEL" \
        --max_input_len "$MAX_INPUT_LEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --num_gaps "$NUM_GAPS" \
        --gap_size "$GAP_SIZE"
done

echo ""
echo "============================================================"
echo "Done. Results in: $OUTPUT_DIR/"
echo "  baseline.jsonl"
for GAP_SIZE in 3648; do
    echo "  rope_gap_${NUM_GAPS}x${GAP_SIZE}.jsonl"
done
echo "============================================================"