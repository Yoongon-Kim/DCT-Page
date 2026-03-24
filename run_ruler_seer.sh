#!/bin/bash
# RULER Evaluation — SeerAttention-R
# Sweeps token_budget values by rewriting seer_attn/config.py.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-qwen-3}"
TOKENIZER_FAMILY="${TOKENIZER_FAMILY:-qwen3}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ruler/seer_attention}"

# Sequence lengths to evaluate
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}" # "${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072}"

# Pass --prepare to also prepare data (skips if already exists)
PREPARE_FLAG=""
if [[ "$*" == *"--prepare"* ]]; then
    PREPARE_FLAG="--prepare"
fi

# Fixed seer parameters
SEER_MODEL="${SEER_MODEL:-SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates}"
START_LAYER=0

CONFIG_FILE="seer_attn/config.py"

write_config() {
    local seer_model="$1"
    local sparsity_method="$2"
    local token_budget="$3"
    local threshold="$4"
    local start_layer="$5"
    cat > "$CONFIG_FILE" <<PYEOF
"""
SeerAttention-R evaluation configuration.

Edit this file to change model checkpoint, sparsity method, and budget/threshold
before running eval_ruler.py with --mode seer_attention.

Available HF checkpoints (SeerAttention-R, decode sparse only):
  - SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates
  - SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates
  - SeerAttention/SeerAttention-Decode-Qwen3-14B-AttnGates
  - SeerAttention/SeerAttention-Decode-DeepSeek-R1-Distill-Qwen-... (Qwen2-based)

Sparsity methods:
  - "token_budget": keep top-k tokens per decode step (controlled by token_budget)
  - "threshold":    keep blocks where gate score > threshold
"""

SEER_ATTN_CONFIG = {
    # HF Hub ID or local path to SeerAttention-R checkpoint
    "seer_model": "${seer_model}",

    # Sparsity method: "token_budget" or "threshold"
    "sparsity_method": "${sparsity_method}",

    # Max active tokens per decode step (only used when sparsity_method="token_budget")
    "token_budget": ${token_budget},

    # Gate score threshold (only used when sparsity_method="threshold")
    "threshold": ${threshold},

    # First layer to apply sparse attention (layers below use dense); 0 = all layers
    "start_layer": ${start_layer},
}
PYEOF
}

# ---- Sweep token_budget ----
for TOKEN_BUDGET in 1156 2180; do
    RUN_NAME="seer_budget${TOKEN_BUDGET}"

    echo ""
    echo "===================================================================="
    echo "SEER ATTENTION: sparsity_method=token_budget, token_budget=${TOKEN_BUDGET}"
    echo "===================================================================="

    write_config "$SEER_MODEL" "token_budget" "$TOKEN_BUDGET" "0.0" "$START_LAYER"

    python eval_ruler.py \
        --mode seer_attention \
        --base_model "$BASE_MODEL" \
        $PREPARE_FLAG --model_template_type "$MODEL_TEMPLATE" --tokenizer_family "$TOKENIZER_FAMILY" \
        --seq_lengths $SEQ_LENGTHS \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME"
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"
