#!/bin/bash
# LongBench v2 Evaluation — SeerAttention-R
# Sweeps token_budget values by rewriting seer_attn/config.py.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-127500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-results_longbench_v2/seer_attention}"

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
before running eval_longbench_v1.py or eval_longbench_v2.py with --mode seer_attention.

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
for TOKEN_BUDGET in 1024 2048 4096 8192; do
    RUN_NAME="seer_budget${TOKEN_BUDGET}"

    echo ""
    echo "===================================================================="
    echo "SEER ATTENTION: sparsity_method=token_budget, token_budget=${TOKEN_BUDGET}"
    echo "===================================================================="

    write_config "$SEER_MODEL" "token_budget" "$TOKEN_BUDGET" "0.0" "$START_LAYER"

    python eval_longbench_v2.py \
        --mode seer_attention \
        --base_model "$BASE_MODEL" \
        --max_input_len "$MAX_INPUT_LEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME"
done

# ---- Summarize all results ----
echo ""
echo "============================================================"
echo "SUMMARIZING ALL RESULTS"
echo "============================================================"
python3 summarize_longbench_v2.py "$OUTPUT_DIR"
