#!/bin/bash
# RULER Evaluation — Multipole Attention
# Sweeps percent_clusters and percentiles values by rewriting multipole_attn/config.py.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-qwen3}"
TOKENIZER_FAMILY="${TOKENIZER_FAMILY:-qwen3}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ruler/multipole_attention/${MODEL_FAMILY}}"

# Sequence lengths to evaluate
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}" #"${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072}"

# Pass --prepare to also prepare data (skips if already exists)
PREPARE_FLAG=""
if [[ "$*" == *"--prepare"* ]]; then
    PREPARE_FLAG="--prepare"
fi

# Fixed multipole parameters
USE_CENTROIDS=True
USE_REPLACEMENT=False
CLUSTER_INTERVAL=128
INFERENCE_TP=1

CONFIG_FILE="multipole_attn/config.py"

write_config() {
    local pct_clusters="$1"
    local percentiles="$2"
    local use_replacement="$3"
    local cluster_interval="$4"
    cat > "$CONFIG_FILE" <<PYEOF
"""
Multipole Attention evaluation configuration.

Edit this file to change model, clustering parameters, and attention behavior
before running eval_ruler.py with --mode multipole_attention.

Key parameters:
  - percent_clusters_lst: percentage of keys to retain per hierarchy level
  - percentiles_lst:      importance threshold (token budget) per level
  - use_replacement:      if True, use centroid value approximation for non-selected tokens
  - cluster_interval:     number of new tokens between re-clustering during generation
"""

MULTIPOLE_ATTN_CONFIG = {
    # Base model (HF Hub ID or local path)
    "base_model": "${BASE_MODEL}",

    # Enable centroid-based sparse attention
    "use_centroids": True,

    # Percentage of keys to retain per hierarchy level (list, one per level)
    "percent_clusters_lst": [${pct_clusters}],

    # Importance threshold per level (token budget for centroid selection)
    "percentiles_lst": [${percentiles}],

    # Use centroid value approximation for non-selected tokens
    "use_replacement": ${use_replacement},

    # Number of new tokens between re-clustering during generation
    "cluster_interval": ${cluster_interval},

    # Tensor parallelism degree (1 = single GPU)
    "inference_tp": ${INFERENCE_TP},
}
PYEOF
}

# ---- Sweep percent_clusters x percentiles x use_replacement ----
for PCT_CLUSTERS in 6.25; do
    for PERCENTILES in 1156 2180; do
        for REPL in False; do
            RUN_NAME="${MODEL_FAMILY}_multipole_pct${PCT_CLUSTERS}_ptl${PERCENTILES}_repl${REPL}"

            echo ""
            echo "===================================================================="
            echo "MULTIPOLE: pct_clusters=${PCT_CLUSTERS}, percentiles=${PERCENTILES}, use_replacement=${REPL}"
            echo "===================================================================="

            write_config "$PCT_CLUSTERS" "$PERCENTILES" "$REPL" "$CLUSTER_INTERVAL"

            python eval_ruler.py \
                --mode multipole_attention \
                --base_model "$BASE_MODEL" \
                $PREPARE_FLAG --model_template_type "$MODEL_TEMPLATE" --tokenizer_family "$TOKENIZER_FAMILY" \
                --seq_lengths $SEQ_LENGTHS \
                --num_samples "$NUM_SAMPLES" \
                --output_dir "$OUTPUT_DIR" \
                --run_name "$RUN_NAME"
        done
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"
