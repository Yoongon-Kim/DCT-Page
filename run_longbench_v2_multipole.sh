#!/bin/bash
# LongBench v2 Evaluation — Multipole Attention
# Sweeps percent_clusters and percentiles values by rewriting multipole_attn/config.py.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-127500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-results_longbench_v2/multipole_attention}"

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
before running eval_longbench_v1.py or eval_longbench_v2.py with --mode multipole_attention.

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

            python eval_longbench_v2.py \
                --mode multipole_attention \
                --base_model "$BASE_MODEL" \
                --max_input_len "$MAX_INPUT_LEN" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --num_samples "$NUM_SAMPLES" \
                --prefill_chunk_size "$PREFILL_CHUNK_SIZE" \
                --output_dir "$OUTPUT_DIR" \
                --run_name "$RUN_NAME"
        done
    done
done

# ---- Summarize all results ----
echo ""
echo "============================================================"
echo "SUMMARIZING ALL RESULTS"
echo "============================================================"
python3 summarize_longbench_v2.py "$OUTPUT_DIR"
