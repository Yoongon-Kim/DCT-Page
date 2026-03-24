#!/bin/bash
# LongBench v1 Evaluation — Multipole Attention
# Sweeps percent_clusters and percentiles values by rewriting multipole_attn/config.py.
# Runs the same tasks as run_longbench_v1.sh.
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-127500}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-results_longbench_v1/multipole_attention}"

# Same tasks as run_longbench_v1.sh
TASKS="${TASKS:-narrativeqa qasper gov_report 2wikimqa multifieldqa_en triviaqa}"

# Fixed multipole parameters
USE_CENTROIDS=True
USE_REPLACEMENT=False
CLUSTER_INTERVAL=128
INFERENCE_TP=1

# Build task args
TASK_ARGS=""
if [ -n "$TASKS" ]; then
    TASK_ARGS="--tasks $TASKS"
fi

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
            RUN_NAME="llama_multipole_pct${PCT_CLUSTERS}_ptl${PERCENTILES}_repl${REPL}"

            echo ""
            echo "===================================================================="
            echo "MULTIPOLE: pct_clusters=${PCT_CLUSTERS}, percentiles=${PERCENTILES}, use_replacement=${REPL}"
            echo "===================================================================="

            write_config "$PCT_CLUSTERS" "$PERCENTILES" "$REPL" "$CLUSTER_INTERVAL"

            python eval_longbench_v1.py \
                --mode multipole_attention \
                --base_model "$BASE_MODEL" \
                --max_input_len "$MAX_INPUT_LEN" \
                --num_samples "$NUM_SAMPLES" \
                --output_dir "$OUTPUT_DIR" \
                --run_name "$RUN_NAME" \
                $TASK_ARGS
        done
    done
done

# ---- Summarize all results ----
echo ""
echo "============================================================"
echo "SUMMARIZING ALL RESULTS"
echo "============================================================"
python3 summarize_longbench_v1.py "$OUTPUT_DIR"
