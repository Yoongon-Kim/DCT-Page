#!/bin/bash
# RULER Evaluation — SnapKV
# Sweeps max_capacity_prompt × pooling.
#
# SnapKV requires a dedicated conda env with transformers==4.37.2,
# flash-attn==2.4.0.post1+, and the vendored baselines/snap_kv/ shim.
# Set SNAPKV_ENV_NAME (default: snap_kv) to override the env name.
set -euo pipefail

# ---- Conda activation ----
SNAPKV_ENV_NAME="${SNAPKV_ENV_NAME:-snap_kv}"
eval "$(conda shell.bash hook)"
conda activate "$SNAPKV_ENV_NAME"

# ---- Configuration (env defaults, overridable via env vars) ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ruler}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
KERNEL_SIZE="${KERNEL_SIZE:-5}"
MAX_CAPACITY_PROMPT_LIST="${MAX_CAPACITY_PROMPT_LIST:-1024 2048 4096}"
POOLING_LIST="${POOLING_LIST:-avgpool maxpool}"

# SnapKV only supports Llama — hard-fail otherwise.
case "${BASE_MODEL,,}" in
    *llama*) ;;
    *) echo "snap_kv only supports Llama (got: $BASE_MODEL)"; exit 1 ;;
esac

echo "======================================================================"
echo "SnapKV RULER sweep"
echo "  BASE_MODEL : $BASE_MODEL"
echo "  SEQ_LENGTHS: $SEQ_LENGTHS"
echo "  NUM_SAMPLES: $NUM_SAMPLES"
echo "  OUTPUT_DIR : $OUTPUT_DIR"
echo "  WINDOW_SIZE: $WINDOW_SIZE  KERNEL_SIZE: $KERNEL_SIZE"
echo "======================================================================"

for CAP in $MAX_CAPACITY_PROMPT_LIST; do
    for POOL in $POOLING_LIST; do
        RUN_NAME="llama_snapkv_cap${CAP}_${POOL}_win${WINDOW_SIZE}_ks${KERNEL_SIZE}"

        echo ""
        echo "----------------------------------------------------------------------"
        echo "SnapKV: max_capacity_prompt=${CAP}, pooling=${POOL}, window=${WINDOW_SIZE}, kernel=${KERNEL_SIZE}"
        echo "----------------------------------------------------------------------"

        python eval_ruler.py \
            --mode snap_kv \
            --base_model "$BASE_MODEL" \
            --seq_lengths $SEQ_LENGTHS \
            --num_samples "$NUM_SAMPLES" \
            --snapkv_window_size "$WINDOW_SIZE" \
            --snapkv_max_capacity_prompt "$CAP" \
            --snapkv_kernel_size "$KERNEL_SIZE" \
            --snapkv_pooling "$POOL" \
            --output_dir "$OUTPUT_DIR" \
            --run_name "$RUN_NAME" \
            --skip_existing
    done
done

echo ""
echo "======================================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "======================================================================"
