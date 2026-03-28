#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dct}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-127500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"

PAGE_SIZE=32
COMPRESS_RATIO=0.125
SINK_SIZE=4
RECENT_SIZE=128
SCORING_METHOD=max
GROUP_AGG_METHOD=mean
UNSELECTED_MODE=drop

BASELINE_GPU="${BASELINE_GPU:-0}"
TOPK64_GPU="${TOPK64_GPU:-1}"
TOPK32_GPU="${TOPK32_GPU:-2}"

RUN_TAG="${RUN_TAG:-longbench_v2_ps32_comp4_topk64_vs_topk32_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/longbench_v2/$RUN_TAG}"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_python() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$CONDA_ENV" python "$@"
}

echo "============================================================"
echo "LongBench v2 Compare Run"
echo "============================================================"
echo "output_dir      : $OUTPUT_DIR"
echo "base_model      : $BASE_MODEL"
echo "num_samples     : $NUM_SAMPLES"
echo "max_input_len   : $MAX_INPUT_LEN"
echo "max_new_tokens  : $MAX_NEW_TOKENS"
echo "baseline_gpu    : $BASELINE_GPU"
echo "topk64_gpu      : $TOPK64_GPU"
echo "topk32_gpu      : $TOPK32_GPU"
echo ""

run_python "$BASELINE_GPU" eval_longbench_v2.py \
  --mode baseline \
  --base_model "$BASE_MODEL" \
  --max_input_len "$MAX_INPUT_LEN" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --run_name baseline \
  > "$LOG_DIR/baseline.log" 2>&1 &
PID_BASELINE=$!

run_python "$TOPK64_GPU" eval_longbench_v2.py \
  --mode page_attention \
  --base_model "$BASE_MODEL" \
  --max_input_len "$MAX_INPUT_LEN" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --run_name ps32_topk64_comp4_haar_drop \
  --page_size "$PAGE_SIZE" \
  --top_k 64 \
  --sink_size "$SINK_SIZE" \
  --recent_size "$RECENT_SIZE" \
  --compress_ratio "$COMPRESS_RATIO" \
  --scoring_method "$SCORING_METHOD" \
  --group_agg_method "$GROUP_AGG_METHOD" \
  --unselected_mode "$UNSELECTED_MODE" \
  > "$LOG_DIR/ps32_topk64_comp4_haar_drop.log" 2>&1 &
PID_TOPK64=$!

run_python "$TOPK32_GPU" eval_longbench_v2.py \
  --mode page_attention \
  --base_model "$BASE_MODEL" \
  --max_input_len "$MAX_INPUT_LEN" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --run_name ps32_topk32_comp4_haar_drop \
  --page_size "$PAGE_SIZE" \
  --top_k 32 \
  --sink_size "$SINK_SIZE" \
  --recent_size "$RECENT_SIZE" \
  --compress_ratio "$COMPRESS_RATIO" \
  --scoring_method "$SCORING_METHOD" \
  --group_agg_method "$GROUP_AGG_METHOD" \
  --unselected_mode "$UNSELECTED_MODE" \
  > "$LOG_DIR/ps32_topk32_comp4_haar_drop.log" 2>&1 &
PID_TOPK32=$!

echo "Launched:"
echo "  baseline pid : $PID_BASELINE"
echo "  topk64 pid   : $PID_TOPK64"
echo "  topk32 pid   : $PID_TOPK32"
echo ""
echo "Logs:"
echo "  $LOG_DIR/baseline.log"
echo "  $LOG_DIR/ps32_topk64_comp4_haar_drop.log"
echo "  $LOG_DIR/ps32_topk32_comp4_haar_drop.log"

wait "$PID_BASELINE"
wait "$PID_TOPK64"
wait "$PID_TOPK32"

echo ""
echo "Done. Outputs:"
echo "  $OUTPUT_DIR/baseline.jsonl"
echo "  $OUTPUT_DIR/baseline_summary.json"
echo "  $OUTPUT_DIR/ps32_topk64_comp4_haar_drop.jsonl"
echo "  $OUTPUT_DIR/ps32_topk64_comp4_haar_drop_summary.json"
echo "  $OUTPUT_DIR/ps32_topk32_comp4_haar_drop.jsonl"
echo "  $OUTPUT_DIR/ps32_topk32_comp4_haar_drop_summary.json"
