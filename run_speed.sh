#!/bin/bash
set -euo pipefail

# Unified wrapper for decode throughput benchmarking.
#
# Defaults:
# - warmup 1
# - measured repeats 3
# - max_new_tokens 128
# - baseline + current default sparse setting + ps32/topk64/comp4 candidate
#
# Example:
#   bash run_speed.sh
#   CONTEXT_LENGTHS="8192,16384,32768" bash run_speed.sh
#   RUN_COMP4=0 bash run_speed.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dct}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-8192,16384,32768,65536,131072}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
NUM_REPEATS="${NUM_REPEATS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/results/speed_test_dummy}"
GPU="${GPU:-0}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_DEFAULT="${RUN_DEFAULT:-1}"
RUN_COMP4="${RUN_COMP4:-1}"

COMMON_ARGS=(
  --model "$MODEL"
  --context_lengths "$CONTEXT_LENGTHS"
  --warmup_steps "$WARMUP_STEPS"
  --num_repeats "$NUM_REPEATS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --chunk_size "$CHUNK_SIZE"
  --output_dir "$OUTPUT_ROOT"
)

run_python() {
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python "$@"
}

echo "============================================================"
echo "Decode Throughput Benchmark"
echo "============================================================"
echo "model            : $MODEL"
echo "context_lengths  : $CONTEXT_LENGTHS"
echo "warmup_steps     : $WARMUP_STEPS"
echo "num_repeats      : $NUM_REPEATS"
echo "max_new_tokens   : $MAX_NEW_TOKENS"
echo "chunk_size       : $CHUNK_SIZE"
echo "gpu              : $GPU"
echo "output_root      : $OUTPUT_ROOT"
echo ""

if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "============================================================"
  echo "BASELINE"
  echo "============================================================"
  run_python speed_test_dummy.py \
    "${COMMON_ARGS[@]}" \
    --mode baseline \
    --run_name baseline_w${WARMUP_STEPS}_r${NUM_REPEATS}_m${MAX_NEW_TOKENS}
fi

if [[ "$RUN_DEFAULT" == "1" ]]; then
  echo ""
  echo "============================================================"
  echo "DCT DEFAULT: ps32 / topk64 / comp1 / Haar / drop"
  echo "============================================================"
  run_python speed_test_dummy.py \
    "${COMMON_ARGS[@]}" \
    --mode dct \
    --page_size 32 \
    --top_k 64 \
    --sink_size 4 \
    --recent_size 128 \
    --compress_ratio 0.03125 \
    --scoring_method max \
    --group_agg_method mean \
    --unselected_mode drop \
    --run_name ps32_topk64_comp1_haar_drop_w${WARMUP_STEPS}_r${NUM_REPEATS}_m${MAX_NEW_TOKENS}
fi

if [[ "$RUN_COMP4" == "1" ]]; then
  echo ""
  echo "============================================================"
  echo "DCT CANDIDATE: ps32 / topk64 / comp4 / Haar / drop"
  echo "============================================================"
  run_python speed_test_dummy.py \
    "${COMMON_ARGS[@]}" \
    --mode dct \
    --page_size 32 \
    --top_k 64 \
    --sink_size 4 \
    --recent_size 128 \
    --compress_ratio 0.125 \
    --scoring_method max \
    --group_agg_method mean \
    --unselected_mode drop \
    --run_name ps32_topk64_comp4_haar_drop_w${WARMUP_STEPS}_r${NUM_REPEATS}_m${MAX_NEW_TOKENS}
fi

echo ""
echo "Done. Summaries:"
for name in \
  "baseline_w${WARMUP_STEPS}_r${NUM_REPEATS}_m${MAX_NEW_TOKENS}" \
  "ps32_topk64_comp1_haar_drop_w${WARMUP_STEPS}_r${NUM_REPEATS}_m${MAX_NEW_TOKENS}" \
  "ps32_topk64_comp4_haar_drop_w${WARMUP_STEPS}_r${NUM_REPEATS}_m${MAX_NEW_TOKENS}"
do
  if [[ -f "$OUTPUT_ROOT/$name/summary.json" ]]; then
    echo "  $OUTPUT_ROOT/$name/summary.json"
  fi
done
