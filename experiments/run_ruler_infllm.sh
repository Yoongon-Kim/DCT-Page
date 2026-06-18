#!/bin/bash

# Run from repo root regardless of where this script is invoked: it now
# lives in experiments/, but its relative paths (eval_*.py, baselines/,
# results/) are repo-root-relative.
cd "$(dirname "$0")/.." || exit 1
# RULER Evaluation — InfLLM (accuracy sweep)
# Varies the 5 accuracy-relevant knobs entirely through CLI flags on
# eval_ruler.py: --inf_llm_topk, --inf_llm_block_size, --inf_llm_n_local,
# --inf_llm_n_init, --inf_llm_repr_topk. max_cached_block (=2*topk) and
# chunk_size also flow via CLI. The static baselines/infllm/config.py is
# no longer rewritten — it provides safe defaults for ad-hoc invocations
# and stable values for the non-sweep knobs (attn_type, exc_block_size,
# fattn, base, distance_scale, score_decay, async_global_stream, faiss,
# perhead).
#
# InfLLM runs in the main DCT_Page env (transformers 5.5.4); upstream inf_llm is
# vendored under baselines/infllm/, so no separate env or activation is needed.
set -e

# ---- Configuration (env defaults, overridable via CLI flags below) ----
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
PREPARE_FLAG=""

# ---- Parse CLI flags ----
usage() {
    echo "Usage: $0 [--base_model MODEL] [--num_samples N] [--prepare]"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model)   BASE_MODEL="$2"; shift 2 ;;
        --num_samples)  NUM_SAMPLES="$2"; shift 2 ;;
        --prepare)      PREPARE_FLAG="--prepare"; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# InfLLM upstream only supports Llama 3.x here. Hard-fail otherwise.
case "$(echo "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')" in
    *llama*)  MODEL_TAG="llama" ;;
    *) echo "InfLLM baseline supports Llama only (got: $BASE_MODEL)"; exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-results/ruler}"

# Sequence lengths to evaluate
SEQ_LENGTHS="${SEQ_LENGTHS:-32768}"

# Fixed (non-swept) knob: GreedySearch prefill chunk size.
CHUNK_SIZE=8192

# ---- Accuracy sweep: topk x block_size x n_local x n_init x repr_topk ----
# topk      = blocks attended per decode step (main sparsity dial).
# block_size= tokens per block (retrieval granularity).
# n_local   = sliding-window of always-attended recent tokens.
# n_init    = sink token count (attention-anchor).
# repr_topk = representative tokens per block (block-summary fidelity).
# MAX_CACHED_BLOCK = 2*TOPK (must be >= topk; raising it only costs GPU memory).
# Tight initial grid: 2*1*1*2*2 = 8 cells. Add values to any axis to widen.
for TOPK in 64; do
    for BLOCK_SIZE in 32; do
        for N_LOCAL in 4096; do
            for N_INIT in 4 128; do
                for REPR_TOPK in 2 4; do
                    MAX_CACHED_BLOCK=$(( TOPK * 2 ))
                    RUN_NAME="${MODEL_TAG}_inf_llm_topk${TOPK}_bs${BLOCK_SIZE}_nlocal${N_LOCAL}_nini${N_INIT}_repr${REPR_TOPK}"

                    echo ""
                    echo "===================================================================="
                    echo "INFLLM: topk=${TOPK}, block_size=${BLOCK_SIZE}, n_local=${N_LOCAL}, n_init=${N_INIT}, repr_topk=${REPR_TOPK}"
                    echo "===================================================================="

                    python eval_ruler.py \
                        --mode inf_llm \
                        --base_model "$BASE_MODEL" \
                        --skip_existing \
                        $PREPARE_FLAG \
                        --seq_lengths $SEQ_LENGTHS \
                        --num_samples "$NUM_SAMPLES" \
                        --inf_llm_topk "$TOPK" \
                        --inf_llm_block_size "$BLOCK_SIZE" \
                        --inf_llm_n_local "$N_LOCAL" \
                        --inf_llm_n_init "$N_INIT" \
                        --inf_llm_repr_topk "$REPR_TOPK" \
                        --inf_llm_max_cached_block "$MAX_CACHED_BLOCK" \
                        --inf_llm_chunk_size "$CHUNK_SIZE" \
                        --output_dir "$OUTPUT_DIR" \
                        --run_name "$RUN_NAME"
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE. Results in: $OUTPUT_DIR/"
echo "============================================================"
