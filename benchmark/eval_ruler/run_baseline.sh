#!/bin/bash
# Baseline (Full Attention) RULER Evaluation Script
# Usage: bash run_baseline.sh [MODEL_NAME] [ROOT_DIR] [MODEL_TEMPLATE_TYPE]
# Example: bash run_baseline.sh meta-llama/Llama-3.1-8B-Instruct results_ruler llama-3

export TOKENIZERS_PARALLELISM=false
RULER_PATH=$(dirname $0)
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"

SEQ_LENGTHS=(
    4096
    8192
    16384
    32768
    65536
    131072
)

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

# Experiment Setup
NUM_SAMPLES=25
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# Arguments
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
ROOT_DIR=${2:-"results_ruler"}
MODEL_TEMPLATE_TYPE=${3:-"llama-3"}

BENCHMARK="synthetic"
MODEL_FRAMEWORK="hf"

MODEL_NAME_FOR_PATH=$(basename ${MODEL_NAME})

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${ROOT_DIR}/baseline/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python ${RULER_PATH}/data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${MODEL_NAME} \
            --tokenizer_type "hf" \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES}

        python ${RULER_PATH}/pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_NAME} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P}
    done

    python ${RULER_PATH}/eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done
