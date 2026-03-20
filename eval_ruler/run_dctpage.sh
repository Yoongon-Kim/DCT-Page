#!/bin/bash
# DCT-Page RULER Evaluation Script
#
# Step 1: Prepare data (once):
#   bash run_dctpage.sh prepare <MODEL_NAME> <ROOT_DIR> [MODEL_TEMPLATE_TYPE]
#
# Step 2: Run prediction + eval (per config):
#   bash run_dctpage.sh predict <MODEL_NAME> <ROOT_DIR> <DCT_TOP_K> <COMPRESS_RATIO> [MODEL_TEMPLATE_TYPE] [DCT_PAGE_SIZE] [DCT_UNSELECTED_MODE]

export TOKENIZERS_PARALLELISM=false
RULER_PATH=$(dirname $0)

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

NUM_SAMPLES=25
BENCHMARK="synthetic"

MODE=${1:?"Usage: $0 <prepare|predict> ..."}
shift

if [ "$MODE" == "prepare" ]; then
    # ---- DATA PREPARATION ----
    MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
    ROOT_DIR=${2:-"results_ruler"}
    MODEL_TEMPLATE_TYPE=${3:-"llama-3"}

    python -c "import nltk; nltk.download('punkt')"
    python -c "import nltk; nltk.download('punkt_tab')"

    for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
        DATA_DIR="${ROOT_DIR}/data/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        mkdir -p ${DATA_DIR}

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
        done
    done

    echo "Data preparation done: ${ROOT_DIR}/data/"

elif [ "$MODE" == "predict" ]; then
    # ---- PREDICTION + EVAL ----
    MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
    ROOT_DIR=${2:-"results_ruler"}
    DCT_TOP_K=${3:-8}
    COMPRESS_RATIO=${4:-0.032}
    MODEL_TEMPLATE_TYPE=${5:-"llama-3"}
    DCT_PAGE_SIZE=${6:-128}
    DCT_UNSELECTED_MODE=${7:-"compressed"}

    for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
        DATA_DIR="${ROOT_DIR}/data/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        PRED_DIR="${ROOT_DIR}/DCTPage_topk${DCT_TOP_K}_cr${COMPRESS_RATIO}_ps${DCT_PAGE_SIZE}_${DCT_UNSELECTED_MODE}/${BENCHMARK}/${MAX_SEQ_LENGTH}/pred"
        mkdir -p ${PRED_DIR}

        for TASK in "${TASKS[@]}"; do
            python ${RULER_PATH}/pred/predict_dctpage.py \
                --data_dir ${DATA_DIR} \
                --save_dir ${PRED_DIR} \
                --benchmark ${BENCHMARK} \
                --task ${TASK} \
                --model_name_or_path ${MODEL_NAME} \
                --dct_top_k ${DCT_TOP_K} \
                --dct_compress_ratio ${COMPRESS_RATIO} \
                --dct_page_size ${DCT_PAGE_SIZE} \
                --dct_unselected_mode ${DCT_UNSELECTED_MODE}
        done

        python ${RULER_PATH}/eval/evaluate.py \
            --data_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK}
    done

else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 <prepare|predict> ..."
    exit 1
fi
