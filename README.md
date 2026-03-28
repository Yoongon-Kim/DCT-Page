# DCT-Page

Long-context decode-time page attention with training-free proxy scoring and fused kernels.

현재 기본 설정은 다음과 같습니다.

- score proxy: `Haar` lowpass
- page attention default: `page_size=32`, `top_k=64`, `compress_ratio=0.03125` (`comp_size=1`)
- default execution mode: `unselected_mode="drop"`

핵심 아이디어는 decode 시 KV cache를 page 단위로 나누고, 각 page의 작은 proxy로 relevance를 점수화한 뒤, 선택된 page만 full KV로 attention에 넣는 것입니다.

## Layout

decode 시 기본 구조는 아래와 같습니다.

```text
[sink] [page 0] [page 1] ... [page N-1] [recent]
```

기본 `drop` 모드에서는:

- `sink`: 항상 유지
- `recent`: 항상 유지
- `selected pages`: full KV 사용
- `unselected pages`: 제거

`hybrid` 모드에서는:

- `selected pages`: full KV 사용
- `unselected pages`: Haar lowpass proxy KV 사용

## Main Files

```text
dct_page_attention.py     Core page-attention implementation
triton_kernels.py         Triton kernels for score / topk / assemble / RoPE
config.py                 Default config values

compare_baseline_dct.py   Ad-hoc pairwise baseline vs DCT/Haar comparison
run_ruler_eval.py         Single-mode RULER runner with flat task jsonl outputs
compare_ruler_runs.py     Post-hoc comparison across completed RULER runs
eval_longbench_v1.py      LongBench v1 evaluation
eval_longbench_v2.py      LongBench v2 evaluation
speed_test_dummy.py       Decode throughput benchmark
profile_decode.py         Decode-path CUDA event profiling

summarize_longbench_v1.py LongBench v1 summary CSV/JSON generator
summarize_longbench_v2.py LongBench v2 summary helper
watch_longbench_v1_partial.py  Periodic partial-summary refresher for in-flight v1 runs
```

## Default Config

Current defaults in [config.py](/home/jiwonsong/DCT-Page/config.py):

| Parameter | Default |
|---|---:|
| `page_size` | `32` |
| `top_k` | `64` |
| `sink_size` | `4` |
| `recent_size` | `128` |
| `compress_ratio` | `0.03125` |
| `scoring_method` | `"max"` |
| `group_agg_method` | `"mean"` |
| `unselected_mode` | `"drop"` |
| `continuous_rope` | `True` |
| `score_use_haar_proxy` | `True` |

## Basic Usage

Monkey-patch before loading the model.

```python
from dct_page_attention import replace_llama_attn
from transformers import AutoModelForCausalLM
import torch

replace_llama_attn(
    page_size=32,
    top_k=64,
    sink_size=4,
    recent_size=128,
    compress_ratio=0.03125,
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",   # or "hybrid"
    continuous_rope=True,
    score_use_haar_proxy=True,
    use_triton=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
```

Qwen 계열은 `replace_qwen2_attn(...)`를 사용하면 됩니다.

## RULER

이 repo는 두 가지 방식으로 RULER를 돌릴 수 있습니다.

### 1. prepared synthetic data 사용

이미 생성해 둔 synthetic RULER data는 다음 위치에 있습니다.

```text
results_ruler/data/synthetic/4096/
results_ruler/data/synthetic/8192/
results_ruler/data/synthetic/16384/
results_ruler/data/synthetic/32768/
```

단일 모드로 한 번만 돌리고, 비교는 나중에 별도 스크립트로 하는 것이 기본 workflow입니다.

예: drop / Haar run

```bash
python run_ruler_eval.py \
  --mode page_attention \
  --context_len 16384 \
  --tasks niah_multikey_3 \
  --tag niah_multikey_3_16k_drop_haar \
  --num_samples 25 \
  --max_new_tokens 128 \
  --cuda_device 0 \
  --dct_page_size 32 \
  --dct_top_k 64 \
  --dct_sink_size 4 \
  --dct_recent_size 128 \
  --dct_compress_ratio 0.03125 \
  --dct_unselected_mode drop
```

baseline run:

```bash
python run_ruler_eval.py \
  --mode baseline \
  --context_len 16384 \
  --tasks niah_multikey_3 \
  --tag niah_multikey_3_16k_baseline \
  --num_samples 25 \
  --max_new_tokens 128 \
  --cuda_device 0
```

기본은 Haar proxy입니다. low DCT로 강제로 내리고 싶으면:

```bash
  --dct_score_use_low_proxy
```

run 디렉터리 안에는 task별로 별도 폴더를 만들지 않고:

```text
<run_dir>/
  manifest.json
  run.log
  summary.csv
  summary.json
  niah_multikey_3.jsonl
  ...
```

pairwise compare가 필요하면:

```bash
python compare_ruler_runs.py \
  --run_dirs results/ruler_runs/run_a,results/ruler_runs/run_b \
  --labels baseline,drop_haar \
  --output_dir results/ruler_compare \
  --tag baseline_vs_drop_haar
```

기존 quick pairwise compare workflow가 필요할 때만 아래 스크립트를 씁니다.

```bash
python compare_baseline_dct.py \
  --data_path results_ruler/data/synthetic/16384/niah_multikey_3/validation.jsonl \
  --tag niah_multikey_3_16k_ps32_top64_drop \
  --num_samples 25 \
  --max_new_tokens 128 \
  --cuda_device 0 \
  --dct_page_size 32 \
  --dct_top_k 64 \
  --dct_sink_size 4 \
  --dct_recent_size 128 \
  --dct_compress_ratio 0.03125 \
  --dct_unselected_mode drop
```

hybrid 실행 예:

```bash
python run_ruler_eval.py \
  --mode page_attention \
  --context_len 32768 \
  --tasks niah_multivalue \
  --tag niah_multivalue_32k_ps32_top64_hybrid \
  --num_samples 25 \
  --max_new_tokens 128 \
  --cuda_device 0 \
  --dct_page_size 32 \
  --dct_top_k 64 \
  --dct_sink_size 4 \
  --dct_recent_size 128 \
  --dct_compress_ratio 0.03125 \
  --dct_unselected_mode hybrid
```

oracle selection upper bound 실행 예:

```bash
python run_ruler_oracle_selection.py \
  --context_len 32768 \
  --page_sizes 32,64,128 \
  --selected_token_budget 2048 \
  --compress_ratio 0.03125 \
  --num_samples 25 \
  --max_new_tokens 128 \
  --cuda_device 0
```

이 스크립트는 `drop` 모드에서 `--dct_select_with_oracle_page_scores`를 켠 upper bound를
여러 page size에 대해 반복 실행하고, 각 page-size subrun 안에는 task별 jsonl만 flat하게 저장합니다.

가정:

- `selected full-token budget`을 고정합니다.
- 따라서 기본 비교는
  - `page_size=32 -> top_k=64`
  - `page_size=64 -> top_k=32`
  - `page_size=128 -> top_k=16`
  입니다.
- `compress_ratio=0.03125`를 그대로 유지하므로 `comp_size`는 각각 `1, 2, 4`가 됩니다.

주요 결과 파일:

- `results/ruler_oracle_selection/<run>/summary.tsv`
- `results/ruler_oracle_selection/<run>/summary_avg.tsv`
- `results/ruler_oracle_selection/<run>/summary.json`
- `results/ruler_oracle_selection/<run>/commands.sh`
- `results/ruler_oracle_selection/<run>/ps32_topk64/<task>.jsonl`
- `results/ruler_oracle_selection/<run>/ps64_topk32/<task>.jsonl`
- `results/ruler_oracle_selection/<run>/ps128_topk16/<task>.jsonl`

### 2. predict script 사용

RULER prediction-only 경로:

```bash
python eval_ruler/pred/predict_dctpage.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_file results_ruler/data/synthetic/8192/qa_1/validation.jsonl \
  --output_file outputs.jsonl \
  --dct_page_size 32 \
  --dct_top_k 64 \
  --dct_compress_ratio 0.03125 \
  --dct_unselected_mode drop
```

## LongBench v1

### Data

LongBench v1 English jsonl data는 로컬에 다음 경로를 사용합니다.

```text
longbench_v1_data/data/*.jsonl
```

현재 `eval_longbench_v1.py`는 이 로컬 데이터를 우선 사용하도록 되어 있습니다.

### Baseline

```bash
python eval_longbench_v1.py \
  --mode baseline \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v1/v1_baseline \
  --run_name baseline
```

### Drop

```bash
python eval_longbench_v1.py \
  --mode page_attention \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v1/v1_drop \
  --run_name drop_ps32_top64_comp1_haar \
  --page_size 32 \
  --top_k 64 \
  --sink_size 4 \
  --recent_size 128 \
  --compress_ratio 0.03125 \
  --unselected_mode drop
```

### Hybrid

```bash
python eval_longbench_v1.py \
  --mode page_attention \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v1/v1_hybrid \
  --run_name hybrid_ps32_top64_comp1_haar \
  --page_size 32 \
  --top_k 64 \
  --sink_size 4 \
  --recent_size 128 \
  --compress_ratio 0.03125 \
  --unselected_mode hybrid
```

### Partial Summary While Running

`eval_longbench_v1.py`는 task 하나가 끝날 때마다 아래 파일을 갱신합니다.

- `summary.json`
- `summary.csv`

이미 돌고 있는 런에 대해서는 watcher를 붙일 수 있습니다.

```bash
conda run -n dct python watch_longbench_v1_partial.py \
  --root results/longbench_v1 \
  --interval 60
```

완료 후 수동 집계:

```bash
python summarize_longbench_v1.py \
  results/longbench_v1/v1_baseline/baseline \
  results/longbench_v1/v1_drop/drop_ps32_top64_comp1_haar \
  results/longbench_v1/v1_hybrid/hybrid_ps32_top64_comp1_haar
```

## LongBench v2

LongBench v2는 multiple-choice 형식 평가입니다.

### Baseline

```bash
python eval_longbench_v2.py \
  --mode baseline \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v2/v2_baseline \
  --run_name baseline
```

### Drop

```bash
python eval_longbench_v2.py \
  --mode page_attention \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v2/v2_drop \
  --run_name drop_ps32_top64_comp1_haar \
  --page_size 32 \
  --top_k 64 \
  --sink_size 4 \
  --recent_size 128 \
  --compress_ratio 0.03125 \
  --unselected_mode drop
```

### Hybrid

```bash
python eval_longbench_v2.py \
  --mode page_attention \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v2/v2_hybrid \
  --run_name hybrid_ps32_top64_comp1_haar \
  --page_size 32 \
  --top_k 64 \
  --sink_size 4 \
  --recent_size 128 \
  --compress_ratio 0.03125 \
  --unselected_mode hybrid
```

완료 후 요약:

```bash
python summarize_longbench_v2.py results/longbench_v2/v2_drop
```

## Speed Measurement

### Throughput benchmark

decode throughput은 [speed_test_dummy.py](/home/jiwonsong/DCT-Page/speed_test_dummy.py)로 재고, 보통은 wrapper인 [run_speed.sh](/home/jiwonsong/DCT-Page/run_speed.sh)를 쓰는 게 편합니다.

기본 실행:

```bash
bash run_speed.sh
```

single GPU로 고정하려면:

```bash
GPU=1 bash run_speed.sh
```

이 스크립트는 기본적으로 아래 3개를 같은 protocol로 돌립니다.

- `baseline`
- 현재 기본 sparse 설정: `ps32 / topk64 / comp1 / Haar / drop`
- 정확도 후보 설정: `ps32 / topk64 / comp4 / Haar / drop`

기본 protocol:

- `warmup_steps=1`
- `num_repeats=3`
- `max_new_tokens=128`

길이나 반복 수를 바꾸고 싶으면 env var로 넘기면 됩니다.

```bash
CONTEXT_LENGTHS="8192,16384,32768" NUM_REPEATS=5 bash run_speed.sh
```

직접 baseline만 재려면:

```bash
python speed_test_dummy.py \
  --mode baseline \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --context_lengths 8192,16384,32768 \
  --warmup_steps 1 \
  --num_repeats 3 \
  --max_new_tokens 128 \
  --output_dir results/speed_test_dummy
```

직접 DCT 설정 하나만 재려면:

```bash
python speed_test_dummy.py \
  --mode dct \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --context_lengths 8192,16384,32768 \
  --warmup_steps 1 \
  --num_repeats 3 \
  --max_new_tokens 128 \
  --page_size 32 \
  --top_k 64 \
  --sink_size 4 \
  --recent_size 128 \
  --compress_ratio 0.03125 \
  --scoring_method max \
  --group_agg_method mean \
  --unselected_mode drop
```

주요 산출물:

- `<run_dir>/summary.json`
- `<run_dir>/samples.jsonl`

### Decode profile

세부 decode 단계 프로파일은 [profile_decode.py](/home/jiwonsong/DCT-Page/profile_decode.py)를 사용합니다.

```bash
python profile_decode.py \
  --context_length 32768 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --page_size 32 \
  --top_k 64 \
  --sink_size 4 \
  --recent_size 128 \
  --compress_ratio 0.03125 \
  --unselected_mode drop
```

이 스크립트는 다음 같은 단계별 시간을 보여줍니다.

- `qkv`
- `score_cache_update`
- `score_pages_kernel`
- `topk`
- `assemble_drop_and_final_k_original_rope`
- `sdpa`
- `o_proj`

## Notes

- 현재 기본 score proxy는 DCT low-frequency가 아니라 `Haar lowpass`입니다.
- `hybrid`는 정확도 실험용 기능이며, 속도 최적화는 `drop` 경로를 우선했습니다.
- LongBench v1은 FastKV eval semantics에 맞춰 prompt / no-chat task / metric 계산을 조정해 두었습니다.
- LongBench v1 현재 no-chat task:
  - `trec`, `triviaqa`, `samsum`, `lsht`, `lcc`, `repobench-p`

## Results Pointers

최근 주요 결과 폴더:

- RULER full13 drop:
  - [ps32_top64_comp1_drop_full13_16k32k](/home/jiwonsong/DCT-Page/results/debug_compare/ps32_top64_comp1_drop_full13_16k32k)
- RULER full13 hybrid:
  - [ps32_top64_comp1_hybrid_full13_16k32k](/home/jiwonsong/DCT-Page/results/debug_compare/ps32_top64_comp1_hybrid_full13_16k32k)
- LongBench current rerun:
  - [results_longbench_compare_20260321_fix6_full](/home/jiwonsong/DCT-Page/results_longbench_compare_20260321_fix6_full)
- Speed benchmarks:
  - [results/speed_test_dummy](/home/jiwonsong/DCT-Page/results/speed_test_dummy)
