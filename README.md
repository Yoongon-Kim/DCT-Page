# DCT-Page

Decode-time sparse page attention for long-context LLMs, with training-free proxy scoring (**DCT-lowpass-IDCT**) and fused Triton kernels. The repo bundles the DCT-Page implementation together with five baseline methods (SeerAttention-R, Multipole, Quest, DuoAttention, InfLLM), a full oracle/diagnostic suite, and benchmark runners for RULER, LongBench v1/v2, AIME25, and GPQA.

## Idea

During autoregressive decoding, the KV cache is divided into fixed-size pages. Each page is represented by a compact DCT-lowpass-IDCT proxy (e.g. 32 tokens → 1 or a few tokens). The query attends against these proxies to score pages, keeps the **top-k most relevant pages** at full precision, and either drops or replaces the remainder with their compressed representation. Prefill runs full attention unchanged.

Default configuration:
- score proxy: **DCT-lowpass-IDCT**
- `page_size=32`, `top_k=64`, `compress_ratio=0.03125` (`comp_size=1`)
- `unselected_mode="drop"` (speed) — switch to `"compressed"` for a quality floor

## Layout

Decode-time KV layout:

```text
[sink] [page 0] [page 1] ... [page N-1] [recent]
```

- `sink`: always kept (default 4 tokens)
- `recent`: always kept (default 128 tokens, absorbs the last partial page)
- `selected pages`: top-k by proxy score, full KV
- `unselected pages`:
  - `drop` — removed entirely
  - `compressed` — replaced with `comp_size` DCT-lowpass-IDCT tokens per page

## Repository map

### Core

| File | Role |
|---|---|
| [config.py](config.py) | `DCTPageConfig` dataclass (every hyperparameter with defaults) |
| [dct_page_attention.py](dct_page_attention.py) | Main forward, compression cache, RoPE handling, monkey-patch entry points |
| [triton_kernels.py](triton_kernels.py) | Fused Triton kernels (scoring, top-k, KV assembly, RoPE) + PyTorch fallbacks |

### Evaluation

| File | Benchmark | Notes |
|---|---|---|
| [eval_ruler.py](eval_ruler.py) | RULER (13 synthetic tasks, default 32k seq len) | Modes: baseline, page_attention, seer_attention, seer_prefill, multipole_attention, quest_attention, duo_attention, inf_llm. Llama 3.x and Qwen3. |
| [eval_longbench_v1.py](eval_longbench_v1.py) | LongBench v1 (16 English tasks) | Modes: baseline, page_attention, seer_attention, multipole_attention, quest_attention, duo_attention, inf_llm. |
| [eval_longbench_v2.py](eval_longbench_v2.py) | LongBench v2 (503 multiple-choice) | Adds `rope_gap` mode for RoPE interpolation studies. |
| [eval_aime25.py](eval_aime25.py) | AIME 2025 (30 problems, pass@1) | Qwen3-8B only. |
| [eval_gpqa.py](eval_gpqa.py) | GPQA (diamond/main/extended, MC accuracy) | Qwen3-8B only. |

### Baselines (`baselines/`)

| Folder | Baseline | Model support |
|---|---|---|
| [baselines/duo_attn/](baselines/duo_attn/) | DuoAttention (head streaming + recent window) | Llama 3.x (requires dedicated env: `transformers==4.45.2`, `flash-attn==2.6.3`) |
| [baselines/infllm/](baselines/infllm/) | InfLLM (retrieval-based block attention) | Llama 3.x (main DCT_Page env, vendored) |
| [baselines/seer_attn/](baselines/seer_attn/) | SeerAttention-R (learned gate-based sparsity, decode + optional prefill) | Llama 3.x, Qwen2/3 |
| [baselines/multipole_attn/](baselines/multipole_attn/) | Multipole Attention (hierarchical k-means clustering) | Llama 3.x, Qwen2/3 |
| [baselines/quest_attn/](baselines/quest_attn/) | Quest (per-page min/max key metadata; custom CUDA kernels) | Llama 2/3.x, Mistral, Qwen3 |

Each baseline folder has a `config.py` with defaults (pattern paths, model names, sparsity budgets, etc.). Run scripts rewrite these configs in place for sweeps.

### Observations & oracle diagnostics (`observations/`)

| File | Purpose |
|---|---|
| [observations/dct_page_energy.py](observations/dct_page_energy.py) | Standalone observation: runs a model and measures per-page DCT-lowpass energy of K/V (proxy sanity/visualization). |
| [observations/oracle_ruler.py](observations/oracle_ruler.py) | Standalone RULER runner; flat per-task JSONL output. |
| [observations/scoring_method_recall.py](observations/scoring_method_recall.py) | Compares ~30 proxy scoring methods against a configurable ground truth (`oracle_max` or `output_contribution`). |
| [observations/attention_mass_recall_ruler.py](observations/attention_mass_recall_ruler.py) | Dense-trajectory reference: full-KV forward, computes per-selector attention-mass recall (DCT, Quest, ShadowKV, oracle_max, mass-topk ceiling). `--mode full` (default) or `--mode quest` (Quest-native geometry). |
| [observations/oracle_hybrid_ruler.py](observations/oracle_hybrid_ruler.py) | Oracle-selection + hybrid-unselected sweeps. |

### Speed / profiling (`speed/`)

| File | Purpose |
|---|---|
| [speed/speed_dummy.py](speed/speed_dummy.py) | Decode throughput benchmark with dummy (random) token inputs. Modes: `baseline`, `dct`, `multipole` (one per run). |
| [speed/profile_decode.py](speed/profile_decode.py) | Per-stage decode profile with chained CUDA events. |
| [speed/compare_speed_dummy.sh](speed/compare_speed_dummy.sh) | Wrapper that runs baseline + DCT configs and prints a tok/s table. |

### Benchmark infrastructure (`benchmark/`)

- `benchmark/data/` — prepared `longbench_v1_data/` and `ruler_data/`
- `benchmark/eval_ruler/` — RULER pipeline (`data/prepare.py`, `eval/evaluate.py`, `synthetic.yaml`, `config_tasks.sh`)
  - `pred/predict_dctpage.py` — prediction-only DCT-Page path compatible with the official RULER evaluator.

### Run scripts (in `experiments/`)

Each sweep script calls the corresponding `eval_*.py` with a parameter grid. All use `--skip_existing` so interrupted runs resume cleanly.

- RULER: `experiments/run_ruler.sh` (Qwen3), `experiments/run_ruler_llama.sh`, `experiments/run_ruler_seer.sh`, `experiments/run_ruler_multipole.sh`, `experiments/run_ruler_duo.sh`, `experiments/run_ruler_quest.sh`
- LongBench v1: `experiments/run_longbench_v1.sh`, `experiments/run_longbench_v1_llama.sh`, `experiments/run_longbench_v1_seer.sh`, `experiments/run_longbench_v1_multipole.sh`, `experiments/run_longbench_v1_duo.sh`
- LongBench v2: `experiments/run_longbench_v2.sh`, `experiments/run_longbench_v2_llama.sh`, `experiments/run_longbench_v2_seer.sh`, `experiments/run_longbench_v2_multipole.sh`, `experiments/run_longbench_v2_duo.sh`

## Setup

```bash
pip install -r requirements.txt
# Core (DCT_Page conda env): torch 2.11.0, transformers 5.5.4, triton 3.6.0
```

Baseline-specific setup:

- **DuoAttention** — dedicated conda env (`duo_env`) with `transformers==4.45.2`, `flash-attn==2.6.3`, and `pip install -e /path/to/duo-attention`. Patterns live under `attn_patterns/` in the upstream repo; set `PATTERN_ROOT` / `PATTERN_SUBDIR` or edit [baselines/duo_attn/config.py](baselines/duo_attn/config.py).
- **InfLLM** — runs in the main DCT_Page env (`transformers==5.5.4`); upstream `inf_llm` is vendored under [baselines/infllm/upstream/](baselines/infllm/upstream/), so no separate env or editable install is needed. See [baselines/infllm/config.py](baselines/infllm/config.py).
- **Quest** — build the custom CUDA extension once: `bash baselines/quest_attn/build_kernels.sh`.
- **SeerAttention-R** — checkpoints pulled from Hugging Face Hub on first run (e.g. `SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates`).

### Benchmark data

Benchmark data lives under `benchmark/data/` and is **gitignored** (~136 MB), so a fresh clone starts without it. Both sources regenerate on demand — no manual download step:

- **RULER** — synthetic. Pass `--prepare` to an `eval_ruler.py` run (or any `experiments/run_ruler*.sh`) to generate and cache it under `benchmark/data/ruler_data/` via [benchmark/eval_ruler/data/prepare.py](benchmark/eval_ruler/data/prepare.py).
- **LongBench v1** — `eval_longbench_v1.py` auto-downloads each task from the `THUDM/LongBench` Hugging Face dataset on first run when no local `benchmark/data/longbench_v1_data/<task>.jsonl` exists (needs network + HF access).

## Basic usage (DCT-Page)

Monkey-patch **before** loading the model.

```python
from dct_page_attention import replace_llama_attn
from transformers import AutoModelForCausalLM
import torch

replace_llama_attn(
    page_size=32,
    top_k=64,
    num_sink_pages=1,
    num_recent_pages=5,
    compress_ratio=0.03125,
    scoring_method="max",
    group_agg_method="mean",
    unselected_mode="drop",          # or "compressed"
    compressed_token_rope="mixed",
    weight_compressed_by_population=True,
    use_triton=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
```

For Qwen3-8B use `replace_qwen3_attn(...)`.

## Default config ([config.py](config.py))

| Parameter | Default |
|---|---:|
| `page_size` | `32` |
| `top_k` | `64` |
| `num_sink_pages` | `1` |
| `num_recent_pages` | `5` |
| `compress_ratio` | `0.03125` |
| `min_decode_kv_len_for_paging` | `8192` |
| `scoring_method` | `"max"` |
| `group_agg_method` | `"mean"` |
| `unselected_mode` | `"drop"` |
| `compressed_token_rope` | `"mixed"` |
| `continuous_rope` | `False` (disabled) |
| `score_use_quest_minmax` | `False` |
| `select_with_oracle_page_scores` | `False` |
| `use_triton` | `True` |
| `weight_compressed_by_population` | `False` |
| `max_unselected_compressed` | `-1` |
| `comp_kv_quant` | `"none"` |
| `comp_kv_quant_granularity` | `"per_page"` |

## RULER

### Single run

```bash
# DCT-Page (drop mode, Qwen3-8B)
python eval_ruler.py \
  --mode page_attention \
  --base_model Qwen/Qwen3-8B \
  --seq_lengths 32768 \
  --num_samples 25 \
  --page_size 32 --top_k 64 \
  --compress_ratio 0.125 \
  --unselected_mode drop \
  --output_dir results/ruler \
  --run_name qwen_drop_ps32_t64
```

```bash
# Baseline
python eval_ruler.py \
  --mode baseline \
  --base_model Qwen/Qwen3-8B \
  --seq_lengths 32768 \
  --num_samples 25 \
  --output_dir results/ruler \
  --run_name baseline
```

Tasks default to all 13 RULER tasks:
```
niah_single_{1,2,3}, niah_multikey_{1,2,3}, niah_multivalue, niah_multiquery,
vt, cwe, fwe, qa_1, qa_2
```

Add `--prepare` to generate synthetic data on demand (cached under `benchmark/data/ruler_data/`).

### Baselines

```bash
# SeerAttention-R
python eval_ruler.py --mode seer_attention --base_model Qwen/Qwen3-8B \
  --output_dir results/ruler --run_name seer_budget1024
# (token budget set in baselines/seer_attn/config.py or via seer CLI flags)

# Multipole
python eval_ruler.py --mode multipole_attention --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/ruler --run_name multipole

# Quest
python eval_ruler.py --mode quest_attention --base_model Qwen/Qwen3-8B \
  --page_size 16 --top_k 128 --output_dir results/ruler --run_name quest

# DuoAttention (requires duo_env)
python eval_ruler.py --mode duo_attention --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/ruler --run_name duo

# InfLLM (runs in main DCT_Page env)
python eval_ruler.py --mode inf_llm --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/ruler --run_name inf_llm
```

### Sweep scripts

```bash
BASE_MODEL=Qwen/Qwen3-8B           bash experiments/run_ruler.sh
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct bash experiments/run_ruler_llama.sh
bash experiments/run_ruler_seer.sh
bash experiments/run_ruler_multipole.sh
bash experiments/run_ruler_duo.sh              # activates duo_env
bash experiments/run_ruler_quest.sh            # launches LLaMA on GPU 2, Qwen3 on GPU 3
```

### Prediction-only path

```bash
python benchmark/eval_ruler/pred/predict_dctpage.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --data_dir benchmark/data/ruler_data/llama/8192 \
  --save_dir results/ruler/predict_dctpage \
  --task qa_1 \
  --dct_page_size 32 --dct_top_k 64 \
  --dct_compress_ratio 0.03125 \
  --dct_unselected_mode drop
```

## LongBench v1

16 English tasks: single-doc QA, multi-doc QA, summarization, few-shot, synthetic, code completion. Metrics are the official per-task ones (F1, ROUGE-L, classification accuracy, code similarity).

```bash
# Baseline
python eval_longbench_v1.py --mode baseline \
  --base_model Qwen/Qwen3-8B \
  --output_dir results/longbench_v1 --run_name baseline

# DCT-Page drop
python eval_longbench_v1.py --mode page_attention \
  --base_model Qwen/Qwen3-8B \
  --page_size 32 --top_k 64 --compress_ratio 0.03125 \
  --unselected_mode drop \
  --output_dir results/longbench_v1 --run_name drop_ps32_top64_comp1

# DCT-Page compressed
python eval_longbench_v1.py --mode page_attention \
  --base_model Qwen/Qwen3-8B \
  --page_size 32 --top_k 64 --compress_ratio 0.03125 \
  --unselected_mode compressed \
  --output_dir results/longbench_v1 --run_name compressed_ps32_top64_comp1
```

No-chat tasks (prompt passed without chat template): `trec`, `triviaqa`, `samsum`, `lcc`, `repobench-p`.

`eval_longbench_v1.py` updates `summary.json` / `summary.csv` after every task so in-flight runs can be inspected without waiting to completion.

## LongBench v2

Multiple-choice format (503 questions, 4 options); accuracy is reported overall, by difficulty (easy/hard), and by context length (short/medium/long).

```bash
python eval_longbench_v2.py --mode page_attention \
  --base_model Qwen/Qwen3-8B \
  --page_size 32 --top_k 64 --compress_ratio 0.03125 \
  --unselected_mode drop \
  --output_dir results/longbench_v2 --run_name drop_ps32_top64_comp1
```

## AIME 2025 / GPQA (Qwen3-8B only)

Reasoning benchmarks. Both scripts accept the full method-mode list for argparse parity, but guard against non-Qwen3 models at runtime.

```bash
python eval_aime25.py --mode page_attention \
  --max_new_tokens 16384 \
  --page_size 32 --top_k 64 --unselected_mode drop \
  --output_dir results/aime25 --run_name drop

python eval_gpqa.py --mode page_attention \
  --gpqa_subset diamond \
  --max_new_tokens 8192 \
  --page_size 32 --top_k 64 --unselected_mode drop \
  --output_dir results_gpqa --run_name drop
```

## Speed measurement

### Throughput (dummy inputs)

```bash
bash speed/compare_speed_dummy.sh
GPU=1 CONTEXT_LENGTHS="8192,16384,32768,65536" NUM_REPEATS=5 bash speed/compare_speed_dummy.sh
```

Direct invocation:

```bash
python speed/speed_dummy.py \
  --mode dct \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --context_lengths 8192,16384,32768 \
  --warmup_steps 1 --num_repeats 3 --max_new_tokens 128 \
  --page_size 32 --top_k 64 --num_sink_pages 1 --num_recent_pages 5 \
  --compress_ratio 0.03125 \
  --unselected_mode drop
```

Outputs: `<run_dir>/summary.json`, `<run_dir>/samples.jsonl`.

### Per-stage profile

```bash
python speed/profile_decode.py \
  --context_length 32768 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --page_size 32 --top_k 64 \
  --num_sink_pages 1 --num_recent_pages 5 \
  --compress_ratio 0.03125 \
  --unselected_mode drop
```

Reports chained-CUDA-event timings for `qkv`, `score_cache_update`, `score_pages_kernel`, `topk`, `assemble_drop_and_final_k_original_rope`, `sdpa`, `o_proj`.

## Oracle diagnostics

```bash
# Scoring-method comparison vs. full-KV ground truth
python observations/scoring_method_recall.py \
  --ground_truths oracle_max,output_contribution \
  --context_len 16384 \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct

# Dense-trajectory attention-mass recall (DCT / Quest / ShadowKV / oracle_max / mass-topk)
python observations/attention_mass_recall_ruler.py \
  --context_len 32768 \
  --page_size 32 --top_k 64

# Standalone RULER runner (independent of eval_ruler.py)
python observations/oracle_ruler.py \
  --mode page_attention \
  --context_len 16384 \
  --tasks niah_multikey_3 \
  --tag oracle_demo \
  --num_samples 25 \
  --cuda_device 0 \
  --dct_page_size 32 --dct_top_k 64 \
  --dct_unselected_mode drop
```

Mass-recall reports three metric families per query head (see the module docstring in [observations/attention_mass_recall_ruler.py](observations/attention_mass_recall_ruler.py) for the full math):

- **Full-KV mass** — recall including the sink + recent floor.
- **Selected-page mass** — fraction of total softmax mass landing on the selector's top-k pages.
- **Paged-only mass** — same fraction with the sink+recent floor stripped out of both numerator and denominator.

## Results directories

All benchmark and diagnostic outputs live under a single `results/` root (gitignored), one subdirectory per benchmark:

- `results/ruler/` — RULER runs, plus oracle diagnostics under `oracle/`, `oracle_hybrid/`, `ablations/`, `scoring_methods_diagnostic/`
- `results/longbench_v1/`, `results/longbench_v2/` — LongBench v1/v2 runs
- `results/aime25/`, `results/gpqa/`, `results/math500/` — reasoning benchmarks
- `results/speed/` — decode-throughput speed tests
- `results/energy/` — per-page DCT energy measurements
- `results/attention_mass_recall/` — per-configuration mass-recall outputs (e.g. `mass_dense_ps16_topk128_cr0.125_fp8_e4m3/`)

## Notes

- The only active score proxy is **DCT-lowpass-IDCT**. Haar, Walsh-Hadamard, direct-spectral, and alternate frequency layouts (`low_high`, `low_mid_high`, `spread`) have been removed.
- Supported `scoring_method`: `"max"`, `"mean"`, `"sum"` (plus the QUEST-style min/max variant via `--score_use_quest_minmax`). `dc_ac` / `spectral_recon_max` / `hybrid_multi` families were removed.
- `compressed` mode is for accuracy experiments; speed optimisation targets `drop`.
- Below `min_decode_kv_len_for_paging=8192`, the patch falls back to baseline decode attention.
- `max_unselected_compressed` (default `-1`) limits how many unselected pages contribute compressed KV: `-1` = unlimited, `0` = drop-equivalent, `N` = top-N by score.
- DuoAttention and InfLLM currently support Llama 3.x only. Quest supports LLaMA-family models and Qwen3.
- LongBench v1 semantics follow the FastKV adjustments (prompt formatting, no-chat tasks, metric computation).
