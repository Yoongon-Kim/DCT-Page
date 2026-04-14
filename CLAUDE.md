# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DCT-Page is a decode-time sparse page attention mechanism for long-context LLMs. During autoregressive decoding, it divides the KV cache into fixed-size pages, scores each page with a lightweight Haar/DCT proxy (32:1 compression), selects top-k pages for full attention, and drops or compresses the rest. Prefill uses standard full attention unchanged.

KV layout at decode time:
```
[sink (4 tokens)] [page 0] [page 1] ... [page N-1] [recent (128 tokens)]
```

- **Drop mode** (default): unselected pages discarded entirely (fastest).
- **Compressed/hybrid mode**: unselected pages use Haar/DCT compressed KV (quality floor).

## Architecture

### Core modules

| File | Role |
|---|---|
| `dct_page_attention.py` | Main attention forward, compression, RoPE, monkey-patching (~2000 lines) |
| `triton_kernels.py` | Fused Triton kernels: score, topk, assemble, RoPE (~2100 lines) |
| `config.py` | `DCTPageConfig` dataclass with all hyperparameters |

### Evaluation & benchmarking

| File | Role |
|---|---|
| `eval_ruler.py` | RULER benchmark (synthetic long-context tasks, exact match) |
| `eval_longbench_v1.py` | LongBench v1 (16 English tasks, F1/ROUGE/accuracy) |
| `eval_longbench_v2.py` | LongBench v2 (multiple-choice format) |
| `run_ruler_eval.py` | Single-mode RULER runner with flat task jsonl outputs |
| `compare_ruler_runs.py` | Post-hoc comparison across completed RULER runs |
| `compare_baseline_dct.py` | Ad-hoc pairwise baseline vs DCT/Haar comparison |
| `speed_test_dummy.py` | Decode throughput benchmark |
| `profile_decode.py` | Layer-by-layer decode-path CUDA event profiling |
| `oracle/run_ruler_eval.py` | Standalone RULER runner for oracle experiments |
| `oracle/eval_ruler_hybridmulti.py` | hybrid_multi scoring RULER sweep (M, alpha params) |

### Key functions in dct_page_attention.py

- `dct_page_attention_forward()` — replacement forward. Prefill: standard attention + cache pre-allocation. Decode: score pages -> topk select -> assemble -> SDPA.
- `replace_llama_attn()`, `replace_qwen2_attn()`, `replace_qwen3_attn()` — monkey-patch entry points. Call **before** `from_pretrained()`.
- `_update_comp_cache()` — incremental page compression (Haar or DCT), only compresses new pages since last decode step.
- `_build_haar_lowpass_projection_matrix()` — default scoring proxy (block averaging).
- `PreAllocatedLayer` — fixed-stride KV buffer replacing `DynamicLayer` for O(1) decode append.

### Key functions in triton_kernels.py

Each kernel has a `*_triton()` wrapper and a pure-PyTorch fallback:
- `_score_pages_fused_kernel` — scores all pages via query-compressed-key dot products
- `_topk_sort_kernel` — parallel topk via bitonic sort
- `_assemble_kv_full_kernel` — gathers sink + selected pages + recent, applies RoPE
- `_apply_rope_q_kernel` — single-token decode query RoPE

## Commands

### Setup

```bash
pip install -r requirements.txt
# Requires: torch 2.10, transformers 4.54, triton 3.6
```

### RULER evaluation

```bash
# DCT-Page drop mode
python run_ruler_eval.py --mode page_attention --context_len 16384 \
  --tasks niah_multikey_3 --tag my_run --num_samples 25 --max_new_tokens 128 \
  --cuda_device 0 --dct_page_size 32 --dct_top_k 64 --dct_unselected_mode drop

# Baseline
python run_ruler_eval.py --mode baseline --context_len 16384 \
  --tasks niah_multikey_3 --tag baseline --num_samples 25 --cuda_device 0

# Compare runs
python compare_ruler_runs.py \
  --run_dirs results/ruler_runs/run_a,results/ruler_runs/run_b \
  --labels baseline,drop_haar --output_dir results/ruler_compare

# Llama RULER sweep (page_size x top_k)
bash run_ruler_llama.sh

# Oracle hybrid_multi scoring sweep
python oracle/eval_ruler_hybridmulti.py --context_len 32768 \
  --base_model meta-llama/Llama-3.1-8B-Instruct
```

### LongBench evaluation

```bash
# v1 baseline / drop / hybrid / quest_attention
python eval_longbench_v1.py --mode {baseline|page_attention|seer_attention|multipole_attention|quest_attention} \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v1/<name> --run_name <name> \
  --page_size 32 --top_k 64 --unselected_mode {drop|hybrid}

# v2 (same modes, uses eval_longbench_v2.py)
python eval_longbench_v2.py --mode page_attention \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir results/longbench_v2/<name> --run_name <name> \
  --page_size 32 --top_k 64 --unselected_mode drop

# Llama sweep scripts (baseline + top_k sweep)
bash run_longbench_v1_llama.sh
bash run_longbench_v2_llama.sh

# Summarize completed runs
python summarize_longbench_v1.py <run_dir1> <run_dir2> ...
python summarize_longbench_v2.py <run_dir>
```

### Speed measurement

```bash
# All three modes via wrapper
bash run_speed.sh
GPU=1 bash run_speed.sh  # single GPU

# Direct single-mode
python speed_test_dummy.py --mode {baseline|dct} \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --context_lengths 8192,16384,32768 --output_dir results/speed

# Per-layer decode profile
python profile_decode.py --context_length 32768 \
  --model meta-llama/Llama-3.1-8B-Instruct --page_size 32 --top_k 64
```

### Oracle diagnostics

```bash
python oracle/diagnose_l2_scoring.py \
  --ground_truths oracle_max output_contribution \
  --scoring_methods oracle_max oracle_mean proxy_max proxy_mean dc_ac \
  --context_length 16384 --model meta-llama/Llama-3.1-8B-Instruct
```

## Conventions

- **Monkey-patch pattern**: set global config (`_dct_page_cfg` module variable) then patch `forward`. Always call `replace_*_attn()` before `from_pretrained()`.
- **Tensor naming**: `paged_*` = reshaped to `[..., num_pages, page_size, ...]`. `comp_*` = compressed. `*_buf` = pre-allocated buffer.
- **Buffer caching**: projection matrices and kernel caches are stored on `attn_module` attributes (lazy init, shape/device checked each call via `_get_or_build_*` functions).
- **Triton kernels**: `@triton.jit` with constexpr block sizes; wrappers handle grid launch and PyTorch fallback when `use_triton=False`.
- **No test suite**: validation through benchmark runs (RULER, LongBench).
- **Supported models**: Llama 3.x (`replace_llama_attn`), Qwen 2.x (`replace_qwen2_attn`), Qwen 3.x with QK-norm (`replace_qwen3_attn`).
- **RoPE**: supports default, Yarn, and Llama3 rope types. `continuous_rope` currently disabled; `compressed_token_rope="mixed"` active.
- **Run naming**: encodes params, e.g. `drop_ps32_top64_comp1_haar`.
- **DCTPageConfig defaults**: `page_size=32`, `top_k=64`, `sink_size=4`, `recent_size=128`, `compress_ratio=0.03125` (comp_size=1), `scoring_method="max"`, `score_use_haar_proxy=True`, `unselected_mode="drop"`, `use_triton=True`.

## Data paths

- RULER synthetic data: `results_ruler/data/synthetic/{seq_len}/`
- LongBench v1 data: `longbench_v1_data/data/*.jsonl` or `benchmark/data/longbench_v1_data/*.jsonl`
- Results: `results/`, `results_ruler/`

## Notes

- Default score proxy is **Haar lowpass** (block averaging), not DCT low-frequency.
- `hybrid` mode is for accuracy experiments; speed optimization targets `drop` mode.
- LongBench v1 no-chat tasks: `trec`, `triviaqa`, `samsum`, `lsht`, `lcc`, `repobench-p`.
- `min_decode_kv_len_for_paging=8192`: below this KV length, falls back to baseline decode attention.
- `max_unselected_compressed` (default `-1`): in compressed mode, limits how many unselected pages contribute compressed KV. `-1` = unlimited, `0` = drop all unselected (equivalent to drop mode), `N` = keep top-N by score.
- Parametric sweep scripts: `run_*.sh` files at repo root (`run_ruler_llama.sh`, `run_ruler_multipole.sh`, `run_longbench_v1_llama.sh`, `run_longbench_v2_llama.sh`, etc.).
- Baselines in `baselines/`: SEER Attention, Multipole Attention, Quest Attention. All eval scripts add `baselines/` to `sys.path` at startup so baseline packages are importable.
- **Quest Attention** (`baselines/quest_attn/`): uses a custom model class (not monkey-patching). Requires `quest_init()` after loading. Supports LLaMA-family models (Llama-2/3.x, Mistral) and Qwen3. Has custom CUDA kernels in `ops/csrc/` built via `build_kernels.sh`. Config in `quest_attn/config.py`. Model classes: `LlamaForCausalLM` (llama.py), `Qwen3ForCausalLM` (qwen3.py). Both share `QuestAttention` which conditionally applies QK-norm for Qwen3.
- `oracle/` folder contains standalone experiment scripts: `run_ruler_eval.py` (independent RULER runner), `eval_ruler_hybridmulti.py` (hybrid_multi scoring sweeps), `diagnose_l2_scoring.py` (proxy scoring diagnostics).
