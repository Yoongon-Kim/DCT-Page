# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

DCT-Page is a research platform for **decode-time sparse page attention** on long-context LLMs. During autoregressive decoding it divides the KV cache into fixed-size pages, scores each page with a **DCT-lowpass-IDCT** proxy, selects top-k pages for full attention, and either drops or compresses the rest. Prefill uses standard full attention unchanged.

KV layout at decode time:
```
[sink (first S pages)] [page 0] [page 1] ... [page P-1] [recent (last R pages, includes open page)]
```

Modes:
- **`drop`** (default): unselected pages are discarded; attention = sink + recent + top-k selected (fastest).
- **`compressed`** (aka hybrid): unselected pages contribute DCT-lowpass-IDCT compressed KV tokens in addition to the top-k selected full pages (quality floor).

The repo also hosts side-by-side **baselines** (SeerAttention-R, Multipole, Quest, DuoAttention, InfLLM), a suite of **oracle diagnostics** (attention-mass recall, scoring-method comparisons, oracle upper-bound sweeps), and **speed/profiling** tools.

## Architecture

### Core modules (repo root)

| File | Role |
|---|---|
| `config.py` | `DCTPageConfig` dataclass with all hyperparameters |
| `dct_page_attention.py` | Main attention forward, compression, RoPE, monkey-patch entry points (~1600+ lines) |
| `triton_kernels.py` | Fused Triton kernels for score / topk / KV-assemble / RoPE with PyTorch fallbacks (~2100 lines) |

Key functions in `dct_page_attention.py`:
- `dct_page_attention_forward()` — replacement forward. Prefill: standard attention + KV pre-allocation. Decode: score pages → topk → assemble → SDPA.
- `replace_llama_attn()` (line 1528), `replace_qwen2_attn()` (line 1390), `replace_qwen3_attn()` (line 1458) — monkey-patch entry points. Call **before** `from_pretrained()`.
- `_update_comp_cache()` — incremental DCT-lowpass-IDCT page compression; only processes new pages each step.
- `_build_dct_projection_matrix()` — builds the lowpass-IDCT projection used for both scoring and compressed-mode KV.
- `PreAllocatedLayer` — fixed-stride KV buffer replacing `DynamicLayer` for O(1) decode append.

Key kernels in `triton_kernels.py` (every kernel has a PyTorch fallback):
- `score_pages_triton` (+ `_score_pages_fused_kernel`, specialized `*_c4_g4`, `*_c1_g4` variants)
- `topk_sort_triton` / `_topk_sort_kernel` (parallel bitonic topk)
- `assemble_kv_split_triton` (hybrid mode) / `assemble_kv_drop_triton` (drop mode), both reusing `_copy_full_segments_kernel`; `build_assemble_stride_cache` caches strides for the split path
- `apply_rope_q_direct` (single-token decode query RoPE, zero-alloc)

### `DCTPageConfig` fields (full list, from `config.py`)

| Field | Default | Purpose |
|---|---|---|
| `page_size` | `32` | Tokens per page |
| `top_k` | `64` | Pages selected for full attention |
| `num_sink_pages` | `1` | First N physical pages always attended (sink) |
| `num_recent_pages` | `5` | Number of full recent pages always attended; EXCLUDES the currently-open partial page (open page is implicit, always also attended). Total recent slots = `num_recent_pages + 1`. |
| `compress_ratio` | `0.03125` | Per-page compression (e.g. 32 → 1 token) |
| `min_decode_kv_len_for_paging` | `8192` | Fallback to baseline decode attention below this KV length |
| `scoring_method` | `"max"` | `"mean" \| "max" \| "sum"` |
| `group_agg_method` | `"mean"` | `"mean" \| "max" \| "topp"` — GQA per-group aggregation |
| `unselected_mode` | `"drop"` | `"drop" \| "compressed"` |
| `compressed_token_rope` | `"mixed"` | `"mixed" \| "block_center"` — RoPE handling for compressed tokens |
| `continuous_rope` | `False` | Temporarily disabled |
| `score_use_quest_minmax` | `False` | Use QUEST-style per-channel min/max key metadata for scoring |
| `select_with_oracle_page_scores` | `False` | Debug/upper-bound: use full-page oracle scores for selection |
| `use_triton` | `True` | Fused Triton kernels (False = pure PyTorch) |
| `weight_compressed_by_population` | `False` | Scale unselected-page rep's softmax mass by `log(page_size/comp_size)` bias |
| `max_unselected_compressed` | `-1` | Max unselected pages contributing compressed KV (`-1`=all, `0`=drop-equivalent, `N`=top-N) |
| `comp_kv_quant` | `"none"` | Fake-quant of compressed K/V: `"fp8_e4m3" \| "fp8_e5m2" \| "int8" \| "int4"` |
| `comp_kv_quant_granularity` | `"per_page"` | `"per_page" \| "per_comp_token"` |

### Evaluation scripts (repo root)

| File | Benchmark | Supported modes |
|---|---|---|
| `eval_ruler.py` | RULER synthetic long-context (13 tasks × configurable seq_lengths, default 32k) | baseline, page_attention, seer_attention, seer_prefill, multipole_attention, quest_attention, duo_attention, inf_llm |
| `eval_longbench_v1.py` | LongBench v1 (16 English tasks, F1 / ROUGE-L / accuracy / code similarity) | baseline, page_attention, seer_attention, multipole_attention, quest_attention, duo_attention, inf_llm |
| `eval_longbench_v2.py` | LongBench v2 (503 multiple-choice, by difficulty/length) | baseline, page_attention, rope_gap, seer_attention, multipole_attention, quest_attention, duo_attention, inf_llm |
| `eval_aime25.py` | AIME 2025 (30 problems, pass@1) — **Qwen3-8B only** | baseline, page_attention, seer_attention, seer_prefill, multipole_attention, quest_attention, duo_attention |
| `eval_gpqa.py` | GPQA (diamond/main/extended, MC accuracy) — **Qwen3-8B only** | same set as AIME |

All eval scripts prepend `baselines/` to `sys.path` so baseline packages are importable.

Model support: **Llama 3.x** (`replace_llama_attn`) and **Qwen3** (`replace_qwen3_attn`, with q_norm/k_norm). Qwen2 patch exists but is not wired into modern eval scripts.

### `baselines/`

| Folder | Baseline | Model support | Notes |
|---|---|---|---|
| `duo_attn/` | DuoAttention (head streaming + recent window) | Llama 3.x only | Requires dedicated env: `transformers==4.45.2`, `flash-attn==2.6.3`, upstream `duo-attention` installed. Config: `duo_attn/config.py` (`pattern_root`, `pattern_subdir`, `sparsity`, `sink_size`, `recent_size`). |
| `infllm/` | InfLLM (retrieval-based block attention) | Llama 3.x only | Targets **transformers 5.5.4** (the DCT_Page main env). **Self-contained** — upstream `inf_llm` package vendored under `baselines/infllm/upstream/`, no separate `pip install -e /home/yoongonkim/InfLLM/` and no separate `infllm` conda env needed. Config: `infllm/config.py` (`attn_type`, `block_size`, `n_init`, `n_local`, `n_recent`, `topk`, `repr_topk`, `max_cached_block`, `chunk_size`). The `n_recent` knob (default None → uses `n_local`) decouples the local-output sliding window from the block-scoring horizon; only the torch-impl path supports it (fattn=True asserts at construction time). **Note:** wrapper directory must remain named `infllm` (no underscore) to avoid ambiguity with the upstream `inf_llm` import path. |
| `seer_attn/` | SeerAttention-R (learned gate-based sparsity, decode-only + optional prefill) | Llama 3.x, Qwen2/3 | Has `decode_sparse/`, `prefill_sparse/`, `kernels/`, `modules/`. Configs: `config.py` (decode) and `prefill_config.py`. Loads HF checkpoints like `SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates`. |
| `multipole_attn/` | Multipole Attention (hierarchical k-means clustering) | Llama 3.x, Qwen2/3 | Modules: `attention_forward.py`, `clustering.py`, `kernels.py`, `kernel_wrappers.py`, `kmeans_ops_sequential.py`. Config: `percent_clusters_lst`, `percentiles_lst`, `use_replacement`, `cluster_interval`. |
| `quest_attn/` | Quest (per-page min/max key metadata) | Llama 2/3.x, Mistral, Qwen3 | Has its own model classes (`models/llama.py`, `models/qwen3.py`) — not monkey-patch based. Custom CUDA kernels under `ops/csrc/` built via `build_kernels.sh`. Config: `page_size`, `max_seq_len`, `token_budget`. |

### `observations/` — diagnostics, observations, and oracle upper bounds

| File | Purpose |
|---|---|
| `dct_page_energy.py` | Standalone observation tool: runs a model and measures per-page DCT-lowpass energy of K/V (proxy sanity/visualization). |
| `oracle_ruler.py` | Standalone RULER runner for oracle experiments. Flat per-task JSONL output. |
| `scoring_method_recall.py` | Compares ~30 scoring methods (oracle_max/mean, proxy_max/mean, l2_energy, dc_ac_*, spectral_recon_*, continuous_cosine_max, hybrid_*) against a configurable ground truth (`oracle_max` or `output_contribution`). |
| `attention_mass_recall_ruler.py` | Dense-trajectory reference: runs **unmodified full-KV forward**, observes Q/K/V per decode step, computes per-selector mass-recall (DCT, Quest, ShadowKV, oracle_max, mass-topk ceiling). Reports full-KV / selected-page / paged-only metric families. Two experiments via `--mode`: `full` (default, multi-selector) and `quest` (Quest-native geometry: last-page keep, skip layers 0–1). |
| `oracle_hybrid_ruler.py` | Oracle-selection + hybrid-unselected sweeps (unselected pages kept as DCT-lowpass-IDCT compressed proxy). |

### `speed/`

| File | Purpose |
|---|---|
| `speed_test_dummy.py` | Decode throughput benchmark with dummy (random) token inputs; measures baseline vs DCT. |
| `speed_test_dummy_multipole.py` | Legacy variant for Multipole Attention speed tests. |
| `profile_decode.py` | Per-stage decode-path timing with chained CUDA events (`qkv`, `score_cache_update`, `score_pages_kernel`, `topk`, `assemble_drop_and_final_k_original_rope`, `sdpa`, `o_proj`). |
| `run_speed_test_dummy.sh` | Wrapper that runs baseline + DCT configurations and prints a tok/s comparison table. |

### `benchmark/`

- `benchmark/data/` — prepared `longbench_v1_data/` and `ruler_data/`
- `benchmark/eval_ruler/` — RULER infrastructure (`data/prepare.py`, `eval/evaluate.py`, `synthetic.yaml`, `config_tasks.sh`). `pred/predict_dctpage.py` is a prediction-only path that mirrors the official RULER pipeline.

### Run scripts (`run_*.sh` at repo root)

Sweep scripts — each invokes `eval_ruler.py` / `eval_longbench_v{1,2}.py` with a parameter grid and `--skip_existing` so interrupted runs resume cleanly.

| Script | Calls | Notes |
|---|---|---|
| `run_ruler.sh` | RULER DCT-Page | Default `Qwen/Qwen3-8B`, sweeps `(page_size,top_k)` × `compress_ratio` × `unselected_mode` × `compressed_token_rope` × `weight_compressed_by_population`. |
| `run_ruler_llama.sh` | RULER DCT-Page | Llama variant of the above. |
| `run_ruler_seer.sh` | RULER SeerAttention-R | Sweeps `token_budget`. |
| `run_ruler_multipole.sh` | RULER Multipole | Sweeps `percent_clusters`, `percentiles`, `use_replacement`. |
| `run_ruler_duo.sh` | RULER DuoAttention | Sweeps `sparsity`; requires `duo_env`. Rewrites `baselines/duo_attn/config.py` in place. |
| `run_ruler_quest.sh` | RULER Quest-minmax | Runs `page_attention` with `--score_use_quest_minmax`; launches LLaMA on GPU 2 and Qwen3 on GPU 3 in parallel. |
| `run_longbench_v1.sh`, `run_longbench_v1_llama.sh`, `run_longbench_v1_seer.sh`, `run_longbench_v1_multipole.sh`, `run_longbench_v1_duo.sh` | LongBench v1 per method | — |
| `run_longbench_v2.sh`, `run_longbench_v2_llama.sh`, `run_longbench_v2_seer.sh`, `run_longbench_v2_multipole.sh`, `run_longbench_v2_duo.sh` | LongBench v2 per method | — |

## Commands

### Setup

```bash
pip install -r requirements.txt
# Core (DCT_Page conda env): torch 2.11.0, transformers 5.5.4, triton 3.6.0
# DuoAttention requires a separate env pinned to transformers==4.45.2 + flash-attn==2.6.3
# InfLLM now runs in the main env (transformers 5.5.4); the legacy `infllm` conda env is retired
# Quest needs baselines/quest_attn/build_kernels.sh for the CUDA extension
```

### RULER

```bash
# DCT-Page (drop)
python eval_ruler.py --mode page_attention \
  --base_model Qwen/Qwen3-8B \
  --seq_lengths 32768 --num_samples 25 \
  --page_size 32 --top_k 64 --compress_ratio 0.125 \
  --unselected_mode drop --output_dir results_ruler --run_name qwen_drop_ps32_t64

# Baseline
python eval_ruler.py --mode baseline --base_model Qwen/Qwen3-8B \
  --seq_lengths 32768 --output_dir results_ruler --run_name baseline

# Baseline sweep
bash run_ruler.sh            # Qwen3-8B default
bash run_ruler_llama.sh      # Llama-3.1-8B-Instruct

# Other methods (may need a dedicated env; see baselines/<name>/config.py)
bash run_ruler_seer.sh
bash run_ruler_multipole.sh
bash run_ruler_duo.sh         # activates duo_env
bash run_ruler_quest.sh       # Quest-minmax, parallel on two GPUs
```

### LongBench

```bash
python eval_longbench_v1.py --mode page_attention \
  --base_model Qwen/Qwen3-8B \
  --page_size 32 --top_k 64 --compress_ratio 0.03125 \
  --unselected_mode drop \
  --output_dir results/longbench_v1 --run_name drop_ps32_top64_comp1

python eval_longbench_v2.py --mode baseline \
  --base_model Qwen/Qwen3-8B \
  --output_dir results/longbench_v2 --run_name baseline
```

### AIME / GPQA (Qwen3-8B only)

```bash
python eval_aime25.py --mode page_attention --max_new_tokens 16384 \
  --page_size 32 --top_k 64 --unselected_mode drop \
  --output_dir results_aime25 --run_name aime25_drop

python eval_gpqa.py --mode page_attention --gpqa_subset diamond \
  --max_new_tokens 8192 \
  --page_size 32 --top_k 64 --unselected_mode drop \
  --output_dir results_gpqa --run_name gpqa_drop
```

### Speed / profiling

```bash
bash speed/run_speed_test_dummy.sh

python speed/speed_test_dummy.py --mode dct \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --context_lengths 8192,16384,32768,65536 \
  --page_size 32 --top_k 64 --compress_ratio 0.03125 \
  --unselected_mode drop

python speed/profile_decode.py --context_length 32768 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --page_size 32 --top_k 64
```

### Oracle diagnostics

```bash
# Scoring-method comparison (no DCT patch; uses full-KV ground truth)
python observations/scoring_method_recall.py \
  --ground_truths oracle_max,output_contribution \
  --context_len 16384 \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct

# Dense-trajectory mass recall across selectors (DCT, Quest, ShadowKV, oracle_max)
python observations/attention_mass_recall_ruler.py --context_len 32768 \
  --page_size 32 --top_k 64

# Standalone RULER runner for ad-hoc experiments
python observations/oracle_ruler.py --mode page_attention --context_len 16384 \
  --tasks niah_multikey_3 --tag my_run --num_samples 25 --cuda_device 0 \
  --dct_page_size 32 --dct_top_k 64 --dct_unselected_mode drop
```

## Conventions

- **Monkey-patch pattern**: set module-level `_dct_page_cfg` then patch `forward`. Always call `replace_*_attn()` **before** `from_pretrained()`.
- **Tensor naming**: `paged_*` = reshaped `[..., num_pages, page_size, ...]`; `comp_*` = compressed; `*_buf` = pre-allocated buffer.
- **Buffer caching**: projection matrices and kernel caches live on `attn_module` attributes (lazy init via `_get_or_build_*`, shape/device checked each call).
- **Triton kernels**: `@triton.jit` with constexpr block sizes; wrappers handle grid launch and switch to pure-PyTorch when `use_triton=False`.
- **Run naming convention**: encodes params, e.g. `drop_ps32_top64_comp1`, `qwen_ps32_topk64_cr0.125_drop_tokenropemixed_popw`.
- **`--top_k` semantics**: in `eval_*.py` (page_attention mode) `--top_k` means TOTAL pages
  (sink + middle + recent); in `observations/*` scripts `--dct_top_k` still means MIDDLE pages.
- **No unit tests**: validation is through benchmark runs (RULER / LongBench / AIME / GPQA) and the oracle diagnostics.

## Data paths

- RULER synthetic data (on-disk cache from `benchmark/eval_ruler/data/prepare.py`):
  - `benchmark/data/ruler_data/{model_family}/{seq_len}/{task}/validation.jsonl` (canonical)
  - `results_ruler/data/synthetic/{seq_len}/` (legacy, used by some oracle scripts)
- LongBench v1: `longbench_v1_data/data/*.jsonl` or `benchmark/data/longbench_v1_data/*.jsonl`
- Results roots: `results/`, `results_ruler/`, `results_attention_mass_recall/`, `results_proxy_slice_overlap/`, `results_quest_mass_recall/`

## Notes

- **Score proxy**: DCT-lowpass-IDCT only. Haar, Walsh-Hadamard, direct-spectral, and alternate frequency layouts have been removed.
- **Supported `scoring_method`**: `"max"`, `"mean"`, `"sum"` (and the QUEST-style min/max variant via `score_use_quest_minmax=True`). `dc_ac`, `spectral_recon_max`, `hybrid_multi` scoring methods were removed (their `observations/` sweep wrappers were deleted along with them).
- **`drop` vs `compressed`**: `drop` is the speed path; `compressed` is for accuracy experiments.
- **`min_decode_kv_len_for_paging=8192`**: below this KV length, the patch falls back to baseline decode attention.
- **`max_unselected_compressed`** (default `-1`): caps how many unselected pages contribute compressed KV. `-1`=unlimited, `0`=drop-equivalent, `N`=top-N by score.
- **LongBench v1 no-chat tasks**: `trec`, `triviaqa`, `samsum`, `lcc`, `repobench-p`.
- **AIME25 / GPQA** are Qwen3-8B only and shell out to the RULER eval helpers in `eval_ruler.py` for monkey-patching; CLI choices expose the full mode list for argparse parity but guard against non-Qwen3 at runtime.
- **Quest baseline** is not monkey-patched — it loads its own `LlamaForCausalLM` / `Qwen3ForCausalLM` classes and must call `quest_init()` after model load. Needs the compiled CUDA extension from `baselines/quest_attn/build_kernels.sh`.
- **DuoAttention, InfLLM** do not yet support Qwen3 and only run with Llama 3.x.
