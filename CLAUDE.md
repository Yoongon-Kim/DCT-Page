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

The repo also hosts side-by-side **baselines** (SeerAttention-R, Multipole, Quest, DuoAttention, InfLLM, ShadowKV), a suite of **oracle diagnostics** (attention-mass recall, scoring-method comparisons, oracle upper-bound sweeps), and **speed/profiling** tools.

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
| `num_sink_pages` | `0` | First N physical pages always attended (sink). 0 = no sink forcing. |
| `num_recent_pages` | `0` | Number of full recent pages always attended; EXCLUDES the currently-open partial page (open page is implicit, always also attended). |
| `compress_ratio` | `0.03125` | Per-page compression (e.g. 32 → 1 token) |
| `proxy_method` | `"dct"` | `"dct"` (lowpass-IDCT, default) `\| "haar"` (block-mean) `\| "haar_c2f"` (forward Haar coarse-to-fine, no inverse) `\| "dct_haar"` (DCT lowpass + Haar detail rows) `\| "harp"` (DCT-Page Adaptive Resolution: L_3 block-mean + H_3 detail, per-page top-K_d adaptive expansion at scoring) — empirically `dct` dominates all variants on RULER mk3 |
| `haar_detail_per_block` | `0` | For `haar`/`dct_haar`: number of per-block Haar detail rows. `0` = lowpass-only. With `haar_detail_with_negation`, each detail row gets a `−`-pair so `max`-scoring picks `\|Q·detail\|`. |
| `haar_detail_with_negation` | `False` | Duplicate every detail row with its negation. |
| `harp_detail_topk` | `4` | HARP: # blocks per page whose H_3 contributes via expansion at scoring time (top-K_d by `\|H_3\|` L2 norm). K=cs ≡ Haar 8 equiv; K=0 ≡ Haar 4. |
| `min_decode_kv_len_for_paging` | `8192` | Fallback to baseline decode attention below this KV length |
| `scoring_method` | `"max"` | `"mean" \| "max" \| "sum"` |
| `group_agg_method` | `"mean"` | `"mean" \| "max" \| "topp"` — GQA per-group aggregation |
| `unselected_mode` | `"drop"` | `"drop" \| "compressed"` |
| `compressed_token_rope` | `"mixed"` | `"mixed" \| "block_center"` — RoPE handling for compressed tokens |
| `continuous_rope` | `False` | Temporarily disabled |
| `score_use_quest_minmax` | `False` | Use QUEST-style per-channel min/max key metadata for scoring (mode B = Quest scoring + DCT-Page SDPA path) |
| `score_combine_quest_dct` | `False` | Best-rank fusion of DCT lowpass and Quest min/max scores. |
| `select_with_oracle_page_scores` | `False` | Debug/upper-bound: use full-page oracle scores for selection |
| `use_triton` | `True` | Fused Triton kernels (False = pure PyTorch) |
| `weight_compressed_by_population` | `False` | Scale unselected-page rep's softmax mass by `log(page_size/comp_size)` bias |
| `max_unselected_compressed` | `-1` | Max unselected pages contributing compressed KV (`-1`=all, `0`=drop-equivalent, `N`=top-N) |
| `comp_kv_quant` | `"none"` | Fake-quant of compressed K/V: `"fp8_e4m3" \| "fp8_e5m2" \| "int8" \| "int4"` |
| `comp_kv_quant_granularity` | `"per_page"` | `"per_page" \| "per_comp_token"` |
| `outlier_budget` | `0` | 0 = off; >0 = top-M outlier tokens per kv_head appended to assembled KV at decode (always-attended supplement to DCT page selection). |
| `outlier_detector` | `"lastq_mean"` | `"knorm"` (L2 norm of K — weak baseline) `\| "lastq_mean"` (top-M by GQA-mean of first-decode Q · K — picks query-aligned tokens) `\| "cluster_dyn"` (k-means clusters + per-step top-K cluster Q·centroid → Q·K refinement within selected clusters; best quality, ~50% decode overhead). |
| `cluster_outlier_N` | `256` | cluster_dyn: k-means cluster count per kv-head. N=1024 (≈ 1 page = 1 cluster) is the sweet spot on Qwen3-8B/32K — matches/exceeds DCT-only RULER overall (85.84 vs 85.24 at N=256), but +110% decode at 32k due to per-step Q·K refinement over T_pageable. |
| `cluster_outlier_iters` | `5` | k-means Lloyd iterations. |
| `cluster_outlier_top_k` | `8` | # of top clusters per step before Q·K refinement. |
| `cluster_outlier_q_agg` | `"mean"` | `"mean"` (group-mean Q first) `\| "max"` (per-qo-head dot, then max). `max` gives best cwe + lowest loop rate. |
| `cluster_outlier_scoring` | `"centroid"` | `"centroid"` (mean K, default) `\| "minmax"` (Quest-style upper-bound). `centroid` empirically wins — `minmax`'s over-optimism on noisy pages hurts. |

### Evaluation scripts (repo root)

| File | Benchmark | Supported modes |
|---|---|---|
| `eval_ruler.py` | RULER synthetic long-context (13 tasks × configurable seq_lengths, default 32k) | baseline, page_attention, seer_attention, seer_prefill, multipole_attention, quest_attention, duo_attention, shadowkv, inf_llm |
| `eval_longbench_v1.py` | LongBench v1 (16 English tasks, F1 / ROUGE-L / accuracy / code similarity) | baseline, page_attention, seer_attention, multipole_attention, quest_attention, duo_attention, inf_llm |
| `eval_longbench_v2.py` | LongBench v2 (503 multiple-choice, by difficulty/length) | baseline, page_attention, rope_gap, seer_attention, multipole_attention, quest_attention, duo_attention, inf_llm |
| `eval_aime25.py` | AIME 2025 (30 problems, pass@1) — **Qwen3-8B only** | baseline, page_attention, seer_attention, seer_prefill, multipole_attention, quest_attention, duo_attention, shadowkv |
| `eval_gpqa.py` | GPQA (diamond/main/extended, MC accuracy) — **Qwen3-8B only** | same set as AIME |

All eval scripts prepend `baselines/` to `sys.path` so baseline packages are importable.

Model support: **Llama 3.x** (`replace_llama_attn`) and **Qwen3** (`replace_qwen3_attn`, with q_norm/k_norm). Qwen2 patch exists but is not wired into modern eval scripts.

### `baselines/`

| Folder | Baseline | Model support | Notes |
|---|---|---|---|
| `duo_attn/` | DuoAttention (head streaming + recent window) | Llama 3.x only | Requires dedicated env: `transformers==4.45.2`, `flash-attn==2.6.3`, upstream `duo-attention` installed. Config: `duo_attn/config.py` (`pattern_root`, `pattern_subdir`, `sparsity`, `sink_size`, `recent_size`). |
| `infllm/` | InfLLM (retrieval-based block attention) | Llama 3.x only | Requires `transformers==4.37.2`, upstream `InfLLM` installed. Config: `infllm/config.py` (`attn_type`, `block_size`, `n_init`, `n_local`, `topk`, `repr_topk`, `max_cached_block`, `chunk_size`). **Note:** directory must not be named `inf_llm` — that collides with the upstream package and breaks the shim's internal `from inf_llm import patch_hf`. Any name without the underscore (e.g. `infllm`) is fine. |
| `seer_attn/` | SeerAttention-R (learned gate-based sparsity, decode-only + optional prefill) | Llama 3.x, Qwen2/3 | Has `decode_sparse/`, `prefill_sparse/`, `kernels/`, `modules/`. Configs: `config.py` (decode) and `prefill_config.py`. Loads HF checkpoints like `SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates`. |
| `multipole_attn/` | Multipole Attention (hierarchical k-means clustering) | Llama 3.x, Qwen2/3 | Modules: `attention_forward.py`, `clustering.py`, `kernels.py`, `kernel_wrappers.py`, `kmeans_ops_sequential.py`. Config: `percent_clusters_lst`, `percentiles_lst`, `use_replacement`, `cluster_interval`. |
| `quest_attn/` | Quest (per-page min/max key metadata) | Llama 2/3.x, Mistral, Qwen3 | Has its own model classes (`models/llama.py`, `models/qwen3.py`) — not monkey-patch based. Custom CUDA kernels under `ops/csrc/` built via `build_kernels.sh`. Config: `page_size`, `max_seq_len`, `token_budget`. Input must span ≥2 pages (i.e. >page_size tokens) at decode time or `controller.py:99` assertion fires. |
| `shadow_kv/` | ShadowKV (SVD-compressed key cache + CPU-offloaded V) | Llama 3.x only | Compiled C++/CUDA kernels in `build/`. Config: `cache_mode` (`shadowkv` or `shadowkv_cpu`), `sparse_budget`, `rank`, `chunk_size`. Qwen3 unsupported (no QK-norm in upstream Qwen2 class). |
| `snap_kv/` | SnapKV (prompt-attention-pooled KV compression) | Llama 3.x only | Vendored shim (`_vendor.py`) that monkey-patches `LlamaFlashAttention2`. Requires dedicated env: `transformers==4.37.2` + compatible `flash-attn`. Config: `snap_kv/config.py` (`window_size`, `max_capacity_prompt`, `kernel_size`, `pooling`). Invoked via `run_ruler_snapkv.sh` (currently only wired through that runner). |

### `oracle/` — diagnostics and oracle upper bounds

| File | Purpose |
|---|---|
| `oracle_ruler.py` | Standalone RULER runner for oracle experiments. Flat per-task JSONL output. |
| `diagnose_scoring_methods.py` | Compares ~30 scoring methods (oracle_max/mean, proxy_max/mean, l2_energy, dc_ac_*, spectral_recon_*, continuous_cosine_max, hybrid_*) against a configurable ground truth (`oracle_max` or `output_contribution`). |
| `attention_mass_recall_ruler.py` | Dense-trajectory reference: runs **unmodified full-KV forward**, observes Q/K/V per decode step, computes per-selector mass-recall (DCT, Quest, ShadowKV, oracle_max, mass-topk ceiling). Reports full-KV / selected-page / paged-only metric families. |
| `attention_mass_recall_ruler_quest.py` | Quest-specific variant of the mass-recall diagnostic. |
| `dc_ac_ruler.py` | Sweeps `dc_ac` / `proxy_dc_ac` scoring methods with lambda tuning on RULER (relies on removed scoring methods; kept for historical comparison). |
| `hybridmulti_ruler.py` | Sweeps the `hybrid_multi` budgeted scoring method (`M`, `alpha`). (Relies on removed scoring methods; kept for historical comparison.) |
| `oracle_hybrid_ruler.py` | Oracle-selection + hybrid-unselected sweeps (oracle pages kept as Haar lowpass proxy). |
| `run_ruler_oracle_selection.py` | Orchestrates oracle-selection upper-bound sweeps across page sizes at a fixed selected-token budget. |

### `speed/`

| File | Purpose |
|---|---|
| `speed_test_dummy.py` | Decode throughput benchmark with dummy (random) token inputs; measures baseline vs DCT. |
| `speed_test_dummy_multipole.py` | Legacy variant for Multipole Attention speed tests. |
| `profile_decode.py` | Per-stage decode-path timing with chained CUDA events (`qkv`, `score_cache_update`, `score_pages_kernel`, `topk`, `assemble_drop_and_final_k_original_rope`, `sdpa`, `o_proj`). Selects attention backend via `--attention_backend {sdpa,quest,flashinfer}`. |
| `profile_decode_flash_infer.py` | Eager-mode profiling driver that wires DCT-Page into an internal FlashInfer paged-decode wrapper; supports `torch.profiler` trace export to sidestep the chained-CUDA-event CPU-dispatch-lag bias of `profile_decode.py`. Modes: `baseline`, `dct_sdpa`, `dct_quest`, `dct_flashinfer`. |
| `profile_decode_upstream_flash_infer.py` | Same idea as the above but against the **upstream** FlashInfer paged-decode kernel (the path used by `eval_*.py --attention_backend upstream_flashinfer`). |
| `flashinfer_backend.py` | DCT-internal FlashInfer adapter (fork at `/home/yoongonkim/flashinfer-dct`) with per-head indices, native bf16, drop-mode only. |
| `upstream_flashinfer_backend.py` | Adapter onto stock upstream FlashInfer via a virtual-batch-per-(batch, KV head) layout. This is the path `eval_*.py --attention_backend upstream_flashinfer` selects. |
| `quest_backend.py` | DCT-Page → Quest paged-decode CUDA kernel adapter (used by `profile_decode.py --attention_backend quest`). |
| `run_speed_test_dummy.sh` | Wrapper that runs baseline + DCT configurations and prints a tok/s comparison table. |
| `run_speed_ps64.sh` (repo root) | Unified speed wrapper: baseline + current sparse default + ps32/topk64/comp4 candidate across a context-length sweep. Configurable via env vars (`CONTEXT_LENGTHS`, `RUN_COMP4`, `MODEL`, `CONDA_ENV`, …). |

### `benchmark/`

- `benchmark/data/` — prepared `longbench_v1_data/` and `ruler_data/`
- `benchmark/eval_ruler/` — RULER infrastructure (`data/prepare.py`, `eval/evaluate.py`, `synthetic.yaml`, `config_tasks.sh`). `pred/predict_dctpage.py` is a prediction-only path that mirrors the official RULER pipeline. `eval_ruler.py` resolves `RULER_DIR` to this folder (not the top-level `eval_ruler/`, which is a runtime cache).

### Other top-level utilities

- `verify_kernels.py` — numerical verification harness. For each Triton kernel in `triton_kernels.py`, runs both the Triton and PyTorch implementations on the same inputs and reports max/mean absolute error. Pass criteria: bf16 mean < 1e-2 / max < 0.1, fp32 mean < 1e-5 / max < 1e-3. Run with `CUDA_VISIBLE_DEVICES=0 python verify_kernels.py`.
- `observations/dct_page_energy.py` — empirical sanity check for the DCT-lowpass-IDCT proxy. Runs an unmodified prefill, takes a per-page DCT-II of K, and reports per-bin energy fraction and cumulative energy under a lowpass cutoff. Supports comparing multiple prior runs with `--compare_runs`.

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
| `run_ruler_infllm.sh` | RULER InfLLM | Sweeps `topk` / `block_size` / `n_local` by overriding `baselines/infllm/config.py`. Requires the InfLLM env. |
| `run_ruler_snapkv.sh` | RULER SnapKV | Sweeps `max_capacity_prompt` × `pooling`. Requires `SNAPKV_ENV_NAME` (default `snap_kv`) conda env with `transformers==4.37.2`. Llama-only (hard-fails on Qwen). |
| `run_ruler_1.sh`, `run_ruler_2.sh` | RULER DCT-Page (split sweeps) | Two halves of the `(page_size, top_k)` × `compress_ratio` grid; designed to be run in parallel on separate GPUs. |
| `run_longbench_v1.sh`, `run_longbench_v1_llama.sh`, `run_longbench_v1_seer.sh`, `run_longbench_v1_multipole.sh`, `run_longbench_v1_duo.sh`, `run_longbench_v1_infllm.sh` | LongBench v1 per method | — |
| `run_longbench_v2.sh`, `run_longbench_v2_llama.sh`, `run_longbench_v2_seer.sh`, `run_longbench_v2_multipole.sh`, `run_longbench_v2_duo.sh`, `run_longbench_v2_infllm.sh` | LongBench v2 per method | — |

## Commands

### Setup

```bash
pip install -r requirements.txt
# Core: torch 2.10.0+cu130, transformers 5.8.0, triton 3.6.0
# SeerAttention-R (in dct env): needs flash-attn 2.8.3 — source-build for CUDA 13:
#   CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=16 \
#   pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
# Quest CUDA kernels (in dct env): build for sm120 only:
#   cd baselines/quest_attn && CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
#   bash build_kernels.sh /home/yoongonkim/Quest/kernels
#   (ops/CMakeLists.txt uses RAFT header-only mode to avoid CUDA-13 CAGRA build failures)
# DuoAttention requires a separate env pinned to transformers==4.45.2 + flash-attn==2.6.3
# InfLLM requires a separate env pinned to transformers==4.37.2
# SnapKV requires a separate env pinned to transformers==4.37.2 + compatible flash-attn (Llama only)
# upstream_flashinfer attention backend assumes a working FlashInfer install in the active env
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
python oracle/diagnose_scoring_methods.py \
  --ground_truths oracle_max,output_contribution \
  --context_len 16384 \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct

# Dense-trajectory mass recall across selectors (DCT, Quest, ShadowKV, oracle_max)
python oracle/attention_mass_recall_ruler.py --context_len 32768 \
  --page_size 32 --top_k 64

# Oracle upper-bound selection sweep across page sizes (fixed selected-token budget)
python oracle/run_ruler_oracle_selection.py \
  --context_len 32768 --page_sizes 32,64,128 \
  --selected_token_budget 2048 --compress_ratio 0.03125

# Standalone RULER runner for ad-hoc experiments
python oracle/oracle_ruler.py --mode page_attention --context_len 16384 \
  --tasks niah_multikey_3 --tag my_run --num_samples 25 --cuda_device 0 \
  --dct_page_size 32 --dct_top_k 64 --dct_unselected_mode drop
```

## Conventions

- **Monkey-patch pattern**: set module-level `_dct_page_cfg` then patch `forward`. Always call `replace_*_attn()` **before** `from_pretrained()`.
- **Tensor naming**: `paged_*` = reshaped `[..., num_pages, page_size, ...]`; `comp_*` = compressed; `*_buf` = pre-allocated buffer.
- **Buffer caching**: projection matrices and kernel caches live on `attn_module` attributes (lazy init via `_get_or_build_*`, shape/device checked each call).
- **Triton kernels**: `@triton.jit` with constexpr block sizes; wrappers handle grid launch and switch to pure-PyTorch when `use_triton=False`.
- **Run naming convention**: encodes params, e.g. `drop_ps32_top64_comp1`, `qwen_ps32_topk64_cr0.125_drop_tokenropemixed_popw`, `llama_shadowkv_shadowkv_cpu_sb2192_r160_cs8`.
- **`--top_k` semantics**: in `eval_*.py` (page_attention mode) `--top_k` means TOTAL pages
  (sink + middle + recent); in `oracle/*` scripts `--dct_top_k` still means MIDDLE pages.
- **`--attention_backend` (eval `page_attention` mode, drop-only)**: `sdpa` = `torch.scaled_dot_product_attention` on assembled KV (production default); `upstream_flashinfer` = stock FlashInfer paged-decode kernel via a virtual-batch-per-(batch, KV head) layout. The flag's default in the parser is the misspelled `upastream_flashinfer` — passing nothing falls through the choice list. Use `--verify_upstream_fi` to enable per-layer SDPA shadow comparison (bf16 max-abs-diff noise floor ≈ 0.05).
- **No unit tests** for end-to-end behavior — validation is through benchmark runs (RULER / LongBench / AIME / GPQA) and the oracle diagnostics. `verify_kernels.py` is the only numerical regression harness and is scoped to individual Triton kernels.

## Data paths

- RULER synthetic data (on-disk cache from `benchmark/eval_ruler/data/prepare.py`):
  - `benchmark/data/ruler_data/{model_family}/{seq_len}/{task}/validation.jsonl` (canonical)
  - `results_ruler/data/synthetic/{seq_len}/` (legacy, used by some oracle scripts)
- LongBench v1: `longbench_v1_data/data/*.jsonl` or `benchmark/data/longbench_v1_data/*.jsonl`
- Results roots: `results/`, `results_ruler/`, `results_attention_mass_recall/`, `results_proxy_slice_overlap/`, `results_quest_mass_recall/`

## Notes

- **Score proxy**: DCT-lowpass-IDCT only. Haar, Walsh-Hadamard, direct-spectral, and alternate frequency layouts have been removed.
- **Supported `scoring_method`**: `"max"`, `"mean"`, `"sum"` (and the QUEST-style min/max variant via `score_use_quest_minmax=True`). `dc_ac`, `spectral_recon_max`, `hybrid_multi` scoring methods were removed; the `oracle/dc_ac_ruler.py` and `oracle/hybridmulti_ruler.py` sweep wrappers remain but are no longer functional without those scoring methods.
- **`drop` vs `compressed`**: `drop` is the speed path; `compressed` is for accuracy experiments.
- **`min_decode_kv_len_for_paging=8192`**: below this KV length, the patch falls back to baseline decode attention.
- **`max_unselected_compressed`** (default `-1`): caps how many unselected pages contribute compressed KV. `-1`=unlimited, `0`=drop-equivalent, `N`=top-N by score.
- **LongBench v1 no-chat tasks**: `trec`, `triviaqa`, `samsum`, `lcc`, `repobench-p`.
- **AIME25 / GPQA** are Qwen3-8B only and shell out to the RULER eval helpers in `eval_ruler.py` for monkey-patching; CLI choices expose the full mode list for argparse parity but guard against non-Qwen3 at runtime.
- **Quest baseline** is not monkey-patched — it loads its own `LlamaForCausalLM` / `Qwen3ForCausalLM` classes and must call `quest_init()` after model load. Needs the compiled CUDA extension from `baselines/quest_attn/build_kernels.sh`.
- **ShadowKV, DuoAttention, InfLLM** do not yet support Qwen3 and only run with Llama 3.x.

## Diagnostic findings (2026-05, Qwen3-8B 32K)

- **DCT 4 baseline** (cr=0.125, sink=1, recent=4, top_k=64, scoring=max, group_agg=max): mk3=76, cwe=45.6. Re-validated after all proxy/outlier code additions (`qwen_dct4_smax_gmax_recheck`).
- **Proxy-method sweep summary**: every Haar/HARP/dct_haar variant tested is ≤ DCT 4 on mk3. Detail rows hurt under `max` scoring (sign cancellation). Adaptive expansion (HARP K=2..8) ≈ Haar 8 ≈ DCT 4 — no meaningful gain from changing proxy alone.
- **Scoring/group_agg 2×2 (`oracle/attention_mass_recall_ruler.py` default `score=max group=max`)**: `max/max` is needle-friendly (mk3=76); switching either axis to `mean` collapses mk3 (52, 20, 24).
- **Cluster_dyn outlier (real lever)**: DCT 4 + cluster_dyn (N=1024, K_top=8, q_agg=max, scoring=centroid) → cwe **+27.2** (45.6 → 72.8) at mk3=76 maintained. Full-RULER overall 85.84 (vs ceiling 86.06). Decode +83~110% slowdown at 32k (`speed/bench_cluster_outlier.py`).
- **Mass_recall is a misleading selector-quality metric for needle tasks**: averaged over 20 decode steps and over softmax mass, Quest gets higher mass_recall (0.83) and top1_hit (0.72) than DCT (0.79, 0.51) but mk3 RULER score is 8 vs 76. Page mass peaks on sink-like / structural tokens, not the needle.
- **set_recall is closer to RULER**: DCT 0.46 > Quest 0.31 aligns with mk3 direction. Absolute values are still moderate (DCT's 0.46 still beats Quest's 0.31 because DCT spreads picks across mid-rank oracle pages, not because it matches oracle top-1 better).
- **Rank-resolved hit profile (`per_task[*]['rank_profile']` in mass_recall summary)** is the cleanest paper figure: Quest peaks at r0 (0.72) and crashes at r4-r16 (0.33-0.44); DCT sits flatter (~0.55-0.66 across r0-r16); InfLLM (paper-faithful) sits between.
- **CMR_α metric** (mass^α-weighted hit) added (`cmr2_*`, `cmr4_*`, `top1_*`): α→∞ = oracle top-1 hit. Surprisingly Quest has higher `top1_hit` than DCT — confirms the "oracle top-1 is sink, not needle" structural artifact. Use rank profile + set_recall, not mass-weighted metrics, when correlating with RULER.
- **`compute_infllm_scores` was a Q-aware substitute** (per-step top-R by current Q·K). Paper-faithful version (`compute_infllm_paper_scores` + `_compute_prefill_rm`) is now wired: fires on the prefill-phase payload (`"phase": "prefill"`) from `_install_recording_forward`, computes r_m via local-window-attention (Xiao 2024 Eq. 1), selects per-page representatives, then dots current Q against the fixed reps each decode step. CLI: `--infllm_local_window` (paper l_L = 2K Vicuna / 4K LLaMA-Mistral, default 2048). Q-aware variant kept as fallback if prefill payload is missing.
- **Practical InfLLM prefill cost note**: r_m needs Q·K dot products across the local window, which vanilla FlashAttention does NOT expose. Paper relies on chunked prefill (encoding chunk = 512) or a custom kernel. DCT/Quest impose no such prefill constraint and slot into stock FlashAttention prefill directly.
- **Two oracle definitions, both reported as ceilings** (RULER 32K, `top_k=64`, smax/gmax no-softmax, both with group_agg=max for consistency):
  - **mass-oracle (canonical recall ceiling)**: top-K pages by Σ_t exp(q·K[p, t]) per page (= softmax-mass per page). This is the **Bayes-optimal selector for attention output approximation** — by construction, the K pages that preserve the most softmax mass of the full-attention output. Measured RULER: mk3=76 / cwe=70. Use as the **default recall denominator** because it is the theoretically principled "attention output fidelity" ceiling, task-agnostic.
  - **oracle_max (RULER-mk3-aligned ceiling)**: top-K pages by max(q·K) per page → group_agg=max. This is the **Bayes-optimal selector for single-token retrieval** (the page containing the strongest sharp-peak token). Measured RULER: mk3=80 / cwe=72.4. Specifically aligned with mk3-style needle tasks where the answer is a single high-spike token.
  - **Why both differ on RULER**: mk3 (needle retrieval) rewards oracle_max because the answer is a single-token spike; reasoning/multi-evidence tasks (LongBench v2) reward mass-oracle because the attention output must preserve broad context evidence. Empirical: LongBench v2 final on Qwen3-8B budget=2048 — Quest-min/max (max-upper-bound proxy that empirically aligns with high-mass page picks) **34.8%** vs DCT cs=4 (lowpass, max-aligned) **33.0%** — Quest's mass-favored selection wins on reasoning, opposite of mk3 ordering.
- **Quest is NOT mass-aligned**, despite empirically picking high-mass pages. Its scoring formula `Σ_d max(q[d]·K_max[p,d], q[d]·K_min[p,d])` = per-channel **optimistic upper bound on max(q·K)** for any token in the page. Pages with wide channel-wise K range (large min-max gap) get high Quest scores — these happen to be attention-active "sink-like" pages with high softmax mass, but the mechanism is *max-upper-bound estimation*, not mass measurement. The bias toward mass-heavy pages is a correlation, not a definition.
- **Canonical "attention recall" ceiling = mass-oracle (group_agg=max)**, NOT oracle_max. `oracle/attention_mass_recall_ruler.py` emits `mass_recall_oraclemax` / `selected_mass_oraclemax` (RULER-mk3-aligned) AND `mass_recall_proxy` etc. against mass-topk (canonical). `attention_recall_X = selected_mass_X / selected_mass_mass_topk` is the canonical ratio (task-agnostic). `set_recall_proxy` (vs mass_topk) is the canonical set-overlap metric; `set_recall_proxy_vs_oraclemax` is reported as a secondary RULER-mk3-aligned diagnostic. Rank-resolved hit profile is emitted in both orderings.
- **cs vs RULER (smax/gmax no-softmax)**: cs=4 → mk3=76, cs=32 → mk3=80 (cs=32 = identity proxy = oracle_max selection). 4-point compression cost is the gap that proxy improvements / oracle-ranking convergence buy. CLAUDE-internal naming: this is the *compression-induced selection loss*. Separately, switching group_agg to mean+softmax (smax/gmean+softmax) at the SAME cs costs an additional 8 mk3 points (cs=32 mk3 = 72 with that head-aggregation scheme even though sr_vs_oraclemax=1.0 by construction) — *head-aggregation choice loss* is a different axis and currently a bigger lever than compression for retrieval tasks.
- **Adaptive bin selection thoroughly explored, all fail at cs=4 budget** (RULER 32K mk3, set_recall_vs_oraclemax):
  - Fixed band-pass `[0,8,16,24]` spread / `[0,1,30,31]` extremes / `[1,8,16,24]` DC-skip: all WORSE than lowpass `[0,1,2,3]` (0.487). DC bin dominant (23% pick share in any adaptive scheme), bin 16+ near-noise.
  - **Q-aware top-N + freq-domain `sum_abs`**: 0.241 at top=32 (full info) — formulation doesn't approximate max(Q·K).
  - **Q-aware top-N + signed-bin IDCT `recon_max`**: 0.516 at top=4 (+0.029 over lowpass) — **the only working adaptive variant**, but requires cs=page_size storage (full info, not feasible).
  - **K-aware ||DCT(K)||₂ top-N (production-feasible)**: 0.477 at top=4 — ~ lowpass.
  - **lastq (Q from first decode step) top-N**: 0.481 at top=4 — Q-drift kills the +0.029 Q-aware advantage. Cliff at small cs, +0.013 at cs=8.
  - **Conclusion**: lowpass cs=4 is *essentially optimal* among Q-independent paged-storage proxies. Adaptive bin selection within compressed storage is a dead-end.
- **per_head_union group_agg drops mk3 -20, cwe -12** (RULER 32K, DCT cs=4): selection rule that lets each qo-head pick its own K/G pages with kv-head set = union shrinks the effective per-kv-head set when heads agree (which they do for needle tasks). Even the per_head_union *oracle* (cs=32) gives mk3=76 vs max-oracle mk3=80 — head aggregation choice fundamentally hurts retrieval.
- **per-layer needle_hit diagnostic (`oracle/needle_hit_diagnostic.py`)**: directly measures whether the page containing the RULER answer UUID is in each selector's top-K, per (layer, decode_step, kv-head). Key findings on Qwen3-8B 32K mk3:
  - **Retrieval layers = 16-34** (peak at layer 24: proxy hits 0.986, oracle 0.953). Layer 35 drops to 0.29 (output prep).
  - **At late layers (24-35), DCT proxy (0.663) BEATS oracle_max (0.619) on needle hit** — counter-intuitive but consistent: oracle ranks pages by single-token max(q·K), which can pick non-needle pages with stronger syntactic spikes; DCT lowpass averages within-page, downweighting one-token spikes and surfacing needle pages whose tokens are semantically packed.
  - **needle_hit ordering ≠ RULER ordering**: shadowkv (0.459) > proxy (0.454) > infllm (0.445) > oracle (0.442) > quest (0.389) aggregate. ShadowKV (Llama-only on RULER) and proxy are essentially tied on this metric.
  - **The 4-point RULER mk3 gap (76 → 80 oracle) is NOT from needle selection** (proxy actually beats oracle here). It's from **supporting-context page coverage** — pages with intermediate q·K containing the question line, key-match line, surrounding context for answer formation. Oracle_max's broader ranking surfaces those better than DCT lowpass's smoothed ranking.
  - **Per-decode-step**: needle hit grows monotonically across decode steps (0.35 → 0.54 from step 0 → 3). Model commits to retrieval as it generates answer tokens.
  - **Implication for paper**: selector-level mk3 levers are exhausted. The remaining gap is structural — addressable only via outlier banks or supporting-context augmentation, not by tweaking the page-scoring function.
- **Main config locked: DCT lowpass cs=4 + scoring=max + group_agg=max + sink=1 + recent=4 + drop mode** (Qwen3-8B 32K RULER: mk3=76, cwe=45.6). This is the production recommended baseline. Subsequent levers explored and deferred:
  - **`scoring=lse` (log-sum-exp over comp axis)**: free win, mk3=76 preserved, cwe=48.0 (+2.4). LSE ≈ max + log(spread) → spike-dominant page ranking preserved while spread bonus boosts mass-aligned pages. *Status: implemented but not default; deferred to future ablation.*
  - **`dense_first_n_layers=2` + LSE**: mk3=80 (= oracle ceiling, +4 from baseline), cwe=48.0. Layers 0,1 dense (Quest-baseline pattern) recovers the 4-point mk3 gap. *Status: implemented but not default; layers 0/1 dense incurs full attention compute on a per-decode-step basis — speed cost not yet measured. Deferred.*
  - **`km_quest_split=8` (K-M union with Quest mass-heavy supplement)**: mk3=80, cwe=48.0 — matches LSE+dense_first_2 result with a different lever. DCT picks top-(K-M)=56 + Quest picks 8 mass-heavy pages NOT in DCT's set. Recall diagnostics: attn_recall mk3 0.581 → 0.769 (+19), top1 0.552 → 0.832 (+28). Production-feasible (Quest min/max metadata cacheable per-layer). *Status: implemented but not default; deferred.*
  - **`score_combine_quest_dct` (best-rank fusion DCT+Quest)**: failed — mk3=60 (-16, Quest's mass-heavy picks bump out DCT's needle picks). *Status: not recommended.*
  - **`per_head_union` group_agg**: failed — mk3=56 (-20). *Status: not recommended.*
- **Future direction (open lever): `cluster_dyn` outlier bank**. Already known: DCT cs=4 + cluster_dyn (N=1024, K_top=8, q_agg=max, scoring=centroid) → cwe **+27.2** (45.6 → 72.8) at mk3=76 maintained. Full-RULER overall 85.84 (vs ceiling 86.06). Decode +83~110% slowdown at 32k. Open question: can cluster_dyn variants also move mk3? Pending sweep on cluster_dyn N/K_top/q_agg/scoring tuned for mk3-style needle retrieval.
