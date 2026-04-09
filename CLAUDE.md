# DCT-Page

Decode-time sparse page attention for long-context LLMs with training-free Haar/DCT proxy scoring and fused Triton kernels.

## What it does

During autoregressive decoding (q_len=1), DCT-Page divides the KV cache into fixed-size **pages** (default 32 tokens), scores each page with a lightweight Haar lowpass or DCT proxy (compressed 32:1), selects the top-k most relevant pages for full attention, and either drops or compresses the rest. Prefill uses standard full attention unchanged.

KV layout at decode time:
```
[sink (4 tokens)] [page 0] [page 1] ... [page N-1] [recent (128 tokens)]
```

- **Drop mode** (default): unselected pages are discarded entirely (fastest).
- **Compressed/hybrid mode**: unselected pages use Haar/DCT compressed KV (maintains a quality floor).

## Repo structure

```
dct_page_attention.py   Core attention forward, compression, RoPE, monkey-patching
triton_kernels.py       Fused Triton kernels: score, topk, assemble, RoPE
config.py               DCTPageConfig dataclass with all hyperparameters

eval_ruler.py           RULER benchmark evaluation (synthetic long-context tasks)
eval_longbench_v1.py    LongBench v1 (16 English tasks, F1/ROUGE/accuracy)
eval_longbench_v2.py    LongBench v2 (multiple-choice format)
run_*.sh                Shell scripts for parametric sweeps

baselines/              Multipole Attention and SEER Attention implementations
benchmark/              RULER benchmark data prep and eval utilities
oracle/                 Oracle/upper-bound evaluation and proxy diagnostics
speed/                  Decode throughput and per-layer profiling scripts
results/                Benchmark outputs (RULER, LongBench, speed)
```

## Key modules

### dct_page_attention.py
- `dct_page_attention_forward()` -- replacement forward for LlamaAttention/Qwen2Attention/Qwen3Attention. Prefill: standard attention + cache pre-allocation. Decode: score pages -> topk select -> assemble -> SDPA.
- `replace_llama_attn()`, `replace_qwen2_attn()`, `replace_qwen3_attn()` -- monkey-patch entry points. Call **before** loading the model.
- `_update_comp_cache()` -- incremental page compression (Haar or DCT).
- `_build_haar_lowpass_projection_matrix()` -- default scoring proxy (block averaging).
- `PreAllocatedLayer` -- fixed-stride KV buffer for O(1) decode append.

### triton_kernels.py
- `_score_pages_fused_kernel` -- scores all pages via query-compressed-key dot products.
- `_topk_sort_kernel` -- parallel topk via bitonic sort.
- `_assemble_kv_full_kernel` -- gathers sink + selected pages + recent, applies RoPE.
- `_apply_rope_q_kernel` -- single-token decode query RoPE.
- Each kernel has a `*_triton()` wrapper and a PyTorch fallback path.

### config.py
`DCTPageConfig` dataclass. Key defaults: `page_size=32`, `top_k=64`, `sink_size=4`, `recent_size=128`, `compress_ratio=0.03125`, `scoring_method="max"`, `unselected_mode="drop"`, `score_use_haar_proxy=True`, `use_triton=True`.

## Setup

```bash
pip install -r requirements.txt
# Requires: torch 2.10, transformers 4.54, triton 3.6
```

## Running evaluations

**RULER** (synthetic long-context):
```bash
python eval_ruler.py --mode page_attention --base_model meta-llama/Llama-3.1-8B-Instruct \
  --seq_lengths 16384 32768 --tasks niah_multikey_3 --num_samples 25 \
  --page_size 32 --top_k 64 --unselected_mode drop --output_dir results_ruler --run_name my_run
```

**LongBench v1/v2**:
```bash
python eval_longbench_v1.py --mode page_attention --base_model meta-llama/Llama-3.1-8B-Instruct \
  --page_size 32 --top_k 64 --output_dir results_longbench_v1 --run_name my_run
```

**Speed test**:
```bash
python speed/speed_test_dummy.py --mode dct --model meta-llama/Llama-3.1-8B-Instruct \
  --context_lengths 8192,16384,32768 --page_size 32 --top_k 64 --output_dir results/speed
```

**Parametric sweeps**: use `run_ruler.sh`, `run_longbench_v1.sh`, etc.

**Baselines** (`--mode baseline`, `seer_attention`, or `multipole_attention` in eval scripts).

## Conventions

- **Monkey-patch pattern**: set global config then patch `forward`. Always call `replace_*_attn()` before `from_pretrained()`.
- **Tensor shapes**: `[bsz, num_heads, seq_len, head_dim]` after transpose. `paged_*` = reshaped to `[..., num_pages, page_size, ...]`. `comp_*` = compressed. `*_buf` = pre-allocated buffer.
- **Buffer caching**: kernels and projection matrices are cached on `attn_module` attributes (lazy init, shape/device checked each call).
- **Triton kernels**: `@triton.jit` with constexpr block sizes, wrapper functions handle grid launch and fallback.
- **No unit test suite**: validation is through benchmark runs (RULER, LongBench).
- **Supported models**: Llama 3.x, Qwen 2.x, Qwen 3.x (with QK-norm).
- **RoPE**: supports default, yarn, and llama3 rope types. `continuous_rope` is currently disabled; `block_center` RoPE for compressed tokens is active.
- **Config naming in run scripts**: run names encode params, e.g. `drop_ps32_top64_comp1_haar`.
