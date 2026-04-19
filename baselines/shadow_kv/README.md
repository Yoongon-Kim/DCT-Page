# ShadowKV baseline

Vendored from https://github.com/bytedance/ShadowKV (ICML 2025 spotlight).

ShadowKV pins `torch==2.3.1`, `transformers==4.43.1`, `vllm==0.5.3.post1`,
which are **incompatible** with DCT-Page's main env (`torch==2.10`,
`transformers==4.54`). Run this baseline in its own conda env.

## Install

```bash
conda create -n dct_shadowkv python=3.10 -y
conda activate dct_shadowkv

cd /path/to/DCT-Page/baselines/shadow_kv
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# eval_ruler.py also needs these
pip install pyyaml pandas tqdm

# CUTLASS for the CUDA kernel build
mkdir -p 3rdparty
git clone https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass

# Build ShadowKV's CUDA extension
python setup.py build_ext --inplace
# produces ./kernels/shadowkv*.so
```

Verify the build:

```bash
python -c "from kernels import shadowkv; print(dir(shadowkv))"
# expect batch_gemm_softmax, gather_copy_with_offsets, apply_rotary_pos_emb_new, ...
```

## Run

From the repo root, in the `dct_shadowkv` env:

```bash
python eval_ruler.py --mode shadowkv \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --shadowkv_cache_mode shadowkv_cpu \
    --sparse_budget 2048 --rank 160 --chunk_size 8 \
    --output_dir results_ruler
```

CLI flags override defaults in `config.py`:
- `--shadowkv_cache_mode` `shadowkv_cpu` (production, V on CPU) | `shadowkv` (GPU-only, batch=1)
- `--sparse_budget` tokens attended to per decode step (default 2048)
- `--rank` SVD rank for compressed key cache (default 160)
- `--chunk_size` tokens per landmark chunk (default 8)

## Scope of fidelity

The adapter (`adapter.py`) replicates ShadowKV's `LLM.generate` loop
(`models/base.py:191-263`) line-by-line in token-id space — bit-identical
inference behavior. It does **not** reproduce ShadowKV's published RULER
scores, because:

1. DCT-Page's RULER prompts (`benchmark/eval_ruler/data/template.py:12`)
   omit ShadowKV's "You are a helpful assistant" system message.
2. DCT-Page uses RULER's official metric_fn for all baselines; ShadowKV's
   own `data/metrics.py` is slightly different.

Both differences are intentional — they keep `--mode shadowkv` directly
comparable to DCT-Page's other baselines on identical inputs and scoring.

## Supported models

Llama-3.x only. Qwen3 needs QK-norm in ShadowKV's `pre_attention_compute`,
which isn't ported yet.

## Layout

```
baselines/shadow_kv/
├── config.py          SHADOWKV_CONFIG dict
├── __init__.py        package exports
├── adapter.py         ShadowKVLLMAdapter, build_shadowkv_llm
├── README.md          this file
├── setup.py           vendored from ShadowKV/setup.py
├── requirements.txt   vendored from ShadowKV/requirements.txt
├── models/            vendored ShadowKV/models/
├── kernels/           vendored ShadowKV/kernels/ (.cu, .cuh, .h)
└── 3rdparty/cutlass/  cloned per-machine, gitignored
```
