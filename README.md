# DCT Page Attention

**Efficient long-context LLM inference via DCT-compressed KV cache page selection**

DCT Page Attention is a sparse attention mechanism that reduces memory and compute costs during LLM decoding by compressing the KV cache into fixed-size pages using the Discrete Cosine Transform (DCT), scoring page relevance against the current query, and attending only to the most informative pages. It achieves significant decode speedups on long contexts (32K-120K+ tokens) while preserving generation quality.

## Motivation

Transformer-based LLMs face quadratic attention cost as context length grows. During autoregressive decoding, each new token must attend to the entire KV cache, making long-context inference (32K+ tokens) a bottleneck. Existing approaches like sliding window attention lose long-range information, while full attention is prohibitively expensive.

DCT Page Attention addresses this by:
1. **Compressing** KV cache pages into compact frequency-domain representations using DCT
2. **Scoring** page importance cheaply against the current query using compressed keys
3. **Selecting** only the top-k most relevant pages for full-precision attention

This is inspired by recent works including Quest (ICML 2024), MoBA (Moonshot/Kimi 2025), and NSA (DeepSeek 2025), but introduces DCT-based compression as the scoring proxy — providing a principled, training-free signal for page relevance.

## Key Features

- **DCT-based page compression**: FFT-based O(N log N) DCT compresses each 128-token page down to 32 tokens (configurable ratio), preserving dominant frequency components as scoring proxies
- **Selective page attention**: During decode, only top-k scored pages (default 8) are fetched at full precision, alongside always-kept sink and recent tokens
- **Fused Triton kernels**: 5 custom GPU kernels (page scoring, top-k sort, KV assembly + RoPE, Q-RoPE) eliminate intermediate allocations and minimize kernel launches
- **Incremental compression cache**: Only newly-finalized pages are compressed each decode step — amortized O(1) per token
- **Pre-allocated KV cache**: Fixed-size buffers replace dynamic concatenation, enabling O(1) cache writes with stable memory strides
- **Continuous RoPE mode**: Stores pre-RoPE KV values and applies position embeddings after page assembly, avoiding redundant RoPE computation
- **GQA-aware scoring**: Per-head scoring within Grouped Query Attention groups with configurable aggregation (mean/max/top-p voting)
- **Drop-in replacement**: Monkey-patch integration for Qwen2/2.5 and Llama models via HuggingFace Transformers

## How It Works

### KV Cache Layout (Decode Phase)

```
[Sink (4)] [Page 0 (128)] [Page 1 (128)] ... [Page N (128)] [Recent (128+)]
  always       scored via DCT-compressed keys            always
  kept         top-k selected for full attention          kept
```

### Pipeline

1. **Prefill**: Standard full causal attention (exact KV cache, no approximation)
2. **Decode** (per token):
   - Compress newly-finalized pages via DCT (cached incrementally)
   - Score all pages: `Q @ compressed_K^T` with per-page max reduction
   - Select top-k pages, assemble KV: `[sink | selected_pages | (compressed_unselected) | recent]`
   - Run standard scaled dot-product attention on the assembled subset

### DCT Compression

The DCT transforms each page's key vectors into frequency domain, retaining only the lowest-frequency coefficients (default 25%). Energy correction via Parseval's theorem (`sqrt(comp_size / page_size)`) preserves signal magnitude. A precomputed projection matrix replaces ~30 FFT kernel launches with a single `einsum`.

## Architecture

```
dct_page_attention.py      Core implementation (DCT, scoring, assembly, forward hook)
dct_page_attention_copy.py Optimized variant with pre-allocated cache + stride caching
triton_kernels.py          Fused Triton GPU kernels for decode path
config.py                  DCTPageConfig dataclass
```

### Evaluation & Benchmarking

```
eval_longbench_v1.py       LongBench v1 (16 English tasks: QA, summarization, code, etc.)
eval_longbench_v2.py       LongBench v2 (503 multi-choice questions, difficulty-stratified)
speed_test_dummy.py        Decode throughput benchmarks (4K-32K+ contexts)
profile_decode.py          Per-layer CUDA event profiling with operation breakdown
```

### Testing

```
test_triton_correctness.py Triton vs. PyTorch reference (scoring, assembly, RoPE)
test_stride_cache_correctness.py  Stride-cached assembly validation
test_noncontig.py          Non-contiguous tensor handling
```

## Usage

```python
from dct_page_attention import replace_qwen2_attn
from transformers import AutoModelForCausalLM

# Monkey-patch before model loading
replace_qwen2_attn(
    page_size=128,
    top_k=8,
    compress_ratio=0.25,
    scoring_method="max",
    unselected_mode="compressed",   # or "drop"
    continuous_rope=True,
    use_triton=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Model now uses DCT Page Attention during decode — no other changes needed
```

Llama models are also supported:
```python
from dct_page_attention import replace_llama_attn
replace_llama_attn(top_k=8, compress_ratio=0.25)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `page_size` | 128 | Tokens per KV cache page |
| `top_k` | 8 | Number of pages selected for full attention |
| `sink_size` | 4 | Initial tokens always retained (attention sink) |
| `recent_size` | 128 | Recent tokens always retained |
| `compress_ratio` | 0.25 | DCT compression ratio (e.g., 128 -> 32 tokens) |
| `scoring_method` | `"max"` | Per-page score reduction: `max` / `mean` / `sum` |
| `group_agg_method` | `"mean"` | GQA group aggregation: `mean` / `max` / `topp` |
| `unselected_mode` | `"drop"` | Unselected pages: `drop` (discard) or `compressed` (keep DCT representation) |
| `continuous_rope` | `True` | Apply RoPE after assembly (stores pre-RoPE KV) |
| `use_triton` | `True` | Use fused Triton kernels (vs. PyTorch fallback) |

## Evaluation

Evaluated on **LongBench v1** (long-context understanding) and **LongBench v2** (multi-choice reasoning) with Llama-3.1-8B-Instruct, covering tasks including NarrativeQA, QASPER, GovReport, and more across input lengths up to 120K tokens.

Run the benchmark:
```bash
bash run_longbench_v1.sh
python summarize_longbench_v1.py
```

Speed benchmarks:
```bash
python speed_test_dummy.py --context_lengths 4096 8192 16384 32768
```

## Technical Highlights

- **O(N log N) DCT** via FFT with orthonormal normalization, cast to float32 for numerical stability
- **Projection matrix optimization**: Precomputed DCT-truncate-IDCT pipeline as a single matmul, cached per attention module
- **5 total GPU kernel launches** per decode step (score + topk + assemble + Q-RoPE + SDPA)
- **Zero intermediate allocations** in the Triton assembly kernel — writes directly to output buffer with fused RoPE
- **Incremental compression**: Only new pages are compressed; previous compressed pages are cached and reused

## Related Work

| Method | Venue | Approach |
|---|---|---|
| Quest | ICML 2024 | Min/max per-channel scoring with upper-bound guarantee |
| MoBA | Moonshot/Kimi 2025 | Mean-pooled key gating, deployed in production |
| NSA | DeepSeek 2025 | Learned MLP compression + shared scoring |
| BLASST | 2024 | Block-max vs. running-max thresholding |
| SeerAttention | Microsoft 2024 | Learned gate with multi-pool training target |

DCT Page Attention contributes a **training-free, frequency-domain compression** approach with **fused GPU kernels** and **energy-preserving scoring**, offering a complementary design point in the sparse attention landscape.
