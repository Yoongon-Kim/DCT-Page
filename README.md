# DCT-Page

**Decode-time sparse page attention for long-context LLMs, with a training-free DCT-lowpass-IDCT proxy for page selection.**

DCT-Page accelerates autoregressive decoding over long contexts by attending to only a small, query-relevant subset of the KV cache. Each fixed-size *page* of past keys/values is summarized by a cheap spectral proxy; the current query scores pages against these proxies, keeps the **top-k** at full precision, and discards (or compresses) the rest. Selection is **training-free** — no auxiliary gates or fine-tuning — and the hot path is implemented as fused Triton kernels.

## Method

At decode time the KV cache is laid out as fixed-size pages, with a small always-kept floor at each end:

```text
[ sink ] [ page 0 ] [ page 1 ] ... [ page N-1 ] [ recent ]
  kept      <----------- scored & selected ----------->     kept
```

Prefill runs standard full attention. For each decode step:

1. **Proxy.** Each page (e.g. 32 tokens) is compressed to a compact **DCT-lowpass-IDCT** representation — a low-frequency reconstruction that preserves the directions a query is most likely to attend to. Compression is incremental: only newly completed pages are processed.
2. **Score.** The query attends against the per-page proxies to produce one relevance score per page.
3. **Select.** The **top-k** pages by score are kept at full precision. The always-kept `sink` (first tokens) and `recent` (last window, including the open page) are added unconditionally.
4. **Assemble & attend.** Unselected pages are either
   - **dropped** — removed entirely (fastest), or
   - **compressed** — replaced by a few DCT-lowpass-IDCT tokens per page (quality floor),
   then standard attention runs over the assembled, much shorter KV.

Because selection is a byproduct of a lightweight spectral summary rather than a learned module, DCT-Page applies to off-the-shelf models with a single monkey-patch.

**Default configuration:** `page_size=32`, `top_k=64`, `compress_ratio=0.125` (32→4 proxy tokens), `unselected_mode="drop"`. Below `min_decode_kv_len_for_paging=8192` tokens the patch falls back to dense decode attention. See [config.py](config.py) for the full hyperparameter set.

## Results

Evaluated on **Llama-3.1-8B-Instruct** and **Qwen3-8B** against dense full attention and five sparse/streaming baselines. Numbers below are accuracy/score; higher is better. *(to be filled in)*

### Long-context quality

| Benchmark | Full (dense) | DCT-Page (drop) | DCT-Page (comp) | Quest | SeerAttn-R | Multipole | InfLLM | DuoAttn |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RULER (32k) | – | – | – | – | – | – | – | – |
| LongBench v1 | – | – | – | – | – | – | – | – |
| LongBench v2 | – | – | – | – | – | – | – | – |

### Reasoning (Qwen3-8B)

| Benchmark | Full (dense) | DCT-Page (drop) | DCT-Page (comp) |
|---|--:|--:|--:|
| AIME 2025 (pass@1) | – | – | – |
| GPQA-diamond | – | – | – |
| MATH-500 | – | – | – |

### Decode throughput (Llama-3.1-8B, tok/s)

| Context | Full (dense) | DCT-Page (drop) | Speedup |
|---|--:|--:|--:|
| 32k | – | – | – |
| 64k | – | – | – |
| 128k | – | – | – |

> DCT-Page's decode cost is dominated by attention over `sink + recent + top-k` tokens rather than the full context, so the relative speedup grows with context length; the crossover vs. dense attention sits around 64k.

## Quickstart

```bash
pip install -r requirements.txt
# Core env (DCT_Page conda): torch 2.11.0, transformers 5.5.4, triton 3.6.0
```

Monkey-patch **before** loading the model:

```python
import torch
from transformers import AutoModelForCausalLM
from dct_page_attention import replace_llama_attn   # or replace_qwen3_attn

replace_llama_attn(
    page_size=32,
    top_k=64,
    compress_ratio=0.125,
    unselected_mode="drop",   # or "compressed"
    use_triton=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
```

Reproduce a benchmark point (RULER @ 32k, Qwen3-8B):

```bash
# DCT-Page
python eval_ruler.py --mode page_attention --base_model Qwen/Qwen3-8B \
  --seq_lengths 32768 --page_size 32 --top_k 64 --unselected_mode drop \
  --output_dir results/ruler --run_name dctpage

# Dense baseline
python eval_ruler.py --mode baseline --base_model Qwen/Qwen3-8B \
  --seq_lengths 32768 --output_dir results/ruler --run_name baseline
```

Full sweeps over models and budgets live in [experiments/](experiments/) (`run_*.sh`).

## Baselines

Side-by-side implementations under [baselines/](baselines/), selected via `--mode` in the eval scripts:

| Method | Mechanism | Models |
|---|---|---|
| **SeerAttention-R** | learned gate-based decode sparsity | Llama 3.x, Qwen2/3 |
| **Multipole** | hierarchical k-means page clustering | Llama 3.x, Qwen2/3 |
| **Quest** | per-page min/max key metadata + CUDA kernels | Llama 2/3.x, Mistral, Qwen3 |
| **InfLLM** | retrieval-based block attention (vendored) | Llama 3.x |
| **DuoAttention** | streaming/retrieval head split | Llama 3.x (separate env) |

Setup notes (extra envs, kernel builds, checkpoints) are in [CLAUDE.md](CLAUDE.md#baselines).

## Repository

- **Method & kernels:** [config.py](config.py), [dct_page_attention.py](dct_page_attention.py), [triton_kernels.py](triton_kernels.py)
- **Evaluation:** `eval_ruler.py`, `eval_longbench_v1.py`, `eval_longbench_v2.py`, `eval_aime25.py`, `eval_gpqa.py`, `eval_math500.py`
- **Diagnostics:** [observations/](observations/) — oracle upper bounds, attention-mass recall, scoring-method comparisons
- **Speed/profiling:** [speed/](speed/) — throughput benchmark and per-stage decode profiler
- **Sweeps & data:** [experiments/](experiments/), [benchmark/](benchmark/)

See **[CLAUDE.md](CLAUDE.md)** for the full file-by-file map, architecture details, the complete command catalog, and per-baseline setup.

## Notes

- The only active score proxy is **DCT-lowpass-IDCT**; earlier Haar / Walsh-Hadamard / direct-spectral variants have been removed.
- `drop` mode targets speed; `compressed` mode is for accuracy studies (sets a quality floor by retaining a compressed view of unselected pages).
- Benchmark data under `benchmark/data/` is gitignored (~136 MB) and regenerates on demand (RULER synthetic via `--prepare`; LongBench auto-downloaded from Hugging Face).
- LongBench v1 semantics follow the FastKV adjustments (prompt formatting, no-chat tasks, metric computation).
