"""
Streaming perplexity evaluation for DCT Page Attention.

Prefills the first `prefill_len` tokens with full attention (standard),
then decodes remaining tokens one-at-a-time with page attention enabled.
Measures perplexity on the decoded tokens.
"""

import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers

from dct_page_attention import replace_qwen2_attn


def parse_config():
    parser = argparse.ArgumentParser(description='DCT Page Attention Evaluation')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (1 recommended for streaming eval)')
    parser.add_argument('--base_model', type=str, required=True,
                        help='path to Qwen2.5 model')
    parser.add_argument('--seq_len', type=int, default=8192,
                        help='total sequence length to evaluate')
    parser.add_argument('--prefill_len', type=int, default=512,
                        help='tokens to prefill with full attention before decode')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to tokenized data (.bin, uint16 memmap)')
    parser.add_argument('--sliding_window', type=int, default=256,
                        help='stride for sliding window over data')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='max number of samples to evaluate (-1 = all)')

    # DCT Page Attention params
    parser.add_argument('--page_size', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=8)
    parser.add_argument('--sink_size', type=int, default=4)
    parser.add_argument('--recent_size', type=int, default=128)
    parser.add_argument('--compress_ratio', type=float, default=0.25)
    parser.add_argument('--scoring_method', type=str, default='max',
                        choices=['mean', 'max', 'sum'])
    parser.add_argument('--unselected_mode', type=str, default='drop',
                        choices=['drop', 'compressed'])

    args = parser.parse_args()
    return args


def get_samples(data, seq_length, sliding_window=256, num_samples=-1):
    """Yield individual sequences from the memmap data."""
    all_ix = list(range(0, len(data) - seq_length - 1, sliding_window))
    if num_samples > 0:
        all_ix = all_ix[:num_samples]

    for i in all_ix:
        x = torch.from_numpy(data[i:i + seq_length + 1].astype(np.int64))
        yield x


def evaluate_streaming(model, data, device, seq_len, prefill_len, args):
    """
    Streaming evaluation: prefill + token-by-token decode.

    For each sample:
      1. Prefill first `prefill_len` tokens (full attention, builds KV cache)
      2. Decode remaining tokens one at a time (page attention on KV cache)
      3. Compute cross-entropy loss on decoded tokens
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    sample_losses = []

    samples = list(get_samples(
        data['val'], seq_len,
        sliding_window=args.sliding_window,
        num_samples=args.num_samples,
    ))

    print(f"Evaluating {len(samples)} samples, seq_len={seq_len}, prefill_len={prefill_len}")
    print(f"Decode tokens per sample: {seq_len - prefill_len}")

    with torch.no_grad():
        for sample_idx, tokens in enumerate(tqdm(samples, desc="Samples")):
            tokens = tokens.to(device)  # [seq_len + 1]
            input_ids = tokens[:seq_len].unsqueeze(0)    # [1, seq_len]
            labels = tokens[1:seq_len + 1]                # [seq_len] (shifted by 1)

            sample_loss = 0.0
            sample_token_count = 0

            # Phase 1: Prefill (full attention)
            prefill_input = input_ids[:, :prefill_len]
            outputs = model(
                input_ids=prefill_input,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            # Compute loss on prefill tokens (these use full attention, for reference)
            prefill_logits = outputs.logits[0]  # [prefill_len, vocab_size]
            prefill_labels = labels[:prefill_len]  # [prefill_len]
            # Loss on positions 0..prefill_len-1 predicting tokens 1..prefill_len
            prefill_loss = torch.nn.functional.cross_entropy(
                prefill_logits, prefill_labels, reduction='sum'
            )

            # Phase 2: Decode token-by-token (page attention)
            decode_loss = 0.0
            decode_count = 0

            for i in range(prefill_len, seq_len):
                decode_input = input_ids[:, i:i + 1]  # [1, 1]
                outputs = model(
                    input_ids=decode_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                # Loss: logits predict the NEXT token
                logits = outputs.logits[0, 0]  # [vocab_size]
                if i + 1 <= seq_len:
                    target = labels[i]  # token at position i+1
                    loss = torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0), target.unsqueeze(0)
                    )
                    decode_loss += loss.item()
                    decode_count += 1

            # Aggregate
            if decode_count > 0:
                avg_decode_loss = decode_loss / decode_count
                sample_losses.append(avg_decode_loss)
                total_loss += decode_loss
                total_tokens += decode_count

            if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
                running_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                print(f"  Sample {sample_idx + 1}/{len(samples)}, "
                      f"running decode PPL: {running_ppl:.2f}")

    # Final stats
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    stats = {
        'decode_loss': avg_loss,
        'decode_perplexity': perplexity,
        'total_decode_tokens': total_tokens,
        'num_samples': len(samples),
    }
    return stats


def main(args):
    device = "cuda:0"
    seed = 42
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}
    print(f"Num validation tokens: {len(data['val'])}")

    # Apply monkey-patch BEFORE loading model
    replace_qwen2_attn(
        page_size=args.page_size,
        top_k=args.top_k,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        compress_ratio=args.compress_ratio,
        scoring_method=args.scoring_method,
        unselected_mode=args.unselected_mode,
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    print(f"\nModel loaded: {args.base_model}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Evaluate
    stats = evaluate_streaming(
        model, data, device,
        seq_len=args.seq_len,
        prefill_len=args.prefill_len,
        args=args,
    )

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Decode perplexity: {stats['decode_perplexity']:.2f}")
    print(f"  Decode loss: {stats['decode_loss']:.4f}")
    print(f"  Total decode tokens: {stats['total_decode_tokens']}")
    print(f"  Num samples: {stats['num_samples']}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_config()
    main(args)
