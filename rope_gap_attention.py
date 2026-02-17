"""
RoPE Gap Experiment: Artificial position gaps to isolate the effect of
positional discontinuity on generation quality.

Full attention is used (no compression, no page selection). Only position_ids
are modified to introduce random gaps, mimicking the positional discontinuity
that DCT page attention creates when dropping/selecting pages.

Usage:
    from rope_gap_attention import replace_qwen2_with_rope_gaps
    replace_qwen2_with_rope_gaps(num_gaps=8, gap_size=128, seed=42)
    # Then load model and run generate() as usual
"""

import random
from dataclasses import dataclass
from typing import Optional

import torch
import transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RoPEGapConfig:
    num_gaps: int = 8        # Number of position gaps to insert
    gap_size: int = 128      # Size of each gap (in position units)
    seed: int = 42           # Random seed for gap placement


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_gap_cfg: Optional[RoPEGapConfig] = None
_total_offset: int = 0
_original_forward = None


# ---------------------------------------------------------------------------
# Gap computation
# ---------------------------------------------------------------------------
def compute_gapped_position_ids(seq_len, num_gaps, gap_size, seed=42):
    """
    Create position_ids with artificial gaps.

    Inserts num_gaps gaps of gap_size at random, well-spaced locations.
    Gaps are placed with a margin from both ends (10%, min 32 tokens)
    and minimum spacing between gaps to avoid clustering.

    Args:
        seq_len: number of tokens
        num_gaps: number of gaps to insert
        gap_size: size of each gap in position units
        seed: random seed for reproducibility

    Returns:
        position_ids: [1, seq_len] tensor with gaps
        total_offset: total accumulated position offset
    """
    rng = random.Random(seed)

    # Leave margin at start and end
    margin = max(32, seq_len // 10)

    if seq_len - 2 * margin < num_gaps:
        # Sequence too short for gaps
        return torch.arange(seq_len, dtype=torch.long).unsqueeze(0), 0

    # Minimum spacing between gap points to avoid clustering
    usable = seq_len - 2 * margin
    min_spacing = max(1, usable // (num_gaps * 2))

    # Greedy random selection with spacing constraint
    candidates = list(range(margin, seq_len - margin))
    gap_points = []
    available = set(candidates)
    for _ in range(num_gaps):
        if not available:
            break
        point = rng.choice(sorted(available))
        gap_points.append(point)
        # Exclude nearby candidates
        for j in range(point - min_spacing, point + min_spacing + 1):
            available.discard(j)

    gap_points.sort()

    # Build gapped positions: each gap_point adds gap_size to all subsequent
    positions = torch.arange(seq_len, dtype=torch.long)
    for point in gap_points:
        positions[point + 1:] += gap_size

    total_offset = len(gap_points) * gap_size
    return positions.unsqueeze(0), total_offset


# ---------------------------------------------------------------------------
# Monkey-patched forward
# ---------------------------------------------------------------------------
def gapped_qwen2_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    cache_position=None,
    **kwargs,
):
    """
    Wrapper around Qwen2Model.forward that injects gapped position_ids.

    - Prefill (seq_len > 1): compute gapped positions from scratch
    - Decode (seq_len == 1): shift position by total accumulated offset
    """
    global _total_offset
    cfg = _gap_cfg

    # Determine sequence length and device
    if inputs_embeds is not None:
        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
    elif input_ids is not None:
        seq_len = input_ids.shape[1]
        device = input_ids.device
    else:
        # Fallback: let original handle the error
        return _original_forward(
            self, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            cache_position=cache_position, **kwargs,
        )

    # Resolve cache_position if not provided (same logic as original)
    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen, past_seen + seq_len, device=device)

    if seq_len > 1:
        # PREFILL: apply gapped positions
        gapped_pos, offset = compute_gapped_position_ids(
            seq_len, cfg.num_gaps, cfg.gap_size, cfg.seed
        )
        position_ids = gapped_pos.to(device)
        _total_offset = offset
    else:
        # DECODE: shift by accumulated offset from prefill gaps
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        position_ids = position_ids + _total_offset

    return _original_forward(
        self, input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, use_cache=use_cache,
        cache_position=cache_position, **kwargs,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def replace_qwen2_with_rope_gaps(num_gaps=8, gap_size=128, seed=42):
    """
    Monkey-patch Qwen2Model.forward to introduce artificial RoPE position gaps.

    Must be called BEFORE loading the model.
    """
    global _gap_cfg, _original_forward

    _gap_cfg = RoPEGapConfig(num_gaps=num_gaps, gap_size=gap_size, seed=seed)
    _original_forward = Qwen2Model.forward

    total_max_offset = num_gaps * gap_size
    print(f"RoPE Gap Experiment config:")
    print(f"  num_gaps={num_gaps}, gap_size={gap_size}, seed={seed}")
    print(f"  Max total position offset: {total_max_offset}")
    print(f"  Full attention (no compression), only position_ids modified")

    Qwen2Model.forward = gapped_qwen2_model_forward