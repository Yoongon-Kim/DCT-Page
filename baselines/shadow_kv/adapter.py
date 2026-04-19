"""
ShadowKV LLM adapter.

Wraps ShadowKV's custom LLM class (`models/base.py:31`) in an HF-shaped
interface so eval_ruler.predict_task can drive it the same way it drives
HF's `model.generate(input_ids, max_new_tokens=...)`.

The generate loop is line-by-line isomorphic to ShadowKV's `LLM.generate`
(`models/base.py:191-263`) — same prefill, same first-token sample, same
H2D() placement, same `while n < gen_len` ordering, same EOS-included
behavior. NO decode/re-tokenize round trip.
"""

import torch


def build_shadowkv_llm(cfg):
    """Constructs the ShadowKV LLM and wraps it in ShadowKVLLMAdapter.

    Imports are lazy so this module can be imported without ShadowKV's
    pinned torch/transformers being available — the actual import happens
    inside the dct_shadowkv conda env when the adapter is constructed.
    """
    from .models import choose_model_class

    if "llama" not in cfg["base_model"].lower():
        raise NotImplementedError(
            f"shadowkv baseline only supports Llama-3.x for now "
            f"(got {cfg['base_model']!r}). Qwen3 needs QK-norm in ShadowKV's "
            "pre_attention_compute — not yet ported."
        )

    LLM = choose_model_class(cfg["base_model"])
    llm = LLM(
        model_name=cfg["base_model"],
        batch_size=1,
        max_length=cfg["max_length"],
        device="cuda:0",
        dtype=torch.bfloat16,
        attn_mode=cfg["cache_mode"],
        sparse_budget=cfg["sparse_budget"],
        rank=cfg["rank"],
        chunk_size=cfg["chunk_size"],
        minference=cfg["minference"],
    )
    return ShadowKVLLMAdapter(llm)


class ShadowKVLLMAdapter:
    """HF-like wrapper around ShadowKV's LLM.

    Returns concatenated [input_ids ; generated_ids] from generate() so
    eval_ruler.predict_task's `output_ids[0, input_len:]` slice +
    tokenizer.decode works without further special-casing.
    """

    def __init__(self, llm):
        self._llm = llm
        self.device = torch.device(llm.device)
        self.tokenizer = llm.tokenizer
        # Llama-3 stop ids: ShadowKV's base.py:237 + :239 break on
        # eos_token_id and on tokens that decode to "<|eot_id|>". For
        # Llama-3.1 both resolve to 128009; the set dedupes.
        self._stop_ids = {self.tokenizer.eos_token_id}
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != self.tokenizer.unk_token_id:
            self._stop_ids.add(eot)

    def eval(self):
        return self

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, do_sample=False,
                 use_cache=True, **_unused):
        from .models.tensor_op import sample_token

        # === isomorphic to base.py:191-263, gen_len := max_new_tokens ===
        # base.py:200 — prefill (auto-clears cache via base.py:76)
        logits = self._llm.prefill(input_ids)
        # base.py:205 — first token (temp=0 ⇒ argmax in tensor_op:290)
        next_token = sample_token(
            logits[:, -1, :], temperature=0.0, top_p=1.0, top_k=50
        )
        # base.py:209-210 — first token enters generated_ids before loop
        generated = [int(next_token[0].item())]
        # base.py:212 — H2D() between prefill and decode loop
        self._llm.kv_cache.H2D()

        # base.py:217 — `while n < gen_len`
        n = 0
        while n < max_new_tokens:
            # base.py:218-219 — inference + sample
            logits = self._llm.inference(
                input_ids=next_token,
                position_ids=self._llm.get_ctx(next_token),
            )
            next_token = sample_token(
                logits[:, -1, :], temperature=0.0, top_p=1.0, top_k=50
            )
            # base.py:221-222 — increment, append BEFORE stop-check
            n += 1
            tok = int(next_token[0].item())
            generated.append(tok)
            # base.py:237-240 — Llama-3 stops (EOS already appended)
            if tok in self._stop_ids:
                break
            # base.py:241-248 (yi/glm/phi stops) intentionally omitted —
            # adapter only supports Llama-3.

        # base.py:257 (trailing inference) intentionally omitted: it
        # advances kv_cache by 1 step but produces no output token. The
        # next sample's prefill() self-clears the cache anyway, so this
        # call is unobservable.
        # base.py:259-261 (gc/empty_cache/synchronize) omitted — pure
        # housekeeping, eval_ruler manages resources itself.

        gen_ids = torch.tensor(
            [generated], device=self.device, dtype=input_ids.dtype
        )
        return torch.cat([input_ids, gen_ids], dim=1)

    def shadowkv_clear(self):
        self._llm.kv_cache.clear()
