# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from MInference for DCT-Page evaluation
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import torch


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )

        if "Yarn-Llama" in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        try:

            if "llama-3" in name_or_path.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=name_or_path,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=model_kwargs,
                )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")
        # When not sampling, remove temperature/top_k/top_p to avoid
        # HuggingFace division-by-zero with temperature=0.0
        if not self.generation_kwargs.get("do_sample", True):
            self.generation_kwargs.pop("temperature", None)
            self.generation_kwargs.pop("top_k", None)
            self.generation_kwargs.pop("top_p", None)

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, use_cache=True, **self.generation_kwargs)
            generated_text = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
        else:
            output = self.pipeline(
                text_inputs=prompt,
                **self.generation_kwargs,
            )
            assert len(output) == 1
            generated_text = output[0]["generated_text"]

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}


class DCTPageModel:
    def __init__(
        self,
        name_or_path: str,
        dct_page_size=128,
        dct_top_k=8,
        dct_sink_size=4,
        dct_recent_size=128,
        dct_compress_ratio=0.25,
        dct_scoring_method="max",
        dct_group_agg_method="mean",
        dct_unselected_mode="drop",
        dct_continuous_rope=True,
        dct_use_triton=True,
        **generation_kwargs,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Add repo root to sys.path so we can import dct_page_attention
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Auto-detect model family and apply monkey-patch BEFORE model loading
        model_name_lower = name_or_path.lower()
        if "llama" in model_name_lower:
            from dct_page_attention import replace_llama_attn
            replace_llama_attn(
                page_size=dct_page_size,
                top_k=dct_top_k,
                sink_size=dct_sink_size,
                recent_size=dct_recent_size,
                compress_ratio=dct_compress_ratio,
                scoring_method=dct_scoring_method,
                group_agg_method=dct_group_agg_method,
                unselected_mode=dct_unselected_mode,
                continuous_rope=dct_continuous_rope,
                use_triton=dct_use_triton,
            )
            print(f"Applied DCT-Page monkey-patch for Llama: top_k={dct_top_k}, compress_ratio={dct_compress_ratio}")
        else:
            from dct_page_attention import replace_qwen2_attn
            replace_qwen2_attn(
                page_size=dct_page_size,
                top_k=dct_top_k,
                sink_size=dct_sink_size,
                recent_size=dct_recent_size,
                compress_ratio=dct_compress_ratio,
                scoring_method=dct_scoring_method,
                group_agg_method=dct_group_agg_method,
                unselected_mode=dct_unselected_mode,
                continuous_rope=dct_continuous_rope,
                use_triton=dct_use_triton,
            )
            print(f"Applied DCT-Page monkey-patch for Qwen: top_k={dct_top_k}, compress_ratio={dct_compress_ratio}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )

        # Use sdpa: prefill needs memory-efficient attention for long sequences;
        # the DCT page attention monkey-patch handles decode itself.
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")
        # When not sampling, remove temperature/top_k/top_p to avoid
        # HuggingFace division-by-zero with temperature=0.0
        if not self.generation_kwargs.get("do_sample", True):
            self.generation_kwargs.pop("temperature", None)
            self.generation_kwargs.pop("top_k", None)
            self.generation_kwargs.pop("top_p", None)

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        output = self.model.generate(input_ids, use_cache=True, **self.generation_kwargs)
        generated_text = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        # remove the input from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}
