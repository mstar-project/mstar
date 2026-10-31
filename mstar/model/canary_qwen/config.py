"""Config for NVIDIA Canary-Qwen-2.5B (NeMo's SALM): a FastConformer encoder
from canary-1b-flash, a linear projection, and Qwen3-1.7B with LoRA on its
query and value projections.

The checkpoint ships NeMo's ``config.json`` and weights only; the LLM's config
and tokenizer come from the base LLM's repo (``pretrained_llm``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mstar.model.components.audio_features import check_nemo_preprocessor
from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.components.qwen3_lm import Qwen3LMConfig

KV_CACHE = "kv_cache"
ATTN = "attn"
ROPE = "rope"
SAMPLER = "sampler"
PREFILL_WALK = "prefill"
DECODE_WALK = "decode"
DECODE_LOOP = "decode_loop"

# the featurizer this model serves with: NemoLogMel's defaults, unpadded
MEL_SETTINGS = dict(
    sample_rate=16_000, n_fft=512, window_size=0.025, window_stride=0.01, normalize="per_feature", pad_to=0,
    preemph=0.97,
)


@dataclass
class CanaryQwenModelConfig:
    encoder: FastConformerConfig
    llm: Qwen3LMConfig
    llm_repo: str
    audio_locator_tag: str = "<|audioplaceholder|>"
    lora_scale: float = 2.0
    sampling_rate: int = 16_000

    @classmethod
    def from_pretrained(cls, local_dir: str | Path, llm_dir: str | Path) -> CanaryQwenModelConfig:
        nemo = json.loads((Path(local_dir) / "config.json").read_text())
        enc = nemo["perception"]["encoder"]
        check_nemo_preprocessor(nemo["perception"].get("preprocessor", {}), **MEL_SETTINGS, features=enc["feat_in"])
        encoder = FastConformerConfig.from_nemo(enc)
        if encoder.causal:
            raise ValueError("Canary-Qwen's encoder is expected to attend to the whole clip")
        lora = nemo["lora"]
        llm = Qwen3LMConfig.from_hf(json.loads((Path(llm_dir) / "config.json").read_text()))
        return cls(
            encoder=encoder,
            llm=llm,
            llm_repo=nemo["pretrained_llm"],
            audio_locator_tag=nemo.get("audio_locator_tag", "<|audioplaceholder|>"),
            lora_scale=lora["lora_alpha"] / lora["r"],
        )
