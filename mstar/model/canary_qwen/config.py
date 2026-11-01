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

from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.components.qwen3_lm import Qwen3LMConfig

KV_CACHE = "kv_cache"
ATTN = "attn"
ROPE = "rope"
SAMPLER = "sampler"
PREFILL_WALK = "prefill"
DECODE_WALK = "decode"
DECODE_LOOP = "decode_loop"

# NeMo's preprocessor settings that NemoLogMel implements
MEL_SETTINGS = dict(
    sample_rate=16_000, n_fft=512, window_size=0.025, window_stride=0.01, window="hann",
    normalize="per_feature", log=True, frame_splicing=1, pad_to=0,
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
        if enc.get("self_attention_model") != "rel_pos" or enc.get("subsampling") != "dw_striding":
            raise ValueError("only rel_pos / dw_striding FastConformer encoders are supported")
        if list(enc.get("att_context_size", [-1, -1])) != [-1, -1]:
            raise ValueError("Canary-Qwen's encoder is expected to attend to the whole clip")
        mel = nemo["perception"].get("preprocessor", {})
        expected = {**MEL_SETTINGS, "features": enc["feat_in"]}
        unsupported = {k: mel.get(k) for k, v in expected.items() if mel.get(k) != v}
        if unsupported:
            raise ValueError(f"unsupported mel front end settings {unsupported}; expected {expected}")
        encoder = FastConformerConfig(
            hidden_size=enc["d_model"],
            num_layers=enc["n_layers"],
            num_heads=enc["n_heads"],
            intermediate_size=enc["d_model"] * enc["ff_expansion_factor"],
            conv_kernel_size=enc["conv_kernel_size"],
            num_mel_bins=enc["feat_in"],
            subsampling_factor=enc["subsampling_factor"],
            subsampling_channels=enc["subsampling_conv_channels"],
            # NeMo scales the input unless told not to
            scale_input=enc.get("xscaling", True),
        )
        lora = nemo["lora"]
        llm = Qwen3LMConfig.from_hf(json.loads((Path(llm_dir) / "config.json").read_text()))
        return cls(
            encoder=encoder,
            llm=llm,
            llm_repo=nemo["pretrained_llm"],
            audio_locator_tag=nemo.get("audio_locator_tag", "<|audioplaceholder|>"),
            lora_scale=lora["lora_alpha"] / lora["r"],
        )
