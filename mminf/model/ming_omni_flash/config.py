"""Ming-flash-omni-2.0 config skeleton.

WIP scaffold. The released checkpoint is ``inclusionAI/Ming-flash-omni-2.0``
(Ling-2.0 sparse-MoE; 100B total / 6B active params, 42 safetensors shards).
The canonical config schema lives in the vllm-omni port at::

    /sgl-workspace/vllm-omni/vllm_omni/transformers_utils/configs/ming_flash_omni.py

That file defines :class:`MingFlashOmniConfig` with sub-configs for:

  * ``thinker`` — Ling-2.0 MoE LLM + multimodal heads
  * ``talker`` — TTS LLM (CFM-based)
  * ``audio_encoder`` — Whisper-style audio encoder
  * ``audio_vae`` — VAE that produces the talker's training-time audio targets
  * ``vision`` — ViT-style image / video encoder
  * ``image_gen`` — :class:`MingImageGenConfig` for the ZImage DiT pipeline

The mminf side needs an equivalent dataclass tree plus the helpers mminf's
base.Model loader expects (``from_pretrained`` reading config.json and any
processor configs). Mirror the structure of
``mminf/model/qwen3_omni/config.py`` (544 lines) once the upstream vllm-omni
config has been ported field-for-field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MingFlashOmniModelConfig:
    """Placeholder. Port the field set from vllm-omni's MingFlashOmniConfig.

    Required surface (per qwen3_omni/config.py reference):
      * ``thinker_text`` — text-side hidden_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, max_position_embeddings,
        rope_theta, MoE expert counts, etc.
      * ``talker_text`` — talker LLM config (CFM head wraps a smaller LLM)
      * ``audio_encoder`` — feature dim, downsample factor
      * ``vision`` — patch size, image size, num_layers
      * ``audio_vae`` — latent dim, codec hop length
      * ``image_gen`` — DiT layer count, ByT5 dim, query-token count
        (defaults to 256 per ``img_gen_scales=[16]`` on the released ckpt)
    """

    model_path_hf: str = "inclusionAI/Ming-flash-omni-2.0"

    @classmethod
    def from_pretrained(cls, local_dir: str) -> "MingFlashOmniModelConfig":
        raise NotImplementedError(
            "Ming-flash-omni-2.0 config port is incomplete. "
            "See vllm-omni source at "
            "vllm_omni/transformers_utils/configs/ming_flash_omni.py "
            "and mirror the field tree in this dataclass. Until then the "
            "model can be benchmarked via --inference-system vllm_omni "
            "against a vllm-omni server."
        )
