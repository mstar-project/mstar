"""Configuration for the Qwen-Image family, read from the checkpoint's config files.

The architecture comes from ``transformer/config.json``, ``text_encoder/config.json`` (the
Qwen2.5-VL language model and its vision tower), ``vae/config.json`` and
``scheduler/scheduler_config.json``. What diffusers keeps in the pipeline class rather than in a
config (the prompt template and how many of its tokens are dropped, the image areas, the "Picture
N:" prefix, how many images an edit takes) is the pipeline's recipe, keyed by
``model_index.json``'s ``_class_name``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.diffusion.flow_match import FlowMatchConfig
from mstar.model.components.diffusion.qwen.encoder import QwenEncoderConfig
from mstar.model.components.qwen_vision import Qwen25VisionConfig
from mstar.model.qwen_image.components.transformer import QwenImageTransformerConfig

QWEN_IMAGE_EDIT_2511 = "Qwen/Qwen-Image-Edit-2511"

# Resource key of the dit's ragged joint attention, and the loop its graph and cap agree on.
DIT_ATTN = "dit_attn"
DENOISE_LOOP = "denoise_loop"

# The VAE downsamples 8x and the transformer patches 2x2 latents: a token is 16 pixels a side.
TOKEN_PIXELS = 16

# What the edit template's first rows hold before the user turn: the system prompt the pipeline drops.
EDIT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the image. Generate a new "
    "image that meets the user's requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PICTURE_PREFIX = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


@dataclass(frozen=True)
class QwenImageRecipe:
    """A pipeline class's constants (``pipeline_qwenimage_edit_plus.py`` and its siblings)."""

    template: str
    drop_tokens: int          # leading prompt tokens whose hidden states the DiT never sees
    condition_area: int       # pixels of each image the VL encoder sees
    vae_area: int             # pixels of each reference the VAE encodes, and of the default output
    max_images: int
    default_steps: int
    default_guidance_scale: float
    default_negative_prompt: str


RECIPES = {
    "QwenImageEditPlusPipeline": QwenImageRecipe(
        template=EDIT_TEMPLATE, drop_tokens=64, condition_area=384 * 384, vae_area=1024 * 1024, max_images=3,
        # the model card's: 40 steps, true CFG 4.0, a one-space negative prompt
        default_steps=40, default_guidance_scale=4.0, default_negative_prompt=" ",
    ),
}
# Siblings on the same transformer, placed but not built: each is its own recipe and parity run.
LATER = {
    "QwenImagePipeline": "text-to-image (its template drops 34 tokens and runs no vision tower)",
    "QwenImageEditPipeline": "the one-image edit (its VL encoder sees the image at 1024^2)",
    "QwenImageLayeredPipeline": "layered output (an RGBA VAE, 3-D layer RoPE and an extra timestep condition)",
}


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """The pipelines' ``calculate_dimensions``: ``(width, height)`` of about ``target_area`` pixels at
    ``ratio`` (width / height), each rounded to a multiple of 32."""
    width = (target_area * ratio) ** 0.5
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


@dataclass(frozen=True)
class QwenImageConfig:
    recipe: QwenImageRecipe = RECIPES["QwenImageEditPlusPipeline"]
    transformer: QwenImageTransformerConfig = field(default_factory=QwenImageTransformerConfig)
    text_encoder: QwenEncoderConfig = field(default_factory=lambda: QwenEncoderConfig(
        vocab_size=152064, hidden_size=3584, intermediate_size=18944, num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, rope_theta=1_000_000.0, hidden_state_layers=(28,), max_sequence_length=8192,
        fused_qkv=False, qk_norm=False, qkv_bias=True, image_token_id=151655, mrope_section=(16, 24, 24),
        mrope_interleaved=False,
    ))
    vision: Qwen25VisionConfig = field(default_factory=lambda: Qwen25VisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16, in_channels=3, patch_size=14,
        temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=3584, window_size=112,
        fullatt_block_indexes=(7, 15, 23, 31),
    ))
    scheduler: FlowMatchConfig = field(default_factory=lambda: FlowMatchConfig.from_scheduler_config({
        "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 8192, "max_shift": 0.9,
        "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": 0.02, "time_shift_type": "exponential",
        "use_dynamic_shifting": True,
    }))
    # the loop's ceiling; a request past it is refused at intake
    max_denoise_steps: int = 200

    def latent_grid(self, height: int, width: int) -> tuple[int, int]:
        """``(h, w)`` in tokens of an image ``height x width`` pixels."""
        return height // TOKEN_PIXELS, width // TOKEN_PIXELS

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "QwenImageConfig":
        snapshot = Path(snapshot)

        def read(*parts: str) -> dict:
            return json.loads((snapshot.joinpath(*parts)).read_text())

        pipeline = read("model_index.json")["_class_name"]
        if pipeline not in RECIPES:
            if pipeline in LATER:
                raise ValueError(f"{pipeline} is not served yet: {LATER[pipeline]}")
            raise ValueError(f"{snapshot} is a {pipeline}, which is not a Qwen-Image pipeline this model serves")
        text = read("text_encoder", "config.json")
        layers = int(text.get("num_hidden_layers") or text["text_config"]["num_hidden_layers"])
        return cls(
            recipe=RECIPES[pipeline],
            transformer=QwenImageTransformerConfig.from_dict(read("transformer", "config.json")),
            text_encoder=QwenEncoderConfig.from_qwen2_5_vl(
                # the normed last hidden state, of a prompt at its own length
                text, hidden_state_layers=(layers,),
                max_sequence_length=int(text.get("max_position_embeddings", 128_000)), fused_qkv=False,
            ),
            vision=Qwen25VisionConfig.from_dict(text["vision_config"]),
            scheduler=FlowMatchConfig.from_scheduler_config(read("scheduler", "scheduler_config.json")),
        )
