"""Meta's Muse Glimmer 30B: a dense LLM with a windowed ViT perception encoder.

Two nodes, ``vision_encoder`` and ``LLM``. Text-only prompts prefill on
``prefill_text``; a prompt with images runs one ``prefill_vision`` walk that
encodes every image and prefills the whole interleaving as one row. Decode is a
loop over the LLM.
"""
import json
import logging
from pathlib import Path
from typing import Any

import torch
from jinja2 import TemplateError

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
)
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.multimodal import PromptPart, messages_from_parts, vision_encoder
from mstar.model.multimodal_decoder import (
    SAMPLING_KEYS,
    MediaIntake,
    MultimodalDecoderModel,
    PreparedMedia,
)
from mstar.model.muse.config import (
    ATTN_FULL,
    ATTN_SLIDING,
    KV_CACHE,
    ROPE,
    SAMPLER,
    VISION_ATTN,
    MuseGlimmerConfig,
    MuseGlimmerVisionConfig,
)
from mstar.model.muse.processing import (
    IMAGE_END,
    IMAGE_PAD,
    IMAGE_START,
    load_tokenizer,
    preprocess_image,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import load_generation_defaults
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)

_REASONING_STRENGTHS = ("low", "medium", "high", "xhigh")
# what `preprocess_image` builds in, besides the patch geometry: RGB, Lanczos
# (PIL's 1), then (x / 255 - 0.5) / 0.5
_IMAGE_PROCESSOR = {
    "image_processor_type": "MuseGlimmerImageProcessor",
    "do_convert_rgb": True, "do_resize": True, "do_rescale": True, "do_normalize": True,
    "resample": 1, "rescale_factor": 1 / 255, "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
}


class MuseGlimmerModel(MultimodalDecoderModel):
    NAME = "Muse Glimmer"
    METADATA_FILES = (
        "config.json", "generation_config.json", "processor_config.json",
        "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
    )

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs: Any):
        super().__init__(model_path_hf, cache_dir)
        self.config = MuseGlimmerConfig.from_hf(self.local_dir)
        self.vision_config = MuseGlimmerVisionConfig.from_hf(self.local_dir)
        self.tokenizer = load_tokenizer(self.local_dir)
        ids = {t: self.tokenizer.convert_tokens_to_ids(t) for t in (IMAGE_PAD, IMAGE_START, IMAGE_END)}
        unknown = [t for t, i in ids.items() if i is None or i == self.tokenizer.unk_token_id]
        if unknown:
            raise ValueError(f"the tokenizer has no {unknown}; this checkpoint cannot take images")
        self._image_pad_id, self._image_start_id, self._image_end_id = ids.values()
        # the vision tower, whose output fills the images' pads in prompt order
        self.media = (MediaIntake(
            "image", vision_encoder((self._image_pad_id,)), self._image_pad_id, self._image_intake,
            sentinels=(self._image_start_id, self._image_end_id),
        ),)
        if self._image_pad_id != self.config.image_token_id:
            raise ValueError(
                f"tokenizer's {IMAGE_PAD} is {self._image_pad_id}, config's image "
                f"token {self.config.image_token_id}"
            )
        self._bos_id = self.tokenizer.bos_token_id
        # temperature 1, top_p 0.95 and top_k 64 for the 30B
        self.sampling_defaults = {
            k: v for k, v in load_generation_defaults(self.local_dir).items() if k in SAMPLING_KEYS
        }
        self._processor = self._read_processor_config()

    def _read_processor_config(self) -> dict:
        """`preprocess_image`'s arguments: the patch geometry is the tower's,
        and the token cap the processor's, 4096 where it sets none. A
        ``processor_config.json`` key that disagrees with what intake builds
        in, or that it does not read, is refused."""
        vision = self.vision_config
        sizes = {
            "patch_size": vision.patch_size,
            "merge_size": vision.merge_size,
            "temporal_patch_size": vision.patch_temporal,
        }
        path = Path(self.local_dir) / "processor_config.json"
        raw = json.loads(path.read_text()) if path.exists() else {}
        # under "image_processor", or flat
        image = raw.get("image_processor", raw)
        ours = {**_IMAGE_PROCESSOR, **sizes}
        for key, value in image.items():
            # the token cap is read below; the others name the HF classes, not a setting
            if key in ("max_image_tokens", "processor_class", "auto_map"):
                continue
            if key not in ours:
                raise ValueError(f"processor_config.json's {key} is not one this port reads")
            if value != ours[key]:
                raise ValueError(f"processor_config.json's {key} is {value}, this port's {ours[key]}")
        return {**sizes, "max_tokens": image.get("max_image_tokens", 4096)}

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        cfg = self.config
        vision = self.vision_config
        max_patches = self._processor["max_tokens"] * self._processor["merge_size"] ** 2
        # a window per 32x32 patches; at most a few more at the edges
        max_windows = max_patches // vision.window_patches ** 2 + 8
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    max_seq_len=cfg.max_position_embeddings,
                    num_qo_heads=cfg.num_attention_heads,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN_SLIDING, nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE, sliding_window=cfg.sliding_window, sm_scale=cfg.attn_scale,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN_FULL, nodes={"LLM"},
                config=AttentionConfig(kv_cache=KV_CACHE, sm_scale=cfg.attn_scale),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                config=PositionConfig(kv_cache=KV_CACHE, rope_theta=cfg.rope_theta),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"LLM"},
                vocab_size=cfg.vocab_size,
                enable_repetion_penalty=True,
            ),
            RaggedAttentionSpec(
                resource_key=VISION_ATTN, nodes={"vision_encoder"},
                config=RaggedAttentionConfig(
                    num_qo_heads=vision.num_attention_heads,
                    num_kv_heads=vision.num_attention_heads,
                    head_dim=vision.head_dim,
                    max_segments_per_request=max_windows,
                ),
            ),
        ]

    # -----------------------------------------------------------------------
    # Graph
    # -----------------------------------------------------------------------

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        # eager for now: no mixed prefill/decode step
        return {}

    # -----------------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------------

    def max_input_files(self) -> dict[str, int]:
        # refused before any file is decoded
        return {"video": 0, "audio": 0}

    def _encode(self, text: str) -> torch.Tensor:
        """The rendered template opens with BOS; the tokenizer's own
        post-processor would add a second on the short-prompt path."""
        ids = encode_ids(self.tokenizer, text)
        if len(ids) >= 2 and ids[0] == ids[1] == self._bos_id:
            ids = ids[1:]
        return torch.tensor(ids, dtype=torch.long)

    def render_prompt(self, parts: list[PromptPart], **kwargs) -> str:
        template_kwargs = {}
        strength = kwargs.get("reasoning_strength") or kwargs.get("reasoning_effort")
        if strength is None and kwargs.get("enable_thinking") is False:
            # the chat adapters' switch: the template has no off, only "low"
            strength = "low"
        if strength is not None:
            if strength not in _REASONING_STRENGTHS:
                raise ValueError(
                    f"{self.NAME}'s reasoning strength is one of {_REASONING_STRENGTHS}, "
                    f"got {strength!r}"
                )
            template_kwargs["reasoning_strength"] = strength
        if kwargs.get("current_date") is not None:
            template_kwargs["current_date"] = kwargs["current_date"]
        try:
            return self.tokenizer.apply_chat_template(
                messages_from_parts(parts),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TemplateError as exc:
            raise ValueError(f"Muse Glimmer's chat template refused the chat: {exc}") from exc

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """Each attachment's placeholder (an image's ``<|patch|>``) is the
        template's; intake expands it between its sentinels, as the reference
        processor does."""
        return self._encode(self.render_prompt(parts, **kwargs))

    def _image_intake(self, image: torch.Tensor, metadata: dict) -> PreparedMedia:
        pixels, grid = preprocess_image(image, **self._processor)
        grid = torch.tensor(grid, dtype=torch.long)
        return PreparedMedia(
            {"pixel_values": pixels}, [int(grid.prod()) // self._processor["merge_size"] ** 2], [grid],
        )

    # -----------------------------------------------------------------------
    # Submodules
    # -----------------------------------------------------------------------

    def _create_submodule(
        self, node_name: str, device: str, autocast_dtype: torch.dtype | None, tp_group,
    ) -> NodeSubmodule | None:
        from mstar.model.muse import weight_loader
        from mstar.model.muse.submodules import (
            LLMSubmodule,
            VisionEncoderSubmodule,
        )

        if node_name not in ("LLM", "vision_encoder"):
            return None
        dtype = autocast_dtype or torch.bfloat16
        weights_dir = self._resolve_weights()
        if node_name == "vision_encoder":
            from mstar.model.muse.components.vision import MuseGlimmerVisionModel

            tower = self._build(lambda: MuseGlimmerVisionModel(self.vision_config), dtype, device)
            weight_loader.load_vision_weights(tower, weights_dir, device=device)
            tower.requires_grad_(False).eval()
            logger.info("Loaded Muse Glimmer vision encoder onto %s", device)
            return VisionEncoderSubmodule(
                tower, self.vision_config, self.encoder(node_name),
            )

        from mstar.model.muse.components.text import MuseGlimmerForCausalLM

        model = self._build(lambda: MuseGlimmerForCausalLM(self.config, tp_group), dtype, device)
        weight_loader.load_text_weights(model, weights_dir, device=device)
        model.requires_grad_(False).eval()
        logger.info("Loaded Muse Glimmer LLM onto %s", device)
        return LLMSubmodule(model, self.config, self.encoders)
