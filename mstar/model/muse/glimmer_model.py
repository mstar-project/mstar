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

from mstar.engine.state import (
    Array,
    Attend,
    Block,
    Causal,
    KVEntry,
    Positions,
    Read,
    Rope,
    Sampler,
    SampleSettings,
    Window,
)
from mstar.model.multimodal import PromptPart, messages_from_parts, vision_encoder
from mstar.model.multimodal_decoder import (
    MIXED_SPLIT,
    SAMPLING_KEYS,
    MediaIntake,
    MultimodalDecoderModel,
    PreparedMedia,
)
from mstar.model.muse.config import (
    ATTN_FULL,
    ATTN_SLIDING,
    KV_CACHE,
    SAMPLER,
    VISION_ATTN,
    VISION_FRAME_ATTN,
    VISION_FRAME_TOKENS,
    VISION_FULL,
    VISION_WINDOW,
    VISION_WINDOW_TOKENS,
    WINDOW_ATTENTION,
    MuseGlimmerConfig,
    MuseGlimmerVisionConfig,
)
from mstar.model.muse.processing import (
    EOT,
    IMAGE_END,
    IMAGE_PAD,
    IMAGE_START,
    load_tokenizer,
    preprocess_image,
)
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import NodeSubmodule
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
        "config.json", "processor_config.json",
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
        eot = self.tokenizer.convert_tokens_to_ids(EOT)
        if eot is not None and eot != self.tokenizer.unk_token_id:
            self.config.eot_token_id = eot
        # the vision tower, whose output fills the images' pads in prompt order
        self.media = (MediaIntake(
            "image", vision_encoder(), self._image_pad_id, self._image_intake,
            sentinels=(self._image_start_id, self._image_end_id),
        ),)
        if self._image_pad_id != self.config.image_token_id:
            raise ValueError(
                f"tokenizer's {IMAGE_PAD} is {self._image_pad_id}, config's image "
                f"token {self.config.image_token_id}"
            )
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

    def rebuild_policies(self) -> dict[str, RebuildPolicy]:
        # not yet dropped: no drop and rebuild has run through its sliding-window KV pool
        return {}

    def declare_state(self) -> list[Array | Sampler]:
        cfg, vision = self.config, self.vision_config
        every = frozenset(range(cfg.num_hidden_layers))
        # the cache keeps only the window of the layers the sliding read runs
        sliding = frozenset(i for i, s in enumerate(cfg.is_sliding) if s)
        attend = Attend(q_heads=cfg.num_attention_heads, scale=cfg.attn_scale)
        tower = frozenset(range(len(vision.layer_types)))
        windowed = frozenset(i for i, kind in enumerate(vision.layer_types) if kind == WINDOW_ATTENTION)

        def tokens(name: str, read: str, label: str, layers: frozenset[int]) -> Array:
            return Array(
                name, nodes={"vision_encoder"}, layers=layers, labels={label}, max_len=0,
                entry=KVEntry(heads=vision.num_attention_heads, head_dim=vision.head_dim),
                reads={read: Read(mask=Block(), op=Attend(q_heads=vision.num_attention_heads))},
            )

        return [
            Array(
                KV_CACHE, nodes={"LLM"}, layers=every,
                entry=KVEntry(heads=cfg.num_key_value_heads, head_dim=cfg.head_dim),
                max_len=cfg.max_position_embeddings, positions=Positions(rope=Rope(theta=cfg.rope_theta)),
                reads={
                    ATTN_SLIDING: Read(mask=Window(cfg.sliding_window), op=attend, layers=sliding),
                    ATTN_FULL: Read(mask=Causal(), op=attend, layers=every - sliding),
                },
            ),
            Sampler(
                SAMPLER, nodes={"LLM"}, vocab_size=cfg.vocab_size, penalty=True,
                min_p=self._sampler_spec(cfg.vocab_size).enable_min_p,
            ),
            # windows first: both reads plan on the resource of the first, under its key
            tokens(VISION_WINDOW_TOKENS, VISION_ATTN, VISION_WINDOW, windowed),
            tokens(VISION_FRAME_TOKENS, VISION_FRAME_ATTN, VISION_FULL, tower - windowed),
        ]

    def kernel_settings(self) -> dict[str, dict]:
        # Segments a captured block-loop replay plans under each label
        # (windows, frames). Every replay plans for this many, so it is sized
        # to real prompts rather than the worst case: the top bucket's patches
        # in whole windows, four times over for partial edge windows and
        # several images. A step with more runs the block loop eagerly
        # (`PackedBlockLoopEncoder._fits_capture`); YAML can raise it.
        from mstar.model.muse.submodules import VisionEncoderSubmodule

        top_bucket = VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1]
        return {
            ATTN_SLIDING: {"splits": (MIXED_SPLIT,)}, ATTN_FULL: {"splits": (MIXED_SPLIT,)},
            VISION_ATTN: {"max_segments_per_step": 4 * (top_bucket // self.vision_config.window_patches ** 2)},
        }

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, SampleSettings]:
        config = super().get_request_resource_configs(input_modalities, output_modalities, model_kwargs)[SAMPLER]
        return {SAMPLER: SampleSettings(**{k: getattr(config, k) for k in SAMPLING_KEYS})}

    # -----------------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        return torch.tensor(encode_ids(self.tokenizer, text), dtype=torch.long)

    def request_kwargs(self) -> frozenset[str]:
        return super().request_kwargs() | {"reasoning_strength", "reasoning_effort", "current_date"}

    def render_prompt(self, parts: list[PromptPart], **kwargs) -> str:
        template_kwargs = {}
        enable_thinking = kwargs.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError(f"enable_thinking must be a boolean; got {enable_thinking!r}")
        if kwargs.get("current_date") is not None and not isinstance(kwargs["current_date"], str):
            raise ValueError(f"current_date must be a string; got {kwargs['current_date']!r}")
        strength = kwargs.get("reasoning_strength") or kwargs.get("reasoning_effort")
        if strength is None and enable_thinking is False:
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
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        from mstar.model.muse import weight_loader
        from mstar.model.muse.submodules import (
            LLMSubmodule,
            VisionEncoderSubmodule,
        )

        if node_name not in ("LLM", "vision_encoder"):
            return None
        weights_dir = self._resolve_weights()
        if node_name == "vision_encoder":
            from mstar.model.muse.components.vision import MuseGlimmerVisionModel

            tower = self._build(lambda: MuseGlimmerVisionModel(self.vision_config), dtype, device)
            weight_loader.load_vision_weights(tower, weights_dir, device=device)
            tower.requires_grad_(False).eval()
            logger.info("Loaded Muse Glimmer vision encoder onto %s", device)
            return VisionEncoderSubmodule(
                tower, self.vision_config, self.encoder(node_name).output,
            )

        from mstar.model.muse.components.text import MuseGlimmerForCausalLM

        model = self._build(lambda: MuseGlimmerForCausalLM(self.config, tp_group), dtype, device)
        weight_loader.load_text_weights(model, weights_dir, device=device)
        model.requires_grad_(False).eval()
        logger.info("Loaded Muse Glimmer LLM onto %s", device)
        return LLMSubmodule(model, self.config, self.encoders)
