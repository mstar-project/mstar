"""QwenImageModel: Qwen-Image-Edit-2511 (and 2509), editing 1-3 reference images from a prompt.

Architecture (4 nodes; the dit optionally owns one ragged attention resource):

    text_encoder  Qwen2.5-VL-7B: the vision tower over the condition images, then the language model over the
                  positive prompt (and the negative, when guided) with the images spliced in
    vae_encoder   the references' VAE latents, packed (Wan2.1-architecture VAE, 16 channels)
    dit           the 60-block dual-stream MMDiT, one Euler step per Loop iteration, true CFG with norm rescale
    vae_decoder   VAE decode -> uint8 image

Graph walks, all run for every request: ``encode_edit`` (text_encoder), ``encode_refs`` (vae_encoder), then
``image_edit``: Loop("denoise_loop", dit) -> vae_decoder -> EMIT image. The loop's cap is the request's step count.

Per-request knobs (``model_kwargs``): ``height``, ``width`` (default: the last image's aspect at ~1024^2, floored
to 16 as the pipeline does), ``num_inference_steps`` (40), ``guidance_scale`` (true CFG, 4.0; 1 turns it off),
``negative_prompt`` (" "), ``seed``. ``true_cfg_scale``, diffusers' name for the CFG scale, is refused with a pointer
to ``guidance_scale``: a request copied from the model card sends it next to ``guidance_scale: 1.0``, and taking that
1.0 would run without guidance.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import NodeResourceSpec, RaggedAttentionConfig, RaggedAttentionSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import Model, ProcessPromptOutput, TensorAndMetadata, WalkStep
from mstar.model.components.diffusion.denoise_loop import LATENTS
from mstar.model.components.diffusion.guidance import GUIDANCE_SCALE
from mstar.model.components.diffusion.image_io import OUTPUT_FORMAT_KWARGS, encode_image, image_output_format
from mstar.model.media_limits import PILLOW_IMAGE_FORMATS, check_image_size, decoding_upload, open_upright_image
from mstar.model.qwen_image.components.transformer import ROPE_POSITIONS
from mstar.model.qwen_image.config import (
    DENOISE_LOOP,
    DIT_ATTN,
    PICTURE_PREFIX,
    QWEN_IMAGE_EDIT_2511,
    SNAPSHOT_PATTERNS,
    TOKEN_PIXELS,
    QwenImageConfig,
    calculate_dimensions,
)
from mstar.model.qwen_image.submodules import (
    IMAGE_INPUTS,
    IMAGE_OUTPUT,
    PROMPT_LENGTHS,
    REF_LATENTS,
    TEXT_EMBEDS_NEG,
    TEXT_EMBEDS_POS,
    TEXT_INPUTS,
    VL_GRIDS,
    VL_PIXELS,
    QwenImageDenoiseSubmodule,
    QwenImageTextEncoderSubmodule,
    QwenImageVaeDecoderSubmodule,
    QwenImageVaeEncoderSubmodule,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

ENCODE_EDIT_WALK = "encode_edit"
ENCODE_REFS_WALK = "encode_refs"
IMAGE_EDIT_WALK = "image_edit"

ATTENTION_BACKENDS = ("flashinfer", "sdpa")


class QwenImageModel(Model):
    """Qwen-Image-Edit-Plus checkpoints (Edit-2511 by default, Edit-2509 by ``model_path_hf``)."""

    INTERNAL_KWARGS = (VL_GRIDS,)

    def __init__(
        self,
        model_path_hf: str = QWEN_IMAGE_EDIT_2511,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        attention_backend: str = "flashinfer",
        compile: bool = False,
        max_image_area: int = 2048 * 2048,
        **kwargs,
    ):
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {attention_backend!r}")
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # Dummy mode: get_submodule returns None for every node (CPU tests, no weights).
        self.skip_weight_loading = skip_weight_loading
        self.attention_backend = attention_backend
        # torch.compile per transformer block, static but for the text axis
        self.compile = bool(compile)
        # the largest output a request may ask for: the attention is quadratic in it
        self.max_image_area = int(max_image_area)
        self._snapshot = None
        self._config: QwenImageConfig | None = None
        self._processor = None
        self._vae: torch.nn.Module | None = None

    # ------------------------------------------------------------------ config
    @property
    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = resolve_snapshot_dir(self.model_path_hf, self.cache_dir, allow_patterns=SNAPSHOT_PATTERNS)
        return self._snapshot

    @property
    def config(self) -> QwenImageConfig:
        """Read from the checkpoint on first use; ``set_config`` pins one for tests."""
        if self._config is None:
            self._config = QwenImageConfig.from_snapshot(self.snapshot)
        return self._config

    def set_config(self, config: QwenImageConfig) -> None:
        self._config = config

    # ------------------------------------------------------------ structure
    def get_node_resources(self) -> list[NodeResourceSpec]:
        if self.attention_backend != "flashinfer":
            return []
        t = self.config.transformer
        return [RaggedAttentionSpec(
            resource_key=DIT_ATTN, nodes={"dit"},
            config=RaggedAttentionConfig(
                num_qo_heads=t.num_attention_heads, num_kv_heads=t.num_attention_heads, head_dim=t.attention_head_dim,
                # two guidance branches a request, each a real span and possibly a pad
                max_segments_per_request=4, dtype=torch.bfloat16, compile_op=self.compile,
            ),
        )]

    def get_default_sharding_config(self) -> ShardingConfig:
        # the dit's linears split over tp_size; its guidance branches over cfg_size
        return ShardingConfig(
            groups=[], tp_enabled_nodes={"dit"}, shard_dim={},
            cfg_enabled_nodes={"dit": len(QwenImageDenoiseSubmodule.GUIDANCE.branches)},
        )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_edit = GraphNode(
            name="text_encoder", input_names=[TEXT_INPUTS, VL_PIXELS],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS_POS, persist=True),
                     GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS_NEG, persist=True)],
        )
        encode_refs = GraphNode(
            name="vae_encoder", input_names=[IMAGE_INPUTS],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=REF_LATENTS, persist=True)],
        )
        loop = Loop(
            name=DENOISE_LOOP,
            section=GraphNode(
                name="dit", input_names=[TEXT_EMBEDS_POS, TEXT_EMBEDS_NEG, REF_LATENTS, LATENTS],
                # latents is the only loop-carried edge; the step index is the loop counter
                outputs=[GraphEdge(next_node="dit", name=LATENTS)],
            ),
            max_iters=self.config.max_denoise_steps,
            outputs=[GraphEdge(next_node="vae_decoder", name=LATENTS)],
        )
        decoder = GraphNode(
            name="vae_decoder", input_names=[LATENTS],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=IMAGE_OUTPUT, output_modality="image")],
        )
        return {
            ENCODE_EDIT_WALK: encode_edit,
            ENCODE_REFS_WALK: encode_refs,
            IMAGE_EDIT_WALK: Sequential([loop, decoder]),
        }

    # ---------------------------------------------------------------- inputs
    def _processor_for(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(str(self.snapshot / "processor"))
        return self._processor

    def max_input_files(self) -> dict[str, int]:
        return {"image": self.config.recipe.max_images}

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """The upload upright by its EXIF orientation and in RGB, as diffusers' ``load_image`` reads it, uint8
        ``[3, H, W]`` at its own size: ``process_prompt`` makes the two sizes the pipeline uses from it."""
        check_image_size(filepath, formats=PILLOW_IMAGE_FORMATS)
        with (
            decoding_upload(filepath, "the reference image"),
            open_upright_image(filepath, formats=PILLOW_IMAGE_FORMATS) as image,
        ):
            width, height = image.size
            if width < 32 or height < 32:
                raise ValueError(f"reference image too small: {width}x{height}; both sides must be >= 32 px")
            if max(width / height, height / width) > 8:
                raise ValueError(f"reference image aspect ratio too extreme: {width}x{height} (max 8:1)")
            array = np.array(image.convert("RGB"))
        return TensorAndMetadata(torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device))

    def request_kwargs(self) -> frozenset[str]:
        # true_cfg_scale is read to refuse it
        return frozenset({
            "height", "width", "num_inference_steps", GUIDANCE_SCALE, "negative_prompt", "true_cfg_scale",
            *OUTPUT_FORMAT_KWARGS,
        })

    def _knobs(self, kwargs: dict) -> dict:
        """The request's steps, guidance scale and negative prompt, validated."""
        recipe = self.config.recipe
        if kwargs.get("true_cfg_scale") is not None:
            raise ValueError("true_cfg_scale is not a knob of this server: send the CFG scale as guidance_scale")
        steps = kwargs.get("num_inference_steps")
        steps = recipe.default_steps if steps is None else steps
        fewest, most = self.config.scheduler.min_steps, self.config.max_denoise_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or not fewest <= steps <= most:
            raise ValueError(f"num_inference_steps must be an integer in [{fewest}, {most}]; got {steps!r}")
        scale = kwargs.get(GUIDANCE_SCALE)
        scale = recipe.default_guidance_scale if scale is None else scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
            raise ValueError(f"guidance_scale must be a finite number; got {scale!r}")
        negative = kwargs.get("negative_prompt")
        negative = recipe.default_negative_prompt if negative is None else negative
        if not isinstance(negative, str):
            raise ValueError(f"negative_prompt must be a string; got {negative!r}")
        return {"num_inference_steps": steps, GUIDANCE_SCALE: float(scale), "negative_prompt": negative}

    def _resolve_size(self, model_kwargs: dict, last_ref: tuple[int, int]) -> tuple[int, int]:
        """``(height, width)`` of the output: the request's, else the last reference's (its aspect at the VAE's area,
        the pipeline's default), each floored to a multiple of 16 as the pipeline does."""
        height = model_kwargs.get("height") or last_ref[0]
        width = model_kwargs.get("width") or last_ref[1]
        for name, value in (("height", height), ("width", width)):
            if isinstance(value, bool) or not isinstance(value, int) or value < TOKEN_PIXELS:
                raise ValueError(f"Qwen-Image {name} must be an integer of at least {TOKEN_PIXELS}; got {value!r}")
        height, width = height // TOKEN_PIXELS * TOKEN_PIXELS, width // TOKEN_PIXELS * TOKEN_PIXELS
        if height * width > self.max_image_area:
            raise ValueError(
                f"Qwen-Image {width}x{height} exceeds this server's max_image_area of {self.max_image_area} pixels"
            )
        return height, width

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> ProcessPromptOutput:
        """The pipeline's ``_get_qwen_prompt_embeds`` inputs: each image at the VL encoder's and the VAE's size, and
        the templated prompts with one "Picture i:" per image, through the checkpoint's own processor."""
        from PIL import Image

        if (output_modalities or ["image"])[0] != "image":
            raise ValueError(f"Qwen-Image-Edit generates images; got output modality {output_modalities!r}")
        if prompt is None or not prompt.strip():
            raise ValueError("Qwen-Image-Edit requires a non-empty text prompt")
        images = (tensors or {}).get(IMAGE_INPUTS) or []
        if not images:
            raise ValueError("Qwen-Image-Edit edits images: attach 1 to "
                             f"{self.config.recipe.max_images} reference images")
        knobs = self._knobs(kwargs)
        image_output_format(kwargs)  # fail a bad output knob now, not after the denoise
        recipe = self.config.recipe
        condition, vae_images = [], []
        for tensor in images:
            pil = Image.fromarray(tensor.cpu().permute(1, 2, 0).numpy())
            ratio = pil.width / pil.height
            condition.append(pil.resize(calculate_dimensions(recipe.condition_area, ratio), Image.Resampling.LANCZOS))
            resized = pil.resize(calculate_dimensions(recipe.vae_area, ratio), Image.Resampling.LANCZOS)
            vae_images.append(torch.from_numpy(np.array(resized)).permute(2, 0, 1).contiguous())
        size = self._resolve_size(kwargs, tuple(vae_images[-1].shape[-2:]))
        prefix = "".join(PICTURE_PREFIX.format(i + 1) for i in range(len(images)))
        processor = self._processor_for()

        def encode(text: str):
            return processor(text=[recipe.template.format(prefix + text)], images=condition, padding=True,
                             return_tensors="pt")

        positive = encode(prompt)
        prompts = [positive["input_ids"][0]]
        if QwenImageDenoiseSubmodule.GUIDANCE.active(knobs):
            prompts.append(encode(knobs["negative_prompt"])["input_ids"][0])
        # The text's rotary positions start past the widest image axis's half, in a table of ROPE_POSITIONS (the
        # reference's too, which fails the same request mid-run); a longer half also overruns the image's own.
        grids = [self.config.latent_grid(*size), *(self.config.latent_grid(*t.shape[-2:]) for t in vae_images)]
        start = max(max(h // 2, w // 2) for h, w in grids)
        text = max(len(ids) for ids in prompts) - recipe.drop_tokens
        if start + text > ROPE_POSITIONS:
            raise ValueError(
                f"Qwen-Image-Edit: a {text}-token prompt after a {start}-token half image axis exceeds the "
                f"transformer's {ROPE_POSITIONS} rotary positions; shorten the prompt or the image's longer side"
            )
        tensors_out = {TEXT_INPUTS: prompts, VL_PIXELS: [positive["pixel_values"]], IMAGE_INPUTS: vae_images}
        return ProcessPromptOutput(tensors_out, {VL_GRIDS: positive["image_grid_thw"].tolist()})

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the loop ends on the request's own step count, and is never speculated past it
        return {DENOISE_LOOP: self._knobs(model_kwargs)["num_inference_steps"]}

    def walk_plan(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict,
    ) -> list[WalkStep]:
        refs = [(int(info.dims[-2]), int(info.dims[-1])) for info in input_signals.get(IMAGE_INPUTS) or []]
        if not input_signals.get(TEXT_INPUTS) or not refs:
            raise ValueError("Qwen-Image-Edit requires a text prompt and reference images")
        height, width = self._resolve_size(model_kwargs, refs[-1])
        knobs = self._knobs(model_kwargs)
        meta = {
            "height": height,
            "width": width,
            "num_inference_steps": knobs["num_inference_steps"],
            GUIDANCE_SCALE: knobs[GUIDANCE_SCALE],
            "ref_grids": [list(self.config.latent_grid(h, w)) for h, w in refs],
            VL_GRIDS: model_kwargs[VL_GRIDS],
            PROMPT_LENGTHS: [int(info.dims[0]) for info in input_signals[TEXT_INPUTS]],
        }
        logger.info(
            "Qwen-Image-Edit request: %dx%d steps=%d guidance=%s refs=%s seed=%s", width, height,
            meta["num_inference_steps"], meta[GUIDANCE_SCALE], meta["ref_grids"], model_kwargs.get("seed", "auto"),
        )
        return [WalkStep(walk, step_metadata=meta) for walk in (ENCODE_EDIT_WALK, ENCODE_REFS_WALK, IMAGE_EDIT_WALK)]

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality != "image":
            raise ValueError(f"unsupported output modality for Qwen-Image-Edit: {modality!r}")
        return encode_image(output, request_kwargs)

    # ------------------------------------------------------------- loading
    def get_autocast_dtype(self):
        # the checkpoint dtypes (bf16) throughout, as the reference pipeline runs
        return None

    def get_submodule(
        self, node_name: str, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None, moe_parallel=None,
        cfg_group=None,
    ) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        from mstar.model.qwen_image import weight_loader

        config, snapshot = self.config, self.snapshot
        if node_name == "text_encoder":
            return QwenImageTextEncoderSubmodule(
                weight_loader.build_text_encoder(config, snapshot, device),
                weight_loader.build_vision_tower(config, snapshot, device), config,
            )
        if node_name in ("vae_encoder", "vae_decoder"):
            if self._vae is None:
                self._vae = weight_loader.build_vae(snapshot, device)
            if node_name == "vae_encoder":
                return QwenImageVaeEncoderSubmodule(self._vae)
            return QwenImageVaeDecoderSubmodule(self._vae, config)
        if node_name == "dit":
            return QwenImageDenoiseSubmodule(
                weight_loader.build_transformer(config, snapshot, device, comm_group=tp_group), config,
                loop_name=DENOISE_LOOP, attn_resource_key=DIT_ATTN if self.attention_backend == "flashinfer" else None,
                compile_transformer=self.compile, cfg_group=cfg_group,
            )
        logger.warning("Qwen-Image has no submodule for node %r; running it dummy", node_name)
        return None
