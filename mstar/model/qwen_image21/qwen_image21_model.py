"""QwenImage21Model: Qwen-Image-2.1 (and Turbo) text-to-image and editing with up to 10 references.

Nodes (submodules.py): ``text_encoder`` (Qwen3-VL-8B), ``vae_encoder``, ``dit`` (one paged KV cache
and its attention), ``vae_decoder``.

Walks, in a request's plan:

    encode / encode_images    the prompt (with the references through the vision tower) and the
                              references' latents; persisted
    prefix_text / prefix_image one step per prompt span, in order: text₀, img₁, text₁, …, imgₙ, textₙ,
                              each committing its K/V over the spans before it (layout.py)
    image_gen                 Loop("denoise_loop", dit) over that cache -> vae_decoder -> EMIT image

The spans through the last image are keyed for the prefix cache, so repeat edits of the same images
reuse their K/V whatever the prompt. CFG (opt-in: ``guidance_scale`` > 1 with a ``negative_prompt``)
forks ``uncond`` off ``main`` after the last image, writes the negative prompt's tail on it, and
denoises both labels in one step. Request knobs: docs/models.rst.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import (
    KVLabel,
    LabelLayout,
    Model,
    PrefixKeys,
    ProcessPromptOutput,
    TensorAndMetadata,
    WalkStep,
)
from mstar.model.components.diffusion.denoise_loop import LATENTS
from mstar.model.components.diffusion.guidance import GUIDANCE_SCALE
from mstar.model.components.diffusion.image_io import OUTPUT_FORMAT_KWARGS, encode_image, image_output_format
from mstar.model.media_limits import PILLOW_IMAGE_FORMATS, check_image_size, decoding_upload, open_upright_image
from mstar.model.qwen_image21.components.transformer import ROPE_POSITIVE
from mstar.model.qwen_image21.config import (
    ATTN,
    ATTN_GEN,
    DEFAULT_OUTPUT_RESOLUTION,
    DENOISE_LOOP,
    DIT_NODE,
    ENCODE_IMAGES_WALK,
    ENCODE_WALK,
    IMAGE_GEN_WALK,
    KV,
    MAIN,
    MAX_REFERENCE_IMAGES,
    PREFIX_IMAGE_WALK,
    PREFIX_TEXT_WALK,
    QWEN_IMAGE21,
    SIZE_MULTIPLE,
    SYSTEM_PROMPT,
    TEXT_ENCODER_NODE,
    UNCOND,
    VAE_DECODER_NODE,
    VAE_ENCODER_NODE,
    VAE_SCALE,
    QwenImage21Config,
)
from mstar.model.qwen_image21.layout import PREFIX_IDS, RequestLayout, prompt_spans
from mstar.model.qwen_image21.submodules import (
    DROP,
    GRIDS,
    IMAGE_OUTPUT,
    LAYOUT,
    PROMPT_LENGTHS,
    REF_LATENTS,
    REF_PIXELS,
    SPAN,
    STEPS,
    TEXT_EMBEDS,
    TEXT_INPUTS,
    VISION_PIXELS,
    DitKey,
    QwenImage21DenoiseSubmodule,
    QwenImage21TextEncoderSubmodule,
    QwenImage21VaeDecoderSubmodule,
    QwenImage21VaeEncoderSubmodule,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

# A ceiling for the loop graph; a request runs its own step count (get_loop_caps).
MAX_INFERENCE_STEPS = 200
IMAGE_PAD = "<|image_pad|>"
VISION_TOKENS = ("<|vision_start|>", IMAGE_PAD, "<|vision_end|>")
# The prefix cache keys a reference by its file and these; the tokens after it by what the
# preprocessing makes of the prompt.
PREPROCESS_VERSION = "qwen_image21/1:lanczos-rgba:alpha-over-white"


def calculate_dimensions(target_area: float, ratio: float) -> tuple[int, int]:
    """The pipeline's ``calculate_dimensions``: ``(width, height)`` of about ``target_area`` px at
    ``ratio``, each rounded to the 32-px grid."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / SIZE_MULTIPLE) * SIZE_MULTIPLE, round(height / SIZE_MULTIPLE) * SIZE_MULTIPLE


def _template(prompt: str, images: int) -> str:
    """The pipeline's raw template strings (not ``apply_chat_template``: they tokenize differently)."""
    head = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    vision = "".join(
        f"{' ' if i else ''}<image{i + 1}><|vision_start|>{IMAGE_PAD}<|vision_end|>" for i in range(images)
    )
    return f"{head}{vision}{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _over_white(image):
    """The Qwen3-VL copy of an RGBA reference: alpha composited over white, as the checkpoint was trained."""
    from PIL import Image

    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


class QwenImage21Model(Model):
    """Qwen-Image-2.1 (7B single-stream DiT whose condition is a cached block-causal prefix)."""

    def __init__(
        self,
        model_path_hf: str = QWEN_IMAGE21,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        max_seq_len: int = 65_536,
        max_num_pages: int = 512,
        max_batch_size: int = 4,
        accelerator_graph: bool = False,
        compile: bool = False,
        capture_sizes: list[list[int]] | None = None,
        capture_batch_sizes: list[int] | None = None,
        async_scheduling: bool = True,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # dummy mode: get_submodule returns None for every node (CPU tests)
        self.skip_weight_loading = skip_weight_loading
        # a request's prefix plus its target, in tokens; the pool holds max_num_pages of 128
        self.max_seq_len = int(max_seq_len)
        self.max_num_pages = int(max_num_pages)
        self.max_batch_size = int(max_batch_size)
        self.accelerator_graph = bool(accelerator_graph)
        # torch.compile the denoise target's blocks and the VAE's decoder (not bit-exact: gated against the fp32
        # pipeline; the compiled encoder missed more edit cases than eager, so it stays eager)
        self.compile = bool(compile)
        self.capture_sizes = [tuple(int(v) for v in s) for s in (capture_sizes or [[1024, 1024]])]
        self.capture_batch_sizes = [int(b) for b in (capture_batch_sizes or [1, 2, 4])]
        self.async_scheduling = bool(async_scheduling)
        self._snapshot = None
        self._config: QwenImage21Config | None = None
        self._processor = None
        self._drop_idx: int | None = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # ------------------------------------------------------------------ config
    @property
    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = resolve_snapshot_dir(self.model_path_hf, self.cache_dir)
        return self._snapshot

    @property
    def config(self) -> QwenImage21Config:
        """Read from the checkpoint on first use; ``set_config`` pins one for tests."""
        if self._config is None:
            self._config = QwenImage21Config.from_snapshot(self.snapshot)
        return self._config

    def set_config(self, config: QwenImage21Config) -> None:
        self._config = config

    def checkpoint_path(self) -> str | None:
        return str(self.snapshot)

    def preprocess_fingerprint(self) -> str:
        """Everything outside a reference's file and size that changes the K/V of it and of the text
        after it: the system turn the encoder reads, Qwen3-VL's pixel bounds, the resize and the
        alpha-over-white rule for the vision copy."""
        cfg = self.config
        return f"{type(self).__name__}:{PREPROCESS_VERSION}:{SYSTEM_PROMPT}:{cfg.min_pixels}:{cfg.max_pixels}"

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        # the K/V the reference caches is bf16; another cache dtype is another model
        dtype = config.get("autocast_dtype", "bfloat16")
        if str(dtype).removeprefix("torch.") not in ("bfloat16", "bf16"):
            raise ValueError(f"{config_path}: Qwen-Image-2.1 caches its prefix K/V in bf16; got autocast_dtype={dtype}")

    # ------------------------------------------------------------ structure
    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The dit's paged KV (the prefix, then each denoise step's scratch target) and its attention: paged for
        the prefix walks, which write the cache; dense for eager denoise steps, which attend the frozen prefix
        and their own K/V and write nothing a later step reads (a captured step stays paged).
        No position resource: the K/V are cached after the model's own 3-axis rope."""
        cfg = self.config.transformer
        heads = cfg.num_attention_heads
        specs = [
            KVSpec(resource_key=KV, nodes={DIT_NODE}, config=PagedKVConfig(
                num_layers=cfg.num_layers, num_kv_heads=heads, head_dim=cfg.attention_head_dim, num_qo_heads=heads,
                max_seq_len=self.max_seq_len, max_num_pages=self.max_num_pages,
            )),
            AttentionSpec(resource_key=ATTN, nodes={DIT_NODE}, config=AttentionConfig(kv_cache=KV)),
        ]
        if not self.accelerator_graph:
            specs.append(AttentionSpec(resource_key=ATTN_GEN, nodes={DIT_NODE},
                                       config=AttentionConfig(kv_cache=KV, backend=AttnBackend.DENSE)))
        return specs

    def kv_labels(self):
        return {KV: {MAIN: KVLabel(keys=PrefixKeys(PREFIX_IDS, (PREFIX_TEXT_WALK, PREFIX_IMAGE_WALK))),
                     UNCOND: KVLabel()}}

    def get_request_resource_configs(self, input_modalities, output_modalities, model_kwargs):
        return {KV: KVReqConfig(needed_labels=[MAIN, UNCOND] if self._guided(model_kwargs or {}) else [MAIN])}

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def text_node(*inputs: str) -> GraphNode:
            return GraphNode(
                name=TEXT_ENCODER_NODE, input_names=list(inputs), enable_async_scheduling=self.async_scheduling,
                outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS, persist=True)],
            )

        def prefix_node(source: str) -> GraphNode:
            return GraphNode(name=DIT_NODE, input_names=[source], outputs=[],
                             enable_async_scheduling=self.async_scheduling)

        encode_images = Parallel([
            text_node(TEXT_INPUTS, VISION_PIXELS),
            GraphNode(
                name=VAE_ENCODER_NODE, input_names=[REF_PIXELS], enable_async_scheduling=self.async_scheduling,
                outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=REF_LATENTS, persist=True)],
            ),
        ])
        loop = Loop(
            name=DENOISE_LOOP,
            section=GraphNode(
                name=DIT_NODE, input_names=[LATENTS], outputs=[GraphEdge(next_node=DIT_NODE, name=LATENTS)],
                enable_async_scheduling=self.async_scheduling,
            ),
            max_iters=MAX_INFERENCE_STEPS,
            outputs=[GraphEdge(next_node=VAE_DECODER_NODE, name=LATENTS)],
        )
        decoder = GraphNode(
            name=VAE_DECODER_NODE, input_names=[LATENTS], enable_async_scheduling=self.async_scheduling,
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=IMAGE_OUTPUT, output_modality="image")],
        )
        return {
            ENCODE_WALK: text_node(TEXT_INPUTS),
            ENCODE_IMAGES_WALK: encode_images,
            PREFIX_TEXT_WALK: prefix_node(TEXT_EMBEDS),
            PREFIX_IMAGE_WALK: prefix_node(REF_LATENTS),
            IMAGE_GEN_WALK: Sequential([loop, decoder]),
        }

    # ---------------------------------------------------------------- inputs
    def _ensure_processor(self):
        """Qwen3-VL's processor (its tokenizer and image processor), and the system turn's length,
        which the pipeline drops from the encoder's hidden states."""
        if self._processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(str(self.snapshot / "processor"))
            system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
            self._drop_idx = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
            self._processor = processor
        return self._processor

    def warmup_preprocess(self) -> None:
        self._ensure_processor()

    def _tokenize(self, text: str) -> torch.Tensor:
        return torch.tensor(self._ensure_processor().tokenizer(text)["input_ids"], dtype=torch.long)

    def _vision_inputs(self, images: list) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        out = self._ensure_processor().image_processor(images=images, return_tensors="pt")
        return out["pixel_values"].to(torch.float32), [tuple(int(v) for v in g) for g in out["image_grid_thw"]]

    def max_input_files(self) -> dict[str, int]:
        return {"image": MAX_REFERENCE_IMAGES, "audio": 0, "video": 0}

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """A reference as decoded RGBA, uint8 ``[4, H, W]``, upright: ``process_prompt`` resizes it with
        PIL, as the reference does, once the request's ``output_resolution`` is known."""
        check_image_size(filepath, formats=PILLOW_IMAGE_FORMATS)
        with (
            decoding_upload(filepath, "the reference image"),
            open_upright_image(filepath, formats=PILLOW_IMAGE_FORMATS) as image,
        ):
            rgba = image.convert("RGBA")
        return TensorAndMetadata(torch.from_numpy(np.asarray(rgba).copy()).permute(2, 0, 1).contiguous())

    def request_kwargs(self) -> frozenset[str]:
        # true_cfg_scale is read only to refuse it (the knob is guidance_scale)
        return frozenset({
            "height", "width", "num_inference_steps", "guidance_scale", "negative_prompt", "output_resolution",
            "true_cfg_scale", *OUTPUT_FORMAT_KWARGS,
        })

    @staticmethod
    def _guided(model_kwargs: dict) -> bool:
        """CFG runs, as the pipeline's ``true_cfg_scale``, only with a scale above 1 and a negative prompt."""
        scale = model_kwargs.get("guidance_scale")
        if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))
                                  or not math.isfinite(scale)):
            raise ValueError(f"guidance_scale must be a finite number; got {scale!r}")
        return scale is not None and float(scale) > 1.0 and model_kwargs.get("negative_prompt") is not None

    def _output_resolution(self, model_kwargs: dict) -> int:
        value = model_kwargs.get("output_resolution")
        value = DEFAULT_OUTPUT_RESOLUTION if value is None else value
        if isinstance(value, bool) or not isinstance(value, int) or value < SIZE_MULTIPLE:
            raise ValueError(f"Qwen-Image-2.1 output_resolution must be an integer of at least {SIZE_MULTIPLE}")
        return value

    def _steps(self, model_kwargs: dict) -> int:
        cfg = self.config
        steps = model_kwargs.get("num_inference_steps")
        if cfg.sample_sigmas is not None:
            if steps is not None and int(steps) != len(cfg.sample_sigmas):
                raise ValueError(
                    f"this distilled Qwen-Image-2.1 checkpoint samples exactly {len(cfg.sample_sigmas)} steps "
                    f"(its sample_sigmas); got num_inference_steps={steps}"
                )
            return len(cfg.sample_sigmas)
        steps = cfg.default_num_inference_steps if steps is None else steps
        fewest = cfg.scheduler.min_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or not fewest <= steps <= MAX_INFERENCE_STEPS:
            raise ValueError(
                f"Qwen-Image-2.1 num_inference_steps must be an integer in [{fewest}, {MAX_INFERENCE_STEPS}]")
        return steps

    def _output_size(self, model_kwargs: dict, references: list, resolution: int) -> tuple[int, int]:
        """``(height, width)``: as asked, else at ``resolution``^2 in the last reference's aspect ratio
        (a square without one), snapped down to the 32-px grid as the pipeline does."""
        height, width = model_kwargs.get("height"), model_kwargs.get("width")
        if references:
            w, h = references[-1].size
            default_w, default_h = calculate_dimensions(resolution * resolution, w / h)
        else:
            default_w = default_h = resolution
        height, width = height or default_h, width or default_w
        for name, value in (("height", height), ("width", width)):
            if isinstance(value, bool) or not isinstance(value, int) or value < SIZE_MULTIPLE:
                raise ValueError(f"Qwen-Image-2.1 {name} must be an integer of at least {SIZE_MULTIPLE}")
        return height // SIZE_MULTIPLE * SIZE_MULTIPLE, width // SIZE_MULTIPLE * SIZE_MULTIPLE

    def _fit_reference(self, image: torch.Tensor, resolution: int):
        """The reference at about ``resolution``^2 px on the 32-px grid, RGBA, LANCZOS (the pipeline's
        ``VaeImageProcessor.resize``); both copies are made from it."""
        from PIL import Image

        rgba = Image.fromarray(image.permute(1, 2, 0).contiguous().cpu().numpy(), mode="RGBA")
        width, height = calculate_dimensions(resolution * resolution, rgba.size[0] / rgba.size[1])
        cfg = self.config
        if not cfg.min_pixels <= width * height <= cfg.max_pixels:
            raise ValueError(
                f"Qwen-Image-2.1 resizes a {rgba.size[0]}x{rgba.size[1]} reference to {width}x{height} at "
                f"output_resolution {resolution}, outside Qwen3-VL's {cfg.min_pixels} to {cfg.max_pixels} px, "
                "which would resize it again and break its 2x2 latents per vision slot"
            )
        return rgba, rgba.resize((width, height), Image.Resampling.LANCZOS)

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> ProcessPromptOutput:
        """Validate the request (a 400 here), template and tokenize the prompt (and the negative one under
        CFG) around the references, resize the references, and lay out the DiT's prefix spans."""
        if (output_modalities or ["image"]) != ["image"]:
            raise ValueError(f"Qwen-Image-2.1 generates one image; got output modalities {output_modalities}")
        if any(m not in ("text", "image") for m in input_modalities or []):
            raise ValueError(f"Qwen-Image-2.1 takes a prompt and reference images; got inputs {input_modalities}")
        if kwargs.get("true_cfg_scale") is not None:
            raise ValueError("Qwen-Image-2.1 takes its guidance scale as guidance_scale, not true_cfg_scale")
        if image_output_format(kwargs) == "jpeg":
            raise ValueError("Qwen-Image-2.1 generates RGBA images, which JPEG cannot carry; ask for png or webp")
        images = list((tensors or {}).get("image_inputs") or [])
        if "image" in (input_modalities or []) and not images:
            raise ValueError("request declared an image input but no image arrived")
        resolution = self._output_resolution(kwargs)
        self._steps(kwargs)
        guided = self._guided(kwargs)
        for text in (prompt, kwargs.get("negative_prompt") if guided else None):
            if text and any(token in text for token in VISION_TOKENS):
                # the references' features are spliced in at image pads, so a typed one would shift them
                raise ValueError(f"Qwen-Image-2.1 prompts can't contain Qwen's vision tokens {list(VISION_TOKENS)}")
        if not guided and kwargs.get("guidance_scale") is not None and float(kwargs["guidance_scale"]) > 1.0:
            logger.info("guidance_scale is ignored without a negative_prompt (classifier-free guidance stays off)")

        fitted = [self._fit_reference(image, resolution) for image in images]
        originals, resized = [pair[0] for pair in fitted], [pair[1] for pair in fitted]
        height, width = self._output_size(kwargs, originals, resolution)
        grids: list[tuple[int, int, int]] = []
        pixel_values = None
        if resized:
            pixel_values, grids = self._vision_inputs([_over_white(image) for image in resized])
            for image, grid in zip(resized, grids, strict=True):
                if grid != (1, image.size[1] // VAE_SCALE, image.size[0] // VAE_SCALE):
                    raise ValueError(f"unexpected vision grid {grid} for a {image.size[0]}x{image.size[1]} reference")
        merge = self.config.vision.spatial_merge_size

        def ids_for(text: str) -> torch.Tensor:
            text = _template(text or " ", len(resized))  # Qwen has no bos: an empty prompt reads a space
            for t, h, w in grids:
                text = text.replace(IMAGE_PAD, "<|placeholder|>" * (t * h * w // merge**2), 1)
            return self._tokenize(text.replace("<|placeholder|>", IMAGE_PAD))

        ids = ids_for(prompt or "")
        uncond_ids = ids_for(kwargs["negative_prompt"]) if guided else None
        self._ensure_processor()
        drop = self._drop_idx
        latent_grids = [(image.size[1] // VAE_SCALE, image.size[0] // VAE_SCALE) for image in resized]
        layout = prompt_spans(
            ids[drop:], self.config.text.image_token_id, latent_grids, (height // VAE_SCALE, width // VAE_SCALE),
            None if uncond_ids is None else uncond_ids[drop:],
        )
        kv_layout = layout.kv_layout(resolution)
        self._check_layout(layout, kv_layout)
        body = ids[drop:]
        out: NameToTensorList = {
            TEXT_INPUTS: [ids if uncond_ids is None else torch.cat([ids, uncond_ids])],
            # the spans the prefix cache keys: the text before the last reference
            PREFIX_IDS: [body[s.rows:s.rows + s.length] for s in layout.spans[:-1] if s.is_text],
            # the decoded uploads are spent; the store would hold them for the whole request
            "image_inputs": [],
        }
        if resized:
            out[VISION_PIXELS] = [pixel_values]
            out[REF_PIXELS] = [torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous()
                               for image in resized]
        metadata = {
            "kv_layout": {KV: kv_layout},
            LAYOUT: layout.to_metadata(),
            PROMPT_LENGTHS: [int(ids.numel())] + ([] if uncond_ids is None else [int(uncond_ids.numel())]),
            DROP: drop,
            GRIDS: [list(grid) for grid in grids],
        }
        return ProcessPromptOutput(out, metadata)

    def _check_layout(self, layout: RequestLayout, kv_layout: dict[str, LabelLayout]) -> None:
        """A 400 for a request the rope or the cache cannot hold, before anything runs. The engine bounds
        each label's positions, which an image advances by ``max(h, w)``, not its ``h * w`` tokens, so
        each label's tokens (``uncond``'s negative prompt too) are bounded here."""
        if max(layout.frames()) >= ROPE_POSITIVE:
            raise ValueError(f"Qwen-Image-2.1's rope covers {ROPE_POSITIVE} positions; this prompt reaches "
                             f"{max(layout.frames())}")
        for label, stream in kv_layout.items():
            tokens = sum(span.length for span in stream.spans) + stream.scratch
            if tokens > self.max_seq_len:
                raise ValueError(
                    f"Qwen-Image-2.1 request holds {tokens} tokens of {label!r} K/V (prompt, references and "
                    f"target), above this server's max_seq_len of {self.max_seq_len}"
                )

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {DENOISE_LOOP: self._steps(model_kwargs)}

    # ------------------------------------------------------------ walk plan
    def walk_plan(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict,
    ) -> list[WalkStep]:
        """Encode, one prefix step per span of the layout ``process_prompt`` made, then denoise."""
        model_kwargs = model_kwargs or {}
        if LAYOUT not in model_kwargs:
            raise ValueError("Qwen-Image-2.1 requires a prompt laid out by process_prompt")
        layout = RequestLayout.from_metadata(model_kwargs[LAYOUT])
        guided = self._guided(model_kwargs)
        params = {
            LAYOUT: model_kwargs[LAYOUT], PROMPT_LENGTHS: model_kwargs[PROMPT_LENGTHS], DROP: model_kwargs[DROP],
            GRIDS: model_kwargs[GRIDS], STEPS: self._steps(model_kwargs),
            # 1 (the dit's Guidance inactive) unless the request runs CFG
            GUIDANCE_SCALE: float(model_kwargs["guidance_scale"]) if guided else 1.0,
        }
        encode = ENCODE_IMAGES_WALK if layout.images else ENCODE_WALK
        prefix = [
            WalkStep(PREFIX_TEXT_WALK if span.is_text else PREFIX_IMAGE_WALK, step_metadata={**params, SPAN: k})
            for k, span in enumerate(layout.spans)
        ]
        h, w = layout.target
        logger.info(
            "Qwen-Image-2.1 request: %dx%d steps=%d references=%d spans=%d cfg=%s seed=%s", w * VAE_SCALE,
            h * VAE_SCALE, params[STEPS], layout.images, len(layout.spans), guided,
            model_kwargs.get("seed", "auto"),
        )
        return [WalkStep(encode, step_metadata=params), *prefix, WalkStep(IMAGE_GEN_WALK, step_metadata=params)]

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality != "image":
            raise ValueError(f"unsupported output modality for Qwen-Image-2.1: {modality!r}")
        return encode_image(output, request_kwargs)

    # ------------------------------------------------------------- loading
    def get_autocast_dtype(self):
        # the KV cache's dtype; every node opts out of autocast and keeps the reference's dtypes
        return torch.bfloat16

    def get_default_sharding_config(self) -> ShardingConfig:
        # the dit's linears split over tp_size; its guidance branches over cfg_size (each cfg rank holds both labels'
        # prefixes, so its branch attends its own label's)
        return ShardingConfig(
            groups=[], tp_enabled_nodes={DIT_NODE}, shard_dim={}, sp_enabled_nodes=set(),
            cfg_enabled_nodes={DIT_NODE: len(QwenImage21DenoiseSubmodule.GUIDANCE.branches)},
        )

    def capture_buckets(self) -> list[tuple[str, Hashable]]:
        if not self.accelerator_graph:
            return []
        return [(IMAGE_GEN_WALK, DitKey(h // VAE_SCALE, w // VAE_SCALE)) for h, w in self.capture_sizes]

    def get_submodule(self, node_name: str, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None,
                      moe_parallel=None, cfg_group=None):
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, tp_group, cfg_group)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(self, node_name: str, device, tp_group=None, cfg_group=None) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        from mstar.model.qwen_image21 import weight_loader

        snapshot, config = self.snapshot, self.config
        if node_name == TEXT_ENCODER_NODE:
            return QwenImage21TextEncoderSubmodule(
                weight_loader.build_text_encoder(config, snapshot, device),
                weight_loader.build_vision_tower(snapshot, device),
            )
        if node_name == VAE_ENCODER_NODE:
            return QwenImage21VaeEncoderSubmodule(weight_loader.build_vae(snapshot, device))
        if node_name == VAE_DECODER_NODE:
            return QwenImage21VaeDecoderSubmodule(weight_loader.build_vae(snapshot, device), compile=self.compile)
        if node_name == DIT_NODE:
            return QwenImage21DenoiseSubmodule(
                weight_loader.build_transformer(config.transformer, snapshot, device, comm_group=tp_group), config,
                loop_name=DENOISE_LOOP, max_batch_size=self.max_batch_size,
                compile_transformer=self.compile, cfg_group=cfg_group,
                capture_buckets=self.capture_buckets(), capture_batch_sizes=self.capture_batch_sizes,
            )
        logger.warning("Qwen-Image-2.1 has no submodule for node %r; running it dummy", node_name)
        return None
