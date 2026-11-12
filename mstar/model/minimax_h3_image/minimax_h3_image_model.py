"""H3ImageModel: MiniMax-H3-Image text-to-image and reference-image editing.

Nodes:
    text_encoder   Qwen3-VL-32B to layer 50 (references through its vision tower), then
                   condition_proj and the token refiner; once per request (H3's node)
    vae_encoder    the references' conditioning latents
    dit            MiniMax-H3's packed-sequence DiT without audio, one forward per step of a
                   fixed six-step grid, each step through its own output head
                   (guidance-distilled: no unconditional pass)
    vae_decoder    the 2D ViT decoder, in tiles

Walks: encode_text or encode_references, then generate_image or generate_image_edit: the
denoise loop and the decoder. Request knobs: docs/models.rst.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import NodeResourceSpec, RaggedAttentionConfig, RaggedAttentionSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import Model, TensorAndMetadata, WalkStep
from mstar.model.components.diffusion.image_io import OUTPUT_FORMAT_KWARGS, encode_image, image_output_format
from mstar.model.media_limits import PILLOW_IMAGE_FORMATS, check_image_size, decoding_upload, open_upright_image
from mstar.model.minimax_h3.config import (
    DENOISE_LOOP,
    DIT_ATTN,
    DIT_NODE,
    REFINER_ATTN,
    TEXT_ENCODER_NODE,
    VAE_DECODER_NODE,
    VAE_ENCODER_NODE,
)
from mstar.model.minimax_h3.keyframes import presentation, vision_spans
from mstar.model.minimax_h3.minimax_h3_model import (
    VISION_TOKENS,
    _positive_int,
    import_image_processor,
    ulysses_options,
)
from mstar.model.minimax_h3.submodules import (
    COND_LATENTS,
    LATENTS,
    TEXT_EMBEDS,
    TEXT_INPUTS,
    VISION_PIXELS,
    H3TextEncoderSubmodule,
)
from mstar.model.minimax_h3_image.config import (
    CANVAS_MULTIPLE,
    DEFAULT_MIN_GENERATION_AREA,
    DEFAULT_SIZE,
    H3_IMAGE_REPO,
    MAX_REFERENCES,
    MAX_TEXT_TOKENS,
    MIN_GENERATION_AREA_RANGE,
    PICTURE_LABEL,
    REFERENCE_SHORT_EDGE,
    SNAPSHOT_PATTERNS,
    VISION_SHORT_EDGE,
    H3ImageConfig,
    bound_short_edge,
    canvas_size,
)
from mstar.model.minimax_h3_image.layout import H3ImageLayout
from mstar.model.minimax_h3_image.submodules import (
    IMAGE_OUTPUT,
    REFERENCE_PIXELS,
    H3ImageDecoderSubmodule,
    H3ImageDenoiseSubmodule,
    H3ImageEncoderSubmodule,
    layout_from_metadata,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

ENCODE_TEXT_WALK = "encode_text"
ENCODE_REFERENCES_WALK = "encode_references"
GENERATE_WALK = "generate_image"
GENERATE_EDIT_WALK = "generate_image_edit"  # the dit also reads cond_latents
ATTENTION_BACKENDS = ("flashinfer", "sdpa")
IMAGE_INPUTS = "image_inputs"
VAE_RATIO = 16
# A ceiling for the loop graph; a request runs the checkpoint's grid (get_loop_caps).
MAX_STEPS = 16


class H3ImageModel(Model):
    """MiniMax-H3-Image (33B DiT, guidance- and step-distilled) text-to-image and editing."""

    def __init__(
        self,
        model_path_hf: str = H3_IMAGE_REPO,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        attention_backend: str = "flashinfer",
        max_packed_rows: int = 131_072,
        async_scheduling: bool = True,
        compile: bool = False,
        row_bucket: int | None = None,
        vae_tile_batch: int = 1,
        vae_compile: bool = False,
        ulysses_head_groups: int = 1,
        ulysses_group_streams: bool = False,
    ):
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {attention_backend!r}")
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # dummy mode: get_submodule returns None for every node (CPU tests)
        self.skip_weight_loading = skip_weight_loading
        self.attention_backend = attention_backend
        # Prompt, reference and generated rows; a step's attention grows with their square. The
        # default covers the reference's largest request: a 16 MP image (16,384 rows) with 16
        # references at a 2048-px short edge (4,096 rows each, ~1k vision tokens each) and a
        # 4096-token prompt, ~102k rows.
        self.max_packed_rows = int(max_packed_rows)
        self.async_scheduling = bool(async_scheduling)
        self.compile = bool(compile)  # the DiT's blocks
        # the DiT's packed rows pad to a multiple of this (and of the SP size): one compiled
        # graph per 256 prompt tokens, not per prompt length
        self.row_bucket = (256 if self.compile else 1) if row_bucket is None else int(row_bucket)
        if self.row_bucket < 1:
            raise ValueError(f"row_bucket must be a positive int, got {row_bucket!r}")
        self.vae_tile_batch = int(vae_tile_batch)  # 1 decodes one tile a call, as the reference
        self.vae_compile = bool(vae_compile)
        # under SP: the DiT's exchange pipelined over head groups on the copy engines (MiniMax-H3's knobs)
        self.ulysses_head_groups, self.ulysses_group_streams = ulysses_options(ulysses_head_groups,
                                                                               ulysses_group_streams)
        self._snapshot = None
        self._config: H3ImageConfig | None = None
        self.tokenizer = None
        self._image_processor = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # ------------------------------------------------------------------ config
    @property
    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = resolve_snapshot_dir(self.model_path_hf, self.cache_dir, allow_patterns=SNAPSHOT_PATTERNS)
        return self._snapshot

    @property
    def config(self) -> H3ImageConfig:
        """Read from the checkpoint on first use; ``set_config`` pins one for tests."""
        if self._config is None:
            self._config = H3ImageConfig.from_snapshot(self.snapshot)
        return self._config

    def set_config(self, config: H3ImageConfig) -> None:
        self._config = config

    def checkpoint_path(self) -> str | None:
        return str(self.snapshot)

    # ------------------------------------------------------------ structure
    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The DiT's attention over the packed sequence (sharded by head under SP) and the
        token refiner's over the prompt; ``"sdpa"`` declares nothing."""
        if self.attention_backend != "flashinfer":
            return []
        cfg = self.config.transformer

        def spec(key, node, shard_over_sp, segments, head_groups=1):
            return RaggedAttentionSpec(
                resource_key=key, nodes={node},
                config=RaggedAttentionConfig(
                    num_qo_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_attention_heads,
                    head_dim=cfg.attention_head_dim, max_segments_per_request=segments, dtype=torch.bfloat16,
                    shard_over_sp=shard_over_sp, workspace_mb=128, head_groups=head_groups,
                ),
            )

        # the DiT's padding rows are a second segment of each request
        return [spec(DIT_ATTN, DIT_NODE, True, 2, self.ulysses_head_groups),
                spec(REFINER_ATTN, TEXT_ENCODER_NODE, False, 1)]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def text_node(*inputs: str) -> GraphNode:
            return GraphNode(
                name=TEXT_ENCODER_NODE, enable_async_scheduling=self.async_scheduling, input_names=list(inputs),
                outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS, persist=True)],
            )

        encode_references = Parallel([
            text_node(TEXT_INPUTS, VISION_PIXELS),
            GraphNode(
                name=VAE_ENCODER_NODE, enable_async_scheduling=self.async_scheduling, input_names=[REFERENCE_PIXELS],
                outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=COND_LATENTS, persist=True)],
            ),
        ])
        return {
            ENCODE_TEXT_WALK: text_node(TEXT_INPUTS),
            ENCODE_REFERENCES_WALK: encode_references,
            GENERATE_WALK: self._generate_walk(edit=False),
            GENERATE_EDIT_WALK: self._generate_walk(edit=True),
        }

    def _generate_walk(self, edit: bool) -> GraphSection:
        loop = Loop(
            name=DENOISE_LOOP,
            section=GraphNode(
                name=DIT_NODE, input_names=[TEXT_EMBEDS, LATENTS, *([COND_LATENTS] if edit else [])],
                outputs=[GraphEdge(next_node=DIT_NODE, name=LATENTS)],
                enable_async_scheduling=self.async_scheduling,
            ),
            max_iters=MAX_STEPS,
            outputs=[GraphEdge(next_node=VAE_DECODER_NODE, name=LATENTS)],
        )
        decoder = GraphNode(
            name=VAE_DECODER_NODE, enable_async_scheduling=self.async_scheduling, input_names=[LATENTS],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=IMAGE_OUTPUT, output_modality="image")],
        )
        return Sequential([loop, decoder])

    # ---------------------------------------------------------------- inputs
    def _ensure_tokenizer(self):
        # verbatim, no special tokens added: the reference's Qwen2TokenizerFast call
        if self.tokenizer is None:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(self.snapshot / "tokenizer" / "tokenizer.json"))
        return self.tokenizer

    @property
    def image_processor(self):
        """Qwen3-VL's image processor (``processor/``), as the reference uses it."""
        if self._image_processor is None:
            self._image_processor = import_image_processor().from_pretrained(str(self.snapshot / "processor"))
        return self._image_processor

    def warmup_preprocess(self) -> None:
        self._ensure_tokenizer()
        self.image_processor  # noqa: B018

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse to start where references could not be laid out, rather than 500 on the
        first edit."""
        if self.skip_weight_loading:
            return
        try:
            import_image_processor()
        except RuntimeError as exc:
            raise ValueError(f"{config_path}: {exc}") from exc

    def max_input_files(self) -> dict[str, int]:
        return {"image": MAX_REFERENCES, "audio": 0, "video": 0}

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """A reference as decoded RGB, uint8 ``[3, H, W]``, upright by its EXIF orientation:
        ``process_prompt`` resizes it with PIL, as the reference does."""
        check_image_size(filepath, formats=PILLOW_IMAGE_FORMATS)
        with decoding_upload(filepath, "the reference image"), \
                open_upright_image(filepath, formats=PILLOW_IMAGE_FORMATS) as image:
            rgb = image.convert("RGB")
        return TensorAndMetadata(torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).contiguous())

    def _prompt_ids(self, prompt: str) -> list[int]:
        """The prompt's token ids, cut to the reference's 4096 and refusing Qwen's vision
        tokens: reference features are spliced in at image pads."""
        tokenizer = self._ensure_tokenizer()
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids[:MAX_TEXT_TOKENS]
        reserved = {tokenizer.token_to_id(token) for token in VISION_TOKENS} - {None}
        if reserved.intersection(ids):
            raise ValueError(f"MiniMax-H3-Image prompts can't contain Qwen's vision tokens {list(VISION_TOKENS)}")
        return ids

    def tokenize(self, prompt: str) -> torch.Tensor:
        return torch.tensor(self._prompt_ids(prompt), dtype=torch.long)

    def resolve_request(self, model_kwargs: dict) -> dict:
        """The request's output size, canvas and generation area, validated."""
        height = _positive_int(model_kwargs, "height", None)
        width = _positive_int(model_kwargs, "width", None)
        if (height is None) != (width is None):
            raise ValueError("MiniMax-H3-Image height and width have to be given together, or neither")
        if height is None:
            height = width = DEFAULT_SIZE
        area = _positive_int(model_kwargs, "min_generation_area", DEFAULT_MIN_GENERATION_AREA)
        low, high = MIN_GENERATION_AREA_RANGE
        if not low <= area <= high:
            raise ValueError(f"MiniMax-H3-Image min_generation_area must be within [{low}, {high}], got {area}")
        canvas_h, canvas_w = canvas_size(height, width, area)
        return {"height": height, "width": width, "canvas_height": canvas_h, "canvas_width": canvas_w,
                "min_generation_area": area}

    def _check_rows(self, request: dict, text_rows: int, references: list[tuple[int, int]]) -> None:
        layout = H3ImageLayout(text_rows, request["canvas_height"] // VAE_RATIO, request["canvas_width"] // VAE_RATIO,
                               tuple((h // 2 * 2, w // 2 * 2) for h, w in references))
        if layout.total_rows > self.max_packed_rows:
            raise ValueError(
                f"MiniMax-H3-Image {request['width']}x{request['height']} (generated at {request['canvas_width']}x"
                f"{request['canvas_height']}) with a {text_rows}-token prompt and {len(references)} reference(s) "
                f"packs {layout.total_rows} rows, above this server's max_packed_rows of {self.max_packed_rows}"
            )

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        """Validate the request at the 400-producing seam; tokenize the prompt, and with
        references resize them for the VAE and the text encoder and lay out the prompt
        around their vision blocks (``MiniMaxH3ImageEditResizeStep``, ``...EditTextEncoderStep``)."""
        if any(m not in ("text", "image") for m in input_modalities or []):
            raise ValueError(f"MiniMax-H3-Image takes a text prompt and reference images; got {input_modalities}")
        if (output_modalities or ["image"]) != ["image"]:
            raise ValueError(f"MiniMax-H3-Image only generates images; got output modalities {output_modalities!r}")
        if prompt is None or not prompt.strip():
            raise ValueError("MiniMax-H3-Image requires a non-empty text prompt")
        images = list((tensors or {}).get(IMAGE_INPUTS) or [])
        if "image" in (input_modalities or []) and not images:
            raise ValueError("request declared an image input but no image arrived")
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3-Image takes at most {MAX_REFERENCES} reference images; got {len(images)}")
        request = self.resolve_request(kwargs)
        image_output_format(kwargs)  # fail a bad output knob now, not after the denoise
        prompt_ids = self._prompt_ids(prompt)
        if not images:
            self._check_rows(request, len(prompt_ids), [])
            return {TEXT_INPUTS: [torch.tensor(prompt_ids, dtype=torch.long)]}
        from PIL import Image

        pil = [Image.fromarray(img.permute(1, 2, 0).contiguous().cpu().numpy()) for img in images]
        for image in pil:
            if min(image.size) < CANVAS_MULTIPLE:
                raise ValueError(f"a reference image needs at least {CANVAS_MULTIPLE} pixels on each side, "
                                 f"got {image.size[0]}x{image.size[1]}")
        for_vae = [_resize(image, *REFERENCE_SHORT_EDGE) for image in pil]
        for_vision = [_resize(image, *VISION_SHORT_EDGE) for image in pil]
        # a reference keeps its own size, cropped at the top left to whole VAE cells
        pixels = [
            torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[
                :, : image.size[1] // VAE_RATIO * VAE_RATIO, : image.size[0] // VAE_RATIO * VAE_RATIO
            ].contiguous()
            for image in for_vae
        ]
        vision = self.image_processor(images=for_vision, return_tensors="pt")
        grids = [tuple(int(v) for v in grid) for grid in vision["image_grid_thw"]]
        ids = presentation(self._ensure_tokenizer(), "", self.vision_tokens(grids), label=PICTURE_LABEL).token_ids
        ids += prompt_ids
        self._check_rows(request, len(ids), [(p.shape[1] // VAE_RATIO, p.shape[2] // VAE_RATIO) for p in pixels])
        # one [grid_h, grid_w, patch_dim] view of the processor's rows per reference: the walk
        # plan reads each grid off its dims
        rows = vision["pixel_values"].to(torch.float32).split([t * h * w for t, h, w in grids])
        if any(t != 1 for t, _, _ in grids):
            raise ValueError(f"unexpected vision grids {grids} for still images")
        return {
            TEXT_INPUTS: [torch.tensor(ids, dtype=torch.long)],
            VISION_PIXELS: [r.view(h, w, -1) for r, (_, h, w) in zip(rows, grids, strict=True)],
            REFERENCE_PIXELS: pixels,
            # the decoded uploads are spent (no node reads them); the store would hold
            # them for the whole request
            IMAGE_INPUTS: [],
        }

    def vision_tokens(self, grids: list[tuple[int, int, int]]) -> list[int]:
        """Each reference's merged vision tokens: its 16-px patches merged 2x2."""
        return [t * h * w // 4 for t, h, w in grids]

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"height", "width", "min_generation_area", *OUTPUT_FORMAT_KWARGS})

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {DENOISE_LOOP: self.config.num_steps}

    # --------------------------------------------------------------- plan
    def walk_plan(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict,
    ) -> list[WalkStep]:
        """Encode the prompt (and any references), then denoise and decode; both steps carry
        the request's resolved facts, which every node reads its layout from."""
        if not input_signals.get(TEXT_INPUTS):
            raise ValueError("MiniMax-H3-Image requires a text prompt (text_inputs)")
        request = self.resolve_request(model_kwargs or {})
        text_len = int(input_signals[TEXT_INPUTS][0].dims[0])
        references = [(int(info.dims[-2]) // VAE_RATIO, int(info.dims[-1]) // VAE_RATIO)
                      for info in input_signals.get(REFERENCE_PIXELS) or []]
        request.update(text_len=text_len, latent_height=request["canvas_height"] // VAE_RATIO,
                       latent_width=request["canvas_width"] // VAE_RATIO,
                       reference_latent_grids=[list(grid) for grid in references])
        if references:
            grids = [(1, int(info.dims[0]), int(info.dims[1])) for info in input_signals[VISION_PIXELS]]
            spans = vision_spans(self._ensure_tokenizer(), self.vision_tokens(grids), label=PICTURE_LABEL)
            request.update(keyframe_grids=[list(grid) for grid in grids],
                           text_image_spans=[list(span) for span in spans])
        layout = layout_from_metadata(request)
        logger.info(
            "MiniMax-H3-Image request: %dx%d (canvas %dx%d) prompt=%d tokens refs=%d rows=%d seed=%s",
            request["width"], request["height"], request["canvas_width"], request["canvas_height"], text_len,
            len(references), layout.total_rows, (model_kwargs or {}).get("seed", "auto"),
        )
        first = ENCODE_REFERENCES_WALK if references else ENCODE_TEXT_WALK
        final = GENERATE_EDIT_WALK if references else GENERATE_WALK
        return [WalkStep(first, step_metadata=request), WalkStep(final, step_metadata=request)]

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """uint8 ``[3, H, W]`` on the canvas -> the requested size (LANCZOS on the 8-bit image,
        as the reference resizes) -> PNG (or the request's ``output_format``)."""
        if modality != "image":
            raise ValueError(f"unsupported output modality for MiniMax-H3-Image: {modality!r}")
        request = self.resolve_request(request_kwargs or {})
        image = output[0] if output.ndim == 4 else output
        if tuple(image.shape[-2:]) != (request["height"], request["width"]):
            from PIL import Image

            pil = Image.fromarray(image.permute(1, 2, 0).contiguous().cpu().numpy())
            pil = pil.resize((request["width"], request["height"]), Image.Resampling.LANCZOS)
            image = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1)
        return encode_image(image, request_kwargs)

    # ------------------------------------------------------------- loading
    def get_autocast_dtype(self):
        # the checkpoint's dtypes: bf16, with the reference's fp32 modules and step math
        return None

    def get_default_sharding_config(self) -> ShardingConfig:
        # The dit shards its rows by rank (Ulysses), the decoder its tiles. No tensor parallelism.
        return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={},
                              sp_enabled_nodes={DIT_NODE, VAE_DECODER_NODE})

    def get_submodule(self, node_name: str, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None):
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, tp_group, sp_group)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("loaded MiniMax-H3-Image submodule for node %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device, tp_group=None, sp_group=None) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        from mstar.model.minimax_h3_image import weight_loader

        ragged = self.attention_backend == "flashinfer"
        snapshot, config = self.snapshot, self.config
        if node_name == TEXT_ENCODER_NODE:
            return H3TextEncoderSubmodule(
                weight_loader.build_text_encoder(config, snapshot, device),
                weight_loader.build_text_conditioner(config, snapshot, device),
                attn_resource_key=REFINER_ATTN if ragged else None,
                vision=weight_loader.build_vision_tower(snapshot, device), config=config,
            )
        if node_name == VAE_ENCODER_NODE:
            return H3ImageEncoderSubmodule(weight_loader.build_image_encoder(config, snapshot, device))
        if node_name == DIT_NODE:
            return H3ImageDenoiseSubmodule(
                weight_loader.build_transformer(config, snapshot, device, comm_group=tp_group, sp_group=sp_group,
                                                head_groups=self.ulysses_head_groups,
                                                group_streams=self.ulysses_group_streams),
                config, loop_name=DENOISE_LOOP, attn_resource_key=DIT_ATTN if ragged else None,
                max_batch_size=1, compile_blocks=self.compile, row_bucket=self.row_bucket,
            )
        if node_name == VAE_DECODER_NODE:
            return H3ImageDecoderSubmodule(weight_loader.build_image_decoder(config, snapshot, device), config,
                                           tile_batch=self.vae_tile_batch, compile_blocks=self.vae_compile,
                                           sp_group=sp_group)
        logger.warning("MiniMax-H3-Image has no submodule for node %r; running it dummy", node_name)
        return None


def _resize(image, min_short_edge: int, max_short_edge: int):
    """A reference for one consumer: kept when its short edge is within the bounds, else
    LANCZOS-resized so it lands on the nearer one (``VaeImageProcessor.resize``)."""
    from PIL import Image

    size = bound_short_edge(*image.size, min_short_edge, max_short_edge)
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)
