"""Flux2KleinModel: FLUX.2 [klein] text-to-image and multi-reference image editing.

Architecture (4 nodes; the dit declares the array its joint attention reads):

    text_encoder  Qwen3 (4B / 8B) hidden-state taps -> [1, 512, joint_dim] prompt embeddings
    vae_encoder   FLUX.2 VAE encode of the reference images -> packed normalized tokens (edits)
    dit           rectified-flow transformer, one Euler step per Loop iteration (no CFG: distilled)
    vae_decoder   FLUX.2 VAE decode -> uint8 image

Graph walks:

    encode_text   text_encoder; persists text_embeds
    encode_image  vae_encoder;  persists ref_latents                      (edit requests only)
    image_gen     Loop("denoise_loop", dit{text_embeds, latents}) -> vae_decoder -> EMIT image
    image_edit    same, with ref_latents as an extra dit input

The walk plan (``walk_plan``) is fixed at admission from the input modalities:

    T2I:  encode_text -> image_gen
    edit: encode_text -> encode_image -> image_edit

The denoise loop is dynamic: ``Loop.max_iters`` is the config ceiling and the
request's ``num_inference_steps`` is the loop's cap (``get_loop_caps``).

Per-request knobs (``model_kwargs``): ``width``, ``height`` (multiples of 16),
``num_inference_steps`` (default 4), ``seed`` (honoured by the conductor). The
distilled checkpoints take no guidance; ``guidance_scale`` is accepted and ignored
with a log line, as in the reference pipeline.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Hashable

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.resources.lora.checkpoint import LoraFile
from mstar.engine.state import Array, Attend, Block, Read
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import Model, TensorAndMetadata, WalkStep
from mstar.model.components.diffusion.attention import dit_entry, refuse_attention_backend_kwarg
from mstar.model.components.diffusion.image_io import OUTPUT_FORMAT_KWARGS, encode_image, image_output_format
from mstar.model.components.diffusion.qwen.submodule import (
    TEXT_EMBEDS,
    TEXT_INPUTS,
    TEXT_LEN,
    QwenTextEncoderSubmodule,
)
from mstar.model.components.quant_linear import (
    check_linear_dtype,
    check_linear_dtype_steps,
    check_linear_keep,
    quantize_linears,
    select_linears,
)
from mstar.model.flux2_klein.components.transformer import QUANTIZABLE_LINEARS
from mstar.model.flux2_klein.config import (
    DENOISE_LOOP,
    DIT_ATTN,
    FLUX2_KLEIN_4B,
    JOINT_TOKENS,
    Flux2KleinConfig,
)
from mstar.model.flux2_klein.submodules import (
    IMAGE_INPUTS,
    IMAGE_OUTPUT,
    LATENTS,
    REF_LATENTS,
    KleinDenoiseSubmodule,
    KleinShape,
    KleinVaeDecoderSubmodule,
    KleinVaeEncoderSubmodule,
)
from mstar.model.media_limits import PILLOW_IMAGE_FORMATS, check_image_size, decoding_upload, open_upright_image
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

ENCODE_TEXT_WALK = "encode_text"
ENCODE_IMAGE_WALK = "encode_image"
IMAGE_GEN_WALK = "image_gen"
IMAGE_EDIT_WALK = "image_edit"



class Flux2KleinModel(Model):
    """FLUX.2 [klein] (4B by default; the 9B checkpoint loads through the same class)."""

    def __init__(
        self,
        model_path_hf: str = FLUX2_KLEIN_4B,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        compile: bool = True,
        compile_eager_rounding: bool = True,
        compile_exact_ops: bool | list[str] = False,
        accelerator_graph: bool = True,
        capture_sizes: list[list[int]] | None = None,
        capture_edit_sizes: list[list[int]] | None = None,
        capture_batch_sizes: list[int] | None = None,
        max_batch_size: int = 8,
        max_image_area: int = 2048 * 2048,
        vae_compile: bool = False,
        async_scheduling: bool = True,
        lora: list | None = None,
        linear_dtype: str = "bf16",
        linear_keep_bf16: list[str] | None = None,
        linear_dtype_steps: int | list[str] | None = None,
        **kwargs,
    ):
        refuse_attention_backend_kwarg(kwargs)
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # Dummy mode: get_submodule returns None for every node, so the engine runs
        # the graph without weights, GPU or network (CPU tests).
        self.skip_weight_loading = skip_weight_loading
        self.compile_transformer = bool(compile)
        self.compile_eager_rounding = bool(compile_eager_rounding)
        # True: every norm / activation class stays eager inside the compiled forward; a list picks classes
        self.compile_exact_ops = compile_exact_ops if isinstance(compile_exact_ops, list) else bool(compile_exact_ops)
        self.accelerator_graph = bool(accelerator_graph)
        # Default capture: the model's native 1024x1024 text-to-image shape.
        self.capture_sizes = [tuple(int(v) for v in s) for s in (capture_sizes or [[1024, 1024]])]
        # Edit buckets: one reference image of the output size (an edit's default output size is its
        # reference's), captured for these output sizes; None = the capture_sizes, [] = none.
        self.capture_edit_sizes = (
            list(self.capture_sizes) if capture_edit_sizes is None
            else [tuple(int(v) for v in s) for s in capture_edit_sizes]
        )
        self.capture_batch_sizes = [int(b) for b in (capture_batch_sizes or [1, 2, 4, 8])]
        self.max_batch_size = int(max_batch_size)
        # largest output (pixels) a request may ask for: an unbounded size lets one request occupy the worker for
        # minutes (8192^2 = 262k tokens of quadratic attention); requests above it are rejected before scheduling
        self.max_image_area = int(max_image_area)
        self.vae_compile = bool(vae_compile)
        # Speculative (asynchronous) scheduling of the image nodes: the worker assembles each batch's next denoise
        # step while the current one runs (the loop's persisted inputs count as ready, the step runs under its own
        # loop counter) and merges requests that became ready meanwhile, so requests at different steps share a
        # forward. On by default since 2026-09-23: same-GPU rows put klein-4B at 0.365-0.372 s vs 0.377-0.390 s
        # lockstep at B=1 and +3% images/s at 4 and 16 concurrent requests (9B +2%), with the served images
        # unchanged (>= 53 dB from the eager path on all 100 protocol prompts). The loop's cap keeps speculation from
        # running past a request's last step. `false` restores lockstep
        # (wait for the step, then batch everything ready), which is within a few percent.
        self.async_scheduling = bool(async_scheduling)
        # LoRA adapters folded into the transformer at load time (static merge).
        self.loras = [LoraFile.parse(item) for item in (lora or [])]
        # The transformer's block linears in a low-precision format (mxfp8, nvfp4), with this deployment's extra
        # bf16 patterns on top of the model's declaration, and optionally some denoise steps in bf16 on a twin.
        # Trades image fidelity for speed; see docs/models.rst.
        self.linear_dtype = check_linear_dtype(linear_dtype)
        self.linear_keep_bf16 = check_linear_keep(linear_keep_bf16)
        self.step_dtypes = check_linear_dtype_steps(linear_dtype_steps, self.linear_dtype)

        self._snapshot = None
        self._config: Flux2KleinConfig | None = None
        self.tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._vae: torch.nn.Module | None = None

    # ------------------------------------------------------------------ config
    @property
    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = resolve_snapshot_dir(self.model_path_hf, self.cache_dir)
        return self._snapshot

    @property
    def config(self) -> Flux2KleinConfig:
        """Read from the checkpoint on first use; ``set_config`` pins one for tests."""
        if self._config is None:
            self._config = Flux2KleinConfig.from_snapshot(self.snapshot)
        return self._config

    def set_config(self, config: Flux2KleinConfig) -> None:
        self._config = config

    # ------------------------------------------------------------ structure
    def declare_state(self) -> list[Array]:
        """The dit's joint attention over each request's tokens; nothing else here keeps state."""
        t = self.config.transformer
        return [Array(
            JOINT_TOKENS, nodes={"dit"}, layers=frozenset(range(t.num_layers + t.num_single_layers)), max_len=0,
            entry=dit_entry(t.num_attention_heads, t.attention_head_dim),
            reads={DIT_ATTN: Read(mask=Block(), op=Attend(q_heads=t.num_attention_heads), packed=False)},
        )]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_text = GraphNode(
            name="text_encoder", enable_async_scheduling=self.async_scheduling,
            input_names=[TEXT_INPUTS],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS, persist=True)],
        )
        encode_image = GraphNode(
            name="vae_encoder", enable_async_scheduling=self.async_scheduling,
            input_names=[IMAGE_INPUTS],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=REF_LATENTS, persist=True)],
        )
        return {
            ENCODE_TEXT_WALK: encode_text,
            ENCODE_IMAGE_WALK: encode_image,
            IMAGE_GEN_WALK: self._denoise_walk(edit=False),
            IMAGE_EDIT_WALK: self._denoise_walk(edit=True),
        }

    def _denoise_walk(self, edit: bool) -> GraphSection:
        dit_inputs = [TEXT_EMBEDS, LATENTS] + ([REF_LATENTS] if edit else [])
        loop = Loop(
            name=DENOISE_LOOP,
            section=GraphNode(
                name="dit",
                input_names=dit_inputs,
                # latents is the only loop-carried edge; the step index is the loop counter
                outputs=[GraphEdge(next_node="dit", name=LATENTS)],
                # speculative by default (see the async_scheduling knob); `false` = lockstep
                enable_async_scheduling=self.async_scheduling,
            ),
            max_iters=self.config.max_denoise_steps,
            outputs=[GraphEdge(next_node="vae_decoder", name=LATENTS)],
        )
        decoder = GraphNode(
            name="vae_decoder", enable_async_scheduling=self.async_scheduling,
            input_names=[LATENTS],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=IMAGE_OUTPUT, output_modality="image")],
        )
        return Sequential([loop, decoder])

    # ---------------------------------------------------------------- inputs
    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot / "tokenizer"))
        return self.tokenizer

    def render_prompt(self, prompt: str) -> str:
        """The pipeline's chat-templated prompt (user turn, generation prompt, thinking off)."""
        return self._ensure_tokenizer().apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Unpadded token ids ``[L]`` (``L <= max_sequence_length``), truncated like the
        pipeline; the text encoder node pads them to ``max_sequence_length`` as it does."""
        enc = self._ensure_tokenizer()(
            self.render_prompt(prompt), return_tensors="pt", truncation=True,
            max_length=self.config.text_encoder.max_sequence_length,
        )
        return enc["input_ids"][0].to(torch.long)

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """Reference image with the pipeline's preprocessing: validity checks, LANCZOS
        shrink to at most ``ref_image_max_area`` pixels, floor of both sides to the 16-px
        token grid, center crop, RGB; returned as uint8 ``[3, H, W]``, upright by its EXIF
        orientation as diffusers' ``load_image`` reads it."""
        from PIL import Image

        check_image_size(filepath, formats=PILLOW_IMAGE_FORMATS)
        with (
            decoding_upload(filepath, "the reference image"),
            open_upright_image(filepath, formats=PILLOW_IMAGE_FORMATS) as image,
        ):
            width, height = image.size
            if width < 64 or height < 64:
                raise ValueError(f"reference image too small: {width}x{height}; both sides must be >= 64 px")
            if max(width / height, height / width) > 8:
                raise ValueError(f"reference image aspect ratio too extreme: {width}x{height} (max 8:1)")
            if width * height > self.config.ref_image_max_area:
                scale = math.sqrt(self.config.ref_image_max_area / (width * height))
                width, height = int(width * scale), int(height * scale)
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            align = self.config.spatial_alignment
            crop_w, crop_h = (width // align) * align, (height // align) * align
            left, top = (width - crop_w) // 2, (height - crop_h) // 2
            image = image.crop((left, top, left + crop_w, top + crop_h))
            # RGB last, as the reference image processor does: resampling an RGBA or palette
            # image and then dropping the alpha differs from resampling its RGB conversion
            image = image.convert("RGB")
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
        return TensorAndMetadata(tensor.to(device))

    def _resolve_size(self, model_kwargs: dict, ref_dims: list[tuple[int, int]]) -> tuple[int, int]:
        """(height, width) of the output: request kwargs, else the first reference image's
        size (the pipeline's default for edits), else the config default."""
        default_h, default_w = self.config.default_height, self.config.default_width
        if ref_dims:
            default_h, default_w = ref_dims[0]
        height = model_kwargs.get("height", default_h)
        width = model_kwargs.get("width", default_w)
        align = self.config.spatial_alignment
        for name, value in (("height", height), ("width", width)):
            if value <= 0 or value % align:
                raise ValueError(
                    f"FLUX.2 klein {name}={value} must be a positive multiple of {align} "
                    f"(VAE stride {self.config.vae.spatial_compression} x 2x2 latent patch)"
                )
        if height * width > self.max_image_area:
            raise ValueError(
                f"FLUX.2 klein {width}x{height} exceeds this server's max_image_area of {self.max_image_area} pixels "
                f"({int(self.max_image_area ** 0.5)}^2)"
            )
        return height, width

    def _resolve_steps(self, model_kwargs: dict) -> int:
        return model_kwargs.get("num_inference_steps", self.config.default_num_inference_steps)

    def request_kwargs(self) -> frozenset[str]:
        # guidance_scale is read only to warn that the distilled checkpoint ignores it
        return frozenset({"height", "width", "num_inference_steps", "guidance_scale", *OUTPUT_FORMAT_KWARGS})

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the loop ends on the request's own step count, and is never speculated past it;
        # past the loop's max_iters the conductor refuses the request
        return {DENOISE_LOOP: self._resolve_steps(model_kwargs)}

    def max_input_files(self) -> dict[str, int]:
        # the data worker refuses more reference images than this before decoding any
        return {"image": self.config.max_ref_images}

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        """Validate the request at the 400-producing seam and tokenize the prompt."""
        if (output_modalities or ["image"])[0] != "image":
            raise ValueError(f"FLUX.2 klein only generates images; got output modality {output_modalities!r}")
        if prompt is None or not prompt.strip():
            raise ValueError("FLUX.2 klein requires a non-empty text prompt")
        images = (tensors or {}).get(IMAGE_INPUTS) or []
        if "image" in input_modalities and not images:
            raise ValueError("request declared an image input but no image arrived; editing needs the reference image")
        ref_dims = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
        self._resolve_size(kwargs, ref_dims)
        self._resolve_steps(kwargs)
        image_output_format(kwargs)  # fail a bad output knob now, not after the denoise
        # Ungated on is_distilled: a non-distilled checkpoint is the case that most
        # needed telling, and it was the one case that stayed silent. Only distilled
        # checkpoints load at all now (Flux2KleinConfig.from_snapshot), so this fires
        # whenever a caller's guidance_scale is about to be dropped.
        if kwargs.get("guidance_scale", 1.0) != 1.0:
            logger.warning("guidance_scale is ignored: FLUX.2 klein is step-distilled and runs without CFG")
        return {TEXT_INPUTS: [self.tokenize(prompt)]}

    # ------------------------------------------------------------ walk plan
    def walk_plan(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict,
    ) -> list[WalkStep]:
        if not input_signals.get(TEXT_INPUTS):
            raise ValueError("FLUX.2 klein requires a text prompt (text_inputs)")
        ref_dims = [(int(info.dims[-2]), int(info.dims[-1])) for info in input_signals.get(IMAGE_INPUTS) or []]
        height, width = self._resolve_size(model_kwargs, ref_dims)
        request = {
            "height": height,
            "width": width,
            "num_inference_steps": self._resolve_steps(model_kwargs),
            "ref_grids": [list(self.config.latent_grid(h, w)) for h, w in ref_dims],
            TEXT_LEN: int(input_signals[TEXT_INPUTS][0].dims[0]),
        }
        logger.info(
            "FLUX.2 klein request: %dx%d steps=%d refs=%s seed=%s", width, height, request["num_inference_steps"],
            request["ref_grids"] or "-", model_kwargs.get("seed", "auto"),
        )
        walks = [ENCODE_TEXT_WALK] + ([ENCODE_IMAGE_WALK, IMAGE_EDIT_WALK] if ref_dims else [IMAGE_GEN_WALK])
        return [WalkStep(walk, step_metadata=request) for walk in walks]

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality != "image":
            raise ValueError(f"unsupported output modality for FLUX.2 klein: {modality!r}")
        # output_format / output_compression / png_compress_level from the request (OpenAI knobs)
        return encode_image(output, request_kwargs)

    # ------------------------------------------------------------- loading
    def get_autocast_dtype(self):
        # None: the engine neither autocasts nor casts modules; numerics follow the
        # checkpoint dtypes, which is what the parity tests pin.
        return None

    def capture_buckets(self) -> list[tuple[str, Hashable]]:
        if not self.accelerator_graph:
            return []
        text_len = self.config.text_encoder.max_sequence_length
        shapes: list[tuple[str, Hashable]] = [
            (IMAGE_GEN_WALK, KleinShape(grid=self.config.latent_grid(h, w), text_len=text_len))
            for h, w in self.capture_sizes
        ]
        for h, w in self.capture_edit_sizes:
            grid = self.config.latent_grid(h, w)
            shapes.append((IMAGE_EDIT_WALK, KleinShape(grid=grid, text_len=text_len, ref_grids=(grid,))))
        return shapes

    def warmup_sizes(self) -> list[tuple[int, int]]:
        """Output sizes whose VAE decode is warmed at load: the generation and edit capture sizes."""
        return list(dict.fromkeys([*self.capture_sizes, *self.capture_edit_sizes]))

    def get_submodule(self, node_name: str, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None):
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("loaded FLUX.2 klein submodule for node %s", node_name)
        return submodule

    def _vae_module(self, device):
        if self._vae is None:
            from mstar.model.flux2_klein.weight_loader import build_vae

            self._vae = build_vae(self.config, self.snapshot, device)
        return self._vae

    def _create_submodule(self, node_name: str, device) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        from mstar.model.flux2_klein.weight_loader import build_text_encoder, build_transformer

        if node_name == "text_encoder":
            encoder = build_text_encoder(self.config.text_encoder, self.snapshot, device)
            return QwenTextEncoderSubmodule(
                encoder, self.config.text_encoder, max_batch_size=self.max_batch_size,
                capture_walk=ENCODE_TEXT_WALK if self.accelerator_graph else None,
            )
        if node_name == "vae_encoder":
            return KleinVaeEncoderSubmodule(self._vae_module(device), self.config)
        if node_name == "vae_decoder":
            return KleinVaeDecoderSubmodule(
                self._vae_module(device), self.config, max_batch_size=self.max_batch_size,
                compile_decode=self.vae_compile,
                warmup_grids=[self.config.latent_grid(h, w) for h, w in self.warmup_sizes()],
            )
        if node_name == "dit":
            transformer = build_transformer(self.config, self.snapshot, device)
            if self.loras:
                from mstar.model.flux2_klein.weight_loader import apply_transformer_loras

                apply_transformer_loras(transformer, self.loras)
            # after the adapters: those fold into the bf16 weights this quantizes
            selection = select_linears(
                transformer, self.linear_dtype, QUANTIZABLE_LINEARS, self.linear_keep_bf16, "FLUX.2 klein transformer",
            )
            # a schedule's bf16 steps run a copy taken before quantizing; nothing quantized, every step is bf16
            step_dtypes = self.step_dtypes if selection.names else None
            bf16_transformer = copy.deepcopy(transformer) if step_dtypes is not None else None
            quantize_linears(transformer, selection)
            return KleinDenoiseSubmodule(
                transformer, self.config, loop_name=DENOISE_LOOP,
                compile_transformer=self.compile_transformer,
                compile_eager_rounding=self.compile_eager_rounding, compile_exact_ops=self.compile_exact_ops,
                max_batch_size=self.max_batch_size,
                capture_buckets=self.capture_buckets(), capture_batch_sizes=self.capture_batch_sizes,
                step_dtypes=step_dtypes, bf16_transformer=bf16_transformer,
            )
        logger.warning("FLUX.2 klein has no submodule for node %r; running it dummy", node_name)
        return None
