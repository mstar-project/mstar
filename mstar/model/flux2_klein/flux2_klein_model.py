"""Flux2KleinModel: FLUX.2 [klein] text-to-image and multi-reference image editing.

Architecture (4 nodes; the dit optionally owns one ragged attention resource):

    text_encoder  Qwen3 (4B / 8B) hidden-state taps -> [1, 512, joint_dim] prompt embeddings
    vae_encoder   FLUX.2 VAE encode of the reference images -> packed normalized tokens (edits)
    dit           rectified-flow transformer, one Euler step per Loop iteration (no CFG: distilled)
    vae_decoder   FLUX.2 VAE decode -> uint8 image

Graph walks:

    encode_text   text_encoder; persists text_embeds
    encode_image  vae_encoder;  persists ref_latents                      (edit requests only)
    image_gen     Loop("denoise_loop", dit{text_embeds, latents}) -> vae_decoder -> EMIT image
    image_edit    same, with ref_latents as an extra dit input

The walk sequence is fixed at admission from the input modalities and stepped
through ``metadata.kwargs["walk_step"]``:

    T2I:  encode_text -> image_gen
    edit: encode_text -> encode_image -> image_edit

The denoise loop is dynamic: ``Loop.max_iters`` is the config ceiling and the
request's ``num_inference_steps`` stops it through the dit's ``check_stop``.

Per-request knobs (``model_kwargs``): ``width``, ``height`` (multiples of 16),
``num_inference_steps`` (default 4), ``seed`` (honoured by the conductor). The
distilled checkpoints take no guidance; ``guidance_scale`` is accepted and ignored
with a log line, as in the reference pipeline.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import NodeResourceSpec, RaggedAttentionConfig, RaggedAttentionSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.components.diffusion.image_io import encode_image
from mstar.model.components.diffusion.lora import LoraSpec
from mstar.model.flux2_klein.config import (
    DENOISE_LOOP,
    DIT_ATTN,
    FLUX2_KLEIN_4B,
    Flux2KleinConfig,
    resolve_snapshot_dir,
)
from mstar.model.flux2_klein.submodules import (
    IMAGE_INPUTS,
    IMAGE_OUTPUT,
    LATENTS,
    REF_LATENTS,
    TEXT_EMBEDS,
    TEXT_INPUTS,
    TEXT_MASK,
    KleinDenoiseSubmodule,
    KleinShape,
    KleinTextEncoderSubmodule,
    KleinVaeDecoderSubmodule,
    KleinVaeEncoderSubmodule,
)
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

ENCODE_TEXT_WALK = "encode_text"
ENCODE_IMAGE_WALK = "encode_image"
IMAGE_GEN_WALK = "image_gen"
IMAGE_EDIT_WALK = "image_edit"

ATTENTION_BACKENDS = ("flashinfer", "sdpa")


class Flux2KleinModel(Model):
    """FLUX.2 [klein] (4B by default; the 9B checkpoint loads through the same class)."""

    def __init__(
        self,
        model_path_hf: str = FLUX2_KLEIN_4B,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        attention_backend: str = "flashinfer",
        compile: bool = True,
        cuda_graph: bool = True,
        capture_sizes: list[list[int]] | None = None,
        capture_batch_sizes: list[int] | None = None,
        max_batch_size: int = 8,
        vae_compile: bool = False,
        lora: list | None = None,
        **kwargs,
    ):
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {attention_backend!r}")
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # Dummy mode: get_submodule returns None for every node, so the engine runs
        # the graph without weights, GPU or network (CPU tests).
        self.skip_weight_loading = skip_weight_loading
        self.attention_backend = attention_backend
        self.compile_transformer = bool(compile)
        self.cuda_graph = bool(cuda_graph)
        # Default capture: the model's native 1024x1024 text-to-image shape.
        self.capture_sizes = [tuple(int(v) for v in s) for s in (capture_sizes or [[1024, 1024]])]
        self.capture_batch_sizes = [int(b) for b in (capture_batch_sizes or [1, 2, 4, 8])]
        self.max_batch_size = int(max_batch_size)
        self.vae_compile = bool(vae_compile)
        # LoRA adapters folded into the transformer at load time (static merge).
        self.loras = [LoraSpec.parse(item) for item in (lora or [])]

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
    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The dit's joint attention through the engine's ragged (cacheless) resource,
        when the FlashInfer backend is selected; nothing else here keeps state."""
        if self.attention_backend != "flashinfer":
            return []
        tcfg = self.config.transformer
        return [
            RaggedAttentionSpec(
                resource_key=DIT_ATTN, nodes={"dit"},
                config=RaggedAttentionConfig(
                    num_qo_heads=tcfg.num_attention_heads,
                    num_kv_heads=tcfg.num_attention_heads,
                    head_dim=tcfg.attention_head_dim,
                    # one request == one bidirectional segment over [txt | img | ref]
                    max_segments_per_request=1,
                    # the DiT's activation dtype: there is no KV cache to inherit one from
                    dtype=torch.bfloat16,
                ),
            )
        ]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_text = GraphNode(
            name="text_encoder", enable_async_scheduling=False,
            input_names=[TEXT_INPUTS, TEXT_MASK],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS, persist=True)],
        )
        encode_image = GraphNode(
            name="vae_encoder", enable_async_scheduling=False,
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
                # Lockstep, not speculative: the worker otherwise launches each request's next
                # step alone while the current one runs (38 of 43 steps unbatched at 8 concurrent
                # requests), so concurrent requests never share a batch. Waiting for the step to
                # finish costs ~2 ms of launch overlap per step and batches everything ready.
                enable_async_scheduling=False,
            ),
            max_iters=self.config.max_denoise_steps,
            outputs=[GraphEdge(next_node="vae_decoder", name=LATENTS)],
        )
        decoder = GraphNode(
            name="vae_decoder", enable_async_scheduling=False,
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

    def tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(input_ids, attention_mask)``, each ``[max_sequence_length]`` int64, right padded."""
        enc = self._ensure_tokenizer()(
            self.render_prompt(prompt), return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.config.text_encoder.max_sequence_length,
        )
        return enc["input_ids"][0].to(torch.long), enc["attention_mask"][0].to(torch.long)

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """Reference image with the pipeline's preprocessing: validity checks, LANCZOS
        shrink to at most ``ref_image_max_area`` pixels, floor of both sides to the 16-px
        token grid, center crop, RGB; returned as float ``[3, H, W]`` in ``[0, 1]``."""
        from PIL import Image

        with Image.open(filepath) as raw:
            raw.load()
            image = raw
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
        return TensorAndMetadata(tensor.to(torch.float32).div_(255.0).to(device))

    def _resolve_size(self, model_kwargs: dict, ref_dims: list[tuple[int, int]]) -> tuple[int, int]:
        """(height, width) of the output: request kwargs, else the first reference image's
        size (the pipeline's default for edits), else the config default."""
        default_h, default_w = self.config.default_height, self.config.default_width
        if ref_dims:
            default_h, default_w = ref_dims[0]
        height = int(model_kwargs.get("height", default_h))
        width = int(model_kwargs.get("width", default_w))
        align = self.config.spatial_alignment
        for name, value in (("height", height), ("width", width)):
            if value <= 0 or value % align:
                raise ValueError(
                    f"FLUX.2 klein {name}={value} must be a positive multiple of {align} "
                    f"(VAE stride {self.config.vae.spatial_compression} x 2x2 latent patch)"
                )
        return height, width

    def _resolve_steps(self, model_kwargs: dict) -> int:
        steps = int(model_kwargs.get("num_inference_steps", self.config.default_num_inference_steps))
        if steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
        if steps > self.config.max_denoise_steps:
            logger.info("clamping num_inference_steps %d to the loop ceiling %d", steps, self.config.max_denoise_steps)
            steps = self.config.max_denoise_steps
        return steps

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        """Validate the request at the 400-producing seam and tokenize the prompt."""
        if (output_modalities or ["image"])[0] != "image":
            raise ValueError(f"FLUX.2 klein only generates images; got output modality {output_modalities!r}")
        if prompt is None:
            raise ValueError("FLUX.2 klein requires a text prompt")
        images = (tensors or {}).get(IMAGE_INPUTS) or []
        if "image" in input_modalities and not images:
            raise ValueError("request declared an image input but no image arrived; editing needs the reference image")
        if len(images) > self.config.max_ref_images:
            raise ValueError(f"at most {self.config.max_ref_images} reference images are supported, got {len(images)}")
        ref_dims = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
        self._resolve_size(kwargs, ref_dims)
        self._resolve_steps(kwargs)
        if float(kwargs.get("guidance_scale", 1.0)) != 1.0 and self.config.is_distilled:
            logger.info("guidance_scale is ignored: FLUX.2 klein is step-distilled and runs without CFG")
        ids, mask = self.tokenize(prompt)
        return {TEXT_INPUTS: [ids], TEXT_MASK: [mask]}

    # --------------------------------------------------------- state machine
    def _step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        kw = metadata.kwargs
        return {
            "is_prefill": metadata.is_prefill,
            "height": kw["height"], "width": kw["width"],
            "num_inference_steps": kw["num_inference_steps"],
            "ref_grids": kw["ref_grids"],
        }

    def get_initial_forward_pass_args(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        model_kwargs = model_kwargs or {}
        if not input_signals.get(TEXT_INPUTS):
            raise ValueError("FLUX.2 klein requires a text prompt (text_inputs)")
        ref_infos = input_signals.get(IMAGE_INPUTS) or []
        ref_dims = [(int(info.dims[-2]), int(info.dims[-1])) for info in ref_infos]
        height, width = self._resolve_size(model_kwargs, ref_dims)
        edit = bool(ref_infos)
        schedule = [ENCODE_TEXT_WALK] + ([ENCODE_IMAGE_WALK, IMAGE_EDIT_WALK] if edit else [IMAGE_GEN_WALK])
        kwargs = {
            "walk_schedule": schedule,
            "walk_step": 0,
            "height": height,
            "width": width,
            "num_inference_steps": self._resolve_steps(model_kwargs),
            "ref_grids": [list(self.config.latent_grid(h, w)) for h, w in ref_dims],
        }
        logger.info(
            "FLUX.2 klein request: %dx%d steps=%d refs=%s seed=%s", width, height, kwargs["num_inference_steps"],
            kwargs["ref_grids"] or "-", model_kwargs.get("seed", "auto"),
        )
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities, output_modalities=output_modalities,
            graph_walk=schedule[0], is_prefill=True, kwargs=kwargs,
        )
        inputs = []
        for name in (TEXT_INPUTS, TEXT_MASK):
            edge = GraphEdge(next_node="text_encoder", name=name)
            edge.tensor_info = input_signals[name]
            inputs.append(edge)
        return ForwardPassArgs(
            full_metadata=metadata, inputs=inputs,
            unpersist_tensors=sum([e.tensor_info for e in inputs], start=[]),
            step_metadata=self._step_metadata(metadata),
        )

    def get_partition_forward_pass_args(
        self, partition_name: str, partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        inputs: list[GraphEdge] = []
        request_done = step >= len(schedule)
        if not request_done:
            metadata.kwargs["walk_step"] = step
            walk = schedule[step]
            metadata.graph_walk = walk
            metadata.is_prefill = walk == ENCODE_IMAGE_WALK
            if walk == ENCODE_IMAGE_WALK:
                edge = GraphEdge(next_node="vae_encoder", name=IMAGE_INPUTS)
                edge.tensor_info = persist_signals.get(IMAGE_INPUTS, [])
                inputs.append(edge)
            else:
                persisted = [TEXT_EMBEDS] + ([REF_LATENTS] if walk == IMAGE_EDIT_WALK else [])
                for name in persisted:
                    edge = GraphEdge(next_node="dit", name=name)
                    edge.tensor_info = persist_signals.get(name, [])
                    inputs.append(edge)
                # the loop-back edge arrives empty; the dit seeds it at iteration 0
                inputs.append(GraphEdge(next_node="dit", name=LATENTS))
        return ForwardPassArgs(
            full_metadata=metadata, inputs=inputs,
            unpersist_tensors=sum([e.tensor_info for e in inputs], start=[]),
            step_metadata=self._step_metadata(metadata), request_done=request_done,
        )

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

    def capture_shapes(self) -> list[tuple[str, Hashable]]:
        if not self.cuda_graph:
            return []
        text_len = self.config.text_encoder.max_sequence_length
        return [
            (IMAGE_GEN_WALK, KleinShape(grid=self.config.latent_grid(h, w), text_len=text_len))
            for h, w in self.capture_sizes
        ]

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
            return KleinTextEncoderSubmodule(encoder, self.config, max_batch_size=self.max_batch_size)
        if node_name == "vae_encoder":
            return KleinVaeEncoderSubmodule(self._vae_module(device), self.config)
        if node_name == "vae_decoder":
            return KleinVaeDecoderSubmodule(
                self._vae_module(device), self.config, max_batch_size=self.max_batch_size,
                compile_decode=self.vae_compile,
                warmup_grids=[self.config.latent_grid(h, w) for h, w in self.capture_sizes],
                decode_batch_sizes=self.capture_batch_sizes,
            )
        if node_name == "dit":
            transformer = build_transformer(self.config, self.snapshot, device)
            if self.loras:
                from mstar.model.flux2_klein.weight_loader import apply_transformer_loras

                apply_transformer_loras(transformer, self.loras)
            return KleinDenoiseSubmodule(
                transformer, self.config, loop_name=DENOISE_LOOP,
                attn_resource_key=DIT_ATTN if self.attention_backend == "flashinfer" else None,
                compile_transformer=self.compile_transformer, max_batch_size=self.max_batch_size,
                capture_shapes=self.capture_shapes(), capture_batch_sizes=self.capture_batch_sizes,
            )
        logger.warning("FLUX.2 klein has no submodule for node %r; running it dummy", node_name)
        return None
