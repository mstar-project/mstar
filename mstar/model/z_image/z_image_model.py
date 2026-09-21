"""ZImageModel: Z-Image-Turbo text-to-image (8 steps, no CFG) on the DiT scaffold.

Architecture (3 nodes; the dit optionally owns one ragged attention resource):

    text_encoder  Qwen3-4B hidden states after layer 35 for the real prompt tokens
    dit           single-stream flow transformer (2 + 2 refiner blocks, 30 blocks), fp32 Euler
    vae_decoder   FLUX.1 VAE decode -> uint8 image

Graph walks:

    encode_text   text_encoder; persists text_embeds
    image_gen     Loop("denoise_loop", dit{text_embeds, latents}) -> vae_decoder -> EMIT image

Per-request knobs (``model_kwargs``): ``width`` / ``height`` (multiples of 16),
``num_inference_steps`` (default 8), ``seed``. The Turbo checkpoint runs without guidance;
``guidance_scale`` is accepted and ignored with a log line.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import NodeResourceSpec, RaggedAttentionConfig, RaggedAttentionSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.components.diffusion.image_io import encode_image
from mstar.model.flux2_klein.submodules import VAE_DECODE_BATCH_SIZES
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.z_image.config import DENOISE_LOOP, DIT_ATTN, Z_IMAGE_TURBO, ZImageConfig, resolve_snapshot_dir
from mstar.model.z_image.submodules import (
    IMAGE_OUTPUT,
    LATENTS,
    TEXT_EMBEDS,
    TEXT_INPUTS,
    ZImageDenoiseSubmodule,
    ZImageTextEncoderSubmodule,
    ZImageVaeDecoderSubmodule,
    ZShape,
    padded_length,
)

logger = logging.getLogger(__name__)

ENCODE_TEXT_WALK = "encode_text"
IMAGE_GEN_WALK = "image_gen"
ATTENTION_BACKENDS = ("flashinfer", "sdpa")


class ZImageModel(Model):
    def __init__(
        self,
        model_path_hf: str = Z_IMAGE_TURBO,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        attention_backend: str = "flashinfer",
        compile: bool = True,
        compile_eager_rounding: bool = True,
        compile_exact_ops: bool | list[str] = False,
        cuda_graph: bool = True,
        capture_sizes: list[list[int]] | None = None,
        capture_caption_lengths: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        max_batch_size: int = 8,
        max_image_area: int = 2048 * 2048,
        vae_compile: bool = False,
        **kwargs,
    ):
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {attention_backend!r}")
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        self.attention_backend = attention_backend
        self.compile_transformer = bool(compile)
        self.compile_eager_rounding = bool(compile_eager_rounding)
        # True: every norm / activation class stays eager inside the compiled forward; a list picks classes
        self.compile_exact_ops = compile_exact_ops if isinstance(compile_exact_ops, list) else bool(compile_exact_ops)
        self.cuda_graph = bool(cuda_graph)
        self.capture_sizes = [tuple(int(v) for v in s) for s in (capture_sizes or [[1024, 1024]])]
        # Captions round up to a multiple of 32 tokens; short prompts land in the first two buckets.
        self.capture_caption_lengths = [int(n) for n in (capture_caption_lengths or [32, 64])]
        self.capture_batch_sizes = [int(b) for b in (capture_batch_sizes or [1, 2, 4, 8])]
        self.max_batch_size = int(max_batch_size)
        # largest output (pixels) a request may ask for; larger requests are rejected before scheduling
        self.max_image_area = int(max_image_area)
        self.vae_compile = bool(vae_compile)
        self._snapshot = None
        self._config: ZImageConfig | None = None
        self.tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = resolve_snapshot_dir(self.model_path_hf, self.cache_dir)
        return self._snapshot

    @property
    def config(self) -> ZImageConfig:
        if self._config is None:
            self._config = ZImageConfig.from_snapshot(self.snapshot)
        return self._config

    def set_config(self, config: ZImageConfig) -> None:
        self._config = config

    # ------------------------------------------------------------ structure
    def get_node_resources(self) -> list[NodeResourceSpec]:
        if self.attention_backend != "flashinfer":
            return []
        tcfg = self.config.transformer
        return [
            RaggedAttentionSpec(
                resource_key=DIT_ATTN, nodes={"dit"},
                config=RaggedAttentionConfig(
                    num_qo_heads=tcfg.n_heads, num_kv_heads=tcfg.n_heads, head_dim=tcfg.head_dim,
                    # per label (image / caption / main span) one segment per request
                    max_segments_per_request=1,
                    # the DiT's activation dtype: there is no KV cache to inherit one from
                    dtype=torch.bfloat16,
                ),
            )
        ]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_text = GraphNode(
            name="text_encoder", input_names=[TEXT_INPUTS], enable_async_scheduling=False,
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS, persist=True)],
        )
        loop = Loop(
            name=DENOISE_LOOP,
            section=GraphNode(
                name="dit", input_names=[TEXT_EMBEDS, LATENTS],
                # lockstep scheduling so concurrent requests batch (see Flux2KleinModel)
                outputs=[GraphEdge(next_node="dit", name=LATENTS)], enable_async_scheduling=False,
            ),
            max_iters=self.config.max_denoise_steps,
            outputs=[GraphEdge(next_node="vae_decoder", name=LATENTS)],
        )
        decoder = GraphNode(
            name="vae_decoder", input_names=[LATENTS], enable_async_scheduling=False,
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=IMAGE_OUTPUT, output_modality="image")],
        )
        return {ENCODE_TEXT_WALK: encode_text, IMAGE_GEN_WALK: Sequential([loop, decoder])}

    # ---------------------------------------------------------------- inputs
    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot / "tokenizer"))
        return self.tokenizer

    def render_prompt(self, prompt: str) -> str:
        """The pipeline's chat-templated prompt (user turn, generation prompt, thinking ON)."""
        return self._ensure_tokenizer().apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Unpadded token ids ``[L]`` (``L <= max_sequence_length``), truncated like the pipeline."""
        enc = self._ensure_tokenizer()(
            self.render_prompt(prompt), return_tensors="pt", truncation=True,
            max_length=self.config.text_encoder.max_sequence_length,
        )
        return enc["input_ids"][0].to(torch.long)

    def _resolve_size(self, model_kwargs: dict) -> tuple[int, int]:
        height = int(model_kwargs.get("height", self.config.default_height))
        width = int(model_kwargs.get("width", self.config.default_width))
        align = self.config.spatial_alignment
        for name, value in (("height", height), ("width", width)):
            if value <= 0 or value % align:
                raise ValueError(f"Z-Image {name}={value} must be a positive multiple of {align}")
        if height * width > self.max_image_area:
            raise ValueError(
                f"Z-Image {width}x{height} exceeds this server's max_image_area of {self.max_image_area} pixels "
                f"({int(self.max_image_area ** 0.5)}^2)"
            )
        return height, width

    def _resolve_steps(self, model_kwargs: dict) -> int:
        steps = int(model_kwargs.get("num_inference_steps", self.config.default_num_inference_steps))
        if steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
        return min(steps, self.config.max_denoise_steps)

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        if (output_modalities or ["image"])[0] != "image":
            raise ValueError(f"Z-Image only generates images; got output modality {output_modalities!r}")
        if prompt is None:
            raise ValueError("Z-Image requires a text prompt")
        if "image" in input_modalities:
            raise ValueError("Z-Image-Turbo is text-to-image only; reference images are not supported")
        self._resolve_size(kwargs)
        self._resolve_steps(kwargs)
        if float(kwargs.get("guidance_scale", 0.0)) > 0.0:
            logger.info("guidance_scale is ignored: Z-Image-Turbo is distilled and runs without CFG")
        return {TEXT_INPUTS: [self.tokenize(prompt)]}

    # --------------------------------------------------------- state machine
    def _step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        kw = metadata.kwargs
        return {
            "is_prefill": metadata.is_prefill, "height": kw["height"], "width": kw["width"],
            "num_inference_steps": kw["num_inference_steps"], "text_len": kw["text_len"], "cap_len": kw["cap_len"],
        }

    def get_initial_forward_pass_args(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        model_kwargs = model_kwargs or {}
        if not input_signals.get(TEXT_INPUTS):
            raise ValueError("Z-Image requires a text prompt (text_inputs)")
        text_len = int(input_signals[TEXT_INPUTS][0].dims[0])
        height, width = self._resolve_size(model_kwargs)
        kwargs = {
            "walk_schedule": [ENCODE_TEXT_WALK, IMAGE_GEN_WALK], "walk_step": 0,
            "height": height, "width": width, "num_inference_steps": self._resolve_steps(model_kwargs),
            "text_len": text_len, "cap_len": padded_length(text_len),
        }
        logger.info("Z-Image request: %dx%d steps=%d tokens=%d seed=%s", width, height,
                    kwargs["num_inference_steps"], text_len, model_kwargs.get("seed", "auto"))
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities, output_modalities=output_modalities,
            graph_walk=ENCODE_TEXT_WALK, is_prefill=True, kwargs=kwargs,
        )
        edge = GraphEdge(next_node="text_encoder", name=TEXT_INPUTS)
        edge.tensor_info = input_signals[TEXT_INPUTS]
        return ForwardPassArgs(
            full_metadata=metadata, inputs=[edge], unpersist_tensors=list(edge.tensor_info),
            step_metadata=self._step_metadata(metadata),
        )

    def get_partition_forward_pass_args(
        self, partition_name: str, partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        step = metadata.kwargs["walk_step"] + 1
        schedule = metadata.kwargs["walk_schedule"]
        inputs: list[GraphEdge] = []
        request_done = step >= len(schedule)
        if not request_done:
            metadata.kwargs["walk_step"] = step
            metadata.graph_walk = schedule[step]
            metadata.is_prefill = False
            edge = GraphEdge(next_node="dit", name=TEXT_EMBEDS)
            edge.tensor_info = persist_signals.get(TEXT_EMBEDS, [])
            inputs = [edge, GraphEdge(next_node="dit", name=LATENTS)]
        return ForwardPassArgs(
            full_metadata=metadata, inputs=inputs,
            unpersist_tensors=sum([e.tensor_info for e in inputs], start=[]),
            step_metadata=self._step_metadata(metadata), request_done=request_done,
        )

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality != "image":
            raise ValueError(f"unsupported output modality for Z-Image: {modality!r}")
        return encode_image(output, request_kwargs)

    # ------------------------------------------------------------- loading
    def get_autocast_dtype(self):
        return None

    def capture_shapes(self) -> list[tuple[str, Hashable]]:
        if not self.cuda_graph:
            return []
        return [
            (IMAGE_GEN_WALK, ZShape(grid=self.config.latent_grid(h, w), cap_len=cap_len))
            for h, w in self.capture_sizes
            for cap_len in self.capture_caption_lengths
        ]

    def get_submodule(self, node_name: str, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None):
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(self, node_name: str, device) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        from mstar.model.z_image.weight_loader import build_text_encoder, build_transformer, build_vae

        if node_name == "text_encoder":
            return ZImageTextEncoderSubmodule(build_text_encoder(self.config, self.snapshot, device), self.config,
                                              max_batch_size=self.max_batch_size)
        if node_name == "vae_decoder":
            return ZImageVaeDecoderSubmodule(
                build_vae(self.config, self.snapshot, device), self.config, max_batch_size=self.max_batch_size,
                compile_decode=self.vae_compile,
                warmup_grids=[self.config.latent_grid(h, w) for h, w in self.capture_sizes],
                decode_batch_sizes=[s for s in VAE_DECODE_BATCH_SIZES if s <= self.max_batch_size],  # see klein
            )
        if node_name == "dit":
            return ZImageDenoiseSubmodule(
                build_transformer(self.config, self.snapshot, device), self.config, loop_name=DENOISE_LOOP,
                attn_resource_key=DIT_ATTN if self.attention_backend == "flashinfer" else None,
                compile_transformer=self.compile_transformer,
                compile_eager_rounding=self.compile_eager_rounding, compile_exact_ops=self.compile_exact_ops,
                max_batch_size=self.max_batch_size,
                capture_shapes=self.capture_shapes(), capture_batch_sizes=self.capture_batch_sizes,
            )
        logger.warning("Z-Image has no submodule for node %r; running it dummy", node_name)
        return None
