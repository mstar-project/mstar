"""LTX25Model: LTX-2.5 text-to-audio+video, the distilled recipe.

Architecture (4 nodes; the dit owns two ragged attention resources):
    text_encoder   Gemma-4 hidden states + text connectors -> video / audio text embeddings
    dit            joint audio-video DiT, one Euler step per Loop iteration (distilled: no guidance)
    vae_decoder    video VAE decode -> uint8 frames
    audio_decoder  audio VAE decode + vocoder -> 48 kHz stereo waveform

Graph walks:
    encode_text     text_encoder; persists text_video, text_audio
    generate_av     Loop("denoise_loop", dit) -> Parallel(vae_decoder -> video, audio_decoder -> audio)
    generate_video  the same loop -> vae_decoder only     (output_modalities == ["video"])
    generate_audio  the same loop -> audio_decoder only   (output_modalities == ["audio"])
    stage1          Loop("denoise_loop", dit) at half resolution -> latent_upsampler;
                    persists refine_latents, refine_audio                (recipe "two_stage")
    refine_av|video|audio
                    Loop("denoise_loop", dit) from the upsampled latents (3 sigmas) -> decoders

A request runs ``encode_text`` then one generate walk (recipe "single", the default),
or ``encode_text`` -> ``stage1`` -> one refine walk ("two_stage", the card's
better-quality recipe; ``height`` / ``width`` are the final size). Walks are stepped
through ``metadata.kwargs["walk_step"]``. The model always denoises video and audio
jointly; the walk only decides which decoders run.

Per-request knobs (``model_kwargs``): ``height``, ``width`` (multiples of 32),
``num_frames`` (8k + 1), ``fps`` (enters the video RoPE and the container),
``seed``, ``recipe`` ("single" | "two_stage"; two-stage sizes are multiples of 64).
The distilled checkpoint runs a fixed 8-sigma schedule with no guidance;
``num_inference_steps`` and the guidance scales are not honoured.
"""

from __future__ import annotations

import io
import logging
from fractions import Fraction

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    NodeResourceSpec,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.ltx2_5.config import (
    AUDIO_DECODER_NODE,
    DENOISE_LOOP,
    DISTILLED_SIGMA_VALUES,
    DIT_NODE,
    LTX25_REPO,
    SNAPSHOT_PATTERNS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    TEXT_ENCODER_NODE,
    UPSAMPLER_NODE,
    VAE_DECODER_NODE,
    LTX25Config,
)
from mstar.model.ltx2_5.submodules import (
    AUDIO_ATTN,
    AUDIO_LATENTS,
    AUDIO_OUTPUT,
    AUDIO_XATTN,
    LATENTS,
    REFINE_AUDIO,
    REFINE_LATENTS,
    TEXT_AUDIO,
    TEXT_INPUTS,
    TEXT_VIDEO,
    VIDEO_ATTN,
    VIDEO_OUTPUT,
    VIDEO_XATTN,
    LTXAudioDecoderSubmodule,
    LTXDenoiseSubmodule,
    LTXLatentUpsamplerSubmodule,
    LTXShape,
    LTXTextEncoderSubmodule,
    LTXVideoDecoderSubmodule,
    shape_from_metadata,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

ENCODE_TEXT_WALK = "encode_text"
GENERATE_AV_WALK = "generate_av"
GENERATE_VIDEO_WALK = "generate_video"
GENERATE_AUDIO_WALK = "generate_audio"
GENERATE_WALKS = {
    frozenset({"video", "audio"}): GENERATE_AV_WALK,
    frozenset({"video"}): GENERATE_VIDEO_WALK,
    frozenset({"audio"}): GENERATE_AUDIO_WALK,
}
# The two-stage recipe: stage 1 at half resolution, the latent upsampler, then a
# refine walk at full resolution that decodes.
STAGE1_WALK = "stage1"
REFINE_AV_WALK = "refine_av"
REFINE_VIDEO_WALK = "refine_video"
REFINE_AUDIO_WALK = "refine_audio"
REFINE_WALKS = {
    frozenset({"video", "audio"}): REFINE_AV_WALK,
    frozenset({"video"}): REFINE_VIDEO_WALK,
    frozenset({"audio"}): REFINE_AUDIO_WALK,
}
RECIPES = ("single", "two_stage")
ATTENTION_BACKENDS = ("flashinfer", "sdpa")


class LTX25Model(Model):
    """LTX-2.5 (22B, distilled) joint audio-video generation."""

    def __init__(
        self,
        model_path_hf: str = LTX25_REPO,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        attention_backend: str = "flashinfer",
        compile: bool = True,
        compile_eager_rounding: bool = True,
        cuda_graph: bool = True,
        capture_shapes: list[list[float]] | None = None,
        capture_refine_shapes: list[list[float]] | None = None,
        capture_batch_sizes: list[int] | None = None,
        max_batch_size: int = 4,
        vae_tile_min_pixels: int = 1280 * 720,
        max_video_tokens: int = 64 * 1024,
        async_scheduling: bool = True,
        **kwargs,
    ):
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {attention_backend!r}")
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # Dummy mode: get_submodule returns None for every node (CPU tests).
        self.skip_weight_loading = skip_weight_loading
        self.attention_backend = attention_backend
        self.compile_transformer = bool(compile)
        self.compile_eager_rounding = bool(compile_eager_rounding)
        self.cuda_graph = bool(cuda_graph)
        # (height, width, num_frames, fps) request shapes the denoise step is captured for;
        # other shapes run the compiled eager path.
        self.capture_shapes = [
            (int(h), int(w), int(f), float(fps)) for h, w, f, fps in (capture_shapes or [[544, 960, 121, 24.0]])
        ]
        # the two-stage recipe's stage 2, by final (height, width, num_frames, fps); its stage 1
        # replays the single-stage capture of the half size
        self.capture_refine_shapes = [
            (int(h), int(w), int(f), float(fps)) for h, w, f, fps in (capture_refine_shapes or [])
        ]
        self.capture_batch_sizes = [int(b) for b in (capture_batch_sizes or [1, 2])]
        self.max_batch_size = int(max_batch_size)
        self.vae_tile_min_pixels = int(vae_tile_min_pixels)
        # Largest latent grid (frames x rows x columns) a request may ask for; the DiT's
        # self-attention is quadratic in it, so an unbounded request could hold a worker
        # for minutes. 64k tokens is ~2x the 1080p, 121-frame grid.
        self.max_video_tokens = int(max_video_tokens)
        # The denoise loop knows its step count at ingestion, so the speculated step past
        # the last is vetoed in prepare_inputs (DenoiseLoopSubmodule) at no GPU cost.
        self.async_scheduling = bool(async_scheduling)
        self._snapshot = None
        self._config: LTX25Config | None = None
        self.tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # ------------------------------------------------------------------ config
    @property
    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = resolve_snapshot_dir(self.model_path_hf, self.cache_dir, allow_patterns=SNAPSHOT_PATTERNS)
        return self._snapshot

    @property
    def config(self) -> LTX25Config:
        """Read from the checkpoint on first use; ``set_config`` pins one for tests."""
        if self._config is None:
            self._config = LTX25Config.from_snapshot(self.snapshot)
        return self._config

    def set_config(self, config: LTX25Config) -> None:
        self._config = config

    def checkpoint_path(self) -> str | None:
        return str(self.snapshot)

    # ------------------------------------------------------------ structure
    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The dit's ragged attentions, one resource per (kind, head geometry): self-
        and cross-attention over the video stream's heads, and over the audio stream's
        (which both cross-modal attentions use). Nothing here is cached; ``"sdpa"``
        declares nothing."""
        if self.attention_backend != "flashinfer":
            return []
        cfg = self.config.transformer
        return [
            spec_cls(
                resource_key=key, nodes={DIT_NODE},
                config=RaggedAttentionConfig(
                    num_qo_heads=heads, num_kv_heads=heads, head_dim=head_dim,
                    # one span per label (or per pair) per request
                    max_segments_per_request=1,
                    dtype=torch.bfloat16,
                ),
            )
            for spec_cls, key, heads, head_dim in (
                (RaggedAttentionSpec, VIDEO_ATTN, cfg.num_attention_heads, cfg.attention_head_dim),
                (RaggedAttentionSpec, AUDIO_ATTN, cfg.audio_num_attention_heads, cfg.audio_attention_head_dim),
                (RaggedCrossAttentionSpec, VIDEO_XATTN, cfg.num_attention_heads, cfg.attention_head_dim),
                (RaggedCrossAttentionSpec, AUDIO_XATTN, cfg.audio_num_attention_heads,
                 cfg.audio_attention_head_dim),
            )
        ]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_text = GraphNode(
            name=TEXT_ENCODER_NODE, enable_async_scheduling=self.async_scheduling,
            input_names=[TEXT_INPUTS],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_VIDEO, persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_AUDIO, persist=True),
            ],
        )
        return {
            ENCODE_TEXT_WALK: encode_text,
            GENERATE_AV_WALK: self._generate_walk(video=True, audio=True),
            GENERATE_VIDEO_WALK: self._generate_walk(video=True, audio=False),
            GENERATE_AUDIO_WALK: self._generate_walk(video=False, audio=True),
            STAGE1_WALK: self._stage1_walk(),
            REFINE_AV_WALK: self._generate_walk(video=True, audio=True, refine=True),
            REFINE_VIDEO_WALK: self._generate_walk(video=True, audio=False, refine=True),
            REFINE_AUDIO_WALK: self._generate_walk(video=False, audio=True, refine=True),
        }

    def _denoise_loop(self, outputs: list[GraphEdge], refine: bool) -> Loop:
        inputs = [TEXT_VIDEO, TEXT_AUDIO, LATENTS, AUDIO_LATENTS] + ([REFINE_LATENTS, REFINE_AUDIO] if refine else [])
        return Loop(
            name=DENOISE_LOOP,
            section=GraphNode(
                name=DIT_NODE,
                input_names=inputs,
                outputs=[
                    GraphEdge(next_node=DIT_NODE, name=LATENTS),
                    GraphEdge(next_node=DIT_NODE, name=AUDIO_LATENTS),
                ],
                enable_async_scheduling=self.async_scheduling,
            ),
            max_iters=len(STAGE_2_DISTILLED_SIGMA_VALUES if refine else DISTILLED_SIGMA_VALUES),
            outputs=outputs,
        )

    def _stage1_walk(self) -> GraphSection:
        to_upsampler = [GraphEdge(next_node=UPSAMPLER_NODE, name=name) for name in (LATENTS, AUDIO_LATENTS)]
        loop = self._denoise_loop(to_upsampler, refine=False)
        upsampler = GraphNode(
            name=UPSAMPLER_NODE, enable_async_scheduling=self.async_scheduling,
            input_names=[LATENTS, AUDIO_LATENTS],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name=REFINE_LATENTS, persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name=REFINE_AUDIO, persist=True),
            ],
        )
        return Sequential([loop, upsampler])

    def _generate_walk(self, video: bool, audio: bool, refine: bool = False) -> GraphSection:
        decoders: list[GraphSection] = []
        loop_outputs: list[GraphEdge] = []
        if video:
            loop_outputs.append(GraphEdge(next_node=VAE_DECODER_NODE, name=LATENTS))
            decoders.append(GraphNode(
                name=VAE_DECODER_NODE, enable_async_scheduling=self.async_scheduling,
                input_names=[LATENTS],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=VIDEO_OUTPUT, output_modality="video")],
            ))
        if audio:
            loop_outputs.append(GraphEdge(next_node=AUDIO_DECODER_NODE, name=AUDIO_LATENTS))
            decoders.append(GraphNode(
                name=AUDIO_DECODER_NODE, enable_async_scheduling=self.async_scheduling,
                input_names=[AUDIO_LATENTS],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=AUDIO_OUTPUT, output_modality="audio")],
            ))
        loop = self._denoise_loop(loop_outputs, refine)
        return Sequential([loop, decoders[0] if len(decoders) == 1 else Parallel(decoders)])

    # ---------------------------------------------------------------- inputs
    def _ensure_tokenizer(self):
        # The `tokenizers` file directly: the checkpoint's tokenizer_config is written
        # for transformers 5.x. It reproduces the pipeline's ids exactly (no BOS).
        if self.tokenizer is None:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(self.snapshot / "tokenizer" / "tokenizer.json"))
        return self.tokenizer

    def warmup_preprocess(self) -> None:
        self._ensure_tokenizer()

    def tokenize(self, prompt: str) -> torch.Tensor:
        """The pipeline's prompt ids, truncated to 1024, without its left padding."""
        ids = self._ensure_tokenizer().encode(prompt.strip()).ids[: self.config.text_max_seq_len]
        return torch.tensor(ids, dtype=torch.long)

    def resolve_request(self, model_kwargs: dict) -> dict:
        """The request's geometry, validated. Raises ``ValueError`` (a 400 when called
        from ``process_prompt``)."""
        cfg, geo = self.config, self.config.geometry

        def integer(name: str, default: int) -> int:
            raw = model_kwargs.get(name)
            try:
                value = int(default if raw is None else raw)
            except (TypeError, ValueError):
                raise ValueError(f"LTX-2.5 {name} must be an integer; got {raw!r}") from None
            if value <= 0:
                raise ValueError(f"LTX-2.5 {name} must be positive; got {value}")
            return value

        recipe = model_kwargs.get("recipe") or "single"
        if recipe not in RECIPES:
            raise ValueError(f"LTX-2.5 recipe must be one of {RECIPES}; got {recipe!r}")
        two_stage = recipe == "two_stage"
        default_h, default_w = cfg.default_height, cfg.default_width
        if two_stage:
            # the card's two-stage example runs stage 1 at the single-stage default size
            default_h, default_w = 2 * default_h, 2 * default_w
        height, width = integer("height", default_h), integer("width", default_w)
        # stage 1 runs at half the size, which must itself be on the VAE's grid
        align = geo.spatial_compression * (2 if two_stage else 1)
        for name, value in (("height", height), ("width", width)):
            if value % align:
                raise ValueError(
                    f"LTX-2.5 {name}={value} must be a multiple of {align} "
                    f"(the VAE's spatial stride{', at half size for stage 1' if two_stage else ''})"
                )
        num_frames = integer("num_frames", cfg.default_num_frames)
        if (num_frames - 1) % geo.temporal_compression:
            lower = (num_frames - 1) // geo.temporal_compression * geo.temporal_compression + 1
            raise ValueError(
                f"LTX-2.5 num_frames={num_frames} must be {geo.temporal_compression}k+1 (the VAE compresses time "
                f"by {geo.temporal_compression} after the first frame); nearest: {lower} or "
                f"{lower + geo.temporal_compression}"
            )
        raw_fps = model_kwargs.get("fps")
        try:
            fps = float(cfg.default_fps if raw_fps is None else raw_fps)
        except (TypeError, ValueError):
            raise ValueError(f"LTX-2.5 fps must be a number; got {raw_fps!r}") from None
        if fps <= 0:
            raise ValueError(f"LTX-2.5 fps must be positive; got {raw_fps!r}")
        stride = geo.spatial_compression
        tokens = geo.latent_frames(num_frames) * (height // stride) * (width // stride)
        if tokens > self.max_video_tokens:
            raise ValueError(
                f"LTX-2.5 {width}x{height}x{num_frames} is {tokens} latent tokens, above this server's "
                f"max_video_tokens of {self.max_video_tokens}"
            )
        return {"height": height, "width": width, "num_frames": num_frames, "fps": fps, "recipe": recipe}

    @staticmethod
    def _generate_walk_for(output_modalities: list[str] | None, recipe: str = "single") -> str:
        wanted = frozenset(output_modalities or ["video", "audio"])
        walk = (REFINE_WALKS if recipe == "two_stage" else GENERATE_WALKS).get(wanted)
        if walk is None:
            raise ValueError(
                f"LTX-2.5 generates video and/or audio; got output modalities {sorted(wanted)}"
            )
        return walk

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        """Validate the request at the 400-producing seam and tokenize the prompt."""
        self._generate_walk_for(output_modalities)
        if any(m != "text" for m in input_modalities or []):
            raise ValueError(f"LTX-2.5 (this port) takes a text prompt only; got inputs {input_modalities}")
        if prompt is None or not prompt.strip():
            raise ValueError("LTX-2.5 requires a non-empty text prompt")
        self.resolve_request(kwargs)
        for name in ("num_inference_steps", "guidance_scale"):
            if kwargs.get(name) is not None:
                logger.warning("LTX-2.5 %s is ignored: the distilled checkpoint runs a fixed schedule, unguided", name)
        return {TEXT_INPUTS: [self.tokenize(prompt)]}

    # --------------------------------------------------------- state machine
    def _step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        # The request's geometry lives under its own key: the conductor merges each
        # pass's step metadata into ``metadata.kwargs``, so stage 1's half-size
        # "height" would otherwise overwrite the request's for every later walk.
        kw = metadata.kwargs["request"]
        scale = 2 if metadata.graph_walk == STAGE1_WALK else 1
        return {
            "is_prefill": metadata.is_prefill,
            "height": kw["height"] // scale, "width": kw["width"] // scale,
            "num_frames": kw["num_frames"], "fps": kw["fps"],
        }

    def get_initial_forward_pass_args(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        # Backstops: process_prompt rejected these already, where a raise is a 400; a
        # raise here runs at the conductor.
        if not input_signals.get(TEXT_INPUTS):
            raise ValueError("LTX-2.5 requires a text prompt (text_inputs)")
        model_kwargs = model_kwargs or {}
        request = self.resolve_request(model_kwargs)
        final_walk = self._generate_walk_for(output_modalities, request["recipe"])
        middle = [STAGE1_WALK] if request["recipe"] == "two_stage" else []
        kwargs = {"walk_schedule": [ENCODE_TEXT_WALK, *middle, final_walk], "walk_step": 0, "request": request}
        logger.info(
            "LTX-2.5 request: %dx%d frames=%d fps=%s recipe=%s outputs=%s seed=%s", request["width"],
            request["height"], request["num_frames"], request["fps"], request["recipe"], output_modalities,
            model_kwargs.get("seed", "auto"),
        )
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities, output_modalities=output_modalities,
            graph_walk=ENCODE_TEXT_WALK, is_prefill=True, kwargs=kwargs,
        )
        edge = GraphEdge(next_node=TEXT_ENCODER_NODE, name=TEXT_INPUTS)
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
        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        inputs: list[GraphEdge] = []
        request_done = step >= len(schedule)
        unpersist: list[TensorPointerInfo] = []
        if not request_done:
            metadata.kwargs["walk_step"] = step
            walk = metadata.graph_walk = schedule[step]
            metadata.is_prefill = False
            persisted = [TEXT_VIDEO, TEXT_AUDIO]
            if walk in REFINE_WALKS.values():
                persisted += [REFINE_LATENTS, REFINE_AUDIO]
            for name in persisted:
                edge = GraphEdge(next_node=DIT_NODE, name=name)
                edge.tensor_info = persist_signals.get(name, [])
                inputs.append(edge)
                # the text embeddings outlive stage 1: the refine walk reads them too
                if walk != STAGE1_WALK:
                    unpersist += edge.tensor_info
            # the loop-back edges arrive empty; the dit seeds them at iteration 0
            inputs += [GraphEdge(next_node=DIT_NODE, name=LATENTS), GraphEdge(next_node=DIT_NODE, name=AUDIO_LATENTS)]
        return ForwardPassArgs(
            full_metadata=metadata, inputs=inputs, unpersist_tensors=unpersist,
            step_metadata=self._step_metadata(metadata), request_done=request_done,
        )

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """``video``: uint8 ``[3, F, H, W]`` -> H.264 mp4 at the request's fps. ``audio``:
        ``[2, samples]`` in ``[-1, 1]`` -> headerless interleaved 16-bit PCM (the API layer
        wraps it with ``get_output_sample_rate``)."""
        if modality == "audio":
            x = output[0] if output.ndim == 3 else output
            pcm = (x.detach().to(torch.float32).clamp(-1, 1) * 32767.0).round().to(torch.int16)
            return pcm.T.contiguous().cpu().numpy().tobytes()
        if modality != "video":
            raise ValueError(f"unsupported output modality for LTX-2.5: {modality!r}")
        import av

        fps = self.resolve_request(request_kwargs or {})["fps"]
        frames = (output[0] if output.ndim == 5 else output).permute(1, 2, 3, 0).cpu().numpy()
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        # PyAV wants a rational rate: 24 -> 24, 23.976 -> 2997/125
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        stream.height, stream.width = frames.shape[1], frames.shape[2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return buffer.getvalue()

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.geometry.output_sample_rate

    def get_output_audio_channels(self, modality: str = "audio") -> int:
        return self.config.geometry.output_audio_channels

    def get_output_frame_rate(self, modality: str = "video_frame", request_kwargs: dict | None = None) -> float:
        return self.resolve_request(request_kwargs or {})["fps"]

    # ------------------------------------------------------------- loading
    def get_autocast_dtype(self):
        # None: no engine autocast or blanket cast; numerics follow the checkpoint
        # dtypes (bf16 weights, fp32 latents and step math), as the parity tests pin.
        return None

    def capture_buckets(self) -> list[tuple[str, LTXShape]]:
        if not self.cuda_graph:
            return []
        def shape(h, w, f, fps):
            return shape_from_metadata(self.config, {"height": h, "width": w, "num_frames": f, "fps": fps})

        return [(GENERATE_AV_WALK, shape(*s)) for s in self.capture_shapes] + [
            (REFINE_AV_WALK, shape(*s)) for s in self.capture_refine_shapes
        ]

    def get_default_sharding_config(self) -> ShardingConfig:
        # The dit's linears and qk-norms shard by head (tensor parallelism); what crosses
        # node edges stays replicated, so nothing needs a shard_dim.
        return ShardingConfig(groups=[], tp_enabled_nodes={DIT_NODE}, shard_dim={})

    def get_submodule(self, node_name: str, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None):
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, tp_group)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("loaded LTX-2.5 submodule for node %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device, tp_group=None) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        from mstar.model.ltx2_5 import weight_loader

        snapshot = self.snapshot
        if node_name == TEXT_ENCODER_NODE:
            return LTXTextEncoderSubmodule(
                weight_loader.build_text_encoder(self.config, snapshot, device),
                weight_loader.build_connectors(self.config, snapshot, device),
                self.config, max_batch_size=self.max_batch_size,
            )
        if node_name == DIT_NODE:
            return LTXDenoiseSubmodule(
                weight_loader.build_transformer(self.config, snapshot, device, comm_group=tp_group), self.config,
                loop_name=DENOISE_LOOP, use_ragged_attention=self.attention_backend == "flashinfer",
                refine_walks=frozenset(REFINE_WALKS.values()),
                compile_transformer=self.compile_transformer, compile_eager_rounding=self.compile_eager_rounding,
                max_batch_size=self.max_batch_size, capture_buckets=self.capture_buckets(),
                capture_batch_sizes=self.capture_batch_sizes,
                # one capture serves every generate walk: the dit's step is the same in each
                replay_walks={
                    GENERATE_AV_WALK: [GENERATE_VIDEO_WALK, GENERATE_AUDIO_WALK, STAGE1_WALK],
                    REFINE_AV_WALK: [REFINE_VIDEO_WALK, REFINE_AUDIO_WALK],
                },
            )
        if node_name == VAE_DECODER_NODE:
            from diffusers import AutoencoderKLLTX2Video

            vae = AutoencoderKLLTX2Video.from_pretrained(str(snapshot / "vae"), torch_dtype=torch.bfloat16)
            return LTXVideoDecoderSubmodule(
                vae.to(device).eval(), self.config, tile_min_pixels=self.vae_tile_min_pixels,
            )
        if node_name == UPSAMPLER_NODE:
            from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

            upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                str(snapshot / "latent_upsampler"), torch_dtype=torch.bfloat16,
            )
            # the latent statistics only; the VAEs themselves live on the decoder nodes
            vae_cfg = AutoencoderKLLTX2Video.load_config(str(snapshot / "vae"))
            vae = AutoencoderKLLTX2Video.from_pretrained(str(snapshot / "vae"), torch_dtype=torch.float32)
            audio_vae = AutoencoderKLLTX2Audio.from_pretrained(str(snapshot / "audio_vae"), torch_dtype=torch.float32)
            submodule = LTXLatentUpsamplerSubmodule(
                upsampler.eval(),
                (vae.latents_mean, vae.latents_std, float(vae_cfg["scaling_factor"])),
                (audio_vae.latents_mean, audio_vae.latents_std),
                self.config,
            )
            del vae, audio_vae
            return submodule.to(device)
        if node_name == AUDIO_DECODER_NODE:
            from diffusers import AutoencoderKLLTX2Audio
            from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

            audio_vae = AutoencoderKLLTX2Audio.from_pretrained(str(snapshot / "audio_vae"), torch_dtype=torch.bfloat16)
            vocoder = LTX2VocoderWithBWE.from_pretrained(str(snapshot / "vocoder"), torch_dtype=torch.bfloat16)
            return LTXAudioDecoderSubmodule(audio_vae.to(device).eval(), vocoder.to(device).eval(), self.config)
        logger.warning("LTX-2.5 has no submodule for node %r; running it dummy", node_name)
        return None
