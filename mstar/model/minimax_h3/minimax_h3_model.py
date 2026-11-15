"""H3Model: MiniMax-H3 text- and first/last-frame-to-audio+video (the FL2VA checkpoint).

Nodes:
    text_encoder   Qwen3-VL-32B to layer 50 (keyframes through its vision tower), then
                   condition_proj and the token refiner; once per request
    vae_encoder    the keyframes' conditioning latents
    dit            the 50-block packed-sequence DiT, one step of both schedules per loop
                   iteration (guidance-distilled: one forward per step)
    vae_decoder    the video VAE's ViT decoder (in the dit's SP group, a share of the clips per rank)
    audio_decoder  BigVGAN, 32 kHz stereo

Walks: encode_text or encode_keyframes, then generate_{av,video,audio}[_keyframes]: the
denoise loop and the decoders asked for. Video and audio always denoise together; the
walk only picks the decoders. Request knobs: docs/models.rst.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import SeqShardSpec
from mstar.engine.state import Array, Attend, Block, Read, SeqShard
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import Model, TensorAndMetadata, WalkStep
from mstar.model.components.diffusion.attention import dit_entry, refuse_attention_backend_kwarg
from mstar.model.components.video_encode import encode_mp4
from mstar.model.media_limits import PILLOW_IMAGE_FORMATS, check_image_size
from mstar.model.minimax_h3.components.layout import H3Layout
from mstar.model.minimax_h3.config import (
    AUDIO_CHANNELS,
    AUDIO_DECODER_NODE,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DENOISE_LOOP,
    DIT_ATTN,
    DIT_NODE,
    DIT_ROWS,
    DIT_SHARD,
    FPS,
    H3_REPO,
    MAX_DURATION,
    MIN_DURATION,
    PARTITION,
    PROMPT_ROWS,
    REFINER_ATTN,
    SNAPSHOT_PATTERNS,
    TEXT_ENCODER_NODE,
    VAE_DECODER_NODE,
    VAE_ENCODER_NODE,
    VAE_SHARD,
    H3Config,
    H3TransformerConfig,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from mstar.model.minimax_h3.keyframes import ANCHORS, fit_keyframes, presentation, vision_spans
from mstar.model.minimax_h3.schedule import ScheduleArgs, h3_schedule
from mstar.model.minimax_h3.submodules import (
    AUDIO_LATENTS,
    AUDIO_OUTPUT,
    COND_LATENTS,
    KEYFRAME_PIXELS,
    LATENTS,
    TEXT_EMBEDS,
    TEXT_INPUTS,
    TEXT_SPAN,
    VIDEO_OUTPUT,
    VISION_PIXELS,
    H3AudioDecoderSubmodule,
    H3DenoiseSubmodule,
    H3TextEncoderSubmodule,
    H3VaeEncoderSubmodule,
    H3VideoDecoderSubmodule,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

ENCODE_TEXT_WALK = "encode_text"
ENCODE_KEYFRAMES_WALK = "encode_keyframes"
GENERATE_AV_WALK = "generate_av"
GENERATE_VIDEO_WALK = "generate_video"
GENERATE_AUDIO_WALK = "generate_audio"
GENERATE_WALKS = {
    frozenset({"video", "audio"}): GENERATE_AV_WALK,
    frozenset({"video"}): GENERATE_VIDEO_WALK,
    frozenset({"audio"}): GENERATE_AUDIO_WALK,
}
KEYFRAMES_SUFFIX = "_keyframes"  # a generate walk whose dit also reads cond_latents


def _positive_int(model_kwargs: dict, name: str, default: int | None) -> int | None:
    raw = model_kwargs.get(name)
    if raw is None:
        return default
    try:
        # a bool, a string, or a float int() would truncate (768.5, inf), is not an integer;
        # the API normalizes the knobs it knows before this
        if isinstance(raw, (bool, str)):
            raise ValueError
        value = int(raw)
        if isinstance(raw, float) and raw != value:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"MiniMax-H3 {name} must be an integer; got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"MiniMax-H3 {name} must be positive; got {value}")
    return value


def _positive_float(model_kwargs: dict, name: str, default: float) -> float:
    raw = model_kwargs.get(name)
    try:
        # a string is refused, not parsed: the API normalizes the knobs it knows before this
        if isinstance(raw, (bool, str)):
            raise ValueError
        value = float(default if raw is None else raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"MiniMax-H3 {name} must be a number; got {raw!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"MiniMax-H3 {name} must be positive and finite; got {raw!r}")
    return value


# only the keyframes' own layout may carry these
VISION_TOKENS = ("<|vision_start|>", "<|image_pad|>", "<|video_pad|>", "<|vision_end|>")


def import_image_processor():
    """transformers' ``AutoImageProcessor``. Importing it imports torchaudio when that is
    installed, which fails for a torchaudio built for another torch; the error says so."""
    try:
        from transformers import AutoImageProcessor
    except OSError as exc:
        if "torchaudio" not in str(exc):
            raise
        raise RuntimeError(
            f"MiniMax-H3 lays keyframes out with transformers' image processor, and importing it "
            f"imports torchaudio, which does not load with torch {torch.__version__} (a torchaudio "
            f"built for another torch): {exc}. Install the torchaudio build that matches this torch."
        ) from exc
    return AutoImageProcessor


MAX_KEYFRAMES = 2
# A ceiling for the loop graph; a request runs its own step count (get_loop_caps).
MAX_INFERENCE_STEPS = 200
CANVAS_MULTIPLE = 32  # the VAE's 16 times the transformer's 2-wide patch


def dit_state(t: H3TransformerConfig) -> list[Array | SeqShard]:
    """The DiT's attention over each request's packed rows, its padding a segment of its own in the same row, the
    token refiner's over the prompt, and the DiT's sequence shard; nothing here keeps state. On SP ranks the
    shard splits the packed rows (one row a step, read segmented, so cut contiguously), which the attention
    exchanges for heads (Ulysses); on SDPA it can't run there. Shared by the MiniMax-H3 family's models."""
    heads, entry = t.num_attention_heads, dit_entry(t.num_attention_heads, t.attention_head_dim)
    return [
        Array(
            DIT_ROWS, nodes={DIT_NODE}, layers=frozenset(range(t.num_layers)), entry=entry, max_len=0,
            reads={DIT_ATTN: Read(mask=Block(), op=Attend(q_heads=heads), packed=False, segmented=True)},
        ),
        Array(
            PROMPT_ROWS, nodes={TEXT_ENCODER_NODE}, layers=frozenset(range(t.token_refiner_num_layers)),
            entry=entry, labels={TEXT_SPAN}, max_len=0,
            reads={REFINER_ATTN: Read(mask=Block(), op=Attend(q_heads=heads), packed=False)},
        ),
        SeqShard(DIT_SHARD, nodes={DIT_NODE}, labels={H3DenoiseSubmodule.MAIN_SPAN}),
    ]


# the DiT's two segments are a request's rows and their padding
DIT_KERNEL_SETTINGS = {
    DIT_ATTN: {"max_segments_per_request": 2, "workspace_mb": 128}, REFINER_ATTN: {"workspace_mb": 128},
}


class H3Model(Model):
    """MiniMax-H3 (33B DiT, guidance-distilled) joint audio-video generation."""

    def __init__(
        self,
        model_path_hf: str = H3_REPO,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        max_batch_size: int = 1,
        max_packed_rows: int = 131_072,
        max_prompt_tokens: int | None = None,
        async_scheduling: bool = True,
        compile: bool = False,
        row_bucket: int | None = None,
        vae_tile_batch: int = 1,
        vae_compile: bool = False,
        **kwargs,
    ):
        refuse_attention_backend_kwarg(kwargs)
        if kwargs:
            # **kwargs is here for the refusal above, so any other key is a misspelling that has to fail the load
            raise TypeError(f"H3Model got unexpected model_kwargs {sorted(kwargs)}")
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # dummy mode: get_submodule returns None for every node (CPU tests)
        self.skip_weight_loading = skip_weight_loading
        # a batch's rows would have to agree on the step's first-step decisions (keyframe
        # noising, a one-timestep embedding)
        if int(max_batch_size) != 1:
            raise ValueError(f"MiniMax-H3 runs one request per denoise step; got max_batch_size={max_batch_size}")
        # Prompt, keyframe, audio and video rows; a step's attention grows with their square.
        # The default covers the longest 768p clip (14.4 s, ~105k rows) with a ~25k-token prompt.
        self.max_packed_rows = int(max_packed_rows)
        # keyframe tokens included; None is the text encoder's positions, the reference's only cap
        self.max_prompt_tokens = None if max_prompt_tokens is None else int(max_prompt_tokens)
        self.async_scheduling = bool(async_scheduling)
        self.compile = bool(compile)  # the DiT's blocks
        # the DiT's packed rows pad to a multiple of this (and of the SP size): one compiled
        # graph per 256 prompt tokens, not per prompt length
        self.row_bucket = (256 if self.compile else 1) if row_bucket is None else int(row_bucket)
        if self.row_bucket < 1:
            raise ValueError(f"row_bucket must be a positive int, got {row_bucket!r}")
        self.vae_tile_batch = int(vae_tile_batch)  # 1 decodes one tile a call, as the reference
        self.vae_compile = bool(vae_compile)
        self._snapshot = None
        self._config: H3Config | None = None
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
    def config(self) -> H3Config:
        """Read from the checkpoint on first use; ``set_config`` pins one for tests."""
        if self._config is None:
            self._config = H3Config.from_snapshot(self.snapshot)
        return self._config

    def set_config(self, config: H3Config) -> None:
        self._config = config

    def checkpoint_path(self) -> str | None:
        return str(self.snapshot)

    # ------------------------------------------------------------ structure
    def declare_state(self) -> list[Array | SeqShard]:
        return dit_state(self.config.transformer)

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        return {key: dict(settings) for key, settings in DIT_KERNEL_SETTINGS.items()}

    def get_node_resources(self) -> list[SeqShardSpec]:
        """The video decoder's sequence shard, which splits each request's clips, which nothing exchanges
        (``exchanges=False``); every request of a decoder batch is one layout, so one clip count (``axis`` 1). The
        decoder has no arrays, so its shard is written by hand, not a `SeqShard`."""
        return [
            SeqShardSpec(
                resource_key=VAE_SHARD, nodes={VAE_DECODER_NODE}, labels=(H3VideoDecoderSubmodule.CLIPS,), axis=1,
                exchanges=False,
            ),
        ]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def text_node(*inputs: str) -> GraphNode:
            return GraphNode(
                name=TEXT_ENCODER_NODE, enable_async_scheduling=self.async_scheduling, input_names=list(inputs),
                outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TEXT_EMBEDS, persist=True)],
            )

        encode_keyframes = Parallel([
            text_node(TEXT_INPUTS, VISION_PIXELS),
            GraphNode(
                name=VAE_ENCODER_NODE, enable_async_scheduling=self.async_scheduling, input_names=[KEYFRAME_PIXELS],
                outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=COND_LATENTS, persist=True)],
            ),
        ])
        walks: dict[str, GraphSection] = {ENCODE_TEXT_WALK: text_node(TEXT_INPUTS),
                                          ENCODE_KEYFRAMES_WALK: encode_keyframes}
        for name, (video, audio) in ((GENERATE_AV_WALK, (True, True)), (GENERATE_VIDEO_WALK, (True, False)),
                                     (GENERATE_AUDIO_WALK, (False, True))):
            walks[name] = self._generate_walk(video, audio, keyframes=False)
            walks[name + KEYFRAMES_SUFFIX] = self._generate_walk(video, audio, keyframes=True)
        return walks

    def _generate_walk(self, video: bool, audio: bool, keyframes: bool) -> GraphSection:
        decoders: list[GraphSection] = []
        loop_outputs: list[GraphEdge] = []
        if video:
            loop_outputs.append(GraphEdge(next_node=VAE_DECODER_NODE, name=LATENTS))
            decoders.append(GraphNode(
                name=VAE_DECODER_NODE, enable_async_scheduling=self.async_scheduling, input_names=[LATENTS],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=VIDEO_OUTPUT, output_modality="video")],
            ))
        if audio:
            loop_outputs.append(GraphEdge(next_node=AUDIO_DECODER_NODE, name=AUDIO_LATENTS))
            decoders.append(GraphNode(
                name=AUDIO_DECODER_NODE, enable_async_scheduling=self.async_scheduling, input_names=[AUDIO_LATENTS],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=AUDIO_OUTPUT, output_modality="audio")],
            ))
        loop = Loop(
            name=DENOISE_LOOP,
            section=GraphNode(
                name=DIT_NODE,
                input_names=[TEXT_EMBEDS, LATENTS, AUDIO_LATENTS, *([COND_LATENTS] if keyframes else [])],
                outputs=[GraphEdge(next_node=DIT_NODE, name=LATENTS),
                         GraphEdge(next_node=DIT_NODE, name=AUDIO_LATENTS)],
                enable_async_scheduling=self.async_scheduling,
            ),
            max_iters=MAX_INFERENCE_STEPS,
            outputs=loop_outputs,
        )
        return Sequential([loop, decoders[0] if len(decoders) == 1 else Parallel(decoders)])

    # ---------------------------------------------------------------- inputs
    def _ensure_tokenizer(self):
        # verbatim, no special tokens added: the reference's Qwen2TokenizerFast call
        if self.tokenizer is None:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(self.snapshot / PARTITION / "tokenizer" / "tokenizer.json"))
        return self.tokenizer

    @property
    def image_processor(self):
        """Qwen3-VL's image processor (``FL2VA/processor``), as the reference uses it."""
        if self._image_processor is None:
            self._image_processor = import_image_processor().from_pretrained(
                str(self.snapshot / PARTITION / "processor"))
        return self._image_processor

    def warmup_preprocess(self) -> None:
        self._ensure_tokenizer()
        self.image_processor  # noqa: B018

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse to start where keyframes could not be laid out, rather than 500 on the
        first first/last-frame request."""
        if self.skip_weight_loading:
            return
        try:
            import_image_processor()
        except RuntimeError as exc:
            raise ValueError(f"{config_path}: {exc}") from exc

    def max_input_files(self) -> dict[str, int]:
        # refused before the data worker decodes anything; process_prompt takes only these
        return {"image": MAX_KEYFRAMES, "audio": 0, "video": 0}

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """A keyframe as decoded RGB, uint8 ``[3, H, W]``: ``process_prompt`` puts it on the
        canvas with PIL, as the reference does, so no resampling happens here."""
        from PIL import Image

        check_image_size(filepath, formats=PILLOW_IMAGE_FORMATS)
        try:
            with Image.open(filepath, formats=PILLOW_IMAGE_FORMATS) as raw:
                image = raw.convert("RGB")
        except OSError as exc:
            raise ValueError("could not decode image: the file is corrupt or truncated") from exc
        return TensorAndMetadata(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous())

    def tokenize(self, prompt: str) -> torch.Tensor:
        return torch.tensor(self._prompt_ids(prompt), dtype=torch.long)

    def _prompt_ids(self, prompt: str) -> list[int]:
        """The prompt's token ids, refusing Qwen's vision tokens: keyframe features are
        spliced in at image pads, so a typed one would fail the forward."""
        tokenizer = self._ensure_tokenizer()
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        reserved = {tokenizer.token_to_id(token) for token in VISION_TOKENS} - {None}
        if reserved.intersection(ids):
            raise ValueError(f"MiniMax-H3 prompts can't contain Qwen's vision tokens {list(VISION_TOKENS)}")
        return ids

    def _check_prompt_length(self, tokens: int) -> None:
        limit = self.max_prompt_tokens or self.config.text_max_positions
        if tokens > limit:
            raise ValueError(
                f"MiniMax-H3 prompt is {tokens} tokens (keyframe tokens included), above "
                + (f"this server's max_prompt_tokens of {limit}" if self.max_prompt_tokens
                   else f"the text encoder's {limit} positions")
            )

    def resolve_request(self, model_kwargs: dict, text_rows: int = 0, keyframes: int = 0) -> dict:
        """The request's geometry and schedule, validated, and its packed rows (with
        ``text_rows`` prompt rows and ``keyframes`` keyframes) under ``max_packed_rows``."""
        integer = functools.partial(_positive_int, model_kwargs)
        height, width = integer("height", None), integer("width", None)
        if (height is None) != (width is None):
            raise ValueError("MiniMax-H3 height and width have to be given together, or neither")
        if height is None:
            height, width = resolve_canvas_size(16, 9, CANVAS_MULTIPLE)
        for name, value in (("height", height), ("width", width)):
            if value % CANVAS_MULTIPLE:
                raise ValueError(f"MiniMax-H3 {name}={value} must be a multiple of {CANVAS_MULTIPLE}")
        if not 1 / 4 <= width / height <= 4:
            raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {width}x{height}")
        requested = integer("num_frames", DEFAULT_NUM_FRAMES)
        vae = self.config.video_vae
        num_frames = align_num_frames(requested, vae.clip_length, vae.latents_per_chunk)
        duration = num_frames / FPS
        if not MIN_DURATION <= duration <= MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates {MIN_DURATION:g} to {MAX_DURATION:g} s at {FPS} fps: num_frames {requested} "
                f"(snapped up to {num_frames}, the 17n+5 the VAE decodes) is {duration:.2f} s"
            )
        steps, video_shift, audio_shift = self.resolve_schedule(model_kwargs)
        request = {
            "height": height, "width": width, "num_frames": num_frames, "num_inference_steps": steps,
            "video_shift": video_shift, "audio_shift": audio_shift,
        }
        rows = self.request_layout(request, text_rows, ANCHORS[:keyframes]).total_rows
        if rows > self.max_packed_rows:
            raise ValueError(
                f"MiniMax-H3 {width}x{height}x{num_frames} with a {text_rows}-token prompt and {keyframes} "
                f"keyframe(s) packs {rows} rows, above this server's max_packed_rows of {self.max_packed_rows}"
            )
        return request

    def resolve_schedule(self, model_kwargs: dict) -> ScheduleArgs:
        steps = _positive_int(model_kwargs, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        if not 2 <= steps <= MAX_INFERENCE_STEPS + 1:
            raise ValueError(f"MiniMax-H3 num_inference_steps must be in [2, {MAX_INFERENCE_STEPS + 1}]; got {steps}")
        video_shift = _positive_float(model_kwargs, "flow_shift", self.config.video_shift)
        audio_shift = _positive_float(model_kwargs, "audio_flow_shift", self.config.audio_shift)
        h3_schedule(steps, video_shift, audio_shift)  # a grid the two schedules can't share is a 400
        return ScheduleArgs(steps, video_shift, audio_shift)

    @staticmethod
    def _generate_walk_for(output_modalities: list[str] | None, keyframes: bool = False) -> str:
        wanted = frozenset(output_modalities or ["video", "audio"])
        walk = GENERATE_WALKS.get(wanted)
        if walk is None:
            raise ValueError(f"MiniMax-H3 generates video and/or audio; got output modalities {sorted(wanted)}")
        return walk + KEYFRAMES_SUFFIX if keyframes else walk

    @staticmethod
    def keyframe_anchors(model_kwargs: dict, count: int) -> tuple[str, ...]:
        """Which end of the video each image input anchors, positional with the images:
        ``keyframe_anchors``, by default ``first`` (and ``last`` for a second image)."""
        anchors = model_kwargs.get("keyframe_anchors")
        if anchors is None:
            anchors = ANCHORS[:count]
        anchors = (anchors,) if isinstance(anchors, str) else anchors
        if not isinstance(anchors, (list, tuple)) or not all(isinstance(a, str) for a in anchors):
            raise ValueError(f"MiniMax-H3 keyframe_anchors must be a list of {ANCHORS}; got {anchors!r}")
        anchors = tuple(anchors)
        if len(anchors) != count or len(set(anchors)) != count or not set(anchors) <= set(ANCHORS):
            raise ValueError(
                f"MiniMax-H3 keyframe_anchors must name {count} distinct of {ANCHORS} for {count} image(s); "
                f"got {list(anchors)}"
            )
        return anchors

    def keyframe_tokens(self, height: int, width: int) -> int:
        """One keyframe's merged vision tokens: 16-px patches merged 2x2, on a canvas Qwen's
        processor keeps as is (a multiple of 32)."""
        return (height // 32) * (width // 32)

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        """Validate the request at the 400-producing seam; tokenize the prompt, and with
        keyframes put them on the canvas and lay out the prompt around them."""
        if any(m not in ("text", "image") for m in input_modalities or []):
            raise ValueError(f"MiniMax-H3 takes a text prompt and up to two keyframes; got inputs {input_modalities}")
        if prompt is None or not prompt.strip():
            raise ValueError("MiniMax-H3 requires a non-empty text prompt")
        images = list((tensors or {}).get("image_inputs") or [])
        if "image" in (input_modalities or []) and not images:
            raise ValueError("request declared an image input but no image arrived")
        self._generate_walk_for(output_modalities, keyframes=bool(images))
        prompt_ids = self._prompt_ids(prompt)
        if not images:
            self._check_prompt_length(len(prompt_ids))
            self.resolve_request(kwargs, text_rows=len(prompt_ids))
            return {TEXT_INPUTS: [torch.tensor(prompt_ids, dtype=torch.long)]}
        if len(images) > MAX_KEYFRAMES:
            raise ValueError(f"MiniMax-H3 takes at most {MAX_KEYFRAMES} keyframes (first and last); got {len(images)}")
        from PIL import Image

        anchors = self.keyframe_anchors(kwargs, len(images))
        if kwargs.get("height") is not None or kwargs.get("width") is not None:
            # a size the server would refuse is a 400 before any keyframe is resized to it
            self.resolve_request(kwargs, text_rows=len(prompt_ids), keyframes=len(images))
        height, width = self._requested_canvas(kwargs)
        pil = {anchor: Image.fromarray(img.permute(1, 2, 0).contiguous().cpu().numpy())
               for anchor, img in zip(anchors, images, strict=True)}
        # packed first, then last, whatever order the images came in
        fitted = fit_keyframes(pil.get("first"), pil.get("last"), height, width)
        tokens = [self.keyframe_tokens(fitted.height, fitted.width)] * len(images)
        ids = presentation(self._ensure_tokenizer(), prompt, tokens).token_ids
        self._check_prompt_length(len(ids))
        self.resolve_request({**kwargs, "height": fitted.height, "width": fitted.width}, text_rows=len(ids),
                             keyframes=len(images))
        vision = self.image_processor(images=fitted.images, return_tensors="pt")
        grids = vision["image_grid_thw"].tolist()
        if [int(t * h * w) // 4 for t, h, w in grids] != tokens:
            raise ValueError(f"unexpected vision grids {grids} for a {fitted.width}x{fitted.height} canvas")
        pixels = torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1) for image in fitted.images])
        return {
            TEXT_INPUTS: [torch.tensor(ids, dtype=torch.long)],
            VISION_PIXELS: [vision["pixel_values"].to(torch.float32)],
            KEYFRAME_PIXELS: [pixels.contiguous()],
            # the decoded uploads are spent (no node reads them); the store would hold
            # them for the whole request
            "image_inputs": [],
        }

    @staticmethod
    def _requested_canvas(model_kwargs: dict) -> tuple[int | None, int | None]:
        height, width = model_kwargs.get("height"), model_kwargs.get("width")
        return (None, None) if height is None and width is None else (int(height), int(width))

    def request_kwargs(self) -> frozenset[str]:
        # the adapter's ``audio`` and ``last_image`` become output modalities and an image input
        return frozenset({
            "height", "width", "num_frames", "num_inference_steps", "flow_shift", "audio_flow_shift",
            "keyframe_anchors",
        })

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {DENOISE_LOOP: h3_schedule(*self.resolve_schedule(model_kwargs)).num_steps}

    # --------------------------------------------------------------- plan
    def request_layout(self, request: dict, text_len: int, anchors: tuple[str, ...] = ()) -> H3Layout:
        vae = self.config.video_vae
        return H3Layout(
            text_len=text_len,
            latent_frames=video_latent_num_frames(request["num_frames"], vae.clip_length, vae.latents_per_chunk),
            latent_height=request["height"] // vae.spatial_compression,
            latent_width=request["width"] // vae.spatial_compression,
            audio_latents=audio_latent_num_frames(request["num_frames"]),
            anchors=anchors,
        )

    def walk_plan(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict,
    ) -> list[WalkStep]:
        """Encode the prompt (and any keyframes), then denoise; both steps carry the
        request's resolved facts, which every node reads its layout from."""
        # Backstops: process_prompt rejected these already, where a raise is a 400.
        if not input_signals.get(TEXT_INPUTS):
            raise ValueError("MiniMax-H3 requires a text prompt (text_inputs)")
        model_kwargs = model_kwargs or {}
        keyframes = input_signals.get(KEYFRAME_PIXELS)
        anchors: tuple[str, ...] = ()
        if keyframes:
            # the canvas the keyframes were put on (process_prompt)
            count, _, height, width = (int(d) for d in keyframes[0].dims)
            model_kwargs = {**model_kwargs, "height": height, "width": width}
            anchors = tuple(a for a in ANCHORS if a in self.keyframe_anchors(model_kwargs, count))
        text_len = int(input_signals[TEXT_INPUTS][0].dims[0])
        request = self.resolve_request(model_kwargs, text_rows=text_len, keyframes=len(anchors))
        layout = self.request_layout(request, text_len)
        request.update(
            text_len=text_len, latent_frames=layout.latent_frames, latent_height=layout.latent_height,
            latent_width=layout.latent_width, audio_latents=layout.audio_latents, anchors=list(anchors),
        )
        if anchors:
            tokens = self.keyframe_tokens(request["height"], request["width"])
            request.update(
                text_video_spans=[
                    list(span) for span in vision_spans(self._ensure_tokenizer(), [tokens] * len(anchors))
                ],
                keyframe_grids=[[1, request["height"] // 16, request["width"] // 16]] * len(anchors),
            )
        final_walk = self._generate_walk_for(output_modalities, keyframes=bool(anchors))
        first_walk = ENCODE_KEYFRAMES_WALK if anchors else ENCODE_TEXT_WALK
        logger.info(
            "MiniMax-H3 request: %dx%d frames=%d steps=%d prompt=%d tokens rows=%d outputs=%s seed=%s",
            request["width"], request["height"], request["num_frames"], request["num_inference_steps"], text_len,
            layout.total_rows, output_modalities, model_kwargs.get("seed", "auto"),
        )
        return [WalkStep(first_walk, step_metadata=request), WalkStep(final_walk, step_metadata=request)]

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """``video``: uint8 ``[F, H, W, 3]`` -> H.264 mp4 at 24 fps. ``audio``: ``[2, samples]``
        in ``[-1, 1]`` -> headerless interleaved 16-bit PCM (the API layer wraps it with
        ``get_output_sample_rate``)."""
        if modality == "audio":
            x = output[0] if output.ndim == 3 else output
            pcm = (x.detach().to(torch.float32).clamp(-1, 1) * 32767.0).round().to(torch.int16)
            return pcm.T.contiguous().cpu().numpy().tobytes()
        if modality != "video":
            raise ValueError(f"unsupported output modality for MiniMax-H3: {modality!r}")
        frames = (output[0] if output.ndim == 5 else output).cpu().numpy()
        # CRF 18, ultrafast: the encode is on the request's critical path
        return encode_mp4(frames, FPS, options={"crf": "18", "preset": "ultrafast", "threads": "0"})

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.audio_vae.sample_rate

    def get_output_audio_channels(self, modality: str = "audio") -> int:
        return AUDIO_CHANNELS

    def get_output_frame_rate(self, modality: str = "video_frame", request_kwargs: dict | None = None) -> float:
        return float(FPS)

    # ------------------------------------------------------------- loading
    def get_autocast_dtype(self):
        # the checkpoint's dtypes: bf16, with the reference's fp32 modules and step math
        return None

    def get_default_sharding_config(self) -> ShardingConfig:
        # The dit shards its rows by rank (its sequence shard). No tensor parallelism: the
        # linears support it, but no config or test runs it.
        return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})

    def get_submodule(self, node_name: str, device="cpu", tp_group=None, autocast_dtype=None):
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, tp_group)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("loaded MiniMax-H3 submodule for node %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device, tp_group=None) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        from mstar.model.minimax_h3 import weight_loader

        snapshot, config = self.snapshot, self.config
        if node_name == TEXT_ENCODER_NODE:
            return H3TextEncoderSubmodule(
                weight_loader.build_text_encoder(config, snapshot, device),
                weight_loader.build_text_conditioner(config, snapshot, device),
                vision=weight_loader.build_vision_tower(snapshot, device), config=config,
            )
        if node_name == VAE_ENCODER_NODE:
            return H3VaeEncoderSubmodule(weight_loader.build_keyframe_encoder(config, snapshot, device))
        if node_name == DIT_NODE:
            return H3DenoiseSubmodule(
                weight_loader.build_transformer(config, snapshot, device, comm_group=tp_group),
                config, loop_name=DENOISE_LOOP,
                max_batch_size=1, compile_blocks=self.compile, row_bucket=self.row_bucket,
            )
        if node_name == VAE_DECODER_NODE:
            return H3VideoDecoderSubmodule(weight_loader.build_video_decoder(config, snapshot, device), config,
                                           tile_batch=self.vae_tile_batch, compile_blocks=self.vae_compile)
        if node_name == AUDIO_DECODER_NODE:
            return H3AudioDecoderSubmodule(weight_loader.build_audio_vae(config, snapshot, device), config)
        logger.warning("MiniMax-H3 has no submodule for node %r; running it dummy", node_name)
        return None
