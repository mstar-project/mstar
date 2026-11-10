"""Wan22Model: Wan2.2-TI2V-5B text/image-to-video generation.

Architecture (4 nodes; nothing here is cached, the dit's self-attention reads an array with max_len 0):
    text_encoder - UMT5-XXL prompt encoder (thin HF wrapper)
    vae_encoder  - Wan2.2-VAE encode of the conditioning frame (I2V)
    dit          - dense 5B video DiT; CFG as one batch-2 forward,
                   UniPC (bh2, order 2, flow prediction) run inline
    vae_decoder  - Wan2.2-VAE latent -> pixel decode

Graph walks (4):
    encode_text   - text_encoder; persists text_embeds_pos / text_embeds_neg
    encode_image  - vae_encoder; persists image_latent (I2V requests only)
    video_gen     - T2V: Loop("denoise_loop") over dit, then vae_decoder ->
                    EMIT_TO_CLIENT (modality "video")
    video_gen_i2v - I2V: same topology; dit additionally consumes the
                    persisted image_latent for first-frame injection

The walk plan (``walk_plan``) is fixed up front from the input modalities:

    T2V: encode_text -> video_gen
    I2V: encode_text -> encode_image -> video_gen_i2v

The denoise loop is dynamic: ``Loop.max_iters`` is the config ceiling, and the
request's ``num_inference_steps`` is the loop's cap (``get_loop_caps``).
"""

import html
import logging
import math
import os
import re
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.distributed.base import ShardingConfig
from mstar.engine.state import Array, Attend, Block, Read
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import Model, WalkStep
from mstar.model.components.diffusion.attention import dit_entry, refuse_attention_backend_kwarg
from mstar.model.components.video_encode import encode_mp4
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.wan22.config import DIT_TOKENS, SELF_ATTN, WAN22_VARIANT_TI2V_5B, Wan22Config
from mstar.model.wan22.submodules import (
    DENOISE_LOOP_NAME,
    Wan22DitSubmodule,
    Wan22TextEncoderSubmodule,
    Wan22VaeDecoderSubmodule,
    Wan22VaeEncoderSubmodule,
    normalize_decode_tiling_mode,
)

logger = logging.getLogger(__name__)

# libx264, which postprocess encodes with, refuses a frame wider or taller than this
_MP4_MAX_SIDE = 16384

# Every model_kwargs key a request may carry; the video adapter maps ``size``
# onto height/width and forwards num_frames and fps.
REQUEST_KWARGS = frozenset({
    "height", "width", "num_frames", "fps", "num_inference_steps", "guidance_scale", "negative_prompt",
})


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class Wan22Model(Model):
    """Wan2.2-TI2V-5B video generation model (TI2V-5B variant only)."""

    ENCODE_TEXT_WALK = "encode_text"
    ENCODE_IMAGE_WALK = "encode_image"
    VIDEO_GEN_WALK = "video_gen"
    VIDEO_GEN_I2V_WALK = "video_gen_i2v"

    DENOISE_LOOP_NAME = DENOISE_LOOP_NAME

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        variant: str = WAN22_VARIANT_TI2V_5B,
        compile_dit_sizes: tuple[str, ...] | None = None,
        dit_max_batch_size: int = Wan22Config.dit_max_batch_size,
        max_video_pixels: int = Wan22Config.max_video_pixels,
        skip_weight_loading: bool = False,
        vae_decode_fast: bool = True,
        vae_decode_fast_sizes: tuple[str, ...] | None = None,
        **kwargs,
    ):
        refuse_attention_backend_kwarg(kwargs)
        if variant != WAN22_VARIANT_TI2V_5B:
            raise NotImplementedError(
                f"Wan2.2 variant {variant!r} is not implemented; only "
                f"{WAN22_VARIANT_TI2V_5B!r} is. The A14B variants are a MoE dual-DiT."
            )
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = Wan22Config(
            variant=variant, compile_dit_sizes=compile_dit_sizes,
            dit_max_batch_size=self._require_positive_int("dit_max_batch_size", dit_max_batch_size),
            max_video_pixels=self._require_positive_int("max_video_pixels", max_video_pixels),
            vae_decode_fast=vae_decode_fast, vae_decode_fast_sizes=vae_decode_fast_sizes,
        )
        self.config.compile_dit_grids()  # a malformed list fails at startup
        # Dummy mode: get_submodule returns None for every node, so engines run
        # without weights, GPU or network.
        self.skip_weight_loading = skip_weight_loading

        # Loaded lazily on the data worker, so constructing the model never hits
        # the network. Dummy tests pin tokenizer=None to take the byte fallback.
        self.tokenizer = None
        self._tokenizer_initialized = False

        # A worker only instantiates the nodes it serves. The encoder and decoder
        # own separate VAE instances (see _create_submodule).
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._encode_vae: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def declare_state(self) -> list[Array]:
        """The DiT's self-attention over its tokens; nothing here keeps state.

        Nothing in this model reads K/V back across steps. The text encoder
        and both VAE passes are one-shot forwards, and the DiT recomputes
        its self-attention over the whole latent grid every denoise step —
        the only state that crosses a step is the UniPC solver's, which
        rides the loop-back edges as ordinary tensors rather than a cache.

        Self-attention only: the cross-attention over the 512 text tokens runs
        SDPA (cuDNN), which measured faster than FlashInfer's ragged kernel at
        that shape (0.46 vs 0.59 ms per layer at 1280x704x81 on an H200), while
        FA3 beats it on the self-attention (12.4 vs 14.8 ms).
        """
        c = self.config
        return [Array(
            DIT_TOKENS, nodes={"dit"}, layers=frozenset(range(c.num_layers)), max_len=0, seq_sharded=True,
            entry=dit_entry(c.num_attention_heads, c.attention_head_dim),
            reads={SELF_ATTN: Read(mask=Block(), op=Attend(q_heads=c.num_attention_heads), packed=False)},
        )]

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        # one segment per guidance branch; the DiT is compiled, not captured, so attention stays in its graph (under
        # SP, the fragment between the all-to-all breaks)
        return {SELF_ATTN: {"max_segments_per_request": Wan22DitSubmodule.CFG_BRANCHES, "compile_op": True}}

    def get_default_sharding_config(self) -> ShardingConfig:
        # The dit's two guidance branches split over cfg_size ranks, which
        # all-gather the noise predictions (see Wan22DitSubmodule), and its
        # tokens over sp_size (Ulysses, see Wan22DiT). Signals stay replicated:
        # no shard_dim entries.
        return ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={}, sp_enabled_nodes={"dit"},
            cfg_enabled_nodes={"dit": Wan22DitSubmodule.CFG_BRANCHES},
        )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # -- encode_text: UMT5 forward; both CFG embeddings persist at the
        # -- conductor for the video_gen walk.
        encode_text = GraphNode(
            name="text_encoder",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_pos", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_neg", persist=True),
            ],
        )

        # -- encode_image (I2V only): VAE-encode the conditioning frame; the
        # -- normalized latent persists for the video_gen_i2v walk. The
        # -- first-frame mask is deterministic given the latent dims and is
        # -- rebuilt inside the dit submodule, not persisted.
        encode_image = GraphNode(
            name="vae_encoder",
            input_names=["image_inputs"],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="image_latent", persist=True),
            ],
        )

        return {
            self.ENCODE_TEXT_WALK: encode_text,
            self.ENCODE_IMAGE_WALK: encode_image,
            self.VIDEO_GEN_WALK: self._build_video_gen_walk(i2v=False),
            self.VIDEO_GEN_I2V_WALK: self._build_video_gen_walk(i2v=True),
        }

    def _build_video_gen_walk(self, i2v: bool) -> GraphSection:
        """Denoise loop + VAE decode.

        T2V and I2V are separate walks over fresh GraphNode instances, so the dit's
        ``input_names`` are exact per mode rather than carrying I2V-only inputs as
        empty edges in T2V.
        """
        dit_inputs = ["text_embeds_pos", "text_embeds_neg"]
        if i2v:
            dit_inputs.append("image_latent")
        # Loop-carried state: see Wan22DitSubmodule's docstring for the
        # definitive shape/dtype inventory (latents, the order-2 UniPC ring
        # buffer, and the corrector's last_sample).
        dit_inputs += [
            "latents",
            "unipc_model_outputs",
            "unipc_last_sample",
        ]

        denoise_loop = Loop(
            name=DENOISE_LOOP_NAME,
            section=GraphNode(
                name="dit",
                input_names=dit_inputs,
                outputs=[
                    GraphEdge(next_node="dit", name="latents"),
                    GraphEdge(next_node="dit", name="unipc_model_outputs"),
                    GraphEdge(next_node="dit", name="unipc_last_sample"),
                ],
                # the loop's cap keeps speculation off a request's last step
                enable_async_scheduling=True,
            ),
            # Ceiling only; the request's step count is the loop's cap.
            max_iters=self.config.max_denoise_steps,
            outputs=[
                # Matched by name against the section's loop-back edges: the final
                # iteration's latents route onward to the VAE decoder.
                GraphEdge(next_node="vae_decoder", name="latents"),
            ],
        )

        vae_decoder = GraphNode(
            name="vae_decoder",
            input_names=["latents"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="video_output",
                    output_modality="video",
                ),
            ],
        )
        return Sequential([denoise_loop, vae_decoder])

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def _ensure_tokenizer(self):
        """Load the UMT5 tokenizer on first use (data-worker side)."""
        if self._tokenizer_initialized:
            return
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path_hf, subfolder="tokenizer", cache_dir=self.cache_dir,
        )
        self._tokenizer_initialized = True

    @staticmethod
    def _prompt_clean(text: str) -> str:
        """diffusers ``prompt_clean`` minus ftfy: double html-unescape + strip,
        then whitespace collapse. The reference additionally runs
        ``ftfy.fix_text`` first, which is the identity on well-formed ASCII
        (all equivalence prompts) — mstar deliberately skips the ftfy
        dependency, so mojibake in non-ASCII prompts is the one tokenization
        divergence from the reference pipeline.
        """
        text = html.unescape(html.unescape(text)).strip()
        return re.sub(r"\s+", " ", text).strip()

    def _encode_text(self, text: str) -> torch.Tensor:
        """Tokenize one prompt string, falling back to raw bytes when no
        tokenizer is available (dummy-model test path)."""
        text = self._prompt_clean(text)
        if self.tokenizer is not None:
            ids = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.text_max_seq_len,
                add_special_tokens=True,
            ).input_ids
            return torch.tensor(ids, dtype=torch.long)
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)

    @staticmethod
    def _require_positive_int(name: str, raw) -> int:
        """A request geometry knob, which must be a positive JSON integer, or ``ValueError``."""
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"Wan2.2 {name} must be an integer; got {raw!r}.")
        value = raw
        if value <= 0:
            raise ValueError(f"Wan2.2 {name} must be positive; got {value}.")
        return value

    def _validate_generation_size(self, model_kwargs: dict) -> None:
        """Reject a request geometry the model cannot produce. Raises ``ValueError``.

        height and width must be multiples of 32 (VAE downsample 16 x DiT patch 2),
        num_frames must be 4k+1 (the VAE compresses time by 4 around an anchor
        frame), height x width x num_frames must not exceed ``max_video_pixels``, each
        axis must fit the DiT's rotary table and each side the mp4 encoder, and fps
        must be positive.

        This runs in process_prompt, so a bad request is a 400 before any work.
        Left to fail later, none of these fail well:
        an unaligned size hits the DiT as a mid-forward shape mismatch and faults
        the WORKER; a non-positive num_frames makes the latent time extent <= 0 and
        crashes on a negative tensor dim; an unaligned num_frames does not fail at
        all but is silently floored (ask for 32 frames, get 29); and a bad fps is
        only caught after the whole video is generated, on the result path.
        """
        cfg = self.config
        size = {}
        for name, patch, align, default in (
            ("height", cfg.patch_size[1], cfg.spatial_alignment[0], cfg.default_height),
            ("width", cfg.patch_size[2], cfg.spatial_alignment[1], cfg.default_width),
        ):
            value = size[name] = self._require_positive_int(name, self._get(model_kwargs, name, default))
            if value % align:
                lower = value // align * align
                upper = lower + align
                nearest = f"{lower} or {upper}" if lower else str(upper)
                raise ValueError(
                    f"Wan2.2 {name}={value} is not a multiple of {align} "
                    f"(VAE spatial factor {cfg.vae_scale_factor_spatial} x DiT patch {patch}), "
                    f"which this model requires for both height and width. "
                    f"Nearest valid {name}: {nearest}."
                )

        temporal = cfg.vae_scale_factor_temporal
        frames = self._require_positive_int(
            "num_frames", self._get(model_kwargs, "num_frames", cfg.default_num_frames)
        )
        if (frames - 1) % temporal:
            lower = (frames - 1) // temporal * temporal + 1
            upper = lower + temporal
            raise ValueError(
                f"Wan2.2 num_frames={frames} is not of the form {temporal}k+1 "
                f"(the VAE compresses time by {temporal} around an anchor frame). "
                f"An unaligned count is silently floored rather than honored. "
                f"Nearest valid num_frames: {lower} or {upper}."
            )

        height, width = size["height"], size["width"]
        if height * width * frames > cfg.max_video_pixels:
            raise ValueError(
                f"Wan2.2 height x width x num_frames = {height}x{width}x{frames} = "
                f"{height * width * frames} exceeds this server's max_video_pixels={cfg.max_video_pixels}."
            )
        # One rotary position per post-patch cell on each axis.
        rope = cfg.rope_max_seq_len
        max_h, max_w = (rope * a for a in cfg.spatial_alignment)
        max_frames = (rope - 1) * temporal + 1
        if height > max_h or width > max_w or frames > max_frames:
            raise ValueError(
                f"Wan2.2 supports at most height {max_h}, width {max_w} and num_frames "
                f"{max_frames} (the DiT's {rope} rotary positions per axis); got "
                f"{height}x{width}x{frames}."
            )
        # postprocess encodes the clip with libx264, which opens no stream past this
        # side; checked here, not after the video is generated
        if max(height, width) > _MP4_MAX_SIDE:
            raise ValueError(
                f"Wan2.2 supports at most {_MP4_MAX_SIDE} pixels per side (the mp4 encoder's "
                f"limit); got {height}x{width}."
            )

        # An explicit ``fps: null`` means "unset", same as an absent key.
        raw_fps = model_kwargs.get("fps")
        if raw_fps is not None:
            if isinstance(raw_fps, bool) or not isinstance(raw_fps, (int, float)):
                raise ValueError(f"Wan2.2 fps must be a number; got {raw_fps!r}.")
            if not math.isfinite(raw_fps) or raw_fps <= 0:
                raise ValueError(f"Wan2.2 fps must be positive and finite; got {raw_fps!r}.")

    @staticmethod
    def _get(model_kwargs: dict, name: str, default):
        """``model_kwargs[name]``, with null the same as absent."""
        value = model_kwargs.get(name)
        return default if value is None else value

    def _resolve_denoise_knobs(self, model_kwargs: dict) -> dict:
        """num_inference_steps, guidance_scale and negative_prompt, validated; ``ValueError`` names the bad one."""
        cfg = self.config
        steps = self._get(model_kwargs, "num_inference_steps", cfg.default_num_inference_steps)
        # past the denoise loop's max_iters, the conductor refuses it (get_loop_caps)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"Wan2.2 num_inference_steps must be an integer >= 1; got {steps!r}.")
        guidance = self._get(model_kwargs, "guidance_scale", cfg.guidance_scale)
        if not _is_number(guidance):
            raise ValueError(f"Wan2.2 guidance_scale must be a finite number; got {guidance!r}.")
        negative = self._get(model_kwargs, "negative_prompt", cfg.default_negative_prompt)
        if not isinstance(negative, str):
            raise ValueError(f"Wan2.2 negative_prompt must be a string; got {negative!r}.")
        return {"num_inference_steps": steps, "guidance_scale": float(guidance), "negative_prompt": negative}

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Validate the request, then tokenize the prompts on the data worker.

        Returns ``text_inputs`` = ``[positive_ids, negative_ids]``, truncated to
        ``text_max_seq_len``; padding and masks are the text encoder's job. The
        I2V image rides through as ``image_inputs`` for the vae_encoder to
        preprocess, but its size is checked here.

        Every malformed-request check runs here, before tokenizing, so a bad
        request is a 400 without any work; ``walk_plan`` repeats the cheap ones
        as backstops.
        """
        self._validate_request(prompt, input_modalities, output_modalities, tensors)
        self._validate_generation_size(kwargs)
        self._validate_conditioning_image(input_modalities, tensors, kwargs)
        negative_prompt = self._resolve_denoise_knobs(kwargs)["negative_prompt"]
        self._ensure_tokenizer()
        return {
            "text_inputs": [
                self._encode_text(prompt),
                self._encode_text(negative_prompt),
            ]
        }

    def _validate_request(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None,
    ) -> None:
        """Reject a malformed request at the 400-producing seam. Raises ``ValueError``."""
        target_output = output_modalities[0] if output_modalities else "video"
        if target_output != "video":
            raise ValueError(
                f"Wan2.2 only generates video; got output modality {target_output!r}."
            )
        if prompt is None:
            # Without a prompt there is no text_inputs edge, so the encode_text
            # node could never become ready and the request would stall forever.
            raise ValueError(
                "Wan2.2 requires a text prompt for both text-to-video and "
                "image-to-video requests."
            )
        if "image" in input_modalities and not (tensors or {}).get("image_inputs"):
            # A declared-but-missing input is a malformed request, not a mode
            # preference — do not silently downgrade an I2V request to T2V.
            raise ValueError(
                "Request declared an image input but no image arrived; "
                "image-to-video requires the conditioning image."
            )

    def _validate_conditioning_image(
        self, input_modalities: list[str], tensors: NameToTensorList | None, model_kwargs: dict,
    ) -> None:
        """Reject an I2V image whose size isn't the request's. Raises ``ValueError``.

        Runs at this seam like the other geometry checks, so a size mismatch is a 400
        rather than a shape fault on the vae_encoder worker. The server does not
        resize; the image must already be the request's height x width (validated
        just above, so it is known aligned here).
        """
        if "image" not in input_modalities:
            return
        images = (tensors or {}).get("image_inputs")
        if not images:
            return  # presence is _validate_request's job
        expected = (
            self._get(model_kwargs, "height", self.config.default_height),
            self._get(model_kwargs, "width", self.config.default_width),
        )
        got = tuple(images[0].shape[-2:])
        if got != expected:
            raise ValueError(
                f"Wan2.2 conditioning image is {got[0]}x{got[1]} (HxW) but the request is "
                f"{expected[0]}x{expected[1]}; resize it client-side (the server does not resize)."
            )

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the loop ends on the request's own step count, and is never speculated past it;
        # past the loop's max_iters the conductor refuses the request
        steps = self._get(model_kwargs, "num_inference_steps", self.config.default_num_inference_steps)
        if isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1:
            return {DENOISE_LOOP_NAME: steps}
        return {}

    def request_kwargs(self) -> frozenset[str]:
        return REQUEST_KWARGS

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """Mux the emitted uint8 ``[F, H, W, 3]`` frames to mp4 bytes.

        PyAV is the encoder because it bundles its own FFmpeg; torchcodec needs
        system FFmpeg libraries. ``fps`` is a PLAYBACK rate, not a generation knob:
        the model always makes a fixed ``num_frames`` clip at an implied 24 fps, so
        another value only rescales its duration.
        """
        if modality != "video":
            raise ValueError(f"Unsupported modality for Wan2.2: {modality!r}")

        # An explicit ``fps: null`` means "unset", same as an absent key.
        raw_fps = (request_kwargs or {}).get("fps")
        if raw_fps is None:
            raw_fps = self.config.video_fps
        try:
            fps = float(raw_fps)
        except (TypeError, ValueError):
            raise ValueError(f"Wan2.2 fps must be a number; got {raw_fps!r}") from None
        if fps <= 0:
            raise ValueError(f"Wan2.2 fps must be positive; got {raw_fps!r}")

        return encode_mp4(output.cpu().numpy(), fps)

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def _log_config_echo(self, model_kwargs: dict, resolved: dict) -> None:
        """Emit one ``key=value`` log line per request at generation start, echoing every
        resolved setting so an operator can read back exactly what ran instead of
        asserting the knobs. ``decode`` is the configured tiling POLICY
        (auto/tiled/untiled, resolved from config + env); the actual per-request
        untiled-vs-tiled decision is logged separately at decode time by the
        vae_decoder. ``compile`` reflects ``config.compile_dit``.
        """
        negative = model_kwargs.get("negative_prompt", self.config.default_negative_prompt) or ""
        raw_fps = model_kwargs.get("fps")
        fps = raw_fps if raw_fps is not None else self.config.video_fps
        seed = model_kwargs.get("seed")
        # Same normalizer the decoder gate uses, so the echoed policy matches the
        # path the decoder will actually resolve (a bogus override echoes "auto").
        decode_policy = normalize_decode_tiling_mode(
            os.environ.get("WAN22_VAE_DECODE_TILING", self.config.vae_decode_tiling)
        )
        logger.info(
            "Wan2.2 request: height=%d width=%d frames=%d steps=%d guidance=%s seed=%s "
            "negative_prompt=%s fps=%s decode=%s compile=%s",
            resolved["height"], resolved["width"], resolved["num_frames"],
            resolved["num_inference_steps"], resolved["guidance_scale"],
            seed if seed is not None else "auto",
            "present" if negative.strip() else "absent", fps, decode_policy,
            "on" if self.config.compile_dit else "off",
        )

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        # Backstops; process_prompt already rejected each of these. A
        # ValueError here is a 400 too.
        target_output = output_modalities[0] if output_modalities else "video"
        if target_output != "video":
            raise ValueError(
                f"Wan2.2 only generates video; got output modality {target_output!r}."
            )
        if not input_signals.get("text_inputs"):
            raise ValueError(
                "Wan2.2 requires a text prompt (text_inputs) for both "
                "text-to-video and image-to-video requests."
            )
        is_i2v = "image" in input_modalities and bool(input_signals.get("image_inputs"))
        if "image" in input_modalities and not is_i2v:
            raise ValueError(
                "Request declared an image input but no image_inputs tensor "
                "arrived; image-to-video requires the conditioning image."
            )
        walks = [self.ENCODE_TEXT_WALK]
        if is_i2v:
            walks.append(self.ENCODE_IMAGE_WALK)
        walks.append(self.VIDEO_GEN_I2V_WALK if is_i2v else self.VIDEO_GEN_WALK)

        self._validate_generation_size(model_kwargs)
        knobs = self._resolve_denoise_knobs(model_kwargs)
        # what the submodules read from ``request_info.step_metadata``
        request = {
            "num_inference_steps": knobs["num_inference_steps"],
            "guidance_scale": knobs["guidance_scale"],
            "height": self._get(model_kwargs, "height", self.config.default_height),
            "width": self._get(model_kwargs, "width", self.config.default_width),
            "num_frames": self._get(model_kwargs, "num_frames", self.config.default_num_frames),
        }
        self._log_config_echo(model_kwargs, request)
        return [WalkStep(walk, step_metadata=request) for walk in walks]

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    # None disables both autocast and the engine's blanket cast, so numerics are
    # governed by the load-time dtypes (bf16 with fp32 islands) rather than by the
    # engine — which is what keeps them equal to the reference pipeline.
    def get_autocast_dtype(self):
        return None

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None, cfg_group=None,
    ) -> torch.nn.Module | None:
        # ``autocast_dtype`` exists for interface parity only: weights load in the
        # checkpoint's own dtypes. ``sp_group`` and ``cfg_group`` reach the dit only.
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, sp_group=sp_group, cfg_group=cfg_group)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded Wan2.2 submodule for node %s", node_name)
        return submodule

    def _create_submodule(
        self, node_name: str, device: str = "cpu", sp_group=None, cfg_group=None,
    ) -> NodeSubmodule | None:
        """Construct one node's submodule from the checkpoint.

        The wrapped components load on CPU and the EngineManager moves them; the
        native DiT materializes straight onto ``device``, with no CPU staging copy
        of the 5B weights. Returns None in dummy mode and for unknown nodes, which
        makes the engine run that node without real computation.
        """
        if self.skip_weight_loading:
            return None
        if node_name == "text_encoder":
            from transformers import UMT5EncoderModel

            text_encoder = UMT5EncoderModel.from_pretrained(
                self.model_path_hf, subfolder="text_encoder",
                torch_dtype=torch.bfloat16, cache_dir=self.cache_dir,
            ).eval()
            return Wan22TextEncoderSubmodule(text_encoder, self.config)
        if node_name == "vae_encoder":
            return Wan22VaeEncoderSubmodule(self._get_encode_vae(), self.config)
        if node_name == "vae_decoder":
            from diffusers import AutoencoderKLWan

            # Its own instance, in the checkpoint dtype: the decode dtype follows
            # cuDNN while the I2V encode stays bf16, and one shared instance would
            # re-cast the whole VAE whenever the two disagree.
            vae = AutoencoderKLWan.from_pretrained(
                self.model_path_hf, subfolder="vae", cache_dir=self.cache_dir,
            ).eval()
            return Wan22VaeDecoderSubmodule(vae, self.config)
        if node_name == "dit":
            from mstar.model.wan22.weight_loader import build_wan22_dit

            self._refresh_checkpoint_defaults()
            transformer = build_wan22_dit(
                self.config, self.model_path_hf, device=device, cache_dir=self.cache_dir,
                sp_group=sp_group,
            )
            return Wan22DitSubmodule(transformer, self.config, cfg_group=cfg_group)
        logger.warning("Wan2.2 has no submodule for node %r; running it dummy.", node_name)
        return None

    def _get_encode_vae(self) -> torch.nn.Module:
        """The vae_encoder's bf16 Wan2.2-VAE instance — the exact loading the
        I2V encode is equivalence-tested against; the decoder owns a separate
        checkpoint-dtype instance (see ``_create_submodule``)."""
        if self._encode_vae is None:
            from diffusers import AutoencoderKLWan

            self._encode_vae = AutoencoderKLWan.from_pretrained(
                self.model_path_hf, subfolder="vae",
                torch_dtype=torch.bfloat16, cache_dir=self.cache_dir,
            ).eval()
        return self._encode_vae

    def _refresh_checkpoint_defaults(self):
        """Re-read the scheduler defaults from the checkpoint at weight-load time.

        ``flow_shift`` follows the checkpoint. ``solver_order`` and
        ``num_train_timesteps`` are hardcoded into the inline port's math and
        ring-buffer layout, so a checkpoint that drifts on those cannot be honored
        and fails loudly rather than serving wrong solver math.
        """
        from diffusers import UniPCMultistepScheduler

        from mstar.model.components.diffusion.unipc import NUM_TRAIN_TIMESTEPS, SOLVER_ORDER

        scheduler_config = UniPCMultistepScheduler.load_config(
            self.model_path_hf, subfolder="scheduler", cache_dir=self.cache_dir,
        )
        for key, ported in [("solver_order", SOLVER_ORDER),
                            ("num_train_timesteps", NUM_TRAIN_TIMESTEPS)]:
            checkpoint_value = scheduler_config.get(key, ported)
            if checkpoint_value != ported:
                raise ValueError(
                    f"Checkpoint scheduler {key}={checkpoint_value} but the inline "
                    f"UniPC port is built for {ported}; refusing to serve with "
                    "silently wrong solver math."
                )
        checkpoint_shift = scheduler_config.get("flow_shift")
        if checkpoint_shift is not None and checkpoint_shift != self.config.flow_shift:
            logger.warning(
                "Wan2.2 checkpoint scheduler flow_shift=%s differs from the documented "
                "default %s; using the checkpoint value.",
                checkpoint_shift, self.config.flow_shift,
            )
            self.config.flow_shift = checkpoint_shift
