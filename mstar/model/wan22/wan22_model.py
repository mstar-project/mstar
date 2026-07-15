"""Wan22Model: Wan2.2-TI2V-5B text/image-to-video generation.

Architecture (4 nodes, all STATELESS — no KV cache anywhere):
    text_encoder (enc_dec) - UMT5-XXL prompt encoder (thin HF wrapper)
    vae_encoder  (enc_dec) - Wan2.2-VAE encode of the conditioning frame (I2V)
    dit          (enc_dec) - dense 5B video DiT; CFG as one batch-2 forward,
                             UniPC (bh2, order 2, flow prediction) run inline
    vae_decoder  (enc_dec) - Wan2.2-VAE latent -> pixel decode

Graph walks (4):
    encode_text   - text_encoder; persists text_embeds_pos / text_embeds_neg
    encode_image  - vae_encoder; persists image_latent (I2V requests only)
    video_gen     - T2V: Loop("denoise_loop") over dit, then vae_decoder ->
                    EMIT_TO_CLIENT (modality "video")
    video_gen_i2v - I2V: same topology; dit additionally consumes the
                    persisted image_latent for first-frame injection

The walk sequence is fixed up front from the input modalities and stepped through
``metadata.kwargs["walk_step"]``:

    T2V: encode_text -> video_gen
    I2V: encode_text -> encode_image -> video_gen_i2v

The denoise loop is dynamic: ``Loop.max_iters`` is the config ceiling, and the
request's ``num_inference_steps`` stops it via ``Wan22DitSubmodule.check_stop``.
"""

import html
import io
import logging
import os
import re
from fractions import Fraction

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.wan22.config import WAN22_VARIANT_TI2V_5B, Wan22Config
from mstar.model.wan22.submodules import (
    DENOISE_LOOP_NAME,
    Wan22DitSubmodule,
    Wan22TextEncoderSubmodule,
    Wan22VaeDecoderSubmodule,
    Wan22VaeEncoderSubmodule,
    normalize_decode_tiling_mode,
)

logger = logging.getLogger(__name__)


class Wan22Model(Model):
    """Wan2.2-TI2V-5B video generation model (TI2V-5B variant only)."""

    ENCODE_TEXT_WALK = "encode_text"
    ENCODE_IMAGE_WALK = "encode_image"
    VIDEO_GEN_WALK = "video_gen"
    VIDEO_GEN_I2V_WALK = "video_gen_i2v"

    # Denoise loop name — referenced by ``Wan22DitSubmodule.check_stop`` via
    # ``request_info.dynamic_loop_iter_counts[...]``.
    DENOISE_LOOP_NAME = DENOISE_LOOP_NAME

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        variant: str = WAN22_VARIANT_TI2V_5B,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        if variant != WAN22_VARIANT_TI2V_5B:
            raise NotImplementedError(
                f"Wan2.2 variant {variant!r} is not implemented; only "
                f"{WAN22_VARIANT_TI2V_5B!r} is. The A14B variants are a MoE dual-DiT."
            )
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = Wan22Config(variant=variant)
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

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return []  # all nodes are stateless; no KV cache

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "text_encoder": EngineType.STATELESS,
            "vae_encoder": EngineType.STATELESS,
            "dit": EngineType.STATELESS,
            "vae_decoder": EngineType.STATELESS,
        }

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
        # definitive shape/dtype inventory (latents, 0-based step index, the
        # order-2 UniPC ring buffer, and the corrector's last_sample).
        dit_inputs += [
            "latents",
            "time_index",
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
                    GraphEdge(next_node="dit", name="time_index"),
                    GraphEdge(next_node="dit", name="unipc_model_outputs"),
                    GraphEdge(next_node="dit", name="unipc_last_sample"),
                ],
                # Don't let async scheduling overshoot check_stop by an iteration.
                enable_async_scheduling=False,
            ),
            # Ceiling only; the request's step count stops the loop early.
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
        """Coerce a request geometry knob, or raise ``ValueError``."""
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Wan2.2 {name} must be an integer; got {raw!r}.") from None
        if value <= 0:
            raise ValueError(f"Wan2.2 {name} must be positive; got {value}.")
        return value

    def _validate_generation_size(self, model_kwargs: dict) -> None:
        """Reject a request geometry the model cannot produce. Raises ``ValueError``.

        height and width must be multiples of 32 (VAE downsample 16 x DiT patch 2),
        num_frames must be 4k+1 (the VAE compresses time by 4 around an anchor
        frame), and fps must be positive.

        This runs on the data worker because that is where a ValueError becomes a
        400 for the one bad request. Left to fail later, none of these fail well:
        an unaligned size hits the DiT as a mid-forward shape mismatch and faults
        the WORKER; a non-positive num_frames makes the latent time extent <= 0 and
        crashes on a negative tensor dim; an unaligned num_frames does not fail at
        all but is silently floored (ask for 32 frames, get 29); and a bad fps is
        only caught after the whole video is generated, on the result path, where
        the raise is swallowed and the client hangs.
        """
        cfg = self.config
        for name, patch, align, default in (
            ("height", cfg.patch_size[1], cfg.spatial_alignment[0], cfg.default_height),
            ("width", cfg.patch_size[2], cfg.spatial_alignment[1], cfg.default_width),
        ):
            value = self._require_positive_int(name, model_kwargs.get(name, default))
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
            "num_frames", model_kwargs.get("num_frames", cfg.default_num_frames)
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

        # An explicit ``fps: null`` means "unset", same as an absent key.
        raw_fps = model_kwargs.get("fps")
        if raw_fps is not None:
            try:
                fps = float(raw_fps)
            except (TypeError, ValueError):
                raise ValueError(f"Wan2.2 fps must be a number; got {raw_fps!r}.") from None
            if fps <= 0:
                raise ValueError(f"Wan2.2 fps must be positive; got {raw_fps!r}.")

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

        Every malformed-request check lives here rather than in
        ``get_initial_forward_pass_args``, even where that also guards: a
        ValueError here becomes a 400, while the same raise at the conductor is
        swallowed and the client hangs.
        """
        self._validate_request(prompt, input_modalities, output_modalities, tensors)
        self._validate_generation_size(kwargs)
        self._validate_conditioning_image(input_modalities, tensors, kwargs)
        negative_prompt = kwargs.get("negative_prompt", self.config.default_negative_prompt)
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
            int(model_kwargs.get("height", self.config.default_height)),
            int(model_kwargs.get("width", self.config.default_width)),
        )
        got = tuple(images[0].shape[-2:])
        if got != expected:
            raise ValueError(
                f"Wan2.2 conditioning image is {got[0]}x{got[1]} (HxW) but the request is "
                f"{expected[0]}x{expected[1]}; resize it client-side (the server does not resize)."
            )

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """Mux the emitted uint8 ``[1, 3, F, H, W]`` video tensor to mp4 bytes.

        PyAV is the encoder because it bundles its own FFmpeg; torchcodec needs
        system FFmpeg libraries. ``fps`` is a PLAYBACK rate, not a generation knob:
        the model always makes a fixed ``num_frames`` clip at an implied 24 fps, so
        another value only rescales its duration.
        """
        if modality != "video":
            raise ValueError(f"Unsupported modality for Wan2.2: {modality!r}")
        import av

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

        frames = output[0].permute(1, 2, 3, 0).cpu().numpy()
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        # PyAV rejects float rates, so the rate is handed over as an exact
        # Fraction: integer rates stay integral (24 -> 24) and fractional ones
        # become a rational the container can carry (23.976 -> 2997/125).
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        stream.height, stream.width = frames.shape[1], frames.shape[2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():  # flush encoder delay
            container.mux(packet)
        container.close()
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def _get_step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        """Per-pass metadata the submodules read from ``request_info.step_metadata``."""
        kw = metadata.kwargs
        return {
            "is_prefill": metadata.is_prefill,
            "num_inference_steps": kw["num_inference_steps"],
            "guidance_scale": kw["guidance_scale"],
            "height": kw["height"],
            "width": kw["width"],
            "num_frames": kw["num_frames"],
        }

    def _log_config_echo(self, model_kwargs: dict, resolved: dict) -> None:
        """Emit one vllm-style line per request at generation start, echoing every
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

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        model_kwargs = model_kwargs or {}
        # These are backstops, not the primary guard. ``process_prompt`` already
        # rejected every one of these on the data worker, where a ValueError
        # becomes a 400; a raise HERE runs at the conductor, whose main loop
        # swallows it, so the client would hang instead. They remain because a
        # malformed request must never reach the graph, but they should be
        # unreachable in practice.
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
        schedule = [self.ENCODE_TEXT_WALK]
        if is_i2v:
            schedule.append(self.ENCODE_IMAGE_WALK)
        schedule.append(self.VIDEO_GEN_I2V_WALK if is_i2v else self.VIDEO_GEN_WALK)

        requested_steps = int(
            model_kwargs.get("num_inference_steps", self.config.default_num_inference_steps)
        )
        num_inference_steps = max(1, min(requested_steps, self.config.max_denoise_steps))
        if num_inference_steps != requested_steps:
            logger.info(
                "Clamped num_inference_steps from %d to %d (max_denoise_steps=%d)",
                requested_steps, num_inference_steps, self.config.max_denoise_steps,
            )

        kwargs = {
            "walk_schedule": schedule,
            "walk_step": 0,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": float(model_kwargs.get("guidance_scale", self.config.guidance_scale)),
            "height": int(model_kwargs.get("height", self.config.default_height)),
            "width": int(model_kwargs.get("width", self.config.default_width)),
            "num_frames": int(model_kwargs.get("num_frames", self.config.default_num_frames)),
        }
        self._log_config_echo(model_kwargs, kwargs)
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=schedule[0],
            is_prefill=True,
            kwargs=kwargs,
        )

        text_edge = GraphEdge(next_node="text_encoder", name="text_inputs")
        text_edge.tensor_info = input_signals["text_inputs"]
        inputs = [text_edge]

        # text_inputs is consumed only by encode_text; image_inputs stays
        # persisted for the encode_image pass (I2V).
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._get_step_metadata(full_metadata),
        )

    def _video_gen_inputs(
        self, walk: str, persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """External inputs seeding the denoise walk.

        The loop-back edges are sent empty; the dit submodule seeds them at
        iteration 0.
        """
        inputs: list[GraphEdge] = []
        persisted = ["text_embeds_pos", "text_embeds_neg"]
        if walk == self.VIDEO_GEN_I2V_WALK:
            persisted.append("image_latent")
        for name in persisted:
            edge = GraphEdge(next_node="dit", name=name)
            edge.tensor_info = persist_signals.get(name, [])
            inputs.append(edge)
        inputs += [
            GraphEdge(next_node="dit", name="latents"),
            GraphEdge(next_node="dit", name="time_index"),
            GraphEdge(next_node="dit", name="unipc_model_outputs"),
            GraphEdge(next_node="dit", name="unipc_last_sample"),
        ]
        return inputs

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Step through the request's fixed walk schedule; done after video_gen."""
        metadata = partition_metadata
        request_done = False
        inputs: list[GraphEdge] = []

        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        if step < len(schedule):
            metadata.kwargs["walk_step"] = step
            walk = schedule[step]
            metadata.graph_walk = walk
            metadata.is_prefill = walk in (self.ENCODE_TEXT_WALK, self.ENCODE_IMAGE_WALK)
            if walk == self.ENCODE_IMAGE_WALK:
                edge = GraphEdge(next_node="vae_encoder", name="image_inputs")
                edge.tensor_info = persist_signals.get("image_inputs", [])
                inputs.append(edge)
            else:
                inputs = self._video_gen_inputs(walk, persist_signals)
        else:
            # video_gen(_i2v) completed — one video per request.
            request_done = True

        # Each input is consumed for the last time in its pass; the denoise loop
        # re-injects its own worker-side.
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._get_step_metadata(metadata),
            request_done=request_done,
        )

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
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        # ``autocast_dtype`` and ``sp_group`` exist for interface parity only.
        # Weights load in the checkpoint's own dtypes, and wan22 declares no
        # sp-enabled nodes.
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded Wan2.2 submodule for node %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device: str = "cpu") -> NodeSubmodule | None:
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
            )
            return Wan22DitSubmodule(transformer, self.config)
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

        from mstar.model.wan22.components.unipc import NUM_TRAIN_TIMESTEPS, SOLVER_ORDER

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
