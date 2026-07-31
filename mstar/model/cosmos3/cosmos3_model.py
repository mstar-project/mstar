"""Cosmos3Model: NVIDIA Cosmos3 omni generator on the mstar engine.

Cosmos3 is a text-conditioned diffusion model: a dual-pathway Mixture-of-
Transformers DiT denoises image/video (and optionally sound) latents, which a
Wan VAE decodes to pixels. An optional action head extends the same backbone to
robot-action generation.

Nodes (2 for image generation):
    dit          (kv_cache)  - dual-pathway DiT. The understanding (text)
                               tower prefills the conditioning K/V; the
                               generation tower runs the denoise loop, reading
                               that frozen K/V each step (it is timestep-
                               independent, so caching it once is exact).
    vae_encoder  (stateless) - Wan VAE: conditioning image/video -> clean
                               anchor latents, in parallel with the prefill
                               (conditioned requests only).
    vae_decoder  (stateless) - Wan VAE: final latents -> pixels.

Graph walks (image generation):
    prefill    - the understanding tower runs over the text prompt and writes
                 its per-layer K/V (causal self-attention over text).
    image_gen  - an N-step denoising loop. Each iteration the generation tower
                 attends to [frozen text K/V | current generation tokens],
                 predicts flow velocity, and applies one scheduler step; the
                 final latents go to the VAE decoder, which emits the image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.distributed.base import ShardingConfig
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.cosmos3 import constants
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import (
    ACTION_GEN_LOOP,
    ACTION_VIDEO_GEN_LOOP,
    IMAGE_GEN_LOOP,
    VIDEO_GEN_LOOP,
    VIDEO_SOUND_GEN_LOOP,
    Cosmos3AudioDecoderSubmodule,
    Cosmos3DiTSubmodule,
    Cosmos3VAEDecoderSubmodule,
    Cosmos3VAEEncoderSubmodule,
)

logger = logging.getLogger(__name__)

DIT_NODE = "dit"
VAE_ENCODER_NODE = "vae_encoder"
VAE_DECODER_NODE = "vae_decoder"
AUDIO_DECODER_NODE = "audio_decoder"


class Cosmos3Model(Model):
    """NVIDIA Cosmos3 generator implementation."""

    PREFILL_WALK = constants.PREFILL_WALK
    PREFILL_COND_WALK = constants.PREFILL_COND_WALK
    PREFILL_COND_VIDEO_WALK = constants.PREFILL_COND_VIDEO_WALK
    IMAGE_GEN_WALK = constants.IMAGE_GEN_WALK
    VIDEO_GEN_WALK = constants.VIDEO_GEN_WALK
    VIDEO_SOUND_GEN_WALK = constants.VIDEO_SOUND_GEN_WALK
    ACTION_GEN_WALK = constants.ACTION_GEN_WALK
    ACTION_VIDEO_GEN_WALK = constants.ACTION_VIDEO_GEN_WALK

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        self._yaml_config_overrides: dict = dict(kwargs)

        self._repo_dir: Path | None = None
        self.config: Cosmos3Config = self._load_config()
        self.tokenizer = self._load_tokenizer()

        self._submodule_cache: dict[str, torch.nn.Module | None] = {}
        # The Wan VAE is shared between the DiT submodule (conditioning encode)
        # and the decoder submodule, so build it once.
        self._vae = None

    # ------------------------------------------------------------------
    # Config + tokenizer
    # ------------------------------------------------------------------

    def _ensure_repo(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir
        candidate = Path(self.model_path_hf)
        if candidate.exists():
            self._repo_dir = candidate
        else:
            from huggingface_hub import snapshot_download

            self._repo_dir = Path(
                snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir)
            )
        return self._repo_dir

    def _load_config(self) -> Cosmos3Config:
        if self.skip_weight_loading:
            cfg = Cosmos3Config()
        else:
            try:
                cfg = Cosmos3Config.from_pretrained(self._ensure_repo())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load Cosmos3 config from %s (%s); using Nano defaults.",
                    self.model_path_hf, exc,
                )
                cfg = Cosmos3Config()

        # Overlay yaml model_kwargs last (so they win over file + defaults).
        if self._yaml_config_overrides:
            valid = {f.name for f in Cosmos3Config.__dataclass_fields__.values()}
            for k, v in self._yaml_config_overrides.items():
                if k in valid:
                    setattr(cfg, k, v)
                else:
                    logger.warning(
                        "Cosmos3Model: yaml model_kwargs key %r is not a Cosmos3Config "
                        "field; ignored.", k,
                    )
        return cfg

    def _load_tokenizer(self):
        if self.skip_weight_loading:
            return None
        from transformers import AutoTokenizer

        repo = self._ensure_repo()
        # The published checkpoint ships the Qwen2 text tokenizer under
        # ``text_tokenizer/``; fall back to the repo root for layouts that
        # keep the tokenizer files at the top level.
        for sub in (repo / "text_tokenizer", repo):
            try:
                return AutoTokenizer.from_pretrained(str(sub), use_fast=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cosmos3 tokenizer load from %s failed (%s).", sub, exc)
        logger.warning("All Cosmos3 tokenizer sources failed; proceeding without one.")
        return None

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return [
            KVCacheConfig(
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_position_embeddings,
                num_qo_heads=self.config.num_attention_heads,
                attention_backend=self.config.attention_backend,
            )
        ]

    def _sound_serving_enabled(self) -> bool:
        """Whether the opt-in sound walk (and its audio_decoder node) is served.

        Requires the model capability (``sound_gen``), the serving knob
        (``enable_sound``, yaml-overridable), and — with real weights — the
        checkpoint's ``sound_tokenizer/`` component."""
        if not (self.config.sound_gen and self.config.enable_sound):
            return False
        if self.skip_weight_loading:
            return True
        return (self._ensure_repo() / "sound_tokenizer" / "config.json").exists()

    def get_node_engine_types(self) -> dict[str, EngineType]:
        types = {
            DIT_NODE: EngineType.KV_CACHE,
            VAE_ENCODER_NODE: EngineType.STATELESS,
            VAE_DECODER_NODE: EngineType.STATELESS,
        }
        if self._sound_serving_enabled():
            types[AUDIO_DECODER_NODE] = EngineType.STATELESS
        return types

    def get_default_sharding_config(self) -> ShardingConfig:
        # The DiT supports tensor parallelism: per layer the attention heads and
        # the MLP intermediate dim shard across ranks, the residual stream stays
        # full, and the row-parallel out/down projections all-reduce. Signals
        # between nodes stay replicated (empty shard_dim) — the sharding is
        # in-module, Megatron-style. The VAE decoder runs un-sharded on one rank.
        return ShardingConfig(
            groups=[], tp_enabled_nodes={DIT_NODE}, shard_dim={},
            sp_enabled_nodes={DIT_NODE},
        )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # prefill: the understanding tower runs over the text prompt and writes
        # its conditioning K/V. No graph output — completion notifies the
        # conductor, and the generation loop reads the K/V from the shared cache.
        prefill = GraphNode(
            name=DIT_NODE,
            input_names=["text_inputs"],
            outputs=[],
        )

        # prefill_cond: the DiT prefills the text prompt while, in parallel, the
        # vae_encoder node encodes the conditioning image into the clean anchor
        # latents that seed the denoise loop. The latents come back to the
        # conductor as a persist signal and enter the generation walk as its
        # ``cond_latents`` input edge.
        def _prefill_cond_walk(cond_input: str) -> Parallel:
            return Parallel(
                [
                    GraphNode(name=DIT_NODE, input_names=["text_inputs"], outputs=[]),
                    GraphNode(
                        name=VAE_ENCODER_NODE,
                        input_names=[cond_input],
                        outputs=[
                            GraphEdge(
                                next_node=EMPTY_DESTINATION,
                                name="cond_latents",
                                persist=True,
                            ),
                        ],
                    ),
                ]
            )

        prefill_cond = _prefill_cond_walk("image_inputs")
        # prefill_cond_video: video conditioning (action inverse-dynamics /
        # video-to-video) encodes the request video instead.
        prefill_cond_video = _prefill_cond_walk("video_inputs")

        # image_gen: denoising loop -> VAE decode -> emit image. The loop body
        # threads the latents + denoise-step index back to itself each iteration;
        # on the final iteration the latents route to the decoder. max_iters is an
        # upper bound — each request stops the loop at its own denoise-step count
        # (Cosmos3DiTSubmodule.check_stop), so one graph serves image and video
        # (and any per-request num_inference_steps) without a rebuild.
        # image_gen and video_gen are the same denoise loop + VAE decode; they
        # differ only in the emitted modality (one frame vs an encoded clip), so
        # the request's output modality selects between them.
        def _gen_walk(loop_name: str, emit_name: str, modality: str) -> Sequential:
            return Sequential(
                [
                    Loop(
                        name=loop_name,
                        # Async (speculative) scheduling pre-dispatches each
                        # request's next denoise step; its yield-away lets other
                        # ready requests join, so concurrent requests still batch
                        # into one forward (see can_batch/forward_batched).
                        # ``cond_latents`` is the loop's external input (no
                        # loop-back edge produces it): the vae_encoder node's
                        # persist signal enters once at the walk transition —
                        # empty for unconditioned requests — and the loop
                        # re-presents the same worker-local tensor each
                        # iteration (no per-step transfer).
                        section=GraphNode(
                            name=DIT_NODE,
                            input_names=["latents", "time_index", "cond_latents"],
                            outputs=[
                                GraphEdge(next_node=DIT_NODE, name="latents"),
                                GraphEdge(next_node=DIT_NODE, name="time_index"),
                            ],
                            enable_async_scheduling=True,
                        ),
                        max_iters=self.config.max_inference_steps,
                        outputs=[
                            GraphEdge(next_node=VAE_DECODER_NODE, name="latents"),
                        ],
                    ),
                    GraphNode(
                        name=VAE_DECODER_NODE,
                        input_names=["latents"],
                        outputs=[
                            GraphEdge(
                                next_node=EMIT_TO_CLIENT,
                                name=emit_name,
                                output_modality=modality,
                            ),
                        ],
                    ),
                ]
            )

        image_gen = _gen_walk(IMAGE_GEN_LOOP, "image_output", "image")
        video_gen = _gen_walk(VIDEO_GEN_LOOP, "video_output", "video")

        # video_sound_gen (opt-in): the video denoise loop with a jointly
        # denoised sound band threaded alongside the video latents. On the final
        # iteration the video latents route to the Wan VAE and the sound latents
        # to the AVAE audio decoder; the walk emits both a video and an audio
        # output, which the API layer muxes into one file.
        video_sound_gen = Sequential(
            [
                Loop(
                    name=VIDEO_SOUND_GEN_LOOP,
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=["latents", "sound_latents", "time_index", "cond_latents"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="sound_latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
                        enable_async_scheduling=True,
                    ),
                    max_iters=self.config.max_inference_steps,
                    outputs=[
                        GraphEdge(next_node=VAE_DECODER_NODE, name="latents"),
                        GraphEdge(next_node=AUDIO_DECODER_NODE, name="sound_latents"),
                    ],
                ),
                # The two decoders are independent: each runs as soon as its
                # own latents arrive from the loop.
                Parallel(
                    [
                        GraphNode(
                            name=VAE_DECODER_NODE,
                            input_names=["latents"],
                            outputs=[
                                GraphEdge(
                                    next_node=EMIT_TO_CLIENT,
                                    name="video_output",
                                    output_modality="video",
                                ),
                            ],
                        ),
                        GraphNode(
                            name=AUDIO_DECODER_NODE,
                            input_names=["sound_latents"],
                            outputs=[
                                GraphEdge(
                                    next_node=EMIT_TO_CLIENT,
                                    name="audio_output",
                                    output_modality="audio",
                                ),
                            ],
                        ),
                    ]
                ),
            ]
        )

        # action_gen: like image_gen but the loop body jointly denoises the video
        # and action latents (threaded as two self-edges), and the predicted
        # action — not a decoded video — is what the request emits.
        action_gen = Sequential(
            [
                Loop(
                    name=ACTION_GEN_LOOP,
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=["latents", "action_latents", "time_index", "cond_latents"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="action_latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
                        enable_async_scheduling=True,
                    ),
                    max_iters=self.config.max_inference_steps,
                    # The loop's terminal output is matched into the section by
                    # name (Loop.__post_init__ filters to the section's own output
                    # edges), so it must reuse a loop-back name: on the final
                    # iteration the predicted action latents go to the client
                    # instead of back into the loop.
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="action_latents",
                            output_modality="action",
                        ),
                    ],
                ),
            ]
        )

        # action_video_gen (forward dynamics): the same joint video+action denoise,
        # but the action is the clean condition and the predicted video is decoded
        # and emitted. The loop's terminal output reuses the "latents" loop-back
        # name; on the final iteration the video latents route to the VAE decoder
        # instead of back into the loop.
        action_video_gen = Sequential(
            [
                Loop(
                    name=ACTION_VIDEO_GEN_LOOP,
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=["latents", "action_latents", "time_index", "cond_latents"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="action_latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
                        enable_async_scheduling=True,
                    ),
                    max_iters=self.config.max_inference_steps,
                    outputs=[
                        GraphEdge(next_node=VAE_DECODER_NODE, name="latents"),
                    ],
                ),
                GraphNode(
                    name=VAE_DECODER_NODE,
                    input_names=["latents"],
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="video_output",
                            output_modality="video",
                        ),
                    ],
                ),
            ]
        )

        walks = {
            self.PREFILL_WALK: prefill,
            self.PREFILL_COND_WALK: prefill_cond,
            self.PREFILL_COND_VIDEO_WALK: prefill_cond_video,
            self.ACTION_VIDEO_GEN_WALK: action_video_gen,
            self.IMAGE_GEN_WALK: image_gen,
            self.VIDEO_GEN_WALK: video_gen,
            self.ACTION_GEN_WALK: action_gen,
        }
        # The sound walk references the audio_decoder node, which only exists
        # (and only needs a node_groups entry) when sound serving is enabled.
        if self._sound_serving_enabled():
            walks[self.VIDEO_SOUND_GEN_WALK] = video_sound_gen
        return walks

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if prompt is None:
            return {}
        if self.tokenizer is None:
            # Tokenizer-less fallback used by structural unit tests.
            return {
                "text_inputs": [
                    torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)
                ]
            }
        # Both the conditional (positive) and unconditional (negative) prompts are
        # tokenized up front; the denoiser reads the second only when guidance is
        # on. Image/video prompts get the chat template + resolution/duration
        # sentences; action prompts are tokenized raw.
        from mstar.model.cosmos3.components.packing import tokenize_prompt

        negative_prompt = kwargs.get("negative_prompt")
        p = self._resolve_gen_params(kwargs, input_modalities, output_modalities)
        # The chat system prompt and the resolution/duration metadata sentences
        # are opt-in, off by default: the model sees the bare user prompt, which
        # matches the reference serving pipeline (its system-prompt and
        # resolution/duration templates default off too). A request may re-enable
        # any of them. Action prompts never use them — they are just the
        # chat-templated user text plus the end-of-text + start-of-generation
        # markers (matching the NVIDIA action references).
        is_action = "action" in output_modalities
        allow_templates = not is_action
        cond_ids, uncond_ids = tokenize_prompt(
            self.tokenizer, prompt, negative_prompt,
            num_frames=p["num_frames"], height=p["height"], width=p["width"], fps=p["fps"],
            use_system_prompt=allow_templates and bool(kwargs.get("use_system_prompt", False)),
            add_resolution_template=allow_templates and bool(kwargs.get("use_resolution_template", False)),
            add_duration_template=allow_templates and bool(kwargs.get("use_duration_template", False)),
            max_sequence_length=p["max_sequence_length"],
        )
        return {
            "text_inputs": [
                torch.tensor(cond_ids, dtype=torch.long),
                torch.tensor(uncond_ids, dtype=torch.long),
            ]
        }

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == "image":
            import io
            import os

            from PIL import Image

            # The decoder emits 8-bit frames [B, C, T, H, W]; take the first one.
            x = output
            if x.ndim == 5:
                x = x[0, :, 0]
            elif x.ndim == 4:
                x = x[0]
            arr = x.permute(1, 2, 0).cpu().numpy()  # H, W, C uint8
            buf = io.BytesIO()
            # PNG is lossless at every compression level, so the level only trades
            # encode time for file size. PIL defaults to 6, which spends ~0.75 s on a
            # 720p frame and dominates the serving latency. Level 0 (no deflate) is
            # the fastest and matches what the OpenAI image endpoint emits at full
            # quality; the decoded pixels are identical regardless. Override with
            # COSMOS3_PNG_COMPRESS for A/B.
            compress_level = int(os.environ.get("COSMOS3_PNG_COMPRESS", "0"))
            Image.fromarray(arr).save(buf, format="PNG", compress_level=compress_level)
            return buf.getvalue()
        if modality == "video":
            import os

            # The decoder emits 8-bit frames [B, C, T, H, W]; encode all of them as
            # an H.264 mp4. The frames already reflect the request fps (it modulates
            # the temporal positions during generation), and the container carries
            # that same rate so playback runs at the requested speed.
            #
            # CRF 18 keeps the H.264 output near-visually-lossless; libx264
            # otherwise defaults to 23, which is visibly lossier. The "ultrafast"
            # preset and multithreading (threads=0) target the same CRF/quality
            # but encode several times faster than libx264's default "medium"
            # preset, which otherwise dominates the serving latency for a
            # many-frame clip. Both are overridable via COSMOS3_X264_PRESET.
            x = output[0] if output.ndim == 5 else output  # [C, T, H, W] uint8
            fps = float((request_kwargs or {}).get("fps", self.config.fps))
            preset = os.environ.get("COSMOS3_X264_PRESET", "ultrafast")
            try:
                # Preferred: torchcodec (torchvision >= 0.27 removed write_video).
                from torchcodec.encoders import VideoEncoder

                frames = x.permute(1, 0, 2, 3).contiguous().cpu()  # [T, C, H, W] uint8
                encoded = VideoEncoder(frames, frame_rate=fps).to_tensor(
                    "mp4",
                    codec="libx264",
                    crf=18,
                    preset=preset,
                    extra_options={"threads": "0"},
                )
                data = encoded.numpy().tobytes()
            except ImportError:
                # Fallback for environments without torchcodec (or with the
                # older decode-only torchcodec that lacks VideoEncoder), where
                # torchvision still ships write_video.
                import tempfile

                from torchvision.io import write_video

                frames = x.permute(1, 2, 3, 0).cpu()  # [T, H, W, C] uint8
                fd, path = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                try:
                    write_video(
                        path,
                        frames,
                        fps=fps,
                        video_codec="libx264",
                        options={"crf": "18", "preset": preset, "threads": "0"},
                    )
                    with open(path, "rb") as f:
                        data = f.read()
                finally:
                    os.remove(path)
            return data
        if modality == "action":
            # The predicted action latents [1, chunk, action_dim] -> [chunk,
            # action_dim] float32 bytes. Columns beyond the request's
            # raw_action_dim are zero padding (the client keeps the first
            # raw_action_dim, the real action width for its embodiment).
            x = output[0] if output.ndim == 3 else output
            return x.detach().to(torch.float32).cpu().numpy().tobytes()
        if modality == "audio":
            # The audio decoder emits a [channels, samples] waveform in [-1, 1];
            # the serving convention for audio is headerless interleaved 16-bit
            # PCM (the API layer wraps it with the model's sample rate).
            x = output[0] if output.ndim == 3 else output
            pcm = (x.detach().to(torch.float32).clamp(-1, 1) * 32767.0).round().to(torch.int16)
            return pcm.T.contiguous().cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Cosmos3: {modality!r}")

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return int(self.config.sound_sample_rate)

    def get_output_audio_channels(self, modality: str = "audio") -> int:
        return 2

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def _resolve_gen_params(
        self, model_kwargs: dict | None, input_modalities: list[str], output_modalities: list[str],
    ) -> dict:
        """Resolve the per-request generation knobs (size, steps, guidance, …)
        from request ``model_kwargs``, applying defaults. Used by both
        ``process_prompt`` (for resolution-aware tokenization) and the forward-
        pass metadata, so the two stay consistent."""
        mk = model_kwargs or {}
        width = height = 1024
        size = mk.get("size")
        if isinstance(size, str) and "x" in size.lower():
            sw, sh = size.lower().split("x", 1)
            try:
                width, height = int(sw), int(sh)
            except ValueError:
                pass

        # A video request without an explicit frame count gets the video default
        # (>1); image requests stay single-frame.
        default_frames = (
            self.config.num_frames_video if "video" in (output_modalities or []) else 1
        )
        num_frames = int(mk.get("num_frames", default_frames))
        # The image and video cookbook step counts differ (image 50, video 35);
        # default by mode and let the request override. The denoise loop runs this
        # many steps and stops early (Cosmos3DiTSubmodule.check_stop), so the value
        # is only bounded above by the loop's static max_iters.
        default_steps = (
            self.config.num_inference_steps_video if num_frames > 1
            else self.config.num_inference_steps
        )
        steps = int(mk.get("num_inference_steps", default_steps))
        steps = max(1, min(steps, self.config.max_inference_steps))
        params = {
            "width": int(mk.get("width", width)),
            "height": int(mk.get("height", height)),
            "num_frames": num_frames,
            "fps": float(mk.get("fps", self.config.fps)),
            "guidance_scale": float(mk.get("guidance_scale", 6.0)),
            "num_inference_steps": steps,
            "has_image_condition": "image" in (input_modalities or []),
            "use_karras_sigma": mk.get("use_karras_sigmas"),
            # Prompt-token truncation cap (reference serving default 4096),
            # request-overridable; floor 1 so a bad value can't empty the prompt.
            "max_sequence_length": max(
                1, int(mk.get("max_sequence_length", constants.DEFAULT_MAX_SEQUENCE_LENGTH))
            ),
        }
        # Video-to-video: a non-action video input pins clean conditioning
        # latent frames taken from the request video (reference recipe defaults:
        # indexes (0, 1), keep "first", flow_shift 10.0). Validated here so a
        # malformed request fails at submission rather than mid-denoise.
        has_video_condition = "video" in (input_modalities or []) and "action_mode" not in mk
        if has_video_condition:
            from mstar.model.cosmos3.components.packing import normalize_condition_frame_indexes

            if num_frames <= 1:
                raise ValueError("Cosmos3 video conditioning requires a video request (num_frames > 1).")
            indexes = normalize_condition_frame_indexes(
                mk.get("condition_frame_indexes_vision"),
                constants.DEFAULT_CONDITION_FRAME_INDEXES_VISION,
            )
            latent_frames = 1 + (num_frames - 1) // self.config.vae.scale_factor_temporal
            if indexes[-1] >= latent_frames:
                raise ValueError(
                    f"Cosmos3 condition_frame_indexes_vision {indexes} is outside the latent "
                    f"video ({latent_frames} latent frames for num_frames={num_frames})."
                )
            keep = str(mk.get("condition_video_keep") or constants.DEFAULT_CONDITION_VIDEO_KEEP).strip().lower()
            if keep not in ("first", "last"):
                raise ValueError("Cosmos3 condition_video_keep must be 'first' or 'last'.")
            params["has_video_condition"] = True
            params["condition_frame_indexes_vision"] = indexes
            params["condition_video_keep"] = keep
        # Text-to-image (single frame, no visual conditioning) follows the
        # reference Cosmos3 t2i recipe: classifier-free guidance only on the
        # timestep interval [400, 1000] (outside it the denoise step runs the
        # conditional branch alone) and flow_shift 3.0. Request kwargs override;
        # video-to-video defaults to the reference V2V flow_shift; other
        # image-conditioned / video paths keep their own defaults (full CFG,
        # scheduler-config flow_shift).
        is_t2i = num_frames == 1 and not params["has_image_condition"]
        fs = mk.get("flow_shift")
        if fs is None and is_t2i:
            fs = 3.0
        if fs is None and has_video_condition:
            fs = constants.V2V_DEFAULT_FLOW_SHIFT
        if fs is not None:
            params["flow_shift"] = float(fs)
        gi = mk.get("guidance_interval")
        if gi is None and is_t2i:
            gi = (400.0, 1000.0)
        if gi is not None:
            params["guidance_interval"] = (float(gi[0]), float(gi[1]))
        # Action requests carry a few extra keys straight through (``action`` is
        # the clean conditioning action chunk for forward-dynamics).
        for k in ("action_mode", "action_chunk_size", "raw_action_dim", "domain_id",
                  "action_fps", "action"):
            if k in mk:
                params[k] = mk[k]
        # Opt-in sound generation: video-only (image and action requests carry
        # no sound band), and only when the served checkpoint/config enable it.
        if mk.get("generate_sound") or mk.get("sound_gen"):
            if num_frames <= 1 or "action_mode" in params:
                raise ValueError(
                    "Cosmos3 sound generation is supported only for video requests "
                    "(num_frames > 1, no action mode)."
                )
            if not self._sound_serving_enabled():
                raise ValueError(
                    "Cosmos3 sound generation was requested, but sound serving is "
                    "disabled or the checkpoint has no sound_tokenizer/ component."
                )
            params["generate_sound"] = True
            if mk.get("sound_duration") is not None:
                params["sound_duration"] = float(mk["sound_duration"])
        return params

    def _step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        md = {"is_prefill": metadata.is_prefill}
        md.update(metadata.kwargs)
        return md

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        params = self._resolve_gen_params(model_kwargs, input_modalities, output_modalities)
        # Visual conditioning routes through a conditioned prefill that also feeds
        # the DiT the input to VAE-encode: a video (action inverse-dynamics) or an
        # image (image-to-video, action policy/forward-dynamics). Fall back to the
        # text-only prefill if no conditioning signal actually arrived.
        video_cond = "video" in input_modalities and "video_inputs" in input_signals
        image_cond = params.get("has_image_condition") and "image_inputs" in input_signals
        if video_cond:
            walk = self.PREFILL_COND_VIDEO_WALK
        elif image_cond:
            walk = self.PREFILL_COND_WALK
        else:
            walk = self.PREFILL_WALK
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=walk,
            is_prefill=True,
            kwargs=params,
        )

        inputs: list[GraphEdge] = []
        if "text_inputs" in input_signals:
            edge = GraphEdge(next_node=DIT_NODE, name="text_inputs")
            edge.tensor_info = input_signals["text_inputs"]
            inputs.append(edge)
        cond_signal = "video_inputs" if video_cond else ("image_inputs" if image_cond else None)
        if cond_signal:
            edge = GraphEdge(next_node=VAE_ENCODER_NODE, name=cond_signal)
            edge.tensor_info = input_signals[cond_signal]
            inputs.append(edge)

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._step_metadata(full_metadata),
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False
        inputs: list[GraphEdge] = []

        # Forward-dynamics conditions on a clean action chunk and emits the
        # predicted video; inverse-dynamics / policy emit the action.
        is_fd = metadata.kwargs.get("action_mode") == "forward_dynamics"
        is_action = "action" in metadata.output_modalities
        is_video = "video" in metadata.output_modalities
        joint_action = is_fd or is_action  # walks that also thread action latents
        if metadata.graph_walk in (
            self.PREFILL_WALK, self.PREFILL_COND_WALK, self.PREFILL_COND_VIDEO_WALK
        ):
            metadata.is_prefill = False
            # Pick the denoise walk: forward-dynamics runs the joint denoise but
            # decodes the predicted video; inverse-dynamics / policy emit the
            # action; image and video share the loop but differ in what the VAE
            # node emits.
            if is_fd:
                metadata.graph_walk = self.ACTION_VIDEO_GEN_WALK
            elif is_action:
                metadata.graph_walk = self.ACTION_GEN_WALK
            elif is_video and metadata.kwargs.get("generate_sound"):
                metadata.graph_walk = self.VIDEO_SOUND_GEN_WALK
            elif is_video:
                metadata.graph_walk = self.VIDEO_GEN_WALK
            else:
                metadata.graph_walk = self.IMAGE_GEN_WALK
            # The first denoise iteration's initial noise + step index are
            # sampled inside the DiT submodule's preprocess. Action and sound
            # walks also thread their extra latents through the loop.
            inputs = [
                GraphEdge(next_node=DIT_NODE, name="latents"),
                GraphEdge(next_node=DIT_NODE, name="time_index"),
            ]
            if joint_action:
                inputs.insert(1, GraphEdge(next_node=DIT_NODE, name="action_latents"))
            elif metadata.graph_walk == self.VIDEO_SOUND_GEN_WALK:
                inputs.insert(1, GraphEdge(next_node=DIT_NODE, name="sound_latents"))
            # The vae_encoder node's clean conditioning latents (persisted at
            # the conductor during the prefill walk) seed the loop's first
            # iteration; unconditioned requests carry an empty edge.
            cond_edge = GraphEdge(next_node=DIT_NODE, name="cond_latents")
            cond_edge.tensor_info = persist_signals.get("cond_latents", [])
            inputs.append(cond_edge)
        elif metadata.graph_walk in (
            self.IMAGE_GEN_WALK, self.VIDEO_GEN_WALK, self.VIDEO_SOUND_GEN_WALK,
            self.ACTION_GEN_WALK, self.ACTION_VIDEO_GEN_WALK,
        ):
            request_done = True

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._step_metadata(metadata),
            request_done=request_done,
        )

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        # autocast_dtype is accepted for interface parity (the engine manager
        # passes it to every model). Cosmos3 already casts the meta module to
        # bf16 before to_empty in _build_transformer, so params are allocated
        # directly in the checkpoint dtype and the hint is redundant here.
        # sp_group is the DiT's sequence-parallel comm group (trivial unless the
        # config sets sp_size > 1); it is orthogonal to the tp_group.
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, tp_group, sp_group)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded Cosmos3 submodule for %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device: str, tp_group=None, sp_group=None):
        if node_name == DIT_NODE:
            return Cosmos3DiTSubmodule(
                transformer=self._build_transformer(device, tp_group=tp_group, sp_group=sp_group),
                config=self.config,
                scheduler=self._build_scheduler(),
            )
        if node_name == VAE_ENCODER_NODE:
            return Cosmos3VAEEncoderSubmodule(
                vae=self._build_encode_vae(device), config=self.config
            )
        if node_name == VAE_DECODER_NODE:
            return Cosmos3VAEDecoderSubmodule(
                vae=self._build_vae(device), config=self.config
            )
        if node_name == AUDIO_DECODER_NODE:
            return Cosmos3AudioDecoderSubmodule(
                sound_tokenizer=self._build_sound_tokenizer(device), config=self.config
            )
        return None

    def _build_scheduler(self):
        if self.skip_weight_loading:
            return None
        from diffusers import UniPCMultistepScheduler

        return UniPCMultistepScheduler.from_pretrained(str(self._ensure_repo() / "scheduler"))

    def _build_transformer(self, device: str, tp_group=None, sp_group=None):
        from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
        from mstar.model.cosmos3.loader import load_transformer_weights

        # Build on the meta device (shapes only, no storage), pin the
        # checkpoint's bf16 dtype, then materialize uninitialized tensors on the
        # target device and overwrite with the checkpoint weights — the same
        # path the other model packages use. bf16 matches the published
        # checkpoint exactly and halves resident weight memory vs the float32
        # meta default; the engine additionally runs the forward under a bf16
        # autocast (a no-op here).
        with torch.device("meta" if not self.skip_weight_loading else "cpu"):
            model = Cosmos3OmniTransformer(self.config, comm_group=tp_group, sp_group=sp_group)
        model = model.to(torch.bfloat16)
        if self.skip_weight_loading:
            return model.to_empty(device=device)

        model.to_empty(device=device)
        load_transformer_weights(model, self._ensure_repo(), device=device)
        # Keep the timestep embedder in fp32, like diffusers'
        # ``_keep_in_fp32_modules=["time_embedder"]`` (the upcast is lossless from
        # the bf16 checkpoint and matches diffusers' numerics).
        model.time_embedder.to(torch.float32)
        model.eval()
        return model

    def _build_vae(self, device: str):
        if self.skip_weight_loading:
            return None
        if self._vae is not None:
            return self._vae
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(str(self._ensure_repo() / "vae"))
        self._vae = vae.to(device).eval()
        return self._vae

    def _build_encode_vae(self, device: str):
        # The encoder node keeps its own fp32 instance instead of sharing the
        # decoder's: encode always runs fp32 while the decode dtype is
        # cuDNN-gated (bf16 from 9.16), so a shared instance would re-cast the
        # full VAE weights on every encode/decode interleave.
        if self.skip_weight_loading:
            return None
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(str(self._ensure_repo() / "vae"))
        return vae.float().to(device).eval()

    def _build_sound_tokenizer(self, device: str):
        if self.skip_weight_loading:
            return None
        from mstar.model.cosmos3.components.sound_tokenizer import Cosmos3SoundTokenizer

        tokenizer = Cosmos3SoundTokenizer.from_pretrained(
            self._ensure_repo(), device=device, dtype=torch.bfloat16
        )
        logger.info(
            "Loaded Cosmos3 sound tokenizer (sr=%d, channels=%d, latent_ch=%d, hop=%d)",
            tokenizer.sample_rate, tokenizer.audio_channels, tokenizer.latent_ch, tokenizer.hop_size,
        )
        return tokenizer
