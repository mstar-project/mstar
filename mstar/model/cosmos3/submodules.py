"""NodeSubmodule wrappers for the Cosmos3 generator nodes.

Two nodes:
  Cosmos3DiTSubmodule         -- dual-pathway DiT (KV_CACHE). Dispatches by
                                 graph_walk between ``prefill`` (the
                                 understanding tower runs once over the text
                                 prompt and writes its per-layer K/V) and
                                 ``image_gen`` (one denoising step of the
                                 generation tower per loop iteration, attending
                                 to the frozen understanding K/V plus the
                                 current generation tokens, then one scheduler
                                 step). Classifier-free guidance keeps the
                                 conditional and unconditional prompts in two
                                 cache labels and combines their velocities.
  Cosmos3VAEDecoderSubmodule  -- Wan VAE decode (STATELESS): final latents to
                                 pixels.

Because the text tokens never receive a timestep embedding, the understanding
K/V is denoise-step independent, so writing it once and re-reading it every step
matches running the whole transformer each step.
"""

from __future__ import annotations

import logging
import math
import os

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import (
    BasicBatchedCudaGraphConfig,
    FlashInferPackedCudaGraphConfig,
)
from mstar.model.cosmos3.components.packing import (
    action_start_frame_offset,
    build_action_static_inputs,
    build_static_inputs,
    vision_condition_frame_indexes,
)
from mstar.model.cosmos3.constants import (
    ACTION_GEN_WALK,
    ACTION_VIDEO_GEN_WALK,
    IMAGE_GEN_WALK,
    PREFILL_COND_VIDEO_WALK,
    PREFILL_COND_WALK,
    PREFILL_WALK,
    VIDEO_GEN_WALK,
    VIDEO_SOUND_GEN_WALK,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

# image_gen and video_gen run the identical denoise step (the DiT loop is
# shape-general over the frame count); they differ only in the emitted output
# modality (a single image frame vs an encoded video), which the graph fixes per
# walk, so the submodule treats them the same.
GEN_WALKS = (IMAGE_GEN_WALK, VIDEO_GEN_WALK)

# All prefill variants run the same understanding-tower prefill; the conditioned
# ones additionally VAE-encode an image (prefill_cond) or video
# (prefill_cond_video) into anchor latents.
PREFILL_WALKS = (PREFILL_WALK, PREFILL_COND_WALK, PREFILL_COND_VIDEO_WALK)

# Names of the denoise loops in the graph walks. The loops are built with a fixed
# upper-bound iteration count and each request stops its loop early at its own
# denoise-step count (see ``check_stop``), so one graph serves any per-request
# step count.
IMAGE_GEN_LOOP = "image_gen_loop"
VIDEO_GEN_LOOP = "video_gen_loop"
VIDEO_SOUND_GEN_LOOP = "video_sound_gen_loop"
ACTION_GEN_LOOP = "action_gen_loop"
ACTION_VIDEO_GEN_LOOP = "action_video_gen_loop"

# Both action walks run the joint video+action denoise loop body; they differ
# only in what they emit (the predicted action vs the predicted video).
ACTION_WALKS = (ACTION_GEN_WALK, ACTION_VIDEO_GEN_WALK)

# Opt-in sound generation: the same video denoise loop with an AVAE-latent sound
# band appended to the generation block, jointly denoised and threaded through
# the loop next to the video latents.
SOUND_WALKS = (VIDEO_SOUND_GEN_WALK,)

# Conditional prompt K/V lives under the primary label; the unconditional
# (negative) prompt's K/V lives under a second label for classifier-free
# guidance. Both are written once at prefill and read every denoise step.
COND_LABEL = "main"
UNCOND_LABEL = "uncond"

# Combined label for the single FlashInfer plan that runs both guidance branches
# in one forward (see cache_manager.plan_attention_batched_cfg).
CFG_BATCHED_LABEL = "_cfg_batched"


class Cosmos3DiTSubmodule(ARNodeSubmodule):
    """Dual-pathway DiT node (understanding tower + generation denoiser)."""

    # The denoise loop is data-dependent (per-step timestep .item(), scheduler
    # step, classifier-free guidance combine), so torch.compile graph-breaks and
    # buys little; CUDA-graph capture of the fixed-shape step is the accelerator.
    disable_torch_compile = True

    # Run the two classifier-free-guidance branches as a single batched forward
    # per denoise step instead of two sequential forwards. The math is the same;
    # set False to fall back to the sequential path.
    batched_cfg = True

    # Cap on how many requests share one batched denoise step. Concurrent
    # requests at the image-generation walk run their step in a single forward.
    max_gen_batch_size = 8

    # Image resolutions (height, width) to capture a denoise-step CUDA graph for.
    # Requests at other resolutions fall back to the eager path. num_frames is
    # fixed at 1 (text-to-image). The graph accelerates the single-request
    # (batch size 1) denoise step, where the forward is launch-bound: the win is
    # large at low resolution (~2.5x at 320x192) and shrinks as the step becomes
    # compute-bound at higher resolution. Concurrent requests batch via the eager
    # path regardless. The default covers the three standard generation tiers;
    # override with COSMOS3_GEN_CAPTURE_RES. The served graph output is identical
    # to the eager path (compare with COSMOS3_DISABLE_CUDA_GRAPH=1).
    gen_capture_resolutions: tuple[tuple[int, int], ...] = (
        (192, 320), (480, 832), (720, 1280),
    )
    # Batch sizes to capture per resolution.
    gen_capture_batch_sizes: tuple[int, ...] = (1,)

    # Understanding-tower (text prefill) CUDA-graph capture. The text length is
    # prompt-dependent, so capture a few combined (cond+uncond) token buckets and
    # round up at replay; prompts past the largest bucket fall back to eager.
    # Targets the small-prompt, non-compute-bound regime where the launch
    # overhead the graph removes actually matters. Override with
    # COSMOS3_PREFILL_CAPTURE_TOKENS / COSMOS3_PREFILL_CAPTURE_BS.
    prefill_capture_token_buckets: tuple[int, ...] = (16, 32, 64, 128, 256)
    prefill_capture_batch_sizes: tuple[int, ...] = (1,)

    def __init__(self, transformer, config, scheduler=None, vae=None):
        super().__init__()
        self.transformer = transformer
        self.config = config
        # Template scheduler; a fresh instance (with its own multistep state) is
        # built per request from this one's config.
        self._scheduler_template = scheduler
        # Wan VAE (shared with the decoder node) — used to encode the
        # conditioning frame for image-to-video / action conditioning. None for
        # text-only generation.
        self.vae = vae
        self._video_processor = None
        # Per-request denoising state: packed static inputs (cond/uncond),
        # scheduler, guidance scale, latent shape.
        self._req: dict[str, dict] = {}
        # torch.compile the pure denoise compute (~1.2-1.3x per step; the compiled
        # kernels bake into the per-resolution CUDA graphs at capture).
        # fullgraph=False leaves the FlashInfer attention an opaque graph break;
        # disable_torch_compile stays True so the engine does not also compile the
        # data-dependent submodule wrapper. ``config.compile_denoise=False`` runs
        # the eager step (the bit-exact parity tests rely on it).
        if config.compile_denoise and transformer is not None:
            self.transformer.denoise_step = torch.compile(
                self.transformer.denoise_step, fullgraph=False, dynamic=False,
            )
            self.transformer.denoise_step_batched_cfg = torch.compile(
                self.transformer.denoise_step_batched_cfg, fullgraph=False, dynamic=False,
            )
            logger.info("Cosmos3 denoise compute torch.compile enabled")

    def to(self, *args, **kwargs):
        # The engine casts this submodule to bf16; the timestep embedder must stay
        # fp32 (diffusers keeps it in _keep_in_fp32_modules, and the multi-step
        # video denoise scrambles when it runs in bf16), so re-assert fp32 after
        # any cast — paired with the autocast-disabled forward below.
        super().to(*args, **kwargs)
        te = getattr(self.transformer, "time_embedder", None)
        if te is not None:
            te.float()
        return self

    def get_needed_cache_labels(
        self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str] | None:
        return [COND_LABEL, UNCOND_LABEL]

    # ------------------------------------------------------------------
    # Static packing + scheduler helpers
    # ------------------------------------------------------------------

    def _latent_shape(
        self, height: int, width: int, num_frames: int = 1
    ) -> tuple[int, int, int, int, int]:
        s = self.config.vae.scale_factor_spatial
        t = 1 if num_frames == 1 else 1 + (num_frames - 1) // self.config.vae.scale_factor_temporal
        return (1, self.config.latent_channel, t, height // s, width // s)

    def _build_static(
        self, ids: list[int], height: int, width: int, num_frames: int,
        fps: float, has_image_condition: bool, device,
        sound_latent_frames: int | None = None,
    ) -> dict:
        static = build_static_inputs(
            list(ids), self._latent_shape(height, width, num_frames), self.config,
            self.config.vae.scale_factor_temporal, fps, device,
            has_image_condition=has_image_condition,
            sound_latent_frames=sound_latent_frames,
        )
        # proj_out runs on the generation token block, so shift the joint-sequence
        # mse indexes to be relative to the generation tokens.
        static["mse_gen_indexes"] = static["vision_mse_loss_indexes"] - static["und_len"]
        if sound_latent_frames is not None:
            static["sound_mse_gen_indexes"] = static["sound_mse_loss_indexes"] - static["und_len"]
        return static

    def _resolve_sound_frames(self, md: dict) -> tuple[int, int]:
        """Resolve a sound request's target sample count and latent frame count.

        The sound duration defaults to the video duration (``num_frames / fps``,
        the request's ``sound_duration`` overrides), the target sample count is
        that duration at the AVAE sample rate, and the latent frame count rounds
        it up to whole tokenizer hops (the decoded tail past the target is
        trimmed by the audio decode node)."""
        num_frames = int(md.get("num_frames", 1))
        fps = float(md.get("fps", 24.0))
        duration = md.get("sound_duration")
        if duration is None:
            duration = num_frames / fps
        duration = max(float(duration), 1.0 / max(fps, 1.0))
        target_samples = max(1, int(round(duration * self.config.sound_sample_rate)))
        hop = int(round(self.config.sound_sample_rate / self.config.sound_latent_fps))
        return target_samples, max(1, math.ceil(target_samples / hop))

    def _new_scheduler(self, num_inference_steps: int, device, use_karras_sigma=None, flow_shift=None):
        from diffusers import UniPCMultistepScheduler

        # The checkpoint scheduler config carries the trained sigma schedule
        # (karras sigmas); keep it unless the request explicitly overrides a
        # field. Forcing karras off uses the wrong schedule and corrupts the
        # larger model's high-resolution text-to-video.
        overrides = {}
        if use_karras_sigma is not None:
            overrides["use_karras_sigmas"] = use_karras_sigma
        if flow_shift is not None:
            overrides["flow_shift"] = flow_shift

        scheduler = UniPCMultistepScheduler.from_config(self._scheduler_template.config, **overrides)
        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler

    def _build_action_static(
        self, ids: list[int], height: int, width: int, num_frames: int, action_chunk: int,
        mode: str, fps: float, action_fps: float, action_offset: int, device,
    ) -> dict:
        static = build_action_static_inputs(
            list(ids), self._latent_shape(height, width, num_frames), action_chunk, mode,
            self.config, self.config.vae.scale_factor_temporal, fps, action_fps, action_offset, device,
        )
        # proj_out runs on the generation token block; shift the joint-sequence
        # mse indexes to be relative to the [vision | action] generation tokens.
        static["mse_gen_indexes"] = static["vision_mse_loss_indexes"] - static["und_len"]
        static["action_mse_gen_indexes"] = static["action_mse_loss_indexes"] - static["und_len"]
        return static

    # ------------------------------------------------------------------
    # prepare_inputs
    # ------------------------------------------------------------------

    def prepare_inputs(
        self, graph_walk, fwd_info, inputs, seen_token_mask=None, pos_info={},
    ) -> ARNodeInputs:
        device = self.get_device()
        if graph_walk in PREFILL_WALKS:
            return self._prepare_prefill(fwd_info, inputs, device)
        if graph_walk in GEN_WALKS:
            return self._prepare_image_gen(fwd_info, inputs, device)
        if graph_walk in SOUND_WALKS:
            return self._prepare_video_sound_gen(fwd_info, inputs, device)
        if graph_walk in ACTION_WALKS:
            return self._prepare_action_gen(fwd_info, inputs, device)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _prepare_prefill(self, fwd_info, inputs, device) -> ARNodeInputs:
        md = fwd_info.step_metadata
        height, width = int(md.get("height", 256)), int(md.get("width", 256))
        fps = float(md.get("fps", 24.0))
        gs = float(md.get("guidance_scale", 6.0))
        steps = int(md.get("num_inference_steps", self.config.num_inference_steps))
        cond_ids = inputs["text_inputs"][0].tolist()
        uncond_ids = inputs["text_inputs"][1].tolist() if gs != 1.0 else None

        action_mode = md.get("action_mode")
        if action_mode:
            return self._prepare_action_prefill(
                fwd_info, md, inputs, cond_ids, uncond_ids, height, width, fps, gs, steps, device
            )

        num_frames = int(md.get("num_frames", 1))
        # Image-to-video: latent frame 0 is a clean conditioning anchor supplied
        # in the first denoise step's ``latents``; it stays in the sequence but is
        # not denoised. (Text-to-image / text-to-video have no clean anchor.)
        has_image_condition = bool(md.get("has_image_condition", False))

        # Opt-in sound: append a jointly denoised AVAE-latent band to the
        # generation block. Video-only (single-frame image and action requests
        # are rejected at request resolution).
        sound_frames = None
        target_audio_samples = None
        if md.get("generate_sound"):
            if num_frames <= 1:
                raise ValueError("Cosmos3 sound generation requires a video request (num_frames > 1)")
            target_audio_samples, sound_frames = self._resolve_sound_frames(md)

        cond = self._build_static(
            cond_ids, height, width, num_frames, fps, has_image_condition, device,
            sound_latent_frames=sound_frames,
        )
        uncond = None
        if uncond_ids is not None:
            uncond = self._build_static(
                uncond_ids, height, width, num_frames, fps, has_image_condition, device,
                sound_latent_frames=sound_frames,
            )

        self._req[fwd_info.request_id] = {
            "cond": cond,
            "uncond": uncond,
            "gs": gs,
            "guidance_interval": md.get("guidance_interval"),
            "scheduler": self._new_scheduler(
                steps, device, flow_shift=md.get("flow_shift"),
                use_karras_sigma=md.get("use_karras_sigma")
            ),
            "num_noisy": cond["num_noisy_vision_tokens"],
            "num_vision": cond["num_vision_tokens"],
            "latent_shape": self._latent_shape(height, width, num_frames),
            "num_sound": sound_frames,
            "target_audio_samples": target_audio_samples,
        }
        # Image-to-video: encode the conditioning frame now (the understanding
        # tower and the VAE encode are both prefill-time, per-request work) and
        # stash its clean anchor latents for the denoise loop to inject.
        if has_image_condition:
            image = (inputs or {}).get("image_inputs")
            if image:
                self._req[fwd_info.request_id]["cond_latents"] = self._encode_conditioning(
                    image[0], height, width, num_frames, device, anchor_only=True
                )

        return self._get_prefill_node_inputs(cond, uncond)

    def _get_prefill_node_inputs(self, cond, uncond):
        statics = {COND_LABEL: cond}
        if uncond is not None:
            statics[UNCOND_LABEL] = uncond
        tensor_inputs = {}
        for label, s in statics.items():
            tensor_inputs[f"input_ids_{label}"] = s["input_ids"]
            tensor_inputs[f"text_mrope_ids_{label}"] = s["text_mrope_ids"]
        return ARNodeInputs(
            input_seq_len=sum(s["und_len"] for s in statics.values()),
            tensor_inputs=tensor_inputs,
            kwargs=dict(
                cfg=uncond is not None,
                seq_lens={label: s["und_len"] for label, s in statics.items()},
            ),
        )

    def _encode_conditioning(self, image, height, width, num_frames, device, anchor_only=False):
        """VAE-encode a conditioning frame into clean anchor latents.

        Mirrors the fused pipeline's image-to-video latent prep: the frame is
        resized and normalized to [-1, 1], repeat-padded across the clip, and
        Wan-VAE encoded with the pipeline-side latent normalization. Latent
        frame 0 is the clean anchor the denoise loop keeps fixed.

        Image-to-video only consumes latent frame 0, and the Wan VAE encodes
        frame 0 as a standalone causal anchor, so ``anchor_only`` skips the
        repeat-pad and encodes the single frame (a bit-identical frame 0)
        instead of the whole clip — at video lengths the full encode is the
        bulk of the conditioning cost. The encode runs in fp32 outside autocast:
        the VAE's 3D convs are far faster in fp32 (TF32) than bf16 on this cuDNN
        and the reference pipeline encodes in fp32 (matching the decoder)."""
        from diffusers.video_processor import VideoProcessor

        vae = self.vae
        if next(vae.parameters()).dtype != torch.float32:
            vae.float()
        dtype = self.transformer.proj_in.weight.dtype
        if self._video_processor is None:
            self._video_processor = VideoProcessor(
                vae_scale_factor=self.config.vae.scale_factor_spatial, resample="bilinear"
            )
        # load_image gives [C, H, W] in [0, 1]; preprocess -> [1, 3, H, W] in [-1, 1].
        frame = self._video_processor.preprocess(image, height=height, width=width).to(
            device=device, dtype=torch.float32
        )
        vision = frame.unsqueeze(2)
        if num_frames > 1 and not anchor_only:
            vision = vision.expand(-1, -1, num_frames, -1, -1)
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device)).view(
            1, -1, 1, 1, 1
        )
        with torch.autocast(device_type=vision.device.type, enabled=False):
            raw_mu = vae.encode(vision).latent_dist.mode()
        return ((raw_mu - mean) * inv_std).to(dtype)

    def _prepare_action_prefill(
        self, fwd_info, md, inputs, cond_ids, uncond_ids, height, width, fps, gs, steps, device,
    ) -> ARNodeInputs:
        mode = md["action_mode"]
        action_chunk = int(md["action_chunk_size"])
        num_frames = int(md.get("num_frames") or action_chunk + 1)
        raw_action_dim = int(md["raw_action_dim"])
        domain_id = int(md.get("domain_id", 0))
        action_fps = float(md.get("action_fps", fps))
        action_offset = action_start_frame_offset(action_chunk, num_frames)

        cond = self._build_action_static(
            cond_ids, height, width, num_frames, action_chunk, mode, fps, action_fps, action_offset, device
        )
        uncond = None
        if uncond_ids is not None:
            uncond = self._build_action_static(
                uncond_ids, height, width, num_frames, action_chunk, mode, fps, action_fps, action_offset, device
            )

        latent_shape = self._latent_shape(height, width, num_frames)
        t_lat = latent_shape[2]
        dtype = self.transformer.proj_in.weight.dtype
        action_dim = self.transformer.action_dim
        vmask = torch.zeros((1, 1, t_lat, 1, 1), device=device, dtype=dtype)
        for f in vision_condition_frame_indexes(mode, t_lat):
            vmask[:, :, f] = 1.0
        action_clean = torch.zeros((1, action_chunk, 1), device=device, dtype=dtype)
        if mode == "forward_dynamics":
            action_clean[:] = 1.0

        # Encode the visual conditioning to clean anchor latents: inverse-dynamics
        # conditions on the whole video (all frames), forward-dynamics / policy on
        # a single frame (frame 0). The per-mode vmask above selects which latent
        # frames are kept clean from these.
        cond_video = (inputs or {}).get("video_inputs")
        cond_image = (inputs or {}).get("image_inputs")
        if cond_video:
            cond_latents = self._encode_conditioning_video(cond_video[0], height, width, num_frames, device)
        elif cond_image:
            cond_latents = self._encode_conditioning(cond_image[0], height, width, num_frames, device)
        else:
            cond_latents = torch.zeros(latent_shape, device=device, dtype=dtype)

        # Forward-dynamics conditions on a clean action chunk supplied with the
        # request; the other modes predict the action (clean values are zero).
        clean_action = torch.zeros((1, action_chunk, action_dim), device=device, dtype=dtype)
        raw_act = md.get("action")
        if mode == "forward_dynamics" and raw_act is not None:
            act = torch.as_tensor(raw_act, device=device, dtype=dtype)
            if act.ndim == 3:
                act = act[0]
            if act.shape[0] < action_chunk:
                act = torch.cat([act, act[-1:].repeat(action_chunk - act.shape[0], 1)], dim=0)
            elif act.shape[0] > action_chunk:
                act = act[:action_chunk]
            clean_action[:, :, :raw_action_dim] = act[:, :raw_action_dim]

        self._req[fwd_info.request_id] = {
            "cond": cond,
            "uncond": uncond,
            "gs": gs,
            "scheduler": self._new_scheduler(
                steps, device, flow_shift=md.get("flow_shift"),
                use_karras_sigma=md.get("use_karras_sigma")
            ),
            "num_noisy": cond["num_noisy_vision_tokens"],
            "num_noisy_action": cond["num_noisy_action_tokens"],
            "num_vision": cond["num_vision_tokens"],
            "num_action": cond["num_action_tokens"],
            "latent_shape": latent_shape,
            "action_mode": mode,
            "action_chunk": action_chunk,
            "action_dim": action_dim,
            "raw_action_dim": raw_action_dim,
            "domain_t": torch.tensor([domain_id], dtype=torch.long, device=device),
            "vmask": vmask,
            "velocity_mask": 1.0 - vmask,
            "action_clean_mask": action_clean,
            "action_velocity_mask": 1.0 - action_clean,
            "cond_video_latents": cond_latents,
            "clean_action": clean_action,
        }
        return self._get_prefill_node_inputs(cond, uncond)

    def _encode_conditioning_video(self, video, height, width, num_frames, device):
        """VAE-encode a conditioning video clip into clean anchor latents.

        Used by action inverse-dynamics, which conditions on the whole observed
        clip. load_video gives [T, C, H, W] in [0, 1]; each frame is resized and
        normalized to [-1, 1] (matching the fused pipeline) and the clip is
        Wan-VAE encoded with the pipeline-side latent normalization."""
        from diffusers.video_processor import VideoProcessor

        vae = self.vae
        if next(vae.parameters()).dtype != torch.float32:
            vae.float()
        dtype = self.transformer.proj_in.weight.dtype
        if self._video_processor is None:
            self._video_processor = VideoProcessor(
                vae_scale_factor=self.config.vae.scale_factor_spatial, resample="bilinear"
            )
        clip = video[:num_frames]
        frames = [
            self._video_processor.preprocess(clip[i], height=height, width=width).squeeze(0)
            for i in range(clip.shape[0])
        ]
        # fp32 outside autocast: the VAE 3D convs are much faster in fp32 (TF32)
        # than bf16 on this cuDNN, and the reference pipeline encodes in fp32.
        vision = torch.stack(frames, dim=1).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,3,T,H,W]
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device)).view(
            1, -1, 1, 1, 1
        )
        with torch.autocast(device_type=vision.device.type, enabled=False):
            raw_mu = vae.encode(vision).latent_dist.mode()
        return ((raw_mu - mean) * inv_std).to(dtype)

    def _prepare_image_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        st = self._req[fwd_info.request_id]
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            gen = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
            latents = torch.randn(
                st["latent_shape"], generator=gen, device=device, dtype=self.transformer.proj_in.weight.dtype
            )
            cond_latents = st.get("cond_latents")
            if cond_latents is not None:
                # Image-to-video: latent frame 0 is the clean conditioning anchor;
                # the rest is noise. It stays clean through the loop because the
                # predicted velocity is zero on conditioning frames (unpatchify
                # only fills the noisy frames), matching the fused pipeline.
                latents[:, :, 0] = cond_latents[:, :, 0].to(latents.dtype)
            time_index = torch.zeros(1, dtype=torch.long, device=device)
        else:
            latents = inputs["latents"][0]
            time_index = inputs["time_index"][0]

        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        if step_index >= len(scheduler.timesteps):
            return None
        tensors = {"latents": latents, "time_index": time_index}
        # The CUDA-graph capture reads the timestep and rotary positions as static
        # buffers (it can't reach the per-request scheduler at replay), so
        # materialize them here. The eager path ignores these and recomputes from
        # per-request state. Only built in the two-branch guidance regime — the
        # one the graph captures.
        if st["uncond"] is not None:
            # The denoise loop may dispatch one extra (discarded) step past this
            # request's step count; clamp so materializing the static timestep
            # buffer can't index past the schedule.
            n_steps = len(st["scheduler"].timesteps)
            idx = time_index.reshape(-1).clamp(max=n_steps - 1)
            t = st["scheduler"].timesteps[idx].to(torch.float32)
            tensors["vision_timesteps"] = t.expand(st["num_noisy"]).contiguous()
            tensors["position_ids_cond"] = st["cond"]["vision_mrope_ids"]
            tensors["position_ids_uncond"] = st["uncond"]["vision_mrope_ids"]
        return ARNodeInputs(
            input_seq_len=st["num_vision"],
            tensor_inputs=tensors,
        )

    def _prepare_video_sound_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        st = self._req[fwd_info.request_id]
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            # First iteration: video noise first, then sound noise from the same
            # generator (the reference pipeline's RNG order), so the video band
            # of a sound request is bit-identical to the same-seed video-only
            # request.
            dtype = self.transformer.proj_in.weight.dtype
            gen = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
            latents = torch.randn(st["latent_shape"], generator=gen, device=device, dtype=dtype)
            cond_latents = st.get("cond_latents")
            if cond_latents is not None:
                latents[:, :, 0] = cond_latents[:, :, 0].to(latents.dtype)
            sound_latents = torch.randn(
                (1, self.config.sound_dim, st["num_sound"]), generator=gen, device=device, dtype=dtype
            )
            time_index = torch.zeros(1, dtype=torch.long, device=device)
        else:
            latents = inputs["latents"][0]
            sound_latents = inputs["sound_latents"][0]
            time_index = inputs["time_index"][0]

        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        if step_index >= len(scheduler.timesteps):
            return None
        return ARNodeInputs(
            input_seq_len=st["num_vision"] + st["num_sound"],
            tensor_inputs={
                "latents": latents, "sound_latents": sound_latents, "time_index": time_index,
            },
        )

    def _prepare_action_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        st = self._req[fwd_info.request_id]
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            # First iteration: build the joint [video | action] latents. Per the
            # mode masks, conditioning frames/action are clean and the predicted
            # ones start from noise; the clean anchors are then carried in the
            # looped latents (re-injected each step). Action noise is drawn before
            # the video noise to match the fused pipeline's RNG order.
            from diffusers.utils.torch_utils import randn_tensor

            dtype = self.transformer.proj_in.weight.dtype
            gen = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
            chunk, adim, raw = st["action_chunk"], st["action_dim"], st["raw_action_dim"]
            a_noise = randn_tensor((1, chunk, adim), generator=gen, device=device, dtype=dtype)
            a_noise[..., raw:] = 0
            action_latents = (
                st["action_clean_mask"] * st["clean_action"] + st["action_velocity_mask"] * a_noise
            )
            action_latents[..., raw:] = 0
            v_noise = randn_tensor(st["latent_shape"], generator=gen, device=device, dtype=dtype)
            latents = st["vmask"] * st["cond_video_latents"] + st["velocity_mask"] * v_noise
            time_index = torch.zeros(1, dtype=torch.long, device=device)
        else:
            latents = inputs["latents"][0]
            action_latents = inputs["action_latents"][0]
            time_index = inputs["time_index"][0]

        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        if step_index >= len(scheduler.timesteps):
            return None
        return ARNodeInputs(
            input_seq_len=st["num_vision"] + st["num_action"],
            tensor_inputs={"latents": latents, "action_latents": action_latents, "time_index": time_index},
        )

    # ------------------------------------------------------------------
    # preprocess: plan paged attention for the labels this walk touches.
    # ------------------------------------------------------------------

    def _plan_gen(self, cm, st, num_gen: int, cfg_active: bool = True) -> None:
        """Plan a denoise step's non-causal attention: one batched plan covering
        both guidance branches when they run together, else a plan per label.
        ``cfg_active`` False (a guidance_interval out-of-interval step, or
        gs==1) plans the conditional branch alone — matching the cond-only
        forward — so an interval step costs no wasted uncond/batched plan."""
        if st["uncond"] is None or not cfg_active:
            cm.plan_attention(seq_lens=[num_gen], is_causal=False, label=COND_LABEL, write_store=False)
        elif self.batched_cfg:
            cm.plan_attention_batched_cfg(
                labels=[COND_LABEL, UNCOND_LABEL], seq_lens=[num_gen],
                is_causal=False, write_store=False,
            )
        else:
            cm.plan_attention(seq_lens=[num_gen], is_causal=False, label=COND_LABEL, write_store=False)
            cm.plan_attention(seq_lens=[num_gen], is_causal=False, label=UNCOND_LABEL, write_store=False)

    def _preprocess_image_gen_captured(self, cm, inputs) -> dict:
        """Plan a denoise step for the CUDA-graph path.

        Runs with synthetic request ids (no per-request state), so it derives the
        token count from ``input_seq_len``. Both guidance branches are planned as
        one combined attention (``plan_attention_batched_cfg``) so the captured
        forward runs a single transformer pass over both — one weight load instead
        of two. The static-input tensors (latents, timestep, rotary positions) are
        stacked on a leading batch dim, so one captured graph spans a whole
        concurrent batch (a batch of one for the single-request latency path); the
        replay side copies each request's tensors into these fixed buffers.

        Separate from the eager ``preprocess`` by contract, not just output
        formation: this path takes pre-derived timestep/rotary buffers with no
        per-request scheduler state and must always plan both guidance branches
        (the graph shape is fixed), while the eager path derives its step inputs
        from ``time_index`` + per-request state and skips the uncond plan on
        guidance-interval steps.
        """
        seq_lens = [inp.input_seq_len for inp in inputs]
        cm.plan_attention_batched_cfg(
            labels=[COND_LABEL, UNCOND_LABEL], seq_lens=seq_lens,
            is_causal=False, write_store=False,
        )
        return {
            "latents": torch.stack([inp.tensor_inputs["latents"] for inp in inputs]),
            "vision_timesteps": torch.stack([inp.tensor_inputs["vision_timesteps"] for inp in inputs]),
            "position_ids_cond": torch.stack([inp.tensor_inputs["position_ids_cond"] for inp in inputs]),
            "position_ids_uncond": torch.stack([inp.tensor_inputs["position_ids_uncond"] for inp in inputs]),
        }

    def preprocess(
        self, graph_walk, engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs]
    ) -> dict:
        cm = engine_inputs.cache_manager

        if graph_walk == IMAGE_GEN_WALK and getattr(cm, "_cuda_graph_mode", False):
            return self._preprocess_image_gen_captured(cm, inputs)

        if graph_walk in PREFILL_WALKS:
            has_cfg = inputs[0].kwargs.get("cfg")
            labels = [COND_LABEL, UNCOND_LABEL] if has_cfg else [COND_LABEL]
            if has_cfg:
                cm.plan_attention_batched_cfg(
                    labels=labels,
                    seq_lens={
                        key: [inp.kwargs["seq_lens"][key] for inp in inputs] \
                            for key in labels
                    }, is_causal=True, write_store=False
                )
            else:
                cm.plan_attention(
                    seq_lens=[inp.input_seq_len for inp in inputs],
                    is_causal=True, label=COND_LABEL,
                    write_store=False
                )
            # Pack label-major (all cond segments, then all uncond) to match the
            # (label, request) batch order of the plan above.
            return {
                "cfg": has_cfg,
                "input_ids": torch.concat([
                    inp.tensor_inputs[f"input_ids_{label}"]
                    for label in labels for inp in inputs
                ]),
                "text_mrope_ids": torch.concat([
                    inp.tensor_inputs[f"text_mrope_ids_{label}"]
                    for label in labels for inp in inputs
                ], dim=1),
            }
        
        st = self._req[engine_inputs.request_ids[0]]

        if graph_walk in GEN_WALKS:
            rids = engine_inputs.request_ids
            if len(rids) > 1:
                # Cross-request batch: one batched plan over every request's two
                # guidance branches, each with its own page set and token count.
                cm.plan_attention_batched_cfg(
                    labels=[COND_LABEL, UNCOND_LABEL],
                    seq_lens=[self._req[r]["num_vision"] for r in rids],
                    is_causal=False, write_store=False,
                )
                return {
                    "latents": {r: inp.tensor_inputs["latents"] for r, inp in zip(rids, inputs, strict=True)},
                    "time_index": {r: inp.tensor_inputs["time_index"] for r, inp in zip(rids, inputs, strict=True)},
                }
            ti = inputs[0].tensor_inputs["time_index"]
            step_index = int(ti.reshape(-1)[0].item())
            self._plan_gen(
                cm, st, st["num_vision"], cfg_active=self._cfg_active(st, step_index)
            )
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "time_index": ti,
            }

        if graph_walk in SOUND_WALKS:
            rids = engine_inputs.request_ids
            if len(rids) > 1:
                # Cross-request batch: one batched plan over every request's
                # [vision | sound] block (sound requests batch only with sound
                # requests — batches are per graph walk).
                cm.plan_attention_batched_cfg(
                    labels=[COND_LABEL, UNCOND_LABEL],
                    seq_lens=[self._req[r]["num_vision"] + self._req[r]["num_sound"] for r in rids],
                    is_causal=False, write_store=False,
                )
                return {
                    "latents": {r: inp.tensor_inputs["latents"] for r, inp in zip(rids, inputs, strict=True)},
                    "sound_latents": {
                        r: inp.tensor_inputs["sound_latents"] for r, inp in zip(rids, inputs, strict=True)
                    },
                    "time_index": {r: inp.tensor_inputs["time_index"] for r, inp in zip(rids, inputs, strict=True)},
                }
            ti = inputs[0].tensor_inputs["time_index"]
            step_index = int(ti.reshape(-1)[0].item())
            self._plan_gen(
                cm, st, st["num_vision"] + st["num_sound"],
                cfg_active=self._cfg_active(st, step_index),
            )
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "sound_latents": inputs[0].tensor_inputs["sound_latents"],
                "time_index": ti,
            }

        if graph_walk in ACTION_WALKS:
            rids = engine_inputs.request_ids
            if len(rids) > 1:
                # Cross-request batch: one batched plan over every request's joint
                # [video | action] block, each with its own page set and token
                # count. A single label when guidance is off (the common
                # guidance-scale-1 case), both labels with classifier-free
                # guidance.
                sts = [self._req[r] for r in rids]
                labels = (
                    [COND_LABEL, UNCOND_LABEL] if sts[0]["uncond"] is not None else [COND_LABEL]
                )
                cm.plan_attention_batched_cfg(
                    labels=labels,
                    seq_lens=[s["num_vision"] + s["num_action"] for s in sts],
                    is_causal=False, write_store=False,
                )
                return {
                    "latents": {r: inp.tensor_inputs["latents"] for r, inp in zip(rids, inputs, strict=True)},
                    "action_latents": {
                        r: inp.tensor_inputs["action_latents"] for r, inp in zip(rids, inputs, strict=True)
                    },
                    "time_index": {r: inp.tensor_inputs["time_index"] for r, inp in zip(rids, inputs, strict=True)},
                }
            self._plan_gen(cm, st, st["num_vision"] + st["num_action"])
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "action_latents": inputs[0].tensor_inputs["action_latents"],
                "time_index": inputs[0].tensor_inputs["time_index"],
            }
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    # Run the prefill/denoise in the model's native bf16, NOT under the engine's
    # autocast. The fused reference pipeline runs the transformer in pure bf16;
    # autocast keeps normalization in fp32, which perturbs the predicted velocity
    # by ~1 ULP per step. A single image step stays well within tolerance, but the
    # multi-step video denoise amplifies that perturbation geometrically into a
    # scrambled latent. The cache-once engine path must reproduce the reference,
    # so this submodule opts out of autocast (the VAE decoder does the same).
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, **kwargs):
        cm = engine_inputs.cache_manager
        rid = engine_inputs.request_ids[0]
        if graph_walk in PREFILL_WALKS:
            return self._forward_prefill(cm, **kwargs)
        if graph_walk in GEN_WALKS:
            return self._forward_image_gen(cm, self._req[rid], **kwargs)
        if graph_walk in SOUND_WALKS:
            return self._forward_video_sound_gen(cm, self._req[rid], **kwargs)
        if graph_walk in ACTION_WALKS:
            return self._forward_action_gen(cm, self._req[rid], **kwargs)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _forward_prefill(
        self, cm,
        input_ids: torch.Tensor,
        text_mrope_ids: torch.Tensor,
        cfg=False,
        **kwargs
    ) -> dict:
        if cfg:
            cm.set_active_label(CFG_BATCHED_LABEL)
        else:
            cm.set_active_label(COND_LABEL)

        self.transformer.prefill_und(input_ids, text_mrope_ids, cm)
        return {}

    def _denoise(self, cm, static, latents, vision_timesteps):
        return self.transformer.denoise_step(
            latents,
            vision_timesteps,
            static["vision_mrope_ids"],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
            cm,
        )

    def _cfg_active(self, st, step_index: int) -> bool:
        """Whether this denoise step runs classifier-free guidance (both
        branches combined). False ⇒ the conditional branch runs alone — the
        guidance_scale==1 case and, for the t2i recipe, steps whose timestep
        falls outside the guidance_interval [lo, hi]. ``preprocess`` and
        ``_forward_image_gen`` both call this for the same step so the planned
        attention (batched vs cond-only) matches the forward that runs."""
        if st["uncond"] is None:
            return False
        gi = st.get("guidance_interval")
        if gi is None:
            return True
        sched = st["scheduler"]
        if step_index >= len(sched.timesteps):
            return False
        t = float(sched.timesteps[step_index].item())
        return gi[0] <= t <= gi[1]

    def _forward_image_gen(self, cm, st, latents, time_index, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        t = scheduler.timesteps[step_index]
        vision_timesteps = torch.full((st["num_noisy"],), t.item(), device=latents.device)

        # Classifier-free guidance is applied only when an uncond branch exists
        # (guidance_scale != 1) and, for the text-to-image recipe, only on the
        # configured timestep interval. Outside the interval the step runs the
        # conditional branch alone (cond-only velocity), matching the recipe.
        cfg_active = self._cfg_active(st, step_index)

        if not cfg_active:
            cm.set_active_label(COND_LABEL)
            velocity = self._denoise(cm, st["cond"], latents, vision_timesteps)
        elif self.batched_cfg:
            cm.set_active_label(CFG_BATCHED_LABEL)
            cond_v, uncond_v = self.transformer.denoise_step_batched_cfg(
                latents,
                vision_timesteps,
                st["cond"]["vision_mrope_ids"],
                st["uncond"]["vision_mrope_ids"],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                cm,
            )
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
        else:
            cm.set_active_label(COND_LABEL)
            cond_v = self._denoise(cm, st["cond"], latents, vision_timesteps)
            cm.set_active_label(UNCOND_LABEL)
            uncond_v = self._denoise(cm, st["uncond"], latents, vision_timesteps)
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)

        new_latents = scheduler.step(
            velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)
        return {"latents": [new_latents], "time_index": [time_index + 1]}

    def _sound_kwargs(self, static, sound_latents, sound_ts) -> dict:
        return dict(
            sound_latents=sound_latents,
            sound_token_shapes=static["sound_token_shapes"],
            sound_noisy_frame_indexes=static["sound_noisy_frame_indexes"],
            sound_mse_gen_indexes=static["sound_mse_gen_indexes"],
            sound_timesteps=sound_ts,
        )

    def _sound_scheduler_step(self, st, latents, sound_latents, video_v, sound_v, t):
        """One joint [video | sound] scheduler step: flatten both bands into one
        packed state so the request's scheduler advances them under a single
        multistep history (the flow update is linear per element), then split
        back. Every sound frame is noisy, so no clean re-injection is needed on
        the sound band; the video band's i2v anchor stays clean because its
        predicted velocity is zero (as in the video-only path)."""
        nv = video_v.numel()
        packed = torch.cat([video_v.reshape(1, -1), sound_v.reshape(1, -1)], dim=1)
        packed_lat = torch.cat([latents.reshape(1, -1), sound_latents.reshape(1, -1)], dim=1)
        packed_next = st["scheduler"].step(packed, t, packed_lat, return_dict=False)[0]
        new_latents = packed_next[:, :nv].reshape(latents.shape)
        new_sound = packed_next[:, nv:].reshape(sound_latents.shape)
        return new_latents, new_sound

    def _forward_video_sound_gen(self, cm, st, latents, sound_latents, time_index, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        t = scheduler.timesteps[step_index]
        vts = torch.full((st["num_noisy"],), t.item(), device=latents.device)
        sts = torch.full((st["num_sound"],), t.item(), device=latents.device)
        cfg_active = self._cfg_active(st, step_index)

        if not cfg_active:
            cm.set_active_label(COND_LABEL)
            velocity, sound_v = self.transformer.denoise_step(
                latents, vts, st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["cond"]["vision_token_shapes"], st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"], cm,
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
        elif self.batched_cfg:
            cm.set_active_label(CFG_BATCHED_LABEL)
            (cond_v, s_c), (uncond_v, s_u) = self.transformer.denoise_step_batched_cfg(
                latents, vts,
                st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["uncond"]["position_ids"][:, st["uncond"]["und_len"]:],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"], cm,
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            sound_v = s_u + st["gs"] * (s_c - s_u)
        else:
            cm.set_active_label(COND_LABEL)
            cond_v, s_c = self.transformer.denoise_step(
                latents, vts, st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["cond"]["vision_token_shapes"], st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"], cm,
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
            cm.set_active_label(UNCOND_LABEL)
            uncond_v, s_u = self.transformer.denoise_step(
                latents, vts, st["uncond"]["position_ids"][:, st["uncond"]["und_len"]:],
                st["uncond"]["vision_token_shapes"], st["uncond"]["vision_noisy_frame_indexes"],
                st["uncond"]["mse_gen_indexes"], cm,
                **self._sound_kwargs(st["uncond"], sound_latents, sts),
            )
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            sound_v = s_u + st["gs"] * (s_c - s_u)

        new_latents, new_sound = self._sound_scheduler_step(
            st, latents, sound_latents, velocity, sound_v, t
        )
        return {
            "latents": [new_latents],
            "sound_latents": [new_sound],
            "time_index": [time_index + 1],
        }

    def _denoise_action(self, cm, static, latents, action_latents, vts, ats, domain):
        und_len = static["und_len"]
        return self.transformer.denoise_step(
            latents,
            vts,
            static["position_ids"][:, und_len:],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
            cm,
            action_latents=action_latents,
            action_token_shapes=static["action_token_shapes"],
            action_noisy_frame_indexes=static["action_noisy_frame_indexes"],
            action_mse_gen_indexes=static["action_mse_gen_indexes"],
            action_timesteps=ats,
            action_domain_id=domain,
        )

    def _action_scheduler_step(self, st, latents, action_latents, video_v, action_v, t):
        """One joint [video | action] scheduler step for an action request: mask
        the predicted velocities to their noisy bands, step the request's own
        scheduler over the packed [video | action] state, then re-inject the clean
        conditioning anchors (conditioning frames / action stay clean each step,
        their masked-in values invariant). Shared by the single-request and
        cross-request batched action forwards."""
        raw, chunk, adim = st["raw_action_dim"], st["action_chunk"], st["action_dim"]
        video_v = video_v * st["velocity_mask"]
        action_v = action_v * st["action_velocity_mask"]
        action_v[..., raw:] = 0
        nv = video_v.numel()
        packed = torch.cat([video_v.reshape(1, -1), action_v.reshape(1, -1)], dim=1)
        packed_lat = torch.cat([latents.reshape(1, -1), action_latents.reshape(1, -1)], dim=1)
        packed_next = st["scheduler"].step(packed, t, packed_lat, return_dict=False)[0]
        new_latents = packed_next[:, :nv].reshape(latents.shape)
        new_action = packed_next[:, nv:].reshape(1, chunk, adim)
        new_latents = st["velocity_mask"] * new_latents + st["vmask"] * latents
        new_action = st["action_velocity_mask"] * new_action + st["action_clean_mask"] * action_latents
        new_action[..., raw:] = 0
        return new_latents, new_action

    def _forward_action_gen(self, cm, st, latents, action_latents, time_index, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        t = scheduler.timesteps[step_index]
        device = latents.device
        vts = torch.full((st["num_noisy"],), t.item(), device=device)
        ats = torch.full((st["num_noisy_action"],), t.item(), device=device)
        domain = st["domain_t"]

        if st["uncond"] is None:
            cm.set_active_label(COND_LABEL)
            video_v, action_v = self._denoise_action(cm, st["cond"], latents, action_latents, vts, ats, domain)
        elif self.batched_cfg:
            cm.set_active_label(CFG_BATCHED_LABEL)
            (video_v, action_v), (v_u, a_u) = self.transformer.denoise_step_batched_cfg(
                latents,
                vts,
                st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["uncond"]["position_ids"][:, st["uncond"]["und_len"]:],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                cm,
                action_latents=action_latents,
                action_token_shapes=st["cond"]["action_token_shapes"],
                action_noisy_frame_indexes=st["cond"]["action_noisy_frame_indexes"],
                action_mse_gen_indexes=st["cond"]["action_mse_gen_indexes"],
                action_timesteps=ats,
                action_domain_id=domain,
            )
            video_v = v_u + st["gs"] * (video_v - v_u)
            action_v = a_u + st["gs"] * (action_v - a_u)
        else:
            cm.set_active_label(COND_LABEL)
            video_v, action_v = self._denoise_action(cm, st["cond"], latents, action_latents, vts, ats, domain)
            cm.set_active_label(UNCOND_LABEL)
            v_u, a_u = self._denoise_action(cm, st["uncond"], latents, action_latents, vts, ats, domain)
            video_v = v_u + st["gs"] * (video_v - v_u)
            action_v = a_u + st["gs"] * (action_v - a_u)

        new_latents, new_action = self._action_scheduler_step(
            st, latents, action_latents, video_v, action_v, t
        )
        return {
            "latents": [new_latents],
            "action_latents": [new_action],
            "time_index": [time_index + 1],
        }

    # ------------------------------------------------------------------
    # Cross-request batching: run several requests' denoise step together.
    # ------------------------------------------------------------------

    def can_batch(self, batch, model_inputs) -> bool:
        # The denoise step batches across concurrent requests at the same walk.
        # The batched forward packs each request's own token shapes, so requests
        # at different resolutions / frame counts (and, for action, different
        # modes / embodiment domains) can share the batch. One request stays on
        # the simpler single-request path.
        if not self.batched_cfg or len(batch.request_ids) < 2:
            return False
        sts = [self._req.get(rid) for rid in batch.request_ids]
        if any(st is None for st in sts):
            return False
        if (
            batch.graph_walk in GEN_WALKS
            or batch.graph_walk in SOUND_WALKS
            or batch.graph_walk in PREFILL_WALKS
        ):
            # Image/video batch only in the two-branch guidance regime, so one
            # batched-CFG plan covers them. (Batches are per graph walk, so
            # sound requests only ever batch with sound requests.)
            return all(st["uncond"] is not None for st in sts)
        if batch.graph_walk in ACTION_WALKS:
            # Action batches when all requests share the guidance regime (all
            # single-branch -- guidance-scale-1 inverse/forward-dynamics and base
            # policy -- or all two-branch), so one plan covers the batch. Modes
            # and embodiment domains may differ: each request's masks, scheduler
            # and domain-aware action projection are applied per request.
            return len({st["uncond"] is not None for st in sts}) == 1
        return False

    def max_batch_size(self, graph_walk: str):
        if graph_walk in GEN_WALKS or graph_walk in SOUND_WALKS or graph_walk in ACTION_WALKS:
            return self.max_gen_batch_size
        return None

    # Native bf16, not the engine autocast — see the note on forward(). The
    # cross-request batched denoise must match the per-request path exactly.
    @torch.autocast(device_type="cuda", enabled=False)
    def forward_batched(
        self, graph_walk, engine_inputs: ModelInputsFromEngine,
        latents=None, time_index=None, action_latents=None, sound_latents=None,
        input_ids=None, text_mrope_ids=None, **kwargs,
    ):
        cm = engine_inputs.cache_manager
        if graph_walk in PREFILL_WALKS:
            if kwargs.get("cfg"):
                cm.set_active_label(CFG_BATCHED_LABEL)
            else:
                cm.set_active_label(COND_LABEL)

            self.transformer.prefill_und(input_ids, text_mrope_ids, cm)
            return {}
        if graph_walk in ACTION_WALKS:
            return self._forward_batched_action(engine_inputs, latents, action_latents, time_index)
        if graph_walk in SOUND_WALKS:
            return self._forward_batched_sound(engine_inputs, latents, sound_latents, time_index)
        if graph_walk not in GEN_WALKS:
            raise ValueError(f"Cosmos3 batched forward only supports generation walks, got {graph_walk!r}")
        cm.set_active_label(CFG_BATCHED_LABEL)
        reqs, meta = [], []
        for rid in engine_inputs.request_ids:
            st = self._req[rid]
            lat, ti = latents[rid], time_index[rid]
            step_index = int(ti.reshape(-1)[0].item())
            n_steps = len(st["scheduler"].timesteps)
            # A request may be one step past its denoise count (a discarded extra
            # step) while others in the batch are still running; clamp its
            # timestep so the shared forward can't index past the schedule, and
            # skip its scheduler step below.
            t = st["scheduler"].timesteps[min(step_index, n_steps - 1)]
            reqs.append({
                "latents": lat,
                "vision_timesteps": torch.full((st["num_noisy"],), t.item(), device=lat.device),
                "position_ids_cond": st["cond"]["vision_mrope_ids"],
                "position_ids_uncond": st["uncond"]["vision_mrope_ids"],
                "vision_token_shapes": st["cond"]["vision_token_shapes"],
                "vision_noisy_frame_indexes": st["cond"]["vision_noisy_frame_indexes"],
                "vision_mse_loss_indexes": st["cond"]["mse_gen_indexes"],
            })
            meta.append((rid, st, lat, ti, t))

        results = self.transformer.denoise_step_batched(reqs, cm)

        out = {}
        for (rid, st, lat, ti, t), (cond_v, uncond_v) in zip(meta, results, strict=True):
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            new_latents = st["scheduler"].step(
                velocity.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            out[rid] = {"latents": [new_latents], "time_index": [ti + 1]}
        return out

    def _forward_batched_sound(self, engine_inputs, latents, sound_latents, time_index):
        """Run several sound requests' joint [video | sound] denoise step in one
        forward. Mirrors the image batched path with per-request sound bands:
        one batched transformer pass, then per request the guidance combine and
        its own joint [video | sound] scheduler step."""
        cm = engine_inputs.cache_manager
        cm.set_active_label(CFG_BATCHED_LABEL)
        reqs, meta = [], []
        for rid in engine_inputs.request_ids:
            st = self._req[rid]
            lat, snd, ti = latents[rid], sound_latents[rid], time_index[rid]
            step_index = int(ti.reshape(-1)[0].item())
            n_steps = len(st["scheduler"].timesteps)
            # A request may be one (discarded) step past its denoise count while
            # others in the batch are still running; clamp its timestep so the
            # shared forward can't index past the schedule, and skip its
            # scheduler step below.
            t = st["scheduler"].timesteps[min(step_index, n_steps - 1)]
            cond, unc = st["cond"], st["uncond"]
            reqs.append({
                "latents": lat,
                "vision_timesteps": torch.full((st["num_noisy"],), t.item(), device=lat.device),
                "position_ids_cond": cond["position_ids"][:, cond["und_len"]:],
                "position_ids_uncond": unc["position_ids"][:, unc["und_len"]:],
                "vision_token_shapes": cond["vision_token_shapes"],
                "vision_noisy_frame_indexes": cond["vision_noisy_frame_indexes"],
                "vision_mse_loss_indexes": cond["mse_gen_indexes"],
                "sound_latents": snd,
                "sound_timesteps": torch.full((st["num_sound"],), t.item(), device=lat.device),
                "sound_token_shapes": cond["sound_token_shapes"],
                "sound_noisy_frame_indexes": cond["sound_noisy_frame_indexes"],
                "sound_mse_gen_indexes": cond["sound_mse_gen_indexes"],
            })
            meta.append((rid, st, lat, snd, ti, t))

        results = self.transformer.denoise_step_batched(reqs, cm)

        out = {}
        for (rid, st, lat, snd, ti, t), ((cond_v, s_c), (uncond_v, s_u)) in zip(meta, results, strict=True):
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            sound_v = s_u + st["gs"] * (s_c - s_u)
            new_latents, new_sound = self._sound_scheduler_step(st, lat, snd, velocity, sound_v, t)
            out[rid] = {
                "latents": [new_latents],
                "sound_latents": [new_sound],
                "time_index": [ti + 1],
            }
        return out

    def _forward_batched_action(self, engine_inputs, latents, action_latents, time_index):
        """Run several action requests' joint [video | action] denoise step in one
        forward. Mirrors the image batched path: build each request's static gen
        inputs (clamping a request that has run one step past its denoise count),
        run one batched transformer pass, then per request combine the guidance
        branches (when present) and apply its own joint scheduler step."""
        cm = engine_inputs.cache_manager
        cm.set_active_label(CFG_BATCHED_LABEL)
        rids = engine_inputs.request_ids
        with_cfg = self._req[rids[0]]["uncond"] is not None
        reqs, meta = [], []
        for rid in rids:
            st = self._req[rid]
            lat, act, ti = latents[rid], action_latents[rid], time_index[rid]
            step_index = int(ti.reshape(-1)[0].item())
            n_steps = len(st["scheduler"].timesteps)
            # A request may be one (discarded) step past its denoise count while
            # others in the batch are still running; clamp its timestep so the
            # shared forward can't index past the schedule, and skip its scheduler
            # step below.
            t = st["scheduler"].timesteps[min(step_index, n_steps - 1)]
            cond = st["cond"]
            und = cond["und_len"]
            req = {
                "latents": lat,
                "action_latents": act,
                "vision_timesteps": torch.full((st["num_noisy"],), t.item(), device=lat.device),
                "action_timesteps": torch.full((st["num_noisy_action"],), t.item(), device=lat.device),
                "position_ids_cond": cond["position_ids"][:, und:],
                "vision_token_shapes": cond["vision_token_shapes"],
                "vision_noisy_frame_indexes": cond["vision_noisy_frame_indexes"],
                "vision_mse_loss_indexes": cond["mse_gen_indexes"],
                "action_token_shapes": cond["action_token_shapes"],
                "action_noisy_frame_indexes": cond["action_noisy_frame_indexes"],
                "action_mse_gen_indexes": cond["action_mse_gen_indexes"],
                "action_domain_id": st["domain_t"],
            }
            if with_cfg:
                unc = st["uncond"]
                req["position_ids_uncond"] = unc["position_ids"][:, unc["und_len"]:]
            reqs.append(req)
            meta.append((rid, st, lat, act, ti, t))

        results = self.transformer.denoise_step_action_batched(reqs, cm, with_cfg)

        out = {}
        for (rid, st, lat, act, ti, t), branches in zip(meta, results, strict=True):
            if with_cfg:
                (cond_video, cond_action), (uncond_video, uncond_action) = branches
                video_v = uncond_video + st["gs"] * (cond_video - uncond_video)
                action_v = uncond_action + st["gs"] * (cond_action - uncond_action)
            else:
                (video_v, action_v), = branches
            new_latents, new_action = self._action_scheduler_step(st, lat, act, video_v, action_v, t)
            out[rid] = {
                "latents": [new_latents],
                "action_latents": [new_action],
                "time_index": [ti + 1],
            }
        return out

    # ------------------------------------------------------------------
    # CUDA-graph capture of the denoise step. Only the transformer velocity
    # computation is captured; the guidance combine and the (Python, multistep)
    # scheduler step run eagerly afterwards.
    # ------------------------------------------------------------------

    def get_cuda_graph_configs(self, device, tp_world_size: int = 1):
        """Declare one fixed-shape capture of the image denoise step per
        resolution. Requests at other resolutions, or without guidance, fall back
        to the eager path. The per-resolution token layout is prompt-independent,
        so bake it once here and key it by latent shape; the per-prompt rotary
        positions, the latents and the timestep flow in as static-buffer inputs.

        Set ``COSMOS3_DISABLE_CUDA_GRAPH=1`` to skip capture and run the denoise
        loop eagerly (escape hatch for a misbehaving driver, and an A/B switch).
        Set ``COSMOS3_GEN_CAPTURE_RES`` (e.g. ``"192x320,480x832"``, height x
        width) to override which resolutions are captured, and
        ``COSMOS3_GEN_CAPTURE_BS`` (e.g. ``"1,4,8"``) to also capture batched
        denoise steps so concurrent requests replay a padded graph instead of
        falling back to the eager path."""
        disable_env = os.environ.get("COSMOS3_DISABLE_CUDA_GRAPH")
        disabled = bool(disable_env) if disable_env is not None else not self.config.cuda_graph
        if self.transformer is None or disabled:
            return []
        res_env = os.environ.get("COSMOS3_GEN_CAPTURE_RES")
        if res_env:
            resolutions = tuple(
                tuple(int(x) for x in pair.split("x")) for pair in res_env.split(",")
            )
        else:
            resolutions = self.gen_capture_resolutions
        bs_env = os.environ.get("COSMOS3_GEN_CAPTURE_BS")
        if bs_env:
            capture_batch_sizes = [int(x) for x in bs_env.split(",")]
        else:
            capture_batch_sizes = list(self.gen_capture_batch_sizes)
        dtype = self.transformer.proj_in.weight.dtype
        # The 2D (TP x SP) captured denoise graph hits a CUDA illegal access at
        # 480p+ in the combined Ulysses all-gather + TP all-reduce path (pure-TP and
        # pure-SP capture cleanly; only the combination at this scale). For 2D,
        # capture only the smallest tier; 480p+ runs eager (correct, just no graph).
        two_d = self.transformer.sp_group.world_size > 1 and self.transformer.comm_group.world_size > 1
        self._capture_layout: dict[tuple, dict] = {}
        configs = []
        for height, width in resolutions:
            latent_shape = self._latent_shape(height, width, num_frames=1)
            # The denoise CUDA-graph capture is bit-faithful at every resolution
            # (the rotary outer product is a broadcast multiply, not a batched
            # matmul over stride-0 views — see Cosmos3RotaryEmbedding; the matmul
            # form was mis-captured by the graph memory pool at some sequence
            # lengths). The graph only HELPS the launch-bound tiers, though: its
            # per-step input copies scale with resolution, so at large latents
            # (720p+) the graph is net-slower than the eager dense path. So
            # capture only below COSMOS3_GRAPH_MAX_LATENT_AREA (latent H*W): 256p
            # (240) and 480p (1560) graph; 720p (3600) and video run eager+dense.
            latent_area = latent_shape[3] * latent_shape[4]
            max_area = int(os.environ.get(
                "COSMOS3_GRAPH_MAX_LATENT_AREA", self.config.graph_max_latent_area))
            if two_d:
                max_area = min(max_area, 1000)  # 256p latent 240 captures; 480p 1560 does not
            if latent_area > max_area:
                logger.info(
                    "Cosmos3: skipping CUDA-graph capture for %dx%d (latent H*W "
                    "%d > %d -> graph net-slower than eager dense here -> eager)",
                    height, width, latent_area, max_area,
                )
                continue
            static = self._build_static(
                [0] * 8, height, width, num_frames=1, fps=24.0,
                has_image_condition=False, device=device,
            )
            num_vision = static["num_vision_tokens"]
            num_noisy = static["num_noisy_vision_tokens"]
            self._capture_layout[tuple(latent_shape)] = {
                "vision_token_shapes": static["vision_token_shapes"],
                "vision_noisy_frame_indexes": static["vision_noisy_frame_indexes"],
                "mse_gen_indexes": static["mse_gen_indexes"],
            }
            single = ARNodeInputs(
                input_seq_len=num_vision,
                tensor_inputs={
                    "latents": torch.zeros(latent_shape, device=device, dtype=dtype),
                    "vision_timesteps": torch.zeros(num_noisy, device=device, dtype=torch.float32),
                    "position_ids_cond": static["vision_mrope_ids"].clone(),
                    "position_ids_uncond": static["vision_mrope_ids"].clone(),
                },
            )
            configs.append(BasicBatchedCudaGraphConfig(
                capture_graph_walk=IMAGE_GEN_WALK,
                single_request_inputs=single,
                requires_cfg=False,
                labels=[COND_LABEL, UNCOND_LABEL],
                capture_forward_method="forward_captured",
                advance_seq_lens=False,
                compile=False,
                capture_batch_sizes=capture_batch_sizes,
                # The captured sizes (default just bs=1, for single-request
                # latency; COSMOS3_GEN_CAPTURE_BS adds batched sizes) are an
                # acceleration subset, not a batch ceiling: a concurrent batch at
                # an uncaptured size or mixed resolution still runs the eager
                # batched denoise (forward_batched), so don't let this capture cap
                # max_batch_size to the captured sizes.
                caps_eager_batch_size=False,
            ))

        # Understanding-tower text prefill: cond+uncond packed into one combined
        # sequence (batched CFG). The dummy zeros are placeholders — the real
        # input_ids / mrope ids are copied into the static buffers at replay.
        if not os.environ.get("COSMOS3_DISABLE_PREFILL_CUDA_GRAPH"):
            tok_env = os.environ.get("COSMOS3_PREFILL_CAPTURE_TOKENS")
            prefill_tokens = (
                [int(x) for x in tok_env.split(",")] if tok_env
                else list(self.prefill_capture_token_buckets)
            )
            bs_env = os.environ.get("COSMOS3_PREFILL_CAPTURE_BS")
            prefill_bs = (
                [int(x) for x in bs_env.split(",")] if bs_env
                else list(self.prefill_capture_batch_sizes)
            )
            mrope_dtype = torch.float32 if self.config.enable_fps_modulation else torch.long

            def _prefill_template(n):
                return {
                    "input_ids": torch.zeros(n, dtype=torch.long, device=device),
                    "text_mrope_ids": torch.zeros((3, n), dtype=mrope_dtype, device=device),
                    "cfg": True,
                }

            configs.append(FlashInferPackedCudaGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                replay_graph_walks=list(PREFILL_WALKS),
                packed_seq_len_to_inputs={n: _prefill_template(n) for n in prefill_tokens},
                requires_cfg=False,
                labels=[COND_LABEL, UNCOND_LABEL],
                batched_cfg=True,
                causal_attention=True,
                compile=False,
                capture_batch_sizes=prefill_bs,
                caps_eager_batch_size=False,
                zero_padding_input=ARNodeInputs(
                    input_seq_len=0,
                    tensor_inputs={
                        "input_ids_" + COND_LABEL: torch.zeros(0, dtype=torch.long, device=device),
                        "input_ids_" + UNCOND_LABEL: torch.zeros(0, dtype=torch.long, device=device),
                        "text_mrope_ids_" + COND_LABEL: torch.zeros((3, 0), dtype=mrope_dtype, device=device),
                        "text_mrope_ids_" + UNCOND_LABEL: torch.zeros((3, 0), dtype=mrope_dtype, device=device),
                    },
                    kwargs=dict(
                        cfg=True,
                        seq_lens={COND_LABEL: 0, UNCOND_LABEL: 0},
                    ),
                ),
            ))
        return configs

    def can_use_cuda_graphs(self, batch, model_inputs) -> bool:
        # Text prefill is captured only with two-branch guidance (the combined
        # cond+uncond packed sequence); the runner's can_run picks the token
        # bucket, so a prompt past the largest bucket falls back to eager here.
        if batch.graph_walk in PREFILL_WALKS:
            return all(
                (st := self._req.get(rid)) is not None and st["uncond"] is not None
                for rid in batch.request_ids
            )
        # Only the image denoise step is captured, only with two-branch guidance,
        # and only at a resolution we captured a graph for. A batched capture is a
        # single fixed resolution, so a concurrent batch must be uniform-resolution
        # to share one captured (batch size, token count) bucket; mixed-resolution
        # batches fall back to the eager cross-request denoise.
        if batch.graph_walk != IMAGE_GEN_WALK:
            return False
        layout = getattr(self, "_capture_layout", None)
        if not layout:
            return False
        shapes = set()
        for rid in batch.request_ids:
            st = self._req.get(rid)
            if st is None or st["uncond"] is None:
                return False
            shape = tuple(st["latent_shape"])
            if shape not in layout:
                return False
            shapes.add(shape)
        return len(shapes) == 1

    def forward_captured(
        self, graph_walk, engine_inputs: ModelInputsFromEngine,
        latents, vision_timesteps, position_ids_cond, position_ids_uncond, **kwargs,
    ) -> dict:
        """Velocity-only denoise forward captured into a CUDA graph: both guidance
        branches in one pass (the combined plan), no scheduler step. The token
        layout is baked per resolution; the latents, timestep and rotary positions
        are static-buffer inputs stacked on a leading batch dim. A single request
        keeps the two-branch path; a concurrent batch runs the per-request denoise
        (the same compute as the eager cross-request forward), one transformer pass
        over the whole batch."""
        cm = engine_inputs.cache_manager
        cm.set_active_label(CFG_BATCHED_LABEL)
        layout = self._capture_layout[tuple(latents.shape[1:])]
        rids = engine_inputs.request_ids
        if latents.shape[0] == 1:
            # This forward is captured into the denoise CUDA graph. Under SP the
            # Ulysses exchange must use all-gather, not all-to-all: the latter is
            # grouped point-to-point send/recv and does not replay from a graph.
            # The flag holds across warmup, capture and replay (all routed here),
            # so the all-gather kernels are compiled during eager warmup rather
            # than mid-capture. No-op without SP.
            cond_v, uncond_v = self.transformer.denoise_step_batched_cfg(
                latents[0], vision_timesteps[0], position_ids_cond[0], position_ids_uncond[0],
                layout["vision_token_shapes"], layout["vision_noisy_frame_indexes"],
                layout["mse_gen_indexes"], cm, prefer_all_gather=True,
            )
            return {rids[0]: {"cond_v": [cond_v], "uncond_v": [uncond_v]}}
        reqs = [
            {
                "latents": latents[i],
                "vision_timesteps": vision_timesteps[i],
                "position_ids_cond": position_ids_cond[i],
                "position_ids_uncond": position_ids_uncond[i],
                "vision_token_shapes": layout["vision_token_shapes"],
                "vision_noisy_frame_indexes": layout["vision_noisy_frame_indexes"],
                "vision_mse_loss_indexes": layout["mse_gen_indexes"],
            }
            for i in range(latents.shape[0])
        ]
        results = self.transformer.denoise_step_batched(reqs, cm)
        return {
            rid: {"cond_v": [cond_v], "uncond_v": [uncond_v]}
            for rid, (cond_v, uncond_v) in zip(rids, results, strict=True)
        }

    def postprocess_captured(
        self, request_ids: list[str],
        inputs: list,
        per_request_info: dict[str, CurrentForwardPassInfo],
        outputs: dict
    ) -> dict:
        """Eager tail run after graph replay: the classifier-free-guidance combine
        and the (Python, multistep) scheduler step the graph can't hold. Mirrors
        the tail of ``_forward_image_gen``."""
        if per_request_info[request_ids[0]].graph_walk not in GEN_WALKS:
            return outputs
        for rid, inp in zip(request_ids, inputs, strict=True):
            st = self._req[rid]
            cond_v = outputs[rid]["cond_v"][0]
            uncond_v = outputs[rid]["uncond_v"][0]
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            latents = inp.tensor_inputs["latents"]
            time_index = inp.tensor_inputs["time_index"]
            step_index = int(time_index.reshape(-1)[0].item())
            sched = st["scheduler"]
            sched_idx = sched.step_index
            if step_index >= len(sched.timesteps) or (
                sched_idx is not None and sched_idx >= len(sched.timesteps)
            ):
                # Discarded extra step past this request's denoise count. Guard on
                # BOTH the engine step counter and the scheduler's own step_index:
                # with concurrent (bs>1) requests the batched loop dispatches one
                # extra step whose engine time_index can lag the scheduler, and
                # stepping an already-exhausted UniPC scheduler trips
                # `assert self.this_order > 0`.
                outputs[rid] = {"latents": [latents], "time_index": [time_index]}
                continue
            t = st["scheduler"].timesteps[step_index]
            new_latents = st["scheduler"].step(
                velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            outputs[rid] = {"latents": [new_latents], "time_index": [time_index + 1]}
        return outputs

    def check_stop(self, request_id, request_info, outputs) -> set[str]:
        """Stop this request's denoise loop once it has run its own step count.

        The loop is built with a fixed upper-bound iteration count
        (``config.max_inference_steps``); each request runs only as many steps as
        its scheduler holds (e.g. image 50, video 35, action 30, distilled policy
        ~4), which can differ between concurrent requests. Runs on the worker's
        slow-postprocess path, so reading the per-request step count is fine. The
        one extra step the loop dispatches before this stop takes effect is a
        no-op (see the ``step_index >=`` guards in the forward methods)."""
        st = self._req.get(request_id)
        if st is None:
            return set()
        loop = {
            ACTION_GEN_WALK: ACTION_GEN_LOOP,
            ACTION_VIDEO_GEN_WALK: ACTION_VIDEO_GEN_LOOP,
            VIDEO_GEN_WALK: VIDEO_GEN_LOOP,
            VIDEO_SOUND_GEN_WALK: VIDEO_SOUND_GEN_LOOP,
        }.get(request_info.graph_walk, IMAGE_GEN_LOOP)
        iter_idx = request_info.dynamic_loop_iter_counts.get(loop, 0)
        if iter_idx + 1 >= len(st["scheduler"].timesteps):
            return {loop}
        return set()

    def cleanup_request(self, request_id: str):
        self._req.pop(request_id, None)


class Cosmos3AudioDecoderSubmodule(NodeSubmodule):
    """AVAE sound decode node (STATELESS): final denoised sound latents
    ``[1, C, T]`` -> stereo waveform ``[channels, samples]`` in [-1, 1],
    trimmed to the request's target sample count.

    The target is re-derived from the request metadata (duration x sample rate)
    rather than read from the DiT node's per-request state, so this node stays
    stateless and placeable on any rank."""

    disable_torch_compile = True

    def __init__(self, sound_tokenizer, config):
        super().__init__()
        self.sound_tokenizer = sound_tokenizer
        self.config = config
        if sound_tokenizer is not None and sound_tokenizer.sample_rate != config.sound_sample_rate:
            raise ValueError(
                f"Cosmos3 sound tokenizer sample rate {sound_tokenizer.sample_rate} does not match "
                f"config.sound_sample_rate {config.sound_sample_rate}; the packed sound band length "
                "would disagree with the decoded audio length."
            )

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        md = fwd_info.step_metadata
        duration = md.get("sound_duration")
        if duration is None:
            duration = int(md.get("num_frames", 1)) / float(md.get("fps", 24.0))
        duration = max(float(duration), 1.0 / max(float(md.get("fps", 24.0)), 1.0))
        target = max(1, int(round(duration * self.config.sound_sample_rate)))
        return NodeInputs(tensor_inputs={
            "sound_latents": inputs["sound_latents"][0],
            "target_samples": torch.tensor([target], dtype=torch.long),
        })

    # Native dtype, not the engine autocast (the tokenizer runs in its own bf16;
    # matching the decode dtype keeps the waveform deterministic across walks).
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, sound_latents, target_samples, **kwargs):
        audio = self.sound_tokenizer.decode(sound_latents)  # [1, channels, T * hop]
        target = int(target_samples.reshape(-1)[0].item())
        if audio.shape[-1] > target:
            audio = audio[..., :target]
        elif audio.shape[-1] < target:
            audio = torch.nn.functional.pad(audio, (0, target - audio.shape[-1]))
        return {"audio_output": [audio[0].to(torch.float32)]}


class Cosmos3VAEDecoderSubmodule(NodeSubmodule):
    """Wan VAE decode node: final denoised latents -> pixel frames.

    Applies the pipeline-side latent normalization (the VAE itself returns raw
    latents) before decoding, matching the fused t2i pipeline's decode.
    """

    # One-shot decode per request; CUDA-graph capture (not torch.compile) is the
    # speedup path.
    disable_torch_compile = True

    def __init__(self, vae, config):
        super().__init__()
        self.vae = vae
        self.config = config
        self._decode_dtype_cached = None
        # The Wan VAE decode is 3D-conv bound and is not captured into a CUDA
        # graph (it runs once per request at request-specific frame/resolution
        # shapes). torch.compile fuses the pointwise epilogues around those convs;
        # fullgraph=False lets dynamo break around the VAE's Python-level
        # causal-conv feature cache, and dynamic=False gives the best per-shape
        # kernels at the cost of a one-time trace per new (frames, height, width)
        # — fine for the few fixed generation tiers (the first request at each
        # shape pays the trace). On by default — it cuts the 189-frame video VAE
        # decode ~20-30% (a large share of video latency) and is neutral for the
        # tiny single-frame image decode; set COSMOS3_COMPILE_VAE=0 to disable
        # (A/B against the eager decode, which is identical bar fp rounding). The
        # compile wraps the autocast-off decode below.
        self._decode = vae.decode if vae is not None else None
        if vae is not None and os.environ.get("COSMOS3_COMPILE_VAE", "1").lower() not in (
            "0", "false", "no", "off",
        ):
            self._decode = torch.compile(vae.decode, fullgraph=False, dynamic=False)
            logger.info("Cosmos3 VAE decode torch.compile enabled")
        # Resolve + log the decode dtype now (a cheap cuDNN-version read) so the
        # choice is fixed at startup, not on the first request.
        self._decode_dtype()

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={"latents": inputs["latents"][0]})

    def _decode_dtype(self):
        # cuDNN ships the fast Hopper bf16 conv3d kernels for the Wan-VAE decode
        # only from 9.16 on; before that (e.g. the 9.10 cu12.8 wheel) bf16/fp16
        # fall back and a 256p-video decode is ~10s vs ~1s in fp32/TF32, so use
        # fp32 there. Measured single-variable (same torch/GPU, swapping only the
        # cuDNN libs): bf16 vid256 decode 10.3s at 9.13 -> 0.85s at 9.16; fp32/TF32
        # ~1.1s throughout. Gate on the live cuDNN version (a cuDNN upgrade flips
        # it automatically); COSMOS3_VAE_DECODE_FP32 overrides.
        if self._decode_dtype_cached is not None:
            return self._decode_dtype_cached
        override = os.environ.get("COSMOS3_VAE_DECODE_FP32")
        if override is not None:
            on = override.lower() not in ("0", "false", "no", "off")
            self._decode_dtype_cached = torch.float32 if on else torch.bfloat16
        else:
            ver = torch.backends.cudnn.version() or 0
            self._decode_dtype_cached = torch.bfloat16 if ver >= 91600 else torch.float32
        logger.info("Cosmos3 VAE decode dtype = %s (cuDNN %s)",
                    self._decode_dtype_cached, torch.backends.cudnn.version())
        return self._decode_dtype_cached

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, latents, **kwargs):
        vae = self.vae
        vae_dtype = self._decode_dtype()
        if next(vae.parameters()).dtype != vae_dtype:
            vae = vae.to(vae_dtype)
        mean = torch.tensor(vae.config.latents_mean, dtype=vae_dtype, device=latents.device).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=vae_dtype, device=latents.device)).view(
            1, -1, 1, 1, 1
        )
        # Force z to the VAE dtype. An outer autocast(enabled=True) context is
        # active during this forward and promotes the `1.0 / std` division to
        # fp32, which would silently make z fp32 even though latents/mean/vae_dtype
        # are all bf16 -> the bf16 VAE conv3d below (run under autocast-off) would
        # then hit "Input type (float) and bias type (BFloat16)". The explicit cast
        # is a no-op for the fp32 path (S1) and unblocks bf16 decode (cuDNN >= 9.16).
        z = (latents.to(vae_dtype) / inv_std + mean).to(vae_dtype)
        with torch.autocast(device_type="cuda", enabled=False):
            decoded = self._decode(z).sample  # [1, 3, T, H, W] in [-1, 1]
        # Quantize to 8-bit here (the output is an 8-bit image/mp4 either way) so
        # only the uint8 frames cross the SHM edge to the data worker, not a 4x
        # larger fp32 tensor — the decoded video transfer dominates the fixed cost
        # at higher resolutions.
        image = (decoded / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)
        # Route the decoded tensor to the active walk's emit edge: image_gen
        # emits "image_output" (one frame); the video walks (plain, sound,
        # forward-dynamics) emit "video_output".
        out_name = (
            "video_output"
            if graph_walk in (VIDEO_GEN_WALK, VIDEO_SOUND_GEN_WALK, ACTION_VIDEO_GEN_WALK)
            else "image_output"
        )
        return {out_name: [image]}
