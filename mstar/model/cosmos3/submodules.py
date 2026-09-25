"""NodeSubmodule wrappers for the Cosmos3 generator nodes.

Nodes:
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
  Cosmos3VAEEncoderSubmodule  -- Wan VAE conditioning encode (STATELESS): the
                                 request's conditioning image/video to clean
                                 anchor latents, in parallel with the DiT
                                 prefill.
  Cosmos3VAEDecoderSubmodule  -- Wan VAE decode (STATELESS): final latents to
                                 pixels.
  Cosmos3VAEDecoderARSubmodule -- streaming Wan VAE decode (STATELESS) for
                                 windowed AR video: one committed window's
                                 latents per stream chunk, decoded behind
                                 re-decoded context and emitted per window or
                                 assembled.
  Cosmos3VisionEncoderSubmodule -- Edge reasoner vision tower (STATELESS):
                                 packed image/video patches to text-space
                                 tokens for the reasoner prefill.
  Cosmos3ReasonerSubmodule    -- the understanding tower as a causal VLM
                                 (shares the DiT's transformer instance and
                                 kv/attn resources, plus its own sampler):
                                 ``reasoner_prefill`` / ``reasoner_prefill_vision``
                                 write the prompt's K/V and sample the first
                                 token, ``reasoner_decode`` one token per loop
                                 iteration.

Because the text tokens never receive a timestep embedding, the understanding
K/V is denoise-step independent, so writing it once and re-reading it every step
matches running the whole transformer each step.
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    PackedCudaGraphConfig,
)
from mstar.engine.resources import AttentionStep, KVStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.windowing import WindowedKVSession, WindowSchedule
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
    REASONER_DECODE_WALK,
    REASONER_PREFILL_VISION_WALK,
    REASONER_PREFILL_WALK,
    VIDEO_GEN_AR_WALK,
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
# walk, so the submodule treats them the same. video_gen_ar runs the same step
# too, over one window's latents at a time, with per-window state swaps at
# window boundaries (and, in kv mode, one commit iteration per window).
GEN_WALKS = (IMAGE_GEN_WALK, VIDEO_GEN_WALK, VIDEO_GEN_AR_WALK)

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
VIDEO_GEN_AR_LOOP = "video_gen_ar_loop"
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

# Combined plan key for the single plan that runs both guidance branches in
# one forward (``KVStep.combined_labels`` in the step declaration).
CFG_BATCHED_LABEL = "_cfg_batched"

# Resource labels the DiT node declares (Cosmos3Model.get_node_resources) and
# keys its resource steps by. ATTN is the paged backend: the understanding
# tower's prefill writes its K/V through it, and a captured denoise graph
# replays against it. ATTN_GEN is the dense backend the eager denoise steps
# use, where the paged path's per-step K/V write and wrapper plan are pure
# overhead — the model declares it only when the config asks for it.
# The reasoner node shares KV_CACHE and ATTN (same weights, same pool) and
# adds SAMPLER for its token sampling.
KV_CACHE = "kv"
ATTN = "attn"
ATTN_GEN = "attn_gen"
SAMPLER = "sampler"

# The reasoner's walks and its decode loop.
REASONER_PREFILL_WALKS = (REASONER_PREFILL_WALK, REASONER_PREFILL_VISION_WALK)
REASONER_DECODE_LOOP = "reasoner_decode_loop"
# The reasoner keeps every request's text under one cache label.
REASONER_LABEL = "main"


def native_flow_sigmas(num_inference_steps: int, num_train_timesteps: int):
    """The native flow-matching sigma grid: ``num_inference_steps`` values
    linearly spaced from ``1 - 1/T`` toward 0, the endpoint dropped. A numpy
    array: UniPC's explicit-sigma path applies the flow shift arithmetically
    to it (a Python list raises inside ``set_timesteps``)."""
    import numpy as np

    return np.linspace(1.0 - 1.0 / num_train_timesteps, 0.0, num_inference_steps + 1)[:-1]


@dataclass(frozen=True)
class GenStepInfo:
    """What a denoise step's declaration needs and cannot read off the batch.

    All three are per-request facts resolved in ``prepare_inputs`` (which has
    the request's state and its step index) and carried to ``declare_step`` on
    ``NodeInputs.resource_step_info``, since the declaration runs with only
    the request ids and their prepared inputs.

    ``capture_key`` is this request's half of ``cg_key_info`` — the capture
    bucket its shape belongs to, or None for "no captured graph for this".
    Agreeing with ``cg_key_info`` matters: the engine leases the slot from
    that and then checks the declaration against the lease. What is left over
    is the batch's size and token count, which no row knows, so a step every
    row of which is capturable may still find no bucket and run eager.
    Declaring the paged backend for such a step only costs the dense path's
    speedup; declaring the dense one for a step that does get a slot would try
    to capture an eager-only backend, so the safe direction over-declares.

    ``cfg_active`` narrows the eager single-request declaration only. It is
    deliberately not part of ``capture_key`` — see ``cg_key_info``.
    """
    cfg: bool                 # this request has an unconditional branch at all
    cfg_active: bool          # ...and this step is inside its guidance interval
    capture_key: object | None  # this request's capture bucket, if any
    # A kv-mode windowed commit iteration: the span is appended to every live
    # branch and committed (paged, non-causal) instead of recomputed and
    # dropped — see ``Cosmos3DiTSubmodule._commit_step``.
    commit: bool = False


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

    # Cap on how many concurrent requests share one batched denoise step.
    max_gen_batch_size = 8

    # t2i (num_frames=1) resolutions to capture a bs=1 denoise-step CUDA graph
    # for; others run eager. The graph removes launch overhead (biggest at low
    # resolution) and replays identically to eager. Override with
    # COSMOS3_GEN_CAPTURE_RES; concurrent requests batch via the eager path.
    gen_capture_resolutions: tuple[tuple[int, int], ...] = (
        (192, 320), (480, 832), (720, 1280),
    )
    # Batch sizes to capture per resolution.
    gen_capture_batch_sizes: tuple[int, ...] = (1,)

    # Understanding-tower prefill capture: combined cond+uncond token buckets,
    # rounded up at replay; prompts past the largest bucket run eager. Override
    # with COSMOS3_PREFILL_CAPTURE_TOKENS / COSMOS3_PREFILL_CAPTURE_BS.
    prefill_capture_token_buckets: tuple[int, ...] = (16, 32, 64, 128, 256)
    prefill_capture_batch_sizes: tuple[int, ...] = (1,)

    def __init__(self, transformer, config, scheduler=None):
        super().__init__()
        self.transformer = transformer
        self.config = config
        # Template scheduler; a fresh instance (with its own multistep state) is
        # built per request from this one's config.
        self._scheduler_template = scheduler
        # Per-request denoise state lives in the engine-managed
        # ``request_states`` store from the NodeSubmodule base.
        # Windowed sessions: the last window's clean latents per session id,
        # most recent last; a follow-up request with ``resume_session``
        # re-pins its head from here (see ``_prepare_windowed_prefill``).
        self._session_tails: OrderedDict[str, torch.Tensor] = OrderedDict()
        # Compile the pure denoise compute (~1.2-1.3x/step; the kernels bake
        # into the CUDA graphs at capture). fullgraph=False breaks at the
        # attention; ``config.compile_denoise=False`` keeps the eager step for
        # the bit-exact parity tests.
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

    def cg_key_info(self, graph_walk: str, per_request_info: dict) -> object | None:
        """Which of this walk's capture buckets the batch belongs to, or None
        for "run eager".

        The engine leases the replay slot before the step is declared, so this
        answers from per-request state rather than from prepared inputs. It is
        the sole gate on capture — the v1 engine has no ``can_use_cuda_graphs``
        — so every condition the old gate checked lives here:

        * only the two-branch guidance regime is captured (both the prefill's
          combined cond+uncond pack and the denoise step's batched CFG);
        * of the denoise walks only ``image_gen``, and only at a resolution a
          graph was captured for (``_capture_layout``);
        * a batch shares one captured (batch size, token count) bucket, so a
          mixed-resolution batch falls back to the eager cross-request denoise.

        Every fact this reads is fixed for a request's whole lifetime (its
        guidance regime and its latent shape, both settled at prefill). That is
        a requirement, not a coincidence: the speculative pre-plan path leases
        a slot for step N+1 while step N is still in flight, before N+1 has
        been through ``prepare_inputs``, so a key derived from anything that
        advances per step (the denoise step index, say) would be read one
        iteration stale and could lease a bucket the declaration then
        contradicts. ``Engine.exec`` checks the two against each other.

        The guidance interval is the fact this deliberately does not consult:
        a captured step always runs both branches, so an out-of-interval step
        gets the guidance combine where the eager path would run the
        conditional branch alone (``postprocess`` combines unconditionally).
        That predates the v1 migration and matches the fused reference
        pipeline, which applies guidance on every step; honoring the interval
        here instead would make the key step-dependent, which is exactly what
        the pre-plan path cannot support.
        """
        rids = list(per_request_info)
        states = [self.request_states.get(rid) for rid in rids]
        if not states or any(
            st is None or st.get("uncond") is None for st in states
        ):
            return None
        if graph_walk in PREFILL_WALKS:
            return True
        if graph_walk not in GEN_WALKS:
            return None
        keys = {self._capture_key(graph_walk, st) for st in states}
        if len(keys) != 1:
            return None
        return keys.pop()

    def _step_info(self, graph_walk: str, st, step_index: int) -> GenStepInfo:
        """The per-request facts ``declare_step`` needs, resolved here where
        the request's state and its step index are both in hand."""
        return GenStepInfo(
            cfg=st.get("uncond") is not None,
            cfg_active=self._cfg_active(st, step_index),
            capture_key=self._capture_key(graph_walk, st),
        )

    def _capture_key(self, graph_walk: str, st) -> object | None:
        """This one request's half of ``cg_key_info``: everything that does
        not depend on the rest of the batch. See ``GenStepInfo.capture_key``."""
        if graph_walk not in GEN_WALKS or st.get("uncond") is None:
            return None
        if graph_walk == VIDEO_GEN_AR_WALK and st.get("ar_kv_mode"):
            # kv windows interleave commit iterations — a different step
            # declaration the lease could not know about — so they run eager.
            return None
        # The latent shape is fixed for the request's lifetime (a windowed
        # request's windows all share the padded window shape), and the
        # clean/noisy layout rides along as a mask input, so one graph per
        # shape serves t2i, t2v, i2v and chained windows alike.
        shape = tuple(st["latent_shape"])
        layout = getattr(self, "_capture_layout", None) or {}
        return shape if shape in layout else None

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
        noisy_frames: list[int] | None = None,
        start_frame_offset: int = 0,
    ) -> dict:
        static = build_static_inputs(
            list(ids), self._latent_shape(height, width, num_frames), self.config,
            self.config.vae.scale_factor_temporal, fps, device,
            has_image_condition=has_image_condition,
            sound_latent_frames=sound_latent_frames,
            noisy_frames=noisy_frames,
            start_frame_offset=start_frame_offset,
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
        if self.config.distilled_sigmas:
            return self._new_distilled_scheduler(device)
        from diffusers import UniPCMultistepScheduler

        # The checkpoint scheduler config carries the trained sigma schedule
        # (karras sigmas); keep it unless the request explicitly overrides a
        # field. Forcing karras off uses the wrong schedule and corrupts the
        # larger model's high-resolution text-to-video.
        overrides = {}
        native_flow = bool(self.config.use_native_flow_schedule)
        if use_karras_sigma is not None:
            overrides["use_karras_sigmas"] = use_karras_sigma
        elif native_flow:
            # The native flow schedule hands the scheduler its sigmas
            # explicitly; the karras transform would re-space them. The
            # reference recipes for these checkpoints pass
            # ``use_karras_sigmas=False`` for the same reason.
            overrides["use_karras_sigmas"] = False
        if flow_shift is not None:
            overrides["flow_shift"] = flow_shift

        scheduler = UniPCMultistepScheduler.from_config(self._scheduler_template.config, **overrides)
        if native_flow:
            # Linspaced flow sigmas from 1 - 1/T down to (not including) 0,
            # the ``use_native_flow_schedule`` pipeline path; the scheduler
            # applies its flow shift on top.
            num_train = int(scheduler.config.num_train_timesteps)
            sigmas = native_flow_sigmas(num_inference_steps, num_train)
            scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler

    def _new_distilled_scheduler(self, device):
        """The 4-step distilled sampler: a FlowMatchEuler scheduler with the
        checkpoint's stochastic (SDE) step over the fixed sigma list — every
        step re-noises with ``x' = (1 - sigma') (x - sigma v) + sigma' eps``
        from the request's generator (see ``_scheduler_step``). Flow shift and
        karras spacing do not apply: the sigmas are explicit."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        template = self._scheduler_template
        if template is not None:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(template.config)
        else:
            sc = self.config.scheduler
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=int(sc.num_train_timesteps), shift=1.0,
                stochastic_sampling=bool(sc.stochastic_sampling),
            )
        scheduler.set_timesteps(sigmas=[float(x) for x in self.config.distilled_sigmas], device=device)
        return scheduler

    @staticmethod
    def _scheduler_step(st, velocity, t, latents):
        """One scheduler update of a ``[C, T, H, W]`` latent. A distilled
        request carries its SDE generator (``sde_generator``, the same one its
        initial noise came from, as in the reference), so the stochastic step
        is seedable; UniPC takes no generator."""
        kwargs = {}
        gen = st.get("sde_generator")
        if gen is not None:
            kwargs["generator"] = gen
        return st["scheduler"].step(
            velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False, **kwargs,
        )[0].squeeze(0)

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
        self, graph_walk, fwd_info, inputs, seen_token_mask=None, pos_info={}, **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        if graph_walk in PREFILL_WALKS:
            return self._prepare_prefill(fwd_info, inputs, device)
        if graph_walk in GEN_WALKS:
            return self._prepare_image_gen(graph_walk, fwd_info, inputs, device)
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
        # Video-to-video: the request's condition_frame_indexes_vision pin clean
        # latent frames from the conditioning video; the complement is predicted.
        condition_indexes = md.get("condition_frame_indexes_vision")
        noisy_frames = None
        if condition_indexes:
            t_lat = self._latent_shape(height, width, num_frames)[2]
            noisy_frames = [f for f in range(t_lat) if f not in set(condition_indexes)]

        # Windowed AR video: statics, scheduler and noise are built per window
        # at the window's own latent shape (window boundaries swap them); the
        # text prefill below is identical either way — its K/V is written once
        # and read by every window's steps.
        if md.get("window_mode") is not None:
            return self._prepare_windowed_prefill(
                fwd_info, md, cond_ids, uncond_ids, height, width, fps, gs, steps, device,
            )

        # Opt-in sound: append a jointly denoised AVAE-latent band to the
        # generation block. Video-only (single-frame image and action requests
        # are rejected at request resolution).
        sound_frames = None
        if md.get("generate_sound"):
            if num_frames <= 1:
                raise ValueError("Cosmos3 sound generation requires a video request (num_frames > 1)")
            _, sound_frames = self._resolve_sound_frames(md)

        cond = self._build_static(
            cond_ids, height, width, num_frames, fps, has_image_condition, device,
            sound_latent_frames=sound_frames, noisy_frames=noisy_frames,
        )
        uncond = None
        if uncond_ids is not None:
            uncond = self._build_static(
                uncond_ids, height, width, num_frames, fps, has_image_condition, device,
                sound_latent_frames=sound_frames, noisy_frames=noisy_frames,
            )

        node_inputs = self._get_prefill_node_inputs(cond, uncond)
        self._slim_statics(cond, uncond)
        st = self.request_state(fwd_info.request_id)
        st.add_all(
            cond=cond,
            uncond=uncond,
            gs=gs,
            guidance_interval=md.get("guidance_interval"),
            scheduler=self._new_scheduler(
                steps, device, flow_shift=md.get("flow_shift"),
                use_karras_sigma=md.get("use_karras_sigma")
            ),
            latent_shape=self._latent_shape(height, width, num_frames),
            num_sound=sound_frames,
        )
        # Video-to-video: pin the requested latent frames (the complement starts
        # from noise). Only the pin mask is derived here — the clean latents are
        # VAE-encoded by the parallel vae_encoder node and reach the denoise
        # loop as its ``cond_latents`` input edge (see ``_ingest_cond_latents``).
        if condition_indexes:
            latent_shape = st["latent_shape"]
            dtype = self.transformer.proj_in.weight.dtype
            vmask = torch.zeros((1, 1, latent_shape[2], 1, 1), device=device, dtype=dtype)
            for f in condition_indexes:
                vmask[:, :, f] = 1.0
            st.add("vmask", vmask)

        return node_inputs

    # ------------------------------------------------------------------
    # Windowed AR video (ported from #198, merceod)
    # ------------------------------------------------------------------

    def _window_latent_shape(self, height, width, units):
        s = self.config.vae.scale_factor_spatial
        return (1, self.config.latent_channel, units, height // s, width // s)

    def _build_window_statics(
        self, cond_ids, uncond_ids, height, width, units, fps,
        has_image_condition, cond_units, device,
        first_window=True, start_unit=0,
    ):
        """Packed statics for one window of ``units`` latent frames. The
        leading ``cond_units`` frames are clean overlap conditioning (the
        previous window's tail); the image anchor applies only to the first
        window. A window packs exactly like a clip of the same latent length.
        Chained windows position from frame 0 (each window in-distribution
        for the clip-trained checkpoint); kv windows pass their absolute
        first frame as ``start_unit`` so relative distances to the committed
        context K/V — rotated at absolute positions — stay right."""
        tf = self.config.vae.scale_factor_temporal
        num_frames = 1 + (units - 1) * tf
        noisy = list(range(cond_units, units)) if cond_units else None
        anchored = has_image_condition and first_window
        cond = self._build_static(
            cond_ids, height, width, num_frames, fps, anchored, device,
            noisy_frames=noisy, start_frame_offset=start_unit,
        )
        uncond = None
        if uncond_ids is not None:
            uncond = self._build_static(
                uncond_ids, height, width, num_frames, fps, anchored, device,
                noisy_frames=noisy, start_frame_offset=start_unit,
            )
        return cond, uncond

    def _prepare_windowed_prefill(
        self, fwd_info, md, cond_ids, uncond_ids, height, width, fps, gs, steps, device,
    ) -> ARNodeInputs:
        # kv mode appends each window's clean K/V to the cache (one commit
        # iteration per window, hence steps + 1 loop iterations) and lets the
        # pool release context beyond the horizon at each commit; chained
        # re-pins the previous tail as clean conditioning instead and never
        # touches committed state.
        is_kv = md.get("window_mode") == "kv"
        schedule = WindowSchedule(
            total_units=int(md["total_latent_units"]),
            window_units=int(md["window_latent_units"]),
            context_units=int(md.get("context_latent_units", 0)) if is_kv else 0,
            overlap_units=int(md["overlap_latent_units"]),
        )
        w0 = schedule.window(0)
        has_image_condition = bool(md.get("has_image_condition", False))
        # A resumed session re-pins the stored tail as window 0's clean head:
        # the same clean-frame layout a chained window uses for its overlap.
        resume_units = int(md.get("resume_latent_units", 0) or 0)
        resume_tail = None
        if resume_units:
            resume_tail = self._session_tail(str(md.get("session_id")), resume_units, height, width)
        cond, uncond = self._build_window_statics(
            cond_ids, uncond_ids, height, width, w0.units, fps,
            has_image_condition=has_image_condition, cond_units=resume_units, device=device,
        )
        node_inputs = self._get_prefill_node_inputs(cond, uncond)
        tokens_per_unit = cond["num_vision_tokens"] // w0.units
        self._slim_statics(cond, uncond)
        iters_per_window = steps + 1 if is_kv else steps
        st = self.request_state(fwd_info.request_id)
        if resume_tail is not None:
            shape = self._window_latent_shape(height, width, w0.units)
            dtype = self.transformer.proj_in.weight.dtype
            vmask = torch.zeros((1, 1, w0.units, 1, 1), device=device, dtype=dtype)
            vmask[:, :, :resume_units] = 1.0
            cond_video = torch.zeros(shape, device=device, dtype=dtype)
            cond_video[:, :, :resume_units] = resume_tail.to(device=device, dtype=dtype)
            st.add_all(vmask=vmask, cond_video_latents=cond_video)
        st.add_all(
            ar_session_id=md.get("session_id"),
            cond=cond,
            uncond=uncond,
            gs=gs,
            guidance_interval=md.get("guidance_interval"),
            scheduler=self._new_scheduler(
                steps, device, flow_shift=md.get("flow_shift"),
                use_karras_sigma=md.get("use_karras_sigma"),
            ),
            latent_shape=self._window_latent_shape(height, width, w0.units),
            num_sound=None,
            ar_schedule=schedule,
            ar_steps=steps,
            ar_iters_per_window=iters_per_window,
            ar_total_iters=schedule.num_windows * iters_per_window,
            ar_kv_mode=is_kv,
            ar_rid=fwd_info.request_id,
            ar_tokens_per_unit=tokens_per_unit,
            ar_cond_ids=list(cond_ids),
            ar_uncond_ids=list(uncond_ids) if uncond_ids is not None else None,
            ar_has_image_condition=has_image_condition,
            ar_flow_shift=md.get("flow_shift"),
            ar_karras=md.get("use_karras_sigma"),
            ar_size=(int(height), int(width)),
            ar_fps=fps,
            # WindowPlan-keyed slimmed (cond, uncond) cache; chained windows
            # share entries across boundaries (their positions restart at 0),
            # kv windows are position-distinct and each get their own.
            ar_statics={},
        )
        return node_inputs

    def _session_tail(self, session_id: str, units: int, height: int, width: int) -> torch.Tensor:
        """The stored last-window latents a resumed request pins its head
        with: the newest ``units`` latent frames, at the request's latent
        size. Unknown (or evicted) sessions and size mismatches are request
        errors — silently starting from scratch would break the client's
        frame accounting."""
        tail = self._session_tails.get(session_id)
        if tail is None:
            raise ValueError(
                f"Cosmos3 resume_session: unknown or expired session {session_id!r}."
            )
        shape = self._window_latent_shape(height, width, units)
        if tail.shape[2] < units or tuple(tail.shape[3:]) != tuple(shape[3:]):
            raise ValueError(
                f"Cosmos3 resume_session: session {session_id!r} holds latents of shape "
                f"{tuple(tail.shape)}, which cannot seed a {height}x{width} window."
            )
        self._session_tails.move_to_end(session_id)
        return tail[:, :, -units:]

    def _store_session_tail(self, st, window_latents: torch.Tensor) -> None:
        """The rollout's final window, kept for a ``resume_session`` follow-up
        (most recent ``session_store_size`` sessions)."""
        session_id = st.get("ar_session_id")
        if not session_id:
            return
        self._session_tails[str(session_id)] = window_latents.detach().clone()
        self._session_tails.move_to_end(str(session_id))
        while len(self._session_tails) > max(1, int(self.config.session_store_size)):
            self._session_tails.popitem(last=False)

    def _window_statics_for(self, st, plan, device):
        # Chained windows restart their positions at 0, so every window past
        # the first shares one statics entry (cached). kv windows carry
        # absolute positions and are each used once, so they are built on
        # demand and never cached: a long rollout's state stays flat.
        kv = bool(st.get("ar_kv_mode"))
        start_unit = plan.start if kv else 0
        key = (plan.units, plan.cond_units, start_unit, plan.index == 0)
        cached = None if kv else st["ar_statics"].get(key)
        if cached is not None:
            return cached
        height, width = st["ar_size"]
        cond, uncond = self._build_window_statics(
            st["ar_cond_ids"], st["ar_uncond_ids"], height, width, plan.units,
            st["ar_fps"], has_image_condition=st["ar_has_image_condition"],
            cond_units=plan.cond_units, device=device,
            first_window=plan.index == 0, start_unit=start_unit,
        )
        self._slim_statics(cond, uncond)
        if not kv:
            st["ar_statics"][key] = (cond, uncond)
        return cond, uncond

    @staticmethod
    def _window_step(st, step_index: int) -> tuple[int, int, bool]:
        """A windowed request's global loop counter -> (window index,
        within-window step, whether this is a kv-mode commit iteration)."""
        per = st["ar_iters_per_window"]
        local = step_index % per
        commit = bool(st.get("ar_kv_mode")) and local == st["ar_steps"]
        return step_index // per, local, commit

    def _bind_window_retention(self, st, kv) -> None:
        """At the first kv-mode commit, before its pass runs: hand each
        guidance branch's text prefix and the schedule's context horizon to
        the pool as the stream's retention policy. The pool then releases
        aged-out frame pages inside every commit (see ``KVManager.commit``) —
        between steps as far as the planners are concerned, so nothing here
        races the engine's pre-plan. Metadata only, so it is safe under this
        step's own admission."""
        if kv is None:
            raise RuntimeError(
                "Cosmos3 windowed kv mode needs the DiT node's KV resource"
            )
        branches = [(COND_LABEL, st["cond"])]
        if st["uncond"] is not None:
            branches.append((UNCOND_LABEL, st["uncond"]))
        for label, static in branches:
            WindowedKVSession(
                kv, st["ar_rid"], label, st["ar_schedule"],
                tokens_per_unit=st["ar_tokens_per_unit"],
            ).bind(static["und_len"])
        st.add("ar_retention_bound", True)

    def _kv_resource(self, engine_inputs: ModelInputsFromEngine):
        resources = engine_inputs.resources or self.node_resources or {}
        return resources.get(KV_CACHE)

    def _prepare_commit(self, st, window_index, latents, time_index) -> ARNodeInputs:
        """Inputs of a kv-mode commit iteration: the window's newly generated
        span, which the declaration appends and commits under every live
        guidance branch."""
        plan = st["ar_schedule"].window(window_index)
        span = (plan.commit_end - plan.commit_start) * st["ar_tokens_per_unit"]
        cfg = st["uncond"] is not None
        return ARNodeInputs(
            input_seq_len=span,
            tensor_inputs={"latents": latents, "time_index": time_index},
            resource_step_info=GenStepInfo(
                cfg=cfg, cfg_active=cfg, capture_key=None, commit=True,
            ),
        )

    def _commit_step(
        self, request_ids: list[str], spans: tuple[int, ...], cfg: bool,
    ) -> SubmoduleStep:
        """A kv-mode window commit: the finished window's new span appended
        under every live guidance branch and committed — the frame-token
        analogue of the prefill. Paged, so the K/V lands in the pages the
        later windows' steps read (the dense backend never writes them);
        non-causal like every generation plan. Both branches commit
        regardless of any guidance interval: each branch's future windows
        read its own context."""
        labels = (COND_LABEL, UNCOND_LABEL) if cfg else (COND_LABEL,)
        combined = cfg and self.batched_cfg
        return SubmoduleStep(
            segments=self._segments(request_ids, labels, [spans] * len(labels)),
            steps={
                KV_CACHE: KVStep(
                    commit=True,
                    combined_labels={labels: CFG_BATCHED_LABEL} if combined else {},
                ),
                ATTN: AttentionStep(causal=False),
            },
        )

    def _commit_window(self, attn, st, latents, time_index, window_index, kv=None) -> dict:
        """kv-mode commit iteration: run the generation tower over the
        window's finished clean latents (no timestep embedding — the clean-
        conditioning convention), appending their K/V to both guidance
        branches' cache streams, then stage the next window. The release of
        context past the horizon is the pool's, at this step's commit, under
        the retention the first commit installs here."""
        if not st.get("ar_retention_bound"):
            self._bind_window_retention(st, kv)
        plan = st["ar_schedule"].window(window_index)
        stride = st["ar_tokens_per_unit"]
        dtype = self.transformer.proj_in.weight.dtype
        commit_latents = (
            latents[:, :, plan.cond_units:] if plan.cond_units else latents
        ).to(dtype)
        # The commit span's positions are the tail slice of the window's
        # statics (kv statics carry absolute frames, so the slice is already
        # at the right absolute positions). Both guidance branches commit,
        # packed into one pass or sequentially per the batched_cfg regime.
        offset = plan.cond_units * stride
        cond_pos = st["cond"]["vision_mrope_ids"][:, offset:]
        if st["uncond"] is not None and self.batched_cfg:
            self.transformer.commit_window(
                commit_latents,
                [cond_pos, st["uncond"]["vision_mrope_ids"][:, offset:]],
                CFG_BATCHED_LABEL, attn,
            )
        else:
            self.transformer.commit_window(commit_latents, [cond_pos], COND_LABEL, attn)
            if st["uncond"] is not None:
                self.transformer.commit_window(
                    commit_latents,
                    [st["uncond"]["vision_mrope_ids"][:, offset:]],
                    UNCOND_LABEL, attn,
                )
        return self._finish_window(st, latents, time_index, window_index)

    def _finish_window(self, st, window_latents, time_index, window_index) -> dict:
        """End of a window: emit the window's clean latents on the streaming
        edge and stage the next window — fresh scheduler, fresh noise, and
        (chained mode) the finished tail re-pinned as clean overlap
        conditioning through the same vmask machinery video-to-video uses."""
        schedule = st["ar_schedule"]
        outputs = {
            "latents": [window_latents],
            "time_index": [time_index + 1],
            "window_latents": [window_latents],
        }
        if window_index + 1 >= schedule.num_windows:
            self._store_session_tail(st, window_latents)
            return outputs
        plan = schedule.window(window_index + 1)
        device = window_latents.device
        dtype = self.transformer.proj_in.weight.dtype
        cond, uncond = self._window_statics_for(st, plan, device)
        st.add("cond", cond)
        st.add("uncond", uncond)
        st.add("scheduler", self._new_scheduler(
            st["ar_steps"], device, flow_shift=st.get("ar_flow_shift"),
            use_karras_sigma=st.get("ar_karras"),
        ))
        height, width = st["ar_size"]
        shape = self._window_latent_shape(height, width, plan.units)
        st.add("latent_shape", shape)
        next_latents = torch.randn(
            shape, generator=st["ar_generator"], device=device, dtype=dtype
        )
        if plan.cond_units > 0:
            tail = window_latents[:, :, -plan.cond_units:].to(dtype)
            vmask = torch.zeros((1, 1, plan.units, 1, 1), device=device, dtype=dtype)
            vmask[:, :, :plan.cond_units] = 1.0
            cond_video = torch.zeros(shape, device=device, dtype=dtype)
            cond_video[:, :, :plan.cond_units] = tail
            st.add("vmask", vmask)
            st.add("cond_video_latents", cond_video)
            next_latents = vmask * cond_video + (1.0 - vmask) * next_latents
        else:
            st.remove(["vmask", "cond_video_latents"])
        outputs["latents"] = [next_latents]
        return outputs

    @staticmethod
    def _slim_statics(cond: dict, uncond: dict | None) -> None:
        """Drop packed-static fields the denoise loop never reads: the token
        ids and und-tower rotary ids ride the prefill node inputs, and the raw
        joint-sequence mse indexes are superseded by the gen-relative
        ``*mse_gen_indexes``."""
        for s in (cond, uncond) if uncond is not None else (cond,):
            for key in (
                "input_ids", "text_mrope_ids", "vision_mse_loss_indexes",
                "sound_mse_loss_indexes", "action_mse_loss_indexes",
            ):
                s.pop(key, None)

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

    def _ingest_cond_latents(self, st, inputs, device) -> None:
        """First denoise iteration: adopt the vae_encoder node's conditioning
        latents (the gen walk's ``cond_latents`` edge; empty for unconditioned
        requests). Video/action conditioning stores the full-latent-shape
        pinned latents under ``cond_video_latents``; image-to-video stores the
        single-frame anchor under ``cond_latents``. An action request without
        visual conditioning pins zeros (matching its all-zero clean anchors);
        a video-conditioned request whose video never arrived cannot serve."""
        incoming = (inputs or {}).get("cond_latents")
        latents = incoming[0].to(device) if incoming else None
        if st.get("vmask") is not None:
            if latents is not None:
                st.add("cond_video_latents", latents)
            elif st.get("cond_video_latents") is not None:
                # A resumed windowed session pinned its head from the stored
                # tail at prefill; nothing arrives on the edge.
                pass
            elif "action_chunk" in st:
                st.add("cond_video_latents", torch.zeros(
                    st["latent_shape"], device=device,
                    dtype=self.transformer.proj_in.weight.dtype,
                ))
            else:
                raise ValueError(
                    "Cosmos3 video conditioning was requested but no conditioning video arrived."
                )
        elif latents is not None:
            st.add("cond_latents", latents)

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

        # The visual conditioning (inverse-dynamics: the whole observed video;
        # forward-dynamics / policy: the conditioning frame) is VAE-encoded by
        # the parallel vae_encoder node and reaches the denoise loop as its
        # ``cond_latents`` input edge; the per-mode vmask above selects which
        # latent frames are kept clean from it.

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

        node_inputs = self._get_prefill_node_inputs(cond, uncond)
        self._slim_statics(cond, uncond)
        self.request_state(fwd_info.request_id).add_all(
            cond=cond,
            uncond=uncond,
            gs=gs,
            scheduler=self._new_scheduler(
                steps, device, flow_shift=md.get("flow_shift"),
                use_karras_sigma=md.get("use_karras_sigma")
            ),
            latent_shape=latent_shape,
            action_chunk=action_chunk,
            action_dim=action_dim,
            raw_action_dim=raw_action_dim,
            domain_t=torch.tensor([domain_id], dtype=torch.long, device=device),
            vmask=vmask,
            action_clean_mask=action_clean,
            clean_action=clean_action,
        )
        return node_inputs

    def _prepare_image_gen(
        self, graph_walk, fwd_info, inputs, device,
    ) -> ARNodeInputs:
        st = self.request_states[fwd_info.request_id]
        windowed = "ar_schedule" in st
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            self._ingest_cond_latents(st, inputs, device)
            gen = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
            if windowed:
                # Later windows draw their noise from the same generator, so a
                # seeded windowed request is deterministic end to end.
                st.add("ar_generator", gen)
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
            if self.config.distilled_sigmas:
                # The distilled SDE step re-noises every position from this
                # generator (the reference passes the pipeline generator), and
                # the i2v anchor must be re-pinned after each step — the same
                # mask re-injection video-to-video uses.
                st.add("sde_generator", gen)
                if cond_latents is not None and st.get("vmask") is None:
                    vmask = torch.zeros((1, 1, latents.shape[2], 1, 1), device=device, dtype=latents.dtype)
                    vmask[:, :, 0] = 1.0
                    cond_video = torch.zeros_like(latents)
                    cond_video[:, :, 0] = cond_latents[:, :, 0].to(latents.dtype)
                    st.add_all(vmask=vmask, cond_video_latents=cond_video)
            if st.get("vmask") is not None:
                # Video-to-video: pinned latent frames start clean, the rest
                # from the noise drawn above (the reference RNG order).
                latents = st["vmask"] * st["cond_video_latents"] + (1.0 - st["vmask"]) * latents
            time_index = torch.zeros(1, dtype=torch.long, device=device)
        else:
            latents = inputs["latents"][0]
            time_index = inputs["time_index"][0]

        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        if windowed:
            # The loop counter is global over every window's iterations; the
            # schedule index is the within-window step, and in kv mode the
            # extra per-window iteration is the commit pass.
            if step_index >= st["ar_total_iters"]:
                return None
            window_index, local, commit = self._window_step(st, step_index)
            if commit:
                return self._prepare_commit(st, window_index, latents, time_index)
            step_index = local
        elif step_index >= len(scheduler.timesteps):
            return None
        tensors = {"latents": latents, "time_index": time_index}
        # The CUDA-graph capture reads the timestep, rotary positions and the
        # clean/noisy token mask as static buffers (it can't reach the
        # per-request scheduler at replay), so materialize them here. The eager
        # path ignores these and recomputes from per-request state. Only built
        # for a request that could land on a captured graph (two-branch
        # guidance at a captured shape; see ``_capture_key``).
        if st["uncond"] is not None and self._capture_key(graph_walk, st) is not None:
            # The denoise loop may dispatch one extra (discarded) step past this
            # request's step count; clamp so materializing the static timestep
            # buffer can't index past the schedule. ``step_index`` is the
            # within-window step for a windowed request.
            n_steps = len(scheduler.timesteps)
            t = scheduler.timesteps[min(step_index, n_steps - 1)].to(torch.float32)
            # Every graph is built with all frames declared noisy, so the
            # timestep buffer spans every generation token.
            tensors["vision_timesteps"] = t.reshape(1).expand(st["cond"]["num_vision_tokens"]).contiguous()
            tensors["position_ids_cond"] = st["cond"]["vision_mrope_ids"]
            tensors["position_ids_uncond"] = st["uncond"]["vision_mrope_ids"]
            tensors["noisy_token_mask"] = self._noisy_masks(st, device)[0]
        return ARNodeInputs(
            input_seq_len=st["cond"]["num_vision_tokens"],
            tensor_inputs=tensors,
            resource_step_info=self._step_info(graph_walk, st, step_index),
        )

    def _noisy_masks(self, st, device) -> tuple[torch.Tensor, torch.Tensor]:
        """The request's current clean/noisy layout as data for a captured
        graph: a per-token mask (``[num_vision_tokens]``, 1 on noisy frames'
        tokens) and the same per frame (``[1, T, 1, 1]``). Cached per
        statics object, so a windowed request refreshes it at each window."""
        cond = st["cond"]
        cache = st.get("noisy_masks")
        if cache is not None and cache[0] is cond:
            return cache[1], cache[2]
        ((t_frames, patch_h, patch_w),) = cond["vision_token_shapes"]
        frame_mask = torch.zeros(t_frames, device=device, dtype=torch.float32)
        frame_mask[cond["vision_noisy_frame_indexes"][0].to(device)] = 1.0
        token_mask = frame_mask.repeat_interleave(patch_h * patch_w).contiguous()
        frame_mask = frame_mask.view(1, t_frames, 1, 1)
        st.add("noisy_masks", (cond, token_mask, frame_mask))
        return token_mask, frame_mask

    def _prepare_video_sound_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        st = self.request_states[fwd_info.request_id]
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            self._ingest_cond_latents(st, inputs, device)
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
            if st.get("vmask") is not None:
                latents = st["vmask"] * st["cond_video_latents"] + (1.0 - st["vmask"]) * latents
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
            input_seq_len=st["cond"]["num_vision_tokens"] + st["num_sound"],
            tensor_inputs={
                "latents": latents, "sound_latents": sound_latents, "time_index": time_index,
            },
            resource_step_info=self._step_info(
                VIDEO_SOUND_GEN_WALK, st, step_index
            ),
        )

    def _prepare_action_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        st = self.request_states[fwd_info.request_id]
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            self._ingest_cond_latents(st, inputs, device)
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
                st["action_clean_mask"] * st["clean_action"]
                + (1.0 - st["action_clean_mask"]) * a_noise
            )
            action_latents[..., raw:] = 0
            v_noise = randn_tensor(st["latent_shape"], generator=gen, device=device, dtype=dtype)
            latents = st["vmask"] * st["cond_video_latents"] + (1.0 - st["vmask"]) * v_noise
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
            input_seq_len=st["cond"]["num_vision_tokens"] + st["cond"]["num_action_tokens"],
            tensor_inputs={"latents": latents, "action_latents": action_latents, "time_index": time_index},
            resource_step_info=self._step_info(ACTION_GEN_WALK, st, step_index),
        )

    # ------------------------------------------------------------------
    # declare_step: what each walk's step is. The runner drives the plans
    # and commits; preprocess below only marshals data.
    # ------------------------------------------------------------------

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """This batch's step: which cache streams it touches, by how much, and
        which attention backend runs over them.

        Two guidance branches are two labels on the one cache. Whether a
        request has both, and whether this step uses both, is a per-request
        fact that rides in on ``NodeInputs.resource_step_info`` from
        ``prepare_inputs`` (see ``GenStepInfo``) rather than being read off
        the batch here.

        The attention key is the other half of the declaration. ``attn`` is
        paged, ``attn_gen`` dense; a step that may replay from a captured
        graph must name the paged one, because the dense backend's prefix
        gather is shaped by the step and cannot be captured.
        """
        if graph_walk in PREFILL_WALKS:
            # The text prefix is written once and committed; the denoise steps
            # read it frozen for the rest of the request. Both guidance
            # branches prefill together under one combined plan when present.
            # Always paged: the denoise steps read this K/V back out of the
            # pages, which the dense backend never writes.
            cfg = bool(inputs[0].kwargs.get("cfg"))
            labels = (COND_LABEL, UNCOND_LABEL) if cfg else (COND_LABEL,)
            return SubmoduleStep(
                cg_key_info=cfg or None,
                segments=self._segments(
                    request_ids, labels,
                    [
                        tuple(inp.kwargs["seq_lens"][label] for inp in inputs)
                        for label in labels
                    ],
                ),
                steps={
                    KV_CACHE: KVStep(
                        commit=True,
                        combined_labels=(
                            {labels: CFG_BATCHED_LABEL} if cfg else {}
                        ),
                    ),
                    ATTN: AttentionStep(causal=True),
                },
            )

        if (
            graph_walk not in GEN_WALKS
            and graph_walk not in SOUND_WALKS
            and graph_walk not in ACTION_WALKS
        ):
            raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

        # Denoise steps read the frozen prefix and recompute the same
        # generation span every step, so nothing commits. The fresh tokens are
        # still declared as ordinary spans: that is what gives the dense
        # backend its query lengths, and it keeps the two backends drop-in for
        # one declaration (see DenseAttentionManager).
        spans = tuple(inp.input_seq_len for inp in inputs)
        infos = [self._gen_step_info(inp) for inp in inputs]

        if len(infos) == 1 and infos[0].commit:
            # A kv-mode windowed commit (never batched: see can_batch).
            return self._commit_step(request_ids, spans, infos[0].cfg)

        # Mirrors cg_key_info: one capture bucket shared by every row. Under a
        # lease the padding rows carry the bucket's own key, which keeps `keys`
        # a singleton. The lease is what selects the branch, though — a key
        # only says this batch *could* have been captured, and a bucket dropped
        # at warmup leaves it set with nothing leased.
        keys = {info.capture_key for info in infos}
        capture_key = keys.pop() if len(keys) == 1 else None
        captured = slot_lease is not None and capture_key is not None
        attn_key = ATTN if captured else self._eager_gen_attn_key()

        if captured:
            # The captured denoise graph has a fixed shape and always runs both
            # guidance branches — including outside a request's guidance
            # interval, which the eager path below honors. See `cg_key_info`.
            return self._gen_step(
                request_ids, spans, (COND_LABEL, UNCOND_LABEL),
                combined=True, attn_key=attn_key, cg_key_info=capture_key,
            )

        if len(infos) > 1:
            # Cross-request batch: one combined plan over every request's
            # guidance branches (a single branch when guidance is off). The
            # batched forward always applies guidance when the branch exists,
            # so the interval does not narrow the batched declaration.
            labels = (
                (COND_LABEL, UNCOND_LABEL) if infos[0].cfg else (COND_LABEL,)
            )
            return self._gen_step(
                request_ids, spans, labels, combined=True, attn_key=attn_key,
            )

        if not infos[0].cfg_active:
            # Guidance off, or a guidance_interval out-of-interval step: the
            # conditional branch runs alone, so nothing else plans.
            return self._gen_step(
                request_ids, spans, (COND_LABEL,),
                combined=False, attn_key=attn_key,
            )
        # Batched CFG packs both branches into one plan; the sequential path
        # plans each branch on its own label and runs two forwards.
        return self._gen_step(
            request_ids, spans, (COND_LABEL, UNCOND_LABEL),
            combined=self.batched_cfg, attn_key=attn_key,
        )

    @staticmethod
    def _segments(
        request_ids: list[str],
        labels: tuple[str, ...],
        spans_per_label: list[tuple[int, ...]],
    ) -> list[Segment]:
        """Label-major, matching how a combined plan concatenates its sources.
        This ordering is what every per-token array in the forward is packed
        in, so it is declared once, here."""
        return [
            Segment(rid, label, span)
            for label, spans in zip(labels, spans_per_label, strict=True)
            for rid, span in zip(request_ids, spans, strict=True)
        ]

    def _gen_step(
        self, request_ids: list[str], spans: tuple[int, ...],
        labels: tuple[str, ...], combined: bool, attn_key: str,
        cg_key_info: object | None = None,
    ) -> SubmoduleStep:
        """A denoise step: the same span under each live label, nothing
        committed, non-causal attention over [frozen prefix | fresh tokens]."""
        return SubmoduleStep(
            cg_key_info=cg_key_info,
            segments=self._segments(
                request_ids, labels, [spans] * len(labels)
            ),
            steps={
                KV_CACHE: KVStep(
                    commit=False,
                    combined_labels=(
                        {labels: CFG_BATCHED_LABEL} if combined else {}
                    ),
                ),
                attn_key: AttentionStep(causal=False),
            },
        )

    def _eager_gen_attn_key(self) -> str:
        """The backend an eager denoise step runs on: the dense one when the
        model declared it (``attention_backend="dense_gen"``), else paged."""
        return ATTN_GEN if ATTN_GEN in self.node_resources else ATTN

    @staticmethod
    def _gen_step_info(inp: ARNodeInputs) -> GenStepInfo:
        info = inp.resource_step_info
        if isinstance(info, GenStepInfo):
            return info
        # A caller that built its own inputs (tests, direct harnesses) gets the
        # single-branch eager declaration.
        return GenStepInfo(cfg=False, cfg_active=False, capture_key=None)

    def _preprocess_image_gen_captured(self, inputs) -> dict:
        """Pack a denoise step's inputs for the CUDA-graph path.

        Runs with synthetic request ids (no per-request state). The
        static-input tensors (latents, timestep, rotary positions) are
        stacked on a leading batch dim, so one captured graph spans a whole
        concurrent batch (a batch of one for the single-request latency
        path); the replay side copies each request's tensors into these
        fixed buffers. The attention plan comes from the step declaration,
        which always covers both guidance branches here (the graph shape is
        fixed), while the eager declaration skips the uncond plan on
        guidance-interval steps.
        """
        return {
            "latents": torch.stack([inp.tensor_inputs["latents"] for inp in inputs]),
            "vision_timesteps": torch.stack([inp.tensor_inputs["vision_timesteps"] for inp in inputs]),
            "position_ids_cond": torch.stack([inp.tensor_inputs["position_ids_cond"] for inp in inputs]),
            "position_ids_uncond": torch.stack([inp.tensor_inputs["position_ids_uncond"] for inp in inputs]),
            "noisy_token_mask": torch.stack([inp.tensor_inputs["noisy_token_mask"] for inp in inputs]),
        }

    def preprocess(
        self, graph_walk, engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs]
    ) -> dict:
        if graph_walk in GEN_WALKS and getattr(engine_inputs, "captured", False):
            return self._preprocess_image_gen_captured(inputs)

        if graph_walk in PREFILL_WALKS:
            has_cfg = inputs[0].kwargs.get("cfg")
            labels = [COND_LABEL, UNCOND_LABEL] if has_cfg else [COND_LABEL]
            # Pack label-major (all cond segments, then all uncond) to match
            # the (label, request) batch order of the declared plan.
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

        rids = engine_inputs.request_ids
        if graph_walk in GEN_WALKS:
            if len(rids) > 1:
                return {
                    "latents": {r: inp.tensor_inputs["latents"] for r, inp in zip(rids, inputs, strict=True)},
                    "time_index": {r: inp.tensor_inputs["time_index"] for r, inp in zip(rids, inputs, strict=True)},
                }
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "time_index": inputs[0].tensor_inputs["time_index"],
            }

        if graph_walk in SOUND_WALKS:
            if len(rids) > 1:
                return {
                    "latents": {r: inp.tensor_inputs["latents"] for r, inp in zip(rids, inputs, strict=True)},
                    "sound_latents": {
                        r: inp.tensor_inputs["sound_latents"] for r, inp in zip(rids, inputs, strict=True)
                    },
                    "time_index": {r: inp.tensor_inputs["time_index"] for r, inp in zip(rids, inputs, strict=True)},
                }
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "sound_latents": inputs[0].tensor_inputs["sound_latents"],
                "time_index": inputs[0].tensor_inputs["time_index"],
            }

        if graph_walk in ACTION_WALKS:
            if len(rids) > 1:
                return {
                    "latents": {r: inp.tensor_inputs["latents"] for r, inp in zip(rids, inputs, strict=True)},
                    "action_latents": {
                        r: inp.tensor_inputs["action_latents"] for r, inp in zip(rids, inputs, strict=True)
                    },
                    "time_index": {r: inp.tensor_inputs["time_index"] for r, inp in zip(rids, inputs, strict=True)},
                }
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "action_latents": inputs[0].tensor_inputs["action_latents"],
                "time_index": inputs[0].tensor_inputs["time_index"],
            }
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    # Run in the model's native bf16, NOT the engine autocast: autocast keeps
    # norms in fp32, perturbing the velocity ~1 ULP/step — harmless for one
    # image step but amplified geometrically over the multi-step video denoise.
    # The reference pipeline runs pure bf16; parity requires matching it.
    @torch.autocast(device_type="cuda", enabled=False)
    def _states(self, engine_inputs: ModelInputsFromEngine) -> dict:
        """The batch's per-request states: the engine-injected view when
        present, else the submodule's own store (paths that build their own
        ``ModelInputsFromEngine``, e.g. tests and capture harnesses)."""
        states = engine_inputs.per_request_states
        return states if states is not None else self.request_states

    def _gen_attn(self, engine_inputs: ModelInputsFromEngine):
        """The attention resource this step's denoise was planned against.

        Read off the step rather than re-derived: ``declare_step`` names the
        dense backend or the paged one depending on whether the step could
        land on a capture slot, and only the one it named was planned."""
        step = engine_inputs.step
        key = ATTN_GEN if step is not None and ATTN_GEN in step else ATTN
        resources = engine_inputs.resources or self.node_resources
        return resources[key]

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, **kwargs):
        rid = engine_inputs.request_ids[0]
        if graph_walk in PREFILL_WALKS:
            return self._forward_prefill(**kwargs)
        states = self._states(engine_inputs)
        attn = self._gen_attn(engine_inputs)
        if graph_walk in GEN_WALKS:
            return self._forward_image_gen(
                attn, states[rid], kv=self._kv_resource(engine_inputs), **kwargs,
            )
        if graph_walk in SOUND_WALKS:
            return self._forward_video_sound_gen(attn, states[rid], **kwargs)
        if graph_walk in ACTION_WALKS:
            return self._forward_action_gen(attn, states[rid], **kwargs)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _forward_prefill(
        self,
        input_ids: torch.Tensor,
        text_mrope_ids: torch.Tensor,
        cfg=False,
        **kwargs
    ) -> dict:
        label = CFG_BATCHED_LABEL if cfg else COND_LABEL
        self.transformer.prefill_und(input_ids, text_mrope_ids, label)
        return {}

    def _denoise(self, attn, static, latents, vision_timesteps, label):
        return self.transformer.denoise_step(
            latents,
            vision_timesteps,
            static["vision_mrope_ids"],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
            label,
            attn,
        )

    def _cfg_active(self, st, step_index: int) -> bool:
        """Whether this eager denoise step runs classifier-free guidance (both
        branches combined). False ⇒ the conditional branch runs alone — the
        guidance_scale==1 case and, for the t2i recipe, steps whose timestep
        falls outside the guidance_interval [lo, hi].

        ``prepare_inputs`` (through ``_step_info``) and ``_forward_image_gen``
        call this for the same step, so the declared attention (batched vs
        cond-only) matches the forward that runs.

        The captured path does not consult it at all: its graph is fixed at
        both branches and ``postprocess`` combines unconditionally, so an
        out-of-interval step that lands on a capture slot gets guidance where
        the eager one would not. See ``cg_key_info`` for why the interval
        cannot be folded into the capture key."""
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

    def _forward_image_gen(self, attn, st, latents, time_index, kv=None, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        windowed = "ar_schedule" in st
        if windowed:
            # The loop counter is global; each window runs its own fresh
            # scheduler over ar_steps steps, and in kv mode the extra
            # per-window iteration commits the finished window's K/V.
            if step_index >= st["ar_total_iters"]:
                return {"latents": [latents], "time_index": [time_index]}
            window_index, local, commit = self._window_step(st, step_index)
            if commit:
                return self._commit_window(attn, st, latents, time_index, window_index, kv=kv)
            step_index = local
        elif step_index >= len(scheduler.timesteps):
            # The loop may dispatch one step past this request's own count
            # before its stop signal lands; that step is a no-op.
            return {"latents": [latents], "time_index": [time_index]}
        t = scheduler.timesteps[step_index]
        vision_timesteps = torch.full((st["cond"]["num_noisy_vision_tokens"],), t.item(), device=latents.device)

        # Classifier-free guidance is applied only when an uncond branch exists
        # (guidance_scale != 1) and, for the text-to-image recipe, only on the
        # configured timestep interval. Outside the interval the step runs the
        # conditional branch alone (cond-only velocity), matching the recipe.
        cfg_active = self._cfg_active(st, step_index)

        if not cfg_active:
            velocity = self._denoise(attn, st["cond"], latents, vision_timesteps, COND_LABEL)
        elif self.batched_cfg:
            cond_v, uncond_v = self.transformer.denoise_step_batched_cfg(
                latents,
                vision_timesteps,
                st["cond"]["vision_mrope_ids"],
                st["uncond"]["vision_mrope_ids"],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                CFG_BATCHED_LABEL,
                attn,
            )
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
        else:
            cond_v = self._denoise(attn, st["cond"], latents, vision_timesteps, COND_LABEL)
            uncond_v = self._denoise(attn, st["uncond"], latents, vision_timesteps, UNCOND_LABEL)
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)

        new_latents = self._scheduler_step(st, velocity, t, latents)
        if st.get("vmask") is not None:
            # Video-to-video: re-inject the clean conditioning frames after the
            # scheduler step (the reference pipeline does the same) so scheduler
            # rounding can't drift the pinned latents.
            new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
        if windowed and local + 1 == st["ar_steps"] and not st.get("ar_kv_mode"):
            # Chained: the window ends at its last denoise step. kv windows
            # end at their commit iteration instead.
            return self._finish_window(st, new_latents, time_index, window_index)
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

    def _forward_video_sound_gen(self, attn, st, latents, sound_latents, time_index, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        if step_index >= len(scheduler.timesteps):
            # One-past-the-end dispatch before the stop signal lands: no-op.
            return {
                "latents": [latents],
                "sound_latents": [sound_latents],
                "time_index": [time_index],
            }
        t = scheduler.timesteps[step_index]
        vts = torch.full((st["cond"]["num_noisy_vision_tokens"],), t.item(), device=latents.device)
        sts = torch.full((st["num_sound"],), t.item(), device=latents.device)
        cfg_active = self._cfg_active(st, step_index)

        if not cfg_active:
            velocity, sound_v = self.transformer.denoise_step(
                latents, vts, st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["cond"]["vision_token_shapes"], st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"], COND_LABEL, attn,
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
        elif self.batched_cfg:
            (cond_v, s_c), (uncond_v, s_u) = self.transformer.denoise_step_batched_cfg(
                latents, vts,
                st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["uncond"]["position_ids"][:, st["uncond"]["und_len"]:],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"], CFG_BATCHED_LABEL, attn,
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            sound_v = s_u + st["gs"] * (s_c - s_u)
        else:
            cond_v, s_c = self.transformer.denoise_step(
                latents, vts, st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["cond"]["vision_token_shapes"], st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"], COND_LABEL, attn,
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
            uncond_v, s_u = self.transformer.denoise_step(
                latents, vts, st["uncond"]["position_ids"][:, st["uncond"]["und_len"]:],
                st["uncond"]["vision_token_shapes"], st["uncond"]["vision_noisy_frame_indexes"],
                st["uncond"]["mse_gen_indexes"], UNCOND_LABEL, attn,
                **self._sound_kwargs(st["uncond"], sound_latents, sts),
            )
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            sound_v = s_u + st["gs"] * (s_c - s_u)

        new_latents, new_sound = self._sound_scheduler_step(
            st, latents, sound_latents, velocity, sound_v, t
        )
        if st.get("vmask") is not None:
            # Video-to-video: re-inject the clean conditioning frames on the
            # video band after the joint step (the reference pipeline does the
            # same); the sound band has no clean frames.
            new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
        return {
            "latents": [new_latents],
            "sound_latents": [new_sound],
            "time_index": [time_index + 1],
        }

    def _denoise_action(self, attn, static, latents, action_latents, vts, ats, domain, label):
        und_len = static["und_len"]
        return self.transformer.denoise_step(
            latents,
            vts,
            static["position_ids"][:, und_len:],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
            label,
            attn,
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
        video_v = video_v * (1.0 - st["vmask"])
        action_v = action_v * (1.0 - st["action_clean_mask"])
        action_v[..., raw:] = 0
        nv = video_v.numel()
        packed = torch.cat([video_v.reshape(1, -1), action_v.reshape(1, -1)], dim=1)
        packed_lat = torch.cat([latents.reshape(1, -1), action_latents.reshape(1, -1)], dim=1)
        packed_next = st["scheduler"].step(packed, t, packed_lat, return_dict=False)[0]
        new_latents = packed_next[:, :nv].reshape(latents.shape)
        new_action = packed_next[:, nv:].reshape(1, chunk, adim)
        new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * latents
        new_action = (
            (1.0 - st["action_clean_mask"]) * new_action
            + st["action_clean_mask"] * action_latents
        )
        new_action[..., raw:] = 0
        return new_latents, new_action

    def _forward_action_gen(self, attn, st, latents, action_latents, time_index, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        if step_index >= len(scheduler.timesteps):
            # One-past-the-end dispatch before the stop signal lands: no-op.
            return {
                "latents": [latents],
                "action_latents": [action_latents],
                "time_index": [time_index],
            }
        t = scheduler.timesteps[step_index]
        device = latents.device
        vts = torch.full((st["cond"]["num_noisy_vision_tokens"],), t.item(), device=device)
        ats = torch.full((st["cond"]["num_noisy_action_tokens"],), t.item(), device=device)
        domain = st["domain_t"]

        if st["uncond"] is None:
            video_v, action_v = self._denoise_action(
                attn, st["cond"], latents, action_latents, vts, ats, domain, COND_LABEL
            )
        elif self.batched_cfg:
            (video_v, action_v), (v_u, a_u) = self.transformer.denoise_step_batched_cfg(
                latents,
                vts,
                st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["uncond"]["position_ids"][:, st["uncond"]["und_len"]:],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                CFG_BATCHED_LABEL,
                attn,
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
            video_v, action_v = self._denoise_action(
                attn, st["cond"], latents, action_latents, vts, ats, domain, COND_LABEL
            )
            v_u, a_u = self._denoise_action(
                attn, st["uncond"], latents, action_latents, vts, ats, domain, UNCOND_LABEL
            )
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
        sts = [self.request_states.get(rid) for rid in batch.request_ids]
        if any(st is None for st in sts):
            return False
        if (
            batch.graph_walk in GEN_WALKS
            or batch.graph_walk in SOUND_WALKS
            or batch.graph_walk in PREFILL_WALKS
        ):
            # Image/video batch only in the two-branch guidance regime, so one
            # batched-CFG plan covers them. (Batches are per graph walk, so
            # sound requests only ever batch with sound requests.) Windowed
            # requests join denoise batches — the batched forward maps their
            # loop counter to the within-window step and handles chained
            # window boundaries — but a kv commit iteration is a different
            # step (an append) and drops the batch to the sequential path.
            if not all(st["uncond"] is not None for st in sts):
                return False
            if batch.graph_walk not in GEN_WALKS:
                # Windowed requests join only generation-loop batches; their
                # prefill stays sequential (these passes carry no time_index,
                # and the committed kv text prefix must not depend on which
                # requests happened to arrive together).
                return all("ar_schedule" not in st for st in sts)
            for st, inp in zip(sts, model_inputs, strict=True):
                if "ar_schedule" not in st:
                    continue
                ti = inp.tensor_inputs["time_index"]
                if self._window_step(st, int(ti.reshape(-1)[0].item()))[2]:
                    return False
            return True
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
        if graph_walk in PREFILL_WALKS:
            label = CFG_BATCHED_LABEL if kwargs.get("cfg") else COND_LABEL
            self.transformer.prefill_und(input_ids, text_mrope_ids, label)
            return {}
        if graph_walk in ACTION_WALKS:
            return self._forward_batched_action(engine_inputs, latents, action_latents, time_index)
        if graph_walk in SOUND_WALKS:
            return self._forward_batched_sound(engine_inputs, latents, sound_latents, time_index)
        if graph_walk not in GEN_WALKS:
            raise ValueError(f"Cosmos3 batched forward only supports generation walks, got {graph_walk!r}")
        states = self._states(engine_inputs)
        attn = self._gen_attn(engine_inputs)
        reqs, meta = [], []
        for rid in engine_inputs.request_ids:
            st = states[rid]
            lat, ti = latents[rid], time_index[rid]
            step_index = int(ti.reshape(-1)[0].item())
            if "ar_schedule" in st:
                # Windowed: the loop counter is global; the schedule index is
                # the within-window step (commit iterations never batch).
                step_index = self._window_step(st, step_index)[1]
            n_steps = len(st["scheduler"].timesteps)
            # A request may be one step past its denoise count (a discarded extra
            # step) while others in the batch are still running; clamp its
            # timestep so the shared forward can't index past the schedule, and
            # skip its scheduler step below.
            t = st["scheduler"].timesteps[min(step_index, n_steps - 1)]
            reqs.append({
                "latents": lat,
                "vision_timesteps": torch.full((st["cond"]["num_noisy_vision_tokens"],), t.item(), device=lat.device),
                "position_ids_cond": st["cond"]["vision_mrope_ids"],
                "position_ids_uncond": st["uncond"]["vision_mrope_ids"],
                "vision_token_shapes": st["cond"]["vision_token_shapes"],
                "vision_noisy_frame_indexes": st["cond"]["vision_noisy_frame_indexes"],
                "vision_mse_loss_indexes": st["cond"]["mse_gen_indexes"],
            })
            meta.append((rid, st, lat, ti, t, step_index))

        results = self.transformer.denoise_step_batched(reqs, CFG_BATCHED_LABEL, attn)

        out = {}
        for (rid, st, lat, ti, t, local), (cond_v, uncond_v) in zip(meta, results, strict=True):
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            new_latents = self._scheduler_step(st, velocity, t, lat)
            if st.get("vmask") is not None:
                # Video-to-video: re-inject the clean conditioning frames, as in
                # the single-request path.
                new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
            if (
                "ar_schedule" in st
                and local + 1 == st["ar_steps"]
                and not st.get("ar_kv_mode")
            ):
                # Chained window boundary inside a batch: emit + stage, as in
                # the single-request path.
                window_index = self._window_step(st, int(ti.reshape(-1)[0].item()))[0]
                out[rid] = self._finish_window(st, new_latents, ti, window_index)
            else:
                out[rid] = {"latents": [new_latents], "time_index": [ti + 1]}
        return out

    def _forward_batched_sound(self, engine_inputs, latents, sound_latents, time_index):
        """Run several sound requests' joint [video | sound] denoise step in one
        forward. Mirrors the image batched path with per-request sound bands:
        one batched transformer pass, then per request the guidance combine and
        its own joint [video | sound] scheduler step."""
        states = self._states(engine_inputs)
        attn = self._gen_attn(engine_inputs)
        reqs, meta = [], []
        for rid in engine_inputs.request_ids:
            st = states[rid]
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
                "vision_timesteps": torch.full((st["cond"]["num_noisy_vision_tokens"],), t.item(), device=lat.device),
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

        results = self.transformer.denoise_step_batched(reqs, CFG_BATCHED_LABEL, attn)

        out = {}
        for (rid, st, lat, snd, ti, t), ((cond_v, s_c), (uncond_v, s_u)) in zip(meta, results, strict=True):
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            sound_v = s_u + st["gs"] * (s_c - s_u)
            new_latents, new_sound = self._sound_scheduler_step(st, lat, snd, velocity, sound_v, t)
            if st.get("vmask") is not None:
                # Video-to-video: re-inject the clean conditioning frames on the
                # video band, as in the single-request path.
                new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
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
        states = self._states(engine_inputs)
        attn = self._gen_attn(engine_inputs)
        rids = engine_inputs.request_ids
        with_cfg = states[rids[0]]["uncond"] is not None
        reqs, meta = [], []
        for rid in rids:
            st = states[rid]
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
                "vision_timesteps": torch.full((st["cond"]["num_noisy_vision_tokens"],), t.item(), device=lat.device),
                "action_timesteps": torch.full((st["cond"]["num_noisy_action_tokens"],), t.item(), device=lat.device),
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

        results = self.transformer.denoise_step_action_batched(
            reqs, CFG_BATCHED_LABEL, attn, with_cfg
        )

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
        # The capture is bit-faithful at every resolution (the rotary uses a
        # broadcast multiply — see Cosmos3RotaryEmbedding), but only HELPS
        # the launch-bound tiers: the graph's per-step input copies grow
        # with resolution and lose to the eager dense path at large latents.
        # Capture only below COSMOS3_GRAPH_MAX_LATENT_AREA (latent H*W).
        max_area = int(os.environ.get(
            "COSMOS3_GRAPH_MAX_LATENT_AREA", self.config.graph_max_latent_area))
        if two_d:
            max_area = min(max_area, 1000)  # 256p latent 240 captures; 480p 1560 does not
        # Video tiers (height, width, frames): a plain clip length and/or the
        # windowed rollout's window; one graph per latent shape serves t2v,
        # i2v and chained windows (see ``_capture_key``).
        video_env = os.environ.get("COSMOS3_GEN_CAPTURE_VIDEO")
        if video_env:
            video_tiers = tuple(
                tuple(int(x) for x in tier.split("x")) for tier in video_env.split(",") if tier.strip()
            )
        else:
            video_tiers = tuple(tuple(int(x) for x in tier) for tier in (self.config.gen_capture_video or ()))
        tiers = [(h, w, 1, IMAGE_GEN_WALK, None) for h, w in resolutions] + [
            (h, w, f, VIDEO_GEN_WALK, [VIDEO_GEN_WALK, VIDEO_GEN_AR_WALK]) for h, w, f in video_tiers
        ]
        for height, width, frames, walk, replay_walks in tiers:
            latent_shape = self._latent_shape(height, width, num_frames=frames)
            latent_area = latent_shape[3] * latent_shape[4]
            if latent_area > max_area:
                logger.info(
                    "Cosmos3: skipping CUDA-graph capture for %dx%dx%d (latent H*W "
                    "%d > %d -> graph net-slower than eager dense here -> eager)",
                    height, width, frames, latent_area, max_area,
                )
                continue
            if tuple(latent_shape) in self._capture_layout:
                continue
            configs.append(self._gen_capture_config(
                latent_shape, height, width, frames, device, dtype, capture_batch_sizes,
                walk, replay_walks,
            ))

        configs.extend(self._prefill_capture_configs(device))
        return configs

    def _gen_capture_config(
        self, latent_shape, height, width, frames, device, dtype, capture_batch_sizes,
        walk, replay_walks,
    ) -> BatchedCudaGraphConfig:
        """One denoise-step capture bucket per latent shape. The graph is
        built with every frame declared noisy (its token layout is baked); the
        request's clean/noisy layout rides in as the ``noisy_token_mask``
        static input, which zeroes the timestep embedding on clean frames —
        the same tokens the eager scatter-add skips — so the noisy frames'
        velocities match the eager step exactly, and ``postprocess`` zeroes
        the clean frames' velocities the way the eager unpatchify would."""
        static = self._build_static(
            [0] * 8, height, width, num_frames=frames, fps=24.0,
            has_image_condition=False, device=device,
        )
        num_vision = static["num_vision_tokens"]
        self._capture_layout[tuple(latent_shape)] = {
            "vision_token_shapes": static["vision_token_shapes"],
            "vision_noisy_frame_indexes": static["vision_noisy_frame_indexes"],
            "mse_gen_indexes": static["mse_gen_indexes"],
        }
        single = ARNodeInputs(
            input_seq_len=num_vision,
            tensor_inputs={
                "latents": torch.zeros(latent_shape, device=device, dtype=dtype),
                "vision_timesteps": torch.zeros(num_vision, device=device, dtype=torch.float32),
                "position_ids_cond": static["vision_mrope_ids"].clone(),
                "position_ids_uncond": static["vision_mrope_ids"].clone(),
                "noisy_token_mask": torch.ones(num_vision, device=device, dtype=torch.float32),
            },
            # What the capture's own `declare_step` reads: this bucket's
            # step is the two-branch one, over the paged backend. Padding
            # rows carry it too, so a partly-filled replay declares the
            # same segments the capture did.
            resource_step_info=GenStepInfo(
                cfg=True, cfg_active=True, capture_key=tuple(latent_shape),
            ),
        )
        return BatchedCudaGraphConfig(
            capture_graph_walk=walk,
            replay_graph_walks=replay_walks,
            single_request_inputs=single,
            # One bucket per latent shape: the token layout is baked into the
            # capture, so a request at another latent shape must not land
            # here. `cg_key_info` returns this same latent shape, and
            # `declare_step` stamps it on the step.
            additional_key_info=tuple(latent_shape),
            capture_forward_method="forward_captured",
            compile=False,
            capture_batch_sizes=capture_batch_sizes,
            # The captured sizes (default bs=1; COSMOS3_GEN_CAPTURE_BS adds
            # more) are an acceleration subset, not a batch ceiling —
            # uncaptured sizes / mixed resolutions run the eager batched
            # denoise, so don't cap max_batch_size to them.
            caps_eager_batch_size=False,
            # This bucket's step always runs both guidance branches combined
            # into one KV plan (``resource_step_info`` above is
            # cfg=True/cfg_active=True unconditionally) — `single.input_seq_len`
            # is one branch's span (declare_step replicates it per label), but
            # the combined plan commits both branches' tokens, so the static
            # buffer needs double the capacity or the real replay's KV plan
            # overruns it (KVPlanState.copy_ shape mismatch).
            total_tokens_multiplier=2,
        )

    def _prefill_capture_configs(self, device) -> list:
        configs = []
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

            configs.append(PackedCudaGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                replay_graph_walks=list(PREFILL_WALKS),
                capture_token_lengths=prefill_tokens,
                make_node_input=lambda n: self._prefill_capture_input(
                    n, device, mrope_dtype,
                ),
                # Only the two-branch prefill is captured; `cg_key_info`
                # answers True for exactly that case. A guidance-off prefill
                # declares one label over a different token count and runs
                # eager.
                additional_key_info=True,
                compile=False,
                capture_batch_sizes=prefill_bs,
                caps_eager_batch_size=False,
            ))
        return configs

    def _prefill_capture_input(
        self, num_tokens: int, device, mrope_dtype,
    ) -> ARNodeInputs:
        """One row of the combined cond+uncond prefill pack, at capture shape.

        ``num_tokens`` is the row's whole contribution — both branches — since
        that is what the runner buckets on. How it splits between them is
        arbitrary here: the capture only fixes the packed token count, and the
        per-label spans a replay attends over come from the real request's
        declaration, which is re-planned into the same buffers before the
        replay. ``num_tokens == 0`` is the padding row.
        """
        lens = {
            COND_LABEL: num_tokens - num_tokens // 2,
            UNCOND_LABEL: num_tokens // 2,
        }
        tensors = {}
        for label, n in lens.items():
            tensors[f"input_ids_{label}"] = torch.zeros(
                n, dtype=torch.long, device=device,
            )
            tensors[f"text_mrope_ids_{label}"] = torch.zeros(
                (3, n), dtype=mrope_dtype, device=device,
            )
        return ARNodeInputs(
            input_seq_len=num_tokens,
            tensor_inputs=tensors,
            kwargs=dict(cfg=True, seq_lens=lens),
        )

    # Native bf16, not the engine autocast — as forward()/forward_batched():
    # the runner captures under the engine's autocast scope, and a graph
    # captured that way replays the autocast'd kernels for the request's
    # whole denoise (measured: ~23 dB latent PSNR from the native step after
    # one iteration on Edge t2i, which compounds over the loop).
    @torch.autocast(device_type="cuda", enabled=False)
    def forward_captured(
        self, graph_walk, engine_inputs: ModelInputsFromEngine,
        latents, vision_timesteps, position_ids_cond, position_ids_uncond, noisy_token_mask,
        **kwargs,
    ) -> dict:
        """Velocity-only denoise forward captured into a CUDA graph: both guidance
        branches in one pass (the combined plan), no scheduler step. The token
        layout is baked per resolution; the latents, timestep and rotary positions
        are static-buffer inputs stacked on a leading batch dim. A single request
        keeps the two-branch path; a concurrent batch runs the per-request denoise
        (the same compute as the eager cross-request forward), one transformer pass
        over the whole batch."""
        attn = self._gen_attn(engine_inputs)
        layout = self._capture_layout[tuple(latents.shape[1:])]
        rids = engine_inputs.request_ids
        if latents.shape[0] == 1:
            # Captured into the denoise CUDA graph. Under SP the Ulysses
            # exchange must be all-gather (all-to-all is grouped p2p send/recv —
            # not graph-replayable); the flag holds across warmup/capture/replay
            # so those kernels compile during eager warmup. No-op without SP.
            cond_v, uncond_v = self.transformer.denoise_step_batched_cfg(
                latents[0], vision_timesteps[0], position_ids_cond[0], position_ids_uncond[0],
                layout["vision_token_shapes"], layout["vision_noisy_frame_indexes"],
                layout["mse_gen_indexes"], CFG_BATCHED_LABEL, attn,
                prefer_all_gather=True, noisy_token_mask=noisy_token_mask[0],
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
                "noisy_token_mask": noisy_token_mask[i],
            }
            for i in range(latents.shape[0])
        ]
        results = self.transformer.denoise_step_batched(reqs, CFG_BATCHED_LABEL, attn)
        return {
            rid: {"cond_v": [cond_v], "uncond_v": [uncond_v]}
            for rid, (cond_v, uncond_v) in zip(rids, results, strict=True)
        }

    def postprocess(self, request_id, request_info, outputs, inputs=None, **kwargs):
        """Captured-path tail: the classifier-free-guidance combine and the
        (Python, multistep) scheduler step the graph can't hold, finished from
        the step's own ``inputs``. Mirrors the tail of ``_forward_image_gen``.
        Eager forwards already emit finished ``latents``/``time_index`` and
        pass through untouched (no ``cond_v`` in their outputs)."""
        if request_info.graph_walk not in GEN_WALKS or "cond_v" not in outputs:
            return
        st = self.request_states[request_id]
        velocity = outputs["uncond_v"][0] + st["gs"] * (
            outputs["cond_v"][0] - outputs["uncond_v"][0]
        )
        latents = inputs.tensor_inputs["latents"]
        time_index = inputs.tensor_inputs["time_index"]
        outputs.clear()
        # step_index is in range here: prepare_inputs vetoes the loop's extra
        # dispatched step and the engine prunes vetoed requests before postprocess.
        step_index = int(time_index.reshape(-1)[0].item())
        windowed = "ar_schedule" in st
        window_index = local = None
        if windowed:
            # Chained windows only (kv requests never take a lease); the loop
            # counter is global, the schedule index is the within-window step.
            window_index, local, _ = self._window_step(st, step_index)
            step_index = local
        # The graph predicts every frame; clean frames (i2v anchor, chained
        # overlap) get the zero velocity the eager unpatchify gives them.
        velocity = velocity * self._noisy_masks(st, velocity.device)[1].to(velocity.dtype)
        t = st["scheduler"].timesteps[step_index]
        new_latents = self._scheduler_step(st, velocity, t, latents)
        if st.get("vmask") is not None:
            new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
        if windowed and local + 1 == st["ar_steps"]:
            outputs.update(self._finish_window(st, new_latents, time_index, window_index))
            return
        outputs["latents"] = [new_latents]
        outputs["time_index"] = [time_index + 1]

    def check_stop(self, request_id, request_info, outputs) -> set[str]:
        """Stop this request's denoise loop once it has run its own step count.

        The loop is built with a fixed upper-bound iteration count
        (``config.max_inference_steps``); each request runs only as many steps as
        its scheduler holds (e.g. image 50, video 35, action 30, distilled policy
        ~4), which can differ between concurrent requests. Runs on the worker's
        slow-postprocess path, so reading the per-request step count is fine. The
        one extra step the loop dispatches before this stop takes effect is
        vetoed by prepare_inputs (the forwards also guard it for direct
        callers, e.g. tests)."""
        st = self.request_states.get(request_id)
        if st is None:
            return set()
        loop = {
            ACTION_GEN_WALK: ACTION_GEN_LOOP,
            ACTION_VIDEO_GEN_WALK: ACTION_VIDEO_GEN_LOOP,
            VIDEO_GEN_WALK: VIDEO_GEN_LOOP,
            VIDEO_GEN_AR_WALK: VIDEO_GEN_AR_LOOP,
            VIDEO_SOUND_GEN_WALK: VIDEO_SOUND_GEN_LOOP,
        }.get(request_info.graph_walk, IMAGE_GEN_LOOP)
        iter_idx = request_info.dynamic_loop_iter_counts.get(loop, 0)
        # Windowed requests run every window's iterations in one loop; others
        # stop at their scheduler's step count.
        total = (
            st["ar_total_iters"] if "ar_schedule" in st
            else len(st["scheduler"].timesteps)
        )
        if iter_idx + 1 >= total:
            return {loop}
        return set()


class Cosmos3VAEEncoderSubmodule(NodeSubmodule):
    """Wan VAE conditioning-encode node (STATELESS): the request's conditioning
    image or video -> clean anchor latents for the denoise loop.

    Runs in parallel with the DiT's understanding-tower prefill (the conditioned
    prefill walks put the two nodes in a ``Parallel`` section) and emits
    ``cond_latents`` as a persist signal the conductor threads into the
    generation walk's first iteration. Everything is re-derived from the request
    metadata so the node stays stateless and placeable on any rank. Per mode:
      - image-to-video: the single conditioning frame, encoded standalone (the
        Wan VAE is temporally causal, so frame 0 encodes bit-identically alone);
      - action policy / forward-dynamics: the frame repeated across the clip,
        full-clip encoded (mirrors the reference pipelines);
      - action inverse-dynamics: the whole observed clip;
      - video-to-video: the ``condition_video_keep`` prefix (short clips padded
        by repeating the last frame), encoded and scattered into the pinned
        frames of a full-latent-shape tensor (zeros elsewhere).
    """

    # One-shot per-request encode at request-specific shapes; nothing to gain
    # from compiling the submodule wrapper.
    disable_torch_compile = True

    def __init__(self, vae, config):
        super().__init__()
        # Dedicated fp32 instance (not shared with the decoder node): encode
        # runs fp32 everywhere (reference behavior; TF32 is the fast conv3d
        # path pre-9.16 cuDNN) while the decode dtype is cuDNN-gated — sharing
        # would re-cast the full VAE weights every encode/decode interleave.
        self.vae = vae
        self.config = config
        self._video_processor = None

    def _latent_t(self, num_frames: int) -> int:
        return 1 if num_frames == 1 else 1 + (num_frames - 1) // self.config.vae.scale_factor_temporal

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        """Shape the conditioning pixels for the encode: mode-dependent frame
        selection/padding plus the [-1, 1] resize-normalize, all derived from
        the request metadata (no per-request state)."""
        from diffusers.video_processor import VideoProcessor

        md = fwd_info.step_metadata
        height, width = int(md.get("height", 256)), int(md.get("width", 256))
        num_frames = int(md.get("num_frames", 1))
        is_action = bool(md.get("action_mode"))
        device = self.get_device()
        if self._video_processor is None:
            self._video_processor = VideoProcessor(
                vae_scale_factor=self.config.vae.scale_factor_spatial, resample="bilinear"
            )

        out_kwargs: dict = {}
        video = (inputs or {}).get("video_inputs")
        image = (inputs or {}).get("image_inputs")
        if video:
            # load_video gives [T, C, H, W] in [0, 1].
            clip = video[0]
            condition_indexes = None if is_action else md.get("condition_frame_indexes_vision")
            if condition_indexes:
                # Video-to-video: only the first max(indexes)*tcf + 1 pixel
                # frames feed the pinned latent frames (the Wan VAE is temporally
                # causal), taken from the start or end per ``condition_video_keep``;
                # short inputs pad by repeating the last frame.
                tcf = self.config.vae.scale_factor_temporal
                cpf = min(max(condition_indexes) * tcf + 1, num_frames)
                clip = clip[-cpf:] if md.get("condition_video_keep") == "last" else clip[:cpf]
                if clip.shape[0] < cpf:
                    clip = torch.cat([clip, clip[-1:].expand(cpf - clip.shape[0], -1, -1, -1)], dim=0)
                s = self.config.vae.scale_factor_spatial
                out_kwargs = {
                    "condition_indexes": tuple(condition_indexes),
                    "latent_shape": (
                        1, self.config.latent_channel, self._latent_t(num_frames),
                        height // s, width // s,
                    ),
                }
            else:
                # Action inverse-dynamics: the whole observed clip.
                clip = clip[:num_frames]
            frames = [
                self._video_processor.preprocess(clip[i], height=height, width=width).squeeze(0)
                for i in range(clip.shape[0])
            ]
            vision = torch.stack(frames, dim=1).unsqueeze(0).to(device=device, dtype=torch.float32)
        elif image:
            # load_image gives [C, H, W] in [0, 1]. Image-to-video follows the
            # deployment's conditioning_resize recipe: "stretch" is the
            # diffusers VideoProcessor resize-normalize the Nano/Super
            # checkpoints were validated against (also what the action modes
            # use for their repeated frame), "aspect_crop" the diffusers 0.40 /
            # vLLM-Omni cover-scale + center-crop recipe of the Edge yamls.
            if is_action or self.config.conditioning_resize == "stretch":
                frame = self._video_processor.preprocess(image[0], height=height, width=width).to(
                    device=device, dtype=torch.float32
                )
                vision = frame.unsqueeze(2)
            else:
                from mstar.model.cosmos3.components.conditioning import prepare_conditioning_frames

                vision = prepare_conditioning_frames(
                    image[0], height, width, self.config.conditioning_resize,
                ).to(device=device, dtype=torch.float32)
            if is_action and num_frames > 1:
                # Policy / forward-dynamics condition on latent frame 0 only
                # (their vmask), and the Wan VAE is temporally causal: frame 0's
                # latent is bit-identical whether the frame is encoded alone or
                # repeated over the clip as the reference pipelines do
                # (measured 0.0 on Edge at 480p). Encode the one frame — 38 ms
                # instead of ~350 ms for a 33-frame clip — and let ``forward``
                # place it in the full latent shape the denoise loop pins.
                s = self.config.vae.scale_factor_spatial
                out_kwargs = {
                    "condition_indexes": (0,),
                    "latent_shape": (
                        1, self.config.latent_channel, self._latent_t(num_frames),
                        height // s, width // s,
                    ),
                }
        else:
            raise ValueError("Cosmos3 vae_encoder received neither an image nor a video conditioning input.")
        return NodeInputs(tensor_inputs={"vision": vision}, kwargs=out_kwargs)

    def forward(
        self, graph_walk, engine_inputs: ModelInputsFromEngine, vision,
        condition_indexes=None, latent_shape=None, **kwargs,
    ):
        vae = self.vae
        # The engine may cast the submodule; re-assert the fp32 encode weights.
        if next(vae.parameters()).dtype != torch.float32:
            vae.float()
        device = vision.device
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device)).view(
            1, -1, 1, 1, 1
        )
        # fp32 outside autocast; pipeline-side latent normalization; the DiT
        # consumes the latents in its bf16 checkpoint dtype.
        with torch.autocast(device_type=device.type, enabled=False):
            raw_mu = vae.encode(vision).latent_dist.mode()
        latents = ((raw_mu - mean) * inv_std).to(torch.bfloat16)
        if condition_indexes:
            full = torch.zeros(latent_shape, device=device, dtype=latents.dtype)
            for f in condition_indexes:
                full[:, :, f] = latents[:, :, f]
            latents = full
        return {"cond_latents": [latents]}


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
        # The decode is 3D-conv bound, one-shot per request at request-specific
        # shapes, and not graph-captured; torch.compile fuses its pointwise
        # epilogues (~20-30% off the 189-frame video decode, neutral for
        # images) at the cost of one trace per new shape. fullgraph=False
        # breaks around the VAE's causal-conv feature cache.
        # COSMOS3_COMPILE_VAE=0 disables.
        self._decode = vae.decode if vae is not None else None
        self._decode_compiled = False
        self._warmed_decode_shapes: set[tuple[int, ...]] = set()
        if vae is not None and os.environ.get("COSMOS3_COMPILE_VAE", "1").lower() not in (
            "0", "false", "no", "off",
        ):
            self._decode = torch.compile(vae.decode, fullgraph=False, dynamic=False)
            self._decode_compiled = True
            logger.info("Cosmos3 VAE decode torch.compile enabled")
        # Resolve + log the decode dtype now (a cheap cuDNN-version read) so the
        # choice is fixed at startup, not on the first request.
        self._decode_dtype()

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={"latents": inputs["latents"][0]})

    def _decode_dtype(self):
        # cuDNN ships fast Hopper bf16 conv3d for the Wan-VAE decode only from
        # 9.16; before that bf16 is 2-10x slower than fp32/TF32 (measured via a
        # single-variable cuDNN swap), so gate on the live version — an upgrade
        # flips it automatically. COSMOS3_VAE_DECODE_FP32 overrides.
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

    def _decode_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> uint8 pixel frames ``[1, 3, T, H, W]``."""
        vae = self.vae
        vae_dtype = self._decode_dtype()
        if next(vae.parameters()).dtype != vae_dtype:
            vae = vae.to(vae_dtype)
        mean = torch.tensor(vae.config.latents_mean, dtype=vae_dtype, device=latents.device).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=vae_dtype, device=latents.device)).view(
            1, -1, 1, 1, 1
        )
        # Force z to the VAE dtype: the engine's outer autocast promotes the
        # `1.0 / std` division to fp32, and a fp32 z into the autocast-off bf16
        # conv3d fails ("Input type (float) and bias type (BFloat16)"). No-op
        # on the fp32-decode path.
        z = (latents.to(vae_dtype) / inv_std + mean).to(vae_dtype)
        with torch.autocast(device_type="cuda", enabled=False):
            if (
                self._decode_compiled
                and vae_dtype == torch.bfloat16
                and tuple(z.shape) not in self._warmed_decode_shapes
            ):
                # A process's first execution of the compiled bf16 decode
                # returns corrupted frames on some torch/cuDNN stacks — even
                # when the kernels come from a warm on-disk inductor cache —
                # while every later call of the same graphs is correct. Decode
                # once per new latent shape and discard, so served frames
                # always come from warmed graphs. The fp32 decode does not
                # exhibit this and skips the warm-up.
                self._decode(z)
                self._warmed_decode_shapes.add(tuple(z.shape))
            decoded = self._decode(z).sample  # [1, 3, T, H, W] in [-1, 1]
        # Quantize to 8-bit here (the output is an 8-bit image/mp4 either way) so
        # only the uint8 frames cross the SHM edge to the data worker, not a 4x
        # larger fp32 tensor — the decoded video transfer dominates the fixed cost
        # at higher resolutions.
        return (decoded / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, latents, **kwargs):
        image = self._decode_pixels(latents)
        # Route the decoded tensor to the active walk's emit edge: image_gen
        # emits "image_output" (one frame); the video walks (plain, sound,
        # forward-dynamics) emit "video_output".
        out_name = (
            "video_output"
            if graph_walk in (VIDEO_GEN_WALK, VIDEO_SOUND_GEN_WALK, ACTION_VIDEO_GEN_WALK)
            else "image_output"
        )
        return {out_name: [image]}


class Cosmos3VAEDecoderARSubmodule(Cosmos3VAEDecoderSubmodule):
    """Streaming Wan VAE decode for windowed AR video (ported from #198).

    Consumes one committed window's latents per stream chunk. Each window is
    decoded behind a re-decoded left context (the last few latents of the
    stream so far) so the causal conv stack is warm at the kept frames; the
    context- and overlap-derived pixels are trimmed, and the video is either
    emitted per window (``stream_video``) or assembled and emitted once the
    last window lands. The chunk count is the completion signal — it is known
    per request up front — so the stream's terminal flush (an empty pass, and
    the only pass a non-windowed request's idle stream ever delivers) runs as
    a no-op forward: the pass must complete normally for the partition to
    report done, so it is not vetoed.
    """

    def __init__(self, vae, config):
        super().__init__(vae, config)
        # Per-session decode context (the latents behind a session's last
        # frames), so a resumed rollout's first window decodes seamlessly.
        self._session_tails: OrderedDict[str, torch.Tensor] = OrderedDict()

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        chunks = (inputs or {}).get("window_latents") or []
        if not chunks:
            return NodeInputs(tensor_inputs={"latents": torch.empty(0)})
        return NodeInputs(tensor_inputs={"latents": chunks[0]})

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, latents, **kwargs):
        if latents.numel() == 0:
            return {}
        rid = engine_inputs.request_ids[0]
        st = self.request_state(rid)
        if "ar_chunks" not in st:
            md = engine_inputs.per_request_info[rid].step_metadata
            session_id = md.get("session_id")
            resume = int(md.get("resume_latent_units", 0) or 0)
            st.add_all(
                ar_chunks=0,
                ar_windows=int(md["num_windows"]),
                ar_overlap=int(md["overlap_latent_units"]),
                ar_ctx=max(1, int(self.config.windowed_decode_context_latents)),
                # The schedule may be padded up to whole windows; the request's
                # frame count is what the assembled video is trimmed to.
                ar_out_frames=int(md["num_frames"]),
                ar_stream=bool(md.get("stream_video")),
                ar_emitted=0,
                ar_pixels=[],
                ar_session_id=str(session_id) if session_id else None,
                # A resumed session: window 0's pinned head duplicates frames
                # the client already has, so it is trimmed like an overlap,
                # and the session's decode context (when this node still
                # holds it) warms the conv stack behind the first new frame.
                ar_resume=resume,
            )
            if resume and session_id and str(session_id) in self._session_tails:
                st.add("ar_tail", self._session_tails[str(session_id)])
        index = st["ar_chunks"]
        overlap = st["ar_overlap"] if index > 0 else st["ar_resume"]
        new = latents[:, :, overlap:] if overlap else latents
        tail = st.get("ar_tail")
        # The retained tail is already capped at ar_ctx latents, so appending
        # the window's new latents yields the [context | window] decode input
        # and the next tail in one tensor.
        stream_tail = new if tail is None else torch.cat([tail, new.to(tail.dtype)], dim=2)
        pixels = self._decode_pixels(stream_tail)
        if tail is not None:
            # A mid-stream latent decodes to scale_factor_temporal frames; the
            # leading context (and the clip-start special frame, which falls
            # inside it) is exactly the part being trimmed.
            keep = new.shape[2] * self.config.vae.scale_factor_temporal
            pixels = pixels[:, :, -keep:]
        st.add("ar_tail", stream_tail[:, :, -st["ar_ctx"]:])
        st.add("ar_chunks", index + 1)
        if index + 1 >= st["ar_windows"] and st["ar_session_id"]:
            self._session_tails[st["ar_session_id"]] = st["ar_tail"]
            self._session_tails.move_to_end(st["ar_session_id"])
            while len(self._session_tails) > max(1, int(self.config.session_store_size)):
                self._session_tails.popitem(last=False)
        if st["ar_stream"]:
            # Deliver each window as its own chunk, capped at the frames still
            # owed (the padded final window can outrun the requested count);
            # nothing is retained across windows.
            chunk = pixels[:, :, : st["ar_out_frames"] - st["ar_emitted"]]
            st.add("ar_emitted", st["ar_emitted"] + chunk.shape[2])
            return {"video_output": [chunk]} if chunk.shape[2] else {}
        st["ar_pixels"].append(pixels)
        if index + 1 < st["ar_windows"]:
            return {}
        video = torch.cat(st["ar_pixels"], dim=2)[:, :, : st["ar_out_frames"]]
        return {"video_output": [video]}


class Cosmos3VisionEncoderSubmodule(NodeSubmodule):
    """The Edge reasoner's vision tower + projector (STATELESS): the request's
    packed image/video patches -> one text-space token per merged 2x2 block,
    in prompt order, for the reasoner prefill to scatter over its
    ``<|image_pad|>`` / ``<|video_pad|>`` tokens.

    The pixel patches and their grids are computed CPU-side in
    ``Cosmos3Model.process_prompt`` (the token count must be known when the
    prompt is rendered), so this node only runs the encoder.
    """

    # One packed forward per request at request-specific patch counts.
    disable_torch_compile = True

    def __init__(self, vision_model, config):
        super().__init__()
        self.vision_model = vision_model
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        pixel_values = inputs["pixel_values"][0]
        grid_thw = inputs["vision_grid_thw"][0]
        return NodeInputs(
            tensor_inputs={"pixel_values": pixel_values, "vision_grid_thw": grid_thw},
            input_seq_len=int(pixel_values.shape[0]),
        )

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, pixel_values, vision_grid_thw, **kwargs):
        grids = [tuple(int(x) for x in row) for row in vision_grid_thw.tolist()]
        embeds = self.vision_model(pixel_values, grids)
        return {"vision_embeds": [embeds]}


class Cosmos3ReasonerSubmodule(ARNodeSubmodule):
    """The understanding tower served as a causal VLM.

    Shares the DiT node's ``Cosmos3OmniTransformer`` instance (one copy of the
    text weights) and its ``kv`` / ``attn`` resources; declares its own
    ``sampler``. The prefill walks embed the rendered prompt, scatter the
    vision encoder's tokens over the media placeholders, run the text tower
    (writing the raw K/V under one label), and sample the first token from
    the last position; the decode loop feeds each sampled token back as the
    next step's single-token input. Positions are the prompt's 3D mRoPE ids
    (computed in ``process_prompt``) and, past the prompt, the scalar cursor
    ``max(position) + 1`` on all three axes, carried in per-request state.
    """

    # The token loop is data-dependent at the Python level (per-step state,
    # sampling); CUDA-graph capture of the decode step is the accelerator.
    disable_torch_compile = True

    # Decode batch sizes captured as CUDA graphs (bucketed; larger batches
    # run the eager batched forward).
    decode_capture_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    def __init__(self, transformer, config):
        super().__init__()
        self.transformer = transformer
        self.config = config
        reasoner = config.reasoner
        self.eos_token_id = reasoner.eos_token_id if reasoner is not None else None
        self.image_token_id = reasoner.image_token_id if reasoner is not None else -1
        self.video_token_id = reasoner.video_token_id if reasoner is not None else -1
        # The prefill's text tower, compiled (dynamic over the prompt length;
        # fullgraph=False breaks at the attention op like the denoise). Decode
        # keeps the captured graph. CUDA only: the CPU tests run eager.
        env = os.environ.get("COSMOS3_REASONER_PREFILL_COMPILE")
        want = (env == "1") if env is not None else bool(config.compile_reasoner_prefill)
        on_cuda = transformer is not None and any(p.is_cuda for p in transformer.parameters())
        if want and on_cuda:
            self._prefill_text_forward = torch.compile(transformer.text_forward, fullgraph=False, dynamic=True)
            logger.info("Cosmos3 reasoner prefill torch.compile enabled")
        else:
            self._prefill_text_forward = transformer.text_forward if transformer is not None else None

    # ------------------------------------------------------------------
    # prepare_inputs
    # ------------------------------------------------------------------

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> ARNodeInputs:
        if graph_walk in REASONER_PREFILL_WALKS:
            input_ids = inputs["text_inputs"][0].reshape(-1)
            position_ids = inputs["position_ids"][0]
            if position_ids.ndim != 2 or position_ids.shape[0] != 3 or position_ids.shape[1] != input_ids.numel():
                raise ValueError(
                    "Cosmos3 reasoner prefill needs [3, N] mRoPE position ids matching the prompt; got "
                    f"{tuple(position_ids.shape)} for {input_ids.numel()} tokens."
                )
            tensors = {"position_ids": position_ids}
            vision = (inputs or {}).get("vision_embeds")
            if graph_walk == REASONER_PREFILL_VISION_WALK:
                if not vision:
                    raise ValueError("Cosmos3 reasoner vision prefill received no vision embeddings.")
                tensors["vision_embeds"] = vision[0]
            # Decoding continues at max(position) + 1 on every axis.
            self.request_state(fwd_info.request_id).add_all(
                next_pos=int(position_ids.max().item()) + 1,
            )
            return ARNodeInputs(
                input_ids=input_ids,
                input_seq_len=int(input_ids.numel()),
                tensor_inputs=tensors,
            )
        if graph_walk == REASONER_DECODE_WALK:
            st = self.request_states[fwd_info.request_id]
            token = inputs["text_inputs"][0].reshape(-1)[-1:]
            pos = st["next_pos"]
            st.add("next_pos", pos + 1)
            # On the token's device: a captured decode pads the batch with the
            # capture config's device-resident rows, and `preprocess` concatenates
            # every row's position ids into one tensor.
            return ARNodeInputs(
                input_ids=token,
                input_seq_len=1,
                tensor_inputs={"position_ids": torch.full((3, 1), pos, dtype=torch.long, device=token.device)},
            )
        raise ValueError(f"Unknown Cosmos3 reasoner graph walk: {graph_walk!r}")

    # ------------------------------------------------------------------
    # declare_step
    # ------------------------------------------------------------------

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """One causal span per request under the reasoner's label, committed
        (the text is context for every later token); the sampler tracks the
        prompt tokens at prefill for the repetition penalty."""
        from mstar.engine.resources import SamplerStep

        prefill = graph_walk in REASONER_PREFILL_WALKS
        if not prefill and graph_walk != REASONER_DECODE_WALK:
            raise ValueError(f"Unknown Cosmos3 reasoner graph walk: {graph_walk!r}")
        segments = [
            Segment(rid, REASONER_LABEL, inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        sampler = SamplerStep(
            prefill_tracked_tokens={
                rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
                if inp.input_ids is not None
            } if prefill else {},
        )
        return SubmoduleStep(
            segments=segments,
            steps={
                KV_CACHE: KVStep(commit=True),
                ATTN: AttentionStep(causal=True),
                SAMPLER: sampler,
            },
        )

    # ------------------------------------------------------------------
    # preprocess / forward
    # ------------------------------------------------------------------

    def preprocess(self, graph_walk, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]) -> dict:
        # One device for the whole batch: prompt rows arrive on the model's
        # device, decode rows follow their token, and a captured step's padding
        # rows are the capture config's. Sequential requests never mix them;
        # a padded batch does.
        device = inputs[0].input_ids.device
        out = {
            "input_ids": torch.cat([inp.input_ids.to(device) for inp in inputs]),
            "position_ids": torch.cat([inp.tensor_inputs["position_ids"].to(device) for inp in inputs], dim=1),
            "seq_lens": [int(inp.input_seq_len) for inp in inputs],
        }
        vision = [inp.tensor_inputs["vision_embeds"] for inp in inputs if "vision_embeds" in inp.tensor_inputs]
        if vision:
            out["vision_embeds"] = torch.cat(vision, dim=0)
        return out

    def _embed(self, input_ids: torch.Tensor, vision_embeds: torch.Tensor | None) -> torch.Tensor:
        embeds = self.transformer.embed_tokens(input_ids)
        if vision_embeds is not None:
            mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
            if int(mask.sum().item()) != vision_embeds.shape[0]:
                raise ValueError(
                    f"Cosmos3 reasoner: {int(mask.sum().item())} media placeholder tokens but "
                    f"{vision_embeds.shape[0]} vision tokens."
                )
            embeds = embeds.masked_scatter(mask.unsqueeze(-1), vision_embeds.to(embeds.dtype))
        return embeds

    def _sample(self, request_ids: list[str], logits: torch.Tensor, engine_inputs: ModelInputsFromEngine):
        resources = engine_inputs.resources or self.node_resources
        tokens = resources[SAMPLER].sample(request_ids, logits)
        # The sampler reuses its output buffer across calls; keep our own copy.
        return tokens.clone()

    def _run(self, graph_walk, engine_inputs, input_ids, position_ids, seq_lens, vision_embeds=None):
        # Native bf16 (see the DiT node): the reference text tower runs pure bf16.
        with torch.autocast(device_type="cuda", enabled=False):
            embeds = self._embed(input_ids, vision_embeds)
            prefill = graph_walk in REASONER_PREFILL_WALKS
            text_forward = self._prefill_text_forward if prefill else self.transformer.text_forward
            hidden = text_forward(embeds, position_ids.to(embeds.device), REASONER_LABEL)
            if prefill:
                # The last position of each request's span predicts its first token.
                ends = torch.tensor(seq_lens, device=hidden.device).cumsum(0) - 1
                hidden = hidden[ends]
            logits = self.transformer.lm_head(hidden)
        return logits.float()

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, input_ids, position_ids, seq_lens,
                vision_embeds=None, **kwargs):
        logits = self._run(graph_walk, engine_inputs, input_ids, position_ids, seq_lens, vision_embeds)
        tokens = self._sample(engine_inputs.request_ids, logits, engine_inputs)
        return {"new_token": [tokens[:1]]}

    def forward_batched(self, graph_walk, engine_inputs: ModelInputsFromEngine, input_ids, position_ids, seq_lens,
                        vision_embeds=None, **kwargs):
        logits = self._run(graph_walk, engine_inputs, input_ids, position_ids, seq_lens, vision_embeds)
        tokens = self._sample(engine_inputs.request_ids, logits, engine_inputs)
        return {
            rid: {"new_token": [token]}
            for rid, token in zip(engine_inputs.request_ids, tokens.split(1), strict=True)
        }

    def can_batch(self, batch, model_inputs) -> bool:
        # Continuous batching for the token loop and for text-only prefills;
        # vision prefills carry per-request packed embeddings and run alone.
        return batch.graph_walk in (REASONER_DECODE_WALK, REASONER_PREFILL_WALK)

    def get_cuda_graph_configs(self, device, tp_world_size: int = 1):
        """Capture the decode step (one token per request) per batch-size
        bucket; prefills run eager (they are one-shot and shape-varied)."""
        if self.transformer is None or os.environ.get("COSMOS3_DISABLE_CUDA_GRAPH"):
            return []
        bs_env = os.environ.get("COSMOS3_REASONER_CAPTURE_BS")
        sizes = [int(x) for x in bs_env.split(",")] if bs_env else list(self.decode_capture_batch_sizes)
        # Compile the captured decode step (inductor, then the graph): at bs=1
        # the eager step is ~1240 kernels of which ~1000 are the norms', the
        # rotary's and the residuals' pointwise pieces — 2.1 of its 3.8 ms on
        # an H100 — and the fusion is what closes the gap to the weight-
        # streaming floor (measured 4.2 -> 2.05 ms/token, 382 kernels).
        env = os.environ.get("COSMOS3_REASONER_COMPILE")
        compile_decode = (env == "1") if env is not None else bool(self.config.compile_reasoner_decode)
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=REASONER_DECODE_WALK,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                    tensor_inputs={"position_ids": torch.zeros((3, 1), dtype=torch.long, device=device)},
                ),
                capture_batch_sizes=sizes,
                caps_eager_batch_size=False,
                compile=compile_decode,
            ),
        ]

    # ------------------------------------------------------------------
    # postprocess / check_stop
    # ------------------------------------------------------------------

    def postprocess(self, request_id, request_info, outputs, inputs=None, **kwargs):
        # The sampled token is both the emitted text chunk and the next
        # decode step's input.
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(self, request_id, request_info, outputs) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = int(outputs["new_token"][0].reshape(-1)[0].item())
        sampling = request_info.resource_configs.get(SAMPLER)
        ignore_eos = bool(getattr(sampling, "ignore_eos", False))
        generated = request_info.dynamic_loop_iter_counts.get(REASONER_DECODE_LOOP, 0) + 1
        if (not ignore_eos and self.eos_token_id is not None and token == self.eos_token_id) or (
            generated + 1 >= request_info.max_tokens
        ):
            return {REASONER_DECODE_LOOP}
        return set()
