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
  Cosmos3VAEDecoderARSubmodule -- streaming Wan VAE decode (per-request
                                 decoder cache; no engine resource) for
                                 windowed AR video: one committed window's
                                 latents per stream chunk, decoded from the
                                 stream's warm decoder cache (behind
                                 re-decoded context where it has none) and
                                 emitted per window or assembled.
  Cosmos3VisionEncoderSubmodule -- Edge reasoner vision tower (STATELESS):
                                 packed image/video patches to text-space
                                 tokens for the reasoner prefill.
  Cosmos3ReasonerSubmodule    -- the understanding tower as a causal VLM, a
                                 ``SampledTokenDecoder`` on the VLM scaffold
                                 (shares the DiT's transformer instance and
                                 kv/attn resources, plus its own sampler and
                                 positions): ``prefill_text`` /
                                 ``prefill_vision`` write the prompt's K/V and
                                 sample the first token, ``decode`` one token
                                 per iteration of ``decode_loop``.

Because the text tokens never receive a timestep embedding, the understanding
K/V is denoise-step independent, so writing it once and re-reading it every step
matches running the whole transformer each step.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import torch
from torch.fx.experimental import _config as fx_config

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    check_inductor_options,
)
from mstar.engine.fp32_precision import cudnn_benchmark
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    at_last_token,
    host,
    kernel_setting,
    last_sampled,
    position,
    positions,
    sample,
)
from mstar.engine.windowing import WindowSchedule
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.components.diffusion.decode_utils import latent_affine, wan_vae_decode_dtype
from mstar.model.components.diffusion.unipc import (
    LAST_SAMPLE,
    MODEL_OUTPUTS,
    SOLVER_STATE,
    UniPCState,
    initial_state,
    unipc_step,
)
from mstar.model.components.mrope import layout_position_ids, mrope_position_ids
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.cosmos3.components.packing import (
    action_start_frame_offset,
    build_action_static_inputs,
    build_static_inputs,
    build_text_segment,
    vision_condition_frame_indexes,
)
from mstar.model.cosmos3.constants import (
    ACTION_GEN_WALK,
    ACTION_VIDEO_GEN_WALK,
    IMAGE_GEN_WALK,
    KV_CACHE,
    PREFILL_COND_VIDEO_WALK,
    PREFILL_COND_WALK,
    PREFILL_WALK,
    PROMPT_LENGTHS,
    VIDEO_GEN_AR_WALK,
    VIDEO_GEN_WALK,
    VIDEO_SOUND_GEN_WALK,
    VISION_TOKENS,
)
from mstar.model.cosmos3.sessions import SessionStore
from mstar.model.keys import MIXED, VISION_ATTN
from mstar.model.multimodal import MediaEncoder
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    StepOrder,
)
from mstar.model.utils import warn_retired_ladder_env
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.knobs import ingraph_decode_rope

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
# upper-bound iteration count and each request caps its loop at its own
# denoise-step count (``Cosmos3Model._denoise_cap``), so one graph serves any
# per-request step count.
IMAGE_GEN_LOOP = "image_gen_loop"
VIDEO_GEN_LOOP = "video_gen_loop"
VIDEO_GEN_AR_LOOP = "video_gen_ar_loop"
VIDEO_SOUND_GEN_LOOP = "video_sound_gen_loop"
ACTION_GEN_LOOP = "action_gen_loop"
ACTION_VIDEO_GEN_LOOP = "action_video_gen_loop"
# Each denoise walk's loop; any other walk runs the image loop.
DENOISE_LOOP_FOR_WALK = {
    ACTION_GEN_WALK: ACTION_GEN_LOOP,
    ACTION_VIDEO_GEN_WALK: ACTION_VIDEO_GEN_LOOP,
    VIDEO_GEN_WALK: VIDEO_GEN_LOOP,
    VIDEO_GEN_AR_WALK: VIDEO_GEN_AR_LOOP,
    VIDEO_SOUND_GEN_WALK: VIDEO_SOUND_GEN_LOOP,
}

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

# A captured denoise step's static-buffer inputs, each stacked on a leading
# batch dim: the latents, the timestep, each branch's rotary table and the
# clean/noisy token mask.
CAPTURED_GEN_INPUTS = (
    "latents", "vision_timesteps", "cos_cond", "sin_cond", "cos_uncond", "sin_uncond", "noisy_token_mask",
)

# The ``kv`` array's reads (Cosmos3Model.declare_state): ATTN, causal, for the
# prefills and the reasoner, and ATTN_GEN, bidirectional, for the generation
# steps. The reasoner shares the array and adds SAMPLER for its tokens.
# PROGRESS holds where a request's draws stand between steps.
ATTN = "attn"
ATTN_GEN = "attn_gen"
SAMPLER = "sampler"
PROGRESS = "progress"
# A windowed request's loop-back edges beside its latents: the frames its
# window pins, and the decode context its session keeps.
WINDOW_COND, SESSION_TAIL = "window_cond", "session_tail"

# Denoise state (initial noise, latents between steps, the guidance combine and
# the scheduler step) is fp32 for image, video, sound and action requests, as
# diffusers' pipeline keeps it; only the transformer input is bf16. Windowed
# rollouts keep their latents in the model dtype.
LATENT_DTYPE = torch.float32

# The reasoner's decode loop (the VLM scaffold's name).
REASONER_DECODE_LOOP = "decode_loop"
# The reasoner keeps every request's text under one cache label.
REASONER_LABEL = "main"


def denoise_compile_kwargs(mode: str) -> dict:
    """``torch.compile`` arguments for the denoise step; see
    ``Cosmos3Config.denoise_compile_mode``. "deterministic" and "default" are
    the engine's compile modes of those names, "level" its -O level."""
    from mstar.engine.accelerator_graph_runner import compile_kwargs

    return compile_kwargs(None if mode == "level" else mode)


def latent_frames(num_frames: int, config) -> int:
    return 1 if num_frames == 1 else 1 + (num_frames - 1) // config.vae.scale_factor_temporal


def frame_tokens(height: int, width: int, config) -> int:
    """Generation tokens per latent frame of a ``height`` x ``width`` request. `_build_static` asserts the count."""
    s, p = config.vae.scale_factor_spatial, config.latent_patch_size
    return math.ceil(height // s / p) * math.ceil(width // s / p)


def sound_frames(md: dict, config) -> tuple[int, int]:
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
    target_samples = max(1, int(round(duration * config.sound_sample_rate)))
    hop = int(round(config.sound_sample_rate / config.sound_latent_fps))
    return target_samples, max(1, math.ceil(target_samples / hop))


def _loop_step(fwd_info, graph_walk: str) -> int:
    """The denoise iteration a step runs, from the engine's loop counter: global
    over a windowed request's windows and commits. A value carried on a loop-back
    edge would be the previous step's device output, and reading it on the host
    waits for that step to finish, which stalls the speculative preparation of this
    one."""
    return int(fwd_info.dynamic_loop_iter_counts[DENOISE_LOOP_FOR_WALK.get(graph_walk, IMAGE_GEN_LOOP)])


def _has_guidance(md: dict) -> bool:
    """Whether a generation request runs classifier-free guidance, which gives it an unconditional branch."""
    return float(md.get("guidance_scale", 6.0)) != 1.0


def _schedule(md: dict) -> WindowSchedule:
    """A windowed request's schedule, from its step metadata."""
    return WindowSchedule(
        total_units=int(md["total_latent_units"]),
        window_units=int(md["window_latent_units"]),
        context_units=int(md.get("context_latent_units", 0)) if md.get("window_mode") == "kv" else 0,
        overlap_units=int(md["overlap_latent_units"]),
    )


def _first_iteration(inputs, step_index: int, request_id: str) -> bool:
    """Whether this step starts the loop: no loop-back latents, which past iteration
    0 is a request being replayed over a finished loop. Raised from
    ``prepare_inputs``, that fails this request alone."""
    if "latents" in inputs and len(inputs["latents"]) > 0:
        return False
    if step_index != 0:
        raise RuntimeError(
            f"Cosmos3 DiT: request {request_id} was dispatched denoise iteration "
            f"{step_index} with no loop-back latents; refusing to restart its loop"
        )
    return True


class _UniPCSampler:
    """A request's UniPC: the scaffold's step over this request's tables. The tables
    go up once, and the solver state rides the loop-back edges, so a step reads
    nothing on the host and holds nothing between steps."""

    def __init__(self, scheduler, device):
        self.config = scheduler.config
        self.host_sigmas = scheduler.sigmas
        self.host_timesteps = scheduler.timesteps.tolist()
        self.sigmas = to_device_nonblocking(scheduler.sigmas, device)
        self.timesteps = to_device_nonblocking(scheduler.timesteps, device)

    def initial_state(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        return initial_state(sample)

    def step(self, model_output, sample, step_index: int, solver, noise=None):
        step = unipc_step(
            model_output, sample,
            UniPCState(model_outputs=solver[MODEL_OUTPUTS], last_sample=solver[LAST_SAMPLE]),
            self.host_sigmas, self.sigmas, step_index, len(self.host_timesteps),
        )
        return step.sample, {MODEL_OUTPUTS: step.model_outputs, LAST_SAMPLE: step.last_sample}


class _DistilledSampler:
    """The 4-step distilled sampler: diffusers' FlowMatchEuler update with the checkpoint's
    stochastic (SDE) step, over the scheduler's sigmas at the loop's step index. Its noise
    comes in as a tensor, so it has no state to carry on an edge or between steps."""

    def __init__(self, scheduler):
        self.config = scheduler.config
        self.host_timesteps = scheduler.timesteps.tolist()
        self.timesteps = scheduler.timesteps
        self.sigmas = scheduler.sigmas

    def initial_state(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}

    def noise(self, sample: torch.Tensor, generator) -> torch.Tensor | None:
        """The SDE step's noise for ``sample``, drawn as `FlowMatchEulerDiscreteScheduler.step` draws it."""
        if not self.config.stochastic_sampling:
            return None
        from diffusers.utils.torch_utils import randn_tensor

        return randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=torch.float32)

    def step(self, model_output, sample, step_index: int, solver, noise=None):
        # `FlowMatchEulerDiscreteScheduler.step`'s update, in its order and dtypes
        sample = sample.to(torch.float32)
        sigma, sigma_next = self.sigmas[step_index], self.sigmas[step_index + 1]
        if self.config.stochastic_sampling:
            prev = (1.0 - sigma_next) * (sample - sigma * model_output) + sigma_next * noise
        else:
            prev = sample + (sigma_next - sigma) * model_output
        return prev.to(model_output.dtype), {}


def _host_timesteps(scheduler) -> list:
    """The scheduler's timesteps as Python numbers (the values ``.item()`` returns),
    from the host copy of the samplers above: a per-step read of the staged table
    would be a host sync."""
    host = getattr(scheduler, "host_timesteps", None)
    return host if host is not None else scheduler.timesteps.tolist()


def native_flow_sigmas(num_inference_steps: int, num_train_timesteps: int):
    """The native flow-matching sigma grid: ``num_inference_steps`` values
    linearly spaced from ``1 - 1/T`` toward 0, the endpoint dropped. A numpy
    array: UniPC's explicit-sigma path applies the flow shift arithmetically
    to it (a Python list raises inside ``set_timesteps``)."""
    import numpy as np

    return np.linspace(1.0 - 1.0 / num_train_timesteps, 0.0, num_inference_steps + 1)[:-1]


def _guided(cond_v: torch.Tensor, uncond_v: torch.Tensor, gs: float, dtype: torch.dtype) -> torch.Tensor:
    """Classifier-free guidance combine in the loop-state dtype."""
    cond_v = cond_v.to(dtype)
    uncond_v = uncond_v.to(dtype)
    return uncond_v + gs * (cond_v - uncond_v)


def _nbytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v) for v in value)
    # a sampler's tables
    if isinstance(value, (_UniPCSampler, _DistilledSampler)):
        return _nbytes(vars(value))
    return 0


class _Memo:
    """Device tensors kept by the key that builds them, the least recently used dropped first past ``max_bytes``.
    A value over ``largest`` bytes, the whole budget unless given, is built per call and not kept."""

    def __init__(self, max_bytes: int, largest: int | None = None):
        self.max_bytes = max_bytes
        self.largest = max_bytes if largest is None else largest
        self.entries: OrderedDict[Hashable, Any] = OrderedDict()
        self.num_bytes = 0
        self._sizes: dict[Hashable, int] = {}

    def get(self, key: Hashable, build: Callable[[], Any]) -> Any:
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
            return value
        value = build()
        size = _nbytes(value)
        if size > self.largest:
            return value
        self.entries[key] = value
        self._sizes[key] = size
        self.num_bytes += size
        while self.num_bytes > self.max_bytes:
            dropped, _ = self.entries.popitem(last=False)
            self.num_bytes -= self._sizes.pop(dropped)
        return value


@dataclass(frozen=True)
class GenStepInfo:
    """What a denoise step's declaration needs and cannot read off the batch.

    These are per-request facts resolved in ``prepare_inputs`` (which has the
    request's settings and its step index) and carried to ``declare_step`` and
    ``can_batch`` on ``NodeInputs.resource_step_info``, since those run with
    only the request ids and their prepared inputs.

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
    # A kv-mode windowed commit iteration: the span is added to every live
    # branch and kept instead of recomputed and dropped — see
    # ``Cosmos3DiTSubmodule._commit_step``.
    commit: bool = False
    # the request's generation tokens, which ``can_batch`` caps a batch's sum of
    vision_tokens: int = 0


def _warmup_kv(served: Any) -> Any:
    """The ``kv`` handle ``Cosmos3DiTSubmodule.warmup_compiles`` binds, since no request has planned a step yet: a copy
    of the served one whose add keeps nothing and whose read returns zeros of the query's size. A copy, so the layer
    stack compiled over it guards on the type of the handle it then serves with."""
    kv = copy.copy(served)
    kv._add_read = lambda layer, q, *, keep=None, **keywords: torch.zeros_like(q)
    return kv


class Cosmos3DiTSubmodule(ARNodeSubmodule):
    """Dual-pathway DiT node (understanding tower + generation denoiser)."""

    # The denoise loop is data-dependent (scheduler step, classifier-free guidance
    # combine), so torch.compile graph-breaks and buys little; CUDA-graph capture of
    # the fixed-shape step is the accelerator.
    disable_torch_compile = True

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
    # rounded up at replay; prompts past the largest bucket run eager. The
    # serving config's `prefill_token_buckets` sets the token buckets, and
    # COSMOS3_PREFILL_CAPTURE_BS the batch sizes.
    prefill_capture_token_buckets: tuple[int, ...] = (16, 32, 64, 128, 256)
    prefill_capture_batch_sizes: tuple[int, ...] = (1,)

    # Statics, masks and samplers by what builds them, shared by requests of one shape. A label's statics are ~6.4 MB
    # at 480x832 and 121 frames. Every step looks them up, so any value the budget holds is kept.
    STATICS_BYTES = 256 << 20

    def __init__(self, transformer, config, scheduler=None, head_decode_vae=None, head_encode_vae=None):
        super().__init__()
        warn_retired_ladder_env("COSMOS3_PREFILL_CAPTURE_TOKENS", "prefill_token_buckets: {dit: {prefill: [...]}}")
        self.transformer = transformer
        self.config = config
        # The loop-back edges of this deployment's scheduler, beside the latents.
        self._solver_state = () if config.distilled_sigmas else SOLVER_STATE
        # Re-encodes the frames a chained window (or a resumed session)
        # continues from; built only for deployments serving windowed video.
        self._window_head = (
            ChainedWindowHead(head_decode_vae, head_encode_vae, config) if head_decode_vae is not None else None
        )
        # Template scheduler; a fresh instance (with its own multistep state) is
        # built per request from this one's config.
        self._scheduler_template = scheduler
        self._statics = _Memo(self.STATICS_BYTES)
        self.progress = None
        # Windowed sessions: the last window's clean latents and the stream's
        # decode context per session id. A follow-up request with
        # ``resume_session`` re-pins its head from the latents, and the decoder
        # decodes its first window behind the context (`_window_start`,
        # `_end_window`). A session is pinned while a request generates under
        # it and idles out after its timeout.
        self._sessions = SessionStore(
            config.session_store_size, config.session_timeout_s, config.session_timeout_max_s,
        )
        # Compile the pure denoise compute (~1.2-1.3x/step; the kernels bake
        # into the CUDA graphs at capture). fullgraph=False breaks at the
        # attention; ``config.compile_denoise=False`` keeps the eager step for
        # the bit-exact parity tests.
        if config.compile_denoise and transformer is not None:
            kwargs = denoise_compile_kwargs(config.denoise_compile_mode)
            self.transformer.denoise_step = torch.compile(self.transformer.denoise_step, **kwargs)
            self.transformer.denoise_step_batched_cfg = torch.compile(
                self.transformer.denoise_step_batched_cfg, **kwargs,
            )
            if config.compile_denoise_batched:
                # Only the cross-request step's layer stack, with the pack
                # length dynamic: one compile, built by `warmup_compiles`
                # before the first request, serves every batch size and mix of
                # resolutions. The per-request patchify, rotary and unpatchify
                # around it run eager.
                self.transformer.gen_layers_batched = torch.compile(
                    self.transformer.gen_layers_batched, **{**kwargs, "dynamic": True},
                )
            logger.info("Cosmos3 denoise compute torch.compile enabled (%s)", config.denoise_compile_mode)

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.progress = handles[PROGRESS]

    def _bind_transformer(self, handles: Mapping[str, Any]) -> None:
        for module in self.transformer.modules():
            bind = getattr(module, "bind_state", None)
            if bind is not None:
                bind(handles)

    def warmup_compiles(self) -> None:
        """Build the compiled cross-request layer stack before the first
        request: one two-request step at the default size, through
        ``denoise_step_batched`` with a stand-in for the ``kv`` handle, since no
        request has planned a step yet. The handle's calls are graph breaks, so
        what the stand-in returns never enters a compiled graph, and the graph
        does not depend on the size it was built at."""
        if not (self.config.compile_denoise and self.config.compile_denoise_batched) or self.transformer is None:
            return
        cfg = self.config
        frames = cfg.num_frames_video if cfg.video_size_default is not None else 1
        width, height = (int(x) for x in (cfg.video_size_default or cfg.image_size_default))
        device = self.get_device()
        static = self._build_static(
            [0] * 8, height, width, num_frames=frames, fps=24.0, has_image_condition=False, device=device,
        )
        row = self.transformer.denoise_step_row(
            static,
            torch.zeros(
                self._latent_shape(height, width, frames), device=device,
                dtype=self.transformer.proj_in.weight.dtype,
            ),
            torch.zeros(static["num_noisy_vision_tokens"], device=device),
            static["rotary"],
            static["rotary"],
        )
        t0 = time.perf_counter()
        # as `forward_batched` runs it: native dtype, no autocast. Without duck sizing, which would
        # tie a pack as long as the hidden size (Nano's default, 4096) to that static size and
        # compile again at the first other length.
        self._bind_transformer({**self.state_handles, KV_CACHE: _warmup_kv(self.state_handles[KV_CACHE])})
        with torch.autocast(device_type=device.type, enabled=False), fx_config.patch(use_duck_shape=False):
            self.transformer.denoise_step_batched([row, row])
        self._bind_transformer(self.state_handles)
        logger.info(
            "Cosmos3 cross-request layer stack compiled in %.1f s (%dx%dx%d, two requests)",
            time.perf_counter() - t0, width, height, frames,
        )

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

    def cg_key_info(
        self, graph_walk: str, per_request_info: dict, per_request_input_metadata=None, **kwargs,
    ) -> object | None:
        """Which of this walk's capture buckets the batch belongs to, or None
        for "run eager".

        The engine leases the replay slot before the step is declared, so this
        answers from each request's step metadata rather than from prepared
        inputs. It is
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
        a captured step always runs both branches, and ``postprocess`` keeps
        the conditional one outside the interval, as the eager path does.
        Honoring the interval here instead would make the key step-dependent,
        which is exactly what the pre-plan path cannot support.
        """
        del per_request_input_metadata, kwargs
        mds = [info.step_metadata for info in per_request_info.values()]
        if not mds or not all(_has_guidance(md) for md in mds):
            return None
        if graph_walk in PREFILL_WALKS:
            return True
        if graph_walk not in GEN_WALKS:
            return None
        keys = {self._capture_key(graph_walk, md) for md in mds}
        if len(keys) != 1:
            return None
        return keys.pop()

    def _step_info(self, graph_walk: str, md: dict, st, step_index: int) -> GenStepInfo:
        """The per-request facts ``declare_step`` and ``can_batch`` need, resolved here where the request's settings
        and its step index are both in hand."""
        return GenStepInfo(
            cfg=st["uncond"] is not None,
            cfg_active=self._cfg_active(st, step_index),
            capture_key=self._capture_key(graph_walk, md),
            vision_tokens=st["cond"]["num_vision_tokens"],
        )

    def _capture_key(self, graph_walk: str, md: dict) -> object | None:
        """This one request's half of ``cg_key_info``: everything that does
        not depend on the rest of the batch. See ``GenStepInfo.capture_key``."""
        if graph_walk not in GEN_WALKS or not _has_guidance(md):
            return None
        if graph_walk == VIDEO_GEN_AR_WALK and md.get("window_mode") == "kv":
            # kv windows interleave commit iterations — a different step
            # declaration the lease could not know about — so they run eager.
            return None
        # The latent shape is fixed for the request's lifetime (a windowed
        # request's windows all share the padded window shape), and the
        # clean/noisy layout rides along as a mask input, so one graph per
        # shape serves t2i, t2v, i2v and chained windows alike.
        height, width = int(md.get("height", 256)), int(md.get("width", 256))
        if md.get("window_mode") is not None:
            shape = self._window_latent_shape(height, width, _schedule(md).window(0).units)
        else:
            shape = self._latent_shape(height, width, int(md.get("num_frames", 1)))
        layout = getattr(self, "_capture_layout", None) or {}
        return shape if shape in layout else None

    # ------------------------------------------------------------------
    # Static packing + scheduler helpers
    # ------------------------------------------------------------------

    def _latent_shape(
        self, height: int, width: int, num_frames: int = 1
    ) -> tuple[int, int, int, int, int]:
        s = self.config.vae.scale_factor_spatial
        return (1, self.config.latent_channel, latent_frames(num_frames, self.config), height // s, width // s)

    def _build_static(
        self, ids: list[int], height: int, width: int, num_frames: int,
        fps: float, has_image_condition: bool, device,
        sound_latent_frames: int | None = None,
        noisy_frames: list[int] | None = None,
        start_frame_offset: int = 0,
    ) -> dict:
        latent_shape = self._latent_shape(height, width, num_frames)
        static = build_static_inputs(
            list(ids), latent_shape, self.config,
            self.config.vae.scale_factor_temporal, fps, device,
            has_image_condition=has_image_condition,
            sound_latent_frames=sound_latent_frames,
            noisy_frames=noisy_frames,
            start_frame_offset=start_frame_offset,
        )
        assert static["num_vision_tokens"] == latent_shape[2] * frame_tokens(height, width, self.config), (
            f"packing made {static['num_vision_tokens']} vision tokens of {latent_shape}, "
            "not the count `frame_tokens` gives a layout to size it by"
        )
        # proj_out runs on the generation token block, so shift the joint-sequence
        # mse indexes to be relative to the generation tokens.
        static["mse_gen_indexes"] = static["vision_mse_loss_indexes"] - static["und_len"]
        if sound_latent_frames is not None:
            static["sound_mse_gen_indexes"] = static["sound_mse_loss_indexes"] - static["und_len"]
        self._attach_rotary(static, device)
        return static

    def _attach_rotary(self, static: dict, device) -> None:
        """The generation band's mRoPE table, ``rotary = (cos, sin)``
        ([num_gen, head_dim], the tower's dtype). It depends only on the
        positions, so it is built once here per request or window and handed
        to every denoise step. None without a transformer (bookkeeping tests)."""
        tf = self.transformer
        if tf is not None:
            static["rotary"] = tf._rotary(static["gen_position_ids"], device, tf.proj_in.weight.dtype)

    def _static(
        self, ids: list[int], height: int, width: int, num_frames: int, fps: float, has_image_condition: bool,
        device, sound_latent_frames: int | None = None, noisy_frames: list[int] | None = None,
        start_frame_offset: int = 0,
    ) -> dict:
        """`_build_static`'s statics, slimmed and with their layout masks, from the memo. Past the prompt's length
        they don't depend on its tokens."""
        key = (
            "static", len(ids), height, width, num_frames, fps, has_image_condition, sound_latent_frames,
            None if noisy_frames is None else tuple(noisy_frames), start_frame_offset,
        )

        def build() -> dict:
            static = self._build_static(
                ids, height, width, num_frames, fps, has_image_condition, device,
                sound_latent_frames=sound_latent_frames, noisy_frames=noisy_frames,
                start_frame_offset=start_frame_offset,
            )
            self._slim_statics(static, None)
            self._add_layout_masks(static, device)
            return static

        return self._statics.get(key, build)

    def _action_static(
        self, ids: list[int], height: int, width: int, num_frames: int, action_chunk: int, mode: str, fps: float,
        action_fps: float, action_offset: int, device, state_rows: int,
    ) -> dict:
        """`_build_action_static`'s statics, slimmed, from the memo."""
        key = (
            "action", len(ids), height, width, num_frames, action_chunk, mode, fps, action_fps, action_offset,
            state_rows,
        )

        def build() -> dict:
            static = self._build_action_static(
                ids, height, width, num_frames, action_chunk, mode, fps, action_fps, action_offset, device,
                state_rows,
            )
            self._slim_statics(static, None)
            return static

        return self._statics.get(key, build)

    @staticmethod
    def _add_layout_masks(static: dict, device) -> None:
        """The static's clean/noisy layout as data for a captured graph: a per-token mask (``noisy_token_mask``,
        ``[num_vision_tokens]``, 1 on noisy frames' tokens) and the same per frame (``noisy_frame_mask``,
        ``[1, T, 1, 1]``)."""
        ((t_frames, patch_h, patch_w),) = static["vision_token_shapes"]
        frame_mask = torch.zeros(t_frames, device=device, dtype=torch.float32)
        frame_mask[static["vision_noisy_frame_indexes"][0].to(device)] = 1.0
        static["noisy_token_mask"] = frame_mask.repeat_interleave(patch_h * patch_w).contiguous()
        static["noisy_frame_mask"] = frame_mask.view(1, t_frames, 1, 1)

    def _pin_mask(self, frames: int, pinned: Sequence[int], device, dtype) -> torch.Tensor:
        """``[1, 1, frames, 1, 1]``, 1 on the ``pinned`` latent frames, which every step re-pins to their clean
        latents."""
        pinned = tuple(pinned)

        def build() -> torch.Tensor:
            mask = torch.zeros((1, 1, frames, 1, 1), device=device, dtype=dtype)
            mask[:, :, list(pinned)] = 1.0
            return mask

        return self._statics.get(("pin", frames, pinned, dtype), build)

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
        # on the host: the sampler stages its tables without syncing the stream
        if native_flow:
            # Linspaced flow sigmas from 1 - 1/T down to (not including) 0,
            # the ``use_native_flow_schedule`` pipeline path; the scheduler
            # applies its flow shift on top.
            num_train = int(scheduler.config.num_train_timesteps)
            sigmas = native_flow_sigmas(num_inference_steps, num_train)
            scheduler.set_timesteps(num_inference_steps, device="cpu", sigmas=sigmas)
        else:
            scheduler.set_timesteps(num_inference_steps, device="cpu")
        return _UniPCSampler(scheduler, device)

    def _new_distilled_scheduler(self, device):
        """A FlowMatchEuler scheduler with the checkpoint's stochastic (SDE) step
        over the fixed sigma list: every step re-noises with
        ``x' = (1 - sigma') (x - sigma v) + sigma' eps`` from the request's generator
        (see ``_scheduler_step``). Flow shift and karras spacing do not apply: the
        sigmas are explicit."""
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
        return _DistilledSampler(scheduler)

    def _sampler(self, steps: int, flow_shift, use_karras_sigma, device):
        """A request's sampler, from the memo: it holds the schedule's tables and nothing of the request's."""
        return self._statics.get(
            ("sampler", steps, flow_shift, use_karras_sigma),
            lambda: self._new_scheduler(steps, device, flow_shift=flow_shift, use_karras_sigma=use_karras_sigma),
        )

    @staticmethod
    def _scheduler_step(st, velocity, latents, step_index: int, solver):
        """One scheduler update of a ``[C, T, H, W]`` latent, and the solver state
        its loop-back edges carry. A distilled request's SDE noise was drawn in
        ``prepare_inputs`` from its generator (the one its initial noise came
        from, as in the reference), so the stochastic step is seedable; UniPC
        takes no noise."""
        new_latents, solver = st["scheduler"].step(
            velocity.unsqueeze(0), latents.unsqueeze(0), step_index, solver, noise=st.get("sde_noise"),
        )
        return new_latents.squeeze(0), solver

    def _build_action_static(
        self, ids: list[int], height: int, width: int, num_frames: int, action_chunk: int,
        mode: str, fps: float, action_fps: float, action_offset: int, device, state_rows: int = 0,
    ) -> dict:
        latent_shape = self._latent_shape(height, width, num_frames)
        static = build_action_static_inputs(
            list(ids), latent_shape, action_chunk, mode,
            self.config, self.config.vae.scale_factor_temporal, fps, action_fps, action_offset, device,
            state_rows=state_rows,
        )
        assert static["num_vision_tokens"] == latent_shape[2] * frame_tokens(height, width, self.config), (
            f"packing made {static['num_vision_tokens']} vision tokens of {latent_shape}, "
            "not the count `frame_tokens` gives a layout to size it by"
        )
        # proj_out runs on the generation token block; shift the joint-sequence
        # mse indexes to be relative to the [vision | action] generation tokens.
        static["mse_gen_indexes"] = static["vision_mse_loss_indexes"] - static["und_len"]
        static["action_mse_gen_indexes"] = static["action_mse_loss_indexes"] - static["und_len"]
        self._attach_rotary(static, device)
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
            return self._prepare_action_gen(graph_walk, fwd_info, inputs, device)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _prepare_prefill(self, fwd_info, inputs, device) -> ARNodeInputs:
        md = fwd_info.step_metadata
        cond_ids = inputs["text_inputs"][0].tolist()
        uncond_ids = inputs["text_inputs"][1].tolist() if _has_guidance(md) else None
        # Opt-in sound appends a jointly denoised AVAE-latent band to the
        # generation block. Video-only (single-frame image and action requests
        # are rejected at request resolution).
        if md.get("generate_sound") and int(md.get("num_frames", 1)) <= 1:
            raise ValueError("Cosmos3 sound generation requires a video request (num_frames > 1)")
        windowed = md.get("window_mode") is not None
        if windowed and md.get("session_id"):
            self._begin_session(fwd_info, md)
        if not windowed:
            # built here, so that the request's denoise steps only look them up
            self._memoized(md, device)
        return self._get_prefill_node_inputs(cond_ids, uncond_ids, device, windowed=windowed)

    def _settings(self, md: dict, step_index: int, inputs, device) -> dict:
        """A generation request's settings for one step: what its step metadata says, its statics and sampler from
        the memo, and what its inputs carry, such as the latents it pins. A windowed request's are its window's."""
        st = {"gs": float(md.get("guidance_scale", 6.0)), "guidance_interval": md.get("guidance_interval")}
        if md.get("window_mode") is not None:
            height, width, fps, steps, ids = self._shape(md)
            st["scheduler"] = self._sampler(steps, md.get("flow_shift"), md.get("use_karras_sigma"), device)
            st.update(self._window_settings(md, *ids, height, width, fps, steps, step_index, inputs, device))
            return st
        st.update(self._memoized(md, device))
        self._ingest_cond_latents(st, inputs, device)
        if self.config.distilled_sigmas and st.get("cond_latents") is not None:
            # The distilled SDE step re-noises every position from the request's
            # generator, so the i2v anchor is re-pinned after each step, by the
            # mask re-injection video-to-video uses.
            st["vmask"] = self._pin_mask(st["latent_shape"][2], (0,), device, LATENT_DTYPE)
            cond_video = torch.zeros(st["latent_shape"], device=device, dtype=LATENT_DTYPE)
            cond_video[:, :, 0] = st["cond_latents"][:, :, 0].to(LATENT_DTYPE)
            st["cond_video_latents"] = cond_video
        return st

    def _shape(self, md: dict) -> tuple:
        """A request's height, width, fps and step count, and stand-ins of its prompts' lengths for its statics."""
        cond_len, uncond_len = md[PROMPT_LENGTHS]
        return (
            int(md.get("height", 256)), int(md.get("width", 256)), float(md.get("fps", 24.0)),
            int(md.get("num_inference_steps", self.config.num_inference_steps)),
            ([0] * cond_len, [0] * uncond_len if _has_guidance(md) else None),
        )

    def _memoized(self, md: dict, device) -> dict:
        """What a request that isn't windowed takes from the memo: its sampler, and its statics and masks."""
        height, width, fps, steps, ids = self._shape(md)
        settings = self._action_settings if md.get("action_mode") else self._clip_settings
        return {
            "scheduler": self._sampler(steps, md.get("flow_shift"), md.get("use_karras_sigma"), device),
            **settings(md, *ids, height, width, fps, device),
        }

    def _clip_settings(self, md, cond_ids, uncond_ids, height, width, fps, device) -> dict:
        num_frames = int(md.get("num_frames", 1))
        latent_shape = self._latent_shape(height, width, num_frames)
        # Video-to-video: the request's condition_frame_indexes_vision pin clean
        # latent frames from the conditioning video; the complement is predicted.
        condition_indexes = md.get("condition_frame_indexes_vision")
        noisy_frames = None
        if condition_indexes:
            noisy_frames = [f for f in range(latent_shape[2]) if f not in set(condition_indexes)]
        num_sound = sound_frames(md, self.config)[1] if md.get("generate_sound") else None
        # Image-to-video: latent frame 0 is a clean conditioning anchor supplied
        # in the first denoise step's ``latents``; it stays in the sequence but is
        # not denoised.
        static = partial(
            self._static, height=height, width=width, num_frames=num_frames, fps=fps,
            has_image_condition=bool(md.get("has_image_condition", False)), device=device,
            sound_latent_frames=num_sound, noisy_frames=noisy_frames,
        )
        st = {
            "cond": static(cond_ids), "uncond": None if uncond_ids is None else static(uncond_ids),
            "latent_shape": latent_shape, "num_sound": num_sound,
        }
        if condition_indexes:
            dtype = self.transformer.proj_in.weight.dtype
            st["vmask"] = self._pin_mask(latent_shape[2], condition_indexes, device, dtype)
        return st

    def _action_settings(self, md, cond_ids, uncond_ids, height, width, fps, device) -> dict:
        mode = md["action_mode"]
        # The action band: the chunk the reply carries, behind a policy
        # request's robot_state token (clean, at video frame 0).
        state_rows = int(md.get("action_state_rows", 0))
        action_chunk = int(md["action_chunk_size"]) + state_rows
        num_frames = int(md.get("num_frames") or action_chunk + 1)
        static = partial(
            self._action_static, height=height, width=width, num_frames=num_frames, action_chunk=action_chunk,
            mode=mode, fps=fps, action_fps=float(md.get("action_fps", fps)),
            action_offset=action_start_frame_offset(action_chunk, num_frames), device=device, state_rows=state_rows,
        )
        latent_shape = self._latent_shape(height, width, num_frames)
        dtype = self.transformer.proj_in.weight.dtype
        domain_id = int(md.get("domain_id", 0))
        # The visual conditioning (inverse-dynamics: the whole observed video;
        # forward-dynamics / policy: the conditioning frame) is VAE-encoded by
        # the parallel vae_encoder node and reaches the denoise loop as its
        # ``cond_latents`` input edge; the per-mode vmask selects which latent
        # frames are kept clean from it.
        return {
            "cond": static(cond_ids), "uncond": None if uncond_ids is None else static(uncond_ids),
            "latent_shape": latent_shape,
            "action_chunk": action_chunk,
            "action_dim": self.transformer.action_dim,
            "raw_action_dim": int(md["raw_action_dim"]),
            "domain_t": self._statics.get(
                ("domain", domain_id), lambda: torch.tensor([domain_id], dtype=torch.long, device=device),
            ),
            "vmask": self._pin_mask(
                latent_shape[2], vision_condition_frame_indexes(mode, latent_shape[2]), device, dtype,
            ),
            "action_clean_mask": self._statics.get(
                ("action_clean", mode, action_chunk, state_rows, dtype),
                lambda: self._action_clean_mask(mode, action_chunk, state_rows, device, dtype),
            ),
        }

    @staticmethod
    def _action_clean_mask(mode: str, action_chunk: int, state_rows: int, device, dtype) -> torch.Tensor:
        """``[1, action_chunk, 1]``, 1 on the action rows that stay clean: forward-dynamics' whole chunk, or a policy
        request's robot_state rows."""
        mask = torch.zeros((1, action_chunk, 1), device=device, dtype=dtype)
        if mode == "forward_dynamics":
            mask[:] = 1.0
        elif state_rows:
            mask[:, :state_rows] = 1.0
        return mask

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

    def _window_statics(
        self, cond_ids, uncond_ids, height, width, units, fps, has_image_condition, cond_units, device,
        first_window=True, start_unit=0,
    ):
        """`_build_window_statics`, slimmed and with their layout masks, from the memo."""
        key = (
            "window", len(cond_ids), None if uncond_ids is None else len(uncond_ids), height, width, units, fps,
            has_image_condition and first_window, cond_units, start_unit,
        )

        def build():
            statics = self._build_window_statics(
                cond_ids, uncond_ids, height, width, units, fps, has_image_condition, cond_units, device,
                first_window=first_window, start_unit=start_unit,
            )
            for static in statics:
                if static is not None:
                    self._slim_statics(static, None)
                    self._add_layout_masks(static, device)
            return statics

        return self._statics.get(key, build)

    def _window_settings(
        self, md, cond_ids, uncond_ids, height, width, fps, steps, step_index, inputs, device,
    ) -> dict:
        # kv mode appends each window's clean K/V to the cache (one commit
        # iteration per window, hence steps + 1 loop iterations), and the window
        # the request's layout keeps releases context beyond the horizon at each
        # commit; chained re-pins the previous tail as clean conditioning instead
        # and never touches committed state.
        schedule = _schedule(md)
        is_kv = md.get("window_mode") == "kv"
        per = steps + 1 if is_kv else steps
        plan = schedule.window(step_index // per)
        # A resumed session re-pins the stored tail as window 0's clean head:
        # the same clean-frame layout a chained window uses for its overlap.
        resume_units = int(md.get("resume_latent_units", 0) or 0) if plan.index == 0 else 0
        cond_units = resume_units or plan.cond_units
        # Chained windows restart their positions at 0, so every window past the
        # first has the same statics. kv windows carry absolute positions.
        cond, uncond = self._window_statics(
            cond_ids, uncond_ids, height, width, plan.units, fps,
            has_image_condition=bool(md.get("has_image_condition", False)), cond_units=cond_units, device=device,
            first_window=plan.index == 0, start_unit=plan.start if is_kv else 0,
        )
        session_id = md.get("session_id")
        pinned, context = (inputs or {}).get(WINDOW_COND) or [], (inputs or {}).get(SESSION_TAIL) or []
        st = {
            "cond": cond,
            "uncond": uncond,
            "latent_shape": self._window_latent_shape(height, width, plan.units),
            "num_sound": None,
            "ar_schedule": schedule,
            "ar_steps": steps,
            "ar_iters_per_window": per,
            "ar_kv_mode": is_kv,
            "ar_tokens_per_unit": cond["num_vision_tokens"] // plan.units,
            "window_index": plan.index,
            "cond_units": cond_units,
            "cond_video_latents": pinned[0] if pinned and pinned[0].numel() else None,
            "session_id": str(session_id) if session_id else None,
            "end_session": bool(md.get("end_session")),
            "session_ttl": md.get("session_timeout_s"),
            "session_tail": context[0] if context and context[0].numel() else None,
        }
        if cond_units:
            st["vmask"] = self._pin_mask(plan.units, range(cond_units), device, self.transformer.proj_in.weight.dtype)
        if resume_units:
            # The prefill pinned the session, and refused the request if it was missing.
            st["resume_latents"], st["session_tail"] = self._sessions.get(st["session_id"])
            st["resume_context"] = st["session_tail"]
        elif plan.index == 0:
            self._ingest_cond_latents(st, inputs, device)
        return st

    def _begin_session(self, fwd_info, md) -> None:
        """Pin the session a windowed request names: a second request on a live
        session is refused, and the store never evicts it mid-rollout. A resumed
        session the store doesn't hold, or holds at another latent size, is a
        request error, refused here once: silently starting from scratch would
        break the client's frame accounting."""
        session_id = str(md["session_id"])
        self._sessions.begin(session_id, fwd_info.rid_handle)
        units = int(md.get("resume_latent_units", 0) or 0)
        if not units:
            return
        stored = self._sessions.get(session_id)  # promotes the session
        if stored is None:
            raise ValueError(f"Cosmos3 resume_session: unknown or expired session {session_id!r}.")
        latents, _ = stored
        height, width = int(md.get("height", 256)), int(md.get("width", 256))
        shape = self._window_latent_shape(height, width, units)
        if latents.shape[2] < units or tuple(latents.shape[3:]) != tuple(shape[3:]):
            raise ValueError(
                f"Cosmos3 resume_session: session {session_id!r} holds latents of shape "
                f"{tuple(latents.shape)}, which cannot seed a {height}x{width} window."
            )

    def _keep_session(self, st, window_latents: torch.Tensor, context: torch.Tensor) -> None:
        """The rollout's final window and the stream's decode context behind it,
        kept for a ``resume_session`` follow-up until the session's timeout;
        ``end_session`` drops the session instead. Either way the request lets
        go of its pin."""
        if st["end_session"]:
            self._sessions.end(st["session_id"])
            return
        self._sessions.put(
            st["session_id"], (window_latents.detach().clone(), context.detach().clone()), st["session_ttl"],
        )

    def cleanup_request(self, request_id: str):
        # A request removed mid-rollout (cancel, failure) lets go of its session.
        self._sessions.release(request_id)
        super().cleanup_request(request_id)

    @staticmethod
    def _window_step(st, step_index: int) -> tuple[int, int, bool]:
        """A windowed request's global loop counter -> (window index,
        within-window step, whether this is a kv-mode commit iteration)."""
        per = st["ar_iters_per_window"]
        local = step_index % per
        commit = bool(st.get("ar_kv_mode")) and local == st["ar_steps"]
        return step_index // per, local, commit

    def _prepare_commit(self, st, window_index, latents, solver) -> ARNodeInputs:
        """Inputs of a kv-mode commit iteration: the window's newly generated
        span, which the declaration adds and keeps under every live guidance
        branch."""
        plan = st["ar_schedule"].window(window_index)
        span = (plan.commit_end - plan.commit_start) * st["ar_tokens_per_unit"]
        cfg = st["uncond"] is not None
        return ARNodeInputs(
            input_seq_len=span,
            tensor_inputs={"latents": latents, **solver},
            kwargs={"settings": st},
            resource_step_info=GenStepInfo(cfg=cfg, cfg_active=cfg, capture_key=None, commit=True),
        )

    def _commit_step(self, request_ids: list[str], spans: tuple[int, ...], cfg: bool) -> SubmoduleStep:
        """A kv-mode window commit: the finished window's new span added under
        every live guidance branch and kept — the frame-token analogue of the
        prefill, bidirectional like every generation step. Both branches commit
        regardless of any guidance interval: each branch's future windows read
        its own context. What the window the request's layout keeps no longer
        reaches goes back to the pool at this step's commit."""
        labels = (COND_LABEL, UNCOND_LABEL) if cfg else (COND_LABEL,)
        return SubmoduleStep(
            segments=self._segments(request_ids, labels, [spans] * len(labels)), state={KV_CACHE: ArrayStep()},
        )

    def _commit_window(self, st, latents, window_index, solver) -> dict:
        """kv-mode commit iteration: run the generation tower over the
        window's finished clean latents (no timestep embedding — the clean-
        conditioning convention), adding their K/V to both guidance branches'
        streams in one pass, then end the window."""
        plan = st["ar_schedule"].window(window_index)
        stride = st["ar_tokens_per_unit"]
        dtype = self.transformer.proj_in.weight.dtype
        commit_latents = (
            latents[:, :, plan.cond_units:] if plan.cond_units else latents
        ).to(dtype)
        # The commit span's rotary rows are the tail of the window's table
        # (kv statics carry absolute frames, so those rows are already at the
        # right absolute positions).
        offset = plan.cond_units * stride
        rotary = [tuple(t[offset:] for t in st["cond"]["rotary"])]
        if st["uncond"] is not None:
            rotary.append(tuple(t[offset:] for t in st["uncond"]["rotary"]))
        self.transformer.commit_window(commit_latents, rotary)
        return self._end_window(st, latents, solver)

    def _window_start(self, st, generator, window_index, device) -> torch.Tensor:
        """A window's first latents: noise from the request's generator, behind
        the frames the window pins. Window 0 pins a resumed session's head,
        re-encoded here as a clip of its own (``ChainedWindowHead``), or
        image-to-video's anchor frame; a later chained window pins the previous
        window's last frames, which that window re-encoded as it ended
        (``_end_window``). The pinned frames go through the vmask
        video-to-video uses."""
        dtype = self.transformer.proj_in.weight.dtype
        latents = torch.randn(st["latent_shape"], generator=generator, device=device, dtype=dtype)
        units = st["cond_units"]
        if units and window_index == 0:
            cond_video = torch.zeros(st["latent_shape"], device=device, dtype=dtype)
            cond_video[:, :, :units] = self._window_head(st["resume_latents"], units).to(device=device, dtype=dtype)
            st["cond_video_latents"] = cond_video
        elif not units and st.get("cond_latents") is not None:
            latents[:, :, 0] = st["cond_latents"][:, :, 0].to(dtype)
        if st.get("vmask") is not None:
            latents = st["vmask"] * st["cond_video_latents"] + (1.0 - st["vmask"]) * latents
        return latents

    def _end_window(self, st, window_latents, solver) -> dict:
        """End of a window: its clean latents leave on the streaming edge, a
        resumed session's decode context ahead of the first window's, and ride
        ``latents`` to the next window's first step (``_window_start``), with
        the frames a chained next window pins on ``window_cond``. A session's
        decode context gains the window's new latents."""
        context = st["session_tail"]
        if st["session_id"]:
            new = window_latents[:, :, st["cond_units"]:]
            context = new if context is None else torch.cat([context, new.to(context.dtype)], dim=2)
            context = context[:, :, -max(1, int(self.config.windowed_decode_context_latents)):]
        stream = window_latents
        if st.get("resume_context") is not None:
            stream = torch.cat([st["resume_context"].to(window_latents.dtype), window_latents], dim=2)
        return {
            "latents": [window_latents], "window_latents": [stream], **self._solver_edges(solver),
            **self._window_edges(latents=window_latents, pinned=self._next_pinned(st, window_latents), context=context),
        }

    def _next_pinned(self, st, window_latents: torch.Tensor) -> torch.Tensor | None:
        """The next window's pinned frames at its latent size: this window's last frames, re-encoded as a clip of
        their own (``ChainedWindowHead``). None after the last window, and before one that pins nothing."""
        schedule, index = st["ar_schedule"], st["window_index"] + 1
        if index >= schedule.num_windows or not schedule.window(index).cond_units:
            return None
        plan = schedule.window(index)
        dtype = self.transformer.proj_in.weight.dtype
        pinned = window_latents.new_zeros((*st["latent_shape"][:2], plan.units, *st["latent_shape"][3:]), dtype=dtype)
        pinned[:, :, :plan.cond_units] = self._window_head(window_latents, plan.cond_units).to(dtype)
        return pinned

    @staticmethod
    def _window_edges(
        latents: torch.Tensor, pinned: torch.Tensor | None, context: torch.Tensor | None,
    ) -> dict[str, list[torch.Tensor]]:
        """A windowed step's loop-back edges besides its latents: the frames the window pins and the session's decode
        context, each empty where there is none."""
        empty = latents.new_empty(0)
        return {
            WINDOW_COND: [empty if pinned is None else pinned], SESSION_TAIL: [empty if context is None else context],
        }

    def _step_outputs(self, st, latents: torch.Tensor, solver) -> dict[str, list[torch.Tensor]]:
        outputs = {"latents": [latents], **self._solver_edges(solver)}
        if "ar_schedule" in st:
            outputs.update(
                self._window_edges(latents=latents, pinned=st.get("cond_video_latents"), context=st["session_tail"]),
            )
        return outputs

    @staticmethod
    def _first_solver_state(st, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        # over the sample ``_scheduler_step`` steps
        return st["scheduler"].initial_state(latents.unsqueeze(0))

    @staticmethod
    def _solver_edges(solver: Mapping[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        return {name: [tensor] for name, tensor in solver.items()}

    @staticmethod
    def _slim_statics(cond: dict, uncond: dict | None) -> None:
        """Drop packed-static fields the denoise loop never reads: the token
        ids and und-tower rotary ids ride the prefill node inputs, the raw
        joint-sequence mse indexes are superseded by the gen-relative
        ``*mse_gen_indexes``, and the generation band's position ids by its
        ``rotary`` table."""
        for s in (cond, uncond) if uncond is not None else (cond,):
            for key in (
                "input_ids", "text_mrope_ids", "vision_mse_loss_indexes",
                "sound_mse_loss_indexes", "action_mse_loss_indexes",
                "position_ids", "gen_position_ids", "vision_mrope_ids",
                "sound_mrope_ids", "action_mrope_ids",
            ):
                s.pop(key, None)

    def _get_prefill_node_inputs(self, cond_ids, uncond_ids, device, windowed: bool = False):
        statics = {COND_LABEL: build_text_segment(cond_ids, self.config, device)}
        if uncond_ids is not None:
            statics[UNCOND_LABEL] = build_text_segment(uncond_ids, self.config, device)
        tensor_inputs = {}
        for label, s in statics.items():
            tensor_inputs[f"input_ids_{label}"] = s["input_ids"]
            tensor_inputs[f"text_mrope_ids_{label}"] = s["text_mrope_ids"]
        return ARNodeInputs(
            input_seq_len=sum(s["und_len"] for s in statics.values()),
            tensor_inputs=tensor_inputs,
            kwargs=dict(
                cfg=uncond_ids is not None,
                seq_lens={label: s["und_len"] for label, s in statics.items()},
                windowed=windowed,
            ),
        )

    def _ingest_cond_latents(self, st, inputs, device) -> None:
        """Adopt the vae_encoder node's conditioning latents, which the gen
        walk's ``cond_latents`` edge presents every iteration (empty for
        unconditioned requests). Video/action conditioning pins the
        full-latent-shape latents under ``cond_video_latents``; image-to-video
        takes the single-frame anchor under ``cond_latents``. An action request
        without visual conditioning pins zeros (matching its all-zero clean
        anchors); a video-conditioned request whose video never arrived cannot
        serve."""
        incoming = (inputs or {}).get("cond_latents")
        latents = incoming[0].to(device) if incoming else None
        if st.get("vmask") is not None:
            if latents is not None:
                st["cond_video_latents"] = latents
            elif "action_chunk" in st:
                shape, model_dtype = tuple(st["latent_shape"]), self.transformer.proj_in.weight.dtype
                st["cond_video_latents"] = self._statics.get(
                    ("zeros", shape, model_dtype), lambda: torch.zeros(shape, device=device, dtype=model_dtype),
                )
            else:
                raise ValueError(
                    "Cosmos3 video conditioning was requested but no conditioning video arrived."
                )
        elif latents is not None:
            st["cond_latents"] = latents

    def _generator(self, fwd_info, device) -> torch.Generator:
        """The request's generator: seeded from its seed, then where its last
        step's draws left it, which ``PROGRESS`` holds once a step has drawn."""
        generator = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
        rng = host(self.progress, fwd_info.rid_handle, COND_LABEL, 0)["rng"]
        if rng:
            generator.set_state(torch.frombuffer(bytearray(rng), dtype=torch.uint8))
        return generator

    @staticmethod
    def _initial_latents(st, generator, device) -> torch.Tensor:
        latents = torch.randn(st["latent_shape"], generator=generator, device=device, dtype=LATENT_DTYPE)
        cond_latents = st.get("cond_latents")
        if cond_latents is not None:
            # Image-to-video: latent frame 0 is the clean conditioning anchor;
            # the rest is noise. It stays clean through the loop because the
            # predicted velocity is zero on conditioning frames (unpatchify
            # only fills the noisy frames), matching the fused pipeline.
            latents[:, :, 0] = cond_latents[:, :, 0].to(latents.dtype)
        if st.get("vmask") is not None:
            # Video-to-video: pinned latent frames start clean, the rest
            # from the noise drawn above (the reference RNG order).
            latents = st["vmask"] * st["cond_video_latents"] + (1.0 - st["vmask"]) * latents
        return latents

    def _clean_action(self, md, st, inputs, device) -> torch.Tensor:
        """The action band's clean values. Forward-dynamics conditions on a
        clean action chunk supplied with the request, and a policy request's
        robot_state (in the policy's convention, see
        Cosmos3Model._action_prompt_output) is the band's first row; both ride
        the ``action_inputs`` edge. Everything else is predicted (clean values
        are zero)."""
        chunk, raw, dtype = st["action_chunk"], st["raw_action_dim"], self.transformer.proj_in.weight.dtype
        state_rows = int(md.get("action_state_rows", 0))
        clean_action = torch.zeros((1, chunk, st["action_dim"]), device=device, dtype=dtype)
        action_inputs = (inputs or {}).get("action_inputs")
        raw_act = action_inputs[0] if action_inputs else None
        if md["action_mode"] == "forward_dynamics" and raw_act is not None:
            act = torch.as_tensor(raw_act, device=device, dtype=dtype)
            if act.ndim == 3:
                act = act[0]
            if act.shape[0] < chunk:
                act = torch.cat([act, act[-1:].repeat(chunk - act.shape[0], 1)], dim=0)
            elif act.shape[0] > chunk:
                act = act[:chunk]
            clean_action[:, :, :raw] = act[:, :raw]
        elif state_rows and raw_act is not None:
            act = torch.as_tensor(raw_act, device=device, dtype=dtype).reshape(state_rows, -1)
            clean_action[:, :state_rows, :raw] = act[:, :raw]
        return clean_action

    def _prepare_image_gen(
        self, graph_walk, fwd_info, inputs, device,
    ) -> ARNodeInputs:
        md = fwd_info.step_metadata
        step_index = _loop_step(fwd_info, graph_walk)
        st = self._settings(md, step_index, inputs, device)
        windowed = "ar_schedule" in st
        first = _first_iteration(inputs, step_index, fwd_info.request_id)
        if windowed:
            # The loop counter is global over every window's iterations; the
            # schedule index is the within-window step, and in kv mode the
            # extra per-window iteration is the commit pass.
            window_index, step_index, commit = self._window_step(st, step_index)
            if commit:
                solver = {name: inputs[name][0] for name in self._solver_state}
                return self._prepare_commit(st, window_index, inputs["latents"][0], solver)
        generator = None
        if first or (windowed and step_index == 0):
            # A windowed request's later windows draw their noise from the same
            # generator, so a seeded windowed request is deterministic end to end.
            generator = self._generator(fwd_info, device)
            if windowed:
                latents = self._window_start(st, generator, window_index, device)
            else:
                latents = self._initial_latents(st, generator, device)
            solver = self._first_solver_state(st, latents)
        else:
            latents = inputs["latents"][0]
            solver = {name: inputs[name][0] for name in self._solver_state}
        if self.config.distilled_sigmas:
            # The distilled SDE step re-noises every position from the same
            # generator (the reference passes the pipeline generator).
            generator = generator or self._generator(fwd_info, device)
            st["sde_noise"] = st["scheduler"].noise(latents.unsqueeze(0), generator)
        tensors = {"latents": latents, **solver}
        # The CUDA-graph capture reads the timestep, rotary tables and the
        # clean/noisy token mask as static buffers (it can't reach the
        # request's sampler at replay), so materialize them here. The eager
        # path ignores these and recomputes from the request's settings. Only
        # built for a request that could land on a captured graph (two-branch
        # guidance at a captured shape; see ``_capture_key``).
        if st["uncond"] is not None and self._capture_key(graph_walk, md) is not None:
            # ``step_index`` is the within-window step for a windowed request
            t = st["scheduler"].timesteps[step_index].to(torch.float32)
            # Every graph is built with all frames declared noisy, so the
            # timestep buffer spans every generation token.
            tensors["vision_timesteps"] = t.reshape(1).expand(st["cond"]["num_vision_tokens"]).contiguous()
            tensors["cos_cond"], tensors["sin_cond"] = st["cond"]["rotary"]
            tensors["cos_uncond"], tensors["sin_uncond"] = st["uncond"]["rotary"]
            tensors["noisy_token_mask"] = st["cond"]["noisy_token_mask"]
        return ARNodeInputs(
            input_seq_len=st["cond"]["num_vision_tokens"],
            tensor_inputs=tensors,
            kwargs={"settings": st, "rng": self._drawn(generator, windowed)},
            resource_step_info=self._step_info(graph_walk, md, st, step_index),
        )

    def _drawn(self, generator: torch.Generator | None, windowed: bool) -> bytes | None:
        """The generator's state for ``PROGRESS`` after this step's draws, where a later step draws again: a windowed
        request's next window, or the distilled sampler's next step. None where this step drew nothing."""
        if generator is None or not (windowed or self.config.distilled_sigmas):
            return None
        return generator.get_state().numpy().tobytes()

    def _prepare_video_sound_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        md = fwd_info.step_metadata
        step_index = _loop_step(fwd_info, VIDEO_SOUND_GEN_WALK)
        st = self._settings(md, step_index, inputs, device)
        if _first_iteration(inputs, step_index, fwd_info.request_id):
            # First iteration: video noise first, then sound noise from the same
            # generator (the reference pipeline's RNG order), so the video band
            # of a sound request is bit-identical to the same-seed video-only
            # request. Both in the fp32 loop dtype, as the video-only path draws.
            generator = self._generator(fwd_info, device)
            latents = self._initial_latents(st, generator, device)
            sound_latents = torch.randn(
                (1, self.config.sound_dim, st["num_sound"]), generator=generator, device=device, dtype=LATENT_DTYPE
            )
            solver = st["scheduler"].initial_state(self._packed(latents, sound_latents))
        else:
            latents = inputs["latents"][0]
            sound_latents = inputs["sound_latents"][0]
            solver = {name: inputs[name][0] for name in self._solver_state}

        return ARNodeInputs(
            input_seq_len=st["cond"]["num_vision_tokens"] + st["num_sound"],
            tensor_inputs={"latents": latents, "sound_latents": sound_latents, **solver},
            kwargs={"settings": st},
            resource_step_info=self._step_info(VIDEO_SOUND_GEN_WALK, md, st, step_index),
        )

    def _prepare_action_gen(self, graph_walk, fwd_info, inputs, device) -> ARNodeInputs:
        md = fwd_info.step_metadata
        step_index = _loop_step(fwd_info, graph_walk)
        st = self._settings(md, step_index, inputs, device)
        if _first_iteration(inputs, step_index, fwd_info.request_id):
            # First iteration: build the joint [video | action] latents. Per the
            # mode masks, conditioning frames/action are clean and the predicted
            # ones start from noise; the clean anchors are then carried in the
            # looped latents (re-injected each step). Action noise is drawn before
            # the video noise to match the fused pipeline's RNG order.
            from diffusers.utils.torch_utils import randn_tensor

            dtype = LATENT_DTYPE
            generator = self._generator(fwd_info, device)
            chunk, adim, raw = st["action_chunk"], st["action_dim"], st["raw_action_dim"]
            a_noise = randn_tensor((1, chunk, adim), generator=generator, device=device, dtype=dtype)
            a_noise[..., raw:] = 0
            action_latents = (
                st["action_clean_mask"] * self._clean_action(md, st, inputs, device)
                + (1.0 - st["action_clean_mask"]) * a_noise
            )
            action_latents[..., raw:] = 0
            v_noise = randn_tensor(st["latent_shape"], generator=generator, device=device, dtype=dtype)
            latents = st["vmask"] * st["cond_video_latents"] + (1.0 - st["vmask"]) * v_noise
            solver = st["scheduler"].initial_state(self._packed(latents, action_latents))
        else:
            latents = inputs["latents"][0]
            action_latents = inputs["action_latents"][0]
            solver = {name: inputs[name][0] for name in self._solver_state}

        return ARNodeInputs(
            input_seq_len=st["cond"]["num_vision_tokens"] + st["cond"]["num_action_tokens"],
            tensor_inputs={"latents": latents, "action_latents": action_latents, **solver},
            kwargs={"settings": st},
            resource_step_info=self._step_info(ACTION_GEN_WALK, md, st, step_index),
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
        """This batch's step: which guidance branches it touches on the ``kv``
        array, and by how much.

        Two guidance branches are two labels on the one array, which a step
        over both plans as one. Whether a request has both, and whether this
        step uses both, is a per-request fact that rides in on
        ``NodeInputs.resource_step_info`` from ``prepare_inputs`` (see
        ``GenStepInfo``) rather than being read off the batch here.
        """
        if graph_walk in PREFILL_WALKS:
            # The text prefix is written once and kept; the denoise steps read
            # it for the rest of the request. Both guidance branches prefill
            # together when present.
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
                state={KV_CACHE: ArrayStep()},
            )

        if (
            graph_walk not in GEN_WALKS
            and graph_walk not in SOUND_WALKS
            and graph_walk not in ACTION_WALKS
        ):
            raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

        # Denoise steps read the kept prefix and recompute the same generation
        # span every step, so nothing is kept. The fresh tokens are still
        # declared as ordinary spans: that is what gives the dense kernel its
        # query lengths.
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
        if slot_lease is not None and capture_key is not None:
            # The captured denoise graph has a fixed shape and always runs both
            # guidance branches — including outside a request's guidance
            # interval, where postprocess keeps the conditional one. See `cg_key_info`.
            return self._gen_step(request_ids, inputs, spans, (COND_LABEL, UNCOND_LABEL), cg_key_info=capture_key)
        if len(infos) > 1:
            # Cross-request batch: one step over every request's guidance
            # branches (a single branch when guidance is off). The image and
            # sound batched forwards run both branches whenever the branch
            # exists. The action one runs the conditional branch alone when no
            # row is inside its guidance interval, as a single request does
            # (rows outside it in a guided batch drop the other branch).
            cfg = infos[0].cfg and (graph_walk not in ACTION_WALKS or any(info.cfg_active for info in infos))
        else:
            # Guidance off, or a guidance_interval out-of-interval step, runs
            # the conditional branch alone.
            cfg = infos[0].cfg_active
        return self._gen_step(request_ids, inputs, spans, (COND_LABEL, UNCOND_LABEL) if cfg else (COND_LABEL,))

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
        self, request_ids: list[str], inputs: list[ARNodeInputs], spans: tuple[int, ...],
        labels: tuple[str, ...], cg_key_info: object | None = None,
    ) -> SubmoduleStep:
        """A denoise step: the same span under each live label, nothing kept,
        bidirectional over [kept prefix | fresh tokens], and under ``PROGRESS``
        where each row's generator stands after the step's draws."""
        state = {KV_CACHE: ArrayStep(keep=False)}
        drawn = [
            (rid, inp.kwargs["rng"], span)
            for rid, inp, span in zip(request_ids, inputs, spans, strict=True) if inp.kwargs.get("rng") is not None
        ]
        if drawn:
            # PROGRESS has only the conditional label, which every row opens
            state[PROGRESS] = ArrayStep(
                segments=tuple(Segment(rid, COND_LABEL, span) for rid, _, span in drawn),
                host={(rid, COND_LABEL, 0): {"rng": rng} for rid, rng, _ in drawn},
            )
        return SubmoduleStep(
            cg_key_info=cg_key_info,
            segments=self._segments(request_ids, labels, [spans] * len(labels)),
            state=state,
        )

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
        static-input tensors (``CAPTURED_GEN_INPUTS``) are stacked on a
        leading batch dim, so one captured graph spans a whole
        concurrent batch (a batch of one for the single-request latency
        path); the replay side copies each request's tensors into these
        fixed buffers. The attention plan comes from the step declaration,
        which always covers both guidance branches here (the graph shape is
        fixed), while the eager declaration skips the uncond plan on
        guidance-interval steps.
        """
        return {
            key: torch.stack([inp.tensor_inputs[key] for inp in inputs])
            for key in CAPTURED_GEN_INPUTS
        }

    def preprocess(
        self, graph_walk, engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs]
    ) -> dict:
        if graph_walk in GEN_WALKS and getattr(engine_inputs, "captured", False):
            return self._preprocess_image_gen_captured(inputs)

        if graph_walk in PREFILL_WALKS:
            labels = [COND_LABEL, UNCOND_LABEL] if inputs[0].kwargs.get("cfg") else [COND_LABEL]
            # Pack label-major (all cond segments, then all uncond) to match
            # the (label, request) order of the declared segments.
            return {
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
            names = ("latents", *self._solver_state)
        elif graph_walk in SOUND_WALKS:
            names = ("latents", "sound_latents", *self._solver_state)
        elif graph_walk in ACTION_WALKS:
            names = ("latents", "action_latents", *self._solver_state)
        else:
            raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")
        if len(rids) > 1:
            return {
                **{name: {r: inp.tensor_inputs[name] for r, inp in zip(rids, inputs, strict=True)} for name in names},
                "settings": {r: inp.kwargs["settings"] for r, inp in zip(rids, inputs, strict=True)},
            }
        return {**{name: inputs[0].tensor_inputs[name] for name in names}, "settings": inputs[0].kwargs["settings"]}

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    # Run the transformer in the model's native bf16, NOT the engine autocast:
    # autocast keeps norms in fp32, perturbing the velocity ~1 ULP/step, which
    # the multi-step denoise amplifies, and it would run the fp32 scheduler
    # step's einsum in bf16. The reference runs the transformer in pure bf16
    # and the loop state around it in fp32 (LATENT_DTYPE).
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, **kwargs):
        if graph_walk in PREFILL_WALKS:
            return self._forward_prefill(**kwargs)
        st = kwargs.pop("settings")
        k = _loop_step(engine_inputs.per_request_info[engine_inputs.request_ids[0]], graph_walk)
        if graph_walk in GEN_WALKS:
            return self._forward_image_gen(st, k, **kwargs)
        if graph_walk in SOUND_WALKS:
            return self._forward_video_sound_gen(st, k, **kwargs)
        if graph_walk in ACTION_WALKS:
            return self._forward_action_gen(st, k, **kwargs)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _forward_prefill(self, input_ids: torch.Tensor, text_mrope_ids: torch.Tensor, **kwargs) -> dict:
        self.transformer.prefill_und(input_ids, text_mrope_ids)
        return {}

    def _denoise(self, static, latents, vision_timesteps):
        return self.transformer.denoise_step(
            latents,
            vision_timesteps,
            static["rotary"],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
        )

    def _cfg_active(self, st, step_index: int) -> bool:
        """Whether this eager denoise step runs classifier-free guidance (both
        branches combined). False ⇒ the conditional branch runs alone — the
        guidance_scale==1 case and steps whose timestep falls outside the
        guidance_interval [lo, hi] (the t2i recipe's default, or the request's).

        ``prepare_inputs`` (through ``_step_info``) and ``_forward_image_gen`` /
        ``_forward_action_gen`` call this for the same step, so the declared
        attention (batched vs cond-only) matches the forward that runs.

        The captured path and the cross-request batch always run both
        branches; ``postprocess`` and ``forward_batched`` call this to keep
        the conditional one outside the interval. See ``cg_key_info`` for why
        the interval is not folded into the capture key."""
        if st["uncond"] is None:
            return False
        gi = st.get("guidance_interval")
        if gi is None:
            return True
        t = float(_host_timesteps(st["scheduler"])[step_index])
        return gi[0] <= t <= gi[1]

    def _forward_image_gen(self, st, k, latents, **solver) -> dict:
        step_index = k
        windowed = "ar_schedule" in st
        if windowed:
            # The loop counter is global; each window runs its own schedule
            # over ar_steps steps, and in kv mode the extra per-window
            # iteration commits the finished window's K/V.
            window_index, local, commit = self._window_step(st, step_index)
            if commit:
                return self._commit_window(st, latents, window_index, solver)
            step_index = local
        t = _host_timesteps(st["scheduler"])[step_index]
        vision_timesteps = torch.full((st["cond"]["num_noisy_vision_tokens"],), t, device=latents.device)

        # Classifier-free guidance is applied only when an uncond branch exists
        # (guidance_scale != 1) and, for the text-to-image recipe, only on the
        # configured timestep interval. Outside the interval the step runs the
        # conditional branch alone (cond-only velocity), matching the recipe.
        if not self._cfg_active(st, step_index):
            velocity = self._denoise(st["cond"], latents, vision_timesteps).to(latents.dtype)
        else:
            cond_v, uncond_v = self.transformer.denoise_step_batched_cfg(
                latents,
                vision_timesteps,
                st["cond"]["rotary"],
                st["uncond"]["rotary"],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
            )
            velocity = _guided(cond_v, uncond_v, st["gs"], latents.dtype)

        new_latents, solver = self._scheduler_step(st, velocity, latents, step_index, solver)
        if st.get("vmask") is not None:
            # Video-to-video: re-inject the clean conditioning frames after the
            # scheduler step (the reference pipeline does the same) so scheduler
            # rounding can't drift the pinned latents.
            new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
        if windowed and local + 1 == st["ar_steps"] and not st.get("ar_kv_mode"):
            # Chained: the window ends at its last denoise step. kv windows
            # end at their commit iteration instead.
            return self._end_window(st, new_latents, solver)
        return self._step_outputs(st, new_latents, solver)

    def _sound_kwargs(self, static, sound_latents, sound_ts) -> dict:
        return dict(
            sound_latents=sound_latents,
            sound_token_shapes=static["sound_token_shapes"],
            sound_noisy_frame_indexes=static["sound_noisy_frame_indexes"],
            sound_mse_gen_indexes=static["sound_mse_gen_indexes"],
            sound_timesteps=sound_ts,
        )

    @staticmethod
    def _packed(latents: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        """The joint ``[1, N]`` state of the video latents and a second band."""
        return torch.cat([latents.reshape(1, -1), band.reshape(1, -1)], dim=1)

    def _sound_scheduler_step(self, st, latents, sound_latents, video_v, sound_v, step_index, solver):
        """One joint [video | sound] scheduler step: flatten both bands into one
        packed state so the request's scheduler advances them under a single
        multistep history (the flow update is linear per element), then split
        back. Every sound frame is noisy, so no clean re-injection is needed on
        the sound band; the video band's i2v anchor stays clean because its
        predicted velocity is zero (as in the video-only path). The step runs in
        the latents' dtype, like the reference's fp32 step."""
        video_v, sound_v = video_v.to(latents.dtype), sound_v.to(sound_latents.dtype)
        nv = video_v.numel()
        packed_next, solver = st["scheduler"].step(
            self._packed(video_v, sound_v), self._packed(latents, sound_latents), step_index, solver,
        )
        new_latents = packed_next[:, :nv].reshape(latents.shape)
        new_sound = packed_next[:, nv:].reshape(sound_latents.shape)
        return new_latents, new_sound, solver

    def _forward_video_sound_gen(self, st, k, latents, sound_latents, **solver) -> dict:
        step_index = k
        t = _host_timesteps(st["scheduler"])[step_index]
        vts = torch.full((st["cond"]["num_noisy_vision_tokens"],), t, device=latents.device)
        sts = torch.full((st["num_sound"],), t, device=latents.device)

        if not self._cfg_active(st, step_index):
            velocity, sound_v = self.transformer.denoise_step(
                latents, vts, st["cond"]["rotary"],
                st["cond"]["vision_token_shapes"], st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
        else:
            (cond_v, s_c), (uncond_v, s_u) = self.transformer.denoise_step_batched_cfg(
                latents, vts,
                st["cond"]["rotary"],
                st["uncond"]["rotary"],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                **self._sound_kwargs(st["cond"], sound_latents, sts),
            )
            velocity = _guided(cond_v, uncond_v, st["gs"], latents.dtype)
            sound_v = _guided(s_c, s_u, st["gs"], sound_latents.dtype)

        new_latents, new_sound, solver = self._sound_scheduler_step(
            st, latents, sound_latents, velocity, sound_v, step_index, solver,
        )
        if st.get("vmask") is not None:
            # Video-to-video: re-inject the clean conditioning frames on the
            # video band after the joint step (the reference pipeline does the
            # same); the sound band has no clean frames.
            new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
        return {"latents": [new_latents], "sound_latents": [new_sound], **self._solver_edges(solver)}

    def _denoise_action(self, static, latents, action_latents, vts, ats, domain):
        return self.transformer.denoise_step(
            latents,
            vts,
            static["rotary"],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
            action_latents=action_latents,
            action_token_shapes=static["action_token_shapes"],
            action_noisy_frame_indexes=static["action_noisy_frame_indexes"],
            action_mse_gen_indexes=static["action_mse_gen_indexes"],
            action_timesteps=ats,
            action_domain_id=domain,
        )

    def _action_scheduler_step(self, st, latents, action_latents, video_v, action_v, step_index, solver):
        """One joint [video | action] scheduler step for an action request: mask
        the predicted velocities to their noisy bands, step the request's own
        scheduler over the packed [video | action] state, then re-inject the clean
        conditioning anchors (conditioning frames / action stay clean each step,
        their masked-in values invariant). Shared by the single-request and
        cross-request batched action forwards. The step runs in the latents'
        dtype, like the reference's fp32 step."""
        raw, chunk, adim = st["raw_action_dim"], st["action_chunk"], st["action_dim"]
        video_v = video_v.to(latents.dtype) * (1.0 - st["vmask"])
        action_v = action_v.to(action_latents.dtype) * (1.0 - st["action_clean_mask"])
        action_v[..., raw:] = 0
        nv = video_v.numel()
        packed_next, solver = st["scheduler"].step(
            self._packed(video_v, action_v), self._packed(latents, action_latents), step_index, solver,
        )
        new_latents = packed_next[:, :nv].reshape(latents.shape)
        new_action = packed_next[:, nv:].reshape(1, chunk, adim)
        new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * latents
        new_action = (
            (1.0 - st["action_clean_mask"]) * new_action
            + st["action_clean_mask"] * action_latents
        )
        new_action[..., raw:] = 0
        return new_latents, new_action, solver

    def _forward_action_gen(self, st, k, latents, action_latents, **solver) -> dict:
        step_index = k
        t = _host_timesteps(st["scheduler"])[step_index]
        device = latents.device
        vts = torch.full((st["cond"]["num_noisy_vision_tokens"],), t, device=device)
        ats = torch.full((st["cond"]["num_noisy_action_tokens"],), t, device=device)
        domain = st["domain_t"]

        if not self._cfg_active(st, step_index):
            video_v, action_v = self._denoise_action(st["cond"], latents, action_latents, vts, ats, domain)
        else:
            (video_v, action_v), (v_u, a_u) = self.transformer.denoise_step_batched_cfg(
                latents,
                vts,
                st["cond"]["rotary"],
                st["uncond"]["rotary"],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                action_latents=action_latents,
                action_token_shapes=st["cond"]["action_token_shapes"],
                action_noisy_frame_indexes=st["cond"]["action_noisy_frame_indexes"],
                action_mse_gen_indexes=st["cond"]["action_mse_gen_indexes"],
                action_timesteps=ats,
                action_domain_id=domain,
            )
            video_v = _guided(video_v, v_u, st["gs"], latents.dtype)
            action_v = _guided(action_v, a_u, st["gs"], action_latents.dtype)

        new_latents, new_action, solver = self._action_scheduler_step(
            st, latents, action_latents, video_v, action_v, step_index, solver,
        )
        return {"latents": [new_latents], "action_latents": [new_action], **self._solver_edges(solver)}

    # ------------------------------------------------------------------
    # Cross-request batching: run several requests' denoise step together.
    # ------------------------------------------------------------------

    def can_batch(self, batch, model_inputs) -> bool:
        # The denoise step batches across concurrent requests at the same walk.
        # The batched forward packs each request's own token shapes, so requests
        # at different resolutions / frame counts (and, for action, different
        # modes / embodiment domains) can share the batch. One request stays on
        # the simpler single-request path.
        if len(batch.request_ids) < 2:
            return False
        if batch.graph_walk in PREFILL_WALKS:
            # Only in the two-branch guidance regime, so one step covers the
            # batch. A windowed request's prefill stays sequential: the kept kv
            # text prefix must not depend on which requests arrived together.
            return all(inp.kwargs["cfg"] and not inp.kwargs["windowed"] for inp in model_inputs)
        infos = [inp.resource_step_info for inp in model_inputs]
        if batch.graph_walk in ACTION_WALKS:
            # Action batches when all requests share the guidance regime (all
            # single-branch -- guidance-scale-1 inverse/forward-dynamics and base
            # policy -- or all two-branch), so one step covers the batch. Modes
            # and embodiment domains may differ: each request's masks, scheduler
            # and domain-aware action projection are applied per request.
            return len({info.cfg for info in infos}) == 1
        if batch.graph_walk not in GEN_WALKS and batch.graph_walk not in SOUND_WALKS:
            return False
        # Image/video/sound batch only in the two-branch guidance regime.
        # (Batches are per graph walk, so sound requests only ever batch with
        # sound requests.) Windowed requests join denoise batches — the
        # batched forward maps their loop counter to the within-window step and
        # handles chained window boundaries — but a kv commit iteration is a
        # different step (an add that keeps) and drops the batch to the
        # sequential path.
        cap = self.config.max_batched_vision_tokens
        if not all(info.cfg for info in infos) or (
            cap is not None and sum(info.vision_tokens for info in infos) > cap
        ):
            return False
        return not any(info.commit for info in infos)

    def max_batch_size(self, graph_walk: str):
        if graph_walk in GEN_WALKS or graph_walk in SOUND_WALKS or graph_walk in ACTION_WALKS:
            return self.max_gen_batch_size
        return None

    def step_order(self, graph_walk: str) -> StepOrder:
        """Chained windows stream to the decoder as they finish, so a
        session's stream keeps moving while others run. Every other walk's
        output comes at the end of its loop."""
        return StepOrder.TURNS if graph_walk == VIDEO_GEN_AR_WALK else StepOrder.ARRIVAL

    # Native bf16, not the engine autocast, as forward() runs: the same dtypes as
    # the per-request path, so batching (and the compiled layer stack) changes
    # only kernel-level rounding.
    @torch.autocast(device_type="cuda", enabled=False)
    def forward_batched(
        self, graph_walk, engine_inputs: ModelInputsFromEngine,
        latents=None, action_latents=None, sound_latents=None,
        input_ids=None, text_mrope_ids=None, settings=None, **kwargs,
    ):
        if graph_walk in PREFILL_WALKS:
            self.transformer.prefill_und(input_ids, text_mrope_ids)
            return {}
        if graph_walk in ACTION_WALKS:
            return self._forward_batched_action(graph_walk, engine_inputs, settings, latents, action_latents, kwargs)
        if graph_walk in SOUND_WALKS:
            return self._forward_batched_sound(graph_walk, engine_inputs, settings, latents, sound_latents, kwargs)
        if graph_walk not in GEN_WALKS:
            raise ValueError(f"Cosmos3 batched forward only supports generation walks, got {graph_walk!r}")
        reqs, meta = [], []
        for rid in engine_inputs.request_ids:
            st = settings[rid]
            lat = latents[rid]
            step_index = _loop_step(engine_inputs.per_request_info[rid], graph_walk)
            if "ar_schedule" in st:
                # Windowed: the loop counter is global; the schedule index is
                # the within-window step (commit iterations never batch).
                step_index = self._window_step(st, step_index)[1]
            timesteps = _host_timesteps(st["scheduler"])
            # A request may be one step past its denoise count (a discarded extra
            # step) while others in the batch are still running; clamp its
            # timestep so the shared forward can't index past the schedule, and
            # skip its scheduler step below.
            t = timesteps[min(step_index, len(timesteps) - 1)]
            reqs.append(self.transformer.denoise_step_row(
                st["cond"],
                lat,
                torch.full((st["cond"]["num_noisy_vision_tokens"],), t, device=lat.device),
                st["cond"]["rotary"],
                st["uncond"]["rotary"],
            ))
            meta.append((rid, st, lat, step_index))

        results = self.transformer.denoise_step_batched(reqs)

        out = {}
        for (rid, st, lat, local), (cond_v, uncond_v) in zip(meta, results, strict=True):
            if self._cfg_active(st, local):
                velocity = _guided(cond_v, uncond_v, st["gs"], lat.dtype)
            else:  # outside the guidance_interval, as in _forward_image_gen
                velocity = cond_v.to(lat.dtype)
            solver = {name: kwargs[name][rid] for name in self._solver_state}
            new_latents, solver = self._scheduler_step(st, velocity, lat, local, solver)
            if st.get("vmask") is not None:
                # Video-to-video: re-inject the clean conditioning frames, as in
                # the single-request path.
                new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
            if (
                "ar_schedule" in st
                and local + 1 == st["ar_steps"]
                and not st.get("ar_kv_mode")
            ):
                # Chained window boundary inside a batch, as in the
                # single-request path.
                out[rid] = self._end_window(st, new_latents, solver)
            else:
                out[rid] = self._step_outputs(st, new_latents, solver)
        return out

    def _forward_batched_sound(self, graph_walk, engine_inputs, states, latents, sound_latents, solvers):
        """Run several sound requests' joint [video | sound] denoise step in one
        forward. Mirrors the image batched path with per-request sound bands:
        one batched transformer pass, then per request the guidance combine and
        its own joint [video | sound] scheduler step."""
        reqs, meta = [], []
        for rid in engine_inputs.request_ids:
            st = states[rid]
            lat, snd = latents[rid], sound_latents[rid]
            step_index = _loop_step(engine_inputs.per_request_info[rid], graph_walk)
            timesteps = _host_timesteps(st["scheduler"])
            # A request may be one (discarded) step past its denoise count while
            # others in the batch are still running; clamp its timestep so the
            # shared forward can't index past the schedule, and skip its
            # scheduler step below.
            t = timesteps[min(step_index, len(timesteps) - 1)]
            cond, unc = st["cond"], st["uncond"]
            reqs.append(self.transformer.denoise_step_row(
                cond,
                lat,
                torch.full((cond["num_noisy_vision_tokens"],), t, device=lat.device),
                cond["rotary"],
                unc["rotary"],
                sound_latents=snd,
                sound_timesteps=torch.full((st["num_sound"],), t, device=lat.device),
            ))
            meta.append((rid, st, lat, snd, step_index))

        results = self.transformer.denoise_step_batched(reqs)

        out = {}
        for (rid, st, lat, snd, k), ((cond_v, s_c), (uncond_v, s_u)) in zip(meta, results, strict=True):
            if self._cfg_active(st, k):
                velocity = _guided(cond_v, uncond_v, st["gs"], lat.dtype)
                sound_v = _guided(s_c, s_u, st["gs"], snd.dtype)
            else:  # outside the guidance_interval, as in _forward_image_gen
                velocity, sound_v = cond_v.to(lat.dtype), s_c.to(snd.dtype)
            solver = {name: solvers[name][rid] for name in self._solver_state}
            new_latents, new_sound, solver = self._sound_scheduler_step(st, lat, snd, velocity, sound_v, k, solver)
            if st.get("vmask") is not None:
                # Video-to-video: re-inject the clean conditioning frames on the
                # video band, as in the single-request path.
                new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
            out[rid] = {"latents": [new_latents], "sound_latents": [new_sound], **self._solver_edges(solver)}
        return out

    def _forward_batched_action(self, graph_walk, engine_inputs, states, latents, action_latents, solvers):
        """Run several action requests' joint [video | action] denoise step in one
        forward. Mirrors the image batched path: build each request's static gen
        inputs (clamping a request that has run one step past its denoise count),
        run one batched transformer pass, then per request combine the guidance
        branches (when present) and apply its own joint scheduler step. When no
        row is inside its guidance interval the batch runs the conditional
        branch alone, as ``declare_step`` planned it."""
        rids = engine_inputs.request_ids
        with_cfg = states[rids[0]]["uncond"] is not None and any(
            self._cfg_active(states[rid], _loop_step(engine_inputs.per_request_info[rid], graph_walk))
            for rid in rids
        )
        reqs, meta = [], []
        for rid in rids:
            st = states[rid]
            lat, act = latents[rid], action_latents[rid]
            step_index = _loop_step(engine_inputs.per_request_info[rid], graph_walk)
            timesteps = _host_timesteps(st["scheduler"])
            # A request may be one (discarded) step past its denoise count while
            # others in the batch are still running; clamp its timestep so the
            # shared forward can't index past the schedule, and skip its scheduler
            # step below.
            t = timesteps[min(step_index, len(timesteps) - 1)]
            cond = st["cond"]
            req = {
                "latents": lat,
                "action_latents": act,
                "vision_timesteps": torch.full((st["cond"]["num_noisy_vision_tokens"],), t, device=lat.device),
                "action_timesteps": torch.full((st["cond"]["num_noisy_action_tokens"],), t, device=lat.device),
                "rotary_cond": cond["rotary"],
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
                req["rotary_uncond"] = unc["rotary"]
            reqs.append(req)
            meta.append((rid, st, lat, act, step_index))

        results = self.transformer.denoise_step_action_batched(reqs, with_cfg)

        out = {}
        for (rid, st, lat, act, k), branches in zip(meta, results, strict=True):
            video_v, action_v = branches[0]
            # The batch runs both branches, but a step outside the request's
            # guidance_interval keeps the conditional one, as the single path does.
            if with_cfg and self._cfg_active(st, k):
                uncond_video, uncond_action = branches[1]
                video_v = _guided(video_v, uncond_video, st["gs"], lat.dtype)
                action_v = _guided(action_v, uncond_action, st["gs"], act.dtype)
            solver = {name: solvers[name][rid] for name in self._solver_state}
            new_latents, new_action, solver = self._action_scheduler_step(st, lat, act, video_v, action_v, k, solver)
            out[rid] = {"latents": [new_latents], "action_latents": [new_action], **self._solver_edges(solver)}
        return out

    # ------------------------------------------------------------------
    # CUDA-graph capture of the denoise step. Only the transformer velocity
    # computation is captured; the guidance combine and the (Python, multistep)
    # scheduler step run eagerly afterwards.
    # ------------------------------------------------------------------

    def get_accelerator_graph_configs(self, device, tp_world_size: int = 1):
        """Declare one fixed-shape capture of the image denoise step per
        resolution. Requests at other resolutions, or without guidance, fall back
        to the eager path. The per-resolution token layout is prompt-independent,
        so bake it once here and key it by latent shape; the per-prompt rotary
        tables, the latents and the timestep flow in as static-buffer inputs.

        Set ``COSMOS3_DISABLE_CUDA_GRAPH=1`` to skip capture and run the denoise
        loop eagerly (escape hatch for a misbehaving driver, and an A/B switch).
        Set ``COSMOS3_GEN_CAPTURE_RES`` (e.g. ``"192x320,480x832"``, height x
        width) to override which resolutions are captured, and
        ``COSMOS3_GEN_CAPTURE_BS`` (e.g. ``"1,4,8"``) to also capture batched
        denoise steps so concurrent requests replay a padded graph instead of
        falling back to the eager path."""
        if (
            self.transformer is None or not self.config.accelerator_graph
            or os.environ.get("COSMOS3_DISABLE_ACCELERATOR_GRAPH")
        ):
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
        # the static latents buffer holds the fp32 loop state; windowed bf16
        # latents copy into it exactly
        dtype = LATENT_DTYPE
        self._capture_layout: dict[tuple, dict] = {}
        configs = []
        # The capture is bit-faithful at every resolution (the rotary tables
        # are inputs, built outside the graph), but only HELPS
        # the launch-bound tiers: the graph's per-step input copies grow
        # with resolution and lose to the eager dense path at large latents.
        # Capture only below COSMOS3_GRAPH_MAX_LATENT_AREA (latent H*W).
        max_area = int(os.environ.get(
            "COSMOS3_GRAPH_MAX_LATENT_AREA", self.config.graph_max_latent_area))
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
                    "Cosmos3: skipping accelerator graph capture for %dx%dx%d (latent H*W "
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
    ) -> BatchedAcceleratorGraphConfig:
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
                "cos_cond": static["rotary"][0].clone(),
                "sin_cond": static["rotary"][1].clone(),
                "cos_uncond": static["rotary"][0].clone(),
                "sin_uncond": static["rotary"][1].clone(),
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
        return BatchedAcceleratorGraphConfig(
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
        if not os.environ.get("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH"):
            bs_env = os.environ.get("COSMOS3_PREFILL_CAPTURE_BS")
            prefill_bs = (
                [int(x) for x in bs_env.split(",")] if bs_env
                else list(self.prefill_capture_batch_sizes)
            )
            mrope_dtype = torch.float32 if self.config.enable_fps_modulation else torch.long

            configs.append(PackedAcceleratorGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                replay_graph_walks=list(PREFILL_WALKS),
                capture_token_lengths=list(self.prefill_capture_token_buckets),
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
        latents, vision_timesteps, cos_cond, sin_cond, cos_uncond, sin_uncond, noisy_token_mask,
        **kwargs,
    ) -> dict:
        """Velocity-only denoise forward captured into a CUDA graph: both guidance
        branches in one pass, no scheduler step. The token
        layout is baked per resolution; the latents, timestep and each branch's
        rotary cos and sin are static-buffer inputs stacked on a leading batch
        dim. A single request keeps the two-branch path; a concurrent batch
        runs the per-request denoise (the same compute as the eager
        cross-request forward), one transformer pass over the whole batch."""
        layout = self._capture_layout[tuple(latents.shape[1:])]
        rids = engine_inputs.request_ids
        if latents.shape[0] == 1:
            # Captured into the denoise CUDA graph.
            cond_v, uncond_v = self.transformer.denoise_step_batched_cfg(
                latents[0], vision_timesteps[0], (cos_cond[0], sin_cond[0]), (cos_uncond[0], sin_uncond[0]),
                layout["vision_token_shapes"], layout["vision_noisy_frame_indexes"],
                layout["mse_gen_indexes"], noisy_token_mask=noisy_token_mask[0],
            )
            return {rids[0]: {"cond_v": [cond_v], "uncond_v": [uncond_v]}}
        reqs = [
            self.transformer.denoise_step_row(
                layout,
                latents[i],
                vision_timesteps[i],
                (cos_cond[i], sin_cond[i]),
                (cos_uncond[i], sin_uncond[i]),
                noisy_token_mask=noisy_token_mask[i],
            )
            for i in range(latents.shape[0])
        ]
        results = self.transformer.denoise_step_batched(reqs)
        return {
            rid: {"cond_v": [cond_v], "uncond_v": [uncond_v]}
            for rid, (cond_v, uncond_v) in zip(rids, results, strict=True)
        }

    def postprocess(self, request_id, request_info, outputs, inputs=None, **kwargs):
        """Captured-path tail: the classifier-free-guidance combine and the
        (Python, multistep) scheduler step the graph can't hold, finished from
        the step's own ``inputs``. Mirrors the tail of ``_forward_image_gen``.
        Eager forwards already emit finished ``latents`` (no ``cond_v`` in their
        outputs). A windowed rollout's last window keeps the session it names."""
        if request_info.graph_walk not in GEN_WALKS:
            return
        st = inputs.kwargs["settings"]
        if "cond_v" in outputs:
            self._finish_captured(st, request_info, outputs, inputs)
        if st.get("session_id") and "window_latents" in outputs:
            window_index = self._window_step(st, _loop_step(request_info, request_info.graph_walk))[0]
            if window_index + 1 == st["ar_schedule"].num_windows:
                self._keep_session(st, outputs["latents"][0], outputs[SESSION_TAIL][0])

    def _finish_captured(self, st, request_info, outputs, inputs) -> None:
        latents = inputs.tensor_inputs["latents"]
        # in range: the request's step count is the loop's cap (Cosmos3Model._denoise_cap)
        step_index = _loop_step(request_info, request_info.graph_walk)
        windowed = "ar_schedule" in st
        if windowed:
            # Chained windows only (kv requests never take a lease); the loop
            # counter is global, the schedule index is the within-window step.
            step_index = self._window_step(st, step_index)[1]
        # the graph runs both branches; a step outside the guidance_interval keeps the conditional one
        if self._cfg_active(st, step_index):
            velocity = _guided(outputs["cond_v"][0], outputs["uncond_v"][0], st["gs"], latents.dtype)
        else:
            velocity = outputs["cond_v"][0].to(latents.dtype)
        solver = {name: inputs.tensor_inputs[name] for name in self._solver_state}
        outputs.clear()
        # The graph predicts every frame; clean frames (i2v anchor, chained
        # overlap) get the zero velocity the eager unpatchify gives them.
        velocity = velocity * st["cond"]["noisy_frame_mask"].to(velocity.dtype)
        new_latents, solver = self._scheduler_step(st, velocity, latents, step_index, solver)
        if st.get("vmask") is not None:
            new_latents = (1.0 - st["vmask"]) * new_latents + st["vmask"] * st["cond_video_latents"]
        if windowed and step_index + 1 == st["ar_steps"]:
            outputs.update(self._end_window(st, new_latents, solver))
        else:
            outputs.update(self._step_outputs(st, new_latents, solver))



class Cosmos3VAEEncoderSubmodule(NodeSubmodule):
    """Wan VAE conditioning-encode node (STATELESS): the request's conditioning
    image or video -> clean anchor latents for the denoise loop.

    Runs in parallel with the DiT's understanding-tower prefill (the conditioned
    prefill walks put the two nodes in a ``Parallel`` section) and emits
    ``cond_latents`` as a persist signal, which the generation walk's loop hands
    to every iteration. Everything is re-derived from the request
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
        # Dedicated instance (not shared with the decoder node): the encode
        # runs in this VAE's own dtype (bf16, see Cosmos3Model._build_encode_vae)
        # while the decode dtype is cuDNN-gated, and sharing would re-cast the
        # full VAE weights every encode/decode interleave.
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
            # deployment's conditioning_resize recipe (components/conditioning.py).
            # Action modes keep diffusers' VideoProcessor resize, since their
            # reference recipes (resolution buckets and padding in diffusers)
            # are not ported.
            if is_action:
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
        # In the VAE's own dtype, like the reference pipeline's encode and latent
        # normalization: bf16 as built and as the engine casts this node. An
        # fp32 encode moves the anchor latent enough to move the whole i2v
        # trajectory.
        dtype = next(vae.parameters()).dtype
        device = vision.device
        mean, inv_std = latent_affine(vae, dtype, device)
        with torch.autocast(device_type=device.type, enabled=False):
            raw_mu = vae.encode(vision.to(dtype)).latent_dist.mode()
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


def _wan_vae():
    # The Wan VAE helpers import diffusers, an optional dependency; only the
    # decode paths load them, so this module imports without it.
    from mstar.model.components import wan_vae

    return wan_vae


class ChainedWindowHead:
    """The clean head of a window that continues earlier frames: a chained
    window's overlap, or a resumed session's first window.

    The checkpoint was trained on clips whose first latent frame encodes one
    pixel frame alone and every later one the four frames after it, so a
    finished window's tail latents (mid-stream latents) are not what a
    window's first positions expect; pinned there, the rollout washes out
    within a few windows. Like the reference's video-to-video conditioning,
    the head is the last ``1 + (units - 1) * tf`` frames, decoded to 8 bits
    and encoded again as a clip of their own. The decode is the whole-clip
    decode the ``vae_decoder`` node runs (compiled, channels-last, the
    cuDNN-gated dtype, the frames quantized as served), behind
    ``chained_head_context_latents`` latents of the same window so the causal
    decoder is warm at the kept frames; the encode runs in the encode VAE's
    own dtype (bf16), as the encoder node's and the reference's
    video-to-video encode do. Both VAEs are the model's shared copies when
    the nodes share a process. Held outside the module tree, so the engine's
    casts of the DiT never reach them.
    """

    def __init__(self, decode_vae, encode_vae, config):
        self.decoder = Cosmos3VAEDecoderSubmodule(decode_vae, config, channels_last=config.vae_decode_channels_last)
        self.encode_vae = encode_vae
        self.config = config

    def __call__(self, window_latents: torch.Tensor, units: int) -> torch.Tensor:
        # at least one: a clip's latent 0 is a clip start, a mid-stream latent is not
        ctx = max(1, int(self.config.chained_head_context_latents))
        z = window_latents[:, :, -min(window_latents.shape[2], units + ctx):]
        frames = 1 + (units - 1) * self.config.vae.scale_factor_temporal
        pixels = self.decoder._decode_pixels(z)[:, :, -frames:]
        vae = self.encode_vae
        # read, never re-cast: the instance is the encoder node's when they share a process
        dtype = next(vae.parameters()).dtype
        mean, inv_std = latent_affine(vae, dtype, z.device)
        with torch.autocast(device_type=z.device.type, enabled=False):
            raw_mu = vae.encode((pixels.to(torch.float32) / 127.5 - 1).to(dtype)).latent_dist.mode()
        return ((raw_mu - mean) * inv_std).to(window_latents.dtype)


class Cosmos3VAEDecoderSubmodule(NodeSubmodule):
    """Wan VAE decode node: final denoised latents -> pixel frames.

    Applies the pipeline-side latent normalization (the VAE itself returns raw
    latents) before decoding, matching the fused t2i pipeline's decode.
    """

    # One-shot decode per request; CUDA-graph capture (not torch.compile) is the
    # speedup path.
    disable_torch_compile = True
    def __init__(self, vae, config, channels_last: bool = False, fast: bool = False):
        super().__init__()
        self.vae = vae
        self.config = config
        self._decode_dtype_cached = None
        # ``fast`` (config.vae_decode_fast, whole-clip node only): clips decode
        # through the shared Wan VAE decoder, always channels-last.
        fast = vae is not None and fast
        if fast:
            # Cast before the layout switch and the compile trace, so both see
            # the weights the decode runs with.
            self.vae = vae = vae.to(self._decode_dtype())
        # Channels-last weights and activations (cuDNN's native conv3d layout),
        # converted in place before any compile traces the decode; only the
        # instance given here is converted (see Cosmos3Model).
        self._channels_last = vae is not None and (channels_last or fast)
        if self._channels_last:
            _wan_vae().to_channels_last(vae)
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
            # with ``fast``, only images (one latent) decode through this
            self._decode = _wan_vae().compile_decode(vae.decode)
            self._decode_compiled = True
            logger.info("Cosmos3 VAE decode torch.compile enabled")
        # Resolve + log the decode dtype now (a cheap cuDNN-version read) so the
        # choice is fixed at startup, not on the first request.
        self._decode_dtype()
        self._fast = None
        if fast:
            self._fast = _wan_vae().FastDecoder(vae, compile=self._decode_compiled)
            logger.info("Cosmos3 VAE decode: video clips on the shared Wan VAE decoder (channels-last), "
                        "images through the stock decode")

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={"latents": inputs["latents"][0]})

    def _decode_dtype(self):
        if self._decode_dtype_cached is not None:
            return self._decode_dtype_cached
        self._decode_dtype_cached = wan_vae_decode_dtype()
        logger.info("Cosmos3 VAE decode dtype = %s (cuDNN %s)",
                    self._decode_dtype_cached, torch.backends.cudnn.version())
        return self._decode_dtype_cached

    def _vae_input(self, latents: torch.Tensor) -> torch.Tensor:
        """DiT latents -> the VAE's raw latents ``z``, in the decode dtype."""
        vae = self.vae
        vae_dtype = self._decode_dtype()
        if next(vae.parameters()).dtype != vae_dtype:
            vae = vae.to(vae_dtype)
        mean, inv_std = latent_affine(vae, vae_dtype, latents.device)
        # Force z to the VAE dtype: the engine's outer autocast promotes the
        # `1.0 / std` division to fp32, and a fp32 z into the autocast-off bf16
        # conv3d fails ("Input type (float) and bias type (BFloat16)"). No-op
        # on the fp32-decode path.
        z = (latents.to(vae_dtype) / inv_std + mean).to(vae_dtype)
        if self._channels_last:
            # Also the layout the compiled decode's warm-up traces with.
            z = z.contiguous(memory_format=torch.channels_last_3d)
        return z

    @staticmethod
    def _to_uint8(decoded: torch.Tensor) -> torch.Tensor:
        # Quantize to 8-bit here (the output is an 8-bit image/mp4 either way) so
        # only the uint8 frames cross the SHM edge to the data worker, not a 4x
        # larger fp32 tensor — the decoded video transfer dominates the fixed cost
        # at higher resolutions. Contiguous whatever layout the decoder ended in.
        # fp32 and rounded like diffusers' postprocess (bf16 math and truncation cost ~1 LSB)
        return (decoded.float() / 2 + 0.5).clamp(0, 1).mul(255).round().to(torch.uint8).contiguous()

    def _decode_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> uint8 pixel frames ``[1, 3, T, H, W]``."""
        vae_dtype = self._decode_dtype()
        z = self._vae_input(latents)
        # one latent (an image) is faster through the stock decode (24 vs 36 ms on an H200)
        if self._fast is not None and z.shape[2] > 1:
            return self._to_uint8(self._fast.decode(z))
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
        return self._to_uint8(decoded)

    @staticmethod
    def _video_frames(pixels: torch.Tensor) -> torch.Tensor:
        """``[1, 3, T, H, W]`` pixels -> ``[T, 3, H, W]`` frames for ``video_output``:
        the layout the API server's encoder takes, so the transpose runs here on
        the GPU and not as a host copy of the whole clip before the encode."""
        return pixels[0].transpose(0, 1).contiguous()

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, latents, **kwargs):
        pixels = self._decode_pixels(latents)
        # image_gen emits "image_output" (one frame); the video walks (plain,
        # sound, forward-dynamics) emit "video_output".
        if graph_walk in (VIDEO_GEN_WALK, VIDEO_SOUND_GEN_WALK, ACTION_VIDEO_GEN_WALK):
            return {"video_output": [self._video_frames(pixels)]}
        return {"image_output": [pixels]}


# The streaming decoder's loop over a request's windows, and its loop-back edges: the latents behind the next
# window's first frame, the frames sent to the client so far (a host count), the frames a request that doesn't
# stream has decoded so far, and the VAE decoder's cache where the warm cache is on.
VIDEO_DECODE_AR_LOOP = "video_decode_ar_loop"
DECODE_TAIL, DECODE_SENT, DECODE_FRAMES, DECODE_CACHE = "decode_tail", "decode_sent", "decode_frames", "decode_cache"


class Cosmos3VAEDecoderARSubmodule(Cosmos3VAEDecoderSubmodule):
    """Streaming Wan VAE decode for windowed AR video (ported from #198).

    Runs once per committed window, in a loop over the request's window
    stream (``VIDEO_DECODE_AR_LOOP``); the overlap a window re-pins is
    trimmed, and the video is either emitted per window (``stream_video``)
    or assembled on a loop-back edge and emitted once the last window lands.

    With ``windowed_decode_warm_cache`` the VAE decoder's causal conv cache
    rides a loop-back edge (``DECODE_CACHE``), so each window decodes only
    its new latents, continuing the cache the previous window left: the
    frames are those of one decode of the whole stream. A stream's first
    window (fresh or resumed) is decoded behind a re-decoded left context
    (the last few latents of the resumed session, which the DiT sends ahead of
    the window's own), so the conv stack is warm at the kept frames, and the
    context-derived pixels are trimmed; that decode leaves the cache the
    stream continues from. With the warm cache off, every window takes that
    stateless re-decode. Sessions keep only the latent context, never the
    cache.

    The stream's terminal flush (an empty pass, and the only pass a
    non-windowed request's idle stream ever delivers) is declined, which
    ends the loop.
    """

    def __init__(self, vae, config):
        super().__init__(vae, config, channels_last=config.vae_decode_channels_last)
        self._warm_enabled = vae is not None and bool(config.windowed_decode_warm_cache)
        # the warm cache's decoder; the stateless re-decode never runs it
        self._stream = None
        if self._warm_enabled:
            self._stream = _wan_vae().FastDecoder(
                vae, channels_last=self._channels_last, compile=self._decode_compiled, deterministic=False,
            )

    def step_order(self, graph_walk: str) -> StepOrder:
        """The decoder feeds the DiT's chained windows to the client, so a
        session's stream keeps moving while others run."""
        del graph_walk
        return StepOrder.TURNS

    def prepare_inputs(
        self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs,
    ) -> NodeInputs | None:
        chunks = (inputs or {}).get("window_latents") or []
        if not chunks:
            assert is_final_stream_chunk, "an empty chunk before the end of the stream"
            return None
        return NodeInputs(
            tensor_inputs={"latents": chunks[0]},
            kwargs={
                # no count until a step of this partition has run, and the first window is its first
                "window": int(fwd_info.dynamic_loop_iter_counts.get(VIDEO_DECODE_AR_LOOP, 0)),
                "edges": {
                    name: inputs[name] for name in (DECODE_TAIL, DECODE_SENT, DECODE_FRAMES, DECODE_CACHE)
                    if name in inputs
                },
            },
        )

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, latents, window, edges, **kwargs):
        md = engine_inputs.per_request_info[engine_inputs.request_ids[0]].step_metadata
        if window == 0:
            # A resumed session: window 0's pinned head duplicates frames the
            # client already has, so it is trimmed like an overlap, and the
            # session's decode context, which the DiT sends ahead of the window's
            # latents, warms the conv stack behind the first new frame.
            overlap = int(md.get("resume_latent_units", 0) or 0)
            context = 0
            if overlap:
                context = latents.shape[2] - min(int(md["window_latent_units"]), int(md["total_latent_units"]))
            tail, latents = (latents[:, :, :context], latents[:, :, context:]) if context else (None, latents)
            cache, sent, frames = None, 0, None
        else:
            overlap = int(md["overlap_latent_units"])
            (tail,), (frames,) = edges[DECODE_TAIL], edges[DECODE_FRAMES]
            cache = _wan_vae().decoder_cache(self.vae, edges[DECODE_CACHE]) if self._warm_enabled else None
            sent = int(edges[DECODE_SENT][0])
        new = latents[:, :, overlap:] if overlap else latents
        # The tail is already capped at the context length, so appending the
        # window's new latents yields the [context | window] decode input and
        # the next tail in one tensor.
        stream_tail = new if tail is None else torch.cat([tail, new.to(tail.dtype)], dim=2)
        if cache is not None:
            pixels = self._decode_stream(new, cache, clip_start=False)
        else:
            if self._warm_enabled:
                cache = _wan_vae().new_feat_cache(self.vae)
                pixels = self._decode_stream(stream_tail, cache, clip_start=True)
            else:
                pixels = self._decode_pixels(stream_tail)
            if tail is not None:
                # A mid-stream latent decodes to scale_factor_temporal frames; the
                # leading context (and the clip-start special frame, which falls
                # inside it) is exactly the part being trimmed.
                keep = new.shape[2] * self.config.vae.scale_factor_temporal
                pixels = pixels[:, :, -keep:]
        tail = stream_tail[:, :, -max(1, int(self.config.windowed_decode_context_latents)):]
        last = window + 1 >= int(md["num_windows"])
        # The schedule may be padded up to whole windows; the request's frame
        # count caps what the client gets.
        chunk = pixels[:, :, : int(md["num_frames"]) - sent]
        outputs = {DECODE_TAIL: [tail], DECODE_SENT: [torch.tensor([sent + chunk.shape[2]])]}
        if self._warm_enabled:
            outputs[DECODE_CACHE] = _wan_vae().to_edge(cache, latents.device)
        if md.get("stream_video"):
            # each window goes to the client as its own chunk, and nothing is kept across windows
            outputs[DECODE_FRAMES] = [chunk[:, :, :0]]
            if chunk.shape[2]:
                outputs["video_output"] = [self._video_frames(chunk)]
            return outputs
        frames = chunk if frames is None else torch.cat([frames, chunk], dim=2)
        outputs[DECODE_FRAMES] = [frames]
        if last:
            outputs["video_output"] = [self._video_frames(frames)]
        return outputs

    def _decode_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.config.vae_decode_cudnn_benchmark:
            return super()._decode_pixels(latents)
        with cudnn_benchmark(True):
            return super()._decode_pixels(latents)

    def _decode_stream(self, latents: torch.Tensor, feat_cache: list, clip_start: bool) -> torch.Tensor:
        """``_decode_pixels`` continuing the stream whose decoder cache is
        ``feat_cache`` (updated in place)."""
        z = self._vae_input(latents)
        # Only steady windows group latents per call: a stream's first window
        # (the clip start, or a resumed session's context ahead of its new
        # latents) runs one latent per call, so its peak stays the default's.
        per_call = 1 if clip_start else int(self.config.windowed_decode_latents_per_call)
        with cudnn_benchmark(self.config.vae_decode_cudnn_benchmark):
            decoded = self._stream.decode_stream(z, feat_cache, clip_start=clip_start, latents_per_call=per_call)
        return self._to_uint8(decoded)


class _VisionNodeConfig(NamedTuple):
    """What `PackedBlockLoopEncoder` reads of its config."""

    hidden_size: int
    head_dim: int
    spatial_merge_size: int


class Cosmos3VisionEncoderSubmodule(PackedBlockLoopEncoder):
    """The Edge reasoner's vision tower + projector (STATELESS): a step's
    requests' packed image / video patches -> one text-space token per merged
    2x2 block, in prompt order, for the reasoner's media walk to splice in at
    its pads.

    The pixel patches are computed CPU-side in ``Cosmos3Model.process_prompt``
    and the grids ride the walk's ``MEDIA_LAYOUT`` (one run per image or video
    frame), so no device tensor is read for them. Each frame attends to
    itself alone, a segment on the ``vision_tokens`` array, and the encoder
    blocks replay as one captured region per patch bucket, a step's requests
    together. A request too large for a replay runs the blocks eagerly, in a
    step of its own (`cg_key_info`).
    """

    # A 512x341 image is ~700 patches; a step past the top bucket runs the
    # block loop eagerly.
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    # A burst of image requests shares one replay and pays the per-step host
    # cost once per this many.
    MAX_BATCH_SIZE = 8
    # Per-request shapes vary; the block loop's capture is the accelerator.
    disable_torch_compile = True
    # The tower is bf16 already. Under autocast each LayerNorm upcasts its input
    # and its fp32 output is cast back once per projection reading it, ~18% of
    # the block loop's GPU time for the same rounding (the norm accumulates in
    # fp32 either way).
    disable_autocast = True
    BLOCK_LOOP_DTYPE = torch.bfloat16
    # Resized position tables kept on the device within a byte budget: one is
    # h*w x hidden, ~151 MB at the processor's largest frame, so a count alone
    # could hold gigabytes.
    POSITION_TABLE_BYTES = 256 << 20

    def __init__(self, vision_model, config):
        vision = config.reasoner.vision if config.reasoner is not None else None
        node_config = None if vision is None else _VisionNodeConfig(
            vision.hidden_size, vision.hidden_size // vision.num_attention_heads, vision.spatial_merge_size,
        )
        super().__init__(vision_model, node_config, output="vision_embeds")
        self.model_config = config
        self._position_tables = _Memo(self.POSITION_TABLE_BYTES, largest=self.POSITION_TABLE_BYTES // 8)
        self.vision = None

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.vision = handles[VISION_TOKENS]

    def _declare(
        self, rows: Sequence[tuple[Any, Mapping[str, Sequence[int]]]], attention: bool = True,
    ) -> SubmoduleStep:
        state = {VISION_TOKENS: ArrayStep()} if attention else {}
        return SubmoduleStep(segments=super()._declare(rows).segments, state=state)

    def max_segments_per_step(self) -> int:
        return kernel_setting(self.vision, VISION_ATTN, "max_segments_per_step")

    def _position_table(self, h: int, w: int) -> torch.Tensor:
        """The frame's resized position grid, built on the device once per (h, w)."""
        return self._position_tables.get((h, w), lambda: self.model.position_table(h, w))

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        frames = [(h, w) for t, h, w in self.grids(fwd_info) for _ in range(t)]
        pixel_values = inputs["pixel_values"][0]
        patches = sum(h * w for h, w in frames)
        if pixel_values.shape[0] != patches:
            raise ValueError(f"{pixel_values.shape[0]} pixel patches for the layout's {patches}")
        return NodeInputs(
            # what the engine leases the captured block loop's bucket by
            input_seq_len=patches,
            tensor_inputs={
                "pixel_values": pixel_values,
                "pos_embeds": torch.cat([self._position_table(h, w) for h, w in frames]),
            },
            # one segment per frame: a frame attends to itself alone
            resource_step_info={"main": [h * w for h, w in frames]},
        )

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        return True

    def cg_key_info(self, graph_walk: str, per_request_info: dict, **kwargs) -> Any:
        """Requests that each fit a replay pack together (None); one too large
        for a replay is a group of its own, so it doesn't hold the others'
        embeddings, and their prefills, until it is encoded."""
        def too_large(info: CurrentForwardPassInfo) -> bool:
            grid = self.grids(info)
            return not self._fits_replay(sum(t * h * w for t, h, w in grid), sum(t for t, _, _ in grid))

        return tuple(rid for rid, info in per_request_info.items() if too_large(info)) or None


class Cosmos3TextTower(torch.nn.Module):
    """The understanding tower of the DiT's transformer as the reasoner's
    decoder module (`SampledTokenDecoder`'s ``embed`` / ``forward`` /
    ``logits``): the same instance, so one copy of the text weights. The
    forward and the head run in the weights' bf16, autocast off, as the
    reference text tower does (see the DiT node's ``forward``)."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    @property
    def embed_dtype(self) -> torch.dtype:
        return self.transformer.embed_tokens.weight.dtype

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_tokens(input_ids)

    def build_cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotary tables for ``[3, N]`` mRoPE ids."""
        return self.transformer.build_cos_sin(position_ids, self.embed_dtype)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self, input_embeds: torch.Tensor, *, label: str, cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # the kv handle takes the step's label from its segments
        del label
        return self.transformer.text_layers(input_embeds, *cos_sin)

    @torch.autocast(device_type="cuda", enabled=False)
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.transformer.lm_head(hidden).float()


class Cosmos3ReasonerSubmodule(SampledTokenDecoder):
    """The understanding tower served as a causal VLM, on the VLM scaffold.

    Shares the DiT node's ``Cosmos3OmniTransformer`` instance and its ``kv``
    array, whose position counter is the reasoner's; declares its own sampler. A
    prompt with images or videos prefills as one row on ``prefill_vision``,
    the vision encoder's tokens spliced in at its pads. Positions are 3D mRoPE
    ids: a text prompt's and each decode step's are the array's positions
    on all three axes; a vision prompt places its own from the intake's
    layout (`prompt_positions`) and moves the counter to ``max(position) + 1``.
    """

    # Eager steps stay uncompiled; the captures compile their own forward.
    disable_torch_compile = True
    # Ladders trimmed from the scaffold's: every capture here is extra startup
    # and graph memory beside the DiT's, and one more Dynamo recompile of the
    # layer code the DiT prefill shares. A prompt with one 512x341 image is
    # ~220 tokens; a step past a ladder runs eagerly.
    # MSTAR_DECODE_BUCKETS narrows the decode ladder, as for every model. 24 and
    # 48 halve the padding of a step between the powers of two, which a busy
    # server's decode batches mostly are (+7% at c=32, +14% at c=64).
    DECODE_CAPTURE_BATCH_SIZES = [
        int(x) for x in os.environ.get("MSTAR_DECODE_BUCKETS", "1,2,4,8,16,24,32,48,64").split(",")
    ]
    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    MIXED_CAPTURE_BATCH_SIZES = [4, 16, 64]
    PREFILL_MEDIA_TOKEN_BUCKETS = [256, 512, 1024, 2048]
    # Image and video prompts join decode steps too, so a burst of them does
    # not stop every decode stream for a step each
    MIXED_MEDIA = True
    # A long prompt (a video, many images) runs in chunks of at most this many
    # tokens, a step's decode rows included, rather than as one step that
    # stops every decode stream for its length: the top of the prefill ladders.
    CHUNKS_MEDIA = True
    MAX_BATCH_TOKENS = 2048

    def __init__(self, transformer, config, encoders: Sequence[MediaEncoder] = ()):
        super().__init__(Cosmos3TextTower(transformer), config, encoders)

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[KV_CACHE]
        self.sampler = handles[SAMPLER]

    def node_step(
        self, segments: list[Segment], prefill_tokens: Mapping[int, torch.Tensor], advance: tuple[int, ...] | None,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=segments,
            state={KV_CACHE: ArrayStep(advance=advance), SAMPLER: SampleStep(prompt_tokens=prefill_tokens)},
        )

    def next_position(self, rid_handle: int) -> int:
        return position(self.kv, rid_handle, REASONER_LABEL)

    def loopback_tokens(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        return last_sampled(self.sampler)

    def last_token_hidden(self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor) -> torch.Tensor:
        return at_last_token(self.kv, hidden)

    def sample_tokens(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        return sample(self.sampler, logits)

    def capture_compile_mode(self, graph_walk: str) -> str | None:
        # Decode keeps the mode it was tuned with; prefills take the
        # deployment's level (mstar serve -O).
        return "max-autotune-no-cudagraphs" if graph_walk == self.DECODE_WALK else None

    def get_accelerator_graph_configs(self, device, tp_world_size: int = 1):
        # the deployment switch every Cosmos3 node's captures follow
        if not self.config.accelerator_graph or os.environ.get("COSMOS3_DISABLE_ACCELERATOR_GRAPH"):
            return []
        configs = super().get_accelerator_graph_configs(device, tp_world_size)
        if os.environ.get("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH"):
            # the understanding tower's prefill captures, as the DiT's
            configs = [c for c in configs if c.capture_graph_walk == self.DECODE_WALK]
        for c in configs:
            if c.capture_graph_walk == self.DECODE_WALK:
                # cuBLAS for the decode GEMMs, max-autotune for everything else.
                # Autotune's timings of these decode-width GEMMs swing ~50% between
                # processes on a host, and in its slow mode it picks Triton
                # templates that run ~2x slower in the graph than cuBLAS (o / up /
                # down projections at 8-32 rows, H200), baking a boot's luck into
                # every replay.
                c.inductor_options = check_inductor_options({"max_autotune_gemm_backends": "ATEN"})
        return configs

    # ------------------------------------------------------------------
    # mRoPE
    # ------------------------------------------------------------------

    def prompt_positions(self, start: int, layout: dict, num_tokens: int) -> tuple[torch.Tensor, int]:
        # built on the host: the scaffold moves them, and a chunk's cut rebuilds them
        return layout_position_ids(layout, self.config.reasoner.vision.spatial_merge_size, num_tokens, start)

    def resident_walks(self) -> frozenset[str]:
        # a captured decode step builds its rope inside the graph (`_rope_in_graph`),
        # so its `preprocess` reads nothing the tables fill
        return frozenset({self.DECODE_WALK})

    def _rope_in_graph(self, engine_inputs: ModelInputsFromEngine, graph_walk: str) -> bool:
        """A captured step of a text walk, whose rotary tables come from the
        planned positions inside the graph (as Qwen3.5's): always for decode,
        with ``ingraph_decode_rope`` for text prefill. Media rows bring their own
        3D positions, so a mixed step does too while media rows can join it
        (``MIXED_MEDIA``)."""
        if not engine_inputs.captured:
            return False
        text_walks = ("prefill_text",) if self._mixes_media else ("prefill_text", MIXED)
        return graph_walk == "decode" or (graph_walk in text_walks and ingraph_decode_rope())

    def preprocess(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]) -> dict:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        if self._rope_in_graph(engine_inputs, graph_walk):
            return out
        # the tables of the step's [3, tokens] positions: the array's on text and
        # decode rows, a vision prompt's own on its row
        out["cos"], out["sin"] = self.model.build_cos_sin(mrope_position_ids(positions(self.kv), inputs))
        return out

    def decoder_kwargs(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, num_tokens: int,
        cos: torch.Tensor | None = None, sin: torch.Tensor | None = None, **tensors: torch.Tensor,
    ) -> dict:
        if self._rope_in_graph(engine_inputs, graph_walk) or cos is None or sin is None:
            # one token a row, no custom grids: the same scalar on all three axes
            pos = positions(self.kv)
            assert pos is not None, "the positions have no plan for this step; `plan` must run before the forward"
            cos, sin = self.model.build_cos_sin(pos[:num_tokens].unsqueeze(0).expand(3, -1))
        return {"cos_sin": (cos, sin)}
