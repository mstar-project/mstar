"""Wan2.2-TI2V-5B node submodules (nothing here is cached across steps; the dit's
attention runs on the engine's ragged attention resources).

    text_encoder -> Wan22TextEncoderSubmodule (UMT5-XXL, thin HF wrapper)
    vae_encoder  -> Wan22VaeEncoderSubmodule  (Wan2.2-VAE encode, I2V only)
    dit          -> Wan22DenoiseSubmodule     (the DiT scaffold's loop: native 5B DiT + UniPC,
                                               batched, captured per served shape)
                    Wan22DitSubmodule         (the one-request loop, bit-exact; dit_loop: legacy)
    vae_decoder  -> Wan22VaeDecoderSubmodule  (Wan2.2-VAE decode)

All of them set ``disable_torch_compile``, so the engine never wraps their forwards: the
three wrapped nodes run once per request with no graph to amortize, and the legacy dit's
whole forward trips Inductor on the CPU-resident UniPC sigma. Instead the dit compiles its
inner transformer region alone (``config.compile_dit``) for the shapes in
``config.compile_dit_sizes``, leaving the solver and every other shape eager.

Numerics are governed by the checkpoint dtypes, not by the engine.
``Wan22Model.get_autocast_dtype`` returns None so the engine neither autocasts
nor blanket-casts the modules, and each forward enters ``_inference_ctx``
itself — reference parity must not depend on what context wraps the call, and
the v1 engine supplies neither the autocast nor the no-grad wrapper.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import (
    AttentionStep,
    RaggedCrossAttentionStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    cross_label,
)
from mstar.engine.resources.convenience import RaggedAttentionCallable
from mstar.model.components.diffusion.attention import RaggedAttentionFn
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule, StepScalar
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.wan22.components.unipc import (
    SOLVER_ORDER,
    UniPCCoefficients,
    UniPCState,
    make_unipc_tables,
    unipc_coefficient_step,
    unipc_coefficients,
    unipc_convert_model_output,
    unipc_corrector_step,
    unipc_effective_order,
    unipc_predictor_step,
)
from mstar.model.wan22.config import DIT_ATTN, DIT_XATTN, TEXT_SPAN, VIDEO_SPAN, Wan22Config

logger = logging.getLogger(__name__)

# Stop signals are registered by loop name, so the model and the stop logic below
# must agree on this literal.
DENOISE_LOOP_NAME = "denoise_loop"

# --- VAE-decode tiling gate calibration (Wan22VaeDecoderSubmodule) -----------
# Untiled AutoencoderKLWan.decode peak scales with output SPATIAL area H*W, nearly
# flat in frame count (the VAE decodes in causal temporal chunks). Fit:
# peak_GiB ~= (H*W / 1e6) * (SPATIAL + FRAME * num_frames), fp32 basis scaled by the
# live decode dtype's element size. Measured on RTX 5090, residual < 3% and biased
# high; see test/wan22 measurement.
_DECODE_PEAK_SPATIAL_GIB_PER_MPIX = 21.5   # per output spatial-megapixel, fp32
_DECODE_PEAK_FRAME_GIB_PER_MPIX = 0.013    # extra per frame, per spatial-Mpix, fp32
# Free VRAM must clear the estimate by this factor before decoding untiled.
_DECODE_UNTILED_SAFETY_MARGIN = 1.3
# Runtime override of config.vae_decode_tiling: "auto" | "tiled" | "untiled".
_DECODE_TILING_ENV = "WAN22_VAE_DECODE_TILING"
_DECODE_TILING_MODES = ("auto", "tiled", "untiled")


def normalize_decode_tiling_mode(raw: str) -> str:
    """Canonicalize a tiling-policy string to one of ``_DECODE_TILING_MODES``, else
    "auto". Shared by the decoder gate and the request config echo so both report the
    same resolved policy."""
    mode = (raw or "").lower()
    return mode if mode in _DECODE_TILING_MODES else "auto"


@contextmanager
def _inference_ctx():
    """The context every wan22 forward runs under: module dtypes, no grad.

    Autocast off because reference parity is governed by the checkpoint
    dtypes, not by whatever wraps the call. ``inference_mode`` because the
    v1 engine — unlike the stateless engine these nodes were first written
    against — does not wrap the eager forward in a grad-disabling context of
    its own, so a 5B DiT would otherwise build an autograd graph on every one
    of a request's denoise steps. ``inference_mode`` rather than ``no_grad``
    to match what the old engine gave this model (it picked ``inference_mode``
    for any node whose autocast dtype was None, which is all four of these).
    """
    with torch.amp.autocast("cuda", enabled=False), torch.inference_mode():
        yield


class _Fp32IslandMixin:
    """Re-pin the fp32 islands after a blanket ``.to(dtype=...)`` cast.

    ``EngineManager`` casts a submodule wholesale when its engine has autocast,
    which would drag the fp32 islands to bf16 and change every fp32-path result
    downstream. This restores the islands' dtype, not their values — a cast that
    actually fired has already rounded them. wan22 never casts
    (``get_autocast_dtype`` returns None), so this is a backstop against
    engine-config drift, not a license to serve under autocast.
    """

    def _record_fp32_islands(self):
        self._fp32_param_names = {n for n, p in self.named_parameters() if p.dtype == torch.float32}
        self._fp32_buffer_names = {n for n, b in self.named_buffers() if b.dtype == torch.float32}

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        for name, param in result.named_parameters():
            if name in self._fp32_param_names and param.dtype != torch.float32:
                param.data = param.data.float()
        for name, buf in result.named_buffers():
            if name in self._fp32_buffer_names and buf.dtype != torch.float32:
                _set_buffer(result, name, buf.float())
        return result


class _SingleRequestMixin:
    """Serve one request per step, through the engine's batched entry point.

    The v1 engine always dispatches to ``forward_batched`` — the worker builds
    every batch with ``running_batched=True``, and nothing flips it back — so a
    submodule that only defines ``forward`` never runs. None of the wan22 nodes
    batch: a request's latent grid is sized by its own height/width/num_frames,
    so two requests in one step would have to be padded to a common grid, and
    the DiT step is large enough that the padding would cost more than the
    batching saves. So they cap the batch at one and hand that row to
    ``forward``, rather than pretending to a batching they do not implement.

    ``max_batch_size`` is what actually keeps the batch at one: the micro
    scheduler reads it (``Engine.get_max_batch_size``) and chunks accordingly.
    The assert below is the backstop, not the mechanism.
    """

    def max_batch_size(self, graph_walk: str):
        return 1

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        request_ids = engine_inputs.request_ids
        assert len(request_ids) == 1, (
            f"{type(self).__name__} does not batch; got {len(request_ids)} "
            "requests in one step (max_batch_size should have capped it at 1)"
        )
        return {
            request_ids[0]: self.forward(
                graph_walk, engine_inputs=engine_inputs, **kwargs
            )
        }


def _set_buffer(module: nn.Module, dotted_name: str, value: torch.Tensor):
    owner_path, _, attr = dotted_name.rpartition(".")
    owner = module.get_submodule(owner_path) if owner_path else module
    setattr(owner, attr, value)


def _loop_step(fwd_info: CurrentForwardPassInfo, time_index: torch.Tensor) -> int:
    """The denoise iteration this step runs, from the engine's loop counter.

    ``time_index`` holds the same value, but it is the previous step's device
    output: reading it on the host waits for that step to finish, which stalls
    the speculative preparation of this one. Callers that drive the submodule
    without a loop counter (tests, harnesses) fall back to the edge.
    """
    counts = getattr(fwd_info, "dynamic_loop_iter_counts", None) or {}
    if DENOISE_LOOP_NAME in counts:
        return int(counts[DENOISE_LOOP_NAME])
    return int(time_index.reshape(-1)[0].item())


def _latent_grid(config: Wan22Config, step_metadata: dict) -> tuple[int, int, int]:
    """(T_lat, H_lat, W_lat) for the request's pixel-space dims (diffusers
    ``prepare_latents``: T=(F-1)//4+1, spatial /16)."""
    t_lat = (int(step_metadata["num_frames"]) - 1) // config.vae_scale_factor_temporal + 1
    h_lat = int(step_metadata["height"]) // config.vae_scale_factor_spatial
    w_lat = int(step_metadata["width"]) // config.vae_scale_factor_spatial
    return t_lat, h_lat, w_lat


class _GridCompiledDiT:
    """The DiT's forward compiled for the latent grids of ``config.compile_dit_sizes``.

    The inner DiT region only (tensors in, tensor out; nothing CPU-resident crosses
    it), compiled IN PLACE so the transformer keeps its identity: fp32 islands,
    ``.dtype``, the tests' transformer swap. ``dynamic=False`` traces one graph per
    input shape (batch included, so CFG and no-CFG are distinct graphs);
    ``fullgraph=False`` allows the attention graph breaks. Every other grid runs the
    transformer's own eager forward, so a request at an unlisted size never costs a
    compile on the shared worker.
    """

    def __init__(self, transformer: nn.Module | None, config: Wan22Config):
        self.enabled = bool(config.compile_dit)
        self.grids = config.compile_dit_grids()
        self._shapes: set[tuple[int, ...]] = set()
        self._eager = None
        if self.enabled and transformer is not None:
            # with the transformer it belongs to: one swapped in later runs its own forward
            self._eager = (transformer, transformer.forward)
            transformer.forward = torch.compile(transformer.forward, fullgraph=False, dynamic=False)
            logger.info(
                "Wan2.2 DiT: torch.compile enabled on the transformer region "
                "(fullgraph=False, dynamic=False) for latent (frames, height, width) %s; other "
                "shapes and the UniPC solver run eager.",
                sorted(self.grids),
            )

    def forward_for(self, transformer: nn.Module, hidden_states: torch.Tensor):
        """The callable to run ``hidden_states`` (``[B, C, F, H, W]``) through."""
        if self.enabled and tuple(hidden_states.shape[2:]) in self.grids:
            shape = tuple(hidden_states.shape)
            if shape not in self._shapes:
                # so an operator watching the log knows why the first request at a new
                # resolution is slow
                self._shapes.add(shape)
                logger.info(
                    "Wan2.2 DiT: first forward at shape %s — torch.compile is tracing "
                    "this resolution now (one-time pause); later requests reuse it.",
                    shape,
                )
            return transformer
        if self._eager is not None and self._eager[0] is transformer:
            return self._eager[1]
        return transformer


class Wan22StepInfo(NamedTuple):
    """What one request's denoise step attends over: the video tokens of each
    guidance branch, the text tokens each branch's cross-attention reads, and
    whether both branches run (CFG). Built in ``prepare_inputs`` from host facts,
    read back by ``declare_step``."""

    video_tokens: int
    text_tokens: int
    cfg: bool


# ---------------------------------------------------------------------------
# text_encoder
# ---------------------------------------------------------------------------

class Wan22TextEncoderSubmodule(_SingleRequestMixin, _Fp32IslandMixin, NodeSubmodule):
    """UMT5-XXL prompt encoder.

    Consumes ``text_inputs`` = ``[positive_ids, negative_ids]``; emits
    ``text_embeds_pos`` / ``text_embeds_neg``, each ``[1, text_max_seq_len,
    text_dim]``, zero-padded past the true sequence length. The two prompts run as
    two batch-1 forwards, as the reference's separate ``encode_prompt`` calls do.
    With CFG off (``guidance_scale <= 1.0``) the negative prompt is not encoded
    at all: the dit never reads it, so ``text_embeds_neg`` is a 1-element
    placeholder that only keeps the persisted edge populated.
    """

    disable_torch_compile = True

    def __init__(self, text_encoder: nn.Module, config: Wan22Config):
        super().__init__()
        self.text_encoder = text_encoder
        self.config = config
        self._record_fp32_islands()

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        tensor_inputs = {"positive_ids": inputs["text_inputs"][0]}
        # CFG off: skip the negative prompt's UMT5 forward entirely (see the
        # class docstring); forward emits the placeholder when the key is absent.
        if float(fwd_info.step_metadata["guidance_scale"]) > 1.0:
            tensor_inputs["negative_ids"] = inputs["text_inputs"][1]
        return NodeInputs(tensor_inputs=tensor_inputs)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        with _inference_ctx():
            pos = self._encode_one(positive_ids)
            neg = self._encode_one(negative_ids) if negative_ids is not None else pos.new_zeros(1)
            return {
                "text_embeds_pos": [pos],
                "text_embeds_neg": [neg],
            }

    def _encode_one(self, ids: torch.Tensor) -> torch.Tensor:
        device = self.get_device()
        max_len = self.config.text_max_seq_len
        ids = ids.to(device=device, dtype=torch.long)[:max_len]
        seq_len = ids.shape[0]
        # T5's pad token id is 0, so zero-fill == tokenizer(padding="max_length").
        padded = torch.zeros(1, max_len, dtype=torch.long, device=device)
        padded[0, :seq_len] = ids
        mask = torch.zeros(1, max_len, dtype=torch.long, device=device)
        mask[0, :seq_len] = 1
        embeds = self.text_encoder(padded, mask).last_hidden_state
        # Zero-pad to text_max_seq_len because the reference's encode_prompt
        # does: the DiT cross-attends the full fixed-length text stream with NO
        # mask, so the zero rows participate in attention and the equivalence
        # gates pin their exact values. (Reference keeps u[:seq_len] and re-pads
        # with zeros; in-place zeroing of the tail is the same tensor.)
        embeds[:, seq_len:] = 0
        return embeds


# ---------------------------------------------------------------------------
# vae_encoder
# ---------------------------------------------------------------------------

class Wan22VaeEncoderSubmodule(_SingleRequestMixin, _Fp32IslandMixin, NodeSubmodule):
    """Wan2.2-VAE first-frame encoder (I2V requests only).

    Consumes ``image_inputs`` (one ``[C, H, W]`` image, float in ``[0, 1]`` or
    uint8, already at the request's height/width — the server does not resize, and
    a mismatch is an error). Emits ``image_latent`` ``[1, vae_z_dim, 1, H/16,
    W/16]``: the mode sample of the encoded frame, normalized with the checkpoint's
    per-channel statistics in float32. The dit node injects it at frame 0 and
    rebuilds the first-frame mask itself, since the mask follows from the latent
    dims and does not need persisting.
    """

    disable_torch_compile = True

    def __init__(self, vae: nn.Module, config: Wan22Config):
        super().__init__()
        self.vae = vae
        self.config = config
        self._record_fp32_islands()

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        # The image's size is checked at the request seam (Wan22Model.process_prompt),
        # so a mismatch is a 400 rather than a fault here on the compute worker.
        image = inputs["image_inputs"][0]
        # uint8 -> float [0, 1] here rather than in forward: prepare_inputs is
        # the per-request preprocessing seam, so a future batched forward sees
        # dtype-uniform inputs.
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return NodeInputs(tensor_inputs={"image": image})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        image: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        with _inference_ctx():
            device = self.get_device()
            # VideoProcessor.preprocess normalizes to [-1, 1].
            video = (image.float() * 2.0 - 1.0).to(device=device, dtype=self.vae.dtype)
            video = video[None, :, None]  # [1, C, 1, H, W]
            latent = self.vae.encode(video).latent_dist.mode()
            # prepare_latents: cast to fp32, then (latent - mean) * (1 / std)
            # with mean / reciprocal-std built at the latent dtype.
            latent = latent.to(torch.float32)
            z_dim = self.config.vae_z_dim
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(
                latent.device, latent.dtype
            )
            latent = (latent - latents_mean) * latents_std
        return {"image_latent": [latent]}


# ---------------------------------------------------------------------------
# dit
# ---------------------------------------------------------------------------

class Wan22DitSubmodule(_SingleRequestMixin, _Fp32IslandMixin, NodeSubmodule):
    """Dense 5B video DiT with the UniPC step run inline.

    One loop iteration is one batch-2 transformer forward (the positive and
    negative prompt embeddings stacked on the batch axis) plus one UniPC
    predictor/corrector update from ``components/unipc.py``.

    Loop-carried edges, all float32 over the latent grid:

        latents              current sample x_t
        time_index           [1] int64, the 0-based step index k
        unipc_model_outputs  ring buffer of the last two converted outputs;
                             see ``UniPCState``
        unipc_last_sample    the sample the previous predictor was given

    Nothing else is carried. The sigmas, the order ramp and the timestep pair all
    follow from ``time_index`` and the per-request tables, so they are recomputed
    rather than shipped across an edge.

    At iteration 0 the conductor sends the four edges empty and this submodule
    seeds them (noise from the request's seed, on a CPU generator as diffusers
    does). I2V also consumes the persisted ``image_latent``: frame 0 of the input
    is replaced by it and its per-token timestep zeroed, and the conditioning is
    written back into the output latents only on the final iteration, matching the
    reference's post-loop injection.

    With ``use_ragged_attention`` the transformer's attention runs on the engine's
    ragged resources (``DIT_ATTN`` over each guidance branch's video tokens,
    ``DIT_XATTN`` over the text): ``prepare_inputs`` stamps the step's spans on
    ``NodeInputs.resource_step_info`` and ``declare_step`` declares them, one
    segment per branch in the order of the batch-2 forward, so the engine plans
    the layouts before the forward. Otherwise the transformer runs SDPA.
    """

    # Mirrors Wan22Model.VIDEO_GEN_I2V_WALK; importing the model here would be a
    # circular import.
    _I2V_WALK = "video_gen_i2v"

    # The WHOLE forward cannot be one compiled region: Inductor would fuse the UniPC
    # step's `sample - sigma * output` and pass the 0-dim CPU-resident sigma as a
    # device pointer, which fails on the first call. Moving sigma to CUDA would fix
    # the launch but break bit-exactness against the reference (components/unipc.py).
    # So the engine wrap stays OFF; instead the inner transformer region alone
    # (patchify -> blocks -> head) is compiled in __init__ (config.compile_dit),
    # leaving the solver, CFG combine, and everything touching sigma eager.
    disable_torch_compile = True

    def __init__(self, transformer: nn.Module, config: Wan22Config, use_ragged_attention: bool = False):
        super().__init__()
        self.transformer = transformer
        self.config = config
        self._use_ragged_attention = bool(use_ragged_attention)
        # one callable per attention kind for the life of a resource binding: the
        # compiled region guards on their identity
        self._ragged_fns: tuple[RaggedAttentionFn, RaggedAttentionFn] | None = None
        self._record_fp32_islands()
        self._compiled = _GridCompiledDiT(transformer, config)

    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._ragged_fns = None  # the memoized callables hold the old resources

    def _ragged_attention(self) -> tuple[RaggedAttentionFn, RaggedAttentionFn] | None:
        """The self- and cross-attention callables over the step's packed rows,
        or None when the transformer runs SDPA (no resource bound)."""
        if not self._use_ragged_attention:
            return None
        if self._ragged_fns is None:
            self_attn = self.node_resources.get(DIT_ATTN)
            cross_attn = self.node_resources.get(DIT_XATTN)
            if self_attn is None or cross_attn is None:
                return None
            self._ragged_fns = (
                RaggedAttentionCallable(self_attn, VIDEO_SPAN),
                RaggedAttentionCallable(cross_attn, cross_label(VIDEO_SPAN, TEXT_SPAN)),
            )
        return self._ragged_fns

    def _step_info(self, tensor_inputs: dict[str, torch.Tensor], step_metadata: dict) -> Wan22StepInfo:
        """The spans this step's attention declares, from host facts: the latent
        grid's token count (the patch over ``[1, C, T, H, W]``), the text length,
        and whether CFG runs both branches (as ``_noise_prediction`` decides)."""
        _, _, t_lat, h_lat, w_lat = tensor_inputs["latents"].shape
        p_t, p_h, p_w = self.config.patch_size
        guidance_scale = float(step_metadata.get("guidance_scale", self.config.guidance_scale))
        return Wan22StepInfo(
            video_tokens=(t_lat // p_t) * (h_lat // p_h) * (w_lat // p_w),
            text_tokens=int(tensor_inputs["text_embeds_pos"].shape[1]),
            cfg=guidance_scale > 1.0,
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases=None,
        **kwargs,
    ) -> SubmoduleStep | None:
        """The ragged resources' step: one video segment per guidance branch, in
        the row order of the CFG forward (conditional first), and the text span
        each branch's cross-attention reads."""
        if not self._use_ragged_attention:
            return None
        video, text = [], []
        for rid, inp in zip(request_ids, inputs, strict=True):
            info: Wan22StepInfo = inp.resource_step_info
            for _ in range(2 if info.cfg else 1):
                video.append(Segment(rid, VIDEO_SPAN, info.video_tokens))
                text.append(Segment(rid, TEXT_SPAN, info.text_tokens))
        return SubmoduleStep(
            steps={
                DIT_ATTN: AttentionStep(segments=tuple(video), causal=False),
                DIT_XATTN: RaggedCrossAttentionStep(
                    segments=tuple(video + text), pairs=((VIDEO_SPAN, TEXT_SPAN),),
                ),
            },
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs | None:
        tensor_inputs = {
            "text_embeds_pos": inputs["text_embeds_pos"][0],
            "text_embeds_neg": inputs["text_embeds_neg"][0],
        }
        if graph_walk == self._I2V_WALK:
            tensor_inputs["image_latent"] = inputs["image_latent"][0]

        if "latents" not in inputs or len(inputs["latents"]) == 0:
            # Iteration 0: the loop-back edges arrive empty, so seed them here.
            device = self.get_device()
            t_lat, h_lat, w_lat = _latent_grid(self.config, fwd_info.step_metadata)
            shape = (1, self.config.in_channels, t_lat, h_lat, w_lat)
            # randn_tensor parity: CPU generator, CPU randn, then move.
            generator = torch.Generator(device="cpu").manual_seed(fwd_info.random_seed)
            latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
            tensor_inputs["latents"] = latents
            # 1-D, not 0-dim: the worker's output fanout reads dims[0] of every
            # routed edge.
            tensor_inputs["time_index"] = torch.zeros(1, dtype=torch.int64, device=device)
            tensor_inputs["unipc_model_outputs"] = torch.zeros(
                (SOLVER_ORDER, *shape), dtype=torch.float32, device=device
            )
            tensor_inputs["unipc_last_sample"] = torch.zeros(shape, dtype=torch.float32, device=device)
        else:
            time_index = inputs["time_index"][0]
            k = _loop_step(fwd_info, time_index)
            num_steps = int(fwd_info.step_metadata["num_inference_steps"])
            if k >= num_steps:
                # Async scheduling dispatched an iteration past this request's
                # stop (check_stop fired at k == N - 1 but the conductor had
                # already sent iteration N). Veto it: None makes the engine
                # skip this forward — the cosmos3 pattern.
                logger.info(
                    "Wan2.2 dit: skipping async-overshoot iteration %d "
                    "(request %s runs %d steps)",
                    k, fwd_info.request_id, num_steps,
                )
                return None
            tensor_inputs["latents"] = inputs["latents"][0]
            tensor_inputs["time_index"] = time_index
            tensor_inputs["unipc_model_outputs"] = inputs["unipc_model_outputs"][0]
            tensor_inputs["unipc_last_sample"] = inputs["unipc_last_sample"][0]
        step_info = self._step_info(tensor_inputs, fwd_info.step_metadata) if self._use_ragged_attention else None
        return NodeInputs(tensor_inputs=tensor_inputs, resource_step_info=step_info)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        latents: torch.Tensor,
        time_index: torch.Tensor,
        unipc_model_outputs: torch.Tensor,
        unipc_last_sample: torch.Tensor,
        text_embeds_pos: torch.Tensor,
        text_embeds_neg: torch.Tensor,
        image_latent: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        step_metadata = engine_inputs.single_request_info.step_metadata
        num_steps = int(step_metadata["num_inference_steps"])
        guidance_scale = float(step_metadata["guidance_scale"])
        k = int(time_index.item())
        device = latents.device

        with _inference_ctx():
            sigmas, timesteps = make_unipc_tables(num_steps, self.config.flow_shift)
            t = timesteps[k].to(device)

            # All-ones for T2V; frame 0 zeroed for I2V.
            mask = torch.ones(1, 1, *latents.shape[2:], dtype=torch.float32, device=device)
            if graph_walk == self._I2V_WALK:
                mask[:, :, 0] = 0
                model_input = (1 - mask) * image_latent + mask * latents
            else:
                model_input = latents
            model_input = model_input.to(self.transformer.dtype)

            # expand_timesteps: per-token timestep over the post-patch grid
            # ([:, ::2, ::2] == patch_size (1, 2, 2)); I2V frame-0 tokens get 0.
            temp_ts = (mask[0, 0][:, ::2, ::2] * t).flatten()
            noise_pred = self._noise_prediction(
                model_input, temp_ts, text_embeds_pos, text_embeds_neg, guidance_scale
            )

            # Inline UniPC step (exact port; see components/unipc.py).
            m_k = unipc_convert_model_output(noise_pred, latents, sigmas, k)
            sample = latents
            if k > 0:
                sample = unipc_corrector_step(
                    UniPCState(model_outputs=unipc_model_outputs, last_sample=unipc_last_sample),
                    this_model_output=m_k,
                    this_sample=sample,
                    sigmas=sigmas,
                    step_index=k,
                    order=unipc_effective_order(k - 1, num_steps),
                )
            new_ring = torch.stack([unipc_model_outputs[1], m_k])
            new_latents = unipc_predictor_step(
                UniPCState(model_outputs=new_ring, last_sample=sample),
                sample=sample,
                sigmas=sigmas,
                step_index=k,
                order=unipc_effective_order(k, num_steps),
            )

            if graph_walk == self._I2V_WALK and k + 1 >= num_steps:
                # The reference injects post-loop, and the final iteration is where
                # the latents edge routes to the VAE decoder.
                new_latents = (1 - mask) * image_latent + mask * new_latents

        return {
            "latents": [new_latents],
            "time_index": [time_index + 1],
            "unipc_model_outputs": [new_ring],
            "unipc_last_sample": [sample],
        }

    def _noise_prediction(
        self,
        model_input: torch.Tensor,
        temp_ts: torch.Tensor,
        text_embeds_pos: torch.Tensor,
        text_embeds_neg: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """CFG noise prediction; one batch-2 forward instead of the
        reference's two sequential forwards. The combine runs in the
        transformer's output dtype (bf16), like the reference."""
        do_cfg = guidance_scale > 1.0
        batch = 2 if do_cfg else 1
        hidden_states = model_input.repeat(batch, 1, 1, 1, 1)
        timestep = temp_ts.unsqueeze(0).expand(batch, -1)
        encoder_hidden_states = (
            torch.cat([text_embeds_pos, text_embeds_neg], dim=0) if do_cfg else text_embeds_pos
        ).to(self.transformer.dtype)
        transformer = self._compiled.forward_for(self.transformer, hidden_states)
        attention = {}
        ragged = self._ragged_attention()
        if ragged is not None:
            attention = {"ragged_self": ragged[0], "ragged_cross": ragged[1]}
        out = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **attention,
        )
        if not do_cfg:
            return out
        noise_cond, noise_uncond = out[0:1], out[1:2]
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop the denoise loop after exactly ``num_inference_steps`` iterations.

        While iteration k (0-based) is being postprocessed the iteration count
        still reads k, and the stop registered here ends the loop at the end of
        that iteration. So for N steps it must fire at k == N - 1, i.e. when
        ``k + 1 >= N``. The dit node runs with async scheduling, so the
        conductor may dispatch a speculative iteration N before this stop
        lands; ``prepare_inputs`` vetoes that overshoot by returning None, and
        ``>=`` keeps the stop firing when the deferred count reads N.
        """
        iter_idx = request_info.dynamic_loop_iter_counts.get(DENOISE_LOOP_NAME, 0)
        requested = int(request_info.step_metadata.get("num_inference_steps", 0) or 0)
        if requested > 0 and iter_idx + 1 >= requested:
            return {DENOISE_LOOP_NAME}
        return set()


# Loop-back edges of the scaffold's denoise loop beyond ``latents``: UniPC's ring of the
# last two converted outputs and the sample its predictor last started from.
UNIPC_MODEL_OUTPUTS = "unipc_model_outputs"
UNIPC_LAST_SAMPLE = "unipc_last_sample"


@dataclass(frozen=True)
class Wan22Shape:
    """What has to match for two requests to share a denoise forward: the latent grid,
    the text length, whether CFG runs both branches and whether frame 0 is an I2V
    conditioning frame. Requests batch, and accelerator-graph buckets key, on it."""

    grid: tuple[int, int, int]  # latent (frames, height, width)
    text_len: int
    cfg: bool
    i2v: bool
    video_tokens: int  # per guidance branch, after the patch

    @property
    def rows(self) -> int:
        """Transformer rows a request contributes: one per guidance branch."""
        return 2 if self.cfg else 1


def wan22_shape(config: Wan22Config, step_metadata: dict, i2v: bool) -> Wan22Shape:
    grid = _latent_grid(config, step_metadata)
    p_t, p_h, p_w = config.patch_size
    guidance_scale = float(step_metadata.get("guidance_scale", config.guidance_scale))
    return Wan22Shape(
        grid=grid, text_len=config.text_max_seq_len, cfg=guidance_scale > 1.0, i2v=i2v,
        video_tokens=(grid[0] // p_t) * (grid[1] // p_h) * (grid[2] // p_w),
    )


@dataclass(frozen=True)
class Wan22Schedule(FlowMatchSchedule):
    """One request's UniPC schedule: the reference's sigma and timestep tables, the
    request's guidance scale and the solver's per-step weights."""

    guidance_scale: float
    unipc: UniPCCoefficients


class Wan22DenoiseSubmodule(_Fp32IslandMixin, DenoiseLoopSubmodule):
    """One Wan2.2 denoise step on the DiT scaffold: CFG as one forward over both
    guidance branches, then UniPC in its coefficient form.

    Unlike ``Wan22DitSubmodule``, every per-step decision (the solver's order ramp,
    the corrector at step 0, I2V's final write-back) is a per-row weight staged with
    the request's schedule, so requests at different steps batch (equal
    ``Wan22Shape`` only) and each served shape replays a captured graph. Loop-carried,
    float32 over the latent grid ``[C, F, H, W]``: ``latents``, the
    ``unipc_model_outputs`` ring ``[2, C, F, H, W]`` and ``unipc_last_sample``.

    The UniPC update matches the step functions in ``components/unipc.py`` to float32
    rounding; the CFG combine and the x0 conversion round through bf16 where the
    reference does.
    """

    _I2V_WALK = Wan22DitSubmodule._I2V_WALK

    SOLVER_STATE = (UNIPC_MODEL_OUTPUTS, UNIPC_LAST_SAMPLE)
    # Every scalar is a plain [B] and is viewed per row in ``denoise``: the latents are
    # [B, C, F, H, W], which the base's [B, 1, 1] would not broadcast over.
    STEP_SCALARS = {
        "sigma": StepScalar("sigmas", broadcast=False),
        "timestep": StepScalar("timesteps", broadcast=False, capture_fill=1000.0),
        "guidance": StepScalar("guidance", broadcast=False, capture_fill=5.0),
        "final": StepScalar("final", broadcast=False, capture_fill=0.0),
        **{name: StepScalar(name, broadcast=False, capture_fill=0.0) for name in UniPCCoefficients._fields},
    }

    def __init__(
        self,
        transformer: nn.Module,
        config: Wan22Config,
        *,
        use_ragged_attention: bool = False,
        max_batch_size: int = 2,
        capture_buckets=(),
        capture_batch_sizes=(1, 2),
    ):
        super().__init__(
            loop_name=DENOISE_LOOP_NAME, max_batch_size=max_batch_size, attn_resource_key=None,
            capture_buckets=capture_buckets, capture_batch_sizes=capture_batch_sizes,
        )
        self.transformer = transformer
        self.config = config
        self.use_ragged_attention = bool(use_ragged_attention)
        self._attention: dict[str, RaggedAttentionFn] | None = None
        self._record_fp32_islands()
        self._compiled = _GridCompiledDiT(transformer, config)

    # hooks -----------------------------------------------------------------
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> Wan22Shape:
        return wan22_shape(self.config, fwd_info.step_metadata, i2v=fwd_info.graph_walk == self._I2V_WALK)

    def schedule_for(self, fwd_info: CurrentForwardPassInfo, bucket_key: Wan22Shape) -> Wan22Schedule:
        num_steps = int(fwd_info.step_metadata["num_inference_steps"])
        sigmas, timesteps = make_unipc_tables(num_steps, self.config.flow_shift)
        return Wan22Schedule(
            # the integer timesteps are exact in float32, the dtype the reference's
            # per-token timestep grid has
            sigmas=sigmas, timesteps=timesteps.to(torch.float32), mu=None,
            guidance_scale=float(fwd_info.step_metadata.get("guidance_scale", self.config.guidance_scale)),
            unipc=unipc_coefficients(sigmas, num_steps),
        )

    def _step_scalar_sources(self, schedule: Wan22Schedule, device: torch.device) -> dict[str, torch.Tensor]:
        # Every per-step table in one pinned copy instead of a blocking copy each.
        n = schedule.num_steps
        final = torch.zeros(n, dtype=torch.float32)
        final[-1] = 1.0
        sources = {
            "sigmas": schedule.sigmas[:n],
            "timesteps": schedule.timesteps,
            "guidance": torch.full((n,), schedule.guidance_scale, dtype=torch.float32),
            "final": final,
            **schedule.unipc._asdict(),
        }
        table = self.stage_seed(torch.stack(list(sources.values())), device)
        return dict(zip(sources, table.unbind(0), strict=True))

    def seed_loop_back(self, fwd_info, bucket_key: Wan22Shape, generator: torch.Generator) -> dict[str, torch.Tensor]:
        # randn_tensor parity: the reference's [1, C, F, H, W] fp32 draw on the CPU generator
        shape = (self.config.in_channels, *bucket_key.grid)
        device = self.get_device()
        return {
            self.LATENTS: torch.randn((1, *shape), generator=generator, dtype=torch.float32)[0],
            # zeros until written; the step-0 weights never read them
            UNIPC_MODEL_OUTPUTS: torch.zeros((SOLVER_ORDER, *shape), dtype=torch.float32, device=device),
            UNIPC_LAST_SAMPLE: torch.zeros(shape, dtype=torch.float32, device=device),
        }

    def request_inputs(self, fwd_info, inputs: NameToTensorList, bucket_key: Wan22Shape) -> dict[str, torch.Tensor]:
        tensors = {"text_embeds_pos": inputs["text_embeds_pos"][0][0]}
        if bucket_key.cfg:
            tensors["text_embeds_neg"] = inputs["text_embeds_neg"][0][0]
        if bucket_key.i2v:
            tensors["image_latent"] = inputs["image_latent"][0][0]
        return tensors

    def num_tokens(self, bucket_key: Wan22Shape) -> int:
        return bucket_key.video_tokens * bucket_key.rows

    def capture_request_inputs(self, bucket_key: Wan22Shape, device) -> dict[str, torch.Tensor]:
        shape = (self.config.in_channels, *bucket_key.grid)
        text = torch.zeros(bucket_key.text_len, self.config.text_dim, dtype=torch.bfloat16, device=device)
        tensors = {
            self.LATENTS: torch.zeros(shape, dtype=torch.float32, device=device),
            UNIPC_MODEL_OUTPUTS: torch.zeros((SOLVER_ORDER, *shape), dtype=torch.float32, device=device),
            UNIPC_LAST_SAMPLE: torch.zeros(shape, dtype=torch.float32, device=device),
            "text_embeds_pos": text,
        }
        if bucket_key.cfg:
            tensors["text_embeds_neg"] = text.clone()
        if bucket_key.i2v:
            tensors["image_latent"] = torch.zeros(
                (self.config.in_channels, 1, *bucket_key.grid[1:]), dtype=torch.float32, device=device,
            )
        return tensors

    def build_layout(self, bucket_key: Wan22Shape, device) -> tuple[torch.Tensor, torch.Tensor] | None:
        """I2V's first-frame mask over the latent grid and over the patched tokens
        (frame 0 keeps the conditioning and is denoised at timestep 0); None for T2V."""
        if not bucket_key.i2v:
            return None
        mask = torch.ones(1, 1, *bucket_key.grid, dtype=torch.float32, device=device)
        mask[:, :, 0] = 0
        p_t, p_h, p_w = self.config.patch_size
        return mask, mask[0, 0][::p_t, ::p_h, ::p_w].flatten()

    def denoise(
        self, engine_inputs, bucket_key: Wan22Shape, latents, unipc_model_outputs, unipc_last_sample,
        sigma, timestep, guidance, final, text_embeds_pos, text_embeds_neg=None, image_latent=None, **coeffs,
    ):
        batch = latents.shape[0]

        def per_row(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, 1, 1, 1, 1)

        with _inference_ctx():
            dtype = self.transformer.dtype
            if bucket_key.i2v:
                mask, token_mask = self.layout(bucket_key, latents.device)
                model_input = (1 - mask) * image_latent + mask * latents
                timesteps = token_mask * timestep.view(batch, 1)
            else:
                model_input = latents
                timesteps = timestep.view(batch, 1).expand(batch, bucket_key.video_tokens)
            # one transformer row per guidance branch, a request's two adjacent (cond
            # first), the order declare_step gives the attention segments
            rows = bucket_key.rows
            hidden_states = model_input.to(dtype).repeat_interleave(rows, dim=0)
            text = torch.stack([text_embeds_pos, text_embeds_neg], dim=1).flatten(0, 1) if bucket_key.cfg \
                else text_embeds_pos
            out = self._compiled.forward_for(self.transformer, hidden_states)(
                hidden_states=hidden_states,
                timestep=timesteps.repeat_interleave(rows, dim=0),
                encoder_hidden_states=text.to(dtype),
                **self.attention(),
            )
            if bucket_key.cfg:
                cond, uncond = out.unflatten(0, (batch, 2)).unbind(1)
                # the reference's bf16 combine with a Python-float scale: the product
                # in float32, rounded to bf16
                out = uncond + (per_row(guidance) * (cond - uncond).float()).to(out.dtype)
            # x0 conversion, rounded to bf16 as the reference's CPU-scalar product is
            this_output = latents - (per_row(sigma) * out.float()).to(out.dtype)
            new_latents, ring, corrected = unipc_coefficient_step(
                latents, this_output, unipc_model_outputs, unipc_last_sample,
                UniPCCoefficients(**{name: per_row(coeffs[name]) for name in UniPCCoefficients._fields}),
            )
            if bucket_key.i2v:
                # the reference writes the conditioning frame back after the loop; the
                # last step's latents are what reach the decoder
                inject = per_row(final) * (1 - mask)
                new_latents = inject * image_latent + (1 - inject) * new_latents
        return {self.LATENTS: new_latents, UNIPC_MODEL_OUTPUTS: ring, UNIPC_LAST_SAMPLE: corrected}

    # attention ------------------------------------------------------------
    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._attention = None  # the memoized callables hold the old resources

    def attention(self) -> dict[str, RaggedAttentionFn]:
        """The transformer's ``ragged_self`` / ``ragged_cross`` callables, built once
        per resource binding (the compiled region guards on their identity); empty
        for SDPA."""
        if self._attention is None:
            self_attn = self.node_resources.get(DIT_ATTN) if self.use_ragged_attention else None
            cross_attn = self.node_resources.get(DIT_XATTN) if self.use_ragged_attention else None
            self._attention = {}
            if self_attn is not None and cross_attn is not None:
                self._attention = {
                    "ragged_self": RaggedAttentionCallable(self_attn, VIDEO_SPAN),
                    "ragged_cross": RaggedAttentionCallable(cross_attn, cross_label(VIDEO_SPAN, TEXT_SPAN)),
                }
        return self._attention

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases=None, **kwargs,
    ) -> SubmoduleStep | None:
        """One video segment per guidance branch in the forward's row order, and the
        text span each branch's cross-attention reads."""
        if not self.use_ragged_attention:
            return None
        video, text = [], []
        for rid, inp in zip(request_ids, inputs, strict=True):
            shape: Wan22Shape = inp.resource_step_info
            for _ in range(shape.rows):
                video.append(Segment(rid, VIDEO_SPAN, shape.video_tokens))
                text.append(Segment(rid, TEXT_SPAN, shape.text_len))
        return SubmoduleStep(
            steps={
                DIT_ATTN: AttentionStep(segments=tuple(video), causal=False),
                DIT_XATTN: RaggedCrossAttentionStep(segments=tuple(video + text), pairs=((VIDEO_SPAN, TEXT_SPAN),)),
            },
            cg_key_info=self._uniform_key(inp.resource_step_info for inp in inputs),
        )


# ---------------------------------------------------------------------------
# vae_decoder
# ---------------------------------------------------------------------------

class Wan22VaeDecoderSubmodule(_SingleRequestMixin, _Fp32IslandMixin, NodeSubmodule):
    """Wan2.2-VAE latent -> pixel decoder (tiled).

    Consumes the loop's final ``latents`` and emits ``video_output``
    ``[1, 3, num_frames, H, W]`` as uint8. Quantizing here, at the worker boundary,
    keeps the edge to the data worker at one byte per pixel instead of a 4x larger
    float tensor; the output is 8-bit mp4 either way, and ``Wan22Model.postprocess``
    does the muxing.

    The node owns its own VAE instance because the decode dtype depends on the cuDNN
    version (see ``_decode_dtype``) while the I2V encode stays bf16 — one shared
    instance would re-cast the whole VAE whenever the two disagree.

    Decode calls ``tiled_decode`` / ``decode`` directly rather than
    ``enable_tiling``, whose flag is instance state: the encoder's numerics must not
    depend on which nodes happen to be colocated. Tiling bounds the workspace by
    tile count; untiled wants a conv3d workspace that scales with output spatial
    area (~20 GiB at dense fp32, enough to OOM a 32 GiB card when the DiT is
    co-resident) but decodes faster with no tile-seam error. So the path is
    VRAM-gated (Feature 1): decode untiled when free VRAM at decode time comfortably
    exceeds the estimated untiled peak for the requested size, tile otherwise. The
    estimate is the calibrated formula above; the decision reads live free memory
    via ``torch.cuda.mem_get_info``; ``config.vae_decode_tiling`` /
    ``WAN22_VAE_DECODE_TILING`` force either path; the chosen path is logged per
    request. The tiled/untiled numeric gap is priced by the tiled-vs-untiled test.

    With ``config.vae_decode_fast`` the untiled path at a size in
    ``config.vae_decode_fast_sizes`` runs the shared Wan VAE decoder
    (``mstar.model.components.wan_vae``) instead of ``vae.decode``: the same
    per-latent decoder calls from an empty cache, compiled, on channels-last
    weights and activations where the platform's conv library wants them
    (``wan_vae.decode_prefers_channels_last``; not MIOpen). With channels-last,
    every other decode switches the VAE back to the layout it loaded in first
    (an exact copy of the decoder weights), so it is stock ``vae.decode`` /
    ``tiled_decode`` and compiles nothing. The engine
    still never wraps the forward; only the decoder's steady-state call is
    compiled.
    """

    disable_torch_compile = True

    def __init__(self, vae: nn.Module, config: Wan22Config):
        super().__init__()
        self.config = config
        self._decode_dtype_cached = None
        # Resolve the decode dtype now (a cheap cuDNN-version read) so the
        # choice is fixed at startup rather than on the first request. Cast
        # BEFORE recording the fp32 islands: the recorded set must describe
        # the dtype the decode will actually run in, not the checkpoint's
        # load dtype, or the mixin re-pins the weights back afterwards.
        # ``vae`` is never None here: dummy mode returns from
        # ``_create_submodule`` before any submodule is constructed.
        self.vae = vae.to(self._decode_dtype())
        self._record_fp32_islands()
        self._fast_step = None
        self._fast_sizes = config.fast_decode_sizes()
        self._channels_last = False
        if config.vae_decode_fast:
            from mstar.model.components import wan_vae

            # Before the compile traces the decoder; this VAE instance is the
            # decoder node's own.
            self._channels_last = wan_vae.decode_prefers_channels_last()
            if self._channels_last:
                wan_vae.to_channels_last(self.vae)
            self._fast_step = wan_vae.DecoderStep(self.vae, compile=True, deterministic=True)
            logger.info(
                "Wan2.2 VAE decode: untiled %s on the shared Wan VAE decoder (%s, compiled); "
                "other sizes decode stock",
                ", ".join(f"{w}x{h}" for w, h in sorted(self._fast_sizes)),
                "channels-last" if self._channels_last else "default layout",
            )

    def _decode_dtype(self):
        # see wan_vae.decode_dtype, WAN22_VAE_DECODE_FP32 overrides
        if self._decode_dtype_cached is not None:
            return self._decode_dtype_cached
        from mstar.model.components import wan_vae

        self._decode_dtype_cached = wan_vae.decode_dtype("WAN22_VAE_DECODE_FP32")
        logger.info("Wan2.2 VAE decode dtype = %s (%s)",
                    self._decode_dtype_cached, wan_vae.conv_library_version())
        return self._decode_dtype_cached

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        latents = inputs["latents"][0]
        # the scaffold's loop carries one request's latents without the batch dim
        return NodeInputs(tensor_inputs={"latents": latents if latents.dim() == 5 else latents.unsqueeze(0)})

    def _output_size(self, latents: torch.Tensor) -> tuple[int, int, int]:
        """(height, width, num_frames) the latent grid decodes to — the inverse of
        the VAE's spatial (/16) and temporal (4k+1) compression."""
        _, _, t_lat, h_lat, w_lat = latents.shape
        height = h_lat * self.config.vae_scale_factor_spatial
        width = w_lat * self.config.vae_scale_factor_spatial
        num_frames = (t_lat - 1) * self.config.vae_scale_factor_temporal + 1
        return height, width, num_frames

    @staticmethod
    def _estimated_untiled_peak_bytes(
        height: int, width: int, num_frames: int, elt_bytes: int
    ) -> int:
        """Estimated untiled-decode peak workspace (bytes) for one output clip,
        from the spatial-area calibration above. fp32-calibrated, scaled by the
        decode dtype's element size."""
        spatial_mpix = height * width / 1e6
        gib = (elt_bytes / 4) * spatial_mpix * (
            _DECODE_PEAK_SPATIAL_GIB_PER_MPIX + _DECODE_PEAK_FRAME_GIB_PER_MPIX * num_frames
        )
        return int(gib * 2**30)

    @staticmethod
    def _decode_should_tile(mode: str, free_bytes: int, estimated_peak_bytes: int) -> bool:
        """Pure tiling decision (True == tile). ``mode`` forces a path
        ("tiled"/"untiled"); "auto" tiles unless free VRAM clears the estimated
        untiled peak by the safety margin. Kept side-effect-free so the gate is
        unit-tested with injected free-memory values."""
        if mode == "tiled":
            return True
        if mode == "untiled":
            return False
        return free_bytes < estimated_peak_bytes * _DECODE_UNTILED_SAFETY_MARGIN

    def _resolve_tiling_mode(self) -> str:
        """config.vae_decode_tiling, overridden by the env var, canonicalized via
        the shared normalizer. An unknown value falls back to "auto" loudly (warned
        once here) rather than silently forcing a path."""
        raw = os.environ.get(_DECODE_TILING_ENV, self.config.vae_decode_tiling)
        mode = normalize_decode_tiling_mode(raw)
        if mode != (raw or "").lower():
            logger.warning(
                "Wan2.2 VAE decode: unknown tiling mode %r (config.vae_decode_tiling / %s); "
                "using 'auto'.", raw, _DECODE_TILING_ENV,
            )
        return mode

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        latents: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        with _inference_ctx():
            vae = self.vae
            vae_dtype = self._decode_dtype()
            if next(vae.parameters()).dtype != vae_dtype:
                vae = vae.to(vae_dtype)
            device = self.get_device()
            latents = latents.to(device=device, dtype=vae_dtype)
            z_dim = self.config.vae_z_dim
            latents_mean = (
                torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            # Feature 1: pick untiled vs tiled from live free VRAM at decode time.
            height, width, num_frames = self._output_size(latents)
            elt_bytes = torch.finfo(vae_dtype).bits // 8
            estimate = self._estimated_untiled_peak_bytes(height, width, num_frames, elt_bytes)
            mode = self._resolve_tiling_mode()
            free_bytes, _total = torch.cuda.mem_get_info(device)
            tiled = self._decode_should_tile(mode, free_bytes, estimate)
            logger.info(
                "Wan2.2 VAE decode: %s (%dx%d x%df, est untiled peak %.1f GiB, "
                "free %.1f GiB, margin %.2f, mode=%s)",
                "tiled" if tiled else "untiled", height, width, num_frames,
                estimate / 2**30, free_bytes / 2**30, _DECODE_UNTILED_SAFETY_MARGIN, mode,
            )
            fast = self._fast_step is not None and not tiled and (width, height) in self._fast_sizes
            if self._fast_step is not None:
                from mstar.model.components import wan_vae
            if self._channels_last:
                # One request per step, so nothing else decodes on this VAE
                # while its layout switches. Outside inference mode: a weight
                # made in it is an inference tensor, whose dispatch keys the
                # compiled graph's parameter guards reject (a recompile).
                with torch.inference_mode(False):
                    (wan_vae.to_channels_last if fast else wan_vae.to_default_layout)(vae)
            if tiled:
                video = vae.tiled_decode(latents, return_dict=False)[0]
            elif fast:
                video = wan_vae.decode_causal(
                    vae, latents, wan_vae.new_feat_cache(vae), self._fast_step, clip_start=True,
                )
            else:
                video = vae.decode(latents, return_dict=False)[0]
            # Quantize to 8-bit at the worker boundary (see class docstring).
            video = (video / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)
        return {"video_output": [video]}
