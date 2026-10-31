"""Wan2.2-TI2V-5B node submodules (all four declare no engine resources).

    text_encoder -> Wan22TextEncoderSubmodule (UMT5-XXL, thin HF wrapper)
    vae_encoder  -> Wan22VaeEncoderSubmodule  (Wan2.2-VAE encode, I2V only)
    dit          -> Wan22DitSubmodule         (native 5B DiT + the scaffold's UniPC)
    vae_decoder  -> Wan22VaeDecoderSubmodule  (Wan2.2-VAE decode)

All four set ``disable_torch_compile``, so the engine never wraps their forwards: the
three wrapped nodes run once per request with no graph to amortize, and the dit's whole
forward trips Inductor on the CPU-resident UniPC sigma. Instead the dit compiles its inner
transformer region alone (``config.compile_dit``) for the shapes in
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
from typing import NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.denoise_loop import UniPCDenoiseLoopSubmodule
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.components.diffusion.unipc import make_unipc_tables
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.wan22.config import Wan22Config

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


def _loop_step(fwd_info: CurrentForwardPassInfo) -> int:
    """The denoise iteration this step runs, from the engine's loop counter.

    A value carried on a loop-back edge would be the previous step's device
    output, and reading it on the host waits for that step to finish, which
    stalls the speculative preparation of this one.
    """
    return int(fwd_info.dynamic_loop_iter_counts[DENOISE_LOOP_NAME])


def _latent_grid(config: Wan22Config, step_metadata: dict) -> tuple[int, int, int]:
    """(T_lat, H_lat, W_lat) for the request's pixel-space dims (diffusers
    ``prepare_latents``: T=(F-1)//4+1, spatial /16)."""
    t_lat = (int(step_metadata["num_frames"]) - 1) // config.vae_scale_factor_temporal + 1
    h_lat = int(step_metadata["height"]) // config.vae_scale_factor_spatial
    w_lat = int(step_metadata["width"]) // config.vae_scale_factor_spatial
    return t_lat, h_lat, w_lat


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

class WanStepKey(NamedTuple):
    """What has to match for two rows to share a dit forward."""

    frames: int
    height: int
    width: int
    # one combine per forward; above 1.0 is what turns the negative branch on
    guidance: float
    # I2V rows carry ``image_latent`` and keep frame 0 clean
    i2v: bool


class Wan22DitSubmodule(_Fp32IslandMixin, UniPCDenoiseLoopSubmodule):
    """Dense 5B video DiT on the scaffold's denoise loop, stepped by its UniPC.

    One loop iteration is one transformer forward over every row's positive and
    negative embeddings, plus a UniPC predictor/corrector update per row.

    Loop-carried edges, all float32 over ``[1, C, T, H, W]``:

        latents              current sample x_t
        unipc_model_outputs  ring buffer of the last two converted outputs;
                             see ``UniPCState``
        unipc_last_sample    the sample the previous predictor was given

    Nothing else is carried. The sigmas, the order ramp and the timestep pair all
    follow from the engine's loop counter and the per-request tables, so they are
    recomputed rather than shipped across an edge.

    At iteration 0 this submodule seeds the three edges (noise from the request's
    seed, on a CPU generator as diffusers does). I2V also consumes the persisted
    ``image_latent``: frame 0 of the input is replaced by it and its per-token
    timestep zeroed, and the conditioning is written back into the output latents
    only on the final iteration, matching the reference's post-loop injection.
    """

    # Mirrors Wan22Model.VIDEO_GEN_I2V_WALK; importing the model here would be a
    # circular import.
    _I2V_WALK = "video_gen_i2v"

    def __init__(self, transformer: nn.Module, config: Wan22Config):
        super().__init__(loop_name=DENOISE_LOOP_NAME, max_batch_size=config.dit_max_batch_size)
        self.transformer = transformer
        self.config = config
        self._record_fp32_islands()
        # The WHOLE forward cannot be one compiled region: Inductor would fuse the
        # UniPC step's `sample - sigma * output` and pass the 0-dim CPU-resident sigma
        # as a device pointer, which fails on the first call. Moving sigma to CUDA
        # would fix the launch but break bit-exactness against the reference
        # (components/diffusion/unipc.py). So compile the inner DiT region only
        # (tensors in, tensor out; nothing CPU-resident crosses it), IN PLACE so
        # ``self.transformer`` keeps its identity — fp32 islands, ``.dtype``, the
        # tests' transformer swap. dynamic=False traces one graph per shape (first
        # request at a new resolution pauses; logged in ``_noise_prediction``);
        # fullgraph=False allows the SDPA graph break, matching cosmos3. Only the
        # latent grids of ``config.compile_dit_sizes`` reach the compiled forward;
        # every other grid runs ``_eager_forward``, the transformer's own forward.
        self._compile_dit = bool(config.compile_dit)
        self._compile_grids = config.compile_dit_grids()
        self._compiled_shapes: set[tuple[int, ...]] = set()
        self._eager_forward = None
        if self._compile_dit and transformer is not None:
            # with the transformer it belongs to: one swapped in later runs its own forward
            self._eager_forward = (transformer, transformer.forward)
            transformer.forward = torch.compile(
                transformer.forward, fullgraph=False, dynamic=False,
            )
            logger.info(
                "Wan2.2 DiT: torch.compile enabled on the transformer region "
                "(fullgraph=False, dynamic=False) for latent (frames, height, width) %s; other "
                "shapes and the UniPC solver run eager.",
                sorted(self._compile_grids),
            )

    def step_index(self, fwd_info: CurrentForwardPassInfo) -> int:
        return _loop_step(fwd_info)

    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> WanStepKey:
        meta = fwd_info.step_metadata
        return WanStepKey(
            *_latent_grid(self.config, meta), guidance=float(meta["guidance_scale"]),
            i2v=fwd_info.graph_walk == self._I2V_WALK,
        )

    def schedule_for(self, fwd_info: CurrentForwardPassInfo, bucket_key: WanStepKey) -> FlowMatchSchedule:
        sigmas, timesteps = make_unipc_tables(
            int(fwd_info.step_metadata["num_inference_steps"]), self.config.flow_shift,
        )
        return FlowMatchSchedule(sigmas=sigmas, timesteps=timesteps, mu=None)

    def seed_latents(
        self, fwd_info: CurrentForwardPassInfo, bucket_key: WanStepKey, generator: torch.Generator,
    ) -> torch.Tensor:
        shape = (1, self.config.in_channels, bucket_key.frames, bucket_key.height, bucket_key.width)
        # randn_tensor parity: CPU generator, CPU randn, then move.
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    def request_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, bucket_key: WanStepKey,
    ) -> dict[str, torch.Tensor]:
        tensors = {name: inputs[name][0] for name in ("text_embeds_pos", "text_embeds_neg")}
        if bucket_key.i2v:
            tensors["image_latent"] = inputs["image_latent"][0]
        return tensors

    def num_tokens(self, bucket_key: WanStepKey) -> int:
        return bucket_key.frames * bucket_key.height * bucket_key.width // 4  # patch_size (1, 2, 2)

    @staticmethod
    def _first_frame_mask(sample: torch.Tensor) -> torch.Tensor:
        """Zero on the frame an I2V request keeps clean, one elsewhere."""
        mask = torch.ones(1, 1, *sample.shape[-3:], dtype=torch.float32, device=sample.device)
        mask[:, :, 0] = 0
        return mask

    def denoise(
        self, engine_inputs: ModelInputsFromEngine, bucket_key: WanStepKey, latents: torch.Tensor,
        timestep: torch.Tensor, **kwargs,
    ) -> dict[str, torch.Tensor]:
        """A row of ``latents`` is ``[1, C, T, H, W]``: it keeps the batch dim the
        transformer and the solver see it with."""
        with _inference_ctx():
            out = super().denoise(engine_inputs, bucket_key, latents, timestep, **kwargs)
            if bucket_key.i2v:
                # The reference injects post-loop, and the final iteration is where
                # the latents edge routes to the VAE decoder.
                mask = self._first_frame_mask(latents)
                infos = (engine_inputs.per_request_info[rid] for rid in engine_inputs.request_ids)
                out[self.LATENTS] = torch.stack([
                    (1 - mask) * image_latent + mask * sample
                    if self.step_index(info) + 1 >= self.request_state(info.rid_handle)["num_steps"] else sample
                    for info, image_latent, sample in zip(infos, kwargs["image_latent"], out[self.LATENTS], strict=True)
                ])
        return out

    def model_output(
        self,
        engine_inputs: ModelInputsFromEngine,
        bucket_key: WanStepKey,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds_pos: torch.Tensor,
        text_embeds_neg: torch.Tensor,
        image_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = latents.flatten(0, 1)
        if bucket_key.i2v:
            mask = self._first_frame_mask(batch)
            model_input = (1 - mask) * image_latent.flatten(0, 1) + mask * batch
        else:
            mask = torch.ones(1, 1, *batch.shape[2:], dtype=torch.float32, device=batch.device)
            model_input = batch

        # expand_timesteps: every token of a latent frame takes its frame's
        # timestep (patch_size (1, 2, 2) keeps the frame axis); I2V frame 0 gets 0.
        temp_ts = (mask[0, 0, :, 0, 0] * timestep[:, None]).squeeze(0)
        output = self._noise_prediction(
            model_input.to(self.transformer.dtype), temp_ts, text_embeds_pos.flatten(0, 1),
            text_embeds_neg.flatten(0, 1), bucket_key.guidance,
        )
        return output[:, None]

    def _noise_prediction(
        self,
        model_input: torch.Tensor,
        temp_ts: torch.Tensor,
        text_embeds_pos: torch.Tensor,
        text_embeds_neg: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """CFG noise prediction; one forward over every row's two branches instead
        of the reference's two sequential forwards. The combine runs in the
        transformer's output dtype (bf16), like the reference. ``temp_ts`` is per
        row, ``[B, frames]``, or one row's ``[frames]``."""
        do_cfg = guidance_scale > 1.0
        branches = 2 if do_cfg else 1
        hidden_states = model_input.repeat(branches, 1, 1, 1, 1)
        timestep = temp_ts.repeat(branches, 1)
        # dynamic=False keys the compiled trace on the full input shape (batch
        # included, so each row count and CFG on or off are distinct graphs). Announce
        # the one-time compile pause the first time a shape is seen, so an operator
        # watching the log knows why the first request at a new resolution is slow.
        compiled = self._compile_dit and tuple(hidden_states.shape[2:]) in self._compile_grids
        if compiled:
            shape_key = tuple(hidden_states.shape)
            if shape_key not in self._compiled_shapes:
                self._compiled_shapes.add(shape_key)
                logger.info(
                    "Wan2.2 DiT: first forward at shape %s — torch.compile is tracing "
                    "this resolution now (one-time pause); later requests reuse it.",
                    shape_key,
                )
        encoder_hidden_states = (
            torch.cat([text_embeds_pos, text_embeds_neg], dim=0) if do_cfg else text_embeds_pos
        ).to(self.transformer.dtype)
        transformer = self.transformer
        if not compiled and self._eager_forward is not None and self._eager_forward[0] is transformer:
            transformer = self._eager_forward[1]
        out = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        if not do_cfg:
            return out
        rows = model_input.shape[0]
        noise_cond, noise_uncond = out[:rows], out[rows:]
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


# ---------------------------------------------------------------------------
# vae_decoder
# ---------------------------------------------------------------------------

class Wan22VaeDecoderSubmodule(_SingleRequestMixin, _Fp32IslandMixin, NodeSubmodule):
    """Wan2.2-VAE latent -> pixel decoder (tiled).

    Consumes the loop's final ``latents`` and emits ``video_output`` as uint8
    ``[num_frames, H, W, 3]`` frames. Quantizing here, at the worker boundary,
    keeps the edge to the data worker at one byte per pixel instead of a 4x larger
    float tensor; the output is 8-bit mp4 either way, and ``Wan22Model.postprocess``
    does the muxing. The frames are laid out as PyAV's rgb24 frames here, on the
    GPU, so the encoder reads them without copying each one out of a strided
    view on the host.

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
    per-latent decoder calls from an empty cache, on channels-last weights and
    activations, compiled. Every other decode switches the VAE back to the
    layout it loaded in first (an exact copy of the decoder weights), so it is
    stock ``vae.decode`` / ``tiled_decode`` and compiles nothing. The engine
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
        if config.vae_decode_fast:
            from mstar.model.components import wan_vae

            # Before the compile traces the decoder; this VAE instance is the
            # decoder node's own.
            wan_vae.to_channels_last(self.vae)
            self._fast_step = wan_vae.DecoderStep(self.vae, compile=True, deterministic=True)
            logger.info(
                "Wan2.2 VAE decode: untiled %s on the shared Wan VAE decoder (channels-last, compiled); "
                "other sizes decode stock",
                ", ".join(f"{w}x{h}" for w, h in sorted(self._fast_sizes)),
            )

    def _decode_dtype(self):
        # TODO: pull conv3d-dtype resolution into a common helper shared with
        # cosmos3 (same cuDNN-version issue there).
        # cuDNN ships fast bf16 conv3d for this VAE decode only from 9.16. On
        # older builds bf16 is 2-10x SLOWER than fp32/TF32, so read the live cuDNN
        # version and pick accordingly; an upgrade then flips the choice on its own.
        # WAN22_VAE_DECODE_FP32 forces fp32.
        if self._decode_dtype_cached is not None:
            return self._decode_dtype_cached
        override = os.environ.get("WAN22_VAE_DECODE_FP32")
        if override is not None:
            on = override.lower() not in ("0", "false", "no", "off")
            self._decode_dtype_cached = torch.float32 if on else torch.bfloat16
        else:
            ver = torch.backends.cudnn.version() or 0
            self._decode_dtype_cached = torch.bfloat16 if ver >= 91600 else torch.float32
        logger.info("Wan2.2 VAE decode dtype = %s (cuDNN %s)",
                    self._decode_dtype_cached, torch.backends.cudnn.version())
        return self._decode_dtype_cached

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(tensor_inputs={"latents": inputs["latents"][0]})

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
            # Quantize to 8-bit at the worker boundary, as rgb24 frames (see class docstring).
            frames = (video[0] / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
        return {"video_output": [frames]}
