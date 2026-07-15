"""Wan2.2-TI2V-5B node submodules (all four STATELESS).

    text_encoder -> Wan22TextEncoderSubmodule (UMT5-XXL, thin HF wrapper)
    vae_encoder  -> Wan22VaeEncoderSubmodule  (Wan2.2-VAE encode, I2V only)
    dit          -> Wan22DitSubmodule         (native 5B DiT + inline UniPC step)
    vae_decoder  -> Wan22VaeDecoderSubmodule  (Wan2.2-VAE decode)

All four set ``disable_torch_compile``, so the engine never wraps their forwards: the
three wrapped nodes run once per request with no graph to amortize, and the dit's whole
forward trips Inductor on the CPU-resident UniPC sigma. Instead the dit compiles its inner
transformer region alone (``config.compile_dit``), leaving the solver eager.

Numerics are governed by the checkpoint dtypes, not by the engine.
``Wan22Model.get_autocast_dtype`` returns None so the engine neither autocasts
nor blanket-casts the modules, and each forward disables any ambient autocast
itself — reference parity must not depend on what context wraps the call.
"""

import logging
import os

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.wan22.components.unipc import (
    SOLVER_ORDER,
    UniPCState,
    make_unipc_tables,
    unipc_convert_model_output,
    unipc_corrector_step,
    unipc_effective_order,
    unipc_predictor_step,
)
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


def _no_autocast():
    """Reference forwards run under module dtypes, never autocast."""
    return torch.amp.autocast("cuda", enabled=False)


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


def _set_buffer(module: nn.Module, dotted_name: str, value: torch.Tensor):
    owner_path, _, attr = dotted_name.rpartition(".")
    owner = module.get_submodule(owner_path) if owner_path else module
    setattr(owner, attr, value)


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

class Wan22TextEncoderSubmodule(_Fp32IslandMixin, NodeSubmodule):
    """UMT5-XXL prompt encoder.

    Consumes ``text_inputs`` = ``[positive_ids, negative_ids]``; emits
    ``text_embeds_pos`` / ``text_embeds_neg``, each ``[1, text_max_seq_len,
    text_dim]``, zero-padded past the true sequence length. The two prompts run as
    two batch-1 forwards, as the reference's separate ``encode_prompt`` calls do.
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
        return NodeInputs(
            tensor_inputs={
                "positive_ids": inputs["text_inputs"][0],
                "negative_ids": inputs["text_inputs"][1],
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        with _no_autocast():
            return {
                "text_embeds_pos": [self._encode_one(positive_ids)],
                "text_embeds_neg": [self._encode_one(negative_ids)],
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
        # Reference keeps u[:seq_len] and re-pads with zeros; in-place zeroing
        # of the tail is the same tensor.
        embeds[:, seq_len:] = 0
        return embeds


# ---------------------------------------------------------------------------
# vae_encoder
# ---------------------------------------------------------------------------

class Wan22VaeEncoderSubmodule(_Fp32IslandMixin, NodeSubmodule):
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
        return NodeInputs(tensor_inputs={"image": inputs["image_inputs"][0]})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        image: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        with _no_autocast():
            device = self.get_device()
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
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

class Wan22DitSubmodule(_Fp32IslandMixin, NodeSubmodule):
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

    def __init__(self, transformer: nn.Module, config: Wan22Config):
        super().__init__()
        self.transformer = transformer
        self.config = config
        self._record_fp32_islands()
        # Feature 2: compile the inner DiT region only. The region takes tensors
        # (device latents, device per-token timestep grid, device text embeds) and
        # returns a tensor — nothing CPU-resident crosses it, so Inductor traces it
        # cleanly. Compile ``forward`` IN PLACE (not by wrapping the module) so
        # ``self.transformer`` stays the same nn.Module: parameter names, the fp32
        # islands, ``.dtype`` and ``.to()`` are all untouched, and the reference
        # tests' transformer-swap trick keeps working. ``dynamic=False`` traces one
        # graph per input shape, so the FIRST request at a new resolution pauses to
        # compile (logged in ``_noise_prediction``); later requests reuse the trace.
        # ``fullgraph=False`` allows a graph break at the SDPA op, matching cosmos3.
        self._compile_dit = bool(config.compile_dit)
        self._compiled_shapes: set[tuple[int, ...]] = set()
        if self._compile_dit and transformer is not None:
            transformer.forward = torch.compile(
                transformer.forward, fullgraph=False, dynamic=False,
            )
            logger.info(
                "Wan2.2 DiT: torch.compile enabled on the transformer region "
                "(fullgraph=False, dynamic=False); UniPC solver stays eager."
            )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
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
            tensor_inputs["latents"] = inputs["latents"][0]
            tensor_inputs["time_index"] = inputs["time_index"][0]
            tensor_inputs["unipc_model_outputs"] = inputs["unipc_model_outputs"][0]
            tensor_inputs["unipc_last_sample"] = inputs["unipc_last_sample"][0]
        return NodeInputs(tensor_inputs=tensor_inputs)

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

        with _no_autocast():
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
        # dynamic=False keys the compiled trace on the full input shape (batch
        # included, so CFG and no-CFG are distinct graphs). Announce the one-time
        # compile pause the first time a shape is seen, so an operator watching the
        # log knows why the first request at a new resolution is slow.
        if self._compile_dit:
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
        out = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
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
        ``k + 1 >= N``. Overshoot cannot happen: the dit node sets
        ``enable_async_scheduling=False``, and an iteration N would fault on
        the N-element timestep table anyway. ``>=`` over ``==`` is plain
        defensive arithmetic, not overshoot handling.
        """
        iter_idx = request_info.dynamic_loop_iter_counts.get(DENOISE_LOOP_NAME, 0)
        requested = int(request_info.step_metadata.get("num_inference_steps", 0) or 0)
        if requested > 0 and iter_idx + 1 >= requested:
            return {DENOISE_LOOP_NAME}
        return set()


# ---------------------------------------------------------------------------
# vae_decoder
# ---------------------------------------------------------------------------

class Wan22VaeDecoderSubmodule(_Fp32IslandMixin, NodeSubmodule):
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
    """

    disable_torch_compile = True

    def __init__(self, vae: nn.Module, config: Wan22Config):
        super().__init__()
        self.vae = vae
        self.config = config
        self._decode_dtype_cached = None
        if vae is not None:
            # Resolve the decode dtype now (a cheap cuDNN-version read) so the
            # choice is fixed at startup rather than on the first request. Cast
            # BEFORE recording the fp32 islands: the recorded set must describe
            # the dtype the decode will actually run in, not the checkpoint's
            # load dtype, or the mixin re-pins the weights back afterwards.
            self.vae = vae.to(self._decode_dtype())
        self._record_fp32_islands()

    def _decode_dtype(self):
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
        with _no_autocast():
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
            if tiled:
                video = vae.tiled_decode(latents, return_dict=False)[0]
            else:
                video = vae.decode(latents, return_dict=False)[0]
            # Quantize to 8-bit at the worker boundary (see class docstring).
            video = (video / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)
        return {"video_output": [video]}
