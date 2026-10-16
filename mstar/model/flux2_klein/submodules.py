"""FLUX.2 [klein] node submodules (four nodes, one optional engine resource).

    text_encoder -> KleinTextEncoderSubmodule  native Qwen3 taps; batches every request (fixed 512 tokens)
    vae_encoder  -> KleinVaeEncoderSubmodule   reference images -> normalized packed latent tokens (edits)
    dit          -> KleinDenoiseSubmodule      DenoiseLoopSubmodule + the klein transformer + Euler step
    vae_decoder  -> KleinVaeDecoderSubmodule   packed tokens -> uint8 image; batches equal shapes

Numerics are governed by the checkpoint dtypes (bf16 everywhere), not by the
engine: ``Flux2KleinModel.get_autocast_dtype`` returns None, so nothing is
autocast or blanket-cast, and each forward mirrors the reference pipeline's op
order (see ``notes`` in the PR description and the equivalence tests).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.compile_utils import ExactOps, compile_transformer_forward
from mstar.model.components.diffusion.decode_utils import (
    VAE_DECODE_BATCH_SIZES,
    compile_vae_decode,
    decode_in_chunks,
)
from mstar.model.components.diffusion.denoise_loop import LATENTS, DenoiseLoopSubmodule
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule, euler_step
from mstar.model.components.diffusion.image_io import (
    image_grid_ids,
    normalize_pixels,
    pack_latents,
    patchify_latents,
    pixels_to_uint8,
    text_ids,
    unpack_latents,
    unpatchify_latents,
)
from mstar.model.components.diffusion.rope import MultiAxisRoPE
from mstar.model.flux2_klein.config import Flux2KleinConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)

# Edge names shared with the model file.
TEXT_INPUTS, TEXT_MASK, TEXT_EMBEDS = "text_inputs", "text_mask", "text_embeds"
IMAGE_INPUTS, REF_LATENTS, IMAGE_OUTPUT = "image_inputs", "ref_latents", "image_output"


@dataclass(frozen=True)
class KleinShape:
    """What a klein denoise step's shape is: the latent token grid, the (fixed) text
    length and the reference-image grids. Requests batch, and accelerator graph buckets key, on it."""

    grid: tuple[int, int]                       # (h, w) latent tokens of the generated image
    text_len: int
    ref_grids: tuple[tuple[int, int], ...] = ()  # per reference image, in order

    @property
    def image_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def ref_tokens(self) -> int:
        return sum(h * w for h, w in self.ref_grids)

    @property
    def total_tokens(self) -> int:
        return self.text_len + self.image_tokens + self.ref_tokens


def shape_from_metadata(config: Flux2KleinConfig, step_metadata: dict) -> KleinShape:
    return KleinShape(
        grid=config.latent_grid(int(step_metadata["height"]), int(step_metadata["width"])),
        text_len=config.text_encoder.max_sequence_length,
        ref_grids=tuple(tuple(g) for g in step_metadata.get("ref_grids", ())),
    )


class _BatchedRows:
    """``forward_batched`` for nodes whose ``forward`` already takes a stacked batch and
    returns one output per row: split the rows back out by request id."""

    output_key: str

    def forward_batched(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs):
        out = self.forward(graph_walk, engine_inputs=engine_inputs, **kwargs)[self.output_key][0]
        return {rid: {self.output_key: [out[i:i + 1]]} for i, rid in enumerate(engine_inputs.request_ids)}


# ---------------------------------------------------------------------------
# text_encoder
# ---------------------------------------------------------------------------

class KleinTextEncoderSubmodule(_BatchedRows, NodeSubmodule):
    """Qwen3 hidden-state taps over the chat-templated, 512-token right-padded prompt.

    Every request has the same token count, so the node batches unconditionally:
    ``text_inputs`` / ``text_mask`` rows stack into ``[B, 512]`` and each request gets
    its ``[1, 512, joint_dim]`` slice of the output. All 512 positions (pad tokens
    included) are emitted, as the reference feeds them all to the DiT.
    """

    disable_torch_compile = True
    output_key = TEXT_EMBEDS

    def __init__(self, encoder: nn.Module, config: Flux2KleinConfig, max_batch_size: int = 16):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self._max_batch_size = max_batch_size

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={TEXT_INPUTS: inputs[TEXT_INPUTS][0], TEXT_MASK: inputs[TEXT_MASK][0]},
            input_seq_len=self.config.text_encoder.max_sequence_length,
        )

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {
            TEXT_INPUTS: torch.stack([inp.tensor_inputs[TEXT_INPUTS] for inp in inputs]),
            TEXT_MASK: torch.stack([inp.tensor_inputs[TEXT_MASK] for inp in inputs]),
        }

    def forward(self, graph_walk, engine_inputs, text_inputs: torch.Tensor, text_mask: torch.Tensor, **kwargs):
        device = self.get_device()
        embeds = self.encoder(text_inputs.to(device=device, dtype=torch.long), text_mask.to(device=device))
        return {TEXT_EMBEDS: [embeds.to(self.encoder.dtype)]}


# ---------------------------------------------------------------------------
# vae_encoder (reference images for editing)
# ---------------------------------------------------------------------------

class KleinVaeEncoderSubmodule(NodeSubmodule):
    """Reference images -> one packed, BatchNorm-normalized token sequence
    ``[1, R, 128]`` (all references concatenated, in request order).

    Consumes ``image_inputs``: ``[3, H, W]`` tensors already sized to the pipeline's
    rule (``Flux2KleinModel.load_image``), float in ``[0, 1]``. Images can differ in
    size, so each is encoded on its own; requests do not batch here.
    """

    disable_torch_compile = True

    def __init__(self, vae: nn.Module, config: Flux2KleinConfig):
        super().__init__()
        self.vae = vae
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={f"image_{i}": img for i, img in enumerate(inputs[IMAGE_INPUTS])})

    def forward(self, graph_walk, engine_inputs, **images: torch.Tensor):
        device = self.get_device()
        tokens = []
        for key in sorted(images, key=lambda name: int(name.split("_")[1])):
            pixels = normalize_pixels(images[key]).to(device=device, dtype=self.vae.dtype).unsqueeze(0)
            # The reference's image processor hands the VAE an NHWC-strided view (numpy HWC transposed
            # to NCHW); cuDNN picks different convolution kernels for that layout than for a dense
            # NCHW tensor, and the one-ulp differences propagate into the edit. Same layout, same
            # kernels, bit-exact reference latents.
            pixels = pixels.contiguous(memory_format=torch.channels_last)
            latent = self.vae.normalize_latents(patchify_latents(self.vae.encode(pixels)))
            tokens.append(pack_latents(latent))
        return {REF_LATENTS: [torch.cat(tokens, dim=1)]}


# ---------------------------------------------------------------------------
# dit (denoise loop body)
# ---------------------------------------------------------------------------

class KleinDenoiseSubmodule(DenoiseLoopSubmodule):
    """One klein Euler step over ``[txt | img | ref]`` tokens; see ``DenoiseLoopSubmodule``.

    Per-shape derived state is the joint rotary table ``(cos, sin)`` over the text ids
    ``(0,0,0,l)``, the image grid ``(0,h,w,0)`` and each reference grid ``(10(i+1),h,w,0)``.
    The transformer is compiled per shape when ``compile`` is on; captured graphs
    (``capture_buckets``) record those kernels together with the Euler update.
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: Flux2KleinConfig,
        *,
        loop_name: str,
        attn_resource_key: str | None,
        compile_transformer: bool = True, compile_eager_rounding: bool = True, compile_exact_ops: ExactOps = False,
        max_batch_size: int = 8,
        capture_buckets=(),
        capture_batch_sizes=(1, 2, 4, 8),
    ):
        super().__init__(
            loop_name=loop_name, max_batch_size=max_batch_size, attn_resource_key=attn_resource_key,
            capture_buckets=capture_buckets, capture_batch_sizes=capture_batch_sizes,
        )
        self.transformer = transformer
        self.config = config
        self.rope = MultiAxisRoPE(config.transformer.rope_theta, config.transformer.axes_dims_rope)
        self._compiled_shapes: set[tuple] = set()
        self._compile = bool(compile_transformer) and transformer is not None
        if self._compile:
            # In place, so the module keeps its identity (dtype property, tests' swaps);
            # dynamic=False traces one graph per (batch, shape) — announced per shape.
            compile_transformer_forward(
                transformer, eager_rounding=compile_eager_rounding, exact_ops=compile_exact_ops,
            )

    # hooks -----------------------------------------------------------------
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> KleinShape:
        return shape_from_metadata(self.config, fwd_info.step_metadata)

    def schedule_for(self, fwd_info, bucket_key: KleinShape) -> FlowMatchSchedule:
        return FlowMatchSchedule.build(
            self.config.scheduler, int(fwd_info.step_metadata["num_inference_steps"]), bucket_key.image_tokens,
        )

    def seed_to_device(self, graph_walk: str) -> bool:
        # pack_latents is a reshape + permute, so it is value-identical run after the
        # copy rather than before it (measured: test_diffusion_noise_stager.py). That
        # lets the draw go straight into the stager's pinned buffer, skipping both the
        # host memcpy and the pageable copy's host stall. Return False to fall back.
        del graph_walk
        return True

    def seed_latents(self, fwd_info, bucket_key: KleinShape, generator: torch.Generator) -> torch.Tensor:
        # randn_tensor parity: drawn in the packed [C, h, w] layout in bf16 on the CPU
        # generator, then packed row-major to tokens.
        h, w = bucket_key.grid
        shape = (1, self.config.transformer.in_channels, h, w)
        if self.seed_to_device(fwd_info.graph_walk):
            noise = self.noise_stager(self.transformer.dtype).randn_to_device(
                shape, fwd_info.random_seed, self.get_device(),
            )
        else:
            noise = torch.randn(shape, generator=generator, dtype=self.transformer.dtype)
        return pack_latents(noise)[0]

    def request_inputs(self, fwd_info, inputs: NameToTensorList, bucket_key: KleinShape) -> dict[str, torch.Tensor]:
        tensors = {TEXT_EMBEDS: inputs[TEXT_EMBEDS][0][0]}
        if bucket_key.ref_grids:
            tensors[REF_LATENTS] = inputs[REF_LATENTS][0][0]
        return tensors

    def num_tokens(self, bucket_key: KleinShape) -> int:
        return bucket_key.total_tokens

    def capture_request_inputs(self, bucket_key: KleinShape, device) -> dict[str, torch.Tensor]:
        dtype = self.transformer.dtype
        tcfg = self.config.transformer
        tensors = {
            LATENTS: torch.zeros(bucket_key.image_tokens, tcfg.in_channels, dtype=dtype, device=device),
            TEXT_EMBEDS: torch.zeros(bucket_key.text_len, tcfg.joint_attention_dim, dtype=dtype, device=device),
        }
        if bucket_key.ref_grids:
            tensors[REF_LATENTS] = torch.zeros(
                bucket_key.ref_tokens, self.config.transformer.in_channels, dtype=dtype, device=device,
            )
        return tensors

    def build_layout(self, bucket_key: KleinShape, device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [text_ids(bucket_key.text_len), image_grid_ids(*bucket_key.grid)]
        scale = self.config.ref_image_time_scale
        ids += [image_grid_ids(h, w, t=scale * (i + 1)) for i, (h, w) in enumerate(bucket_key.ref_grids)]
        cos, sin = self.rope(torch.cat(ids))
        return cos.to(device), sin.to(device)

    def denoise(self, engine_inputs, bucket_key: KleinShape, latents, timestep, sigma, sigma_next, **cond):
        text_embeds = cond[TEXT_EMBEDS]
        ref_latents = cond.get(REF_LATENTS)
        rope = self.layout(bucket_key, latents.device)
        model_input = latents if ref_latents is None else torch.cat([latents, ref_latents], dim=1)
        # The reference hands the transformer ``t.to(latents.dtype) / 1000``; the double
        # rounding through bf16 is part of what the checkpoint saw.
        timestep_in = timestep.to(latents.dtype) / 1000
        if self._compile:
            key = (tuple(model_input.shape), tuple(text_embeds.shape))
            if key not in self._compiled_shapes:
                self._compiled_shapes.add(key)
                logger.info("FLUX.2 klein: compiling the transformer for shape %s (one-time pause)", key)
        velocity = self.transformer(model_input, text_embeds, timestep_in, rope, ragged=self._ragged())
        velocity = velocity[:, : latents.shape[1]]
        return euler_step(latents, velocity, sigma, sigma_next)


# ---------------------------------------------------------------------------
# vae_decoder
# ---------------------------------------------------------------------------

class KleinVaeDecoderSubmodule(_BatchedRows, NodeSubmodule):
    """Final packed tokens ``[L, 128]`` per request -> uint8 images ``[B, 3, H, W]``.

    Unpacks to the grid, undoes the BatchNorm normalization (bf16, as the reference),
    unpatchifies to 32 channels and decodes; quantizes with the reference's
    ``round`` at the worker boundary so the edge to the data worker is one byte per
    pixel. Requests at the same grid batch.
    """

    disable_torch_compile = True
    output_key = IMAGE_OUTPUT

    def __init__(
        self, vae: nn.Module, config: Flux2KleinConfig, max_batch_size: int = 8, compile_decode: bool = False,
        warmup_grids: Sequence[tuple[int, int]] = (), decode_batch_sizes: Sequence[int] = VAE_DECODE_BATCH_SIZES,
    ):
        super().__init__()
        self.vae = vae
        self.config = config
        self._max_batch_size = max_batch_size
        # torch.compile (max-autotune, no cudagraphs) takes the 1024^2 decode from 89 to 29 ms on an
        # H100; its fused bf16 reductions move the image by <= 5.6e-2 in [-1, 1] (~56 dB), so it is a
        # deployment knob and stays off for the bit-exact parity path. CUDA graphs would add nothing:
        # the compiled decode is not launch-bound (measured), so the node declares no captures.
        self._compiled = bool(compile_decode)
        self._decode_one = compile_vae_decode(vae) if compile_decode else vae.decode
        # compiled: only these batch sizes are ever decoded (larger batches are split into them)
        self._decode_batch_sizes = tuple(decode_batch_sizes) if compile_decode else ()
        # latent shapes (batch, H, W) the compiled decode was warmed with; any other shape decodes eager:
        # a fresh max-autotune compile inside a request cost 40-100 s (measured on 1024x768 / 512^2 edits)
        self._warmed: set[tuple[int, int, int]] = set()
        # Before the warm-up, not after: the loader leaves requires_grad=True and
        # Engine.load_model only clears it later. Dynamo guards on requires_grad, so
        # a graph warmed with it set fails its guard on the first served decode and
        # recompiles under max-autotune *inside* the request -- the warm-up is thrown
        # away exactly when it matters.
        vae.requires_grad_(False)
        self.warmup(warmup_grids)

    def _decode_chunk(self, latent: torch.Tensor) -> torch.Tensor:
        key = (int(latent.shape[0]), int(latent.shape[-2]), int(latent.shape[-1]))
        if self._compiled and key not in self._warmed:
            return self.vae.decode(latent)
        return self._decode_one(latent)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return decode_in_chunks(self._decode_chunk, latents, self._decode_batch_sizes)

    def warmup(self, grids: Sequence[tuple[int, int]]) -> None:
        """Decode zeros at every decode batch size for every token grid at load, so the compiles
        and their autotuning happen before the first request (only when the decode is compiled)."""
        if not self._compiled:
            return
        for h, w in grids:
            patch = self.config.vae.patch_size
            for bs in self._decode_batch_sizes:
                latent = torch.zeros(
                    bs, self.config.vae.latent_channels, h * patch[0], w * patch[1],
                    device=self.get_device(), dtype=self.vae.dtype,
                )
                with torch.no_grad():
                    self._decode_one(latent)
                self._warmed.add((int(latent.shape[0]), int(latent.shape[-2]), int(latent.shape[-1])))

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        grid = self.config.latent_grid(int(fwd_info.step_metadata["height"]), int(fwd_info.step_metadata["width"]))
        return NodeInputs(tensor_inputs={LATENTS: inputs[LATENTS][0]}, resource_step_info=grid)

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1 and len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {
            LATENTS: torch.stack([inp.tensor_inputs[LATENTS] for inp in inputs]),
            "grid": inputs[0].resource_step_info,
        }

    def forward(self, graph_walk, engine_inputs, latents: torch.Tensor, grid: tuple[int, int], **kwargs):
        latents = latents.to(device=self.get_device(), dtype=self.vae.dtype)
        patched = self.vae.denormalize_latents(unpack_latents(latents, *grid))
        image = self._decode(unpatchify_latents(patched))
        return {IMAGE_OUTPUT: [pixels_to_uint8(image)]}
