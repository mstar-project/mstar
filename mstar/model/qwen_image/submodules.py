"""Qwen-Image node submodules.

    text_encoder -> QwenImageTextEncoderSubmodule  Qwen2.5-VL: the vision tower over the condition images, then
                                                   the language model over each prompt with them spliced in
    vae_encoder  -> QwenImageVaeEncoderSubmodule   reference images -> normalized packed latent tokens
    dit          -> QwenImageDenoiseSubmodule      DenoiseLoopSubmodule + the transformer; guided by the scaffold
    vae_decoder  -> QwenImageVaeDecoderSubmodule   packed tokens -> uint8 image

Numerics follow the checkpoint dtypes (bf16) and the reference pipeline's op order;
``QwenImageModel.get_autocast_dtype`` is None.
"""

from __future__ import annotations

import functools
import logging
from typing import NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.batched_rows import BatchedRows
from mstar.model.components.diffusion.attention import joint_attention
from mstar.model.components.diffusion.compile_utils import compile_blocks
from mstar.model.components.diffusion.decode_utils import latent_affine, latent_std
from mstar.model.components.diffusion.denoise_loop import LATENTS, DenoiseLoopSubmodule, StepKey
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.components.diffusion.guidance import Guidance, cfg_renorm
from mstar.model.components.diffusion.image_io import (
    pack_latents,
    patchify_latents,
    pixels_to_uint8,
    unpack_latents,
    unpatchify_latents,
    vae_pixels,
)
from mstar.model.components.diffusion.qwen.vl_prompt import VisionTokens
from mstar.model.components.qwen_vision import VISION_ATTN, FrameSDPA, qwen25_tower_inputs
from mstar.model.keys import VISION_FULL
from mstar.model.qwen_image.components.transformer import QwenStepLayout, qwen_rope_tables
from mstar.model.qwen_image.config import QwenImageConfig
from mstar.model.submodule_base import NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking

logger = logging.getLogger(__name__)

# Edge names shared with the model file.
TEXT_INPUTS = "text_inputs"
VL_PIXELS = "vl_pixels"
IMAGE_INPUTS = "image_inputs"
TEXT_EMBEDS_POS, TEXT_EMBEDS_NEG = "text_embeds_pos", "text_embeds_neg"
REF_LATENTS = "ref_latents"
IMAGE_OUTPUT = "image_output"
# The condition images' patch grids, which only process_prompt sets (an internal model kwarg) and the walks carry as
# step metadata, and each prompt's token count.
VL_GRIDS, PROMPT_LENGTHS = "qwen_image_vl_grids", "prompt_lengths"


def pack(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` latents -> ``[B, H/2 * W/2, 4C]`` tokens (the pipelines' ``_pack_latents``)."""
    return pack_latents(patchify_latents(latents))


def unpack(tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """The inverse of :func:`pack` for a ``grid`` of tokens (``_unpack_latents``, without its frame axis)."""
    return unpatchify_latents(unpack_latents(tokens, *grid))


# ---------------------------------------------------------------------------
# text_encoder
# ---------------------------------------------------------------------------

class QwenImageTextEncoderSubmodule(NodeSubmodule):
    """The condition images through the Qwen2.5-VL tower once, then each prompt (the positive, and the
    negative when guided) alone through the language model with the images' tokens spliced in: the
    reference's batch-1, mask-free calls. Emits each prompt's normed last hidden states past the
    template's dropped tokens, ``[L - drop, hidden]`` bf16. Unguided, the negative edge carries a
    one-element placeholder the dit never reads."""

    disable_torch_compile = True

    def __init__(self, encoder: nn.Module, vision: nn.Module, config: QwenImageConfig):
        super().__init__()
        self.encoder = encoder
        self.vision = vision
        self.config = config
        # The tower runs once per request on a few thousand patches: SDPA within each window or frame, a stand-in
        # per layout, not a node resource.
        self._windows, self._frames = FrameSDPA(), FrameSDPA()
        for block in vision.blocks:
            block.attn.bind_resources({VISION_ATTN: self._frames if block.attn.label == VISION_FULL else self._windows})

    def max_batch_size(self, graph_walk: str):
        return 1

    def can_batch(self, batch, model_inputs) -> bool:
        return False

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        grids = tuple(tuple(int(v) for v in g) for g in fwd_info.step_metadata[VL_GRIDS])
        prompts = inputs[TEXT_INPUTS]
        return NodeInputs(
            tensor_inputs={VL_PIXELS: inputs[VL_PIXELS][0], **{f"prompt_{i}": ids for i, ids in enumerate(prompts)}},
            input_seq_len=sum(int(ids.shape[0]) for ids in prompts), resource_step_info=grids,
        )

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {**inputs[0].tensor_inputs, "grids": inputs[0].resource_step_info}

    def forward(self, graph_walk, engine_inputs, vl_pixels: torch.Tensor, grids, **prompts) -> NameToTensorList:
        device = self.get_device()
        images = self.encode_images(vl_pixels.to(device), grids)
        embeds = [self.encode_prompt(prompts[f"prompt_{i}"].to(device), images) for i in range(len(prompts))]
        neg = embeds[1] if len(embeds) > 1 else torch.zeros(1, dtype=embeds[0].dtype, device=device)
        return {TEXT_EMBEDS_POS: [embeds[0]], TEXT_EMBEDS_NEG: [neg]}

    def encode_images(self, pixels: torch.Tensor, grids) -> VisionTokens:
        tower = qwen25_tower_inputs(grids, self.vision.config)
        self._windows.lengths, self._frames.lengths = list(tower.window_lengths), list(tower.frame_lengths)
        device = pixels.device
        merged = self.vision(
            pixels.to(self.vision.patch_embed.proj.weight.dtype), to_device_nonblocking(tower.rope_positions, device),
            to_device_nonblocking(tower.window_index, device), to_device_nonblocking(tower.restore_index, device),
        )
        return VisionTokens(grids, self.vision.config.spatial_merge_size, merged)

    def encode_prompt(self, ids: torch.Tensor, images: VisionTokens) -> torch.Tensor:
        ids = ids.to(torch.long)
        hidden = self.encoder(ids[None], torch.ones_like(ids[None]), padded=False, images=images)
        return hidden[0, self.config.recipe.drop_tokens:].to(self.encoder.dtype)


# ---------------------------------------------------------------------------
# vae_encoder
# ---------------------------------------------------------------------------

# the image size (pixels) the compiled encoder and decoder are warmed on. Dynamo gives equal sizes one symbol (duck
# sizing), and a symbol tied to a fixed size is fixed: height == width would tie the two, and a side equal to a
# channel count would pin it, so every other size would recompile on a live request
_WARM_PIXELS = (448, 320)


def _encode_mode(vae: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """``vae.encode(pixels).latent_dist.mode()`` for one frame, bit for bit, without its feature cache: a one-frame
    encode only writes the cache (a clone of every conv input), never reads it back."""
    return vae.quant_conv(vae.encoder(pixels)).chunk(2, dim=1)[0]


def _decode(vae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """``vae.decode(z)`` for one frame, bit for bit, without its feature cache (as :func:`_encode_mode`)."""
    return vae.decoder(vae.post_quant_conv(z)).clamp(-1.0, 1.0)


class QwenImageVaeEncoderSubmodule(NodeSubmodule):
    """Reference images, uint8 ``[3, H, W]`` at the VAE's size each -> one packed sequence of normalized latent
    tokens ``[sum(h*w), 64]``, the references in order: the posterior's mode, ``(z - mean) / std`` in the
    VAE's dtype, as the pipeline encodes, from pixels laid out as its image processor hands them
    (:func:`vae_pixels`). Images differ in size, so each runs alone."""

    disable_torch_compile = True

    def __init__(self, vae: nn.Module, compile: bool = False):
        super().__init__()
        self.vae = vae
        self._compile = compile

    def compile_regions(self) -> dict[str, list[tuple]]:
        if not self._compile:
            return {}
        # laid out as forward's input (vae_pixels' NHWC strides): Dynamo guards on strides
        image = torch.zeros(3, *_WARM_PIXELS, dtype=torch.uint8)
        return {"vae.encoder": [(vae_pixels(image).to(device=self.get_device(), dtype=self.vae.dtype),)]}

    def max_batch_size(self, graph_walk: str):
        return 1

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={f"image_{i}": img for i, img in enumerate(inputs[IMAGE_INPUTS])})

    def forward(self, graph_walk, engine_inputs, **images: torch.Tensor) -> NameToTensorList:
        device, dtype = self.get_device(), self.vae.dtype
        mean, std = latent_affine(self.vae, dtype, device)[0], latent_std(self.vae, dtype, device)
        tokens = []
        for key in sorted(images, key=lambda name: int(name.split("_")[1])):
            pixels = vae_pixels(images[key]).to(device=device, dtype=dtype)  # [1, 3, 1, H, W], NHWC-strided
            latent = (_encode_mode(self.vae, pixels) - mean) / std
            tokens.append(pack(latent[:, :, 0])[0])
        return {REF_LATENTS: [torch.cat(tokens)]}


# ---------------------------------------------------------------------------
# dit
# ---------------------------------------------------------------------------

class QwenEditKey(NamedTuple):
    """What has to match for two rows to share a dit forward: the target's and each reference's token grid,
    and each prompt's length (the negative's 0 when the request is unguided)."""

    grid: tuple[int, int]
    ref_grids: tuple[tuple[int, int], ...]
    text_lengths: tuple[int, int]

    @property
    def image_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def joint_image_tokens(self) -> int:
        return self.image_tokens + sum(h * w for h, w in self.ref_grids)


def text_rows(key: StepKey) -> int:
    """The step's text length: the longest of this rank's branches. Shorter ones are left-padded to it."""
    return max(key.bucket.text_lengths[b] for b in key.branches)


class QwenImageDenoiseSubmodule(DenoiseLoopSubmodule):
    """One Euler step of the Qwen-Image transformer; the scaffold runs the guidance branches (the positive and
    the negative prompt) as rows and combines them with Qwen-Image's norm-rescaled CFG.

    Loop state is the packed bf16 latent ``[h*w, 64]``, as the reference carries it. A row's input is the
    latent followed by the request's reference tokens; each branch's text is left-padded to the step's text
    length, and the pad attends only itself (``attention_segments``)."""

    GUIDANCE = Guidance(
        branches=({"text_embeds": TEXT_EMBEDS_POS}, {"text_embeds": TEXT_EMBEDS_NEG}), combine=cfg_renorm,
    )
    # the noise is a function of the latent grid and the seed alone
    PREFETCH_SEEDS = True

    def __init__(
        self, transformer: nn.Module, config: QwenImageConfig, *, loop_name: str, attn_resource_key: str | None,
        compile_transformer: bool = False, cfg_group=None,
    ):
        super().__init__(loop_name=loop_name, attn_resource_key=attn_resource_key, cfg_group=cfg_group)
        self.transformer = transformer
        self.config = config
        # SDPA within each segment where no ragged resource is bound (the reference's own kernel)
        self._sdpa = FrameSDPA()
        self._compiled = bool(compile_transformer) and transformer is not None
        if self._compiled:
            # per block, static except the text axis: a new image shape compiles the one block graph once, and any
            # prompt length reuses it (one graph for lengths 9-50, checked on CPU)
            compile_blocks(transformer.transformer_blocks, dynamic=False)
            transformer.dynamic_text = True

    # hooks -----------------------------------------------------------------
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> QwenEditKey:
        meta = fwd_info.step_metadata
        lengths = [int(n) - self.config.recipe.drop_tokens for n in meta[PROMPT_LENGTHS]]
        return QwenEditKey(
            grid=self.config.latent_grid(int(meta["height"]), int(meta["width"])),
            ref_grids=tuple(tuple(int(v) for v in g) for g in meta["ref_grids"]),
            text_lengths=(lengths[0], lengths[1] if len(lengths) > 1 else 0),
        )

    def schedule_for(self, fwd_info, bucket_key: QwenEditKey) -> FlowMatchSchedule:
        steps = int(fwd_info.step_metadata["num_inference_steps"])
        return FlowMatchSchedule.build(self.config.scheduler, steps, bucket_key.image_tokens)

    def seed_latents(self, fwd_info, bucket_key: QwenEditKey, generator: torch.Generator) -> torch.Tensor:
        # randn_tensor parity: (1, 1, C, H, W) in the transformer's dtype on the CPU generator, then packed; C is
        # the VAE's latent channels, which the transformer outputs
        h, w = bucket_key.grid
        shape = (1, 1, self.config.transformer.out_channels, 2 * h, 2 * w)
        return pack(torch.randn(shape, generator=generator, dtype=self.transformer.dtype)[:, 0])[0]

    def request_inputs(self, fwd_info, inputs: NameToTensorList, key: StepKey) -> dict[str, torch.Tensor]:
        rows = text_rows(key)
        tensors = {REF_LATENTS: inputs[REF_LATENTS][0]}
        for b in key.branches:  # each run branch's prompt, left-padded to the step's rows (Guidance.branches)
            edge = self.GUIDANCE.branches[b]["text_embeds"]
            embeds = inputs[edge][0]
            tensors[edge] = torch.nn.functional.pad(embeds, (0, 0, rows - embeds.shape[0], 0))
        return tensors

    def num_tokens(self, key: StepKey) -> int:
        return text_rows(key) + key.bucket.joint_image_tokens

    def attention_segments(self, key: StepKey):
        rows, image = text_rows(key), key.bucket.joint_image_tokens
        segments = []
        for b in key.branches:
            length = key.bucket.text_lengths[b]
            if rows > length:
                segments.append((self.MAIN_SPAN, rows - length))
            segments.append((self.MAIN_SPAN, length + image))
        return tuple(segments)

    def capture_request_inputs(self, key: StepKey, device) -> dict[str, torch.Tensor]:
        raise NotImplementedError("the Qwen-Image dit is not captured: a step is GPU-bound at this size")

    def build_layout(self, key: StepKey, device) -> QwenStepLayout:
        bucket: QwenEditKey = key.bucket
        img_freqs, txt_freqs = qwen_rope_tables(
            self.config.transformer.axes_dims_rope, [bucket.grid, *bucket.ref_grids],
            [bucket.text_lengths[b] for b in key.branches], text_rows(key),
        )
        return QwenStepLayout(img_freqs.to(device), txt_freqs.to(device), bucket.image_tokens)

    def model_output(self, engine_inputs, key: StepKey, latents, timestep, text_embeds, ref_latents, **cond):
        layout = self.layout(key, latents.device)
        attend = self._attend(key, latents.shape[0])
        # the reference hands the transformer t.to(latents.dtype) / 1000, rounded through bf16
        timestep_in = timestep.to(latents.dtype) / 1000
        out = self.transformer(
            torch.cat([latents, ref_latents.to(latents.dtype)], dim=1), text_embeds.to(self.transformer.dtype),
            timestep_in, layout, attend,
        )
        return out[:, :latents.shape[1]]

    def _attend(self, key: StepKey, rows: int):
        ragged = self.ragged_for(self.MAIN_SPAN)
        if ragged is None:
            spans = [span for _, span in self.attention_segments(key)]
            self._sdpa.lengths = spans * (rows // len(key.branches))
            ragged = self._sdpa.run
        return functools.partial(joint_attention, ragged=ragged)


# ---------------------------------------------------------------------------
# vae_decoder
# ---------------------------------------------------------------------------

class QwenImageVaeDecoderSubmodule(BatchedRows, NodeSubmodule):
    """Final packed tokens ``[h*w, 64]`` per request -> uint8 images ``[B, 3, H, W]``: unpacked, ``z / (1 / std) +
    mean`` in the VAE's dtype, decoded as one frame, quantized with the reference's rounding. Requests at the same
    grid batch."""

    disable_torch_compile = True
    output_keys = (IMAGE_OUTPUT,)
    keep_batch_dim = True

    def __init__(self, vae: nn.Module, config: QwenImageConfig, max_batch_size: int = 4, compile: bool = False):
        super().__init__()
        self.vae = vae
        self.config = config
        self._max_batch_size = max_batch_size
        self._compile = compile

    def compile_regions(self) -> dict[str, list[tuple]]:
        if not self._compile:
            return {}
        # built as run_batch builds it, at batch 1 and 2: Dynamo specializes a size of 1, so the batched decodes
        # (2 up to max_batch_size) need their own warm-up
        grid = self.config.latent_grid(*_WARM_PIXELS)
        tokens = torch.zeros(2, grid[0] * grid[1], 4 * self.vae.config.z_dim)
        return {"vae.decoder": [(self._vae_latents(tokens[:1], grid),), (self._vae_latents(tokens, grid),)]}

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        grid = self.config.latent_grid(int(fwd_info.step_metadata["height"]), int(fwd_info.step_metadata["width"]))
        return NodeInputs(tensor_inputs={LATENTS: inputs[LATENTS][0]}, resource_step_info=grid)

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1 and len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        latents = torch.stack([inp.tensor_inputs[LATENTS] for inp in inputs])
        return {LATENTS: latents, "grid": inputs[0].resource_step_info}

    def _vae_latents(self, latents: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Packed tokens ``[B, h*w, 64]`` -> the decoder's input ``[B, C, 1, H, W]``: unpacked, ``z / (1 / std) + mean``."""
        device, dtype = self.get_device(), self.vae.dtype
        mean, inv_std = latent_affine(self.vae, dtype, device)
        return unpack(latents.to(device=device, dtype=dtype), grid)[:, :, None] / inv_std + mean

    def run_batch(self, latents: torch.Tensor, grid: tuple[int, int], **kwargs):
        image = _decode(self.vae, self._vae_latents(latents, grid))[:, :, 0]
        return {IMAGE_OUTPUT: pixels_to_uint8(image)}
