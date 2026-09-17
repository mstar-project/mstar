"""Z-Image-Turbo node submodules on the DiT scaffold.

    text_encoder -> ZImageTextEncoderSubmodule  Qwen3-4B layer-35 hidden states, unpadded prompt length
    dit          -> ZImageDenoiseSubmodule      DenoiseLoopSubmodule + the Z-Image transformer, fp32 Euler
    vae_decoder  -> ZImageVaeDecoderSubmodule   FLUX.1 VAE decode (scaling/shift latents) -> uint8 image

Layout facts the reference pipeline pins (see ``components/transformer.py``): image
patches and caption features are each padded to a multiple of 32 with learned pad
tokens; captions keep only their real tokens (pads dropped after the encoder); the
caption's rotary axis-0 positions are ``1..cap_len``, the image's is ``cap_len + 1``, and
image pad tokens sit at ``(0, 0, 0)``. Latents stay fp32 across steps (the velocity is
negated and the Euler update runs in fp32); the transformer runs in bf16.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.denoise_loop import LATENTS, DenoiseLoopSubmodule
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule, euler_step
from mstar.model.components.diffusion.image_io import pixels_to_uint8
from mstar.model.submodule_base import NodeInputs, NodeSubmodule
from mstar.model.z_image.components.transformer import ZImageRoPE, patchify_image, unpatchify_image
from mstar.model.z_image.config import SEQ_MULTIPLE, ZImageConfig

logger = logging.getLogger(__name__)

TEXT_INPUTS, TEXT_EMBEDS, IMAGE_OUTPUT = "text_inputs", "text_embeds", "image_output"
CAP_PAD_MASK = "cap_pad_mask"


def padded_length(n: int) -> int:
    return n + (-n) % SEQ_MULTIPLE


@dataclass(frozen=True)
class ZShape:
    """Batching / capture key: the latent token grid and the padded caption length."""

    grid: tuple[int, int]
    cap_len: int  # multiple of SEQ_MULTIPLE

    @property
    def image_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def image_tokens_padded(self) -> int:
        return padded_length(self.image_tokens)

    @property
    def total_tokens(self) -> int:
        return self.image_tokens_padded + self.cap_len


def shape_from_metadata(config: ZImageConfig, step_metadata: dict) -> ZShape:
    return ZShape(
        grid=config.latent_grid(int(step_metadata["height"]), int(step_metadata["width"])),
        cap_len=int(step_metadata["cap_len"]),
    )


# ---------------------------------------------------------------------------
# text_encoder
# ---------------------------------------------------------------------------

class ZImageTextEncoderSubmodule(NodeSubmodule):
    """Caption features ``[1, cap_len, 2560]`` for one request: Qwen3 hidden states after
    layer 35 at the real token positions, zero elsewhere (the DiT swaps those rows for its
    learned pad token). ``text_inputs`` arrives unpadded; rows are right-padded to the
    batch's longest padded length, so requests of any length batch together — the hidden
    states of real tokens do not depend on how much padding follows them."""

    disable_torch_compile = True

    def __init__(self, encoder: nn.Module, config: ZImageConfig, max_batch_size: int = 16):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self._max_batch_size = max_batch_size

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        ids = inputs[TEXT_INPUTS][0]
        return NodeInputs(tensor_inputs={TEXT_INPUTS: ids}, input_seq_len=int(ids.shape[0]))

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        lengths = [int(inp.tensor_inputs[TEXT_INPUTS].shape[0]) for inp in inputs]
        width = padded_length(max(lengths))
        pad_id = self.config.text_encoder.pad_token_id
        ids = torch.full((len(inputs), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(inputs), width), dtype=torch.long)
        for row, inp in enumerate(inputs):
            n = lengths[row]
            ids[row, :n] = inp.tensor_inputs[TEXT_INPUTS].to(torch.long)
            mask[row, :n] = 1
        return {TEXT_INPUTS: ids, "text_mask": mask, "lengths": lengths}

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = self.get_device()
        hidden = self.encoder(ids.to(device), mask.to(device)).to(self.encoder.dtype)
        # zero the pad rows: the DiT replaces them, and the parity tests compare real rows only
        return hidden * mask.to(device=device, dtype=hidden.dtype)[..., None]

    def forward(self, graph_walk, engine_inputs, text_inputs, text_mask, lengths, **kwargs):
        hidden = self._encode(text_inputs, text_mask)
        return {TEXT_EMBEDS: [hidden[:, : padded_length(lengths[0])]]}

    def forward_batched(self, graph_walk, engine_inputs, text_inputs, text_mask, lengths, **kwargs):
        hidden = self._encode(text_inputs, text_mask)
        return {
            rid: {TEXT_EMBEDS: [hidden[i : i + 1, : padded_length(lengths[i])]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }


# ---------------------------------------------------------------------------
# dit
# ---------------------------------------------------------------------------

class ZImageDenoiseSubmodule(DenoiseLoopSubmodule):
    """One Z-Image Euler step; see ``DenoiseLoopSubmodule``.

    Loop state is the fp32 latent ``[16, H/8, W/8]`` (the reference keeps latents fp32 and
    casts to bf16 only for the transformer). Per-shape layout: image-token pad mask and the
    complex rotary tables for the image and caption spans.
    """

    def __init__(
        self, transformer: nn.Module, config: ZImageConfig, *, loop_name: str, attn_resource_key: str | None,
        compile_transformer: bool = True, max_batch_size: int = 8, capture_shapes=(), capture_batch_sizes=(1, 2, 4, 8),
    ):
        super().__init__(
            loop_name=loop_name, max_batch_size=max_batch_size, attn_resource_key=attn_resource_key,
            capture_shapes=capture_shapes, capture_batch_sizes=capture_batch_sizes,
        )
        self.transformer = transformer
        self.config = config
        tcfg = config.transformer
        self.rope = ZImageRoPE(tcfg.rope_theta, tcfg.axes_dims, tcfg.axes_lens)
        self._compile = bool(compile_transformer) and transformer is not None
        if self._compile:
            transformer.forward = torch.compile(transformer.forward, fullgraph=False, dynamic=False)

    def shape_key_for(self, fwd_info: CurrentForwardPassInfo) -> ZShape:
        return shape_from_metadata(self.config, fwd_info.step_metadata)

    def schedule_for(self, fwd_info, shape_key: ZShape) -> FlowMatchSchedule:
        return FlowMatchSchedule.build(
            self.config.scheduler, int(fwd_info.step_metadata["num_inference_steps"]), shape_key.image_tokens,
        )

    def seed_latents(self, fwd_info, shape_key: ZShape, generator: torch.Generator) -> torch.Tensor:
        h, w = shape_key.grid
        patch = self.config.transformer.patch_size
        # randn_tensor parity: fp32 on the CPU generator, in the unpacked [C, H/8, W/8] layout
        return torch.randn(
            (self.config.transformer.in_channels, h * patch, w * patch), generator=generator, dtype=torch.float32,
        )

    def request_inputs(self, fwd_info, inputs: NameToTensorList, shape_key: ZShape) -> dict[str, torch.Tensor]:
        embeds = inputs[TEXT_EMBEDS][0][0]
        text_len = int(fwd_info.step_metadata["text_len"])
        pad_mask = torch.arange(shape_key.cap_len, device=embeds.device) >= text_len
        return {TEXT_EMBEDS: embeds, CAP_PAD_MASK: pad_mask}

    def num_tokens(self, shape_key: ZShape) -> int:
        return shape_key.total_tokens

    def capture_request_inputs(self, shape_key: ZShape, device) -> dict[str, torch.Tensor]:
        tcfg = self.config.transformer
        h, w = shape_key.grid
        return {
            LATENTS: torch.zeros(tcfg.in_channels, h * tcfg.patch_size, w * tcfg.patch_size, device=device),
            TEXT_EMBEDS: torch.zeros(shape_key.cap_len, tcfg.cap_feat_dim, dtype=self.transformer.dtype, device=device),
            CAP_PAD_MASK: torch.zeros(shape_key.cap_len, dtype=torch.bool, device=device),
        }

    def build_layout(self, shape_key: ZShape, device):
        h, w = shape_key.grid
        n_img, n_pad = shape_key.image_tokens, shape_key.image_tokens_padded - shape_key.image_tokens
        cap_ids = torch.zeros(shape_key.cap_len, 3, dtype=torch.long)
        cap_ids[:, 0] = torch.arange(1, shape_key.cap_len + 1)
        hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        img_ids = torch.stack([torch.full((n_img,), shape_key.cap_len + 1), hh.flatten(), ww.flatten()], dim=-1)
        img_ids = torch.cat([img_ids, torch.zeros(n_pad, 3, dtype=torch.long)])
        image_pad_mask = torch.zeros(shape_key.image_tokens_padded, dtype=torch.bool)
        image_pad_mask[n_img:] = True
        return {
            "image_freqs": self.rope(img_ids).to(device), "caption_freqs": self.rope(cap_ids).to(device),
            "image_pad_mask": image_pad_mask.to(device),
        }

    def denoise(self, engine_inputs, shape_key: ZShape, latents, timestep, sigma, sigma_next, **cond):
        layout = self.layout(shape_key, latents.device)
        patch = self.config.transformer.patch_size
        tokens = torch.stack([patchify_image(row, patch) for row in latents.to(self.transformer.dtype)])
        n_pad = shape_key.image_tokens_padded - shape_key.image_tokens
        if n_pad:
            tokens = torch.cat([tokens, tokens[:, -1:].expand(-1, n_pad, -1)], dim=1)
        # the reference conditions on (1000 - t) / 1000 with the scheduler's fp32 timestep
        t_cond = (1000.0 - timestep) / 1000.0
        out = self.transformer(
            tokens, cond[TEXT_EMBEDS], cond[CAP_PAD_MASK], layout["image_pad_mask"].expand(tokens.shape[0], -1),
            t_cond, layout["image_freqs"], layout["caption_freqs"], ragged=self._ragged(),
        )
        channels = self.config.transformer.in_channels
        velocity = -torch.stack([unpatchify_image(row, shape_key.grid, patch, channels) for row in out.float()])
        return euler_step(latents, velocity, sigma, sigma_next)


# ---------------------------------------------------------------------------
# vae_decoder
# ---------------------------------------------------------------------------

class ZImageVaeDecoderSubmodule(NodeSubmodule):
    """fp32 latent ``[16, H/8, W/8]`` -> ``z / scaling + shift`` in the VAE dtype -> decode -> uint8."""

    disable_torch_compile = True

    def __init__(self, vae: nn.Module, config: ZImageConfig, max_batch_size: int = 8):
        super().__init__()
        self.vae = vae
        self.config = config
        self._max_batch_size = max_batch_size

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        latents = inputs[LATENTS][0]
        return NodeInputs(tensor_inputs={LATENTS: latents}, resource_step_info=tuple(latents.shape[-2:]))

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1 and len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {LATENTS: torch.stack([inp.tensor_inputs[LATENTS] for inp in inputs])}

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.get_device(), dtype=self.vae.dtype)
        return pixels_to_uint8(self.vae.decode(self.vae.unscale_latents(latents)))

    def forward(self, graph_walk, engine_inputs, latents: torch.Tensor, **kwargs):
        return {IMAGE_OUTPUT: [self._decode(latents)]}

    def forward_batched(self, graph_walk, engine_inputs, latents: torch.Tensor, **kwargs):
        images = self._decode(latents)
        return {rid: {IMAGE_OUTPUT: [images[i : i + 1]]} for i, rid in enumerate(engine_inputs.request_ids)}
