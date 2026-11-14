# Copyright (c) 2026 Robbyant Team.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the causal-fast chunk loop split
# into M* graph-node submodules, one chunk per rollout iteration.
#
# Original file (Robbyant/lingbot-world, wan/image2video_fast.py) was released
# under Apache-2.0, with the full license text available at
# https://github.com/Robbyant/lingbot-world/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

"""LingBot-World-V2 node submodules.

    text_encoder -> wan22's ``Wan22TextEncoderSubmodule`` (UMT5-XXL), reused as is
    dit          -> LingBotDitSubmodule (one chunk per rollout iteration: the
                    chunk's I2V condition and Plucker rays, the DiT, the decode)

The dit declares its state (``LingBotWorldModel.declare_state``): the
self-attention K/V of each chunk in ``DIT_KV``, of which the engine's rolling
store keeps the sink and the latest window; each chunk's text K/V in ``TEXT``, which
no step keeps; and where the request's draws stand in ``PROGRESS``. The VAE
encoder's and decoder's causal-conv caches are ``VAE_CACHES``, a sized Value:
the engine admits a rollout by their bytes at its latent size and holds them
from one chunk to the next.

Numerics are the model's, not the engine's: ``LingBotWorldModel.get_autocast_dtype``
is bf16, which sizes the K/V pool and the attention kernels, and no node runs
under autocast. The DiT enters the reference's bf16 autocast itself, and the VAE
encode runs fp32 as the reference does; the decode runs bf16 (cuDNN >= 9.16)
unless ``reference_numerics``.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import ArrayStep, Entry, Segment, Size, SubmoduleStep, TensorType, host
from mstar.model.components import wan_vae
from mstar.model.components.diffusion.decode_utils import latent_affine, wan_vae_decode_dtype
from mstar.model.components.diffusion.image_io import truncate_to_uint8
from mstar.model.lingbot_world.components.camera import plucker_embedding
from mstar.model.lingbot_world.components.dit import DIT_KV, TEXT, LingBotCausalDiT
from mstar.model.lingbot_world.components.schedule import add_noise, flow_to_x0, make_fast_schedule
from mstar.model.lingbot_world.config import LingBotWorldConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule, StepOrder

logger = logging.getLogger(__name__)

ROLLOUT_LOOP_NAME = "rollout_loop"
ROLLOUT_WALK = "rollout"

# the declared state beside the blocks' arrays: where the request's draws stand, and the VAE's causal-conv caches
PROGRESS, VAE_CACHES = "progress", "vae_caches"
# VAE_CACHES' fields: the encoder's slots and its steady output, and the decoder's slots
ENCODER, STEADY, DECODER = "enc", "enc_steady", "dec"
# the reads of DIT_KV and TEXT
SELF_ATTN, TEXT_ATTN = "self_attn", "text_attn"
# what a chunk adds to its outputs under keep_chunk_outputs
CHUNK_OUTPUTS = ("chunk_latents", "chunk_pixels", "chunk_condition")


def vae_cache_entry(config: LingBotWorldConfig) -> Entry:
    """``VAE_CACHES``' fields at a request's latent size: the encoder's cache and its steady output, which the encode
    runs in fp32, and the decoder's cache in the decode's dtype."""
    if config.chunk_size < 2:
        raise ValueError(
            f"LingBotWorldConfig.chunk_size is {config.chunk_size}, and a chunk of one latent ends with diffusers' "
            f"'Rep' marker in the decoder's temporal upsamples, which {VAE_CACHES!r} has no field for; make it at "
            "least 2")
    decode = torch.float32 if config.reference_numerics else torch.bfloat16
    return Entry({
        **wan_vae.stream_cache_fields("encoder", torch.float32, ENCODER),
        STEADY: TensorType(
            (1, 2 * config.vae_z_dim, 1, Size(wan_vae.LAT_H), Size(wan_vae.LAT_W)), dtype=torch.float32,
        ),
        **wan_vae.stream_cache_fields("decoder", decode, DECODER),
    })


def first_frame_mask(num_latent_frames: int, lat_h: int, lat_w: int, device) -> torch.Tensor:
    """The reference's ``[4, lat_f, lat_h, lat_w]`` I2V mask: the first video
    frame is known, repeated over the first latent's four temporal slots."""
    num_frames = (num_latent_frames - 1) * 4 + 1
    msk = torch.ones(1, num_frames, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    return msk.transpose(1, 2)[0]


class EncoderStream:
    """A request's place in its condition encode: the encoder's causal-conv
    cache, then, once the black frames have driven that cache to a fixed
    point, the one encoder output every later call would return.

    It goes from one chunk to the next in ``VAE_CACHES``: the cache's slots,
    or the one steady output in their place. ``held`` is what the last chunk
    left there, nothing at a stream's first."""

    def __init__(self, vae: nn.Module, held: Mapping[str, torch.Tensor | None] | None = None):
        held = held or {}
        counts = getattr(vae, "_cached_conv_counts", None)
        if counts is not None:
            num_convs = counts["encoder"]
        else:
            num_convs = sum(isinstance(m, wan_vae.WanCausalConv3d) for m in vae.encoder.modules())
        self.steady: torch.Tensor | None = held.get(STEADY)
        # One slot per causal conv, as ``AutoencoderKLWan.clear_cache`` builds it.
        self.cache: list | None = None
        if self.steady is None:
            self.cache = wan_vae.cache_from_fields(num_convs, held, ENCODER)
        # whether a call on this stream object reached the steady output, so the slots it was given go
        self.settled = False

    def fields(self) -> dict[str, torch.Tensor]:
        """What the next chunk continues from, by ``VAE_CACHES``' field names."""
        return {STEADY: self.steady} if self.steady is not None else wan_vae.cache_fields(self.cache, ENCODER)

    def encode_black(self, vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """One encoder call on black frames. When a call leaves every cache
        slot equal to what it held before, the next call sees the same input
        and the same cache, so it, and every call after it, returns this
        call's output; the cache is dropped from then on. The encoder replaces
        its slots rather than writing into them; a slot it wrote in place
        never counts as unchanged."""
        if self.steady is not None:
            return self.steady
        before = list(self.cache)
        out = vae.encoder(x, feat_cache=self.cache, feat_idx=[0])
        if all(_same_slot(a, b) for a, b in zip(before, self.cache, strict=True)):
            self.steady, self.cache, self.settled = out, None, True
        return out


def _same_slot(before, after) -> bool:
    if torch.is_tensor(before) and torch.is_tensor(after):
        return before is not after and torch.equal(before, after)
    return not torch.is_tensor(before) and not torch.is_tensor(after) and before == after


def encode_condition_chunk(
    vae: nn.Module, image: torch.Tensor, stream: EncoderStream, first_latent: int, num_latents: int,
    config: LingBotWorldConfig,
) -> torch.Tensor:
    """Latent frames ``[first_latent, first_latent + num_latents)`` of the I2V
    condition, ``[20, num_latents, lat_h, lat_w]``: the first-frame mask over
    the Wan2.1 VAE latents of ``image`` followed by black frames, in fp32.

    The reference encodes the whole clip in one ``vae.encode``, which diffusers
    runs as one encoder call for the image, then one per ``vae_stride[0]``
    black frames against the causal-conv cache. Called for consecutive ranges
    with the same ``stream``, this makes the same encoder calls in the same
    order, until the cache stops changing (``EncoderStream.encode_black``).
    ``image`` is ``[3, H, W]`` in [-1, 1] on the VAE's device.
    """
    _, h, w = image.shape
    stride_t = config.vae_stride[0]
    outs = []
    with torch.amp.autocast("cuda", enabled=False), torch.inference_mode():
        for latent in range(first_latent, first_latent + num_latents):
            if latent == 0:
                outs.append(vae.encoder(image.float()[None, :, None], feat_cache=stream.cache, feat_idx=[0]))
            else:
                x = torch.zeros(1, 3, stride_t, h, w, dtype=torch.float32, device=image.device)
                outs.append(stream.encode_black(vae, x))
        enc = vae.quant_conv(torch.cat(outs, dim=2))
        z = config.vae_z_dim
        latent = enc[:, :z].float()
        mean, inv_std = latent_affine(vae, latent.dtype, latent.device)
        latent = ((latent - mean) * inv_std)[0]
        lat_h, lat_w = latent.shape[2:]
        if first_latent == 0:
            msk = first_frame_mask(num_latents, lat_h, lat_w, image.device)
        else:
            msk = torch.zeros(4, num_latents, lat_h, lat_w, device=image.device)
        return torch.concat([msk, latent])


class LingBotDitSubmodule(NodeSubmodule):
    """One rollout iteration is one chunk of ``chunk_size`` latent frames:

      * the chunk's I2V condition (``encode_condition_chunk``, continuing the
        request's VAE encoder cache) and its Plucker rays, then
      * ``len(timesteps_index)`` denoise forwards, each converting the flow
        prediction to x0 and re-noising it to the next timestep, then
      * one commit forward over the clean x0 at t=0, whose K/V the step keeps
        for the chunks after it, then
      * a decode of the chunk's latents that continues the request's Wan VAE
        decoder cache (#425's ``decode_causal``), so the frames are those of one
        decode of the whole rollout, emitted as ``video_output``.

    ``prepare_inputs`` draws the chunk's noise from the request's seed and the
    generator state ``PROGRESS`` holds: the rollout's initial noise in one call,
    as the reference does, which is what makes the rollout reproducible bit for
    bit and requires its length up front, then the chunk's re-noise draws.

    ``image`` arrives resized and normalized to [-1, 1] by
    ``LingBotWorldModel.process_prompt``, which runs the reference's CPU bicubic
    resize so the pixels match it exactly. ``encoder_vae`` is its own Wan2.1
    VAE: the decoder's copy may be cast to bf16. Each copy's own path may be
    switched to channels-last.
    """

    disable_torch_compile = True
    # the DiT enters the reference's autocast contexts itself, and the VAE runs in the dtype it was cast to
    disable_autocast = True

    def __init__(self, dit: LingBotCausalDiT, vae: nn.Module, config: LingBotWorldConfig,
                 encoder_vae: nn.Module | None = None, warmup_latent_sizes: list[tuple[int, int]] = ()):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.encoder_vae = encoder_vae
        self.config = config
        self.schedule = make_fast_schedule(config.sample_shift, config.timesteps_index, config.num_train_timesteps)
        fast = not config.reference_numerics
        self._decode_dtype = torch.float32
        if encoder_vae is not None and fast:
            wan_vae.encoder_to_channels_last(encoder_vae)
        if vae is not None and fast:
            self._decode_dtype = wan_vae_decode_dtype()
            if self._decode_dtype != torch.bfloat16:
                raise RuntimeError(
                    f"LingBot-World decodes in bf16, the dtype {VAE_CACHES!r} declares its decoder cache in, and "
                    f"cuDNN {torch.backends.cudnn.version()} has no fast bf16 conv3d; serve it with cuDNN 9.16 or "
                    "later, or with `reference_numerics`, which decodes in fp32")
            vae.to(torch.bfloat16)
        self._decoder = wan_vae.FastDecoder(vae, channels_last=fast, compile=fast) if vae is not None else None
        if vae is not None:
            # the de-normalize's constants, fp32 on the decoder's device
            device = next(vae.parameters()).device
            self._latents_mean, self._latents_inv_std = latent_affine(vae, torch.float32, device)
            if fast:
                fields = wan_vae.stream_cache_fields("decoder", self._decode_dtype, DECODER)
                for lat_hw in warmup_latent_sizes:
                    wan_vae.check_stream_cache(self.warmup_decode(lat_hw), fields, DECODER, lat_hw)
        if dit is not None and fast:
            dit.compile_blocks(autotune=config.dit_compile_autotune)
        self._encoder_fields = tuple(wan_vae.stream_cache_fields("encoder", torch.float32, ENCODER))
        self._decoder_slots = len(wan_vae.new_feat_cache(vae)) if vae is not None else 0
        self.progress = self.vae_caches = None

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.progress = handles[PROGRESS]
        self.vae_caches = handles[VAE_CACHES]

    def step_order(self, graph_walk: str) -> StepOrder:
        """Every rollout chunk emits frames to the client, so a session's
        stream keeps moving while others run."""
        del graph_walk
        return StepOrder.TURNS

    def max_batch_size(self, graph_walk: str):
        # not batched: each request rolls out its own K/V, draws and VAE caches
        return 1

    def prepare_inputs(self, graph_walk, fwd_info: CurrentForwardPassInfo, inputs, **kwargs) -> NodeInputs:
        cfg = self.config
        # the loop's cap is the request's chunk count, so this stays in range
        chunk = int(fwd_info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME])
        image = inputs["image_inputs"][0]
        lat_h, lat_w = image.shape[1] // cfg.vae_stride[1], image.shape[2] // cfg.vae_stride[2]
        latent, renoise, rng = self._draws(
            chunk, int(fwd_info.random_seed), int(fwd_info.step_metadata["num_latent_frames"]), lat_h, lat_w,
            host(self.progress, fwd_info.rid_handle, "main", 0)["rng"],
        )
        return NodeInputs(
            tensor_inputs={
                "text_embeds": inputs["text_embeds_pos"][0],
                "image": image,
                "camera_poses": inputs["camera_poses"][0],
                "intrinsics": inputs["intrinsics"][0],
                "latent": latent,
            },
            kwargs={"chunk": chunk, "renoise": renoise, "rng": rng},
            resource_step_info=cfg.chunk_size * cfg.frame_tokens(lat_h, lat_w),
        )

    def _draws(
        self, chunk: int, seed: int, num_latent_frames: int, lat_h: int, lat_w: int, rng: bytes,
    ) -> tuple[torch.Tensor, list[torch.Tensor], bytes]:
        """Chunk ``chunk``'s initial noise and re-noise draws, and the generator state after them, which the step
        writes to ``PROGRESS``. The rollout's initial noise is one draw from the seed, as the reference makes it, so
        each chunk draws it again and keeps its own frames. ``rng`` is the state the last chunk left, empty before the
        first, where the draws go on from the initial noise's."""
        cfg = self.config
        device = self.get_device()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        shape = (cfg.out_dim, num_latent_frames, lat_h, lat_w)
        noise = torch.randn(shape, dtype=torch.float32, generator=generator, device=device)
        # a copy, so the rollout's noise, up to 940 MB, is freed here and not held through the step
        latent = noise[:, chunk * cfg.chunk_size:(chunk + 1) * cfg.chunk_size].clone()
        if rng:
            generator.set_state(torch.frombuffer(bytearray(rng), dtype=torch.uint8))
        renoise = [
            torch.randn(latent.shape, generator=generator, device=device, dtype=torch.float32)
            for _ in self.schedule.step_timesteps[1:]
        ]
        return latent, renoise, generator.get_state().numpy().tobytes()

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs) -> SubmoduleStep:
        """A segment of the chunk's tokens a row, the text's under ``TEXT``, the draws' state under ``PROGRESS``, and
        the row's VAE caches."""
        state = {
            DIT_KV: ArrayStep(),
            VAE_CACHES: ArrayStep(),
            PROGRESS: ArrayStep(host={
                (rid, "main", 0): {"rng": inp.kwargs["rng"]} for rid, inp in zip(request_ids, inputs, strict=True)
            }),
        }
        if not self.config.reference_numerics:
            state[TEXT] = ArrayStep(segments=tuple(Segment(rid, "text", self.config.text_len) for rid in request_ids))
        rows = zip(request_ids, inputs, strict=True)
        return SubmoduleStep(segments=[Segment(rid, "main", inp.resource_step_info) for rid, inp in rows], state=state)

    def forward(
        self, graph_walk, engine_inputs: ModelInputsFromEngine, text_embeds, image, camera_poses, intrinsics,
        latent, chunk, renoise, **kwargs,
    ) -> NameToTensorList:
        cfg = self.config
        device = self.get_device()
        image = image.to(device)
        context = text_embeds.to(device)
        context = context[0] if context.dim() == 3 else context
        cs = cfg.chunk_size
        frames = slice(chunk * cs, (chunk + 1) * cs)
        stream, feat_cache = self._take_caches(chunk)
        y = encode_condition_chunk(self.encoder_vae, image, stream, chunk * cs, cs, cfg)
        _, height, width = image.shape
        plucker = plucker_embedding(
            camera_poses[frames].float().to(device), intrinsics.float(), height, width,
            height // cfg.vae_stride[1], width // cfg.vae_stride[2],
        ).to(torch.bfloat16)

        with torch.inference_mode():
            # per chunk, and freed before the decode rather than kept with the request
            text_kv = self.dit.text_kv(context)
            conditioning = self.dit.chunk_conditioning(plucker, chunk * cs)
            steps = self.schedule.step_timesteps
            for i, t in enumerate(steps):
                timestep = torch.stack([t]).to(device)
                # each denoise pass adds the chunk's K/V for its own read, and the step keeps the commit pass's
                flow = self.dit(latent, timestep, text_kv, y, conditioning, keep=False)
                x0 = flow_to_x0(flow, latent, t, self.schedule)
                if i < len(steps) - 1:
                    latent = add_noise(x0, renoise[i], steps[i + 1], self.schedule)
            commit_t = torch.stack([steps[-1] * 0.0]).to(device)
            self.dit(x0, commit_t, text_kv, y, conditioning, keep=True)
            del conditioning, text_kv
            pixels = self._decode(x0, feat_cache, clip_start=chunk == 0)
        self.vae_caches.write(0, **stream.fields(), **wan_vae.cache_fields(feat_cache, DECODER))
        if stream.settled:
            self.vae_caches.clear(0, *self._encoder_fields)

        # quantize from fp32; a bf16 decode's frames round once, here
        frames_u8 = truncate_to_uint8(pixels[0].float()).permute(1, 2, 3, 0).contiguous()
        outputs = {
            "clock": [torch.full((1,), chunk + 1, dtype=torch.int64, device=device)],
            "video_output": [frames_u8],
        }
        if cfg.keep_chunk_outputs:
            outputs.update(chunk_latents=[x0], chunk_pixels=[pixels], chunk_condition=[y])
        return outputs

    def _take_caches(self, chunk: int) -> tuple[EncoderStream, wan_vae.FeatCache]:
        """The request's encoder stream and decoder cache from ``VAE_CACHES``, each empty at chunk 0. The read hands
        the tensors over and nothing here keeps them, so a slot is freed where the encode or the decode replaces it."""
        held = self.vae_caches.read(0).fields
        # built whole, so `FeatCache`'s compaction doesn't run again on the slots it wrote
        feat_cache = wan_vae.FeatCache(wan_vae.cache_from_fields(self._decoder_slots, held, DECODER))
        if chunk and not any(torch.is_tensor(slot) for slot in feat_cache):
            raise RuntimeError(f"LingBot dit: chunk {chunk} found no decoder cache in {VAE_CACHES!r}, where chunk "
                               f"{chunk - 1} left it")
        return EncoderStream(self.encoder_vae, held), feat_cache

    def warmup_decode(self, lat_hw: tuple[int, int]) -> wan_vae.FeatCache:
        """Decode two chunks of zero latents through a fresh stream at
        ``lat_hw``, so every call signature of the compiled decode is compiled
        before a request at that size. Returns the stream's cache."""
        cfg = self.config
        device = next(self.vae.parameters()).device
        feat_cache = wan_vae.new_feat_cache(self.vae)
        with torch.inference_mode():
            for k in range(2):
                self._decode(torch.zeros(cfg.out_dim, cfg.chunk_size, *lat_hw, device=device), feat_cache,
                             clip_start=k == 0)
        return feat_cache

    def _decode(self, latents: torch.Tensor, feat_cache: list, clip_start: bool) -> torch.Tensor:
        """``[1, 3, frames, H, W]`` in [-1, 1]: the chunk decoded as the
        continuation of the stream whose cache is ``feat_cache``."""
        # the de-normalize in fp32 whatever the decode's dtype
        z = (latents[None].float() / self._latents_inv_std + self._latents_mean).to(self._decode_dtype)
        return self._decoder.decode_stream(z, feat_cache, clip_start=clip_start)
