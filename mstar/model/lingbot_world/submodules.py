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

The rolling self-attention K/V and the text K/V live in the request's
``PerRequestState`` (dense, model-owned, freed with the request), the VAE
encoder's and decoder's causal-conv caches likewise. The dit's one engine
resource is the attention over them: a ragged cross-attention (``DIT_ATTN``)
whose step pairs the chunk's queries with its K/V window and with the text.

Numerics are the model's, not the engine's: ``LingBotWorldModel.get_autocast_dtype``
returns None, the DiT enters the reference's bf16 autocast itself, and the VAE
runs fp32 as the reference does.
"""

import functools
import logging
from typing import NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import RaggedCrossAttentionStep, Segment, SubmoduleStep, cross_label
from mstar.engine.resources.convenience import RaggedAttentionCallable
from mstar.model.components import wan_vae
from mstar.model.components.diffusion.attention import joint_attention
from mstar.model.lingbot_world.components.camera import plucker_embedding
from mstar.model.lingbot_world.components.dit import (
    REFERENCE_ATTENTION,
    DitAttention,
    LingBotCausalDiT,
    attended_frames,
)
from mstar.model.lingbot_world.components.schedule import add_noise, flow_to_x0, make_fast_schedule
from mstar.model.lingbot_world.config import LingBotWorldConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.wan22.submodules import _SingleRequestMixin

logger = logging.getLogger(__name__)

ROLLOUT_LOOP_NAME = "rollout_loop"
ROLLOUT_WALK = "rollout"

DIT_ATTN = "dit_attn"
CHUNK_SPAN, WINDOW_SPAN, TEXT_SPAN = "chunk", "window", "text"
DIT_ATTN_PAIRS = ((CHUNK_SPAN, WINDOW_SPAN), (CHUNK_SPAN, TEXT_SPAN))


class ChunkSpans(NamedTuple):
    """Tokens of one chunk's attentions: its queries, the rolling K/V window
    they attend (sink, history, the chunk itself) and the padded text."""

    chunk: int
    window: int
    text: int


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
    point, the one encoder output every later call would return."""

    def __init__(self, vae: nn.Module):
        counts = getattr(vae, "_cached_conv_counts", None)
        if counts is not None:
            num_convs = counts["encoder"]
        else:
            num_convs = sum(isinstance(m, wan_vae.WanCausalConv3d) for m in vae.encoder.modules())
        # One slot per causal conv, as ``AutoencoderKLWan.clear_cache`` builds it.
        self.cache: list | None = [None] * num_convs
        self.steady: torch.Tensor | None = None

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
            self.steady, self.cache = out, None
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
        mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(latent)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(latent)
        latent = ((latent - mean) * inv_std)[0]
        lat_h, lat_w = latent.shape[2:]
        if first_latent == 0:
            msk = first_frame_mask(num_latents, lat_h, lat_w, image.device)
        else:
            msk = torch.zeros(4, num_latents, lat_h, lat_w, device=image.device)
        return torch.concat([msk, latent])


class LingBotDitSubmodule(_SingleRequestMixin, NodeSubmodule):
    """One rollout iteration is one chunk of ``chunk_size`` latent frames:

      * the chunk's I2V condition (``encode_condition_chunk``, continuing the
        request's VAE encoder cache) and its Plucker rays, then
      * ``len(timesteps_index)`` denoise forwards, each converting the flow
        prediction to x0 and re-noising it to the next timestep from the
        request's generator, then
      * one commit forward over the clean x0 at t=0, which leaves the chunk's
        settled K/V in the rolling cache for the chunks after it, then
      * a decode of the chunk's latents that continues the request's Wan VAE
        decoder cache (#425's ``decode_causal``), so the frames are those of one
        decode of the whole rollout, emitted as ``video_output``.

    At the first iteration the request's state is seeded: the K/V cache, the
    generator (seeded with the request seed) and the full rollout's initial
    noise drawn from it in one call. The reference draws the initial noise the
    same way, which is what makes the rollout reproducible bit for bit; it
    requires the rollout length up front.

    ``image`` arrives resized and normalized to [-1, 1] by
    ``LingBotWorldModel.process_prompt``, which runs the reference's CPU bicubic
    resize so the pixels match it exactly. ``encoder_vae`` is its own Wan2.1
    VAE: the decoder's copy may be switched to channels-last.
    """

    disable_torch_compile = True

    def __init__(self, dit: LingBotCausalDiT, vae: nn.Module, config: LingBotWorldConfig,
                 encoder_vae: nn.Module | None = None):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.encoder_vae = encoder_vae
        self.config = config
        self.schedule = make_fast_schedule(config.sample_shift, config.timesteps_index, config.num_train_timesteps)
        if vae is not None and config.vae_decode_channels_last:
            wan_vae.to_channels_last(vae)
        self._decode_step = wan_vae.DecoderStep(vae, compile=False) if vae is not None else None
        self._attention: DitAttention | None = None
        # Debug hook for the parity harness: keep the last chunk's latents and
        # float frames in the request state.
        self.keep_chunk_outputs = False

    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._attention = None

    def attention(self) -> DitAttention:
        """The DiT's attentions through ``DIT_ATTN`` when the engine bound it,
        else the reference's FA2 (``reference_attention``, the parity harness)."""
        if self._attention is None:
            resource = self.node_resources.get(DIT_ATTN)
            if resource is None:
                self._attention = REFERENCE_ATTENTION
            else:
                def on(kv_label):
                    ragged = RaggedAttentionCallable(resource, cross_label(CHUNK_SPAN, kv_label))
                    return functools.partial(joint_attention, ragged=ragged)
                self._attention = DitAttention(window=on(WINDOW_SPAN), text=on(TEXT_SPAN))
        return self._attention

    def chunk_spans(self, chunk: int, image_hw: tuple[int, int]) -> ChunkSpans:
        cfg = self.config
        lat_h, lat_w = image_hw[0] // cfg.vae_stride[1], image_hw[1] // cfg.vae_stride[2]
        frame_seqlen = lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
        return ChunkSpans(
            chunk=cfg.chunk_size * frame_seqlen,
            window=attended_frames(cfg, chunk) * frame_seqlen,
            text=cfg.text_len,
        )

    def prepare_inputs(self, graph_walk, fwd_info: CurrentForwardPassInfo, inputs, **kwargs) -> NodeInputs | None:
        state = self.request_state(fwd_info.rid_handle)
        num_chunks = int(fwd_info.step_metadata["num_chunks"])
        chunk = int(state.get("chunk", 0))
        if chunk >= num_chunks:
            logger.info(
                "LingBot dit: skipping async-overshoot chunk %d (request %s runs %d chunks)",
                chunk, fwd_info.request_id, num_chunks,
            )
            return None
        image = inputs["image_inputs"][0]
        return NodeInputs(
            tensor_inputs={
                "text_embeds": inputs["text_embeds_pos"][0],
                "image": image,
                "camera_poses": inputs["camera_poses"][0],
                "intrinsics": inputs["intrinsics"][0],
            },
            resource_step_info=self.chunk_spans(chunk, tuple(image.shape[1:])),
        )

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs) -> SubmoduleStep | None:
        if DIT_ATTN not in self.node_resources:
            return None
        segments = []
        for rid, inp in zip(request_ids, inputs, strict=True):
            spans: ChunkSpans = inp.resource_step_info
            segments += [Segment(request_id=rid, label=CHUNK_SPAN, span=spans.chunk),
                         Segment(request_id=rid, label=WINDOW_SPAN, span=spans.window),
                         Segment(request_id=rid, label=TEXT_SPAN, span=spans.text)]
        return SubmoduleStep(steps={DIT_ATTN: RaggedCrossAttentionStep(segments=tuple(segments), pairs=DIT_ATTN_PAIRS)})

    def _seed_request(self, state, md: dict, seed: int, image: torch.Tensor, device):
        cfg = self.config
        lat_f = int(md["num_latent_frames"])
        lat_h, lat_w = image.shape[1] // cfg.vae_stride[1], image.shape[2] // cfg.vae_stride[2]
        frame_seqlen = lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(cfg.out_dim, lat_f, lat_h, lat_w, dtype=torch.float32, generator=generator, device=device)
        state.add_all(
            chunk=0,
            frame_seqlen=frame_seqlen,
            generator=generator,
            noise=noise,
            kv=self.dit.new_cache(frame_seqlen, lat_f, device),
            encoder_stream=EncoderStream(self.encoder_vae) if self.encoder_vae is not None else None,
            feat_cache=wan_vae.new_feat_cache(self.vae) if self.vae is not None else None,
        )

    def forward(
        self, graph_walk, engine_inputs: ModelInputsFromEngine, text_embeds, image,
        camera_poses, intrinsics, **kwargs,
    ) -> NameToTensorList:
        cfg = self.config
        rid = engine_inputs.request_ids[0]
        info = engine_inputs.per_request_info[rid]
        state = self.request_state(rid)
        device = self.get_device()
        image = image.to(device)
        if "kv" not in state:
            self._seed_request(state, info.step_metadata, info.random_seed, image, device)
        chunk = int(state["chunk"])
        frame_seqlen = int(state["frame_seqlen"])
        cs = cfg.chunk_size
        frames = slice(chunk * cs, (chunk + 1) * cs)
        latent = state["noise"][:, frames]
        y = encode_condition_chunk(self.encoder_vae, image, state["encoder_stream"], chunk * cs, cs, cfg)
        _, height, width = image.shape
        plucker = plucker_embedding(
            camera_poses[frames].float().to(device), intrinsics.float(), height, width,
            height // cfg.vae_stride[1], width // cfg.vae_stride[2],
        ).to(torch.bfloat16)
        context = text_embeds.to(device)
        context = context[0] if context.dim() == 3 else context
        current_start = chunk * cs * frame_seqlen
        kv = state["kv"]
        generator = state["generator"]
        # the step declared its K/V window from the chunk index; the cache must agree
        window = kv.window(kv.plan(current_start, cs * frame_seqlen))
        if window.stop - window.start != attended_frames(cfg, chunk) * frame_seqlen:
            raise RuntimeError(f"LingBot dit: chunk {chunk} attends {window} of the rolling K/V, not the declared span")
        attention = self.attention()

        with torch.inference_mode():
            steps = self.schedule.step_timesteps
            for i, t in enumerate(steps):
                timestep = torch.stack([t]).to(device)
                flow = self.dit(latent, timestep, context, y, plucker, kv, current_start, frame_seqlen, attention)
                x0 = flow_to_x0(flow, latent, t, self.schedule)
                if i < len(steps) - 1:
                    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                    latent = add_noise(x0, noise, steps[i + 1], self.schedule)
            commit_t = torch.stack([steps[-1] * 0.0]).to(device)
            self.dit(x0, commit_t, context, y, plucker, kv, current_start, frame_seqlen, attention)
            pixels = self._decode(x0, state, clip_start=chunk == 0)

        state.add("chunk", chunk + 1)
        if self.keep_chunk_outputs:
            state.add_all(chunk_latents=x0.clone(), chunk_pixels=pixels.clone(), chunk_condition=y.clone())
        frames_u8 = (pixels[0] / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
        return {
            "clock": [torch.full((1,), chunk + 1, dtype=torch.int64, device=device)],
            "video_output": [frames_u8],
        }

    def _decode(self, latents: torch.Tensor, state, clip_start: bool) -> torch.Tensor:
        """``[1, 3, frames, H, W]`` in [-1, 1]: the chunk decoded as the
        continuation of the request's stream."""
        cfg = self.config
        dtype = torch.float32 if cfg.vae_decode_fp32 else torch.bfloat16
        z = latents[None].to(dtype)
        mean = torch.tensor(self.vae.config.latents_mean).view(1, cfg.vae_z_dim, 1, 1, 1).to(z)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, cfg.vae_z_dim, 1, 1, 1).to(z)
        z = z / inv_std + mean
        with torch.amp.autocast("cuda", enabled=False):
            return wan_vae.decode_causal(
                self.vae, z, state["feat_cache"], self._decode_step, clip_start=clip_start, latents_per_call=1,
            )

    def check_stop(self, request_id, request_info: CurrentForwardPassInfo, outputs) -> set[str]:
        done = request_info.dynamic_loop_iter_counts.get(ROLLOUT_LOOP_NAME, 0)
        if done + 1 >= int(request_info.step_metadata.get("num_chunks", 0) or 0):
            return {ROLLOUT_LOOP_NAME}
        return set()
