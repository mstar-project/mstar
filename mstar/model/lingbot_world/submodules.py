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

The rolling self-attention K/V, the text K/V (computed once per request) and
the VAE encoder's and decoder's causal-conv caches live in each request's region
of the engine's request pool (``DIT_POOL``), declared when the request opens
(``rollout_state_buffers``) so it waits in line for room. The rolling KV window
(``DIT_KV``) keeps the K/V's sink plus latest window compacted in their buffer,
so a chunk's keys are one contiguous range; the attention over them is a ragged
cross-attention (``DIT_ATTN``) whose step pairs the chunk's queries with that
range and with the text. The encoder and decoder write their caches straight
into the pool (``wan_vae.PoolFeatCache``, ``wan_vae.encoder_saves_uncopied``,
``wan_vae.decoder_saves_uncopied``). How many rollouts run at once is the pool's
to decide: a rollout waits in line until its region fits. Only the noise drawn
for the whole rollout (at most 4.4% of the rollout's region) and small host-side
bookkeeping stay in the request's ``PerRequestState``, outside the pool.

Numerics are the model's, not the engine's: ``LingBotWorldModel.get_autocast_dtype``
returns None, the DiT enters the reference's bf16 autocast itself, and the VAE
encode runs fp32 as the reference does; the decode runs bf16 (cuDNN >= 9.16)
unless ``reference_numerics``.
"""

import functools
import logging
from typing import NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import RaggedCrossAttentionStep, Segment, SubmoduleStep, cross_label
from mstar.engine.resources.kv.rolling import RollingKVConfig, RollingKVStep, RollingPosition
from mstar.engine.resources.request_pool import RequestPoolStep
from mstar.model.components import wan_vae
from mstar.model.components.diffusion.attention import joint_attention, sdpa_attention
from mstar.model.components.diffusion.decode_utils import latent_affine, wan_vae_decode_dtype
from mstar.model.components.diffusion.image_io import truncate_to_uint8
from mstar.model.lingbot_world.components.camera import plucker_embedding
from mstar.model.lingbot_world.components.dit import (
    REFERENCE_ATTENTION,
    ChunkOnlyStore,
    DitAttention,
    LingBotCausalDiT,
    attended_frames,
)
from mstar.model.lingbot_world.components.schedule import add_noise, flow_to_x0, make_fast_schedule
from mstar.model.lingbot_world.config import LingBotWorldConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule, StepOrder

logger = logging.getLogger(__name__)

ROLLOUT_LOOP_NAME = "rollout_loop"
ROLLOUT_WALK = "rollout"

DIT_ATTN = "dit_attn"
DIT_KV = "dit_kv"
# the request pool whose regions hold each rollout's state: its rolling K/V (the
# DIT_KV buffer), its text K/V, the VAE encoder's causal-conv cache (its slots, then
# the one output it keeps once it settles), and the VAE decoder's
DIT_POOL = "dit_pool"
TEXT_KV = "text_kv"
ENCODER_OUT = "enc.out"
CHUNK_SPAN, WINDOW_SPAN, TEXT_SPAN = "chunk", "window", "text"
DIT_ATTN_PAIRS = ((CHUNK_SPAN, WINDOW_SPAN), (CHUNK_SPAN, TEXT_SPAN))


@torch.compiler.disable
def _warmup_ragged(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SDPA over one packed ``[L, H, D]`` sequence, standing in for the
    ragged callable at load: compiled blocks trace the same graphs around it."""
    return sdpa_attention(q[None], k[None], v[None])[0]


class ChunkSpans(NamedTuple):
    """Tokens of one chunk's attentions: its queries, the rolling K/V window
    they attend (sink, history, the chunk itself) and the padded text; and the
    chunk's place in the rolling K/V stream: the tokens written before it, the
    sink, and the trailing tokens kept with the chunk's own."""

    chunk: int
    window: int
    text: int
    position: RollingPosition


def rolling_extent(
    cfg: LingBotWorldConfig, image_hw: tuple[int, int], num_latent_frames: int | None = None,
) -> tuple[int, int]:
    """A rollout's rolling K/V per layer, ``(sink, window)`` in tokens, at the
    output size ``image_hw``. ``num_latent_frames`` is needed only without a
    window, where the K/V keep the whole rollout."""
    lat_h, lat_w = image_hw[0] // cfg.vae_stride[1], image_hw[1] // cfg.vae_stride[2]
    frame_seqlen = lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
    if cfg.local_attn_size > -1:
        return cfg.sink_size * frame_seqlen, (cfg.local_attn_size - cfg.sink_size) * frame_seqlen
    if num_latent_frames is None:
        raise ValueError("without a window, the rolling K/V extent needs the rollout's num_latent_frames")
    return 0, num_latent_frames * frame_seqlen


def rolling_config(cfg: LingBotWorldConfig) -> RollingKVConfig:
    """The dit's rolling K/V window, over the ``DIT_KV`` buffer of each rollout's pool region."""
    return RollingKVConfig(num_layers=cfg.num_layers, num_heads=cfg.num_heads, head_dim=cfg.head_dim,
                           pool=DIT_POOL, dtype=torch.bfloat16)


class CacheSlot(NamedTuple):
    """One causal conv's cache slot once its stream is past its first call:
    ``[1, channels, frames, h, w]``, where ``h2`` and ``w2`` are ``h`` and ``w``
    at a 2x2 latent; every slot scales with the latent's size."""

    channels: int
    frames: int
    h2: int
    w2: int

    def shape(self, lat_h: int, lat_w: int) -> tuple[int, ...]:
        return (1, self.channels, self.frames, self.h2 * lat_h // 2, self.w2 * lat_w // 2)


class EncoderCacheLayout(NamedTuple):
    slots: tuple[CacheSlot | None, ...]
    # one encoder call's output, which the stream keeps once its cache settles
    out: CacheSlot


@functools.cache
def encoder_cache_layout() -> EncoderCacheLayout:
    """The Wan2.1 VAE encoder's cache slots, found by running the stock-config VAE
    (the checkpoint's config is the class default) on the meta device at a 2x2
    latent: no weights and no memory, so the conductor can size a rollout's state
    before any worker has the VAE. A slot no call fills is None."""
    from diffusers import AutoencoderKLWan

    with torch.device("meta"):
        vae = AutoencoderKLWan()
    with torch.inference_mode():
        stream = EncoderStream(vae)
        vae.encoder(torch.empty(1, 3, 1, 16, 16, device="meta"), feat_cache=stream.cache, feat_idx=[0])
        out = vae.encoder(torch.empty(1, 3, 4, 16, 16, device="meta"), feat_cache=stream.cache, feat_idx=[0])
    return EncoderCacheLayout(tuple(CacheSlot(*t.shape[1:]) if torch.is_tensor(t) else None for t in stream.cache),
                              CacheSlot(*out.shape[1:]))


def encoder_slot_name(i: int) -> str:
    return f"enc.{i}"


@functools.cache
def decoder_cache_layout() -> tuple[CacheSlot | None, ...]:
    """The Wan2.1 VAE decoder's cache slots, found as ``encoder_cache_layout``'s are:
    the stock-config VAE run on the meta device at a 2x2 latent. A slot no call
    fills is None."""
    from diffusers import AutoencoderKLWan

    with torch.device("meta"):
        vae = AutoencoderKLWan()
    with torch.inference_mode():
        cache = wan_vae.new_feat_cache(vae)
        z = torch.empty(1, 16, 1, 2, 2, device="meta")
        vae.decoder(z, feat_cache=cache, feat_idx=[0], first_chunk=True)
        vae.decoder(z, feat_cache=cache, feat_idx=[0])
    return tuple(CacheSlot(*t.shape[1:]) if torch.is_tensor(t) else None for t in cache)


def decoder_cache_dtype(cfg: LingBotWorldConfig) -> torch.dtype:
    """The dtype the dit's VAE decode runs, and keeps its cache, in."""
    return torch.float32 if cfg.reference_numerics else wan_vae_decode_dtype()


def decoder_slot_name(i: int) -> str:
    return f"dec.{i}"


def rollout_state_buffers(
    cfg: LingBotWorldConfig, lat_hw: tuple[int, int], kv_tokens: int,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """What a rollout at a ``lat_hw`` latent keeps in the pool between chunks: its
    rolling K/V (``kv_tokens`` a layer), its text K/V (every block's, keys then
    values), each slot of its VAE encoder's cache plus the output it keeps once
    settled (fp32, as the encode runs), and each slot of its VAE decoder's cache
    (in the decode's dtype)."""
    buffers = {
        DIT_KV: rolling_config(cfg).buffer(kv_tokens),
        TEXT_KV: ((cfg.num_layers, 2, 1, cfg.text_len, cfg.num_heads, cfg.head_dim), torch.bfloat16),
    }
    layout = encoder_cache_layout()
    for i, slot in enumerate(layout.slots):
        if slot is not None:
            buffers[encoder_slot_name(i)] = (slot.shape(*lat_hw), torch.float32)
    buffers[ENCODER_OUT] = (layout.out.shape(*lat_hw), torch.float32)
    dtype = decoder_cache_dtype(cfg)
    for i, slot in enumerate(decoder_cache_layout()):
        if slot is not None:
            buffers[decoder_slot_name(i)] = (slot.shape(*lat_hw), dtype)
    return buffers


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

    @classmethod
    def over(cls, cache: list | None, steady: torch.Tensor | None) -> "EncoderStream":
        """A stream continuing from a kept cache (or, once settled, a kept output)."""
        stream = cls.__new__(cls)
        stream.cache, stream.steady = cache, steady
        return stream

    def encode_black(self, vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """One encoder call on black frames. When a call leaves every cache
        slot equal to what it held before, the next call sees the same input
        and the same cache, so it, and every call after it, returns this
        call's output; the cache is dropped from then on. A ``PoolFeatCache``
        compares each slot as it is overwritten, on the device, and the answer
        is read once a call (``PoolFeatCache.watch``); with a plain list the
        encoder replaces its slots, and they are compared after the call (a
        slot written in place never counts as unchanged)."""
        if self.steady is not None:
            return self.steady
        if isinstance(self.cache, wan_vae.PoolFeatCache):
            self.cache.watch()
            out = vae.encoder(x, feat_cache=self.cache, feat_idx=[0])
            if self.cache.unchanged():
                self.steady, self.cache = out, None
            return out
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
    VAE: the decoder's copy may be cast to bf16. Each copy's own path may be
    switched to channels-last.
    """

    disable_torch_compile = True

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
        if encoder_vae is not None:
            # each call saves its cache straight into the request's pool region (same math)
            wan_vae.encoder_saves_uncopied(encoder_vae)
        if encoder_vae is not None and fast:
            wan_vae.encoder_to_channels_last(encoder_vae)
        if vae is not None and fast:
            self._decode_dtype = wan_vae_decode_dtype()
            if self._decode_dtype == torch.bfloat16:
                vae.to(torch.bfloat16)
            else:
                logger.info("LingBot-World: decoding fp32, cuDNN %s has no fast bf16 conv3d (9.16 does)",
                            torch.backends.cudnn.version())
        if vae is not None:
            # each call writes its cache straight into the request's pool region (same math);
            # before FastDecoder, whose step binds the decoder's forward
            wan_vae.decoder_saves_uncopied(vae)
        self._decoder = wan_vae.FastDecoder(vae, channels_last=fast, compile=fast) if vae is not None else None
        if vae is not None:
            # the de-normalize's constants, fp32 on the decoder's device
            device = next(vae.parameters()).device
            self._latents_mean, self._latents_inv_std = latent_affine(vae, torch.float32, device)
            if fast:
                for lat_hw in warmup_latent_sizes:
                    self.warmup_decode(lat_hw)
        if dit is not None and fast:
            dit.compile_blocks(autotune=config.dit_compile_autotune)
            for lat_hw in warmup_latent_sizes:
                self.warmup_dit(lat_hw)
        self._attention: DitAttention | None = None
        # Debug hook for the parity harness: keep the last chunk's latents and
        # float frames in the request state.
        self.keep_chunk_outputs = False

    def step_order(self, graph_walk: str) -> StepOrder:
        """Every rollout chunk emits frames to the client, so a session's
        stream keeps moving while others run."""
        del graph_walk
        return StepOrder.TURNS

    def max_batch_size(self, graph_walk: str):
        # not batched: each request rolls out its own K/V cache, generator and VAE caches
        return 1

    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._attention = None

    def attention(self) -> DitAttention:
        """The DiT's attentions through ``DIT_ATTN``, or the reference's FA2
        under ``reference_numerics`` (the parity harness binds no resource)."""
        if self._attention is None:
            resource = self.node_resources.get(DIT_ATTN)
            if resource is None:
                if not self.config.reference_numerics:
                    raise RuntimeError(
                        f"LingBot dit: no {DIT_ATTN} resource is bound; only reference_numerics attends without one"
                    )
                self._attention = REFERENCE_ATTENTION
            else:
                def on(kv_label):
                    ragged = functools.partial(resource.run, label=cross_label(CHUNK_SPAN, kv_label))
                    return functools.partial(joint_attention, ragged=ragged)
                self._attention = DitAttention(window=on(WINDOW_SPAN), text=on(TEXT_SPAN))
        return self._attention

    def chunk_spans(self, chunk: int, image_hw: tuple[int, int], num_latent_frames: int | None = None) -> ChunkSpans:
        """``num_latent_frames`` (the rollout's) is needed only without a window,
        where the rolling K/V keep the whole rollout."""
        cfg = self.config
        lat_h, lat_w = image_hw[0] // cfg.vae_stride[1], image_hw[1] // cfg.vae_stride[2]
        frame_seqlen = lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
        chunk_tokens = cfg.chunk_size * frame_seqlen
        written = chunk * chunk_tokens
        sink, trailing = rolling_extent(cfg, image_hw, num_latent_frames)
        return ChunkSpans(
            chunk=chunk_tokens,
            window=attended_frames(cfg, chunk) * frame_seqlen,
            text=cfg.text_len,
            position=RollingPosition(written=written, sink=sink, window=trailing),
        )

    def prepare_inputs(self, graph_walk, fwd_info: CurrentForwardPassInfo, inputs, **kwargs) -> NodeInputs:
        # the loop's cap is the request's chunk count, so this stays in range
        chunk = int(fwd_info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME])
        image = inputs["image_inputs"][0]
        return NodeInputs(
            tensor_inputs={
                "text_embeds": inputs["text_embeds_pos"][0],
                "image": image,
                "camera_poses": inputs["camera_poses"][0],
                "intrinsics": inputs["intrinsics"][0],
            },
            resource_step_info=self.chunk_spans(
                chunk, tuple(image.shape[1:]), int(fwd_info.step_metadata["num_latent_frames"])),
        )

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs) -> SubmoduleStep | None:
        steps = {}
        spans: list[ChunkSpans] = [inp.resource_step_info for inp in inputs]
        if DIT_POOL in self.node_resources:
            steps[DIT_POOL] = RequestPoolStep(
                segments=tuple(Segment(request_id=rid, label=CHUNK_SPAN, span=s.chunk)
                               for rid, s in zip(request_ids, spans, strict=True)),
            )
        if DIT_KV in self.node_resources:
            steps[DIT_KV] = RollingKVStep(
                segments=tuple(Segment(request_id=rid, label=CHUNK_SPAN, span=s.chunk)
                               for rid, s in zip(request_ids, spans, strict=True)),
                positions={rid: s.position for rid, s in zip(request_ids, spans, strict=True)},
            )
        if DIT_ATTN in self.node_resources:
            segments = []
            for rid, s in zip(request_ids, spans, strict=True):
                segments += [Segment(request_id=rid, label=CHUNK_SPAN, span=s.chunk),
                             Segment(request_id=rid, label=WINDOW_SPAN, span=s.window),
                             Segment(request_id=rid, label=TEXT_SPAN, span=s.text)]
            steps[DIT_ATTN] = RaggedCrossAttentionStep(segments=tuple(segments), pairs=DIT_ATTN_PAIRS)
        return SubmoduleStep(steps=steps) if steps else None

    def state_pool(self):
        """The engine's ``DIT_POOL``: the rollout's state lives in its region."""
        pool = self.node_resources.get(DIT_POOL)
        if pool is None:
            raise RuntimeError(f"LingBot dit: no {DIT_POOL} resource is bound; the rollout's state lives there")
        return pool

    def kv_store(self):
        """The rolling K/V the DiT writes and attends, the engine's ``DIT_KV``."""
        store = self.node_resources.get(DIT_KV)
        if store is None:
            raise RuntimeError(f"LingBot dit: no {DIT_KV} resource is bound; the rolling K/V live there")
        return store

    def _seed_request(self, state, md: dict, seed: int, image: torch.Tensor, context: torch.Tensor, device):
        cfg = self.config
        lat_f = int(md["num_latent_frames"])
        lat_h, lat_w = image.shape[1] // cfg.vae_stride[1], image.shape[2] // cfg.vae_stride[2]
        frame_seqlen = lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(cfg.out_dim, lat_f, lat_h, lat_w, dtype=torch.float32, generator=generator, device=device)
        text_kv = self.state_pool().view(TEXT_KV)
        with torch.inference_mode():
            keys, values = self.dit.text_kv(context)
            for i, (k, v) in enumerate(zip(keys, values, strict=True)):
                text_kv[i, 0].copy_(k)
                text_kv[i, 1].copy_(v)
        state.add_all(
            frame_seqlen=frame_seqlen,
            generator=generator,
            noise=noise,
            # the encode stream (its cache, slots in the pool) and where in the pool its views were taken
            encoder_stream=None,
            encoder_stream_at=None,
            # the decoder cache (its slots in the pool) and where in the pool its views were taken
            decoder_cache=None,
            decoder_cache_at=None,
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
        context = text_embeds.to(device)
        context = context[0] if context.dim() == 3 else context
        if "generator" not in state:
            self._seed_request(state, info.step_metadata, info.random_seed, image, context, device)
        chunk = int(info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME])
        frame_seqlen = int(state["frame_seqlen"])
        cs = cfg.chunk_size
        frames = slice(chunk * cs, (chunk + 1) * cs)
        latent = state["noise"][:, frames]
        stream = self._encoder_stream(state)
        y = encode_condition_chunk(self.encoder_vae, image, stream, chunk * cs, cs, cfg)
        self._keep_settled_output(state, stream)
        _, height, width = image.shape
        plucker = plucker_embedding(
            camera_poses[frames].float().to(device), intrinsics.float(), height, width,
            height // cfg.vae_stride[1], width // cfg.vae_stride[2],
        ).to(torch.bfloat16)
        store = self.kv_store()
        # Every block's text K/V, views of the pool good for this step (a rearrange can move
        # the region); taken in inference mode, as the warmup's are, since the compiled
        # blocks guard on their inputs' dispatch keys.
        with torch.inference_mode():
            text = self.state_pool().view(TEXT_KV)
            text_kv = (list(text[:, 0].unbind(0)), list(text[:, 1].unbind(0)))
        generator = state["generator"]
        # the attention step declared the window from the chunk index; the store's plan must agree
        kept = store.window(0)[0].shape[1]
        if kept != attended_frames(cfg, chunk) * frame_seqlen:
            raise RuntimeError(f"LingBot dit: chunk {chunk} attends {kept} tokens of the rolling K/V, "
                               "not the declared window")
        attention = self.attention()

        with torch.inference_mode():
            # per chunk, and freed before the decode rather than kept with the request
            conditioning = self.dit.chunk_conditioning(plucker, chunk * cs)
            steps = self.schedule.step_timesteps
            for i, t in enumerate(steps):
                timestep = torch.stack([t]).to(device)
                # each pass rewrites the chunk's K/V in the store; the commit pass's stay
                flow = self.dit(latent, timestep, text_kv, y, conditioning, store, attention)
                x0 = flow_to_x0(flow, latent, t, self.schedule)
                if i < len(steps) - 1:
                    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                    latent = add_noise(x0, noise, steps[i + 1], self.schedule)
            commit_t = torch.stack([steps[-1] * 0.0]).to(device)
            self.dit(x0, commit_t, text_kv, y, conditioning, store, attention)
            del conditioning
            pixels = self._decode(x0, self._decoder_cache(state), clip_start=chunk == 0)

        if self.keep_chunk_outputs:
            state.add_all(chunk_latents=x0.clone(), chunk_pixels=pixels.clone(), chunk_condition=y.clone())
        # quantize from fp32; a bf16 decode's frames round once, here
        frames_u8 = truncate_to_uint8(pixels[0].float()).permute(1, 2, 3, 0).contiguous()
        return {
            "clock": [torch.full((1,), chunk + 1, dtype=torch.int64, device=device)],
            "video_output": [frames_u8],
        }

    def warmup_dit(self, lat_hw: tuple[int, int]) -> None:
        """One chunk's DiT forward on zeros at ``lat_hw`` against a scratch
        store that keeps nothing (chunk 0's window), so the compiled blocks are
        traced and tuned before a request at that size. The window and chunk
        index never reach a compiled graph, so one chunk covers a rollout; the
        attention has the bound one's call structure, which decides where the
        graphs split."""
        cfg = self.config
        device = self.dit.patch_embedding.weight.device
        lat_h, lat_w = lat_hw
        zeros = functools.partial(torch.zeros, device=device)
        with torch.inference_mode():
            conditioning = self.dit.chunk_conditioning(
                zeros(1, cfg.control_dim * 64, cfg.chunk_size, lat_h, lat_w, dtype=torch.bfloat16), 0)
            latents = zeros(cfg.out_dim, cfg.chunk_size, lat_h, lat_w)
            text = zeros(cfg.text_len, cfg.text_dim, dtype=torch.bfloat16)
            condition = zeros(cfg.in_dim - cfg.out_dim, cfg.chunk_size, lat_h, lat_w)
            if cfg.reference_numerics:
                attention = REFERENCE_ATTENTION
            else:
                attend = functools.partial(joint_attention, ragged=_warmup_ragged)
                attention = DitAttention(window=attend, text=attend)
            self.dit(latents, zeros(1), self.dit.text_kv(text), condition, conditioning, ChunkOnlyStore(), attention)

    def warmup_decode(self, lat_hw: tuple[int, int]) -> None:
        """Decode two chunks of zero latents through a fresh stream at
        ``lat_hw``, so every call signature of the compiled decode is compiled
        before a request at that size."""
        cfg = self.config
        device = next(self.vae.parameters()).device
        with torch.inference_mode():
            # laid out as a request's cache in the pool, so the served call signatures compile here
            shapes = [None if slot is None else slot.shape(*lat_hw) for slot in decoder_cache_layout()]
            cache = wan_vae.PoolFeatCache.fresh(shapes, self._decode_dtype, device)
            for k in range(2):
                self._decode(torch.zeros(cfg.out_dim, cfg.chunk_size, *lat_hw, device=device), cache, clip_start=k == 0)

    def _decode(self, latents: torch.Tensor, cache: list, clip_start: bool) -> torch.Tensor:
        """``[1, 3, frames, H, W]`` in [-1, 1]: the chunk decoded as the
        continuation of the stream whose decoder cache is ``cache``."""
        # the de-normalize in fp32 whatever the decode's dtype
        z = (latents[None].float() / self._latents_inv_std + self._latents_mean).to(self._decode_dtype)
        return self._decoder.decode_stream(z, cache, clip_start=clip_start)

    def _decoder_cache(self, state) -> wan_vae.PoolFeatCache:
        """The rollout's decoder cache, its slots views of the request's pool buffers.
        Kept between chunks, and taken again when the pool has moved the region (a
        rearrange copies it elsewhere, so the old views no longer point at it)."""
        pool = self.state_pool()
        slots = decoder_cache_layout()
        cache = state["decoder_cache"]
        with torch.inference_mode():
            first = pool.view(decoder_slot_name(next(i for i, s in enumerate(slots) if s is not None)))
            if cache is not None and state["decoder_cache_at"] == first.data_ptr():
                return cache
            buffers = [None if slot is None else pool.view(decoder_slot_name(i)) for i, slot in enumerate(slots)]
            # a filled slot sits at the start of its buffer (PoolFeatCache)
            items = [b.as_strided(t.shape, t.stride()) if torch.is_tensor(t) else t
                     for b, t in zip(buffers, cache if cache is not None else [None] * len(slots), strict=True)]
        cache = wan_vae.PoolFeatCache(items, buffers)
        state.add_all(decoder_cache=cache, decoder_cache_at=first.data_ptr())
        return cache

    def _encoder_stream(self, state) -> EncoderStream:
        """The rollout's encode stream, its cache's slots views of the request's pool
        buffers. Kept between chunks, and taken again when the pool has moved the
        region (a rearrange copies it elsewhere, so the old views no longer point at it)."""
        pool = self.state_pool()
        slots = encoder_cache_layout().slots
        stream = state["encoder_stream"]
        with torch.inference_mode():
            anchor = pool.view(ENCODER_OUT)
            if stream is not None and state["encoder_stream_at"] == anchor.data_ptr():
                return stream
            if stream is not None and stream.cache is None:
                # settled: the output it keeps, at the start of its buffer
                stream = EncoderStream.over(None, anchor.as_strided(stream.steady.shape, stream.steady.stride()))
            else:
                buffers = [None if slot is None else pool.view(encoder_slot_name(i)) for i, slot in enumerate(slots)]
                old = stream.cache if stream is not None else [None] * len(slots)
                # a filled slot sits at the start of its buffer (PoolFeatCache)
                items = [b.as_strided(t.shape, t.stride()) if torch.is_tensor(t) else t
                         for b, t in zip(buffers, old, strict=True)]
                stream = EncoderStream.over(wan_vae.PoolFeatCache(items, buffers), None)
        state.add_all(encoder_stream=stream, encoder_stream_at=anchor.data_ptr())
        return stream

    def _keep_settled_output(self, state, stream: EncoderStream) -> None:
        """Once the stream settles, keep the output it returns from then on in the
        pool, where the cache was: copied there once."""
        steady = stream.steady
        if steady is None:
            return
        with torch.inference_mode():
            buf = self.state_pool().view(ENCODER_OUT)
            if steady.data_ptr() == buf.data_ptr():
                return
            kept = buf.as_strided(steady.shape, wan_vae._compact_strides(steady))
            kept.copy_(steady)
        state.add("encoder_stream", EncoderStream.over(None, kept))
