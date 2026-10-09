"""LTX-2.5 node submodules.

    text_encoder  -> LTXTextEncoderSubmodule   Gemma-4 hidden states + the two text connectors
    dit           -> LTXDenoiseSubmodule       DenoiseLoopSubmodule over the joint audio-video DiT
    vae_decoder   -> LTXVideoDecoderSubmodule  packed video latents -> uint8 frames
    audio_decoder -> LTXAudioDecoderSubmodule  packed audio latents -> mel -> 48 kHz stereo waveform

Numerics follow the checkpoint dtypes (bf16 weights, fp32 latents between steps and
fp32 step math), not the engine's autocast: ``LTX25Model.get_autocast_dtype`` is
None. Each forward mirrors ``LTX2Pipeline``'s op order; see
``integration_testing/ltx25/parity_notes.md`` for what was verified against it.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from dataclasses import dataclass

import numpy as np
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
from mstar.model.components.batched_rows import BatchedRows
from mstar.model.components.diffusion.attention import joint_attention, sdpa_attention
from mstar.model.components.diffusion.compile_utils import compile_transformer_forward
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule, euler_step
from mstar.model.ltx2_5.components.transformer import LTX2Attends, build_rope
from mstar.model.ltx2_5.config import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES, LTX25Config
from mstar.model.submodule_base import NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)

# Edge names shared with the model file.
TEXT_INPUTS = "text_inputs"
TEXT_VIDEO, TEXT_AUDIO = "text_video", "text_audio"
LATENTS, AUDIO_LATENTS = "latents", "audio_latents"
# the two-stage recipe's stage-2 starting point, made by the latent upsampler
REFINE_LATENTS, REFINE_AUDIO = "refine_latents", "refine_audio"
VIDEO_OUTPUT, AUDIO_OUTPUT = "video_output", "audio_output"

# The dit's ragged attention resources, one per (kind, head geometry): the video
# stream's 32 x 128 heads, and the 32 x 64 heads of the audio stream and of both
# cross-modal attentions.
VIDEO_ATTN, AUDIO_ATTN = "dit_attn_video", "dit_attn_audio"            # self-attention
VIDEO_XATTN, AUDIO_XATTN = "dit_xattn_video", "dit_xattn_audio"        # cross-attention
# (q_label, kv_label) pairs each cross resource plans
VIDEO_XATTN_PAIRS = (("video", "text"),)
AUDIO_XATTN_PAIRS = (("audio", "text"), ("video", "audio"), ("audio", "video"))
# Spans each request contributes to a dit step.
VIDEO_SPAN, AUDIO_SPAN, TEXT_SPAN = "video", "audio", "text"


@dataclass(frozen=True)
class LTXShape:
    """What a denoise step's shape is. Requests batch, and CUDA-graph buckets key, on it."""

    frames: int        # latent frames
    height: int        # latent rows
    width: int         # latent columns
    audio_frames: int  # audio latent frames (tokens)
    fps: float         # enters the video RoPE's time axis
    text_len: int

    @property
    def video_tokens(self) -> int:
        return self.frames * self.height * self.width

    @property
    def total_tokens(self) -> int:
        return self.video_tokens + self.audio_frames + self.text_len


def shape_from_metadata(config: LTX25Config, step_metadata: dict) -> LTXShape:
    geo = config.geometry
    num_frames, fps = int(step_metadata["num_frames"]), float(step_metadata["fps"])
    return LTXShape(
        frames=geo.latent_frames(num_frames),
        height=int(step_metadata["height"]) // geo.spatial_compression,
        width=int(step_metadata["width"]) // geo.spatial_compression,
        audio_frames=geo.audio_frames(num_frames, fps),
        fps=fps,
        text_len=config.text_max_seq_len,
    )


def distilled_schedule(sigma_values=DISTILLED_SIGMA_VALUES) -> FlowMatchSchedule:
    """An explicit sigma list as ``FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...)``
    builds it for this checkpoint (shift 1.0, no dynamic shifting): float32 sigmas used
    as given, ``timesteps = 1000 * sigmas``, a terminal 0 appended."""
    sigmas = torch.from_numpy(np.array(sigma_values).astype(np.float32))
    return FlowMatchSchedule(sigmas=torch.cat([sigmas, torch.zeros(1)]), timesteps=sigmas * 1000, mu=None)


def ltx_step(sample: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor):
    """``LTX2Pipeline``'s unguided step: the velocity goes to x0 and back in fp32 (the
    path its guidance terms are combined on), then the Euler update."""
    velocity = velocity.float()
    x0 = sample - velocity * sigma
    return euler_step(sample, (sample - x0) / sigma, sigma, sigma_next)


def pack_video(latents: torch.Tensor) -> torch.Tensor:
    """``_pack_latents`` at patch size 1: ``[B, C, F, H, W] -> [B, F*H*W, C]``."""
    return latents.flatten(2).transpose(1, 2)


def unpack_video(tokens: torch.Tensor, shape: LTXShape) -> torch.Tensor:
    return tokens.transpose(1, 2).unflatten(2, (shape.frames, shape.height, shape.width))


def pack_audio(latents: torch.Tensor) -> torch.Tensor:
    """``_pack_audio_latents``: ``[B, C, L, M] -> [B, L, C*M]``."""
    return latents.transpose(1, 2).flatten(2, 3)


def unpack_audio(tokens: torch.Tensor, mel_bins: int) -> torch.Tensor:
    return tokens.unflatten(2, (-1, mel_bins)).transpose(1, 2)


# ---------------------------------------------------------------------------
# text_encoder
# ---------------------------------------------------------------------------

class LTXTextEncoderSubmodule(BatchedRows, NodeSubmodule):
    """Prompt token ids -> the DiT's video ``[1024, 4096]`` and audio ``[1024, 2048]``
    text embeddings.

    One fused node: Gemma's 49 stacked hidden states are ~376 MB per prompt at full
    length, the connectors' outputs ~12 MB, so splitting the two would ship the large
    tensor across a node edge. Gemma runs each prompt on its real tokens alone (see
    ``Gemma4TextEncoder``), so requests of different lengths batch here by running
    Gemma per prompt and the fixed-length connectors once over the batch.

    Attention here runs through SDPA rather than an engine resource: Gemma's global
    layers have head dim 512, which no FlashInfer prefill kernel is built for. The
    node is eager and runs once per request.
    """

    disable_torch_compile = True
    output_keys = (TEXT_VIDEO, TEXT_AUDIO)

    def __init__(self, gemma: nn.Module, connectors: nn.Module, config: LTX25Config, max_batch_size: int = 8):
        super().__init__()
        self.gemma = gemma
        self.connectors = connectors
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
        return {TEXT_INPUTS: [inp.tensor_inputs[TEXT_INPUTS] for inp in inputs]}

    def run_batch(self, text_inputs: list[torch.Tensor], **kwargs):
        device = self.get_device()
        seq_len = self.config.text_max_seq_len
        valid = [int(ids.shape[0]) for ids in text_inputs]
        hidden = []
        for ids in text_inputs:
            n = int(ids.shape[0])
            # the positions the reference's left padding gives the real tokens
            positions = torch.arange(seq_len - n, seq_len, device=device)
            hidden.append(self.gemma(ids.to(device=device, dtype=torch.long)[None], positions[None])[0])
        longest = max(valid)
        stacked = torch.stack([
            torch.nn.functional.pad(h, (0, 0, 0, 0, 0, longest - h.shape[0])) for h in hidden
        ])
        video, audio = self.connectors(stacked, valid, seq_len, sdpa_attention)
        return {TEXT_VIDEO: video, TEXT_AUDIO: audio}


# ---------------------------------------------------------------------------
# dit (denoise loop body)
# ---------------------------------------------------------------------------

class LTXDenoiseSubmodule(DenoiseLoopSubmodule):
    """One distilled denoise step of the joint audio-video DiT; see ``DenoiseLoopSubmodule``.

    The loop carries the video latents (``latents``) and the audio latents
    (``audio_latents``, the scaffold's ``SOLVER_STATE``), both as fp32 packed tokens;
    each step casts them to bf16 for the DiT and updates them in fp32. Per-shape
    derived state is the four rotary tables (``build_rope``).

    Every attention in the DiT goes through a ragged resource (see ``VIDEO_ATTN``):
    three spans per request, ``video``, ``audio`` and ``text``; two self-attentions
    and four cross-attentions between them, on four resources (self / cross x the
    two head geometries). With no resource bound
    (``attention_backend="sdpa"``), every attention runs SDPA, which is what the
    reference pipeline runs.
    """

    SOLVER_STATE = (AUDIO_LATENTS,)

    def __init__(
        self,
        transformer: nn.Module,
        config: LTX25Config,
        *,
        loop_name: str,
        use_ragged_attention: bool,
        refine_walks: frozenset[str] = frozenset(),
        compile_transformer: bool = False,
        compile_eager_rounding: bool = True,
        max_batch_size: int = 4,
        capture_buckets=(),
        capture_batch_sizes=(1, 2, 4),
        replay_walks=None,
    ):
        super().__init__(
            loop_name=loop_name, max_batch_size=max_batch_size, attn_resource_key=None,
            capture_buckets=capture_buckets, capture_batch_sizes=capture_batch_sizes, replay_walks=replay_walks,
        )
        self.transformer = transformer
        self.config = config
        self.use_ragged_attention = use_ragged_attention
        # walks that run the two-stage recipe's stage 2 (the refine loop)
        self.refine_walks = frozenset(refine_walks)
        self._attends: LTX2Attends | None = None
        if compile_transformer and transformer is not None:
            # Fuses the AdaLN modulation, norms, RoPE and gating chains (44% of an eager
            # step's kernel time at 544x960x121) around the GEMMs and attention; one static
            # graph per shape, which the captured graphs then record.
            compile_transformer_forward(transformer, eager_rounding=compile_eager_rounding)

    # hooks -----------------------------------------------------------------
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> LTXShape:
        return shape_from_metadata(self.config, fwd_info.step_metadata)

    def schedule_for(self, fwd_info, bucket_key: LTXShape) -> FlowMatchSchedule:
        if fwd_info.graph_walk in self.refine_walks:
            return distilled_schedule(STAGE_2_DISTILLED_SIGMA_VALUES)
        return distilled_schedule()

    def initial_loop_back(self, fwd_info, inputs, bucket_key: LTXShape, generator: torch.Generator):
        """Stage 2 of the two-stage recipe starts from the upsampled stage-1 latents,
        blended with noise at its first sigma (``LTX2Pipeline._create_noised_state``).

        The reference threads one generator through both stages, so stage 2's noise
        continues the stream stage 1 drew from: the stage-1 draws are replayed (and
        dropped) first. Stage 2 draws its video noise in bf16 in the packed layout (it
        takes the dtype of the upsampled latents) and its audio noise in fp32."""
        if fwd_info.graph_walk not in self.refine_walks:
            return self.seed_loop_back(fwd_info, bucket_key, generator)
        stage1 = dataclasses.replace(bucket_key, height=bucket_key.height // 2, width=bucket_key.width // 2)
        self.seed_loop_back(fwd_info, stage1, generator)
        device = self.get_device()
        refine_video = inputs[REFINE_LATENTS][0].to(device)
        refine_audio = inputs[REFINE_AUDIO][0].to(device)
        video_noise = torch.randn((1, *refine_video.shape), generator=generator, dtype=refine_video.dtype)[0]
        audio_noise = torch.randn((1, *refine_audio.shape), generator=generator, dtype=refine_audio.dtype)[0]
        scale = STAGE_2_DISTILLED_SIGMA_VALUES[0]
        video = scale * video_noise.to(device) + (1 - scale) * refine_video
        audio = scale * audio_noise.to(device) + (1 - scale) * refine_audio
        return {LATENTS: video.to(torch.float32), AUDIO_LATENTS: audio.to(torch.float32)}

    def seed_loop_back(self, fwd_info, bucket_key: LTXShape, generator: torch.Generator) -> dict[str, torch.Tensor]:
        # LTX2Pipeline draws the video noise, then the audio noise, from one generator,
        # both fp32 in the unpacked layout, then packs them.
        geo = self.config.geometry
        video = torch.randn(
            (1, geo.latent_channels, bucket_key.frames, bucket_key.height, bucket_key.width),
            generator=generator, dtype=torch.float32,
        )
        audio = torch.randn(
            (1, geo.audio_latent_channels, bucket_key.audio_frames, geo.audio_latent_mel_bins),
            generator=generator, dtype=torch.float32,
        )
        return {LATENTS: pack_video(video)[0], AUDIO_LATENTS: pack_audio(audio)[0]}

    def request_inputs(self, fwd_info, inputs: NameToTensorList, bucket_key: LTXShape) -> dict[str, torch.Tensor]:
        return {TEXT_VIDEO: inputs[TEXT_VIDEO][0], TEXT_AUDIO: inputs[TEXT_AUDIO][0]}

    def num_tokens(self, bucket_key: LTXShape) -> int:
        return bucket_key.total_tokens

    def capture_request_inputs(self, bucket_key: LTXShape, device) -> dict[str, torch.Tensor]:
        cfg = self.config.transformer
        dtype = self.transformer.dtype
        return {
            LATENTS: torch.zeros(bucket_key.video_tokens, cfg.in_channels, dtype=torch.float32, device=device),
            AUDIO_LATENTS: torch.zeros(bucket_key.audio_frames, cfg.audio_in_channels, dtype=torch.float32,
                                       device=device),
            TEXT_VIDEO: torch.zeros(bucket_key.text_len, cfg.cross_attention_dim, dtype=dtype, device=device),
            TEXT_AUDIO: torch.zeros(bucket_key.text_len, cfg.audio_cross_attention_dim, dtype=dtype, device=device),
        }

    def build_layout(self, bucket_key: LTXShape, device):
        return build_rope(
            self.config.transformer, bucket_key.frames, bucket_key.height, bucket_key.width, bucket_key.fps,
            bucket_key.audio_frames, device,
        )

    def denoise(
        self, engine_inputs, bucket_key: LTXShape, latents, timestep, sigma, sigma_next,
        audio_latents, text_video, text_audio, **cond,
    ):
        dtype = self.transformer.dtype
        velocity, audio_velocity = self.transformer(
            latents.to(dtype), audio_latents.to(dtype), text_video, text_audio, timestep,
            self.layout(bucket_key, latents.device), self.attends(),
        )
        return {
            LATENTS: ltx_step(latents, velocity, sigma, sigma_next),
            AUDIO_LATENTS: ltx_step(audio_latents, audio_velocity, sigma, sigma_next),
        }

    # attention ------------------------------------------------------------
    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._attends = None  # the memoized callables hold the old resources

    def attends(self) -> LTX2Attends:
        """The six attentions' callables, built once per resource binding: a compiled
        region guards on their identity (see ``DenoiseLoopSubmodule.ragged_for``)."""
        if self._attends is None:
            keys = (VIDEO_ATTN, AUDIO_ATTN, VIDEO_XATTN, AUDIO_XATTN)
            bound = [self.node_resources.get(key) for key in keys] if self.use_ragged_attention else []
            if len(bound) < len(keys) or any(r is None for r in bound):
                self._attends = LTX2Attends(*([sdpa_attention] * 6))
            else:
                video, audio, video_x, audio_x = bound

                def on(resource, label):
                    # every row has the bucket's spans, so [B, L, H, D] packs per request
                    return functools.partial(joint_attention, ragged=RaggedAttentionCallable(resource, label))
                self._attends = LTX2Attends(
                    video_self=on(video, VIDEO_SPAN),
                    audio_self=on(audio, AUDIO_SPAN),
                    video_text=on(video_x, cross_label(VIDEO_SPAN, TEXT_SPAN)),
                    audio_text=on(audio_x, cross_label(AUDIO_SPAN, TEXT_SPAN)),
                    audio_to_video=on(audio_x, cross_label(VIDEO_SPAN, AUDIO_SPAN)),
                    video_to_audio=on(audio_x, cross_label(AUDIO_SPAN, VIDEO_SPAN)),
                )
        return self._attends

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases=None, **kwargs,
    ) -> SubmoduleStep | None:
        if not self.use_ragged_attention:
            return None
        video, audio, text = [], [], []
        for rid, inp in zip(request_ids, inputs, strict=True):
            shape: LTXShape = inp.resource_step_info
            video.append(Segment(request_id=rid, label=VIDEO_SPAN, span=shape.video_tokens))
            audio.append(Segment(request_id=rid, label=AUDIO_SPAN, span=shape.audio_frames))
            text.append(Segment(request_id=rid, label=TEXT_SPAN, span=shape.text_len))
        return SubmoduleStep(
            steps={
                VIDEO_ATTN: AttentionStep(segments=tuple(video), causal=False),
                AUDIO_ATTN: AttentionStep(segments=tuple(audio), causal=False),
                VIDEO_XATTN: RaggedCrossAttentionStep(segments=tuple(video + text), pairs=VIDEO_XATTN_PAIRS),
                AUDIO_XATTN: RaggedCrossAttentionStep(
                    segments=tuple(video + audio + text), pairs=AUDIO_XATTN_PAIRS,
                ),
            },
            cg_key_info=self._uniform_key(inp.resource_step_info for inp in inputs),
        )


class _ShapeBatchedBase:
    """Requests batch when their latent shape matches."""

    _max_batch_size: int

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1 and len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size


# ---------------------------------------------------------------------------
# latent_upsampler (two-stage recipe)
# ---------------------------------------------------------------------------

class LTXLatentUpsamplerSubmodule(_ShapeBatchedBase, BatchedRows, NodeSubmodule):
    """Stage 1's final latents -> stage 2's starting latents, before the noise blend.

    Mirrors the card's recipe through the pipelines' latent space round trips: stage 1
    returns denormalized latents (fp32), ``LTX2LatentUpsamplePipeline`` runs the x2
    spatial upsampler in bf16, and stage 2's ``prepare_latents`` re-normalizes (in bf16)
    and packs. The audio passes through the same denormalize / re-normalize round trip
    in fp32. Emits ``refine_latents`` ``[L2, 128]`` bf16 and ``refine_audio`` ``[La, 128]``.
    """

    disable_torch_compile = True
    output_keys = (REFINE_LATENTS, REFINE_AUDIO)

    def __init__(self, upsampler: nn.Module, vae_stats: tuple[torch.Tensor, torch.Tensor, float],
                 audio_stats: tuple[torch.Tensor, torch.Tensor], config: LTX25Config, max_batch_size: int = 2):
        super().__init__()
        self.upsampler = upsampler
        mean, std, scaling = vae_stats
        self.register_buffer("vae_mean", mean.view(1, -1, 1, 1, 1).clone(), persistent=False)
        self.register_buffer("vae_std", std.view(1, -1, 1, 1, 1).clone(), persistent=False)
        self.scaling = float(scaling)
        self.register_buffer("audio_mean", audio_stats[0].clone(), persistent=False)
        self.register_buffer("audio_std", audio_stats[1].clone(), persistent=False)
        self.config = config
        self._max_batch_size = max_batch_size

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={LATENTS: inputs[LATENTS][0], AUDIO_LATENTS: inputs[AUDIO_LATENTS][0]},
            resource_step_info=shape_from_metadata(self.config, fwd_info.step_metadata),
        )

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {
            LATENTS: torch.stack([inp.tensor_inputs[LATENTS] for inp in inputs]),
            AUDIO_LATENTS: torch.stack([inp.tensor_inputs[AUDIO_LATENTS] for inp in inputs]),
            "shape": inputs[0].resource_step_info,
        }

    def run_batch(self, latents: torch.Tensor, audio_latents: torch.Tensor, shape: LTXShape, **kwargs):
        device = self.vae_mean.device
        x = unpack_video(latents.to(device=device, dtype=torch.float32), shape)
        x = x * self.vae_std.float() / self.scaling + self.vae_mean.float()
        x = self.upsampler(x.to(self.upsampler.dtype))
        dtype = x.dtype
        x = (x - self.vae_mean.to(dtype)) * self.scaling / self.vae_std.to(dtype)
        a = audio_latents.to(device=device, dtype=torch.float32)
        a = (a * self.audio_std + self.audio_mean - self.audio_mean) / self.audio_std
        return {REFINE_LATENTS: pack_video(x), REFINE_AUDIO: a}


# ---------------------------------------------------------------------------
# decoders
# ---------------------------------------------------------------------------

class LTXVideoDecoderSubmodule(_ShapeBatchedBase, BatchedRows, NodeSubmodule):
    """Packed fp32 video latents ``[Lv, 128]`` -> uint8 frames ``[3, F, H, W]``, through the
    convolutional VAE. The reference casts the final latents to bf16 before
    denormalizing; quantizing at the worker boundary keeps the edge one byte per channel."""

    disable_torch_compile = True
    output_keys = (VIDEO_OUTPUT,)

    def __init__(self, vae: nn.Module, config: LTX25Config, max_batch_size: int = 2,
                 tile_min_pixels: int = 1280 * 720):
        super().__init__()
        self.vae = vae
        self.config = config
        self._max_batch_size = max_batch_size
        # Frames larger than this decode in spatial tiles (the card's recipe tiles its
        # 1088x1920 stage-2 decode); smaller ones decode whole, as the single-stage
        # pipeline does. Tiling changes the output near tile seams.
        self.tile_min_pixels = int(tile_min_pixels)

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={LATENTS: inputs[LATENTS][0]},
            resource_step_info=shape_from_metadata(self.config, fwd_info.step_metadata),
        )

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {LATENTS: torch.stack([inp.tensor_inputs[LATENTS] for inp in inputs]),
                "shape": inputs[0].resource_step_info}

    def run_batch(self, latents: torch.Tensor, shape: LTXShape, **kwargs):
        dtype = self.vae.dtype
        x = unpack_video(latents.to(device=self.get_device(), dtype=dtype), shape)
        mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(x.device, dtype)
        std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(x.device, dtype)
        x = x * std / self.vae.config.scaling_factor + mean
        sc = self.config.geometry.spatial_compression
        self.vae.use_tiling = shape.height * sc * shape.width * sc > self.tile_min_pixels
        video = self.vae.decode(x, None, return_dict=False)[0]
        frames = ((video / 2 + 0.5).clamp(0, 1).float() * 255).round().to(torch.uint8)
        return {VIDEO_OUTPUT: frames}


class LTXAudioDecoderSubmodule(_ShapeBatchedBase, BatchedRows, NodeSubmodule):
    """Packed fp32 audio latents ``[La, 128]`` -> a ``[2, samples]`` fp32 waveform at the
    vocoder's 48 kHz: denormalize over the packed features, unpack to
    ``[8, La, 16]``, audio VAE -> mel, vocoder (with bandwidth extension)."""

    disable_torch_compile = True
    output_keys = (AUDIO_OUTPUT,)

    def __init__(self, audio_vae: nn.Module, vocoder: nn.Module, config: LTX25Config, max_batch_size: int = 4):
        super().__init__()
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.config = config
        self._max_batch_size = max_batch_size

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={AUDIO_LATENTS: inputs[AUDIO_LATENTS][0]},
            resource_step_info=shape_from_metadata(self.config, fwd_info.step_metadata),
        )

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {AUDIO_LATENTS: torch.stack([inp.tensor_inputs[AUDIO_LATENTS] for inp in inputs])}

    def run_batch(self, audio_latents: torch.Tensor, **kwargs):
        x = audio_latents.to(device=self.get_device(), dtype=torch.float32)
        x = x * self.audio_vae.latents_std.to(x.device, x.dtype) + self.audio_vae.latents_mean.to(x.device, x.dtype)
        x = unpack_audio(x, self.config.geometry.audio_latent_mel_bins).to(self.audio_vae.dtype)
        mel = self.audio_vae.decode(x, return_dict=False)[0]
        return {AUDIO_OUTPUT: self.vocoder(mel).float()}


__all__ = [
    "LTXShape", "LTXTextEncoderSubmodule", "LTXDenoiseSubmodule", "LTXVideoDecoderSubmodule",
    "LTXAudioDecoderSubmodule", "shape_from_metadata",
]
