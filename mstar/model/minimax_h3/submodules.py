"""MiniMax-H3 node submodules.

    text_encoder   -> H3TextEncoderSubmodule   Qwen3-VL layer 50 (keyframes through its tower)
                                               -> condition_proj -> token refiner
    vae_encoder    -> H3VaeEncoderSubmodule    keyframes -> conditioning latent rows
    dit            -> H3DenoiseSubmodule       DenoiseLoopSubmodule over the packed-sequence DiT
    vae_decoder    -> H3VideoDecoderSubmodule  video latent rows -> uint8 frames
    audio_decoder  -> H3AudioDecoderSubmodule  audio latent rows -> 32 kHz stereo waveform

Numerics follow the checkpoint's dtypes (bf16, with the reference's fp32 modules and
step math), not the engine's autocast. ``test/minimax_h3/parity.py`` checks each stage
against diffusers' pipeline.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup
from mstar.engine.state import ArrayStep, Segment, SubmoduleStep
from mstar.model.components.batched_rows import BatchedRows
from mstar.model.components.diffusion.attention import FrameSDPA, sdpa_attention
from mstar.model.components.diffusion.compile_utils import compile_blocks as compile_blocks_
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule, StepScalar
from mstar.model.components.diffusion.qwen3.encoder import qwen3_inv_freq
from mstar.model.components.diffusion.qwen3.vl_prompt import qwen3_vl_prompt
from mstar.model.components.qwen3_vit import vision_interpolation, vision_position_ids, vision_seq_lengths
from mstar.model.components.qwen_vision import VISION_ATTN, QwenVisionModel
from mstar.model.minimax_h3.components.layout import (
    H3Layout,
    modality_tags,
    patchify_video,
    position_ids,
    timestep_classes,
    unpatchify_video,
)
from mstar.model.minimax_h3.components.transformer import (
    H3Inputs,
    H3StepLayout,
    rope_inv_freq,
    rope_table,
)
from mstar.model.minimax_h3.config import (
    AUDIO_CHANNELS,
    DIT_ROWS,
    KEYFRAME_ENCODE_SEED,
    KEYFRAME_NOISE_AUG,
    NUM_MODALITIES,
    PIXEL_MEAN,
    PIXEL_STD,
    PROMPT_ROWS,
    H3Config,
)
from mstar.model.minimax_h3.schedule import H3Schedule, ScheduleArgs, h3_schedule, h3_step
from mstar.model.submodule_base import NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking

logger = logging.getLogger(__name__)

# Edge names shared with the model file.
TEXT_INPUTS, TEXT_EMBEDS = "text_inputs", "text_embeds"
LATENTS, AUDIO_LATENTS = "latents", "audio_latents"
VIDEO_OUTPUT, AUDIO_OUTPUT = "video_output", "audio_output"
# a denoise step input, not an edge: whether this is a request's first step (host-side)
FIRST_STEP = "first_step"
# FL2VA: the keyframes as the Qwen3-VL image processor lays them out (``[patches, 1536]``),
# the keyframes' uint8 pixels for the VAE encoder, and their normalized latent rows
VISION_PIXELS, KEYFRAME_PIXELS, COND_LATENTS = "vision_pixels", "keyframe_pixels", "cond_latents"
TEXT_SPAN = "text"


def layout_from_metadata(step_metadata: dict) -> H3Layout:
    """The request's packed layout, from the host values its step metadata carries."""
    return H3Layout(
        text_len=int(step_metadata["text_len"]),
        latent_frames=int(step_metadata["latent_frames"]),
        latent_height=int(step_metadata["latent_height"]),
        latent_width=int(step_metadata["latent_width"]),
        audio_latents=int(step_metadata["audio_latents"]),
        anchors=tuple(step_metadata.get("anchors", ())),
        text_video_spans=tuple(tuple(span) for span in step_metadata.get("text_video_spans", ())),
    )


# ---------------------------------------------------------------------------
# text_encoder
# ---------------------------------------------------------------------------

class TextStep(NamedTuple):
    length: int
    grids: tuple[tuple[int, int, int], ...] = ()   # each keyframe's (t, h, w) patch grid


class H3TextEncoderSubmodule(NodeSubmodule):
    """Prompt token ids ``[L]`` -> the DiT's text rows ``[L, hidden]`` bf16: Qwen3-VL-32B
    through layer 50, then the DiT's ``condition_proj`` and token refiner. The refiner's
    output doesn't depend on the step, so it runs here, once per request. With keyframes
    the prompt is laid out by ``qwen3_vl_prompt`` around the vision tower's tokens.
    """

    disable_torch_compile = True

    def __init__(self, encoder: nn.Module, conditioner: nn.Module, *, vision: QwenVisionModel | None = None,
                 config: H3Config | None = None):
        super().__init__()
        self.encoder = encoder
        self.conditioner = conditioner
        self.vision = vision
        self.h3_config = config
        if vision is not None:
            # per-frame SDPA, not a node resource: the tower runs once per request on a few
            # thousand patches
            self._frames = FrameSDPA()
            for block in vision.blocks:
                block.attn.bind_resources({VISION_ATTN: self._frames})
            text = config.text
            self.register_buffer("mrope_inv_freq", qwen3_inv_freq(text.head_dim, text.rope_theta), persistent=False)

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        ids = inputs[TEXT_INPUTS][0]
        tensors = {TEXT_INPUTS: ids}
        grids: tuple[tuple[int, int, int], ...] = ()
        if inputs.get(VISION_PIXELS):
            tensors[VISION_PIXELS] = inputs[VISION_PIXELS][0]
            grids = tuple(tuple(int(v) for v in g) for g in fwd_info.step_metadata["keyframe_grids"])
        return NodeInputs(tensor_inputs=tensors, input_seq_len=int(ids.shape[0]),
                          resource_step_info=TextStep(int(ids.shape[0]), grids))

    def can_batch(self, batch, model_inputs) -> bool:
        return False

    def max_batch_size(self, graph_walk: str):
        return 1

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {**inputs[0].tensor_inputs, "step": inputs[0].resource_step_info}

    def forward(self, graph_walk, engine_inputs, text_inputs: torch.Tensor, step: TextStep,
                vision_pixels: torch.Tensor | None = None, **kwargs) -> NameToTensorList:
        device = self.get_device()
        ids = text_inputs.to(device=device, dtype=torch.long)
        if vision_pixels is None:
            hidden = self.encoder(ids[None], torch.ones_like(ids[None]), padded=False)
        else:
            hidden = self._encode_with_images(ids, vision_pixels.to(device), list(step.grids))
        return {TEXT_EMBEDS: [self.conditioner(hidden)[0]]}

    def _encode_with_images(self, ids: torch.Tensor, pixels: torch.Tensor, grids: list) -> torch.Tensor:
        vision, device = self.vision, ids.device
        merge = vision.config.spatial_merge_size
        self._frames.lengths = vision_seq_lengths(grids)
        indices, weights = vision_interpolation(grids, vision.config.num_grid_per_side, merge)
        positions = vision_position_ids(grids, merge)
        merged, deepstack = vision(
            pixels, to_device_nonblocking(indices, device), to_device_nonblocking(weights, device),
            to_device_nonblocking(positions, device),
        )
        prompt = qwen3_vl_prompt(
            ids, self.h3_config.image_token_id, grids, merge, merged, deepstack, self.encoder.embed_tokens,
            self.mrope_inv_freq, list(self.h3_config.mrope_section),
        )
        ones = torch.ones_like(ids[None])
        return self.encoder(ids[None], ones, padded=False, inputs_embeds=prompt.inputs_embeds,
                            rotary=prompt.rotary, deepstack=prompt.deepstack)

    def declare_step(self, graph_walk, request_ids, inputs: list[NodeInputs], slot_lease=None, piecewise_leases=None,
                     **kwargs) -> SubmoduleStep:
        segments = [Segment(request_id=rid, label=TEXT_SPAN, span=int(inp.input_seq_len))
                    for rid, inp in zip(request_ids, inputs, strict=True)]
        return SubmoduleStep(segments=segments, state={PROMPT_ROWS: ArrayStep(segments=tuple(segments))})


# ---------------------------------------------------------------------------
# vae_encoder (FL2VA keyframes)
# ---------------------------------------------------------------------------

class H3VaeEncoderSubmodule(NodeSubmodule):
    """The keyframes, uint8 ``[n, 3, H, W]`` on the canvas -> their normalized latents as
    rows ``[n * rows_per_frame, 96]`` fp32, in packed order: each encoded alone, its
    posterior sampled under the keyframe seed and rounded to fp16 (``encode_vae_condition``).
    Once per request."""

    disable_torch_compile = True
    output_keys = (COND_LATENTS,)

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={KEYFRAME_PIXELS: inputs[KEYFRAME_PIXELS][0]},
                          resource_step_info=layout_from_metadata(fwd_info.step_metadata))

    def can_batch(self, batch, model_inputs) -> bool:
        return False

    def max_batch_size(self, graph_walk: str):
        return 1

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {KEYFRAME_PIXELS: inputs[0].tensor_inputs[KEYFRAME_PIXELS], "layout": inputs[0].resource_step_info}

    def forward(self, graph_walk, engine_inputs, keyframe_pixels: torch.Tensor, layout: H3Layout,
                **kwargs) -> NameToTensorList:
        device = self.get_device()
        frames = keyframe_pixels.to(device)
        rows = [patchify_video(self.encoder.encode(frame, KEYFRAME_ENCODE_SEED)[0], layout.patch) for frame in frames]
        return {COND_LATENTS: [torch.cat(rows).to(device)]}


# ---------------------------------------------------------------------------
# dit (denoise loop body)
# ---------------------------------------------------------------------------

class H3DenoiseSubmodule(DenoiseLoopSubmodule):
    """One denoise step of the packed audio-video DiT (``DenoiseLoopSubmodule``). The
    loop carries the video rows (keyframe rows first) and the audio rows, fp32; only the
    generated rows are stepped. The fp32 GEMMs run at full fp32, as the reference's do.
    """

    SOLVER_STATE = (AUDIO_LATENTS,)
    STEP_SCALARS = {
        "sigma": StepScalar("sigmas"),
        "sigma_next": StepScalar("sigmas", offset=1, capture_fill=0.0),
        "timestep": StepScalar("timesteps", broadcast=False, capture_fill=0.5),
        "audio_sigma": StepScalar("audio_sigmas"),
        "audio_sigma_next": StepScalar("audio_sigmas", offset=1, capture_fill=0.0),
        "audio_timestep": StepScalar("audio_timesteps", broadcast=False, capture_fill=0.5),
        "cond_timestep": StepScalar("cond_timesteps", broadcast=False, capture_fill=1.0),
    }
    PREFETCH_SEEDS = True  # the noise depends on the layout and the seed alone
    float32_matmul_precision = "highest"

    def __init__(self, transformer: nn.Module, config: H3Config, *, loop_name: str, max_batch_size: int = 1,
                 compile_blocks: bool = False, row_bucket: int = 1):
        if max_batch_size != 1:
            # a batch's rows would have to agree on the step's first-step decisions (denoise)
            raise ValueError(f"the MiniMax-H3 dit runs one request per step; got max_batch_size={max_batch_size}")
        if int(row_bucket) < 1:
            raise ValueError(f"row_bucket must be a positive int; got {row_bucket!r}")
        super().__init__(loop_name=loop_name, max_batch_size=1)
        self.transformer = transformer
        self.config = config
        self.compiled = bool(compile_blocks)
        sp_group = getattr(transformer, "sp_group", None)
        # The packed rows pad up to a multiple of this, as a tail segment of their own: the
        # compiled blocks see one shape per bucket rather than per prompt length, and the rows
        # split evenly over the SP ranks, which keeps the exchanges on their copy-free path.
        self.row_multiple = math.lcm(int(row_bucket), sp_group.world_size if sp_group is not None else 1)
        if compile_blocks and transformer is not None:
            # around the attention and the exchanges, static per row bucket: each new bucket
            # compiles once (5-8 s), and a step then runs the same kernels, at static speed, whatever
            # was served before it (dynamic-shape kernels ran 14-19% slower per step)
            compile_blocks_(transformer.blocks, dynamic=False)

    # hooks -----------------------------------------------------------------
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> H3Layout:
        return layout_from_metadata(fwd_info.step_metadata)

    def schedule_settings(self, fwd_info: CurrentForwardPassInfo) -> ScheduleArgs:
        meta = fwd_info.step_metadata
        return ScheduleArgs(int(meta["num_inference_steps"]), float(meta["video_shift"]), float(meta["audio_shift"]))

    def schedule_for(self, walk: str, settings: ScheduleArgs, bucket_key: H3Layout) -> H3Schedule:
        return h3_schedule(*settings)

    def seed_loop_back(self, bucket_key: H3Layout, generator: torch.Generator) -> dict[str, torch.Tensor]:
        # the reference's draw order from the one generator: each keyframe, the video
        # latents, then the audio rows
        cfg = self.config.transformer
        cond_noise = [
            patchify_video(torch.randn((1, cfg.latents_dim, 1, bucket_key.latent_height, bucket_key.latent_width),
                                       generator=generator, dtype=torch.float32)[0], bucket_key.patch)
            for _ in bucket_key.anchors
        ]
        video = torch.randn(
            (1, cfg.latents_dim, bucket_key.latent_frames, bucket_key.latent_height, bucket_key.latent_width),
            generator=generator, dtype=torch.float32,
        )
        audio = torch.randn((bucket_key.num_audio_rows, cfg.audio_latents_dim), generator=generator,
                            dtype=torch.float32)
        rows = patchify_video(video[0], bucket_key.patch)
        if cond_noise:
            rows = torch.cat([*cond_noise, rows])
        return {LATENTS: rows, AUDIO_LATENTS: audio}

    def request_inputs(self, fwd_info, inputs: NameToTensorList, bucket_key: H3Layout) -> dict[str, torch.Tensor]:
        # host-side, so what depends on the first step is decided without a sync
        tensors = {TEXT_EMBEDS: inputs[TEXT_EMBEDS][0], FIRST_STEP: torch.tensor(self.step_index(fwd_info) == 0)}
        if bucket_key.num_condition_rows:
            tensors[COND_LATENTS] = inputs[COND_LATENTS][0]
        return tensors

    @property
    def row_multiple(self) -> int:
        """The packed rows pad up to a multiple of this, as a tail segment of their own: the
        compiled blocks see one shape per bucket rather than per prompt length, and the rows
        split evenly over the sequence shard's SP ranks, which keeps the exchanges on their
        copy-free path."""
        shard = self.node_resources.get(self.shard_resource_key) if self.shard_resource_key else None
        return math.lcm(self.row_bucket, 1 if shard is None else shard.group.world_size)

    def padded_rows(self, bucket_key: H3Layout) -> int:
        return -(-bucket_key.total_rows // self.row_multiple) * self.row_multiple

    def num_tokens(self, bucket_key: H3Layout) -> int:
        return self.padded_rows(bucket_key)

    def attention_segments(self, bucket_key: H3Layout) -> dict[str, tuple[tuple[str, int], ...]]:
        rows, pad = bucket_key.total_rows, self.padded_rows(bucket_key) - bucket_key.total_rows
        return {DIT_ROWS: ((self.MAIN_SPAN, rows),) + (((self.MAIN_SPAN, pad),) if pad else ())}

    def capture_request_inputs(self, bucket_key: H3Layout, device) -> dict[str, torch.Tensor]:
        cfg = self.config.transformer
        return {
            LATENTS: torch.zeros(bucket_key.num_condition_rows + bucket_key.num_video_rows, cfg.patch_dim,
                                 device=device),
            AUDIO_LATENTS: torch.zeros(bucket_key.num_audio_rows, cfg.audio_latents_dim, device=device),
            TEXT_EMBEDS: torch.zeros(bucket_key.text_len, cfg.hidden_size, dtype=self.transformer.dtype,
                                     device=device),
            FIRST_STEP: torch.tensor(False),
            **({COND_LATENTS: torch.zeros(bucket_key.num_condition_rows, cfg.patch_dim, device=device)}
               if bucket_key.num_condition_rows else {}),
        }

    def build_layout(self, bucket_key: H3Layout, device) -> H3StepLayout:
        pad = self.padded_rows(bucket_key) - bucket_key.total_rows

        def padded(rows: torch.Tensor) -> torch.Tensor:
            # a padding row sits at position 0 (rope cos 1, sin 0) on AdaLN table row 0
            return torch.cat([rows, rows.new_zeros((pad, *rows.shape[1:]))])

        # the positions travel to the device in fp64 and round to fp32 there, as the reference's do
        pos = padded(position_ids(bucket_key)).to(device)
        cfg = self.config.transformer
        rope = rope_table(pos, rope_inv_freq(cfg), self.transformer.dtype, cfg.attention_head_dim)
        classes, tags = padded(timestep_classes(bucket_key)), padded(modality_tags(bucket_key))
        return H3StepLayout(
            rope=rope, row_index=(classes * NUM_MODALITIES + tags).to(device), row_class=classes.to(device),
            no_class=torch.zeros_like(classes).to(device), tags=tags.to(device), text_len=bucket_key.text_len,
            num_condition_rows=bucket_key.num_condition_rows, num_audio_rows=bucket_key.num_audio_rows,
            num_pad_rows=pad,
        )

    def denoise(
        self, engine_inputs, bucket_key: H3Layout, latents, timestep, sigma, sigma_next,
        audio_latents, audio_timestep, audio_sigma, audio_sigma_next, cond_timestep, text_embeds,
        first_step, cond_latents=None, **cond,
    ):
        layout = self.layout(bucket_key, latents.device)
        n = layout.num_condition_rows
        first = bool(first_step.all())  # a CPU tensor: no sync
        if n and first:
            # the keyframe rows arrive as noise (seed_loop_back), are noised to t = 0.999
            # here and carried unchanged after
            t = torch.full((), KEYFRAME_NOISE_AUG, dtype=torch.float32, device=latents.device)
            latents = torch.cat([t * cond_latents + (1.0 - t) * latents[:, :n], latents[:, n:]], dim=1)
        if first and not n and not self.compiled:
            # video and audio both at t = 0: one timestep, as the reference embeds it. Compiled
            # (not bit-exact anyway) step 0 keeps the three-entry table, the same values, so a
            # row bucket compiles one graph rather than a second for step 0
            timesteps = timestep.view(-1, 1).float()
        else:
            timesteps = torch.stack([timestep, audio_timestep, cond_timestep], dim=-1).float()
        inputs = H3Inputs(video=latents, audio=audio_latents, text=text_embeds, timesteps=timesteps)
        video_v, audio_v = self.transformer(inputs, layout)
        generated = h3_step(latents[:, n:], video_v[:, n:], timestep.view(-1, 1, 1), sigma, sigma_next)
        return {
            LATENTS: torch.cat([latents[:, :n], generated], dim=1) if n else generated,
            AUDIO_LATENTS: h3_step(audio_latents, audio_v, audio_timestep.view(-1, 1, 1), audio_sigma,
                                   audio_sigma_next),
        }


# ---------------------------------------------------------------------------
# decoders
# ---------------------------------------------------------------------------

class _LayoutBatched:
    """Requests batch when their layout matches."""

    _max_batch_size: int

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1 and len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size


class _VideoFrames(nn.Module):
    """The decode's ends: latent rows to the VAE's input, and its output to frames."""

    def __init__(self, config: H3Config):
        super().__init__()
        self.config = config
        vcfg = config.video_vae
        self.register_buffer("latents_mean", torch.tensor(vcfg.latents_mean).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(vcfg.latents_std).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1), persistent=False)

    def latents(self, rows: torch.Tensor, layout: H3Layout) -> torch.Tensor:
        """Video rows ``[B, Nv, 96]`` -> denormalized latents ``[B, 24, F, H, W]``, keyframe rows dropped."""
        rows = rows.to(dtype=torch.float32)[:, layout.num_condition_rows:]
        z = unpatchify_video(rows, layout, self.config.video_vae.latent_channels)
        return z * self.latents_std + self.latents_mean

    def frames(self, video: torch.Tensor) -> torch.Tensor:
        """The VAE's ImageNet-normalized output -> uint8 rgb24 ``[B, F, H, W, 3]``."""
        video = (video.float() * self.pixel_std + self.pixel_mean).clamp(0, 1)
        return (video * 255).round().to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous()


class H3VideoDecoderSubmodule(_LayoutBatched, BatchedRows, NodeSubmodule):
    """Video rows ``[Nv, 96]`` fp32 -> uint8 rgb24 frames ``[F, H, W, 3]``: keyframe rows
    dropped, unpatchified, denormalized with the VAE's latent statistics, decoded in
    clips and tiles, the ImageNet normalization undone, clamped and rounded.

    On SP ranks its sequence shard (``shard_resource_key``, ``exchanges=False``: nothing mixes
    clips but the cross-fade after the gather) splits each request's clips: each rank decodes
    its contiguous share and the shares are gathered on every rank, the DiT's ranks decoding
    in lockstep, scheduled by its rank 0 between denoise steps."""

    # the label a step declares a request's clips under
    CLIPS = "clips"

    disable_torch_compile = True
    output_keys = (VIDEO_OUTPUT,)

    def __init__(self, vae: nn.Module, config: H3Config, *, max_batch_size: int = 1, tile_batch: int = 1,
                 compile_blocks: bool = False, shard_resource_key: str | None = None):
        super().__init__()
        self.vae = vae
        self.shard_resource_key = shard_resource_key
        if compile_blocks and vae is not None:
            # static per tile count: a rank's tiles (its clips' times the canvas's) go
            # tile_batch at a time, so a count tile_batch doesn't divide, e.g. 2 x 16 at
            # 768x768 in calls of 28, compiles its remainder too
            compile_blocks_(vae.decoder.transformer_blocks)
        self.ends = _VideoFrames(config)
        self._max_batch_size = max_batch_size
        # spatial tiles decoded per decoder call: 1 is the reference's order of work
        self.tile_batch = int(tile_batch)

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={LATENTS: inputs[LATENTS][0]},
                          resource_step_info=layout_from_metadata(fwd_info.step_metadata))

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {LATENTS: torch.stack([inp.tensor_inputs[LATENTS] for inp in inputs]),
                "layout": inputs[0].resource_step_info}

    def declare_step(self, graph_walk, request_ids, inputs: list[NodeInputs], **kwargs) -> SubmoduleStep | None:
        if self.shard_resource_key is None:
            return None
        # a batch shares one layout (`can_batch`), so every request's clips are one count
        segments = [
            Segment(rid, self.CLIPS, self.vae.walk.num_clips(inp.resource_step_info.latent_frames))
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        return SubmoduleStep(steps={}, segments=segments)

    def run_batch(self, latents: torch.Tensor, layout: H3Layout, **kwargs):
        # SDPA within each fixed-size tile, as the reference: dense, one segment, no capture,
        # so a ragged resource would add planning for nothing
        z = self.ends.latents(latents.to(self.get_device()), layout)
        shard = self.node_resources.get(self.shard_resource_key) if self.shard_resource_key else None
        split = None if shard is None else shard.split_range(self.CLIPS)
        if split is None:
            return {VIDEO_OUTPUT: self.ends.frames(self.vae.decode(z, sdpa_attention, tile_batch=self.tile_batch))}
        return {VIDEO_OUTPUT: self.ends.frames(self._decode_shared(z, shard, split))}

    def _decode_shared(self, z: torch.Tensor, shard, split) -> torch.Tensor:
        """This rank's range of the clips (``split``), gathered in clip order on every rank and
        cross-faded there. A rank with no clips (fewer clips than ranks) decodes nothing."""
        num_latents = z.shape[2]
        z, num_clips = self.vae.clip_plan(z)
        if sum(split.counts) != num_clips:
            # the declared count is the same on every rank, so every rank refuses alike
            raise ValueError(f"the step declared {sum(split.counts)} clips, the decode plans {num_clips}")
        clips = self.vae.decode_clips(z, range(split.start, split.start + split.count), sdpa_attention,
                                      self.tile_batch)
        mine = torch.stack(clips) if clips else self.vae.empty_clip(z)[None][:0]
        gathered = shard.gather(self.CLIPS, mine, dim=0)
        return self.vae.walk.assemble(list(gathered.unbind(0)), num_latents)


class H3AudioDecoderSubmodule(_LayoutBatched, BatchedRows, NodeSubmodule):
    """Audio rows ``[2 * A, 32]`` fp32 (channel-major) -> ``[2, A * 800]`` fp32 waveform at
    32 kHz: the two stereo channels are two batch items of the mono audio VAE."""

    disable_torch_compile = True
    output_keys = (AUDIO_OUTPUT,)

    def __init__(self, audio_vae: nn.Module, config: H3Config, max_batch_size: int = 4):
        super().__init__()
        self.audio_vae = audio_vae
        self.config = config
        self._max_batch_size = max_batch_size
        acfg = config.audio_vae
        self.register_buffer("latents_mean", torch.tensor(acfg.latents_mean).view(1, -1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(acfg.latents_std).view(1, -1, 1), persistent=False)

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={AUDIO_LATENTS: inputs[AUDIO_LATENTS][0]},
                          resource_step_info=layout_from_metadata(fwd_info.step_metadata))

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {AUDIO_LATENTS: torch.stack([inp.tensor_inputs[AUDIO_LATENTS] for inp in inputs])}

    def run_batch(self, audio_latents: torch.Tensor, **kwargs):
        rows = audio_latents.to(device=self.get_device(), dtype=torch.float32)
        batch, channels = rows.shape[0], rows.shape[-1]
        z = rows.reshape(batch * AUDIO_CHANNELS, -1, channels).permute(0, 2, 1)
        z = z * self.latents_std + self.latents_mean
        audio = self.audio_vae(z)
        return {AUDIO_OUTPUT: audio.reshape(batch, AUDIO_CHANNELS, -1)}


__all__ = [
    "H3TextEncoderSubmodule", "H3VaeEncoderSubmodule", "H3DenoiseSubmodule", "H3VideoDecoderSubmodule",
    "H3AudioDecoderSubmodule", "layout_from_metadata",
]
