"""MiniMax-H3-Image node submodules.

    text_encoder   -> H3TextEncoderSubmodule (H3's)  Qwen3-VL layer 50 (references through its
                                                     tower) -> condition_proj -> token refiner
    vae_encoder    -> H3ImageEncoderSubmodule        references -> conditioning latent rows
    dit            -> H3ImageDenoiseSubmodule        H3's packed DiT, one output head per step
    vae_decoder    -> H3ImageDecoderSubmodule        generated rows -> uint8 image

Numerics follow the checkpoint's dtypes (bf16, with the reference's fp32 modules and step
math), not the engine's autocast. ``test/minimax_h3_image/parity.py`` checks each stage
against diffusers' pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import Segment, SubmoduleStep
from mstar.model.components.batched_rows import BatchedRows
from mstar.model.components.diffusion.attention import sdpa_attention
from mstar.model.components.diffusion.compile_utils import compile_blocks as compile_blocks_
from mstar.model.components.diffusion.denoise_loop import StepScalar
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.minimax_h3.components.layout import patchify_video
from mstar.model.minimax_h3.components.transformer import H3Inputs, H3StepLayout, StepModulation
from mstar.model.minimax_h3.components.video_vae import stitch_tiles
from mstar.model.minimax_h3.config import PIXEL_MEAN, PIXEL_STD
from mstar.model.minimax_h3.submodules import (
    COND_LATENTS,
    FIRST_STEP,
    LATENTS,
    TEXT_EMBEDS,
    H3PackedDenoise,
)
from mstar.model.minimax_h3_image.config import REFERENCE_NOISE_AUG, H3ImageConfig
from mstar.model.minimax_h3_image.layout import H3ImageLayout, modality_tags, position_ids, timestep_classes
from mstar.model.submodule_base import NodeInputs, NodeSubmodule

# Edge names shared with the model file: the references' uint8 pixels (one tensor each, at
# its own size) and the generated image
REFERENCE_PIXELS, IMAGE_OUTPUT = "reference_pixels", "image_output"


def layout_from_metadata(step_metadata: dict) -> H3ImageLayout:
    """The request's packed layout, from the host values its step metadata carries."""
    grids = tuple(reference_grids(step_metadata))
    references = tuple((h // 2 * 2, w // 2 * 2) for h, w in grids)
    return H3ImageLayout(
        text_len=int(step_metadata["text_len"]),
        latent_height=int(step_metadata["latent_height"]),
        latent_width=int(step_metadata["latent_width"]),
        references=references,
        text_image_spans=tuple(tuple(span) for span in step_metadata.get("text_image_spans", ())),
        reference_grids=() if grids == references else grids,
    )


def reference_grids(step_metadata: dict) -> list[tuple[int, int]]:
    """Each reference's whole latent grid (its pixels over 16), before the crop to whole patches."""
    return [(int(h), int(w)) for h, w in step_metadata.get("reference_latent_grids", ())]


# ---------------------------------------------------------------------------
# vae_encoder
# ---------------------------------------------------------------------------

class H3ImageEncoderSubmodule(NodeSubmodule):
    """The references, uint8 ``[3, h, w]`` each at its own size (whole VAE cells) -> their
    normalized latents as rows ``[sum of rows, 96]`` fp32, in packed order: each encoded
    alone in tiles, its posterior sampled from the request's generator (the first draws the
    reference makes from it, in reference order), rounded to fp16 and cropped to whole
    patches (``MiniMaxH3ImageEditVaeEncoderStep``). Once per request."""

    disable_torch_compile = True
    output_keys = (COND_LATENTS,)

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={REFERENCE_PIXELS: list(inputs[REFERENCE_PIXELS])},
                          resource_step_info=fwd_info.random_seed)

    def can_batch(self, batch, model_inputs) -> bool:
        return False

    def max_batch_size(self, graph_walk: str):
        return 1

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {REFERENCE_PIXELS: inputs[0].tensor_inputs[REFERENCE_PIXELS], "seed": inputs[0].resource_step_info}

    def forward(self, graph_walk, engine_inputs, reference_pixels: list[torch.Tensor], seed: int,
                **kwargs) -> NameToTensorList:
        device = self.get_device()
        generator = torch.Generator().manual_seed(seed)
        rows = []
        for pixels in reference_pixels:
            latents = self.encoder.posterior_latents(self.encoder.frame_moments(pixels.to(device)), generator)
            latents = latents[..., : latents.shape[-2] // 2 * 2, : latents.shape[-1] // 2 * 2]
            rows.append(patchify_video(latents[0], (2, 2)))
        return {COND_LATENTS: [torch.cat(rows)]}


# ---------------------------------------------------------------------------
# dit (denoise loop body)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class H3ImageSchedule(FlowMatchSchedule):
    """The fixed grid, the references' timestep at every step, and the head of each step."""

    reference_timesteps: torch.Tensor | None = None
    heads: torch.Tensor | None = None


def h3_image_schedule(sigmas: tuple[float, ...]) -> H3ImageSchedule:
    """The released sampler's grid (``MiniMaxH3ImageSetTimestepsStep``): every sigma rounds
    through its fp32 timestep ``1 - sigma`` and back, which moves those below 0.5 by an ulp."""
    timesteps = torch.tensor([1.0 - sigma for sigma in sigmas[:-1]], dtype=torch.float32)
    steps = timesteps.shape[0]
    return H3ImageSchedule(
        sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]), timesteps=timesteps, mu=None,
        reference_timesteps=torch.full((steps,), REFERENCE_NOISE_AUG, dtype=torch.float32),
        heads=torch.arange(steps, dtype=torch.float32),
    )


def h3_image_step(sample: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor,
                  sigma_next: torch.Tensor) -> torch.Tensor:
    """``MiniMaxH3Scheduler.step`` with ``euler_update_form="velocity"``, fp32: the heads were
    distilled with this form, and the blend form differs from it in the last bit."""
    return sample + (sigma - sigma_next) * velocity.float()


class H3ImageDenoiseSubmodule(H3PackedDenoise):
    """One step of the fixed six-step grid over H3's packed DiT, with the step's head. The
    loop carries the image rows, reference rows first, fp32; only the generated rows are
    stepped. The references arrive as noise (seed_loop_back), are noised to t = 0.999 on the
    first step and carried unchanged after.

    The step's layout travels as inputs (``LAYOUT_FROM_INPUTS``), so the prompts of one row
    bucket share a captured graph (:meth:`graph_key`): ``capture_buckets`` lists text-to-image
    graph keys. An edit's first step differs from the rest on the host, so edits run eager.

    Every request walks the same grid, so each step's AdaLN rows (13B of the DiT's weights
    read for one or two rows) are computed once a process, for text-to-image and for edits,
    and a step reads its own by its head, which is its index on the grid."""

    LAYOUT_FROM_INPUTS = True

    STEP_SCALARS = {
        "sigma": StepScalar("sigmas"),
        "sigma_next": StepScalar("sigmas", offset=1, capture_fill=0.0),
        "timestep": StepScalar("timesteps", broadcast=False, capture_fill=0.5),
        "reference_timestep": StepScalar("reference_timesteps", broadcast=False, capture_fill=1.0),
        "head": StepScalar("heads", broadcast=False, capture_fill=0.0),
    }

    def __init__(self, transformer: nn.Module, config: H3ImageConfig, **kwargs):
        super().__init__(transformer, config, **kwargs)
        edits = [key for _, key in self.capture_buckets if key.references]
        if edits:
            raise ValueError(f"MiniMax-H3-Image captures text-to-image steps only; got edit buckets {edits}")
        # timesteps a step embeds (1: text-to-image; 2: an edit's, with the references') -> the grid's rows
        self._modulation: dict[int, StepModulation] = {}

    def step_modulation(self, timesteps: int, device) -> StepModulation:
        """The grid's AdaLN rows for steps that embed ``timesteps`` timesteps, built on first use,
        which for a captured step is :meth:`capture_request_inputs`, before the capture."""
        modulation = self._modulation.get(timesteps)
        if modulation is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                # a capture records the GEMMs rather than running them: the rows would hold nothing
                raise RuntimeError("MiniMax-H3-Image builds its AdaLN rows before a capture, not in it")
            schedule = h3_image_schedule(self.config.sigmas)
            grid = schedule.timesteps.view(-1, 1)
            if timesteps == 2:
                grid = torch.stack([schedule.timesteps, schedule.reference_timesteps], dim=-1)
            modulation = self._modulation[timesteps] = self.transformer.step_modulation(grid.to(device))
        return modulation

    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> H3ImageLayout:
        return layout_from_metadata(fwd_info.step_metadata)

    def graph_key(self, bucket_key: H3ImageLayout) -> H3ImageLayout:
        # the key at the longest prompt its row bucket holds; the prompt's vision spans are in its tags
        capacity = self.padded_rows(bucket_key) - bucket_key.total_rows + bucket_key.text_len
        return dataclasses.replace(bucket_key, text_len=capacity, text_image_spans=())

    def schedule_settings(self, fwd_info: CurrentForwardPassInfo) -> None:
        return None  # the checkpoint's fixed grid

    def schedule_for(self, walk: str, settings: None, bucket_key: H3ImageLayout) -> H3ImageSchedule:
        return h3_image_schedule(self.config.sigmas)

    def seed_loop_back(self, bucket_key: H3ImageLayout, generator: torch.Generator) -> dict[str, torch.Tensor]:
        # The reference's draw order from the one generator: the references' posterior
        # samples (drawn by the vae_encoder node from its own generator on the same seed, so
        # skipped here: their shapes are the uncropped grids), the generated image's noise,
        # then each reference's noise.
        channels = self.config.transformer.latents_dim
        for height, width in bucket_key.reference_grids or bucket_key.references:
            torch.randn((1, channels, height, width), generator=generator, dtype=torch.float32)
        target = torch.randn((1, channels, 1, bucket_key.latent_height, bucket_key.latent_width),
                             generator=generator, dtype=torch.float32)
        references = [
            patchify_video(torch.randn((1, channels, height, width), generator=generator,
                                       dtype=torch.float32)[:, :, None][0], bucket_key.patch)
            for height, width in bucket_key.references
        ]
        return {LATENTS: torch.cat([*references, patchify_video(target[0], bucket_key.patch)])}

    def request_inputs(self, fwd_info, inputs: NameToTensorList, bucket_key: H3ImageLayout) -> dict[str, torch.Tensor]:
        # host-side, so whether this is the first step is known without a sync
        text = inputs[TEXT_EMBEDS][0]
        tensors = {TEXT_EMBEDS: self.padded_prompt(text, bucket_key),
                   FIRST_STEP: torch.tensor(self.step_index(fwd_info) == 0),
                   **self.layout_inputs(bucket_key, text.device)}
        if bucket_key.num_condition_rows:
            tensors[COND_LATENTS] = inputs[COND_LATENTS][0]
        return tensors

    def capture_request_inputs(self, bucket_key: H3ImageLayout, device) -> dict[str, torch.Tensor]:
        cfg = self.config.transformer
        self.step_modulation(1, device)  # the rows a captured step reads, outside its capture
        return {
            LATENTS: torch.zeros(bucket_key.num_condition_rows + bucket_key.num_video_rows, cfg.patch_dim,
                                 device=device),
            TEXT_EMBEDS: torch.zeros(bucket_key.text_len, cfg.hidden_size, dtype=self.transformer.dtype,
                                     device=device),
            FIRST_STEP: torch.tensor(False),
            **self.layout_inputs(bucket_key, device),
            **({COND_LATENTS: torch.zeros(bucket_key.num_condition_rows, cfg.patch_dim, device=device)}
               if bucket_key.num_condition_rows else {}),
        }

    def build_layout(self, bucket_key: H3ImageLayout, device) -> H3StepLayout:
        return self.step_layout(bucket_key, position_ids(bucket_key), timestep_classes(bucket_key),
                                modality_tags(bucket_key), device)

    def denoise(self, engine_inputs, bucket_key: H3ImageLayout, latents, timestep, sigma, sigma_next,
                reference_timestep, head, text_embeds, first_step, cond_latents=None, **cond):
        layout = self.staged_layout(bucket_key, cond, latents.device)
        n = layout.num_condition_rows
        if n and bool(first_step.all()):  # a CPU tensor: no sync
            t = torch.full((), REFERENCE_NOISE_AUG, dtype=torch.float32, device=latents.device)
            latents = torch.cat([t * cond_latents + (1.0 - t) * latents[:, :n], latents[:, n:]], dim=1)
        # the reference embeds a step's distinct timesteps: the step's alone for a
        # text-to-image request, then the references' 0.999 for an edit
        timesteps = torch.stack([timestep, reference_timestep], dim=-1) if n else timestep.view(-1, 1)
        inputs = H3Inputs(video=latents, audio=None, text=text_embeds, timesteps=timesteps.float())
        head = head.long()
        velocity, _ = self.transformer(inputs, layout, shard=self.shard, head=head,
                                       modulation=self.step_modulation(timesteps.shape[-1], latents.device), step=head)
        generated = h3_image_step(latents[:, n:], velocity[:, n:], sigma, sigma_next)
        return torch.cat([latents[:, :n], generated], dim=1) if n else generated


# ---------------------------------------------------------------------------
# vae_decoder
# ---------------------------------------------------------------------------

class H3ImageDecoderSubmodule(BatchedRows, NodeSubmodule):
    """Image rows ``[N, 96]`` fp32 -> uint8 ``[3, H, W]`` on the canvas: reference rows dropped,
    unpatchified, denormalized with the VAE's latent statistics, decoded in tiles, the
    ImageNet normalization undone, clamped, and quantized to 8 bits by truncation, as the
    released pipeline does (the resize to the requested size is the model's postprocess).

    On SP ranks its sequence shard (``vae_shard``, ``exchanges=False``: nothing mixes tiles
    but the stitch after the gather) splits each request's tiles: each rank decodes its
    contiguous share and the shares are gathered on every rank, which then stitch the same
    image: the DiT's ranks decode in lockstep, as MiniMax-H3's video decoder does."""

    # the label a step declares a request's tiles under
    TILES = "tiles"

    disable_torch_compile = True
    output_keys = (IMAGE_OUTPUT,)

    def __init__(self, vae: nn.Module, config: H3ImageConfig, *, max_batch_size: int = 1, tile_batch: int = 1,
                 compile_blocks: bool = False):
        super().__init__()
        self.vae = vae
        self._max_batch_size = max_batch_size
        # spatial tiles decoded per decoder call: 1 is the reference's order of work
        self.tile_batch = int(tile_batch)
        if compile_blocks and vae is not None:
            compile_blocks_(vae.decoder.transformer_blocks)
        cfg = config.vae
        self.register_buffer("latents_mean", torch.tensor(cfg.latents_mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(cfg.latents_std).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD).view(1, -1, 1, 1), persistent=False)
        self.channels = cfg.latent_channels

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1 and len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={LATENTS: inputs[LATENTS][0]},
                          resource_step_info=layout_from_metadata(fwd_info.step_metadata))

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        return {LATENTS: torch.stack([inp.tensor_inputs[LATENTS] for inp in inputs]),
                "layout": inputs[0].resource_step_info}

    def declare_step(self, graph_walk, request_ids, inputs: list[NodeInputs], **kwargs) -> SubmoduleStep | None:
        if self.sequence_shard is None:
            return None
        # a batch shares one layout (`can_batch`), so every request's tiles are one count
        segments = [
            Segment(rid, self.TILES, self.num_tiles(inp.resource_step_info))
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        return SubmoduleStep(steps={}, segments=segments)

    def num_tiles(self, layout: H3ImageLayout) -> int:
        grid, _, _ = self.vae.tile_slices(layout.latent_height, layout.latent_width)
        return sum(len(row) for row in grid)

    def run_batch(self, latents: torch.Tensor, layout: H3ImageLayout, **kwargs):
        rows = latents.to(device=self.get_device(), dtype=torch.float32)[:, layout.num_condition_rows:]
        ph, pw = layout.patch
        b, h, w = rows.shape[0], layout.latent_height, layout.latent_width
        z = rows.reshape(b, h // ph, w // pw, self.channels, ph, pw).permute(0, 3, 1, 4, 2, 5)
        z = z.reshape(b, self.channels, h, w) * self.latents_std + self.latents_mean
        # SDPA within each fixed-size tile, as the reference: dense, one segment
        shard = self.sequence_shard
        split = None if shard is None else shard.split_range(self.TILES)
        if split is None:
            image = self.vae.decode(z, sdpa_attention, tile_batch=self.tile_batch)
        else:
            image = self._decode_shared(z, shard, split)
        image = (image.float() * self.pixel_std + self.pixel_mean).clamp(0, 1)
        return {IMAGE_OUTPUT: (image * 255).clamp(0, 255).to(torch.uint8)}

    def _decode_shared(self, z: torch.Tensor, shard, split) -> torch.Tensor:
        """This rank's range of the tiles (``split``), gathered in tile order on every rank and
        stitched there. A rank with no tiles (fewer tiles than ranks) decodes nothing."""
        vae = self.vae
        z = z.to(vae.post_quant_conv.weight.dtype)
        grid, y_overlaps, x_overlaps = vae.tile_slices(z.shape[-2], z.shape[-1])
        tiles = [z[..., ys, xs] for row in grid for ys, xs in row]
        if sum(split.counts) != len(tiles):
            # the declared count is the same on every rank, so every rank refuses alike
            raise ValueError(f"the step declared {sum(split.counts)} tiles, the decode plans {len(tiles)}")
        decoded = vae.decode_tiles(tiles[split.start:split.start + split.count], sdpa_attention,
                                   max(1, self.tile_batch))
        if decoded:
            mine = torch.stack(decoded)
        else:
            first = tiles[0]
            mine = first.new_zeros(0, first.shape[0], vae.decoder.out_channels, first.shape[-2] * vae.ratio,
                                   first.shape[-1] * vae.ratio, dtype=vae.decoder.proj_out.weight.dtype)
        gathered = iter(shard.gather(self.TILES, mine, dim=0).unbind(0))
        return stitch_tiles([[next(gathered) for _ in row] for row in grid], y_overlaps, x_overlaps)


__all__ = [
    "H3ImageEncoderSubmodule", "H3ImageDenoiseSubmodule", "H3ImageDecoderSubmodule", "h3_image_schedule",
    "h3_image_step", "layout_from_metadata",
]
