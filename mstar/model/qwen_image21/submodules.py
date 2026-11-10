"""Qwen-Image-2.1's nodes.

    text_encoder  Qwen3-VL-8B to its last layer, before the final norm, with the images through its
                  vision tower (one tower pass feeds both prompts under CFG); the system turn dropped
    vae_encoder   each reference image's normalized latents, all references end to end
    dit           one committing step per prompt span (``prefix_text`` causal, ``prefix_image``
                  bidirectional) over the paged KV, then the denoise loop: the target over that cache
    vae_decoder   the latents to RGBA pixels
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, Segment, SubmoduleStep
from mstar.engine.resources.convenience import AttentionCallable
from mstar.model.components.diffusion.decode_utils import latent_affine, latent_std
from mstar.model.components.diffusion.denoise_loop import LATENTS, DenoiseLoopSubmodule, StepKey
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.components.diffusion.guidance import Guidance, cfg
from mstar.model.components.diffusion.image_io import pixels_to_uint8
from mstar.model.components.diffusion.qwen.vl_prompt import qwen3_vl_tokens
from mstar.model.components.qwen_vision import VISION_ATTN, FrameSDPA, QwenVisionModel
from mstar.model.qwen_image21.components.transformer import (
    QwenImage21Transformer,
    image_positions,
    text_positions,
)
from mstar.model.qwen_image21.config import (
    ATTN,
    KV,
    MAIN,
    PREFIX_IMAGE_WALK,
    PREFIX_TEXT_WALK,
    UNCOND,
    QwenImage21Config,
)
from mstar.model.qwen_image21.layout import RequestLayout
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking

TEXT_INPUTS, VISION_PIXELS, REF_PIXELS = "text_inputs", "vision_pixels", "ref_pixels"
TEXT_EMBEDS, REF_LATENTS, IMAGE_OUTPUT = "text_embeds", "ref_latents", "image_output"
# step_metadata keys
LAYOUT, SPAN, PROMPT_LENGTHS, DROP, GRIDS = "layout", "span", "prompt_lengths", "drop", "vision_grids"
STEPS = "num_inference_steps"
# the guidance branches' KV labels, in branch order, and the plan label a guided step attends both under
BRANCH_LABELS = (MAIN, UNCOND)
CFG_LABEL = "_cfg"
# each branch's target frame position (its label's prefix length), a per-branch input
FRAME, MAIN_FRAME, UNCOND_FRAME = "frame", "main_frame", "uncond_frame"
PREFIX_WALKS = (PREFIX_TEXT_WALK, PREFIX_IMAGE_WALK)
POSITIONS = "positions"


# ---------------------------------------------------------------------------
# text_encoder
# ---------------------------------------------------------------------------

class QwenImage21TextEncoderSubmodule(NodeSubmodule):
    """The templated prompt (and the negative one under CFG), token ids end to end, -> their last
    layer's hidden states past the system turn, end to end ``[L, hidden]``."""

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, encoder: nn.Module, vision: QwenVisionModel):
        super().__init__()
        self.encoder = encoder
        self.vision = vision
        # per-frame SDPA, not a node resource: the tower runs once per request
        self._frames = FrameSDPA()
        for block in vision.blocks:
            block.attn.bind_resources({VISION_ATTN: self._frames})

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        tensors = {TEXT_INPUTS: inputs[TEXT_INPUTS][0]}
        if inputs.get(VISION_PIXELS):
            tensors[VISION_PIXELS] = inputs[VISION_PIXELS][0]
        meta = fwd_info.step_metadata
        return NodeInputs(tensor_inputs=tensors, kwargs={
            PROMPT_LENGTHS: tuple(meta[PROMPT_LENGTHS]), DROP: int(meta[DROP]),
            GRIDS: tuple(tuple(int(v) for v in grid) for grid in meta.get(GRIDS, ())),
        })

    def can_batch(self, batch, model_inputs) -> bool:
        return False

    def max_batch_size(self, graph_walk: str):
        return 1

    @torch.no_grad()
    def forward(self, graph_walk, engine_inputs, text_inputs: torch.Tensor, prompt_lengths: tuple[int, ...],
                drop: int, vision_grids: tuple = (), vision_pixels: torch.Tensor | None = None,
                **kwargs) -> NameToTensorList:
        device = self.get_device()
        ids = text_inputs.to(device=device, dtype=torch.long)
        images = None if vision_pixels is None else qwen3_vl_tokens(
            self.vision, self._frames, vision_pixels.to(device), vision_grids,
        )
        out = [
            self.encoder(prompt[None], torch.ones_like(prompt[None]), padded=False, images=images)[0, drop:]
            for prompt in ids.split(list(prompt_lengths))
        ]
        return {TEXT_EMBEDS: [torch.cat(out)]}


# ---------------------------------------------------------------------------
# vae_encoder / vae_decoder
# ---------------------------------------------------------------------------

def vae_pixels(image: torch.Tensor) -> torch.Tensor:
    """uint8 ``[C, H, W]`` -> ``[1, C, 1, H, W]`` in ``[-1, 1]`` fp32, as ``VaeImageProcessor.preprocess``
    hands it to the VAE: the division on the CPU (CUDA divides by a scalar through its reciprocal,
    which rounds differently) and the memory NHWC-strided (cuDNN picks its kernels by layout). Both
    change the latents' bits."""
    pixels = image.permute(1, 2, 0).contiguous().float() / 255.0
    return (2.0 * pixels - 1.0)[None].permute(0, 3, 1, 2).unsqueeze(2)


class QwenImage21VaeEncoderSubmodule(NodeSubmodule):
    """RGBA uint8 ``[4, H, W]`` references -> their normalized latents (the posterior's mode) as
    rows ``[h * w, C]``, every reference end to end. Images differ in size: one encode each."""

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={f"image_{i}": img for i, img in enumerate(inputs[REF_PIXELS])})

    @torch.no_grad()
    def forward(self, graph_walk, engine_inputs, **images: torch.Tensor) -> NameToTensorList:
        device, dtype, rows = self.get_device(), self.vae.dtype, []
        mean, std = latent_affine(self.vae, dtype, device)[0], latent_std(self.vae, dtype, device)
        for key in sorted(images, key=lambda name: int(name.split("_")[1])):
            pixels = vae_pixels(images[key]).to(device=device, dtype=dtype)
            latents = (self.vae.encode(pixels).latent_dist.mode() - mean) / std
            rows.append(latents.flatten(2).transpose(1, 2)[0])
        return {REF_LATENTS: [torch.cat(rows)]}


class QwenImage21VaeDecoderSubmodule(NodeSubmodule):
    """Latents ``[h * w, C]`` -> uint8 RGBA ``[4, H, W]``, as the pipeline unpacks, denormalizes,
    decodes and post-processes them."""

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        target = RequestLayout.from_metadata(fwd_info.step_metadata[LAYOUT]).target
        return NodeInputs(tensor_inputs={LATENTS: inputs[LATENTS][0]}, kwargs={"target": target})

    def can_batch(self, batch, model_inputs) -> bool:
        return False

    def max_batch_size(self, graph_walk: str):
        return 1

    @torch.no_grad()
    def forward(self, graph_walk, engine_inputs, latents: torch.Tensor, target: tuple[int, int],
                **kwargs) -> NameToTensorList:
        h, w = target
        device, dtype = self.get_device(), self.vae.dtype
        z = latents.to(device=device, dtype=dtype).transpose(0, 1).reshape(1, -1, 1, h, w)
        z = z * latent_std(self.vae, dtype, device) + latent_affine(self.vae, dtype, device)[0]
        image = self.vae.decode(z, return_dict=False)[0][:, :, 0]
        return {IMAGE_OUTPUT: [pixels_to_uint8(image)[0]]}


# ---------------------------------------------------------------------------
# dit
# ---------------------------------------------------------------------------

class DitKey(NamedTuple):
    """What has to match for two denoise rows to share a forward and a captured graph: the target's
    latent grid. Whether the request runs its guidance branches is the step key's."""

    height: int
    width: int


class _PrefixRow(NamedTuple):
    main: int      # this row's main tokens (the step's segment)
    uncond: int    # its uncond tail's, under CFG on the prompt's last span, else 0
    fork: bool     # the row forks uncond off main after this step


class PrefixPass:
    """The dit's prefix walks: one committing step per prompt span, a text span causal and an image
    span bidirectional (the walk decides), each over the spans before it. Under CFG the last image's
    step forks ``uncond`` off ``main``, and the tail step writes both labels' tails."""

    def __init__(self, sub: "QwenImage21DenoiseSubmodule"):
        self.sub = sub

    @staticmethod
    def _row(info: CurrentForwardPassInfo, input_seq_len: int) -> _PrefixRow:
        meta = info.step_metadata
        layout = RequestLayout.from_metadata(meta[LAYOUT])
        k = int(meta[SPAN])
        last = k == len(layout.spans) - 1
        uncond = layout.uncond_tail[1] if last and layout.uncond_tail is not None else 0
        fork = layout.uncond_tail is not None and layout.images > 0 and k == len(layout.spans) - 2
        return _PrefixRow(input_seq_len - uncond, uncond, fork)

    def prepare_inputs(self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList):
        meta = fwd_info.step_metadata
        layout = RequestLayout.from_metadata(meta[LAYOUT])
        k = int(meta[SPAN])
        span = layout.spans[k]
        if span.is_text != (graph_walk == PREFIX_TEXT_WALK):
            raise RuntimeError(f"{graph_walk} step {k} names a {'text' if span.is_text else 'image'} span")
        if span.is_text:
            embeds = inputs[TEXT_EMBEDS][0]
            rows = [embeds[span.rows:span.rows + span.length]]
            positions = [text_positions(span.position, span.length)]
            if k == len(layout.spans) - 1 and layout.uncond_tail is not None:
                start, length = layout.uncond_tail
                rows.append(embeds[start:start + length])
                positions.append(text_positions(span.position, length))
        else:
            rows = [inputs[REF_LATENTS][0][span.rows:span.rows + span.length]]
            positions = [image_positions(span.position, span.height, span.width)]
        rows = torch.cat(rows) if len(rows) > 1 else rows[0]
        positions = to_device_nonblocking(torch.cat(positions), self.sub.get_device())
        return ARNodeInputs(input_embeds=rows, input_seq_len=rows.shape[0], tensor_inputs={POSITIONS: positions})

    @staticmethod
    def split_inputs(inputs: ARNodeInputs, start: int, end: int) -> ARNodeInputs:
        # a cached head of the span: the rest runs at its own rope positions
        return ARNodeInputs(
            input_embeds=inputs.input_embeds[start:end], input_seq_len=end - start,
            tensor_inputs={POSITIONS: inputs.tensor_inputs[POSITIONS][start:end]},
        )

    def preprocess(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]) -> dict:
        info = engine_inputs.per_request_info
        rows = [
            self._row(info[rid], inp.input_seq_len)
            for rid, inp in zip(engine_inputs.request_ids, inputs, strict=True)
        ]
        parts = {MAIN: ([], []), UNCOND: ([], [])}
        for row, inp in zip(rows, inputs, strict=True):
            embeds, positions = inp.input_embeds, inp.tensor_inputs[POSITIONS]
            parts[MAIN][0].append(embeds[:row.main])
            parts[MAIN][1].append(positions[:row.main])
            if row.uncond:
                parts[UNCOND][0].append(embeds[row.main:])
                parts[UNCOND][1].append(positions[row.main:])
        return {label: (torch.cat(e), torch.cat(p)) for label, (e, p) in parts.items() if e}

    def forward(self, graph_walk: str, **labels) -> None:
        dit = self.sub.transformer
        embed = dit.txt_in if graph_walk == PREFIX_TEXT_WALK else dit.img_in
        attend = self.sub.attend()
        for label in (MAIN, UNCOND):
            if label in labels:
                embeds, positions = labels[label]
                attend.bind_step(label)
                dit.prefix(embed(embeds.to(dit.dtype)), positions, attend)

    def declare_step(self, graph_walk: str, request_ids: list, inputs: list[ARNodeInputs], per_request_info):
        rows = [
            self._row(per_request_info[rid], inp.input_seq_len) for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        segments = tuple(
            [Segment(rid, MAIN, row.main) for rid, row in zip(request_ids, rows, strict=True)]
            + [Segment(rid, UNCOND, row.uncond) for rid, row in zip(request_ids, rows, strict=True) if row.uncond]
        )
        forks = frozenset(rid for rid, row in zip(request_ids, rows, strict=True) if row.fork)
        return SubmoduleStep(steps={
            KV: KVStep(
                segments=segments, commit=True,
                post_forks=((MAIN, UNCOND),) if forks else (), fork_rids=forks or None,
            ),
            ATTN: AttentionStep(segments=segments, causal=graph_walk == PREFIX_TEXT_WALK),
        })


class QwenImage21DenoiseSubmodule(DenoiseLoopSubmodule):
    """The scaffold's denoise loop over a paged KV cache, plus the prefix walks (``PrefixPass``).

    The guidance branches are the KV labels ``main`` and ``uncond``, in that order: a guided row of
    branch ``b`` attends ``BRANCH_LABELS[b]``'s prefix and takes the target at that label's frame
    position, and the scaffold combines the branches' velocities with plain CFG."""

    disable_autocast = True
    PREFETCH_SEEDS = True
    # the prefix walks carry each token's rope position, cut with the tokens
    cuts_tensor_inputs = True
    GUIDANCE = Guidance(branches=({FRAME: MAIN_FRAME}, {FRAME: UNCOND_FRAME}), combine=cfg)

    def __init__(self, transformer: QwenImage21Transformer, config: QwenImage21Config, *, loop_name: str, **kwargs):
        super().__init__(loop_name=loop_name, **kwargs)
        self.transformer = transformer
        self.qi_config = config
        self.prefix_pass = PrefixPass(self)
        self._attend: AttentionCallable | None = None

    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._attend = None

    def attend(self):
        if self._attend is None:
            self._attend = AttentionCallable(kv=self.node_resources[KV], attn=self.node_resources[ATTN])
        return self._attend

    # ------------------------------------------------------------ prefix walks
    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        if graph_walk in PREFIX_WALKS:
            return self.prefix_pass.prepare_inputs(graph_walk, fwd_info, inputs)
        return super().prepare_inputs(graph_walk, fwd_info, inputs, **kwargs)

    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        if graph_walk not in PREFIX_WALKS:
            return super().split_inputs(graph_walk, fwd_info, inputs, start, end)
        return self.prefix_pass.split_inputs(inputs, start, end)

    def preprocess(self, graph_walk, engine_inputs, inputs):
        if graph_walk in PREFIX_WALKS:
            return self.prefix_pass.preprocess(graph_walk, engine_inputs, inputs)
        return super().preprocess(graph_walk, engine_inputs, inputs)

    def forward(self, graph_walk, engine_inputs, **kwargs):
        if graph_walk in PREFIX_WALKS:
            self.prefix_pass.forward(graph_walk, **kwargs)
            return {}
        return super().forward(graph_walk, engine_inputs, **kwargs)

    def forward_batched(self, graph_walk, engine_inputs, **kwargs):
        if graph_walk in PREFIX_WALKS:
            self.prefix_pass.forward(graph_walk, **kwargs)
            return {rid: {} for rid in engine_inputs.request_ids}
        return super().forward_batched(graph_walk, engine_inputs, **kwargs)

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, piecewise_leases=None,
                     per_request_info=None, **kwargs):
        if graph_walk in PREFIX_WALKS:
            return self.prefix_pass.declare_step(graph_walk, request_ids, inputs, per_request_info)
        keys = [inp.resource_step_info for inp in inputs]
        # rows batch at one step key, so every row runs the same branches
        labels = tuple(BRANCH_LABELS[b] for b in keys[0].branches)
        segments = tuple(
            Segment(rid, label, self.num_tokens(key))
            for label in labels for rid, key in zip(request_ids, keys, strict=True)
        )
        return SubmoduleStep(cg_key_info=self._uniform_key(keys), steps={
            KV: KVStep(segments=segments, commit=False,
                       combined_labels={labels: CFG_LABEL} if len(labels) > 1 else {}),
            ATTN: AttentionStep(segments=segments, causal=False),
        })

    def cg_key_info(self, graph_walk, per_request_info, per_request_input_metadata=None, **kwargs):
        if graph_walk in PREFIX_WALKS:
            return None
        return super().cg_key_info(graph_walk, per_request_info, per_request_input_metadata, **kwargs)

    # ------------------------------------------------------------ denoise hooks
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> DitKey:
        return DitKey(*RequestLayout.from_metadata(fwd_info.step_metadata[LAYOUT]).target)

    def schedule_for(self, fwd_info: CurrentForwardPassInfo, bucket_key: DitKey) -> FlowMatchSchedule:
        cfg = self.qi_config
        if cfg.sample_sigmas is not None:
            return FlowMatchSchedule.from_sigmas(cfg.sample_sigmas)
        steps = int(fwd_info.step_metadata[STEPS])
        return FlowMatchSchedule.build(cfg.scheduler, steps, bucket_key.height * bucket_key.width)

    def seed_latents(self, fwd_info, bucket_key: DitKey, generator: torch.Generator) -> torch.Tensor:
        """``randn_tensor`` of the packed target, in the transformer's dtype as the pipeline draws it."""
        h, w = bucket_key.height, bucket_key.width
        channels = self.qi_config.transformer.in_channels
        noise = torch.randn((1, 1, channels, h, w), generator=generator, dtype=self.transformer.dtype)
        return noise.view(channels, h * w).transpose(0, 1)

    def num_tokens(self, key: StepKey) -> int:
        """The target, which each branch writes on its own label."""
        return key.bucket.height * key.bucket.width

    def request_inputs(self, fwd_info, inputs, key: StepKey) -> dict[str, torch.Tensor]:
        """Each branch's target frame position (``main``'s, and ``uncond``'s when guided): staged once."""
        state = self.request_state(fwd_info.rid_handle)
        staged = state.get("frames")
        if staged is None:
            frames = RequestLayout.from_metadata(fwd_info.step_metadata[LAYOUT]).frames()
            device = self.get_device()
            staged = to_device_nonblocking(torch.tensor(frames, dtype=torch.long), device)
            state.add("frames", staged)
        names = (MAIN_FRAME, UNCOND_FRAME)
        return {names[b]: staged[b:b + 1] for b in key.branches}

    def capture_request_inputs(self, key: StepKey, device) -> dict[str, torch.Tensor]:
        out = {
            LATENTS: torch.zeros(self.num_tokens(key), self.qi_config.transformer.in_channels,
                                 dtype=self.transformer.dtype, device=device),
        }
        names = (MAIN_FRAME, UNCOND_FRAME)
        out.update({names[b]: torch.zeros(1, dtype=torch.long, device=device) for b in key.branches})
        return out

    def build_layout(self, key: StepKey, device) -> torch.Tensor:
        """The target's height/width positions ``[h * w, 2]``; the frame axis is per row."""
        return image_positions(0, key.bucket.height, key.bucket.width)[:, 1:].to(device)

    def model_output(self, engine_inputs, key: StepKey, latents, timestep, frame: torch.Tensor) -> torch.Tensor:
        """The velocity of each row, request-major as the scaffold stacks them (``frame [R, 1]`` the row's
        branch's). A guided step's KV plan packs its labels label-major (``main``'s rows, then ``uncond``'s),
        so the rows run in that order."""
        local = len(key.branches)

        def label_major(x: torch.Tensor) -> torch.Tensor:
            return x if local == 1 else x.unflatten(0, (-1, local)).transpose(0, 1).flatten(0, 1)

        rows, tokens = latents.shape[:2]
        hw = self.layout(key, latents.device)
        positions = torch.cat(
            [label_major(frame)[:, None, :].expand(-1, tokens, 1), hw[None].expand(rows, -1, -1)], dim=-1,
        )
        attend = self.attend()
        attend.bind_step(CFG_LABEL if local > 1 else BRANCH_LABELS[key.branches[0]])
        velocity = self.transformer.target(label_major(latents), label_major(timestep), positions, attend)
        return velocity if local == 1 else velocity.unflatten(0, (local, -1)).transpose(0, 1).flatten(0, 1)
