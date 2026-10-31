# ---------------------------------------------------------------------------
# NodeSubmodule wrappers
# ---------------------------------------------------------------------------


import functools
import logging
import os
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.bagel.components.language_model import BagelForCausalLM
from mstar.model.bagel.components.modeling_utils import (
    ImageTransform,
    PositionEmbedding,
    TimestepEmbedder,
    get_flattened_position_ids_extrapolate,
    patchify,
    vllm_vae_resize,
    vllm_vae_size,
    vllm_vit_resize,
)
from mstar.model.bagel.components.vit_encoder import VIT_ATTN
from mstar.model.bagel.config import MIXED_TEXT, BagelModelConfig
from mstar.model.higgs_audio.config import SAMPLER
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    StackingMethod,
)
from mstar.utils.h2d import to_device_nonblocking

logger = logging.getLogger(__name__)

# Private key the LLM's inner forwards hand their logits back under; `forward`
# and `forward_batched` sample it and never emit it. Only they know whether a
# sampled token is one flat entry or one per request.
_LOGITS = "_logits"


class DenoiseWalk(NamedTuple):
    """The shape of a denoise walk: a ``Loop`` named ``denoise_loop(walk)``."""
    # the CFG branches every step of the walk runs, in ``NODE_TO_CFG_LABEL``
    # order: main, then cfg_text and cfg_img (`cfg_branches`)
    branches: int
    # each branch on its own node, combined in combine_cfg (CFG-parallel
    # deployments); else the one LLM node runs them as one combined forward
    split: bool
    # ends in the VAE decode, or persists latents and time_index for the next walk
    decode: bool


IMAGE_GEN = "image_gen"
# A guided request's phases (`image_gen_phases`): the steps ahead of the CFG
# window, the window, and the VAE decode alone when the window runs to the
# last step.
IMAGE_GEN_HEAD = "image_gen_head"
IMAGE_DECODE = "image_decode"
DENOISE_WALKS = {
    IMAGE_GEN: DenoiseWalk(branches=1, split=False, decode=True),
    IMAGE_GEN_HEAD: DenoiseWalk(branches=1, split=False, decode=False),
    "image_gen_guided2": DenoiseWalk(branches=2, split=False, decode=False),
    "image_gen_guided3": DenoiseWalk(branches=3, split=False, decode=False),
    "image_gen_cfg2": DenoiseWalk(branches=2, split=True, decode=False),
    "image_gen_cfg3": DenoiseWalk(branches=3, split=True, decode=False),
}
# the CFG-parallel walks, one node per branch
CFG_WALKS = {walk: d.branches for walk, d in DENOISE_WALKS.items() if d.split}
# a guided window's walk, by its branch count and whether the deployment splits the branches
WINDOW_WALKS = {(d.branches, d.split): walk for walk, d in DENOISE_WALKS.items() if d.branches > 1}
SINGLE_NODE_DENOISE_WALKS = frozenset(walk for walk, d in DENOISE_WALKS.items() if not d.split)


def denoise_loop(walk: str) -> str:
    """A denoise walk's Loop. Named, and per walk, so a request's loop cap
    (`get_loop_caps`) names one phase."""
    return f"{walk}_loop"


# The LLM nodes' walks, by whether the deployment splits the CFG branches
# (`image_gen_phases` runs a guided window on the branch nodes there, else on
# LLM alone, never both), and the CFG branch each node owns. Module level
# because both the submodule (per step) and the model (per request, in
# get_request_resource_configs) have to answer the same question.
LLM_GRAPH_WALKS = {
    cfg_parallel: ("prefill_text", "prefill_vit", "prefill_vae", "decode", *(
        walk for walk, d in DENOISE_WALKS.items() if d.branches == 1 or d.split == cfg_parallel
    ))
    for cfg_parallel in (False, True)
}
NODE_TO_CFG_LABEL = {
    "LLM": "main",
    "LLM_cfg_text": "cfg_text",
    "LLM_cfg_img": "cfg_img",
}

# The ViT encoder's block loop, as a piecewise capture region
VIT_BLOCK_LOOP = "vit_block_loop"


def requires_cfg_for_inputs(inputs: list) -> bool:
    """Whether this batch runs the guidance branches, read off the rows.

    `prepare_inputs` stamps it per request and a capture's template rows carry
    it too, so this answers for a replay and for the capture that recorded it —
    which the per-request info cannot (capture passes `dummy_metadata`).
    """
    return any(bool(inp.resource_step_info) for inp in inputs)


def active_labels(graph_walk: str, cfg: bool, node_name: str) -> list[str]:
    """Cache labels a node touches on this walk.

    Guidance adds a branch label on the walks that carry one; with guidance
    off every walk is just "main". On a CFG walk each LLM node owns exactly
    one branch.
    """
    if graph_walk in {"prefill_text", "decode", MIXED_TEXT}:
        if cfg:
            return ["main", "cfg_img"]
    elif graph_walk in SINGLE_NODE_DENOISE_WALKS:
        if cfg:
            return ["main", "cfg_text", "cfg_img"]
    elif graph_walk in CFG_WALKS:
        return [NODE_TO_CFG_LABEL.get(node_name, "main")]
    return ["main"]


# Shared at module level so `vit_patches` counts with the same resizes `prepare_inputs` runs.
_VAE_TRANSFORM = ImageTransform(1024, 512, 16)
_VIT_TRANSFORM = ImageTransform(980, 224, 14)
# An image block is its patches between two sentinels, and occupies one position
# regardless of token count. The walks and the prefix layout both read these.
IMAGE_SENTINELS = 2
IMAGE_ADVANCE = 1


def vit_preprocess(image_preprocess: str | None) -> str:
    """The name of the resize the ViT applies for a request's ``image_preprocess``.
    Any name but "vllm" means the default."""
    return "vllm" if image_preprocess == "vllm" else "default"


def vit_patches(
    height: int, width: int, image_preprocess: str,
    patch_size: int, max_num_patches_per_side: int,
) -> int:
    """How many patches the ViT cuts an image of this size into. Its ``prepare_inputs`` asserts the same count."""
    if image_preprocess == "vllm":
        return max_num_patches_per_side ** 2
    height, width = _VAE_TRANSFORM.resize_transform.target_size(height, width)
    height, width = _VIT_TRANSFORM.resize_transform.target_size(height, width)
    return (height // patch_size) * (width // patch_size)


# Output sizes whose image_gen shape inputs an LLM node keeps on the device: above
# the shipped configs' concurrency (16), so concurrent sizes don't evict each other.
# An entry is the size's position ids and indexes, tens of KB at 1024x1024.
_IMAGE_GEN_SHAPES_CACHED = 64


def apply_timestep_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """BAGEL's non-linear timestep remapping: uniform t in [0, 1] to a shifted t
    that spends more time at higher noise levels (shift > 1); shift=1 is identity."""
    return shift * t / (1 + (shift - 1) * t)


def _f32(x: float) -> float:
    """``x`` rounded to float32, as a device compare against a Python scalar sees it."""
    return float(torch.tensor(float(x), dtype=torch.float32))


@functools.lru_cache(maxsize=4096)
def image_gen_timestep(time_index: int, num_timesteps: int, shift: float) -> float:
    """Host copy of the shifted timestep ``_forward_image_gen`` computes on the
    device for iteration ``time_index``, with the same float32 ops in the same
    order, so the host's CFG-interval test sees the device's timestep exactly.
    Memoized: a request's schedule is a handful of distinct values."""
    t_uniform = 1.0 - torch.tensor([float(time_index)], dtype=torch.float32) / (num_timesteps - 1)
    return float(apply_timestep_shift(t=t_uniform, shift=shift)[0])


@functools.lru_cache(maxsize=4096)
def cfg_branches(
    t_value: float, cfg_interval: tuple[float, float], text_scale: float, img_scale: float,
) -> int:
    """How many CFG branches an image_gen step has to run at timestep ``t_value``.

    1 (main only): guidance off, or ``t_value`` outside ``(lo, hi]``. The CFG
       combine at scales of exactly 1.0 is ``v_main`` up to float rounding
       (``a + 1 * (b - a)``, and a renorm ratio ``|v_main| / (|v_main| + 1e-8)``
       clamped to <= 1), so the step returns ``v_main``.
    2 (main, cfg_text): in the interval with ``img_scale <= 1``, where the
       combine never reads ``v_cfg_img``.
    3 (main, cfg_text, cfg_img): in the interval with image guidance on.

    The bounds are compared at float32, the dtype of the device timestep.
    ``cfg_interval`` must be hashable (a tuple); callers convert a list.
    """
    if not (text_scale > 1.0 or img_scale > 1.0):
        return 1
    lo, hi = cfg_interval
    if not (_f32(lo) < t_value <= _f32(hi)):
        return 1
    return 3 if img_scale > 1.0 else 2


class CFGWindow(NamedTuple):
    """The denoise steps ``[start, stop)`` that run guidance, and how many
    branches each of them runs (`cfg_branches`). Every other step runs main
    alone. One window, since t decreases with the step and the interval is one
    range of t; ``start == stop`` when no step is guided."""
    start: int
    stop: int
    branches: int


@functools.lru_cache(maxsize=256)
def cfg_window(
    num_timesteps: int, shift: float,
    cfg_interval: tuple[float, float], text_scale: float, img_scale: float,
) -> CFGWindow:
    """A request's `CFGWindow`, from its guidance kwargs, once at intake.
    ``cfg_interval`` must be hashable (a tuple)."""
    counts = [
        cfg_branches(image_gen_timestep(k, num_timesteps, shift), cfg_interval, text_scale, img_scale)
        for k in range(num_timesteps - 1)
    ]
    guided = [k for k, n in enumerate(counts) if n > 1]
    if not guided:
        return CFGWindow(0, 0, 1)
    window = CFGWindow(guided[0], guided[-1] + 1, counts[guided[0]])
    assert len(guided) == window.stop - window.start, f"guided steps {guided} are not one window"
    return window


def guided(window: CFGWindow | None) -> bool:
    """Whether a request runs guidance on any step: an image request whose window has steps."""
    return window is not None and window.start < window.stop


class ImageGenPhase(NamedTuple):
    """One denoise walk of a request."""
    walk: str
    # its loop's iterations; 0 for IMAGE_DECODE, which has no loop
    steps: int


def image_gen_phases(window: CFGWindow, num_timesteps: int, cfg_parallel: bool) -> tuple[ImageGenPhase, ...]:
    """The denoise walks a request runs, in order, from its `CFGWindow`.

    A request without guided steps runs image_gen alone. Otherwise the window
    is its own walk, with the window's branch count on every step: on the one
    LLM node, or, where the deployment places the branches on their own nodes,
    a CFG walk on them. The steps before it run on LLM alone, and those after
    it run on LLM alone and decode (or, with none left, a decode-only walk).
    Every walk fixes its branch count, so no node reads a loop counter for it.
    """
    n = num_timesteps - 1
    if window.start == window.stop:
        return (ImageGenPhase(IMAGE_GEN, n),)
    phases = []
    if window.start > 0:
        phases.append(ImageGenPhase(IMAGE_GEN_HEAD, window.start))
    phases.append(ImageGenPhase(WINDOW_WALKS[window.branches, cfg_parallel], window.stop - window.start))
    if window.stop < n:
        phases.append(ImageGenPhase(IMAGE_GEN, n - window.stop))
    else:
        phases.append(ImageGenPhase(IMAGE_DECODE, 0))
    return tuple(phases)


def combine_cfg_velocity(
    v_main: torch.Tensor,
    v_cfg_text: torch.Tensor,
    v_cfg_img: torch.Tensor | None,
    text_scale,
    img_scale,
    renorm_type: str,
    renorm_min: float,
) -> torch.Tensor:
    """BAGEL's two-node CFG velocity combine plus renormalization.

    ``v_cfg_img=None`` means the image branch did not run; that is only valid
    where ``img_scale <= 1``, when the image term is skipped anyway.
    """
    apply_img = v_cfg_img is not None and img_scale > 1.0
    if renorm_type == "text_channel":
        # text_channel: renorm AFTER text CFG, THEN apply image CFG
        v_text_guided = v_cfg_text + text_scale * (v_main - v_cfg_text)
        norm_v = torch.norm(v_main, dim=-1, keepdim=True)
        norm_v_text = torch.norm(v_text_guided, dim=-1, keepdim=True)
        scale = (norm_v / (norm_v_text + 1e-8)).clamp(min=renorm_min, max=1.0)
        v_text_renormed = v_text_guided * scale
        if apply_img:
            return v_cfg_img + img_scale * (v_text_renormed - v_cfg_img)
        return v_text_renormed

    # global / channel: apply both text+image CFG, THEN renorm
    v_text_guided = v_cfg_text + text_scale * (v_main - v_cfg_text)
    if apply_img:
        v_combined = v_cfg_img + img_scale * (v_text_guided - v_cfg_img)
    else:
        v_combined = v_text_guided

    if renorm_type == "channel":
        renorm_scale = (
            v_main.norm(dim=-1, keepdim=True) /
            (v_combined.norm(dim=-1, keepdim=True) + 1e-8)
        ).clamp(min=renorm_min, max=1.0)
    else:
        renorm_scale = (
            v_main.norm() / (v_combined.norm() + 1e-8)
        ).clamp(min=renorm_min, max=1.0)
    return v_combined * renorm_scale


def vae_latents(
    height: int, width: int, image_preprocess: str,
    latent_downsample: int, max_latent_size: int,
) -> int:
    """How many latents the VAE encodes an image of this size into. The VAE's
    ``prepare_inputs`` asserts the same count."""
    if image_preprocess == "vllm":
        height, width = vllm_vae_size(height, width, latent_downsample, max_latent_size * latent_downsample)
    else:
        height, width = _VAE_TRANSFORM.resize_transform.target_size(height, width)
    return (height // latent_downsample) * (width // latent_downsample)


class ViTEncoderSubmodule(NodeSubmodule):
    """SigLIP2 ViT + connector + vit_pos_embed: pixel patches -> ViT features.

    Receives preprocessed inputs containing packed pixel values, position IDs,
    cumulative sequence lengths, and max sequence length. Both vit_encoder and
    vae_encoder receive "image_inputs" as their graph input name; routing is
    handled by the graph edge's next_node field.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        connector: nn.Module,
        vit_pos_embed: nn.Module,
        vit_patch_size: int,
        vit_max_num_patch_per_side: int,
        accelerator_graph: bool = False,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.connector = connector
        self.vit_pos_embed = vit_pos_embed

        self.vit_patch_size = vit_patch_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.transform = _VIT_TRANSFORM
        self.vae_transform = _VAE_TRANSFORM

        self._vit_batching = os.environ.get("MSTAR_VIT_BATCHING", "0") == "1"
        # CUDA-graph capture of the ViT block loop, over the engine's ragged
        # attention resource. Off by default: capture costs one graph per
        # (bs, token bucket) and the eager flash-attn path is already fast;
        # the win is removing per-layer launch overhead on small images.
        self._accelerator_graph_enabled = (
            accelerator_graph and os.environ.get("MSTAR_VIT_ACCELERATOR_GRAPH", "0") == "1"
        )
        self._accelerator_graph_token_buckets = [
            int(t) for t in os.environ.get(
                "MSTAR_VIT_ACCELERATOR_GRAPH_TOKEN_BUCKETS", "512,1024,2048,4096,4900"
            ).split(",") if t.strip()
        ]  # 4900 == 70*70, the exact length the "vllm" preprocess option emits
        self._accelerator_graph_batch_sizes = [
            int(b) for b in
            os.environ.get("MSTAR_VIT_ACCELERATOR_GRAPH_BATCH_SIZES", "1,2,4").split(",")
            if b.strip()
        ] if self._vit_batching else [1]
        # `prepare_inputs` emits exactly one varlen segment (one image) per
        # request, so the segment ceiling is the batch size. Raise only if that
        # changes — and raise the resource's `max_segments_per_request` with it.
        self._max_images_per_request = 1

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        image_inputs = inputs["image_inputs"]

        image_preprocess = vit_preprocess(fwd_info.step_metadata.get("image_preprocess"))
        if image_preprocess == "vllm":
            # Fixed-square option: SiglipImageProcessor resizes to a fixed square
            # (size = max_num_patch_per_side * patch_size, e.g. 70*14=980),
            # discarding aspect ratio, then [0.5]/[0.5] normalize.
            square = self.vit_max_num_patch_per_side * self.vit_patch_size
            image_tensor = vllm_vit_resize(image_inputs[0], square)
            image_tensor = self.transform.normalize_transform(image_tensor)
        else:
            image_tensor = self.vae_transform.resize_transform(image_inputs[0])
            image_tensor = self.transform(image_tensor)

        device = image_tensor.device

        position_ids = get_flattened_position_ids_extrapolate(
            image_tensor.size(1), image_tensor.size(2),
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side,
            device=device,
        )
        pixel_values = patchify(image_tensor, self.vit_patch_size)
        # patchify yields (num_patches, p²·C); pre-patchify image_tensor is
        # (C, H, W), so its .shape[0] is the channel count, not the patch
        # count flashattn / cu_seqlens needs.
        num_tokens = pixel_values.shape[0]
        assert num_tokens == vit_patches(
            *image_inputs[0].shape[-2:], image_preprocess,
            self.vit_patch_size, self.vit_max_num_patch_per_side,
        ), (
            f"the ViT cut a {tuple(image_inputs[0].shape[-2:])} image into {num_tokens} patches, "
            "not the count `vit_patches` gives a layout to key it by"
        )

        # not `torch.tensor(..., device=)`: that syncs the stream, which held this request's
        # prepare until the in-flight decode steps of others had finished
        cu_seqlens = to_device_nonblocking([0, num_tokens], device, dtype=torch.int32)

        return NodeInputs(
            tensor_inputs={
                "packed_pixel_values": pixel_values,
                "packed_position_ids": position_ids,
            },
            kwargs={
                "cu_seqlens": cu_seqlens,
                "max_seqlen": num_tokens,
                # CPU-side per-image token count, so the batched preprocess
                # can build cu_seqlens without a GPU→CPU sync per request.
                "_num_tokens_per_image_cpu": [num_tokens],
            },
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **kwargs,
    ) -> NameToTensorList:
        features = self._encode(
            engine_inputs=engine_inputs,
            packed_pixel_values=packed_pixel_values,
            packed_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            seq_lens=[int(packed_pixel_values.shape[0])],
        )
        features = self.connector(features)
        features = features + self.vit_pos_embed(packed_position_ids)
        return {"img_emb": [features]}

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """Capture the ViT block loop, leaving patch-embed and the RoPE gathers
        eager — they are data-dependent indexing with no business in a graph."""
        if not self._accelerator_graph_enabled:
            return {}
        vit_config = self.vit_model.vision_model.config
        hidden = vit_config.hidden_size
        rope_dim = (hidden // vit_config.num_attention_heads) // 2

        def make_static_inputs(shape: PiecewiseCaptureShape):
            statics = {
                "hidden_states": torch.zeros(
                    shape.total_tokens, hidden, dtype=autocast_dtype, device=device
                )
            }
            if vit_config.rope:
                for name in ("cos_h", "sin_h", "cos_w", "sin_w"):
                    statics[name] = torch.zeros(
                        shape.total_tokens, rope_dim,
                        dtype=autocast_dtype, device=device,
                    )
            return statics

        return {
            VIT_BLOCK_LOOP: PiecewisePackedConfig(
                capture_fn=self._capture_block_loop,
                make_static_inputs=make_static_inputs,
                # one segment per row: `_max_images_per_request` is 1, so a
                # request's tokens are one independently-attending image
                attn_key=VIT_ATTN,
                total_tokens=sorted(self._accelerator_graph_token_buckets),
                capture_batch_sizes=sorted(self._accelerator_graph_batch_sizes),
            )
        }

    def _capture_block_loop(
        self, inp: PiecewiseCallInputs,
    ) -> dict[str, torch.Tensor]:
        """The captured region. Reads the runner-owned buffers, never rebinds.

        ``cu_seqlens`` / ``max_seqlen`` go unused on this path: the layout
        lives in the ragged attention resource, which the runner plans outside
        the graph before every replay.
        """
        rope_inputs = {
            name: inp.static_inputs[name]
            for name in ("cos_h", "sin_h", "cos_w", "sin_w")
            if name in inp.static_inputs
        }
        out = self.vit_model.vision_model.encode(
            inp.static_inputs["hidden_states"],
            cu_seqlens=None,
            max_seqlen=0,
            use_ragged=True,
            **rope_inputs,
        )
        return {"hidden_states": out}

    def _encode(
        self,
        engine_inputs: ModelInputsFromEngine,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        seq_lens: list[int],
    ) -> torch.Tensor:
        """ViT features for a packed batch, through the captured block loop
        when one fits and eager flash-attn otherwise.

        Only the replay is kept out of dynamo's reach; ``embed`` and ``encode``
        stay traceable, which is worth ~7% on this tower.
        """
        vision_model = self.vit_model.vision_model
        hidden_states, rope_inputs = vision_model.embed(
            packed_pixel_values, packed_position_ids
        )
        captured = self._replay_block_loop(
            engine_inputs, hidden_states, rope_inputs, cu_seqlens, seq_lens,
        )
        if captured is not None:
            return captured
        return vision_model.encode(
            hidden_states, cu_seqlens, max_seqlen, use_ragged=False, **rope_inputs
        )

    @torch.compiler.disable
    def _replay_block_loop(
        self,
        engine_inputs: ModelInputsFromEngine,
        hidden_states: torch.Tensor,
        rope_inputs: dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor,
        seq_lens: list[int],
    ) -> torch.Tensor | None:
        """The captured block loop's output, or None when it can't serve this
        batch and the caller should run eager."""
        runner = engine_inputs.piecewise_runners.get(VIT_BLOCK_LOOP)
        if runner is None or not runner.can_run(
            len(seq_lens), int(hidden_states.shape[0])
        ):
            return None
        # A captured bucket fixes the segment count at bs * max_images_per_request;
        # a request carrying more images than that takes the eager path rather
        # than tripping the wrapper's guard.
        n_segments = int(cu_seqlens.numel()) - 1
        if n_segments > len(seq_lens) * self._max_images_per_request:
            return None
        out = runner.run(
            static_inputs={"hidden_states": hidden_states, **rope_inputs},
            request_ids=list(engine_inputs.request_ids),
            seq_lens=seq_lens,
        )
        # a view: the connector consumes it within this step
        return out.get_view("hidden_states")

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        # Opt-in via MSTAR_VIT_BATCHING=1.
        if not self._vit_batching:
            return False
        return batch.graph_walk == "prefill_vit" and len(model_inputs) > 1

    def max_batch_size(self, graph_walk: str):
        if not self._vit_batching:
            return 1
        if graph_walk == "prefill_vit":
            # low batch size, as higher batch sizes can cause performance degredation
            return 4
        return None

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        packed_pixel_values = torch.cat(
            [inp.tensor_inputs["packed_pixel_values"] for inp in inputs], dim=0
        )
        packed_position_ids = torch.cat(
            [inp.tensor_inputs["packed_position_ids"] for inp in inputs], dim=0
        )
        per_image_lens: list[int] = []
        seq_lens: list[int] = []
        for inp in inputs:
            tokens_per_image: list[int] = inp.kwargs["_num_tokens_per_image_cpu"]
            per_image_lens.extend(tokens_per_image)
            seq_lens.append(sum(tokens_per_image))
        cu_lens_cpu = [0]
        for n in per_image_lens:
            cu_lens_cpu.append(cu_lens_cpu[-1] + n)
        cu_seqlens = torch.tensor(
            cu_lens_cpu, dtype=torch.int32, device=packed_pixel_values.device
        )
        return {
            "packed_pixel_values": packed_pixel_values,
            "packed_position_ids": packed_position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(per_image_lens) if per_image_lens else 0,
            "seq_lens": seq_lens,
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        seq_lens: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        features = self._encode(
            engine_inputs=engine_inputs,
            packed_pixel_values=packed_pixel_values,
            packed_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            seq_lens=seq_lens,
        )
        features = self.connector(features)
        features = features + self.vit_pos_embed(packed_position_ids)

        out: dict[str, NameToTensorList] = {}
        offset = 0
        for rid, n in zip(engine_inputs.request_ids, seq_lens, strict=True):
            out[rid] = {"img_emb": [features[offset:offset + n]]}
            offset += n
        return out


# Added to the request seed for the VAE posterior draw, so it is a different
# stream from the diffusion's initial noise (seeded with the request seed itself).
_VAE_NOISE_SEED_OFFSET = 0x9E3779B9


class VAEEncoderSubmodule(NodeSubmodule):
    """VAE encode + patchify + vae2llm + time_embedder + latent_pos_embed.

    Encodes an image tensor to VAE latents, patchifies them, and projects
    into the LLM hidden dimension with positional and timestep embeddings.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        vae2llm: nn.Linear,
        time_embedder: nn.Module,
        latent_pos_embed: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
        max_latent_size: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample
        self.max_latent_size = max_latent_size
        self.transform = ImageTransform(1024, 512, 16)


    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:

        """Convert raw images to VAE encoder input format.

        Computes patchified dimensions as Python ints for CUDA graph
        compatibility (no .item() calls in forward).

        Full implementation should include:
        - Image padding to be divisible by latent_downsample * latent_patch_size
        - VAE position ID computation from latent grid
        - Timestep preparation
        """
        image_inputs = inputs["image_inputs"]

        # [C, H, W]
        image_preprocess = fwd_info.step_metadata.get("image_preprocess", "default")
        img = image_inputs[0].contiguous()
        if image_preprocess == "vllm":
            # Stride-multiple resize (aspect-preserving, divisible
            # by latent_downsample, long <= max_latent_size*latent_downsample,
            # short >= 256), then [0.5]/[0.5] normalize.
            image_tensor = vllm_vae_resize(
                img,
                stride=self.latent_downsample,
                max_img_size=self.max_latent_size * self.latent_downsample,
            )
            image_tensor = self.transform.normalize_transform(image_tensor)
        else:
            image_tensor = self.transform(self.transform.resize_transform(img))
        device = image_tensor.device

        # Compute patchified dimensions as ints (CUDA graph compatible)
        p = self.latent_patch_size
        ds = self.latent_downsample
        _, img_h, img_w = image_tensor.shape
        h = (img_h // ds)
        w = (img_w // ds)
        assert h * w == vae_latents(*img.shape[-2:], image_preprocess, ds, self.max_latent_size), (
            f"the VAE encoded a {tuple(img.shape[-2:])} image into {h * w} latents, "
            "not the count `vae_latents` gives a layout to size it by"
        )

        packed_vae_position_ids = get_flattened_position_ids_extrapolate(
            img_h, img_w,
            self.latent_downsample,
            max_num_patches_per_side=self.max_latent_size
        )

        # The VAE samples its posterior. Drawing that noise here from the
        # request's seed makes image-to-image repeatable per seed, like the
        # diffusion's initial noise. The global RNG would differ per run.
        generator = torch.Generator(device=device)
        generator.manual_seed(fwd_info.random_seed + _VAE_NOISE_SEED_OFFSET)
        vae_noise = self.vae_model.posterior_noise(
            img_h, img_w, generator=generator, device=device,
        )

        tensor_inputs = {
            "padded_images": image_tensor.unsqueeze(0),
            "packed_vae_position_ids": packed_vae_position_ids,
            "packed_timesteps": torch.tensor([0.0], device=device),
            "vae_noise": vae_noise,
        }
        kwargs = {
            "h": h,
            "w": w,
        }
        return NodeInputs(tensor_inputs=tensor_inputs, kwargs=kwargs)


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        padded_images: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_timesteps: torch.Tensor,
        vae_noise: torch.Tensor,
        h: int,
        w: int,
        **kwargs,
    ) -> NameToTensorList:
        logger.debug(
            ("Running BAGEL VAE enc with padded_images shape=%s, "
             "packed_vae_position_ids shape=%s, packed_timesteps shape=%s, "
             "h=%d, w=%d"),
            padded_images.shape, packed_vae_position_ids.shape,
            packed_timesteps.shape, h, w
        )

        latent = self.vae_model.encode(padded_images, noise=vae_noise)

        p = self.latent_patch_size
        # h, w are already ints from preprocess (CUDA graph compatible)
        packed_latent = []
        for lat in latent:
            lat = lat[:, :h * p, :w * p].reshape(
                self.latent_channel, h, p, w, p
            )
            lat = torch.einsum("chpwq->hwpqc", lat).reshape(
                -1, p * p * self.latent_channel
            )
            packed_latent.append(lat)
        packed_latent = torch.cat(packed_latent, dim=0)

        # Project to hidden dim with timestep and position embeddings
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        return {"img_emb": [packed_latent]}

def _init_latents_and_time_index(
    config: BagelModelConfig,
    device,
    seed: int,
    H: int,
    W: int,
):

    h, w = (H // config.latent_downsample,
            W // config.latent_downsample)
    num_image_tokens = h * w

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    latents = torch.randn(
        num_image_tokens,
        config.vae_config.z_channels * config.latent_patch_size ** 2,
        generator=g,
        device=device,
    )
    if torch.is_autocast_enabled(latents.device.type):
        latents = latents.to(torch.get_autocast_dtype(latents.device.type))
    time_idx = torch.zeros(latents.shape[0], device=device)
    return latents, time_idx


class LLMSubmodule(ARNodeSubmodule):
    """Fat LLM wrapper that dispatches based on graph walk.

    Absorbs text_emb, lm_head, and flow_proj into a single node to avoid
    unnecessary IPC overhead. Graph walk-based dispatch handles:

      - prefill_text: embed_tokens -> LLM forward (causal, mode="und")
      - prefill_vit:  BOI + vit_emb + EOI -> LLM forward (bidirectional)
      - prefill_vae:  BOI + vae_emb + EOI -> LLM forward (bidirectional)
      - decode:       embed_tokens -> LLM forward -> lm_head -> argmax
      - image_gen:    the step's guidance branches -> llm2vae -> velocity
                      combine -> Euler step

    BOI/EOI tokens (<|vision_start|>, <|vision_end|>) are structural
    delimiters manually inserted around image embeddings during prefill.
    They are NOT predicted by the model (excluded from CE loss during
    training).

    During image_gen, classifier-free guidance runs one LLM pass per branch
    its denoise walk runs (`DenoiseWalk`: main, then cfg_text and cfg_img
    while their scales and the CFG interval keep them), each over its own KV
    cache. On a CFG walk (CFG-parallel deployments) each branch is its
    own node and combine_cfg combines them. With all three the velocities are combined via:
        v_final = v_cfg_img + img_scale * (
            v_cfg_text + text_scale * (v_main - v_cfg_text) - v_cfg_img
        )
    followed by an Euler step: x_{t+1} = x_t + v_final * dt.

    Multi-cache orchestration is driven by the requires_cfg flag in
    per-request metadata. When True, the declared steps cover 3 caches:
      - prefill_text: fork main->cfg_text before, forward [main, cfg_img]
      - prefill_vit/vae: forward [main], fork main->cfg_text at commit
      - decode: forward [main, cfg_img]
      - image_gen: the step's branches, with renormalization
    The runner drives the declaration (forks, plans, commits); every
    forward call names its label.
    """

    # Plan label the three CFG branches share when they run as one
    # combined plan; see declare_step.
    CFG_BATCHED_LABEL = "_cfg_batched"

    def _sample_tokens(
        self, request_ids: list[str], logits: torch.Tensor,
    ) -> torch.Tensor:
        """One token per row of this step's ``[rows, vocab]`` logits.

        ``request_ids`` is the batch the forward ran on, so under capture it is
        the slot's padding ids and ``logits`` has the padded row count — which
        is what the graph sampler's per-row params were gathered for. The
        engine maps the entries back onto the real ids.
        """
        tokens = self.node_resources["sampler"].sample(request_ids, logits)
        # FlashInfer reuses its sampling output buffer across calls, so a view
        # held past the next step aliases that step's token.
        return tokens.clone()

    def _sample_one(
        self, request_ids: list[str], logits: torch.Tensor,
    ) -> NameToTensorList:
        """The single-request forward's output shape: one flat name -> tensors."""
        return {"new_token": [self._sample_tokens(request_ids, logits)]}

    def _sample_per_request(
        self, request_ids: list[str], logits: torch.Tensor,
    ) -> BatchedModelOutput:
        """The batched forward's output: row i of ``new_token`` is request i."""
        return BatchedModelOutput(row_outputs={"new_token": self._sample_tokens(request_ids, logits)})

    def __init__(
        self,
        language_model: BagelForCausalLM,
        llm2vae: nn.Linear,
        vae2llm: nn.Linear,
        time_embedder: TimestepEmbedder,
        latent_pos_embed: PositionEmbedding,
        config: BagelModelConfig,
        boi_token_id: int | None = None,
        eoi_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        node_name: str = "LLM",
    ):
        super().__init__()
        self.node_name = node_name
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.llm2vae = llm2vae
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.config = config
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # least recently used first; bounded, since the output size is the client's
        self._image_gen_shape_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        # (device, dtype) -> the <|vision_start|> and <|vision_end|> embeddings
        self._sentinel_embeds: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_image_pos_ids(
        self, labels: list[str],
        request_id: str,
        device: str,
        seq_len: int,
    ):
        """Every token of an image block shares the stream's current position.

        The starts come from the position resource, which owns the counters
        (the old cache handle handed them in as a ``pos_info`` dict).
        """
        positions = self.node_resources["rope"]
        return {
            label: torch.zeros(
                seq_len, dtype=torch.int32, device=device,
            ) + positions.position(request_id, label)
            for label in labels
        }

    def _get_text_vae_idxs(
        self, seq_len: int, device: str
    ):
        tensor_inputs = {}
        tensor_inputs["text_indexes"] = to_device_nonblocking([0, seq_len - 1], device, dtype=torch.long)
        tensor_inputs["vae_token_indexes"] = torch.arange(
            1, seq_len-1,
            dtype=torch.long, device=device
        )
        tensor_inputs["text_mask"] = torch.zeros(
            seq_len, dtype=torch.bool, device=device
        )
        tensor_inputs["text_mask"][0] = True
        tensor_inputs["text_mask"][-1] = True

        return tensor_inputs

    PREFILL_TEXT_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    # past 4 for mixed_text, whose decode rows ride in these captures
    PREFILL_TEXT_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BatchedAcceleratorGraphConfig | PackedAcceleratorGraphConfig]:
        """Declare CUDA graph captures for ``decode`` (cfg-off + cfg-on) and ``prefill_text`` (cfg-off only).

        cfg-on prefill_text is intentionally NOT captured. BAGEL's cfg-on
        prefill_text declares a pre-forward fork of the main stream
        (main to cfg_text), an allocating operation. Replay addresses the
        cache manager at the real request ids, so the fork would land on
        real state now, but a captured cfg-on prefill has never been
        exercised and stays off until verified on its own. cfg-on prefill_text continues to
        use the eager path; downstream image_gen / decode+cfg captures are
        unaffected (they don't depend on this capture's snapshot
        semantics).
        """
        if not self.config.accelerator_graph:
            return []

        # `additional_key_info` carries what `requires_cfg` / `labels` used to:
        # the cfg-on and cfg-off decodes declare different segments, so they
        # are different capture buckets. `declare_step` stamps the same value
        # on the step (`cg_key_info`) so a replay lands on its own bucket.
        return [
            BatchedAcceleratorGraphConfig(
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk="decode",
                additional_key_info=False,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1
                ),
            ),
            BatchedAcceleratorGraphConfig(
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk="decode",
                additional_key_info=True,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                    resource_step_info=True, # requires cfg
                ),
            ),
            PackedAcceleratorGraphConfig(
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk="prefill_text",
                # decode rows of a mixed step are one-token segments
                replay_graph_walks=["prefill_text", MIXED_TEXT],
                capture_token_lengths=list(self.PREFILL_TEXT_TOKEN_BUCKETS),
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(n, dtype=torch.long, device=device),
                    input_seq_len=n
                ),
                additional_key_info=False,
                compile=True,
                capture_batch_sizes=self.PREFILL_TEXT_CAPTURE_BATCH_SIZES,
            ),
        ]

    def _get_active_labels(self, graph_walk: str, cfg: bool):
        return active_labels(graph_walk, cfg, self.node_name)

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs,
    ):
        """Guidance on/off, which is what separates this walk's two decode
        captures. Same fact `declare_step` stamps on the step, read from the
        batch rather than from the prepared inputs because the lease can be
        taken before `prepare_inputs` runs (the speculation path)."""
        del graph_walk, per_request_input_metadata, kwargs
        return self._batch_get_requires_cfg(per_request_info)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:

        device = self.get_device()
        node_inputs = ARNodeInputs(input_seq_len=0)
        # What `declare_step` needs but cannot see: whether this request runs
        # the guidance branches. It is per-request, so it travels with the
        # request's inputs rather than being read off the batch later.
        node_inputs.resource_step_info = fwd_info.step_metadata.get(
            "requires_cfg", False
        )

        if graph_walk == "prefill_text":
            node_inputs.input_ids = inputs["text_inputs"][0]
            node_inputs.input_seq_len = node_inputs.input_ids.shape[0]

        elif graph_walk == "decode":
            # NOTE: newly-sampled tokens automatically added to the seen token mask
            node_inputs.input_ids = inputs["text_inputs"][0]
            node_inputs.input_seq_len = 1

        if graph_walk in ["prefill_vit", "prefill_vae"]:
            node_inputs.input_embeds = self._wrap_with_boi_eoi(inputs["img_emb"][0])
            seq_len = inputs["img_emb"][0].shape[0] + IMAGE_SENTINELS
            node_inputs.input_seq_len = seq_len

            labels = ["main", "cfg_text", "cfg_img"] # just return all labels since it is cheap

            node_inputs.custom_pos_ids = self._get_image_pos_ids(
                labels, fwd_info.rid_handle, device, seq_len
            )

        if graph_walk == "prefill_vae":
            node_inputs.tensor_inputs = self._get_text_vae_idxs(seq_len, device)

        if graph_walk in DENOISE_WALKS:
            first_iter = "latents" not in inputs or len(inputs["latents"]) == 0
            H = fwd_info.step_metadata.get("height", 1024)
            W = fwd_info.step_metadata.get("width", 1024)
            shape_inputs = self._image_gen_shape_inputs(H, W, device)
            tensor_inputs = {"vae_position_ids": shape_inputs["vae_position_ids"]}
            if first_iter:
                node_inputs.input_embeds, tensor_inputs["time_index"] = _init_latents_and_time_index(
                    self.config, device, seed=fwd_info.random_seed, H=H, W=W
                )
            else:
               node_inputs.input_embeds = inputs["latents"][0]
               tensor_inputs["time_index"] = inputs["time_index"][0]

            tensor_inputs["empty_combined_emb"] = self._sentinel_frame(
                node_inputs.input_embeds.shape[0], node_inputs.input_embeds.dtype, device,
            )
            labels = ["main", "cfg_text", "cfg_img"]
            seq_len = tensor_inputs["empty_combined_emb"].shape[0]
            node_inputs.input_seq_len = seq_len
            node_inputs.custom_pos_ids = self._get_image_pos_ids(
                labels, fwd_info.rid_handle, device, seq_len
            )
            node_inputs.tensor_inputs = {
                **tensor_inputs,
                "text_indexes": shape_inputs["text_indexes"],
                "vae_token_indexes": shape_inputs["vae_token_indexes"],
                "text_mask": shape_inputs["text_mask"],
            }

        return node_inputs

    def _image_gen_shape_inputs(self, H: int, W: int, device) -> dict[str, torch.Tensor]:
        """The image_gen inputs that depend only on the output size, on the device, built once.

        Building them per step copied host data to the device (``torch.tensor(..., device=)`` and
        a CPU ``vae_position_ids`` indexing a device table), each a stream sync that held the next
        step's ``prepare_inputs`` until the current step had finished. Read-only downstream.
        """
        key = (H, W, str(device))
        cache = self._image_gen_shape_cache
        cached = cache.pop(key, None)
        if cached is None:
            pos = get_flattened_position_ids_extrapolate(
                H, W, self.config.latent_downsample, max_num_patches_per_side=self.config.max_latent_size,
                device=device,
            )
            cached = {
                "vae_position_ids": pos,
                **self._get_text_vae_idxs(pos.shape[0] + IMAGE_SENTINELS, device),
            }
        cache[key] = cached
        if len(cache) > _IMAGE_GEN_SHAPES_CACHED:
            del cache[next(iter(cache))]
        return cached

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        """This batch's step: which cache streams it touches, by how much,
        what backs them, and what commits.

        The three CFG branches are three labels on the same cache. Whether
        the batch runs them rides in on ``NodeInputs.resource_step_info`` from
        ``prepare_inputs``.
        """
        requires_cfg = requires_cfg_for_inputs(inputs)
        labels = self._get_active_labels(graph_walk, requires_cfg)
        if graph_walk in SINGLE_NODE_DENOISE_WALKS:
            # Only the branches the walk runs: labels are ordered main,
            # cfg_text, cfg_img, so the first n. One branch is the plain
            # "main" step below.
            labels = labels[:DENOISE_WALKS[graph_walk].branches]
        spans = tuple(inp.input_seq_len for inp in inputs)

        # Label-major, matching how a combined plan concatenates its sources;
        # this tuple is the ordering authority for every per-token array.
        segments = tuple(
            Segment(rid, label, span)
            for label in labels
            for rid, span in zip(request_ids, spans, strict=True)
        )
        per_label_pos_ids = {
            label: [
                inp.custom_pos_ids[label] for inp in inputs
                if isinstance(inp.custom_pos_ids, dict) and label in inp.custom_pos_ids
            ] for label in labels
        }

        if graph_walk in SINGLE_NODE_DENOISE_WALKS and len(labels) > 1:
            # Batched CFG: one combined plan across the active branch labels
            # so the step runs one forward instead of n. The frozen caches
            # never grow during flow matching, so nothing commits.
            pos_ids = [
                tensor for label in labels for tensor in per_label_pos_ids[label]
            ]
            return SubmoduleStep(cg_key_info=requires_cfg, steps={
                "kv": KVStep(
                    segments=segments,
                    commit=False,
                    combined_labels={tuple(labels): self.CFG_BATCHED_LABEL},
                ),
                "attn": AttentionStep(segments=segments, causal=False),
                "rope": PositionStep(
                    segments=segments,
                    # keyed by plan label: the combined plan is one label, and
                    # its ids are concatenated in the same label-major order
                    pos_ids={
                        self.CFG_BATCHED_LABEL: torch.cat(pos_ids)
                    } if pos_ids else None,
                    # the latents sit at the stream's current position every
                    # iteration; nothing here is kept, so nothing advances
                    advance=(0,) * len(segments),
                ),
            })

        info = per_request_info or {}
        guided_rids = sink = frozenset()
        if requires_cfg:
            # A pre-plan declares over the bucket's template rows, which are all
            # guided, so whether a row is guided comes from its request.
            guided_rids = frozenset(
                rid for rid, inp in zip(request_ids, inputs, strict=True)
                if (info[rid].step_metadata.get("requires_cfg", False) if rid in info else inp.resource_step_info)
            )
            sink = frozenset(
                (rid, label) for label in labels if label != "main" for rid in request_ids if rid not in guided_rids
            )
        writes = graph_walk not in DENOISE_WALKS
        if graph_walk in ("prefill_vit", "prefill_vae"):
            pos_advance = (IMAGE_ADVANCE,) * len(segments)
        elif not writes:
            # Flow matching: the caches are frozen and every iteration puts the
            # latents at the same position, so the counters must not move.
            pos_advance = (0,) * len(segments)
        else:
            pos_advance = None
        pos_ids = {
            label: torch.cat(tensors)
            for label, tensors in per_label_pos_ids.items() if tensors
        }

        pre_forks: tuple = ()
        post_forks: tuple = ()
        fork_rids = None
        row_walks = [inp.graph_walk or graph_walk for inp in inputs]
        if requires_cfg:
            if "prefill_text" in row_walks:
                # cfg_text keeps the pre-text context, so it forks before
                # anything plans or writes: the prompt rows only, not a
                # decode row sharing the step
                fork_rids = frozenset(
                    rid for rid, walk in zip(request_ids, row_walks, strict=True)
                    if walk == "prefill_text" and rid in guided_rids
                )
                pre_forks = (("main", "cfg_text"),)
            elif graph_walk in ("prefill_vit", "prefill_vae"):
                # cfg_text tracks the context including this image, so it
                # forks at commit, after the step's writes have landed.
                fork_rids = guided_rids
                post_forks = (("main", "cfg_text"),)

        # `cg_key_info` picks among the walk's capture buckets; it must match
        # the `additional_key_info` on the configs in get_cuda_graph_configs.
        steps: dict = {}
        # the rows whose token `postprocess` keeps
        kept_rids = frozenset(
            rid for rid, walk in zip(request_ids, row_walks, strict=True)
            if walk == "decode" or (
                rid in info and info[rid].step_metadata.get("sample_prefill_token", False)
            )
        ) if graph_walk in {"prefill_text", "prefill_vit", MIXED_TEXT} else None

        if graph_walk in ("prefill_text", MIXED_TEXT):
            # The prompt's tokens enter the repetition-penalty mask here; the
            # sampler resource adds them at plan time. Only prefill carries
            # them — a sampled token is tracked by the sampler itself.
            steps["sampler"] = SamplerStep(
                prefill_tracked_tokens={
                    rid: inp.input_ids
                    for rid, inp, walk in zip(request_ids, inputs, row_walks, strict=True)
                    if inp.input_ids is not None and walk == "prefill_text"
                },
                kept_rids=kept_rids,
            )
        elif graph_walk == "prefill_vit":
            # prefill_vit is the last walk before decode for an image prompt,
            # so it samples the first token too (prefill_vae never does).
            steps["sampler"] = SamplerStep(kept_rids=kept_rids)
        elif graph_walk == "decode":
            steps["sampler"] = SamplerStep()

        steps.update({
            "kv": KVStep(
                segments=segments,
                commit=writes,
                pre_forks=pre_forks,
                post_forks=post_forks,
                fork_rids=fork_rids,
                sink=sink,
            ),
            "attn": AttentionStep(
                segments=segments,
                causal=graph_walk in ("prefill_text", "decode", MIXED_TEXT),
            ),
            "rope": PositionStep(
                segments=segments,
                pos_ids=pos_ids or None,
                advance=pos_advance,
            ),
        })
        return SubmoduleStep(cg_key_info=requires_cfg, steps=steps)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        """Data transform for the forward; the plans come from the step
        declaration the runner drives before this runs."""
        if graph_walk in DENOISE_WALKS:
            assert len(inputs) == 1 , "Batching not supported for image gen"

        # Concatenate lists of tensors into single tensors for each input name
        result = ARNodeInputs.collate(inputs, stacking_method=StackingMethod.CAT)
        result["seq_lens"] = [inp.input_seq_len for inp in inputs]
        # Off the inputs, the same way `declare_step` reads it, so the forward
        # and the step it was declared under cannot disagree during capture, which
        # passes in dummy metadata
        result["requires_cfg"] = requires_cfg_for_inputs(inputs)
        result["sample_token"] = any([
            info.step_metadata.get("sample_prefill_token", True) \
                for info in engine_inputs.per_request_info.values()
        ])
        return result

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs
    ) -> NameToTensorList:

        request_info = engine_inputs.single_request_info
        kwargs.update(request_info.step_metadata)

        logger.debug("Running BAGEL LLM for graph walk %s", graph_walk)

        # The walks that never sample return straight out.
        if graph_walk == "prefill_vae":
            return self._forward_prefill_vae(**kwargs)
        elif graph_walk in SINGLE_NODE_DENOISE_WALKS:
            return self._forward_image_gen(branches=DENOISE_WALKS[graph_walk].branches, **kwargs)
        elif graph_walk in CFG_WALKS:
            return self._forward_image_gen_single_branch(**kwargs)

        if graph_walk == "prefill_text":
            out = self._forward_prefill_text(**kwargs)
        elif graph_walk == "prefill_vit":
            out = self._forward_prefill_vit(**kwargs)
        elif graph_walk == "decode":
            out = self._forward_decode(**kwargs)
        else:
            raise ValueError(f"Unknown LLM graph walk: {graph_walk!r}")

        # A prefill hands back logits only when it is meant to sample this step
        # (`sample_prefill_token`); decode always does.
        if _LOGITS in out:
            return self._sample_one(engine_inputs.request_ids, out.pop(_LOGITS)[0])
        return out

    def _forward_prefill_text(
        self, input_ids: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward (causal, mode='und') -> KV cache update.

        When requires_cfg is True (image generation mode), the declared
        step forks main -> cfg_text before anything plans, and the forward
        runs for main and cfg_img (both see the text tokens).
        """
        emb = self.embed_tokens(input_ids)
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        sample_token = kwargs.pop("sample_prefill_token", True)

        if requires_cfg:
            for label in ["main", "cfg_img"]:
                out = self.language_model(
                    emb, mode="und",
                    label=label, **kwargs
                )
                if label == "main":
                    hidden = out
        else:
            hidden = self.language_model(
                emb, mode="und",
                label="main", **kwargs
            )
        if sample_token:
            last_hidden = self.node_resources["attn"].select_last_hidden(hidden)
            logits = self.lm_head(last_hidden)
            # `_LOGITS` is the private handoff to whichever dispatcher called:
            # the shape of a sampled token is `forward` vs `forward_batched`'s
            # to decide, and this one runs under both.
            return {_LOGITS: [logits]}
        return {}

    def _forward_prefill_vit(
        self, input_embeds: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        """Wrap img_emb with BOI/EOI tokens -> LLM forward (bidirectional).

        When requires_cfg is True the declared step forks main ->
        cfg_text at commit (cfg_text = context including this image).
        """
        kwargs.pop("requires_cfg", None)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        sample_token = kwargs.pop("sample_prefill_token", True)

        hidden = self.language_model(
            input_embeds, mode="und",
            label="main", **kwargs
        )

        if sample_token:
            last_hidden = self.node_resources["attn"].select_last_hidden(hidden)
            logits = self.lm_head(last_hidden)
            return {_LOGITS: [logits]}
        return {}

    def _forward_prefill_vae(
        self, input_embeds: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
        text_mask: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        """VAE image emb -> LLM forward (bidirectional, gen mode).

        When requires_cfg is True the declared step forks main ->
        cfg_text at commit (cfg_text = context including this image).
        """
        kwargs.pop("requires_cfg", None)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        self.language_model(
            input_embeds, mode="gen",
            label="main",
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
            text_mask=text_mask,
            **kwargs
        )

        # NOTE: we will never sample a token from prefill_vae because it is alwaus
        # followed by prefill_vit
        return {}

    def _forward_decode(
        self, input_ids: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward -> lm_head -> logits.

        Returns logits; token sampling is done by the engine post-forward
        (outside CUDA graph capture).

        When requires_cfg is True: also forward for cfg_img to keep its
        KV cache in sync (cfg_img tracks all text, no images).
        """
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        kwargs.pop("temperature", None)
        kwargs.pop("top_k", None)
        kwargs.pop("top_p", None)
        emb = self.embed_tokens(input_ids)

        hidden = self.language_model(
            emb, mode="und",
            label="main", **kwargs
        )

        if requires_cfg:
            self.language_model(
                emb, mode="und",
                label="cfg_img", **kwargs
            )

        logits = self.lm_head(hidden[-1:])
        return {_LOGITS: [logits]}

    def _forward_image_gen(
        self,
        input_embeds: torch.Tensor,
        empty_combined_emb: torch.Tensor,
        vae_position_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_mask: torch.Tensor,
        time_index: torch.Tensor,
        branches: int,
        **kwargs,
    ) -> NameToTensorList:
        """Flow matching Euler step with optional batched CFG.

        Runs against the frozen KV caches (main, cfg_text, cfg_img) through
        one combined plan; the declared step writes no store and commits
        nothing since the caches are frozen during flow matching.

        ``branches`` is how many of those branches the walk's steps run
        (`DenoiseWalk`): 3, 2 (no image guidance) or 1 (guidance off, or
        outside the CFG window: main only), as `declare_step` planned them.

        Renormalization modes (cfg_renorm_type in config):
          - "global": single scalar renorm over all dimensions (default)
          - "channel": per-token renorm (independent scale per latent token)
        """
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        latents = input_embeds

        # latents, vae_position_ids, time_index and empty_combined_emb are all
        # built with a sequence length derived from the same (H, W), but they
        # arrive as separate inputs. Under torch.compile's dynamic shapes each
        # gets an independent symbolic dim, so Dynamo can't prove the adds below
        # line up. Assert the relationship explicitly.
        n_latent = latents.shape[0]
        torch._check(vae_position_ids.shape[0] == n_latent)
        torch._check(time_index.shape[0] == n_latent)
        torch._check(empty_combined_emb.shape[0] == n_latent + 2)

        N = self.config.num_timesteps
        shift = self.config.timestep_shift

        # Compute shifted timestep and step size for this iteration.
        # time_index goes from 0 to N-2 (N-1 total Euler steps).
        t_uniform = 1.0 - time_index / (N - 1)
        t_uniform_next = 1.0 - (time_index + 1) / (N - 1)
        timestep = apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_next = apply_timestep_shift(t=t_uniform_next, shift=shift)
        dt = (timestep - timestep_next)[0]  # positive step size

        pos_embed = self.latent_pos_embed(vae_position_ids)
        timestep_embeds = self.time_embedder(timestep)
        latents_ = self.vae2llm(latents) + timestep_embeds \
            + pos_embed

        empty_combined_emb[1:-1] = latents_
        logger.debug("packed_seq = %s", empty_combined_emb)  # tensor repr: keep lazy

        kwargs.pop("requires_cfg", None)
        cfg_text_scale = kwargs.pop("cfg_text_scale", self.config.cfg_text_scale)
        cfg_img_scale = kwargs.pop("cfg_img_scale", self.config.cfg_img_scale)
        renorm_type = kwargs.pop("cfg_renorm_type", self.config.cfg_renorm_type)
        kwargs.pop("cfg_interval", None)
        cfg_renorm_min = kwargs.pop("cfg_renorm_min", self.config.cfg_renorm_min)
        if branches > 1:
            # In the CFG window (decided at intake): the scales apply as given.
            # Two branches when image guidance is off (img scale <= 1), where
            # the combine never reads v_cfg_img.
            v_main, v_cfg_text, v_cfg_img = self._image_gen_branches_forward(
                branches, empty_combined_emb, text_indexes, vae_token_indexes, text_mask, kwargs,
            )
            v_final = combine_cfg_velocity(
                v_main, v_cfg_text, v_cfg_img,
                cfg_text_scale, cfg_img_scale,
                renorm_type, cfg_renorm_min,
            )
        else:
            # No CFG, or a guidance-on step outside the CFG interval, where
            # the combine reduces to v_main: a single forward pass over the
            # declared "main" plan.
            hidden = self.language_model(
                empty_combined_emb, mode="gen",
                label="main",
                vae_token_indexes=vae_token_indexes,
                text_indexes=text_indexes,
                text_mask=text_mask,
                **kwargs,
            )
            v_final = self.llm2vae(hidden)[1:-1]

        # Euler step: x_{t-dt} = x_t - v * dt  (velocity points data -> noise)
        latents = latents - v_final * dt
        if torch.is_autocast_enabled(latents.device.type):
            latents = latents.to(torch.get_autocast_dtype(latents.device.type))
        return {
            "latents": [latents],
            "time_index": [time_index + 1]
        }

    def _image_gen_branches_forward(
        self,
        n: int,
        empty_combined_emb: torch.Tensor,
        text_indexes: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_mask: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The first ``n`` CFG branches (main, cfg_text[, cfg_img]) in one LLM
        forward over the combined plan `declare_step` made of the same ``n``
        labels: the plan treats each branch as a "virtual request" over its
        own cache. Returns their VAE-space velocities, cfg_img None if n < 3.
        """
        S = empty_combined_emb.shape[0]
        batched_emb = empty_combined_emb.repeat(n, 1)  # [nS, hidden]

        # Expand MoT indexes with absolute offsets for each copy
        batched_text_indexes = torch.cat([
            text_indexes + i * S for i in range(n)
        ])
        batched_vae_indexes = torch.cat([
            vae_token_indexes + i * S for i in range(n)
        ])
        batched_text_mask = text_mask.repeat(n)

        hidden = self.language_model(
            batched_emb, mode="gen",
            label=self.CFG_BATCHED_LABEL,
            vae_token_indexes=batched_vae_indexes,
            text_indexes=batched_text_indexes,
            text_mask=batched_text_mask,
            **kwargs,
        )

        # Split output back per branch and project to VAE space
        v_main = self.llm2vae(hidden[:S])[1:-1]
        v_cfg_text = self.llm2vae(hidden[S:2*S])[1:-1]
        v_cfg_img = self.llm2vae(hidden[2*S:3*S])[1:-1] if n >= 3 else None
        return v_main, v_cfg_text, v_cfg_img

    def _forward_image_gen_single_branch(
        self,
        input_embeds: torch.Tensor,
        empty_combined_emb: torch.Tensor,
        vae_position_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_mask: torch.Tensor,
        time_index: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """Single-branch LLM forward on a CFG walk (``CFG_WALKS``).

        Each parallel LLM node (LLM, LLM_cfg_text, LLM_cfg_img) runs this
        method with its own cache label. The velocity output is sent to the
        combine_cfg node which applies the CFG formula and Euler step.
        """
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        kwargs.pop("requires_cfg", None)
        kwargs.pop("cfg_text_scale", None)
        kwargs.pop("cfg_img_scale", None)
        kwargs.pop("cfg_renorm_type", None)
        kwargs.pop("cfg_interval", None)
        kwargs.pop("cfg_renorm_min", None)

        latents = input_embeds
        label = self._cfg_label
        # See _forward_image_gen: tie the separately-passed sequence dims so
        # torch.compile's dynamic shapes can prove the adds below line up.
        n_latent = latents.shape[0]
        torch._check(vae_position_ids.shape[0] == n_latent)
        torch._check(time_index.shape[0] == n_latent)
        torch._check(empty_combined_emb.shape[0] == n_latent + 2)

        pos_embed = self.latent_pos_embed(vae_position_ids)

        N = self.config.num_timesteps
        shift = self.config.timestep_shift
        t_uniform = 1.0 - time_index / (N - 1)
        timestep = apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_embeds = self.time_embedder(timestep)

        latents_ = self.vae2llm(latents) + timestep_embeds + pos_embed
        empty_combined_emb[1:-1] = latents_

        hidden = self.language_model(
            empty_combined_emb, mode="gen",
            label=label,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
            text_mask=text_mask,
            **kwargs,
        )

        projected = self.llm2vae(hidden[1:-1])

        # Output velocity. Main LLM also passes through latents/time_index
        # for the combine_cfg node.
        result: NameToTensorList = {f"v_{label}": [projected]}
        if self.node_name == "LLM":
            result["time_index"] = [time_index]
        return result

    @property
    def _cfg_label(self) -> str:
        """This node's CFG branch, and its velocity edge ``v_<label>``, on a CFG walk."""
        return NODE_TO_CFG_LABEL.get(self.node_name, "main")

    def _batch_get_requires_cfg(self, per_request_info: dict[str, CurrentForwardPassInfo]):
        return any(
            info.step_metadata.get("requires_cfg", False)
            for info in per_request_info.values()
        )

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None=None,
        input_embeds: torch.Tensor | None=None,
        sample_token: bool=False,
        requires_cfg: bool=False,
        **kwargs
    ) -> dict[str, NameToTensorList]:
        """Batched forward pass for decode and prefill_text.

        Concatenates inputs across requests, runs a single LLM forward with
        one batched attention plan, then splits outputs back per-request.
        """
        request_ids = engine_inputs.request_ids

        if graph_walk == "decode":
            out = self._forward_decode_batched(
                input_ids=input_ids,
                requires_cfg=requires_cfg,
            )
        elif graph_walk in ("prefill_text", MIXED_TEXT):
            out = self._forward_prefill_text(
                input_ids=input_ids,
                requires_cfg=requires_cfg,
                # sample for cuda graph compatibility
                sample_prefill_token=True,
                **kwargs
            )
        elif graph_walk == "prefill_vit":
            out = self._forward_prefill_vit(
                input_embeds=input_embeds,
                requires_cfg=requires_cfg,
                sample_prefill_token=sample_token,
                **kwargs
            )
        else:
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

        # Every walk reaching here samples, except a prefill_vit with
        # `sample_prefill_token` off, which hands back no logits.
        if _LOGITS in out:
            # one `[bs, vocab]` sample for the whole batch, split per request
            return self._sample_per_request(request_ids, out.pop(_LOGITS)[0])
        return out

    def _forward_decode_batched(
        self,
        input_ids: torch.Tensor,
        requires_cfg: bool = False,
    ) -> NameToTensorList:
        """Batched decode: all requests generate 1 token each.

        1. Concatenate embeddings: [N, hidden] where N = num_requests
        2. Single LLM forward over the batch (batched attention)
        3. If any request requires CFG, run a second pass for cfg_img
        4. lm_head -> one [N, vocab] logits tensor, sampled by the caller
        """
        # 1. Embed and concatenate
        embs = self.embed_tokens(input_ids)

        # 2. Single LLM forward (main cache, already planned)
        hidden = self.language_model(
            embs, mode="und",
            label="main",
        )

        # 3. CFG sync pass for cfg_img if needed (already planned)
        if requires_cfg:
            self.language_model(
                embs, mode="und",
                label="cfg_img",
            )

        # 4. lm_head over the whole batch: one [N, vocab] tensor, so the batch
        # samples in one call rather than per-rid slice by slice.
        return {_LOGITS: [self.lm_head(hidden)]}


    def _sentinels(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The <|vision_start|> and <|vision_end|> embeddings, ``[1, hidden]`` each,
        embedded once per (device, dtype): building the ids on the device is a host
        copy that syncs the stream, and the gather and cast (all-reduced under TP)
        need not repeat every step."""
        assert self.boi_token_id is not None and self.eoi_token_id is not None
        embeds = self._sentinel_embeds.get((device, dtype))
        if embeds is None:
            ids = torch.tensor([self.boi_token_id, self.eoi_token_id], device=device)
            with torch.no_grad():
                boi, eoi = self.embed_tokens(ids).to(dtype).split(1)
            embeds = self._sentinel_embeds[(device, dtype)] = (boi, eoi)
        return embeds

    def _wrap_with_boi_eoi(self, emb: torch.Tensor) -> torch.Tensor:
        """Wrap embeddings with <|vision_start|> and <|vision_end|> tokens."""
        boi, eoi = self._sentinels(emb.device, emb.dtype)
        return torch.cat([boi, emb, eoi], dim=0)

    def _sentinel_frame(self, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``[n + 2, hidden]``: the sentinels in the first and last rows around ``n``
        rows the forward fills (an image_gen step's latents)."""
        boi, eoi = self._sentinels(device, dtype)
        frame = torch.empty((n + IMAGE_SENTINELS, self.config.hidden_size), dtype=dtype, device=device)
        frame[:1].copy_(boi)
        frame[-1:].copy_(eoi)
        return frame

    def max_batch_size(self, graph_walk: str):
        if graph_walk == "prefill_vit":
            return 2
        return None

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs]
    ):
        return batch.graph_walk in ["decode", "prefill_text", "prefill_vit", MIXED_TEXT]

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "new_token" not in outputs:
            return
        if request_info.graph_walk != "decode" and (
            not request_info.step_metadata.get("sample_prefill_token", False)
        ):
            outputs.pop("new_token")
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if (not ignore_eos and self.eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2 >= request_info.max_tokens):
            return {"decode_loop"}
        return set()


class VAEDecoderSubmodule(NodeSubmodule):
    """VAE decoder: latent grid -> pixel image."""

    def __init__(
        self,
        vae_model: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        """Prepare VAE decoder inputs.

        Unwraps latents from list. Image dimensions (image_h, image_w)
        are provided via per-request metadata and converted to ints for
        CUDA graph compatibility.
        """
        return NodeInputs(
            tensor_inputs={
                "latents": inputs["latents"][0]
            },
            kwargs={
                "image_h": fwd_info.step_metadata.get("height", 1024),
                "image_w": fwd_info.step_metadata.get("width", 1024)
            }
        )


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        latents: torch.Tensor,
        image_h: int,
        image_w: int,
        **kwargs,
    ) -> NameToTensorList:
        logger.debug(
            "Running BAGEL VAE dec with latents shape %s, h %d, w %d",
            latents.shape, image_h, image_w
        )
        H = image_h
        W = image_w

        p = self.latent_patch_size
        h = H // self.latent_downsample
        w = W // self.latent_downsample

        # Unpatchify: [num_patches, patch_dim] -> [1, C, H_latent, W_latent]
        latent = latents.reshape(1, h, w, p, p, self.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.latent_channel, h * p, w * p
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return {"image_output": [image]}


class CombineCFGSubmodule(NodeSubmodule):
    """Lightweight node: applies CFG formula + Euler step.

    Receives the VAE-space velocities of the walk's branches (v_main and
    v_cfg_text, and v_cfg_img on image_gen_cfg3) plus latents and time_index
    from the parallel LLM branches. Applies the CFG formula with
    renormalization, then performs an Euler step.

    Used in the CFG walks (``CFG_WALKS``, parallel CFG architecture).
    Runs on the same GPU as the main LLM branch (enc_dec engine, no KV cache).
    """

    def __init__(self, config: "BagelModelConfig"):
        super().__init__()
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        # The join has no parameters; incoming branch output identifies the
        # worker device without loading an otherwise-unused model component.
        device = inputs["v_main"][0].device

        first_iter = "latents" not in inputs or not inputs["latents"]
        # the walk's branches: v_cfg_img only on image_gen_cfg3
        result = {
            f"v_{label}": inputs[f"v_{label}"][0]
            for label in NODE_TO_CFG_LABEL.values() if inputs.get(f"v_{label}")
        }
        if first_iter:
            H = fwd_info.step_metadata.get("height", 1024)
            W = fwd_info.step_metadata.get("width", 1024)
            result["latents"], result["time_index"] = _init_latents_and_time_index(
                self.config, device=device, seed=fwd_info.random_seed, H=H, W=W
            )
        else:
            result = {
                "latents": inputs["latents"][0],
                "time_index": inputs["time_index"][0],
                **result,
            }
        return NodeInputs(tensor_inputs=result, kwargs=dict(fwd_info.step_metadata))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        v_main: torch.Tensor,
        v_cfg_text: torch.Tensor,
        latents: torch.Tensor,
        time_index: torch.Tensor,
        v_cfg_img: torch.Tensor | None = None,
        cfg_text_scale: float = None,
        cfg_img_scale: float = None,
        cfg_renorm_type: str = None,
        cfg_renorm_min: float = None,
        **kwargs,
    ) -> NameToTensorList:
        if cfg_text_scale is None:
            cfg_text_scale = self.config.cfg_text_scale
        if cfg_img_scale is None:
            cfg_img_scale = self.config.cfg_img_scale
        if cfg_renorm_type is None:
            cfg_renorm_type = self.config.cfg_renorm_type
        if cfg_renorm_min is None:
            cfg_renorm_min = self.config.cfg_renorm_min

        N = self.config.num_timesteps
        shift = self.config.timestep_shift

        # latents, the three velocity tensors and time_index all share the same
        # (H, W)-derived sequence length but arrive as separate inputs. Tie their
        # symbolic dims so torch.compile can prove the CFG combine / Euler step
        # below line up under dynamic shapes.
        n_latent = latents.shape[0]
        torch._check(v_main.shape[0] == n_latent)
        torch._check(v_cfg_text.shape[0] == n_latent)
        if v_cfg_img is not None:
            torch._check(v_cfg_img.shape[0] == n_latent)
        torch._check(time_index.shape[0] == n_latent)

        # Compute timestep and step size
        t_uniform = 1.0 - time_index / (N - 1)
        t_uniform_next = 1.0 - (time_index + 1) / (N - 1)
        timestep = apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_next = apply_timestep_shift(t=t_uniform_next, shift=shift)
        dt = (timestep - timestep_next)[0]

        # Branches project to VAE space before routing to this join. A CFG walk is
        # the request's guided window (`CFGWindow`), so every step combines.
        v_final = combine_cfg_velocity(
            v_main, v_cfg_text, v_cfg_img,
            cfg_text_scale, cfg_img_scale, cfg_renorm_type, cfg_renorm_min,
        )

        # Euler step
        new_latents = latents - v_final * dt
        if torch.is_autocast_enabled(new_latents.device.type):
            new_latents = new_latents.to(torch.get_autocast_dtype(new_latents.device.type))

        new_time_index = time_index + 1
        return {
            "latents": [new_latents],
            "time_index": [new_time_index],
        }
