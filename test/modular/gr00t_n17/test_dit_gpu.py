"""GR00T N1.7's DT on the GPU: the real ragged kernels against the masked reference.

``test_dit.py`` pins each piece of the DT on the CPU with the attention handed in. This file runs
the whole 32-block stack in bf16 on **FlashInfer's ragged prefill kernels** against NVIDIA's
``AlternateVLDiT``, the masked variant the checkpoint was trained as. The reference hands each
cross block one tensor of bridge rows plus a mask of which it may read; we hand it only those
rows, packed, one segment per request. Same weights and inputs, so the answers must differ only
by rounding: the packing is the mask.

The bar: ours against the reference in fp32 within **1.2x the bf16 floor**, the reference in bf16
against itself in fp32, measured here. Two traps would make a test that passes and says nothing:

1. **Random weights do not round, they diverge.** At ``std=0.3`` the 32 residual blocks send the
   output rms to ~3900 and the "floor" to 643, inside which every control passes. So the
   reference carries the checkpoint's 456 ``action_head.model.*`` tensors, and the output rms
   (~0.2) is asserted.
2. **One copy of the weights.** The stack is 1.09B parameters (4.4 GiB fp32, 2.2 GiB bf16, on a
   12 GiB device). The reference runs in fp32, is converted in place to bf16 for the floor, then
   hands its tensors to ours (built on ``meta``) through ``load_state_dict(..., assign=True)``.

FlashInfer JIT-compiles its kernels, so the tests also need ``nvcc`` (on PATH or under
``CUDA_HOME``) and ``ninja`` on PATH; they skip without either.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from mstar.engine.resources.attn.ragged.wrappers import (
    SUPPORTED_HEAD_DIMS,
    FlashInferRaggedCrossPrefillWrapper,
    FlashInferRaggedPrefillWrapper,
    padded_head_dim,
)
from mstar.model.components.diffusion.attention import joint_attention
from mstar.model.gr00t_n17.components.dit import DiT
from mstar.model.gr00t_n17.config import (
    SA_SPAN,
    VL_IMAGE_SPAN,
    VL_TEXT_SPAN,
    Gr00tN17Config,
)

from .helpers import BACKBONE, GR00T

# NVIDIA's `AlternateVLDiT` from a checkout of Isaac-GR00T, loaded by file path: the `gr00t`
# package's `__init__` imports nothing, and the reference's venv is not this one.
_REFERENCE = Path.home() / "ref/Isaac-GR00T/gr00t/model/modules/dit.py"
# the DT's weights, which `test_dit.py` counts off the safetensors header
_CKPT_PREFIX = "action_head.model."
_CKPT_TENSORS = 456

def _toolchain() -> bool:
    """Whether FlashInfer can JIT-build its kernels here. Without a compiler, a GPU machine fails
    at ``write_ninja`` instead of skipping, an environment failure that would hide a real one.
    ``CUDA_HOME`` alone is not enough: ``ninja`` lives in the venv's ``bin``, and running pytest
    as ``.venv/bin/python`` does not put it on PATH.
    """
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = shutil.which("nvcc") or (cuda_home and (Path(cuda_home) / "bin/nvcc").exists())
    return bool(nvcc and shutil.which("ninja"))


pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available() and GR00T.exists() and bool(BACKBONE)
        and _REFERENCE.exists() and _toolchain()
    ),
    reason=(
        "the full DT on the ragged kernels needs CUDA, the cached checkpoint, the reference at "
        f"{_REFERENCE} and FlashInfer's build toolchain (nvcc via PATH or CUDA_HOME, ninja on PATH)"
    ),
)

_DEVICE = torch.device("cuda:0")
_BF16 = torch.bfloat16

# Spelled out rather than imported: these are the numbers the tests exist to catch changing.
_SA = 41  # the DT's own rows: the state row and the 40 action rows
_IMAGE = 256  # the bridge's image rows, one tile
_EQUAL_TEXT = (37, 37)  # one bucket: both requests the same text length
_RAGGED_TEXT = (37, 45)  # two buckets, and what the reference pads to 45
_TIMESTEPS = (250, 750)  # one denoising step per request, and not the same one
_INPUT_SCALE = 0.3  # the rms the bridge's rows and the DT's own arrive at
_HEADS, _HEAD_DIM = 32, 48  # what the checkpoint fixes, asserted against the config below
_BAR = 1.2  # a multiple of the measured floor, never an absolute tolerance
_WORKSPACE = 128 << 20  # FlashInfer's scratch, per wrapper
# The trained weights, not a fill: a random stack at `_INPUT_SCALE` lands at rms ~3900 (trap 1).
_OUTPUT_RMS = (0.05, 1.0)


# ----------------------------------------------------------------------------
# The reference, and the checkpoint's own weights
# ----------------------------------------------------------------------------


def _reference_module():
    spec = importlib.util.spec_from_file_location("nvidia_gr00t_dit", _REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_tensors() -> dict[str, torch.Tensor]:
    """The ``action_head.model.*`` tensors, keyed as the reference stack names them."""
    weight_map = json.loads((GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for name, shard in sorted(weight_map.items()):
        if name.startswith(_CKPT_PREFIX):
            with safe_open(str(GR00T / shard), framework="pt") as shard_file:
                tensors[name[len(_CKPT_PREFIX) :]] = shard_file.get_tensor(name)
    return tensors


def _build_reference(config: Gr00tN17Config) -> torch.nn.Module:
    """``AlternateVLDiT`` at the checkpoint's shape and weights, fp32, on the device. The keyword
    arguments are what the checkpoint's config implies."""
    reference = _reference_module().AlternateVLDiT(
        num_attention_heads=config.dit.num_attention_heads,
        attention_head_dim=config.dit.attention_head_dim,
        output_dim=config.dit.output_dim,
        num_layers=config.dit.num_layers,
        norm_type="ada_norm",
        norm_elementwise_affine=False,
        interleave_self_attention=True,
        positional_embeddings=None,
        cross_attention_dim=config.backbone_embedding_dim,
        attend_text_every_n_blocks=config.attend_text_every_n_blocks,
    )
    tensors = _checkpoint_tensors()
    # `strict=True`: at `positional_embeddings=None` the blocks carry no `pos_embed`, so the state
    # dict is exactly the checkpoint's 456 tensors and every parameter is a trained one (trap 1).
    assert len(tensors) == _CKPT_TENSORS, len(tensors)
    reference.load_state_dict(tensors, strict=True)
    del tensors
    return reference.to(_DEVICE).eval()


# ----------------------------------------------------------------------------
# One input layout, the reference's answer for it, and the floor it sets
# ----------------------------------------------------------------------------


@dataclass
class _Layout:
    """One batch of inputs, in fp32, plus what the reference makes of it. The inputs are rounded
    at each call site, so input rounding is inside the floor and our error alike."""

    hidden: torch.Tensor  # [B, _SA, inner_dim], the DT's own rows
    text: tuple[torch.Tensor, ...]  # one [1, n, backbone_dim] per request
    image: torch.Tensor  # [B, _IMAGE, backbone_dim]
    timestep: torch.Tensor  # [B]
    want: torch.Tensor = field(default=None)  # the reference in fp32
    floor: tuple[float, ...] = ()  # per-request mean abs error of the reference in bf16
    floor_mean: float = 0.0  # ... and the same over the whole batch

    @property
    def batch(self) -> int:
        return self.hidden.shape[0]

    @property
    def text_rows(self) -> tuple[int, ...]:
        return tuple(rows.shape[1] for rows in self.text)

    def ours_args(self, swap: bool = False) -> tuple[torch.Tensor, ...]:
        """What our ``DiT.forward`` takes, in bf16, for a batch that shares its lengths. ``swap``
        (the negative control) hands the text blocks the image rows and vice versa."""
        assert len(set(self.text_rows)) == 1, "one bucket per distinct length"
        text = torch.cat(self.text, dim=0).to(_BF16)
        image = self.image.to(_BF16)
        return (self.hidden.to(_BF16), image if swap else text, text if swap else image)


def _layout(config: Gr00tN17Config, text_rows: tuple[int, ...], seed: int) -> _Layout:
    """Random rows at ``_INPUT_SCALE``. The text is drawn as one block and sliced per request, so
    the shorter request's rows are the longer one's first rows, as the reference's padding leaves."""
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    batch, width = len(text_rows), config.backbone_embedding_dim

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=_DEVICE) * _INPUT_SCALE

    hidden = randn(batch, _SA, config.dit.inner_dim)
    text = randn(batch, max(text_rows), width)
    image = randn(batch, _IMAGE, width)
    return _Layout(
        hidden=hidden,
        text=tuple(text[r : r + 1, :rows] for r, rows in enumerate(text_rows)),
        image=image,
        timestep=torch.tensor(_TIMESTEPS[:batch], device=_DEVICE),
    )


def _reference_forward(model: torch.nn.Module, layout: _Layout, dtype: torch.dtype) -> torch.Tensor:
    """The reference's one encoder tensor and its two masks: the text padded to the longest
    request with ``backbone_attention_mask`` marking the pad invalid, and ``image_mask``. With
    equal text lengths the pad is empty, so both layouts take this one path."""
    batch, rows = layout.batch, layout.text_rows
    pad, width = max(rows), layout.image.shape[2]
    encoder = torch.zeros(batch, pad + _IMAGE, width, device=_DEVICE, dtype=dtype)
    valid = torch.zeros(batch, pad + _IMAGE, dtype=torch.bool, device=_DEVICE)
    is_image = torch.zeros(batch, pad + _IMAGE, dtype=torch.bool, device=_DEVICE)
    for request, text in enumerate(layout.text):
        encoder[request, : rows[request]] = text[0].to(dtype)
        valid[request, : rows[request]] = True
    encoder[:, pad:] = layout.image.to(dtype)
    valid[:, pad:] = True
    is_image[:, pad:] = True
    with torch.no_grad():
        return model(
            hidden_states=layout.hidden.to(dtype),
            encoder_hidden_states=encoder,
            timestep=layout.timestep,
            image_mask=is_image,
            backbone_attention_mask=valid,
        ).float()


def _errors(got: torch.Tensor, want: torch.Tensor) -> tuple[float, ...]:
    """Mean absolute error per request."""
    difference = (got.float() - want.float()).abs()
    return tuple(difference[request].mean().item() for request in range(difference.shape[0]))


def _mean_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return (got.float() - want.float()).abs().mean().item()


# ----------------------------------------------------------------------------
# The ragged kernels, bound to the three spans
# ----------------------------------------------------------------------------


def _cu(lengths: list[int]) -> torch.Tensor:
    """A packed layout's offsets. On the host, which is what the wrappers prefer."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32)


class _Spans:
    """One self-attention and two cross-attention wrappers, built as ``test_ragged_attention.py``
    builds them, and planned by hand so the resource is not in between."""

    def __init__(self, config: Gr00tN17Config):
        def build(cls):
            return cls(
                workspace_buffer=torch.empty(_WORKSPACE, dtype=torch.uint8, device=_DEVICE),
                num_qo_heads=config.dit.num_attention_heads,
                num_kv_heads=config.dit.num_attention_heads,
                head_dim=config.dit.attention_head_dim,
                device=_DEVICE,
                q_data_type=_BF16,
            )

        self.own = build(FlashInferRaggedPrefillWrapper)
        self.text = build(FlashInferRaggedCrossPrefillWrapper)
        self.image = build(FlashInferRaggedCrossPrefillWrapper)

    def plan(self, batch: int, text_rows: int, image_rows: int = _IMAGE) -> dict:
        """Plan one bucket of ``batch`` requests; return the span-to-attention map ``DiT.forward`` takes."""
        own = _cu([_SA] * batch)
        self.own.plan(own)
        self.text.plan(own, _cu([text_rows] * batch))
        self.image.plan(own, _cu([image_rows] * batch))
        return {
            span: partial(joint_attention, ragged=wrapper.run)
            for span, wrapper in ((SA_SPAN, self.own), (VL_TEXT_SPAN, self.text), (VL_IMAGE_SPAN, self.image))
        }


# ----------------------------------------------------------------------------
# The one build: the reference, the floor, and ours sharing its tensors
# ----------------------------------------------------------------------------


@dataclass
class _Stacks:
    config: Gr00tN17Config
    ours: DiT
    equal: _Layout
    ragged: _Layout


@pytest.fixture(scope="module")
def stacks() -> _Stacks:
    """Both layouts measured against the reference, and ours carrying its tensors. Every fp32
    answer is taken before the in-place bf16 conversion: there is no room for a second copy."""
    config = Gr00tN17Config.from_hf(GR00T, BACKBONE)
    assert (config.dit.num_attention_heads, config.dit.attention_head_dim) == (_HEADS, _HEAD_DIM)
    reference = _build_reference(config)
    equal = _layout(config, _EQUAL_TEXT, seed=3)
    ragged = _layout(config, _RAGGED_TEXT, seed=5)

    for layout in (equal, ragged):
        layout.want = _reference_forward(reference, layout, torch.float32)
    reference = reference.to(_BF16)
    for layout in (equal, ragged):
        rounded = _reference_forward(reference, layout, _BF16)
        layout.floor = _errors(rounded, layout.want)
        layout.floor_mean = _mean_error(rounded, layout.want)

    # on `meta`, so a real build's 4.4 GiB of fp32 parameters is never allocated; `strict=True`
    # says every one was assigned
    with torch.device("meta"):
        ours = DiT(config)
    ours.load_state_dict(reference.state_dict(), strict=True, assign=True)
    del reference  # the parameters stay alive -- `ours` is now the only module holding them
    torch.cuda.empty_cache()
    yield _Stacks(config=config, ours=ours.eval(), equal=equal, ragged=ragged)
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# The packing is the mask
# ----------------------------------------------------------------------------


def test_the_packed_spans_are_the_reference_mask(stacks: _Stacks):
    """Both requests at the same lengths: ours on the ragged kernels in bf16 against the
    reference's masked attention in fp32, at the bar.

    Past the bar by a little points at the kernel or the sm_scale; by orders of magnitude, at the
    span assignment or the alternation, which the negative control separates. The output rms is
    asserted because random weights land at rms ~3900 and a "floor" of 643 that passes anything.
    """
    layout = stacks.equal
    spans = _Spans(stacks.config).plan(layout.batch, layout.text_rows[0])
    with torch.no_grad():
        got = stacks.ours(*layout.ours_args(), layout.timestep, spans).float()

    rms = layout.want.pow(2).mean().sqrt().item()
    assert _OUTPUT_RMS[0] < rms < _OUTPUT_RMS[1], f"output rms {rms:.3g}: not the trained weights"
    error = _mean_error(got, layout.want)
    assert error <= _BAR * layout.floor_mean, (
        f"mean abs error {error:.3e} past the bar {_BAR * layout.floor_mean:.3e} "
        f"({_BAR}x the measured bf16 floor {layout.floor_mean:.3e})"
    )


def test_two_buckets_are_the_reference_padded_to_the_longest(stacks: _Stacks):
    """Two text lengths are two buckets and two forwards (``joint_attention`` needs every segment
    in a plan to share its length); the reference instead pads to the longest and masks the pad.
    Each request must land at its **own** floor, so the padded one may not be worse.

    If this fails, bucketing is not free: the pad leaks into the reference's answer, or a short
    segment reads past its own rows in ours.
    """
    layout = stacks.ragged
    assert len(set(layout.text_rows)) == len(layout.text_rows), "the point is two lengths"
    spans = _Spans(stacks.config)
    for request, rows in enumerate(layout.text_rows):
        bucket = spans.plan(1, rows)
        with torch.no_grad():
            got = stacks.ours(
                layout.hidden[request : request + 1].to(_BF16),
                layout.text[request].to(_BF16),
                layout.image[request : request + 1].to(_BF16),
                layout.timestep[request : request + 1],
                bucket,
            ).float()
        error = _mean_error(got, layout.want[request : request + 1])
        bar = _BAR * layout.floor[request]
        assert error <= bar, (
            f"request {request} (text {rows}, padded to {max(layout.text_rows)} by the "
            f"reference): mean abs error {error:.3e} past the bar {bar:.3e} "
            f"({_BAR}x its floor {layout.floor[request]:.3e})"
        )


# ----------------------------------------------------------------------------
# head_dim 48 is a padded head dim
# ----------------------------------------------------------------------------


def test_the_wrappers_run_at_the_checkpoints_head_dim(stacks: _Stacks):
    """FlashInfer's **padded head_dim path**, the only path the DT has.

    The checkpoint is 32 heads x head_dim 48, and no FlashInfer prefill kernel exists for 48 (an
    unsupported head dim fails to build), so the wrappers run at 64 with q/k/v zero-padded and the
    output sliced back. The tests above already go through it; this pins that, and that the three
    wrappers build and run at the checkpoint's own shape.
    """
    config = stacks.config
    assert config.dit.attention_head_dim not in SUPPORTED_HEAD_DIMS
    assert padded_head_dim(config.dit.attention_head_dim) == 64

    batch = len(_EQUAL_TEXT)
    spans = _Spans(config)
    spans.plan(batch, _EQUAL_TEXT[0])
    generator = torch.Generator(device=_DEVICE).manual_seed(11)

    def shape(rows: int) -> tuple[int, int, int]:
        """``[batch * rows, 32, 48]``, which is what the packed layout above declared."""
        return batch * rows, config.dit.num_attention_heads, config.dit.attention_head_dim

    def packed(rows: int) -> torch.Tensor:
        return torch.randn(shape(rows), generator=generator, device=_DEVICE, dtype=_BF16)

    for wrapper, kv_rows in ((spans.own, _SA), (spans.text, _EQUAL_TEXT[0]), (spans.image, _IMAGE)):
        out = wrapper.run(packed(_SA), packed(kv_rows), packed(kv_rows))
        assert out.shape == shape(_SA), out.shape
        assert torch.isfinite(out).all()
        assert out.abs().max() > 0  # the sliced-back 48 columns, not the padding


# ----------------------------------------------------------------------------
# The negative control: which rows a block reads
# ----------------------------------------------------------------------------


def test_swapping_the_text_and_image_spans_breaks_the_bar(stacks: _Stacks):
    """The control: swap which rows the text and image blocks read, and the bar must reject it.

    ``_span_for`` has a cross block read the text rows when ``idx % 4 == 0``, the image rows
    otherwise. Swapped, every block still runs and every shape fits; only the rows read are wrong.
    Both runs are on **SDPA** (``attends=None``) so the span assignment is the only difference,
    and the correct run must sit inside the bar, so the bar is not just rejecting everything.
    """
    layout = stacks.equal
    with torch.no_grad():
        correct = stacks.ours(*layout.ours_args(), layout.timestep, None).float()
        swapped = stacks.ours(*layout.ours_args(swap=True), layout.timestep, None).float()

    bar = _BAR * layout.floor_mean
    assert _mean_error(correct, layout.want) <= bar, "the control's own baseline misses the bar"
    error = _mean_error(swapped, layout.want)
    assert error > bar, (
        f"swapped spans: mean abs error {error:.3e} is inside the bar {bar:.3e}, so the bar "
        "does not read the span assignment"
    )
