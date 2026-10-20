"""GR00T N1.7's DT on the GPU: the real ragged kernels against the masked reference.

``test_gr00t_n17_dit.py`` pins every piece of the DT on the CPU against inline references,
with the attention handed in as a callable so that no kernel enters the comparison it did
not call itself. This file is the other half: the whole 32-block stack, in bf16, with the
attention bound to **FlashInfer's ragged prefill kernels**, against NVIDIA's own
``AlternateVLDiT`` -- the masked variant, which is the one the checkpoint was trained as.

The two implementations say the same thing two different ways, and that is the point. The
reference hands every cross block **one** tensor of bridge rows plus a boolean mask that
says which of them that block may read. We hand each cross block only the rows it reads,
packed with no mask at all, one segment per request. So this file is the evidence that the
packing is the mask: same weights, same inputs, answers that differ only by rounding.

Three things are measured, and one is deliberately broken:

* **R8, equal spans.** Two requests, text 37 and image 256 each. Ours on the ragged kernels
  against the reference in fp32, at the **T3 bar**: within 1.2x the bf16 floor, and the floor
  is the reference in bf16 against itself in fp32, measured here rather than written down.
* **R8, different spans.** Two requests with text 37 and 45. ``joint_attention`` needs every
  request in a batch to share its span lengths, so two lengths are two **buckets**, one
  forward each (D07). The reference instead pads the text to 45, marks the pad invalid and
  masks it. Each request has to land at its own floor. This is D07's evidence.
* **R9.** The wrappers are built at the checkpoint's 32 heads x head_dim 48, which no
  FlashInfer kernel is instantiated for, so every run above goes through the pad-to-64 path.
* **The negative control.** Swap which rows the text blocks and the image blocks read and
  the answer has to move past the bar. Both sides of it run on SDPA, so the only difference
  is the span assignment -- a swapped *ragged* run against a correct *ragged* one would
  confound that with the kernel.

Two traps are worth naming, because both of them produce a test that passes and says nothing:

1. **Random weights do not round, they diverge.** Filling the 32 residual blocks at
   ``std=0.3`` sends the output rms to ~3900 and the "bf16 floor" with it, to 643. At that
   magnitude every control is inside the floor. So the reference is loaded with the real
   ``action_head.model.*`` tensors (456 of them) and the output rms is ~0.2 instead, which
   ``test_the_packed_spans_are_the_reference_mask`` asserts.
2. **One copy of the weights, not two.** The stack is 1.09B parameters: 4.4 GiB in fp32,
   2.2 GiB in bf16, against 12 GiB of device memory. The reference is built once, run in
   fp32 for the answer, converted in place to bf16 for the floor, and then **hands its
   tensors to ours** through ``load_state_dict(..., assign=True)``. Ours is built on
   ``meta`` so that nothing is allocated for it to throw away.

Running it needs the CUDA shim on ``PATH`` for FlashInfer's JIT, and the venv's ``bin`` with
it or the build fails on a missing ``ninja``::

    CUDA_HOME=$HOME/.local/cuda-home \\
    PATH=$HOME/mstar/.venv/bin:$HOME/.local/cuda-home/bin:$PATH \\
    pytest test/modular/test_gr00t_n17_dit_gpu.py

Not a CPU-core test: it needs a GPU and the cached checkpoint, and it is skipped without
either.
"""

from __future__ import annotations

import glob
import importlib.util
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from mstar.engine.resources.attn.ragged.wrappers import (
    SUPPORTED_HEAD_DIMS,
    RaggedCrossPrefillWrapper,
    RaggedPrefillWrapper,
    padded_head_dim,
)
from mstar.engine.resources.attn.workspace import Workspace
from mstar.model.components.diffusion.attention import joint_attention
from mstar.model.gr00t_n17.components.dit import DiT
from mstar.model.gr00t_n17.config import (
    SA_SPAN,
    VL_IMAGE_SPAN,
    VL_TEXT_SPAN,
    Gr00tN17Config,
)

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)
# NVIDIA's `AlternateVLDiT`, loaded by file path: the `gr00t` package's `__init__` imports
# nothing, so there is no module to import, and the reference's venv is not this one.
_REFERENCE = Path.home() / "ref/Isaac-GR00T/gr00t/model/modules/dit.py"
# the DT's weights, which `test_gr00t_n17_dit.py` counts off the safetensors header
_CKPT_PREFIX = "action_head.model."
_CKPT_TENSORS = 456

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _GR00T.exists() and bool(_BACKBONE) and _REFERENCE.exists()),
    reason="the full DT on the ragged kernels needs CUDA, the cached checkpoint and the reference",
)

_DEVICE = torch.device("cuda:0")
_BF16 = torch.bfloat16

# Spelled out rather than imported, as in the CPU file: these are the numbers the tests exist
# to catch changing.
_SA = 41  # the DT's own rows: the state row and the 40 action rows
_IMAGE = 256  # the bridge's image rows, one tile
_EQUAL_TEXT = (37, 37)  # R8's batch-wide case: both requests the same text length
_RAGGED_TEXT = (37, 45)  # R8's two buckets, and what the reference pads to 45
_TIMESTEPS = (250, 750)  # one denoising step per request, and not the same one
_INPUT_SCALE = 0.3  # the rms the bridge's rows and the DT's own arrive at
_HEADS, _HEAD_DIM = 32, 48  # what the checkpoint fixes, asserted against the config below
_T3 = 1.2  # the T3 bar: a multiple of the measured floor, never an absolute tolerance
_WORKSPACE = 128 << 20  # FlashInfer's scratch, per wrapper
# What says the weights are the trained ones and not a fill: a random stack at `_INPUT_SCALE`
# lands at rms ~3900 (trap 1 in the banner).
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
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for name, shard in sorted(weight_map.items()):
        if name.startswith(_CKPT_PREFIX):
            with safe_open(str(_GR00T / shard), framework="pt") as shard_file:
                tensors[name[len(_CKPT_PREFIX) :]] = shard_file.get_tensor(name)
    return tensors


def _build_reference(config: Gr00tN17Config) -> torch.nn.Module:
    """``AlternateVLDiT`` at the checkpoint's shape, carrying the checkpoint's weights, in
    fp32 on the device. The keyword arguments are the ones the checkpoint's own config
    implies; ``positional_embeddings=None`` is why the blocks have no ``pos_embed``.
    """
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
    # `strict=True`: at `positional_embeddings=None` the blocks carry no `pos_embed`, so the
    # reference's state dict is exactly the checkpoint's 456 tensors and every parameter in
    # the stack is a trained one. That is trap 1 in the banner, and this is where it is held.
    assert len(tensors) == _CKPT_TENSORS, len(tensors)
    reference.load_state_dict(tensors, strict=True)
    del tensors
    return reference.to(_DEVICE).eval()


# ----------------------------------------------------------------------------
# One input layout, the reference's answer for it, and the floor it sets
# ----------------------------------------------------------------------------


@dataclass
class _Layout:
    """One batch of inputs, in fp32, plus what the reference makes of it.

    The inputs stay fp32 and are rounded at each call site, so the input rounding is inside
    the floor and inside our error alike -- the ratio of the two is what the bar reads.
    """

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
        """What our ``DiT.forward`` takes, in bf16, for a batch that shares its lengths.

        ``swap`` is the negative control: the text blocks are handed the image rows and the
        image blocks the text rows, which is the only thing that changes.
        """
        assert len(set(self.text_rows)) == 1, "one bucket per distinct length"
        text = torch.cat(self.text, dim=0).to(_BF16)
        image = self.image.to(_BF16)
        return (self.hidden.to(_BF16), image if swap else text, text if swap else image)


def _layout(config: Gr00tN17Config, text_rows: tuple[int, ...], seed: int) -> _Layout:
    """Random rows at ``_INPUT_SCALE``. The text is drawn as one block and sliced per
    request, so the shorter request's rows are the longer one's first rows -- which is also
    what the reference's padding leaves in place.
    """
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
    """The reference's one encoder tensor and its two masks.

    The text is padded to the longest request and ``backbone_attention_mask`` marks the pad
    invalid; ``image_mask`` says which rows are the image. With equal text lengths the pad is
    empty and the validity mask is all ones, so both layouts run through this one path.
    """
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
    """One self-attention wrapper and two cross-attention wrappers, built the way
    ``test_ragged_attention.py`` builds them.

    In the engine these are one ``RaggedAttentionSpec`` and one ``RaggedCrossAttentionSpec``
    carrying two labels, planned by the resource; here they are planned by hand so that this
    file tests the kernels and the packing without the resource in between.
    """

    def __init__(self, config: Gr00tN17Config):
        def build(cls):
            return cls(
                workspace_buffer=Workspace(torch.empty(_WORKSPACE, dtype=torch.uint8, device=_DEVICE), "test"),
                num_qo_heads=config.dit.num_attention_heads,
                num_kv_heads=config.dit.num_attention_heads,
                head_dim=config.dit.attention_head_dim,
                device=_DEVICE,
                q_data_type=_BF16,
            )

        self.own = build(RaggedPrefillWrapper)
        self.text = build(RaggedCrossPrefillWrapper)
        self.image = build(RaggedCrossPrefillWrapper)

    def plan(self, batch: int, text_rows: int, image_rows: int = _IMAGE) -> dict:
        """Plan one bucket: ``batch`` requests that each read ``text_rows`` and
        ``image_rows``, and return the span-to-attention map ``DiT.forward`` takes.
        """
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
    """Both layouts measured against the reference, and ours carrying its tensors.

    The order is forced by memory, not by taste: every fp32 answer has to be taken before the
    reference is converted, because the conversion is in place and there is no room for a
    second copy of 1.09B parameters.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
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

    # on `meta`, so the 4.4 GiB of fp32 parameters a real build would allocate and then
    # throw away is never allocated; `strict=True` is what says every one was assigned
    with torch.device("meta"):
        ours = DiT(config)
    ours.load_state_dict(reference.state_dict(), strict=True, assign=True)
    del reference  # the parameters stay alive -- `ours` is now the only module holding them
    torch.cuda.empty_cache()
    yield _Stacks(config=config, ours=ours.eval(), equal=equal, ragged=ragged)
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# R8: the packing is the mask
# ----------------------------------------------------------------------------


def test_the_packed_spans_are_the_reference_mask(stacks: _Stacks):
    """R8 with both requests at the same lengths: ours on the ragged kernels in bf16, against
    the reference's masked attention in fp32, at the T3 bar.

    If this fails, the DT's answer is not the trained model's answer, and nothing downstream
    of it means anything -- the actions served would be a different policy. The two ways it
    can fail say different things: past the bar by a little is the kernel or the sm_scale,
    past it by orders of magnitude is the span assignment or the alternation, which is what
    the negative control below separates.

    The output rms is asserted with it, because a stack filled with random weights instead of
    the checkpoint's lands at rms ~3900 and sets a "floor" of 643 -- and inside a floor that
    size this test passes no matter what the kernels do.
    """
    layout = stacks.equal
    spans = _Spans(stacks.config).plan(layout.batch, layout.text_rows[0])
    with torch.no_grad():
        got = stacks.ours(*layout.ours_args(), layout.timestep, spans).float()

    rms = layout.want.pow(2).mean().sqrt().item()
    assert _OUTPUT_RMS[0] < rms < _OUTPUT_RMS[1], f"output rms {rms:.3g}: not the trained weights"
    error = _mean_error(got, layout.want)
    assert error <= _T3 * layout.floor_mean, (
        f"mean abs error {error:.3e} past the T3 bar {_T3 * layout.floor_mean:.3e} "
        f"({_T3}x the measured bf16 floor {layout.floor_mean:.3e})"
    )


def test_two_buckets_are_the_reference_padded_to_the_longest(stacks: _Stacks):
    """R8 with two text lengths, which is D07's evidence.

    ``joint_attention`` packs one segment per request and every segment in a plan shares its
    length, so a batch with two text lengths is two buckets and two forwards. The reference
    has no buckets: it pads the text to the longest request, marks the pad invalid and lets
    the mask drop it. The claim is that those are the same answer, and each request has to
    land at its **own** floor -- so a request whose text was padded is not allowed to be
    worse than one whose text was not.

    If this fails, D07 is wrong and bucketing is not free: either the pad leaks into the
    reference's answer or a short segment reads past its own rows in ours. Either way the
    engine would have to pad and mask like the reference instead of bucketing, which is a
    different resource and a different plan.
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
        bar = _T3 * layout.floor[request]
        assert error <= bar, (
            f"request {request} (text {rows}, padded to {max(layout.text_rows)} by the "
            f"reference): mean abs error {error:.3e} past the T3 bar {bar:.3e} "
            f"({_T3}x its floor {layout.floor[request]:.3e})"
        )


# ----------------------------------------------------------------------------
# R9: head_dim 48 is a padded head dim
# ----------------------------------------------------------------------------


def test_the_wrappers_run_at_the_checkpoints_head_dim(stacks: _Stacks):
    """R9: FlashInfer's **padded head_dim path**, which is the only path the DT has.

    The checkpoint is 32 heads x head_dim 48, and no FlashInfer prefill kernel is
    instantiated for 48 -- an unsupported head dim does not fall back, it fails to build. So
    the wrappers run at the next supported size, 64, with q/k/v zero-padded and the output
    sliced back to 48. Every number the two tests above produce already came through that
    path; this one pins it as the reason, and that the three wrappers build and run at the
    checkpoint's own shape rather than at a size chosen to be convenient.

    If this fails, either the checkpoint's head dim stopped being a padded one (and the two
    tests above stopped covering the padding) or the wrappers cannot be built at it at all,
    in which case the DT has no attention backend on this machine.
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
    """What the T3 bar above would catch, shown by breaking it.

    ``_span_for`` says a cross block reads the text rows when ``idx % 4 == 0`` and the image
    rows otherwise. Swap the two and every cross block still runs, every shape still fits and
    the output is still finite -- the only thing wrong is which rows each block read. That is
    exactly the failure a parity test exists to catch, so the bar has to reject it.

    Both runs are on **SDPA** (``attends=None``), not on the ragged kernels. A swapped ragged
    run against a correct ragged one would differ in two ways at once, and a reader could not
    tell whether the bar was reading the span assignment or the kernel. On one path the span
    assignment is the only difference there is, which is also why the correct SDPA run is
    asserted to sit at the floor: it says the bar is not simply rejecting everything.

    If this fails, the bar in the two tests above is not measuring the alternation, and the
    DT could be reading the image rows where it should read the text rows and still pass.
    """
    layout = stacks.equal
    with torch.no_grad():
        correct = stacks.ours(*layout.ours_args(), layout.timestep, None).float()
        swapped = stacks.ours(*layout.ours_args(swap=True), layout.timestep, None).float()

    bar = _T3 * layout.floor_mean
    assert _mean_error(correct, layout.want) <= bar, "the control's own baseline misses the bar"
    error = _mean_error(swapped, layout.want)
    assert error > bar, (
        f"swapped spans: mean abs error {error:.3e} is inside the T3 bar {bar:.3e}, so the bar "
        "does not read the span assignment"
    )
