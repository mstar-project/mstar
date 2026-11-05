"""Linears in a low-precision dtype, opt-in per deployment (``linear_dtype``).

Three pieces, each declared once:

- a format (``LinearFormat``) owns its weight quantization, its activation quantization and its GEMM, as one
  ``mstar::<format>_linear`` op with a fake implementation, so a compiled forward sees one opaque node per linear;
  ``block_scaled.py`` declares ``mxfp8`` and ``nvfp4``;
- a linear kind says whether it takes a format (``quant_capability``), as ``lora_capability`` does for LoRA:
  ``nn.Linear`` and ``FusedColumnLinear`` do, and a kind that can't says why, so a declared linear is never skipped
  silently;
- a model declares which of its linears may go low precision and which stay bf16 (``QuantizableLinears``), and a
  deployment can keep more of them bf16 (``linear_keep_bf16``).

``apply_linear_dtype`` swaps the selected linears for ``QuantLinear``s once, at load, after any LoRA merge and
before compile. The output moves from the bf16 model's by the format's rounding, so it is a per-model quality call
and off by default.
"""

from __future__ import annotations

import fnmatch
import importlib
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import torch
from torch import nn

from mstar.engine.resources.lora.config import LoraTarget
from mstar.model.components.lora.capable import LoraCapable, LoraOutputAdd

logger = logging.getLogger(__name__)

BF16 = "bf16"
# a format's name -> the module that declares it, imported on first use (it registers its ops)
_FORMAT_MODULES = {
    "mxfp8": "mstar.model.components.block_scaled",
    "nvfp4": "mstar.model.components.block_scaled",
}
LINEAR_DTYPES = (BF16, *_FORMAT_MODULES)
_FORMATS: dict[str, LinearFormat] = {}


class LinearFormat:
    """One low-precision linear format: what ``QuantLinear`` stores for a weight and how it runs a GEMM on it.

    A subclass names the buffers it stores (``buffers``), fills them from a bf16 weight (``quantize_weight``) and
    runs ``x @ W.T`` on them (``linear``), quantizing ``x`` itself. ``linear`` is what a compiled forward traces, so
    it should be one op (``torch.library``) with a fake implementation, or plain torch Inductor can fuse."""

    name: ClassVar[str]
    # the QuantLinear buffers ``linear`` takes after the activation, in order
    buffers: ClassVar[tuple[str, ...]]

    def unavailable_reason(self, device: torch.device) -> str | None:
        """None when this process runs the format's GEMM on ``device``; otherwise why not."""
        raise NotImplementedError

    def check(self, out_features: int, in_features: int, dtype: torch.dtype) -> None:
        """Raise ``ValueError`` for a weight the format's kernels can't take."""

    def quantize_weight(self, weight: torch.Tensor, input_amax: float | None) -> dict[str, torch.Tensor | None]:
        """``buffers`` for one ``[N, K]`` weight. ``input_amax`` is the largest magnitude the model declares this
        linear's input reaches (``QuantizableLinears.input_amax``), for a format whose activation scale has a range
        to place; None when it declares none."""
        raise NotImplementedError

    def linear(self, x: torch.Tensor, *buffers: torch.Tensor | None) -> torch.Tensor:
        """``x [M, K]`` (contiguous) times the stored weight, ``[M, N]`` in ``x``'s dtype."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


def register_format(fmt: LinearFormat) -> LinearFormat:
    if fmt.name in _FORMATS:
        raise ValueError(f"linear format {fmt.name!r} is declared twice")
    _FORMATS[fmt.name] = fmt
    return fmt


def linear_format(name: str) -> LinearFormat:
    """The format ``linear_dtype: <name>`` runs."""
    if name not in _FORMAT_MODULES:
        raise ValueError(f"linear_dtype must be one of {LINEAR_DTYPES}, got {name!r}")
    if name not in _FORMATS:
        importlib.import_module(_FORMAT_MODULES[name])
    return _FORMATS[name]


def check_linear_dtype(linear_dtype: str) -> str:
    """A model config's ``linear_dtype``, checked when the model is constructed."""
    if linear_dtype not in LINEAR_DTYPES:
        raise ValueError(f"linear_dtype must be one of {LINEAR_DTYPES}, got {linear_dtype!r}")
    return linear_dtype


def check_linear_keep(keep: Sequence[str] | None, knob: str = "linear_keep_bf16") -> tuple[str, ...]:
    """A config's keep list: a list of patterns. A bare string would iterate as one-character patterns, and its
    ``*`` would keep every linear."""
    if keep is None:
        return ()
    if isinstance(keep, str) or not all(isinstance(p, str) for p in keep):
        raise ValueError(f"{knob} is a list of module-name patterns, got {keep!r}")
    return tuple(keep)


@dataclass(frozen=True)
class StepDtypes:
    """The dtype a transformer's quantizable linears run in on each denoise step (``linear_dtype_steps``): step
    ``k`` takes ``steps[k]``, and steps past the list its last entry. The DiT scaffold keys a step on it
    (``StepBucket``), so steps of two dtypes batch and capture apart."""

    steps: tuple[str, ...]

    @property
    def dtypes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.steps))

    def at(self, step: int) -> str:
        return self.steps[min(step, len(self.steps) - 1)]


def check_linear_dtype_steps(steps: int | Sequence[str] | None, linear_dtype: str) -> StepDtypes | None:
    """A config's ``linear_dtype_steps``: a count of leading denoise steps that run bf16, or a dtype per step, each
    ``bf16`` or the deployment's ``linear_dtype``. None when every step runs ``linear_dtype``."""
    if steps is None:
        return None
    if isinstance(steps, bool) or not isinstance(steps, int | list | tuple):
        raise ValueError(f"linear_dtype_steps is a count of leading bf16 steps or a list of dtypes, got {steps!r}")
    if isinstance(steps, int):
        if steps < 0:
            raise ValueError(f"linear_dtype_steps must be >= 0, got {steps}")
        steps = [BF16] * steps + [linear_dtype]
    steps = tuple(steps)
    unknown = [dtype for dtype in steps if dtype not in (BF16, linear_dtype)]
    if not steps or unknown:
        raise ValueError(f"linear_dtype_steps entries are {BF16} or linear_dtype ({linear_dtype}), got {list(steps)}")
    if set(steps) == {linear_dtype}:
        return None
    return StepDtypes(steps)


class QuantLinear(nn.Module, LoraCapable):
    """``x @ W.T (+ b)`` with ``W`` held in a ``LinearFormat``; input of any rank, output in its dtype. Replaces the
    module that held the bf16 weight, so the weight's bytes go.

    LoRA-capable with the replaced linear's target: a bound delta reads the same bf16 input and adds to the output,
    so it doesn't see the format."""

    lora_twin = LoraOutputAdd

    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor | None, fmt: LinearFormat, *,
        input_amax: float | None = None, lora_target: LoraTarget | None = None,
    ):
        super().__init__()
        self.fmt = fmt
        self.out_features, self.in_features = weight.shape
        fmt.check(self.out_features, self.in_features, weight.dtype)
        for name, tensor in fmt.quantize_weight(weight.detach(), input_amax).items():
            self.register_buffer(name, tensor)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self._lora_target = lora_target or LoraTarget(
            in_features=self.in_features, out_slices=(self.out_features,), parallel="none",
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear | nn.Module, fmt: LinearFormat, input_amax: float | None = None,
    ) -> QuantLinear:
        target = linear.lora_target() if isinstance(linear, LoraCapable) else None
        return cls(linear.weight, linear.bias, fmt, input_amax=input_amax, lora_target=target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the kernels index rows as ``row * K``
        rows = x.reshape(-1, self.in_features).contiguous()
        out = self.fmt.linear(rows, *(getattr(self, name) for name in self.fmt.buffers))
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features)

    def lora_target(self) -> LoraTarget:
        return self._lora_target

    def lora_local_shape(self) -> tuple[int, int]:
        # the stored weight is packed; the delta is over the linear's own widths
        return self.in_features, self.out_features

    def quant_capability(self) -> QuantCapability:
        return QuantCapability(False, f"already {self.fmt.name}")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, format={self.fmt.name}"


class QuantCapability(NamedTuple):
    """Whether a linear kind can be replaced by a ``QuantLinear``; else ``reason`` says why not."""

    capable: bool
    reason: str | None = None


def quant_capability(module: nn.Module) -> QuantCapability | None:
    """How ``module`` takes a low-precision format, or None when it is no linear (a container, a norm, a conv).
    A linear kind declares it with a ``quant_capability()`` method; a plain ``nn.Linear`` (or a subclass that
    keeps its forward) is capable."""
    kind = type(module)
    declared = getattr(module, "quant_capability", None)
    if declared is not None:
        return declared()
    if not isinstance(module, nn.Linear):
        return None
    if "forward" in module.__dict__:
        return QuantCapability(False, f"{kind.__name__} has its forward replaced on the instance")
    if kind.forward is not nn.Linear.forward:
        return QuantCapability(False, f"{kind.__name__} subclasses nn.Linear with a forward of its own")
    return QuantCapability(True)


_RANGE = re.compile(r"(\d+)-(\d+)")


def matches(name: str, pattern: str) -> bool:
    """Whether a dotted module name matches ``pattern`` segment by segment: each segment is an ``fnmatch`` pattern
    for one name segment (``*`` never spans a dot) or an inclusive index range, so ``single_blocks.0-3.*`` names
    the first four single blocks' linears."""
    parts, pats = name.split("."), pattern.split(".")
    if len(parts) != len(pats):
        return False
    for part, pat in zip(parts, pats, strict=True):
        span = _RANGE.fullmatch(pat)
        if span is not None:
            if not part.isdigit() or not int(span.group(1)) <= int(part) <= int(span.group(2)):
                return False
        elif not fnmatch.fnmatchcase(part, pat):
            return False
    return True


@dataclass(frozen=True)
class QuantizableLinears:
    """A model's declaration over its linears' names (``matches`` patterns): the ones that may run low precision
    (``candidates``) and the ones that stay bf16 (``keep_bf16``). Every linear has to be in one or the other, so a
    new layer can't change dtype silently; ``keep_bf16`` wins where both match.

    ``input_amax`` maps patterns to the largest input magnitude measured on those linears, for a format whose
    activation scale has a range to place (NVFP4's). The first matching pattern counts."""

    candidates: tuple[str, ...]
    keep_bf16: tuple[str, ...]
    input_amax: Mapping[str, float] = field(default_factory=dict)

    def selected(self, name: str, keep: tuple[str, ...] = ()) -> bool:
        if any(matches(name, pattern) for pattern in (*self.keep_bf16, *keep)):
            return False
        if any(matches(name, pattern) for pattern in self.candidates):
            return True
        raise ValueError(f"linear {name!r} is neither a low-precision candidate nor kept bf16")

    def amax_of(self, name: str) -> float | None:
        return next((amax for pattern, amax in self.input_amax.items() if matches(name, pattern)), None)


class LinearSelection(NamedTuple):
    """What ``select_linears`` picked: the format, and the names to quantize with each one's declared input amax;
    empty when nothing will be."""

    fmt: LinearFormat | None
    names: tuple[str, ...]
    input_amax: tuple[float | None, ...] = ()


def select_linears(
    module: nn.Module, linear_dtype: str, declaration: QuantizableLinears, keep: Sequence[str] = (),
    what: str = "transformer",
) -> LinearSelection:
    """The linears of ``module`` (weights loaded) that ``linear_dtype`` would replace, checked: a keep pattern that
    names none of the candidates, or a selected linear whose kind or widths the format can't take, raises before
    anything is replaced. ``bf16``, or a GPU without the format's GEMM, selects nothing."""
    check_linear_dtype(linear_dtype)
    keep = check_linear_keep(keep)
    linears: dict[str, nn.Module] = {}
    for name, child in module.named_modules():
        if quant_capability(child) is not None:
            linears[name] = child
    candidates = [name for name in linears if declaration.selected(name)]
    for pattern in keep:
        kept = [name for name in candidates if matches(name, pattern)]
        if not kept:
            raise ValueError(f"{what}: linear_keep_bf16 entry {pattern!r} names none of its candidate linears")
        logger.info("%s: linear_keep_bf16 %r keeps %d linears bf16: %s", what, pattern, len(kept), ", ".join(kept))
    if linear_dtype == BF16:
        return LinearSelection(None, ())
    fmt = linear_format(linear_dtype)
    names = tuple(name for name in candidates if declaration.selected(name, keep))
    if not names:
        logger.info("%s: linear_dtype %s selects no linear; staying bf16", what, linear_dtype)
        return LinearSelection(fmt, ())
    reason = fmt.unavailable_reason(linears[names[0]].weight.device)
    if reason is not None:
        logger.warning("%s: linear_dtype %s requested, but %s; its linears stay bf16", what, linear_dtype, reason)
        return LinearSelection(fmt, ())
    for name in names:
        capability = quant_capability(linears[name])
        if not capability.capable:
            raise ValueError(f"{what}: {name!r} can't run {linear_dtype}: {capability.reason}")
        out_features, in_features = linears[name].weight.shape
        try:
            fmt.check(out_features, in_features, linears[name].weight.dtype)
        except ValueError as error:
            raise ValueError(f"{what}: {name!r}: {error}") from None
    return LinearSelection(fmt, names, tuple(declaration.amax_of(name) for name in names))


def quantize_linears(module: nn.Module, selection: LinearSelection) -> list[str]:
    """Replace ``selection``'s linears with ``QuantLinear``s, one at a time, so the bf16 weight of each goes before
    the next is quantized. Returns the names replaced."""
    for name, amax in zip(selection.names, selection.input_amax, strict=True):
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name)
        setattr(parent, leaf, QuantLinear.from_linear(getattr(parent, leaf), selection.fmt, amax))
    if selection.names:
        logger.info("linear_dtype %s: %d linears", selection.fmt.name, len(selection.names))
    return list(selection.names)


def apply_linear_dtype(
    module: nn.Module, linear_dtype: str, declaration: QuantizableLinears, keep: Sequence[str] = (),
    what: str = "transformer",
) -> list[str]:
    """``select_linears`` then ``quantize_linears``: serve ``module``'s declared linears in ``linear_dtype``.
    Returns the names replaced, empty when nothing was."""
    return quantize_linears(module, select_linears(module, linear_dtype, declaration, keep, what))
