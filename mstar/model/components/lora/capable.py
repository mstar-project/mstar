"""Linears that take a per-request LoRA delta.

A LoRA-capable module (``LoraCapable``) describes its own target (``lora_target``),
with the checkpoint's (unsharded) shapes and how the base splits them across
tensor-parallel ranks, and its rank-local shape (``lora_local_shape``), which binding
checks against the pool's shard of that target. Its forward knows nothing of LoRA.

Binding a delta swaps the instance to its family's twin class (``lora_capability``), a
``LoraBound`` mixin over the same forward that adds the delta in place before anything
else reads the output: before a column-parallel gather, before a row-parallel
all-reduce (the delta is a partial sum like the base output), after a vocab-parallel
embedding's all-reduce (its delta is whole on every rank). An unbound module keeps its
class, so a deployment without LoRA traces and runs exactly what it did before, with no
attribute read or guard on LoRA. Plain ``nn.Linear``, patch convolutions (a ``ConvNd``
whose kernel is its stride) and subclasses of either that keep their base's forward
take the same swap, to ``LoraLinear`` / ``LoraPatchConvNd`` or a twin made for them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import torch
from torch import nn

from mstar.engine.resources.lora.config import LoraTarget

if TYPE_CHECKING:
    from mstar.engine.resources.lora.pool import LoraLayerDelta


def apply_lora(delta: "LoraLayerDelta | None", x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``y`` with the step's per-row delta of ``x`` added, or ``y`` itself when nothing is
    bound or no row of the step names an adapter (a base-only step, which launches no
    kernel). The one place a capable module calls LoRA, and callers use the return value,
    so the op form (in place on ``y`` today; a functional ``y + delta`` lets Inductor
    schedule fusions around it) changes here alone. Only bound (swapped) modules call it."""
    return y if delta is None else delta(x, y)


class LoraCapable:
    """Mixin for a module that describes its LoRA target (see the module docstring). Its
    own forward knows nothing of LoRA: binding swaps the instance to its twin class,
    which adds the bound delta (``lora_twin``, a ``LoraBound`` mixin over the family's
    forward; ``lora_capability`` makes the twin)."""

    # the forward mixin a bound instance of this family runs, set by the family's base
    lora_twin: ClassVar[type | None] = None

    def lora_target(self) -> LoraTarget:
        """This module's target with the checkpoint's shapes. Row facts (``per_segment``,
        ``token_fold``) and ``layers`` belong to the model's declaration, not here."""
        raise NotImplementedError

    def lora_local_shape(self) -> tuple[int, int]:
        """``(in, out)`` of the delta on this rank: the input columns the module reads
        and the output columns it writes before any collective, which its weight
        (``[out, in, *kernel]``) holds."""
        weight = self.weight
        return math.prod(weight.shape[1:]), weight.shape[0]


class LoraBound:
    """The delta slots of a bound (swapped) module: a forward mixin derives from this."""

    # set at bind (``bind_module_lora``); class-level None until then
    lora_delta: "LoraLayerDelta | None" = None
    # a delta bound for the node to add itself (``module_lora_spec(deferred=...)``): the
    # module's own forward never adds it, since the node also calls the module outside
    # its planned step (e.g. embedding a prompt in ``prepare_inputs``)
    lora_deferred: "LoraLayerDelta | None" = None


class LoraOutputAdd(LoraBound):
    """A bound module whose delta adds to its whole output, after the base forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_lora(self.lora_delta, x, super().forward(x))


class _LoraLinearMixin(LoraOutputAdd, LoraCapable):
    """An ``nn.Linear`` (or a subclass that keeps its forward) plus its bound delta.
    Whole on every rank (``parallel="none"``): a plain linear is not split by a
    tensor-parallel group."""

    def lora_target(self) -> LoraTarget:
        return LoraTarget(in_features=self.in_features, out_slices=(self.out_features,), parallel="none")


class _LoraPatchConvMixin(LoraBound, LoraCapable):
    """A convolution whose kernel is its stride: a linear over each flattened patch, as
    PEFT's conv LoRA (``A [r, c, *kernel]``, ``B [out, r, 1, ...]``) is over it. Each
    output position is a row, in the order ``flatten(2).transpose(1, 2)`` makes tokens of
    them (one per input when a patch embedding is fed pre-cut patches)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.lora_delta is None:
            return y
        n, out, *grid = y.shape
        kernel = self.kernel_size
        # the patches the conv read ([N, C, G1, K1, G2, K2, ...], less any remainder it
        # dropped), each flattened as the weight is: [N * G1 * G2 ..., C * K1 * K2 ...]
        x = x[(..., *(slice(0, g * k) for g, k in zip(grid, kernel, strict=True)))]
        x = x.reshape(n, self.in_channels, *(d for g, k in zip(grid, kernel, strict=True) for d in (g, k)))
        dims = range(len(grid))
        x = x.permute(0, *(2 + 2 * i for i in dims), 1, *(3 + 2 * i for i in dims))
        rows = y.movedim(1, -1).reshape(-1, out)
        rows = apply_lora(self.lora_delta, x.reshape(rows.shape[0], -1), rows)
        return rows.view(n, *grid, out).movedim(-1, 1).contiguous()

    def lora_target(self) -> LoraTarget:
        return LoraTarget(
            in_features=self.in_channels * math.prod(self.kernel_size),
            out_slices=(self.out_channels,), parallel="none",
        )


class LoraLinear(_LoraLinearMixin, nn.Linear):
    pass


class LoraPatchConv1d(_LoraPatchConvMixin, nn.Conv1d):
    pass


class LoraPatchConv2d(_LoraPatchConvMixin, nn.Conv2d):
    pass


class LoraPatchConv3d(_LoraPatchConvMixin, nn.Conv3d):
    pass


_PLAIN = {nn.Linear: LoraLinear, nn.Conv1d: LoraPatchConv1d, nn.Conv2d: LoraPatchConv2d, nn.Conv3d: LoraPatchConv3d}
# a subclass that keeps its base's forward -> its capable twin, made once per class
_TWINS: dict[type, type] = {}


def _twin(kind: type, mixin: type) -> type:
    found = _PLAIN.get(kind) or _TWINS.get(kind)
    if found is None:
        found = _TWINS[kind] = type(f"Lora{kind.__name__}", (mixin, kind), {"__module__": kind.__module__})
    return found


class LoraCapability(NamedTuple):
    """How a module takes a LoRA delta: ``capable`` is the capable class it is, or is
    swapped to at bind; else ``reason`` says why it cannot."""

    capable: type | None
    reason: str | None = None


def _is_patch_conv(conv: nn.Module) -> bool:
    padding = conv.padding
    unpadded = padding == "valid" or (not isinstance(padding, str) and all(p == 0 for p in padding))
    return (
        tuple(conv.stride) == tuple(conv.kernel_size) and unpadded
        and all(d == 1 for d in conv.dilation) and conv.groups == 1
    )


_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def lora_capability(module: nn.Module) -> LoraCapability | None:
    """How ``module`` takes a LoRA delta (``LoraCapability``), or None for a module with
    no weights of its own to adapt (a container, a norm)."""
    kind = type(module)
    if "forward" in module.__dict__:
        return LoraCapability(None, f"{kind.__name__} has its forward replaced on the instance")
    if module._forward_hooks or module._forward_pre_hooks:
        return LoraCapability(
            None, f"{kind.__name__} has forward hooks, which would see (or change) what the delta adds to",
        )
    if isinstance(module, LoraBound):
        return LoraCapability(kind)
    if isinstance(module, LoraCapable):
        family = next(c for c in kind.__mro__ if "lora_twin" in c.__dict__)
        if kind.forward is not family.forward:
            return LoraCapability(None, f"{kind.__name__} subclasses {family.__name__} with a forward of its own")
        return LoraCapability(_twin(kind, family.lora_twin))
    if isinstance(module, nn.Linear):
        if kind.forward is not nn.Linear.forward:
            return LoraCapability(None, f"{kind.__name__} subclasses nn.Linear with a forward of its own")
        return LoraCapability(_twin(kind, _LoraLinearMixin))
    if isinstance(module, _CONVS):
        base = next(c for c in _CONVS if isinstance(module, c))
        if kind.forward is not base.forward:
            return LoraCapability(None, f"{kind.__name__} subclasses {base.__name__} with a forward of its own")
        if not _is_patch_conv(module):
            return LoraCapability(None, (
                f"{kind.__name__} with kernel {tuple(module.kernel_size)}, stride "
                f"{tuple(module.stride)}, groups {module.groups}: only a patch projection "
                "(kernel == stride, no padding, groups 1) is a linear per row"
            ))
        return LoraCapability(_twin(kind, _LoraPatchConvMixin))
    if isinstance(module, nn.Embedding):
        return LoraCapability(None, (
            "nn.Embedding: a lookup whose rows need not be the step's tokens (positions, "
            "codebooks); token embeddings take LoRA as VocabParallelEmbedding"
        ))
    if isinstance(module, nn.modules.conv._ConvNd):
        return LoraCapability(None, f"{kind.__name__}: not a convolution a delta can be added to per row")
    if any(p.dim() >= 2 for p in module.parameters(recurse=False)):
        return LoraCapability(None, f"{kind.__name__} holds weight matrices but is not a LoRA-capable linear")
    return None


def make_lora_capable(module: nn.Module) -> LoraBound:
    """``module`` as a capable module, swapping a plain linear / patch conv's class in
    place (nothing else about it changes). Raises for a module that cannot be."""
    found = lora_capability(module)
    if found is None or found.capable is None:
        raise ValueError(found.reason if found else f"{type(module).__name__} holds no weights to adapt")
    if type(module) is not found.capable:
        module.__class__ = found.capable
    return module
