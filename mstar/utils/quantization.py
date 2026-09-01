"""Quantization scheme descriptors shared by the MoE kernels and model code.

These are pure data — no ``nn.Module``, no kernel imports — so both
:mod:`mstar.utils.fused_moe` (which consumes them at runtime) and
:mod:`mstar.model.components.quantization` (which produces them) can depend on
this module without a cycle. ``model -> utils`` is the established direction;
defining them model-side would invert it.

A quantized GEMM entry point takes one :class:`QuantizationData` instead of a
widening list of optional per-scheme kwargs, and branches on
:attr:`QuantizationData.quant_type` rather than sniffing arguments for ``None``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import torch


class QuantizationType(Enum):
    """Weight/activation precision scheme of a quantized GEMM.

    Members are added alongside a kernel path that implements them — a member
    with no dispatch branch is a promise the kernels cannot keep.
    """

    W4A16 = "w4a16"


class QuantizationData(ABC):
    """Per-call companion data for a quantized GEMM (scales, zero points, layout).

    One subclass per :class:`QuantizationType`. Kernel entry points branch on
    :attr:`quant_type` and read the subclass's fields, so supporting a new scheme
    is a subclass plus one dispatch branch — not another six optional kwargs whose
    legal combinations exist only in a maintainer's head.

    Distinct from a *checkpoint* descriptor such as
    ``CompressedTensorsQuantConfig``: that says how the weights were written to
    disk (format, ignore list, bit width); this carries the live tensors a
    specific kernel call needs. The checkpoint descriptor builds one of these.
    """

    @property
    @abstractmethod
    def quant_type(self) -> QuantizationType:
        """Which scheme this data describes. Kernels dispatch on it."""


@dataclass(frozen=True)
class W4A16Data(QuantizationData):
    """Group-wise INT4 weights with bf16/fp16 activations (compressed-tensors layout).

    Weights reach the kernel as low-order-first int32 containers holding
    :attr:`pack_factor` nibbles each; ``w1_scale`` / ``w2_scale`` are
    ``(num_experts, N, K // group_size)``. ``w1_zp`` / ``w2_zp`` are ``None`` for
    symmetric quantization (Kimi-K2.7's case) — that ``None`` *is* the
    symmetric/asymmetric discriminant, read off one object rather than re-derived
    at each call depth.

    ``frozen=True`` blocks rebinding a field, not mutation of the tensors a field
    points at; it marks the object as a description rather than a workspace.
    """

    # The Triton W4A16 kernel extracts nibbles with ``& 0xF`` and subtracts the
    # 4-bit offset-binary bias, so the width is fixed by the class, not a knob.
    NUM_BITS: ClassVar[int] = 4

    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    group_size: int
    w1_zp: torch.Tensor | None = None
    w2_zp: torch.Tensor | None = None

    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.W4A16

    @property
    def pack_factor(self) -> int:
        """INT4 values per int32 container (8). Derived, so it cannot disagree
        with the bit width the kernel actually implements."""
        return 32 // self.NUM_BITS

    @property
    def symmetric(self) -> bool:
        return self.w1_zp is None and self.w2_zp is None
