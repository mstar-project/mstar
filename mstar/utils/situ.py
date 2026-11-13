"""SiTU-GLU, Kimi K3's gated activation: ``beta * tanh(g / beta) * sigmoid(g) * lb * tanh(u / lb)``.

A gated MLP (dense, shared expert or routed expert) applies it to its gate ``g`` and up ``u``
halves; with ``linear_beta`` None the up half passes through. The math runs in fp32, as the
reference (``SituAndMul`` in the checkpoint's ``modeling_kimi_linear.py``) does, and the result
is cast back to the input dtype. ``SiTU`` (or its ``spec`` string, for the custom ops) is also the
activation the fused MoE kernels take (``act_and_mul_triton``), so the Triton paths and this one
compute the same function.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SiTU:
    beta: float = 1.0
    linear_beta: float | None = None

    def __post_init__(self) -> None:
        if not self.beta > 0 or (self.linear_beta is not None and not self.linear_beta > 0):
            raise ValueError(f"SiTU needs positive betas, got {self}")

    def __call__(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return situ_and_mul(gate, up, self.beta, self.linear_beta)

    @property
    def spec(self) -> str:
        """The activation as a string, for the fused MoE custom ops whose schema takes one."""
        return f"situ:{self.beta!r}:{'none' if self.linear_beta is None else repr(self.linear_beta)}"

    @classmethod
    def parse(cls, spec: str) -> SiTU | None:
        """The ``SiTU`` a ``spec`` string names, None for any other activation name."""
        if not spec.startswith("situ:"):
            return None
        _, beta, linear = spec.split(":")
        return cls(float(beta), None if linear == "none" else float(linear))


def activation_spec(activation) -> str:
    """A gated MLP's activation as the fused kernels name it: ``silu`` / ``gelu``, or a ``SiTU``'s spec."""
    return activation.spec if isinstance(activation, SiTU) else activation


def situ_and_mul(
    gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None,
) -> torch.Tensor:
    """``(..., D)`` gate and up halves -> ``(..., D)`` in ``gate``'s dtype."""
    g = gate.float()
    u = up.float()
    act = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (act * u).to(gate.dtype)
