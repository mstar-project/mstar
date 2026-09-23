"""SiTU-GLU activation (spec E.4).

``situ_and_mul(x, beta, linear_beta)`` takes the fused ``[gate | up]`` projection and
returns ``beta * tanh(gate / beta) * sigmoid(gate) * (linear_beta * tanh(up / linear_beta))``,
computed in fp32 and cast back to the input dtype. With ``linear_beta=None`` the up branch
is used unchanged (the generic Kimi-Linear form).
"""
from __future__ import annotations

import torch


def situ_gate(gate: torch.Tensor, beta: float) -> torch.Tensor:
    gate = gate.float()
    return beta * torch.tanh(gate / beta) * torch.sigmoid(gate)


def situ_and_mul(
    x: torch.Tensor, beta: float = 4.0, linear_beta: float | None = 25.0,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:].float()
    a = situ_gate(gate, beta)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (a * up).to(x.dtype)
