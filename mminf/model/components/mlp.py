"""MLP and SwiGLU-style GatedMLP.

Used for transformer FFN blocks and for small projection MLPs (timestep
embedders, Talker resize projections, etc.).

``GatedMLP``'s gate + up projections start as separate ``nn.Linear``s
that match the HF checkpoint layout. After load, calling
``consolidate_gate_up_weight()`` concatenates them into a single
``gate_up_proj_weight`` buffer (one fused GEMM instead of two) and nulls
out the originals. The forward branches on whether consolidation has
happened.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from mminf.model.components.linear import FusedColumnLinear


def _resolve_activation(activation: str | Callable) -> Callable:
    """Resolve an activation name to a callable. Accepts the canonical
    HF names (``silu``, ``gelu``, ``gelu_tanh``, ``relu``) or a callable.
    """
    if callable(activation):
        return activation
    if activation == "silu":
        return F.silu
    if activation == "gelu":
        return F.gelu
    if activation == "gelu_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if activation == "relu":
        return F.relu
    raise ValueError(f"Unknown activation: {activation!r}")


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP: ``down(act(gate(x)) * up(x))``.

    Args:
        hidden_size: input/output feature dim.
        intermediate_size: gate/up output and down input feature dim.
        activation: activation applied to the gate path. Either an HF
            string (``silu`` / ``gelu`` / ``gelu_tanh``) or a callable.
        bias: whether the linears have a bias term.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str | Callable = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act = _resolve_activation(activation)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def consolidate_gate_up_weight(self) -> None:
        """Fuse ``gate_proj`` and ``up_proj`` weights into a single
        ``gate_up_proj_weight`` buffer and null out the originals.
        Idempotent; safe to call multiple times.
        """
        if self.gate_proj is None:
            return
        if self.gate_proj.bias is not None:
            raise NotImplementedError(
                "consolidate_gate_up_weight does not yet handle biases."
            )
        gate_up = torch.cat(
            (self.gate_proj.weight, self.up_proj.weight), dim=0,
        ).contiguous()
        self.register_buffer("gate_up_proj_weight", gate_up, persistent=False)
        self.gate_proj = None
        self.up_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is not None:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        gate_up = F.linear(x, self.gate_up_proj_weight)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act(gate) * up)


class FusedGatedMLP(nn.Module):
    """SwiGLU-style gated MLP with the gate + up projections fused into a
    single ``FusedColumnLinear`` (one GEMM instead of two): ``down(act(gate)
    * up)``.
    Unlike ``GatedMLP`` (separate Linears + post-load
    ``consolidate_gate_up_weight``), this fuses from construction and loads
    the separate ``gate_proj`` / ``up_proj`` checkpoint tensors straight
    into the fused parameter via the loader's stacked-param rules (gate is
    shard ``0``, up is shard ``1``). Use for models that fuse but don't
    need TP.
    Args:
        hidden_size: input/output feature dim.
        intermediate_size: gate/up output and down input feature dim.
        activation: activation applied to the gate path. Either an HF
            string (``silu`` / ``gelu`` / ``gelu_tanh``) or a callable.
        bias: whether the linears have a bias term.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str | Callable = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act = _resolve_activation(activation)
        self.gate_up_proj = FusedColumnLinear(
            hidden_size,
            {0: intermediate_size, 1: intermediate_size},
            bias=bias,
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act(gate) * up)



class MLP(nn.Module):
    """Plain two-layer MLP: ``out(act(in(x)))``.

    Used for small projection MLPs that aren't gated (Talker resize
    projection, bagel timestep embedder MLP, etc.). For SwiGLU-style
    transformer FFNs, use ``GatedMLP``.

    Args:
        input_size: input feature dim.
        intermediate_size: hidden feature dim.
        output_size: output feature dim. Defaults to ``input_size`` if
            None.
        activation: activation between the two linears.
        bias: whether the linears have a bias term.
    """

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int | None = None,
        activation: str | Callable = "silu",
        bias: bool = True,
    ):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.act = _resolve_activation(activation)
        self.linear_in = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_out = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.act(self.linear_in(x)))
