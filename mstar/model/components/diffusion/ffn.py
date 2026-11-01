"""A DiT feed-forward ``linear_out(gelu_tanh(linear_in(x)))`` with the GELU in
the first GEMM's epilogue when compiled.

Compiled, the up-projection and its GELU are two kernels: the GEMM writes the
``[tokens, ffn_dim]`` activation, and a pointwise pass reads it back, which at a
DiT's FFN width is the largest elementwise pass of the block and bound by the
tanh's arithmetic. cuBLASLt's bias+GELU epilogue (``aten._addmm_activation``,
tanh approximation) applies it to the fp32 accumulator before the one rounding
to bf16, so the separate pass goes away and the result is no further from the
exact GELU than eager's (closer, as it skips a rounding). Eager calls run the
module as written, the form the reference ports are held bit-exact in.
"""

from __future__ import annotations

import torch

from mstar.model.components.mlp import MLP


def gelu_tanh_ffn(x: torch.Tensor, mlp: MLP) -> torch.Tensor:
    """``mlp(x)`` for an ``MLP`` built with ``activation="gelu_tanh"`` and a
    biased ``linear_in``."""
    if x.is_cuda and torch.compiler.is_compiling():
        lin = mlp.linear_in
        hidden = torch._addmm_activation(lin.bias, x.flatten(0, -2), lin.weight.t(), use_gelu=True)
        return mlp.linear_out(hidden.unflatten(0, x.shape[:-1]))
    return mlp(x)
