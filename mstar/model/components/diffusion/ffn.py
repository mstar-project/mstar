"""A DiT up-projection with its bias and tanh-GELU in the GEMM's epilogue.

Run apart, the up-projection and its GELU are two kernels: the GEMM writes the
``[tokens, ffn_dim]`` activation, and a pointwise pass reads it back, which at a
DiT's FFN width is the largest elementwise pass of the block and bound by the
tanh's arithmetic. cuBLASLt's bias+GELU epilogue (``aten._addmm_activation``,
tanh approximation on CUDA) applies it to the fp32 accumulator before the one
rounding to bf16, so the separate pass goes away and the result is no further
from the exact GELU than eager's (closer, as it skips a rounding), though not
bit-exact with it.

* :func:`addmm_gelu_tanh`: the epilogue GEMM itself, for a model that takes it
  eagerly (Qwen-Image's DiTs, whose quality gate is the fp32 yardstick).
* :func:`gelu_tanh_ffn`: a whole ``MLP`` feed-forward that swaps it in only
  when compiled; eager calls run the module as written, the form the reference
  ports are held bit-exact in, and so does a ``linear_in`` bound to a LoRA
  delta, which its forward adds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.components.lora.capable import LoraBound
from mstar.model.components.mlp import MLP


def addmm_gelu_tanh(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``gelu(x @ weight.T + bias, approximate="tanh")`` in one GEMM, ``x [..., K]`` with ``weight [N, K]``.

    On CUDA only: the CPU kernel of ``_addmm_activation`` applies the erf GELU, so elsewhere this is
    ``F.gelu(F.linear(...), approximate="tanh")``."""
    if not x.is_cuda:
        return F.gelu(F.linear(x, weight, bias), approximate="tanh")
    out = torch._addmm_activation(bias, x.flatten(0, -2), weight.t(), use_gelu=True)
    return out.unflatten(0, x.shape[:-1])


def gelu_tanh_ffn(x: torch.Tensor, mlp: MLP) -> torch.Tensor:
    """``mlp(x)`` for an ``MLP`` built with ``activation="gelu_tanh"`` and a
    biased ``linear_in``."""
    if x.is_cuda and torch.compiler.is_compiling() and not isinstance(mlp.linear_in, LoraBound):
        lin = mlp.linear_in
        return mlp.linear_out(addmm_gelu_tanh(x, lin.weight, lin.bias))
    return mlp(x)
