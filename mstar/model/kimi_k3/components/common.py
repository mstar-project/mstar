"""Small shared pieces: the exact ``KimiRMSNorm``, SiTU activation, replicated linear
with a plain copy loader, and dim-0 sharding loaders for per-head parameters."""
from __future__ import annotations

import logging
import os
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.kimi_k3.components.rmsnorm_kernel import kimi_rmsnorm_triton, rmsnorm_supported
from mstar.model.kimi_k3.reference.situ import situ_and_mul


class KimiRMSNorm(nn.Module):
    """``weight * (x / rms(x)).to(dtype)`` with fp32 statistics — the checkpoint's own
    norm, kept bit-compatible with the HF reference (the weight multiply happens in the
    activation dtype)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if rmsnorm_supported(x) and self.weight.dtype == x.dtype:
            # one Triton kernel with the same arithmetic (bit-identical output)
            return kimi_rmsnorm_triton(x, self.weight, self.variance_epsilon)
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * xf.to(x.dtype)


class SiTUAndMul(nn.Module):
    def __init__(self, beta: float = 4.0, linear_beta: float | None = 25.0):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        if gate_up.is_cuda and gate_up.dtype in (torch.bfloat16, torch.float16):
            # one Triton kernel (fp32 math, like the reference) instead of six elementwise ones
            from mstar.utils.fused_moe.mxfp4 import situ_and_mul_triton

            x = gate_up.reshape(-1, gate_up.shape[-1])  # a 2-D view keeps a merged projection's row stride
            if x.stride(-1) != 1:
                x = x.contiguous()
            out = torch.empty(x.shape[0], x.shape[1] // 2, dtype=x.dtype, device=x.device)
            situ_and_mul_triton(x, out, self.beta, self.linear_beta)
            return out.view(*gate_up.shape[:-1], x.shape[1] // 2)
        return situ_and_mul(gate_up, self.beta, self.linear_beta)


class ReplicatedLinear(nn.Module):
    """A full-width linear every rank holds (small replicated projections; the ones that
    share an input now live as replicated segments of a ``MergedParallelLinear``)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def shard_dim0_loader(tp_rank: int, tp_size: int, param: nn.Parameter, loaded: torch.Tensor,
                      loaded_shard_id=None) -> None:
    """Copy this rank's contiguous slice along dim 0 (heads / channels)."""
    assert loaded_shard_id is None
    full = param.data.shape[0] * tp_size
    n = loaded.shape[0]
    if n > full:
        # the released Kimi K3 checkpoints store A_log with head_dim (128) entries for 96
        # heads; the reference module allocates num_heads and vLLM narrows per rank, i.e.
        # both use the first num_heads entries
        _warn_once(f"{tuple(loaded.shape)} -> first {full} entries (parameter is {full} across TP)")
        loaded = loaded.narrow(0, 0, full)
        n = full
    assert n % tp_size == 0, (n, tp_size)
    per = n // tp_size
    src = loaded.narrow(0, tp_rank * per, per)
    assert param.data.shape == src.shape, (tuple(param.data.shape), tuple(src.shape))
    param.data.copy_(src)


_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        logging.getLogger(__name__).warning("dim-0 shard loader: oversized checkpoint tensor %s", msg)


def attach_dim0_loader(param: nn.Parameter, tp_rank: int, tp_size: int) -> None:
    param.weight_loader = partial(shard_dim0_loader, tp_rank, tp_size)


def restore_kept_dtypes(module: nn.Module) -> None:
    """Undo a blanket ``module.to(dtype)`` for parameters tagged ``_keep_dtype`` (fp32
    gates/biases, uint8 packed weights): the engine casts every node to the autocast
    dtype after ``get_submodule``. Call from ``_apply`` overrides."""
    for prm in module.parameters(recurse=False):
        keep = getattr(prm, "_keep_dtype", None)
        if keep is not None and prm.dtype != keep:
            prm.data = prm.data.to(keep)


def replicated_loader(param: nn.Parameter, loaded: torch.Tensor, loaded_shard_id=None) -> None:
    assert loaded_shard_id is None
    assert param.data.shape == loaded.shape, (tuple(param.data.shape), tuple(loaded.shape))
    param.data.copy_(loaded)


def fused_decode_kernels() -> bool:
    """``MSTAR_K3_FUSED_DECODE=0`` falls back to the torch/fla decode paths the fused kernels replaced
    (the KDA recurrence and gated norm reading the projection slices in place, the MLA output),
    for a served A/B on one tree."""
    return os.environ.get("MSTAR_K3_FUSED_DECODE", "1") != "0"
