"""Streaming steps for incremental (stateful) audio codec decoders.

A causal codec decoder streams exactly when each chunk carries each causal
conv's last input samples and each overlapping transposed conv's last input
sample from the chunk before it. These run that step time-major
(``[B, T, C]``): causal convs as 2-D convs on channels-last tensors and
weights, so cuDNN's NHWC kernels need no layout transposes around them, and
transposed convs as one GEMM. Used by the Fish S2-Pro codec.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def tail(xs: torch.Tensor, valid: torch.Tensor, keep: int, dim: int) -> torch.Tensor:
    """Per row, the ``keep`` entries of ``xs`` along ``dim`` that end ``valid[row]``
    entries past its first ``keep`` (``xs`` is ``[keep of state | new]``)."""
    shape = [1] * xs.dim()
    shape[0], shape[dim] = -1, keep
    index = (valid.view(-1, *[1] * (xs.dim() - 1)) + torch.arange(keep, device=xs.device).view(shape))
    size = list(xs.shape)
    size[dim] = keep
    return xs.gather(dim, index.expand(size))


def tconv_gemm(x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
    """Every input sample's transposed-conv output, one GEMM: ``x [B, T, C_in]`` ->
    ``[B, T, kernel, C_out]`` (no bias, no overlap-add). A tensor-core GEMM where
    cuDNN runs these transposed convs on its fp32 dgrad engine."""
    weight = conv.weight   # [C_in, C_out, kernel]; the matrix below is a view if ``to_channels_last`` laid it out
    y = x @ weight.permute(0, 2, 1).reshape(weight.shape[0], -1)
    return y.unflatten(2, (weight.shape[2], weight.shape[1]))


def overlap_tconv(xs: torch.Tensor, conv: nn.Module) -> torch.Tensor:
    """A kernel ``2 * stride`` transposed conv over ``xs [B, 1 + T, C]`` (its
    carried input sample, then ``T`` new ones) -> ``[B, T * stride, C_out]``:
    input i writes outputs i*stride .. i*stride + 2*stride - 1, so each output
    block is its input's first half plus the previous input's second."""
    stride = conv.stride[0]
    y = tconv_gemm(xs, conv).unflatten(2, (2, stride))       # [B, 1 + T, 2, stride, C]
    y = y[:, 1:, 0] + y[:, :-1, 1]
    return y.flatten(1, 2) + conv.bias


def causal_conv(xs: torch.Tensor, conv: nn.Module) -> torch.Tensor:
    """A stride-1 causal conv over ``xs [B, keep + T, C]`` (its carried input
    samples, then ``T`` new ones) -> ``[B, T, C_out]``, on channels-last views."""
    x4 = xs.transpose(1, 2).unsqueeze(2)   # [B, C, 1, T], a channels-last view
    y = F.conv2d(x4, conv.weight.unsqueeze(2), conv.bias, dilation=(1, conv.dilation[0]), groups=conv.groups)
    return y.squeeze(2).transpose(1, 2)


def snake_time_major(x: torch.Tensor, alpha: torch.Tensor, inv_beta: torch.Tensor) -> torch.Tensor:
    """Snake ``x + inv_beta * sin(alpha * x)^2`` over a time-major ``x [B, T, C]`` (the
    channel-major one is ``components.snake.snake``), in fp32 or wider as the codecs' fused
    kernels compute it."""
    dtype = torch.promote_types(x.dtype, torch.float32)
    x32 = x.to(dtype)
    return (x32 + inv_beta.to(dtype) * torch.sin(x32 * alpha.to(dtype)).pow(2)).to(x.dtype)


def to_channels_last(conv: nn.Module) -> None:
    """Lay ``conv``'s weight out channels-last in place: ``weight.unsqueeze(2)``
    is then the NHWC 2-D weight, and the module still runs on the same storage."""
    w = conv.weight
    w.data = w.data.unsqueeze(2).contiguous(memory_format=torch.channels_last).squeeze(2)
