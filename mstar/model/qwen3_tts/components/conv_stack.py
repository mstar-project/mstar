"""Geometry of the 12 Hz codec decoder's conv stack (everything after its transformer).

Every module in the stack is causal: convolutions pad on the left only and the
transposed convolutions trim on the right. So each output sample depends on a
bounded span of earlier conv-stack inputs (one per codec frame), and the audio
of a window's new frames needs only that many frames of context, however much
context the transformer before it needed. ``causal_receptive_field_frames``
derives that span from the modules themselves rather than from constants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.platforms import current_platform


def conv_stack(decoder: nn.Module) -> list[nn.Module]:
    """The decoder's post-transformer modules, in execution order."""
    return [block for blocks in decoder.upsample for block in blocks] + list(decoder.decoder)


def _earliest_input(module: nn.Module, out_index: int) -> int:
    """First input position that output position ``out_index`` of ``module`` depends on.

    Raises on any module whose dependency span is not modelled here, so a
    change to the decoder fails loudly instead of trimming too little context.
    """
    name = type(module).__name__
    if name in ("SnakeBeta", "FusedSnakeBeta"):   # pointwise
        return out_index
    if name.endswith("CausalConvNet"):
        # left-padded by ``padding``; output o reads padded inputs o*stride .. o*stride + kernel - 1
        return out_index * module.stride - module.padding
    if name.endswith("CausalTransConvNet"):
        conv = module.conv
        if (module.left_pad, conv.padding[0], conv.output_padding[0], conv.dilation[0]) != (0, 0, 0, 1):
            raise TypeError(f"unsupported transposed-conv geometry in {name}")
        kernel, stride = conv.kernel_size[0], conv.stride[0]
        # input i writes outputs i*stride .. i*stride + kernel - 1 (the right trim drops none of o's)
        return -((kernel - 1 - out_index) // stride)   # ceil((o - kernel + 1) / stride)
    if name.endswith("ConvNeXtBlock"):   # residual + depthwise conv; the rest is per position
        return _earliest_input(module.dwconv, out_index)
    if name.endswith("DecoderResidualUnit"):   # residual + conv1 -> act -> conv2 (kernel 1)
        return _earliest_input(module.conv1, _earliest_input(module.conv2, out_index))
    if name.endswith("DecoderDecoderBlock"):
        for block in reversed(module.block):
            out_index = _earliest_input(block, out_index)
        return out_index
    raise TypeError(f"no causal receptive-field rule for {name}")


def causal_receptive_field_frames(decoder: nn.Module) -> int:
    """Frames of context the conv stack needs for a frame's audio to equal a whole decode.

    Walks the stack backwards from the frame's first output sample (the
    earliest dependency only moves forward with the sample). Exact for the
    modules modelled in ``_earliest_input``.
    """
    upsample = 1
    for blocks in decoder.upsample:
        upsample *= blocks[0].conv.stride[0]
    for block in decoder.decoder:
        if type(block).__name__.endswith("DecoderDecoderBlock"):
            upsample *= block.block[1].conv.stride[0]
    frame = 10_000   # far from the start, so no left boundary is involved
    index = frame * upsample
    for module in reversed(conv_stack(decoder)):
        index = _earliest_input(module, index)
    return frame - index


def channels_last_applies(device: torch.device) -> bool:
    """Whether ``ChannelsLastConvStack`` runs on this device. cuDNN picks its
    kernels per layout and GPU, and these picks were measured on Blackwell;
    other GPUs keep the reference stack until someone measures them."""
    return device.type == "cuda" and current_platform.has_cuda_capability(10, 0, device)


class _CausalConv(nn.Module):
    """A ``CausalConvNet`` as a 2D conv over ``[B, C, 1, T]``."""

    def __init__(self, module: nn.Module):
        super().__init__()
        conv = module.conv
        self.padding = module.padding
        self.extra_padding = module._get_extra_padding_for_conv1d
        self.stride = (1, conv.stride[0])
        self.dilation = (1, conv.dilation[0])
        self.groups = conv.groups
        # cuDNN has no fast bf16 kernel for a dilated conv: milliseconds where
        # fp32 takes hundreds of microseconds, 100 ms at some lengths.
        dtype = conv.weight.dtype
        if conv.dilation[0] > 1 and dtype == torch.bfloat16:
            dtype = torch.float32
        self.register_buffer("weight", _channels_last_weight(conv, dtype), persistent=False)
        self.register_buffer("bias", conv.bias.detach().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = F.pad(x, (self.padding, self.extra_padding(x)))
        out = F.conv2d(x.to(self.weight.dtype), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        return out.to(dtype)


class _CausalTransConv(nn.Module):
    """A ``CausalTransConvNet`` as a 2D transposed conv over ``[B, C, 1, T]``."""

    def __init__(self, module: nn.Module):
        super().__init__()
        conv = module.conv
        self.right_pad = module.right_pad
        self.stride = (1, conv.stride[0])
        self.register_buffer("weight", _channels_last_weight(conv, conv.weight.dtype), persistent=False)
        self.register_buffer("bias", conv.bias.detach(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv_transpose2d(x, self.weight, self.bias, self.stride)
        return out[..., : out.shape[-1] - self.right_pad].contiguous(memory_format=torch.channels_last)


class _ConvNeXtBlock(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.dwconv = _CausalConv(module.dwconv)
        self.norm, self.pwconv1, self.act, self.pwconv2 = module.norm, module.pwconv1, module.act, module.pwconv2
        self.gamma = module.gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, 1, T] in channels-last memory is already [B, T, C] in place
        h = self.dwconv(x).squeeze(2).permute(0, 2, 1)
        h = self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(h))))
        return x + h.permute(0, 2, 1).unsqueeze(2)


class _ResidualUnit(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.act1, self.act2 = module.act1, module.act2
        self.conv1, self.conv2 = _CausalConv(module.conv1), _CausalConv(module.conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act2(self.conv1(self.act1(x)))) + x


def _channels_last_weight(conv: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    return conv.weight.detach().to(dtype).unsqueeze(2).contiguous(memory_format=torch.channels_last)


def _channels_last_module(module: nn.Module) -> nn.Module:
    name = type(module).__name__
    if name == "FusedSnakeBeta":
        return module
    if name.endswith("CausalConvNet"):
        return _CausalConv(module)
    if name.endswith("CausalTransConvNet"):
        return _CausalTransConv(module)
    if name.endswith("ConvNeXtBlock"):
        return _ConvNeXtBlock(module)
    if name.endswith("DecoderResidualUnit"):
        return _ResidualUnit(module)
    if name.endswith("DecoderDecoderBlock"):
        return nn.Sequential(*map(_channels_last_module, module.block))
    raise TypeError(f"no channels-last form of {name}")


class ChannelsLastConvStack(nn.Module):
    """``conv_stack`` over channels-last ``[B, C, 1, T]`` tensors.

    cuDNN's fast convs are NHWC, so on the reference's ``[B, C, T]`` it transposes
    around most of them. Holding the whole stack in that layout drops the
    transposes and lets cuDNN pick its NHWC kernels: a fifth to a third of the
    captured decode on a B300, within rounding of the reference stack (over 80
    dB) in fp32. The weights are copies in the 2D layout; the decoder keeps its
    own for the reference path and its module walk.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.blocks = nn.Sequential(*map(_channels_last_module, conv_stack(decoder)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` to the stack's ``[B, 1, samples]``."""
        return self.blocks(hidden.unsqueeze(2).contiguous(memory_format=torch.channels_last)).squeeze(2)
