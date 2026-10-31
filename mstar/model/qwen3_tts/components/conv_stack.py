"""Geometry of the 12 Hz codec decoder's conv stack (everything after its transformer).

Every module in the stack is causal: convolutions pad on the left only and the
transposed convolutions trim on the right. So each output sample depends on a
bounded span of earlier conv-stack inputs (one per codec frame), and the audio
of a window's new frames needs only that many frames of context, however much
context the transformer before it needed. ``causal_receptive_field_frames``
derives that span from the modules themselves rather than from constants.
"""

from __future__ import annotations

from torch import nn


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
