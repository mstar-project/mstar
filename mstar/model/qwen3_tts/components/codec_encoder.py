"""The 12 Hz codec's Mimi encoder over clips zero-padded to one width.

Each row's codes equal its own unpadded encode, up to kernel choice, given its
sample count. Every Mimi module is causal (left-padded convs, a causal
sliding-window transformer, per-frame quantizers), so a frame never reads
samples after its own, with one exception: a strided conv's last step of a
clip reads up to a stride past the clip's end, which the unpadded encode fills
with the conv's own right padding (zeros, or the last step replicated). Here
those positions hold the encode of the padding instead, so before each strided
conv they are refilled with what the unpadded conv pads with.

The padding arithmetic is on Python ints rather than Mimi's tensor buffers
(each a device sync), so the encode captures into an accelerator graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiResnetBlock


class PaddedCodecEncoder(nn.Module):
    """``[B, width]`` float32 samples and ``[B]`` sample counts -> ``[B, frames, num_quantizers]`` codes."""

    def __init__(self, codec_encoder: nn.Module, num_quantizers: int):
        super().__init__()
        self.num_quantizers = num_quantizers
        encoder = codec_encoder.encoder
        # transformers 5.21+ nests the encoder half; earlier ones keep it on the model
        nested = hasattr(encoder, "encoder_seanet")
        # Held in a tuple, not as submodules: the owner registers the encoder, and a
        # second registration would put its weights in the state dict twice.
        self._parts = (
            codec_encoder,
            encoder.encoder_seanet if nested else encoder,
            encoder.encoder_transformer if nested else codec_encoder.encoder_transformer,
            encoder.downsample if nested else codec_encoder.downsample,
        )
        for module in codec_encoder.modules():
            if isinstance(module, MimiConv1d) and not module.causal:
                raise TypeError("the padded codec encode assumes Mimi's causal convs")
            if isinstance(module, MimiConv1d) and module.pad_mode not in ("constant", "replicate"):
                # any other mode would pad about the bucket's end, not the clip's
                raise TypeError(f"the padded codec encode does not reproduce pad_mode {module.pad_mode!r}")
        if not isinstance(self._parts[3], MimiConv1d):
            raise TypeError("the padded codec encode assumes Mimi's strided downsample conv")
        self.hop = 1
        for module in (*self.seanet.modules(), self.downsample):
            if isinstance(module, MimiConv1d):
                self.hop *= module.conv.stride[0]

    @property
    def codec_encoder(self) -> nn.Module:
        return self._parts[0]

    @property
    def seanet(self) -> nn.Module:
        return self._parts[1]

    @property
    def transformer(self) -> nn.Module:
        return self._parts[2]

    @property
    def downsample(self) -> nn.Module:
        return self._parts[3]

    @staticmethod
    def _conv(conv: MimiConv1d, hidden: torch.Tensor, samples: torch.Tensor, rate: int) -> torch.Tensor:
        """``MimiConv1d.forward`` of a row's steps at ``rate`` samples a step, as if unpadded."""
        stride = conv.conv.stride[0]
        kernel = (conv.conv.kernel_size[0] - 1) * conv.conv.dilation[0] + 1
        left = kernel - stride
        length = hidden.shape[-1]
        # Mimi pads up to a whole stride on the right
        frames = -(-length // stride)
        if stride > 1:
            steps = (samples + rate - 1) // rate
            outside = (torch.arange(length, device=hidden.device) >= steps.unsqueeze(1)).unsqueeze(1)
            if conv.pad_mode == "replicate":
                fill = hidden.gather(2, (steps - 1).view(-1, 1, 1).expand(-1, hidden.shape[1], 1))
            else:
                fill = hidden.new_zeros(())
            hidden = torch.where(outside, fill, hidden)
        if conv.pad_mode == "constant" and frames * stride - length <= left:
            # The conv's own zero padding, on both sides, rather than a padded copy
            # of the input: at the full sample rate that copy is a fifth of the
            # graph's memory. The outputs past ``frames`` read right padding only.
            weight = conv.conv
            out = F.conv1d(hidden, weight.weight, weight.bias, stride, left, weight.dilation, weight.groups)
            return out[..., :frames]
        hidden = F.pad(hidden, (left, frames * stride - length), mode=conv.pad_mode)
        return conv.conv(hidden)

    def forward(self, waveform: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        latents = self.latents(waveform, samples)
        # [quantizers, B, frames]
        codes = self.codec_encoder.quantizer.encode(latents, self.num_quantizers)
        return codes.permute(1, 2, 0).contiguous()

    def latents(self, waveform: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        """The latents ``[B, dim, frames]`` the quantizers read."""
        hidden = waveform.unsqueeze(1)
        rate = 1
        for layer in self.seanet.layers:
            if isinstance(layer, MimiConv1d):
                hidden = self._conv(layer, hidden, samples, rate)
                rate *= layer.conv.stride[0]
            elif isinstance(layer, MimiResnetBlock):
                residual = hidden
                for sub in layer.block:
                    hidden = self._conv(sub, hidden, samples, rate) if isinstance(sub, MimiConv1d) else sub(hidden)
                shortcut = layer.shortcut
                if isinstance(shortcut, MimiConv1d):
                    residual = self._conv(shortcut, residual, samples, rate)
                hidden = residual + hidden
            elif isinstance(layer, nn.ELU):
                hidden = layer(hidden)
            else:
                raise TypeError(f"the padded codec encode does not know Mimi layer {type(layer).__name__}")
        hidden = self.transformer(hidden.transpose(1, 2), use_cache=False).last_hidden_state.transpose(1, 2)
        return self._conv(self.downsample, hidden, samples, rate)
