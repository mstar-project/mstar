"""BigVGAN generator (inference): transposed-conv upsamplers, each followed by
anti-aliased AMP blocks of SnakeBeta activations. Names are the reference's, so a
checkpoint loads once its weight-norm pairs are folded (``fold_weight_norm``). The stored
resampling filters load as frozen parameters, so the loader's completeness check covers
them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# Both spellings of a weight-norm pair: the classic hook's and the parametrization's.
_WEIGHT_NORM_LEAVES = {
    "weight_g": "g", "weight_v": "v",
    "parametrizations.weight.original0": "g", "parametrizations.weight.original1": "v",
}


def fold_weight_norm(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    """``weights`` with each weight-norm pair replaced by its effective ``<module>.weight``.

    The fold is ``torch._weight_norm(v, g, 0)``, which is what either form evaluates on
    every forward, so on the same device the folded module computes the same bits.
    """
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        for leaf, role in _WEIGHT_NORM_LEAVES.items():
            if name.endswith("." + leaf):
                base = name[: -len(leaf) - 1]
                parts = pending.setdefault(base, {})
                parts[role] = tensor
                if len(parts) == 2:
                    del pending[base]
                    yield f"{base}.weight", torch._weight_norm(parts["v"], parts["g"], 0)
                break
        else:
            yield name, tensor
    if pending:
        raise ValueError(f"weight-norm pairs missing a half: {sorted(pending)[:5]}")


class SnakeBeta(nn.Module):
    """``x + sin(exp(alpha) * x)^2 / (exp(beta) + 1e-9)``, per channel; ``alpha`` and
    ``beta`` are stored in log space."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha)[None, :, None]
        beta = torch.exp(self.beta)[None, :, None]
        return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class _LowPass(nn.Module):
    """Depthwise low-pass filter with a stride: the downsampler."""

    def __init__(self, stride: int, kernel_size: int):
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = nn.Parameter(torch.zeros(1, 1, kernel_size), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class _DownSample(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = _LowPass(ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class _UpSample(nn.Module):
    """Depthwise transposed low-pass filter: the ``ratio``x upsampler."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.filter = nn.Parameter(torch.zeros(1, 1, kernel_size), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.ratio, groups=channels,
        )
        return x[..., self.pad_left:-self.pad_right]


class AliasFreeActivation(nn.Module):
    """SnakeBeta evaluated at twice the sample rate: upsample, activate, low-pass down."""

    def __init__(self, channels: int, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = _UpSample(ratio, kernel_size)
        self.downsample = _DownSample(ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock(nn.Module):
    """Per dilation, a dilated conv and a dilation-1 conv, each after its own activation,
    added back to the input."""

    def __init__(self, channels: int, kernel_size: int, dilations: Sequence[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilations
        )
        self.activations = nn.ModuleList(AliasFreeActivation(channels) for _ in range(2 * len(dilations)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pairs = zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2], strict=True)
        for conv1, conv2, act1, act2 in pairs:
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class BigVGANGenerator(nn.Module):
    """``[batch, in_channels, frames] -> [batch, 1, frames * prod(upsample_rates)]``.

    Stage ``i`` upsamples to ``initial_channels / 2^(i+1)`` channels and averages one AMP
    block per resblock kernel size. The output is ``tanh``-squashed or clamped to
    ``[-1, 1]`` (``final_tanh``), the checkpoint's choice.
    """

    def __init__(
        self,
        in_channels: int,
        initial_channels: int,
        upsample_rates: Sequence[int],
        upsample_kernel_sizes: Sequence[int],
        resblock_kernel_sizes: Sequence[int],
        resblock_dilations: Sequence[Sequence[int]],
        *,
        final_bias: bool = True,
        final_tanh: bool = True,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.final_tanh = final_tanh
        self.conv_pre = nn.Conv1d(in_channels, initial_channels, 7, padding=3)
        # one-element lists: the reference nests each upsampler as ``ups.<i>.0``
        self.ups = nn.ModuleList(
            nn.ModuleList([nn.ConvTranspose1d(
                initial_channels // 2**i, initial_channels // 2 ** (i + 1), kernel, rate, padding=(kernel - rate) // 2,
            )])
            for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True))
        )
        self.resblocks = nn.ModuleList(
            AMPBlock(initial_channels // 2 ** (i + 1), kernel, dilations)
            for i in range(len(self.ups))
            for kernel, dilations in zip(resblock_kernel_sizes, resblock_dilations, strict=True)
        )
        channels = initial_channels // 2 ** len(self.ups)
        self.activation_post = AliasFreeActivation(channels)
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3, bias=final_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, (up,) in enumerate(self.ups):
            x = up(x)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            total = blocks[0](x)
            for block in blocks[1:]:
                total = total + block(x)
            x = total / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.tanh(x) if self.final_tanh else x.clamp(-1.0, 1.0)


__all__ = ["AMPBlock", "AliasFreeActivation", "BigVGANGenerator", "SnakeBeta", "fold_weight_norm"]
