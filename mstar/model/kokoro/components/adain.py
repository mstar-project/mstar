"""Style-conditioned normalization blocks shared by the prosody predictor and
the decoder (StyleTTS2's ``AdaIN1d`` / ``AdainResBlk1d``).

Instance statistics are taken over the *valid* frames of each row only, so a
padded batch normalizes exactly as the single-request reference does.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.kokoro.components.masking import length_mask, mask_channels

INSTANCE_NORM_EPS = 1e-5
LEAKY_SLOPE = 0.2


def masked_instance_norm(x: torch.Tensor, mask: torch.Tensor, eps: float = INSTANCE_NORM_EPS) -> torch.Tensor:
    """Per-(row, channel) normalization over the frames selected by ``mask``.

    Matches ``nn.InstanceNorm1d`` (biased variance, no affine) on the valid
    span and returns zeros on the padded tail.
    """
    m = mask[:, None, :].to(x.dtype)
    count = m.sum(dim=-1, keepdim=True)
    mean = (x * m).sum(dim=-1, keepdim=True) / count
    centered = (x - mean) * m
    var = centered.square().sum(dim=-1, keepdim=True) / count
    return centered * torch.rsqrt(var + eps)


class AdaIN1d(nn.Module):
    """Instance norm whose scale and shift come from the style vector."""

    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(style).chunk(2, dim=1)
        out = (1 + gamma[:, :, None]) * masked_instance_norm(x, mask) + beta[:, :, None]
        return mask_channels(out, mask)


class AdainResBlk1d(nn.Module):
    """Residual block ``AdaIN -> LeakyReLU -> [x2 upsample] -> conv -> AdaIN -> LeakyReLU -> conv``
    with a (possibly learned) shortcut, scaled by ``1/sqrt(2)``.

    Returns the output together with the per-row lengths, which double when
    the block upsamples.
    """

    def __init__(self, dim_in: int, dim_out: int, style_dim: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
        if upsample:
            self.pool = nn.ConvTranspose1d(
                dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1
            )

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = length_mask(lengths, x.shape[-1])
        h = F.leaky_relu(self.norm1(x, style, mask), LEAKY_SLOPE)
        shortcut = x
        if self.upsample:
            lengths = lengths * 2
            mask = length_mask(lengths, 2 * x.shape[-1])
            h = mask_channels(self.pool(h), mask)
            shortcut = F.interpolate(shortcut, scale_factor=2, mode="nearest")
        h = mask_channels(self.conv1(h), mask)
        h = F.leaky_relu(self.norm2(h, style, mask), LEAKY_SLOPE)
        h = mask_channels(self.conv2(h), mask)
        if self.learned_sc:
            shortcut = self.conv1x1(shortcut)
        return (h + shortcut) * math.sqrt(0.5), lengths
