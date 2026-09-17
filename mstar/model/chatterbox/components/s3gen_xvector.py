"""CAMPPlus speaker encoder (the ``speaker_encoder`` of S3Gen).

Reference: ``chatterbox/models/s3gen/xvector.py`` (FunASR / 3D-Speaker
CAMPPlus). Input is an 80-bin Kaldi fbank ``[B, T, 80]``; output the 192-dim
x-vector that conditions the flow decoder. Parameter paths mirror the
checkpoint (``head.*``, ``xvector.*``) so the weights stream in unchanged.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.chatterbox.config import S3GenXVectorConfig


class _ResBlock2d(nn.Module):
    """3x3 residual block whose stride only walks the frequency axis."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class _FrequencyConvHead(nn.Module):
    """FCM: 2-D convolutions over (frequency, time) that fold frequency into channels."""

    def __init__(self, m_channels: int = 32, feat_dim: int = 80):
        super().__init__()
        self.conv1 = nn.Conv2d(1, m_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = nn.Sequential(_ResBlock2d(m_channels, m_channels, 2), _ResBlock2d(m_channels, m_channels, 1))
        self.layer2 = nn.Sequential(_ResBlock2d(m_channels, m_channels, 2), _ResBlock2d(m_channels, m_channels, 1))
        self.conv2 = nn.Conv2d(m_channels, m_channels, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        b, c, f, t = out.shape
        return out.reshape(b, c * f, t)


def _bn_relu(channels: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict([("batchnorm", nn.BatchNorm1d(channels)), ("relu", nn.ReLU(inplace=True))]))


class _TDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False,
        )
        self.nonlinear = _bn_relu(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class _ContextAwareMask(nn.Module):
    """CAM layer: a local conv gated by a sigmoid of global + segment context."""

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        reduction: int = 2,
        seg_len: int = 100,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.linear_local = nn.Conv1d(
            bn_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.seg_len = seg_len

    def _segment_pool(self, x: torch.Tensor) -> torch.Tensor:
        seg = F.avg_pool1d(x, kernel_size=self.seg_len, stride=self.seg_len, ceil_mode=True)
        b, c, n = seg.shape
        seg = seg.unsqueeze(-1).expand(b, c, n, self.seg_len).reshape(b, c, -1)
        return seg[..., : x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self._segment_pool(x)
        context = F.relu(self.linear1(context))
        return y * torch.sigmoid(self.linear2(context))


class _CAMDenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.nonlinear1 = _bn_relu(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = _bn_relu(bn_channels)
        self.cam_layer = _ContextAwareMask(bn_channels, out_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(self.nonlinear1(x))
        return self.cam_layer(self.nonlinear2(x))


class _CAMDenseBlock(nn.ModuleList):
    """Densely connected TDNN block: every layer sees the concat of all before it."""

    def __init__(
        self, num_layers: int, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, dilation: int,
    ):
        super().__init__()
        for i in range(num_layers):
            self.add_module(
                f"tdnnd{i + 1}",
                _CAMDenseLayer(in_channels + i * out_channels, out_channels, bn_channels, kernel_size, dilation),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class _TransitLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.nonlinear = _bn_relu(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(x))


class _StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)


class _DenseOut(nn.Module):
    """1x1 projection followed by a non-affine batch norm (the ``batchnorm_`` head)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.nonlinear = nn.Sequential(OrderedDict([("batchnorm", nn.BatchNorm1d(out_channels, affine=False))]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x.unsqueeze(-1)).squeeze(-1))


class CAMPPlus(nn.Module):
    """``[B, T, feat_dim]`` fbank -> ``[B, embedding_size]`` speaker embedding."""

    def __init__(self, config: S3GenXVectorConfig):
        super().__init__()
        self.config = config
        self.head = _FrequencyConvHead(feat_dim=config.feat_dim)
        channels = self.head.out_channels
        layers: list[tuple[str, nn.Module]] = [
            ("tdnn", _TDNNLayer(channels, config.init_channels, 5, stride=2, dilation=1)),
        ]
        channels = config.init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip(config.block_num_layers, config.block_kernel_sizes, config.block_dilations, strict=True)
        ):
            layers.append((
                f"block{i + 1}",
                _CAMDenseBlock(
                    num_layers, channels, config.growth_rate,
                    config.bn_size * config.growth_rate, kernel_size, dilation,
                ),
            ))
            channels += num_layers * config.growth_rate
            layers.append((f"transit{i + 1}", _TransitLayer(channels, channels // 2)))
            channels //= 2
        layers.append(("out_nonlinear", _bn_relu(channels)))
        layers.append(("stats", _StatsPool()))
        layers.append(("dense", _DenseOut(channels * 2, config.embedding_size)))
        self.xvector = nn.Sequential(OrderedDict(layers))

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        x = fbank.permute(0, 2, 1)
        return self.xvector(self.head(x))
