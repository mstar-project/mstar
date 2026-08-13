from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

CACHE_T = 2


def _silu(_: str = "silu"):
    return nn.SiLU()


class LingBotWanCausalConv3d(nn.Conv3d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x=None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))


class LingBotWanRMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
        normalized = F.normalize(x.float() if needs_fp32 else x, dim=(1 if self.channel_first else -1)).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class LingBotWanResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu") -> None:
        super().__init__()
        self.nonlinearity = _silu(non_linearity)
        self.norm1 = LingBotWanRMSNorm(in_dim, images=False)
        self.conv1 = LingBotWanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = LingBotWanRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LingBotWanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = LingBotWanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)
        x = self.nonlinearity(self.norm1(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        x = self.dropout(self.nonlinearity(self.norm2(x)))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        return x + h


class LingBotWanAttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = LingBotWanRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x).view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class LingBotWanMidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        resnets = [LingBotWanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(LingBotWanAttentionBlock(dim))
            resnets.append(LingBotWanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:], strict=False):
            x = resnet(attn(x), feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class LingBotWanUpsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class LingBotWanResample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None):
        super().__init__()
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                LingBotWanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                LingBotWanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = LingBotWanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                x = self.time_conv(x) if feat_cache[idx] == "Rep" else self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)


class LingBotWanUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, dropout: float = 0.0, upsample_mode=None):
        super().__init__()
        current_dim = in_dim
        self.resnets = nn.ModuleList([])
        for _ in range(num_res_blocks + 1):
            self.resnets.append(LingBotWanResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.upsamplers = nn.ModuleList([LingBotWanResample(out_dim, upsample_mode)]) if upsample_mode else None

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=None):
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx) if feat_cache is not None else resnet(x)
        if self.upsamplers is not None:
            x = (
                self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
                if feat_cache is not None
                else self.upsamplers[0](x)
            )
        return x


class LingBotWanDecoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.nonlinearity = _silu()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = LingBotWanCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = LingBotWanMidBlock(dims[0], dropout, num_layers=1)
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            if i > 0 and not is_residual:
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            mode = "upsample3d" if up_flag and temperal_upsample[i] else ("upsample2d" if up_flag else None)
            self.up_blocks.append(LingBotWanUpBlock(in_dim, out_dim, num_res_blocks, dropout, mode))
        self.norm_out = LingBotWanRMSNorm(out_dim, images=False)
        self.conv_out = LingBotWanCausalConv3d(out_dim, out_channels, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        x = self.nonlinearity(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            return x
        return self.conv_out(x)


class LingBotWanVaeDecoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = SimpleNamespace(**config)
        base_dim = config.get("base_dim", 96)
        decoder_base_dim = config.get("decoder_base_dim") or base_dim
        z_dim = config.get("z_dim", 16)
        dim_mult = config.get("dim_mult", [1, 2, 4, 4])
        num_res_blocks = config.get("num_res_blocks", 2)
        dropout = config.get("dropout", 0.0)
        temperal_upsample = list(reversed(config.get("temperal_downsample", [False, True, True])))
        self.post_quant_conv = LingBotWanCausalConv3d(z_dim, z_dim, 1)
        self.decoder = LingBotWanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=config.get("attn_scales", []),
            temperal_upsample=temperal_upsample,
            dropout=dropout,
            out_channels=config.get("out_channels", 3),
            is_residual=config.get("is_residual", False),
        )
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, LingBotWanCausalConv3d) for m in self.decoder.modules()),
            "encoder": 0,
        }
        self.clear_cache()

    def clear_cache(self) -> None:
        self._conv_num = self._cached_conv_counts["decoder"]
        self._feat_map = [None] * self._conv_num
        self._conv_idx = [0]

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, _, _ = z.shape
        self.clear_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_i = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_i], 2)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return SimpleNamespace(sample=out)
