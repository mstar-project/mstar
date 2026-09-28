"""MoonViT vision tower + patch-merger projector for Kimi-K2.7-Code."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from mstar.model.kimi_k2_7.config import KimiVisionConfig


def _resized_and_padded_dims(
    w: int, h: int, cfg: KimiVisionConfig
) -> tuple[int, int, int, int]:
    patch = cfg.patch_size
    s1 = math.sqrt(
        cfg.in_patch_limit / (max(1.0, w // patch) * max(1.0, h // patch))
    )
    s2 = cfg.patch_limit_on_one_side * patch / w
    s3 = cfg.patch_limit_on_one_side * patch / h
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    new_w = min(new_w, cfg.patch_limit_on_one_side * patch)
    new_h = min(new_h, cfg.patch_limit_on_one_side * patch)

    factor = cfg.merge_kernel_size * patch
    pad_w = (factor - new_w % factor) % factor
    pad_h = (factor - new_h % factor) % factor
    return new_w, new_h, pad_w, pad_h


def num_image_tokens(w: int, h: int, cfg: KimiVisionConfig) -> int:
    """Vision tokens (post spatial-merge) an image of size ``(w, h)`` costs."""
    new_w, new_h, pad_w, pad_h = _resized_and_padded_dims(w, h, cfg)
    factor = cfg.merge_kernel_size * cfg.patch_size
    return ((new_h + pad_h) // factor) * ((new_w + pad_w) // factor)


def preprocess_image(
    img_chw: torch.Tensor, cfg: KimiVisionConfig
) -> tuple[torch.Tensor, int, int]:
    """Resize + pad + normalize + patchify one image.

    ``img_chw`` is a float tensor in [0, 1], shape (3, H, W) as
    ``Model.load_image`` hands back. Returns ``(patches, gh, gw)`` where
    ``patches`` has shape ``(gh * gw, 3, patch_size, patch_size)`` and
    ``(gh, gw)`` is the pre-merge patch grid.
    """
    c, h, w = img_chw.shape
    new_w, new_h, pad_w, pad_h = _resized_and_padded_dims(w, h, cfg)
    uint8_hwc = (
        (img_chw * 255).round().clamp(0, 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
    )
    resized_pil = Image.fromarray(uint8_hwc).resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized = (
        torch.from_numpy(np.array(resized_pil)).to(device=img_chw.device, dtype=img_chw.dtype)
        .permute(2, 0, 1)
        / 255
    )
    padded = F.pad(resized, (0, pad_w, 0, pad_h))

    mean = torch.tensor(cfg.image_mean, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(cfg.image_std, device=img_chw.device).view(3, 1, 1)
    normalized = (padded - mean) / std

    patch = cfg.patch_size
    gh, gw = (new_h + pad_h) // patch, (new_w + pad_w) // patch
    patches = (
        normalized.view(c, gh, patch, gw, patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(gh * gw, c, patch, patch)
    )
    return patches, gh, gw


def _rope_2d_freqs_cis(
    gh: int, gw: int, head_dim: int, device: torch.device, theta_base: float = 10000.0
) -> torch.Tensor:
    dim_range = torch.arange(0, head_dim, 4, device=device).float()[: head_dim // 4]
    freqs = 1.0 / (theta_base ** (dim_range / head_dim))
    flat_pos = torch.arange(gh * gw, device=device).float()
    x_pos, y_pos = flat_pos % gw, flat_pos // gw
    x_cis = torch.polar(torch.ones(gh * gw, len(freqs), device=device), torch.outer(x_pos, freqs))
    y_cis = torch.polar(torch.ones(gh * gw, len(freqs), device=device), torch.outer(y_pos, freqs))
    return torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1).reshape(gh * gw, -1)


def _apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class KimiVisionPosEmbed(nn.Module):
    def __init__(self, height: int, width: int, dim: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.weight = nn.Parameter(torch.empty(height, width, dim))

    def forward(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        if (gh, gw) == (self.height, self.width):
            pos = self.weight.flatten(end_dim=1)
        else:
            pos = (
                F.interpolate(
                    self.weight.float().permute(2, 0, 1).unsqueeze(0),
                    size=(gh, gw),
                    mode="bicubic",
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .reshape(-1, self.weight.shape[-1])
                .to(self.weight.dtype)
            )
        return x + pos


class KimiVisionPatchEmbed(nn.Module):
    def __init__(self, config: KimiVisionConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.pos_emb = KimiVisionPosEmbed(
            config.pos_emb_height, config.pos_emb_width, config.hidden_size,
        )

    def forward(self, patches: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        x = self.proj(patches).view(patches.size(0), -1)
        return self.pos_emb(x, gh, gw)


class KimiVisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc0 = nn.Linear(dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class KimiVisionBlock(nn.Module):
    def __init__(self, config: KimiVisionConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.wqkv = nn.Linear(dim, dim * 3, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)
        self.mlp = KimiVisionMLP(dim, config.intermediate_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        residual = x
        h = self.norm0(x)
        qkv = self.wqkv(h).view(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q, k = _apply_rope(q, k, freqs_cis)
        attn = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )
        attn = attn.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        x = residual + self.wo(attn)

        residual = x
        return residual + self.mlp(self.norm1(x))


class KimiVisionEncoder(nn.Module):
    def __init__(self, config: KimiVisionConfig) -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.blocks = nn.ModuleList(
            [KimiVisionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        freqs_cis = _rope_2d_freqs_cis(gh, gw, self.head_dim, x.device)
        for block in self.blocks:
            x = block(x, freqs_cis)
        return self.final_layernorm(x)


def _patch_merge(x: torch.Tensor, gh: int, gw: int, k: int) -> torch.Tensor:
    d = x.size(-1)
    nh, nw = gh // k, gw // k
    x = x.view(nh, k, nw, k, d).permute(0, 2, 1, 3, 4).contiguous()
    return x.view(nh * nw, k * k, d)


class KimiVisionTower(nn.Module):
    def __init__(self, config: KimiVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = KimiVisionPatchEmbed(config)
        self.encoder = KimiVisionEncoder(config)

    def forward(self, patches: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        x = self.patch_embed(patches, gh, gw)
        x = self.encoder(x, gh, gw)
        return _patch_merge(x, gh, gw, self.config.merge_kernel_size)


class KimiMMProjector(nn.Module):
    def __init__(self, config: KimiVisionConfig) -> None:
        super().__init__()
        in_dim = config.hidden_size * config.merge_kernel_size * config.merge_kernel_size
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.projector_ln_eps)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, config.text_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        return self.proj(x.reshape(x.size(0), -1))
