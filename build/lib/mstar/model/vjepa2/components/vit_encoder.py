"""V-JEPA 2 video encoder (ViT with 3D tubelet patches + 3D RoPE).

Ports ``VJEPA2PatchEmbeddings3D``, ``VJEPA2Embeddings``, ``VJEPA2Encoder``
from HuggingFace ``transformers/models/vjepa2/modeling_vjepa2.py``.

Input layout: ``pixel_values_videos`` of shape ``[B, T, C, H, W]`` (frames
before channels — matches HF default).  The embedding layer permutes
internally to ``[B, C, T, H, W]`` for Conv3d.

Weight layout (matches HF checkpoint keys, prefix ``encoder.``):
    encoder.embeddings.patch_embeddings.proj.{weight,bias}
    encoder.layer.{N}.*
    encoder.layernorm.{weight,bias}
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.vjepa2.components.layers import VJEPA2Layer
from mstar.model.vjepa2.config import VJepa2Config


def apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather per-row patch indices from ``tensor``.

    Args:
        tensor: ``[B, N, D]``.
        masks: list of ``[B, M]`` index tensors.  Outputs stacked along dim 0.

    Returns:
        ``[len(masks) * B, M, D]``.
    """
    all_masked = []
    for mask in masks:
        mask = mask.to(tensor.device)
        mask_keep = mask.unsqueeze(-1).repeat(1, 1, tensor.size(-1))
        all_masked.append(torch.gather(tensor, dim=1, index=mask_keep))
    return torch.cat(all_masked, dim=0)


class VJEPA2PatchEmbeddings3D(nn.Module):
    def __init__(self, config: VJepa2Config, hidden_size: int):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size
        self.hidden_size = hidden_size
        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        # pixel_values_videos: [B, C, T, H, W] (already permuted by caller)
        return self.proj(pixel_values_videos).flatten(2).transpose(1, 2)


class VJEPA2Embeddings(nn.Module):
    def __init__(self, config: VJepa2Config, hidden_size: int):
        super().__init__()
        self.config = config
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(config, hidden_size=hidden_size)

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        num_frames = pixel_values_videos.shape[1]
        # [B, T, C, H, W] -> [B, C, T, H, W] for Conv3d
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)
        if num_frames < self.config.tubelet_size:
            pixel_values_videos = pixel_values_videos.repeat(1, 1, self.config.tubelet_size, 1, 1)
        target_dtype = self.patch_embeddings.proj.weight.dtype
        pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)
        return self.patch_embeddings(pixel_values_videos)


class VJEPA2Encoder(nn.Module):
    def __init__(self, config: VJepa2Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size)
        self.layer = nn.ModuleList(
            [
                VJEPA2Layer(
                    config,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values_videos)
        for layer in self.layer:
            hidden_states = layer(hidden_states, position_mask=None)
        return self.layernorm(hidden_states)
