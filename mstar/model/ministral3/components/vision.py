"""Native Pixtral tower and Ministral projector over packed image patches.

Patches arrive in spatial-merge-block order, with (row, column) coordinates.
The engine's ragged attention segments isolate images. There is no KV cache,
position counter, or request-owned state here. Vision runs on one rank; its
placement is independent of the text node.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mstar.model.components.norm import RMSNorm
from mstar.model.components.qwen3_vit import vision_position_ids
from mstar.model.keys import VISION_ATTN


@dataclass(frozen=True)
class PixtralConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    head_dim: int = 64
    num_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    rope_theta: float = 10000.0
    text_hidden_size: int = 3072
    projector_norm_eps: float = 1e-5

    def __post_init__(self):
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                     "head_dim", "num_channels", "patch_size", "spatial_merge_size", "text_hidden_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"Pixtral {name} must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim or self.head_dim % 4:
            raise ValueError("Pixtral requires hidden_size = heads * head_dim and head_dim divisible by four")
        if self.rope_theta <= 0 or self.projector_norm_eps <= 0:
            raise ValueError("Pixtral rope_theta and projector_norm_eps must be positive")

    @classmethod
    def from_hf(cls, root: dict) -> "PixtralConfig":
        vision = root["vision_config"]
        rope = vision["rope_parameters"]
        if (vision["model_type"] != "pixtral" or vision["hidden_act"] != "silu"
                or rope["rope_type"] not in ("default", "axial")
                or root["vision_feature_layer"] != -1
                or root["projector_hidden_act"] != "gelu" or root["multimodal_projector_bias"]):
            raise ValueError("Unsupported Pixtral tower or Ministral projector configuration")
        return cls(**{name: vision[name] for name in (
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "head_dim", "num_channels", "patch_size",
        )}, spatial_merge_size=root["spatial_merge_size"], rope_theta=rope["rope_theta"],
            text_hidden_size=root["text_config"]["hidden_size"],
            projector_norm_eps=root["text_config"]["rms_norm_eps"])


def pack_image_patches(images: list[torch.Tensor], config: PixtralConfig):
    """Pack already resized/normalized CPU CHW images; preprocessing comes before this.

    The returned grids are host metadata; coordinates and patches concatenate
    without rebasing when the engine batches requests.
    """
    if not images:
        raise ValueError("Pixtral requires at least one image")
    patches, grids = [], []
    p, m = config.patch_size, config.spatial_merge_size
    for image in images:
        if image.device.type != "cpu" or image.ndim != 3 or image.shape[0] != config.num_channels:
            raise ValueError("Expected CPU CHW images with the configured channel count")
        _, height, width = image.shape
        if min(height, width) == 0 or height % (p * m) or width % (p * m):
            raise ValueError("Image dimensions must be positive multiples of patch_size * spatial_merge_size")
        h, w = height // p, width // p
        raster = F.unfold(image.unsqueeze(0), kernel_size=p, stride=p)[0].T
        patches.append(raster.reshape(h // m, m, w // m, m, -1)
                       .permute(0, 2, 1, 3, 4).reshape(h * w, -1))
        grids.append((1, h, w))
    return torch.cat(patches), vision_position_ids(grids, m), grids


def axial_rotary(position_ids: torch.Tensor, config: PixtralConfig, dtype: torch.dtype):
    """Stateless axial rotation from image coordinates, not decoder position tables."""
    inv = 1.0 / (config.rope_theta ** (
        torch.arange(0, config.head_dim, 2, device=position_ids.device, dtype=torch.float32) / config.head_dim
    ))
    angles = torch.cat((position_ids[:, :1].float() * inv[0::2],
                        position_ids[:, 1:].float() * inv[1::2]), dim=-1)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos[:, None] + torch.cat((-b, a), dim=-1) * sin[:, None]


class PixtralAttention(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        # Separate projections preserve the released checkpoint's names/shapes.
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(config.hidden_size, config.hidden_size, bias=False))
        self.ragged_attn = None

    def bind_resources(self, resources: dict):
        self.ragged_attn = resources[VISION_ATTN]

    def forward(self, hidden, cos, sin):
        if self.ragged_attn is None:
            raise ValueError("Pixtral requires its declared ragged attention resource")
        shape = (-1, self.config.num_attention_heads, self.config.head_dim)
        q, k, v = (projection(hidden).reshape(shape) for projection in (self.q_proj, self.k_proj, self.v_proj))
        out = self.ragged_attn.run(_rotate(q, cos, sin), _rotate(k, cos, sin), v, label="main")
        return self.o_proj(out.reshape(-1, self.config.hidden_size))


class PixtralMLP(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class PixtralBlock(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.attention = PixtralAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP(config)

    def forward(self, hidden, cos, sin):
        hidden = hidden + self.attention(self.attention_norm(hidden), cos, sin)
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class PixtralTower(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.patch_conv = nn.Conv2d(config.num_channels, config.hidden_size,
                                    config.patch_size, stride=config.patch_size, bias=False)
        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([PixtralBlock(config) for _ in range(config.num_hidden_layers)])


class PixtralProjector(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.hidden_size, eps=config.projector_norm_eps)
        self.patch_merger = nn.Module()
        self.patch_merger.merging_layer = nn.Linear(
            config.hidden_size * config.spatial_merge_size ** 2, config.hidden_size, bias=False,
        )
        self.linear_1 = nn.Linear(config.hidden_size, config.text_hidden_size, bias=False)
        self.linear_2 = nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=False)

    def forward(self, hidden):
        hidden = self.norm(hidden)
        # HF unfold orders channels before positions within a merge block.
        hidden = hidden.reshape(-1, self.config.spatial_merge_size ** 2, self.config.hidden_size)
        hidden = hidden.transpose(1, 2).flatten(1)
        hidden = self.patch_merger.merging_layer(hidden)
        return self.linear_2(F.gelu(self.linear_1(hidden)))


class Ministral3Vision(nn.Module):
    """PackedBlockLoopEncoder's embed/encode/project tensor computation."""

    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.vision_tower = PixtralTower(config)
        self.multi_modal_projector = PixtralProjector(config)

    def embed(self, pixel_values, position_ids):
        weight = self.vision_tower.patch_conv.weight
        hidden = F.linear(pixel_values.to(weight.dtype), weight.flatten(1))
        hidden = self.vision_tower.ln_pre(hidden)
        cos, sin = axial_rotary(position_ids, self.config, hidden.dtype)
        return hidden, cos, sin

    def encode(self, hidden, cos, sin):
        for layer in self.vision_tower.transformer.layers:
            hidden = layer(hidden, cos, sin)
        return hidden

    def project(self, hidden, **kwargs):
        return self.multi_modal_projector(hidden)
