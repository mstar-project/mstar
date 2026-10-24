"""Command A+'s image encoder: a SigLIP tower, a pixel shuffle and a SwiGLU projector.

Tiles are all ``image_size`` square, so they batch as plain ``[tiles, patches, width]``.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.command_a_plus.config import CommandAPlusVisionConfig


class SiglipAttention(nn.Module):
    def __init__(self, config: CommandAPlusVisionConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tiles, length, width = x.shape
        q, k, v = (
            proj(x).view(tiles, length, self.num_heads, -1).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(tiles, length, width))


class SiglipMLP(nn.Module):
    def __init__(self, config: CommandAPlusVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SiglipLayer(nn.Module):
    def __init__(self, config: CommandAPlusVisionConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SiglipEmbeddings(nn.Module):
    def __init__(self, config: CommandAPlusVisionConfig) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        return patches.flatten(2).transpose(1, 2) + self.position_embedding.weight


class CommandAPlusProjector(nn.Module):
    """Merges each ``downsample_factor``-square block of patches into one LLM token."""

    def __init__(self, config: CommandAPlusVisionConfig, text_hidden_size: int) -> None:
        super().__init__()
        self.factor = config.downsample_factor
        self.linear_1 = nn.Linear(
            config.hidden_size * self.factor ** 2, config.alignment_intermediate_size,
        )
        self.linear_2 = nn.Linear(config.alignment_intermediate_size // 2, text_hidden_size)

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        # Hugging Face's order, which the projector was trained on.
        tiles, length, width = x.shape
        side, f = int(length ** 0.5), self.factor
        x = x.reshape(tiles, side, side // f, width * f).permute(0, 2, 1, 3)
        x = x.reshape(tiles, side // f, side // f, width * f * f).permute(0, 2, 1, 3)
        return x.reshape(tiles, -1, width * f * f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.linear_1(self.pixel_shuffle(x)).chunk(2, dim=-1)
        return self.linear_2(F.silu(gate) * x)


CHECKPOINT_PREFIXES = (
    ("model.vision_tower.vision_model.encoder.", ""),
    ("model.vision_tower.vision_model.", ""),
    ("model.multi_modal_projector.", "projector."),
)


class CommandAPlusVisionEncoder(nn.Module):
    """Image tiles in, one LLM-width embedding per ``<|IMG_PATCH|>`` token out."""

    def __init__(self, config: CommandAPlusVisionConfig, text_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SiglipEmbeddings(config)
        self.layers = nn.ModuleList(SiglipLayer(config) for _ in range(config.num_hidden_layers))
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projector = CommandAPlusProjector(config, text_hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[tiles, 3, size, size]`` to ``[tiles * tokens_per_tile, text_hidden]``, tile-major."""
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x)
        return self.projector(self.post_layernorm(x)).flatten(0, 1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Strict: a missing, unexpected or misshaped weight fails."""
        state = {}
        for name, tensor in weights:
            for prefix, target in CHECKPOINT_PREFIXES:
                if name.startswith(prefix):
                    name = target + name.removeprefix(prefix)
                    break
            state[name] = tensor
        self.load_state_dict(state)
