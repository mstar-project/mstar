"""The Cosmos3-Edge reasoner's vision tower and projector.

A packed, variable-resolution SigLIP2-style encoder (``visual``): a linear
patch embedding over ``patch_size x patch_size`` RGB patches, a learned square
position grid bilinearly resized to every frame's patch grid, ``N`` pre-LayerNorm
encoder blocks whose attention stays within one frame, and a post LayerNorm.
The projector (``projector``) merges each 2x2 block of patches into one token
(LayerNorm per patch -> concat -> Linear -> GELU -> Linear) in the text hidden
size, which the understanding tower consumes in place of the ``<|image_pad|>``
/ ``<|video_pad|>`` tokens.

Parameter names follow ``vision_encoder/model.safetensors`` with its leading
``model.`` stripped (see ``loader.vision_encoder_name_remapper``), so the
checkpoint loads by name. Shapes and the packed patch order come from the
Hugging Face ``Cosmos3EdgeVisionModel`` / ``Cosmos3EdgePatchMerger``.

Frames of one video share a grid, and an image is a single frame, so the
attention runs one batched SDPA per distinct frame size instead of a
block-diagonal mask over the whole pack.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.cosmos3.config import Cosmos3ReasonerConfig, Cosmos3VisionEncoderConfig


def _activation(name: str):
    if name in ("gelu_pytorch_tanh", "gelu_tanh"):
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported vision activation {name!r}")


def resize_position_grid(
    grid_embeds: torch.Tensor, grid_thw: list[tuple[int, int, int]], merge_size: int,
) -> torch.Tensor:
    """Resize the learned ``[S, S, hidden]`` position grid to every frame's
    ``(h, w)`` patch grid and lay it out in the processor's block-major
    (2x2-merge) patch order, repeated over the frame's ``t``.

    Bilinear, ``align_corners=False``, antialiased — the reference's
    ``F.interpolate`` call; fp32 on CPU because antialias has no bf16 kernel
    there. Returns ``[total_patches, hidden]`` in pack order.
    """
    source_dtype = grid_embeds.dtype
    grid = grid_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, hidden, S, S]
    if grid.device.type == "cpu":
        grid = grid.float()
    chunks = []
    for t, h, w in grid_thw:
        resized = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        resized = resized.squeeze(0).permute(1, 2, 0).to(source_dtype)  # [h, w, hidden]
        resized = resized.reshape(h // merge_size, merge_size, w // merge_size, merge_size, -1)
        resized = resized.transpose(1, 2).reshape(h * w, -1)
        chunks.append(resized.repeat(t, 1))
    return torch.cat(chunks, dim=0)


class Cosmos3VisionEmbeddings(nn.Module):
    def __init__(self, config: Cosmos3VisionEncoderConfig):
        super().__init__()
        self.config = config
        patch_dim = config.num_channels * config.patch_size * config.patch_size
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size)
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)
        self.grid_side = int(round(config.num_patches ** 0.5))
        if self.grid_side * self.grid_side != config.num_patches:
            raise ValueError(f"num_patches={config.num_patches} is not a square grid")

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[tuple[int, int, int]]) -> torch.Tensor:
        weight = self.patch_embedding.weight
        embeds = self.patch_embedding(pixel_values.to(weight.dtype))
        grid = self.position_embedding.weight.reshape(self.grid_side, self.grid_side, -1)
        pos = resize_position_grid(grid, grid_thw, self.config.spatial_merge_size)
        if pos.shape[0] != embeds.shape[0]:
            raise ValueError(
                f"packed patch count {embeds.shape[0]} does not match grid_thw {grid_thw}"
            )
        return embeds + pos.to(embeds.dtype)


def frame_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, frame_lens: list[int],
) -> torch.Tensor:
    """Non-causal attention within each frame of a packed ``[N, H, D]`` batch.

    ``frame_lens`` lists the consecutive frame token counts. Frames of equal
    length are batched into one SDPA call; the outputs come back in pack
    order."""
    if len(set(frame_lens)) == 1:
        n, length = len(frame_lens), frame_lens[0]
        qb = q.view(n, length, *q.shape[1:]).transpose(1, 2)
        kb = k.view(n, length, *k.shape[1:]).transpose(1, 2)
        vb = v.view(n, length, *v.shape[1:]).transpose(1, 2)
        out = F.scaled_dot_product_attention(qb, kb, vb)
        return out.transpose(1, 2).reshape(q.shape)
    outputs = [None] * len(frame_lens)
    offsets = [0]
    for length in frame_lens:
        offsets.append(offsets[-1] + length)
    by_len: dict[int, list[int]] = {}
    for i, length in enumerate(frame_lens):
        by_len.setdefault(length, []).append(i)
    for length, idxs in by_len.items():
        sel = torch.cat([torch.arange(offsets[i], offsets[i] + length, device=q.device) for i in idxs])
        out = frame_attention(q[sel], k[sel], v[sel], [length] * len(idxs))
        for j, i in enumerate(idxs):
            outputs[i] = out[j * length:(j + 1) * length]
    return torch.cat(outputs, dim=0)


class Cosmos3VisionAttention(nn.Module):
    def __init__(self, config: Cosmos3VisionEncoderConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        if self.head_dim * self.num_heads != dim:
            raise ValueError(f"hidden_size {dim} is not divisible by {self.num_heads} heads")
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, frame_lens: list[int]) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_heads, self.head_dim)
        out = frame_attention(q, k, v, frame_lens).reshape(n, -1)
        return self.out_proj(out)


class Cosmos3VisionMLP(nn.Module):
    def __init__(self, config: Cosmos3VisionEncoderConfig):
        super().__init__()
        self.act = _activation(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Cosmos3VisionEncoderLayer(nn.Module):
    def __init__(self, config: Cosmos3VisionEncoderConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Cosmos3VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Cosmos3VisionMLP(config)

    def forward(self, x: torch.Tensor, frame_lens: list[int]) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), frame_lens)
        return x + self.mlp(self.layer_norm2(x))


class Cosmos3VisionEncoder(nn.Module):
    def __init__(self, config: Cosmos3VisionEncoderConfig):
        super().__init__()
        self.layers = nn.ModuleList(Cosmos3VisionEncoderLayer(config) for _ in range(config.num_hidden_layers))

    def forward(self, x: torch.Tensor, frame_lens: list[int]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, frame_lens)
        return x


class Cosmos3VisionTransformer(nn.Module):
    """``visual``: embeddings -> encoder -> post LayerNorm, over packed patches."""

    def __init__(self, config: Cosmos3VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = Cosmos3VisionEmbeddings(config)
        self.encoder = Cosmos3VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[tuple[int, int, int]]) -> torch.Tensor:
        frame_lens = [h * w for t, h, w in grid_thw for _ in range(t)]
        x = self.embeddings(pixel_values, grid_thw)
        x = self.encoder(x, frame_lens)
        return self.post_layernorm(x)


class Cosmos3PatchMerger(nn.Module):
    """``projector``: per-patch LayerNorm, 2x2 merge into the channel dim,
    Linear -> GELU -> Linear into the text hidden size."""

    def __init__(self, config: Cosmos3ReasonerConfig):
        super().__init__()
        self.spatial_merge_size = config.vision.spatial_merge_size
        self.input_hidden_size = config.projector_input_hidden_size
        self.hidden_size = self.input_hidden_size * self.spatial_merge_size ** 2
        self.use_postshuffle_norm = config.use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if self.use_postshuffle_norm else self.input_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, config.projector_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.projector_hidden_size, config.projector_out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.spatial_merge_size ** 2, self.input_hidden_size)
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Cosmos3VisionModel(nn.Module):
    """Vision tower + projector: packed pixel patches -> text-space tokens.

    ``pixel_values`` is ``[total_patches, 3 * patch_size**2]`` in the
    processor's block-major order (see ``components.reasoner``), ``grid_thw``
    one ``(t, h, w)`` patch grid per image/video. Returns
    ``[total_patches // merge_size**2, text_hidden]``: one token per merged
    2x2 block, in the order the ``<|image_pad|>`` / ``<|video_pad|>`` tokens
    appear in the prompt."""

    def __init__(self, config: Cosmos3ReasonerConfig):
        super().__init__()
        self.config = config
        self.visual = Cosmos3VisionTransformer(config.vision)
        self.projector = Cosmos3PatchMerger(config)

    @property
    def tokens_per_patch_block(self) -> int:
        return self.config.vision.spatial_merge_size ** 2

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[tuple[int, int, int]]) -> torch.Tensor:
        features = self.visual(pixel_values, grid_thw)
        return self.projector(features)
