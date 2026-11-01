"""MiniMax-H3's video VAE decoder: a ViT over latent voxels (36 blocks, register tokens),
decoded in clips of 5 latents overlapping by 2 and in 256-px tiles overlapping by at
least 64, cross-faded and blended as the reference does. Ported from diffusers'
``autoencoder_kl_minimax_h3.py``; weights from ``FL2VA/video_vae/source``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.minimax_h3.config import H3VideoVAEConfig

# (q, k, v) -> out over [B, N, H, D]
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def vae_rope_inv_freq(rot_dim: int, theta: float, num_axes: int = 3) -> torch.Tensor:
    """``MiniMaxH3VideoRotaryPosEmbed.inv_freq``: ``rot_dim / (2 * axes)`` frequencies, on
    the CPU (the module may be built under a meta device)."""
    return 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / rot_dim, dtype=torch.float32, device="cpu")


def vae_rope_table(frames: int, height: int, width: int, num_suffix: int, inv_freq: torch.Tensor, device):
    """cos / sin ``[N + suffix, 1, rot_dim]`` fp32 of one tile: per axis, voxel centres in
    [-1, 1), times ``2 pi * inv_freq``, the three axes concatenated and doubled; the
    suffix tokens sit at 0."""
    grids = [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=device) / size) - 1.0
        for size in (frames, height, width)
    ]
    pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    pos = torch.cat([pos, pos.new_zeros(num_suffix, 3)])
    angles = 2.0 * math.pi * pos[:, :, None] * inv_freq.to(device)[None, None, :]
    angles = angles.flatten(1, 2).tile(2).unsqueeze(1)
    return angles.cos(), angles.sin()


class H3VaeAttention(nn.Module):
    """q/k/v in one GEMM (per-head ``[q k v]`` rows, the checkpoint's layout), q/k RMSNorm
    without affine in fp32, partial rope, attention, output GEMM; with biases."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.heads, self.head_dim, self.eps = heads, head_dim, eps
        self.to_qkv = nn.Linear(dim, 3 * heads * head_dim)
        self.to_out = nn.Linear(heads * head_dim, dim)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], attend: Attend) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, n, self.heads, 3 * self.head_dim).chunk(3, dim=-1)
        dtype = q.dtype
        q = F.rms_norm(q.float(), (self.head_dim,), eps=self.eps).to(dtype)
        k = F.rms_norm(k.float(), (self.head_dim,), eps=self.eps).to(dtype)
        cos, sin = (t.to(dtype) for t in rope)
        rot = cos.shape[-1]

        def rotate(t):
            t_rot, t_pass = t[..., :rot], t[..., rot:]
            first, second = t_rot.chunk(2, dim=-1)
            return torch.cat([t_rot * cos + torch.cat([-second, first], dim=-1) * sin, t_pass], dim=-1)

        out = attend(rotate(q), rotate(k), v.contiguous())
        return self.to_out(out.flatten(2, 3))


class H3VaeFeedForward(nn.Module):
    """SwiGLU with biases; ``w1`` is ``[gate; value]``."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.inner = dim * mult
        self.w1 = nn.Linear(dim, 2 * self.inner)
        self.w2 = nn.Linear(self.inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w1(x).split([self.inner, self.inner], dim=-1)
        return self.w2(value * F.silu(gate))


class H3VaeBlock(nn.Module):
    """Pre-norm block with fp32 norms and fp32 per-channel residual scales: after the first
    block the residual stream is fp32, as in the reference."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = H3VaeAttention(dim, heads, head_dim, eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = H3VaeFeedForward(dim)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, rope, attend: Attend) -> torch.Tensor:
        dtype = self.attn.to_qkv.weight.dtype
        x = x + self.attn(self.norm1(x.float()).to(dtype), rope, attend) * self.scale1
        return x + self.ff(self.norm2(x.float()).to(dtype)) * self.scale2


class H3ViTDecoder(nn.Module):
    """``[B, C, F, H, W]`` latents (one tile) -> ``[B, 3, 4F, 16H, 16W]`` pixels."""

    def __init__(self, cfg: H3VideoVAEConfig, eps: float = 1e-5, num_register_tokens: int = 4, out_channels: int = 3):
        super().__init__()
        dim = cfg.decoder_heads * cfg.decoder_head_dim
        self.out_channels, self.ps, self.pt = out_channels, cfg.spatial_compression, cfg.temporal_compression
        self.num_register_tokens = num_register_tokens
        self.x_embedder = nn.Linear(cfg.latent_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            H3VaeBlock(dim, cfg.decoder_heads, cfg.decoder_head_dim, eps) for _ in range(cfg.decoder_layers)
        )
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * self.pt * self.ps * self.ps)
        self.rope_inv_freq = vae_rope_inv_freq(int(cfg.decoder_head_dim * cfg.decoder_rope_dim_ratio),
                                               cfg.decoder_rope_theta)

    def forward(self, z: torch.Tensor, attend: Attend) -> torch.Tensor:
        b, c, f, h, w = z.shape
        x = self.x_embedder(z.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c))
        n = x.shape[1]
        suffix = [self.register_tokens.expand(b, -1, -1).to(x.dtype), torch.zeros_like(x[:, :1])]
        x = torch.cat([x, *suffix], dim=1)
        rope = vae_rope_table(f, h, w, self.num_register_tokens + 1, self.rope_inv_freq, z.device)
        for block in self.transformer_blocks:
            x = block(x, rope, attend)
        x = self.proj_out(self.norm_out(x).to(self.proj_out.weight.dtype))[:, :n]
        x = x.view(b, f, h, w, self.out_channels, self.pt, self.ps, self.ps)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(b, self.out_channels, f * self.pt, h * self.ps, w * self.ps)


# ---- spatial tiles, shared by the decoder and the keyframe encoder (the reference's
# ``_split_tiles`` / ``_blend`` / ``_stitch_tiles``)
def split_tiles(length: int, tile: int, min_overlap: int, step: int) -> tuple[list[int], list[int], list[int]]:
    """Tile starts, lengths and overlaps covering ``length`` exactly: overlaps of at least
    ``min_overlap``, widened in whole ``step``s (the latent stride) until the tiles fit."""
    if tile >= length:
        return [0], [length], []
    num = math.ceil(length / tile)
    while tile * num - min_overlap * (num - 1) - length < 0:
        num += 1
    overlaps = [min_overlap] * (num - 1)
    remaining = tile * num - sum(overlaps) - length
    for i in range(remaining // step):
        overlaps[i % (num - 1)] += step
    starts = [0]
    for i in range(num - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, [tile] * num, overlaps


def blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    """``b`` with its first ``extent`` slices cross-faded linearly from ``a``'s last ones."""
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a, weight_b = (1 - positions / extent).view(shape), (positions / extent).view(shape)
    blended = a.narrow(dim, a.shape[dim] - extent, extent) * weight_a + b.narrow(dim, 0, extent) * weight_b
    if extent == b.shape[dim]:
        return blended
    return torch.cat([blended, b.narrow(dim, extent, b.shape[dim] - extent)], dim=dim)


def stitch_tiles(tiles: list[list[torch.Tensor]], h_overlaps: list[int], w_overlaps: list[int]) -> torch.Tensor:
    """A row-major grid of tiles, blended over their overlaps and cropped to one tensor."""
    rows = []
    for i, row in enumerate(tiles):
        out_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(tiles[i - 1][j], tile, h_overlaps[i - 1], dim=-2)
            if j > 0:
                tile = blend(row[j - 1], tile, w_overlaps[j - 1], dim=-1)
            if i < len(tiles) - 1:
                tile = tile[..., : -h_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -w_overlaps[j]]
            out_row.append(tile)
        rows.append(torch.cat(out_row, dim=-1))
    return torch.cat(rows, dim=-2)


class ClipWalk(NamedTuple):
    """The temporal clip walk's geometry (the encoder's ``clip_length`` / ``token_drop``):
    clips of ``chunk`` latents overlapping by ``token_overlap``, cross-faded over
    ``frame_overlap`` frames, the first ``frame_pre_padding`` decoded frames of each dropped."""

    chunk: int
    token_drop: int
    token_overlap: int
    temporal: int
    frame_pre_padding: int
    frame_overlap: int
    clip_length: int

    @classmethod
    def of(cls, cfg: H3VideoVAEConfig) -> "ClipWalk":
        tr = cfg.temporal_compression
        pre = (-cfg.clip_length) % tr
        chunk = math.ceil(cfg.clip_length / tr)
        overlap = (-cfg.token_drop) % chunk
        return cls(chunk, cfg.token_drop, overlap, tr, pre, max(overlap * tr - pre, 0), cfg.clip_length)

    def pad_tokens(self, num_latents: int) -> int:
        return (-(num_latents + self.token_drop)) % self.chunk

    def num_clips(self, num_latents: int) -> int:
        padded = num_latents + self.token_drop + self.pad_tokens(num_latents)
        return padded // self.chunk - int(self.token_drop > 0)

    def assemble(self, clips: list[torch.Tensor], num_latents: int) -> torch.Tensor:
        """The clips' temporal cross-fade, then the frames the padding added cut off again
        (``num_latents`` before padding)."""
        chunk_frames = self.chunk * self.temporal
        decoded, overlap = [], None
        for clip in clips:
            for j in range(int(self.token_drop > 0) + 1):
                piece = clip[:, :, j * chunk_frames: (j + 1) * chunk_frames][:, :, self.frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        piece = blend(overlap, piece, self.frame_overlap, dim=-3)
                    decoded.append(piece)
                else:
                    overlap = piece
        if overlap is not None:
            decoded.append(overlap)
        out = torch.cat(decoded, dim=2)
        pad_tokens = self.pad_tokens(num_latents)
        if pad_tokens:
            tail = self.clip_length % self.temporal
            pad_frames = sum(
                tail if tail and (num_latents + k) % self.chunk == 0 else self.temporal for k in range(pad_tokens)
            )
            out = out[:, :, :-pad_frames]
        return out


class H3VideoDecoder(nn.Module):
    """``post_quant_conv`` + the ViT decoder, and the reference's clip / tile walk."""

    def __init__(self, cfg: H3VideoVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.post_quant_conv = nn.Conv3d(cfg.latent_channels, cfg.latent_channels, kernel_size=1)
        self.decoder = H3ViTDecoder(cfg)
        self.ratio = cfg.spatial_compression
        self.walk = ClipWalk.of(cfg)

    def keep_fp32(self) -> None:
        """The modules the reference keeps fp32 under a bf16 load (diffusers matches its
        ``_keep_in_fp32_modules`` per name component): the norms and the residual scales."""
        for module in self.decoder.modules():
            if isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                module.float()
        for block in self.decoder.transformer_blocks:
            block.scale1 = nn.Parameter(block.scale1.float())
            block.scale2 = nn.Parameter(block.scale2.float())

    def split_tiles(self, length: int) -> tuple[list[int], list[int], list[int]]:
        return split_tiles(length, self.cfg.tile_size, self.cfg.tile_overlap_min, self.ratio)

    def tile_slices(self, height: int, width: int) -> tuple[list[list[tuple[slice, slice]]], list[int], list[int]]:
        """The latent slices of a clip's spatial tiles (row-major) and the pixel overlaps."""
        r = self.ratio
        ys, y_lens, y_overlaps = self.split_tiles(height * r)
        xs, x_lens, x_overlaps = self.split_tiles(width * r)
        grid = [[(slice(y // r, (y + ylen) // r), slice(x // r, (x + xlen) // r))
                 for x, xlen in zip(xs, x_lens, strict=True)] for y, ylen in zip(ys, y_lens, strict=True)]
        return grid, y_overlaps, x_overlaps

    def decode_tiles(self, tiles: list[torch.Tensor], attend: Attend, batch: int) -> list[torch.Tensor]:
        """Decode same-shape tiles ``batch`` at a time. One at a time (``batch = 1``) is the
        reference's order of work; batched, the GEMMs run at other shapes (cuBLAS may then
        round differently) and the tile loop stops being launch-bound."""
        out: list[torch.Tensor] = []
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            decoded = self.decoder(self.post_quant_conv(torch.cat(chunk)), attend)
            out.extend(decoded.split([t.shape[0] for t in chunk]))
        return out

    def clip_plan(self, z: torch.Tensor) -> tuple[torch.Tensor, int]:
        """The temporal walk's input: the latents in the decoder's dtype, padded by repeating
        the last frame to whole chunks; and the clip count."""
        z = z.to(self.post_quant_conv.weight.dtype)
        pad_tokens = self.walk.pad_tokens(z.shape[2])
        num_clips = self.walk.num_clips(z.shape[2])
        if pad_tokens:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        return z, num_clips

    def decode_clips(self, z: torch.Tensor, clips: range, attend: Attend, tile_batch: int = 1) -> list[torch.Tensor]:
        """Clips ``clips`` of padded latents ``z`` (``clip_plan``), each spatially stitched:
        every clip's tiles are cut first and decoded ``tile_batch`` at a time."""
        chunk = self.walk.chunk
        grid, y_overlaps, x_overlaps = self.tile_slices(z.shape[-2], z.shape[-1])
        windows = [z[:, :, i * chunk: i * chunk + chunk + self.walk.token_overlap] for i in clips]
        tiles = [w[..., ys, xs] for w in windows for row in grid for ys, xs in row]
        decoded_tiles = iter(self.decode_tiles(tiles, attend, max(1, int(tile_batch))))
        return [
            stitch_tiles([[next(decoded_tiles) for _ in row] for row in grid], y_overlaps, x_overlaps)
            for _ in windows
        ]

    def decode(self, z: torch.Tensor, attend: Attend, tile_batch: int = 1) -> torch.Tensor:
        """``AutoencoderKLMiniMaxH3._decode``: denormalized latents ``[B, C, F, H, W]`` ->
        pixels in the VAE's ImageNet-normalized space: every clip decoded, then the
        temporal cross-fade."""
        num_latents = z.shape[2]
        z, num_clips = self.clip_plan(z)
        return self.walk.assemble(self.decode_clips(z, range(num_clips), attend, tile_batch), num_latents)
