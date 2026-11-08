"""DeepEncoder: SAM ViT-B/16 into CLIP-L/14, then a linear projector.

A port of DeepSeek-OCR's ``deepencoder.py``. SAM's 16x-downsampled feature map
(1024 channels) is CLIP's patch embedding (CLIP's own patch conv is never run),
and each token's CLIP and SAM features are concatenated and projected to the
decoder's width. Parameter paths match the checkpoint's ``model.`` subtree.

Attention runs on SDPA rather than an attention resource: SAM adds a
decomposed relative-position bias no FlashInfer kernel takes, and CLIP sees a
fixed 1 + N^2 tokens per view, bidirectional, with no cache.

The op order follows the reference under bf16 autocast, which is where its
numerics come from (LayerNorm outputs, and so CLIP's residual stream, are fp32).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

SAM_DIM, SAM_DEPTH, SAM_HEADS, SAM_WINDOW = 768, 12, 12, 14
SAM_GLOBAL_LAYERS = (2, 5, 8, 11)
SAM_PATCH, SAM_IMG = 16, 1024
CLIP_DIM, CLIP_DEPTH, CLIP_HEADS, CLIP_FFN = 1024, 24, 16, 4096
CLIP_POSITIONS = (224 // 14) ** 2 + 1


def _resize_pos_2d(pos: torch.Tensor, side: int) -> torch.Tensor:
    """``(1, S, S, C)`` absolute positions bicubically resized to ``side``."""
    if pos.shape[1] == side:
        return pos
    resized = F.interpolate(
        pos.permute(0, 3, 1, 2).float(), size=(side, side), mode="bicubic", antialias=True, align_corners=False,
    )
    return resized.to(pos.dtype).permute(0, 2, 3, 1)


def _rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Relative position embeddings for every (query, key) offset (SAM's ``get_rel_pos``)."""
    span = 2 * max(q_size, k_size) - 1
    if rel_pos.shape[0] != span:
        resized = F.interpolate(rel_pos.float().reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                                size=span, mode="linear")
        rel_pos = resized.to(rel_pos.dtype).reshape(-1, span).permute(1, 0)
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[relative.long()]


class SamAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.num_heads = SAM_HEADS
        head_dim = SAM_DIM // SAM_HEADS
        self.qkv = nn.Linear(SAM_DIM, SAM_DIM * 3)
        self.proj = nn.Linear(SAM_DIM, SAM_DIM)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        heads = self.num_heads
        qkv = self.qkv(x).reshape(batch, height * width, 3, heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, batch * heads, height * width, -1).unbind(0)
        # decomposed relative positions (MViTv2), as an additive attention bias
        rh, rw = _rel_pos(height, height, self.rel_pos_h), _rel_pos(width, width, self.rel_pos_w)
        rq = q.reshape(batch * heads, height, width, -1)
        rel_h = torch.einsum("bhwc,hkc->bhwk", rq, rh).reshape(batch, heads, height * width, height, 1)
        rel_w = torch.einsum("bhwc,wkc->bhwk", rq, rw).reshape(batch, heads, height * width, 1, width)
        bias = (rel_h + rel_w).view(batch, heads, height * width, height * width)
        q, k, v = (t.view(batch, heads, height * width, -1) for t in (q, k, v))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x.view(batch, heads, height, width, -1).permute(0, 2, 3, 1, 4).reshape(batch, height, width, -1)
        return self.proj(x)


class SamMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(SAM_DIM, SAM_DIM * 4)
        self.lin2 = nn.Linear(SAM_DIM * 4, SAM_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(F.gelu(self.lin1(x)))


def _window_partition(x: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, height, width, channels = x.shape
    pad_h, pad_w = (window - height % window) % window, (window - width % window) % window
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = height + pad_h, width + pad_w
    x = x.view(batch, hp // window, window, wp // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels), (hp, wp)


def _window_unpartition(windows: torch.Tensor, window: int, padded: tuple[int, int], size: tuple[int, int]):
    (hp, wp), (height, width) = padded, size
    batch = windows.shape[0] // (hp * wp // window // window)
    x = windows.view(batch, hp // window, wp // window, window, window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, hp, wp, -1)
    return x[:, :height, :width, :].contiguous() if (hp > height or wp > width) else x


class SamBlock(nn.Module):
    def __init__(self, window: int):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(SAM_DIM, eps=1e-6)
        self.attn = SamAttention(window if window else SAM_IMG // SAM_PATCH)
        self.norm2 = nn.LayerNorm(SAM_DIM, eps=1e-6)
        self.mlp = SamMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window:
            size = x.shape[1], x.shape[2]
            x, padded = _window_partition(x, self.window)
        x = self.attn(x)
        if self.window:
            x = _window_unpartition(x, self.window, padded, size)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class LayerNorm2d(nn.Module):
    """Channel LayerNorm on NCHW, written out as SAM does (its autocast dtypes included)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, SAM_DIM, kernel_size=SAM_PATCH, stride=SAM_PATCH)


class SamEncoder(nn.Module):
    """SAM ViT-B image encoder plus DeepSeek's two stride-2 convs: (B, 3, S, S) -> (B, 1024, S/64, S/64)."""

    def __init__(self):
        super().__init__()
        side = SAM_IMG // SAM_PATCH
        self.patch_embed = _PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, side, side, SAM_DIM))
        self.blocks = nn.ModuleList(
            SamBlock(0 if i in SAM_GLOBAL_LAYERS else SAM_WINDOW) for i in range(SAM_DEPTH)
        )
        self.neck = nn.Sequential(
            nn.Conv2d(SAM_DIM, 256, kernel_size=1, bias=False), LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), LayerNorm2d(256),
        )
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed.proj(x).permute(0, 2, 3, 1)
        x = x + _resize_pos_2d(self.pos_embed, x.shape[1])
        for block in self.blocks:
            x = block(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return self.net_3(self.net_2(x))


class ClipEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(CLIP_DIM))
        # Loaded but never run: SAM's feature map is the patch embedding.
        self.patch_embedding = nn.Conv2d(3, CLIP_DIM, kernel_size=14, stride=14, bias=False)
        self.position_embedding = nn.Embedding(CLIP_POSITIONS, CLIP_DIM)

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(patch_embeds.shape[0], 1, -1)
        x = torch.cat([cls, patch_embeds], dim=1)
        pos = self.position_embedding.weight
        src, tgt = int(math.sqrt(pos.shape[0] - 1)), int(math.sqrt(x.shape[1] - 1))
        if src != tgt:
            grid = _resize_pos_2d(pos[1:].view(1, src, src, CLIP_DIM), tgt).reshape(tgt * tgt, CLIP_DIM)
            pos = torch.cat([pos[:1], grid], dim=0)
        return x + pos[None]


class ClipAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = CLIP_HEADS
        self.qkv_proj = nn.Linear(CLIP_DIM, CLIP_DIM * 3)
        self.out_proj = nn.Linear(CLIP_DIM, CLIP_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.qkv_proj(x).view(batch, seq, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.permute(0, 2, 1, 3).reshape(batch, seq, -1))


class ClipMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CLIP_DIM, CLIP_FFN)
        self.fc2 = nn.Linear(CLIP_FFN, CLIP_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        # quick_gelu in fp32, rounded once: the reference's is a TorchScript
        # function, which its fuser runs in fp32 after the first calls
        y = x.float()
        return self.fc2((y * torch.sigmoid(1.702 * y)).to(x.dtype))


class ClipLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(CLIP_DIM, eps=1e-5)
        self.self_attn = ClipAttention()
        self.layer_norm2 = nn.LayerNorm(CLIP_DIM, eps=1e-5)
        self.mlp = ClipMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


class _ClipTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(ClipLayer() for _ in range(CLIP_DEPTH))


class ClipEncoder(nn.Module):
    """CLIP-L over SAM's feature map: (B, 1024, n, n) -> (B, 1 + n^2, 1024), class token first."""

    def __init__(self):
        super().__init__()
        self.embeddings = ClipEmbeddings()
        self.pre_layrnorm = nn.LayerNorm(CLIP_DIM, eps=1e-5)
        self.transformer = _ClipTransformer()

    def forward(self, sam_features: torch.Tensor) -> torch.Tensor:
        x = self.pre_layrnorm(self.embeddings(sam_features))
        for layer in self.transformer.layers:
            x = layer(x)
        return x


class _Projector(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.layers = nn.Linear(input_dim, dim)


class DeepEncoder(nn.Module):
    """Views to decoder-width rows, laid out as the decoder's image tokens.

    Each image is its tiled local view (if any) row by row, each row ending in
    ``image_newline``, then its global view likewise, then ``view_seperator``.
    """

    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.sam_model = SamEncoder()
        self.vision_model = ClipEncoder()
        self.projector = _Projector(input_dim, dim)
        self.image_newline = nn.Parameter(torch.zeros(dim))
        self.view_seperator = nn.Parameter(torch.zeros(dim))

    def encode_views(self, views: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) -> (B, (S/64)^2, dim) projected token features."""
        sam = self.sam_model(views)
        clip = self.vision_model(sam)
        features = torch.cat((clip[:, 1:], sam.flatten(2).permute(0, 2, 1)), dim=-1)
        return self.projector.layers(features)

    def _with_newlines(self, grid: torch.Tensor) -> torch.Tensor:
        """(H, W, dim) -> (H * (W + 1), dim)."""
        height, _, dim = grid.shape
        newline = self.image_newline[None, None, :].expand(height, 1, dim)
        return torch.cat([grid, newline], dim=1).view(-1, dim)

    def assemble(
        self, global_tokens: torch.Tensor, local_tokens: torch.Tensor | None, cols: int, rows: int,
    ) -> torch.Tensor:
        """One image's rows from its encoded global view ``(n^2, dim)`` and tiles ``(rows * cols, m^2, dim)``."""
        dim = global_tokens.shape[-1]
        side = math.isqrt(global_tokens.shape[0])
        parts = []
        if local_tokens is not None:
            tile = math.isqrt(local_tokens.shape[1])
            grid = (local_tokens.view(rows, cols, tile, tile, dim).permute(0, 2, 1, 3, 4)
                    .reshape(rows * tile, cols * tile, dim))
            parts.append(self._with_newlines(grid))
        parts.append(self._with_newlines(global_tokens.view(side, side, dim)))
        parts.append(self.view_seperator[None, :])
        return torch.cat(parts, dim=0)
