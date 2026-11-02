# Copyright (c) 2026 Robbyant Team.
# Copyright (c) 2024-2025 The Alibaba Wan Team Authors.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: reimplemented as an M* graph-node
# module with the same numerics, camera conditioning only.
#
# Original files (Robbyant/lingbot-world, wan/modules/model_fast.py and
# wan/modules/model.py) were released under Apache-2.0, with the full license
# text available at https://github.com/Robbyant/lingbot-world/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

"""LingBot-World-V2 causal-fast DiT: a Wan2.1-layout I2V transformer with
Plucker camera injection, run chunk by chunk against a rolling KV cache.

Port of LingBot-World v1's ``WanModelFast`` (Robbyant/lingbot-world,
Apache-2.0), which V2's causal-fast checkpoints run unchanged. v1 takes the
rolling KV cache with attention sinks from guandeh17/Self-Forcing and the
transformer blocks from Wan-Video/Wan2.1 (both Apache-2.0). Parameter names are the checkpoint's
(original Wan layout: ``self_attn.q``, ``ffn.0``, ``modulation``, ``norm3`` is
the cross-attention norm), so a state dict loads with no renaming.

Numerics contract: the reference runs the whole DiT under
``torch.amp.autocast("cuda", dtype=bfloat16)`` on bf16 weights, with fp32
autocast islands around the modulation and the residual gates, RoPE in
complex128, and FlashAttention-2 varlen for both attentions. This module enters
the same contexts itself (``LingBotCausalDiT.forward``) in the same order. Its
two attentions are whatever ``DitAttention`` the caller hands it: served, the
engine's ragged attention resource; without one, the reference's FA2 varlen
(``REFERENCE_ATTENTION``), with which the forward is bit-identical to the
reference's on the same inputs. The other differences from the reference are
structural only: batch 1 without the list wrapping, and the rolling K/V held
by the engine's rolling KV store, which keeps the reference's layout (sink,
then the latest window, compacted by the same shift-left eviction) with its
indices planned once per step on the host instead of per layer from
1-element device tensors read back with ``.item()``.

``wan22.components.dit`` is not reused: it is the diffusers layout and its
numerics (explicit ``type_as`` casts, SDPA, fp32-island weights) are the
diffusers pipeline's, not this checkpoint's reference.
"""

import math
from collections.abc import Callable
from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from mstar.model.lingbot_world.config import LingBotWorldConfig


def require_flash_attn() -> None:
    """Fail the node build, not the first request, when flash-attn is missing."""
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "LingBot-World's reference attention is FlashAttention-2 (flash_attn_varlen_func), as the "
            "reference runs; install flash-attn (docs/installation.rst, 'flash-attn (Qwen3-Omni, "
            "LingBot-World)') or serve without reference_numerics."
        ) from e


def _flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The reference ``flash_attention`` for one unpadded sequence: q/k/v
    ``[1, L, H, D]`` bf16 through FA2 varlen; returns ``[1, Lq, H, D]``."""
    import flash_attn

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)
    q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
    cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
    x = flash_attn.flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_q.to(q.device, non_blocking=True),
        cu_seqlens_k=cu_k.to(q.device, non_blocking=True),
        max_seqlen_q=lq, max_seqlen_k=lk,
        dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
    ).unflatten(0, (b, lq))
    return x.type(out_dtype)


# ``(q, k, v) -> out`` on ``[1, L, H, D]`` tensors; ``k`` / ``v`` have their own length.
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class DitAttention(NamedTuple):
    """The DiT's two attentions: the chunk's queries over its rolling K/V
    window, and over the text K/V."""

    window: Attend
    text: Attend


REFERENCE_ATTENTION = DitAttention(window=_flash_attention, text=_flash_attention)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    freqs = torch.outer(
        torch.arange(max_seq_len, device="cpu"),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device="cpu").to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_table(freqs: torch.Tensor, grid: tuple[int, int, int], start_frame: int) -> torch.Tensor:
    """The 3D RoPE rotations of a chunk's tokens, ``[f*h*w, 1, head_dim // 2]``
    complex128, with the temporal axis offset to ``start_frame`` (the chunk's
    absolute latent frame)."""
    c = freqs.size(1)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid
    return torch.cat([
        freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)


def causal_rope_apply(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """RoPE in complex128 on ``x`` ``[1, L, H, D]`` by a ``rope_table``."""
    seq_len, n = table.size(0), x.size(2)
    x_i = torch.view_as_complex(x[0, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
    x_i = torch.view_as_real(x_i * table).flatten(2)
    if seq_len < x.size(1):
        x_i = torch.cat([x_i, x[0, seq_len:]])
    return x_i.unsqueeze(0).type_as(x)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


def attended_frames(config: LingBotWorldConfig, chunk: int) -> int:
    """Latent frames of rolling K/V that chunk ``chunk``'s queries attend, its
    own included: the sink and the latest window once the chunk is written."""
    frames = (chunk + 1) * config.chunk_size
    return min(frames, config.local_attn_size) if config.local_attn_size > -1 else frames


class WindowStore(Protocol):
    """Where the rolling self-attention K/V live: the engine's rolling KV store
    (``RollingKVManager``) when served, a scratch one at warmup. ``write`` puts
    the chunk's ``[1, L, H, D]`` K/V in layer ``layer``; ``window`` returns
    what the chunk attends there, the sink and the latest window ending with
    the chunk, in stream order."""

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None: ...

    def window(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class ChunkOnlyStore:
    """A ``WindowStore`` that keeps nothing: each layer attends the chunk alone,
    as chunk 0 does. For tracing the compiled blocks at load."""

    def __init__(self):
        self._kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._kv[layer] = (k, v)

    def window(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._kv[layer]


class ChunkWindow:
    """One forward's window attention, block after block: write the chunk's K/V
    into the store at the next layer, then attend over that layer's window.
    Eager inside a compiled block, and handed to it as one object that counts
    the layers itself: Dynamo guards on integer arguments, so a layer index
    passed in would recompile every block for every layer."""

    def __init__(self, attend: Attend, store: WindowStore, num_layers: int):
        self.attend = attend
        self.store = store
        self.num_layers = num_layers
        self.layer = 0

    @torch.compiler.disable
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        layer = self.layer
        if layer >= self.num_layers:
            raise RuntimeError(f"ChunkWindow called past its {self.num_layers} layers in one forward")
        self.layer += 1
        self.store.write(layer, k, v)
        return self.attend(q, *self.store.window(layer))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, window: ChunkWindow) -> torch.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        roped_query = causal_rope_apply(q, rope).type_as(v)
        roped_key = causal_rope_apply(k, rope).type_as(v)
        x = window(roped_query, roped_key, v)
        return self.o(x.flatten(2))


class CrossAttention(nn.Module):
    """Cross-attention to the 512-token text stream. The text K/V are computed
    on a request's first forward and reused for the rest of its rollout."""

    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def text_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, d = context.size(0), self.num_heads, self.head_dim
        return self.norm_k(self.k(context)).view(b, -1, n, d), self.v(context).view(b, -1, n, d)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attend: Attend) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        return self.o(attend(q, k, v).flatten(2))


class CausalBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalSelfAttention(dim, num_heads, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def cam_modulation(self, cam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The block's camera (scale, shift): a function of the chunk's rays alone."""
        cam_hidden = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(cam)))
        cam_hidden = cam_hidden + cam
        return self.cam_scale_layer(cam_hidden), self.cam_shift_layer(cam_hidden)

    def forward(
        self, x: torch.Tensor, e: torch.Tensor, cam: tuple[torch.Tensor, torch.Tensor], cross_k: torch.Tensor,
        cross_v: torch.Tensor, rope: torch.Tensor, window: ChunkWindow, text_attend: Attend,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        y = self.self_attn(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2), rope, window)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        cam_scale, cam_shift = cam
        x = (1.0 + cam_scale) * x + cam_shift

        x = x + self.cross_attn(self.norm3(x), cross_k, cross_v, text_attend)
        y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            return self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))


class ChunkConditioning(NamedTuple):
    """What a chunk's forwards (its denoise steps and the commit) share: each
    block's camera (scale, shift) from the chunk's Plucker rays, and the RoPE
    table at the chunk's latent frames."""

    cam: list[tuple[torch.Tensor, torch.Tensor]]
    rope: torch.Tensor


class LingBotCausalDiT(nn.Module):
    def __init__(self, config: LingBotWorldConfig):
        super().__init__()
        self.config = config
        dim, patch = config.dim, config.patch_size
        self.patch_embedding = nn.Conv3d(config.in_dim, dim, kernel_size=patch, stride=patch)
        self.patch_embedding_wancamctrl = nn.Linear(config.control_dim * 64 * math.prod(patch), dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(config.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            CausalBlock(dim, config.ffn_dim, config.num_heads, config.eps) for _ in range(config.num_layers)
        )
        self.head = CausalHead(dim, config.out_dim, patch, config.eps)
        # Derived complex128 tables, not checkpoint state: kept off the module
        # tree so neither the loader nor ``.to(dtype)`` touches them.
        d, n = config.head_dim, config.rope_max_frames
        self._freqs_cpu = torch.cat([
            rope_params(n, d - 4 * (d // 6)),
            rope_params(n, 2 * (d // 6)),
            rope_params(n, 2 * (d // 6)),
        ], dim=1)
        self._freqs: dict[torch.device, torch.Tensor] = {}

    def compile_blocks(self, autotune: bool = False, **compile_kwargs) -> None:
        """Compile each block's forward in place (``compile_utils.compile_blocks``:
        deterministic unless ``autotune``). The K/V write and window attention
        (``ChunkWindow``) and a ragged text attention stay eager, so a block
        traces once for every chunk; the reference's text attention
        (``REFERENCE_ATTENTION``) is not disabled and compiles with the block."""
        from mstar.model.components.diffusion.compile_utils import compile_blocks

        compile_blocks(self.blocks, autotune=autotune, **compile_kwargs)

    def freqs(self, device: torch.device) -> torch.Tensor:
        if device not in self._freqs:
            self._freqs[device] = self._freqs_cpu.to(device)
        return self._freqs[device]

    def text_kv(self, context: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Every block's cross-attention K/V for ``context`` ``[L, 4096]``, zero-padded
        to ``text_len``: computed on a request's first chunk and reused for its rollout."""
        cfg = self.config
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            padded = torch.cat([context, context.new_zeros(cfg.text_len - context.size(0), context.size(1))])
            text = self.text_embedding(padded.unsqueeze(0))
            kv = [block.cross_attn.text_kv(text) for block in self.blocks]
        return [k for k, _ in kv], [v for _, v in kv]

    def chunk_conditioning(self, plucker: torch.Tensor, start_frame: int) -> ChunkConditioning:
        """``ChunkConditioning`` for a chunk's Plucker rays ``[1, 384, f, h, w]``
        bf16 starting at latent frame ``start_frame``. Its forwards see the same
        rays and frames, so this runs once per chunk (four GEMMs a block)."""
        cfg = self.config
        f, h, w = (n // p for n, p in zip(plucker.shape[2:], cfg.patch_size, strict=True))
        freqs = self.freqs(plucker.device)
        if start_frame + f > freqs.shape[0]:
            raise ValueError(f"latent frame {start_frame + f} is past the RoPE table ({freqs.shape[0]} frames)")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            cam = rearrange(
                plucker, "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                c1=cfg.patch_size[0], c2=cfg.patch_size[1], c3=cfg.patch_size[2],
            ).contiguous()
            cam = self.patch_embedding_wancamctrl(cam)
            cam = cam + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(cam)))
            return ChunkConditioning(
                cam=[block.cam_modulation(cam) for block in self.blocks],
                rope=rope_table(freqs, (f, h, w), start_frame),
            )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, text_kv: tuple[list[torch.Tensor], list[torch.Tensor]],
        y: torch.Tensor, chunk: ChunkConditioning, store: WindowStore,
        attention: DitAttention = REFERENCE_ATTENTION,
    ) -> torch.Tensor:
        """One chunk: ``x`` ``[16, f, h, w]`` fp32 noisy latents, ``t`` ``[1]``
        timestep, ``text_kv`` the request's ``text_kv``, ``y`` ``[20, f, h, w]``
        mask + image latent, ``chunk`` its ``chunk_conditioning``. Writes the
        chunk's K/V to ``store`` and attends its window there; returns the fp32
        flow prediction ``[16, f, h, w]``."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            return self._forward(x, t, text_kv, y, chunk, store, attention)

    def _forward(self, x, t, text_kv, y, chunk, store, attention):
        cfg = self.config

        x = self.patch_embedding(torch.cat([x, y], dim=0).unsqueeze(0))
        grid = tuple(x.shape[2:])
        # contiguous: a compiled block guards on its input's strides
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_len = x.size(1)

        per_token = t.dim() > 1
        if not per_token:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            e = self.time_embedding(
                sinusoidal_embedding_1d(cfg.freq_dim, t.flatten()).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, cfg.dim))
        if not per_token:
            # Every token's row is the same bits (embedded per token, as the
            # reference does), so one row broadcasts in the blocks instead of
            # each block adding and scaling [L, 6, dim] fp32 tensors.
            e, e0 = e[:, :1], e0[:, :1]

        cross_k, cross_v = text_kv
        window = ChunkWindow(attention.window, store, len(self.blocks))
        for i, block in enumerate(self.blocks):
            x = block(x, e0, chunk.cam[i], cross_k[i], cross_v[i], chunk.rope, window, attention.text)

        x = self.head(x, e)
        c = cfg.out_dim
        u = x[0, :math.prod(grid)].view(*grid, *cfg.patch_size, c)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        return u.reshape(c, *[i * j for i, j in zip(grid, cfg.patch_size, strict=True)]).float()
