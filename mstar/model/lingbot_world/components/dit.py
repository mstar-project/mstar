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
structural only: batch 1 without the list
wrapping, the cache indices kept as Python ints instead of 1-element device
tensors read back with ``.item()``, and one index computation per forward
instead of one per layer (every layer sees the same indices).

``wan22.components.dit`` is not reused: it is the diffusers layout and its
numerics (explicit ``type_as`` casts, SDPA, fp32-island weights) are the
diffusers pipeline's, not this checkpoint's reference.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

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
            "LingBot-World)') or serve without reference_attention."
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


def causal_rope_apply(
    x: torch.Tensor, grid: tuple[int, int, int], freqs: torch.Tensor, start_frame: int,
) -> torch.Tensor:
    """3D RoPE in complex128 with the temporal axis offset to ``start_frame``
    (the chunk's absolute latent frame); ``x`` ``[1, L, H, D]``."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid
    seq_len = f * h * w
    x_i = torch.view_as_complex(x[0, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
    freqs_i = torch.cat([
        freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(seq_len, 1, -1)
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
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


@dataclass
class RollingKVCache:
    """Per-request self-attention K/V for every layer: ``capacity`` tokens
    (``local_attn_size`` latent frames, or the whole rollout when -1), the
    first ``sink_tokens`` never evicted. ``global_end`` is the absolute token
    the stream has reached, ``local_end`` where it sits in the buffer.

    Eviction is the reference's: when a forward past the frontier would
    overflow the buffer, the tokens behind the sink shift left by the overflow
    and the new chunk lands at the end. A forward that re-writes the current
    chunk (the denoise steps, then the commit) overwrites it in place.
    """

    k: list[torch.Tensor]
    v: list[torch.Tensor]
    sink_tokens: int
    windowed: bool
    global_end: int = 0
    local_end: int = 0
    cross_k: list[torch.Tensor] = field(default_factory=list)
    cross_v: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def allocate(
        cls, num_layers: int, capacity: int, num_heads: int, head_dim: int,
        sink_tokens: int, windowed: bool, device, dtype=torch.bfloat16,
    ) -> "RollingKVCache":
        shape = (1, capacity, num_heads, head_dim)
        return cls(
            k=[torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)],
            v=[torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)],
            sink_tokens=sink_tokens,
            windowed=windowed,
        )

    @property
    def capacity(self) -> int:
        return self.k[0].shape[1]

    def plan(self, current_start: int, num_new: int) -> "_WritePlan":
        current_end = current_start + num_new
        if not self.windowed:
            return _WritePlan(current_start, current_end, 0, 0, current_end)
        if current_end > self.global_end and num_new + self.local_end > self.capacity:
            evicted = num_new + self.local_end - self.capacity
            rolled = self.local_end - evicted - self.sink_tokens
            local_end = self.local_end + current_end - self.global_end - evicted
        else:
            evicted = rolled = 0
            local_end = self.local_end + current_end - self.global_end
        return _WritePlan(local_end - num_new, local_end, evicted, rolled, current_end)

    def window(self, plan: "_WritePlan") -> slice:
        """The buffer tokens a forward under ``plan`` attends: the sink and
        the latest window, ending with the chunk it writes."""
        return slice(max(0, plan.local_end - self.capacity), plan.local_end)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, plan: "_WritePlan") -> None:
        """Apply ``plan`` to one layer: shift the tokens behind the sink left by
        the eviction, then write the chunk's ``[1, L, H, D]`` K/V."""
        k_cache, v_cache = self.k[layer], self.v[layer]
        if plan.evicted:
            src = slice(self.sink_tokens + plan.evicted, self.sink_tokens + plan.evicted + plan.rolled)
            dst = slice(self.sink_tokens, self.sink_tokens + plan.rolled)
            k_cache[:, dst] = k_cache[:, src].clone()
            v_cache[:, dst] = v_cache[:, src].clone()
        k_cache[:, plan.local_start:plan.local_end] = k
        v_cache[:, plan.local_start:plan.local_end] = v


@dataclass(frozen=True)
class _WritePlan:
    local_start: int
    local_end: int
    evicted: int
    rolled: int
    current_end: int


def attended_frames(config: LingBotWorldConfig, chunk: int) -> int:
    """Latent frames of rolling K/V that chunk ``chunk``'s queries attend, its
    own included: what ``RollingKVCache.window`` spans once the chunk is written."""
    frames = (chunk + 1) * config.chunk_size
    return min(frames, config.local_attn_size) if config.local_attn_size > -1 else frames


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

    def forward(
        self, x: torch.Tensor, grid: tuple[int, int, int], freqs: torch.Tensor,
        start_frame: int, cache: RollingKVCache, layer: int, plan: _WritePlan, attend: Attend,
    ) -> torch.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        roped_query = causal_rope_apply(q, grid, freqs, start_frame).type_as(v)
        roped_key = causal_rope_apply(k, grid, freqs, start_frame).type_as(v)
        cache.write(layer, roped_key, v, plan)
        window = cache.window(plan)
        x = attend(roped_query, cache.k[layer][:, window], cache.v[layer][:, window])
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

    def forward(
        self, x: torch.Tensor, e: torch.Tensor, cam: torch.Tensor, cross_k: torch.Tensor,
        cross_v: torch.Tensor, attn_args: dict, text_attend: Attend,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        y = self.self_attn(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2), **attn_args)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        cam_hidden = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(cam)))
        cam_hidden = cam_hidden + cam
        x = (1.0 + self.cam_scale_layer(cam_hidden)) * x + self.cam_shift_layer(cam_hidden)

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

    def freqs(self, device: torch.device) -> torch.Tensor:
        if device not in self._freqs:
            self._freqs[device] = self._freqs_cpu.to(device)
        return self._freqs[device]

    def new_cache(self, frame_seqlen: int, total_latent_frames: int, device) -> RollingKVCache:
        cfg = self.config
        windowed = cfg.local_attn_size > -1
        frames = cfg.local_attn_size if windowed else total_latent_frames
        return RollingKVCache.allocate(
            cfg.num_layers, frame_seqlen * frames, cfg.num_heads, cfg.head_dim,
            sink_tokens=cfg.sink_size * frame_seqlen, windowed=windowed, device=device,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor, y: torch.Tensor,
        plucker: torch.Tensor, cache: RollingKVCache, current_start: int, frame_seqlen: int,
        attention: DitAttention = REFERENCE_ATTENTION,
    ) -> torch.Tensor:
        """One chunk: ``x`` ``[16, f, h, w]`` fp32 noisy latents, ``t`` ``[1]``
        timestep, ``context`` ``[L, 4096]`` text embeds, ``y`` ``[20, f, h, w]``
        mask + image latent, ``plucker`` ``[1, 384, f, h, w]`` bf16. Writes the
        chunk's K/V at absolute token ``current_start``; returns the fp32 flow
        prediction ``[16, f, h, w]``."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            return self._forward(x, t, context, y, plucker, cache, current_start, frame_seqlen, attention)

    def _forward(self, x, t, context, y, plucker, cache, current_start, frame_seqlen, attention):
        cfg = self.config
        device = self.patch_embedding.weight.device
        freqs = self.freqs(device)

        x = self.patch_embedding(torch.cat([x, y], dim=0).unsqueeze(0))
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.size(1)

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            e = self.time_embedding(
                sinusoidal_embedding_1d(cfg.freq_dim, t.flatten()).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, cfg.dim))

        if not cache.cross_k:
            padded = torch.cat([context, context.new_zeros(cfg.text_len - context.size(0), context.size(1))])
            text = self.text_embedding(padded.unsqueeze(0))
            for block in self.blocks:
                k, v = block.cross_attn.text_kv(text)
                cache.cross_k.append(k)
                cache.cross_v.append(v)

        cam = rearrange(
            plucker, "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
            c1=cfg.patch_size[0], c2=cfg.patch_size[1], c3=cfg.patch_size[2],
        ).contiguous()
        cam = self.patch_embedding_wancamctrl(cam)
        cam = cam + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(cam)))

        plan = cache.plan(current_start, seq_len)
        start_frame = current_start // frame_seqlen
        if start_frame + grid[0] > freqs.shape[0]:
            raise ValueError(
                f"latent frame {start_frame + grid[0]} is past the RoPE table ({freqs.shape[0]} frames)")
        for i, block in enumerate(self.blocks):
            attn_args = dict(grid=grid, freqs=freqs, start_frame=start_frame, cache=cache, layer=i, plan=plan,
                             attend=attention.window)
            x = block(x, e0, cam, cache.cross_k[i], cache.cross_v[i], attn_args, attention.text)
        cache.global_end = plan.current_end
        cache.local_end = plan.local_end

        x = self.head(x, e)
        c = cfg.out_dim
        u = x[0, :math.prod(grid)].view(*grid, *cfg.patch_size, c)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        return u.reshape(c, *[i * j for i, j in zip(grid, cfg.patch_size, strict=True)]).float()
