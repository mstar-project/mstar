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
        self.write_layer(self.k[layer], self.v[layer], k, v, plan)

    def write_layer(
        self, k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: "_WritePlan",
    ) -> None:
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


class ChunkWindow:
    """One chunk's window attention for every block: write the chunk's K/V
    into a layer's rolling cache, then attend over the window. Eager inside a
    compiled block, and handed to it as one object rather than as the plan's
    integers: the cache offsets change chunk to chunk, and Dynamo guards on
    integer arguments, so traced (or passed as ints) they would recompile
    every block every chunk."""

    def __init__(self, attend: Attend, cache: "RollingKVCache", plan: "_WritePlan"):
        self.attend = attend
        self.cache = cache
        self.plan = plan

    @torch.compiler.disable
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self.cache.write_layer(*kv, k, v, self.plan)
        window = self.cache.window(self.plan)
        return self.attend(q, kv[0][:, window], kv[1][:, window])


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
        self, x: torch.Tensor, rope: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor], window: ChunkWindow,
    ) -> torch.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        roped_query = causal_rope_apply(q, rope).type_as(v)
        roped_key = causal_rope_apply(k, rope).type_as(v)
        x = window(roped_query, roped_key, v, kv)
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
        cross_v: torch.Tensor, rope: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor], window: ChunkWindow,
        text_attend: Attend,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        y = self.self_attn(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2), rope, kv, window)
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
        """Compile each block's forward in place, rounding every intermediate
        where eager does (``emulate_precision_casts``): by default Inductor's
        defaults in its deterministic mode, so every process compiles the same
        kernels; with ``autotune`` the -O level's options, whose kernel picks
        are per process. The cache write and both attentions stay eager
        (``_window_attention``, the ragged callables), so a block traces once
        for every chunk."""
        from mstar.model.components.diffusion.compile_utils import with_inductor_rounding

        if autotune:
            from mstar.engine.accelerator_graph_runner import compile_kwargs as level_kwargs

            compile_kwargs = {**level_kwargs(None), **compile_kwargs}
        compile_kwargs = {"fullgraph": False, "dynamic": False, **compile_kwargs}
        for block in self.blocks:
            block.forward = with_inductor_rounding(
                torch.compile(block.forward, **compile_kwargs), eager_rounding=True, deterministic=not autotune,
            )

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
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor, y: torch.Tensor,
        chunk: ChunkConditioning, cache: RollingKVCache, current_start: int,
        attention: DitAttention = REFERENCE_ATTENTION,
    ) -> torch.Tensor:
        """One chunk: ``x`` ``[16, f, h, w]`` fp32 noisy latents, ``t`` ``[1]``
        timestep, ``context`` ``[L, 4096]`` text embeds, ``y`` ``[20, f, h, w]``
        mask + image latent, ``chunk`` its ``chunk_conditioning``. Writes the
        chunk's K/V at absolute token ``current_start``; returns the fp32 flow
        prediction ``[16, f, h, w]``."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            return self._forward(x, t, context, y, chunk, cache, current_start, attention)

    def _forward(self, x, t, context, y, chunk, cache, current_start, attention):
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

        if not cache.cross_k:
            padded = torch.cat([context, context.new_zeros(cfg.text_len - context.size(0), context.size(1))])
            text = self.text_embedding(padded.unsqueeze(0))
            for block in self.blocks:
                k, v = block.cross_attn.text_kv(text)
                cache.cross_k.append(k)
                cache.cross_v.append(v)

        plan = cache.plan(current_start, seq_len)
        window = ChunkWindow(attention.window, cache, plan)
        for i, block in enumerate(self.blocks):
            # the layer's buffers, not its index: a compiled block would guard on the int
            x = block(x, e0, chunk.cam[i], cache.cross_k[i], cache.cross_v[i], chunk.rope,
                      (cache.k[i], cache.v[i]), window, attention.text)
        cache.global_end = plan.current_end
        cache.local_end = plan.local_end

        x = self.head(x, e)
        c = cfg.out_dim
        u = x[0, :math.prod(grid)].view(*grid, *cfg.patch_size, c)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        return u.reshape(c, *[i * j for i, j in zip(grid, cfg.patch_size, strict=True)]).float()
