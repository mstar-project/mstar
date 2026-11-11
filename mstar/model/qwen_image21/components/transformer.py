"""The Qwen-Image-2.1 DiT, split at the reference's KV cache into a prefix pass and a target pass.

diffusers' ``QwenImage21Transformer2DModel`` runs one block-causal sequence (text causal, each
image block bidirectional, the target last) on step 0 and caches the prefix's K/V, which is
step-independent because ``causal_condition`` modulates every prefix token from ``t = 0``. Here the
prefix is a span at a time over a paged KV cache (``prefix``: a text span causal, an image span
bidirectional, each over the spans before it) and every denoise step runs the target alone over
that cache (``target``). The prefix's queries never see the target, so this is the same math.

Ported op for op (dtypes, rounding points) from the reference. Where it differs without changing
the numbers:

* The ``t = 0`` modulation is a constant of the weights, computed once (``set_prefix_modulation``).
* A prefix pass's output is only the K/V it writes, so its last block stops after attention and
  the output head never runs.
* On CUDA the q/k norms with the rotation, and the target's gated residuals, run as the shared Qwen-Image
  kernels (``dit_kernels``), which round where the eager ops do.
* Heads shard over ``comm_group`` (q/k/v and the SwiGLU column-wise, their outputs row-wise); the
  modulation, the input projections and the output head are replicated.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.qwen import dit_kernels as kernels
from mstar.model.components.diffusion.rope import qwen_rope_params
from mstar.model.components.distributed.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.norm import QwenRMSNorm
from mstar.model.qwen_image21.config import QwenImage21TransformerConfig

# Each Qwen3-VL image slot stands for a 2x2 group of latent tokens.
TOKENS_PER_SLOT = 4
# QwenImage21Rope's tables: positions [0, 8192) and [-1024, 0).
ROPE_POSITIVE = 8192
ROPE_NEGATIVE = 1024


class Attend(Protocol):
    """The step's attention over the paged cache: K/V written, then attended (``AttentionCallable``)."""

    def set_layer_idx(self, layer_idx: int) -> None: ...

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


def text_positions(start: int, length: int) -> torch.Tensor:
    """``[length, 3]``: a text token sits at the same position on all three axes."""
    return torch.arange(start, start + length, dtype=torch.long)[:, None].expand(length, 3).contiguous()


def image_positions(frame: int, height: int, width: int) -> torch.Tensor:
    """``[height * width, 3]`` row-major: the frame axis held at ``frame``, height and width on a
    grid centred on zero (``QwenImage21Rope``)."""
    hs = torch.arange(-(height - height // 2), height // 2, dtype=torch.long)
    ws = torch.arange(-(width - width // 2), width // 2, dtype=torch.long)
    grid = torch.stack([
        torch.full((height, width), frame, dtype=torch.long),
        hs[:, None].expand(height, width),
        ws[None, :].expand(height, width),
    ], dim=-1)
    return grid.reshape(height * width, 3)


class QwenImage21Rope:
    """Complex rotary tables per axis, built on the CPU as the reference builds them (one copy per
    device, kept: a captured step reads them at fixed addresses), looked up by ``[N, 3]``
    positions; negative heights and widths index the tables' tails."""

    def __init__(self, axes_dim: tuple[int, int, int], theta: int = 10000):
        # on the CPU even when the module is built on the meta device
        index = torch.cat([
            torch.arange(ROPE_POSITIVE, device="cpu"), torch.arange(ROPE_NEGATIVE, device="cpu").flip(0) * -1 - 1,
        ])
        self._cpu = tuple(qwen_rope_params(index, dim, theta) for dim in axes_dim)
        self._tables: dict[torch.device, tuple[torch.Tensor, ...]] = {}

    def tables(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        tables = self._tables.get(device)
        if tables is None:
            tables = self._tables[device] = tuple(table.to(device) for table in self._cpu)
        return tables

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """``positions [..., 3]`` long -> complex ``[..., sum(axes_dim) / 2]``."""
        tables = self.tables(positions.device)
        return torch.cat([table[positions[..., axis]] for axis, table in enumerate(tables)], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """``apply_rotary_emb_qwen(use_real=False)`` on ``[N, H, D]`` with ``freqs [N, D / 2]``."""
    rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(rotated * freqs.unsqueeze(1)).flatten(-2).type_as(x)


def norm_rope(x: torch.Tensor, norm: QwenRMSNorm, freqs: torch.Tensor) -> torch.Tensor:
    """``apply_rope(norm(x), freqs)`` on ``[N, H, D]``: one fused pass on CUDA, the same bits."""
    if kernels.fits(x) and x.shape[-1] == 128:
        return kernels.qk_norm_rope(x, norm.weight, freqs, norm.eps)
    return apply_rope(norm(x).to(x.dtype), freqs)


def gated_add(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x + gate * y``: for the target's ``x``, ``y [R, L, D]`` and ``gate [R, 1, D]`` one fused pass on CUDA, the
    same bits. Under ``torch.compile`` the eager ops, which inductor fuses with the next block's norm."""
    if (not torch.compiler.is_compiling() and x.dim() == 3 and gate.shape == (x.shape[0], 1, x.shape[2])
            and y.shape == x.shape and kernels.fits(x, gate, y) and x.is_contiguous() and y.is_contiguous()):
        return kernels.gated_add(x, gate.reshape(x.shape[0], -1).contiguous(), y, x.shape[1])
    return x + gate * y


class ZeroCenterRMSNorm(nn.Module):
    """``QwenImage21ZeroCenterRMSNorm``: the checkpoint stores ``scale - 1``; fp32 throughout."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        rrms = torch.rsqrt(torch.mean(h**2, dim=-1, keepdim=True) + self.eps)
        return (h * rrms * (self.weight.float() + 1)).to(x.dtype)


class TextProjection(nn.Module):
    """``txt_in``: zero-centred RMSNorm, then linear -> tanh-GELU -> linear."""

    def __init__(self, context_dim: int, hidden: int, eps: float):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepEmbedder(nn.Module):
    """``time_text_embed``: cos-then-sin sinusoid of ``1000 t`` (fp32), linear -> SiLU -> linear."""

    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        half = freq_dim // 2
        # built on the CPU as the reference's buffer is, then kept per device
        self._freqs_cpu = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device="cpu") / half)
        self._freqs: dict[torch.device, torch.Tensor] = {}
        self.timestep_embedder = nn.Module()
        self.timestep_embedder.linear_1 = nn.Linear(freq_dim, hidden, bias=False)
        self.timestep_embedder.linear_2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``t [B]`` in [0, 1], already in ``dtype`` as the reference rounds it."""
        freqs = self._freqs.get(t.device)
        if freqs is None:
            freqs = self._freqs[t.device] = self._freqs_cpu.to(t.device)
        args = (1000.0 * t.float())[:, None] * freqs[None]
        proj = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        e = self.timestep_embedder
        return e.linear_2(F.silu(e.linear_1(proj)))


class Attention(nn.Module):
    def __init__(self, cfg: QwenImage21TransformerConfig, comm_group: CommGroup):
        super().__init__()
        heads, head_dim, dim = cfg.num_attention_heads, cfg.attention_head_dim, cfg.inner_dim
        self.head_dim = head_dim
        self.local_heads = heads // comm_group.world_size
        self.to_qkv = QKVParallelLinear(comm_group, dim, head_dim, heads, heads, bias=False)
        self.norm_q = QwenRMSNorm(head_dim, cfg.eps)
        self.norm_k = QwenRMSNorm(head_dim, cfg.eps)
        self.to_out = RowParallelLinear(comm_group, heads * head_dim, dim, bias=False)

    def attend(self, x: torch.Tensor, rope: torch.Tensor, attend: Attend) -> torch.Tensor:
        """``x [N, dim]`` -> the attention over this step's cache, ``[N, local_heads * head_dim]``."""
        local = self.local_heads * self.head_dim
        q, k, v = self.to_qkv(x).split([local, local, local], dim=-1)
        shape = (-1, self.local_heads, self.head_dim)
        q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
        if kernels.fits(q, k) and self.head_dim == 128 and q.stride() == k.stride():
            # q and k in one launch, the token's rotation read once: the same bits as two norm_rope calls
            q, k = kernels.qk_pair_norm_rope(q, k, self.norm_q.weight, self.norm_k.weight, rope, self.norm_q.eps)
        else:
            q = norm_rope(q, self.norm_q, rope)
            k = norm_rope(k, self.norm_k, rope)
        return attend(q, k, v).flatten(1).type_as(q)


class SwiGLU(nn.Module):
    """``img_mlp``: ``out(silu(gate_layer(x)) * proj(x))``, with gate and proj in one GEMM."""

    def __init__(self, dim: int, inner: int, comm_group: CommGroup):
        super().__init__()
        self.local = inner // comm_group.world_size
        self.gate_proj = MergedColumnParallelLinear(comm_group, dim, [inner, inner], bias=False)
        self.out = RowParallelLinear(comm_group, inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, proj = self.gate_proj(x).split([self.local, self.local], dim=-1)
        return self.out(F.silu(gate) * proj)


class Block(nn.Module):
    """``QwenImage21TransformerBlock``: every block reads the one shared modulation
    ``[..., 4 * dim]`` = attention (scale, gate), MLP (scale, gate)."""

    def __init__(self, cfg: QwenImage21TransformerConfig, comm_group: CommGroup):
        super().__init__()
        self.eps = cfg.eps
        self.attn = Attention(cfg, comm_group)
        self.img_mlp = SwiGLU(cfg.inner_dim, cfg.inner_dim * cfg.mlp_ratio, comm_group)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), eps=self.eps)

    def forward(
        self, x: torch.Tensor, modulation: torch.Tensor, rope: torch.Tensor, attend: Attend, last_prefix: bool = False,
    ) -> torch.Tensor | None:
        """``x [..., dim]`` with ``modulation`` broadcastable to ``[..., 4 * dim]``. ``last_prefix``
        writes this block's K/V and returns None: nothing after it is read."""
        scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
        lead = x.shape[:-1]
        h = (self._norm(x) * (1 + scale1)).reshape(-1, x.shape[-1])
        out = self.attn.attend(h, rope, attend)
        if last_prefix:
            return None
        x = gated_add(x, gate1.tanh(), self.attn.to_out(out).reshape(*lead, -1))
        x = gated_add(x, gate2.tanh(), self.img_mlp(self._norm(x) * (1 + scale2)))
        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)
        return x


class QwenImage21Transformer(nn.Module):
    """Parameter names follow the diffusers checkpoint (q/k/v and gate/proj are stacked on load)."""

    def __init__(self, cfg: QwenImage21TransformerConfig, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        if cfg.patch_size != 1:
            raise ValueError(f"Qwen-Image-2.1 consumes latents unpatched; got patch_size={cfg.patch_size}")
        if not cfg.causal_condition:
            raise ValueError(
                "this Qwen-Image-2.1 checkpoint has causal_condition=false: its prefix is not step-independent, "
                "so it cannot be served from a KV cache"
            )
        self.cfg = cfg
        dim = cfg.inner_dim
        self.pos_embed = QwenImage21Rope(cfg.axes_dims_rope)
        self.time_text_embed = TimestepEmbedder(dim)
        self.txt_in = TextProjection(cfg.context_in_dim, dim, cfg.eps)
        self.img_in = nn.Linear(cfg.in_channels, dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
        self.transformer_blocks = nn.ModuleList(Block(cfg, comm_group) for _ in range(cfg.num_layers))
        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, cfg.out_channels, bias=False)
        # the t = 0 row every prefix token modulates from (set_prefix_modulation)
        self.prefix_modulation: torch.Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self.img_in.weight.dtype

    @torch.no_grad()
    def set_prefix_modulation(self) -> None:
        """The shared modulation at ``t = 0``; call once the weights are loaded."""
        zero = torch.zeros(1, dtype=self.dtype, device=self.img_in.weight.device)
        self.prefix_modulation = self.modulation(self.time_text_embed(zero, self.dtype))[0]

    def prefix(self, hidden: torch.Tensor, positions: torch.Tensor, attend: Attend) -> None:
        """One step's prefix tokens ``hidden [N, dim]`` (``txt_in`` / ``img_in`` applied) at ``positions
        [N, 3]``: every block writes its K/V through ``attend``. The step's causality is the attention's."""
        rope = self.pos_embed(positions)
        blocks = self.transformer_blocks
        for i, block in enumerate(blocks):
            attend.set_layer_idx(i)
            # the class's forward: a span's length varies per request, so the prefix runs eager even when
            # compile_target_blocks compiled the blocks' instance forward
            last = i == len(blocks) - 1
            hidden = Block.forward(block, hidden, self.prefix_modulation, rope, attend, last_prefix=last)

    def compile_target_blocks(self) -> None:
        """Compile each block for the target pass: its shapes are fixed per latent grid and batch, and the
        attention stays an eager call between two compiled graphs. The prefix pass keeps the eager forward."""
        from mstar.model.components.diffusion.compile_utils import compile_blocks

        compile_blocks(self.transformer_blocks, dynamic=False)

    def target(
        self, latents: torch.Tensor, timestep: torch.Tensor, positions: torch.Tensor, attend: Attend,
    ) -> torch.Tensor:
        """The target ``latents [R, S, in]`` at ``timestep [R]`` (``sigma * 1000``, fp32) and
        ``positions [R, S, 3]``, attending the cache through ``attend`` -> velocity ``[R, S, out]``."""
        dtype = self.dtype
        # the pipeline rounds t to the latents' dtype before dividing, the model again after
        t = (timestep.to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(t, dtype)
        modulation = self.modulation(temb)[:, None]
        rope = self.pos_embed(positions.reshape(-1, 3))
        x = self.img_in(latents.to(dtype))
        for i, block in enumerate(self.transformer_blocks):
            attend.set_layer_idx(i)
            x = block(x, modulation, rope, attend)
        scale = self.norm_out.linear(F.silu(temb).to(x.dtype))[:, None]
        x = F.layer_norm(x, (x.shape[-1],), eps=self.cfg.eps) * (1 + scale)
        return self.proj_out(x)
