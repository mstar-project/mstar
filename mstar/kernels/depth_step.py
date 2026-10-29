"""The resident depth-loop kernel (``hip/depth_step.cu``): build, scratch, launch.

ROCm only (gfx942 / gfx950: MFMA bf16, the XCC id register, L2-scoped
synchronisation). ``PersistentExecutor`` is the caller; it lowers a
``StepProgram`` with ``mstar.engine.step_program.lower`` and keeps one
``DepthStepScratch`` per captured batch size.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import torch

from mstar.engine.step_program import DepthDecoderShape, rope_table  # noqa: F401  re-exported
from mstar.kernels.hip.build import JitExtension

SOURCE = Path(__file__).resolve().parent / "hip" / "depth_step.cu"
SCOPE_AGENT, SCOPE_XCD = 0, 1


def extension() -> JitExtension:
    return JitExtension("_mstar_depth_step", [SOURCE])


def op():
    extension().load()
    return torch.ops.mstar_persistent.depth_step


@dataclasses.dataclass
class DepthStepScratch:
    """Device scratch of one launch shape. ``reset`` zeroes the sums of squares
    and the barrier counters (one memset) before every launch."""
    logits: torch.Tensor
    h: torch.Tensor
    qkv: torch.Tensor
    attn: torch.Tensor
    act: torch.Tensor
    kv: torch.Tensor
    zeroed: torch.Tensor
    ss: torch.Tensor
    sync: torch.Tensor
    ctl: torch.Tensor

    @classmethod
    def allocate(cls, shape: DepthDecoderShape, bs: int, groups: int, ss_slots: int, device) -> DepthStepScratch:
        s = shape
        f32 = dict(dtype=torch.float32, device=device)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        n_ss = ss_slots * bs
        zeroed = torch.zeros(n_ss + groups * 64 + 4, dtype=torch.int32, device=device)
        return cls(
            logits=torch.zeros((s.positions - 1, bs, s.vocab), **f32),
            h=torch.zeros((bs, s.hidden), **f32),
            qkv=torch.zeros((bs, (s.heads + 2 * s.kv_heads) * s.head_dim), **f32),
            attn=torch.zeros((bs, s.heads * s.head_dim), **bf16),
            act=torch.zeros((bs, s.intermediate), **bf16),
            kv=torch.zeros((s.layers, bs, 2, s.positions, s.kv_heads, s.head_dim), **bf16),
            zeroed=zeroed, ss=zeroed[:n_ss].view(torch.float32).view(ss_slots, bs),
            sync=zeroed[n_ss:n_ss + groups * 64], ctl=zeroed[n_ss + groups * 64:])

    def reset(self) -> None:
        self.zeroed.zero_()


def launch(table: torch.Tensor, rope: torch.Tensor, last_hidden: torch.Tensor, embed0: torch.Tensor,
           codebook: torch.Tensor, emb_sum: torch.Tensor, codes: torch.Tensor, scratch: DepthStepScratch, *,
           groups: int, scope: int, grid: int, shape: DepthDecoderShape, top_k: int, top_p: float,
           temperature: float, seed: int, forced: torch.Tensor | None = None, stop_after: int = -1) -> None:
    """One frame: zero the scratch counters, run the kernel (both capturable)."""
    scratch.reset()
    op()(table, rope, last_hidden, embed0, codebook, forced, emb_sum, codes, scratch.logits, scratch.h,
         scratch.qkv, scratch.attn, scratch.act, scratch.kv, scratch.ss, scratch.sync, scratch.ctl, groups, scope,
         grid, stop_after, top_k, top_p, temperature, shape.eps, seed)


def default_grid(compute_units: int, groups: int, per_cu: int = 1) -> int:
    """Workgroups for ``groups`` groups: every CU once (or ``per_cu`` times),
    rounded down to a multiple of the group count."""
    return (compute_units * per_cu) // groups * groups


def check_shape(shape: DepthDecoderShape, bs: int, groups: int) -> str | None:
    """Why this kernel cannot run the shape, or None."""
    want = dict(hidden=1024, heads=16, kv_heads=8, head_dim=128, positions=16, vocab=2048, intermediate=3072)
    for name, value in want.items():
        if getattr(shape, name) != value:
            return f"{name}={getattr(shape, name)}, the kernel is built for {value}"
    if math.ceil(bs / groups) > 16:
        return f"{math.ceil(bs / groups)} rows per group, at most 16"
    return None
