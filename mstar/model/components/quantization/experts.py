"""Routed experts held in a checkpoint's quantized format, sharded on the intermediate dim.

The weights stay in the format's containers (``gate_up_proj_<part>`` stacks gate's rows over up's,
``down_proj_<part>``, one leading expert dim), so ``module.to(bf16)`` leaves the bytes alone. The
forward returns this rank's unreduced partial: the fused kernels where the format has them and the
owning block resolved them on the real device, else a reference loop that dequantizes only the
experts a batch hit (host syncs: eager only, the numerics anchor).
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.quantization.formats import WeightFormat
from mstar.model.loader.base import StackedParamRule

QUANT_KERNELS = ("auto", "fused", "reference")


class ExpertMath(NamedTuple):
    """What the experts compute between their two GEMMs, where checkpoints were trained differently.

    ``act(gate, up)`` is the gated activation (None: ``silu(gate) * up`` in the activation dtype,
    DeepSeek-V3's and GLM-5.x's); ``weights_before_down`` scales each token's hidden by its routing
    weight before the down GEMM (DeepSeek-V4.1) instead of scaling the down GEMM's output. The fused
    kernels serve only the default.
    """

    act: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    weights_before_down: bool = False


class QuantizedExperts(nn.Module):
    """``num_experts`` routed SwiGLU experts in ``fmt``, this rank's ``intermediate_size`` shard."""

    def __init__(
        self, fmt: WeightFormat, num_experts: int, intermediate_size: int, hidden_size: int,
        comm_group: CommGroup | None = None, *, kernels: str = "auto",
        math: ExpertMath = ExpertMath(),
    ) -> None:
        super().__init__()
        if kernels not in QUANT_KERNELS:
            raise ValueError(f"quant kernels {kernels!r} is not one of {QUANT_KERNELS}")
        comm_group = comm_group or CommGroup.trivial()
        self.tp_rank, self.tp_size = comm_group.rank, comm_group.world_size
        self.fmt = fmt
        self.kernels = kernels
        self.math = math
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        shard_inter = divide(intermediate_size, self.tp_size)
        self.shard_inter = shard_inter
        # gate and up each shard their rows, down its columns
        fmt.check_shard(intermediate_size, hidden_size, row_split=self.tp_size)
        fmt.check_shard(hidden_size, intermediate_size, col_split=self.tp_size)
        for part in fmt.parts:
            r, c = fmt.shape(part, shard_inter, hidden_size)
            self.register_parameter(f"gate_up_proj_{part.container}", nn.Parameter(
                torch.empty(num_experts, 2 * r, c, dtype=part.dtype), requires_grad=False))
            r, c = fmt.shape(part, hidden_size, shard_inter)
            self.register_parameter(f"down_proj_{part.container}", nn.Parameter(
                torch.empty(num_experts, r, c, dtype=part.dtype), requires_grad=False))
        # resolved by resolve_kernels on the real device; host tests stay on the reference loop
        self.fused = False
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        from mstar.model.components.moe import (
            _down_proj_fp8_weight_loader,
            _gate_up_fp8_weight_loader,
        )

        rank, size, inter = self.tp_rank, self.tp_size, self.intermediate_size
        for part in self.fmt.parts:
            getattr(self, f"gate_up_proj_{part.container}").weight_loader = partial(
                _gate_up_fp8_weight_loader, rank, size, inter, part.unit[0])
            getattr(self, f"down_proj_{part.container}").weight_loader = partial(
                _down_proj_fp8_weight_loader, rank, size, inter, part.unit[1])

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def resolve_kernels(self, device) -> None:
        """Pick the kernels on the real device: ``fused`` must not silently downgrade."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        ok = (self.kernels != "reference" and self.math == ExpertMath()
              and self.fmt.fused_available(dev))
        if self.kernels == "fused" and not ok:
            raise RuntimeError(
                f"quant kernels 'fused' but no fused MoE kernel serves {self.fmt.name} experts "
                f"on {dev} with this expert math. Use 'auto' to fall back to the reference "
                "dispatch.")
        self.fused = ok

    def expert_weights(self, kind: str, expert: int, out_dtype: torch.dtype) -> torch.Tensor:
        """Expert ``expert``'s dense ``gate_up_proj`` (2 * shard_inter, hidden) or ``down_proj``
        (hidden, shard_inter), dequantized."""
        rows, cols = ((2 * self.shard_inter, self.hidden_size) if kind == "gate_up_proj"
                      else (self.hidden_size, self.shard_inter))
        parts = {p.container: getattr(self, f"{kind}_{p.container}")[expert]
                 for p in self.fmt.parts}
        if kind == "down_proj":
            return self.fmt.dequantize(parts, rows, cols, out_dtype)
        # gate's rows over up's: each dequantizes on its own (a scale block never spans both)
        halves = [{k: v.chunk(2, dim=0)[i] for k, v in parts.items()} for i in (0, 1)]
        return torch.cat([self.fmt.dequantize(h, rows // 2, cols, out_dtype) for h in halves])

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.Tensor) -> torch.Tensor:
        """This rank's routed partial, unreduced; routing weights stay fp32 into the combine."""
        if not self.fused:
            return self.reference(x, topk_weights, topk_ids)
        from mstar.utils.fused_moe import fused_experts_fp8

        return fused_experts_fp8(
            x, self.gate_up_proj_fp8, self.down_proj_fp8, self.gate_up_proj_scale_inv,
            self.down_proj_scale_inv, topk_weights, topk_ids, block_size=self.fmt.block)

    def reference(self, x: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor) -> torch.Tensor:
        """Per-expert loop over the experts this batch hit, each dequantized to ``x``'s dtype."""
        act, before_down = self.math
        final = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = F.one_hot(topk_ids, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_t in expert_hit:
            e = expert_idx_t[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            tokens = x[token_idx]
            gate_up_w = self.expert_weights("gate_up_proj", e, x.dtype)
            down_w = self.expert_weights("down_proj", e, x.dtype)
            gate, up = torch.mm(tokens, gate_up_w.T).chunk(2, dim=-1)
            hidden = F.silu(gate) * up if act is None else act(gate, up)
            weight = topk_weights[token_idx, top_k_pos, None]
            if before_down:
                out = torch.mm((hidden * weight).to(x.dtype), down_w.T)
            else:
                out = torch.mm(hidden.to(x.dtype), down_w.T) * weight
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final


def expert_param_rules(
    fmt: WeightFormat, num_experts: int, source: Callable[[str, int], str],
) -> list[StackedParamRule]:
    """Rules stacking expert ``i``'s checkpoint tensors ``<source(proj, i)>.<part>`` into the
    :class:`QuantizedExperts` containers; ``proj`` is ``gate_proj``, ``up_proj`` or ``down_proj``
    (a checkpoint that calls them w1 / w3 / w2 maps them in ``source``)."""
    # first-win substring matching: ".weight" is a prefix of ".weight_scale_inv"
    parts = sorted(fmt.parts, key=lambda p: -len(p.checkpoint))
    rules: list[StackedParamRule] = []
    for i in range(num_experts):
        for proj, sid, kind in (("gate_proj", f"gate:{i}", "gate_up_proj"),
                                ("up_proj", f"up:{i}", "gate_up_proj"),
                                ("down_proj", f"down:{i}", "down_proj")):
            for p in parts:
                rules.append(StackedParamRule(
                    target_suffix=f".experts.{kind}_{p.container}",
                    source_suffix=f"{source(proj, i)}.{p.checkpoint}",
                    shard_id=sid,
                ))
    return rules


def expert_read_slice(
    fmt: WeightFormat, checkpoint_suffix: str, proj: str, intermediate_size: int, tp_rank: int,
    tp_size: int,
) -> tuple[int, int, int] | None:
    """This rank's ``(dim, start, stop)`` slice of one expert tensor for a sliced checkpoint read:
    rows of gate / up, columns of down, in stored elements. None reads it whole (an aux part)."""
    part = fmt.part(checkpoint_suffix)
    if part is None:
        return None
    dim = 1 if proj == "down_proj" else 0
    n = divide(intermediate_size, tp_size) // part.unit[dim]
    return dim, tp_rank * n, (tp_rank + 1) * n


__all__ = [
    "QUANT_KERNELS",
    "ExpertMath",
    "QuantizedExperts",
    "expert_param_rules",
    "expert_read_slice",
]
