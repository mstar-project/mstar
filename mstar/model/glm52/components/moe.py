"""GLM-5.2 fine-grained MoE: groupless sigmoid router + FP8-resident experts.

The router is DeepSeek-V3 noaux_tc math (identical to ``KimiMoEGate``) minus
the group machinery — GLM-5.2 has no ``n_group``/``topk_group`` fields, i.e.
n_group=1, where group selection is the identity. A CPU test pins parity
against ``KimiMoEGate(n_group=1, topk_group=1)``.

Routed experts stay FP8: bytes in uint8 containers (ints dodge the
module-wide autocast like Kimi's packed int32 weights) with fp32
``weight_scale_inv`` block scales. Dispatch v1 dequantizes only the experts
a batch actually routes to and reuses the shared bf16 per-expert loop — the
"simplest correct" resolution of the FP8 kernel decision. At decode that
touches top-k+shared experts (~30 MB/layer); large prefills touch most of
the 256 and pay full dequant traffic. Perf debt, on the M4 ledger: port
sglang's fp8_w8a8 branch back into ``utils/fused_moe`` (it was stripped —
M* has no fp8 kernel today) or adopt DeepGEMM.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.moe import (
    _dispatch,
    _down_proj_weight_loader,
    _gate_up_weight_loader,
    dispatch_experts_fused,
)
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.quantization import FP8_DTYPE, dequantize_fp8_block_weight

logger = logging.getLogger(__name__)

_BACKEND_LOGGED = False


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _gate_up_fp8_loader(
    tp_rank: int, tp_size: int, full_inter: int, row_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """Route one expert's gate/up tensor into the stacked per-rank param.

    ``row_unit`` is 1 for the fp8 bytes and block_size[0] for the scale rows;
    the same slicing logic covers both because scales tile the row axis.
    Shape-driven: a full checkpoint tensor is TP-sliced here; a pre-sliced
    shard (the ``slice_spec`` fast read path — each rank reads only its
    bytes) is written as-is.
    """
    assert loaded_shard_id is not None
    kind, expert_str = str(loaded_shard_id).split(":")
    expert_idx = int(expert_str)
    rows = divide(divide(full_inter, tp_size), row_unit)
    full_rows = divide(full_inter, row_unit)
    if loaded_weight.dtype == FP8_DTYPE:
        loaded_weight = loaded_weight.view(torch.uint8)
    if loaded_weight.shape[0] == full_rows:
        start = tp_rank * rows
        loaded_weight = loaded_weight[start:start + rows, :]
    elif loaded_weight.shape[0] != rows:
        raise ValueError(
            f"expert gate/up tensor has {loaded_weight.shape[0]} rows; expected "
            f"the full {full_rows} or the per-rank {rows}"
        )
    if kind == "gate":
        param.data[expert_idx, :rows, :] = loaded_weight
    else:
        param.data[expert_idx, rows:2 * rows, :] = loaded_weight


def _down_fp8_loader(
    tp_rank: int, tp_size: int, full_inter: int, col_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """Column (contraction-dim) sharding twin of :func:`_gate_up_fp8_loader`."""
    assert loaded_shard_id is not None
    expert_idx = int(str(loaded_shard_id).split(":")[1])
    cols = divide(divide(full_inter, tp_size), col_unit)
    full_cols = divide(full_inter, col_unit)
    if loaded_weight.dtype == FP8_DTYPE:
        loaded_weight = loaded_weight.view(torch.uint8)
    if loaded_weight.shape[1] == full_cols:
        start = tp_rank * cols
        loaded_weight = loaded_weight[:, start:start + cols]
    elif loaded_weight.shape[1] != cols:
        raise ValueError(
            f"expert down tensor has {loaded_weight.shape[1]} cols; expected "
            f"the full {full_cols} or the per-rank {cols}"
        )
    param.data[expert_idx, :, :] = loaded_weight


class Glm52MoEGate(nn.Module):
    """DeepSeek-V3 noaux_tc sigmoid router, groupless (n_group=1).

    Bias-added scores drive expert *selection*; raw sigmoid scores drive the
    combine weights. The selection bias is fp32 by checkpoint convention
    (``restore_fp32_params`` re-widens it before loading).
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        self.weight = nn.Parameter(torch.zeros(n_routed_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(n_routed_experts, dtype=torch.float32)
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = hidden_states.reshape(-1, self.hidden_size).float()
        scores = F.linear(h, self.weight.float()).sigmoid()  # (T, E)

        biased = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_ids = torch.topk(biased, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_ids


class Glm52SparseMoeBlock(nn.Module):
    """Routed experts + ungated shared expert (DeepSeek-V3 block shape)."""

    def __init__(
        self, config: Glm52ModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.tp_size = comm_group.world_size
        self.tp_rank = comm_group.rank
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.moe_intermediate_size = config.moe_intermediate_size
        shard_inter = divide(config.moe_intermediate_size, self.tp_size)

        self.fp8_experts = (
            config.quantization_config is not None and config.moe_fp8_resident
        )
        self.quant_kernel = getattr(config, "moe_quant_kernel", "reference")
        # Resolved on the real device by process_weights_after_loading;
        # blocks used without the load hook (CPU tests) stay on reference.
        self._use_fused = False

        self.gate = Glm52MoEGate(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
        )

        self.experts = nn.Module()
        if self.fp8_experts:
            bo, bi = config.quantization_config.weight_block_size
            self.block_size = (bo, bi)
            hidden, E = config.hidden_size, config.n_routed_experts
            assert shard_inter % bo == 0 and shard_inter % bi == 0, (
                f"per-rank intermediate {shard_inter} must be a multiple of the "
                f"fp8 scale block {self.block_size} for clean TP slicing (full "
                f"model: 2048/8=256 per rank, 2 blocks of 128)"
            )
            # fp8 bytes in uint8 containers; e4m3 view happens at dispatch.
            self.experts.gate_up_proj_fp8 = nn.Parameter(
                torch.empty(E, 2 * shard_inter, hidden, dtype=torch.uint8),
                requires_grad=False,
            )
            self.experts.gate_up_proj_scale_inv = nn.Parameter(
                torch.empty(
                    E, 2 * (shard_inter // bo), _ceil_div(hidden, bi),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.experts.down_proj_fp8 = nn.Parameter(
                torch.empty(E, hidden, shard_inter, dtype=torch.uint8),
                requires_grad=False,
            )
            self.experts.down_proj_scale_inv = nn.Parameter(
                torch.empty(
                    E, _ceil_div(hidden, bo), shard_inter // bi,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            self.experts.gate_up_proj = nn.Parameter(
                torch.empty(
                    config.n_routed_experts, 2 * shard_inter, config.hidden_size,
                )
            )
            self.experts.down_proj = nn.Parameter(
                torch.empty(
                    config.n_routed_experts, config.hidden_size, shard_inter,
                )
            )
        self._attach_expert_weight_loaders()

        # One all-reduce per MoE block instead of two (vLLM's DeepseekV2MoE
        # layout): the shared expert returns its per-rank partial, it is
        # added to the routed partial, and the SUM is reduced once. Saves 76
        # collectives per decode step at TP8 (~15-25 us each on NVSwitch
        # [estimate] -> ~1-2 ms of a 19 ms step), MTP off and on alike.
        # DEFAULT OFF: bf16 rounding moves (sum-then-reduce vs reduce-then-
        # sum), so the emitted stream can differ from today's at FP near-ties
        # — a policy call to make with a measurement, not silently.
        # MSTAR_GLM52_MOE_FUSED_ALLREDUCE=1 to enable.
        self._fused_allreduce = (
            self.tp_size > 1
            and os.environ.get("MSTAR_GLM52_MOE_FUSED_ALLREDUCE", "0") == "1"
        )
        self.shared_expert = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            comm_group=comm_group,
            activation=config.hidden_act,
            bias=False,
            reduce_results=not self._fused_allreduce,
        )

    def _attach_expert_weight_loaders(self) -> None:
        """Reattach per-shard loaders after ``_apply`` rebuilds parameters."""
        from functools import partial

        full_inter = self.moe_intermediate_size
        if self.fp8_experts:
            bo, bi = self.block_size
            self.experts.gate_up_proj_fp8.weight_loader = partial(
                _gate_up_fp8_loader, self.tp_rank, self.tp_size, full_inter, 1,
            )
            self.experts.gate_up_proj_scale_inv.weight_loader = partial(
                _gate_up_fp8_loader, self.tp_rank, self.tp_size, full_inter, bo,
            )
            self.experts.down_proj_fp8.weight_loader = partial(
                _down_fp8_loader, self.tp_rank, self.tp_size, full_inter, 1,
            )
            self.experts.down_proj_scale_inv.weight_loader = partial(
                _down_fp8_loader, self.tp_rank, self.tp_size, full_inter, bi,
            )
        else:
            self.experts.gate_up_proj.weight_loader = partial(
                _gate_up_weight_loader, self.tp_rank, self.tp_size, full_inter,
            )
            self.experts.down_proj.weight_loader = partial(
                _down_proj_weight_loader, self.tp_rank, self.tp_size, full_inter,
            )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_expert_weight_loaders()
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size).contiguous()

        topk_weights, topk_ids = self.gate(flat)
        topk_weights = topk_weights.to(flat.dtype)
        if self.fp8_experts:
            if self._use_fused:
                from mstar.utils.fused_moe import fused_experts_fp8

                routed = fused_experts_fp8(
                    flat,
                    self.experts.gate_up_proj_fp8,
                    self.experts.down_proj_fp8,
                    self.experts.gate_up_proj_scale_inv,
                    self.experts.down_proj_scale_inv,
                    topk_weights, topk_ids,
                    block_size=self.block_size,
                )
                if self.tp_size > 1 and not self._fused_allreduce:
                    self.comm_group.all_reduce(routed)
            else:
                routed = self._dispatch_fp8_reference(
                    flat, topk_weights, topk_ids,
                    reduce=not self._fused_allreduce)
        elif self.tp_size == 1:
            routed = _dispatch(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                self.num_experts,
                topk_ids,
                topk_weights,
            )
        else:
            routed = dispatch_experts_fused(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                self.num_experts,
                topk_ids,
                topk_weights,
            )
            if not self._fused_allreduce:
                self.comm_group.all_reduce(routed)
        shared = self.shared_expert(flat)
        out = routed + shared
        if self._fused_allreduce:
            # Both terms are per-rank partials here; one reduce for the sum.
            self.comm_group.all_reduce(out)
        return out.view(input_shape)

    def _dispatch_fp8_reference(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Per-expert loop that dequantizes only the experts this batch hit.

        Same loop shape as the shared ``dispatch_experts_fused``; the weighted
        ``index_add`` makes partial sums linear, so the TP all_reduce of the
        (T, hidden) result equals the stock all_reduce-then-sum path.
        """
        final = torch.zeros_like(flat)

        with torch.no_grad():
            expert_mask = F.one_hot(topk_ids, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            e = expert_idx_t[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            tokens = flat[token_idx]

            gate_up_w = dequantize_fp8_block_weight(
                self.experts.gate_up_proj_fp8[e],
                self.experts.gate_up_proj_scale_inv[e],
                block_size=self.block_size, out_dtype=flat.dtype,
            )
            down_w = dequantize_fp8_block_weight(
                self.experts.down_proj_fp8[e],
                self.experts.down_proj_scale_inv[e],
                block_size=self.block_size, out_dtype=flat.dtype,
            )
            gate, up = torch.mm(tokens, gate_up_w.T).chunk(2, dim=-1)
            out = torch.mm(F.silu(gate) * up, down_w.T)
            out = out * topk_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        if reduce and self.tp_size > 1:
            self.comm_group.all_reduce(final)
        return final

    def process_weights_after_loading(self, device) -> None:
        """Resolve reference-vs-fused dispatch on the real device (kimi
        quant_kernel semantics: explicit "triton" must not silently
        downgrade; "auto" probes; "reference" keeps the bitwise loop)."""
        if not self.fp8_experts:
            return
        dev = torch.device(device) if device is not None else torch.device("cpu")
        kernel = self.quant_kernel
        fused_ok = dev.type == "cuda"
        if fused_ok:
            try:
                from mstar.utils.fused_moe import fused_experts_fp8  # noqa: F401
            except Exception:
                fused_ok = False
        if kernel == "triton" and not fused_ok:
            raise RuntimeError(
                "moe_quant_kernel='triton' requested but the fused fp8 path "
                "is unavailable (needs CUDA + triton). Use 'auto' to fall "
                "back to the reference dispatch."
            )
        self._use_fused = kernel == "triton" or (kernel == "auto" and fused_ok)
        global _BACKEND_LOGGED
        if not _BACKEND_LOGGED:
            logger.info(
                "Glm52SparseMoeBlock routed-expert backend: %s "
                "(moe_quant_kernel=%s, block_size=%s, tp_size=%d).",
                "fused_experts_fp8 W8A8" if self._use_fused
                else "fp8-resident reference dispatch",
                kernel, self.block_size, self.tp_size,
            )
            _BACKEND_LOGGED = True
