"""GLM-5.2 fine-grained MoE: the noaux_tc router (components/moe.py) + FP8-resident experts."""
from __future__ import annotations

import logging

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.moe import (
    NoAuxTcRouter,
    _dispatch,
    _down_proj_weight_loader,
    _gate_up_weight_loader,
    dispatch_experts_fused,
)
from mstar.model.components.quantization import QuantMethod
from mstar.model.glm52.config import Glm52ModelConfig, glm52_quant_method

logger = logging.getLogger(__name__)

_BACKEND_LOGGED = False
# accepted moe_quant_kernel values; see Glm52ModelConfig.moe_quant_kernel
MOE_QUANT_KERNELS = ("reference", "triton", "auto")


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

        self.quant_kernel = getattr(config, "moe_quant_kernel", "reference")
        if self.quant_kernel not in MOE_QUANT_KERNELS:
            # an unknown value would otherwise resolve to the uncapturable
            # reference loop and serve eager behind a healthy /health
            raise ValueError(
                f"moe_quant_kernel={self.quant_kernel!r} is not one of "
                f"{MOE_QUANT_KERNELS}"
            )
        quant: QuantMethod | None = glm52_quant_method(
            config.quantization_config, config.moe_fp8_resident, self.quant_kernel)
        self.fp8_experts = quant is not None and quant.keeps_quantized("routed_experts")

        self.gate = NoAuxTcRouter(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
        )

        if self.fp8_experts:
            self.block_size = quant.format_for("routed_experts").block
            self.experts = quant.routed_experts(
                config.n_routed_experts, config.moe_intermediate_size, config.hidden_size,
                comm_group)
        else:
            self.experts = nn.Module()
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

        self.shared_expert = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            comm_group=comm_group,
            activation=config.hidden_act,
            bias=False,
        )

    @property
    def _use_fused(self) -> bool:
        """Whether the fp8 experts resolved to the fused kernel."""
        return self.fp8_experts and self.experts.fused

    @_use_fused.setter
    def _use_fused(self, value: bool) -> None:
        self.experts.fused = value

    def _attach_expert_weight_loaders(self) -> None:
        """Reattach the bf16 experts' per-shard loaders after ``_apply`` rebuilds parameters
        (fp8 experts keep their own)."""
        from functools import partial

        if self.fp8_experts:
            return
        full_inter = self.moe_intermediate_size
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

        # topk_weights stay fp32 into the combine, as in HF's experts.
        topk_weights, topk_ids, _ = self.gate(flat)
        if self.fp8_experts:
            routed = self.experts(flat, topk_weights, topk_ids)
            if self.tp_size > 1:
                self.comm_group.all_reduce(routed)
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
            self.comm_group.all_reduce(routed)
        out = routed + self.shared_expert(flat)
        return out.view(input_shape)

    def process_weights_after_loading(self, device) -> None:
        """Resolve reference-vs-fused dispatch on the real device: explicit
        "triton" must not silently downgrade, "auto" probes for the fused
        path, "reference" keeps the bitwise loop."""
        self.gate.finalize_weights()
        if not self.fp8_experts:
            return
        self.experts.resolve_kernels(device)
        global _BACKEND_LOGGED
        if not _BACKEND_LOGGED:
            logger.info(
                "Glm52SparseMoeBlock routed-expert backend: %s "
                "(moe_quant_kernel=%s, block_size=%s, tp_size=%d).",
                "fused_experts_fp8 W8A8" if self._use_fused
                else "fp8-resident reference dispatch",
                self.quant_kernel, self.block_size, self.tp_size,
            )
            _BACKEND_LOGGED = True
