"""Qwen3.5-MoE's sparse block: top-k routed experts beside a shared expert scaled by a sigmoid gate.

Every layer of a ``qwen3_5_moe`` checkpoint has one in place of the dense MLP. Under TP each rank holds a slice
of every expert's intermediate width (and of the shared expert's), so both halves come out as this rank's partial
sum, and the block returns their sum unreduced: the decoder loop reduces it in the next norm's fused all-reduce +
add + norm (``run_layers``), as it does the dense MLP's row-parallel output. The sigmoid gate reads the full
(replicated) input, so scaling a partial sum by it commutes with the reduce.
"""
from __future__ import annotations

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.components.moe import (
    ParallelSparseMoeBlockWithSharedExpert,
    TopKRouter,
    _dispatch,
    _route,
)
from mstar.model.qwen3_5.config import Qwen3_5Config


class Qwen3_5SparseMoeBlock(ParallelSparseMoeBlockWithSharedExpert):
    """``ParallelSparseMoeBlockWithSharedExpert`` returning a TP partial sum (see the module docstring). The
    routed experts are ``<block>.experts``, one quantizable layer (``QuantExperts``): bf16, or NVFP4 / block fp8
    as a checkpoint stores them."""

    def __init__(self, config: Qwen3_5Config, comm_group: CommGroup | None = None):
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            shared_expert=ParallelGatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                comm_group=comm_group, activation="silu", reduce_results=False,
            ),
            # HF's Qwen3_5MoeTopKRouter: softmax over every expert in fp32, the top k renormalized
            router=TopKRouter(
                config.hidden_size, config.num_experts, config.num_experts_per_tok, norm_topk_prob=True,
            ),
            comm_group=comm_group,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        if self.experts_quantized:
            weights, experts, _, _ = _route(self.gate, flat, None, None)
            routed = self._experts_quant(flat, weights, experts)
        else:
            # one rank's routing can align the batch for the fused kernel; under TP the kernel aligns for itself
            align = self.experts.gate_up_proj if self.comm_group.world_size == 1 else None
            weights, experts, _, alignment = _route(self.gate, flat, align, None)
            routed = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj, self.num_experts, experts, weights,
                alignment=alignment,
            )
        return (routed + shared).view(hidden_states.shape)
