"""Kimi K3's MLPs: the SiTU-GLU dense MLP and the latent MoE.

The latent MoE routes each token on its hidden state (896 experts, top 16, sigmoid noaux_tc),
projects it down into the routed experts' latent (``routed_expert_down_proj``, 7168 -> 3584),
runs the experts there, normalizes their sum (``routed_expert_norm``) and projects it back up
(``routed_expert_up_proj``); the two shared experts (one MLP twice as wide) read the hidden
state. The block is the shared ``ParallelSparseMoeBlock`` (tensor parallel over the experts'
intermediate dim) or ``ExpertParallelSparseMoeBlock`` (whole experts per rank) over the latent,
so ``<block>.experts`` is the routed experts layer the quantization API sees, and the router
is the shared ``NoAuxTcRouter``. After merceod's #257 (its ``KimiLatentMoE``) without its
fused-kernel and collective rewrites, which main has its own versions of or lacks.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ParallelGatedMLP, RowParallelLinear
from mstar.model.components.moe import (
    ExpertParallelSparseMoeBlock,
    NoAuxTcRouter,
    ParallelSparseMoeBlock,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.utils.situ import SiTU


class KimiSiTUMLP(ParallelGatedMLP):
    """``ParallelGatedMLP`` with SiTU-GLU, which bends the up half too: ``act(gate) * up`` does
    not express it."""

    def __init__(self, *args, situ: SiTU, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.situ = situ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.intermediate_size_per_partition, dim=-1)
        return self.down_proj(self.situ(gate, up))


class _LatentMoE:
    """The latent MoE over a routed-experts block (``ParallelSparseMoeBlock`` or its expert
    parallel twin, the next base in the MRO): this rank's partial of the layer's output, summed
    in one all-reduce with the shared experts' partial."""

    def __init__(self, config: KimiK3ModelConfig, comm_group: CommGroup | None = None) -> None:
        comm_group = comm_group or CommGroup.trivial()
        latent = config.routed_expert_hidden_size
        super().__init__(
            hidden_size=latent,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            router=NoAuxTcRouter(
                hidden_size=config.hidden_size,
                n_routed_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_topk_prob=config.moe_renormalize,
            ),
            comm_group=comm_group,
            activation=config.situ(),
        )
        self.model_hidden_size = config.hidden_size
        # Replicated: one GEMM with no collective, which at decode beats an all-gather of a
        # column-parallel projection (51 MB a layer, read once).
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, latent, bias=False)
        self.routed_expert_norm = (
            RMSNorm(latent, eps=config.rms_norm_eps) if config.latent_moe_use_norm else None)
        # Row-parallel over the (replicated, normed) latent: its partial joins the shared
        # experts' in the layer's one all-reduce.
        self.routed_expert_up_proj = RowParallelLinear(
            comm_group, latent, config.hidden_size, bias=False,
            input_is_parallel=False, reduce_results=False)
        self.shared_experts = None
        if config.num_shared_experts:
            self.shared_experts = KimiSiTUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                comm_group=comm_group, situ=config.situ(), reduce_results=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.model_hidden_size)
        routing_weights, selected_experts, _ = self.gate(x)
        z = self.routed_expert_down_proj(x)
        # summed over the ranks: the norm needs every intermediate shard (or expert)
        y = self.experts_forward(z, routing_weights, selected_experts)
        if self.routed_expert_norm is not None:
            y = self.routed_expert_norm(y)
        out = self.routed_expert_up_proj(y)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        if self.comm_group.world_size > 1:
            out = self.comm_group.all_reduce(out)
        return out.view(shape)


class KimiLatentMoE(_LatentMoE, ParallelSparseMoeBlock):
    """Routed experts sharded over the TP group by intermediate dim (3072 / 8 = 384 at TP8)."""


class KimiLatentMoEExpertParallel(_LatentMoE, ExpertParallelSparseMoeBlock):
    """Routed experts as whole experts per rank (896 / 8 = 112 at EP8)."""


def build_mlp(config: KimiK3ModelConfig, layer_idx: int, comm_group: CommGroup | None = None):
    """Layer ``layer_idx``'s MLP: the dense SiTU MLP below ``first_k_dense_replace``, else the
    latent MoE."""
    if not config.is_moe_layer(layer_idx):
        return KimiSiTUMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            comm_group=comm_group, situ=config.situ())
    cls = KimiLatentMoEExpertParallel if config.moe_expert_parallel else KimiLatentMoE
    return cls(config, comm_group=comm_group)
