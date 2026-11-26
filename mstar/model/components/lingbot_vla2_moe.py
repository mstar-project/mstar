"""LingBot-VLA 2.0's action-decoder MoE MLP, on shared M* expert modules.

The released RoboTwin action tower uses this block on all 36 layers. This
module covers the MLP only: prefix/expert attention, FiLM norms and the graph
that joins their rows are separate parts of the port.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.moe import FusedExperts, NoAuxTcRouter
from mstar.model.loader.base import default_weight_loader
from mstar.model.loader.iterators import iter_safetensors_shards


@dataclass(frozen=True)
class LingBotVLAActionMoeConfig:
    hidden_size: int
    num_experts: int
    top_k: int
    routed_intermediate_size: int
    shared_intermediate_size: int
    routed_scaling_factor: float


class LingBotVLAActionMoe(nn.Module):
    """Sigmoid routing with a correction bias and an ungated shared expert.

    This first component uses the released checkpoint's TP1 layout. The
    router and expert dispatch are M*'s shared implementations.
    """

    def __init__(self, config: LingBotVLAActionMoeConfig):
        super().__init__()
        if min(config.hidden_size, config.num_experts, config.top_k,
               config.routed_intermediate_size, config.shared_intermediate_size) <= 0:
            raise ValueError("LingBot-VLA MoE dimensions must be positive")
        if config.top_k > config.num_experts:
            raise ValueError("LingBot-VLA MoE top_k exceeds num_experts")
        if config.routed_scaling_factor <= 0:
            raise ValueError("LingBot-VLA MoE routing scale must be positive")
        self.config = config
        self.gate = NoAuxTcRouter(
            hidden_size=config.hidden_size,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.top_k,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=True,
        )
        self.experts = FusedExperts(
            num_experts=config.num_experts, hidden_size=config.hidden_size,
            intermediate_size=config.routed_intermediate_size,
        )
        self.shared_expert = ParallelGatedMLP(
            hidden_size=config.hidden_size, intermediate_size=config.shared_intermediate_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("LingBot-VLA MoE input must be [batch, tokens, hidden_size]")
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        weights, expert_ids, _ = self.gate(flat)
        routed = self.experts(flat, weights.to(flat.dtype), expert_ids)
        shared = self.shared_expert(flat)
        return (routed + shared).reshape_as(hidden_states)


def load_action_moe_weights(
    moe: LingBotVLAActionMoe, checkpoint_dir: str | Path, layer: int,
    *, device: str | torch.device = "cpu",
) -> set[str]:
    """Read one layer's eight MoE tensors from the released VLA checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"LingBot-VLA MoE requires {index_path}")
    prefix = f"model.qwenvl_with_expert.qwen_expert.model.layers.{layer}.mlp."
    names = (
        "e_score_correction_bias", "experts.down_proj", "experts.gate_proj", "experts.up_proj",
        "gate.weight", "shared_expert.down_proj.weight", "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
    )
    expected = {prefix + name for name in names}
    actual = {name for name in json.loads(index_path.read_text())["weight_map"] if name.startswith(prefix)}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"LingBot-VLA MoE checkpoint keys differ: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} extra (e.g. {extra[:3]})"
        )
    weights = {
        name.removeprefix(prefix): value
        for name, value in iter_safetensors_shards(checkpoint_dir, device=device, keys=expected)
    }
    if set(weights) != set(names):
        raise ValueError("LingBot-VLA MoE checkpoint did not yield every indexed tensor")
    with torch.no_grad():
        default_weight_loader(moe.gate.weight, weights["gate.weight"])
        default_weight_loader(moe.gate.e_score_correction_bias, weights["e_score_correction_bias"])
        default_weight_loader(moe.experts.gate_up_proj, torch.cat((
            weights["experts.gate_proj"], weights["experts.up_proj"],
        ), dim=1))
        default_weight_loader(moe.experts.down_proj, weights["experts.down_proj"])
        default_weight_loader(moe.shared_expert.gate_up_proj.weight, torch.cat((
            weights["shared_expert.gate_proj.weight"], weights["shared_expert.up_proj.weight"],
        ), dim=0))
        default_weight_loader(moe.shared_expert.down_proj.weight, weights["shared_expert.down_proj.weight"])
    return set(names)
