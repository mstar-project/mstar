"""MoE router (spec E.1): sigmoid scores, bias-corrected selection, unbiased weights.

Matches ``KimiMoEGate.forward`` for ``num_expert_group == 1`` (K3) and implements the
grouped top-k branch for completeness.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def noaux_tc_route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    *,
    scoring: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(topk_idx [T, k] int64, topk_weight [T, k] fp32)``."""
    # routing is fp32 in the reference; keep it so under the engine's bf16 autocast
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        return _route_fp32(
            hidden_states, gate_weight, e_score_correction_bias, top_k, scoring=scoring,
            renormalize=renormalize, routed_scaling_factor=routed_scaling_factor,
            num_expert_group=num_expert_group, topk_group=topk_group,
        )


def _route_fp32(
    hidden_states, gate_weight, e_score_correction_bias, top_k, *, scoring, renormalize,
    routed_scaling_factor, num_expert_group, topk_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    logits = F.linear(x, gate_weight.float())
    if scoring == "sigmoid":
        scores = logits.sigmoid()
    elif scoring == "softmax":
        scores = logits.softmax(dim=-1)
    else:
        raise NotImplementedError(scoring)
    scores_for_choice = scores + e_score_correction_bias.float().unsqueeze(0)
    if num_expert_group > 1 and num_expert_group > topk_group:
        n, e = scores_for_choice.shape
        group_scores = (
            scores_for_choice.view(n, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1).expand(n, num_expert_group, e // num_expert_group).reshape(n, -1)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if top_k > 1 and renormalize:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor
    return topk_idx, topk_weight
