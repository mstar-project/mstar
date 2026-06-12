"""Ling-2.0 MoE router with grouped expert selection.

Ling-2.0 (BailingMoeV2) uses ``router_type: "MultiRouter"``, which differs from
mstar's standard :class:`mstar.model.components.moe.TopKRouter` in four ways:

  * **Sigmoid** activation on the gate logits, not softmax.
  * A learned per-expert bias added to the routing scores before top-k —
    not gradient-trained on this checkpoint (stored as ``requires_grad=False``).
  * **Group-limited top-k**: the ``num_experts`` are partitioned into
    ``n_group`` groups; tokens may only route to experts within the
    ``topk_group`` highest-scoring groups (group score = sum of top-2
    expert scores in that group). This caps cross-group all-to-all
    bandwidth at the cost of expressiveness.
  * Weights are renormalised to sum to 1 across the chosen top-k and then
    multiplied by ``routed_scaling_factor``.

Returns the same 3-tuple as :class:`TopKRouter` (``logits, weights, indices``)
so it can drop into mstar's existing :class:`SparseMoeBlockWithSharedExpert`
and the fused-Triton dispatch path.

Reference: vllm-omni's ``BailingMoeV2Gate``
``/tmp/vllm-omni/vllm_omni/model_executor/models/ming_flash_omni/modeling_bailing_moe_v2.py:211-279``
and Ming upstream ``modeling_bailing_moe_v2.py:696-765``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LingMoeRouter(nn.Module):
    """Ling-2.0 ``MultiRouter`` (group-limited top-k with sigmoid + bias).

    Args:
        hidden_size: input hidden dimension.
        num_experts: total routed experts. Must divide evenly by ``n_group``.
        num_experts_per_tok: top-k experts selected per token.
        n_group: expert groups; the experts are split contiguously by
            ``num_experts // n_group``.
        topk_group: how many groups a single token may route into.
        routed_scaling_factor: post-renormalisation scale applied to the
            top-k weights (matches upstream ``routed_scaling_factor``).

    The gate ``nn.Linear`` weight is **replicated** across TP ranks in the
    parallel build (router decisions must be identical across ranks); for
    this step-3a unit-test scope we just expose a plain ``nn.Linear``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts % n_group != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by n_group={n_group}"
            )
        if topk_group > n_group:
            raise ValueError(
                f"topk_group={topk_group} cannot exceed n_group={n_group}"
            )
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.experts_per_group = num_experts // n_group
        self.routed_scaling_factor = routed_scaling_factor

        # Gate projection — replicated (no bias).
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert bias — not gradient-trained, but stored as a parameter so
        # state_dict loaders see it.
        self.expert_bias = nn.Parameter(
            torch.zeros(num_experts), requires_grad=False,
        )

    def _group_limited_topk(self, scores: torch.Tensor) -> torch.Tensor:
        """Pick the top-k experts under the ``topk_group``-best-groups constraint.

        Args:
            scores: ``(num_tokens, num_experts)``. Already sigmoid + bias.

        Returns:
            ``(num_tokens, top_k)`` int64 expert indices.

        Per-group score = sum of that group's top-2 expert scores. The
        ``topk_group`` groups with the highest per-group scores are kept;
        the rest are masked out before the final top-k.
        """
        num_tokens = scores.size(0)
        # (N, n_group, experts_per_group)
        grouped = scores.view(num_tokens, self.n_group, self.experts_per_group)
        # Per-group score: sum of top-2 expert scores in that group.
        # Matches upstream exactly (``.topk(2, dim=-1)[0].sum(dim=-1)``).
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        # Pick the topk_group best groups.
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        # Broadcast group mask back across experts_per_group.
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.experts_per_group)
            .reshape(num_tokens, -1)
        )
        # Mask un-selected groups' experts to -inf so they can't be picked.
        masked = scores.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: ``(..., hidden_size)``. Flattened internally.

        Returns:
            Three tensors matching :class:`TopKRouter`'s shape:
              - ``router_logits``: ``(N, num_experts)`` raw gate logits
                (pre-sigmoid). Kept as float32 for stability and parity
                with ``TopKRouter``.
              - ``routing_weights``: ``(N, top_k)`` normalised + scaled
                weights for the chosen experts.
              - ``selected_experts``: ``(N, top_k)`` int64 expert indices.
        """
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        # Linear is rank-replicated; the float() cast matches upstream's
        # ``logits = logits.float()`` for numeric stability.
        logits = F.linear(hidden_states, self.gate.weight).float()
        # Per-expert sigmoid (NOT softmax). Bias is added AFTER sigmoid
        # in the routing path; the gathered weights below pull from the
        # un-biased sigmoid scores.
        sigmoid_scores = torch.sigmoid(logits)
        scored_for_routing = sigmoid_scores + self.expert_bias

        selected_experts = self._group_limited_topk(scored_for_routing)
        # Gather the un-biased sigmoid score for the chosen experts.
        chosen_scores = torch.gather(
            sigmoid_scores, dim=1, index=selected_experts,
        ).to(logits.dtype)
        if self.top_k > 1:
            chosen_scores = chosen_scores / (
                chosen_scores.sum(dim=-1, keepdim=True) + 1e-20
            )
        routing_weights = chosen_scores * self.routed_scaling_factor

        return logits, routing_weights, selected_experts
