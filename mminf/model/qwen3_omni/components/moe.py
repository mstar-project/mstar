"""Mixture-of-Experts routing for Qwen3-Omni Thinker and Talker.

Provides three modules:

* :class:`Qwen3OmniMLP` -- SwiGLU MLP used as individual expert *and* as the
  dense MLP in ``mlp_only_layers``.
* :class:`Qwen3OmniSparseMoeBlock` -- Top-K MoE for the **Thinker** text
  backbone (no shared expert).
* :class:`Qwen3OmniTalkerSparseMoeBlock` -- Top-K MoE for the **Talker** text
  backbone (adds a shared expert with sigmoid gating).

Expert weights use fused Parameters matching the HF checkpoint layout
(``experts.gate_up_proj``, ``experts.down_proj``) rather than per-expert
``nn.Linear`` modules.  Dispatch uses a naive PyTorch expert loop that
will be replaced with fused kernels in Phase 3.

Reference
---------
HF ``Qwen3OmniMoeThinkerTextSparseMoeBlock`` and
``Qwen3OmniMoeTalkerTextSparseMoeBlock`` from ``modeling_qwen3_omni_moe.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

# -----------------------------------------------------------------------
# SwiGLU MLP  (used as individual experts AND as dense MLP)
# -----------------------------------------------------------------------

class Qwen3OmniMLP(nn.Module):
    """SwiGLU feedforward: ``down(silu(gate(x)) * up(x))``.

    Used as:
      - An individual expert inside the MoE blocks.
      - The dense MLP for ``mlp_only_layers`` that bypass MoE routing.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension (input & output size).
    intermediate_size : int
        Inner expansion dimension.  For routed experts this is
        ``moe_intermediate_size`` (e.g. 768 for Thinker, 384 for Talker);
        for dense layers it is the full ``intermediate_size``.
    bias : bool
        Whether linear layers have a bias term (default ``False``).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape ``(..., hidden_size)``

        Returns
        -------
        torch.Tensor  shape ``(..., hidden_size)``
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------
# Top-K Router
# -----------------------------------------------------------------------

class _TopKRouter(nn.Module):
    """Softmax top-k router shared by Thinker and Talker MoE blocks.

    Parameters
    ----------
    hidden_size : int
        Input hidden dimension.
    num_experts : int
        Total number of routed experts.
    num_experts_per_tok : int
        Number of experts each token is dispatched to (top-k).
    norm_topk_prob : bool
        If ``True``, renormalize the top-k probabilities so they sum to 1.
        Thinker uses ``True``; Talker uses ``False``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor  shape ``(tokens, hidden_size)``

        Returns
        -------
        router_logits : torch.Tensor  shape ``(tokens, num_experts)``
            Full softmax distribution (for auxiliary losses).
        routing_weights : torch.Tensor  shape ``(tokens, top_k)``
            Top-k probabilities (optionally renormalized).
        selected_experts : torch.LongTensor  shape ``(tokens, top_k)``
            Indices of the chosen experts.
        """
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        # (tokens, num_experts)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)

        # Top-k selection
        router_top_value, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )

        routing_weights = router_top_value.to(router_logits.dtype)
        return router_logits, routing_weights, router_indices


# -----------------------------------------------------------------------
# Naive expert dispatch  (loop over active experts -- fused weight format)
# -----------------------------------------------------------------------

def _dispatch_experts_fused(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Scatter tokens to experts and gather weighted results (fused weights).

    Uses the HF fused expert weight layout where gate and up projections
    are stored in a single ``gate_up_proj`` tensor.

    Parameters
    ----------
    hidden_states : torch.Tensor  shape ``(tokens, hidden_size)``
    gate_up_proj : torch.Tensor  shape ``(num_experts, 2 * moe_intermediate_size, hidden_size)``
        Fused gate + up projection weights for all experts.
    down_proj : torch.Tensor  shape ``(num_experts, hidden_size, moe_intermediate_size)``
        Down projection weights for all experts.
    num_experts : int
        Total number of experts.
    selected_experts : torch.LongTensor  shape ``(tokens, top_k)``
    routing_weights : torch.Tensor  shape ``(tokens, top_k)``

    Returns
    -------
    torch.Tensor  shape ``(tokens, hidden_size)``
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    # Build a mask: (num_experts, top_k, tokens) -- one-hot over expert dim
    with torch.no_grad():
        expert_mask = F.one_hot(
            selected_experts, num_classes=num_experts
        )  # (tokens, top_k, num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, tokens)
        # Identify which experts are actually used
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t[0]
        # top_k_pos: which top-k slot, token_idx: which tokens
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # Fused gate+up projection: [tokens, 2*inter]
        gate_up = torch.mm(current_state, gate_up_proj[expert_idx].T)
        gate, up = gate_up.chunk(2, dim=-1)
        # SwiGLU + down projection: [tokens, hidden]
        current_hidden_states = torch.mm(F.silu(gate) * up, down_proj[expert_idx].T)

        current_hidden_states = (
            current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


# -----------------------------------------------------------------------
# Thinker MoE  (no shared expert)
# -----------------------------------------------------------------------

class Qwen3OmniSparseMoeBlock(nn.Module):
    """Top-K Sparse MoE block for the Qwen3-Omni **Thinker** text backbone.

    Uses fused expert weights matching the HF checkpoint layout:
      - ``experts.gate_up_proj``: ``(num_experts, 2 * moe_intermediate_size, hidden_size)``
      - ``experts.down_proj``: ``(num_experts, hidden_size, moe_intermediate_size)``

    Config defaults (Thinker):
      - ``hidden_size = 2048``
      - ``num_experts = 128``
      - ``num_experts_per_tok = 8``
      - ``moe_intermediate_size = 768``
      - ``norm_topk_prob = True``

    Parameters
    ----------
    hidden_size : int
    num_experts : int
    num_experts_per_tok : int
    moe_intermediate_size : int
    norm_topk_prob : bool
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 768,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = _TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )

        # Fused expert weights matching HF checkpoint layout
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(batch_size, seq_len, hidden_size)`` or
            ``(tokens, hidden_size)`` for packed/disaggregated inputs.

        Returns
        -------
        torch.Tensor  same shape as input.
        """
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        _, routing_weights, selected_experts = self.gate(hidden_states_flat)

        final = _dispatch_experts_fused(
            hidden_states_flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            self.num_experts,
            selected_experts,
            routing_weights,
        )

        return final.view(input_shape)


# -----------------------------------------------------------------------
# Talker MoE  (with shared expert + sigmoid gate)
# -----------------------------------------------------------------------

class Qwen3OmniTalkerSparseMoeBlock(nn.Module):
    """Top-K Sparse MoE block for the Qwen3-Omni **Talker** text backbone.

    Compared to the Thinker MoE, the Talker adds:
      - A **shared expert** (dense MLP) that processes *every* token.
      - A **shared expert gate** (``Linear(hidden_size, 1)`` with sigmoid)
        that modulates the shared expert output before adding it to the
        routed expert output.

    The final output is::

        output = routed_output + sigmoid(shared_gate(x)) * shared_expert(x)

    Uses fused expert weights matching the HF checkpoint layout for routed
    experts.  The shared expert remains a separate :class:`Qwen3OmniMLP`
    (dense format, not fused).

    Config defaults (Talker):
      - ``hidden_size = 1024``
      - ``num_experts = 128``
      - ``num_experts_per_tok = 8``
      - ``moe_intermediate_size = 384``
      - ``norm_topk_prob = False``
      - ``shared_expert_intermediate_size`` -- from config (e.g. 2048)

    Parameters
    ----------
    hidden_size : int
    num_experts : int
    num_experts_per_tok : int
    moe_intermediate_size : int
    norm_topk_prob : bool
    shared_expert_intermediate_size : int
        Intermediate size for the shared expert MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 384,
        norm_topk_prob: bool = False,
        shared_expert_intermediate_size: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        # Routed experts -- fused weights matching HF checkpoint layout
        self.gate = _TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )

        # Shared expert (dense format, not fused)
        self.shared_expert = Qwen3OmniMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(batch_size, seq_len, hidden_size)`` or
            ``(tokens, hidden_size)`` for packed/disaggregated inputs.

        Returns
        -------
        torch.Tensor  same shape as input.
        """
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Shared expert -- applied to ALL tokens
        shared_output = self.shared_expert(hidden_states_flat)

        # Routed experts -- top-k dispatch
        _, routing_weights, selected_experts = self.gate(hidden_states_flat)
        routed_output = _dispatch_experts_fused(
            hidden_states_flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            self.num_experts,
            selected_experts,
            routing_weights,
        )

        # Combine: routed + sigmoid-gated shared
        shared_gate = torch.sigmoid(
            self.shared_expert_gate(hidden_states_flat)
        )
        combined = routed_output + shared_gate * shared_output

        return combined.view(input_shape)
