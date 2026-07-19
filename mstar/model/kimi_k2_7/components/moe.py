"""Kimi-K2.7 / DeepSeek-V3 fine-grained MoE.

mstar's ``model.components.moe`` router is softmax-only and its shared-expert
block gates the shared expert (Qwen-style). Kimi/DeepSeek-V3 needs a different
router and an *ungated* shared expert, so these live here (append, don't modify
the shared abstraction). The expert dispatch itself is reused verbatim — the
fused-expert GEMM (``fused_experts`` via ``model.components.moe._dispatch``) and
the ``(E, 2*moe_inter, hidden)`` / ``(E, hidden, moe_inter)`` fused param layout.

Two pieces:

* :class:`KimiMoEGate` — the router. sigmoid scoring + group-limited top-k
  (``n_group`` / ``topk_group``) + ``noaux_tc`` per-expert
  ``e_score_correction_bias`` (affects *selection* only; the combine weights come
  from the raw sigmoid scores) + optional ``norm_topk_prob`` + a
  ``routed_scaling_factor`` folded into the returned weights. Computed in fp32.
  Exactly mirrors vLLM ``fused_moe/cpu_fused_moe.py::grouped_topk``.
* :class:`KimiSparseMoeBlock` — router + fused routed experts + ungated shared
  expert. ``out = routed(scaled weights) + shared`` (the shared expert does *not*
  get ``routed_scaling_factor``). Mirrors vLLM ``deepseek_v2.py::DeepseekV2MoE``.
"""
from __future__ import annotations

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
)
from mstar.model.kimi_k2_7.config import KimiK2Config


class KimiMoEGate(nn.Module):
    """DeepSeek-V3 group-limited sigmoid router with ``noaux_tc`` bias.

    Args:
        hidden_size: input hidden dim.
        n_routed_experts: number of routed experts (``E``).
        num_experts_per_tok: top-k experts per token.
        n_group: number of expert groups (``E`` split into ``n_group`` contiguous
            groups for group-limited routing).
        topk_group: number of groups kept per token.
        routed_scaling_factor: scale folded into the returned combine weights.
        scoring_func: ``"sigmoid"`` (Kimi/DeepSeek-V3) or ``"softmax"``.
        topk_method: ``"noaux_tc"`` enables the per-expert
            ``e_score_correction_bias`` (selection-only). Anything else disables it.
        norm_topk_prob: renormalize the top-k combine weights to sum to 1.
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob

        # Router projection ``[E, hidden]`` (no bias), like DeepSeek ``MoEGate``.
        self.weight = nn.Parameter(torch.zeros(n_routed_experts, hidden_size))
        if topk_method == "noaux_tc":
            # Per-expert selection bias; fp32, added to scores for group/top-k
            # selection but never to the combine weights.
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(n_routed_experts, dtype=torch.float32)
            )
        else:
            self.register_parameter("e_score_correction_bias", None)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Returns:
            topk_weights: ``(tokens, top_k)`` fp32 combine weights (renormalized
                and scaled by ``routed_scaling_factor``).
            topk_ids: ``(tokens, top_k)`` int64 expert indices.
        """
        # Route in fp32 (DeepSeek runs the router in fp32 for stability).
        h = hidden_states.reshape(-1, self.hidden_size).float()
        gating = F.linear(h, self.weight.float())  # (T, E)

        if self.scoring_func == "sigmoid":
            scores = gating.sigmoid()
        elif self.scoring_func == "softmax":
            scores = gating.softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func!r}")

        num_token = scores.shape[0]
        if self.e_score_correction_bias is not None:
            # noaux_tc: bias-added scores drive group + expert *selection*; the
            # raw sigmoid scores drive the combine weights.
            original_scores = scores
            scores = scores + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores.view(num_token, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # (T, n_group)
        else:
            original_scores = scores
            group_scores = scores.view(num_token, self.n_group, -1).max(dim=-1).values

        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]  # (T, topk_group)
        group_mask = torch.zeros_like(group_scores)  # (T, n_group)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, self.n_group, scores.shape[-1] // self.n_group)
            .reshape(num_token, -1)
        )  # (T, E)
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

        if self.e_score_correction_bias is not None:
            topk_ids = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)[1]
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(
                masked_scores, k=self.top_k, dim=-1, sorted=False
            )

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_ids


class KimiSparseMoeBlock(nn.Module):
    """DeepSeek-V3 MoE block: routed experts + ungated shared expert.

    ``out = routed(x) + shared(x)`` where ``routed`` dispatches the top-k experts
    through the fused-expert GEMM with the router's (scaled) combine weights, and
    ``shared`` is a plain dense SwiGLU MLP added ungated (no sigmoid gate, no
    ``routed_scaling_factor``).

    **TP sharding (intermediate-parallel).** Under tensor parallelism the router
    (:class:`KimiMoEGate`) stays REPLICATED — every rank computes the full
    ``(top_k_ids, weights)`` — and only the expert GEMMs shard, exactly like
    mstar's own ``ParallelSparseMoeBlock``: each rank holds every expert but only
    a ``moe_intermediate_size // tp_size`` slice of its SwiGLU intermediate
    (``gate_up_proj`` column-parallel, ``down_proj`` row-parallel). The per-rank
    partial hidden contributions are summed with a single all-reduce before the
    top-k sum-reduce. The shared expert is a ``ParallelGatedMLP`` on the same comm
    group, so it shards its intermediate and all-reduces internally. This reuses
    the existing fused-expert machinery verbatim and is trivially goldenable
    (tp>1 == tp=1). Its tradeoff: every rank still stores ALL experts' weights, so
    it does NOT reduce per-rank expert memory — the 1T fit needs true
    token-dispatch expert parallelism (all-to-all), which is a Phase-4 concern and
    is deliberately not built here.

    Expert weights use the fused layout reused from ``model.components.moe``,
    sharded to this rank (``full == moe_intermediate_size``,
    ``shard == full // tp_size``):
      - ``experts.gate_up_proj``: ``(E, 2 * shard, hidden)``
      - ``experts.down_proj``:   ``(E, hidden, shard)``
    """

    def __init__(
        self, config: KimiK2Config, comm_group: CommGroup | None = None
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
        # Per-rank slice of each expert's SwiGLU intermediate (== full at tp=1).
        shard_inter = divide(config.moe_intermediate_size, self.tp_size)

        self.gate = KimiMoEGate(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            scoring_func=config.scoring_func,
            topk_method=config.topk_method,
            norm_topk_prob=config.norm_topk_prob,
        )

        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                2 * shard_inter,
                config.hidden_size,
            )
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                config.hidden_size,
                shard_inter,
            )
        )
        # The fused expert params are plain nn.Parameters, so they carry no
        # per-shard ``weight_loader`` by default. The M5 stacked-param rules route
        # each checkpoint expert via a ``"gate:N"/"up:N"/"down:N"`` shard id, so we
        # attach the same fused-expert loaders ``ParallelSparseMoeBlock`` uses.
        # The loaders take ``(tp_rank, tp_size, full_inter)`` and slice this rank's
        # intermediate stripe out of the full-size checkpoint expert — so a single
        # weight path serves tp=1 (full) and tp>1 (sharded).
        self._attach_expert_weight_loaders()

        # Ungated shared expert: a dense SwiGLU MLP with the shared intermediate
        # size (``moe_intermediate_size * n_shared_experts``).
        self.shared_expert = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            comm_group=comm_group,
            activation=config.hidden_act,
            bias=False,
        )

    def _attach_expert_weight_loaders(self) -> None:
        """Give the fused expert params their per-shard ``weight_loader``.

        Mirrors ``ParallelSparseMoeBlock._attach_weight_loaders``. Re-run after
        every ``_apply`` (``.to(dtype)`` / ``to_empty(device)`` rebuild the
        Parameter objects and drop the attribute), so weights load correctly
        through the meta -> to_empty -> load path.
        """
        from functools import partial

        self.experts.gate_up_proj.weight_loader = partial(
            _gate_up_weight_loader, self.tp_rank, self.tp_size, self.moe_intermediate_size,
        )
        self.experts.down_proj.weight_loader = partial(
            _down_proj_weight_loader, self.tp_rank, self.tp_size, self.moe_intermediate_size,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_expert_weight_loaders()
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size).contiguous()

        # Router is replicated: every rank computes the full top-k selection.
        topk_weights, topk_ids = self.gate(flat)
        topk_weights = topk_weights.to(flat.dtype)
        if self.tp_size == 1:
            routed = _dispatch(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                self.num_experts,
                topk_ids,
                topk_weights,
            )
        else:
            routed = self._dispatch_tp(flat, topk_weights, topk_ids)
        # Shared expert is a ParallelGatedMLP on the same comm group: at tp>1 it
        # holds its own intermediate stripe and all-reduces inside its down_proj.
        shared = self.shared_expert(flat)
        return (routed + shared).view(input_shape)

    def _dispatch_tp(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Intermediate-sharded routed dispatch (mirrors
        ``ParallelSparseMoeBlock._dispatch_tp``).

        Each rank's ``fused_experts`` produces its partial hidden contribution per
        (token, top-k slot); an all-reduce sums the intermediate-dim partials
        across ranks, then ``moe_sum_reduce_triton`` folds the top-k dim. The
        combine weights already carry ``routed_scaling_factor`` (folded in by
        :class:`KimiMoEGate`), so the sum-reduce passes ``routed_scaling_factor=1.0``.
        """
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        # (tokens, top_k, hidden) partials — reduce_results=False keeps the
        # per-slot rows so we can all-reduce the intermediate-parallel partials.
        cache3 = fused_experts(
            flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            topk_weights,
            topk_ids,
            reduce_results=False,
        )
        self.comm_group.all_reduce(cache3)
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
        return output
