"""Mixture-of-Experts blocks.

Top-K router + dispatch path for the standard fused-expert checkpoint
layout (``experts.gate_up_proj`` and ``experts.down_proj`` packed as
``(num_experts, ...)`` parameters). Two block flavors:

* :class:`SparseMoeBlock` — Top-K MoE with no shared expert (e.g. the
  Qwen3-Omni Thinker text backbone).
* :class:`SparseMoeBlockWithSharedExpert` — Top-K MoE plus a shared
  expert with sigmoid gating (e.g. the Qwen3-Omni Talker text backbone).
  The shared expert is passed in as an ``nn.Module`` so callers can pick
  any MLP shape they need.

Parallel (TP-aware) variants:

* :class:`ParallelSparseMoeBlock`
* :class:`ParallelSparseMoeBlockWithSharedExpert`

Expert-parallel variant:

* :class:`ExpertParallelSparseMoeBlock` — each rank owns a contiguous
  block of whole experts instead of a slice of every expert.

Expert-parallel variant:

* :class:`ExpertParallelSparseMoeBlock` — each rank owns a contiguous
  block of whole experts instead of a slice of every expert.

When triton is installed and inputs are on CUDA, dispatch goes through
the Triton fused-MoE kernel in :mod:`mstar.utils.fused_moe`; otherwise it
falls back to the naive per-expert loop in :func:`dispatch_experts_fused`.

Router contract
---------------

Each block computes its expert assignment with a router. The block keeps
the router in ``self.gate``. The default router is :class:`TopKRouter`;
:class:`NoAuxTcRouter` is DeepSeek-V3's sigmoid ``noaux_tc`` one (GLM-5.x).

A router is an ``nn.Module``. A router does not extend a base class. The
``forward`` method of the router must obey this contract::

    router(hidden_states, router_states=None)
        -> (routing_weights, selected_experts, router_states_next)

The router reads these inputs:

* ``hidden_states`` is the input tensor. The shape is ``(tokens, hidden_size)``.
* ``router_states`` is the state from the previous call. A stateless router
  ignores this input.

The router returns these outputs:

* ``routing_weights`` is the tensor of expert weights. The shape is
  ``(tokens, top_k)``.
* ``selected_experts`` is the tensor of expert indices. The shape is
  ``(tokens, top_k)``. The type is int64.
* ``router_states_next`` is the state for the next call. A stateless router
  returns ``None``.

A stateless router ignores ``router_states``. A stateless router returns
``None`` for ``router_states_next``.

A stateful router reads ``router_states``. A stateful router returns a new
state. To thread the state through a block, set the ``return_router_states``
flag on the block.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.quant_linear import QuantCapability, QuantGeometry
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight, quantize_fp8_block_weight

logger = logging.getLogger(__name__)

# bf16 x bf16 -> fp32 mm (aten::mm.dtype); older torch lacks the overload
_MM_OUT_DTYPE = hasattr(torch.ops.aten.mm, "dtype")


# Optional fused Triton MoE path. Imports succeed wherever triton is
# installed; the alignment op JIT-builds a vendored CUDA kernel on first use
# (or falls back to a torch impl), so no extra kernel package is needed. The
# actual kernels only run on CUDA -- guarded by the ``is_cuda`` check below.
try:
    from mstar.utils.fused_moe import MoeAlignment, moe_block_m
    from mstar.utils.fused_moe import fused_experts as _fused_experts
    from mstar.utils.fused_moe.route import ROUTE_ALIGN_MAX_SLOTS, route_align
    from mstar.utils.fused_moe.runner import runs_per_pair

    _HAS_FUSED = True
except Exception as e:  # pragma: no cover -- exercised only when triton missing
    _fused_experts = None
    _HAS_FUSED = False
    logger.warning(f"Could not load fused MoE kernel: {e}")


class TopKRouter(nn.Module):
    """Softmax top-k router. This is the default router for all MoE blocks.

    See the module docstring for the router contract. This router is
    stateless. It ignores ``router_states``. It returns ``None`` for
    ``router_states_next``.

    Args:
        hidden_size: input hidden dimension.
        num_experts: total number of routed experts.
        num_experts_per_tok: number of experts each token is dispatched
            to (top-k).
        norm_topk_prob: if True, renormalize the top-k probabilities so
            they sum to 1.
        fused_softmax_topk: on CUDA, route through Triton kernels instead
            of softmax, ``torch.topk`` and the renormalize (see :meth:`route`).
            Equal probabilities go to the lower expert id, where ``torch.topk``
            may pick another; bf16 logits tie often, so a model opts in once it
            has checked how often its outputs diverge.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
        fused_softmax_topk: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.fused_softmax_topk = fused_softmax_topk
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Args:
            hidden_states: ``(tokens, hidden_size)`` router input.
            router_states: unused (stateless router); accepted for
                interface compatibility with stateful routers.

        Returns:
            routing_weights: ``(tokens, top_k)`` top-k probabilities
                (optionally renormalized).
            selected_experts: ``(tokens, top_k)`` int64 indices.
            router_states_next: always ``None`` (stateless).
        """
        routing_weights, router_indices, _ = self.route(hidden_states)
        return routing_weights, router_indices, None

    def route(
        self, hidden_states: torch.Tensor, align_w1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, "MoeAlignment | None"]:
        """The weights and experts, plus the block alignment when one kernel can do both.

        With ``fused_softmax_topk`` on CUDA, a batch of up to ``ROUTE_ALIGN_MAX_SLOTS``
        renormalized slots whose block can consume an alignment (it passes its
        ``gate_up_proj`` as ``align_w1``, and its flat batch) takes ``route_align``,
        unless ``fused_experts`` runs it as per-pair GEMVs, which read no alignment.
        Any other batch takes ``router_topk``, and with the flag off it's softmax,
        ``torch.topk`` and the renormalize.
        """
        fused = self.fused_softmax_topk and hidden_states.is_cuda
        tokens = hidden_states.shape[0]
        if (
            fused and _HAS_FUSED and align_w1 is not None and self.norm_topk_prob
            and tokens * self.top_k <= ROUTE_ALIGN_MAX_SLOTS
            and not runs_per_pair(hidden_states, align_w1)
        ):
            router_logits = F.linear(hidden_states, self.weight)
            # int: the kernel's block sizes are constants, and a compile that keeps the
            # token count symbolic specializes on it here, for at most these few counts
            tokens = int(tokens)
            return route_align(
                router_logits, self.top_k, moe_block_m(tokens, align_w1, self.top_k),
                router_logits.dtype, softmax=True,
            )
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight)
        if fused:
            from mstar.utils.fused_moe.decode import router_topk

            routing_weights, router_indices = router_topk(
                router_logits, None, self.top_k, 1.0, self.norm_topk_prob, softmax=True,
            )
            return routing_weights, router_indices, None
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)

        routing_weights = router_top_value.to(router_logits.dtype)
        return routing_weights, router_indices, None


class NoAuxTcRouter(nn.Module):
    """DeepSeek-V3's ``noaux_tc`` router, groupless (``n_group=1``), as GLM-5.x routes.

    Sigmoid scores in fp32; each token's top-k by score + ``e_score_correction_bias``; weights
    are the unbiased scores of those experts, renormalized if ``norm_topk_prob``, times
    ``routed_scaling_factor``. Stateless (see the router contract above).
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
        # fused_moe's router ops, from ``finalize_weights(topk_kernel=True)``: the decode
        # router (logits and top-k) for up to ``_route_rows`` tokens, its top-k program on
        # GEMM logits past them; both (..., bias, top_k, scale, renormalize) -> (weights, ids)
        self._route_op = None
        self._route_rows = 0
        self._topk_op = None
        # fp32 copy of ``weight``, built once by ``finalize_weights``: the
        # forward needs fp32 and the router never changes, so casting it once
        # per layer per step is pure waste. A plain attribute, not a buffer,
        # so model.to(bf16) cannot downcast it.
        self._weight_fp32: torch.Tensor | None = None

    def finalize_weights(self, topk_kernel: bool = False) -> None:
        """Cache the fp32 router weight; call after the weights are loaded. ``topk_kernel``
        routes CUDA tokens through fused_moe's router kernels, whose ids can differ from
        torch's at exact ties. Up to its MAX_ROUTER_TOKENS rows that is the decode router
        itself, the one a fused decode chain runs, so a block's gate and its decode kernels
        give the same weights bit for bit (a GEMM sums the logits in another order)."""
        self._weight_fp32 = self.weight.detach().float()
        self._route_op = self._topk_op = None
        if topk_kernel:
            from mstar.utils.fused_moe import decode  # registers the ops

            self._route_op, self._topk_op = decode.route_op, decode.router_topk
            self._route_rows = decode.MAX_ROUTER_TOKENS

    def forward(
        self, hidden_states: torch.Tensor, router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        flat = hidden_states.reshape(-1, self.hidden_size)
        if self._route_op is not None and flat.is_cuda and flat.shape[0] <= self._route_rows:
            topk_weights, topk_ids = self._route_op(
                flat.contiguous(), self.weight, self.e_score_correction_bias, self.top_k,
                self.routed_scaling_factor, self.norm_topk_prob)
            return topk_weights, topk_ids, None
        # the engine's bf16 autocast would run F.linear in bf16 and round the logits
        with torch.autocast(flat.device.type, enabled=False):
            if _MM_OUT_DTYPE and flat.is_cuda and flat.dtype == self.weight.dtype == torch.bfloat16:
                # bf16 x bf16 products are exact in fp32: the fp32 router in another
                # summation order, without an fp32 copy of the tokens
                logits = torch.mm(flat, self.weight.t(), out_dtype=torch.float32)
            else:
                w = self._weight_fp32
                if w is None or w.device != self.weight.device:
                    w = self.weight.float()
                logits = F.linear(flat.float(), w)
        if self._topk_op is not None and logits.is_cuda:
            topk_weights, topk_ids = self._topk_op(
                logits, self.e_score_correction_bias, self.top_k, self.routed_scaling_factor,
                self.norm_topk_prob)
            return topk_weights, topk_ids, None
        scores = logits.sigmoid()  # (T, E)

        biased = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_ids = torch.topk(biased, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)

        if self.norm_topk_prob:
            # HF's 1e-20, as the fused kernel adds: it guards an all-zero top-k and rounds
            # away for any sum above 2^-42
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_ids, None


def dispatch_experts_fused(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    skip_invalid: bool = False,
) -> torch.Tensor:
    """Naive per-expert dispatch using the fused HF checkpoint layout.

    Used as a fallback when the Triton fused-MoE kernel isn't available.
    Loops over the experts that received any tokens and runs SwiGLU per
    expert.

    Args:
        hidden_states: ``(tokens, hidden_size)``.
        gate_up_proj: ``(num_experts, 2 * moe_intermediate_size, hidden_size)``.
        down_proj: ``(num_experts, hidden_size, moe_intermediate_size)``.
        selected_experts: ``(tokens, top_k)`` int64.
        routing_weights: ``(tokens, top_k)`` float.
        skip_invalid: set when ``selected_experts`` may hold the EP
            sentinel ``num_experts``; those slots contribute nothing.
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        # one-hot mask over expert dim: (num_experts, top_k, tokens)
        num_classes = num_experts + 1 if skip_invalid else num_experts
        expert_mask = F.one_hot(selected_experts, num_classes=num_classes)
        # Drop the sentinel column so skipped slots never reach an expert.
        expert_mask = expert_mask[..., :num_experts].permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        gate_up = torch.mm(current_state, gate_up_proj[expert_idx].T)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden_states = torch.mm(F.silu(gate) * up, down_proj[expert_idx].T)
        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype),
        )

    return final_hidden_states


def _mask_padding(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    token_valid: torch.Tensor,
    sentinel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route padding rows to ``sentinel`` so no GEMM runs them.

    ``token_valid`` is ``(tokens,)`` bool; a CUDA-graph replay pads the batch
    with rows past the real tokens. Dispatch with ``skip_invalid=True``.
    """
    keep = token_valid[:, None]
    return (
        torch.where(keep, selected_experts, sentinel),
        torch.where(keep, routing_weights, 0),
    )


def _dispatch(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    skip_invalid: bool = False,
    alignment: "MoeAlignment | None" = None,
) -> torch.Tensor:
    """Pick fused-Triton if available, otherwise the naive loop."""
    if _HAS_FUSED and hidden_states.is_cuda:
        return _fused_experts(
            hidden_states, gate_up_proj, down_proj,
            routing_weights, selected_experts, skip_invalid=skip_invalid, alignment=alignment,
        )
    return dispatch_experts_fused(
        hidden_states, gate_up_proj, down_proj,
        num_experts, selected_experts, routing_weights, skip_invalid=skip_invalid,
    )


def _route(
    gate: nn.Module, flat: torch.Tensor, align_w1: torch.Tensor | None, router_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, "MoeAlignment | None"]:
    """``gate``'s output plus the alignment ``fused_experts`` would otherwise
    compute, which a :class:`TopKRouter` may give a block that passes ``align_w1``."""
    if type(gate) is TopKRouter:
        weights, experts, alignment = gate.route(flat, align_w1)
        return weights, experts, None, alignment
    return (*gate(flat, router_states), None)


def _sync_routing(
    comm_group: CommGroup,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make every rank use group rank 0's top-k. Expert parallelism only.

    Ranks compile the router separately, so a near-tie can resolve
    differently per rank; EP dispatch would then give a token more or fewer
    than k experts. Under tensor parallelism each rank runs all k slots on its
    slice of every expert, so a disagreement only mixes two experts' slices in
    one slot, and the all-reduce after the experts has the same shape either
    way: not worth an all-reduce per MoE layer. Both tensors travel in one fp32
    tensor, which holds expert ids below 2**24 and bf16/fp16/fp32 weights
    exactly. Other ranks send zeros, so the all-reduce is an exact broadcast; a
    plain ``dist.broadcast`` is not dynamo-traceable and would break the graph.
    """
    top_k = selected_experts.shape[-1]
    packed = torch.cat([selected_experts.float(), routing_weights.float()], dim=-1)
    if comm_group.rank != 0:
        packed = torch.zeros_like(packed)
    comm_group.all_reduce(packed)
    return packed[:, top_k:].to(routing_weights.dtype), packed[:, :top_k].long()


class SparseMoeBlock(nn.Module):
    """Top-K sparse MoE with fused expert weights, no shared expert.

    Expert weights match the HF fused checkpoint layout:
      - ``experts.gate_up_proj``: ``(num_experts, 2 * moe_intermediate_size, hidden_size)``
      - ``experts.down_proj``: ``(num_experts, hidden_size, moe_intermediate_size)``
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
        router: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = router if router is not None else TopKRouter(
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
        *,
        return_router_states: bool = False,
    ):
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        routing_weights, selected_experts, router_states_next = self.gate(flat, router_states)
        out = _dispatch(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, selected_experts, routing_weights,
        )
        out = out.view(input_shape)
        return (out, router_states_next) if return_router_states else out


class SparseMoeBlockWithSharedExpert(nn.Module):
    """Top-K sparse MoE with a shared expert + sigmoid gating.

    Final output is::

        out = routed(x) + sigmoid(shared_gate(x)) * shared_expert(x)

    The shared expert is supplied by the caller (any ``nn.Module``
    matching the ``hidden_size → hidden_size`` interface). The routed
    path uses the same fused checkpoint layout as :class:`SparseMoeBlock`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert: nn.Module,
        norm_topk_prob: bool = False,
        router: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = router if router is not None else TopKRouter(
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
        self.shared_expert = shared_expert
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
        *,
        return_router_states: bool = False,
    ):
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()

        shared = self.shared_expert(flat)

        routing_weights, selected_experts, router_states_next = self.gate(flat, router_states)
        routed = _dispatch(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, selected_experts, routing_weights,
        )
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        out = (routed + shared_gate * shared).view(input_shape)
        return (out, router_states_next) if return_router_states else out


# ---------------------------------------------------------------------------
# TP-aware MoE blocks
# ---------------------------------------------------------------------------


def _gate_up_weight_loader(
    tp_rank: int, tp_size: int, full_inter: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's gate_proj or up_proj into the fused gate_up_proj param.

    ``loaded_shard_id`` is ``"gate:N"`` or ``"up:N"`` where N is the
    expert index.  ``loaded_weight`` shape is ``(inter, hidden)`` — a
    single expert's projection.  The TP rank's slice is taken and copied
    into the correct position in ``param`` which has shape
    ``(E, 2*shard_inter, hidden)``.
    """
    assert loaded_shard_id is not None
    kind, expert_str = loaded_shard_id.split(":")
    expert_idx = int(expert_str)
    shard_inter = divide(full_inter, tp_size)
    start = tp_rank * shard_inter
    # the full tensor, or this rank's slice from a sliced read
    tp_slice = loaded_weight if loaded_weight.shape[0] == shard_inter else loaded_weight[start:start + shard_inter, :]
    if kind == "gate":
        param.data[expert_idx, :shard_inter, :] = tp_slice
    else:
        param.data[expert_idx, shard_inter:, :] = tp_slice


def _down_proj_weight_loader(
    tp_rank: int, tp_size: int, full_inter: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's down_proj into the fused down_proj param.

    ``loaded_shard_id`` is ``"down:N"``.  ``loaded_weight`` shape is
    ``(hidden, inter)``.  The TP rank's column slice is taken.
    """
    assert loaded_shard_id is not None
    expert_idx = int(loaded_shard_id.split(":")[1])
    shard_inter = divide(full_inter, tp_size)
    start = tp_rank * shard_inter
    if loaded_weight.shape[1] != shard_inter:
        loaded_weight = loaded_weight[:, start:start + shard_inter]
    param.data[expert_idx, :, :] = loaded_weight


def _gate_up_fp8_weight_loader(
    tp_rank: int, tp_size: int, full_inter: int, row_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """:func:`_gate_up_weight_loader` for an fp8 block-scale checkpoint's experts (a stacked
    ``(E, 2 * rows, cols)`` uint8 param): one expert's gate or up weight, ``row_unit`` = 1, or its
    ``weight_scale_inv``, ``row_unit`` = the scale block's rows. The full tensor or this rank's slice."""
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


def _down_proj_fp8_weight_loader(
    tp_rank: int, tp_size: int, full_inter: int, col_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """Column (contraction-dim) sharding twin of :func:`_gate_up_fp8_weight_loader`."""
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


def _f32_container(loader):
    """``loader`` for an fp32 tensor kept in an int32 container: the checkpoint's scales viewed, not cast."""
    def load(param, loaded_weight, loaded_shard_id=None):
        loader(param, loaded_weight.to(torch.float32).view(torch.int32), loaded_shard_id)
    return load


class Fp8Experts:
    """A MoE block's routed experts (``experts.gate_up_proj (E, 2I, H)``, ``experts.down_proj (E, H, I)``, this
    rank's share) as one quantizable layer, ``<block>.experts`` (``mstar.quantization``), held as block fp8: e4m3
    in uint8 containers and their fp32 ``*_scale_inv`` blocks in int32 ones (the engine's node cast would round
    them). Loaded as a block-fp8 checkpoint stores them (``experts_prepare``) or quantized from the loaded bf16
    an expert at a time (``experts_quantize``); dispatched through ``fused_experts_fp8`` (W8A8, activations per
    token per 128), whose sentinel expert id skips a slot as the bf16 dispatch's does.

    The block supplies ``_reattach_loaders()`` (its ``_attach_weight_loaders`` with its own arguments), which
    calls ``_attach_fp8_loaders`` while ``experts_fp8_block`` is set."""

    # (block_n, block_k) once the experts hold block-scaled e4m3
    experts_fp8_block: tuple[int, int] | None = None

    def _reattach_loaders(self) -> None:
        raise NotImplementedError

    def experts_quant_capability(self) -> QuantCapability:
        if self.experts_fp8_block is not None:
            return QuantCapability(False, "already fp8_block")
        return QuantCapability(True)

    def experts_geometry(self) -> QuantGeometry:
        """gate_up's ``[2I, H]`` per rank, its gate and up halves as row spans (down's K is ``I``)."""
        _, two_inter, hidden = self.experts.gate_up_proj.shape
        return QuantGeometry(two_inter, hidden, (two_inter // 2, two_inter // 2), (hidden,))

    def _attach_fp8_loaders(self, gate_up, down) -> None:
        """``gate_up(row_unit)`` / ``down(col_unit)``: the block's loaders for a stored tensor whose one index
        covers that many weight rows / columns (1 for the e4m3 data, the scale block for its scales)."""
        bn, bk = self.experts_fp8_block
        experts = self.experts
        experts.gate_up_proj.weight_loader = gate_up(1)
        experts.down_proj.weight_loader = down(1)
        experts.gate_up_proj_scale_inv.weight_loader = _f32_container(gate_up(bn))
        experts.down_proj_scale_inv.weight_loader = _f32_container(down(bk))

    def _fp8_storage(self, block: tuple[int, int], data: dict[str, torch.Tensor] | None = None) -> None:
        """Replace the bf16 experts with block-fp8 storage, empty or ``data``."""
        bn, bk = block
        experts = self.experts
        e, two_inter, hidden = experts.gate_up_proj.shape
        inter = two_inter // 2
        device = experts.gate_up_proj.device
        shapes = {
            "gate_up_proj": ((e, two_inter, hidden), torch.uint8),
            "gate_up_proj_scale_inv": ((e, -(-two_inter // bn), -(-hidden // bk)), torch.int32),
            "down_proj": ((e, hidden, inter), torch.uint8),
            "down_proj_scale_inv": ((e, -(-hidden // bn), -(-inter // bk)), torch.int32),
        }
        for name, (shape, dtype) in shapes.items():
            tensor = torch.empty(shape, dtype=dtype, device=device) if data is None else data[name]
            assert tuple(tensor.shape) == shape and tensor.dtype == dtype, (name, tensor.shape, tensor.dtype)
            if hasattr(experts, name):
                delattr(experts, name)
            setattr(experts, name, nn.Parameter(tensor, requires_grad=False))
        self.experts_fp8_block = tuple(block)
        self._reattach_loaders()

    def experts_prepare(self, fmt: str, block: tuple[int, int] = (128, 128)) -> None:
        """Hold the experts as a block-fp8 checkpoint stores them, before it loads."""
        assert fmt == "fp8_block", fmt
        self._fp8_storage(block)

    @torch.no_grad()
    def experts_quantize(self, fmt: str, block: tuple[int, int] = (128, 128)) -> None:
        """Quantize this rank's loaded bf16 experts to block fp8, one expert at a time (a whole stack's fp32
        working copy would be several GB)."""
        assert fmt == "fp8_block", fmt
        data = {}
        for name in ("gate_up_proj", "down_proj"):
            weight = getattr(self.experts, name).data
            parts = [quantize_fp8_block_weight(weight[i], block) for i in range(weight.shape[0])]
            data[name] = torch.stack([q for q, _ in parts]).view(torch.uint8)
            data[f"{name}_scale_inv"] = torch.stack([s for _, s in parts]).contiguous().view(torch.int32)
        self._fp8_storage(block, data)

    def _experts_fp8(
        self, flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor,
        skip_invalid: bool = False,
    ) -> torch.Tensor:
        """The routed experts' weighted sum over this rank's experts, ``(tokens, hidden)``; ids equal to the
        local expert count are skipped with ``skip_invalid``. Off CUDA, the weights dequantized as a bf16
        checkpoint would hold them (tests)."""
        experts = self.experts
        w1, w2 = experts.gate_up_proj, experts.down_proj
        s1 = experts.gate_up_proj_scale_inv.view(torch.float32)
        s2 = experts.down_proj_scale_inv.view(torch.float32)
        if not flat.is_cuda:
            bf16 = [
                torch.stack([dequantize_fp8_block_weight(w[i], s[i], self.experts_fp8_block, flat.dtype)
                             for i in range(w.shape[0])])
                for w, s in ((w1, s1), (w2, s2))
            ]
            return dispatch_experts_fused(flat, *bf16, w1.shape[0], selected_experts, routing_weights,
                                          skip_invalid=skip_invalid)
        from mstar.utils.fused_moe import fused_experts_fp8

        return fused_experts_fp8(
            flat, w1, w2, s1, s2, routing_weights, selected_experts,
            block_size=self.experts_fp8_block, skip_invalid=skip_invalid,
        )


class ParallelSparseMoeBlock(Fp8Experts, nn.Module):
    """TP-aware Top-K sparse MoE.

    When ``tp_size == 1``, the forward is identical to
    :class:`SparseMoeBlock` (full fused kernel, no communication).
    When ``tp_size > 1``, expert weights are sharded along the
    intermediate dimension and an all-reduce is inserted between the
    down-projection GEMM and the top-k sum-reduce.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
        router: nn.Module | None = None,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        tp_size = comm_group.world_size
        tp_rank = comm_group.rank

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = router if router is not None else TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )

        shard_inter = divide(moe_intermediate_size, tp_size)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * shard_inter, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, shard_inter)
        )
        self._attach_weight_loaders(tp_rank, tp_size, moe_intermediate_size)

    def _reattach_loaders(self) -> None:
        self._attach_weight_loaders(self.comm_group.rank, self.comm_group.world_size, self.moe_intermediate_size)

    def _attach_weight_loaders(self, tp_rank: int, tp_size: int, full_inter: int):
        from functools import partial

        if self.experts_fp8_block is not None:
            self._attach_fp8_loaders(
                lambda unit: partial(_gate_up_fp8_weight_loader, tp_rank, tp_size, full_inter, unit),
                lambda unit: partial(_down_proj_fp8_weight_loader, tp_rank, tp_size, full_inter, unit),
            )
            return
        self.experts.gate_up_proj.weight_loader = partial(
            _gate_up_weight_loader, tp_rank, tp_size, full_inter,
        )
        self.experts.down_proj.weight_loader = partial(
            _down_proj_weight_loader, tp_rank, tp_size, full_inter,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders(
            self.comm_group.rank, self.comm_group.world_size,
            self.moe_intermediate_size,
        )
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
        *,
        return_router_states: bool = False,
        token_valid: torch.Tensor | None = None,
    ):
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        skip = token_valid is not None
        if self.experts_fp8_block is not None:
            routing_weights, selected_experts, router_states_next, _ = _route(self.gate, flat, None, router_states)
            if skip:
                selected_experts, routing_weights = _mask_padding(
                    selected_experts, routing_weights, token_valid, self.num_experts,
                )
            out = self._experts_fp8(flat, routing_weights, selected_experts, skip_invalid=skip)
            if self.comm_group.world_size > 1:
                self.comm_group.all_reduce(out)
            out = out.view(input_shape)
            return (out, router_states_next) if return_router_states else out
        # the mask below moves experts after routing, and TP's dispatch aligns for itself
        can_align = not skip and self.comm_group.world_size == 1
        routing_weights, selected_experts, router_states_next, alignment = _route(
            self.gate, flat, self.experts.gate_up_proj if can_align else None, router_states,
        )
        if skip:
            selected_experts, routing_weights = _mask_padding(
                selected_experts, routing_weights, token_valid, self.num_experts,
            )

        if self.comm_group.world_size == 1:
            out = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_experts, selected_experts, routing_weights, skip_invalid=skip, alignment=alignment,
            )
        else:
            out = self._dispatch_tp(flat, routing_weights, selected_experts, skip_invalid=skip)
        out = out.view(input_shape)
        return (out, router_states_next) if return_router_states else out

    def _dispatch_tp(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        skip_invalid: bool = False,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        # (tokens, top_k, hidden) — partial results before reduce
        cache3 = fused_experts(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            routing_weights, selected_experts, reduce_results=False,
            skip_invalid=skip_invalid,
        )
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
        self.comm_group.all_reduce(output)
        return output


class ParallelSparseMoeBlockWithSharedExpert(Fp8Experts, nn.Module):
    """TP-aware Top-K sparse MoE with a shared expert + sigmoid gating.

    The shared expert should be a ``ParallelGatedMLP`` constructed with
    the same ``comm_group`` so its all-reduce is handled internally.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert: nn.Module,
        norm_topk_prob: bool = False,
        router: nn.Module | None = None,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        tp_size = comm_group.world_size
        tp_rank = comm_group.rank

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = router if router is not None else TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )

        shard_inter = divide(moe_intermediate_size, tp_size)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * shard_inter, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, shard_inter)
        )
        self.shared_expert = shared_expert
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self._attach_weight_loaders(tp_rank, tp_size, moe_intermediate_size)

    def _reattach_loaders(self) -> None:
        self._attach_weight_loaders(self.comm_group.rank, self.comm_group.world_size, self.moe_intermediate_size)

    def _attach_weight_loaders(self, tp_rank: int, tp_size: int, full_inter: int):
        from functools import partial

        if self.experts_fp8_block is not None:
            self._attach_fp8_loaders(
                lambda unit: partial(_gate_up_fp8_weight_loader, tp_rank, tp_size, full_inter, unit),
                lambda unit: partial(_down_proj_fp8_weight_loader, tp_rank, tp_size, full_inter, unit),
            )
            return
        self.experts.gate_up_proj.weight_loader = partial(
            _gate_up_weight_loader, tp_rank, tp_size, full_inter,
        )
        self.experts.down_proj.weight_loader = partial(
            _down_proj_weight_loader, tp_rank, tp_size, full_inter,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders(
            self.comm_group.rank, self.comm_group.world_size,
            self.moe_intermediate_size,
        )
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
        *,
        return_router_states: bool = False,
    ):
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()

        shared = self.shared_expert(flat)

        if self.experts_fp8_block is not None:
            routing_weights, selected_experts, router_states_next, _ = _route(self.gate, flat, None, router_states)
            routed = self._experts_fp8(flat, routing_weights, selected_experts)
            if self.comm_group.world_size > 1:
                self.comm_group.all_reduce(routed)
            out = (routed + torch.sigmoid(self.shared_expert_gate(flat)) * shared).view(input_shape)
            return (out, router_states_next) if return_router_states else out
        routing_weights, selected_experts, router_states_next, alignment = _route(
            self.gate, flat, self.experts.gate_up_proj if self.comm_group.world_size == 1 else None, router_states,
        )
        if self.comm_group.world_size == 1:
            routed = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_experts, selected_experts, routing_weights, alignment=alignment,
            )
        else:
            routed = self._dispatch_tp(flat, routing_weights, selected_experts)
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        out = (routed + shared_gate * shared).view(input_shape)
        return (out, router_states_next) if return_router_states else out

    def _dispatch_tp(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        cache3 = fused_experts(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            routing_weights, selected_experts, reduce_results=False,
        )
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
        self.comm_group.all_reduce(output)
        return output


# ---------------------------------------------------------------------------
# Expert-parallel MoE block
# ---------------------------------------------------------------------------


def _ep_gate_up_weight_loader(
    expert_start: int, num_local_experts: int, full_inter: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's gate_proj or up_proj whole, if this rank owns it.

    ``loaded_shard_id`` is ``"gate:N"`` or ``"up:N"`` with ``N`` the global
    expert index. Experts outside ``[expert_start, expert_start +
    num_local_experts)`` belong to another rank and are skipped.
    ``param`` has shape ``(E_local, 2*inter, hidden)``.
    """
    assert loaded_shard_id is not None
    kind, expert_str = loaded_shard_id.split(":")
    local_idx = int(expert_str) - expert_start
    if not 0 <= local_idx < num_local_experts:
        return
    assert loaded_weight.shape == (full_inter, param.shape[2]), (
        f"expert {loaded_shard_id} shape {tuple(loaded_weight.shape)} "
        f"!= {(full_inter, param.shape[2])}"
    )
    if kind == "gate":
        param.data[local_idx, :full_inter, :] = loaded_weight
    else:
        param.data[local_idx, full_inter:, :] = loaded_weight


def _ep_down_proj_weight_loader(
    expert_start: int, num_local_experts: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's down_proj whole, if this rank owns it.

    ``loaded_shard_id`` is ``"down:N"``. ``loaded_weight`` has shape
    ``(hidden, inter)``, the same as one expert's slot in ``param``.
    """
    assert loaded_shard_id is not None
    local_idx = int(loaded_shard_id.split(":")[1]) - expert_start
    if not 0 <= local_idx < num_local_experts:
        return
    assert loaded_weight.shape == param.shape[1:], (
        f"expert {loaded_shard_id} shape {tuple(loaded_weight.shape)} "
        f"!= {tuple(param.shape[1:])}"
    )
    param.data[local_idx] = loaded_weight


def _ep_gate_up_fp8_weight_loader(
    expert_start: int, num_local_experts: int, full_inter: int, row_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """:func:`_ep_gate_up_weight_loader` for block-fp8 storage: one expert's gate or up weight (``row_unit``
    1) or its ``weight_scale_inv`` (``row_unit`` the scale block's rows), whole, if this rank owns it."""
    assert loaded_shard_id is not None
    kind, expert_str = str(loaded_shard_id).split(":")
    local_idx = int(expert_str) - expert_start
    if not 0 <= local_idx < num_local_experts:
        return
    rows = -(-full_inter // row_unit)
    if loaded_weight.dtype == FP8_DTYPE:
        loaded_weight = loaded_weight.view(torch.uint8)
    assert loaded_weight.shape == (rows, param.shape[2]), (
        f"expert {loaded_shard_id} tensor {tuple(loaded_weight.shape)} != {(rows, param.shape[2])}"
    )
    if kind == "gate":
        param.data[local_idx, :rows, :] = loaded_weight
    else:
        param.data[local_idx, rows:2 * rows, :] = loaded_weight


def _ep_down_proj_fp8_weight_loader(
    expert_start: int, num_local_experts: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """:func:`_ep_down_proj_weight_loader` for block-fp8 storage (data or scales, whole)."""
    if loaded_weight.dtype == FP8_DTYPE:
        loaded_weight = loaded_weight.view(torch.uint8)
    _ep_down_proj_weight_loader(expert_start, num_local_experts, param, loaded_weight, loaded_shard_id)


class ExpertParallelSparseMoeBlock(Fp8Experts, nn.Module):
    """Expert-parallel Top-K sparse MoE.

    Rank ``r`` of ``P`` owns global experts ``[r*E_local, (r+1)*E_local)``
    at full intermediate width, with ``E_local = num_experts / P``. The
    input and the router are replicated, and :func:`_sync_routing` makes
    every rank use rank 0's top-k. Each rank remaps slots routed to other
    ranks' experts to the sentinel id ``E_local``, which the dispatch skips,
    and one all-reduce sums the disjoint partial outputs.

    The signature matches :class:`ParallelSparseMoeBlock`. When
    ``world_size == 1`` the forward is identical to :class:`SparseMoeBlock`.

    ``track_expert_load`` (off by default, CUDA-graph safe) accumulates
    per-expert slot counts in the ``expert_load`` buffer; read and reset it
    with :meth:`pop_expert_load`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
        router: nn.Module | None = None,
        comm_group: CommGroup | None = None,
        track_expert_load: bool = False,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.num_local_experts = divide(num_experts, comm_group.world_size)
        self.expert_start = comm_group.rank * self.num_local_experts
        self.track_expert_load = track_expert_load
        if track_expert_load:
            # Routing is replicated, so every rank counts every expert.
            self.register_buffer(
                "expert_load", torch.zeros(num_experts, dtype=torch.int64), persistent=False,
            )

        self.gate = router if router is not None else TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )

        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, moe_intermediate_size)
        )
        self._attach_weight_loaders()

    def _reattach_loaders(self) -> None:
        self._attach_weight_loaders()

    def _attach_weight_loaders(self):
        from functools import partial

        if self.experts_fp8_block is not None:
            self._attach_fp8_loaders(
                lambda unit: partial(_ep_gate_up_fp8_weight_loader, self.expert_start, self.num_local_experts,
                                     self.moe_intermediate_size, unit),
                lambda unit: partial(_ep_down_proj_fp8_weight_loader, self.expert_start, self.num_local_experts),
            )
            return
        self.experts.gate_up_proj.weight_loader = partial(
            _ep_gate_up_weight_loader,
            self.expert_start, self.num_local_experts, self.moe_intermediate_size,
        )
        self.experts.down_proj.weight_loader = partial(
            _ep_down_proj_weight_loader, self.expert_start, self.num_local_experts,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        if self.track_expert_load:
            # to_empty leaves the buffer uninitialised.
            self.expert_load.zero_()
        return result

    def pop_expert_load(self) -> torch.Tensor:
        """Return per-expert slot counts since the last pop, and reset them.

        Shape ``(num_experts,)``; rank ``r`` owns ``[r*E_local, (r+1)*E_local)``.
        Syncs the device, so call it from eager code, never inside a forward.
        """
        counts = self.expert_load.to("cpu", copy=True)
        self.expert_load.zero_()
        return counts

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
        *,
        return_router_states: bool = False,
        token_valid: torch.Tensor | None = None,
    ):
        """``token_valid``: ``(tokens,)`` bool; rows marked False are padding and skip every GEMM."""
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        routing_weights, selected_experts, router_states_next = self.gate(flat, router_states)

        if self.track_expert_load:
            ids = selected_experts.flatten()
            if token_valid is None:
                ones = torch.ones_like(ids, dtype=torch.int64)
            else:
                ones = token_valid[:, None].expand_as(selected_experts).flatten().to(torch.int64)
            self.expert_load.index_add_(0, ids, ones)

        if self.comm_group.world_size == 1:
            skip = token_valid is not None
            if skip:
                selected_experts, routing_weights = _mask_padding(
                    selected_experts, routing_weights, token_valid, self.num_experts,
                )
            if self.experts_fp8_block is not None:
                out = self._experts_fp8(flat, routing_weights, selected_experts, skip_invalid=skip)
                out = out.view(input_shape)
                return (out, router_states_next) if return_router_states else out
            out = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_experts, selected_experts, routing_weights, skip_invalid=skip,
            )
        else:
            routing_weights, selected_experts = _sync_routing(
                self.comm_group, routing_weights, selected_experts,
            )
            out = self._dispatch_ep(flat, routing_weights, selected_experts, token_valid)
        out = out.view(input_shape)
        return (out, router_states_next) if return_router_states else out

    def _dispatch_ep(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        token_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Every shape is static, so this stays torch.compile and CUDA-graph safe.
        local = selected_experts - self.expert_start
        owned = (local >= 0) & (local < self.num_local_experts)
        if token_valid is not None:
            owned = owned & token_valid[:, None]
        local = torch.where(owned, local, self.num_local_experts)
        # Skipped slots never reach a GEMM; zeroing their weight is belt-and-braces.
        routing_weights = torch.where(owned, routing_weights, 0)
        if self.experts_fp8_block is not None:
            out = self._experts_fp8(flat, routing_weights, local, skip_invalid=True)
        else:
            out = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_local_experts, local, routing_weights, skip_invalid=True,
            )
        self.comm_group.all_reduce(out)
        return out
