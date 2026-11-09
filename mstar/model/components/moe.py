"""Mixture-of-Experts blocks.

Top-K router + dispatch path for the standard fused-expert checkpoint
layout (``experts.gate_up_proj`` and ``experts.down_proj`` packed as
``(num_experts, ...)`` parameters):

* :class:`FusedExperts` -- the routed experts and their dispatch, sharded
  over a tensor-parallel group by intermediate slice or by whole experts.
* :class:`ParallelSparseMoeBlock` -- router, experts and the all-reduce,
  plus an optional shared expert with sigmoid gating (the Qwen3-Omni
  Talker's) and, under expert parallelism, optional expert-load counts.

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
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.quant_linear import QuantCapability, QuantGeometry
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight, quantize_fp8_block_weight
from mstar.utils.situ import activation_spec

logger = logging.getLogger(__name__)
# (block class, backend) pairs whose routed-expert backend line was logged
_EXPERTS_BACKEND_LOGGED: set[tuple[str, str]] = set()

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
            and not runs_per_pair(hidden_states, align_w1, self.top_k)
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
        return self.route(logits)

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """``forward`` from the fp32 router logits ``(T, E)``, for a caller that computed them
        itself (e.g. in one GEMM with other projections of the same input)."""
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


def _gated(gate_up: torch.Tensor, activation: str | Callable) -> torch.Tensor:
    """``act(gate) * up`` of a ``(..., 2 * inter)`` gate|up row: ``silu``, or a callable taking
    both halves (``mstar.utils.situ.SiTU``)."""
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu":
        return F.silu(gate) * up
    return activation(gate, up)


def dispatch_experts_fused(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    skip_invalid: bool = False,
    activation: str | Callable = "silu",
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
        skip_invalid: set when ``selected_experts`` may hold the sentinel
            ``num_experts`` (a padding row's, or another EP rank's expert);
            those slots contribute nothing.
        activation: ``silu`` (SwiGLU) or a callable on the gate and up halves.
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
        current_hidden_states = torch.mm(_gated(gate_up, activation), down_proj[expert_idx].T)
        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype),
        )

    return final_hidden_states


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
    # an all-reduce may return the sum in a new tensor (tp_allreduce: flashinfer), not in its input
    packed = comm_group.all_reduce(packed)
    return packed[:, top_k:].to(routing_weights.dtype), packed[:, :top_k].long()


class FusedExperts(nn.Module):
    """The routed experts of one MoE layer and their dispatch: ``E`` SwiGLU MLPs stacked
    in the fused checkpoint layout, ``gate_up_proj (E_local, 2 * inter_local, hidden)``
    (gate rows, then up) and ``down_proj (E_local, hidden, inter_local)``.

    Over a tensor-parallel group every rank holds a slice of each expert's intermediate
    (``expert_parallel=False``); under expert parallelism it holds whole experts
    ``[expert_start, expert_start + num_local_experts)``. Either way ``forward`` returns
    this rank's partial sum, which the caller all-reduces.

    :meth:`_dispatch` runs the kernel on these parameters. A fused layer that calls the
    kernel on them directly (Command-A+'s ``fused_branches``) does not go through it.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        comm_group: CommGroup | None = None,
        expert_parallel: bool = False,
        activation: str | Callable = "silu",
    ) -> None:
        """``activation``: the experts' gated activation, ``silu`` or a ``SiTU``."""
        super().__init__()
        group = comm_group or CommGroup.trivial()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        if expert_parallel:
            self.num_local_experts = divide(num_experts, group.world_size)
            self.expert_start = group.rank * self.num_local_experts
            local_inter, self.inter_start = intermediate_size, 0
        else:
            self.num_local_experts, self.expert_start = num_experts, 0
            local_inter = divide(intermediate_size, group.world_size)
            self.inter_start = group.rank * local_inter
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_local_experts, 2 * local_inter, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_local_experts, hidden_size, local_inter))
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        # a block that swapped in quantized storage (`QuantExperts`) attaches that storage's loaders
        for name, loader in (("gate_up_proj", self._load_gate_up), ("down_proj", self._load_down)):
            param = getattr(self, name, None)
            if param is not None and param.dtype.is_floating_point:
                param.weight_loader = loader

    def _apply(self, fn, recurse=True):
        # new parameters (a dtype or device move, ``to_empty``) lose their loaders
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def _parse_shard_id(
        self, loaded_shard_id: str | None, projections: tuple[str, ...],
    ) -> tuple[str, int | None]:
        """``"<proj>:N"`` -> ``(proj, local)``: ``local`` is this rank's index of
        expert ``N``, None if another EP rank's."""
        proj, _, expert = (loaded_shard_id or "").partition(":")
        if proj not in projections or not expert.isdigit() or int(expert) >= self.num_experts:
            raise ValueError(
                f"expert shard id {loaded_shard_id!r} is not '<{'|'.join(projections)}>:N' "
                f"with N in [0, {self.num_experts})"
            )
        local = int(expert) - self.expert_start
        return proj, (local if 0 <= local < self.num_local_experts else None)

    @staticmethod
    def _check_shape(loaded_shard_id: str, loaded_weight: torch.Tensor, expected: tuple[int, int]) -> None:
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(f"expert {loaded_shard_id} shape {tuple(loaded_weight.shape)} != {expected}")

    def _load_gate_up(self, param, loaded_weight, loaded_shard_id=None):
        """One expert's ``gate_proj`` or ``up_proj`` (``"gate:N"`` / ``"up:N"``, ``N``
        the global expert), ``(inter, hidden)``: this rank's rows of it."""
        proj, local = self._parse_shard_id(loaded_shard_id, ("gate", "up"))
        if local is None:
            return
        self._check_shape(loaded_shard_id, loaded_weight, (self.intermediate_size, param.shape[2]))
        inter = param.shape[1] // 2
        offset = 0 if proj == "gate" else inter
        param.data[local, offset:offset + inter] = loaded_weight[self.inter_start:self.inter_start + inter]

    def _load_down(self, param, loaded_weight, loaded_shard_id=None):
        """One expert's ``down_proj`` (``"down:N"``), ``(hidden, inter)``: this rank's
        columns of it."""
        _, local = self._parse_shard_id(loaded_shard_id, ("down",))
        if local is None:
            return
        self._check_shape(loaded_shard_id, loaded_weight, (param.shape[1], self.intermediate_size))
        inter = param.shape[2]
        param.data[local] = loaded_weight[:, self.inter_start:self.inter_start + inter]

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        token_valid: torch.Tensor | None = None,
        alignment: "MoeAlignment | None" = None,
    ) -> torch.Tensor:
        """This rank's ``(tokens, hidden)`` share of the routing-weighted expert sum.

        ``selected_experts`` are global ids. ``token_valid`` (``(tokens,)`` bool) marks
        CUDA-graph padding rows False. Their slots, and under EP the slots routed to other
        ranks' experts, go to the local sentinel ``num_local_experts`` and reach no GEMM.
        Every shape is static, so this stays torch.compile and CUDA-graph safe.

        ``alignment`` is a :class:`TopKRouter`'s alignment of ``selected_experts`` against
        ``gate_up_proj`` (see :meth:`TopKRouter.route`), which the kernel then skips
        computing. It holds only while no slot moves to the sentinel: without
        ``token_valid``, with every expert local.
        """
        routing_weights, selected_experts, skip_invalid = self.local_slots(
            routing_weights, selected_experts, token_valid, alignment)
        return self._dispatch(hidden_states, routing_weights, selected_experts, skip_invalid, alignment)

    def local_slots(
        self,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        token_valid: torch.Tensor | None = None,
        alignment: "MoeAlignment | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """``(routing_weights, selected_experts, skip_invalid)`` over this rank's expert ids, as
        :meth:`forward` dispatches them: the slots of padding rows and of other ranks' experts
        moved to the sentinel ``num_local_experts`` (``skip_invalid`` says whether any may be)."""
        keep = None if token_valid is None else token_valid[:, None]
        if self.num_local_experts != self.num_experts:
            selected_experts = selected_experts - self.expert_start
            owned = (selected_experts >= 0) & (selected_experts < self.num_local_experts)
            keep = owned if keep is None else keep & owned
        if keep is not None:
            if alignment is not None:
                raise ValueError("an alignment of the router's experts is stale once slots move to the sentinel")
            selected_experts = torch.where(keep, selected_experts, self.num_local_experts)
            # skipped slots never reach a GEMM; zeroing their weight is belt-and-braces
            routing_weights = torch.where(keep, routing_weights, 0)
        return routing_weights, selected_experts, keep is not None

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        skip_invalid: bool,
        alignment: "MoeAlignment | None" = None,
    ) -> torch.Tensor:
        """The fused Triton kernel on a GPU, else the naive loop, over local ids."""
        if _HAS_FUSED and hidden_states.is_cuda:
            return _fused_experts(
                hidden_states, self.gate_up_proj, self.down_proj,
                routing_weights, selected_experts, skip_invalid=skip_invalid, alignment=alignment,
                activation=self.activation,
            )
        return dispatch_experts_fused(
            hidden_states, self.gate_up_proj, self.down_proj, self.num_local_experts,
            selected_experts, routing_weights, skip_invalid=skip_invalid, activation=self.activation,
        )


# GLM-5.2 (`_dispatch` and both loaders) and GLM-5.3-Flash (both loaders) still use these
# three - port them onto `FusedExperts` and delete them
def _dispatch(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    skip_invalid: bool = False,
) -> torch.Tensor:
    """Pick fused-Triton if available, otherwise the naive loop."""
    if _HAS_FUSED and hidden_states.is_cuda:
        return _fused_experts(
            hidden_states, gate_up_proj, down_proj,
            routing_weights, selected_experts, skip_invalid=skip_invalid,
        )
    return dispatch_experts_fused(
        hidden_states, gate_up_proj, down_proj,
        num_experts, selected_experts, routing_weights, skip_invalid=skip_invalid,
    )


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

    def experts_prepare(self, fmt: str, block: tuple[int, int] = (128, 128), source=None) -> None:
        """Hold the experts as a block-fp8 checkpoint stores them, before it loads (``source``: how the
        checkpoint stores them, ``checkpoint.StoredBase``; one layout here)."""
        assert fmt == "fp8_block", fmt
        self._fp8_storage(block)

    def experts_scales(self) -> dict[str, torch.dtype]:
        """The experts' scale params and the float each stands for (the loader checks they all arrived)."""
        return {"gate_up_proj_scale_inv": torch.float32, "down_proj_scale_inv": torch.float32}

    def experts_after_load(self) -> None:
        """Put the loaded experts in their kernel's layout; block fp8 runs the checkpoint's as is."""

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
                                          skip_invalid=skip_invalid, activation=self._experts_activation)
        from mstar.utils.fused_moe import fused_experts_fp8

        return fused_experts_fp8(
            flat, w1, w2, s1, s2, routing_weights, selected_experts,
            block_size=self.experts_fp8_block, skip_invalid=skip_invalid,
            activation=activation_spec(self._experts_activation),
        )

    @property
    def _experts_activation(self):
        """The experts' gated activation: the block's ``activation`` (``silu`` or a ``SiTU``)."""
        return getattr(self, "activation", "silu")


# a stored 4-bit tensor's checkpoint suffix -> the stacked experts param's, after ``gate_up_proj`` / ``down_proj``
# (the stacked rules replace ``.experts.<proj>.__expertN__.weight`` and keep what follows)
_EXPERT_SUFFIX = {"weight": "", "weight_packed": "_packed", "weight_scale": "_scale",
                  "weight_zero_point": "_zero_point", "weight_scale_2": "_scale_2", "input_scale": "_input_scale"}
_INT_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def _expert_tensor_loader(fmt, suffix: str, loader):
    """``loader`` for one checkpoint tensor of a 4-bit expert: the format's ``from_checkpoint``, then floating
    data viewed as the param's integer container (the engine's node cast would round it)."""
    def load(param, loaded_weight, loaded_shard_id=None):
        loaded_weight = fmt.from_checkpoint(suffix, loaded_weight)
        if loaded_weight.dtype.is_floating_point or loaded_weight.dtype != param.dtype:
            loaded_weight = loaded_weight.contiguous().view(_INT_OF_SIZE[loaded_weight.element_size()])
        loader(param, loaded_weight, loaded_shard_id)
    return load


def _expert_scalar_loader(expert_start: int, num_local: int, param, loaded_weight, loaded_shard_id=None):
    """One expert's fp32 per-tensor scale (NVFP4's ``weight_scale_2`` / ``input_scale``) into its slot: ``(E, 2)``
    for gate / up, ``(E, 1)`` for down, in an int32 container; another rank's expert is skipped."""
    kind, expert = str(loaded_shard_id).split(":")
    local = int(expert) - expert_start
    if not 0 <= local < num_local:
        return
    param.data[local, 1 if kind == "up" else 0] = loaded_weight.to(torch.float32).reshape(1).view(torch.int32)[0]


class QuantExperts(Fp8Experts):
    """``Fp8Experts`` plus the 4-bit formats, ``w4a16`` (int4 groups), ``mxfp4`` and ``nvfp4``, held as the
    compressed-tensors / modelopt checkpoint stores them, per expert stacked: ``gate_up_proj_packed (E, 2I, H/8)``
    int32 or ``(E, 2I, H/2)`` uint8, ``gate_up_proj_scale``, ``gate_up_proj_zero_point`` (asymmetric int4), NVFP4's
    ``gate_up_proj`` / ``gate_up_proj_scale`` / ``gate_up_proj_scale_2`` / ``gate_up_proj_input_scale``, and down's
    the same. The stacked rules that route ``experts.N.gate_proj.weight`` route every suffix after ``weight``
    the same way, so a model's loader needs no change beyond passing those suffixes through its remap.

    Dispatch: B200/B300 run ``mxfp4`` / ``nvfp4`` through FlashInfer's TRT-LLM FP4 MoE where it has the
    activation (``experts_after_load`` shuffles the weights into its layout on the device); otherwise ``w4a16`` /
    ``mxfp4`` run vLLM's Marlin MoE (``utils.marlin_moe``, converted at load and probed against the Triton kernel)
    where it builds, else ``fused_experts_w4a16`` (Triton, the weight dequantized in registers). Off CUDA, the
    reference: every expert dequantized, NVFP4's activations quantized as its kernel would."""

    # the format (as the checkpoint stores it) once the experts hold 4-bit storage
    experts_packed = None

    @property
    def experts_quantized(self) -> bool:
        return self.experts_fp8_block is not None or self.experts_packed is not None

    def experts_quant_capability(self) -> QuantCapability:
        if self.experts_packed is not None:
            return QuantCapability(False, f"already {self.experts_packed.name}")
        return super().experts_quant_capability()

    def experts_geometry(self) -> QuantGeometry:
        e, two_inter, hidden = self._experts_dims()
        return QuantGeometry(two_inter, hidden, (two_inter // 2, two_inter // 2), (hidden,))

    def _experts_dims(self) -> tuple[int, int, int]:
        dims = getattr(self, "_experts_bf16_dims", None)
        return dims if dims is not None else tuple(self.experts.gate_up_proj.shape)

    def experts_scales(self) -> dict[str, torch.dtype]:
        if self.experts_packed is None:
            return super().experts_scales()
        return {name: view[0] for name, view in self._experts_views.items() if view[0].is_floating_point}

    def _attach_quant_loaders(self, gate_up, down) -> None:
        if self.experts_packed is None:
            self._attach_fp8_loaders(gate_up, down)
            return
        from functools import partial

        start = getattr(self, "expert_start", 0)
        local = self._experts_dims()[0]
        for name, (_, units, proj, suffix) in self._experts_views.items():
            param = getattr(self.experts, name)
            if units == (None, None):
                param.weight_loader = partial(_expert_scalar_loader, start, local)
            elif proj == "gate_up_proj":
                param.weight_loader = _expert_tensor_loader(self.experts_packed, suffix, gate_up(units[0]))
            else:
                param.weight_loader = _expert_tensor_loader(self.experts_packed, suffix, down(units[1]))

    def _packed_storage(self, fmt, data: dict[str, torch.Tensor] | None = None) -> None:
        """Replace the bf16 experts with ``fmt``'s storage, stacked per expert, empty or ``data``."""
        e, two_inter, hidden = self._experts_bf16_dims = self._experts_dims()
        experts = self.experts
        device = next(iter(experts.parameters())).device
        self._experts_views = {}
        for proj, (n, k) in (("gate_up_proj", (two_inter, hidden)), ("down_proj", (hidden, two_inter // 2))):
            if hasattr(experts, proj):
                delattr(experts, proj)
            for st in fmt.storage(n, k):
                name = proj + _EXPERT_SUFFIX[st.name]
                # a per-tensor scale: one per expert per checkpoint linear (gate and up are two)
                shape = (e, 2 if proj == "gate_up_proj" else 1) if st.units == (None, None) else (e, *st.shape)
                tensor = torch.empty(shape, dtype=st.dtype, device=device) if data is None else data[name]
                assert tuple(tensor.shape) == shape and tensor.dtype == st.dtype, (name, tensor.shape, tensor.dtype)
                if hasattr(experts, name):
                    delattr(experts, name)
                setattr(experts, name, nn.Parameter(tensor, requires_grad=False))
                self._experts_views[name] = (st.stands_for, st.units, proj, st.name)
        self.experts_packed = fmt
        self._experts_kernel = None
        self._reattach_loaders()

    def experts_prepare(self, fmt: str, block: tuple[int, int] = (128, 128), source=None) -> None:
        """Hold the experts as the checkpoint stores them (``source``: its layout and params), before it loads."""
        if fmt == "fp8_block":
            super().experts_prepare(fmt, block)
            return
        from mstar.model.components.quant_linear import linear_format

        self._packed_storage(linear_format(fmt).for_stored(source))

    @torch.no_grad()
    def experts_quantize(self, fmt: str, block: tuple[int, int] = (128, 128)) -> None:
        """Quantize this rank's loaded bf16 experts, one expert at a time."""
        if fmt == "fp8_block":
            super().experts_quantize(fmt, block)
            return
        from mstar.model.components.quant_linear import as_container, linear_format

        fmt_ = linear_format(fmt)
        data = {}
        for proj in ("gate_up_proj", "down_proj"):
            weight = getattr(self.experts, proj).data
            parts = [fmt_.quantize_weight(weight[i], None) for i in range(weight.shape[0])]
            for key in parts[0]:
                data[proj + _EXPERT_SUFFIX[key]] = torch.stack([as_container(p[key]) for p in parts])
        self._packed_storage(fmt_, data)

    def _expert_buffers(self, proj: str, index) -> dict[str, torch.Tensor]:
        """One expert's (or, ``index`` a slice, several experts') storage under the format's names, viewed as
        what it stands for."""
        out = {}
        for name, (stands_for, _, p, suffix) in self._experts_views.items():
            if p == proj:
                t = getattr(self.experts, name)[index]
                out[suffix] = t if not stands_for.is_floating_point else t.view(stands_for)
        return out

    # which kernel serves the 4-bit experts: "auto" (FlashInfer's FP4 MoE on B200/B300 where it has the
    # activation, else Marlin), "trtllm" (FlashInfer's or a load error), "marlin" (Marlin, else Triton); a model
    # sets it from its serving config (Kimi K3: model_kwargs.routed_experts_backend)
    routed_experts_backend = "auto"
    # FlashInfer's FP4 MoE tuned at load over these token buckets (fp4_flashinfer.autotune, as vLLM's boot does),
    # its tactics kept per GPU across boots; None the default buckets. Off unless the model asks (Kimi K3 does).
    routed_experts_autotune = False
    # "kernel": a noaux_tc router runs inside FlashInfer's FP4 MoE from the logits (fp4_flashinfer.kernel_routing,
    # checked on the GPU at load); "torch": the block routes and passes ids and weights. Kimi K3 asks for "kernel".
    routed_experts_routing = "torch"
    routed_experts_tune_tokens: tuple[int, ...] | None = None
    # every call passes skip_invalid (a model that masks a padded step's padding rows on every step): tune that call
    routed_experts_skip_invalid = False

    def experts_after_load(self) -> None:
        """Put the loaded experts in their kernel's layout, on the device they serve from (B200/B300's FlashInfer
        FP4 MoE; else, for w4a16 / mxfp4, Marlin MoE where it builds and agrees with the Triton kernel, which reads
        the checkpoint layout as is and serves the rest)."""
        if self.experts_packed is None or self.experts_packed.name not in ("mxfp4", "nvfp4", "w4a16"):
            return
        backend = self.routed_experts_backend
        if backend not in ("auto", "trtllm", "marlin"):
            raise ValueError(f"routed_experts_backend must be auto, trtllm or marlin, got {backend!r}")
        if self.experts_packed.name == "w4a16":
            self._experts_to_marlin()
            return
        if self.experts_packed.name == "nvfp4":
            # one GEMM per expert takes one tensor scale and one activation scale for gate and up together
            for name in ("gate_up_proj_scale_2", "gate_up_proj_input_scale"):
                pair = getattr(self.experts, name).view(torch.float32)
                if not bool((pair[:, 0] == pair[:, 1]).all()):
                    raise ValueError(f"nvfp4 experts: the checkpoint's gate and up {name.rpartition('_proj_')[2]} "
                                     "differ; serve these experts bf16 instead")
        from mstar.utils.fused_moe import fp4_flashinfer

        fmt = self.experts_packed.name
        device = next(iter(self.experts.parameters())).device
        reason = "routed_experts_backend: marlin"
        if backend != "marlin":
            dtype, detail = fp4_flashinfer.choose(self, device)
            if dtype is not None:
                kernel = fp4_flashinfer.TrtllmFp4Experts.convert(self, device, activation=dtype)
                self._experts_kernel = kernel
                if self.routed_experts_routing == "kernel":
                    routing, why = fp4_flashinfer.kernel_routing(self)
                    if routing is not None:
                        why, _ = fp4_flashinfer.probe_routing(self, device, dtype, routing)
                    if why is None:
                        kernel.routing = routing
                        detail = f"{detail}; routing in the kernel (DeepSeekV3, one group)"
                    else:
                        detail = f"{detail}; routing in torch: {why}"
                if self.routed_experts_autotune:
                    # expert parallel blocks pass the sentinel id on every call: tune the call they make
                    total = getattr(self, "num_experts", None)
                    # (a block masking its padding rows too, unless the kernel routes from the logits: no sentinel)
                    ep = (getattr(self, "num_local_experts", total) != total
                          or (self.routed_experts_skip_invalid and kernel.routing is None))
                    tuned = fp4_flashinfer.autotune(self, kernel, device, self.routed_experts_tune_tokens,
                                                    skip_invalid=ep)
                    detail = f"{detail}; {tuned}" if detail else tuned
                self._log_experts_backend(
                    f"FlashInfer trtllm-gen {fmt.upper()} x {dtype.upper()}", detail)
                return
            reason = detail
            if backend == "trtllm":
                raise RuntimeError(f"routed_experts_backend: trtllm, but FlashInfer's FP4 MoE can't serve these "
                                   f"{fmt} experts: {reason}")
            if fp4_flashinfer.unavailable_reason(device) is None:
                # a Blackwell GPU whose FP4 MoE can't take these experts
                logger.warning("%s experts not on FlashInfer's FP4 MoE: %s", fmt, reason)
        if fmt == "mxfp4":
            self._experts_to_marlin()
        if device.type == "cuda":
            kernel = ("Marlin MoE" if self._experts_kernel is not None else
                      "Triton W4A16" if fmt == "mxfp4" else "reference dispatch")
            self._log_experts_backend(f"{kernel} {fmt.upper()} x BF16", f"not FlashInfer's FP4 MoE: {reason}")

    def _log_experts_backend(self, backend: str, detail: str) -> None:
        """One INFO line per block class and backend (a model's ~90 MoE layers would repeat it)."""
        key = (type(self).__name__, backend)
        if key in _EXPERTS_BACKEND_LOGGED:
            return
        _EXPERTS_BACKEND_LOGGED.add(key)
        e, two_inter, hidden = self._experts_dims()
        logger.info("%s routed-expert backend: %s (%s experts x %d x %d, activation %s, setting %s%s)",
                    type(self).__name__, backend, e, two_inter // 2, hidden,
                    activation_spec(self._experts_activation), self.routed_experts_backend,
                    f"; {detail}" if detail else "")

    def _experts_to_marlin(self) -> None:
        """Marlin MoE for w4a16 / mxfp4 experts on CUDA (``utils.marlin_moe.MarlinExperts``: converted, probed
        against the Triton kernel, the checkpoint copy freed); the Triton kernel where it can't."""
        device = next(iter(self.experts.parameters())).device
        if device.type != "cuda":
            return
        from mstar.utils.marlin_moe import MarlinExperts

        self._experts_kernel = MarlinExperts.convert(self, device)

    def _experts_quant(
        self, flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor,
        skip_invalid: bool = False,
    ) -> torch.Tensor:
        if self.experts_packed is None:
            return self._experts_fp8(flat, routing_weights, selected_experts, skip_invalid=skip_invalid)
        return self._experts_4bit(flat, routing_weights, selected_experts, skip_invalid)

    def _experts_4bit(self, flat, routing_weights, selected_experts, skip_invalid):
        fmt = self.experts_packed
        if self._experts_kernel is not None:
            return self._experts_kernel(flat, routing_weights, selected_experts, skip_invalid)
        if flat.is_cuda and fmt.name in ("w4a16", "mxfp4"):
            from mstar.model.components.int4_formats import KIND_E2M1, KIND_INT4_SYM, KIND_INT4_ZP
            from mstar.utils.fused_moe.w4a16 import fused_experts_w4a16

            w1, s1, z1 = (self._expert_buffers("gate_up_proj", slice(None)).get(k)
                          for k in ("weight_packed", "weight_scale", "weight_zero_point"))
            w2, s2, z2 = (self._expert_buffers("down_proj", slice(None)).get(k)
                          for k in ("weight_packed", "weight_scale", "weight_zero_point"))
            kind = KIND_E2M1 if fmt.name == "mxfp4" else KIND_INT4_SYM if z1 is None else KIND_INT4_ZP
            if kind == KIND_E2M1:
                s1, s2 = s1.view(torch.uint8), s2.view(torch.uint8)
            return fused_experts_w4a16(
                flat, w1.view(torch.uint8), w2.view(torch.uint8), s1, s2, z1, z2, routing_weights, selected_experts,
                kind, fmt.group, activation_spec(self._experts_activation), skip_invalid,
            )
        if flat.is_cuda:
            raise RuntimeError(f"{fmt.name} experts need FlashInfer's FP4 MoE (compute capability 10.0)")
        return self._experts_reference(flat, routing_weights, selected_experts, skip_invalid)

    def _experts_reference(self, flat, routing_weights, selected_experts, skip_invalid):
        """What the 4-bit experts compute, in fp32 per expert (tests): the dequantized weights, and for NVFP4 the
        activations quantized with each expert's input scale as the kernel quantizes them."""
        fmt = self.experts_packed
        e = self._experts_dims()[0]
        out = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
        for i in range(e):
            hit = selected_experts == i
            tokens, slot = torch.where(hit)
            if not tokens.numel():
                continue
            x = flat[tokens].float()
            gate_up = self._expert_matmul(fmt, "gate_up_proj", i, x)
            h = _gated(gate_up, self._experts_activation).to(flat.dtype).float()
            y = self._expert_matmul(fmt, "down_proj", i, h)
            out.index_add_(0, tokens, y * routing_weights[tokens, slot, None].float())
        return out.to(flat.dtype)

    def _expert_matmul(self, fmt, proj, i, x):
        buffers = self._expert_buffers(proj, i)
        if fmt.name != "nvfp4":
            return x @ fmt.dequantize(buffers).to(torch.bfloat16).float().t()
        # NVFP4: each checkpoint linear (gate, up) keeps its own tensor scale and activation scale
        halves = []
        rows = buffers["weight"].shape[0] // buffers["weight_scale_2"].numel()
        for j in range(buffers["weight_scale_2"].numel()):
            part = {"weight": buffers["weight"][j * rows:(j + 1) * rows],
                    "weight_scale": buffers["weight_scale"][j * rows:(j + 1) * rows],
                    "weight_scale_2": buffers["weight_scale_2"][j:j + 1],
                    "input_scale": buffers["input_scale"][j:j + 1]}
            halves.append(fmt.reference_stored(x.to(torch.bfloat16), part).float())
        return torch.cat(halves, dim=-1)


def _ep_gate_up_fp8_weight_loader(
    expert_start: int, num_local_experts: int, full_inter: int, row_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """:func:`_gate_up_fp8_weight_loader` under expert parallelism: one expert's gate or up weight (``row_unit``
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
    """Its ``down_proj`` twin (data or scales, whole)."""
    assert loaded_shard_id is not None
    local_idx = int(str(loaded_shard_id).split(":")[1]) - expert_start
    if not 0 <= local_idx < num_local_experts:
        return
    if loaded_weight.dtype == FP8_DTYPE:
        loaded_weight = loaded_weight.view(torch.uint8)
    param.data[local_idx] = loaded_weight


class ParallelSparseMoeBlock(QuantExperts, nn.Module):
    """Top-K sparse MoE over a tensor-parallel group, with an optional shared expert.

    The router is replicated, the experts (:class:`FusedExperts`) compute each rank's
    partial sum, and one all-reduce adds them up. With ``tp_size == 1`` there is no
    communication. With ``expert_parallel`` each rank holds whole experts instead of a
    slice of every expert, and :func:`_sync_routing` first makes every rank use rank 0's
    top-k.

    With a ``shared_expert`` (any ``hidden -> hidden`` module, a ``ParallelGatedMLP`` on
    the same group reducing its own output) the output is::

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    ``token_valid`` (``(tokens,)`` bool) marks padding rows False; they skip every
    routed-expert GEMM.

    ``track_expert_load`` (EP only, off by default, CUDA-graph safe) accumulates
    per-expert slot counts of the valid rows in the ``expert_load`` buffer; read and
    reset it with :meth:`pop_expert_load`.

    ``activation`` is the experts' gated activation, ``silu`` or a ``SiTU``. The experts are
    one quantizable layer (:class:`QuantExperts`): quantized, they keep the storage the
    mixin gave ``experts`` and its dispatch runs them.
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
        shared_expert: nn.Module | None = None,
        expert_parallel: bool = False,
        track_expert_load: bool = False,
        activation: str | Callable = "silu",
    ) -> None:
        super().__init__()
        if track_expert_load and not expert_parallel:
            raise ValueError("track_expert_load needs expert_parallel=True")
        self.comm_group = comm_group or CommGroup.trivial()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.expert_parallel = expert_parallel
        self.activation = activation

        self.gate = router if router is not None else TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = FusedExperts(
            num_experts=num_experts, hidden_size=hidden_size, intermediate_size=moe_intermediate_size,
            comm_group=self.comm_group, expert_parallel=expert_parallel, activation=activation,
        )
        self.shared_expert = shared_expert
        if shared_expert is not None:
            self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self.track_expert_load = track_expert_load
        if track_expert_load:
            # Routing is replicated, so every rank counts every expert.
            self.register_buffer(
                "expert_load", torch.zeros(num_experts, dtype=torch.int64), persistent=False,
            )

    @property
    def num_local_experts(self) -> int:
        return self.experts.num_local_experts

    @property
    def expert_start(self) -> int:
        return self.experts.expert_start

    def _reattach_loaders(self) -> None:
        """The quantized storage's loaders (``FusedExperts`` attaches the bf16 ones itself)."""
        from functools import partial

        if not self.experts_quantized:
            return
        full_inter = self.moe_intermediate_size
        if self.expert_parallel:
            start, local = self.expert_start, self.num_local_experts
            self._attach_quant_loaders(
                lambda unit: partial(_ep_gate_up_fp8_weight_loader, start, local, full_inter, unit),
                lambda unit: partial(_ep_down_proj_fp8_weight_loader, start, local),
            )
            return
        rank, size = self.comm_group.rank, self.comm_group.world_size
        self._attach_quant_loaders(
            lambda unit: partial(_gate_up_fp8_weight_loader, rank, size, full_inter, unit),
            lambda unit: partial(_down_proj_fp8_weight_loader, rank, size, full_inter, unit),
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # new parameters lose their loaders
        self._reattach_loaders()
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
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        shared = None if self.shared_expert is None else self.shared_expert(flat)

        # the experts move slots to the sentinel under a mask or EP, TP > 1 doesn't align, and
        # a quantized dispatch takes no alignment
        can_align = token_valid is None and self.comm_group.world_size == 1 and not self.experts_quantized
        routing_weights, selected_experts, router_states_next, alignment = _route(
            self.gate, flat, self.experts.gate_up_proj if can_align else None, router_states,
        )
        out = self.experts_forward(
            flat, routing_weights, selected_experts, token_valid=token_valid, alignment=alignment,
        )
        if shared is not None:
            out = out + torch.sigmoid(self.shared_expert_gate(flat)) * shared
        out = out.view(input_shape)
        return (out, router_states_next) if return_router_states else out

    def experts_forward(
        self,
        flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        *,
        token_valid: torch.Tensor | None = None,
        alignment: "MoeAlignment | None" = None,
        reduce_results: bool = True,
    ) -> torch.Tensor:
        """The routed experts on ``flat (tokens, hidden)`` given each token's top-k, summed over
        the ranks: ``forward`` after its router, for a block whose router reads another input
        (a latent MoE routes on the hidden state and runs its experts on a projection of it).
        ``reduce_results=False`` returns this rank's partial sum, for a caller that reduces it in
        a fused kernel (e.g. the all-reduce + RMSNorm of ``RMSNorm.forward(x, comm_group)``)."""
        if self.track_expert_load:
            ids = selected_experts.flatten()
            if token_valid is None:
                ones = torch.ones_like(ids, dtype=torch.int64)
            else:
                ones = token_valid[:, None].expand_as(selected_experts).flatten().to(torch.int64)
            self.expert_load.index_add_(0, ids, ones)
        if self.expert_parallel and self.comm_group.world_size > 1:
            routing_weights, selected_experts = _sync_routing(
                self.comm_group, routing_weights, selected_experts,
            )
        if self.experts_quantized:
            routing_weights, selected_experts, skip_invalid = self.experts.local_slots(
                routing_weights, selected_experts, token_valid)
            out = self._experts_quant(flat, routing_weights, selected_experts, skip_invalid=skip_invalid)
        else:
            out = self.experts(
                flat, routing_weights, selected_experts, token_valid=token_valid, alignment=alignment)
        if reduce_results and self.comm_group.world_size > 1:
            out = self.comm_group.all_reduce(out)
        return out
