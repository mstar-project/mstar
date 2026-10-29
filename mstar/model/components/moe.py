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

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.utils.fp8_block import FP8_DTYPE

logger = logging.getLogger(__name__)

# bf16 x bf16 -> fp32 mm (aten::mm.dtype); older torch lacks the overload
_MM_OUT_DTYPE = hasattr(torch.ops.aten.mm, "dtype")


# Optional fused Triton MoE path. Imports succeed wherever triton is
# installed; the alignment op JIT-builds a vendored CUDA kernel on first use
# (or falls back to a torch impl), so no extra kernel package is needed. The
# actual kernels only run on CUDA -- guarded by the ``is_cuda`` check below.
try:
    from mstar.utils.fused_moe import fused_experts as _fused_experts

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
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight)
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
        # fp32 copy of ``weight``, built once by ``finalize_weights``: the
        # forward needs fp32 and the router never changes, so casting it once
        # per layer per step is pure waste. A plain attribute, not a buffer,
        # so model.to(bf16) cannot downcast it.
        self._weight_fp32: torch.Tensor | None = None

    def finalize_weights(self) -> None:
        """Cache the fp32 router weight; call after the weights are loaded."""
        self._weight_fp32 = self.weight.detach().float()

    def forward(
        self, hidden_states: torch.Tensor, router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        flat = hidden_states.reshape(-1, self.hidden_size)
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
        scores = logits.sigmoid()  # (T, E)

        biased = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_ids = torch.topk(biased, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
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
        skip_invalid: set when ``selected_experts`` may hold the sentinel
            ``num_experts`` (a padding row's, or another EP rank's expert);
            those slots contribute nothing.
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


def _sync_routing(
    comm_group: CommGroup,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make every rank use group rank 0's top-k.

    Ranks compile the router separately, so a near-tie can resolve
    differently per rank; sharded dispatch would then give a token more or
    fewer than k experts. Both tensors travel in one fp32 tensor, which holds
    expert ids below 2**24 and bf16/fp16/fp32 weights exactly. Other ranks
    send zeros, so the all-reduce is an exact broadcast; a plain
    ``dist.broadcast`` is not dynamo-traceable and would break the graph.
    """
    top_k = selected_experts.shape[-1]
    packed = torch.cat([selected_experts.float(), routing_weights.float()], dim=-1)
    if comm_group.rank != 0:
        packed = torch.zeros_like(packed)
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
    ) -> None:
        super().__init__()
        group = comm_group or CommGroup.trivial()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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
        self.gate_up_proj.weight_loader = self._load_gate_up
        self.down_proj.weight_loader = self._load_down

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
    ) -> torch.Tensor:
        """This rank's ``(tokens, hidden)`` share of the routing-weighted expert sum.

        ``selected_experts`` are global ids. ``token_valid`` (``(tokens,)`` bool) marks
        CUDA-graph padding rows False. Their slots, and under EP the slots routed to other
        ranks' experts, go to the local sentinel ``num_local_experts`` and reach no GEMM.
        Every shape is static, so this stays torch.compile and CUDA-graph safe.
        """
        keep = None if token_valid is None else token_valid[:, None]
        if self.num_local_experts != self.num_experts:
            selected_experts = selected_experts - self.expert_start
            owned = (selected_experts >= 0) & (selected_experts < self.num_local_experts)
            keep = owned if keep is None else keep & owned
        if keep is not None:
            selected_experts = torch.where(keep, selected_experts, self.num_local_experts)
            # skipped slots never reach a GEMM; zeroing their weight is belt-and-braces
            routing_weights = torch.where(keep, routing_weights, 0)
        return self._dispatch(hidden_states, routing_weights, selected_experts, keep is not None)

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        skip_invalid: bool,
    ) -> torch.Tensor:
        """The fused Triton kernel on a GPU, else the naive loop, over local ids."""
        if _HAS_FUSED and hidden_states.is_cuda:
            return _fused_experts(
                hidden_states, self.gate_up_proj, self.down_proj,
                routing_weights, selected_experts, skip_invalid=skip_invalid,
            )
        return dispatch_experts_fused(
            hidden_states, self.gate_up_proj, self.down_proj, self.num_local_experts,
            selected_experts, routing_weights, skip_invalid=skip_invalid,
        )


# GLM-5.2 still uses these three - port it onto `FusedExperts` and delete them
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
    tp_slice = loaded_weight[start:start + shard_inter, :]
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
    param.data[expert_idx, :, :] = loaded_weight[:, start:start + shard_inter]


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


class ParallelSparseMoeBlock(nn.Module):
    """Top-K sparse MoE over a tensor-parallel group, with an optional shared expert.

    The router is replicated; over several ranks :func:`_sync_routing` makes every rank
    use rank 0's top-k, the experts (:class:`FusedExperts`) compute each rank's partial
    sum, and one all-reduce adds them up. With ``tp_size == 1`` there is no
    communication. With ``expert_parallel`` each rank holds whole experts instead of a
    slice of every expert.

    With a ``shared_expert`` (any ``hidden -> hidden`` module, a ``ParallelGatedMLP`` on
    the same group reducing its own output) the output is::

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    ``token_valid`` (``(tokens,)`` bool) marks padding rows False; they skip every
    routed-expert GEMM.

    ``track_expert_load`` (EP only, off by default, CUDA-graph safe) accumulates
    per-expert slot counts of the valid rows in the ``expert_load`` buffer; read and
    reset it with :meth:`pop_expert_load`.
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
    ) -> None:
        super().__init__()
        if track_expert_load and not expert_parallel:
            raise ValueError("track_expert_load needs expert_parallel=True")
        self.comm_group = comm_group or CommGroup.trivial()
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
        self.experts = FusedExperts(
            num_experts=num_experts, hidden_size=hidden_size, intermediate_size=moe_intermediate_size,
            comm_group=self.comm_group, expert_parallel=expert_parallel,
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

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
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

        routing_weights, selected_experts, router_states_next = self.gate(flat, router_states)
        if self.track_expert_load:
            ids = selected_experts.flatten()
            if token_valid is None:
                ones = torch.ones_like(ids, dtype=torch.int64)
            else:
                ones = token_valid[:, None].expand_as(selected_experts).flatten().to(torch.int64)
            self.expert_load.index_add_(0, ids, ones)
        if self.comm_group.world_size > 1:
            routing_weights, selected_experts = _sync_routing(
                self.comm_group, routing_weights, selected_experts,
            )
        out = self.experts(flat, routing_weights, selected_experts, token_valid=token_valid)
        if self.comm_group.world_size > 1:
            out = self.comm_group.all_reduce(out)
        if shared is not None:
            out = out + torch.sigmoid(self.shared_expert_gate(flat)) * shared
        out = out.view(input_shape)
        return (out, router_states_next) if return_router_states else out
