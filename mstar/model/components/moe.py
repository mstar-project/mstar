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
from mstar.model.components.norm import InputScaledRMSNorm
from mstar.model.loader.iterators import SliceSpec, TensorSlice
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
    from mstar.utils.fused_moe import moe_input_scaled_rms_sum_reduce_triton

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


def _split_fp32(w: torch.Tensor) -> torch.Tensor:
    """``w`` as bf16 ``(hi, mid, lo)`` on a new first dim, summing to ``w`` in fp32's
    24 bits: each part rounds what the parts before it left."""
    w = w.float()
    hi = w.bfloat16()
    mid = (w - hi.float()).bfloat16()
    lo = (w - hi.float() - mid.float()).bfloat16()
    return torch.stack([hi, mid, lo])


class NoAuxTcRouter(nn.Module):
    """DeepSeek-V3's ``noaux_tc`` router, groupless (``n_group=1``), as GLM-5.x routes.

    Sigmoid scores in fp32; each token's top-k by score + ``e_score_correction_bias``; weights
    are the unbiased scores of those experts, renormalized if ``norm_topk_prob``, times
    ``routed_scaling_factor``. ``logit_scale`` multiplies the logits (Muse Spark's
    ``output_multiplier``). Stateless (see the router contract above).

    By default the weight is one ``(E, H)`` parameter, which the engine's cast to bf16
    rounds. ``fp32_weight=True`` routes on an fp32 checkpoint weight at fp32's precision:
    it loads as bf16 parts ``(3, E, H)`` (``hi + mid + lo``; a bf16 checkpoint weight
    as its one part), the logits are one bf16 GEMM against all the parts with fp32
    output, summed after it, and the parts and the fp32 bias keep their dtypes through
    the engine's cast. At Muse Spark's shape on an H200 (H 8192, E 256, 8192 tokens),
    against fp64, that GEMM's logits are within 3.9e-5 in 0.14 ms (IEEE fp32 8.5e-6 in
    0.69 ms, TF32 7.3e-4).
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool = True,
        logit_scale: float = 1.0,
        fp32_weight: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.logit_scale = logit_scale
        self.fp32_weight = fp32_weight

        if fp32_weight:
            self.weight = nn.Parameter(
                torch.zeros(3, n_routed_experts, hidden_size, dtype=torch.bfloat16)
            )
            self.weight.weight_loader = self._load_weight_parts
        else:
            self.weight = nn.Parameter(torch.zeros(n_routed_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(n_routed_experts, dtype=torch.float32)
        )
        # fp32 copy of ``weight``, built once by ``finalize_weights``: the
        # forward needs fp32 and the router never changes, so casting it once
        # per layer per step is pure waste. A plain attribute, not a buffer,
        # so model.to(bf16) cannot downcast it.
        self._weight_fp32: torch.Tensor | None = None

    @staticmethod
    def _load_weight_parts(param: nn.Parameter, loaded: torch.Tensor) -> None:
        if loaded.shape != param.shape[1:]:
            raise ValueError(f"router weight is {tuple(loaded.shape)}, expected {tuple(param.shape[1:])}")
        if loaded.dtype == torch.float32:
            parts = _split_fp32(loaded)
        elif loaded.dtype == torch.bfloat16:
            parts = loaded[None]
        else:
            raise ValueError(f"the router weight is fp32 or bf16 here, not {loaded.dtype}")
        param.data = parts.to(param.device)

    def _apply(self, fn, recurse=True):
        if not self.fp32_weight:
            return super()._apply(fn, recurse)

        # A cast moves the parameters but keeps their dtypes: it would round
        # the bias, and the GEMM reads the weight's parts as bf16.
        def keep_dtype(t: torch.Tensor) -> torch.Tensor:
            out = fn(t)
            return out if out.dtype == t.dtype else t.to(device=out.device)

        return super()._apply(keep_dtype, recurse)

    def _float_weight(self) -> torch.Tensor:
        w = self.weight.detach().float()
        return w.sum(0) if self.fp32_weight else w

    def finalize_weights(self) -> None:
        """Cache the fp32 router weight; call after the weights are loaded."""
        self._weight_fp32 = self._float_weight()

    def forward(
        self, hidden_states: torch.Tensor, router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        flat = hidden_states.reshape(-1, self.hidden_size)
        # the engine's bf16 autocast would run F.linear in bf16 and round the logits
        with torch.autocast(flat.device.type, enabled=False):
            if _MM_OUT_DTYPE and flat.is_cuda and flat.dtype == self.weight.dtype == torch.bfloat16:
                # bf16 x bf16 products are exact in fp32: the fp32 router in another
                # summation order, without an fp32 copy of the tokens
                logits = torch.mm(flat, self.weight.flatten(0, -2).t(), out_dtype=torch.float32)
                if self.fp32_weight:
                    logits = logits.unflatten(-1, (self.weight.shape[0], -1)).sum(-2)
            else:
                w = self._weight_fp32
                if w is None or w.device != self.weight.device:
                    w = self._float_weight()
                logits = F.linear(flat.float(), w)
        if self.logit_scale != 1.0:
            logits = logits * self.logit_scale
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
    reduce_results: bool = True,
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
        reduce_results: False returns each slot's weighted output,
            ``(tokens, top_k, hidden_size)``, zero for a skipped slot, as
            :func:`mstar.utils.fused_moe.fused_experts` does.
    """
    if reduce_results:
        final_hidden_states = torch.zeros_like(hidden_states)
    else:
        final_hidden_states = hidden_states.new_zeros(*selected_experts.shape, hidden_states.shape[-1])

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
        if not reduce_results:
            final_hidden_states[token_idx, top_k_pos] = current_hidden_states.to(final_hidden_states.dtype)
            continue
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
    reduce_results: bool = True,
    zero_skipped: bool = True,
) -> torch.Tensor:
    """Pick fused-Triton if available, otherwise the naive loop (which always
    zeroes skipped slots)."""
    if _HAS_FUSED and hidden_states.is_cuda:
        return _fused_experts(
            hidden_states, gate_up_proj, down_proj,
            routing_weights, selected_experts, skip_invalid=skip_invalid,
            reduce_results=reduce_results, zero_skipped=zero_skipped,
        )
    return dispatch_experts_fused(
        hidden_states, gate_up_proj, down_proj,
        num_experts, selected_experts, routing_weights, skip_invalid=skip_invalid,
        reduce_results=reduce_results,
    )


def _sync_routing(
    comm_group: CommGroup,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make every rank use group rank 0's top-k.

    Ranks compile the router separately, or the router's input itself
    differs in its last bits (a replicated layer before it, such as an EP
    block's ``post_experts``, may be autotuned to another kernel on each
    rank), so a
    near-tie can resolve differently per rank; sharded dispatch would then
    give a token more or fewer than k experts. Both tensors travel in one
    fp32 tensor, which holds expert ids below 2**24 and bf16/fp16/fp32
    weights exactly. Other ranks send zeros, so the all-reduce is an exact
    broadcast; a plain ``dist.broadcast`` is not dynamo-traceable and would
    break the graph.
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

    def _attach_weight_loaders(self, tp_rank: int, tp_size: int, full_inter: int):
        from functools import partial

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
        routing_weights, selected_experts, router_states_next = self.gate(flat, router_states)
        skip = token_valid is not None
        if skip:
            selected_experts, routing_weights = _mask_padding(
                selected_experts, routing_weights, token_valid, self.num_experts,
            )

        if self.comm_group.world_size == 1:
            out = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_experts, selected_experts, routing_weights, skip_invalid=skip,
            )
        else:
            routing_weights, selected_experts = _sync_routing(
                self.comm_group, routing_weights, selected_experts,
            )
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


class ParallelSparseMoeBlockWithSharedExpert(nn.Module):
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

    def _attach_weight_loaders(self, tp_rank: int, tp_size: int, full_inter: int):
        from functools import partial

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

        routing_weights, selected_experts, router_states_next = self.gate(flat, router_states)
        if self.comm_group.world_size == 1:
            routed = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_experts, selected_experts, routing_weights,
            )
        else:
            routing_weights, selected_experts = _sync_routing(
                self.comm_group, routing_weights, selected_experts,
            )
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


def _ep_packed_weight_loader(
    expert_start: int, num_experts: int, param: nn.Parameter, loaded_weight: torch.Tensor,
) -> None:
    """Load a packed expert tensor's rows for this rank's experts.

    Packed is ``gate_up_proj`` as ``(E, 2 * inter, hidden)``, each expert's
    gate rows before its up rows, or ``down_proj`` as ``(E, hidden, inter)``.
    A checkpoint that interleaves gate and up rows has the same shape and would
    load wrong; it is not supported. ``loaded_weight`` holds all ``num_experts``
    rows, or only this rank's when the read was cut to them
    (:func:`ep_expert_slice_spec`).
    """
    if loaded_weight.shape[0] == num_experts:
        loaded_weight = loaded_weight[expert_start:expert_start + param.shape[0]]
    if loaded_weight.shape != param.shape:
        raise ValueError(
            f"packed experts {tuple(loaded_weight.shape)} do not fit this rank's "
            f"{tuple(param.shape)} of {num_experts} experts"
        )
    param.data.copy_(loaded_weight)


def _ep_gate_up_weight_loader(
    expert_start: int, num_local_experts: int, num_experts: int, full_inter: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's gate_proj or up_proj whole, if this rank owns it, or
    a packed ``gate_up_proj`` (``loaded_shard_id`` None).

    ``loaded_shard_id`` is ``"gate:N"`` or ``"up:N"`` with ``N`` the global
    expert index. Experts outside ``[expert_start, expert_start +
    num_local_experts)`` belong to another rank and are skipped.
    ``param`` has shape ``(E_local, 2*inter, hidden)``.
    """
    if loaded_shard_id is None:
        return _ep_packed_weight_loader(expert_start, num_experts, param, loaded_weight)
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
    expert_start: int, num_local_experts: int, num_experts: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's down_proj whole, if this rank owns it, or a packed
    ``down_proj`` (``loaded_shard_id`` None).

    ``loaded_shard_id`` is ``"down:N"``. ``loaded_weight`` has shape
    ``(hidden, inter)``, the same as one expert's slot in ``param``.
    """
    if loaded_shard_id is None:
        return _ep_packed_weight_loader(expert_start, num_experts, param, loaded_weight)
    local_idx = int(loaded_shard_id.split(":")[1]) - expert_start
    if not 0 <= local_idx < num_local_experts:
        return
    assert loaded_weight.shape == param.shape[1:], (
        f"expert {loaded_shard_id} shape {tuple(loaded_weight.shape)} "
        f"!= {tuple(param.shape[1:])}"
    )
    param.data[local_idx] = loaded_weight


# The EP block's packed expert parameters, as a checkpoint names them.
_PACKED_EXPERTS = ("experts.gate_up_proj", "experts.down_proj")


def ep_expert_slice_spec(model: nn.Module) -> SliceSpec | None:
    """A checkpoint read (``slice_spec``) that takes, of each packed expert
    tensor (a key ending in the EP block's ``experts.gate_up_proj`` or
    ``experts.down_proj``), only this rank's experts' rows; None when
    ``model`` has no :class:`ExpertParallelSparseMoeBlock`. The tensors must be
    packed as :func:`_ep_packed_weight_loader` says: gate_up ``(E, 2 * inter,
    hidden)`` with gate rows before up rows, down ``(E, hidden, inter)``."""
    owned = {
        (m.expert_start, m.num_local_experts)
        for m in model.modules() if isinstance(m, ExpertParallelSparseMoeBlock)
    }
    if not owned:
        return None
    if len(owned) > 1:
        raise ValueError(f"the model's EP blocks own different experts: {sorted(owned)}")
    ((start, count),) = owned

    def spec(key: str) -> TensorSlice | None:
        return TensorSlice(0, start, start + count) if key.endswith(_PACKED_EXPERTS) else None

    return spec


class ExpertParallelSparseMoeBlock(nn.Module):
    """Expert-parallel Top-K sparse MoE.

    Rank ``r`` of ``P`` owns global experts ``[r*E_local, (r+1)*E_local)``
    at full intermediate width, with ``E_local = num_experts / P``. The
    input and the router are replicated, and :func:`_sync_routing` makes
    every rank use rank 0's top-k. Each rank remaps slots routed to other
    ranks' experts to the sentinel id ``E_local``, which the dispatch skips,
    and one all-reduce sums the disjoint partial outputs.

    The signature matches :class:`ParallelSparseMoeBlock`. When
    ``world_size == 1`` the forward is identical to :class:`SparseMoeBlock`.
    The expert loaders take a checkpoint's per-expert tensors or packed
    ``(E, ...)`` ones; :func:`ep_expert_slice_spec` reads only this rank's
    rows of the packed ones.

    ``track_expert_load`` (off by default, CUDA-graph safe) accumulates
    per-expert slot counts in the ``expert_load`` buffer; read and reset it
    with :meth:`pop_expert_load`.

    Options, off by default:

    * ``pre_experts`` and ``post_experts`` run the experts in a latent: the
      router (which must then be given) reads the block's input, the experts
      read ``pre_experts`` of it, and ``post_experts`` maps the all-reduced
      output back. ``hidden_size`` is then the latent's width.
    * ``expert_norm``, an :class:`InputScaledRMSNorm`, normalizes each
      expert's output row before its routing weight applies; the rows are
      weighted and summed in fp32. On CUDA the norm runs fused into that sum,
      reading only this rank's slots' rows.
    * ``chunk_tokens`` caps the tokens of one expert call: a longer step
      runs in near-equal chunks, so the experts' scratch is bounded at any
      prompt length. A stopgap, to be deleted once engine chunked prefill
      (#916) caps every step.
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
        pre_experts: nn.Module | None = None,
        post_experts: nn.Module | None = None,
        expert_norm: InputScaledRMSNorm | None = None,
        chunk_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if pre_experts is not None and router is None:
            raise ValueError(
                "pre_experts needs a router: the default one would read the latent's width "
                "while the router reads the block's input"
            )
        if chunk_tokens is not None and chunk_tokens < 1:
            raise ValueError(f"chunk_tokens must be >= 1, not {chunk_tokens}")
        if expert_norm is not None and not isinstance(expert_norm, InputScaledRMSNorm):
            # the CUDA finalize fuses InputScaledRMSNorm's formula from its weight and eps
            raise ValueError(f"expert_norm must be an InputScaledRMSNorm, not {type(expert_norm).__name__}")
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.pre_experts = pre_experts
        self.post_experts = post_experts
        self.chunk_tokens = chunk_tokens

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
        self.expert_norm = expert_norm
        self._attach_weight_loaders()

    def _attach_weight_loaders(self):
        from functools import partial

        self.experts.gate_up_proj.weight_loader = partial(
            _ep_gate_up_weight_loader, self.expert_start, self.num_local_experts,
            self.num_experts, self.moe_intermediate_size,
        )
        self.experts.down_proj.weight_loader = partial(
            _ep_down_proj_weight_loader, self.expert_start, self.num_local_experts, self.num_experts,
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
        routing_weights, selected_experts, router_states_next = self._route(flat, router_states)
        if self.pre_experts is not None:
            flat = self.pre_experts(flat).contiguous()

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
            out = self._local_experts(flat, routing_weights, selected_experts, skip_invalid=skip)
        else:
            out = self._dispatch_ep(flat, routing_weights, selected_experts, token_valid)
        if self.post_experts is not None:
            out = self.post_experts(out)
        out = out.view(input_shape)
        return (out, router_states_next) if return_router_states else out

    def _route(
        self, flat: torch.Tensor, router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The router's outputs, with rank 0's top-k on every rank when the
        experts are split."""
        routing_weights, selected_experts, router_states_next = self.gate(flat, router_states)
        if self.comm_group.world_size > 1:
            routing_weights, selected_experts = _sync_routing(
                self.comm_group, routing_weights, selected_experts,
            )
        return routing_weights, selected_experts, router_states_next

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
        out = self._local_experts(flat, routing_weights, local)
        self.comm_group.all_reduce(out)
        return out

    def _local_experts(
        self, flat: torch.Tensor, routing_weights: torch.Tensor, local: torch.Tensor,
        skip_invalid: bool = True,
    ) -> torch.Tensor:
        """This rank's share of each token's weighted expert sum; ``local``
        holds other ranks' slots and padding rows as the sentinel
        ``num_local_experts``."""
        if self.chunk_tokens is not None and flat.shape[0] > self.chunk_tokens:
            return self._local_experts_chunked(flat, routing_weights, local, skip_invalid)
        if self.expert_norm is None:
            return _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_local_experts, local, routing_weights, skip_invalid=skip_invalid,
            )
        # each slot's unweighted row, normed, then weighted and summed in fp32
        fused = _HAS_FUSED and flat.is_cuda
        rows = _dispatch(
            flat, self.experts.gate_up_proj, self.experts.down_proj, self.num_local_experts,
            local, torch.ones_like(routing_weights, dtype=flat.dtype), skip_invalid=skip_invalid,
            reduce_results=False, zero_skipped=not fused,
        )
        if fused:
            return moe_input_scaled_rms_sum_reduce_triton(
                rows, local.contiguous(), routing_weights.float().contiguous(),
                self.expert_norm.weight, self.expert_norm.eps, self.num_local_experts,
            )
        return (self.expert_norm(rows).float() * routing_weights[..., None]).sum(1).to(flat.dtype)

    @torch.compiler.disable
    def _local_experts_chunked(
        self, flat: torch.Tensor, routing_weights: torch.Tensor, local: torch.Tensor,
        skip_invalid: bool,
    ) -> torch.Tensor:
        """`_local_experts` over the fewest chunks of at most ``chunk_tokens``
        tokens, of near-equal size. Eager, so the chunk bounds are plain ints of
        the shape: no host sync, and no recompile per prompt length. Every rank
        holds the same tokens, so all run the same chunks, none with a
        collective, and the all-reduce after them stays in lockstep."""
        out = torch.empty_like(flat)
        n = flat.shape[0]
        chunks = -(-n // self.chunk_tokens)
        for i in range(chunks):
            rows = slice(n * i // chunks, n * (i + 1) // chunks)
            out[rows] = self._local_experts(flat[rows], routing_weights[rows], local[rows], skip_invalid)
        return out
