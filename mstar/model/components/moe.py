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

When triton is installed and inputs are on CUDA, dispatch goes through
the Triton fused-MoE kernel in :mod:`mstar.utils.fused_moe`; otherwise it
falls back to the naive per-expert loop in :func:`dispatch_experts_fused`.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide

logger = logging.getLogger(__name__)


# Optional fused Triton MoE path. Imports succeed wherever triton is
# installed; the alignment op JIT-builds a vendored CUDA kernel on first use
# (or falls back to a torch impl), so no sgl-kernel/vllm dep is needed. The
# actual kernels only run on CUDA -- guarded by the ``is_cuda`` check below.
try:
    from mstar.utils.fused_moe import fused_experts as _fused_experts

    _HAS_FUSED = True
except Exception as e:  # pragma: no cover -- exercised only when triton missing
    _fused_experts = None
    _HAS_FUSED = False
    logger.warning(f"Could not load fused MoE kernel: {e}")

# Fused softmax+topk router kernel (sgl_kernel). Replaces the
# softmax -> torch.topk -> renorm chain (three separate kernel launches,
# one a bitonic sort) with a single kernel. Bit-identical expert ids,
# weights equal to float rounding, so it's used whenever available; falls
# back to the torch ops below when sgl_kernel isn't importable.
try:
    from sgl_kernel import topk_softmax as _topk_softmax
except Exception:  # pragma: no cover
    _topk_softmax = None

_FUSED_TOPK = _topk_softmax is not None


class TopKRouter(nn.Module):
    """Softmax top-k router shared by all MoE blocks.

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
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            router_logits: ``(tokens, num_experts)`` softmax distribution,
                or ``None`` on the fused kernel path (see below) -- no
                caller uses this value in that case, so it isn't
                materialized.
            routing_weights: ``(tokens, top_k)`` top-k probabilities
                (optionally renormalized).
            selected_experts: ``(tokens, top_k)`` indices, dtype depends on
                the path taken: the fused sgl_kernel path (``_FUSED_TOPK``
                available and CUDA input) returns int32 and ``None`` for
                ``router_logits``; the torch fallback returns int64
                alongside the full softmax distribution.
        """
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight)

        if _FUSED_TOPK and _HAS_FUSED and router_logits.is_cuda:
            # Single-kernel softmax+topk(+renorm). All callers discard the
            # first return value, so the full softmax distribution is not
            # materialized on this path.
            m = router_logits.shape[0]
            routing_weights = torch.empty(
                m, self.top_k, device=router_logits.device, dtype=torch.float32,
            )
            router_indices = torch.empty(
                m, self.top_k, device=router_logits.device, dtype=torch.int32,
            )
            _topk_softmax(
                routing_weights, router_indices,
                router_logits.float(), self.norm_topk_prob,
            )
            return None, routing_weights, router_indices

        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)

        routing_weights = router_top_value.to(router_logits.dtype)
        return router_logits, routing_weights, router_indices


def dispatch_experts_fused(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
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
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        # one-hot mask over expert dim: (num_experts, top_k, tokens)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
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


def _dispatch(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Pick fused-Triton if available, otherwise the naive loop."""
    if _HAS_FUSED and hidden_states.is_cuda:
        return _fused_experts(
            hidden_states, gate_up_proj, down_proj,
            routing_weights, selected_experts,
        )
    return dispatch_experts_fused(
        hidden_states, gate_up_proj, down_proj,
        num_experts, selected_experts, routing_weights,
    )


# --------------------------------------------------------------------------
# Optional block-fp8 (w8a8) expert dispatch. At decode the routed-expert
# GEMMs are memory-bound on expert weight reads, so halving weight bytes
# (bf16 -> fp8_e4m3 with per-(128,128)-block scales) roughly halves the MoE
# slice of the step. Weights are quantized lazily on first forward (before
# CUDA graph capture, which happens after warmup) and the bf16 originals are
# freed to reclaim VRAM.
# --------------------------------------------------------------------------
def _moe_fp8_flag(env_name: str) -> bool:
    return _HAS_FUSED and os.environ.get(env_name, "0") == "1"


def _ensure_fp8_experts(experts: nn.Module):
    cached = getattr(experts, "_fp8_cache", None)
    if cached is not None:
        return cached
    from mstar.utils.fused_moe.fp8 import per_block_cast_to_fp8_weight

    w1, s1 = per_block_cast_to_fp8_weight(experts.gate_up_proj.data)
    w2, s2 = per_block_cast_to_fp8_weight(experts.down_proj.data)
    # Free the bf16 originals: every later forward uses only the fp8 copies.
    experts.gate_up_proj.data = experts.gate_up_proj.data.new_empty(0)
    experts.down_proj.data = experts.down_proj.data.new_empty(0)
    experts._fp8_cache = (w1, s1, w2, s2)
    logger.info(
        "MoE experts quantized to block-fp8 w8a8: w1 %s, w2 %s",
        tuple(w1.shape), tuple(w2.shape),
    )
    return experts._fp8_cache


@torch.compiler.disable
def _dispatch_fp8(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    reduce_results: bool = True,
) -> torch.Tensor:
    # compiler.disable (same pattern as FlashInferDecodeWrapper.run): the
    # lazy quantization mutates module state (frees the bf16 params), which
    # dynamo must not trace — re-tracing a later bucket otherwise sees the
    # freed size-0 param and inductor fails the capture.
    from mstar.utils.fused_moe.fp8 import fused_experts_fp8

    w1, s1, w2, s2 = _ensure_fp8_experts(experts)
    return fused_experts_fp8(
        hidden_states, w1, s1, w2, s2,
        routing_weights, selected_experts,
        reduce_results=reduce_results,
    )


def _dispatch_fp8_maybe_op(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    reduce_results: bool = True,
) -> torch.Tensor:
    """fp8 expert dispatch, custom-op path when available.

    When MSTAR_CUSTOM_OPS is on AND the experts were pre-quantized before
    compile (prequantize_fp8_experts), read the cached fp8 weights and go
    through the mstar::fused_experts_fp8 op -- an opaque in-graph node, no
    graph break. Both operands of the guard are compile-time constants at trace
    (env flag + a populated module attribute), so dynamo folds to the op path
    and never traces the mutating legacy fallback. Otherwise fall back to the
    @torch.compiler.disable legacy dispatch (unchanged default behavior).
    """
    from mstar.engine.compile_ops import custom_ops_enabled

    if custom_ops_enabled() and getattr(experts, "_fp8_cache", None) is not None:
        w1, s1, w2, s2 = experts._fp8_cache
        return torch.ops.mstar.fused_experts_fp8(
            hidden_states, w1, s1, w2, s2,
            routing_weights, selected_experts, reduce_results,
        )
    return _dispatch_fp8(
        experts, hidden_states, selected_experts, routing_weights,
        reduce_results=reduce_results,
    )


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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = TopKRouter(
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        _, routing_weights, selected_experts = self.gate(flat)
        out = _dispatch(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, selected_experts, routing_weights,
        )
        return out.view(input_shape)


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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = TopKRouter(
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()

        shared = self.shared_expert(flat)

        _, routing_weights, selected_experts = self.gate(flat)
        routed = _dispatch(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, selected_experts, routing_weights,
        )
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (routed + shared_gate * shared).view(input_shape)


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

        self.gate = TopKRouter(
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        _, routing_weights, selected_experts = self.gate(flat)

        if self.comm_group.world_size == 1:
            if _moe_fp8_flag("MSTAR_MOE_FP8") and flat.is_cuda:
                out = _dispatch_fp8_maybe_op(
                    self.experts, flat, selected_experts, routing_weights,
                )
            else:
                out = _dispatch(
                    flat, self.experts.gate_up_proj, self.experts.down_proj,
                    self.num_experts, selected_experts, routing_weights,
                )
        else:
            out = self._dispatch_tp(flat, routing_weights, selected_experts)
        return out.view(input_shape)

    def _dispatch_tp(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        # (tokens, top_k, hidden) — partial results before reduce
        if _moe_fp8_flag("MSTAR_MOE_FP8") and flat.is_cuda:
            cache3 = _dispatch_fp8_maybe_op(
                self.experts, flat, selected_experts, routing_weights,
                reduce_results=False,
            )
        else:
            cache3 = fused_experts(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                routing_weights, selected_experts, reduce_results=False,
            )
        self.comm_group.all_reduce(cache3)
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
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

        self.gate = TopKRouter(
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()

        shared = self.shared_expert(flat)

        _, routing_weights, selected_experts = self.gate(flat)
        if self.comm_group.world_size == 1:
            if _moe_fp8_flag("MSTAR_MOE_FP8_TALKER") and flat.is_cuda:
                routed = _dispatch_fp8_maybe_op(
                    self.experts, flat, selected_experts, routing_weights,
                )
            else:
                routed = _dispatch(
                    flat, self.experts.gate_up_proj, self.experts.down_proj,
                    self.num_experts, selected_experts, routing_weights,
                )
        else:
            routed = self._dispatch_tp(flat, routing_weights, selected_experts)
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (routed + shared_gate * shared).view(input_shape)

    def _dispatch_tp(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        if _moe_fp8_flag("MSTAR_MOE_FP8_TALKER") and flat.is_cuda:
            cache3 = _dispatch_fp8_maybe_op(
                self.experts, flat, selected_experts, routing_weights,
                reduce_results=False,
            )
        else:
            cache3 = fused_experts(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                routing_weights, selected_experts, reduce_results=False,
            )
        self.comm_group.all_reduce(cache3)
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
        return output


def prequantize_fp8_experts(module: nn.Module) -> int:
    """Eagerly fp8-quantize the experts of every fp8-eligible MoE block in
    ``module``, BEFORE torch.compile traces it.

    The lazy quant in :func:`_ensure_fp8_experts` mutates module state (frees
    the bf16 params) and so cannot run inside a traced/compiled forward -- which
    is exactly why the legacy fp8 dispatch is ``@torch.compiler.disable`` (a
    graph break). Running the quant here, before compile, leaves the fp8 weights
    cached so the forward reads them as plain tensors and calls the
    mstar::fused_experts_fp8 op with no break. Idempotent (skips already-cached),
    so it is safe to call once per submodule/slot capture. Returns the number of
    blocks quantized this call. Only meaningful under MSTAR_CUSTOM_OPS.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, ParallelSparseMoeBlock) and _moe_fp8_flag("MSTAR_MOE_FP8"):
            _ensure_fp8_experts(m.experts)
            n += 1
        elif isinstance(m, ParallelSparseMoeBlockWithSharedExpert) and _moe_fp8_flag(
            "MSTAR_MOE_FP8_TALKER"
        ):
            _ensure_fp8_experts(m.experts)
            n += 1
    return n
