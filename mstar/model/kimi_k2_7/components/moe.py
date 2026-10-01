"""Kimi-K2.7 fine-grained MoE: sigmoid router plus ungated shared expert."""
from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)

_BACKEND_LOGGED = False

# Tokens per slice when running routed dispatch over a large prefill, to bound
# the MoE working set (partial expert output, all-reduce buffer, Marlin
# internals) instead of letting it scale with the full prompt length.
MOE_PREFILL_SLICE = 8192


def _gate_up_packed_loader(
    tp_rank: int, tp_size: int, full_inter: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
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


def _down_packed_loader(
    tp_rank: int, tp_size: int, full_inter: int, divisor: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    assert loaded_shard_id is not None
    expert_idx = int(str(loaded_shard_id).split(":")[1])
    shard_inter = divide(full_inter, tp_size)
    span = divide(shard_inter, divisor)
    start = tp_rank * span
    param.data[expert_idx, :, :] = loaded_weight[:, start:start + span]


# Optional fused Triton kernel for the n_group == 1 router fast path (the
# Kimi-K2.7 config). KimiMoEGate.forward falls back to the general torch
# path below when triton isn't available or the scores aren't on CUDA.
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _kimi_gate_fused_kernel(
        scores_ptr, bias_ptr, weight_ptr, id_ptr,
        E,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        NORM_TOPK: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """One program per token row: bias-add, top-K by selection score with
        a lowest-index tie-break, gather the original scores as weights.

        K iterations of max + lowest-index argmin over the BLOCK_E-wide row
        replace the topk-of-2-per-group/scatter/expand/masked_fill/expert-topk
        chain that n_group == 1 reduces to an identity mask over.
        """
        row = tl.program_id(0)
        offs_e = tl.arange(0, BLOCK_E)
        mask = offs_e < E
        scores = tl.load(scores_ptr + row * E + offs_e, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_e, mask=mask, other=0.0).to(tl.float32)
            remaining = scores + bias  # padding stays -inf: -inf + 0.0 == -inf
        else:
            remaining = scores

        # K is a power of two (checked by the launcher): offs_k indexes the
        # output slots as a vector so the loop only ever writes through
        # tl.where, never a Python list/index -- required for real compilation.
        offs_k = tl.arange(0, K)
        w_vec = tl.zeros([K], dtype=tl.float32)
        id_vec = tl.zeros([K], dtype=tl.int32)
        for k in tl.static_range(K):
            best_val = tl.max(remaining, axis=0)
            is_best = (remaining == best_val) & mask
            best_idx = tl.min(tl.where(is_best, offs_e, BLOCK_E), axis=0)
            is_winner = offs_e == best_idx
            orig = tl.sum(tl.where(is_winner, scores, 0.0), axis=0)
            w_vec = tl.where(offs_k == k, orig, w_vec)
            id_vec = tl.where(offs_k == k, best_idx, id_vec)
            remaining = tl.where(is_winner, float("-inf"), remaining)

        if NORM_TOPK:
            w_vec = w_vec / tl.sum(w_vec, axis=0)
        if SCALE != 1.0:
            w_vec = w_vec * SCALE
        tl.store(weight_ptr + row * K + offs_k, w_vec)
        tl.store(id_ptr + row * K + offs_k, id_vec.to(tl.int64))

    _HAS_FUSED_GATE = True
except Exception as e:  # pragma: no cover -- exercised only when triton missing
    _kimi_gate_fused_kernel = None
    _HAS_FUSED_GATE = False
    logger.warning(f"Could not load Triton for the fused Kimi router: {e}")


def _fused_group1_topk(
    scores: torch.Tensor,
    bias: torch.Tensor | None,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """n_group == 1 fast path: the group stage is a no-op (every expert is
    eligible), so go straight from scores to the fused top-k kernel."""
    assert top_k & (top_k - 1) == 0, f"_fused_group1_topk requires a power-of-two top_k, got {top_k}"
    num_tokens, num_experts = scores.shape
    scores = scores.contiguous()
    topk_weights = torch.empty(num_tokens, top_k, dtype=torch.float32, device=scores.device)
    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int64, device=scores.device)
    bias_arg = bias if bias is not None else scores  # unread stand-in when HAS_BIAS=False
    _kimi_gate_fused_kernel[(num_tokens,)](
        scores, bias_arg, topk_weights, topk_ids,
        num_experts,
        K=top_k,
        BLOCK_E=triton.next_power_of_2(num_experts),
        HAS_BIAS=bias is not None,
        NORM_TOPK=norm_topk_prob,
        SCALE=routed_scaling_factor,
    )
    return topk_weights, topk_ids


class KimiMoEGate(nn.Module):
    """DeepSeek-V3 group-limited sigmoid router with selection-only bias."""

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
        h = hidden_states.reshape(-1, self.hidden_size).float()
        gating = F.linear(h, self.weight.float())  # (T, E)

        if self.scoring_func == "sigmoid":
            scores = gating.sigmoid()
        elif self.scoring_func == "softmax":
            scores = gating.softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func!r}")

        if (
            self.n_group == 1
            and _HAS_FUSED_GATE
            and scores.is_cuda
            and self.top_k & (self.top_k - 1) == 0
        ):
            return _fused_group1_topk(
                scores, self.e_score_correction_bias, self.top_k,
                self.norm_topk_prob, self.routed_scaling_factor,
            )

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
    """DeepSeek-V3 MoE block: routed experts + ungated shared expert."""

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
        shard_inter = divide(config.moe_intermediate_size, self.tp_size)

        self.packed_experts = (
            config.quantization_config is not None and config.moe_in_kernel_dequant
        )
        # Hold the checkpoint descriptor rather than copying its fields out: it is
        # the single source for group_size / pack_factor / symmetric, and it builds
        # the per-dispatch QuantizationData.
        self.quant_config = config.quantization_config if self.packed_experts else None
        self.quant_kernel = getattr(config, "quant_kernel", "auto")
        self._marlin_method = None
        self._use_marlin = False
        self.moe_prefill_slice = MOE_PREFILL_SLICE

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
        if self.packed_experts:
            qc = self.quant_config
            qc.ensure_kernel_support()
            hidden, gs, pf = config.hidden_size, qc.group_size, qc.pack_factor
            assert hidden % pf == 0 and hidden % gs == 0, (
                f"hidden {hidden} must divide pack_factor {pf} and group_size {gs}"
            )
            assert shard_inter % pf == 0 and shard_inter % gs == 0, (
                f"shard_inter {shard_inter} must divide pack_factor {pf} and group_size {gs}"
            )
            E = config.n_routed_experts
            self.experts.gate_up_proj_packed = nn.Parameter(
                torch.empty(E, 2 * shard_inter, hidden // pf, dtype=torch.int32),
                requires_grad=False,
            )
            self.experts.gate_up_proj_scale = nn.Parameter(
                torch.empty(E, 2 * shard_inter, hidden // gs, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.experts.down_proj_packed = nn.Parameter(
                torch.empty(E, hidden, shard_inter // pf, dtype=torch.int32),
                requires_grad=False,
            )
            self.experts.down_proj_scale = nn.Parameter(
                torch.empty(E, hidden, shard_inter // gs, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
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
        self._attach_expert_weight_loaders()

        self.shared_expert = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            comm_group=comm_group,
            activation=config.hidden_act,
            bias=False,
            # Rank-local partial: forward() sums it with the routed partial
            # and all-reduces once, instead of this MLP reducing on its own.
            reduce_results=False,
        )

    def _attach_expert_weight_loaders(self) -> None:
        """Reattach per-shard loaders after ``_apply`` rebuilds parameters."""
        from functools import partial

        full_inter = self.moe_intermediate_size
        if self.packed_experts:
            pf, gs = self.quant_config.pack_factor, self.quant_config.group_size
            self.experts.gate_up_proj_packed.weight_loader = partial(
                _gate_up_packed_loader, self.tp_rank, self.tp_size, full_inter,
            )
            self.experts.gate_up_proj_scale.weight_loader = partial(
                _gate_up_packed_loader, self.tp_rank, self.tp_size, full_inter,
            )
            self.experts.down_proj_packed.weight_loader = partial(
                _down_packed_loader, self.tp_rank, self.tp_size, full_inter, pf,
            )
            self.experts.down_proj_scale.weight_loader = partial(
                _down_packed_loader, self.tp_rank, self.tp_size, full_inter, gs,
            )
        else:
            self.experts.gate_up_proj.weight_loader = partial(
                _gate_up_weight_loader, self.tp_rank, self.tp_size, full_inter,
            )
            self.experts.down_proj.weight_loader = partial(
                _down_proj_weight_loader, self.tp_rank, self.tp_size, full_inter,
            )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_expert_weight_loaders()
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size).contiguous()

        topk_weights, topk_ids = self.gate(flat)
        num_tokens = flat.shape[0]
        if num_tokens <= self.moe_prefill_slice:
            routed = self._route(flat, topk_weights, topk_ids)
        else:
            # Bound the MoE working set for long prefills: the naive one-shot
            # dispatch allocates partial expert output, all-reduce buffers, and
            # Marlin internals that scale with the full token count.
            routed = torch.empty_like(flat)
            for start in range(0, num_tokens, self.moe_prefill_slice):
                end = min(start + self.moe_prefill_slice, num_tokens)
                routed[start:end] = self._route(
                    flat[start:end], topk_weights[start:end], topk_ids[start:end]
                )
        shared = self.shared_expert(flat)
        out = routed + shared
        if self.tp_size > 1:
            # routed and shared are both rank-local partials now; one
            # all-reduce here replaces the three the block used to issue.
            out = self.comm_group.all_reduce(out)
        return out.view(input_shape)

    def _route(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_marlin:
            # Marlin's GEMM takes fp32 combine weights — pass them BEFORE the bf16
            # cast the other paths need. Otherwise identical TP story.
            return self._dispatch_marlin(flat, topk_weights, topk_ids)
        topk_weights = topk_weights.to(flat.dtype)
        if self.packed_experts:
            return self._dispatch_packed_experts(flat, topk_weights, topk_ids)
        if self.tp_size == 1:
            return _dispatch(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                self.num_experts,
                topk_ids,
                topk_weights,
            )
        return self._dispatch_tp(flat, topk_weights, topk_ids)

    def _dispatch_packed_experts(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        reduce = self.tp_size == 1
        # Built per dispatch, not cached: Module._apply rebuilds these Parameters
        # on .to(device), so a cached descriptor could hold a stale tensor.
        quant = self.quant_config.moe_quant_data(
            self.experts.gate_up_proj_scale, self.experts.down_proj_scale,
        )
        out = fused_experts(
            flat,
            self.experts.gate_up_proj_packed,
            self.experts.down_proj_packed,
            topk_weights,
            topk_ids,
            quant=quant,
            reduce_results=reduce,
        )
        if reduce:
            return out
        # Rank-local partial: sum over top-k here, leave the cross-rank
        # reduce to forward()'s single all-reduce.
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(out, output, routed_scaling_factor=1.0)
        return output

    def process_weights_after_loading(self, device) -> None:
        """Resolve Triton-vs-Marlin after weights land on the real device."""
        if not self.packed_experts:
            return
        from mstar.model.components.quantization import MarlinMoEMethod
        from mstar.utils.marlin import is_marlin_available

        qc = self.quant_config
        dev = torch.device(device)
        shard_inter = divide(self.moe_intermediate_size, self.tp_size)
        legal_shapes = MarlinMoEMethod.shapes_are_legal(
            self.hidden_size, shard_inter, qc.group_size
        )
        eligible = (
            self.quant_kernel != "triton"
            and dev.type == "cuda"
            and torch.cuda.get_device_capability(dev) >= (8, 0)
            and qc.symmetric
            and legal_shapes
            and is_marlin_available()
        )
        if self.quant_kernel == "marlin" and not eligible:
            raise RuntimeError(
                "quant_kernel='marlin' requested but Marlin is ineligible "
                f"(needs CUDA sm80+, symmetric INT4, legal shapes: hidden="
                f"{self.hidden_size}, shard_inter={shard_inter}, "
                f"group_size={qc.group_size}, legal={legal_shapes}). "
                "Use quant_kernel='auto' to fall back to the Triton path."
            )
        global _BACKEND_LOGGED
        if not eligible:
            if not _BACKEND_LOGGED:
                logger.info(
                    "KimiSparseMoeBlock routed-expert backend: Triton W4A16 "
                    "(quant_kernel=%s, marlin ineligible: legal_shapes=%s).",
                    self.quant_kernel, legal_shapes,
                )
                _BACKEND_LOGGED = True
            return

        if not _BACKEND_LOGGED:
            logger.info(
                "KimiSparseMoeBlock routed-expert backend: Marlin W4A16 "
                "(quant_kernel=%s, group_size=%d, tp_size=%d).",
                self.quant_kernel, qc.group_size, self.tp_size,
            )
            _BACKEND_LOGGED = True

        method = MarlinMoEMethod.from_quant_config(qc)
        method.prepare(
            self.experts.gate_up_proj_packed.data,
            self.experts.gate_up_proj_scale.data,
            self.experts.down_proj_packed.data,
            self.experts.down_proj_scale.data,
            dev,
        )
        for name in (
            "gate_up_proj_packed", "gate_up_proj_scale",
            "down_proj_packed", "down_proj_scale",
        ):
            p = getattr(self.experts, name)
            p.data = torch.empty(0, dtype=p.dtype, device=dev)
        self._marlin_method = method
        self._use_marlin = True

    def _dispatch_marlin(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import moe_sum_reduce_triton

        reduce = self.tp_size == 1
        out = self._marlin_method.apply(
            flat, topk_weights, topk_ids, reduce_results=reduce
        )
        if reduce:
            return out
        # Rank-local partial: sum over top-k here, leave the cross-rank
        # reduce to forward()'s single all-reduce.
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(out, output, routed_scaling_factor=1.0)
        return output

    def _dispatch_tp(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        cache3 = fused_experts(
            flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            topk_weights,
            topk_ids,
            reduce_results=False,
        )
        # Rank-local partial: sum over top-k here, leave the cross-rank
        # reduce to forward()'s single all-reduce.
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
        return output
