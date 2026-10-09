"""GLM-5.2 fine-grained MoE: groupless sigmoid router + FP8-resident experts."""
from __future__ import annotations

import logging
import os

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
    dispatch_experts_fused,
)
from mstar.model.glm52.components.fp8_linear import Fp8ParallelGatedMLP, dense_fp8_block
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.quantization import FP8_DTYPE, dequantize_fp8_block_weight

logger = logging.getLogger(__name__)

_BACKEND_LOGGED = False
# accepted moe_quant_kernel values; see Glm52ModelConfig.moe_quant_kernel
MOE_QUANT_KERNELS = ("reference", "triton", "auto")
# bf16 x bf16 -> fp32 mm (aten::mm.dtype); older torch lacks the overload
_MM_OUT_DTYPE = hasattr(torch.ops.aten.mm, "dtype")
# the weight scale block fused_experts_fp8 compiles for
_FUSED_FP8_BLOCK = (128, 128)


def fused_fp8_available(device: torch.device, block_size: tuple[int, int]) -> bool:
    """Whether fused_experts_fp8 can serve these experts: CUDA, an importable
    kernel, and the scale block it compiles for."""
    if device.type != "cuda" or tuple(block_size) != _FUSED_FP8_BLOCK:
        return False
    try:
        from mstar.utils.fused_moe import fused_experts_fp8  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - 'auto' serves the reference loop, loudly
        logger.warning("fused fp8 MoE kernel unavailable (%r): the reference dispatch "
                       "serves eager, without CUDA graphs", exc)
        return False
    return True
# Up to this many tokens moe_decode_kernel routes the block through the decode kernels.
_DECODE_MAX_TOKENS = 64


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _gate_up_fp8_loader(
    tp_rank: int, tp_size: int, full_inter: int, row_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """Route one expert's gate/up tensor into the stacked per-rank param."""
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


def _down_fp8_loader(
    tp_rank: int, tp_size: int, full_inter: int, col_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """Column (contraction-dim) sharding twin of :func:`_gate_up_fp8_loader`."""
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


class Glm52MoEGate(nn.Module):
    """DeepSeek-V3 noaux_tc sigmoid router, groupless (n_group=1)."""

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
        # mstar::moe_router_topk once the fused path resolves with moe_router_kernel
        self._topk_op = None
        # fp32 copy of ``weight``, built once by ``finalize_weights``: the
        # forward needs fp32 and the router never changes, so casting it once
        # per layer per step is pure waste. A plain attribute, not a buffer,
        # so model.to(bf16) cannot downcast it.
        self._weight_fp32: torch.Tensor | None = None

    def finalize_weights(self) -> None:
        """Cache the fp32 router weight; call after the weights are loaded."""
        self._weight_fp32 = self.weight.detach().float()

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if self._topk_op is not None and logits.is_cuda:
            return self._topk_op(logits, self.e_score_correction_bias, self.top_k,
                                 self.routed_scaling_factor, self.norm_topk_prob)
        scores = logits.sigmoid()  # (T, E)

        biased = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_ids = torch.topk(biased, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_ids


class Glm52SparseMoeBlock(nn.Module):
    """Routed experts + ungated shared expert (DeepSeek-V3 block shape)."""

    def __init__(
        self, config: Glm52ModelConfig, comm_group: CommGroup | None = None,
        reduce_results: bool = True,
    ) -> None:
        """``reduce_results=False`` returns this rank's partial (routed + shared) for the
        caller to reduce."""
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.reduce_results = reduce_results
        self.tp_size = comm_group.world_size
        self.tp_rank = comm_group.rank
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.moe_intermediate_size = config.moe_intermediate_size
        shard_inter = divide(config.moe_intermediate_size, self.tp_size)

        self.fp8_experts = (
            config.quantization_config is not None and config.moe_fp8_resident
        )
        self.quant_kernel = getattr(config, "moe_quant_kernel", "reference")
        if self.quant_kernel not in MOE_QUANT_KERNELS:
            # an unknown value would otherwise resolve to the uncapturable
            # reference loop and serve eager behind a healthy /health
            raise ValueError(
                f"moe_quant_kernel={self.quant_kernel!r} is not one of "
                f"{MOE_QUANT_KERNELS}"
            )
        # Resolved on the real device by process_weights_after_loading;
        # blocks used without the load hook (CPU tests) stay on reference.
        self._use_fused = False
        self._decode_kernel = bool(getattr(config, "moe_decode_kernel", False))
        self._router_kernel = bool(getattr(config, "moe_router_kernel", False))
        self._prefill_kernel = bool(getattr(config, "moe_prefill_kernel", False))

        self.gate = Glm52MoEGate(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
        )

        self.experts = nn.Module()
        if self.fp8_experts:
            bo, bi = config.quantization_config.weight_block_size
            self.block_size = (bo, bi)
            hidden, E = config.hidden_size, config.n_routed_experts
            assert shard_inter % bo == 0 and shard_inter % bi == 0, (
                f"per-rank intermediate {shard_inter} must be a multiple of the "
                f"fp8 scale block {self.block_size} for clean TP slicing (full "
                f"model: 2048/8=256 per rank, 2 blocks of 128)"
            )
            # fp8 bytes in uint8 containers; e4m3 view happens at dispatch.
            self.experts.gate_up_proj_fp8 = nn.Parameter(
                torch.empty(E, 2 * shard_inter, hidden, dtype=torch.uint8),
                requires_grad=False,
            )
            self.experts.gate_up_proj_scale_inv = nn.Parameter(
                torch.empty(
                    E, 2 * (shard_inter // bo), _ceil_div(hidden, bi),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.experts.down_proj_fp8 = nn.Parameter(
                torch.empty(E, hidden, shard_inter, dtype=torch.uint8),
                requires_grad=False,
            )
            self.experts.down_proj_scale_inv = nn.Parameter(
                torch.empty(
                    E, _ceil_div(hidden, bo), shard_inter // bi,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            self.experts.gate_up_proj = nn.Parameter(
                torch.empty(
                    config.n_routed_experts, 2 * shard_inter, config.hidden_size,
                )
            )
            self.experts.down_proj = nn.Parameter(
                torch.empty(
                    config.n_routed_experts, config.hidden_size, shard_inter,
                )
            )
        self._attach_expert_weight_loaders()

        # One all-reduce per block instead of two: the shared expert's
        # per-rank partial is summed with the routed partial and reduced once.
        # Off by default — sum-then-reduce rounds differently from
        # reduce-then-sum in bf16, so the emitted tokens can shift at ties.
        # model_kwargs.moe_fused_allreduce turns it on; a set
        # MSTAR_GLM52_MOE_FUSED_ALLREDUCE overrides it either way.
        env = os.environ.get("MSTAR_GLM52_MOE_FUSED_ALLREDUCE", "")
        self._fused_allreduce = self.tp_size > 1 and (
            env == "1" if env else config.moe_fused_allreduce
        )
        # the two partials are summed before any reduce: one here, or none
        self._sum_partials = self.tp_size > 1 and (self._fused_allreduce or not reduce_results)
        shared_inter = config.moe_intermediate_size * config.n_shared_experts
        dense_block = dense_fp8_block(config)
        if dense_block is not None and config.fp8_shared_expert:
            self.shared_expert = Fp8ParallelGatedMLP(
                config.hidden_size, shared_inter, dense_block, comm_group=comm_group,
                reduce_results=not self._sum_partials)
        else:
            self.shared_expert = ParallelGatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_inter,
                comm_group=comm_group,
                activation=config.hidden_act,
                bias=False,
                reduce_results=not self._sum_partials,
            )

    def _attach_expert_weight_loaders(self) -> None:
        """Reattach per-shard loaders after ``_apply`` rebuilds parameters."""
        from functools import partial

        full_inter = self.moe_intermediate_size
        if self.fp8_experts:
            bo, bi = self.block_size
            self.experts.gate_up_proj_fp8.weight_loader = partial(
                _gate_up_fp8_loader, self.tp_rank, self.tp_size, full_inter, 1,
            )
            self.experts.gate_up_proj_scale_inv.weight_loader = partial(
                _gate_up_fp8_loader, self.tp_rank, self.tp_size, full_inter, bo,
            )
            self.experts.down_proj_fp8.weight_loader = partial(
                _down_fp8_loader, self.tp_rank, self.tp_size, full_inter, 1,
            )
            self.experts.down_proj_scale_inv.weight_loader = partial(
                _down_fp8_loader, self.tp_rank, self.tp_size, full_inter, bi,
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

        if self._use_decode(flat):
            # routed + shared partials in one tensor, so one reduction
            out = self._forward_decode(flat)
            if self.tp_size > 1 and self.reduce_results:
                self.comm_group.all_reduce(out)
            return out.view(input_shape)

        # topk_weights stay fp32 into the combine, as in HF's experts.
        topk_weights, topk_ids = self.gate(flat)
        if self.fp8_experts:
            if self._use_fused and self._prefill_kernel and flat.shape[0] > _DECODE_MAX_TOKENS:
                routed = self._dispatch_prefill(flat, topk_weights, topk_ids)
                if self.tp_size > 1 and not self._sum_partials:
                    self.comm_group.all_reduce(routed)
            elif self._use_fused:
                from mstar.utils.fused_moe import fused_experts_fp8

                routed = fused_experts_fp8(
                    flat,
                    self.experts.gate_up_proj_fp8,
                    self.experts.down_proj_fp8,
                    self.experts.gate_up_proj_scale_inv,
                    self.experts.down_proj_scale_inv,
                    topk_weights, topk_ids,
                    block_size=self.block_size,
                )
                if self.tp_size > 1 and not self._sum_partials:
                    self.comm_group.all_reduce(routed)
            else:
                routed = self._dispatch_fp8_reference(
                    flat, topk_weights, topk_ids,
                    reduce=not self._sum_partials)
        elif self.tp_size == 1:
            routed = _dispatch(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                self.num_experts,
                topk_ids,
                topk_weights,
            )
        else:
            routed = dispatch_experts_fused(
                flat,
                self.experts.gate_up_proj,
                self.experts.down_proj,
                self.num_experts,
                topk_ids,
                topk_weights,
            )
            if not self._sum_partials:
                self.comm_group.all_reduce(routed)
        shared = self.shared_expert(flat)
        out = routed + shared
        if self._sum_partials and self.reduce_results:
            # Both terms are per-rank partials here; one reduce for the sum.
            self.comm_group.all_reduce(out)
        return out.view(input_shape)

    def _use_decode(self, flat: torch.Tensor) -> bool:
        """Small batches take the decode kernels when moe_decode_kernel is set on the
        fused path; they need bf16 activations and the shared expert in the routed experts'
        shape and dtype."""
        if not (self._decode_kernel and self._use_fused
                and flat.shape[0] <= _DECODE_MAX_TOKENS):
            return False
        shared = self.shared_expert.down_proj.weight
        return (
            flat.is_cuda
            and shared.dtype == flat.dtype == torch.bfloat16
            and shared.shape[1] == self.experts.down_proj_fp8.shape[2]
        )

    def _dispatch_decode(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Routed + shared experts in the fused_moe decode kernels, launched unchained."""
        from mstar.utils.fused_moe import decode as moe_decode

        exp, shared = self.experts, self.shared_expert
        return moe_decode.experts(
            flat, exp.gate_up_proj_fp8, exp.gate_up_proj_scale_inv,
            exp.down_proj_fp8, exp.down_proj_scale_inv,
            shared.gate_up_proj.weight, shared.down_proj.weight, topk_weights, topk_ids,
            block_size=self.block_size)

    # Triton launches with PDL: dynamo stays out, the kernels run inside the captured graph.
    @torch.compiler.disable
    def _forward_decode(self, flat: torch.Tensor) -> torch.Tensor:
        """Router, routed and shared experts as one fused_moe decode launch chain; returns
        this rank's partial (routed + shared), unreduced."""
        from mstar.utils.fused_moe import decode as moe_decode

        gate, exp, shared = self.gate, self.experts, self.shared_expert
        return moe_decode.forward(
            flat, gate.weight, gate.e_score_correction_bias,
            exp.gate_up_proj_fp8, exp.gate_up_proj_scale_inv,
            exp.down_proj_fp8, exp.down_proj_scale_inv,
            shared.gate_up_proj.weight, shared.down_proj.weight,
            top_k=gate.top_k, scale=gate.routed_scaling_factor, normalize=gate.norm_topk_prob,
            block_size=self.block_size)

    # Triton launches stay outside dynamo, as the runner's fp8 quant does.
    @torch.compiler.disable
    def _dispatch_prefill(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """fused_experts_fp8's output from the fused_moe prefill kernels (same bits)."""
        from mstar.utils.fused_moe import prefill as moe_prefill

        exp = self.experts
        return moe_prefill.experts(
            flat, exp.gate_up_proj_fp8, exp.down_proj_fp8, exp.gate_up_proj_scale_inv,
            exp.down_proj_scale_inv, topk_weights, topk_ids, block_size=self.block_size,
            swiglu_limit=None)

    def _dispatch_fp8_reference(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Per-expert loop that dequantizes only the experts this batch hit."""
        final = torch.zeros_like(flat)

        with torch.no_grad():
            expert_mask = F.one_hot(topk_ids, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            e = expert_idx_t[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            tokens = flat[token_idx]

            gate_up_w = dequantize_fp8_block_weight(
                self.experts.gate_up_proj_fp8[e],
                self.experts.gate_up_proj_scale_inv[e],
                block_size=self.block_size, out_dtype=flat.dtype,
            )
            down_w = dequantize_fp8_block_weight(
                self.experts.down_proj_fp8[e],
                self.experts.down_proj_scale_inv[e],
                block_size=self.block_size, out_dtype=flat.dtype,
            )
            gate, up = torch.mm(tokens, gate_up_w.T).chunk(2, dim=-1)
            out = torch.mm(F.silu(gate) * up, down_w.T)
            out = out * topk_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        if reduce and self.tp_size > 1:
            self.comm_group.all_reduce(final)
        return final

    def process_weights_after_loading(self, device) -> None:
        """Resolve reference-vs-fused dispatch on the real device: explicit
        "triton" must not silently downgrade, "auto" probes for the fused
        path, "reference" keeps the bitwise loop."""
        self.gate.finalize_weights()
        if not self.fp8_experts:
            return
        dev = torch.device(device) if device is not None else torch.device("cpu")
        kernel = self.quant_kernel
        fused_ok = fused_fp8_available(dev, self.block_size)
        if kernel == "triton" and not fused_ok:
            raise RuntimeError(
                "moe_quant_kernel='triton' requested but the fused fp8 path "
                f"is unavailable (needs CUDA, triton and {_FUSED_FP8_BLOCK} "
                f"scale blocks; these are {tuple(self.block_size)}). Use "
                "'auto' to fall back to the reference dispatch."
            )
        self._use_fused = kernel == "triton" or (kernel == "auto" and fused_ok)
        if self._use_fused and self._router_kernel:
            from mstar.utils.fused_moe import decode as moe_decode  # registers the op

            self.gate._topk_op = moe_decode.router_topk
        global _BACKEND_LOGGED
        if not _BACKEND_LOGGED:
            logger.info(
                "Glm52SparseMoeBlock routed-expert backend: %s "
                "(moe_quant_kernel=%s, moe_decode_kernel=%s, block_size=%s, "
                "tp_size=%d).",
                "fused_experts_fp8 W8A8" if self._use_fused
                else "fp8-resident reference dispatch",
                kernel, self._decode_kernel and self._use_fused, self.block_size,
                self.tp_size,
            )
            _BACKEND_LOGGED = True
