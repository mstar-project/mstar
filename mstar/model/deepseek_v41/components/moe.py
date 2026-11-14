"""DeepSeek-V4.1's MoE: the ``noaux_tc`` router on sqrt-softplus scores, routed experts and one
shared expert, with the reference's rounding.

Every expert GEMM reads its input rounded to FP8 in 32-wide groups, and the routing weight
scales an expert's activation before the down projection rounds it. ``_dispatch`` is that
arithmetic expert by expert, the numerics anchor a kernel is held to. ``_grouped`` is the same
arithmetic with every (token, expert) pair a row of one grouped GEMM launch over the ``fp4``
bytes (``moe_kernel``): it differs from ``_dispatch`` only in the order a GEMM sums in fp32,
and its shapes and launches follow from the step's row count alone.

The routed experts are held as ``moe_experts`` says, the same values each way:

- ``fp4``: the checkpoint's bytes, MXFP4. ``gate_up_proj_packed (E, 2 * inter, hidden / 2)`` and
  ``down_proj_packed (E, hidden, inter / 2)`` hold two E2M1 codes a byte along the inputs (the
  even input in the low nibble), gate rows before up rows; ``gate_up_proj_scale (E, 2 * inter,
  hidden / 32)`` and ``down_proj_scale (E, hidden, inter / 32)`` hold one E8M0 byte per row and
  32 inputs. A hit expert is dequantized for the step.
- ``fp8``: their upcast to E4M3 32 x 32 blocks, in ``fused_moe``'s layout.
- ``dense``: the model's dtype.

A TP group gives each rank a slice of every expert's intermediate dims, or under the node
group's ``ep_config: {parallel: ep}`` whole routed experts, an even share of them, of which a
rank runs its own. Either way one all-reduce sums the ranks; the shared expert stays sliced.
"""
from __future__ import annotations

import logging
from functools import lru_cache, partial

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.moe import NoAuxTcRouter, _sync_routing
from mstar.model.components.moe_parallel import MoeParallelConfig
from mstar.model.deepseek_v41.components.linear import fp32_linear
from mstar.model.deepseek_v41.config import DeepseekV41Config
from mstar.utils.fp8_block import E8M0_DTYPE, dequantize_fp8_block_weight
from mstar.utils.mx_round import dequant_row_fp4, round_fp8

logger = logging.getLogger(__name__)

# MXFP4's scale group: one E8M0 scale per 32 inputs of a row
FP4_GROUP = 32
# rows of a step the grouped GEMMs take a pass: bounds their scratch (a row's gate/up outputs
# for every slot) whatever the prefill
GROUPED_ROWS = 4096


class DeepseekV41MoE(nn.Module):
    def __init__(self, config: DeepseekV41Config, comm_group: CommGroup | None = None,
                 moe_parallel: MoeParallelConfig | None = None) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.tp_size = self.comm_group.world_size
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.block = config.fp8_block
        self.layout = config.moe_experts
        self.kernel = config.moe_kernel
        self._grouped_gemm = False  # resolved once the weights are on their device
        # each rank holds a slice of every expert's intermediate dim, whole scale blocks of it
        self.inter = divide(config.moe_intermediate_size, self.tp_size)
        if self.inter % self.block:
            raise ValueError(f"a rank's {self.inter} intermediate dims are not whole {self.block}-wide scale blocks")
        self.expert_parallel = moe_parallel is not None and moe_parallel.parallel == "ep" and self.tp_size > 1
        if self.expert_parallel:  # this rank's whole experts, at full width
            self.local_experts = divide(self.num_experts, self.tp_size)
            self.expert_start = self.comm_group.rank * self.local_experts
            self.expert_inter = config.moe_intermediate_size
        else:
            self.local_experts, self.expert_start, self.expert_inter = self.num_experts, 0, self.inter
        if self.layout == "fp4" and (self.expert_inter % FP4_GROUP or self.hidden_size % FP4_GROUP):
            raise ValueError(
                f"fp4 experts scale {FP4_GROUP} inputs at a time: a rank's {self.expert_inter} intermediate dims and "
                f"the hidden size {self.hidden_size} must be whole groups (moe_experts: fp8 or dense otherwise)")

        self.gate = NoAuxTcRouter(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
            scoring_func=config.scoring_func,
        )
        experts, hidden, inter, block = self.local_experts, self.hidden_size, self.expert_inter, self.block
        self.experts = nn.Module()
        if self.layout == "fp4":
            def stored(rows: int, width: int) -> nn.Parameter:
                return nn.Parameter(torch.empty(experts, rows, width, dtype=torch.uint8), requires_grad=False)

            self.experts.gate_up_proj_packed = stored(2 * inter, hidden // 2)
            self.experts.gate_up_proj_scale = stored(2 * inter, hidden // FP4_GROUP)
            self.experts.down_proj_packed = stored(hidden, inter // 2)
            self.experts.down_proj_scale = stored(hidden, inter // FP4_GROUP)
        elif self.layout == "fp8":
            self.experts.gate_up_proj_fp8 = nn.Parameter(
                torch.empty(experts, 2 * inter, hidden, dtype=torch.uint8), requires_grad=False)
            self.experts.gate_up_proj_scale_inv = nn.Parameter(
                torch.empty(experts, 2 * inter // block, hidden // block, dtype=torch.float32),
                requires_grad=False)
            self.experts.down_proj_fp8 = nn.Parameter(
                torch.empty(experts, hidden, inter, dtype=torch.uint8), requires_grad=False)
            self.experts.down_proj_scale_inv = nn.Parameter(
                torch.empty(experts, hidden // block, inter // block, dtype=torch.float32),
                requires_grad=False)
        else:
            self.experts.gate_up_proj = nn.Parameter(torch.empty(experts, 2 * inter, hidden))
            self.experts.down_proj = nn.Parameter(torch.empty(experts, hidden, inter))
        self.shared = nn.Module()
        self.shared.gate_up_proj = nn.Parameter(torch.empty(2 * self.inter, hidden))
        self.shared.down_proj = nn.Parameter(torch.empty(hidden, self.inter))

    def process_weights_after_loading(self, device) -> None:
        self.gate.finalize_weights()
        self._grouped_gemm = self.kernel == "auto" and self.layout == "fp4" and _grouped_gemm_runs(device)

    @property
    def capturable(self) -> bool:
        """Whether the routed experts run as grouped GEMM launches (static shapes, no host
        sync) rather than ``_dispatch``'s loop over the experts a step hit. Known once the
        weights are loaded."""
        return self._grouped_gemm

    def _expert_weights(self, expert: torch.Tensor, dtype: torch.dtype):
        """One routed expert's ``(gate_up, down)`` in ``dtype``: what ``dense`` holds, whatever
        the layout."""
        exp = self.experts
        if self.layout == "dense":
            return exp.gate_up_proj[expert], exp.down_proj[expert]
        if self.layout == "fp4":
            gate_up_scale = exp.gate_up_proj_scale[expert].view(E8M0_DTYPE)
            down_scale = exp.down_proj_scale[expert].view(E8M0_DTYPE)
            return (dequant_row_fp4(exp.gate_up_proj_packed[expert], gate_up_scale, dtype),
                    dequant_row_fp4(exp.down_proj_packed[expert], down_scale, dtype))
        block = (self.block, self.block)
        return (
            dequantize_fp8_block_weight(exp.gate_up_proj_fp8[expert], exp.gate_up_proj_scale_inv[expert],
                                        block_size=block, out_dtype=dtype),
            dequantize_fp8_block_weight(exp.down_proj_fp8[expert], exp.down_proj_scale_inv[expert],
                                        block_size=block, out_dtype=dtype),
        )

    def _activation(self, gate_up: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """The down GEMM's input from the gate/up GEMM's output: clamped SwiGLU in fp32, scaled
        by the routing ``weights``, cast and rounded to FP8."""
        gate, up = gate_up.float().chunk(2, dim=-1)
        limit = self.swiglu_limit
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        if weights is not None:
            act = weights * act
        return round_fp8(act.to(gate_up.dtype), self.block)

    def _expert(self, tokens: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """One expert over ``tokens`` (already rounded to FP8)."""
        return fp32_linear(self._activation(fp32_linear(tokens, gate_up), weights), down)

    def _dispatch(self, tokens: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """The routed experts' sum in fp32, over this rank's experts that the batch hit."""
        out = torch.zeros(tokens.shape, dtype=torch.float32, device=tokens.device)
        mask = F.one_hot(topk_ids, num_classes=self.num_experts).permute(2, 1, 0)
        mask = mask[self.expert_start:self.expert_start + self.local_experts]
        for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            slot, token = torch.where(mask[expert[0]])
            gate_up, down = self._expert_weights(expert[0], tokens.dtype)
            routed = self._expert(tokens[token], gate_up, down, topk_weights[token, slot, None])
            out.index_add_(0, token, routed.float())
        return out

    def _grouped(self, tokens: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """``_dispatch``'s sum on the grouped GEMM kernel, in the same order: a token's experts
        by ascending id."""
        from mstar.utils.fused_moe.w4a16 import grouped_experts_mxfp4

        exp = self.experts
        topk_ids, order = topk_ids.sort(dim=1)
        topk_weights = topk_weights.gather(1, order)
        if self.expert_parallel:  # another rank's expert: an id the kernel skips
            local = topk_ids - self.expert_start
            topk_ids = torch.where((local >= 0) & (local < self.local_experts), local, self.local_experts)
        sums = []
        for start in range(0, tokens.shape[0], GROUPED_ROWS):
            rows = slice(start, start + GROUPED_ROWS)
            slots = grouped_experts_mxfp4(
                tokens[rows], exp.gate_up_proj_packed, exp.down_proj_packed, exp.gate_up_proj_scale,
                exp.down_proj_scale, topk_ids[rows],
                partial(self._activation, weights=topk_weights[rows].reshape(-1, 1)),
                skip_invalid=self.expert_parallel).float()
            total = slots[:, 0]
            for slot in range(1, slots.shape[1]):
                total = total + slots[:, slot]
            sums.append(total)
        if len(sums) == 1:
            return sums[0]
        return torch.cat(sums) if sums else torch.zeros(tokens.shape, dtype=torch.float32, device=tokens.device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, self.hidden_size)
        topk_weights, topk_ids, _ = self.gate(flat)
        if self.expert_parallel:
            # a token's experts sit on several ranks, which must all route it alike
            topk_weights, topk_ids = _sync_routing(self.comm_group, topk_weights, topk_ids)
        tokens = round_fp8(flat, self.block)
        routed = self._grouped if self._grouped_gemm else self._dispatch
        out = routed(tokens, topk_weights, topk_ids)
        out = out + self._expert(tokens, self.shared.gate_up_proj, self.shared.down_proj).float()
        if self.tp_size > 1:
            out = self.comm_group.all_reduce(out)
        return out.to(hidden_states.dtype).view(hidden_states.shape)


def _grouped_gemm_runs(device) -> bool:
    """Whether the grouped GEMM kernel over fp4 experts runs on ``device``; on a GPU where it
    does not, says why once."""
    try:
        from mstar.utils.fused_moe import w4a16
    except ImportError:
        reason = "triton is not installed"
    else:
        reason = w4a16.grouped_gemm_unavailable(device)
    if reason is not None and device is not None and torch.device(device).type == "cuda":
        _warn_torch_dispatch(reason)
    return reason is None


@lru_cache(maxsize=None)
def _warn_torch_dispatch(reason: str) -> None:
    logger.warning("DeepSeek-V4.1's routed experts run the torch dispatch, eager only: %s", reason)
