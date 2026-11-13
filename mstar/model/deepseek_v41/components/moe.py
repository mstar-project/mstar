"""DeepSeek-V4.1's MoE: the ``noaux_tc`` router on sqrt-softplus scores, routed experts and one
shared expert, with the reference's rounding.

Every expert GEMM reads its input rounded to FP8 in 32-wide groups, and the routing weight
scales an expert's activation before the down projection rounds it. ``_dispatch`` is that
arithmetic expert by expert, the numerics anchor a fused kernel is held to.

The routed experts are held as ``moe_experts`` says, the same values each way:

- ``fp4``: the checkpoint's bytes, MXFP4. ``gate_up_proj_packed (E, 2 * inter, hidden / 2)`` and
  ``down_proj_packed (E, hidden, inter / 2)`` hold two E2M1 codes a byte along the inputs (the
  even input in the low nibble), gate rows before up rows; ``gate_up_proj_scale (E, 2 * inter,
  hidden / 32)`` and ``down_proj_scale (E, hidden, inter / 32)`` hold one E8M0 byte per row and
  32 inputs. A hit expert is dequantized for the step.
- ``fp8``: their upcast to E4M3 32 x 32 blocks, in ``fused_moe``'s layout.
- ``dense``: the model's dtype.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.moe import NoAuxTcRouter
from mstar.model.deepseek_v41.components.linear import fp32_linear
from mstar.model.deepseek_v41.config import DeepseekV41Config
from mstar.utils.fp8_block import E8M0_DTYPE, dequantize_fp8_block_weight
from mstar.utils.mx_round import dequant_row_fp4, round_fp8

# MXFP4's scale group: one E8M0 scale per 32 inputs of a row
FP4_GROUP = 32


class DeepseekV41MoE(nn.Module):
    def __init__(self, config: DeepseekV41Config, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.tp_size = self.comm_group.world_size
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.block = config.fp8_block
        self.layout = config.moe_experts
        # each rank holds a slice of every expert's intermediate dim, whole scale blocks of it
        self.inter = divide(config.moe_intermediate_size, self.tp_size)
        if self.inter % self.block:
            raise ValueError(f"a rank's {self.inter} intermediate dims are not whole {self.block}-wide scale blocks")
        if self.layout == "fp4" and (self.inter % FP4_GROUP or self.hidden_size % FP4_GROUP):
            raise ValueError(
                f"fp4 experts scale {FP4_GROUP} inputs at a time: a rank's {self.inter} intermediate dims and "
                f"the hidden size {self.hidden_size} must be whole groups (moe_experts: fp8 or dense otherwise)")

        self.gate = NoAuxTcRouter(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
            scoring_func=config.scoring_func,
        )
        experts, hidden, inter, block = self.num_experts, self.hidden_size, self.inter, self.block
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
        self.shared.gate_up_proj = nn.Parameter(torch.empty(2 * inter, hidden))
        self.shared.down_proj = nn.Parameter(torch.empty(hidden, inter))

    def process_weights_after_loading(self, device) -> None:
        del device
        self.gate.finalize_weights()

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

    def _expert(self, tokens: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """One expert over ``tokens`` (already rounded to FP8): clamped SwiGLU in fp32, then
        its activation, scaled by the routing ``weights``, rounded to FP8 into the down GEMM."""
        gate, up = fp32_linear(tokens, gate_up).float().chunk(2, dim=-1)
        limit = self.swiglu_limit
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        if weights is not None:
            act = weights * act
        return fp32_linear(round_fp8(act.to(tokens.dtype), self.block), down)

    def _dispatch(self, tokens: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """The routed experts' sum in fp32, over the experts this batch hit."""
        out = torch.zeros(tokens.shape, dtype=torch.float32, device=tokens.device)
        mask = F.one_hot(topk_ids, num_classes=self.num_experts).permute(2, 1, 0)
        for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            slot, token = torch.where(mask[expert[0]])
            gate_up, down = self._expert_weights(expert[0], tokens.dtype)
            routed = self._expert(tokens[token], gate_up, down, topk_weights[token, slot, None])
            out.index_add_(0, token, routed.float())
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, self.hidden_size)
        topk_weights, topk_ids, _ = self.gate(flat)
        tokens = round_fp8(flat, self.block)
        out = self._dispatch(tokens, topk_weights, topk_ids)
        out = out + self._expert(tokens, self.shared.gate_up_proj, self.shared.down_proj).float()
        if self.tp_size > 1:
            out = self.comm_group.all_reduce(out)
        return out.to(hidden_states.dtype).view(hidden_states.shape)
