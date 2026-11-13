"""DeepSeek-V4.1's MoE: the ``noaux_tc`` router on sqrt-softplus scores, routed experts and one
shared expert, with the reference's rounding.

Every expert GEMM reads its input rounded to FP8 in 32-wide groups, and the routing weight
scales an expert's activation before the down projection rounds it. ``_dispatch`` is that
arithmetic expert by expert, the numerics anchor a fused kernel is held to. The routed experts
stay in the checkpoint's bytes' worth of fp8 (``moe_fp8_resident``: its fp4 upcast to E4M3
blocks, in ``fused_moe``'s layout) or in the model's dtype.
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
from mstar.utils.fp8_block import dequantize_fp8_block_weight
from mstar.utils.mx_round import round_fp8


class DeepseekV41MoE(nn.Module):
    def __init__(self, config: DeepseekV41Config, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.tp_size = self.comm_group.world_size
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.block = config.fp8_block
        # each rank holds a slice of every expert's intermediate dim, whole scale blocks of it
        self.inter = divide(config.moe_intermediate_size, self.tp_size)
        if self.inter % self.block:
            raise ValueError(f"a rank's {self.inter} intermediate dims are not whole {self.block}-wide scale blocks")
        self.fp8_experts = config.moe_fp8_resident

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
        if self.fp8_experts:
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
        exp = self.experts
        if not self.fp8_experts:
            return exp.gate_up_proj[expert], exp.down_proj[expert]
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
