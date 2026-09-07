"""Latent MoE reference (spec E): router -> latent down-projection -> routed experts in
the 3584-space -> RMSNorm -> up-projection, plus the fused shared-expert MLP.

Experts are given as dequantized bf16/fp32 tensors stacked over experts:
``w13 [E, 2*inter, latent]`` (gate rows first, then up rows) and ``w2 [E, latent, inter]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mstar.model.kimi_k3.reference.mla import rms_norm
from mstar.model.kimi_k3.reference.router import noaux_tc_route
from mstar.model.kimi_k3.reference.situ import situ_and_mul


@dataclass
class LatentMoEWeights:
    gate_weight: torch.Tensor  # [E, hidden]
    e_score_correction_bias: torch.Tensor  # [E]
    down_proj: torch.Tensor  # [latent, hidden]
    up_proj: torch.Tensor  # [hidden, latent]
    routed_norm: torch.Tensor | None  # [latent]
    w13: torch.Tensor  # [E, 2*inter, latent]
    w2: torch.Tensor  # [E, latent, inter]
    shared_gate_up: torch.Tensor | None  # [2*shared_inter, hidden]
    shared_down: torch.Tensor | None  # [hidden, shared_inter]
    top_k: int = 16
    situ_beta: float = 4.0
    situ_linear_beta: float | None = 25.0
    norm_eps: float = 1e-5
    renormalize: bool = True
    routed_scaling_factor: float = 1.0

    @classmethod
    def from_hf_module(cls, m, dtype: torch.dtype = torch.float32) -> LatentMoEWeights:
        """Stack an HF ``KimiSparseMoeBlock`` (bf16 experts) into the reference layout."""
        w13 = torch.stack(
            [torch.cat([e.w1.weight, e.w3.weight], dim=0) for e in m.experts]
        ).to(dtype)
        w2 = torch.stack([e.w2.weight for e in m.experts]).to(dtype)
        shared = getattr(m, "shared_experts", None)
        act = m.experts[0].act_fn
        return cls(
            gate_weight=m.gate.weight,
            e_score_correction_bias=m.gate.e_score_correction_bias,
            down_proj=m.routed_expert_down_proj.weight,
            up_proj=m.routed_expert_up_proj.weight,
            routed_norm=m.routed_expert_norm.weight if getattr(m, "latent_moe_use_norm", False) else None,
            w13=w13,
            w2=w2,
            shared_gate_up=None if shared is None else torch.cat([shared.gate_proj.weight, shared.up_proj.weight], 0),
            shared_down=None if shared is None else shared.down_proj.weight,
            top_k=m.top_k,
            situ_beta=getattr(act, "beta", 1.0),
            situ_linear_beta=getattr(act, "linear_beta", None),
            norm_eps=m.routed_expert_norm.variance_epsilon if getattr(m, "latent_moe_use_norm", False) else 1e-5,
            renormalize=m.gate.moe_renormalize,
            routed_scaling_factor=m.gate.routed_scaling_factor,
        )


def routed_experts_loop(
    z: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Per-expert loop over the latent inputs ``z [T, latent]``; returns ``[T, latent]``
    in ``z.dtype`` with the routing weights applied in fp32 (like ``moe_infer``)."""
    t, k = topk_idx.shape
    out = torch.zeros(t, k, z.shape[-1], dtype=torch.float32, device=z.device)
    for e in torch.unique(topk_idx).tolist():
        rows, slots = torch.where(topk_idx == e)
        h = F.linear(z[rows], w13[e].to(z.dtype))
        h = situ_and_mul(h, beta, linear_beta)
        y = F.linear(h, w2[e].to(z.dtype))
        out[rows, slots] = y.float()
    out = (out * topk_weight.float().unsqueeze(-1)).sum(dim=1)
    return out.to(z.dtype)


def shared_experts_mlp(
    x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, beta: float, linear_beta: float | None,
) -> torch.Tensor:
    h = F.linear(x, gate_up.to(x.dtype))
    return F.linear(situ_and_mul(h, beta, linear_beta), down.to(x.dtype))


def latent_moe_forward(w: LatentMoEWeights, x: torch.Tensor) -> torch.Tensor:
    """``x [T, hidden] -> [T, hidden]``."""
    topk_idx, topk_weight = noaux_tc_route(
        x, w.gate_weight, w.e_score_correction_bias, w.top_k,
        renormalize=w.renormalize, routed_scaling_factor=w.routed_scaling_factor,
    )
    z = F.linear(x, w.down_proj.to(x.dtype))
    y = routed_experts_loop(z, topk_idx, topk_weight, w.w13, w.w2, w.situ_beta, w.situ_linear_beta)
    if w.routed_norm is not None:
        y = rms_norm(y, w.routed_norm, w.norm_eps)
    y = F.linear(y, w.up_proj.to(x.dtype))
    if w.shared_gate_up is not None:
        y = y + shared_experts_mlp(x, w.shared_gate_up, w.shared_down, w.situ_beta, w.situ_linear_beta)
    return y
