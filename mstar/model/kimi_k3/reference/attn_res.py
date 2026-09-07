"""Block Attention Residuals (spec B.6).

One AttnRes *read* mixes the residual-stack entries ``blocks[:, i, :]`` (i < num_blocks)
and the running intra-block partial sum ``prefix`` with softmax weights. Each candidate
``v`` is scored by ``<RMSNorm_noweight(v), score_weight>`` where
``score_weight = res_norm.weight * res_proj.weight`` (a single 7168-vector), with no
``1/sqrt(d)``; the mixture is over the *unnormalized* candidates, in fp32.

Two reads happen per decoder layer (before attention, before the FFN) and one more at
the model output. See ``spec.KimiK3TextConfig.attn_res_blocks_before`` for how many
stack entries are valid at a given layer.
"""
from __future__ import annotations

import torch


def attn_res_score_weight(norm_weight: torch.Tensor, proj_weight: torch.Tensor) -> torch.Tensor:
    """Fold ``res_norm.weight [D]`` and ``res_proj.weight [1, D]`` into one fp32 vector."""
    return norm_weight.float() * proj_weight.reshape(-1).float()


def attn_res_read(
    prefix: torch.Tensor,
    blocks: torch.Tensor | None,
    score_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Mix ``blocks[:, :m]`` and ``prefix`` for every token.

    Args:
        prefix: ``[T, D]`` running partial sum (any float dtype).
        blocks: ``[T, m, D]`` valid residual-stack entries, or ``None``/``m == 0``.
        score_weight: ``[D]`` fp32, see :func:`attn_res_score_weight`.
        eps: the res_norm epsilon (1e-5 for K3).

    Returns ``[T, D]`` in ``prefix.dtype``. With no blocks the result is ``prefix``
    itself (softmax over one candidate), which matches the reference skipping the read.
    """
    if blocks is None or blocks.shape[1] == 0:
        return prefix
    # the mixture is fp32 in the reference; keep it so under bf16 autocast
    with torch.autocast(device_type=prefix.device.type, enabled=False):
        v = torch.cat((blocks, prefix.unsqueeze(1)), dim=1).float()  # [T, m+1, D]
        rstd = torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        scores = (v * rstd * score_weight).sum(-1)  # [T, m+1]
        probs = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("tm,tmd->td", probs, v)
    return mixed.to(prefix.dtype)


def attn_res_read_and_norm(
    prefix: torch.Tensor,
    blocks: torch.Tensor | None,
    score_weight: torch.Tensor,
    out_norm_weight: torch.Tensor,
    eps: float = 1e-5,
    out_eps: float = 1e-5,
) -> torch.Tensor:
    """AttnRes read followed by the layer's own RMSNorm (``input_layernorm`` or
    ``post_attention_layernorm``), matching ``KimiRMSNorm``: fp32 normalization, then the
    weight multiply in the activation dtype."""
    mixed = attn_res_read(prefix, blocks, score_weight, eps)
    x = mixed.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + out_eps)
    return out_norm_weight * x.to(mixed.dtype)
