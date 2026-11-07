"""Block attention residuals (AttnRes), Kimi K3's residual stream.

Instead of one running sum, a layer reads its input as a softmax mixture of the residual
stack (the stream's value at each block start, ``attn_res_block_size`` layers apart) and the
running partial sum since the last block start. A candidate ``v`` scores
``<rmsnorm_noweight(v), res_norm.weight * res_proj.weight>`` (no 1/sqrt(d)); the mixture is
of the unnormalized candidates, in fp32. Two reads per layer (before attention, before the
MLP) and one before the final norm.

The stack is per token across depth, so it lives only inside one forward: nothing persists
across steps and no resource holds it. Follows ``_apply_attn_res`` in the checkpoint's
``modeling_kimi_linear.py`` and merceod's #257.
"""
from __future__ import annotations

import torch


def attn_res_read(
    prefix: torch.Tensor, blocks: torch.Tensor | None, score_weight: torch.Tensor, eps: float,
) -> torch.Tensor:
    """``prefix (T, D)`` mixed with ``blocks (T, m, D)``; ``prefix`` itself when the stack is
    empty (a softmax over one candidate)."""
    if blocks is None or blocks.shape[1] == 0:
        return prefix
    # fp32 in the reference; keep it so under the engine's bf16 autocast
    with torch.autocast(device_type=prefix.device.type, enabled=False):
        v = torch.cat((blocks, prefix.unsqueeze(1)), dim=1).float()  # (T, m + 1, D)
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        probs = (k * score_weight.float()).sum(-1).softmax(-1)  # (T, m + 1)
        mixed = torch.einsum("tm,tmd->td", probs, v)
    return mixed.to(prefix.dtype)


def score_weight(norm_weight: torch.Tensor, proj_weight: torch.Tensor) -> torch.Tensor:
    """A read site's query, ``<site>_norm.weight [D] * <site>_proj.weight [1, D]``, in fp32."""
    return norm_weight.float() * proj_weight.reshape(-1).float()


def push_block(blocks: torch.Tensor | None, prefix: torch.Tensor) -> torch.Tensor:
    """The stack with ``prefix (T, D)`` appended: ``(T, m + 1, D)``."""
    if blocks is None:
        return prefix.unsqueeze(1)
    return torch.cat((blocks, prefix.unsqueeze(1)), dim=1)
