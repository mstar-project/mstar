"""A drafter's context write: K/V a stack writes into the pool without attending with them."""
from __future__ import annotations

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.convenience import AttentionCallable


@torch.compiler.disable
def norm_rope_write(
    attend: AttentionCallable, k: torch.Tensor, v: torch.Tensor, k_norm: torch.nn.Module, pos_ids: torch.Tensor,
    rope_theta: float,
) -> None:
    """``AttentionCallable.norm_rope_attend``'s write without the attention: k's per-head RMSNorm and RoPE and
    the KV write are one launch of the same kernel, over no query heads. Through the attention's exchange, k is
    normed and rotated in place and written through it."""
    from mstar.utils.attention import qk_norm_rope_inplace, qk_norm_rope_kv_write

    if getattr(k_norm, "gemma_mode", False):
        raise ValueError("the fused k norm kernel scales by weight; a gemma_mode RMSNorm needs its own path")
    eps = float(k_norm.variance_epsilon)
    no_q = k.new_empty(k.shape[0], 0, k.shape[2])
    if isinstance(attend.attn, AttentionManager) and attend.attn.step_exchanges():
        qk_norm_rope_inplace(no_q, k, k_norm.weight, k_norm.weight, pos_ids, eps, eps, rope_theta)
        attend.write(k, v)
        return
    layer, page_idx, slot_idx, n = attend.kv.write_target()
    qk_norm_rope_kv_write(
        no_q, k, v, k_norm.weight, k_norm.weight, pos_ids, layer, page_idx, slot_idx, min(n, k.shape[0]),
        eps, eps, rope_theta,
    )
