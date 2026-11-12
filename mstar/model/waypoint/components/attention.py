"""Waypoint self-attention: fused QKV, value residual, OrthoRoPE, the ring of frames.

The served reference is the fused ``patch_model.MergedQKVAttn``, not three
separate GEMMs, so that fused form is the parity target. ``qkv_proj.weight``
is ``cat([q, k, v], dim=0)``; GQA makes the K/V slabs unequal, so swapping
them loads cleanly but serves wrong video.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from mstar.model.waypoint.components.layers import rms_norm
from mstar.model.waypoint.components.rope import OrthoRoPE
from mstar.model.waypoint.config import WaypointConfig

__all__ = ["WaypointAttention"]


class WaypointAttention(nn.Module):
    """One layer of causal frame attention over the world's ring of frames.

    ``x`` is ``[B, N*T, D]`` (N frames of T tokens, flattened; N is 1 for every
    served call) and the head layout inside is ``[B, H, N*T, d_head]``, 32 query
    heads over 16 KV heads.
    """

    def __init__(self, config: WaypointConfig, layer_idx: int):
        super().__init__()
        if config.gated_attn:
            # This checkpoint is ungated and carries no `gate_proj` for the
            # loader to fill; hard-fail rather than silently mis-serve.
            raise ValueError(
                "WaypointAttention implements the ungated attention path only; "
                "this config declares gated_attn=True."
            )

        self.config = config
        self.layer_idx = layer_idx
        self.value_residual = config.value_residual

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head

        # Split widths in the concat order baked into the weight: Q | K | V.
        self.q_out = self.n_heads * self.d_head
        self.kv_out = self.n_kv_heads * self.d_head

        if self.value_residual:
            # Per-layer scalar; 0.5 is the reference's init, so a structural
            # build without a checkpoint is still sane.
            self.v_lamb = nn.Parameter(torch.tensor(0.5))

        self.qkv_proj = nn.Linear(config.d_model, self.q_out + 2 * self.kv_out, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Per-layer to match the reference's call site; the angles themselves
        # are built once per forward at the DiT root.
        self.rope = OrthoRoPE(config)

        # Bound at load by NodeSubmodule.bind_node_state.
        self.state = None

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.state = handles["kv"]

    def forward(
        self,
        x: Tensor,
        rope_angles: tuple[Tensor, Tensor],
        v1: Tensor | None,
        *,
        commit: bool,
    ) -> tuple[Tensor, Tensor]:
        """``x`` ``[B, N*T, D]`` -> ``(out [B, N*T, D], v1 [B, H_kv, N*T, d_head])``.

        ``v1`` is ``None`` at layer 0 and layer 0's pre-lerp V thereafter.
        ``commit`` keeps this pass's K/V in the ring.
        """
        B, T = x.shape[:2]

        # One GEMM, then slice: the split widths transpose qkv_proj.weight's
        # row order (Q | K | V).
        q, k, v = self.qkv_proj(x).split((self.q_out, self.kv_out, self.kv_out), dim=-1)
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.value_residual:
            # v1 is captured BEFORE the lerp, so layer 0 returns its raw V and
            # caches the self-lerped one. Lerp stays ahead of the norm: what
            # this order produces is what the cache keeps, permanently.
            v1 = v if v1 is None else v1
            v = torch.lerp(v, v1.view_as(v), self.v_lamb)

        # Q/K only: V is neither normed nor rotated, at any layer.
        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, rope_angles), self.rope(k, rope_angles)

        # `k` goes in post-RoPE. The add runs on frozen passes too: its scratch
        # write is what makes this frame visible to itself.
        self.state.add(self.layer_idx, keep=commit, k=k.transpose(1, 2), v=v.transpose(1, 2))
        y = self.state.read(self.layer_idx, q.transpose(1, 2))
        return self.out_proj(y.reshape(B, T, -1)), v1
