"""Waypoint self-attention: fused QKV, value residual, OrthoRoPE, ring cache.

``WorldEngine.__init__`` applies ``apply_inference_patches`` unconditionally, so
the served reference module is the fused ``patch_model.MergedQKVAttn``. That
fused form, not three separate GEMMs, is the parity target.

``qkv_proj.weight`` is ``cat([q, k, v], dim=0)`` — rows 0:2048 Q, 2048:3072 K,
3072:4096 V. GQA makes the slabs unequal, so a wrong order is a shape error for
Q but *not* between K and V: swapping those two loads cleanly and serves wrong
video.

This module reaches the world state only through the two resources bound in
``bind_resources``. It never sees the ring, the slot arithmetic or the
``BlockMask`` — ``visible`` is a ``[capacity]`` bool row it hands straight back
to ``attend``.
"""

import torch
from torch import Tensor, nn

from mstar.model.waypoint.components.layers import rms_norm
from mstar.model.waypoint.components.rope import OrthoRoPE
from mstar.model.waypoint.config import WaypointConfig

__all__ = ["WaypointAttention"]


class WaypointAttention(nn.Module):
    """One layer of causal frame attention over the ring KV cache.

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
        self.enable_gqa = config.enable_gqa

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

        # Bound at load by NodeSubmodule.bind_node_resources.
        self.kv = None
        self.attn = None

    def bind_resources(self, resources: dict) -> None:
        """Resolve the two resources this layer calls.

        ``NodeSubmodule.bind_node_resources`` walks ``self.modules()``, so one
        bind on the submodule reaches all 24 layers. ``.get``: a layer may be
        bound on a node that owns only some of them.
        """
        self.kv = resources.get("kv")
        self.attn = resources.get("attn")

    def forward(
        self,
        x: Tensor,
        frame_pos: Tensor,
        rope_angles: tuple[Tensor, Tensor],
        v1: Tensor | None,
        *,
        commit: bool,
    ) -> tuple[Tensor, Tensor]:
        """``x`` ``[B, N*T, D]`` -> ``(out [B, N*T, D], v1 [B, H_kv, N*T, d_head])``.

        ``frame_pos`` is the ``[]`` int64 ring clock; ``v1`` is ``None`` at
        layer 0 and layer 0's pre-lerp V thereafter. Both stay arguments rather
        than cursors the KV resource keeps: a cursor that drifts from the
        caller does not raise, it rewrites history.
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

        # Argument order is (k, v, layer_idx, frame_pos), NOT the reference's
        # (k, v, pos_ids, layer_idx). `k` goes in post-RoPE.
        if getattr(self.attn, "needs_token_visibility", True):
            k, v, visible = self.kv.upsert(
                k, v, self.layer_idx, frame_pos, commit=commit
            )
        else:
            k, v, visible = self.kv.upsert(
                k,
                v,
                self.layer_idx,
                frame_pos,
                commit=commit,
                build_visibility=False,
            )

        # No `if self.attn.requires_kv_write` guard: the upsert above IS the
        # write, and it has to happen on frozen passes too — the scratch slot
        # is what makes this frame visible to itself.
        y = self.attn.attend(
            q, k, v, visible,
            enable_gqa=self.enable_gqa,
            layer_idx=self.layer_idx,
        )
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(y), v1
