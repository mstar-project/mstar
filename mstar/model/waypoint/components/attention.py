"""Waypoint self-attention: fused QKV, value residual, OrthoRoPE, ring cache.

Port of ``world_engine/src/model/attn.py::Attn`` **as it is actually served**.
``WorldEngine.__init__`` applies ``patch_model.apply_inference_patches``
unconditionally, so the shipped module is ``patch_model.MergedQKVAttn``: three
separate ``q_proj``/``k_proj``/``v_proj`` GEMMs fused into one. This port
implements the fused form (CONTRACTS section 5, DECISIONS D5), which makes the
*patched* reference the parity target -- one fused GEMM is not bit-identical to
three separate ones.

Facts that are load-bearing, in the order they bite:

  * **Fused layout.** ``qkv_proj.weight`` is ``cat([q, k, v], dim=0)`` --
    ``[q_out + 2 * kv_out, d_model] = [4096, 2048]`` here, rows 0:2048 = Q,
    2048:3072 = K, 3072:4096 = V, verified against
    ``patch_model.py:110-112`` (the ``cat``) and ``patch_model.py:117`` (the
    matching ``split((q_out, kv_out, kv_out), dim=-1)``). GQA makes the three
    slabs unequal, so a wrong order is a shape error for Q but *not* between K
    and V -- swapping those two loads cleanly and produces wrong video.
  * **Order inside forward:** value-residual lerp FIRST, then ``rms_norm(q, k)``,
    then RoPE on Q and K, then the cache upsert, then attention. Reordering the
    lerp past the norm changes what enters the cache, permanently, for every
    future frame that attends to it (CONTRACTS section 4.2).
  * **``v1`` is layer 0's V captured PRE-lerp** and threaded unchanged through
    all 24 layers. Layer 0 lerps against itself; every later layer lerps against
    layer 0. The **lerped** V is what the cache stores.
  * **Q and K are RMS-normed and rotated; V is neither.** K enters the ring
    already rotated, so replayed history is never re-rotated.
  * **The backend's argument order is not the reference's.** The reference calls
    ``kv_cache.upsert(k, v, pos_ids, layer_idx)``; the port's protocol is
    ``backend.upsert(k, v, layer_idx, frame_pos)``. Both trailing arguments are
    positional ints/tensors, so a verbatim transcription passes a layer index
    where a frame position belongs, raises nothing, and drifts (CONTRACTS
    section 3).

This module talks to the world state only through ``WaypointKVBackend``. It
never sees the ring, the slot arithmetic or the ``BlockMask``: ``meta`` is
opaque and goes straight back into ``attend``, which is what keeps a future
engine-owned KV implementation a one-class swap (DECISIONS D3).
"""

import torch
from torch import Tensor, nn

from mstar.model.waypoint.components.kv_backend import WaypointKVBackend
from mstar.model.waypoint.components.layers import rms_norm
from mstar.model.waypoint.components.rope import OrthoRoPE
from mstar.model.waypoint.config import WaypointConfig

__all__ = ["WaypointAttention"]


class WaypointAttention(nn.Module):
    """One layer of causal frame attention over the ring KV cache.

    Shapes: ``x`` is ``[B, N*T, D]`` (N frames of T tokens, flattened -- N is 1
    for every served call), and the head layout inside is ``[B, H, N*T, d_head]``
    with 32 query heads over 16 KV heads (GQA, 2:1).
    """

    def __init__(self, config: WaypointConfig, layer_idx: int):
        super().__init__()
        if config.gated_attn:
            # The reference supports a per-head sigmoid gate on the attention
            # output; this checkpoint does not use it and it carries a
            # `gate_proj` the loader could not fill. Same hard-fail stance as
            # Wan22DiT's qk_norm check: a drifting checkpoint must not be
            # silently mis-served.
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

        # Split widths for the fused projection, in the concat order baked into
        # the weight: Q (2048) | K (1024) | V (1024).
        self.q_out = self.n_heads * self.d_head
        self.kv_out = self.n_kv_heads * self.d_head

        if self.value_residual:
            # Per-layer scalar; the checkpoint value is what makes layer 0's V
            # matter more or less deep in the stack. Initialized to the
            # reference's 0.5 so a no-checkpoint structural build is sane.
            self.v_lamb = nn.Parameter(torch.tensor(0.5))

        self.qkv_proj = nn.Linear(config.d_model, self.q_out + 2 * self.kv_out, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Stateless and parameter-free, but kept as a per-layer submodule
        # because that is the reference's shape and the call site reads
        # `self.rope(q, rope_angles)`. The angles themselves are built once per
        # forward at the DiT root and passed down.
        self.rope = OrthoRoPE(config)

    def forward(
        self,
        x: Tensor,
        frame_pos: Tensor,
        rope_angles: tuple[Tensor, Tensor],
        v1: Tensor | None,
        backend: WaypointKVBackend,
    ) -> tuple[Tensor, Tensor]:
        """``x`` ``[B, N*T, D]`` -> ``(out [B, N*T, D], v1 [B, H_kv, N*T, d_head])``.

        ``frame_pos`` is the ``[]`` int64 ring clock (the reference's
        ``pos_ids["f_pos"][0, 0]``); it is the only piece of ``pos_ids`` this
        layer needs, since the RoPE angles are precomputed at the root. ``v1``
        is ``None`` at layer 0 and layer 0's pre-lerp V thereafter.
        """
        B, T = x.shape[:2]

        # One GEMM, then slice: the split widths and their order are the
        # transpose of the qkv_proj.weight row order (Q | K | V).
        q, k, v = self.qkv_proj(x).split((self.q_out, self.kv_out, self.kv_out), dim=-1)
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.value_residual:
            # v1 is captured BEFORE the lerp, so layer 0 returns its raw V while
            # storing the (self-)lerped one. The lerped V is what the cache
            # keeps and what every future frame attends to.
            v1 = v if v1 is None else v1
            v = torch.lerp(v, v1.view_as(v), self.v_lamb)

        # Q/K only: V is neither normed nor rotated, at any layer.
        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, rope_angles), self.rope(k, rope_angles)

        # NOTE the argument order -- (k, v, layer_idx, frame_pos), NOT the
        # reference's (k, v, pos_ids, layer_idx). CONTRACTS section 3.
        # `k` goes in post-RoPE; history comes back already rotated.
        k, v, meta = backend.upsert(k, v, self.layer_idx, frame_pos)

        y = backend.attend(q, k, v, meta, enable_gqa=self.enable_gqa)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(y), v1
