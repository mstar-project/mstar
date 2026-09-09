"""Per-request KDA state: the model's view of the engine's slot-state resource.

KDA layers keep a fixed-size recurrent matrix memory ``S (H, D, D)`` fp32 and
a conv tail ``(3*H*D, kernel-1)`` per request per layer — NOT paged KV. On
the resource-pool engine that state is a ``SlotStateManager`` (declared by
:func:`kda_slot_state_config`): the engine owns the slot pool, the
``request -> slot`` lease, the per-step slot index and the committed-token
count, and drives them through ingest / admit / plan / commit / remove. The
model never allocates or frees anything.

What stays model-side is the layer-facing access this module provides:

- pool layout: ``recurrent (L_kda, S+1, H, D, D)`` fp32 and ``conv (L_kda,
  S+1, 3*H*D, kernel-1)`` in the compute dtype — layer-major so a decode
  step's per-layer gather touches one contiguous ``(S+1, ...)`` plane;
- the decode gather/scatter by the planned ``[bs]`` slot index into
  persistent staging planes (one plane per pool, reused layer to layer —
  stream order serializes scatter before the next gather): fixed shapes, no
  ``.item()``, no per-step allocation, capture/compile friendly;
- in-place slot views for the prefill span loop (host loop, eager phase).

Memory (full model, per TP rank at TP8 once KDA is head-sharded, heads
64 -> 8): 512 KB recurrent + 18 KB conv per layer x 34 layers ~= 17.6
MB/request/rank. Until the sharding lands the pool allocates full width
(~143 MB/request, ~4.6 GB at 32 slots) — size ``max_slots`` on the box.

MTP (M2): the delta rule is not rewindable, so verify must ``snapshot``
before the speculative window and ``restore`` on rejection; both are here
(and double as the test seam for prefill/decode parity across a save/restore).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.engine.resources.slot_state.config import (
    SlotStateConfig,
    SlotStatePlan,
    SlotTensorSpec,
)
from mstar.engine.resources.slot_state.manager import SlotStateManager

RECURRENT = "recurrent"
CONV = "conv"


def kda_slot_state_config(
    config, max_slots: int, conv_dtype: torch.dtype = torch.bfloat16,
) -> SlotStateConfig:
    """The ``SlotStateConfig`` for this model's KDA state.

    ``config`` duck-types ``Glm5NextModelConfig`` (``kda_layer_indices``,
    ``linear_num_heads``, ``linear_head_dim``, ``linear_conv_channels``,
    ``linear_conv_kernel_size``). ``conv_dtype`` must be the KDA projection
    dtype (the continue path ``cat``s the conv tail onto the projected
    activations bit-exactly). ``shard_dim`` is None on both tensors until
    the KDA layer is head-sharded across TP — the pool is replicated like
    the layer.
    """
    num_layers = len(config.kda_layer_indices)
    return SlotStateConfig(
        tensors={
            RECURRENT: SlotTensorSpec(
                shape=(
                    num_layers, config.linear_num_heads,
                    config.linear_head_dim, config.linear_head_dim,
                ),
                dtype=torch.float32,
                slot_dim=1,
            ),
            CONV: SlotTensorSpec(
                shape=(
                    num_layers, config.linear_conv_channels,
                    config.linear_conv_kernel_size - 1,
                ),
                dtype=conv_dtype,
                slot_dim=1,
            ),
        },
        max_slots=max_slots,
    )


@dataclass
class Glm5NextKdaSnapshot:
    """Deep copy of one request's KDA state — the M2 rewind primitive."""

    recurrent: torch.Tensor  # (L_kda, H, D, D) fp32
    conv: torch.Tensor  # (L_kda, 3*H*D, kernel-1)
    committed: int


class Glm5NextKdaStateAccess:
    """Layer-facing gather/scatter over the resource's pools."""

    def __init__(self, resource: SlotStateManager) -> None:
        self.resource = resource
        self._recurrent = resource.pool(RECURRENT)
        self._conv = resource.pool(CONV)
        rows = resource.max_slots + 1
        # Persistent decode staging: one gather plane per pool.
        self._gather_recurrent = torch.zeros(
            (rows, *self._recurrent.shape[2:]),
            dtype=self._recurrent.dtype, device=self._recurrent.device,
        )
        self._gather_conv = torch.zeros(
            (rows, *self._conv.shape[2:]),
            dtype=self._conv.dtype, device=self._conv.device,
        )

    # -- per-step plan ----------------------------------------------------

    def current_plan(self) -> SlotStatePlan:
        return self.resource.current_plan()

    def slot_of(self, request_id: str) -> int | None:
        return self.resource.slot_of(request_id)

    def committed_tokens(self, request_id: str) -> int:
        return self.resource.committed(request_id)

    def tracked_requests(self) -> set[str]:
        return self.resource.tracked_requests()

    # -- layer-facing state access ----------------------------------------

    def state_views(
        self, kda_pos: int, slot: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """In-place slot views ``(recurrent (1, H, D, D), conv (1, C, W))``
        for the prefill span loop — ``Glm5NextLinearAttention.prefill``
        writes back through them, no copies."""
        return (
            self._recurrent[kda_pos, slot].unsqueeze(0),
            self._conv[kda_pos, slot].unsqueeze(0),
        )

    def gather(
        self, kda_pos: int, slot_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched decode read: contiguous ``(bs, ...)`` copies in slot-index
        order, written into the persistent staging plane. ``decode_step``
        mutates the views in place; ``scatter`` commits."""
        batch = slot_index.shape[0]
        recurrent = self._gather_recurrent[:batch]
        conv = self._gather_conv[:batch]
        torch.index_select(self._recurrent[kda_pos], 0, slot_index, out=recurrent)
        torch.index_select(self._conv[kda_pos], 0, slot_index, out=conv)
        return recurrent, conv

    def scatter(
        self,
        kda_pos: int,
        slot_index: torch.Tensor,
        recurrent: torch.Tensor,
        conv: torch.Tensor,
    ) -> None:
        self._recurrent[kda_pos].index_copy_(0, slot_index, recurrent)
        self._conv[kda_pos].index_copy_(0, slot_index, conv)

    # -- snapshot / restore (M2 verify-rewind; test seam today) -----------

    def snapshot(self, request_id: str) -> Glm5NextKdaSnapshot:
        slot = self._slot(request_id)
        return Glm5NextKdaSnapshot(
            recurrent=self._recurrent[:, slot].clone(),
            conv=self._conv[:, slot].clone(),
            committed=self.resource.committed(request_id),
        )

    def restore(self, request_id: str, snap: Glm5NextKdaSnapshot) -> None:
        slot = self._slot(request_id)
        self._recurrent[:, slot].copy_(snap.recurrent)
        self._conv[:, slot].copy_(snap.conv)
        self.resource._committed[request_id] = snap.committed

    def _slot(self, request_id: str) -> int:
        slot = self.resource.slot_of(request_id)
        if slot is None:
            raise KeyError(f"request {request_id!r} holds no KDA state slot")
        return slot
