"""The attention resource's shared machinery: the spec-time factory, the
custom op every backend attends through, and the workspace pool.

The backends themselves live beside this — `flashinfer`, `cross`, `dense` —
and `AttentionManager.build` reaches them by deferred import, so naming a
backend in a spec does not load the other two.
"""

import logging
import os
from typing import NamedTuple

import torch

from mstar.engine.resources.attn.config import AttentionSpec, AttnBackend
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo

logger = logging.getLogger(__name__)


class AttentionManager(AttentionResource):
    # Remains abstract except for build; will build based
    # on the attention backend

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    @classmethod
    def build(cls, spec: AttentionSpec, info: EngineResourceInfo):
        # the wrappers are planned against per-rank head counts; idempotent, so
        # sharing one KVConfig with the KV resource is fine (KVConfig.shard)
        if info.joint_comm_group is not None:
            spec.kv_config.shard(info.joint_comm_group.world_size)
        backend = spec.config.backend
        if backend == AttnBackend.DENSE:
            # A dense backend needs the FlashAttention-3 kernel; where the
            # wheel does not match the installed torch/CUDA build, degrade to
            # the paged backend rather than failing to serve. The two are
            # drop-in for each other: the step declaration is identical
            # (see DenseAttentionManager) and `requires_kv_write` tells the
            # layer which one it is talking to.
            from mstar.engine.resources.attn.dense import (
                DenseAttentionManager,
                _fa3_unavailable_reason,
                _warn_dense_fallback,
            )

            reason = _fa3_unavailable_reason()
            if reason is None:
                return DenseAttentionManager(
                    kv_cache=spec.config.kv_cache,
                    device=info.device,
                    dtype=info.kv_dtype,
                    kv_config=spec.kv_config,
                )
            _warn_dense_fallback(reason)
            backend = AttnBackend.FLASHINFER

        if backend == AttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.flashinfer import FlashInferManager

            return FlashInferManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=spec.kv_config,
                backend=spec.config.flashinfer_backend,
            )
        raise ValueError(f"Unknown attention backend {backend!r}")

    @property
    def requires_kv_write(self) -> bool:
        """Whether a layer must write this step's K/V through the KV resource
        before calling ``run``.

        False only for backends that take the fresh K/V straight into the
        kernel (the dense one). A layer therefore reads

            if self.attn.requires_kv_write:
                self.kv.write_kv(k, v, layer_idx=i, label=label)
            out = self.attn.run(q, label, kv.layer_view(i), k=k, v=v, layer_idx=i)

        and stays correct whichever backend the spec named.
        """
        return True


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    page_indices: tuple
    last_page_lens: tuple


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


# Attention resources reachable from a custom op. A traced forward can't pass
# the resource itself into an op (a schema takes tensors and scalars), so it
# passes this handle. One entry per resource, fixed for the process, so dynamo
# specializing on it costs nothing — a handle that varied per step or per slot
# would reintroduce the recompiles this exists to avoid.
_ATTENDERS: dict[int, "AttentionManager"] = {}


def _register_attender(manager: "AttentionManager") -> int:
    """Give an attention resource a handle its layers can pass into the op."""
    handle = len(_ATTENDERS)
    _ATTENDERS[handle] = manager
    return handle


@torch.library.custom_op("mstar::flashinfer_attend", mutates_args=())
def flashinfer_attend(
    handle: int, label: str, q: torch.Tensor, kv_cache_layer: torch.Tensor,
) -> torch.Tensor:
    """One layer's attention, planned by the resource behind ``handle``.

    Behind an op because FlashInfer's kernel is a TVM-FFI call dynamo can't
    trace and can't run on fake tensors: called directly it breaks the graph
    once per layer, and each break makes the layer body a frame dynamo
    recompiles per ``layer_idx``.
    """
    out = _ATTENDERS[handle].attend(q, label, kv_cache_layer)
    # the planned wrapper runs in its own dtype; the fake below promises q's,
    # and a mismatch there is silent corruption
    return out.to(q.dtype)


@flashinfer_attend.register_fake
def _flashinfer_attend_fake(
    handle: int, label: str, q: torch.Tensor, kv_cache_layer: torch.Tensor,
) -> torch.Tensor:
    # attention is shape-preserving on the query
    return torch.empty_like(q)


class WorkspacePool:
    """FlashInfer workspace buffers, one per (plan label, cg slot).

    Persistent: a wrapper is planned against the buffer it was built with, so
    these outlive any single step.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._size = int(
            os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")
        ) * 1024 * 1024
        self._buffers: dict[str, torch.Tensor] = {}

    def get(self, label: str, cg_slot: int | None = None) -> torch.Tensor:
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        if key not in self._buffers:
            self._buffers[key] = torch.empty(
                self._size, dtype=torch.uint8, device=self._device
            )
        return self._buffers[key]
