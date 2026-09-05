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
        # the KV resource's own config, not a copy; per-rank head counts, and
        # `shard` is idempotent so both builders can call it
        kv_config = info.dependency(spec.config.kv_cache).config
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
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
                    kv_config=kv_config,
                )
            _warn_dense_fallback(reason)
            backend = AttnBackend.FLASHINFER

        if backend == AttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.flashinfer import FlashInferManager

            return FlashInferManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
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
