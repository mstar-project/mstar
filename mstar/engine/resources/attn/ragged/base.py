"""The cacheless attention resource: the spec-time factory.

Sibling of ``attn.base``'s ``AttentionManager``, for attention that is not
backed by a KV cache, e.g., an encoder tower attending within variable-length
segments of one packed forward. There is nothing to page, nothing to advance,
and nothing to carry between steps: the resource holds only planned wrappers.
"""

import torch

from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionSpec,
    RaggedAttnBackend,
    RaggedBlockCausalAttentionSpec,
    RaggedCrossAttentionSpec,
)
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.platforms import current_platform


def resolve_ragged_attn_backend(
    backend: RaggedAttnBackend, device: torch.device,
) -> RaggedAttnBackend:
    """Ragged counterpart of ``attn.base.resolve_attn_backend``. On ROCm,
    ``AUTO`` is AITER's varlen flash attention."""
    if backend is not RaggedAttnBackend.AUTO:
        return backend
    if current_platform.is_rocm():
        from mstar.engine.resources.attn import aiter as rocm
        from mstar.engine.resources.attn.ragged import aiter as rocm_ragged

        aiter_reason = rocm_ragged.aiter_varlen_unavailable_reason()
        if aiter_reason is None:
            return RaggedAttnBackend.AITER
        flashinfer_reason = rocm.flashinfer_rocm_unavailable_reason()
        if flashinfer_reason is None:
            return RaggedAttnBackend.FLASHINFER
        raise RuntimeError(
            "No ragged attention backend on this ROCm install: AITER's varlen "
            f"flash attention is not importable ({aiter_reason}), nor is AMD's "
            f"FlashInfer build ({flashinfer_reason})."
        )
    return RaggedAttnBackend.FLASHINFER


def _build_ragged(spec, info: EngineResourceInfo, manager_cls, **kwargs):
    config = spec.config
    if info.joint_comm_group is not None:
        config.shard(info.joint_comm_group.world_size)
    dtype = config.dtype if config.dtype is not None else info.kv_dtype
    if dtype is None:
        raise ValueError(
            f"ragged attention {spec.resource_key!r}: no activation dtype. Set "
            "RaggedAttentionConfig.dtype on a node without a KV-backed attention "
            "(the engine has no KV dtype to fall back on)."
        )
    return manager_cls(device=info.device, dtype=dtype, config=config, **kwargs)


class RaggedAttnManager(AttentionResource):
    prefix_skip_safe = True

    # Remains abstract except for build; will build based on the backend.

    @classmethod
    def build(cls, spec: RaggedAttentionSpec, info: EngineResourceInfo):
        backend = resolve_ragged_attn_backend(spec.config.backend, info.device)
        if backend is RaggedAttnBackend.FLASHINFER:
            # Deferred so naming the spec does not load FlashInfer, and because
            # `flashinfer` imports this class back.
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedManager,
            )

            return _build_ragged(spec, info, FlashInferRaggedManager)
        if backend is RaggedAttnBackend.AITER:
            from mstar.engine.resources.attn.ragged.aiter import AiterRaggedManager

            return _build_ragged(spec, info, AiterRaggedManager)
        raise ValueError(f"Unknown ragged attention backend {backend!r}")


class RaggedCrossAttnManager(AttentionResource):
    """Cacheless cross-attention between two spans of each request; see
    ``RaggedCrossAttentionSpec``."""

    prefix_skip_safe = True

    @classmethod
    def build(cls, spec: RaggedCrossAttentionSpec, info: EngineResourceInfo):
        backend = resolve_ragged_attn_backend(spec.config.backend, info.device)
        if backend is RaggedAttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedCrossManager,
            )

            return _build_ragged(spec, info, FlashInferRaggedCrossManager)
        if backend is RaggedAttnBackend.AITER:
            from mstar.engine.resources.attn.ragged.aiter import AiterRaggedCrossManager

            return _build_ragged(spec, info, AiterRaggedCrossManager)
        raise ValueError(f"Unknown ragged cross attention backend {backend!r}")


class RaggedBlockCausalAttnManager(AttentionResource):
    """Cacheless block-causal self-attention within each span; see
    ``RaggedBlockCausalAttentionSpec``."""

    prefix_skip_safe = True

    @classmethod
    def build(cls, spec: RaggedBlockCausalAttentionSpec, info: EngineResourceInfo):
        backend = resolve_ragged_attn_backend(spec.config.backend, info.device)
        if backend is RaggedAttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedBlockCausalManager,
            )

            return _build_ragged(
                spec, info, FlashInferRaggedBlockCausalManager, block_size=spec.block_size,
            )
        raise ValueError(f"Unknown ragged block-causal attention backend {backend!r}")
