"""The cacheless attention resource: the spec-time factory.

Sibling of ``attn.base``'s ``AttentionManager``, for attention that is not
backed by a KV cache, e.g., an encoder tower attending within variable-length
segments of one packed forward. There is nothing to page, nothing to advance,
and nothing to carry between steps: the resource holds only planned wrappers.
"""

from mstar.engine.resources.attn.ragged.config import RaggedAttentionSpec
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo


class RaggedAttnManager(AttentionResource):
    # Remains abstract except for build; will build based on the backend.

    @classmethod
    def build(cls, spec: RaggedAttentionSpec, info: EngineResourceInfo):
        # NOTE: only flashinfer backend for now, to add more backends, we would
        # have to add ``backend: AttnBackend`` to the RaggedAttentionSpec.
        # Deferred so naming the spec does not load FlashInfer, and because
        # `flashinfer` imports this class back.
        from mstar.engine.resources.attn.ragged.flashinfer import (
            FlashInferRaggedManager,
        )

        config = spec.config
        if info.joint_comm_group is not None:
            config.shard(info.joint_comm_group.world_size)
        return FlashInferRaggedManager(
            device=info.device,
            dtype=info.kv_dtype,
            config=config,
        )
