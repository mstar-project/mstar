"""The linear-attention resource's spec-time factory.

The variants themselves live beside this — `gdn`, and `kda` when it lands —
and ``LinearAttnManager.build`` reaches them by deferred import, so naming one
in a spec does not load the others.
"""

import logging

from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry

logger = logging.getLogger(__name__)


class LinearAttnManager(AttentionResource):
    # Remains abstract except for build; will build based on the variant.

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    @classmethod
    def build(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        # the pool's own config, not a copy; per-rank shapes, and `shard` is
        # idempotent so both builders can call it
        pool_config = info.dependency(spec.config.recurrent_state).config
        if info.joint_comm_group is not None:
            pool_config.shard(info.joint_comm_group.world_size)

        # Reading geometry off the pool is also the check that the two were
        # built for the same model: `from_blocks` raises on shapes that are not
        # this family's.
        geometry = DeltaNetGeometry.from_blocks(pool_config.blocks)

        backend = spec.config.backend
        if backend is not LinearAttnBackend.FLASHINFER:
            raise ValueError(f"Unknown linear attention backend {backend!r}")

        variant = spec.config.variant
        if variant is LinearAttnVariant.GDN:
            from mstar.engine.resources.linear_attn.gdn import GDNManager

            return GDNManager(
                config=spec.config,
                geometry=geometry,
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["state"].dtype,
                has_sink=not pool_config.disable_sink_slot,
                device=info.device,
            )
        if variant is LinearAttnVariant.KDA:
            raise NotImplementedError(
                "KDA shares this pool's state layout, but its kernels take a "
                "per-K-channel gate the GDN marshalling does not build, and "
                "FlashInfer ships no chunked KDA prefill. Follow-up."
            )
        raise ValueError(f"Unknown linear attention variant {variant!r}")
