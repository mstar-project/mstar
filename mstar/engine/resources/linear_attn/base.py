"""The linear-attention resource's spec-time factory.

``LinearAttnManager.build`` picks the manager for the configured variant.
"""

import logging

from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, Mamba2Geometry
from mstar.engine.rows import RowState

logger = logging.getLogger(__name__)


class LinearAttnManager(AttentionResource):
    # Abstract except for `build`, which dispatches on the variant.

    def row_state(self, space, node_name: str, graph_walk: str) -> RowState:
        # plans from the pool's addressing, which the pool's rows hold
        return RowState()

    @classmethod
    def build(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        # the pool's own config, not a copy; `shard` is idempotent so both
        # builders can call it
        pool_config = info.dependency(spec.config.recurrent_state).config
        if info.joint_comm_group is not None:
            pool_config.shard(info.joint_comm_group.world_size)

        backend = spec.config.backend
        if backend is not LinearAttnBackend.FLASHINFER:
            raise ValueError(f"Unknown linear attention backend {backend!r}")

        variant = spec.config.variant
        if variant is LinearAttnVariant.MAMBA2:
            # Its own kernels (Triton), planned against the same pool.
            from mstar.engine.resources.linear_attn.mamba2 import Mamba2Manager

            return Mamba2Manager(
                config=spec.config,
                geometry=Mamba2Geometry.from_blocks(pool_config.blocks),
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["ssm"].resolved_dtype(info.kv_dtype),
                has_sink=not pool_config.disable_sink_slot,
                device=info.device,
            )

        # also checks the pool's shapes: `from_blocks` raises on another family's
        geometry = DeltaNetGeometry.from_blocks(pool_config.blocks)
        if variant is LinearAttnVariant.GDN:
            from mstar.engine.resources.linear_attn.gdn import GDNManager

            return GDNManager(
                config=spec.config,
                max_rows_in_combined=info.max_rows_in_combined,
                geometry=geometry,
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["state"].resolved_dtype(info.kv_dtype),
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
