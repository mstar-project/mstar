"""The linear-attention resource's spec-time factory.

``LinearAttnManager.build`` picks the manager for the configured variant (`gdn`,
`kda` or `mamba2`), by deferred import so naming one does not load the others.
"""

import logging

from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, Mamba2Geometry

logger = logging.getLogger(__name__)


class LinearAttnManager(AttentionResource):
    # Abstract except for `build`, which dispatches on the variant.

    @classmethod
    def build(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        # the pool's own config, not a copy; `shard` is idempotent so both
        # builders can call it
        pool_config = info.dependency(spec.config.recurrent_state).config
        if info.joint_comm_group is not None:
            pool_config.shard(info.joint_comm_group.world_size)

        backend = spec.config.backend
        variant = spec.config.variant
        if variant is LinearAttnVariant.MAMBA2:
            if backend is not LinearAttnBackend.FLASHINFER:
                raise ValueError(f"Unknown linear attention backend {backend!r}")
            # Its own kernels (Triton), planned against the same pool.
            from mstar.engine.resources.linear_attn.mamba2 import Mamba2Manager

            return Mamba2Manager(
                config=spec.config,
                geometry=Mamba2Geometry.from_blocks(pool_config.blocks),
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["ssm"].dtype,
                has_sink=not pool_config.disable_sink_slot,
                device=info.device,
            )

        # also checks the pool's shapes: `from_blocks` raises on another family's
        geometry = DeltaNetGeometry.from_blocks(pool_config.blocks)
        if variant is LinearAttnVariant.KDA:
            if backend is LinearAttnBackend.FLASHINFER:
                # the config's default, which is GDN's; KDA has no FlashInfer path
                backend = spec.config.backend = LinearAttnBackend.TRITON
            if backend is not LinearAttnBackend.TRITON:
                raise ValueError(f"KDA runs on the Triton backend, not {backend!r}")
            if pool_config.disable_sink_slot:
                raise ValueError(
                    "KDA needs the pool's sink slot: its kernels address padding "
                    "rows there. Unset RecurrentStateConfig.disable_sink_slot."
                )
            from mstar.engine.resources.linear_attn.kda import KDAManager

            return KDAManager(
                config=spec.config,
                geometry=geometry,
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["state"].dtype,
                device=info.device,
                speculative_tokens=DeltaNetGeometry.speculative_tokens_of(pool_config.blocks),
            )

        if backend is not LinearAttnBackend.FLASHINFER:
            raise ValueError(f"Unknown linear attention backend {backend!r}")
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
        raise ValueError(f"Unknown linear attention variant {variant!r}")
