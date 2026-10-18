"""The linear-attention resource's spec-time factory.

``LinearAttnManager.build`` picks the manager for the configured variant, and
for GDN the kernels it runs.
"""

import logging

import torch

from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, Mamba2Geometry
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)


def resolve_linear_attn_backend(
    backend: LinearAttnBackend, device: torch.device,
) -> LinearAttnBackend:
    """The backend a spec's ``backend`` runs on ``device``.

    The one place ``AUTO`` becomes a kernel choice: AITER on an AMD GPU, where
    FlashInfer has no GDN kernels, and FlashInfer everywhere else, as before
    ``AUTO`` existed (a resource on the CPU device runs no kernel at all). Any
    other value is returned as is, so a spec or YAML that names a backend gets
    that backend.
    """
    if backend is not LinearAttnBackend.AUTO:
        return backend
    if device.type == "cuda" and current_platform.is_rocm():
        return LinearAttnBackend.AITER
    return LinearAttnBackend.FLASHINFER


class LinearAttnManager(AttentionResource):
    # Abstract except for `build`, which dispatches on the variant.

    @classmethod
    def build(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        # the pool's own config, not a copy; `shard` is idempotent so both
        # builders can call it
        pool_config = info.dependency(spec.config.recurrent_state).config
        if info.joint_comm_group is not None:
            pool_config.shard(info.joint_comm_group.world_size)

        backend = resolve_linear_attn_backend(spec.config.backend, info.device)

        variant = spec.config.variant
        if variant is LinearAttnVariant.MAMBA2:
            # Its own kernels (Triton) on every backend, planned against the
            # same pool.
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
            from mstar.engine.resources.linear_attn.gdn_kernels import build_gdn_kernels

            state_dtype = pool_config.blocks["state"].resolved_dtype(info.kv_dtype)
            has_sink = not pool_config.disable_sink_slot
            kernels = build_gdn_kernels(
                backend,
                config=spec.config,
                geometry=geometry,
                state_dtype=state_dtype,
                has_sink=has_sink,
                device=info.device,
            )
            return GDNManager(
                config=spec.config,
                max_rows_in_combined=info.max_rows_in_combined,
                geometry=geometry,
                num_layers=pool_config.num_layers,
                state_dtype=state_dtype,
                has_sink=has_sink,
                device=info.device,
                kernels=kernels,
            )
        if variant is LinearAttnVariant.KDA:
            raise NotImplementedError(
                "KDA shares this pool's state layout, but its kernels take a "
                "per-K-channel gate the GDN marshalling does not build, and "
                "FlashInfer ships no chunked KDA prefill. Follow-up."
            )
        raise ValueError(f"Unknown linear attention variant {variant!r}")
