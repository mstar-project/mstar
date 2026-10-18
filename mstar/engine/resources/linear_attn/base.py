"""The linear-attention resource's spec-time factory.

``LinearAttnManager.build`` picks the manager for the configured variant (`gdn`,
`kda` or `mamba2`), by deferred import so naming one does not load the others,
and for GDN the kernels it runs.
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
        pool_config.shard(info.head_shards)

        backend = spec.config.backend
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
        if variant is LinearAttnVariant.KDA:
            if backend in (LinearAttnBackend.AUTO, LinearAttnBackend.FLASHINFER):
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
                state_dtype=pool_config.blocks["state"].resolved_dtype(info.kv_dtype),
                device=info.device,
                speculative_tokens=DeltaNetGeometry.speculative_tokens_of(pool_config.blocks),
                conv_dtype=pool_config.blocks["conv"].resolved_dtype(info.kv_dtype),
            )

        if variant is LinearAttnVariant.GDN:
            from mstar.engine.resources.linear_attn.gdn import GDNManager
            from mstar.engine.resources.linear_attn.gdn_kernels import build_gdn_kernels

            backend = resolve_linear_attn_backend(backend, info.device)
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
        raise ValueError(f"Unknown linear attention variant {variant!r}")
