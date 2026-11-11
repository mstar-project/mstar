"""A request's KV config as the conductor builds it, for a harness that drives
an engine without one."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from mstar.engine.resources.kv.config import KVReqConfig, KVSpec


def conductor_kv_config(spec: KVSpec, kv_layout: Mapping[str, Sequence], **conductor_config) -> KVReqConfig:
    """``kv_layout`` (``process_prompt``'s rows for the pool) and the rest of the
    conductor's config applied, then the pool's own say (`KVSpec.request_layout`,
    a verifying pool's scratch floor), as `Conductor` does before it checks a request."""
    cfg = KVReqConfig()
    cfg.apply_conductor_config(kv_layout=dict(kv_layout), **conductor_config)
    cfg.apply_spec(spec)
    return cfg
