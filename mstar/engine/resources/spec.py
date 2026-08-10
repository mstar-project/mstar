"""Resource declarations models hand to the engine.

The engine builds each node's resources once, at load time, from these
specs. A spec names what to build and its parameters; the model declares,
the engine constructs. The default declaration wraps a model's KV cache
configs unchanged, so a model only overrides it to add resources beyond
what those configs already describe.
"""

from dataclasses import dataclass, field

import torch

from mstar.engine.kv_store import KVCacheConfig


@dataclass(frozen=True)
class ScratchKVSpec:
    """A fixed-shape scratch cache: overwritten every step, slot-indexed
    by batch position, no per-request lifetime. A ``dtype`` of None means
    the engine's KV cache dtype."""
    shape: tuple[int, ...]
    dtype: "torch.dtype | None" = None


@dataclass
class NodeResourceSpec:
    """One KV cache group's resource declaration.

    The cache config derives the self-attention pool, the attention
    backend, the rope embedder, and the cross-attention pools, exactly as
    it always has. ``scratch`` adds keyed fixed-shape caches built
    alongside them (resource key to spec).

    TODO: once get_kv_cache_config retires, allow several named KV cache
    configs per node and split the cross-attention and rope settings out
    of KVCacheConfig into their own spec entries.
    """
    kv_cache_config: KVCacheConfig
    scratch: dict[str, ScratchKVSpec] = field(default_factory=dict)
