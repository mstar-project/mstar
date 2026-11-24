"""The ``store:`` block of a paged KV cache's ``resources:`` entry.

What every backend takes sits at the top of the block, and each backend's own settings sit in a block
named after it, so a second backend adds its block without touching another's:

    store:
      backend: mooncake
      timeout_ms: 1000
      mooncake:
        master: 10.0.0.5:50051
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, NamedTuple


@dataclass(frozen=True)
class MooncakeStoreConfig:
    """The ``mooncake:`` block under ``store:``."""

    # the mooncake_master's address
    master: str
    # the transfer engine's metadata service: ``P2PHANDSHAKE``, or the master's HTTP metadata URL
    metadata: str = "P2PHANDSHAKE"
    protocol: str = "tcp"
    # RDMA devices to use; empty lets Mooncake pick
    rdma_devices: str = ""
    # this process's address as other store clients reach it; empty picks the interface that reaches the
    # master and a free port
    local_hostname: str = ""
    # host memory this process gives the store; 0 leaves the memory to separate store processes, which
    # keep pages across M* restarts
    segment_bytes: int = 0

    @classmethod
    def from_yaml(cls, block: Any) -> MooncakeStoreConfig:
        _check_keys("store.mooncake", block, {f.name for f in fields(cls)})
        if not block.get("master"):
            raise ValueError("store.mooncake: needs master, the mooncake_master address")
        protocol = str(block.get("protocol", "tcp")).lower()
        if protocol not in ("tcp", "rdma"):
            raise ValueError(f"store.mooncake: protocol must be tcp or rdma, got {block['protocol']!r}")
        config = cls(**{**block, "protocol": protocol})
        if isinstance(config.segment_bytes, bool) or not isinstance(config.segment_bytes, int):
            raise ValueError(f"store.mooncake: segment_bytes must be a whole number, got {config.segment_bytes!r}")
        _check_numbers("store.mooncake", {"segment_bytes": config.segment_bytes}, positive=())
        return config


# the longest a request waits for the store, however many bytes it reads
MAX_WAIT_S = 30.0


class Backend(NamedTuple):
    """One backend: the config of the block named after it, and its `KVStoreBackend` (``module:class``),
    imported only when a store opens."""
    config: type
    implementation: str


BACKENDS: dict[str, Backend] = {
    "mooncake": Backend(MooncakeStoreConfig, "mstar.engine.resources.kv.store.mooncake:MooncakeStore"),
}
_COMMON = ("backend", "timeout_ms", "floor_gbps", "min_tokens", "read_share", "write_share")


@dataclass(frozen=True)
class KVStoreConfig:
    """The ``store:`` block."""

    backend: str
    # the backend's own block
    options: MooncakeStoreConfig
    # How long a request waits for the store before it recomputes instead, from when it asks, so time
    # queued behind other store calls counts. A read also gets time for its bytes at ``floor_gbps``, so a
    # long read isn't cut off for being long. Failures are counted against a call's own run time.
    timeout_ms: int = 1000
    floor_gbps: float = 0.5
    # A prompt asks the store only when at least this many tokens of whole pages lie past what the GPU
    # and host hold (at least one page), as SGLang's HiCache does: a short continuation would wait for
    # the store's answer to save little.
    min_tokens: int = 256
    # The fractions of the host pool (``cpu_offload_pages``) store reads, and store writes (queued pages and
    # the put in flight), may each hold, so a slow store can't take the room offloads need. A hit longer than
    # the read share is cut, and its rest recomputed.
    read_share: float = 0.125
    write_share: float = 0.125

    @classmethod
    def from_yaml(cls, block: Any) -> KVStoreConfig:
        if not isinstance(block, Mapping):
            raise TypeError(f"store: expected a mapping, got {type(block).__name__}")
        backend = block.get("backend")
        if backend is None:
            raise ValueError(f"store: needs a backend, one of {sorted(BACKENDS)}")
        if backend not in BACKENDS:
            raise ValueError(f"store: unknown backend {backend!r}; one of {sorted(BACKENDS)}")
        _check_keys("store", block, {*_COMMON, backend})
        if backend not in block:
            raise ValueError(f"store: the {backend} backend needs a {backend}: block with its settings")
        config = cls(
            backend=backend, options=BACKENDS[backend].config.from_yaml(block[backend]),
            **{name: block[name] for name in _COMMON[1:] if name in block},
        )
        numbers = {name: getattr(config, name) for name in _COMMON[1:]}
        _check_numbers("store", numbers, positive=("timeout_ms", "floor_gbps", "read_share", "write_share"))
        if config.read_share + config.write_share > 0.5:
            raise ValueError("store: read_share and write_share together must leave at least half the host pool")
        return config

    def get_timeout(self, nbytes: int) -> float:
        """Seconds a get of ``nbytes`` may run: ``timeout_ms`` plus the bytes at ``floor_gbps``, so the limit
        follows the model's KV size, but never past `MAX_WAIT_S`, as in SGLang."""
        return min(MAX_WAIT_S, self.timeout_ms / 1000 + nbytes / (self.floor_gbps * 1e9))


def _check_keys(where: str, block: Any, known: set[str]) -> None:
    """Unknown keys are a TypeError, as for any unexpected keyword."""
    if not isinstance(block, Mapping):
        raise TypeError(f"{where}: expected a mapping, got {type(block).__name__}")
    unknown = sorted(block.keys() - known)
    if unknown:
        raise TypeError(f"{where}: unknown keys {unknown}; it takes {sorted(known)}")


def _check_numbers(where: str, numbers: dict[str, Any], positive: tuple[str, ...]) -> None:
    for name, value in numbers.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where}: {name} must be a number, got {value!r}")
        if value < 0 or (name in positive and value == 0):
            raise ValueError(f"{where}: {name} must be {'positive' if name in positive else 'at least 0'}")
