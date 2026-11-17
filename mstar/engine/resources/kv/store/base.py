"""A store shared by processes that keeps host pages under their page keys.

The store sits under the host pool: a backend only ever reads or writes pages of the host pool's tensor,
and the KV cache decides what to store and when. Every backend method blocks: `exists`, `put` and `get` run
on the store threads (`StoreIO`), never on the worker's main thread, and `bind` and `close` on the caller's,
at startup and shutdown.
"""

from __future__ import annotations

import enum
import importlib
from abc import ABC, abstractmethod

import torch

from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.store.config import BACKENDS, KVStoreConfig

# Bump this with any change that makes the KV bytes for the same tokens, weights and cache config differ:
# positions, the kernels that write K/V, a page's layout on the host. The store outlives deployments, and
# nothing else in a store key names M*'s code, so without a bump a new release reads an older one's pages.
_FORMAT = "v1"


class GetResult(enum.Enum):
    """What happened to one page of a get."""
    ARRIVED = "arrived"
    # the store doesn't hold it, and nothing was transferred
    MISSING = "missing"
    # a transfer was attempted and failed; the backend stopped it before `get` returned
    FAILED = "failed"


class KVStoreBackend(ABC):
    """Keeps host pages under string keys, in a store that processes share. A call that fails as a whole
    raises; per-key errors are reported in the result."""

    @abstractmethod
    def bind(self, host_pages: torch.Tensor) -> None:
        """Register the host pool's tensor, laid out ``[layers, pages, ...]``. Every put and get names
        pages of it."""

    @abstractmethod
    def exists(self, keys: list[str]) -> list[bool]:
        """Whether the store holds each of ``keys``. Raises if the store can't say for some key."""

    @abstractmethod
    def put(self, keys: list[str], pages: list[int]) -> list[bool]:
        """Store host page ``pages[i]`` under ``keys[i]``. A key the store already has counts as stored."""

    @abstractmethod
    def get(self, keys: list[str], pages: list[int]) -> list[GetResult]:
        """Read ``keys[i]`` into host page ``pages[i]``, and say how each went. A get that returns has stopped
        every transfer it started, so nothing lands in ``pages`` afterwards; one that raises may still write
        any of them, and the caller never reuses those pages."""

    def close(self) -> None:  # noqa: B027 (optional for a backend that holds nothing)
        """Release what `bind` and the connection hold."""


def page_layout(host_pages: torch.Tensor, resource_key: str) -> str:
    """A short name for which KV cache a page belongs to and its byte layout, so a page another cache of the
    model stored, or one stored under another layout, never matches. Within a process each cache has an index
    of its own; in the store they share one namespace. It names the stored object (its layers in order, each a
    block of this shape and dtype), not the host tensor's order, so a host pool laid out another way that stores
    the same bytes must give the same name, or every stored page becomes a miss."""
    shape = (tuple(host_pages.shape[:1]), tuple(host_pages.shape[2:]), str(host_pages.dtype))
    return fingerprint(resource_key, *shape).hex()[:16]


def store_key(layout: str, key: bytes) -> str:
    """The store's name for the page the host index files under ``key``. The key already chains from a
    root naming the weights, preprocessing, cache config and salt."""
    return f"mstar-kv/{_FORMAT}/{layout}/{key.hex()}"


def make_store_backend(config: KVStoreConfig) -> KVStoreBackend:
    module, name = BACKENDS[config.backend].implementation.split(":")
    return getattr(importlib.import_module(module), name)(config.options)
