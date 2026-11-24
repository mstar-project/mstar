"""A KV store backend in a dict, and a host pool on CPU, for the KV store tests. Two KV caches can share one
``pages`` dict to stand in for two processes."""

from __future__ import annotations

import torch

from mstar.engine.resources.kv.arena import new_page_arena
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.kv.store.base import GetResult, KVStoreBackend


class MemoryStore(KVStoreBackend):
    def __init__(self, pages: dict[str, torch.Tensor] | None = None):
        self.pages = {} if pages is None else pages
        self.calls: list[str] = []
        # how many keys each get asked for
        self.got: list[int] = []
        self._host: torch.Tensor | None = None

    def bind(self, host_pages: torch.Tensor) -> None:
        self._host = host_pages

    def exists(self, keys: list[str]) -> list[bool]:
        self.calls.append("exists")
        return [key in self.pages for key in keys]

    def put(self, keys: list[str], pages: list[int]) -> list[bool]:
        self.calls.append("put")
        for key, page in zip(keys, pages, strict=True):
            self.pages.setdefault(key, self._host[:, page].clone())
        return [True] * len(keys)

    def get(self, keys: list[str], pages: list[int]) -> list[GetResult]:
        self.calls.append("get")
        self.got.append(len(keys))
        found = []
        for key, page in zip(keys, pages, strict=True):
            stored = self.pages.get(key)
            ok = stored is not None and stored.shape == self._host[:, page].shape
            if ok:
                self._host[:, page].copy_(stored)
            found.append(GetResult.ARRIVED if ok else GetResult.MISSING)
        return found


class _Done:
    def query(self):
        return True

    def synchronize(self):
        pass


class CpuHostPool(CPUPagePool):
    """A host pool on CPU whose copies run at once, so every copy's event has already completed."""

    def __init__(self, max_cpu_pages: int, page_shape: tuple[int, ...], layers: int):
        self.arena = new_page_arena(max_cpu_pages)
        self.index = PrefixIndex(self.arena)
        self.arena.index = self.index
        self.offloaded, self.reserved = {}, {}
        self._unmoved = 0
        self._refused_logged_at = None
        self._refused_unlogged = 0
        self.dropped = self.written = self.cancelled = self.guaranteed = 0
        self._dropped_logged_at = self._guaranteed_logged_at = None
        self.cpu_kv_cache = torch.zeros((layers, max_cpu_pages, *page_shape))

    def copy_out(self, gpu_kv_cache, copies):
        self.cpu_kv_cache[:, copies.cpu] = gpu_kv_cache[:, copies.gpu]

    def copy_in(self, gpu_kv_cache, copies):
        gpu_kv_cache[:, copies.gpu] = self.cpu_kv_cache[:, copies.cpu]

    write = copy_out

    def record(self):
        return _Done()

    record_write = record

    def copies_queued(self):
        return [_Done()]

    def sync(self):
        pass
