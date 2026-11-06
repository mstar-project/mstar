"""The simulated card's memory: torch's caching allocator's accounting of the emulated
tensors' storages, and the fake addresses of the value-less ones. ``torch.cuda``'s memory
API (:mod:`mstar_sim.cuda`) reads these figures."""

from __future__ import annotations

import atexit
import threading
import weakref

import torch

from mstar_sim import clock
from mstar_sim.cost import Cache

# Simulated device memory, as torch's caching allocator accounts it. Each device has a
# default pool and the private pools graphs capture into. A pool's ``live`` bytes are
# its tensors' storages; ``reserved`` is what it holds from the card: its high-water
# mark, until the cache is emptied (a freed block is reused within its pool, never
# by another). ``graphs`` counts the graphs using a private pool; one no graph uses
# gives its free bytes back on ``empty_cache``, and goes once nothing in it lives.
# ``recording``: the thread whose allocations on its device go to the pool (a capture's,
# or a ``_cuda_beginAllocateToPool`` caller's), or None.
DEFAULT_POOL = (0, 0)


class Pool:
    __slots__ = ("device", "handle", "live", "reserved", "graphs", "recording")

    def __init__(self, device: int, handle: tuple):
        self.device, self.handle = device, handle
        self.live = self.reserved = self.graphs = 0
        self.recording: int | None = None


POOLS: dict[tuple[int, tuple], Pool] = {}
_STORAGES: dict[int, list] = {}  # storage key: [bytes, references, fake address range, pool or None]
ALLOCATED: dict[int, list[int]] = {}  # device index: [current, peak] bytes in tensors
RESERVED_PEAK: dict[int, int] = {}
# reentrant: a finalizer can run (garbage collection) while this thread holds it
_track_lock = threading.RLock()


def pool(index: int, handle=DEFAULT_POOL) -> Pool:
    key = (index, tuple(handle))
    with _track_lock:
        found = POOLS.get(key)
        if found is None:
            found = POOLS[key] = Pool(*key)
        return found


def reserved(index: int) -> int:
    with _track_lock:
        return sum(p.reserved for p in POOLS.values() if p.device == index)


def release_cached(index: int) -> None:
    """``empty_cache``: the default pool, and private pools no graph uses, give back
    what their tensors don't hold."""
    with _track_lock:
        for key, p in list(POOLS.items()):
            if p.device == index and (p.handle == DEFAULT_POOL or p.graphs == 0):
                p.reserved = p.live
                if p.handle != DEFAULT_POOL and p.live == 0 and p.recording is None:
                    del POOLS[key]


def _release(key: int) -> None:
    with _track_lock:
        entry = _STORAGES.get(key)
        if entry is not None:
            entry[1] -= 1
            if entry[1] == 0:
                del _STORAGES[key]
                if entry[3] is not None:
                    entry[3].live -= entry[0]
                    ALLOCATED[entry[3].device][0] -= entry[0]


def _allocate(index: int, nbytes: int) -> Pool:
    """The pool a new storage of ``nbytes`` on device ``index`` comes from: the one this
    thread records into on that device, else the default. Growing a pool past the card
    (its memory fraction) raises out of memory, after emptying the cache once, as torch's
    allocator retries."""
    thread = threading.get_ident()
    into = next((p for p in POOLS.values() if p.recording == thread and p.device == index), None) or pool(index)
    if into.live + nbytes > into.reserved:
        from mstar_sim import cuda

        grow = _segment(nbytes)
        limit = cuda.memory_limit(index)
        if reserved(index) + grow > limit:
            release_cached(index)
            if into.live + nbytes <= into.reserved:
                return into
            if reserved(index) + grow > limit:
                held = ", ".join(f"{'default' if p.handle == DEFAULT_POOL else p.handle}: {p.reserved / 2**20:.0f}"
                                 for p in POOLS.values() if p.device == index and p.reserved)
                from mstar_sim.device import gpu_error

                raise gpu_error(torch.cuda.OutOfMemoryError(
                    f"CUDA out of memory. Tried to allocate {nbytes / 2**20:.2f} MiB. GPU {index} has a total "
                    f"capacity of {limit / 2**30:.2f} GiB, of which {(limit - reserved(index)) / 2**20:.2f} MiB "
                    f"is free (mstar_sim: MiB reserved per pool: {held})"))
        into.reserved += grow
        RESERVED_PEAK[index] = max(RESERVED_PEAK.get(index, 0), reserved(index))
    return into


def _segment(nbytes: int) -> int:
    """The segment torch's caching allocator reserves from the card for a block it has no room
    for (``CUDACachingAllocator``'s ``get_allocation_size``): 2 MiB for a small block (up to
    1 MiB), 20 MiB up to 10 MiB, else the block rounded up to 2 MiB."""
    if nbytes <= 1 << 20:
        return 2 << 20
    if nbytes < 10 << 20:
        return 20 << 20
    return -(-nbytes // (2 << 20)) * (2 << 20)


def track(data: torch.Tensor, index: int) -> None:
    """Count ``data``'s storage, on device ``index``, while it lives. The weak reference
    hangs on the payload, not the wrapper: module moves swap wrappers, and a swap refuses
    weakrefs. A storage opened from another process (CUDA IPC) is that process's memory."""
    if data._base is not None or getattr(data, "_sim_tracked", False):
        return  # a view: its base holds (and counts) the storage
    storage = data.untyped_storage()
    key = storage._cdata
    with _track_lock:
        entry = _STORAGES.get(key)
        if entry is None:
            # a block is a multiple of 512 bytes, as the caching allocator rounds it
            nbytes = -(-storage.nbytes() // 512) * 512
            entry = [nbytes, 0, _fake_range(nbytes), None]
            if not getattr(storage, "_sim_imported", False) and not clock.in_kernel():
                _count(entry, index)  # out of memory: no storage to remember
            _STORAGES[key] = entry
        entry[1] += 1
    data._sim_tracked = True
    _LIVE[weakref.ref(data, _released)] = key


# each tracked payload's weak reference, to the storage it counts: what a dead one releases
# (emptied at exit, before the modules a release reads are torn down)
_LIVE: dict = {}
atexit.register(_LIVE.clear)


def _released(ref, live=_LIVE, begin=clock.compute_begin, end=clock.compute_end) -> None:
    """A tracked payload is gone (a free): the emulator's work, not the host's."""
    key = live.pop(ref, None)
    if key is None:
        return
    depth = begin()
    try:
        _release(key)
    finally:
        end(depth)


def _count(entry: list, index: int) -> None:
    into = entry[3] = _allocate(index, entry[0])
    into.live += entry[0]
    used = ALLOCATED.setdefault(into.device, [0, 0])
    used[0] += entry[0]
    used[1] = max(used[1], used[0])


def count_storage(data: torch.Tensor, index: int) -> None:
    """Count a storage made inside a stand-in kernel (``clock.in_kernel``), now that it
    is known to be one of the kernel's outputs."""
    storage = data.untyped_storage()
    with _track_lock:
        entry = _STORAGES.get(storage._cdata)
        if entry is not None and entry[3] is None and not getattr(storage, "_sim_imported", False):
            _count(entry, index)


# Value-less storages get addresses in [2^60, 2^62): unique per storage, never
# reused, and non-canonical on x86-64 and arm64, so code that compares or offsets
# pointers sees distinct storages, and code that reads through one faults loudly
# (``faulthandler`` prints the Python stack) instead of reading host memory.
_FAKE_NEXT = [1 << 60]
_FAKE_UNTRACKED = Cache()  # by storage address, which nothing tells us is free again: bounded


def _fake_range(nbytes: int) -> int:
    with _track_lock:
        base = _FAKE_NEXT[0]
        _FAKE_NEXT[0] += -(-max(nbytes, 1) // 512) * 512 + 512
    return base


def device_of(storage) -> int:
    """The index of the emulated GPU a storage was counted on (0 if it wasn't)."""
    entry = _STORAGES.get(storage._cdata)
    return entry[3].device if entry is not None and entry[3] is not None else 0


def fake_base(storage) -> int:
    """The fake address of a value-less storage."""
    entry = _STORAGES.get(storage._cdata)
    if entry is not None:
        return entry[2]
    base = _FAKE_UNTRACKED.get(storage._cdata)
    if base is None:
        base = _FAKE_UNTRACKED.put(storage._cdata, _fake_range(storage.nbytes()))
    return base
