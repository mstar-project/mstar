"""Tensor storage, split so the bookkeeping half can have a Rust backend.

``TensorStore`` holds the ``torch.Tensor`` objects and stays Python: Rust must
never own a tensor, or freeing a record inside a ``py.allow_threads`` section
would drop a Python object without the GIL.

``TensorBookkeeping`` holds everything else -- refcounts, the persist flag,
whether the memory has been registered for remote reads -- and sees nothing but
integer uuids. That is what gets ported, so ``complete_and_route_batch`` can
adjust refcounts without a hop back to Python.

Uuids are globally unique now (see ``tensor_uuid``), so nothing here is keyed by
request. The one exception is ``_rid_to_uuids``, which exists only so a request
teardown can find what to free.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from mstar.utils.containers import ParallelList

NameToTensorList = dict[str, list[torch.Tensor]]


@dataclass
class ReferenceInfo:
    ref_cnt: int = 0
    persist: bool = False
    mem_registered: bool = False


class TensorBookkeeping(ABC):
    """Per-uuid reference state. Never sees a tensor or a request id.

    The batched forms are what the graph runtime calls; the defaults loop so a
    backend only overrides where batching buys something.
    """

    @abstractmethod
    def put_tensor(self, uuid: int):
        pass

    def put_tensor_batch(self, uuids: list[int]):
        for uuid in uuids:
            self.put_tensor(uuid)

    @abstractmethod
    def forget_tensor(self, uuid: int):
        """Drop the record. The caller frees the tensor itself."""
        pass

    @abstractmethod
    def is_tracked(self, uuid: int) -> bool:
        pass

    # -- refcounts ----------------------------------------------------------

    @abstractmethod
    def increment_ref(self, uuid: int, n: int = 1):
        pass

    def increment_ref_batch(self, counts: ParallelList[int, int]):
        for uuid, n in counts:
            self.increment_ref(uuid, n)

    @abstractmethod
    def dereference(self, uuid: int, n: int = 1):
        pass

    def dereference_batch(self, counts: ParallelList[int, int]):
        for uuid, n in counts:
            self.dereference(uuid, n)

    # -- flags --------------------------------------------------------------

    @abstractmethod
    def set_persist(self, uuid: int, persist: bool):
        pass

    def set_persist_batch(self, uuids: list[int], persist: bool):
        for uuid in uuids:
            self.set_persist(uuid, persist)

    @abstractmethod
    def set_mem_registered(self, uuid: int, mem_registered: bool):
        pass

    @abstractmethod
    def is_registered(self, uuid: int) -> bool:
        pass

    # -- gc -----------------------------------------------------------------

    @abstractmethod
    def can_gc(self, uuid: int) -> bool:
        """No references left and not being persisted for the conductor."""
        pass

    def collectable(self, uuids: list[int]) -> list[int]:
        """Which of ``uuids`` are now free. One call instead of one per tensor
        after a batch of refcount changes."""
        return [uuid for uuid in uuids if self.can_gc(uuid)]


class PythonTensorBookkeeping(TensorBookkeeping):
    def __init__(self):
        self._ref_info: dict[int, ReferenceInfo] = {}

    def put_tensor(self, uuid: int):
        self._ref_info[uuid] = ReferenceInfo()

    def forget_tensor(self, uuid: int):
        self._ref_info.pop(uuid, None)

    def is_tracked(self, uuid: int) -> bool:
        return uuid in self._ref_info

    def increment_ref(self, uuid: int, n: int = 1):
        info = self._ref_info.get(uuid)
        if info is None:
            return
        assert n >= 0, f"Tried to increment tensor {uuid} reference by {n}"
        info.ref_cnt += n

    def dereference(self, uuid: int, n: int = 1):
        info = self._ref_info.get(uuid)
        if info is None:
            return
        info.ref_cnt -= n

    def set_persist(self, uuid: int, persist: bool):
        info = self._ref_info.get(uuid)
        if info is not None:
            info.persist = persist

    def set_mem_registered(self, uuid: int, mem_registered: bool):
        info = self._ref_info.get(uuid)
        if info is not None:
            info.mem_registered = mem_registered

    def is_registered(self, uuid: int) -> bool:
        info = self._ref_info.get(uuid)
        return info is not None and info.mem_registered

    def can_gc(self, uuid: int) -> bool:
        info = self._ref_info.get(uuid)
        return info is not None and info.ref_cnt <= 0 and not info.persist


class TensorStore:
    """The tensors themselves, plus which request owns each one.

    Reference state lives in ``bookkeeping``; the delegating methods here exist
    so callers have one object to talk to.
    """

    def __init__(self, bookkeeping: TensorBookkeeping | None = None):
        # {UUID -> tensor}
        self._tensors: dict[int, torch.Tensor] = {}
        # Only for teardown: which uuids a request is responsible for.
        self._rid_to_uuids: dict[int, set[int]] = {}
        self.bookkeeping = bookkeeping or PythonTensorBookkeeping()

    # -- tensors ------------------------------------------------------------

    def get_tensor(self, uuid: int) -> torch.Tensor:
        return self._tensors[uuid]

    def put_tensor(self, request_id: int, uuid: int, tensor: torch.Tensor):
        self._tensors[uuid] = tensor
        self._rid_to_uuids.setdefault(request_id, set()).add(uuid)
        self.bookkeeping.put_tensor(uuid)

    def put_tensor_batch(
        self, request_id: int, tensors: ParallelList[int, torch.Tensor],
    ):
        owned = self._rid_to_uuids.setdefault(request_id, set())
        for uuid, tensor in tensors:
            self._tensors[uuid] = tensor
            owned.add(uuid)
        self.bookkeeping.put_tensor_batch(tensors.keys)

    def check_uuid_presence(self, uuid: int) -> bool:
        return uuid in self._tensors

    def remove_tensor(self, uuid: int):
        if self._tensors.pop(uuid, None) is None:
            return
        self.bookkeeping.forget_tensor(uuid)
        # The owning request is usually known by the caller, but a slice can
        # outlive its producer's entry, so find it rather than require it.
        for owned in self._rid_to_uuids.values():
            if uuid in owned:
                owned.discard(uuid)
                break

    def get_all_uuids(self, request_id: int) -> list[int]:
        return list(self._rid_to_uuids.get(request_id, ()))

    def remove_request(self, request_id: int) -> list[int]:
        """Forget the request and return the uuids it owned, so the caller can
        run its own per-uuid cleanup (shm files, arena slots) before they go."""
        uuids = list(self._rid_to_uuids.pop(request_id, ()))
        for uuid in uuids:
            self._tensors.pop(uuid, None)
            self.bookkeeping.forget_tensor(uuid)
        return uuids

    # -- bookkeeping passthrough -------------------------------------------

    def can_gc(self, uuid: int) -> bool:
        return self.bookkeeping.can_gc(uuid)

    def is_registered(self, uuid: int) -> bool:
        return self.bookkeeping.is_registered(uuid)

    def increment_ref(self, uuid: int, n: int = 1):
        self.bookkeeping.increment_ref(uuid, n)

    def dereference(self, uuid: int, n: int = 1):
        self.bookkeeping.dereference(uuid, n)

    def set_metadata(
        self, uuid: int,
        persist: bool | None = None,
        mem_registered: bool | None = None,
    ):
        if persist is not None:
            self.bookkeeping.set_persist(uuid, persist)
        if mem_registered is not None:
            self.bookkeeping.set_mem_registered(uuid, mem_registered)
