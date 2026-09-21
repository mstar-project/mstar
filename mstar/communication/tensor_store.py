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

from mstar.graph.base import TensorPointerInfo
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
    backend only overrides where batching buys something. They take two
    parallel lists rather than a ParallelList because that is what crosses the
    Rust boundary -- pyo3 converts a list of ints directly, where a NamedTuple
    would need unpacking on every call.
    """

    @abstractmethod
    def put_tensor(self, uuid: int, info: TensorPointerInfo):
        pass

    def put_tensor_batch(
        self, uuids: list[int], infos: list[TensorPointerInfo]
    ):
        for uuid, info in zip(uuids, infos, strict=True):
            self.put_tensor(uuid, info)

    @abstractmethod
    def update_info(self, uuid: int, info: TensorPointerInfo):
        """Rebind the descriptor for an existing uuid.

        Needed where the tensor lands before its final descriptor exists: a
        slice re-points an arriving info at a freshly minted uuid, and a fan-in
        consolidation mints one for a tensor it has just concatenated.
        """
        pass

    def update_info_batch(
        self, uuids: list[int], infos: list[TensorPointerInfo]
    ):
        for uuid, info in zip(uuids, infos, strict=True):
            self.update_info(uuid, info)

    @abstractmethod
    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        """The descriptor a peer needs to read this tensor.

        This is what lets an edge be rebuilt from uuids alone: ingestion and
        routing carry uuids, and anything that has to put a descriptor back on
        the wire (a disaggregated loop re-emitting its external inputs, say)
        looks it up here.
        """
        pass

    def get_info_batch(self, uuids: list[int]) -> list[TensorPointerInfo | None]:
        return [self.get_info(uuid) for uuid in uuids]

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

    def increment_ref_batch(self, uuids: list[int], counts: list[int]):
        for uuid, n in zip(uuids, counts, strict=True):
            self.increment_ref(uuid, n)

    @abstractmethod
    def dereference(self, uuid: int, n: int = 1):
        pass

    def dereference_batch(self, uuids: list[int], counts: list[int]):
        for uuid, n in zip(uuids, counts, strict=True):
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
        self._tensor_info: dict[int, TensorPointerInfo] = {}

    def put_tensor(self, uuid: int, info: TensorPointerInfo):
        self._ref_info[uuid] = ReferenceInfo()
        self._tensor_info[uuid] = info

    def update_info(self, uuid: int, info: TensorPointerInfo):
        self._tensor_info[uuid] = info

    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        return self._tensor_info.get(uuid)

    def forget_tensor(self, uuid: int):
        self._ref_info.pop(uuid, None)
        self._tensor_info.pop(uuid, None)

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
        # Only for teardown: which uuids a request is responsible for, and the
        # reverse so a single removal does not scan every request. A sliced
        # tensor can outlive its producer's entry (_slice_existing_tensor mints
        # a new uuid), so ownership is not derivable from the uuid alone.
        self._rid_to_uuids: dict[int, set[int]] = {}
        self._uuid_to_rid: dict[int, int] = {}
        self.bookkeeping = bookkeeping or PythonTensorBookkeeping()

    # -- tensors ------------------------------------------------------------

    def get_tensor(self, uuid: int) -> torch.Tensor:
        return self._tensors[uuid]

    def put_tensor(
        self, rid: int, uuid: int,
        tensor: torch.Tensor,
        info: TensorPointerInfo
    ):
        self._tensors[uuid] = tensor
        self._rid_to_uuids.setdefault(rid, set()).add(uuid)
        self._uuid_to_rid[uuid] = rid
        self.bookkeeping.put_tensor(uuid, info)

    def put_tensor_batch(
        self, rid: int, tensors: ParallelList[int, torch.Tensor],
        info: list[TensorPointerInfo],
    ):
        owned = self._rid_to_uuids.setdefault(rid, set())
        for uuid, tensor in tensors:
            self._tensors[uuid] = tensor
            owned.add(uuid)
            self._uuid_to_rid[uuid] = rid
        self.bookkeeping.put_tensor_batch(tensors.keys, info)

    def check_uuid_presence(self, uuid: int) -> bool:
        return uuid in self._tensors

    def remove_tensor(self, uuid: int):
        if self._tensors.pop(uuid, None) is None:
            return
        self.bookkeeping.forget_tensor(uuid)
        rid = self._uuid_to_rid.pop(uuid, None)
        if rid is not None:
            owned = self._rid_to_uuids.get(rid)
            if owned is not None:
                owned.discard(uuid)
                if not owned:
                    del self._rid_to_uuids[rid]

    def get_all_uuids(self, rid: int) -> list[int]:
        return list(self._rid_to_uuids.get(rid, ()))

    def remove_request(self, rid: int) -> list[int]:
        """Forget the request and return the uuids it owned, so the caller can
        run its own per-uuid cleanup (shm files, arena slots) before they go."""
        uuids = list(self._rid_to_uuids.pop(rid, ()))
        for uuid in uuids:
            self._tensors.pop(uuid, None)
            self._uuid_to_rid.pop(uuid, None)
            self.bookkeeping.forget_tensor(uuid)
        return uuids

    # -- bookkeeping passthrough -------------------------------------------

    def can_gc(self, uuid: int) -> bool:
        return self.bookkeeping.can_gc(uuid)

    def is_registered(self, uuid: int) -> bool:
        return self.bookkeeping.is_registered(uuid)

    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        return self.bookkeeping.get_info(uuid)

    def update_info(self, uuid: int, info: TensorPointerInfo):
        self.bookkeeping.update_info(uuid, info)

    def increment_ref(self, uuid: int, n: int = 1):
        self.bookkeeping.increment_ref(uuid, n)

    def increment_ref_batch(self, uuids: list[int], counts: list[int]):
        self.bookkeeping.increment_ref_batch(uuids, counts)

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
