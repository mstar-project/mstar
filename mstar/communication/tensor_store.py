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
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from mstar.graph.base import TensorPointerInfo
from mstar.utils.containers import ParallelList

NameToTensorList = dict[str, list[torch.Tensor]]

# The worker interns request ids to integer handles; the api-server data and
# preprocess workers never did and still pass the string id. Both work: rid is
# an opaque dict key here, never arithmetic, and it reaches the bookkeeper --
# the half that is actually typed -- not at all.
Rid = int | str


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


def _dtype_name(dtype: torch.dtype) -> str:
    """Same short name the wire codec uses, so there is one mapping."""
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _to_rust(info: TensorPointerInfo) -> dict:
    return {
        "dims": list(info.dims),
        "dtype": _dtype_name(info.dtype),
        "nbytes": info.nbytes,
        "address": info.address,
        "stride": list(info.stride),
        "uuid": info.uuid,
        "source_session_id": info.source_session_id,
        "source_entity": info.source_entity,
        "offset": info.offset,
        "source_tp_size": info.source_tp_size,
        "source_tp_rank": info.source_tp_rank,
        "shm_segment": info.shm_segment,
        "shm_offset": info.shm_offset,
        "source_node_name": info._source_node_name,
        "source_graph_walk": info._source_graph_walk,
    }


def _from_rust(out) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=out.dims,
        dtype=_dtype_from_name(out.dtype),
        nbytes=out.nbytes,
        address=out.address,
        stride=tuple(out.stride),
        uuid=out.uuid,
        source_session_id=out.source_session_id,
        source_entity=out.source_entity,
        offset=out.offset,
        source_tp_size=out.source_tp_size,
        source_tp_rank=out.source_tp_rank,
        shm_segment=out.shm_segment,
        shm_offset=out.shm_offset,
        _source_node_name=out.source_node_name,
        _source_graph_walk=out.source_graph_walk,
    )


class RustTensorBookkeeping(TensorBookkeeping):
    """Every method forwards; the batched ones go over in one call.

    A descriptor handed in is COPIED into Rust, unlike the Python backend which
    keeps the caller's object. Anything relying on seeing a later in-place edit
    (``register_for_send`` stamping ``shm_segment``) has to re-``update_info``.
    """

    def __init__(self):
        # Imported here, not at module scope: tensor_store is imported by
        # effectively everything, and a top-level import would make the whole
        # package unimportable wherever the extension is not built. Same
        # pattern as arena.py.
        from mstar_rust import TensorBookkeeping as _RustBookkeeping

        self._rust = _RustBookkeeping()

    # -- descriptors --------------------------------------------------------

    def put_tensor(self, uuid: int, info: TensorPointerInfo):
        self._rust.put_tensor(uuid, _to_rust(info))

    def put_tensor_batch(
        self, uuids: list[int], infos: list[TensorPointerInfo]
    ):
        self._rust.put_tensor_batch(uuids, [_to_rust(i) for i in infos])

    def update_info(self, uuid: int, info: TensorPointerInfo):
        self._rust.update_info(uuid, _to_rust(info))

    def update_info_batch(
        self, uuids: list[int], infos: list[TensorPointerInfo]
    ):
        self._rust.update_info_batch(uuids, [_to_rust(i) for i in infos])

    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        out = self._rust.get_info(uuid)
        return None if out is None else _from_rust(out)

    def get_info_batch(self, uuids: list[int]) -> list[TensorPointerInfo | None]:
        return [
            None if out is None else _from_rust(out)
            for out in self._rust.get_info_batch(uuids)
        ]

    def forget_tensor(self, uuid: int):
        self._rust.forget_tensor(uuid)

    def is_tracked(self, uuid: int) -> bool:
        return self._rust.is_tracked(uuid)

    # -- refcounts ----------------------------------------------------------

    def increment_ref(self, uuid: int, n: int = 1):
        self._rust.increment_ref(uuid, n)

    def increment_ref_batch(self, uuids: list[int], counts: list[int]):
        self._rust.increment_ref_batch(uuids, counts)

    def dereference(self, uuid: int, n: int = 1):
        self._rust.dereference(uuid, n)

    def dereference_batch(self, uuids: list[int], counts: list[int]):
        self._rust.dereference_batch(uuids, counts)

    # -- flags --------------------------------------------------------------

    def set_persist(self, uuid: int, persist: bool):
        self._rust.set_persist(uuid, persist)

    def set_persist_batch(self, uuids: list[int], persist: bool):
        self._rust.set_persist_batch(uuids, persist)

    def set_mem_registered(self, uuid: int, mem_registered: bool):
        self._rust.set_mem_registered(uuid, mem_registered)

    def is_registered(self, uuid: int) -> bool:
        return self._rust.is_registered(uuid)

    # -- gc -----------------------------------------------------------------

    def can_gc(self, uuid: int) -> bool:
        return self._rust.can_gc(uuid)

    def collectable(self, uuids: list[int]) -> list[int]:
        return self._rust.collectable(uuids)


def _build_tensor_bookkeeping() -> TensorBookkeeping:
    """Gated on ``MSTAR_RUST_GRAPH``, which is what actually needs it: the
    Rust graph runtime holds a SHARE of this object, so the two have to be the
    same implementation.

    Not worth taking on its own. A descriptor is copied into Rust on the way
    in and rebuilt on the way out, which the Python runtime pays for and gets
    nothing back -- the saving is in the crossings the Rust runtime avoids.
    """
    if os.environ.get("MSTAR_RUST_GRAPH", "0") == "1":
        return RustTensorBookkeeping()
    return PythonTensorBookkeeping()


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
        self._rid_to_uuids: dict[Rid, set[int]] = {}
        self._uuid_to_rid: dict[int, Rid] = {}
        self.bookkeeping = bookkeeping or _build_tensor_bookkeeping()

    # -- tensors ------------------------------------------------------------

    def get_tensor(self, uuid: int) -> torch.Tensor:
        return self._tensors[uuid]

    def put_tensor(
        self, rid: Rid, uuid: int,
        tensor: torch.Tensor,
        info: TensorPointerInfo
    ):
        self._tensors[uuid] = tensor
        self._rid_to_uuids.setdefault(rid, set()).add(uuid)
        self._uuid_to_rid[uuid] = rid
        self.bookkeeping.put_tensor(uuid, info)

    def put_tensor_batch(
        self, rid: Rid, tensors: ParallelList[int, torch.Tensor],
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

    def get_all_uuids(self, rid: Rid) -> list[int]:
        return list(self._rid_to_uuids.get(rid, ()))

    def remove_request(self, rid: Rid) -> list[int]:
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

    def update_info_batch(
        self, uuids: list[int], infos: list[TensorPointerInfo]
    ):
        self.bookkeeping.update_info_batch(uuids, infos)

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
