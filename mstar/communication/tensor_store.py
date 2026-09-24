"""Tensor storage, split so the bookkeeping half can have a Rust backend.

``TensorStore`` holds the ``torch.Tensor`` objects and stays Python: Rust must
never own a tensor, or freeing a record inside a ``py.allow_threads`` section
would drop a Python object without the GIL.

``TensorBookkeeping`` holds everything else -- refcounts, the persist flag,
whether the memory has been registered for remote reads -- and sees nothing but
integer uuids. That is what gets ported, so a Rust-side caller can adjust
refcounts without a hop back to Python.

Uuids are globally unique now (see ``tensor_uuid``), so nothing here is keyed by
request. The one exception is ``_rid_to_uuids``, which exists only so a request
teardown can find what to free.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from mstar.graph.base import TensorPointerInfo

NameToTensorList = dict[str, list[torch.Tensor]]

# The worker interns request ids to integer handles; the api-server data and
# preprocess workers never did and still pass the string id. Both work: the id
# is an opaque dict key here, never arithmetic, and it reaches the bookkeeper --
# the half that is actually typed -- not at all.
Rid = int | str


@dataclass
class ReferenceInfo:
    ref_cnt: int = 0
    persist: bool = False
    mem_registered: bool = False


@dataclass
class ColumnarTensorInfo:
    """One node's output batch, marshalled as columns of primitives.

    The fields above the columns are constant for the whole batch, so they
    cross into the bookkeeper once and intern once instead of per tensor.
    ``tp_size``/``tp_rank`` belong there too: the sharding group is a property
    of the node and this rank, not of the request, so one batch -- one node,
    one graph walk, one worker -- cannot be heterogeneous in them.
    """

    source_session_id: str
    source_entity: str
    source_node_name: str
    source_graph_walk: str
    tp_size: int
    tp_rank: int
    uuids: list[int] = field(default_factory=list)
    addresses: list[int] = field(default_factory=list)
    nbytes: list[int] = field(default_factory=list)
    ndims: list[int] = field(default_factory=list)
    dims_flat: list[int] = field(default_factory=list)
    stride_flat: list[int] = field(default_factory=list)
    dtype_idx: list[int] = field(default_factory=list)
    dtype_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._seen_dtypes: dict[torch.dtype, int] = {}

    def add_tensors_canonical(
        self, uuids: list[int],
        canonicals: list[torch.Tensor],
    ):
        self.uuids.extend(uuids)
        for tensor in canonicals:
            self.addresses.append(tensor.data_ptr())
            self.nbytes.append(tensor.nbytes)
            dims = tensor.shape
            self.ndims.append(len(dims))
            self.dims_flat.extend(dims)
            self.stride_flat.extend(tensor.stride())
            idx = self._seen_dtypes.get(tensor.dtype)
            if idx is None:
                idx = self._seen_dtypes[tensor.dtype] = len(self.dtype_names)
                self.dtype_names.append(_dtype_name(tensor.dtype))
            self.dtype_idx.append(idx)


class TensorBookkeeping(ABC):
    """Per-uuid reference state. Never sees a tensor or a request id.

    The defaults of the batched forms loop, so a backend only overrides where
    batching buys something. They take two
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

    @property
    def has_put_tensor_batch_columns(self):
        return False

    def put_tensor_batch_columns(
        self, infos: ColumnarTensorInfo,
    ):
        """Columnar put, for a backend that pays per tensor to marshal.

        Returns whether it handled the batch. The default says no: a Python
        bookkeeper stores the descriptor by reference and has nothing to
        marshal, so flattening it into columns would only add work.
        """
        raise NotImplementedError("Columnar put not implemented for this backend")

    @abstractmethod
    def set_shm_placement(
        self, uuids: list[int], segments: list[str | None],
        offsets: list[int],
    ):
        """Stamp arena placement onto descriptors already held.

        ``register_for_send`` changes only these two fields, so only they
        cross -- not a rebuilt descriptor.
        """
        pass

    @abstractmethod
    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        """The descriptor a peer needs to read this tensor.

        This is what lets an edge be rebuilt from uuids alone: ingestion and
        routing carry uuids, and anything that has to put a descriptor back on
        the wire (a disaggregated loop re-emitting its external inputs, say)
        looks it up here.
        """
        pass

    @abstractmethod
    def forget_tensor(self, uuid: int):
        """Drop the record. The caller frees the tensor itself."""
        pass

    # -- refcounts ----------------------------------------------------------

    @abstractmethod
    def increment_ref(self, uuid: int, n: int = 1):
        pass

    def increment_ref_batch(self, uuids: list[int], counts: list[int]):
        for uuid, n in zip(uuids, counts, strict=True):
            self.increment_ref(uuid, n)

    def increment_ref_batch_uniform(self, uuids: list[int], n: int = 1):
        """``increment_ref_batch`` where every uuid takes the same count.

        The safety hold on a freshly stored output batch is always 1, so the
        caller would otherwise build an n-long list of ones per batch purely
        to satisfy the parallel-list signature.
        """
        for uuid in uuids:
            self.increment_ref(uuid, n)

    @abstractmethod
    def dereference(self, uuid: int, n: int = 1):
        pass

    def dereference_batch(
        self, uuids: list[int], counts: list[int], cleanup: bool = False,
    ) -> tuple[list[int], list[bool]]:
        """Drop ``counts[i]`` references from ``uuids[i]``, answering the gc
        question in the same pass.

        Returns the uuids that became collectable and whether each was
        registered for remote reads -- the one flag a teardown still needs
        after ``cleanup`` has dropped the record. Per uuid the caller would
        otherwise pay a crossing each for the decrement, ``can_gc``,
        ``is_registered`` and ``forget_tensor``.
        """
        return self._drop_refs(zip(uuids, counts, strict=True), cleanup)

    def dereference_batch_uniform(
        self, uuids: list[int], n: int = 1, cleanup: bool = False,
    ) -> tuple[list[int], list[bool]]:
        """``dereference_batch`` for a uniform count, which is what a node's
        consumed inputs and a streamed output batch always are."""
        return self._drop_refs(((uuid, n) for uuid in uuids), cleanup)

    def _drop_refs(
        self, refs: Iterable[tuple[int, int]], cleanup: bool,
    ) -> tuple[list[int], list[bool]]:
        collectable: list[int] = []
        registered: list[bool] = []
        for uuid, n in refs:
            self.dereference(uuid, n)
            if not self.can_gc(uuid):
                continue
            collectable.append(uuid)
            registered.append(self.is_registered(uuid))
        if cleanup:
            for uuid in collectable:
                self.forget_tensor(uuid)
        return collectable, registered

    # -- flags --------------------------------------------------------------

    @abstractmethod
    def set_persist(self, uuid: int, persist: bool):
        pass

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


class PythonTensorBookkeeping(TensorBookkeeping):
    def __init__(self):
        self._ref_info: dict[int, ReferenceInfo] = {}
        self._tensor_info: dict[int, TensorPointerInfo] = {}

    def put_tensor(self, uuid: int, info: TensorPointerInfo):
        self._ref_info[uuid] = ReferenceInfo()
        self._tensor_info[uuid] = info

    def set_shm_placement(
        self, uuids: list[int], segments: list[str | None],
        offsets: list[int],
    ):
        # The descriptor is held by reference, so stamping it in place is the
        # whole update.
        for uuid, seg, off in zip(uuids, segments, offsets, strict=True):
            info = self._tensor_info.get(uuid)
            if info is None:
                continue
            info.shm_segment = seg
            info.shm_offset = off

    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        return self._tensor_info.get(uuid)

    def forget_tensor(self, uuid: int):
        self._ref_info.pop(uuid, None)
        self._tensor_info.pop(uuid, None)

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

    def _drop_refs(
        self, refs: Iterable[tuple[int, int]], cleanup: bool,
    ) -> tuple[list[int], list[bool]]:
        """One record lookup per uuid, where the generic version above pays
        one per question it asks."""
        collectable: list[int] = []
        registered: list[bool] = []
        for uuid, n in refs:
            info = self._ref_info.get(uuid)
            if info is None:
                continue
            info.ref_cnt -= n
            if info.ref_cnt > 0 or info.persist:
                continue
            collectable.append(uuid)
            registered.append(info.mem_registered)
        if cleanup:
            for uuid in collectable:
                self._ref_info.pop(uuid, None)
                self._tensor_info.pop(uuid, None)
        return collectable, registered

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
        # tuple(), like stride below: Rust hands back a list, but the Python
        # bookkeeper stores what the producer built -- a torch.Size, which is
        # a tuple subclass. A list never compares equal to either, so leaving
        # it would make the same descriptor unequal across backends.
        dims=tuple(out.dims),
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
    (``register_for_send`` stamping ``shm_segment``) has to write it back
    (``set_shm_placement``).
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

    @property
    def has_put_tensor_batch_columns(self):
        return True

    def put_tensor_batch_columns(
        self, infos: ColumnarTensorInfo
    ):
        """Store a batch already marshalled as columns of primitives.

        ``_to_rust`` builds a 15-key dict per tensor and pyo3 then allocates a
        String per string field per tensor, which is where storing a batch
        actually spends its time -- batching the *call* alone measured zero.
        Here the caller has filled the columns as it minted, so no
        ``TensorPointerInfo`` is built to be torn apart again, and the fields
        that are invariant across one node's output batch cross and intern
        once rather than per tensor.
        """
        self._rust.put_tensor_batch_columns(
            infos.uuids, infos.addresses, infos.nbytes, infos.ndims,
            infos.dims_flat, infos.stride_flat, infos.dtype_names,
            infos.dtype_idx, infos.tp_size, infos.tp_rank,
            infos.source_session_id, infos.source_entity,
            infos.source_node_name, infos.source_graph_walk,
        )
        return True

    def set_shm_placement(
        self, uuids: list[int], segments: list[str | None],
        offsets: list[int],
    ):
        # One crossing, primitives only -- no descriptor is rebuilt for
        # Python and none is marshalled back.
        self._rust.set_shm_placement(uuids, segments, offsets)

    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        out = self._rust.get_info(uuid)
        return None if out is None else _from_rust(out)

    def forget_tensor(self, uuid: int):
        self._rust.forget_tensor(uuid)

    # -- refcounts ----------------------------------------------------------

    def increment_ref(self, uuid: int, n: int = 1):
        self._rust.increment_ref(uuid, n)

    def increment_ref_batch(self, uuids: list[int], counts: list[int]):
        self._rust.increment_ref_batch(uuids, counts)

    def increment_ref_batch_uniform(self, uuids: list[int], n: int = 1):
        self._rust.increment_ref_batch_uniform(uuids, n)

    def dereference(self, uuid: int, n: int = 1):
        self._rust.dereference(uuid, n)

    def dereference_batch(
        self, uuids: list[int], counts: list[int], cleanup: bool = False,
    ) -> tuple[list[int], list[bool]]:
        return self._rust.dereference_batch(uuids, counts, cleanup)

    def dereference_batch_uniform(
        self, uuids: list[int], n: int = 1, cleanup: bool = False,
    ) -> tuple[list[int], list[bool]]:
        return self._rust.dereference_batch_uniform(uuids, n, cleanup)

    # -- flags --------------------------------------------------------------

    def set_persist(self, uuid: int, persist: bool):
        self._rust.set_persist(uuid, persist)

    def set_mem_registered(self, uuid: int, mem_registered: bool):
        self._rust.set_mem_registered(uuid, mem_registered)

    def is_registered(self, uuid: int) -> bool:
        return self._rust.is_registered(uuid)

    # -- gc -----------------------------------------------------------------

    def can_gc(self, uuid: int) -> bool:
        return self._rust.can_gc(uuid)


def _build_tensor_bookkeeping() -> TensorBookkeeping:
    """The Python bookkeeper, always, for now.

    The Rust one is not worth taking on its own: a descriptor is copied into
    Rust on the way in and rebuilt on the way out, and the saving is in the
    boundary crossings a Rust-side caller avoids by holding the same state.
    Until there is such a caller it is reachable only by passing it to
    ``TensorStore`` explicitly.
    """
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
        # Uuids a batched dereference already dropped from the bookkeeper. The
        # caller still runs its own per-uuid teardown afterwards, and that ends
        # in ``remove_tensor``; without this it would cross again to forget a
        # record that is already gone.
        self._forgotten: set[int] = set()
        self.bookkeeping = bookkeeping or _build_tensor_bookkeeping()

    # -- tensors ------------------------------------------------------------

    def get_tensor(self, uuid: int) -> torch.Tensor:
        return self._tensors[uuid]

    def put_tensor(
        self, request_id: Rid, uuid: int,
        tensor: torch.Tensor,
        info: TensorPointerInfo
    ):
        self._tensors[uuid] = tensor
        self._rid_to_uuids.setdefault(request_id, set()).add(uuid)
        self._uuid_to_rid[uuid] = request_id
        self.bookkeeping.put_tensor(uuid, info)

    @property
    def has_put_tensor_batch_columns(self) -> bool:
        """Whether the caller should marshal into ``ColumnarTensorInfo``.

        The manager asks before it starts minting, because the two forms are
        built differently -- columnar fills its columns as it goes, where the
        other materialises a ``TensorPointerInfo`` per tensor.
        """
        return self.bookkeeping.has_put_tensor_batch_columns

    def put_tensor_batch_multi(
        self, request_ids: list[Rid], uuids: list[int],
        tensors: list[torch.Tensor], infos: list[TensorPointerInfo],
    ):
        """``put_tensor_batch`` spanning requests: one bookkeeping call for a
        whole output batch rather than one per request.

        The maps below are plain dicts, so keeping them in a Python loop costs
        nothing; it is ``bookkeeping.put_tensor_batch`` that is a Rust boundary
        crossing, and taking every request at once collapses it to one.
        """
        self._own(request_ids, uuids, tensors)
        self.bookkeeping.put_tensor_batch(uuids, infos)

    def put_tensor_batch_columns(
        self, request_ids: list[Rid], tensors: list[torch.Tensor],
        columns: ColumnarTensorInfo,
    ):
        """``put_tensor_batch_multi`` for a backend that marshals per tensor.

        Takes the columns the caller filled while minting instead of a
        descriptor per tensor. ``columns.uuids`` is the same order as
        ``tensors``, which is what keeps the ownership maps aligned.
        """
        self._own(request_ids, columns.uuids, tensors)
        self.bookkeeping.put_tensor_batch_columns(columns)

    def _own(
        self, request_ids: list[Rid], uuids: list[int], tensors: list[torch.Tensor],
    ):
        """Record which request owns each tensor. Pure Python dicts, so this
        stays a loop -- only the bookkeeping call crosses into Rust."""
        for request_id, uuid, tensor in zip(request_ids, uuids, tensors, strict=True):
            self._tensors[uuid] = tensor
            self._rid_to_uuids.setdefault(request_id, set()).add(uuid)
            self._uuid_to_rid[uuid] = request_id

    def check_uuid_presence(self, uuid: int) -> bool:
        return uuid in self._tensors

    def remove_tensor(self, uuid: int):
        forgotten = uuid in self._forgotten
        self._forgotten.discard(uuid)
        if self._tensors.pop(uuid, None) is None:
            return
        if not forgotten:
            self.bookkeeping.forget_tensor(uuid)
        request_id = self._uuid_to_rid.pop(uuid, None)
        if request_id is not None:
            owned = self._rid_to_uuids.get(request_id)
            if owned is not None:
                owned.discard(uuid)
                if not owned:
                    del self._rid_to_uuids[request_id]

    def get_all_uuids(self, request_id: Rid) -> list[int]:
        return list(self._rid_to_uuids.get(request_id, ()))

    # -- bookkeeping passthrough -------------------------------------------

    def can_gc(self, uuid: int) -> bool:
        return self.bookkeeping.can_gc(uuid)

    def is_registered(self, uuid: int) -> bool:
        return self.bookkeeping.is_registered(uuid)

    def get_info(self, uuid: int) -> TensorPointerInfo | None:
        return self.bookkeeping.get_info(uuid)

    def increment_ref(self, uuid: int, n: int = 1):
        self.bookkeeping.increment_ref(uuid, n)

    def increment_ref_batch_uniform(self, uuids: list[int], n: int = 1):
        self.bookkeeping.increment_ref_batch_uniform(uuids, n)

    def set_shm_placement(
        self, uuids: list[int], segments: list[str | None],
        offsets: list[int],
    ):
        self.bookkeeping.set_shm_placement(uuids, segments, offsets)

    def increment_ref_batch(self, uuids: list[int], counts: list[int]):
        self.bookkeeping.increment_ref_batch(uuids, counts)

    def dereference_batch(
        self, uuids: list[int], counts: list[int], cleanup: bool = False,
    ) -> tuple[list[int], list[bool]]:
        """``dereference_batch_uniform`` where the count differs per uuid --
        an ack that covers several reads of the same tensor, say."""
        return self._collected(
            self.bookkeeping.dereference_batch(uuids, counts, cleanup), cleanup
        )

    def dereference(self, uuid: int, n: int = 1):
        self.bookkeeping.dereference(uuid, n)

    def dereference_batch_uniform(
        self, uuids: list[int], n: int = 1, cleanup: bool = False,
    ) -> tuple[list[int], list[bool]]:
        """The uuids that became collectable, and whether each was registered.

        ``cleanup`` forgets them in the bookkeeper as part of the same call;
        the tensors themselves stay here until the caller's teardown removes
        them, which is what ``_forgotten`` covers.
        """
        return self._collected(
            self.bookkeeping.dereference_batch_uniform(uuids, n, cleanup),
            cleanup,
        )

    def _collected(
        self, result: tuple[list[int], list[bool]], cleanup: bool,
    ) -> tuple[list[int], list[bool]]:
        if cleanup:
            self.mark_forgotten(result[0])
        return result

    def mark_forgotten(self, uuids: list[int]):
        """These records are already gone from the bookkeeper, so the teardown
        that follows must not cross again to drop them."""
        self._forgotten.update(uuids)

    def set_metadata(
        self, uuid: int,
        persist: bool | None = None,
        mem_registered: bool | None = None,
    ):
        if persist is not None:
            self.bookkeeping.set_persist(uuid, persist)
        if mem_registered is not None:
            self.bookkeeping.set_mem_registered(uuid, mem_registered)
