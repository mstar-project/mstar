"""``TensorBookkeeping`` over the Rust implementation in ``rust/``
(``mstar_rust.TensorBookkeeping``) — a drop-in for the Python class, so
``TensorStore`` does not know which one it holds.

Rust never sees a tensor; that is why only the bookkeeping half is ported (see
``tensor_store``). What crosses here is uuids and descriptors: the descriptor's
strings are interned on the Rust side, so they are passed as plain ``str`` and
rebuilt into a ``TensorPointerInfo`` on the way out.

Build: ``maturin develop --release`` in ``rust/`` (see ``docs/installation.rst``).
"""
from __future__ import annotations

import torch
from mstar_rust import TensorBookkeeping as _RustBookkeeping

from mstar.communication.tensor_store import TensorBookkeeping
from mstar.graph.base import TensorPointerInfo


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
