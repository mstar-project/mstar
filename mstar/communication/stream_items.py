"""STREAM_ITEMS: a step's streamed tensors to one peer worker, as columns.

The sender side (``StreamItemsBuilder``) mirrors the Rust runtime's
``frames::StreamItems``: both runtimes send a step's plain streamed edges this
way, so a receiver decodes, reads and routes them once per step instead of
once per (request, worker) INPUT_SIGNALS. The reader side turns a frame back
into rows, and a row back into what INPUT_SIGNALS carried, for the paths that
still want that (a request not known yet, a transport without a batched read).
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

from mstar.communication.wire import _dtype_from_name, _dtype_name
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.utils.ipc_format import InputSignals, StreamItems


def stream_items_eligible(edge: GraphEdge) -> bool:
    """Whether a streamed edge can ride STREAM_ITEMS: unsharded, one
    producer, with items."""
    return bool(edge.tensor_info) and edge._shard_dim is None and edge._total_fanin == 1


class StreamItemsBuilder:
    """One frame's columns, row by row (``frames::StreamItems`` in Rust)."""

    def __init__(self, partition_name: str):
        self._partition = partition_name
        self._msg: StreamItems | None = None
        self._name_at: dict[str, int] = {}
        self._dtype_at: dict = {}

    def __bool__(self) -> bool:
        return self._msg is not None and bool(self._msg.request_ids)

    def _name(self, name: str) -> int:
        i = self._name_at.get(name)
        if i is None:
            i = self._name_at[name] = len(self._msg.names)
            self._msg.names.append(name)
        return i

    def try_add(self, request_id: str, edge: GraphEdge) -> bool:
        """Add the edge's row; False, adding nothing, when its descriptors
        don't share the frame's fields (or one walk)."""
        infos = edge.tensor_info
        if not infos:
            return False
        first = infos[0]
        shared = (
            first.source_session_id, first.source_entity, first._source_node_name,
            first.source_tp_size, first.source_tp_rank,
        )
        walk = first._source_graph_walk
        if any(
            (i.source_session_id, i.source_entity, i._source_node_name,
             i.source_tp_size, i.source_tp_rank) != shared or i._source_graph_walk != walk
            for i in infos
        ):
            return False
        m = self._msg
        if m is None:
            m = self._msg = StreamItems(
                partition_name=self._partition, source_session_id=shared[0],
                source_entity=shared[1], tp_size=shared[3], tp_rank=shared[4],
                source_node_name=shared[2],
            )
        elif (m.source_session_id, m.source_entity, m.source_node_name, m.tp_size, m.tp_rank) != shared:
            return False
        m.request_ids.append(request_id)
        m.signal_idx.append(self._name(edge.name))
        m.next_node_idx.append(self._name(edge.next_node))
        m.walk_idx.append(-1 if walk is None else self._name(walk))
        m.num_items.append(len(infos))
        for i in infos:
            m.uuids.append(i.uuid)
            m.addresses.append(i.address)
            m.nbytes.append(i.nbytes)
            m.offsets.append(i.offset)
            m.ndims.append(len(i.dims))
            m.dims_flat.extend(i.dims)
            m.stride_flat.extend(i.stride)
            d = self._dtype_at.get(i.dtype)
            if d is None:
                d = self._dtype_at[i.dtype] = len(m.dtype_names)
                m.dtype_names.append(_dtype_name(i.dtype))
            m.dtype_idx.append(d)
            m.shm_segment_idx.append(-1 if i.shm_segment is None else self._name(i.shm_segment))
            m.shm_offsets.append(i.shm_offset)
        return True

    def build(self) -> StreamItems:
        return self._msg


class StreamRow(NamedTuple):
    """One (request, streamed edge) row of a frame, its ``index``-th: its
    items are ``frame.uuids[start:start + count]`` (and likewise every item
    column), and their dims and strides ``frame.dims_flat[dim_start:...]``."""
    index: int
    request_id: str
    signal: str
    next_node: str
    start: int
    count: int
    dim_start: int


def rows(frame: StreamItems) -> Iterator[StreamRow]:
    names, ndims = frame.names, frame.ndims
    start = dim_start = 0
    for index, (rid, s, n, count) in enumerate(zip(
        frame.request_ids, frame.signal_idx, frame.next_node_idx, frame.num_items, strict=True,
    )):
        yield StreamRow(index, rid, names[s], names[n], start, count, dim_start)
        dim_start += sum(ndims[start:start + count])
        start += count


def row_infos(frame: StreamItems, row: StreamRow) -> list[TensorPointerInfo]:
    """The row's tensor descriptors, as INPUT_SIGNALS carried them."""
    out = []
    names = frame.names
    dtypes = [_dtype_from_name(n) for n in frame.dtype_names]
    dim_at = row.dim_start
    walk = frame.walk_idx[row.index]
    for k in range(row.start, row.start + row.count):
        nd = frame.ndims[k]
        seg = frame.shm_segment_idx[k]
        out.append(TensorPointerInfo(
            dims=frame.dims_flat[dim_at:dim_at + nd],
            dtype=dtypes[frame.dtype_idx[k]],
            nbytes=frame.nbytes[k],
            address=frame.addresses[k],
            stride=frame.stride_flat[dim_at:dim_at + nd],
            uuid=frame.uuids[k],
            source_session_id=frame.source_session_id,
            source_entity=frame.source_entity,
            offset=frame.offsets[k],
            source_tp_size=frame.tp_size,
            source_tp_rank=frame.tp_rank,
            shm_segment=None if seg < 0 else names[seg],
            shm_offset=frame.shm_offsets[k],
            _source_node_name=frame.source_node_name,
            _source_graph_walk=None if walk < 0 else names[walk],
        ))
        dim_at += nd
    return out


def row_input_signals(frame: StreamItems, row: StreamRow) -> InputSignals:
    """The row as the INPUT_SIGNALS that carried it before STREAM_ITEMS."""
    return InputSignals(
        request_id=row.request_id,
        inputs=[GraphEdge(
            next_node=row.next_node, name=row.signal,
            tensor_info=row_infos(frame, row), is_streaming=True,
        )],
        request_info=None,
        partition_name=frame.partition_name,
    )
