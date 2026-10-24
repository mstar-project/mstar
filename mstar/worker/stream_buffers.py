"""The consumer side of streaming edges on one worker.

Each request's ``StreamBuffer``s live on its per-request info; this object
creates them, turns their ready chunks into graph edges, takes back the
chunks a step did not keep, and settles which streams a step ends.
"""
from __future__ import annotations

import torch

from mstar.communication.tensors import NameToTensorList, TensorCommunicationManager
from mstar.graph.base import GraphEdge, GraphSection
from mstar.graph.runtime.base import ColumnarEdgeSpecs, InputTensors
from mstar.streaming.stream_buffer import StreamBuffer, StreamChunkInfo, StreamingEdge
from mstar.streaming.topology import Connection
from mstar.worker.node_manager_utils import RequestStateManager


def consumer_nodes(
    connections: list[Connection], walks: dict[str, GraphSection],
) -> dict[str, str]:
    """Map each incoming streaming edge to the node that consumes it.

    Recurses through ``get_nodes()`` so a consumer nested inside a
    ``Loop`` / ``Sequential`` / ``Parallel`` is found too. Matching the
    walk's top-level ``input_names`` only finds a walk that is itself a
    single ``GraphNode``; a decode ``Loop`` whose inner node consumes the
    streamed edge would be missed, and its chunks routed to ``next_node=""``.
    """
    cache: dict[str, str] = {}
    for conn in connections:
        for section in walks.values():
            for node in section.get_nodes().values():
                if conn.edge_name in node.input_names:
                    cache[conn.edge_name] = node.name
    return cache


class StreamBufferManager:
    """The stream buffers of this worker's requests, for the connections whose
    consumer runs here."""

    def __init__(
        self,
        request_state: RequestStateManager,
        tensor_manager: TensorCommunicationManager,
        connections: list[Connection],
        walks: dict[str, GraphSection] | None = None,
        *,
        graph_runtime=None,
        walk_driven_edges: dict[str, str] | None = None,
    ):
        self._request_state = request_state
        self._tensor_manager = tensor_manager
        self._graph_runtime = graph_runtime
        # walk-driven edge here -> the producer-triggered partition it feeds
        self._walk_driven_edges = walk_driven_edges or {}
        self.connections = connections
        # edge name -> the node that consumes it
        self.consumer_node: dict[str, str] = (
            consumer_nodes(connections, walks) if connections and walks else {}
        )
        # edge name -> the partition the stream feeds
        self._partition: dict[str, str] = {c.edge_name: c.to_partition for c in connections}
        # the rids that own a buffer; empty for a text model, so a speculation
        # skips its per-row stream polls
        self._rids: set[int] = set()

    def possible(self) -> bool:
        """Whether any live request owns a stream buffer."""
        return bool(self._rids)

    # -- request lifecycle ----------------------------------------------------

    def add_request(self, request_id: int, request_info) -> None:
        """Create the request's buffers. A request with several partitions
        here arrives once per partition: the buffer the first arrival made is
        kept (it may already hold items)."""
        req_info = self._request_state.per_request_info[request_id]
        for conn in self.connections:
            sbuf = req_info.stream_buffers.get(conn.edge_name)
            if sbuf is None:
                sbuf = StreamBuffer(
                    request_id=request_id,
                    edge_name=conn.edge_name,
                    from_partition=conn.from_partition,
                    policy=conn.chunk_policy_factory(),
                    walk_tagged=conn.consumer_walk is not None,
                )
                req_info.stream_buffers[conn.edge_name] = sbuf
                consumer = self.consumer_node.get(conn.edge_name, "")
                req_info.stream_buffers_by_consumer.setdefault(consumer, {})[conn.edge_name] = sbuf
                self._rids.add(request_id)
            if conn.to_partition == request_info.partition_name:
                lead = request_info.stream_lead_items.get(conn.edge_name)
                if lead:
                    sbuf.prime_context(lead)

    def remove_request(self, request_id: int) -> None:
        self._rids.discard(request_id)

    # -- items in, chunks out -------------------------------------------------

    def route_tensor(self, request_id: int, edge: GraphEdge) -> None:
        """Route a streaming tensor to its request's StreamBuffer for this
        edge. The buffer holds the item by its handle: the read's usage
        reference passes to it, and the tensor is looked up when the
        consuming batch is built (``input_tensors``)."""
        req_info = self._request_state.per_request_info.get(request_id)
        stream_buf = req_info.stream_buffers[edge.name]
        for info in edge.tensor_info:
            stream_buf.put(info.uuid, info)

    def _partition_walk(self, request_id: int, partition: str) -> str | None:
        req_info = self._request_state.per_request_info.get(request_id)
        part_info = req_info and req_info.per_partition_info.get(partition)
        return part_info.current_fwd_info.graph_walk if part_info else None

    def _enter_chunk_walk(
        self, sbuf: StreamBuffer, edge_name: str, request_id: int,
        allow_walk_change: bool,
    ) -> bool:
        """Put the consumer in the walk the next chunk runs under. False when
        the chunk has to wait: the partition is mid-pass in its current walk,
        or the caller cannot change walks (speculation)."""
        walk = sbuf.peek_walk()
        if walk is None:
            return True
        partition = self._walk_driven_edges[edge_name]
        if walk == self._partition_walk(request_id, partition):
            return True
        if not allow_walk_change or not self._graph_runtime.is_partition_idle(
            request_id, partition,
        ):
            return False
        self._graph_runtime.set_walk(request_id, partition, walk)
        # Nothing of the partition is in flight to still read the old walk
        self._request_state.get_fwd_info(request_id, partition).graph_walk = walk
        return True

    def pop_edge(
        self, sbuf: StreamBuffer, edge_name: str, request_id: int,
        allow_walk_change: bool = True,
    ) -> StreamingEdge | None:
        """The buffer's next chunk as a graph edge into its consumer: one a
        step refused earlier first, else a newly ready one; None when there
        is neither. A producer-triggered consumer first enters the chunk's
        walk, if ``allow_walk_change`` and nothing of it is mid-pass."""
        consumer_node = self.consumer_node.get(edge_name, "")
        waiting = sbuf.pop_waiting_edge()
        if waiting is not None:
            return waiting
        partition = self._walk_driven_edges.get(edge_name)
        consumer_walk = (
            self._partition_walk(request_id, partition) if partition else None
        )
        if not sbuf.has_chunk_ready(consumer_walk) or not self._enter_chunk_walk(
            sbuf, edge_name, request_id, allow_walk_change,
        ):
            return None
        chunk = sbuf.pop_chunk()
        # The chunk's items as the buffer holds them: their tensor infos, or
        # none for an empty chunk (producer done, nothing left). The edge
        # carries every item; ``input_tensors`` stacks them when the consuming
        # batch is built.
        items = chunk.data.get("data") or []
        tm = self._tensor_manager
        if items:
            # the edge's own reference, dropped once its node consumes it
            tm.increment_ref_batch_uniform([info.uuid for info in items])
        if chunk.released:
            # the buffer's references to the items it advanced past; after
            # the increment, so an item in both stays alive
            tm.dereference_batch_uniform([info.uuid for info in chunk.released])
        edge = GraphEdge(
            next_node=consumer_node, name=edge_name, tensor_info=list(items),
            _final_stream_chunk=chunk.is_final,
        )
        return StreamingEdge(edge, chunk.info)

    def poll(self) -> list[tuple[int, StreamingEdge]]:
        """Every request's ready chunks, popped."""
        polled: list[tuple[int, StreamingEdge]] = []
        for request_id, req_info in list(self._request_state.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                streaming_edge = self.pop_edge(sbuf, edge_name, request_id)
                if streaming_edge is not None:
                    polled.append((request_id, streaming_edge))
        return polled

    def poll_for(self, request_id: int, node_name: str) -> list[StreamingEdge]:
        """The ready chunks of the request's streams into ``node_name``, for a
        speculated step, which was built for the current walk."""
        req_info = self._request_state.per_request_info.get(request_id)
        if req_info is None:
            return []
        result = []
        for edge_name, sbuf in req_info.stream_buffers_by_consumer.get(node_name, {}).items():
            edge = self.pop_edge(sbuf, edge_name, request_id, allow_walk_change=False)
            if edge is not None:
                result.append(edge)
        return result

    def return_edge(self, request_id: int, streaming_edge: StreamingEdge) -> None:
        """Hand a chunk back to its StreamBuffer after it was refused, by the
        graph's ingest or by a speculation rolling its ingests back."""
        req_info = self._request_state.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(streaming_edge.edge.name)
        if sbuf is not None:
            sbuf.store_uningested_edge(streaming_edge)
        # The step that took the final chunk will not run; the stream is live again.
        edge = streaming_edge.edge
        if edge._final_stream_chunk:
            req_info.ended_streams.discard(edge.name)

    def mark_ingested(self, request_id: int, streaming_edge: StreamingEdge) -> None:
        """Remember the chunk the graph accepted, for the batch that consumes it.

        A single ``ingested_chunk`` field works because streaming edges are
        ingested with ``can_buffer=False`` -- except when speculating a
        loop-back, where the current value is already in flight in the
        forward, so the buffered chunk is the only one a later build can
        still be describing.
        """
        req_info = self._request_state.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(streaming_edge.edge.name)
        if sbuf is not None:
            sbuf.ingested_chunk = streaming_edge.chunk

    # -- what a step consumes ---------------------------------------------------

    def input_tensors(
        self, edges: ColumnarEdgeSpecs, node_name: str, rids, skip_missing: bool = False,
    ) -> InputTensors:
        """``edges.to_input_tensors``, with each stream chunk into ``node_name``
        collated: a chunk's edge carries one uuid per item, and the items are
        stacked here, when the consuming batch is built. A one-item chunk is
        that item, unstacked."""
        out = edges.to_input_tensors(
            self._tensor_manager.get_tensor, rids, skip_missing=skip_missing,
        )
        signals = [e for e, n in self.consumer_node.items() if n == node_name]
        if signals:
            for inputs in out.by_rid.values():
                for name in signals:
                    items = inputs.get(name)
                    if items is not None and len(items) > 1:
                        inputs[name] = [torch.stack(items)]
        return out

    def chunks_for(
        self, request_id: int, node_name: str, inputs: NameToTensorList,
    ) -> dict[str, StreamChunkInfo] | None:
        """Chunk info for the streamed inputs this step consumes, or None."""
        req_info = self._request_state.per_request_info.get(request_id)
        if req_info is None:
            return None
        sbufs = req_info.stream_buffers_by_consumer.get(node_name)
        if not sbufs:
            return None
        chunks = {
            name: sbuf.ingested_chunk for name, sbuf in sbufs.items()
            if name in inputs and sbuf.ingested_chunk is not None
        }
        return chunks or None

    def settle_final(
        self, node_name: str, final_edges: dict[int, set[str]],
    ) -> tuple[set[int], set[int]]:
        """Record the streams whose final chunk this step consumes, per rid.

        Returns two sets, ``(node_done, partition_done)``: the rids whose
        streams into ``node_name`` have all ended once this step runs, and
        the rids whose streams into the partition have. The second is a
        subset of the first. Streams end in any order and over several
        steps, so one final chunk alone means only that its own stream ended. A
        ``continue_after_producer_done`` stream never sends a final chunk;
        it holds neither back.
        """
        node_done: set[int] = set()
        partition_done: set[int] = set()
        for rid, edges in final_edges.items():
            req_info = self._request_state.per_request_info.get(rid)
            if req_info is None:
                node_done.add(rid)
                partition_done.add(rid)
                continue
            req_info.ended_streams |= edges
            partition = self._partition.get(next(iter(edges)))
            live = [
                edge_name for edge_name, sbuf in req_info.stream_buffers.items()
                if edge_name not in req_info.ended_streams
                and not sbuf.policy.continue_after_producer_done()
            ]
            if all(self.consumer_node.get(e) != node_name for e in live):
                node_done.add(rid)
            if all(self._partition.get(e) != partition for e in live):
                partition_done.add(rid)
        return node_done, partition_done

    def consumption(self, request_id: int) -> dict[str, int]:
        """Items consumed so far per stream, as the producer is told."""
        req_info = self._request_state.per_request_info.get(request_id)
        if req_info is None:
            return {}
        return {
            edge_name: sbuf._consumed
            for edge_name, sbuf in req_info.stream_buffers.items()
        }

    def consumption_batch(self, rids: list[int]) -> list[tuple[int, dict[str, int]]]:
        """``consumption`` for the rids that have stream buffers at all; the
        rest would report an empty dict."""
        infos = self._request_state.per_request_info
        out: list[tuple[int, dict[str, int]]] = []
        for rid in rids:
            req_info = infos.get(rid)
            if req_info is None or not req_info.stream_buffers:
                continue
            out.append((rid, {
                edge_name: sbuf._consumed
                for edge_name, sbuf in req_info.stream_buffers.items()
            }))
        return out
