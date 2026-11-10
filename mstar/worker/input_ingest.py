"""Everything inbound: the message dispatch, input reads, ready tensors and
stream polls. Holds no state of its own beside ``LoraLoads``, the runtime LoRA
loads in flight, which it builds and finishes after each message pass: lifecycle
messages are recorded on ``RequestLifecycle`` for the worker to apply at its next
teardown point, and TP-follow messages go to ``Lockstep``."""
from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

from mstar.communication.edge_items import frame_dtypes, row_infos, row_input_signals
from mstar.communication.edge_items import rows as stream_rows
from mstar.communication.event import EventWakeup
from mstar.engine.resources.kv.config import without_layouts
from mstar.graph.base import GraphEdge
from mstar.graph.graph_io import format_graph_edge_list
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.utils.client_errors import client_error_text
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    EdgeItems,
    FailRequests,
    InputSignals,
    MessageSource,
    NewRequest,
    RemoveRequest,
    StopLoops,
    TensorReceived,
    UnpersistTensors,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.profiler import PhaseLaps, phase_end, phase_start, range_pop, range_push
from mstar.worker.lockstep import Lockstep
from mstar.worker.lora_loads import LoraLoads
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.worker_metrics import WorkerMetrics

if TYPE_CHECKING:
    from mstar.engine.resources import ResourceReqConfig

logger = logging.getLogger(__name__)

# worker_phases span per message type
_MSG_SPANS = {t: f"worker.msg.{t.name.lower()}" for t in WorkerMessageType}

_NEEDS_ACTIVE_REQUEST = frozenset({
    WorkerMessageType.REMOVE_REQUEST,
    WorkerMessageType.INPUT_SIGNALS,
    WorkerMessageType.STOP_LOOPS,
})


def _one_rids_edges(rid: int, edges: list[GraphEdge]) -> ColumnarEdgeSpecs:
    """Arriving edges as columns, for the runtime's ingest. Only uuids cross: the
    runtime rebuilds the descriptors from the tensor store, which is why they are
    kept there. Filled straight from the edges: an EdgeSpec per edge in between
    would cost as much as the columns save."""
    block = ColumnarEdgeSpecs.empty()
    for edge in edges:
        block.add_edge(rid, edge)
    return block


class InputIngest:
    def __init__(
        self, services: WorkerServices, layout: WorkerLayout,
        lifecycle: RequestLifecycle, lockstep: Lockstep, wakeup_event: EventWakeup,
        *, metrics: WorkerMetrics | None = None, enable_nvtx: bool = False,
        set_rebuildable: Callable[[int, dict[str, ResourceReqConfig]], Collection[str]] | None = None,
    ):
        self._worker_id = services.worker_id
        self._runtime = services.graph_runtime
        self._engines = services.engine_manager
        self._tensors = services.tensor_manager
        self._request_state = services.request_state
        self._streams = services.streams
        self._communicator = services.communicator
        self._layout = layout
        self._lifecycle = lifecycle
        self._lockstep = lockstep
        self._wakeup_event = wakeup_event
        self._enable_nvtx = enable_nvtx
        # marks each paged KV cache of a request's first NewRequest rebuildable or not, before the engine adds it,
        # and returns the nodes whose log of the request the rebuild pool couldn't hold
        self._set_rebuildable = set_rebuildable
        self.lora_loads = LoraLoads(services)
        self._handlers = {
            WorkerMessageType.NEW_REQUEST: self._add_new_request,
            WorkerMessageType.DRAIN_REQUEST: self._drain_request,
            WorkerMessageType.REMOVE_REQUEST: self._remove_request,
            WorkerMessageType.INPUT_SIGNALS: self._process_new_inputs,
            WorkerMessageType.EDGE_ITEMS: self._process_edge_items,
            WorkerMessageType.TENSOR_RECEIVED: self._handle_tensor_received,
            WorkerMessageType.UNPERSIST_TENSORS: self._unpersist_tensors,
            WorkerMessageType.STOP_LOOPS: self._stop_loops,
            WorkerMessageType.SCHEDULE_TP: lockstep.enqueue_follow,
            WorkerMessageType.TP_NO_SPEC: lockstep.enqueue_no_spec,
            WorkerMessageType.LOAD_LORA_ADAPTER: self.lora_loads.load,
            WorkerMessageType.UNLOAD_LORA_ADAPTER: self.lora_loads.unload,
        }
        if metrics is not None:
            # sent only with metrics on
            self._handlers[WorkerMessageType.ENABLE_METRICS] = lambda _: metrics.start_counting()

    def poll(self) -> None:
        """Messages, then ready tensors, then stream buffers."""
        self.process_messages()
        self.check_ready_tensors()
        self.poll_stream_buffers()

    # -- messages -------------------------------------------------------------

    def process_messages(self) -> None:
        self._dispatch(self._communicator.get_all_new_messages())
        if self.lora_loads:
            self.lora_loads.finish()

    def _dispatch(self, messages: list[WorkerMessage]) -> None:
        handlers = self._handlers
        for message in messages:
            message_type = message.message_type
            # per_request_info is handle-keyed, so resolve the wire string
            # first; an unknown one has no handle and is parked the same way.
            # The conductor's REMOVE is never parked: ``_remove_request``
            # handles an id this worker does not hold.
            if (
                message_type in _NEEDS_ACTIVE_REQUEST
                and not (
                    message_type == WorkerMessageType.REMOVE_REQUEST
                    and message.body.source == MessageSource.CONDUCTOR
                )
                and self._runtime.get_rid_handle(message.body.request_id)
                not in self._request_state.per_request_info
            ):
                if not self._lifecycle.park(message):
                    self.drop_for_removed(message)
                continue
            t0 = phase_start()
            handler = handlers.get(message_type)
            if handler is not None:
                handler(message.body)
            if t0 is not None:
                phase_end(_MSG_SPANS[message_type], t0)

    def drop_for_removed(self, message: WorkerMessage) -> None:
        """Drop a message for a removed request. Inputs are ACKed unread so the
        producer can free its source buffers."""
        logger.debug(
            "Worker %s: dropping %s for removed request %s",
            self._worker_id, message.message_type.value, message.body.request_id,
        )
        if message.message_type == WorkerMessageType.INPUT_SIGNALS:
            self._tensors.ack_unread_tensors(message.body.inputs)

    def _skip_unshared_publish(self, rid: int, wg_to_workers: dict[int, list[str]]) -> None:
        """Turn off publication of each resource whose declaring nodes all run
        on this worker's lockstep instance for this request: published state
        is only read by another instance's ``admit_retrieve``."""
        engine = self._engines.engine
        all_wg_nodes = self._layout.all_wg_nodes
        keys = []
        for key, nodes in engine.publisher_nodes.items():
            owners = {
                frozenset(workers) for wg_id, workers in wg_to_workers.items()
                if not nodes.isdisjoint(all_wg_nodes.get(wg_id, ()))
            }
            if len(owners) == 1 and self._worker_id in next(iter(owners)):
                keys.append(key)
        if keys:
            engine.skip_publish(rid, keys)

    def _add_new_request(self, body: NewRequest) -> None:
        wire_rid = body.request_id
        lifecycle = self._lifecycle
        # Any worker can handle a DRAIN before the conductor's NEW_REQUEST, and a TP follower gets rank 0's. After
        # READS_DONE the worker still takes the request in, without reads, since on a follower rank 0's moves for the
        # request wait until it holds the request.
        unread = not lifecycle.takes_input(wire_rid)
        logger.debug("Worker %s received request %s", self._worker_id, wire_rid)
        laps = PhaseLaps("worker.msg.new_request.")
        runtime = self._runtime
        try:
            # The one place a handle is minted; a request with several partitions
            # on this worker gets the same one for each. Everything below keys on it.
            rid = runtime.add_request(
                request_id=wire_rid,
                partition=body.request_info.partition_name,
                graph_walk=body.request_info.graph_walk,
                partition_worker_graph_ids=body.partition_worker_graph_ids,
                worker_graph_to_workers=ParallelList.from_dict(body.worker_graph_to_workers),
                loop_caps=body.loop_caps,
            )
            body.request_info.rid_handle = rid
            lifecycle.add(rid, wire_rid)
            laps.lap("runtime_add")

            # One NewRequest comes per partition this worker serves, and a later one
            # can land after a step already ran. The tensor manager and the engine
            # hold request-wide state, so only the first opens it. Opening it again
            # would reset that state under the partition that is running.
            first = rid not in self._request_state.per_request_info
            self._request_state.add_request(rid, body.request_info)
            runtime.register_stop_limits(rid, body.request_info)
            if first:
                self._tensors.register_request(rid, runtime.get_sharding_config(rid))
            laps.lap("worker_state")
            if first:
                unheld = () if self._set_rebuildable is None else self._set_rebuildable(
                    rid, body.request_info.resource_configs,
                )
                self._engines.add_request(
                    rid, body.request_info.resource_configs, body.request_info,
                )
                # Rank 0 journals each refused log once the engine holds the request, since a follower's replay
                # changes the request's KV config.
                for node in unheld:
                    self._engines.get_engine(node).keeps_no_ids(node, rid)
                laps.lap("engine_add")
                self._skip_unshared_publish(rid, body.worker_graph_to_workers)
                laps.lap("skip_publish")
            body.request_info.resource_configs = without_layouts(body.request_info.resource_configs)

            if not unread:
                self._streams.add_request(rid, body.request_info)
                laps.lap("stream_buffers")

                # Start RDMA reads for tensors that have tensor_info
                self._wakeup_event.register_futures(self._tensors.start_read_tensors(
                    rid, body.initial_inputs, graph_walk=body.request_info.graph_walk,
                ))
                laps.lap("start_reads")

                # Signal-only edges (tensor_info is None) can be processed immediately
                signal_only = [edge for edge in body.initial_inputs if len(edge.tensor_info) == 0]
                if signal_only:
                    runtime.ingest_inputs_batch(_one_rids_edges(rid, signal_only), can_buffer=True)
                laps.lap("ingest_signals")
        except Exception as err:
            what = f"ingest of request {wire_rid} failed"
            rid = runtime.get_rid_handle(wire_rid)
            if rid is not None and rid in lifecycle.arrivals:
                # a failed engine ingest rolled itself back; the conductor's REMOVE
                # tears down what else the request has here
                self._fail_intake([rid], err, what)
            else:
                # Raised before the lifecycle took the request in, so it can't
                # report it: tell the conductor by its wire id.
                text, _ = client_error_text(err, f"Worker {self._worker_id}: {what}", log=logger)
                self._communicator.send("conductor", ConductorMessage(
                    message_type=ConductorMessageType.FAIL_REQUESTS,
                    body=FailRequests(errors={wire_rid: f"Error in worker: {text}"}),
                ))
        parked = lifecycle.take_parked(wire_rid)
        if unread:
            # its parked inputs are never read, so they are ACKed unread as for a removed request
            for message in parked:
                if message.message_type == WorkerMessageType.INPUT_SIGNALS:
                    self.drop_for_removed(message)
            parked = [message for message in parked if message.message_type != WorkerMessageType.INPUT_SIGNALS]
        self._dispatch(parked)

    def _remove_request(self, body: RemoveRequest) -> None:
        rid = self._runtime.get_rid_handle(body.request_id)
        if rid is not None:
            self._lifecycle.remove(rid, body, self._lockstep.last_consumed_seq)
            return
        # Never admitted here, or already removed: no handle-keyed state to
        # clear, on any rank: the conductor sends a request's NEW_REQUESTs and
        # its REMOVE down one FIFO socket and forgets it after, so no NEW_REQUEST
        # follows. The wire-keyed TP-follow count can be non-empty for a rid that
        # never got a handle, and no later message would ever pop it.
        for message in self._lifecycle.removed(None, body.request_id):
            self.drop_for_removed(message)
        self._lockstep.clear_wire_rid(body.request_id)

    def _drain_request(self, body: DrainRequest) -> None:
        self._lifecycle.drain(body)

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        # Uuids are global, so a late ack needs no rid lookup and works even
        # after the request is gone.
        # One crossing for the ack: a reader confirms a whole edge at a time,
        # and the count differs per uuid (an edge read by several consumers).
        self._tensors.dereference_batch(
            list(body.successful_tensors),
            list(body.successful_tensors.values()),
        )

    def _unpersist_tensors(self, body: UnpersistTensors) -> None:
        for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
            self._tensors.increment_ref(uuid, n=ref_cnt)
            self._tensors.set_persist(uuid, persist=False)

    def _stop_loops(self, body: StopLoops) -> None:
        rid = self._runtime.get_rid_handle(body.request_id)
        if rid is None:
            return
        self._runtime.apply_peer_loop_stops(rid, body.partition_name, body.loop_stop_times)

    # -- inputs ---------------------------------------------------------------

    def _process_new_inputs(self, body: InputSignals) -> None:
        if not self._lifecycle.takes_input(body.request_id):
            return
        if logger.isEnabledFor(logging.DEBUG):  # the edge list is formatted per message
            logger.debug(
                "Received new signals %s at worker %s for request %s",
                format_graph_edge_list(body.inputs), self._worker_id, body.request_id,
            )
        runtime = self._runtime
        rid = runtime.get_rid_handle(body.request_id)
        if rid is None:
            # _dispatch parks or drops signals for an unknown request, so this
            # only happens if that changes; there is nothing to route into.
            logger.warning(
                "Worker %s: dropping input signals for unknown request %s",
                self._worker_id, body.request_id,
            )
            return
        if rid in self._lifecycle.pending_removes:
            self._tensors.ack_unread_tensors(body.inputs)
            return
        req_info = self._request_state.per_request_info.get(rid)
        enable_nvtx = self._enable_nvtx

        if enable_nvtx:
            range_push("process_new_inputs.routing_update")
        # Handle producer_done signal: end the producer's streams it counts.
        # Its sender counts every edge it ends, with 0 items too: the conductor
        # for a producer partition, naming each of its edges into this
        # consumer, and the data worker for the client, naming only the
        # streams the client ended.
        if body.producer_done and req_info:
            counts = body.producer_done_items
            for edge_name, sbuf in req_info.stream_buffers.items():
                if sbuf.from_partition in body.producer_done and edge_name in counts:
                    sbuf.signal_done(counts[edge_name])

        # Separate streaming edges: they'll be handled when tensors are ready
        # (streaming edges with tensor_info go through RDMA, handled in check_ready_tensors)
        non_streaming = [edge for edge in body.inputs if not edge.is_streaming]
        buffers = req_info.stream_buffers if req_info else {}
        streaming_with_tensors = []
        unbuffered = []
        for edge in body.inputs:
            if edge.is_streaming and edge.tensor_info:
                (streaming_with_tensors if edge.name in buffers else unbuffered).append(edge)
        if unbuffered:
            self._ack_unbuffered(rid, body.request_id, unbuffered)

        # Only update fwd_info when there are non-streaming edges (i.e., this is
        # a conductor-triggered forward pass, not just streaming data from another
        # partition). Streaming-only InputSignals must not overwrite the current
        # partition's fwd_info.
        if non_streaming:
            # A fwd_info off the wire carries only the string (or the sender's
            # handle); stamp this worker's so submodules keying per-request
            # state can use it directly.
            body.request_info.rid_handle = rid
            runtime.set_walk(rid, body.partition_name, body.request_info.graph_walk)
            for _rid, _iters in runtime.get_dynamic_loop_iters(
                [rid], partition=body.partition_name,
            ):
                body.request_info.dynamic_loop_iter_counts.update(_iters)
            self._request_state.update_request_info(
                rid, current_fwd_info=body.request_info,
                partition_name=body.partition_name,
            )
            runtime.register_stop_limits(rid, body.request_info)

        if enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.start_read")
        # Start RDMA reads for non-streaming edges with tensor_info; a
        # streamed row of an EDGE_ITEMS replayed from the park has none, and
        # no forward-pass info either
        try:
            if non_streaming:
                self._wakeup_event.register_futures(self._tensors.start_read_tensors(
                    rid, non_streaming, graph_walk=body.request_info.graph_walk,
                ))
            # Start RDMA reads for streaming edges with tensor_info (will be routed
            # to a buffer in check_ready_tensors)
            if streaming_with_tensors:
                self._wakeup_event.register_futures(
                    self._tensors.start_read_tensors(rid, streaming_with_tensors),
                )
                # no forward-pass info comes with a client stream's chunks, nor with
                # a stream whose walk the producer does not decide (see
                # `walk_driven_signals`)
                walks = body.request_info.stream_consumer_walks if body.request_info is not None else {}
                walk_driven = self._streams.walk_driven_edges
                for edge in streaming_with_tensors:
                    req_info.stream_buffers[edge.name].pre_read_register_joined(
                        [info.uuid for info in edge.tensor_info], walks.get(walk_driven.get(edge.name)),
                    )
        except Exception as err:
            if enable_nvtx:
                range_pop(synchronize=False)
            self._fail_intake([rid], err, f"input read of request {body.request_id} failed")
            return
        if enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.process_inputs")

        # Signal-only non-streaming edges can be processed immediately
        signal_only = [edge for edge in non_streaming if len(edge.tensor_info) == 0]
        if signal_only:
            runtime.ingest_inputs_batch(_one_rids_edges(rid, signal_only), can_buffer=True)
        if enable_nvtx:
            range_pop()

    def _process_edge_items(self, frame: EdgeItems) -> None:
        """A step's streamed tensors from one producer (EDGE_ITEMS): each
        row is what one INPUT_SIGNALS of a single streamed edge did, and the
        frame's reads start in one call."""
        reads: list[tuple[int, list[GraphEdge]]] = []
        dtypes = frame_dtypes(frame)
        per_request_info = self._request_state.per_request_info
        for row in stream_rows(frame):
            rid = self._runtime.get_rid_handle(row.request_id)
            if rid is None or rid not in per_request_info:
                # not here yet, or removed: parked or dropped as the
                # INPUT_SIGNALS it replaces
                self._dispatch([WorkerMessage(
                    message_type=WorkerMessageType.INPUT_SIGNALS,
                    body=row_input_signals(frame, row),
                )])
                continue
            if not self._lifecycle.takes_input(row.request_id):
                continue  # as _process_new_inputs: nothing read after READS_DONE
            edge = GraphEdge(
                next_node=row.next_node, name=row.signal,
                tensor_info=row_infos(frame, row, dtypes), is_streaming=True,
            )
            if rid in self._lifecycle.pending_removes:
                self._tensors.ack_unread_tensors([edge])
                continue
            stream_buf = per_request_info[rid].stream_buffers.get(row.signal)
            if stream_buf is None:
                self._ack_unbuffered(rid, row.request_id, [edge])
                continue
            # a row is one edge's items: a chunked producer's pieces join
            stream_buf.pre_read_register_joined(frame.uuids[row.start:row.start + row.count])
            reads.append((rid, [edge]))
        if reads:
            try:
                futures = self._tensors.start_read_tensors_multi(reads)
            except Exception as err:
                self._fail_intake([rid for rid, _ in reads], err, "streamed input read failed")
                return
            self._wakeup_event.register_futures(futures)

    def _fail_intake(self, rids: list[int], err: Exception, what: str) -> None:
        """Fail ``rids`` for an error taking in their inputs (a CUDA OOM in
        the H2D read, say), rather than letting it escape the message loop:
        nothing else would fail them, and the rest of the drained messages
        would be lost. Recorded for a TP step's read check, as a failed read
        is (see ``check_ready_tensors``), and reported once per request."""
        self._lockstep.record_read_failures(
            r for r in rids if r in self._request_state.per_request_info
        )
        rids = [r for r in rids if r not in self._lifecycle.failed_rids]
        if not rids:
            return
        text, _ = client_error_text(err, f"Worker {self._worker_id}: {what}", log=logger)
        self._lifecycle.fail({rid: f"Error in worker: {text}" for rid in rids})

    def _ack_unbuffered(self, rid: int, request_id: str, edges: list[GraphEdge]) -> None:
        """Streamed edges of a known request with no stream buffer here:
        nothing will consume them, so ack them unread for the producer to
        free. Expected when the request's ingest failed on this worker."""
        if rid not in self._lifecycle.failed_rids:
            logger.warning(
                "Worker %s: no stream buffer for %s of request %s; dropping it",
                self._worker_id, [edge.name for edge in edges], request_id,
            )
        self._tensors.ack_unread_tensors(edges)

    # -- ready tensors and streams ----------------------------------------------

    def check_ready_tensors(self) -> None:
        """Poll for completed RDMA transfers, feed ready graph edges to worker graph queues."""
        self._wakeup_event.drain()
        ready = self._tensors.get_ready_tensors()
        failed_reads = self._tensors.take_failed_reads()
        if failed_reads:
            self._lockstep.record_read_failures(
                f.request_id for f in failed_reads
                if f.request_id in self._request_state.per_request_info
            )
            # reported once per request; the tensor manager logged the read's
            # error where it failed
            self._lifecycle.fail({
                f.request_id: "Error in worker: input tensor read failed: " + client_error_text(
                    f.error,
                    f"Worker {self._worker_id}: input tensor read of request {f.request_id} failed",
                    log=logger, traceback=False,
                )[0]
                for f in failed_reads
                if f.request_id not in self._lifecycle.failed_rids
            })
        if ready:
            self._lockstep.note_reads(ready)
        enable_nvtx = self._enable_nvtx
        for rid, edges in ready.items():
            streaming = [e for e in edges if e.is_streaming]
            normal = [e for e in edges if not e.is_streaming]

            if enable_nvtx:
                range_push("check_ready-tensors.route_streaming")
            for edge in streaming:
                self._streams.route_tensor(rid, edge)

            if enable_nvtx:
                range_pop(synchronize=False)
                range_push("process_new_inputs.process_inputs")

            if normal:
                self._runtime.ingest_inputs_batch(_one_rids_edges(rid, normal), can_buffer=True)
            if enable_nvtx:
                range_pop(synchronize=False)

    def poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        # Every ready chunk is popped first and ingested in one call (and one
        # Python <> Rust roundtrip)
        laps = PhaseLaps("worker.main.poll_stream_buffers.")
        streams = self._streams
        polled = streams.poll()
        laps.lap("scan_and_pop")
        if not polled:
            return

        # Streaming edges go through the same path as regular ones:
        # ReadySignals.is_ready_for_streaming flips on as soon as the streaming
        # inputs are the only ones missing. The final-chunk signal rides the
        # synthetic edge to the consuming pass, which reports the partition done
        # in postprocess, NOT here, where an earlier in-flight pass's WGD could
        # read it before the final output chunk is emitted.
        refused = set(self._runtime.ingest_inputs_batch(
            streams.edge_block(polled),
            # important: only ingest for this loop iter!
            can_buffer=False,
            is_streaming=True,
        ))
        # Indices into the block, which is `polled` order.
        for i, (rid, se) in enumerate(polled):
            if i in refused:
                streams.return_edge(rid, se)
            else:
                streams.mark_ingested(rid, se)
        laps.lap("ingest")
