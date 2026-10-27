"""A stream ends only after its last item arrives, whichever message lands first.

The producer worker sends each stream descriptor straight to the consumer
worker, and its WORKER_GRAPHS_DONE to the conductor, which then sends the
consumer ``producer_done``. Those are different peers on the consumer's PULL
socket, so zmq does not order them: ``producer_done`` can arrive first. The
consumer then used to emit an empty final chunk and end its partition, and
the real last item came after as a second final chunk.

These run the whole chain through the real code: a producer graph runtime
(both of them) routes and sends a streamed item, the conductor processes the
WORKER_GRAPHS_DONE and sends ``producer_done``, and a consumer worker takes
the two messages in a chosen order. Only transport reads are stubbed.
"""
from types import SimpleNamespace

import pytest
import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import TensorCommunicationManager
from mstar.communication.wire import decode
from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
    PartitionDefinition,
    PartitionState,
    StreamingConnectionState,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SendInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import ForwardPassArgs, WorkerGraph
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import (
    ConductorMessageType,
    InputSignals,
    WorkerGraphsDone,
    WorkerMessageType,
)
from mstar.worker.worker import Worker

PRODUCER = "worker_0"
CONSUMER = "worker_1"
EDGE = "new_token"
RID = "r1"


class _StubTensorManager:
    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts
    set_persist = TensorCommunicationManager.set_persist
    dereference = TensorCommunicationManager.dereference
    dereference_batch = TensorCommunicationManager.dereference_batch
    dereference_batch_uniform = TensorCommunicationManager.dereference_batch_uniform
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable
    stamps_shm_placement = False

    def __init__(self, bookkeeping):
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def refresh_shm_placement(self, edges):
        pass

    def _cleanup_by_uuid(self, uuid, registered=None):
        if self.tensor_store.check_uuid_presence(uuid):
            self.tensor_store.remove_tensor(uuid)
        elif registered is None:
            self.tensor_store.bookkeeping.forget_tensor(uuid)


class _Capture:
    def __init__(self):
        self.messages = []

    def send(self, dest, msg):
        self.messages.append((dest, msg))


def _info(uuid):
    return TensorPointerInfo(
        dims=[1], dtype=torch.int64, nbytes=8, address=0, stride=[1],
        uuid=uuid, source_session_id="h:1", source_entity=PRODUCER,
    )


class _Producer:
    """``LLM`` on worker_0 streams ``new_token`` into ``snac_decoder`` on
    worker_1, each in its own walk and partition, as Orpheus splits them."""

    def __init__(self, kind, tmp_path):
        llm = WorkerGraph(
            section=GraphNode(
                name="LLM", input_names={"text_inputs"},
                outputs=[GraphEdge(name=EDGE, next_node="snac_decoder",
                                   is_streaming=True)],
            ),
            graph_walks={"decode"}, ranks=[0], worker_graph_id=0,
        )
        common = dict(
            my_worker_id=PRODUCER, my_worker_graphs=[llm],
            all_wg_ids_to_graph_walks={0: {"decode"}, 1: {"snac_chunk"}},
            all_wg_ids_to_dyn_loops={0: set(), 1: set()},
            all_wg_ids_to_nodes={0: {"LLM"}, 1: {"snac_decoder"}},
            node_to_partition={"LLM": "LLM", "snac_decoder": "SNAC"},
            sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        )
        self.kind = kind
        if kind == "python":
            self.book = PythonTensorBookkeeping()
            self.capture = _Capture()
            self.rt = PythonGraphRuntime(
                tensor_manager=_StubTensorManager(self.book),
                communicator=self.capture, **common,
            )
        else:
            pytest.importorskip("mstar_rust")
            from mstar_rust import ZmqCommunicator

            from mstar.communication.tensor_store import RustTensorBookkeeping
            from mstar.graph.runtime.rust import RustGraphRuntime

            self.book = RustTensorBookkeeping()
            comm = ZmqCommunicator(PRODUCER, str(tmp_path))
            self.inboxes = {
                peer: ZmqCommunicator(peer, str(tmp_path))
                for peer in ("conductor", "api_server", CONSUMER)
            }
            self.rt = RustGraphRuntime(
                bookkeeping=self.book,
                communicator=SimpleNamespace(_inner=comm), **common,
            )
        self.rid = self.rt.add_request(
            request_id=RID, partition="LLM", graph_walk="decode",
            partition_worker_graph_ids=[0],
            worker_graph_to_workers=ParallelList([0, 1], [[PRODUCER], [CONSUMER]]),
            loop_caps={},
        )

    def step(self, uuid):
        """One LLM pass that streams one item. Returns (descriptor, wgd)."""
        block = ColumnarEdgeSpecs.empty()
        block.add(self.rid, "text_inputs", [], False, next_node="LLM")
        self.rt.ingest_inputs_batch(block)
        assert self.rt.pop_rids("LLM", "decode", [self.rid]) is not None
        self.book.put_tensor(uuid, _info(uuid))
        self.book.increment_ref(uuid, 1)
        out = self.rt.complete_and_route_batch(RouteInput(
            partition="LLM", graph_walk="decode", node_name="LLM",
            output_signals=[EDGE], wg_ids=ParallelList([self.rid], [0]),
            tensors=[uuid], num_tensors=[1],
        ))
        fwd = CurrentForwardPassInfo(
            request_id=RID, graph_walk="decode", fwd_index=0, random_seed=0,
            max_tokens=8, partition_name="LLM",
        )
        self.rt.send_outputs(SendInput(
            out.completion_id, ParallelList([self.rid], [fwd]), ParallelList([], []),
        ))
        return self._take()

    def _take(self):
        if self.kind == "python":
            got, self.capture.messages = self.capture.messages, []
        else:
            got = [
                (peer, m) for peer in ("conductor", CONSUMER)
                for m in _collect(self.inboxes[peer])
            ]
        to_consumer = [m.body for d, m in got if d == CONSUMER]
        to_conductor = [m.body for d, m in got if d == "conductor"]
        assert len(to_consumer) == 1 and len(to_conductor) == 1
        assert isinstance(to_conductor[0], WorkerGraphsDone)
        return to_consumer[0], to_conductor[0]


def _collect(box):
    out = []
    first = box.recv_timeout(1000)
    if first is None:
        return out
    out.append(decode(first))
    while (more := box.recv_timeout(50)) is not None:
        out.append(decode(more))
    return out


class _EndsLLM:
    """The LLM partition ends on its ``passes``-th pass."""

    def __init__(self, passes):
        self.left = passes

    def get_partition_forward_pass_args(self, partition_name, partition_metadata,
                                        persist_signals, incoming_connections=None):
        assert partition_name == "LLM"
        self.left -= 1
        return ForwardPassArgs(
            full_metadata=partition_metadata, inputs=[], unpersist_tensors=[],
            request_done=self.left == 0,
        )


def _conductor(passes=1):
    c = object.__new__(Conductor)
    c.model = _EndsLLM(passes)
    c.producer_triggered_partitions = set()
    c.enable_prof = False
    c.communicator = _Capture()
    c._all_worker_graph_ids_to_graph_walks = {0: {"decode"}, 1: {"snac_chunk"}}
    c._un_persist_tensors = lambda request_id, tensors: None

    def next_pass(request_id, pname, walk):
        c.requests[request_id].partition_states[pname].current_worker_graph_ids = {0}
    c._set_partition_worker_graph_ids = next_pass
    meta = CurrentForwardConductorMetadata
    req = RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: [PRODUCER], 1: [CONSUMER]},
        all_worker_graph_ids={0, 1}, max_output_tokens=10_000, random_seed=0,
        resource_configs={},
        partition_states={
            "LLM": PartitionState("LLM", meta(graph_walk="decode", is_prefill=False),
                                  current_worker_graph_ids={0}),
            "SNAC": PartitionState("SNAC", meta(graph_walk="snac_chunk", is_prefill=False),
                                   current_worker_graph_ids={1}),
        },
        partition_definitions={
            "LLM": PartitionDefinition(name="LLM", graph_walks={"decode"}),
            "SNAC": PartitionDefinition(name="SNAC", graph_walks={"snac_chunk"},
                                        producer_partitions=["LLM"]),
        },
        streaming_connections={("LLM", "SNAC", EDGE): StreamingConnectionState(
            from_partition="LLM", to_partition="SNAC", edge_name=EDGE,
        )},
    )
    c.requests = {RID: req}
    return c


def _conduct(c, wgds) -> InputSignals:
    """Process the producer's reports; the last one ends LLM. Returns the
    ``producer_done`` the consumer is sent."""
    for wgd in wgds:
        for part in c._process_worker_graphs_done(wgd):
            c._process_done_forward(RID, part)
    sent = [(d, m) for d, m in c.communicator.messages
            if m.message_type == WorkerMessageType.INPUT_SIGNALS]
    assert len(sent) == 1 and sent[0][0] == CONSUMER
    done = sent[0][1].body
    assert done.producer_done == {"LLM"}
    return done


def _consumer(policy):
    """A consumer worker holding one StreamBuffer; reads land at once."""
    w = Worker.__new__(Worker)
    w._walk_drivers = {}
    w._walk_driven_edges = {}
    buf = StreamBuffer(1, EDGE, "LLM", policy)
    w.worker_id = CONSUMER
    w._reads_done_sent = set()
    w.enable_nvtx = False
    w._rid = lambda rid: 1
    w.request_state = SimpleNamespace(per_request_info={
        1: SimpleNamespace(stream_buffers={EDGE: buf}, ended_streams=set()),
    })
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    w.tensor_manager = SimpleNamespace(
        start_read_tensors=lambda *a, **k: [],
        get_tensor=lambda uuid: torch.tensor([uuid]),
        dereference_batch_uniform=lambda ids: None,
        store_and_return_tensor_info=lambda rid, d, **k: {
            name: [object()] for name in d
        },
    )
    w._consumer_node_cache = {EDGE: "snac_decoder"}
    w._stream_partition = {EDGE: "SNAC"}
    return w, buf


def _deliver(w, descriptor):
    """The descriptor arrives and its read completes."""
    w._process_new_inputs(descriptor)
    for edge in descriptor.inputs:
        w._route_streaming_tensor(1, edge)


def _chunks(w, buf):
    """Every chunk the consumer would take now, as (num_items, is_final)."""
    out = []
    while (se := w._pop_streaming_edge(buf, EDGE, 1)) is not None:
        out.append((se.chunk.num_items, se.chunk.is_final))
        if se.chunk.is_final:
            node_done, partition_done = w._settle_final_streams("snac_decoder", {1: {EDGE}})
            assert node_done == partition_done == {1}
    return out


@pytest.fixture(params=["python", "rust"])
def kind(request):
    return request.param


def test_done_before_the_last_item_waits_for_it(kind, tmp_path):
    producer = _Producer(kind, tmp_path)
    descriptor, wgd = producer.step(uuid=11)
    done = _conduct(_conductor(), [wgd])

    w, buf = _consumer(FixedChunkPolicy(25))
    w._process_new_inputs(done)
    assert _chunks(w, buf) == [], "ended the stream before its last item arrived"

    _deliver(w, descriptor)
    assert _chunks(w, buf) == [(1, True)]
    assert _chunks(w, buf) == []


def test_done_before_the_last_of_several_passes(kind, tmp_path):
    """The count spans every pass's report, not just the one that ended it."""
    producer = _Producer(kind, tmp_path)
    first, wgd1 = producer.step(uuid=11)
    second, wgd2 = producer.step(uuid=12)
    done = _conduct(_conductor(passes=2), [wgd1, wgd2])

    w, buf = _consumer(FixedChunkPolicy(1))
    _deliver(w, first)
    w._process_new_inputs(done)
    assert _chunks(w, buf) == [(1, False)]
    _deliver(w, second)
    assert _chunks(w, buf) == [(1, True)]
    assert done.producer_done_items == {EDGE: 2}


@pytest.mark.parametrize("flush", ["one_per_tick", "all_at_once"])
def test_a_send_backlog_that_drains_after_done(kind, tmp_path, flush):
    """With non-blocking sends and a per-peer backlog (#217), descriptors for a
    backed-up consumer wait on the producer while its completions reach an idle
    conductor at once. Every pass's descriptor then lands after
    ``producer_done``, in order, and the stream ends with the last one."""
    producer = _Producer(kind, tmp_path)
    backlog, wgds = [], []
    for uuid in (11, 12, 13):
        descriptor, wgd = producer.step(uuid=uuid)
        backlog.append(descriptor)
        wgds.append(wgd)
    done = _conduct(_conductor(passes=3), wgds)

    if flush == "one_per_tick":
        w, buf = _consumer(FixedChunkPolicy(1))
        w._process_new_inputs(done)
        assert _chunks(w, buf) == [], "ended the stream before its backlog drained"
        _deliver(w, backlog[0])
        assert _chunks(w, buf) == [(1, False)]
        _deliver(w, backlog[1])
        assert _chunks(w, buf) == [(1, False)]
        _deliver(w, backlog[2])
        assert _chunks(w, buf) == [(1, True)]
    else:
        w, buf = _consumer(FixedChunkPolicy(25))
        w._process_new_inputs(done)
        assert _chunks(w, buf) == [], "ended the stream before its backlog drained"
        for descriptor in backlog:
            _deliver(w, descriptor)
        assert _chunks(w, buf) == [(3, True)]
    assert _chunks(w, buf) == []
    assert done.producer_done_items == {EDGE: 3}


def test_items_before_done_end_the_stream_as_before(kind, tmp_path):
    producer = _Producer(kind, tmp_path)
    descriptor, wgd = producer.step(uuid=11)
    done = _conduct(_conductor(), [wgd])

    w, buf = _consumer(FixedChunkPolicy(25))
    _deliver(w, descriptor)
    assert _chunks(w, buf) == []
    w._process_new_inputs(done)
    assert _chunks(w, buf) == [(1, True)]
    assert _chunks(w, buf) == []


def test_a_stream_that_sent_nothing_ends_at_once():
    done = _conduct(_conductor(), [WorkerGraphsDone(
        request_id=RID, worker_graph_ids=[0], is_first_tp_rank=True, partition_name="LLM",
    )])
    w, buf = _consumer(FixedChunkPolicy(25))
    w._process_new_inputs(done)
    assert _chunks(w, buf) == [(0, True)]


def test_a_continuing_stream_waits_for_its_last_item_too(tmp_path):
    """Qwen3-Omni's Thinker to Talker: empty chunks keep the consumer going
    after the producer ends, and one must not stand in for the last item."""
    producer = _Producer("python", tmp_path)
    descriptor, wgd = producer.step(uuid=11)
    done = _conduct(_conductor(), [wgd])

    w, buf = _consumer(FixedChunkPolicy(1, continue_after_done=True))
    w._process_new_inputs(done)
    assert w._pop_streaming_edge(buf, EDGE, 1) is None, "an empty chunk overtook the last item"
    _deliver(w, descriptor)
    se = w._pop_streaming_edge(buf, EDGE, 1)
    assert se.chunk.num_items == 1 and not se.chunk.is_final


def test_counts_are_credited_by_edge_from_every_rank():
    """A report names the stream items its worker sent since the last one,
    whichever partition's completion carries them, and a TP follower's items
    are its own."""
    c = _conductor()
    req = c.requests[RID]
    c._process_worker_graphs_done(WorkerGraphsDone(
        request_id=RID, worker_graph_ids=[], is_first_tp_rank=False,
        partition_name="SNAC", stream_items_sent={EDGE: {CONSUMER: 2, PRODUCER: 1}},
    ))
    c._process_worker_graphs_done(WorkerGraphsDone(
        request_id=RID, worker_graph_ids=[0], is_first_tp_rank=True,
        partition_name="LLM", stream_items_sent={EDGE: {CONSUMER: 1}},
    ))
    assert req.streaming_connections["LLM", "SNAC", EDGE].items_sent == {CONSUMER: 3, PRODUCER: 1}
    c._send_producer_done(RID, "LLM", "SNAC")
    (dest, msg), = c.communicator.messages
    assert dest == CONSUMER and msg.body.producer_done_items == {EDGE: 3}


def test_every_edge_between_two_partitions_gets_its_count():
    """Qwen3-Omni streams ``thinker_states`` and ``thinker_mask`` from Thinker
    to Talker: one connection per edge, and one done signal carrying both
    counts."""
    c = _conductor()
    c.requests[RID].streaming_connections[("LLM", "SNAC", "mask")] = StreamingConnectionState(
        from_partition="LLM", to_partition="SNAC", edge_name="mask",
    )
    done = _conduct(c, [WorkerGraphsDone(
        request_id=RID, worker_graph_ids=[0], is_first_tp_rank=True, partition_name="LLM",
        stream_items_sent={EDGE: {CONSUMER: 4}, "mask": {CONSUMER: 4}},
    )])
    assert done.producer_done_items == {EDGE: 4, "mask": 4}


def test_the_count_survives_the_wire():
    from mstar.communication.wire import encode
    from mstar.utils.ipc_format import ConductorMessage, WorkerMessage

    wgd = ConductorMessage(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
        request_id=RID, worker_graph_ids=[0], is_first_tp_rank=True,
        stream_items_sent={EDGE: {CONSUMER: 3}},
    ))
    assert decode(encode(wgd)).body.stream_items_sent == {EDGE: {CONSUMER: 3}}
    done = WorkerMessage(WorkerMessageType.INPUT_SIGNALS, InputSignals(
        RID, [], None, producer_done={"LLM"}, producer_done_items={EDGE: 3},
    ))
    assert decode(encode(done)).body.producer_done_items == {EDGE: 3}
