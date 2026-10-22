"""The API server holds a finished request open until every chunk the workers
emitted has arrived.

``request_complete`` (worker -> conductor -> API server) and a worker's last
result chunks (worker -> API server) travel on different connections, so
completion can overtake them. The conductor counts the chunks each worker
reports emitting (``WorkerGraphsDone.output_signal_names``, one name per
chunk, from every rank), and the API server compares that with what it has
received.

These drive the real Python graph runtime, the conductor's report handling and
the API server's bookkeeping; ``test/rust/test_delivery_chunk_counts_rust.py``
runs the same request on the Rust runtime.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorker
from mstar.api_server.request_types import ResultTensors, ResultTokens
from mstar.communication.tensor_store import TensorStore
from mstar.communication.tensors import TensorCommunicationManager
from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import CurrentForwardConductorMetadata, PartitionState
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SendInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import ConductorMessageType, WorkerGraphsDone

WORKER = "worker_0"
PREFILL, DECODE, LOOP = "prefill_text", "thinker_decode", "thinker_decode_loop"
NODE = "Thinker"

# (prefill runs, decode steps): Qwen3-Omni's "text, image, text" prompt runs
# prefill_text twice (the image walk emits on its own worker graph, which
# changes nothing here); one prefill with one decode step is any AR model whose
# decode stops after a token (EOS, or max_output_tokens=2).
SHAPES = {
    "text_image_text": (2, 3),
    "one_decode_step": (1, 1),
    "plain": (1, 3),
}


def worker_graphs(decode_steps: int) -> list[WorkerGraph]:
    prefill = GraphNode(name=NODE, input_names=["text_inputs"], outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True),
    ])
    decode = Loop(name=LOOP, max_iters=decode_steps, outputs=[], section=GraphNode(
        name=NODE, input_names=["text_inputs"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
            GraphEdge(next_node=NODE, name="text_inputs"),
        ],
    ))
    return [
        WorkerGraph(section=prefill, graph_walks={PREFILL}, ranks=[0], worker_graph_id=0),
        WorkerGraph(section=decode, graph_walks={DECODE}, ranks=[0], worker_graph_id=1),
    ]


RUNTIME_KWARGS = dict(
    my_worker_id=WORKER,
    all_wg_ids_to_graph_walks={0: {PREFILL}, 1: {DECODE}},
    all_wg_ids_to_dyn_loops={0: set(), 1: {LOOP}},
    all_wg_ids_to_nodes={0: {NODE}, 1: {NODE}},
    node_to_partition={NODE: "default"},
    sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
)


def tensor_info(uuid: int) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[1], dtype=torch.int64, nbytes=8, address=0, stride=[1], uuid=uuid,
        source_session_id="h:1", source_entity=WORKER,
    )


def drive(runtime, put, prefills: int, decode_steps: int, inline: bool = False) -> None:
    """Run one request through ``prefills`` prefill walks and a decode loop.

    ``put(uuid)`` registers a produced tensor with the runtime's store, as the
    worker's prologue does. ``inline`` sends the decode tokens as inline values
    (the Rust runtime's RESULT_TOKENS path).
    """
    rid = runtime.add_request(
        request_id="r1", partition="default", graph_walk=PREFILL,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
        loop_caps={},
    )
    uuid = 100

    def step(walk: str, wg_id: int, inline_value=None):
        nonlocal uuid
        assert runtime.pop_rids(NODE, walk, [rid]) is not None
        signals = runtime.get_output_signals(NODE, walk)
        uuids = []
        for _ in signals:
            uuid += 1
            put(uuid)
            uuids.append(uuid)
        extra = {} if inline_value is None else dict(
            inline_signal="new_token", inline_values=[inline_value],
        )
        out = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk=walk, node_name=NODE, output_signals=signals,
            wg_ids=ParallelList([rid], [wg_id]), tensors=uuids,
            num_tensors=[1] * len(signals), **extra,
        ))
        runtime.send_outputs(SendInput(
            completion_id=out.completion_id, per_request_info=ParallelList([rid], [None]),
            new_token_counts=ParallelList([rid], [{}]),
        ))

    def ingest():
        block = ColumnarEdgeSpecs.empty()
        block.add_edge(rid, GraphEdge(name="text_inputs", next_node=NODE))
        assert runtime.ingest_inputs_batch(block, can_buffer=True) == []

    for _ in range(prefills):
        runtime.set_walk(rid, "default", PREFILL)
        ingest()
        step(PREFILL, 0)
    runtime.set_walk(rid, "default", DECODE)
    ingest()
    for i in range(decode_steps):
        step(DECODE, 1, inline_value=7 + i if inline else None)


def expected_counts(reports: list[WorkerGraphsDone]) -> dict[str, int]:
    """What the conductor puts in ``request_complete``, from the real handler."""
    conductor = Conductor.__new__(Conductor)
    conductor.enable_prof = False
    request = RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: [WORKER], 1: [WORKER]}, all_worker_graph_ids={0, 1},
        max_output_tokens=64, random_seed=0, resource_configs={},
        partition_states={"default": PartitionState(
            partition_name="default",
            metadata=CurrentForwardConductorMetadata(graph_walk=PREFILL, is_prefill=True),
        )},
    )
    conductor.requests = {"r1": request}
    for body in reports:
        conductor._process_worker_graphs_done(body)
    return request.output_chunk_counts


def api_server() -> PreprocessWorker:
    worker = PreprocessWorker.__new__(PreprocessWorker)
    worker.received_chunk_counts = {"r1": {}}
    worker.per_request_reading_tensors = {"r1": 0}
    worker.result_tensor_input_queue = SimpleNamespace(put=lambda item: None)
    worker._signal = lambda: None
    return worker


def deliver(worker: PreprocessWorker, chunk) -> None:
    """One chunk message as the API server's message loop hands it over."""
    if isinstance(chunk, ResultTokens):
        worker.new_result_tokens(chunk.request_ids, chunk.values, chunk.signal, chunk.modality)
    else:
        assert isinstance(chunk, ResultTensors)
        worker.new_result_tensors(chunk)


def assert_holds_until_last_chunk(chunks: list, reports: list[WorkerGraphsDone]) -> None:
    counts = expected_counts(reports)
    assert counts == {"new_token": len(chunks)}
    worker = api_server()
    for i, chunk in enumerate(chunks):
        assert not worker.received_final_chunks("r1", counts), (
            f"request reported delivered with {i} of {len(chunks)} chunks received"
        )
        deliver(worker, chunk)
    assert worker.received_final_chunks("r1", counts)


class _Recorder:
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def chunks(self):
        return [m.body for e, m in self.sent if e == "api_server"]

    def reports(self):
        return [m.body for e, m in self.sent
                if e == "conductor" and m.message_type == ConductorMessageType.WORKER_GRAPHS_DONE]


class _TensorManager:
    """What the Python runtime needs of the manager: a real store underneath."""

    stamps_shm_placement = False
    refresh_shm_placement = TensorCommunicationManager.refresh_shm_placement
    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts

    def __init__(self):
        self.tensor_store = TensorStore()

    def set_persist(self, uuid, persist):
        self.tensor_store.set_metadata(uuid, persist=persist)

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def dereference_batch(self, uuids, counts):
        self.tensor_store.dereference_batch(uuids, counts)

    def dereference_batch_uniform(self, uuids, n=1):
        pass


def python_runtime(decode_steps: int):
    manager = _TensorManager()
    recorder = _Recorder()
    runtime = PythonGraphRuntime(
        my_worker_graphs=worker_graphs(decode_steps), tensor_manager=manager,
        communicator=recorder, **RUNTIME_KWARGS,
    )

    def put(uuid):
        manager.tensor_store.put_tensor("r1", uuid, torch.ones(1), tensor_info(uuid))

    return runtime, put, recorder


@pytest.mark.parametrize("shape", list(SHAPES))
def test_request_is_held_until_its_last_chunk(shape):
    prefills, decode_steps = SHAPES[shape]
    runtime, put, recorder = python_runtime(decode_steps)
    drive(runtime, put, prefills, decode_steps)
    assert len(recorder.chunks()) == prefills + decode_steps
    assert_holds_until_last_chunk(recorder.chunks(), recorder.reports())


class _ScriptedApiComm:
    """The API server's inbox: one scripted batch per loop pass, stopping the
    loop when the script runs out. Records whether the client had been
    released before each pass."""

    def __init__(self, server, batches, req):
        self.server, self.batches, self.req = server, list(batches), req
        self.released_before = []

    def get_all_new_messages(self):
        self.released_before.append(self.req.event.is_set())
        if not self.batches:
            self.server.running = False
            return []
        return self.batches.pop(0)


def test_completion_that_overtakes_the_last_chunks_waits_for_them():
    """Through the API server's real message loop: ``request_complete``
    arrives after two of the five chunks, the rest a pass later each. The
    client is released only once the fifth has been read, never with a
    truncated reply."""
    import collections
    import queue
    import threading

    from mstar.api_server.entrypoint import APIServer, PendingRequest
    from mstar.api_server.request_types import APIServerMessage, RequestComplete, ResultChunk
    from mstar.profile.format import RequestProfile

    runtime, put, recorder = python_runtime(SHAPES["text_image_text"][1])
    drive(runtime, put, *SHAPES["text_image_text"])
    chunks = recorder.chunks()
    complete = APIServerMessage(message_type="request_complete", body=RequestComplete(
        request_id="r1", output_chunk_counts=expected_counts(recorder.reports()),
        conductor_ingest_time=0.0, conductor_finish_time=0.0,
    ))

    worker = PreprocessWorker.__new__(PreprocessWorker)
    worker.received_chunk_counts = {"r1": {}}
    worker.per_request_reading_tensors = {"r1": 0}
    worker.output_queue = queue.Queue()
    worker.profile_queue = queue.Queue()
    worker.reads_done_queue = queue.Queue()
    worker._signal = lambda: None
    # the data worker's thread, collapsed: a queued read completes at once
    worker.result_tensor_input_queue = SimpleNamespace(put=lambda item: worker.output_queue.put(
        ResultChunk(request_id=item.request_id, modality="text", data=b"x", metadata={}),
    ))

    server = object.__new__(APIServer)
    server.running = True
    server._liveness_interval_s = 3600.0
    server._conductor_exited = lambda: None
    server.request_lock = threading.Lock()
    server.recently_completed = collections.OrderedDict()
    server._recently_completed_ttl = 3600.0
    server.log_stats = False
    server.metrics = None
    server.enable_nvtx = False
    server.preprocess_worker = worker
    req = PendingRequest(streaming=False, input_modalities=["text", "image", "text"],
                         output_modalities=["text"], profile=RequestProfile(rid="r1"))
    server.pending_requests = {"r1": req}

    def msg(body):
        return APIServerMessage(message_type="result_tensors", body=body)

    batches = [[msg(chunks[0]), msg(chunks[1])], [complete], []] + [[msg(c)] for c in chunks[2:]] + [[]]
    server.communicator = _ScriptedApiComm(server, batches, req)
    server._process_messages()

    assert req.event.is_set() and req.error is None
    assert len(req.chunks) == len(chunks)
    # Not released until the pass after the one that read the last chunk (the
    # loop prunes at the top of a pass); before the fix it was released right
    # after request_complete.
    released = server.communicator.released_before
    last_chunk_pass = len(batches) - 2
    assert released.index(True) == last_chunk_pass + 1, released


def test_every_rank_reports_the_chunks_it_sent():
    """Under TP a replicated output leaves from rank 0 only and a sharded one
    from every rank, each rank's copy its own message, so the conductor counts
    every rank's report rather than only rank 0's."""
    conductor = Conductor.__new__(Conductor)
    conductor.enable_prof = False
    request = RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: ["worker_0", "worker_1"]}, all_worker_graph_ids={0},
        max_output_tokens=64, random_seed=0, resource_configs={},
        partition_states={"default": PartitionState(
            partition_name="default",
            metadata=CurrentForwardConductorMetadata(graph_walk=DECODE, is_prefill=False),
        )},
    )
    conductor.requests = {"r1": request}
    for first, names in [(True, ["sharded", "replicated"]), (False, ["sharded"])]:
        conductor._process_worker_graphs_done(WorkerGraphsDone(
            request_id="r1", worker_graph_ids=[0], is_first_tp_rank=first,
            output_signal_names=names,
        ))
    assert request.output_chunk_counts == {"sharded": 2, "replicated": 1}
