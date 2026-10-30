"""A tensor the transfer engine refuses to register for send fails only its
own request, and nothing of that request is sent. Before, the refusal raised
out of ``_register_outputs`` and the worker's main-loop handler failed every
request in the batch (#488)."""

import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from test_graph_runtime import (
    _build_on_real_manager,
    _complete,
    _fwd_info,
    _ingest,
    _RecordingCommunicator,
)

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    AsyncMooncakeReader,
    MooncakeCommunicationManager,
    StoredOutputs,
    TensorCommunicationManager,
)
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
from mstar.graph.runtime.base import FreedTensors, RouteOutput, SendInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import BatchedModelOutput
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import ConductorMessageType, RemoveRequest
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker


class _Communicator(BaseCommunicator):
    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


class _Engine:
    """Registers like a transfer engine, but refuses the given pointers the
    way Mooncake refuses an overlapping region (-7)."""

    def __init__(self, refuse=()):
        self.refuse = set(refuse)
        self.registered = []
        self.unregistered = []

    def register_memory(self, ptr, nbytes):
        if ptr in self.refuse:
            return -7
        self.registered.append(ptr)
        return 0

    def unregister_memory(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def batch_transfer_sync_read(self, session_id, local_ptrs, remote_ptrs, nbytes):
        return 0


def _manager(engine):
    mgr = MooncakeCommunicationManager.__new__(MooncakeCommunicationManager)
    TensorCommunicationManager.__init__(mgr, "w0", "w0", "cpu", _Communicator(), engine)
    mgr.protocol = CommProtocol.TCP
    mgr._async_reader = AsyncMooncakeReader(engine, "cpu")
    return mgr


def _store(mgr, rid, *tensors):
    infos = mgr.store_and_return_tensor_info(rid, {"x": list(tensors)})
    return [info.uuid for info in infos["x"]]


def test_a_refused_registration_fails_only_its_request():
    a, bad, c, other = (torch.randn(4) for _ in range(4))
    engine = _Engine(refuse={bad.data_ptr()})
    mgr = _manager(engine)
    try:
        r0 = _store(mgr, "r0", a, bad, c)
        r1 = _store(mgr, "r1", other)

        failed = mgr.register_for_send_uuids(ParallelList(["r0", "r1"], [r0, r1]))

        assert list(failed) == ["r0"]
        assert "request id r0" in failed["r0"] and f"uuid {r0[1]}" in failed["r0"]
        assert [mgr.tensor_store.is_registered(u) for u in r0] == [True, False, False]
        assert mgr.tensor_store.is_registered(r1[0])
        assert engine.registered == [a.data_ptr(), other.data_ptr()]

        # cleanup unregisters only what did register
        mgr.force_cleanup_request("r0")
        assert engine.unregistered == [a.data_ptr()]
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_no_refusal_reports_nothing():
    mgr = _manager(_Engine())
    try:
        uuids = _store(mgr, "r0", torch.randn(4))
        assert mgr.register_for_send_uuids(ParallelList(["r0"], [uuids])) is None
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_the_worker_fails_only_the_refused_request():
    mine, bad, theirs = (torch.randn(4) for _ in range(3))
    mgr = _manager(_Engine(refuse={bad.data_ptr()}))
    try:
        r7 = _store(mgr, 7, mine, bad)
        r8 = _store(mgr, 8, theirs)
        worker = Worker.__new__(Worker)
        worker._walk_drivers = {}
        worker._walk_driven_edges = {}
        worker.tensor_manager = mgr
        worker.scheduler = SimpleNamespace(failed_rids=set())
        reported = []
        worker._fail_requests = reported.append
        route_output = type("RouteOutput", (), {
            "register_uuids": r7 + r8, "register_rids": [7, 7, 8],
        })()

        assert worker._register_outputs(route_output) == {7}

        (errors,) = reported
        assert list(errors) == [7]
        assert errors[7].startswith("Error in worker: Mooncake memory registration failed")
        assert mgr.tensor_store.is_registered(r8[0])
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


@pytest.mark.parametrize("failed", [False, True])
def test_the_python_runtime_sends_nothing_for_a_failed_rid(tmp_path, failed):
    section = GraphNode(
        name="gen", input_names={"tok"},
        outputs=[GraphEdge(name="out", next_node=EMIT_TO_CLIENT)],
    )
    tm, runtime, rid = _build_on_real_manager(section, {"gen"}, tmp_path)
    comm = _RecordingCommunicator()
    runtime._communicator = comm
    _ingest(runtime, rid, [GraphEdge(name="tok", next_node="gen")])
    _, out = _complete(tm, runtime, rid, "gen", {"out": [torch.ones(1)]}, ["out"])

    sent_rids = runtime.send_outputs(SendInput(
        completion_id=out.completion_id,
        per_request_info=ParallelList([rid], [_fwd_info("w")]),
        new_token_counts=ParallelList([], []),
        failed_rids=frozenset({rid}) if failed else frozenset(),
    ))

    sent = sorted(entity for entity, _ in comm.sent)
    if failed:
        assert sent == [] and sent_rids == []
    else:
        assert sent == ["api_server", "conductor"] and sent_rids == [rid]


def _postprocess_worker(refuse, follower=False):
    """A worker whose ``_postprocess_batch`` runs for real, with the engine,
    graph runtime and transfer engine faked around it. The transfer engine
    refuses every tensor of the rids in ``refuse``, step after step."""
    w = Worker.__new__(Worker)
    w._walk_drivers = {}
    w._walk_driven_edges = {}
    w._phase_period = 0
    w.enable_nvtx = w.enable_prof = False
    w.device = torch.device("cpu")
    w.worker_id = "w0"
    w.parallel_nodes = {"llm"} if follower else set()
    w.parallel_leader_nodes = set()
    w._last_active = {}
    w._innermost_loop_by_node = {}
    w.log = SimpleNamespace(stop_checked=[], routed=[], registered=[], sent=[], conductor=[])
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.scheduler.fail_rids = w.scheduler.failed_rids.update
    w.communicator = SimpleNamespace(send=lambda e, m: w.log.conductor.append(m))
    w.request_state = SimpleNamespace(
        per_request_info={0: object(), 1: object()},
        get_fwd_info=lambda rid, p: None,
        get_pending_publish_info=lambda rid, p: None,
        flush_publish_info=lambda rid, p: None,
    )
    w._prematerialize_for_check_stop = (
        lambda outputs, ev, request_ids=None: (outputs.per_rid_outputs, None)
    )
    w._inline_emit_values = lambda *a: (None, None)
    w._count_new_tokens = lambda *a: {}
    w._stream_consumption_batch = lambda rids: []

    def check_stop(node_batch, cpu_outputs, host_rows=None):
        w.log.stop_checked.append(list(node_batch.request_ids))
        return {}

    engine = SimpleNamespace(
        check_stop_for_batch=check_stop,
        extend_prefix_chains=lambda *a, **k: None,
        materialize_publish=lambda *a: None,
        # like the engine's, reads each stopped rid's info
        finalize_stopped_requests=lambda nb, rids: [nb.per_request_info[r] for r in rids],
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda node: engine)

    def store(rids, outputs, signals, **kwargs):
        return StoredOutputs(
            flat_uuids=[100 + r for r in rids], flat_rids=list(rids),
            signal_idxs=[0] * len(rids), num_tensors=[1] * len(rids),
        )

    def register(per_request, skip_cuda_sync):
        w.log.registered.append(list(per_request.keys))
        return {r: "refused" for r in per_request.keys if r in refuse} or None

    w.tensor_manager = SimpleNamespace(
        cleanup_collectable=lambda *a: None,
        store_and_return_tensor_info_batch=store,
        increment_ref_batch_uniform=lambda *a: None,
        dereference_batch_uniform=lambda *a: None,
        register_for_send_uuids=register,
        drop_host_copies=lambda: None,
    )

    def route(route_input):
        rids = list(route_input.wg_ids.keys)
        w.log.routed.append(rids)
        return RouteOutput(
            completion_id=len(w.log.routed),
            register_uuids=list(route_input.tensors), register_rids=rids,
            new_token_output_idxs=[], local_streaming_by_signal={},
            freed_inputs=FreedTensors.none(),
        )

    def send(send_input):
        w.log.sent.append(
            [r for r in send_input.per_request_info.keys
             if r not in send_input.failed_rids]
        )
        return []

    w._graph_runtime = SimpleNamespace(
        cleanup_consumed_inputs=lambda *a: FreedTensors.none(),
        clear_pending_loop_stops=lambda: None,
        stop_loops_batched=lambda **kw: list(kw["loop_names"].keys),
        complete_and_route_batch=route,
        update_dynamic_loop_iters=lambda *a: None,
        send_outputs=send,
        get_rid_string=lambda rid: f"r{rid}",
        supports_inline_emit=False,
    )
    return w


def _decode_step(rids, final_stream=()):
    node_batch = SimpleNamespace(
        request_ids=list(rids), per_request_info=dict.fromkeys(rids),
        failed_requests={}, completion_event=None,
        final_stream_rids=set(final_stream), resource_publish_info={},
        stream_partition_done_rids=set(), cached_rids=set(), withheld_outputs={},
    )
    batch = ScheduledBatch(
        node_name="llm", graph_walk="decode",
        request_to_worker_graph=dict.fromkeys(rids, 0), output_signals=["tok"],
    )
    outputs = BatchedModelOutput(
        per_rid_outputs={r: {"tok": [torch.tensor([r])]} for r in rids},
    )
    pending = SimpleNamespace(
        batch=batch, node_batch=node_batch, node_name="llm",
        partition="p", graph_walk="decode", loop_name="decode",
        speculative_new_iter=False,
    )
    return pending, outputs


def _fail_requests_sent(w):
    return [
        m for m in w.log.conductor
        if m.message_type == ConductorMessageType.FAIL_REQUESTS
    ]


def test_the_speculated_next_step_sends_nothing_of_a_refused_request():
    """Async scheduling submits step N+1 before step N is post-processed, so
    a rid whose registration is refused in N is already in N+1. N+1 must
    neither route, register nor send it, nor report it a second time."""
    w = _postprocess_worker(refuse={0})
    step_n = _decode_step([0, 1])
    step_n1 = _decode_step([0, 1])  # speculated from N while N ran

    Worker._postprocess_batch(w, *step_n)
    assert w.log.sent == [[1]]
    assert w.scheduler.failed_rids == {0}

    Worker._postprocess_batch(w, *step_n1)

    assert w.log.stop_checked == [[0, 1], [1]]
    assert w.log.routed == [[0, 1], [1]]
    assert w.log.registered == [[0, 1], [1]]
    assert w.log.sent == [[1], [1]]
    assert len(_fail_requests_sent(w)) == 1


def test_a_step_of_only_failed_rids_routes_nothing():
    w = _postprocess_worker(refuse=set())
    w.scheduler.failed_rids.add(0)
    pending, outputs = _decode_step([0])

    Worker._postprocess_batch(w, pending, outputs)

    assert w.log.routed == [] and w.log.sent == []
    assert pending.batch.request_to_worker_graph == {}
    assert pending.node_batch.request_ids == []


def test_a_failed_rid_leaves_the_batch_walk_grouping():
    """The batch's rids-by-walk grouping is cached once built; a failed rid
    taken out of an in-flight step must leave it too, or routing reaches the
    rid's row after it is gone."""
    w = _postprocess_worker(refuse=set())
    w.scheduler.failed_rids.add(0)
    pending, outputs = _decode_step([0, 1])
    assert pending.batch.rids_by_walk() == {"decode": [0, 1]}

    Worker._postprocess_batch(w, pending, outputs)

    assert pending.batch.rids_by_walk() == {"decode": [1]}
    assert w.log.routed == [[1]] and w.log.sent == [[1]]


@pytest.mark.parametrize("has_event", [True, False])
def test_a_step_of_only_failed_rids_still_waits_for_its_gpu_work(monkeypatch, has_event):
    """No output of the step leaves, but a later step reuses its capture slot
    while its kernels may still run, so postprocess waits for them anyway."""
    w = _postprocess_worker(refuse=set())
    w.device = SimpleNamespace(type="cuda")
    w.scheduler.failed_rids.add(0)
    pending, outputs = _decode_step([0])
    waits = []
    if has_event:
        pending.node_batch.completion_event = SimpleNamespace(synchronize=lambda: waits.append("event"))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: waits.append("device"))

    Worker._postprocess_batch(w, pending, outputs)

    assert waits == ["event" if has_event else "device"]
    assert w.log.routed == [] and pending.batch.request_to_worker_graph == {}


def test_a_failed_rid_that_took_its_last_stream_chunk_does_not_stop_its_loop():
    """N+1 consumed the failed rid's terminal stream chunk; ending its loop
    would finalize a rid already taken out of the batch, and the raise would
    fail the healthy rows too."""
    w = _postprocess_worker(refuse={0})
    w._innermost_loop_by_node = {("decode", "llm"): "decode"}
    Worker._postprocess_batch(w, *_decode_step([0, 1]))

    pending, outputs = _decode_step([0, 1], final_stream={0, 1})
    Worker._postprocess_batch(w, pending, outputs)

    assert w.log.routed == [[0, 1], [1]]
    assert w.log.sent == [[1], [1]]
    assert pending.node_batch.final_stream_rids == {1}


def test_a_follower_still_routes_a_failed_rid_and_reports_it_once():
    """Rank 0 decides what a follower runs and may still send a step for the
    rid, which needs this step's loop-back; the refused sends stay skipped."""
    w = _postprocess_worker(refuse={0}, follower=True)

    Worker._postprocess_batch(w, *_decode_step([0, 1]))
    Worker._postprocess_batch(w, *_decode_step([0, 1]))

    assert w.log.routed == [[0, 1], [0, 1]]
    assert w.log.sent == [[1], [1]]
    assert len(_fail_requests_sent(w)) == 1


def test_an_overlapping_view_fails_only_its_request_on_real_mooncake():
    """The issue's repro on Mooncake over TCP: a tensor and a view inside it
    overlap, which Mooncake refuses (-7)."""
    pytest.importorskip("mooncake.engine")
    from mstar.communication.tensors import create_tensor_communication_manager

    mgr = create_tensor_communication_manager(
        protocol=CommProtocol.TCP, my_entity_id="w0", hostname="localhost",
        device="cpu", communicator=_Communicator(), tcp_transfer_device="lo",
    )
    h = torch.randn(4, 64)
    r0 = _store(mgr, "r0", h, h[-1])
    r1 = _store(mgr, "r1", torch.randn(4, 64))

    failed = mgr.register_for_send_uuids(ParallelList(["r0", "r1"], [r0, r1]))

    assert list(failed) == ["r0"]
    assert mgr.tensor_store.is_registered(r1[0])
    mgr.force_cleanup_request("r0")
    mgr.force_cleanup_request("r1")


def _real_manager(monkeypatch, rust):
    """Mooncake over TCP, with the bookkeeper the given graph runtime uses."""
    pytest.importorskip("mooncake.engine")
    if rust:
        pytest.importorskip("mstar_rust")
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "1" if rust else "0")
    from mstar.communication.tensors import create_tensor_communication_manager

    mgr = create_tensor_communication_manager(
        protocol=CommProtocol.TCP, my_entity_id="w0", hostname="localhost",
        device="cpu", communicator=_Communicator(), tcp_transfer_device="lo",
    )
    assert type(mgr.tensor_store.bookkeeping).__name__ == (
        "RustTensorBookkeeping" if rust else "PythonTensorBookkeeping"
    )
    return mgr


@pytest.mark.parametrize("rust", [False, True], ids=["python", "rust"])
def test_a_failed_unregister_does_not_stop_the_teardown(monkeypatch, rust):
    """A tensor the engine no longer has registered (Mooncake answers -3)
    leaves no part of the request behind: before, the raise skipped the
    request's other tensors."""
    mgr = _real_manager(monkeypatch, rust)
    tensors = [torch.randn(4, 64) for _ in range(3)]
    uuids = _store(mgr, "r0", *tensors)
    assert mgr.register_for_send_uuids(ParallelList(["r0"], [uuids])) is None
    assert mgr.transfer_engine.unregister_memory(tensors[1].data_ptr()) == 0

    mgr.force_cleanup_request("r0")

    assert mgr.tensor_store.get_all_uuids("r0") == []
    assert not any(mgr.tensor_store.check_uuid_presence(u) for u in uuids)
    # the other two are unregistered: registering them again does not overlap
    for t in (tensors[0], tensors[2]):
        assert mgr.transfer_engine.register_memory(t.data_ptr(), t.nbytes) == 0
        assert mgr.transfer_engine.unregister_memory(t.data_ptr()) == 0


@pytest.mark.parametrize("rust", [False, True], ids=["python", "rust"])
def test_remove_request_finishes_when_an_unregister_fails(monkeypatch, rust):
    """REMOVE_REQUEST runs to the end, so the rid leaves the scheduler and
    the graph runtime; before, the raise stopped it in force cleanup and
    reached the worker's main loop."""
    mgr = _real_manager(monkeypatch, rust)
    t = torch.randn(4, 64)
    uuids = _store(mgr, 3, t)
    assert mgr.register_for_send_uuids(ParallelList([3], [uuids])) is None
    assert mgr.transfer_engine.unregister_memory(t.data_ptr()) == 0
    released = []
    w = Worker.__new__(Worker)
    w._walk_drivers = {}
    w._walk_driven_edges = {}
    w.is_tp_follower = False
    w._rid = lambda wire: 3
    w._in_flight_rids = set()
    w._draining_rids, w._pending_drains, w._reads_done_sent = set(), set(), set()
    w._stream_buffer_rids = set()
    w._pending_removes = set()
    w._last_active = {}
    w.streaming_buffers = {}
    w.tensor_manager = mgr
    w.engine_manager = SimpleNamespace(
        remove_request=lambda rid: None, evictable_nodes=lambda: [],
    )
    w.request_state = SimpleNamespace(remove_request=lambda rid: None)
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.scheduler = SimpleNamespace(
        clear_rid=lambda rid, wire: released.append(("scheduler", rid)),
    )
    w._graph_runtime = SimpleNamespace(
        get_sharding_config=lambda rid: None,
        remove_request=lambda rid: released.append(("runtime", rid)),
    )

    Worker._remove_request(w, RemoveRequest(request_id="r"))

    assert released == [("scheduler", 3), ("runtime", 3)]
    assert mgr.tensor_store.get_all_uuids(3) == []


class _RefusingEngine(_Engine):
    """Refuses every registration and, like Mooncake, answers -3 when asked
    to unregister memory it never registered."""

    def register_memory(self, ptr, nbytes):
        return -7

    def unregister_memory(self, ptr):
        self.unregistered.append(ptr)
        return -3


class _ReadEngine(_Engine):
    """Refuses to register a buffer of ``refuse_nbytes`` bytes, records each
    read, and answers -3 when asked to unregister memory it never registered."""

    def __init__(self, refuse_nbytes):
        super().__init__()
        self.refuse_nbytes = refuse_nbytes
        self.read = []

    def register_memory(self, ptr, nbytes):
        if nbytes == self.refuse_nbytes:
            return -7
        return super().register_memory(ptr, nbytes)

    def unregister_memory(self, ptr):
        self.unregistered.append(ptr)
        return 0 if ptr in self.registered else -3

    def batch_transfer_sync_read(self, session_id, local_ptrs, remote_ptrs, nbytes):
        self.read.extend(local_ptrs)
        return 0


def _edge(*infos):
    edge = GraphEdge(name="x", next_node="n")
    edge.tensor_info = [
        TensorPointerInfo(
            dims=(nbytes // 4,), dtype=torch.float32, stride=(1,), nbytes=nbytes,
            address=0x1000, uuid=uuid, source_session_id="peer:1", source_entity="w1",
        )
        for uuid, nbytes in infos
    ]
    return edge


def test_a_read_whose_buffer_the_engine_refused_fails_only_its_request():
    """r0's second buffer is refused. Its read was submitted anyway, into
    memory the engine never pinned; now r0 fails as on a failed read, r1
    reads as before, and r0's one registered buffer is unregistered."""
    engine = _ReadEngine(refuse_nbytes=16)
    mgr = _manager(engine)
    try:
        f0 = mgr.start_read_tensors("r0", [_edge((70, 32), (71, 16))])
        r1_edge = _edge((80, 32))
        f1 = mgr.start_read_tensors("r1", [r1_edge])
        f1[0].result()
        assert f0[0].done(), "a refused read must not wait on the reader"
        kept_ptr = mgr.tensor_store.get_tensor(70).data_ptr()
        assert not mgr.tensor_store.is_registered(71)

        ready = mgr.get_ready_tensors()
        (failed,) = mgr.take_failed_reads()

        assert list(ready) == ["r1"]
        assert failed.request_id == "r0"
        assert type(failed.error).__name__ == "MemoryRegistrationError"
        assert "uuid 71: -7" in str(failed.error)
        assert engine.read == [mgr.tensor_store.get_tensor(80).data_ptr()]
        assert not mgr.has_inflight_reads("r0")

        # the next poll frees r0's buffers: no transfer ever ran into them
        mgr.get_ready_tensors()
        assert mgr._kept_reads == [] and mgr._unreleasable_buffers == []
        assert engine.unregistered == [kept_ptr]
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_failed_unregister_keeps_the_tensor_alive():
    """Mooncake keeps a region it failed to unregister. Freed, the tensor's
    address went back to the allocator, and the next tensor placed there was
    refused as an overlap (-7), failing an unrelated request."""

    class _StuckEngine(_Engine):
        def unregister_memory(self, ptr):
            super().unregister_memory(ptr)
            return -1

    engine = _StuckEngine()
    mgr = _manager(engine)
    try:
        t = torch.randn(64)
        ptr = t.data_ptr()
        uuids = _store(mgr, "r0", t)
        assert mgr.register_for_send_uuids(ParallelList(["r0"], [uuids])) is None
        del t

        mgr.force_cleanup_request("r0")

        assert engine.unregistered == [ptr]
        assert not mgr.tensor_store.check_uuid_presence(uuids[0])
        assert [b.data_ptr() for b in mgr._unreleasable_buffers] == [ptr], (
            "the tensor was freed while the engine may still hold its region"
        )
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_leader_drops_a_failed_row_its_follower_keeps():
    """One step on rank 0 and on a follower, its row 0 already failed. Rank 0
    takes it out of the in-flight step; the follower, which rank 0 drives,
    keeps it, for the loop-back a step rank 0 may yet send reads. Row 1 runs
    the same on both."""
    leader = _postprocess_worker(refuse=set())
    follower = _postprocess_worker(refuse=set(), follower=True)
    for w in (leader, follower):
        w.scheduler.failed_rids.add(0)

        Worker._postprocess_batch(w, *_decode_step([0, 1]))

    assert leader.log.routed == [[1]] and leader.log.registered == [[1]]
    assert follower.log.routed == [[0, 1]] and follower.log.registered == [[0, 1]]
    assert leader.log.sent == [[1]] and follower.log.sent == [[0, 1]]
