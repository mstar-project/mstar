"""Worker-local rid handles: strings on the wire, recycled ints inside.

The failure mode these guard is silent: a handle that outlives its request
attaches to whichever request is given that handle next, and a string that
reaches a handle-keyed map just misses.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.testing_utils.worker import deliver, make_scheduler, make_worker, tp_sharding
from mstar.utils.ipc_format import (
    ConductorMessageType,
    MessageSource,
    RemoveRequest,
    ScheduleTPNode,
    TensorReceived,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.host_pool import HostPool
from mstar.worker.lockstep import Lockstep
from mstar.worker.micro_scheduler import REMOVED_RID
from mstar.worker.preemption import Preempted
from mstar.worker.rid_table import RidTable
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

# ── the table ──────────────────────────────────────────────────────────────


def test_intern_is_idempotent_across_partitions():
    """One NEW_REQUEST per partition on this worker; all share one handle."""
    t = RidTable()
    h = t.intern("req-a")
    assert t.intern("req-a") == h
    assert len(t) == 1


def test_first_handle_is_zero_and_round_trips():
    """0 is a real handle, so nothing may treat a handle as a truth value."""
    t = RidTable()
    assert t.intern("req-a") == 0
    assert t.handle("req-a") == 0
    assert t.name(0) == "req-a"


def test_unknown_request_has_no_handle():
    assert RidTable().handle("never-seen") is None


def test_released_handle_is_recycled_for_the_next_request():
    t = RidTable()
    a = t.intern("req-a")
    t.intern("req-b")
    t.release(a)
    assert t.handle("req-a") is None
    assert t.intern("req-c") == a
    assert t.name(a) == "req-c"


def test_name_of_a_released_handle_raises():
    t = RidTable()
    h = t.intern("req-a")
    t.release(h)
    with pytest.raises(KeyError):
        t.name(h)


def test_release_is_idempotent():
    t = RidTable()
    h = t.intern("req-a")
    t.release(h)
    t.release(h)
    # A double release must not put the handle on the free list twice, or two
    # later requests would share it.
    assert t.intern("req-b") != t.intern("req-c")


# ── TP-follow messages carry strings ───────────────────────────────────────


def _lockstep(table: RidTable) -> Lockstep:
    return make_scheduler(graph_runtime=SimpleNamespace(get_rid_handle=table.handle)).lockstep


def _follow(*rids: str) -> ScheduleTPNode:
    return ScheduleTPNode(node_name="n", graph_walk="w", request_ids=list(rids))


def test_tp_rids_resolve_to_this_ranks_handles_in_wire_order():
    t = RidTable()
    t.intern("pad")  # so this rank's handles differ from the leader's order
    b, a = t.intern("b"), t.intern("a")
    assert _lockstep(t).tp_rids(_follow("a", "b")) == [a, b]


def test_tp_rid_this_rank_does_not_know_is_the_removed_sentinel():
    t = RidTable()
    a = t.intern("a")
    assert _lockstep(t).tp_rids(_follow("a", "gone")) == [a, REMOVED_RID]


def test_follow_refcount_counts_rids_this_rank_cannot_resolve():
    """Counted by WIRE STRING, so a rid whose NEW_REQUEST has not landed yet
    still gates the drain. Resolving to handles here would drop it, and the
    drain would ACK while its batch sat at the head of the TP FIFO."""
    t = RidTable()
    t.intern("a")
    s = _lockstep(t)
    s.enqueue_follow(_follow("a", "gone"))
    assert (s.pending_follow_count("a"), s.pending_follow_count("gone")) == (1, 1)
    s.pop_follow_head()
    assert (s.pending_follow_count("a"), s.pending_follow_count("gone")) == (0, 0)


# ── the worker's boundary ──────────────────────────────────────────────────


def _worker(
    *rids: str, tensor_manager=None, engine_manager=None, connections=(),
) -> tuple[Worker, RidTable]:
    """A worker whose graph runtime is only the rid table: the boundary is
    what these tests are about, and the runtime owns the table."""
    t = RidTable()
    sent = []
    runtime = SimpleNamespace()
    request_state = SimpleNamespace()
    tensor_manager = tensor_manager or SimpleNamespace()
    scheduler = make_scheduler(graph_runtime=runtime, request_state=request_state)
    w = make_worker(
        communicator=SimpleNamespace(send=lambda e, m: sent.append((e, m))),
        graph_runtime=runtime, request_state=request_state,
        tensor_manager=tensor_manager, engine_manager=engine_manager or SimpleNamespace(),
        streams=StreamBufferManager(request_state, tensor_manager, list(connections)),
        wakeup_event=SimpleNamespace(register_futures=lambda f: None),
        scheduler=scheduler, admission=scheduler.admission, lockstep=scheduler.lockstep,
        profile_info=SimpleNamespace(pop_request=lambda h: None),
    )
    w.sent = sent
    w.removed_in_runtime = []

    def _runtime_remove(h):
        # The handle must be freed only after every worker-side map dropped it.
        assert h not in w.request_state.per_request_info
        w.removed_in_runtime.append(h)
        t.release(h)

    # The runtime mints the handle now; interning is still the whole of it.
    runtime.add_request = lambda request_id, **kw: t.intern(request_id)
    runtime.register_stop_limits = lambda rid, info: None
    runtime.get_rid_handle, runtime.get_rid_string = t.handle, t.name
    runtime.get_sharding_config = lambda h: tp_sharding()
    runtime.remove_request = _runtime_remove
    request_state.per_request_info = {t.intern(r): SimpleNamespace() for r in rids}
    for r in rids:
        w.lifecycle.add(t.handle(r), r)
    request_state.remove_request = lambda h: request_state.per_request_info.pop(h)  # noqa: PLW0108
    return w, t


def test_failures_leave_the_worker_as_wire_strings():
    w, t = _worker("req-a", "req-b")
    w.pipeline._fail_requests({t.handle("req-a"): "boom"})
    [(entity, msg)] = w.sent
    assert entity == "conductor"
    assert msg.message_type == ConductorMessageType.FAIL_REQUESTS
    assert msg.body.errors == {"req-a": "boom"}
    # ...while the scheduler gate stays on the handle.
    assert w.lifecycle.failed_rids == {t.handle("req-a")}


def test_remove_purges_handle_keyed_state_then_frees_the_handle():
    w, t = _worker(
        "req-a", "req-b", engine_manager=SimpleNamespace(remove_request=lambda h: None),
        tensor_manager=SimpleNamespace(force_cleanup_request=lambda h: None),
    )
    a, b = t.handle("req-a"), t.handle("req-b")
    w._rebuilds._pool = HostPool(16, torch.device("cpu"))
    pieces = {rid: w._rebuilds._pool.take(8) for rid in (a, b)}
    w._rebuilds._kept, w._rebuilds._keeps_ids = {"n": {a: None, b: None}}, {"n": dict(pieces)}
    w._rebuilds._prefilling = {"n": {a, b}}
    w._rebuilds._writes = {a: {"n": ["kv"]}, b: {"n": ["kv"]}}
    w._rebuilds._cursors[("n", a)] = None
    w.preemption._preempted[a] = Preempted()
    w.streams._rids.add(a)
    w.scheduler.admission.held_until[a] = float("inf")

    deliver(w, WorkerMessage(
        WorkerMessageType.REMOVE_REQUEST, RemoveRequest(request_id="req-a", source=MessageSource.SELF),
    ))

    assert not w.streams.possible(), "the stream manager forgets the handle"

    assert w.removed_in_runtime == [a]
    assert b in w.request_state.per_request_info
    assert a not in w.scheduler.admission.held_until
    assert w._rebuilds._kept == {"n": {b: None}}, "the next request would rebuild from these ids"
    assert w._rebuilds._keeps_ids == {"n": {b: pieces[b]}}, "the next request would keep ids it can't be dropped with"
    assert w._rebuilds._prefilling == {"n": {b}}, "the next request's steps would be recorded as this one's"
    assert w._rebuilds._writes.keys() == {b}, "rank 0's refusal of the next request's log would find this one's"
    assert not (w._rebuilds._cursors or w.preemption._preempted), (
        "the next request would run this one's rebuild, or be counted as preempted"
    )
    # Freed last, and reused by the next request: nothing of req-a's may remain.
    assert t.handle("req-a") is None
    assert t.intern("req-c") == a


def test_a_teardown_that_raises_stops_the_worker_instead_of_failing_other_requests():
    def remove_request(h):
        raise RuntimeError("engine teardown failed")

    w, _ = _worker(
        "req-a", "req-b", engine_manager=SimpleNamespace(remove_request=remove_request),
        tensor_manager=SimpleNamespace(force_cleanup_request=lambda h: None),
    )

    # SystemExit passes the main loop's ``except Exception``, which would fail
    # the step in flight (req-b's) for a release that may have leaked pages.
    with pytest.raises(SystemExit, match="teardown of request req-a"):
        deliver(w, WorkerMessage(
            WorkerMessageType.REMOVE_REQUEST, RemoveRequest(request_id="req-a", source=MessageSource.SELF),
        ))
    assert not w.sent, "the worker stops; the conductor's dead-worker check fails what is left"


def test_remove_for_an_unknown_request_is_a_no_op():
    w, _ = _worker("req-a")
    w.ingest._remove_request(RemoveRequest(request_id="gone", source=MessageSource.SELF))
    w.pipeline._teardown(set())
    assert len(w.request_state.per_request_info) == 1


def test_tensor_ack_is_keyed_by_uuid_alone():
    """Uuids are global, so an ack -- even a late one for a request this worker
    already removed -- needs no rid lookup; an untracked uuid is a no-op in the
    store."""
    acked = []
    w, _ = _worker(tensor_manager=SimpleNamespace(
        dereference_batch=lambda uuids, counts: acked.append((uuids, counts)),
    ))
    w.ingest._handle_tensor_received(TensorReceived(
        successful_tensors={1: 2, 3: 1}, failed_tensor_ids=[],
    ))
    assert acked == [([1, 3], [2, 1])]


def test_add_new_request_hands_the_handle_to_every_subsystem():
    """The one place a handle is minted. Nothing below NEW_REQUEST sees the
    wire string, so a subsystem handed ``body.request_id`` instead would key
    its state under something no later message resolves to."""
    got: dict[str, object] = {}
    w, t = _worker(
        engine_manager=SimpleNamespace(
            engine=SimpleNamespace(publisher_nodes={}, skip_publish=lambda rid, keys: None),
            add_request=lambda rid, cfgs, info: got.__setitem__("engine", rid),
        ),
        tensor_manager=SimpleNamespace(
            register_request=lambda rid, sc: got.__setitem__("tensors", rid),
            start_read_tensors=lambda rid, inputs, graph_walk=None: [],
        ),
    )
    cfg = SimpleNamespace(sharding_config="cfg", stream_buffers={})

    def _state_add(request_id, request_info):
        got["state"] = request_id
        w.request_state.per_request_info[request_id] = cfg

    w.request_state.add_request = _state_add
    info = SimpleNamespace(
        rid_handle=-1, resource_configs={}, graph_walk="decode",
        partition_name="p",
    )
    body = SimpleNamespace(
        request_id="wire-1", request_info=info,
        partition_worker_graph_ids={}, worker_graph_to_workers={},
        initial_inputs=[], loop_caps={},
    )
    w.ingest._add_new_request(body)

    handle = t.handle("wire-1")
    assert handle is not None
    assert got == {"state": handle, "engine": handle, "tensors": handle}
    assert info.rid_handle == handle


def test_add_new_request_keeps_the_stream_buffer_and_primes_it_for_its_consumer():
    """Both partitions of a request land on one worker: the first NEW makes the
    stream buffer, the consumer's NEW primes it (and must not replace it)."""
    import collections

    import torch

    from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
    from mstar.streaming.topology import Connection

    w, _ = _worker(
        engine_manager=SimpleNamespace(
            engine=SimpleNamespace(publisher_nodes={}, skip_publish=lambda rid, keys: None),
            add_request=lambda rid, cfgs, info: None,
        ),
        tensor_manager=SimpleNamespace(
            register_request=lambda rid, sc: None, start_read_tensors=lambda rid, inputs, graph_walk=None: [],
        ),
        connections=[Connection(
            from_partition="Talker", to_partition="Codec", edge_name="codec_tokens",
            chunk_policy_factory=lambda: ScheduledLeftContextChunkPolicy(schedule=(1,), chunk=2, left_context=3),
        )],
    )
    req = SimpleNamespace(stream_buffers={}, stream_buffers_by_consumer={})
    w.request_state.add_request = lambda request_id, request_info: None
    w.request_state.per_request_info = collections.defaultdict(lambda: req)
    w.streams.consumer_node = {"codec_tokens": "Codec"}

    def new(partition, lead):
        info = SimpleNamespace(
            rid_handle=-1, resource_configs={}, graph_walk="w", partition_name=partition,
            stream_lead_items=lead,
        )
        body = SimpleNamespace(
            request_id="wire-1", request_info=info, partition_worker_graph_ids={},
            worker_graph_to_workers={}, initial_inputs=[], loop_caps={},
        )
        w.ingest._add_new_request(body)

    # the conductor stamps the lead on every partition's NEW; only the consumer primes
    new("Talker", {"codec_tokens": 2})
    sbuf = next(iter(req.stream_buffers.values()))
    sbuf.pre_read_register("t0")
    sbuf.put("t0", torch.tensor([0]))
    new("Codec", {"codec_tokens": 2})
    assert next(iter(req.stream_buffers.values())) is sbuf   # kept, item and all
    assert req.stream_buffers_by_consumer == {"Codec": {"codec_tokens": sbuf}}
    for i in (1, 2):
        sbuf.pre_read_register(f"t{i}")
        sbuf.put(f"t{i}", torch.tensor([i]))
    chunk = sbuf.pop_chunk()
    assert (chunk.context_items, chunk.num_items) == (2, 3)   # two lead items, then the first real one
