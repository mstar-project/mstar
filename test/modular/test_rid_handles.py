"""Worker-local rid handles: strings on the wire, recycled ints inside.

The failure mode these guard is silent: a handle that outlives its request
attaches to whichever request is given that handle next, and a string that
reaches a handle-keyed map just misses.
"""
from types import SimpleNamespace

import pytest

from mstar.utils.ipc_format import (
    ConductorMessageType,
    MessageSource,
    RemoveRequest,
    ScheduleTPNode,
    TensorReceived,
)
from mstar.worker.micro_scheduler import REMOVED_RID, MicroScheduler
from mstar.worker.rid_table import RidTable
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


def _scheduler(table: RidTable) -> MicroScheduler:
    s = MicroScheduler(engine_manager=None)
    s.rid_of = table.handle
    return s


def _follow(*rids: str) -> ScheduleTPNode:
    return ScheduleTPNode(node_name="n", graph_walk="w", request_ids=list(rids))


def test_tp_rids_resolve_to_this_ranks_handles_in_wire_order():
    t = RidTable()
    t.intern("pad")  # so this rank's handles differ from the leader's order
    b, a = t.intern("b"), t.intern("a")
    assert _scheduler(t).tp_rids(_follow("a", "b")) == [a, b]


def test_tp_rid_this_rank_does_not_know_is_the_removed_sentinel():
    t = RidTable()
    a = t.intern("a")
    assert _scheduler(t).tp_rids(_follow("a", "gone")) == [a, REMOVED_RID]


def test_follow_refcount_skips_unknown_rids():
    t = RidTable()
    a = t.intern("a")
    s = _scheduler(t)
    s.register_tp_follow(_follow("a", "gone"))
    assert dict(s.pending_tp_follow_count) == {a: 1}
    s.pop_tp_follow_head()
    assert dict(s.pending_tp_follow_count) == {}


# ── the worker's boundary ──────────────────────────────────────────────────


def _worker(*rids: str) -> tuple[Worker, RidTable]:
    """A worker whose graph runtime is only the rid table: the boundary is
    what these tests are about, and the runtime owns the table."""
    t = RidTable()
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.is_tp_follower = False
    w.sent = []
    w.communicator = SimpleNamespace(send=lambda e, m: w.sent.append((e, m)))
    w.removed_in_runtime = []

    def _runtime_remove(h):
        # The handle must be freed only after every worker-side map dropped it.
        assert h not in w.request_state.per_request_info
        w.removed_in_runtime.append(h)
        t.release(h)

    w._graph_runtime = SimpleNamespace(
        get_rid_handle=t.handle, get_rid_string=t.name,
        get_sharding_config=lambda h: SimpleNamespace(groups=[]),
        remove_request=_runtime_remove,
    )
    w.scheduler = MicroScheduler(engine_manager=None)
    w.scheduler.rid_of = t.handle
    handles = [t.intern(r) for r in rids]
    w.request_state = SimpleNamespace(
        per_request_info={h: SimpleNamespace() for h in handles},
        remove_request=lambda h: w.request_state.per_request_info.pop(h),  # noqa: PLW0108
    )
    return w, t


def test_failures_leave_the_worker_as_wire_strings():
    w, t = _worker("req-a", "req-b")
    w._fail_requests({t.handle("req-a"): "boom"})
    [(entity, msg)] = w.sent
    assert entity == "conductor"
    assert msg.message_type == ConductorMessageType.FAIL_REQUESTS
    assert msg.body.errors == {"req-a": "boom"}
    # ...while the scheduler gate stays on the handle.
    assert w.scheduler.failed_rids == {t.handle("req-a")}


def test_remove_purges_handle_keyed_state_then_frees_the_handle():
    w, t = _worker("req-a", "req-b")
    a, b = t.handle("req-a"), t.handle("req-b")
    w._in_flight_rids = set()
    w._pending_removes = set()
    w._pending_drains, w._draining_rids, w._reads_done_sent = set(), set(), set()
    w._last_active = {}
    w.streaming_buffers = {a: {}}
    w.engine_manager = SimpleNamespace(remove_request=lambda h: None, evictable_nodes=lambda: [])
    w.tensor_manager = SimpleNamespace(force_cleanup_request=lambda h: None)
    w.profile_info = SimpleNamespace(pop_request=lambda h: None)
    w.scheduler.held_until[a] = float("inf")

    w._remove_request(RemoveRequest(request_id="req-a", source=MessageSource.SELF))

    assert w.removed_in_runtime == [a]
    assert b in w.request_state.per_request_info
    assert a not in w.streaming_buffers
    assert a not in w.scheduler.held_until
    # Freed last, and reused by the next request: nothing of req-a's may remain.
    assert t.handle("req-a") is None
    assert t.intern("req-c") == a


def test_remove_for_an_unknown_request_is_a_no_op():
    w, _ = _worker("req-a")
    w._remove_request(RemoveRequest(request_id="gone", source=MessageSource.SELF))
    assert len(w.request_state.per_request_info) == 1


def test_tensor_ack_is_keyed_by_uuid_alone():
    """Uuids are global, so an ack -- even a late one for a request this worker
    already removed -- needs no rid lookup; an untracked uuid is a no-op in the
    store."""
    w, _ = _worker()
    acked = []
    w.tensor_manager = SimpleNamespace(
        dereference_batch=lambda uuids, counts: acked.append((uuids, counts)),
    )
    w._handle_tensor_received(TensorReceived(
        request_id="gone", successful_tensors={1: 2, 3: 1}, failed_tensor_ids=[],
    ))
    assert acked == [([1, 3], [2, 1])]
