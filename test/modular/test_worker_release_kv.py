"""Worker side of RELEASE_KV: the pages of a finished request go back before
its REMOVE_REQUEST, which still follows and clears the rest.

RELEASE_KV is deferred behind an in-flight step exactly as a removal is, a TP
follower acts only on its leader's forward, and a request released early is
never scheduled, speculated on, published or given pages again. The KV tests
(``test_kv_release_at_completion``) cover the pool; these cover the worker's
part, partly against a real pool.
"""

from types import SimpleNamespace

import pytest
from test_kv_release_at_completion import (  # noqa: F401  (_isolated: the pool's autouse stubs)
    _all_free,
    _isolated,
    _manager,
    _pages,
    _ready,
    _running,
    _step,
)

from mstar.communication import wire, wire_types  # noqa: F401  (registers the message types)
from mstar.utils.ipc_format import (
    ConductorMessageType,
    DrainRequest,
    InputSignals,
    MessageSource,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.rid_table import RidTable
from mstar.worker.worker import Worker

LEADER_OF_X = SimpleNamespace(groups=[
    SimpleNamespace(tp_size=2, _tp_rank=0, _workers=["w0", "w1"]),
])


def _worker(
    *rids: str, is_follower=False, in_flight=(), tp_follow=(), sharding=None, kv=None,
):
    """A worker over the real rid table and scheduler, with a pool when ``kv`` is
    given (keyed by handle, as on the real worker) and recorders where it has none."""
    t = RidTable()
    handles = [t.intern(r) for r in rids]
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.is_tp_follower = is_follower
    w.sent, w.released, w.removed, w.forced = [], [], [], []
    w.communicator = SimpleNamespace(send=lambda e, msg: w.sent.append((e, msg)))
    w._graph_runtime = SimpleNamespace(
        get_rid_handle=t.handle, get_rid_string=t.name,
        get_sharding_config=lambda h: sharding,
        remove_request=t.release,
    )
    w.scheduler = MicroScheduler(engine_manager=None)
    w.scheduler.rid_of = t.handle
    for rid in tp_follow:
        w.scheduler.pending_tp_follow_count[rid] = 1
    w._in_flight_rids = {t.handle(r) for r in in_flight}
    w._pending_removes = set()
    w.scheduler.pending_removes = w._pending_removes
    w._pending_releases = set()
    w._pending_drains, w._draining_rids, w._reads_done_sent = set(), set(), set()
    w._last_active, w.streaming_buffers = {}, {}
    w.request_state = SimpleNamespace(
        per_request_info={h: SimpleNamespace() for h in handles},
        remove_request=lambda h: w.request_state.per_request_info.pop(h, None),
    )

    def release(h):
        w.released.append(h)
        if kv is not None:
            kv.release_kv(h)

    def remove(h):
        w.removed.append(h)
        if kv is not None:
            kv.remove_request(h)

    w.engine_manager = SimpleNamespace(
        release_kv=release, remove_request=remove, evictable_nodes=lambda: [],
    )
    w.tensor_manager = SimpleNamespace(
        has_inflight_reads=lambda h: False,
        force_cleanup_request=w.forced.append,
    )
    w.profile_info = SimpleNamespace(pop_request=lambda h: None)
    return w, t


def _release(w, rid="X", source=MessageSource.CONDUCTOR):
    Worker._release_kv(w, RemoveRequest(request_id=rid, source=source))


def _forwards(w):
    return [
        (e, m) for e, m in w.sent
        if m.message_type == WorkerMessageType.RELEASE_KV
    ]


# ── applied, and deferred ───────────────────────────────────────────────


def test_a_release_for_a_request_not_in_flight_gives_its_pages_back_at_once():
    w, t = _worker("X")

    _release(w)

    assert w.released == [t.handle("X")]
    assert not w._pending_releases
    # the request's other state is the removal's
    assert w.removed == [] and w.forced == [] and t.handle("X") is not None


def test_a_release_stops_new_work_for_the_request_but_keeps_it_until_the_removal():
    w, t = _worker("X", "Y")
    x = t.handle("X")

    _release(w)

    assert w.scheduler.failed_rids == {x}, "the scheduler would still start work for a finished request"
    assert w._is_tearing_down(x) and not w._is_tearing_down(t.handle("Y"))
    batch = SimpleNamespace(
        request_ids=[x, t.handle("Y")],
        per_request_info={
            x: SimpleNamespace(request_id="X"), t.handle("Y"): SimpleNamespace(request_id="Y"),
        },
    )
    assert w._publishable_request_ids(batch) == [t.handle("Y")]


def test_a_release_for_a_request_in_flight_waits_until_the_step_lets_go():
    w, t = _worker("X", in_flight=("X",))
    x = t.handle("X")

    _release(w)

    assert w.released == [] and w._pending_releases == {x}
    # a step in flight is the last: nothing speculates on, publishes or schedules the rid
    assert w._is_tearing_down(x)

    w._apply_pending_removes_safe_to_drop({x})  # still in the step
    assert w.released == []

    w._in_flight_rids.clear()  # the step ends
    w._apply_pending_removes_safe_to_drop(set())
    assert w.released == [x] and not w._pending_releases


def test_a_release_waits_for_a_tp_follow_batch_queued_for_the_request():
    w, t = _worker("X", tp_follow=("X",))

    _release(w)
    assert w.released == [] and w._pending_releases == {t.handle("X")}

    w.scheduler.pending_tp_follow_count.pop("X")  # the batch ran
    w._apply_pending_removes_safe_to_drop(set())
    assert w.released == [t.handle("X")]


def test_a_release_for_a_request_this_worker_does_not_know_does_nothing():
    w, _ = _worker()

    _release(w, "gone")

    assert w.released == [] and not w._pending_releases and not w.scheduler.failed_rids


# ── the removal after it ────────────────────────────────────────────────


def test_the_removal_after_a_release_still_cleans_the_rest():
    w, t = _worker("X")
    x = t.handle("X")
    _release(w)

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.removed == [x] and w.forced == [x]
    assert w.scheduler.failed_rids == set(), "clear_rid takes the rid out of the scheduler's gate"
    assert t.handle("X") is None


def test_a_removal_that_beats_a_deferred_release_leaves_no_release_for_a_reused_handle():
    w, t = _worker("X", in_flight=("X",))
    x = t.handle("X")
    _release(w)
    w._in_flight_rids.clear()  # the step ended before the removal was handled

    Worker._remove_request(w, RemoveRequest(request_id="X"))
    w._apply_pending_removes_safe_to_drop(set())

    assert not w._pending_releases and w.released == []
    assert t.intern("Z") == x, "the handle is reused"
    _release(w, "Z")
    assert w.released == [x]


def test_a_release_and_a_removal_both_deferred_apply_in_that_order():
    w, t = _worker("X", in_flight=("X",))
    x = t.handle("X")
    order = []
    w.engine_manager.release_kv = lambda h: order.append(("release", h))
    w.engine_manager.remove_request = lambda h: order.append(("remove", h))
    _release(w)
    Worker._remove_request(w, RemoveRequest(request_id="X"))
    assert order == [] and w._pending_releases == {x} and w._pending_removes == {x}

    w._in_flight_rids.clear()
    w._apply_pending_removes_safe_to_drop(set())

    assert order == [("release", x), ("remove", x)]


# ── TP ──────────────────────────────────────────────────────────────────


def test_a_follower_ignores_the_conductors_release_and_honors_the_leaders():
    w, _ = _worker("X", is_follower=True)

    _release(w)  # source=CONDUCTOR
    assert w.released == [] and not w.scheduler.failed_rids

    _release(w, source=MessageSource.TP_RANK_0)
    assert len(w.released) == 1


def test_a_follower_defers_the_leaders_release_behind_its_own_step():
    w, t = _worker("X", is_follower=True, in_flight=("X",))
    x = t.handle("X")

    _release(w, source=MessageSource.TP_RANK_0)
    assert w.released == [] and w._pending_releases == {x}

    w._in_flight_rids.clear()
    w._apply_pending_removes_safe_to_drop(set())  # re-enters as SELF
    assert w.released == [x]


def test_the_leader_forwards_the_release_to_its_followers_when_it_applies_it():
    w, t = _worker("X", sharding=LEADER_OF_X, in_flight=("X",))

    _release(w)
    assert _forwards(w) == [], "followers were told to free pages the leader's step still reads"

    w._in_flight_rids.clear()
    w._apply_pending_removes_safe_to_drop(set())

    [(entity, msg)] = _forwards(w)
    assert entity == "w1"
    assert msg.body == RemoveRequest(request_id="X", source=MessageSource.TP_RANK_0)
    assert w.released == [t.handle("X")]


def test_a_worker_with_no_followers_forwards_nothing():
    w, _ = _worker("X", sharding=SimpleNamespace(groups=[
        SimpleNamespace(tp_size=1, _tp_rank=0, _workers=["w0"]),
    ]))

    _release(w)

    assert _forwards(w) == [] and len(w.released) == 1


# ── a release, an abort and a removal, in any order ─────────────────────


def _running_worker(**kw):
    """A worker whose request ``X`` (handle 0) has pages in a real pool."""
    kv = _manager(max_num_pages=16)
    w, t = _worker("X", kv=kv, **kw)
    _running(kv, t.handle("X"), prompt=40, max_tokens=20)
    assert _pages(kv, t.handle("X"))
    return w, t, kv


def _reads_done(w):
    return [
        m for e, m in w.sent
        if e == "conductor" and m.message_type == ConductorMessageType.READS_DONE
    ]


@pytest.mark.parametrize("order", [("drain", "release"), ("release", "drain")])
def test_an_abort_and_a_release_in_either_order_free_every_page_once(order):
    w, t, kv = _running_worker()
    x = t.handle("X")

    for op in order:
        if op == "drain":
            Worker._drain_request(w, DrainRequest(request_id="X"))
        else:
            _release(w)
        kv.assert_pages_conserved()
    assert _pages(kv, x) == [] and _all_free(kv)
    assert len(_reads_done(w)) == 1

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    kv.assert_pages_conserved()
    assert _all_free(kv) and x not in kv._streams
    assert w.forced == [x]


def test_an_abort_a_release_and_a_removal_all_deferred_free_every_page_once():
    w, t, kv = _running_worker(in_flight=("X",))
    x = t.handle("X")
    Worker._drain_request(w, DrainRequest(request_id="X"))
    _release(w)
    Worker._remove_request(w, RemoveRequest(request_id="X"))
    assert _pages(kv, x), "the pages were freed under the step in flight"

    w._in_flight_rids.clear()
    w._apply_pending_removes_safe_to_drop(set())
    w._apply_pending_drains(set())

    kv.assert_pages_conserved()
    assert _all_free(kv) and x not in kv._streams and w.forced == [x]


def test_an_abort_alone_still_frees_everything_at_the_removal():
    w, t, kv = _running_worker()

    Worker._drain_request(w, DrainRequest(request_id="X"))
    assert _pages(kv, t.handle("X")), "a drain frees nothing: it waits for the removal"
    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert _all_free(kv)


def test_a_released_request_is_refused_by_the_pool_it_gave_its_pages_to():
    """What stands between a late step and the pages it no longer has."""
    w, t, kv = _running_worker()
    x = t.handle("X")
    _release(w)

    assert not _ready(kv, x).ok
    assert not _step(kv, x, 1).ok
    assert _pages(kv, x) == [] and x not in kv._reserved


# ── the message ─────────────────────────────────────────────────────────


def _stub_for_dispatch(active=("X",)):
    handles = {rid: i for i, rid in enumerate(active)}
    stub = SimpleNamespace(
        request_state=SimpleNamespace(per_request_info={h: object() for h in handles.values()}),
        _unprocessed_messages={}, _rid=handles.get, delivered=[], released=[],
    )
    stub._release_kv = stub.released.append
    stub._remove_request = lambda body: None
    stub._process_new_inputs = stub.delivered.append
    return stub


def _release_message(rid, source=MessageSource.CONDUCTOR):
    return WorkerMessage(
        message_type=WorkerMessageType.RELEASE_KV,
        body=RemoveRequest(request_id=rid, source=source),
    )


def test_a_release_message_reaches_the_handler():
    stub = _stub_for_dispatch()
    message = _release_message("X")

    Worker._process_message_list(stub, [message])

    assert stub.released == [message.body]
    assert stub._unprocessed_messages == {}


def test_a_release_that_beats_its_new_request_is_kept_for_it():
    stub = _stub_for_dispatch(active=())
    message = _release_message("Y")

    Worker._process_message_list(stub, [message])

    assert stub.released == [] and stub._unprocessed_messages == {"Y": [message]}


def test_a_release_replayed_with_its_request_does_not_loop():
    stub = _stub_for_dispatch()
    buffered = [
        _release_message("X", MessageSource.TP_RANK_0),
        WorkerMessage(
            message_type=WorkerMessageType.INPUT_SIGNALS,
            body=InputSignals(request_id="X", inputs=[], request_info=None),
        ),
    ]

    Worker._process_message_list(stub, buffered)

    assert len(stub.released) == 1 and len(stub.delivered) == 1


def test_a_release_message_crosses_the_wire_with_the_body_a_removal_has():
    message = _release_message("X", MessageSource.TP_RANK_0)

    back = wire.decode(wire.encode(message))

    assert back.message_type is WorkerMessageType.RELEASE_KV
    assert back.body == RemoveRequest(request_id="X", source=MessageSource.TP_RANK_0)
