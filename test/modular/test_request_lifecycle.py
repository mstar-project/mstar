"""The order ``RequestLifecycle`` hands teardowns back in, which keeps a TP
follower's page state in step with rank 0's."""
from mstar.utils.ipc_format import DrainRequest, MessageSource, RemoveRequest
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout


class _Runtime:
    """The graph runtime's handle table, as `_add` mints it; a handle stays
    named until a later `_add` reuses it, as the runtime frees it only at the
    teardown's end."""

    def __init__(self):
        self.names: dict[int, str] = {}

    def get_rid_string(self, handle):
        return self.names.get(handle)

    def get_rid_handle(self, wire_rid):
        return next((h for h, w in self.names.items() if w == wire_rid), None)


def _add(lifecycle: RequestLifecycle, rid: int, wire_rid: str) -> None:
    lifecycle._runtime.names[rid] = wire_rid
    lifecycle.add(rid, wire_rid)


def _lifecycle(*rids, follower=False) -> RequestLifecycle:
    lifecycle = RequestLifecycle(layout=WorkerLayout(is_tp_follower=follower), graph_runtime=_Runtime())
    for rid in rids:
        _add(lifecycle, rid, f"r{rid}")
    return lifecycle


def _rank0_remove(rid, seq=-1) -> RemoveRequest:
    return RemoveRequest(request_id=f"r{rid}", source=MessageSource.TP_RANK_0, after_tp_seq=seq)


def test_a_leader_tears_down_whatever_no_step_holds():
    lifecycle = _lifecycle(1, 2, 3)
    for rid in (1, 2, 3):
        lifecycle.remove(rid, RemoveRequest(request_id=f"r{rid}", source=MessageSource.CONDUCTOR), -1)
    assert lifecycle.removes_ready({2}, -1) == [1, 3]
    lifecycle.removed(1, "r1")
    lifecycle.removed(3, "r3")
    assert list(lifecycle.pending_removes) == [2]
    assert lifecycle.removes_ready(set(), -1) == [2]


def test_a_follower_never_tears_down_ahead_of_one_it_deferred():
    lifecycle = _lifecycle(1, 2, follower=True)
    lifecycle.remove(1, _rank0_remove(1), -1)
    lifecycle.remove(2, _rank0_remove(2), -1)
    assert lifecycle.removes_ready({1}, -1) == []
    assert lifecycle.removes_ready(set(), -1) == [1, 2]


def test_a_follower_ignores_the_conductors_remove():
    lifecycle = _lifecycle(1, follower=True)
    lifecycle.remove(1, RemoveRequest(request_id="r1", source=MessageSource.CONDUCTOR), 10)
    assert lifecycle.removes_ready(set(), 10) == []


def test_a_stamped_remove_waits_for_its_step_and_keeps_rank_0s_order():
    lifecycle = _lifecycle(1, 2, 3, follower=True)
    lifecycle.remove(2, _rank0_remove(2, seq=6), 4)
    lifecycle.remove(1, _rank0_remove(1, seq=5), 4)
    assert lifecycle.removes_ready(set(), 4) == []
    assert lifecycle.removes_ready(set(), 5) == [1]
    lifecycle.removed(1, "r1")
    # a landed stamp goes behind what was already waiting on a step in flight
    lifecycle.remove(3, _rank0_remove(3), 5)
    assert lifecycle.removes_ready({3}, 6) == []
    assert lifecycle.removes_ready(set(), 6) == [3, 2]


def test_a_remove_stays_pending_until_its_teardown_finishes():
    """A teardown that raises must not lose the removes after it."""
    lifecycle = _lifecycle(1, 2)
    for rid in (1, 2):
        lifecycle.remove(rid, RemoveRequest(request_id=f"r{rid}", source=MessageSource.CONDUCTOR), 10)
    assert lifecycle.removes_ready(set(), -1) == [1, 2]
    lifecycle.removed(1, "r1")  # 2's teardown never ran
    assert lifecycle.removes_ready(set(), -1) == [2]


def test_a_drain_stays_pending_until_it_begins():
    lifecycle = _lifecycle(1, 2)
    lifecycle.drain(DrainRequest(request_id="r1", source=MessageSource.CONDUCTOR))
    lifecycle.drain(DrainRequest(request_id="r2", source=MessageSource.CONDUCTOR))
    lifecycle.begin_drain("r1")
    assert lifecycle.drains_ready(set()) == ["r2"]


def test_a_stamped_remove_is_purged_with_its_handle():
    """A second stamp for a rid already torn down must not land on whichever
    request gets the handle next."""
    lifecycle = _lifecycle(7, follower=True)
    lifecycle.remove(7, _rank0_remove(7, seq=9), 4)
    lifecycle.removed(7, "r7")
    _add(lifecycle, 7, "next")
    assert lifecycle.removes_ready(set(), 9) == []
    assert lifecycle.arrivals.keys() == {7}


def test_removed_forgets_a_handle_that_may_be_recycled():
    lifecycle = _lifecycle(1)
    lifecycle.fail({1: "boom"})
    lifecycle.remove(1, RemoveRequest(request_id="r1", source=MessageSource.CONDUCTOR), 10)
    lifecycle.removed(1, "r1")
    assert 1 not in lifecycle.failed_rids
    assert 1 not in lifecycle.pending_removes
    assert 1 not in lifecycle.arrivals
    assert lifecycle.take_failures() == {}
    assert not lifecycle.teardowns_pending()


def test_a_failure_keeps_its_first_status_until_reported_or_removed():
    """A failure that isn't a 500 (the request's own fault) carries its status to
    the conductor; the handle's next request starts without it."""
    lifecycle = _lifecycle(1, 2)
    lifecycle.fail({1: "bad grammar"}, status=400)
    lifecycle.fail({1: "Error in worker: internal error"})
    lifecycle.fail({2: "boom"})
    errors = lifecycle.take_failures()
    assert errors == {1: "bad grammar", 2: "boom"}
    assert lifecycle.take_statuses(errors) == {1: 400}
    lifecycle.fail({2: "bad grammar"}, status=400)
    lifecycle.removed(2, "r2")
    assert lifecycle.take_statuses([2]) == {}


def test_a_failure_for_a_request_already_gone_is_dropped():
    lifecycle = _lifecycle()
    lifecycle.fail({7: "late"})
    assert lifecycle.failed_rids == set()
    assert lifecycle.take_failures() == {}


def test_a_drain_defers_behind_a_step_and_a_later_arrival_takes_no_work():
    lifecycle = _lifecycle(1)
    lifecycle.drain(DrainRequest(request_id="r1", source=MessageSource.CONDUCTOR))
    lifecycle.drain(DrainRequest(request_id="r9", source=MessageSource.CONDUCTOR))
    assert lifecycle.drains_ready({1}) == ["r9"]
    assert lifecycle.begin_drain("r9") is None
    # NEW_REQUEST for an id already draining
    _add(lifecycle, 9, "r9")
    assert 9 in lifecycle.failed_rids
    assert lifecycle.is_tearing_down(1)
    assert lifecycle.drains_ready(set()) == ["r1"]
    assert lifecycle.begin_drain("r1") == 1
    assert dict(lifecycle.open_drains()) == {"r1": 1, "r9": 9}
    lifecycle.reads_done("r1")
    assert dict(lifecycle.open_drains()) == {"r9": 9}
    assert not lifecycle.takes_input("r1")


def test_parked_messages_come_back_once_and_a_removed_id_refuses_them():
    class _Msg:
        def __init__(self, rid):
            self.body = DrainRequest(request_id=rid)

    lifecycle = _lifecycle()
    message = _Msg("r1")
    assert lifecycle.park(message)
    _add(lifecycle, 1, "r1")
    assert lifecycle.take_parked("r1") == [message]
    assert lifecycle.take_parked("r1") == []

    assert lifecycle.park(_Msg("r2"))
    assert len(list(lifecycle.removed(None, "r2"))) == 1
    assert not lifecycle.park(_Msg("r2"))


def test_parked_messages_stay_parked_until_the_replay_takes_them():
    """A setup that raises between the arrival and the replay leaves them for
    the request's teardown to drop and ACK."""
    lifecycle = _lifecycle()
    message = type("_Msg", (), {"body": DrainRequest(request_id="r1")})()
    assert lifecycle.park(message)
    _add(lifecycle, 1, "r1")

    assert list(lifecycle.removed(1, "r1")) == [message], "a parked message would never be ACKed"


def test_a_rows_first_error_is_the_one_reported():
    lifecycle = _lifecycle(1)
    lifecycle.fail({1: "Error in worker: request too long"})
    lifecycle.fail({1: "Error in worker: internal error"})

    assert lifecycle.take_failures() == {1: "Error in worker: request too long"}, (
        "the client would get the step's generic error instead of its own"
    )


def test_a_teardown_that_raised_keeps_its_request_out_and_is_not_retried():
    lifecycle = _lifecycle(1)
    lifecycle.remove(1, RemoveRequest(request_id="r1", source=MessageSource.CONDUCTOR), -1)
    assert lifecycle.removes_ready(set(), -1) == [1]
    lifecycle.removed(1, "r1")

    lifecycle.teardown_failed(1)

    assert 1 in lifecycle.failed_rids, "a half torn-down request would be scheduled again"
    assert lifecycle.removes_ready(set(), -1) == [], "the teardown would raise again every pass"
