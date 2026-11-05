"""The order ``RequestLifecycle`` hands teardowns back in, which keeps a TP
follower's page state in step with rank 0's."""
from mstar.utils.ipc_format import DrainRequest, MessageSource, RemoveRequest
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout


def _lifecycle(*rids, follower=False) -> RequestLifecycle:
    lifecycle = RequestLifecycle(WorkerLayout(is_tp_follower=follower))
    for rid in rids:
        lifecycle.add(rid, f"r{rid}")
    return lifecycle


def _rank0_remove(rid, seq=-1) -> RemoveRequest:
    return RemoveRequest(request_id=f"r{rid}", source=MessageSource.TP_RANK_0, after_tp_seq=seq)


def test_a_leader_tears_down_whatever_no_step_holds():
    lifecycle = _lifecycle(1, 2, 3)
    for rid in (1, 2, 3):
        lifecycle.remove(rid, RemoveRequest(request_id=f"r{rid}", source=MessageSource.CONDUCTOR))
    assert lifecycle.take_removes({2}, -1) == [1, 3]
    assert list(lifecycle.pending_removes) == [2]
    assert lifecycle.take_removes(set(), -1) == [2]


def test_a_follower_never_tears_down_ahead_of_one_it_deferred():
    lifecycle = _lifecycle(1, 2, follower=True)
    lifecycle.remove(1, _rank0_remove(1))
    lifecycle.remove(2, _rank0_remove(2))
    assert lifecycle.take_removes({1}, -1) == []
    assert lifecycle.take_removes(set(), -1) == [1, 2]


def test_a_follower_ignores_the_conductors_remove():
    lifecycle = _lifecycle(1, follower=True)
    lifecycle.remove(1, RemoveRequest(request_id="r1", source=MessageSource.CONDUCTOR))
    assert lifecycle.take_removes(set(), 10) == []


def test_a_stamped_remove_waits_for_its_step_and_keeps_rank_0s_order():
    lifecycle = _lifecycle(1, 2, 3, follower=True)
    lifecycle.remove(2, _rank0_remove(2, seq=6))
    lifecycle.remove(1, _rank0_remove(1, seq=5))
    assert lifecycle.take_removes(set(), 4) == []
    assert lifecycle.take_removes(set(), 5) == [1]
    # a landed stamp goes behind what was already waiting on a step in flight
    lifecycle.remove(3, _rank0_remove(3))
    assert lifecycle.take_removes({3}, 6) == []
    assert lifecycle.take_removes(set(), 6) == [3, 2]


def test_removed_forgets_a_handle_that_may_be_recycled():
    lifecycle = _lifecycle(1)
    lifecycle.fail({1: "boom"})
    lifecycle.remove(1, RemoveRequest(request_id="r1", source=MessageSource.CONDUCTOR))
    lifecycle.removed(1, "r1")
    assert 1 not in lifecycle.failed_rids
    assert 1 not in lifecycle.pending_removes
    assert 1 not in lifecycle.arrivals
    assert lifecycle.take_failures() == {}
    assert not lifecycle.teardowns_pending()


def test_a_failure_for_a_request_already_gone_is_dropped():
    lifecycle = _lifecycle()
    lifecycle.fail({7: "late"})
    assert lifecycle.failed_rids == set()
    assert lifecycle.take_failures() == {}


def test_a_drain_defers_behind_a_step_and_a_later_arrival_takes_no_work():
    lifecycle = _lifecycle(1)
    lifecycle.drain(DrainRequest(request_id="r1", source=MessageSource.CONDUCTOR))
    lifecycle.drain(DrainRequest(request_id="r9", source=MessageSource.CONDUCTOR))
    assert lifecycle.take_drains({1}) == ["r9"]
    assert lifecycle.begin_drain("r9") is None
    # NEW_REQUEST for an id already draining
    lifecycle.add(9, "r9")
    assert 9 in lifecycle.failed_rids
    assert lifecycle.is_tearing_down(1)
    assert lifecycle.take_drains(set()) == ["r1"]
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
    assert lifecycle.add(1, "r1") == [message]
    assert list(lifecycle.add(1, "r1")) == []

    assert lifecycle.park(_Msg("r2"))
    assert len(list(lifecycle.removed(None, "r2"))) == 1
    assert not lifecycle.park(_Msg("r2"))
