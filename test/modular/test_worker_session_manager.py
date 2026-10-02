"""Worker-side session bookkeeping, on its own.

``WorkerSessionManager`` holds no engine or transport state, so the rules it
enforces can be read directly: which rids belong to a session, when a resumed
request is allowed in, which deferred removal still ends its session, and when a
held teardown is free to run. The one thing it asks the worker is whether a rid
is still on its way out, which these tests answer with a set.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from mstar.utils.ipc_format import NewRequest
from mstar.worker.sessions import WorkerSessionManager


def _manager(leaving=()):
    """A manager whose ``is_leaving`` answers from a set the test can mutate."""
    still_here = set(leaving)
    manager = WorkerSessionManager(is_leaving=still_here.__contains__)
    return manager, still_here


def _new_request(rid: str, session_id: str | None = None) -> NewRequest:
    return NewRequest(
        request_id=rid,
        partition_worker_graph_ids=[],
        worker_graph_to_workers={},
        initial_inputs=[],
        request_info=None,
        session_id=session_id,
    )


# ── what it holds ───────────────────────────────────────────────────────────

def test_binding_records_both_directions():
    manager, _ = _manager()

    manager.bind("r0", "s")

    assert manager.get_rids("s") == {"r0"}
    assert manager.session_of("r0") == "s"


def test_a_request_with_no_session_is_not_recorded():
    manager, _ = _manager()

    manager.bind("r0", None)

    assert manager.session_of("r0") is None
    assert manager.get_rids("s") == set()


def test_a_session_can_hold_more_than_one_rid():
    manager, _ = _manager()

    manager.bind("r0", "s")
    manager.bind("r1", "s")

    assert manager.get_rids("s") == {"r0", "r1"}


def test_nothing_is_pending_on_a_fresh_manager():
    manager, _ = _manager()

    assert manager.has_pending is False


# ── is the session still busy ───────────────────────────────────────────────

def test_a_session_is_busy_while_one_of_its_requests_is_leaving():
    manager, leaving = _manager(leaving={"r0"})
    manager.bind("r0", "s")

    assert manager.requests_still_leaving("s") is True

    leaving.discard("r0")

    assert manager.requests_still_leaving("s") is False


def test_a_session_nothing_is_known_about_is_not_busy():
    manager, _ = _manager(leaving={"r0"})

    assert manager.requests_still_leaving("other") is False


def test_a_request_does_not_read_its_own_ingest_as_one_leaving():
    # the conductor sends a NEW_REQUEST per partition, so a worker serving two
    # of them sees the same rid twice
    manager, _ = _manager(leaving={"r0"})
    manager.bind("r0", "s")

    assert manager.requests_still_leaving("s", except_rid="r0") is False


# ── ingest ──────────────────────────────────────────────────────────────────

def test_a_resumed_request_is_held_while_the_previous_one_leaves():
    manager, leaving = _manager(leaving={"r0"})
    manager.bind("r0", "s")

    assert manager.hold_if_not_ready(_new_request("r1", "s")) is True
    assert manager.has_pending is True
    # not bound yet: the worker has not ingested it
    assert manager.session_of("r1") is None

    leaving.discard("r0")

    held = manager.take_held_ingests()
    assert [body.request_id for body in held] == ["r1"]
    assert manager.has_pending is False


def test_a_request_with_no_session_is_never_held():
    manager, _ = _manager(leaving={"r0"})

    assert manager.hold_if_not_ready(_new_request("r1")) is False


def test_a_request_whose_session_is_free_is_never_held():
    manager, _ = _manager()
    manager.bind("r0", "s")

    assert manager.hold_if_not_ready(_new_request("r1", "s")) is False


def test_the_second_new_request_for_one_rid_is_not_held_by_the_first():
    # the worker hands over the handle it already minted for this rid
    manager, _ = _manager(leaving={"r0"})
    manager.bind("r0", "s")

    assert manager.hold_if_not_ready(_new_request("r0", "s"), "r0") is False
    # without it the rid would read as one of its own session's stragglers
    assert manager.hold_if_not_ready(_new_request("r0", "s")) is True


def test_a_held_request_that_is_still_not_ready_holds_itself_again():
    manager, leaving = _manager(leaving={"r0"})
    manager.bind("r0", "s")
    manager.hold_if_not_ready(_new_request("r1", "s"))

    for body in manager.take_held_ingests():
        assert manager.hold_if_not_ready(body) is True

    assert manager.has_pending is True
    leaving.discard("r0")
    assert [b.request_id for b in manager.take_held_ingests()] == ["r1"]


# ── removal ─────────────────────────────────────────────────────────────────

def test_release_hands_back_the_session_and_keeps_it():
    manager, _ = _manager()
    manager.bind("r0", "s")

    assert manager.release("r0") == "s"
    assert manager.session_of("r0") is None
    # the session lives on: it keeps what the request built
    assert manager.get_rids("s") == set()


def test_releasing_a_sessionless_request_says_so():
    manager, _ = _manager()

    assert manager.release("stranger") is None


def test_a_deferred_removal_remembers_that_it_ends_its_session():
    manager, _ = _manager()
    manager.bind("r0", "s")

    assert manager.ends_session("r0") is False

    manager.defer_end_session("r0")

    assert manager.ends_session("r0") is True


def test_releasing_clears_the_end_session_flag():
    manager, _ = _manager()
    manager.bind("r0", "s")
    manager.defer_end_session("r0")

    manager.release("r0")

    assert manager.ends_session("r0") is False


# ── teardown ────────────────────────────────────────────────────────────────

def test_a_held_teardown_is_ready_once_its_requests_have_left():
    manager, leaving = _manager(leaving={"r0"})
    manager.bind("r0", "s")

    manager.hold_teardown("s")

    assert manager.ready_teardowns() == []
    assert manager.has_pending is True

    leaving.discard("r0")

    assert manager.ready_teardowns() == ["s"]


def test_ready_teardowns_reports_each_free_session():
    manager, leaving = _manager(leaving={"r0"})
    manager.bind("r0", "busy")
    manager.hold_teardown("busy")
    manager.hold_teardown("free")

    assert manager.ready_teardowns() == ["free"]


def test_forgetting_a_session_drops_every_trace_of_it():
    manager, _ = _manager()
    manager.bind("r0", "s")
    manager.bind("r1", "s")
    manager.defer_end_session("r1")
    manager.hold_teardown("s")

    manager.forget_session("s")

    assert manager.get_rids("s") == set()
    assert manager.session_of("r0") is None
    assert manager.session_of("r1") is None
    assert manager.ends_session("r1") is False
    assert manager.ready_teardowns() == []
    assert manager.has_pending is False


def test_forgetting_a_session_leaves_the_others_alone():
    manager, _ = _manager()
    manager.bind("r0", "s")
    manager.bind("r1", "other")

    manager.forget_session("s")

    assert manager.session_of("r1") == "other"


def test_forgetting_a_session_that_was_never_opened_is_a_no_op():
    manager, _ = _manager()

    manager.forget_session("never-existed")

    assert manager.has_pending is False
