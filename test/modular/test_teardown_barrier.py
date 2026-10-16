"""Teardown barrier (conductor side): the hard RemoveRequest that unlinks a
segment is sent only after every reader has drained (READS_DONE).

Covers the happy path (defer Remove until the preprocess worker finished
delivering outputs, and unpersist still-held signals with correct counts) and
the abort/fail path (drain all workers + preprocess worker, then Remove).
"""

import types
from collections import deque

import pytest
import torch

from mstar.conductor.conductor import Conductor, RequestData, _req_id_to_seed, _request_seed
from mstar.graph.base import TensorPointerInfo
from mstar.utils.ipc_format import (
    FailRequests,
    NewRequestConductor,
    ReadsDone,
    UnpersistTensors,
    WorkerMessageType,
)

PREPROCESS = "api_server_preprocess_worker"


def _request_data(workers=("w0",), persist_signals=None, ref_cnts=None):
    return RequestData(
        persist_signals=persist_signals or {},
        persist_signal_ref_cnt=ref_cnts or {},
        worker_graph_to_workers={0: list(workers)},
        all_worker_graph_ids={0},
        max_output_tokens=1,
        random_seed=0,
        # Post-resource-pool refactor: requests carry per-resource configs
        # rather than a sampling_config. The barrier never reads them.
        resource_configs={},
    )


def _new_request(rid, model_kwargs=None):
    return NewRequestConductor(
        request_id=rid,
        initial_signals={},
        initial_input_modalities=["text"],
        initial_output_modalities=["text"],
        input_metadata={},
        model_kwargs=model_kwargs or {},
    )


def _conductor(requests, drain_ttl_s=120.0):
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.admits = 0
    c.communicator = types.SimpleNamespace(
        send=lambda entity_id, msg: c.sent.append((entity_id, msg))
    )
    c.requests = dict(requests)
    c.draining = {}
    c._drain_ttl_s = drain_ttl_s
    c._early_reads_done = {}
    c._early_abort_requests = set()
    c._draining_deadlines = deque()
    c._early_reads_done_deadlines = deque()
    c._early_abort_deadlines = deque()
    c._torn_down = {}
    c._torn_down_deadlines = deque()
    c._unsent_teardown = []
    c.waiting_queue = []
    c.enable_prof = False
    c.enable_nvtx = False
    c.max_concurrent_requests = None
    c._try_admit_waiting = lambda: setattr(c, "admits", c.admits + 1)
    return c


def _by_type(c, message_type):
    return [
        (e, m) for e, m in c.sent
        if getattr(m, "message_type", None) == message_type
    ]


def _drain_targets(c, rid):
    return {
        e for e, m in _by_type(c, WorkerMessageType.DRAIN_REQUEST)
        if m.body.request_id == rid
    }


def _remove_targets(c, rid):
    return {
        e for e, m in _by_type(c, WorkerMessageType.REMOVE_REQUEST)
        if m.body.request_id == rid
    }


# ── abort / fail path ───────────────────────────────────────────────────────

def test_abort_drains_all_participants_then_removes():
    c = _conductor({"r1": _request_data(workers=("w0", "w1"))})
    c._abort_request("r1")

    # Phase 1: drain every worker + the preprocess worker; nothing removed yet.
    assert _drain_targets(c, "r1") == {"w0", "w1", PREPROCESS}
    assert _remove_targets(c, "r1") == set()
    assert "r1" in c.requests and "r1" in c.draining

    # Partial ACKs do not finalize.
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id="w0"))
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id="w1"))
    assert _remove_targets(c, "r1") == set()
    assert "r1" in c.requests

    # Final ACK (preprocess worker) completes the barrier: hard-remove + admit.
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    assert _remove_targets(c, "r1") == {"w0", "w1", PREPROCESS}
    assert "r1" not in c.requests and "r1" not in c.draining
    assert c.admits == 1


def test_abort_of_unknown_request_sends_nothing():
    c = _conductor({})
    c._abort_request("nope")
    assert c.sent == []


def test_abort_of_queued_request_removes_preprocess_state():
    """A request aborted out of the waiting queue never reached a worker, so the
    preprocess worker is the only holder of its persisted input signals."""
    c = _conductor({})
    c.waiting_queue = [_new_request("r1")]
    c._abort_request("r1")

    assert c.waiting_queue == []
    assert _remove_targets(c, "r1") == {PREPROCESS}


def test_second_abort_while_draining_is_ignored():
    c = _conductor({"r1": _request_data()})
    c._abort_request("r1")
    c.sent.clear()
    c._abort_request("r1")  # already draining
    assert c.sent == []


def test_fail_defers_client_notification_until_barrier_completes():
    c = _conductor({"r1": _request_data()})
    c._fail_requests(FailRequests(errors={"r1": "boom"}))

    # Draining, but the client is not told yet.
    assert _drain_targets(c, "r1") == {"w0", PREPROCESS}
    assert not [m for e, m in c.sent if e == "api_server"]

    c.sent.clear()
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id="w0"))
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    failures = [
        m for e, m in c.sent
        if e == "api_server" and m.message_type == "request_failed"
    ]
    assert len(failures) == 1
    assert failures[0].body.error_message == "boom"
    assert _remove_targets(c, "r1") == {"w0", PREPROCESS}


# ── happy path ──────────────────────────────────────────────────────────────

def test_done_completes_client_immediately_but_defers_remove():
    c = _conductor({"r1": _request_data(workers=("w0", "w1"))})
    c._process_request_done("r1")

    # Client told right away; workers assumed done so only the preprocess worker
    # must ACK — no Remove yet.
    completes = [m for e, m in c.sent if e == "api_server"
                 and m.message_type == "request_complete"]
    assert len(completes) == 1
    assert _remove_targets(c, "r1") == set()
    assert c.draining["r1"].expected_acks == {PREPROCESS}
    assert "r1" in c.requests

    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    assert _remove_targets(c, "r1") == {"w0", "w1", PREPROCESS}
    assert "r1" not in c.requests
    assert c.admits == 1


def test_done_unpersists_still_held_signals_with_ref_counts():
    info = TensorPointerInfo(
        dims=(4,), dtype=torch.float32, stride=(1,), nbytes=16, address=0,
        uuid="u1", source_session_id="s", source_entity="w0",
    )
    rd = _request_data(
        persist_signals={"emb": [info]}, ref_cnts={"u1": 3},
    )
    c = _conductor({"r1": rd})
    c._process_request_done("r1")

    unpersists = [
        (e, m) for e, m in c.sent
        if getattr(m, "message_type", None) == WorkerMessageType.UNPERSIST_TENSORS
    ]
    assert len(unpersists) == 1
    entity, msg = unpersists[0]
    assert entity == "w0"
    assert isinstance(msg.body, UnpersistTensors)
    assert msg.body.uuid_to_ref_count == {"u1": 3}


# ── ordering race: ACK before registration ──────────────────────────────────

def test_reads_done_racing_registration_is_not_lost():
    """The preprocess worker self-drains on abort and can ACK before the
    conductor processes ABORT_REQUEST. That early ACK must still count."""
    c = _conductor({"r1": _request_data()})
    # Preprocess worker's READS_DONE lands before the barrier is registered.
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    assert "r1" in c._early_reads_done

    c._abort_request("r1")
    # Only w0 was still outstanding; deliver it and the barrier finalizes.
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id="w0"))
    assert _remove_targets(c, "r1") == {"w0", PREPROCESS}
    assert "r1" not in c.requests


# ── ordering race: abort before ingest ──────────────────────────────────────

def test_abort_racing_ingest_drops_the_request_and_removes_signals():
    """The preprocess worker forwards ABORT_REQUEST before it finishes
    preprocessing, so the abort can outrun the NEW_REQUEST it aborts. The
    request must never be dispatched, and its input signals must be removed —
    but only once ingest proves they exist."""
    c = _conductor({})
    c._abort_request("r1")
    assert "r1" in c._early_abort_requests
    # Nothing sent yet: a RemoveRequest here would beat the signals into being.
    assert c.sent == []

    c._ingest_request(_new_request("r1"))
    assert "r1" not in c.requests and c.waiting_queue == []
    assert _remove_targets(c, "r1") == {PREPROCESS}
    assert "r1" not in c._early_abort_requests


def test_abort_racing_ingest_discards_its_early_reads_done():
    c = _conductor({})
    c._abort_request("r1")
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    c._ingest_request(_new_request("r1"))
    assert "r1" not in c._early_reads_done


# ── late messages for a torn-down request ───────────────────────────────────

def _complete_and_tear_down(c, rid):
    c._process_request_done(rid)
    c._handle_reads_done(ReadsDone(request_id=rid, entity_id=PREPROCESS))
    assert rid not in c.requests


def _admitting(c):
    admitted = []
    c._admit_request = lambda body: admitted.append(body.request_id)
    return admitted


def test_a_late_abort_does_not_drop_a_rerun_of_the_id():
    """The API server sends an abort when a client disconnects as its answer
    arrives, after the conductor already tore the request down. That abort
    must not tombstone the id: a client re-running it would hang."""
    c = _conductor({"r1": _request_data()})
    admitted = _admitting(c)
    _complete_and_tear_down(c, "r1")

    c._abort_request("r1")
    c._ingest_request(_new_request("r1"))
    assert admitted == ["r1"]
    assert "r1" not in c._early_abort_requests


def test_a_late_reads_done_does_not_pre_ack_a_rerun():
    """After a failure's barrier, the API server's own cleanup still sends a
    READS_DONE. Kept as an early ACK, it would let a re-run of the id finish
    its teardown without waiting for the preprocess worker's reads."""
    c = _conductor({"r1": _request_data()})
    c._fail_requests(FailRequests(errors={"r1": "boom"}))
    for entity in ("w0", PREPROCESS):
        c._handle_reads_done(ReadsDone(request_id="r1", entity_id=entity))
    assert "r1" not in c.requests

    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    assert "r1" not in c._early_reads_done
    c._ingest_request(_new_request("r1"))
    c.requests["r1"] = _request_data()  # what the dispatch registers
    c._process_request_done("r1")
    assert c.draining["r1"].expected_acks == {PREPROCESS}


def test_a_queued_abort_s_reads_done_does_not_pre_ack_a_rerun():
    """The data worker answers its own abort with a READS_DONE, which lands
    after the conductor already dropped the queued request."""
    c = _conductor({})
    c.waiting_queue = [_new_request("r1")]
    c._abort_request("r1")
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    assert "r1" not in c._early_reads_done
    c._ingest_request(_new_request("r1"))
    c.requests["r1"] = _request_data()
    c._process_request_done("r1")
    assert c.draining["r1"].expected_acks == {PREPROCESS}


def test_a_late_abort_after_a_rejection_does_not_drop_a_rerun():
    c = _conductor({})
    admitted = _admitting(c)
    c._ingest_request(_new_request("r1", {"seed": "abc"}))  # rejected with a 400
    c._abort_request("r1")
    c._ingest_request(_new_request("r1"))
    assert admitted == ["r1"]


def test_an_abort_for_an_id_never_seen_still_tombstones_it():
    c = _conductor({})
    admitted = _admitting(c)
    c._abort_request("r1")
    c._ingest_request(_new_request("r1"))
    assert admitted == []
    assert _remove_targets(c, "r1") == {PREPROCESS}


def test_late_messages_after_an_abort_that_outran_its_ingest_spare_a_rerun():
    """The abort outran NEW_REQUEST, so ingest dropped the request. The API
    server's own release still sends a READS_DONE, and may send another abort,
    after that; neither may drop or pre-ack a re-run of the id."""
    c = _conductor({})
    admitted = _admitting(c)
    c._abort_request("r1")
    c._ingest_request(_new_request("r1"))

    c._abort_request("r1")
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))
    assert "r1" not in c._early_reads_done
    c._ingest_request(_new_request("r1"))
    assert admitted == ["r1"]
    c.requests["r1"] = _request_data()
    c._process_request_done("r1")
    assert c.draining["r1"].expected_acks == {PREPROCESS}


def test_an_abort_for_a_rerun_in_flight_tears_it_down():
    c = _conductor({"r1": _request_data()})
    _complete_and_tear_down(c, "r1")
    c._ingest_request(_new_request("r1"))
    c.requests["r1"] = _request_data()
    c._abort_request("r1")
    assert "r1" in c.draining


def test_a_torn_down_id_is_forgotten_after_the_ttl():
    c = _conductor({"r1": _request_data()}, drain_ttl_s=0.0)
    admitted = _admitting(c)
    _complete_and_tear_down(c, "r1")
    c._sweep_expiry()
    assert c._torn_down == {} and not c._torn_down_deadlines
    # Past the window, an abort for the id is again taken as racing its ingest.
    c._abort_request("r1")
    c._ingest_request(_new_request("r1"))
    assert admitted == []


# ── ingest failures ─────────────────────────────────────────────────────────

def _failures(c, rid):
    return [m.body for e, m in _by_type(c, "request_failed") if m.body.request_id == rid]


def test_request_seed_takes_an_int64_or_derives_one():
    assert _request_seed({"seed": 7}, "r1") == 7
    assert _request_seed({"seed": -(2**63)}, "r1") == -(2**63)
    assert _request_seed({}, "r1") == _req_id_to_seed("r1")


@pytest.mark.parametrize("seed", ["abc", "7", 1.5, True, 2**63, -(2**63) - 1, [1]])
def test_bad_seed_is_rejected_with_400_not_dropped(seed):
    """A bad seed used to raise in the run loop: logged, never answered."""
    c = _conductor({})
    c._do_ingest_request = lambda body: pytest.fail("a bad seed must not be dispatched")
    c._ingest_request(_new_request("r1", {"seed": seed}))
    assert "r1" not in c.requests and c.waiting_queue == []
    assert _remove_targets(c, "r1") == {PREPROCESS}
    [failure] = _failures(c, "r1")
    assert failure.status == 400 and "seed" in failure.error_message


def test_ingest_error_before_dispatch_fails_the_request():
    c = _conductor({})

    def boom(body):
        raise RuntimeError("model hook broke")
    c._do_ingest_request = boom
    c._ingest_request(_new_request("r1"))
    assert _remove_targets(c, "r1") == {PREPROCESS}
    [failure] = _failures(c, "r1")
    assert failure.status == 500 and "model hook broke" in failure.error_message


def test_bad_value_registered_but_not_dispatched_is_a_400():
    """A config that fails validation is refused after registration but before
    any NewRequest: no worker holds it, so it is dropped, not drained."""
    c = _conductor({})

    def bad_config(body):
        c.requests[body.request_id] = _request_data(workers=("w0",))
        raise ValueError("top_p must be a finite number in (0, 1]; got -1")
    c._do_ingest_request = bad_config
    c._ingest_request(_new_request("r1"))
    assert "r1" not in c.requests and c.draining == {}
    assert _remove_targets(c, "r1") == {PREPROCESS}
    [failure] = _failures(c, "r1")
    assert failure.status == 400 and "top_p" in failure.error_message


def test_ingest_error_after_dispatch_drains_like_a_worker_failure():
    c = _conductor({})

    def boom(body):
        c.requests[body.request_id] = _request_data(workers=("w0",))
        c.requests[body.request_id].dispatched = True
        raise ValueError("half dispatched")
    c._do_ingest_request = boom
    c._ingest_request(_new_request("r1"))
    assert _drain_targets(c, "r1") == {"w0", PREPROCESS}
    assert _failures(c, "r1") == []  # the client hears once the barrier completes
    for entity in ("w0", PREPROCESS):
        c._handle_reads_done(ReadsDone(request_id="r1", entity_id=entity))
    [failure] = _failures(c, "r1")
    assert failure.status == 500 and "half dispatched" in failure.error_message


def test_queued_request_failing_admission_is_answered():
    """Admission from the waiting queue goes through the same fail-safe."""
    c = _conductor({})
    c.waiting_queue = [_new_request("r1")]

    def boom(body):
        raise ValueError("bad temperature")
    c._do_ingest_request = boom
    Conductor._try_admit_waiting(c)
    [failure] = _failures(c, "r1")
    assert failure.status == 400 and c.waiting_queue == []


# ── expiry sweeps ───────────────────────────────────────────────────────────

def test_drain_barrier_times_out_and_force_finalizes():
    """A participant that never ACKs (crashed/hung) must not hold the request —
    or, on the fail path, the client's notification — forever."""
    c = _conductor({"r1": _request_data()}, drain_ttl_s=0.0)
    c._fail_requests(FailRequests(errors={"r1": "boom"}))
    assert "r1" in c.draining  # w0 and the preprocess worker never ACK

    c._sweep_expiry()
    assert "r1" not in c.draining and "r1" not in c.requests
    assert _remove_targets(c, "r1") == {"w0", PREPROCESS}
    failures = [
        m for e, m in c.sent
        if e == "api_server" and m.message_type == "request_failed"
    ]
    assert len(failures) == 1
    assert c.admits == 1


def test_sweep_leaves_a_live_barrier_alone():
    c = _conductor({"r1": _request_data()})
    c._abort_request("r1")
    c._sweep_expiry()
    assert "r1" in c.draining
    assert _remove_targets(c, "r1") == set()


def test_stale_early_entries_are_swept():
    """Early ACKs and abort tombstones for requests that never come back must
    not pile up."""
    c = _conductor({}, drain_ttl_s=0.0)
    c._handle_reads_done(ReadsDone(request_id="gone", entity_id="w0"))
    c._abort_request("also-gone")
    assert c._early_reads_done and c._early_abort_requests

    c._sweep_expiry()
    assert not c._early_reads_done
    assert not c._early_abort_requests
    assert not c._early_reads_done_deadlines
    assert not c._early_abort_deadlines
