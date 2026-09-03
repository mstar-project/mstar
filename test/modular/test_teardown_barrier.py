"""Teardown barrier (conductor side): the hard RemoveRequest that unlinks a
segment is sent only after every reader has drained (READS_DONE).

Covers the happy path (defer Remove until the preprocess worker finished
delivering outputs, and unpersist still-held signals with correct counts) and
the abort/fail path (drain all workers + preprocess worker, then Remove).
"""

import types

import torch

from mstar.conductor.conductor import Conductor, RequestData
from mstar.graph.base import TensorPointerInfo
from mstar.utils.ipc_format import (
    FailRequests,
    ReadsDone,
    UnpersistTensors,
    WorkerMessageType,
)

PREPROCESS = "api_server_preprocess_worker"


def _request_data(workers=("w0",), persist_signals=None, ref_cnts=None):
    return RequestData(
        persist_signals=persist_signals or {},
        persist_signal_ref_cnt=ref_cnts or {},
        worker_graph_to_workers={"wg": list(workers)},
        all_worker_graph_ids={"wg"},
        max_output_tokens=1,
        random_seed=0,
        sampling_config={},
    )


def _conductor(requests):
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.admits = 0
    c.communicator = types.SimpleNamespace(
        send=lambda entity_id, msg: c.sent.append((entity_id, msg))
    )
    c.requests = dict(requests)
    c.draining = {}
    c._early_reads_done = {}
    c.waiting_queue = []
    c.enable_prof = False
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
