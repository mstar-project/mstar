"""The conductor's pause, continue and weight-update state machine.

Driven through a bare ``Conductor`` with the real ``GenerationControl`` wired the
way ``__init__`` wires it, so the admission gate, the abort path (the per-request
drain barrier) and the answers to the API server are the production code. Worker
ranks are played by hand: each protocol step is answered with a
``WeightControlAck`` as a worker would send it.
"""
from __future__ import annotations

import types
from collections import deque

import pytest

from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.generation_control import GenerationControl
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    ContinueGeneration,
    GetGenerationState,
    NewRequestConductor,
    PauseGeneration,
    ReadsDone,
    UpdateWeightsFromDisk,
    WeightControlAck,
    WeightOp,
    WorkerMessageType,
)

WORKERS = ["worker_0", "worker_1"]
PREPROCESS = "api_server_preprocess_worker"


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _request_data(workers=tuple(WORKERS)):
    return RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: list(workers)}, all_worker_graph_ids={0},
        max_output_tokens=1, random_seed=0, resource_configs={},
    )


def _new_request(rid):
    return NewRequestConductor(
        request_id=rid, initial_signals={}, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
    )


def _conductor(requests=(), timeout_s=30.0):
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.communicator = types.SimpleNamespace(send=lambda peer, msg: c.sent.append((peer, msg)))
    c.requests = dict(requests)
    c.draining = {}
    c._drain_ttl_s = 120.0
    c._early_reads_done = {}
    c._early_abort_requests = set()
    c._draining_deadlines = deque()
    c._early_reads_done_deadlines = deque()
    c._early_abort_deadlines = deque()
    c.waiting_queue = []
    c.enable_prof = False
    c.enable_nvtx = False
    c.max_concurrent_requests = None
    c.admitted = []
    c._admit_request = lambda body: c.admitted.append(body.request_id)
    c._try_admit_waiting = lambda: None
    c.clock = _Clock()
    c._generation = GenerationControl(
        workers=WORKERS,
        send=c.communicator.send,
        reject_queued=lambda: c._reject_waiting_queue("generation is paused", status=503),
        abort_in_flight=c._abort_in_flight_for_pause,
        timeout_s=timeout_s,
        clock=c.clock,
    )
    return c


def _steps(c, op: WeightOp | None = None):
    """The WeightControl messages sent so far, oldest first."""
    return [
        (peer, m.body) for peer, m in c.sent
        if getattr(m, "message_type", None) == WorkerMessageType.WEIGHT_CONTROL
        and (op is None or m.body.op == op.value)
    ]


def _answers(c, op_id=None):
    return [
        m.body for peer, m in c.sent
        if peer == "api_server" and getattr(m, "message_type", None) == "generation_control_result"
        and (op_id is None or m.body.op_id == op_id)
    ]


def _failures(c):
    return {
        m.body.request_id: (m.body.status, m.body.error_message) for peer, m in c.sent
        if peer == "api_server" and getattr(m, "message_type", None) == "request_failed"
    }


def _applied(c) -> int:
    """The epoch of the latest apply step: what a rank that ran it reports."""
    return _steps(c, WeightOp.APPLY)[-1][1].epoch


def _ack(c, op: WeightOp, worker, error=None, version=None, dirty=False, weight_epoch=None):
    epoch = _steps(c, op)[-1][1].epoch
    if weight_epoch is None:
        if op is WeightOp.APPLY and error is None:
            weight_epoch = epoch  # this step's copy
        elif version is None:
            weight_epoch = 0  # the startup checkpoint
        else:
            weight_epoch = _applied(c)  # the latest copy
    c._generation.ack(WeightControlAck(
        epoch=epoch, worker_id=worker, op=op.value, error=error,
        weight_version=version, weight_epoch=weight_epoch, dirty=dirty,
    ))


def _ack_all(c, op: WeightOp, version=None, errors=None):
    for worker in WORKERS:
        _ack(c, op, worker, error=(errors or {}).get(worker), version=version)


def _finish_drain(c, rid):
    for entity in [*WORKERS, PREPROCESS]:
        c._handle_reads_done(ReadsDone(request_id=rid, entity_id=entity))


def _paused(c):
    """Pause an idle conductor and let every rank confirm it."""
    c._generation.pause(PauseGeneration(op_id="pause", mode="wait"))
    c._generation.tick(len(c.requests))
    _ack_all(c, WeightOp.QUIESCE)
    assert _answers(c, "pause")[-1].status == 200
    return c


# ── pausing ─────────────────────────────────────────────────────────────────


def test_wait_mode_lets_requests_finish_and_refuses_new_ones():
    c = _conductor({"r1": _request_data()})
    c.waiting_queue.append(_new_request("queued"))

    c._generation.pause(PauseGeneration(op_id="p", mode="wait"))
    c._ingest_request(_new_request("late"))

    assert _failures(c) == {
        "queued": (503, "generation is paused"), "late": (503, "generation is paused"),
    }
    assert "r1" not in c.draining and c.admitted == [], "the running request was touched"
    c._generation.tick(len(c.requests))
    assert _steps(c) == [] and _answers(c, "p") == [], "paused before the request finished"

    c.requests.pop("r1")  # it completed and was torn down
    c._generation.tick(len(c.requests))
    assert {peer for peer, _ in _steps(c, WeightOp.QUIESCE)} == set(WORKERS)
    _ack(c, WeightOp.QUIESCE, "worker_0")
    assert _answers(c, "p") == [], "answered before every rank confirmed idle"
    _ack(c, WeightOp.QUIESCE, "worker_1")
    answer = _answers(c, "p")[-1]
    assert (answer.status, answer.state, answer.paused) == (200, "paused", True)


def test_abort_mode_fails_requests_through_the_drain_barrier():
    c = _conductor({"r1": _request_data(), "r2": _request_data()})

    c._generation.pause(PauseGeneration(op_id="p", mode="abort"))

    drained = {
        (peer, m.body.request_id) for peer, m in c.sent
        if getattr(m, "message_type", None) == WorkerMessageType.DRAIN_REQUEST
    }
    assert drained == {(e, r) for e in [*WORKERS, PREPROCESS] for r in ("r1", "r2")}
    assert _failures(c) == {}, "a client heard before its readers drained"
    for rid in ("r1", "r2"):
        _finish_drain(c, rid)
    assert {r: s for r, (s, _msg) in _failures(c).items()} == {"r1": 503, "r2": 503}
    assert not c.requests

    c._generation.tick(len(c.requests))
    _ack_all(c, WeightOp.QUIESCE)
    assert _answers(c, "p")[-1].status == 200


def test_a_wait_pause_escalates_to_abort():
    c = _conductor({"r1": _request_data()})
    c._generation.pause(PauseGeneration(op_id="p1", mode="wait"))
    assert "r1" not in c.draining
    c._generation.pause(PauseGeneration(op_id="p2", mode="abort"))
    assert "r1" in c.draining
    _finish_drain(c, "r1")
    c._generation.tick(len(c.requests))
    _ack_all(c, WeightOp.QUIESCE)
    assert [a.status for a in _answers(c)] == [200, 200], "both pause calls are answered"


def test_a_completed_request_is_not_failed_by_an_abort():
    c = _conductor({"done": _request_data()})
    c._register_draining("done", expected_acks={PREPROCESS}, participants={*WORKERS, PREPROCESS})
    c._generation.pause(PauseGeneration(op_id="p", mode="abort"))
    _finish_drain(c, "done")
    assert _failures(c) == {}


def test_pause_is_idempotent_and_checks_its_mode():
    c = _paused(_conductor())
    c._generation.pause(PauseGeneration(op_id="again", mode="abort"))
    assert _answers(c, "again")[-1].status == 200
    assert len({s.epoch for _, s in _steps(c, WeightOp.QUIESCE)}) == 1, "a second pause asked the ranks again"
    c._generation.pause(PauseGeneration(op_id="bad", mode="later"))
    assert _answers(c, "bad")[-1].status == 400


def test_a_rank_that_never_confirms_idle_times_out_and_blocks_updates():
    c = _conductor(timeout_s=10.0)
    c._generation.pause(PauseGeneration(op_id="p", mode="wait"))
    c._generation.tick(0)
    _ack(c, WeightOp.QUIESCE, "worker_0")
    c.clock.now = 11.0
    c._generation.tick(0)
    answer = _answers(c, "p")[-1]
    assert answer.status == 504 and "worker_1" in answer.error_message
    assert answer.paused, "a timed-out pause must not reopen admission"

    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1"))
    assert _answers(c, "u")[-1].status == 409, "updated without every rank idle"

    # pausing again asks every rank once more
    c._generation.pause(PauseGeneration(op_id="p2", mode="wait"))
    _ack_all(c, WeightOp.QUIESCE)
    assert _answers(c, "p2")[-1].status == 200


# ── updating ────────────────────────────────────────────────────────────────


def test_an_update_needs_a_pause():
    c = _conductor()
    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1"))
    assert _answers(c, "u")[-1].status == 409
    assert _steps(c) == []


def test_an_update_checks_then_copies_then_commits():
    c = _paused(_conductor())
    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/ckpt/v1", weight_version="v1"))
    prepare = _steps(c, WeightOp.PREPARE)
    assert {peer for peer, _ in prepare} == set(WORKERS)
    assert prepare[0][1].model_path == "/ckpt/v1"
    assert _steps(c, WeightOp.APPLY) == [], "copied before every rank checked"

    _ack_all(c, WeightOp.PREPARE)
    apply_ = _steps(c, WeightOp.APPLY)
    assert {peer for peer, _ in apply_} == set(WORKERS) and apply_[0][1].weight_version == "v1"
    assert apply_[0][1].epoch > prepare[0][1].epoch

    _ack(c, WeightOp.APPLY, "worker_0", version="v1")
    assert c._generation.weight_version is None, "committed before every rank copied"
    _ack(c, WeightOp.APPLY, "worker_1", version="v1")
    answer = _answers(c, "u")[-1]
    assert (answer.status, answer.weight_version, answer.state) == (200, "v1", "paused")
    assert answer.ranks == dict.fromkeys(WORKERS, "ok")

    c._generation.resume(ContinueGeneration(op_id="c"))
    _ack_all(c, WeightOp.VERIFY, version="v1")
    assert _answers(c, "c")[-1].status == 200
    c._ingest_request(_new_request("next"))
    assert c.admitted == ["next"]


def test_a_refused_checkpoint_keeps_the_old_version_and_can_resume():
    c = _paused(_conductor())
    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/bad", weight_version="v1"))
    _ack_all(c, WeightOp.PREPARE, errors={"worker_1": "WeightReloadRefusedError: shape"})
    answer = _answers(c, "u")[-1]
    assert answer.status == 400 and "worker_1: WeightReloadRefusedError: shape" in answer.error_message
    assert answer.ranks == {"worker_0": "ok", "worker_1": "WeightReloadRefusedError: shape"}
    assert _steps(c, WeightOp.APPLY) == [] and c._generation.weight_version is None

    c._generation.resume(ContinueGeneration(op_id="c"))
    _ack_all(c, WeightOp.VERIFY, version=None)
    assert _answers(c, "c")[-1].status == 200, "nothing was copied, so the old weights can serve"


def test_a_failed_copy_on_a_follower_rank_stays_paused_until_an_update_succeeds():
    c = _paused(_conductor())
    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1"))
    _ack_all(c, WeightOp.PREPARE)
    _ack(c, WeightOp.APPLY, "worker_0", version="v1")
    _ack(c, WeightOp.APPLY, "worker_1", error="OSError: disk", dirty=True)

    answer = _answers(c, "u")[-1]
    assert answer.status == 500 and answer.paused
    assert answer.ranks == {"worker_0": "ok", "worker_1": "OSError: disk"}
    assert answer.rank_versions == {"worker_0": "v1", "worker_1": None}
    assert c._generation.weight_version is None, "a version committed on part of the ranks"

    c._generation.resume(ContinueGeneration(op_id="c1"))
    _ack(c, WeightOp.VERIFY, "worker_0", version="v1")
    _ack(c, WeightOp.VERIFY, "worker_1", version=None, dirty=True)
    refused = _answers(c, "c1")[-1]
    assert refused.status == 409 and refused.paused
    c._ingest_request(_new_request("r"))
    assert c.admitted == [] and _failures(c)["r"][0] == 503, "served mixed weights"

    c._generation.update(UpdateWeightsFromDisk(op_id="u2", model_path="/ckpt", weight_version="v1"))
    _ack_all(c, WeightOp.PREPARE)
    _ack_all(c, WeightOp.APPLY, version="v1")
    c._generation.resume(ContinueGeneration(op_id="c2"))
    _ack_all(c, WeightOp.VERIFY, version="v1")
    assert _answers(c, "c2")[-1].status == 200


def test_a_rank_silent_during_the_copy_times_out_and_its_late_answer_is_ignored():
    c = _paused(_conductor(timeout_s=10.0))
    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1"))
    _ack_all(c, WeightOp.PREPARE)
    _ack(c, WeightOp.APPLY, "worker_0", version="v1")
    late_epoch = _steps(c, WeightOp.APPLY)[-1][1].epoch
    c.clock.now += 11.0
    c._generation.tick(0)
    assert _answers(c, "u")[-1].status == 504
    assert c._generation.weight_version is None

    c._generation.resume(ContinueGeneration(op_id="c"))
    c._generation.ack(WeightControlAck(
        epoch=late_epoch, worker_id="worker_1", op="apply", weight_version="v1",
    ))
    assert _answers(c, "c") == [], "an answer from an old step counted toward this one"
    _ack_all(c, WeightOp.VERIFY, version="v1")
    assert _answers(c, "c")[-1].status == 409, "resumed on a version that never committed"


def test_a_reused_label_on_a_rank_that_missed_the_copy_does_not_resume():
    """Commit v1, then copy other weights under the same label; one rank never
    runs that copy. Both report "v1", but not the same copy."""
    c = _paused(_conductor(timeout_s=10.0))
    c._generation.update(UpdateWeightsFromDisk(op_id="u1", model_path="/ckpt/b", weight_version="v1"))
    _ack_all(c, WeightOp.PREPARE)
    _ack_all(c, WeightOp.APPLY, version="v1")
    first_copy = _applied(c)
    assert _answers(c, "u1")[-1].status == 200

    c._generation.update(UpdateWeightsFromDisk(op_id="u2", model_path="/ckpt/c", weight_version="v1"))
    _ack_all(c, WeightOp.PREPARE)
    _ack(c, WeightOp.APPLY, "worker_0", version="v1")
    c.clock.now += 11.0
    c._generation.tick(0)
    assert _answers(c, "u2")[-1].status == 504

    c._generation.resume(ContinueGeneration(op_id="c"))
    _ack(c, WeightOp.VERIFY, "worker_0", version="v1", weight_epoch=_applied(c))
    _ack(c, WeightOp.VERIFY, "worker_1", version="v1", weight_epoch=first_copy)
    refused = _answers(c, "c")[-1]
    assert refused.status == 409 and refused.paused, "resumed with two copies under one label"
    assert "worker_0" in refused.error_message and "worker_1" not in refused.error_message


def test_a_pause_during_resume_verification_stays_paused():
    """A pause that lands while continue is still being verified used to get
    200 paused, and the verify then resumed anyway."""
    c = _paused(_conductor())
    c._generation.resume(ContinueGeneration(op_id="c"))
    assert c._generation.state.value == "verifying"
    c._generation.pause(PauseGeneration(op_id="p", mode="wait"))
    assert _answers(c, "p") == [], "pause was answered before verify finished"
    _ack_all(c, WeightOp.VERIFY, version=None)
    pause = _answers(c, "p")[-1]
    assert (pause.status, pause.state, pause.paused) == (200, "paused", True)
    assert c._generation.state.value == "paused"
    cont = _answers(c, "c")[-1]
    assert cont.status == 409 and cont.paused and cont.state == "paused"
    c._ingest_request(_new_request("late"))
    assert _failures(c)["late"] == (503, "generation is paused")
    c._generation.resume(ContinueGeneration(op_id="c2"))
    _ack_all(c, WeightOp.VERIFY, version=None)
    assert (_answers(c, "c2")[-1].status, c._generation.state.value) == (200, "serving")


def test_an_apply_no_rank_ran_leaves_the_committed_copy_servable():
    """Every rank timed out before copying (their apply was superseded): all
    still hold the committed copy, so serving it again is safe."""
    c = _paused(_conductor(timeout_s=10.0))
    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1"))
    _ack_all(c, WeightOp.PREPARE)
    c.clock.now += 11.0
    c._generation.tick(0)
    assert _answers(c, "u")[-1].status == 504
    c._generation.resume(ContinueGeneration(op_id="c"))
    _ack_all(c, WeightOp.VERIFY, version=None)
    assert _answers(c, "c")[-1].status == 200


def test_operations_in_the_wrong_order_are_refused():
    c = _conductor({"r1": _request_data()})
    c._generation.resume(ContinueGeneration(op_id="serving"))
    assert _answers(c, "serving")[-1].status == 200, "continue while serving is a no-op"

    c._generation.pause(PauseGeneration(op_id="p", mode="wait"))
    c._generation.resume(ContinueGeneration(op_id="draining"))
    assert _answers(c, "draining")[-1].status == 409
    c.requests.clear()
    c._generation.tick(0)
    _ack_all(c, WeightOp.QUIESCE)

    c._generation.update(UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1"))
    c._generation.update(UpdateWeightsFromDisk(op_id="u2", model_path="/ckpt", weight_version="v2"))
    assert _answers(c, "u2")[-1].status == 409, "two updates ran at once"
    c._generation.resume(ContinueGeneration(op_id="mid"))
    assert _answers(c, "mid")[-1].status == 409, "resumed in the middle of an update"

    c._generation.report(GetGenerationState(op_id="s"))
    state = _answers(c, "s")[-1]
    assert (state.status, state.state, state.paused) == (200, "preparing", True)


@pytest.mark.parametrize("field, value", [("model_path", ""), ("weight_version", "")])
def test_an_update_needs_a_path_and_a_version(field, value):
    c = _paused(_conductor())
    body = UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1")
    setattr(body, field, value)
    c._generation.update(body)
    assert _answers(c, "u")[-1].status == 400 and field in _answers(c, "u")[-1].error_message


def test_a_conductor_without_the_state_machine_refuses_control_messages():
    c = _conductor()
    c._generation = None
    for message_type, body in [
        (ConductorMessageType.PAUSE_GENERATION, PauseGeneration(op_id="p")),
        (ConductorMessageType.CONTINUE_GENERATION, ContinueGeneration(op_id="c")),
        (ConductorMessageType.UPDATE_WEIGHTS_FROM_DISK,
         UpdateWeightsFromDisk(op_id="u", model_path="/ckpt", weight_version="v1")),
        (ConductorMessageType.GET_GENERATION_STATE, GetGenerationState(op_id="s")),
        (ConductorMessageType.WEIGHT_CONTROL_ACK, WeightControlAck(epoch=1, worker_id="worker_0", op="quiesce")),
    ]:
        c._route_message(ConductorMessage(message_type=message_type, body=body), [])
    assert [(a.op_id, a.status, a.paused) for a in _answers(c)] == [
        ("p", 503, False), ("c", 503, False), ("u", 503, False), ("s", 503, False),
    ]
    assert len(c.sent) == 4, "the ack has no caller to answer"
