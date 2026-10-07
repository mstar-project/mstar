"""Conductor side of RELEASE_KV: a completed request's workers are told to give
its KV pages back, when ``MSTAR_KV_RELEASE_AT_COMPLETION`` is on, and only then.

The REMOVE_REQUEST that waits for the preprocess worker's READS_DONE is sent
exactly as it is with the toggle off, so a release is always followed by it.
"""

import importlib.util

import pytest
from test_teardown_barrier import (
    PREPROCESS,
    _by_type,
    _conductor,
    _remove_targets,
    _request_data,
)

from mstar.conductor import conductor as conductor_mod
from mstar.utils.ipc_format import FailRequests, ReadsDone, RemoveRequest, WorkerMessageType


def _release_targets(c, rid):
    return {
        e for e, m in _by_type(c, WorkerMessageType.RELEASE_KV)
        if m.body.request_id == rid
    }


@pytest.fixture
def toggle(monkeypatch):
    def set_to(on):
        monkeypatch.setattr(conductor_mod, "_KV_RELEASE_AT_COMPLETION", on)
    return set_to


def test_a_completed_request_is_released_on_every_worker_with_the_toggle_on(toggle):
    toggle(True)
    c = _conductor({"r1": _request_data(workers=("w0", "w1"))})

    c._process_request_done("r1")

    assert _release_targets(c, "r1") == {"w0", "w1"}, "the preprocess worker holds no KV"
    [(_, message)] = _by_type(c, WorkerMessageType.RELEASE_KV)[:1]
    assert message.body == RemoveRequest(request_id="r1")


def test_nothing_is_released_with_the_toggle_off(toggle):
    toggle(False)
    c = _conductor({"r1": _request_data(workers=("w0", "w1"))})

    c._process_request_done("r1")

    assert _by_type(c, WorkerMessageType.RELEASE_KV) == []


def test_the_removal_still_follows_a_release_once_the_outputs_are_read(toggle):
    toggle(True)
    c = _conductor({"r1": _request_data(workers=("w0", "w1"))})
    c._process_request_done("r1")

    assert _remove_targets(c, "r1") == set(), "the release did not replace the removal's wait"
    assert c.draining["r1"].expected_acks == {PREPROCESS}

    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))

    assert _remove_targets(c, "r1") == {"w0", "w1", PREPROCESS}
    assert "r1" not in c.requests and c.admits == 1
    last_release = max(i for i, (_, m) in enumerate(c.sent)
                       if m.message_type == WorkerMessageType.RELEASE_KV)
    first_remove = min(i for i, (_, m) in enumerate(c.sent)
                       if m.message_type == WorkerMessageType.REMOVE_REQUEST)
    assert last_release < first_remove


def test_a_release_precedes_a_removal_that_is_sent_at_once(toggle):
    """The preprocess worker's READS_DONE can be in before the request is done:
    the barrier then finalizes where it registers, and the order each worker
    sees is the order sent."""
    toggle(True)
    c = _conductor({"r1": _request_data()})
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id=PREPROCESS))

    c._process_request_done("r1")

    to_w0 = [m.message_type for e, m in c.sent if e == "w0"]
    assert to_w0 == [WorkerMessageType.RELEASE_KV, WorkerMessageType.REMOVE_REQUEST]


def test_the_client_is_told_before_the_release_is_sent(toggle):
    toggle(True)
    c = _conductor({"r1": _request_data()})

    c._process_request_done("r1")

    assert c.sent[0][0] == "api_server" and c.sent[0][1].message_type == "request_complete"


def test_an_abort_or_a_failure_sends_no_release(toggle):
    toggle(True)
    c = _conductor({
        "aborted": _request_data(workers=("w0", "w1")),
        "failed": _request_data(workers=("w0",)),
    })

    c._abort_request("aborted")
    c._fail_requests(FailRequests(errors={"failed": "boom"}))

    assert _by_type(c, WorkerMessageType.RELEASE_KV) == []


def test_a_released_request_is_not_drained_by_a_late_abort_or_failure(toggle):
    """The conductor never sends a request both a release and a drain: once it is
    done it is draining, and an abort or a failure after that is ignored."""
    toggle(True)
    c = _conductor({"r1": _request_data(workers=("w0", "w1"))})
    c._process_request_done("r1")
    c.sent.clear()

    c._abort_request("r1")
    c._fail_requests(FailRequests(errors={"r1": "boom"}))

    assert c.sent == []


def _module_with_environment(monkeypatch, value):
    """The conductor's module, run again as a copy so the toggle is read afresh."""
    if value is None:
        monkeypatch.delenv("MSTAR_KV_RELEASE_AT_COMPLETION", raising=False)
    else:
        monkeypatch.setenv("MSTAR_KV_RELEASE_AT_COMPLETION", value)
    spec = importlib.util.spec_from_file_location("conductor_toggle_probe", conductor_mod.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("value", "on"), [
    (None, False), ("0", False), ("", False), ("1", True),
])
def test_the_toggle_is_the_environment_read_once_and_defaults_off(monkeypatch, value, on):
    assert _module_with_environment(monkeypatch, value)._KV_RELEASE_AT_COMPLETION is on
