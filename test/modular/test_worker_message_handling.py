"""Regression tests for Worker._process_message_list out-of-order handling.

A REMOVE handled while replaying a request's buffered messages must not let its
trailing signals re-buffer onto the list being iterated (which would loop).

The stub keys ``per_request_info`` by HANDLE and resolves the wire string
through ``_rid``, because that is what the worker does. Keying it by the
string instead is what let the guard below compare a string against a
handle-keyed dict -- always true, so every message for a LIVE request was
parked as out-of-order and never delivered.
"""
import threading
import types

from mstar.utils.ipc_format import (
    InputSignals,
    MessageSource,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.worker import Worker


def _stub_worker(active_rids):
    """Minimal stub exposing only what _process_message_list touches.

    ``active_rids`` are wire strings; each gets an integer handle, and state is
    keyed by the handle as on the real worker.
    """
    handles = {rid: i for i, rid in enumerate(active_rids)}
    stub = types.SimpleNamespace()
    stub.request_state = types.SimpleNamespace(
        per_request_info={h: object() for h in handles.values()}
    )
    stub._unprocessed_messages = {}
    stub._rid = handles.get
    stub.delivered = []

    # REMOVE drops the handle from per_request_info (mirrors _remove_request's
    # teardown); the other handlers are no-ops we don't exercise here.
    def _remove(body):
        stub.request_state.per_request_info.pop(handles.get(body.request_id), None)

    stub._remove_request = _remove
    stub._add_new_request = lambda body: None
    stub._process_new_inputs = stub.delivered.append
    stub._handle_tensor_received = lambda body: None
    stub._unpersist_tensors = lambda body: None
    stub._stop_loops = lambda body: None
    stub.scheduler = types.SimpleNamespace(register_tp_follow=lambda body: None)
    return stub


def _run_with_timeout(fn, timeout=5.0):
    done = threading.Event()
    err = []

    def target():
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - surfaced via assertion below
            err.append(e)
        finally:
            done.set()

    threading.Thread(target=target, daemon=True).start()
    return done.wait(timeout), (err[0] if err else None)


def test_replay_buffered_remove_then_signal_terminates():
    """A buffered REMOVE that drops the request mid-replay, followed by trailing
    signals, must not spin _process_message_list forever (the signals would
    otherwise re-buffer onto the list being iterated)."""
    stub = _stub_worker(active_rids=["X"])
    buffered = [
        WorkerMessage(
            message_type=WorkerMessageType.REMOVE_REQUEST,
            body=RemoveRequest(request_id="X", source=MessageSource.TP_RANK_0),
        ),
        WorkerMessage(
            message_type=WorkerMessageType.INPUT_SIGNALS,
            body=InputSignals(request_id="X", inputs=[], request_info=None),
        ),
        WorkerMessage(
            message_type=WorkerMessageType.INPUT_SIGNALS,
            body=InputSignals(request_id="X", inputs=[], request_info=None),
        ),
    ]
    stub._unprocessed_messages["X"] = buffered

    finished, err = _run_with_timeout(
        lambda: Worker._process_message_list(stub, stub._unprocessed_messages["X"]),
        timeout=5.0,
    )
    assert err is None, f"unexpected error: {err!r}"
    assert finished, "_process_message_list did not terminate (re-append loop)"
    # The request was removed and not resurrected.
    assert 0 not in stub.request_state.per_request_info


def test_out_of_order_messages_buffer_for_unknown_request():
    """Signals for a not-yet-added request are buffered, not dropped or looped."""
    stub = _stub_worker(active_rids=[])
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="Y", inputs=[], request_info=None),
    )
    finished, err = _run_with_timeout(
        lambda: Worker._process_message_list(stub, [msg]), timeout=5.0
    )
    assert err is None and finished
    assert stub._unprocessed_messages.get("Y") == [msg]


def test_signals_for_a_live_request_are_delivered_not_buffered():
    """The stall this guard caused: INPUT_SIGNALS for an ADMITTED request.

    ``per_request_info`` is keyed by the worker's integer handle, so testing
    the wire string against it is always true. Every INPUT_SIGNALS for a live
    request was parked as out-of-order and never replayed -- NEW_REQUEST is
    not in the guarded set, so admission and the first forward pass worked and
    nothing after them ever did.
    """
    stub = _stub_worker(active_rids=["X"])
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="X", inputs=[], request_info=None),
    )
    finished, err = _run_with_timeout(
        lambda: Worker._process_message_list(stub, [msg]), timeout=5.0
    )
    assert err is None and finished
    assert stub.delivered == [msg.body], "the signals never reached the worker"
    assert stub._unprocessed_messages == {}, "a live request must not buffer"
