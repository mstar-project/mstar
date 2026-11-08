"""Regression tests for InputIngest's out-of-order handling.

A REMOVE handled while replaying a request's buffered messages must not let its
trailing signals re-buffer onto the list being iterated (which would loop).

The stub keys ``per_request_info`` by HANDLE and resolves the wire string
through ``get_rid_handle``, because that is what the worker does. Keying it
by the string instead is what let the guard below compare a string against a
handle-keyed dict -- always true, so every message for a LIVE request was
parked as out-of-order and never delivered.
"""
import threading
import types

from mstar.testing_utils.worker import make_worker
from mstar.utils.ipc_format import (
    InputSignals,
    MessageSource,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)


def _stub_worker(active_rids):
    """A worker with the real ``InputIngest`` and ``RequestLifecycle``, over a
    runtime that only resolves handles. Only the input handler is replaced, to
    record what reaches it.

    ``active_rids`` are wire strings; each gets an integer handle, and state is
    keyed by the handle as on the real worker.
    """
    handles = {rid: i for i, rid in enumerate(active_rids)}
    request_state = types.SimpleNamespace(
        per_request_info={h: object() for h in handles.values()}
    )
    worker = make_worker(
        request_state=request_state,
        _graph_runtime=types.SimpleNamespace(get_rid_handle=handles.get),
    )
    for rid, handle in handles.items():
        worker.lifecycle.add(handle, rid)
    worker.delivered = []
    worker.ingest._handlers[WorkerMessageType.INPUT_SIGNALS] = worker.delivered.append
    return worker


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

    finished, err = _run_with_timeout(lambda: stub.ingest._dispatch(buffered), timeout=5.0)
    assert err is None, f"unexpected error: {err!r}"
    assert finished, "the dispatch did not terminate (re-append loop)"
    # The remove is recorded, and the trailing signals are not parked behind it.
    assert list(stub.lifecycle.pending_removes) == [0]
    assert stub.lifecycle._parked == {}


def test_out_of_order_messages_buffer_for_unknown_request():
    """Signals for a not-yet-added request are buffered, not dropped or looped."""
    stub = _stub_worker(active_rids=[])
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="Y", inputs=[], request_info=None),
    )
    finished, err = _run_with_timeout(
        lambda: stub.ingest._dispatch([msg]), timeout=5.0
    )
    assert err is None and finished
    assert stub.lifecycle._parked.get("Y") == [msg]


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
        lambda: stub.ingest._dispatch([msg]), timeout=5.0
    )
    assert err is None and finished
    assert stub.delivered == [msg.body], "the signals never reached the worker"
    assert stub.lifecycle._parked == {}, "a live request must not buffer"
