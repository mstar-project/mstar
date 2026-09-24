"""Regression tests for Worker._process_message_list out-of-order handling.

A REMOVE handled while replaying a request's buffered messages must not let its
trailing signals re-buffer onto the list being iterated (which would loop).
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
from mstar.worker.rid_table import RidTable
from mstar.worker.worker import Worker


def _stub_worker(active_rids):
    """Minimal stub exposing only what _process_message_list touches."""
    stub = types.SimpleNamespace()
    # Real interning: per_request_info is keyed by handle while messages carry
    # the string, which is the translation _process_message_list relies on.
    rids = RidTable()
    stub._rid = rids.handle
    stub.worker_graphs_manager = types.SimpleNamespace(
        per_request_info={rids.intern(rid): object() for rid in active_rids}
    )
    stub._unprocessed_messages = {}

    # REMOVE drops the rid from per_request_info and frees its handle (mirrors
    # _remove_request's teardown); the other handlers are no-ops we don't
    # exercise here.
    def _remove(body):
        handle = rids.handle(body.request_id)
        stub.worker_graphs_manager.per_request_info.pop(handle, None)
        rids.release(handle)

    stub._remove_request = _remove
    stub._add_new_request = lambda body: None
    stub._process_new_inputs = lambda body: None
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
    assert stub._rid("X") is None
    assert not stub.worker_graphs_manager.per_request_info


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
