"""The data worker's drain sets after a request's hard remove.

For a failed or rejected request the conductor sends REMOVE_REQUEST to the
data worker before ``request_failed`` reaches the API server, so the API
server's release (``finished_reading``, or an abort) lands after the hard
cleanup. No second REMOVE_REQUEST follows it, so that release must not put the
id back into ``_draining_rids`` / ``_reads_done_sent`` or ACK READS_DONE again.
A new request that reuses the id must still read and ACK normally, and nothing
is kept for an id once its last request is removed.

Each test drives the real worker loop on its own thread through the facade's
queues and a fake conductor socket, in the order the conductor sends.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.model.base import Model
from mstar.streaming.topology import PartitionTopology
from mstar.utils.ipc_format import (
    ConductorMessageType,
    DrainRequest,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)


class _Model:
    def __init__(self, fail_preprocess=False):
        self.fail_preprocess = fail_preprocess

    def get_partition_topology(self):
        return PartitionTopology(partitions=["default"])

    def process_prompt(self, *args, **kwargs):
        if self.fail_preprocess:
            raise ValueError("bad knob")

    # the inline path's per-value default, over the stub's postprocess
    postprocess_inline = Model.postprocess_inline

    def postprocess(self, output, modality, request_kwargs=None):
        return f"<{int(output[0])}>".encode()


class _Conductor:
    """The data worker's socket: ``deliver`` queues a conductor message, and
    every pass of the loop polls it once."""

    def __init__(self):
        self.inbox = queue.Queue()
        self.sent = []
        self.polls = 0
        self.cond = threading.Condition()
        # cleared by _Rig.hold: the next pass waits here before doing anything
        self.open = threading.Event()
        self.open.set()
        self.held = threading.Event()

    def get_all_new_messages(self):
        if not self.open.is_set():
            self.held.set()
            self.open.wait()
        msgs = []
        while not self.inbox.empty():
            msgs.append(self.inbox.get())
        with self.cond:
            self.polls += 1
            self.cond.notify_all()
        return msgs

    def send(self, entity, msg):
        self.sent.append((entity, msg))

    # the data worker's outbox encodes with this: identity, so sends arrive as objects
    codec = SimpleNamespace(encode=lambda msg: msg)

    def try_send_bytes(self, entity, msg):
        self.send(entity, msg)
        return True

    def deliver(self, message_type, body):
        self.inbox.put(WorkerMessage(message_type=message_type, body=body))

    def sent_of(self, message_type, rid):
        return [
            m for e, m in self.sent if e == "conductor" and m.message_type == message_type and m.body.request_id == rid
        ]


class _TensorManager:
    def __init__(self):
        self.forced = []

    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass

    def take_failed_reads(self):

        return []


    def get_ready_tensors(self):
        # the loop polls this after its release queues and before in_queue
        hook, self.on_ready = getattr(self, "on_ready", None), None
        if hook is not None:
            hook()
        return {}

    def has_inflight_reads(self, request_id):
        return False

    def force_cleanup_request(self, request_id):
        self.forced.append(request_id)

    def ack_unread_tensors(self, request_id, graph_edges):
        pass


def _input(rid):
    return PreprocessInput(
        request_id=rid,
        text="hi",
        file_paths=None,
        input_modalities=["text"],
        output_modalities=["text"],
        model_kwargs={},
    )


class _Rig:
    def __init__(self, model, start=True, api_pending=False):
        self.conductor = _Conductor()
        # api_pending: the API server's pending set, as is_pending reads it.
        # It lets go of a submission before queueing its abort.
        self.pending: set[str] = set()
        self.tm = _TensorManager()
        self.stop = threading.Event()
        self.worker = PreprocessWorkerThread(
            in_queue=queue.Queue(),
            result_tensor_queue=queue.Queue(),
            out_queue=queue.Queue(),
            profile_queue=queue.Queue(),
            cleanup_request_queue=queue.Queue(),
            abort_request_queue=queue.Queue(),
            reads_done_queue=queue.Queue(),
            discard_tensor_queue=queue.Queue(),
            stop_event=self.stop,
            communicator=self.conductor,
            tensor_manager=self.tm,
            model=model,
            is_pending=(lambda key: key in self.pending) if api_pending else None,
        )
        # the API server's side of the same queues
        facade = PreprocessWorker.__new__(PreprocessWorker)
        facade.request_input_queue = self.worker.in_queue
        facade.result_tensor_input_queue = self.worker.result_tensor_queue
        facade.abort_request_queue = self.worker.abort_request_queue
        facade.reads_done_queue = self.worker.reads_done_queue
        facade.received_chunk_counts = {}
        facade.per_request_reading_tensors = {}
        if api_pending:
            new_request, abort_request = facade.new_request, facade.abort_request

            def submit(input):
                self.pending.add(input.request_id)
                new_request(input)

            def abort(request_id):
                self.pending.discard(request_id)
                abort_request(request_id)

            facade.new_request, facade.abort_request = submit, abort
        self.api = facade
        self.thread = threading.Thread(target=self.worker.run, daemon=True)
        if start:
            self.thread.start()

    def settle(self):
        """Return once a whole loop pass has run after everything queued so far."""
        with self.conductor.cond:
            target = self.conductor.polls + 2
            assert self.conductor.cond.wait_for(lambda: self.conductor.polls >= target, timeout=10)

    def hold(self):
        """Park the loop at the start of a pass, so what is queued next lands in one pass."""
        self.conductor.held.clear()
        self.conductor.open.clear()
        assert self.conductor.held.wait(timeout=10)

    def resume(self):
        self.conductor.open.set()
        self.settle()

    def close(self):
        self.conductor.open.set()
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=10)

    def submit(self, rid):
        self.api.new_request(_input(rid))
        self.settle()

    def fail_in_engine(self, rid):
        """The conductor's failure barrier: drain, then the hard remove."""
        self.conductor.deliver(WorkerMessageType.DRAIN_REQUEST, DrainRequest(rid))
        self.settle()
        self.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest(rid))
        self.settle()

    def reads_done(self, rid):
        return self.conductor.sent_of(ConductorMessageType.READS_DONE, rid)

    def assert_forgotten(self, rid):
        """Nothing is kept for the rid."""
        assert rid not in self.worker._draining_rids
        assert rid not in self.worker._reads_done_sent
        assert rid not in self.worker._live_requests


@pytest.fixture
def rig():
    made = []

    def make(model=None, start=True, api_pending=False):
        made.append(_Rig(model or _Model(), start=start, api_pending=api_pending))
        return made[-1]

    yield make
    for r in made:
        r.close()


@pytest.mark.parametrize("drained", [False, True])
def test_failed_request_release_after_hard_remove_leaves_nothing(rig, drained):
    r = rig()
    r.submit("X")
    r.fail_in_engine("X")
    assert len(r.reads_done("X")) == 1

    r.api.finished_reading("X", drained=drained)
    r.settle()

    r.assert_forgotten("X")
    assert len(r.reads_done("X")) == 1


def test_rejected_request_release_after_hard_remove_leaves_nothing(rig):
    """``_reject_request``: REMOVE_REQUEST with no drain, then request_failed."""
    r = rig()
    r.submit("X")
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()

    r.api.finished_reading("X", drained=False)
    r.settle()

    r.assert_forgotten("X")
    assert not r.reads_done("X")


def test_abort_after_hard_remove_does_not_drain(rig):
    r = rig()
    r.submit("X")
    r.fail_in_engine("X")

    r.api.abort_request("X")
    r.settle()

    r.assert_forgotten("X")
    assert len(r.reads_done("X")) == 1
    # still forwarded: the conductor owns what an abort for an id means
    assert len(r.conductor.sent_of(ConductorMessageType.ABORT_REQUEST, "X")) == 1


def test_preprocess_failure_release_leaves_nothing(rig):
    """A request that never reached the conductor gets no REMOVE_REQUEST at all,
    yet its error chunk still makes the API server release it."""
    r = rig(_Model(fail_preprocess=True))
    r.submit("X")
    assert r.worker.out_queue.get(timeout=10).modality == "error"
    assert not r.conductor.sent_of(ConductorMessageType.NEW_REQUEST, "X")

    r.api.finished_reading("X", drained=False)
    r.settle()

    r.assert_forgotten("X")
    assert not r.reads_done("X")


def test_new_request_reusing_a_removed_id_reads_and_acks(rig):
    r = rig()
    r.submit("X")
    r.fail_in_engine("X")
    r.api.finished_reading("X", drained=False)
    r.settle()

    # the client retries with the same pinned id
    r.submit("X")
    assert len(r.conductor.sent_of(ConductorMessageType.NEW_REQUEST, "X")) == 2
    acked = len(r.reads_done("X"))
    r.api.new_result_tokens(["X"], [7], "text_inputs", "text")
    chunk = r.worker.out_queue.get(timeout=2)
    assert chunk.request_id == "X" and chunk.data == b"<7>"

    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == acked + 1

    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()
    r.assert_forgotten("X")


@pytest.mark.parametrize("drained", [False, True])
def test_abort_of_a_new_request_reusing_a_removed_id_drains(rig, drained):
    r = rig()
    r.submit("X")
    r.fail_in_engine("X")
    r.api.finished_reading("X", drained=drained)
    r.settle()

    r.submit("X")
    acked = len(r.reads_done("X"))
    r.api.abort_request("X")
    r.settle()

    assert "X" in r.worker._draining_rids
    assert len(r.reads_done("X")) == acked + 1


def test_finished_request_keeps_nothing(rig):
    """Happy path: the release comes before the REMOVE_REQUEST it triggers."""
    r = rig()
    r.submit("X")
    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == 1
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()
    r.assert_forgotten("X")


def test_abort_during_a_failure_keeps_nothing(rig):
    """The client goes away between the drain and the hard remove; the API
    server then sends no release for the failure."""
    r = rig()
    r.submit("X")
    r.conductor.deliver(WorkerMessageType.DRAIN_REQUEST, DrainRequest("X"))
    r.settle()
    r.api.abort_request("X")
    r.settle()
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()
    r.assert_forgotten("X")
    assert len(r.reads_done("X")) == 1


def test_abort_queued_before_a_failing_preprocess_leaves_nothing(rig):
    """The abort is handled first in the pass, then preprocessing fails; the
    request never reached the conductor, so no REMOVE_REQUEST comes. (A
    thread with no is_pending preprocesses whatever is queued.)"""
    r = rig(_Model(fail_preprocess=True), start=False)
    r.api.new_request(_input("X"))
    r.api.abort_request("X")
    r.thread.start()
    assert r.worker.out_queue.get(timeout=10).modality == "error"
    r.settle()
    r.assert_forgotten("X")


def test_abort_queued_before_preprocessing_drops_it_unsent(rig):
    """The API server lets go of the request while it waits for
    preprocessing: it is dropped unprocessed and unsent, so no REMOVE_REQUEST
    comes, and nothing is kept for it."""
    r = rig(_Model(fail_preprocess=True), start=False, api_pending=True)
    r.api.new_request(_input("X"))
    r.api.abort_request("X")
    r.thread.start()
    r.settle()
    assert r.worker.out_queue.empty()
    assert r.conductor.sent_of(ConductorMessageType.NEW_REQUEST, "X") == []
    assert len(r.conductor.sent_of(ConductorMessageType.ABORT_REQUEST, "X")) == 1
    r.assert_forgotten("X")


def test_late_remove_of_a_finished_request_does_not_swallow_a_retry(rig):
    """The old request's REMOVE_REQUEST lands after a retry under its id was
    already preprocessed; the retry must still deliver and ACK."""
    r = rig()
    r.submit("X")
    r.api.finished_reading("X", drained=True)
    r.settle()
    r.submit("X")
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()

    r.api.new_result_tokens(["X"], [7], "text_inputs", "text")
    assert r.worker.out_queue.get(timeout=2).data == b"<7>"
    acked = len(r.reads_done("X"))
    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == acked + 1

    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()
    r.assert_forgotten("X")


def test_abort_of_a_queued_retry_does_not_swallow_the_next_retry(rig):
    """A fails; B reuses the id and is aborted before preprocessing (its abort
    also drops A from recently_completed, so A's release never comes); C reuses
    it again and is preprocessed before B's REMOVE_REQUEST lands. C must ACK."""
    r = rig()
    r.submit("X")
    r.fail_in_engine("X")

    r.hold()
    r.api.new_request(_input("X"))
    r.api.abort_request("X")
    r.resume()
    assert len(r.conductor.sent_of(ConductorMessageType.NEW_REQUEST, "X")) == 2

    r.submit("X")
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()

    r.api.new_result_tokens(["X"], [7], "text_inputs", "text")
    assert r.worker.out_queue.get(timeout=2).data == b"<7>"
    acked = len(r.reads_done("X"))
    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == acked + 1

    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()
    r.assert_forgotten("X")


def test_a_queued_retry_dropped_for_its_abort_does_not_swallow_the_next_retry(rig):
    """As above, but the API server let go of B before its abort, so B is
    dropped unsent; C reuses the id again. C must ACK."""
    r = rig(api_pending=True)
    r.submit("X")
    r.fail_in_engine("X")

    r.hold()
    r.api.new_request(_input("X"))
    r.api.abort_request("X")
    r.resume()
    assert len(r.conductor.sent_of(ConductorMessageType.NEW_REQUEST, "X")) == 1

    r.submit("X")
    assert len(r.conductor.sent_of(ConductorMessageType.NEW_REQUEST, "X")) == 2

    r.api.new_result_tokens(["X"], [7], "text_inputs", "text")
    assert r.worker.out_queue.get(timeout=2).data == b"<7>"
    acked = len(r.reads_done("X"))
    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == acked + 1

    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()
    r.assert_forgotten("X")


def test_abort_landing_after_the_pending_check_tears_the_retry_down(rig):
    """As above, but B's abort lands just after the thread found B still
    pending: B goes to the conductor and its abort after it, B is drained as
    any running request, and B's REMOVE_REQUEST comes only after C was
    preprocessed. C must ACK."""
    r = rig(api_pending=True)
    r.submit("X")
    r.fail_in_engine("X")

    check = r.worker._is_pending

    def pending_then_aborted(key):
        held = check(key)
        if held:
            r.worker._is_pending = check
            r.api.abort_request(key)
        return held

    r.worker._is_pending = pending_then_aborted
    r.submit("X")
    sent = [m.message_type for e, m in r.conductor.sent if m.body.request_id == "X"]
    # drained as a running request: with no read in flight it ACKs at once
    assert sent[-3:] == [
        ConductorMessageType.NEW_REQUEST, ConductorMessageType.ABORT_REQUEST, ConductorMessageType.READS_DONE,
    ]

    r.submit("X")
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()

    r.api.new_result_tokens(["X"], [7], "text_inputs", "text")
    assert r.worker.out_queue.get(timeout=2).data == b"<7>"
    acked = len(r.reads_done("X"))
    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == acked + 1


def test_release_queued_just_before_a_retry_does_not_drain_it(rig):
    """A's release is queued after the loop's release stage, and a retry B under
    the id right after it, both before the same pass takes B from in_queue."""
    r = rig()
    r.submit("X")
    r.fail_in_engine("X")

    def release_then_retry():
        r.api.finished_reading("X", drained=False)
        r.api.new_request(_input("X"))

    r.tm.on_ready = release_then_retry
    r.settle()
    assert len(r.conductor.sent_of(ConductorMessageType.NEW_REQUEST, "X")) == 2
    assert "X" not in r.worker._draining_rids

    r.api.new_result_tokens(["X"], [7], "text_inputs", "text")
    assert r.worker.out_queue.get(timeout=2).data == b"<7>"
    acked = len(r.reads_done("X"))
    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == acked + 1


def test_two_late_removals_do_not_swallow_a_retry(rig):
    """A finishes and B is rejected, both released, and a third request C under
    the id starts before either REMOVE_REQUEST lands. C must still ACK."""
    r = rig()
    r.submit("X")
    r.api.finished_reading("X", drained=True)
    r.settle()
    r.submit("X")
    r.api.finished_reading("X", drained=False)
    r.settle()
    r.submit("X")
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()

    r.api.new_result_tokens(["X"], [7], "text_inputs", "text")
    assert r.worker.out_queue.get(timeout=2).data == b"<7>"
    acked = len(r.reads_done("X"))
    r.api.finished_reading("X", drained=True)
    r.settle()
    assert len(r.reads_done("X")) == acked + 1

    r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("X"))
    r.settle()
    r.assert_forgotten("X")


def test_a_backlog_of_late_releases_leaves_nothing(rig):
    """Every removal lands before any of the releases."""
    r = rig()
    rids = [f"r{i}" for i in range(200)]
    for rid in rids:
        r.api.new_request(_input(rid))
    for _ in rids:
        r.settle()
    for rid in rids:
        r.conductor.deliver(WorkerMessageType.DRAIN_REQUEST, DrainRequest(rid))
    r.settle()
    for rid in rids:
        r.conductor.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest(rid))
    r.settle()
    for rid in rids:
        r.api.finished_reading(rid, drained=False)
    r.settle()

    assert not r.worker._draining_rids
    assert not r.worker._reads_done_sent
    assert not r.worker._live_requests
    assert all(len(r.reads_done(rid)) == 1 for rid in rids)
