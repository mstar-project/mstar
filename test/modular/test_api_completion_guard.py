"""The API server must not treat a request with no reported outputs as
fully delivered, and a request that fails preprocessing must reach the
client as an error instead of leaving it to hit the server timeout.

An empty output_chunk_counts dict made received_final_chunks vacuously true, so a
request whose walk emitted nothing (or whose completion message raced ahead
of every result) completed instantly with zero chunks instead of holding for
late results.
"""

import queue
import threading
from types import SimpleNamespace

from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput


def _worker_with(received_chunk_counts):
    worker = PreprocessWorker.__new__(PreprocessWorker)
    worker.received_chunk_counts = received_chunk_counts
    return worker


def test_empty_output_chunk_counts_is_not_done():
    worker = _worker_with({"r1": {}})
    assert worker.received_final_chunks("r1", {}) is False


def test_output_chunk_counts_wait_for_registration_then_complete():
    worker = _worker_with({"r1": {}})
    # Completion reported but the terminal chunk hasn't registered yet.
    assert worker.received_final_chunks("r1", {"video_output": 1}) is False
    worker.received_chunk_counts["r1"]["video_output"] = 1
    assert worker.received_final_chunks("r1", {"video_output": 1}) is True


def test_preprocess_failure_emits_error_chunk():
    """A request rejected during preprocessing (e.g. an invalid model kwarg)
    must produce an "error" chunk so the waiting client is released."""

    class _RejectingModel:
        def process_prompt(self, *args, **kwargs):
            raise ValueError("bad knob")

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.in_queue = queue.Queue()
    wt._is_pending = lambda key: True
    wt.out_queue = queue.Queue()
    wt.result_tensor_queue = queue.Queue()
    wt.cleanup_request_queue = queue.Queue()
    wt.abort_request_queue = queue.Queue()
    wt.reads_done_queue = queue.Queue()
    wt.discard_tensor_queue = queue.Queue()
    wt.stop_event = threading.Event()
    wt.timing_requested = threading.Event()
    wt.communicator = SimpleNamespace(
        get_all_new_messages=lambda: [], send=lambda *args: None,
    )
    cleaned = []
    wt.tensor_manager = SimpleNamespace(
        force_cleanup_request=cleaned.append,
        has_inflight_reads=lambda rid: False,
        get_ready_tensors=lambda: {},
        take_failed_reads=lambda: [],
    )
    wt.model = _RejectingModel()
    wt.device = "cpu"
    wt.tensor_uuid_to_metadata_per_request = {}
    wt.request_model_kwargs = {}
    wt.in_flight_requests = set()
    wt._draining_rids = set()
    wt._reads_done_sent = set()
    wt._live_requests = {}
    wt._keys_shut = False

    wt.in_queue.put(PreprocessInput(
        request_id="r1", text="x", file_paths=None,
        input_modalities=["text"], output_modalities=["video"], model_kwargs={},
    ))
    thread = threading.Thread(target=wt.run, daemon=True)
    thread.start()
    try:
        chunk = wt.out_queue.get(timeout=10)
    finally:
        wt.stop_event.set()
        thread.join(timeout=10)

    assert chunk.request_id == "r1"
    assert chunk.modality == "error"
    assert b"bad knob" in chunk.data
    assert chunk.metadata["status"] == 400
    assert cleaned == ["r1"]
