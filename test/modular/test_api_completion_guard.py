"""The API server must not treat a request with no reported outputs as
fully delivered, and a request that fails preprocessing must reach the
client as an error instead of leaving it to hit the server timeout.

An empty final_outputs dict made received_final_chunks vacuously true, so a
request whose walk emitted nothing (or whose completion message raced ahead
of every result) completed instantly with zero chunks instead of holding for
late results.
"""

import queue
import threading
from types import SimpleNamespace

from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.graph.loop_indices import NestedLoopIndices


def _worker_with(output_loop_idxs):
    worker = PreprocessWorker.__new__(PreprocessWorker)
    worker.output_loop_idxs = output_loop_idxs
    return worker


def test_empty_final_outputs_is_not_done():
    worker = _worker_with({"r1": {}})
    assert worker.received_final_chunks("r1", {}) is False


def test_final_outputs_wait_for_registration_then_complete():
    final = NestedLoopIndices(
        loop_name_order=["denoise"], loop_indices={"denoise": 34}, wg_fwd_pass_idx=0
    )
    worker = _worker_with({"r1": {}})
    # Completion reported but the terminal chunk hasn't registered yet.
    assert worker.received_final_chunks("r1", {"video_output": final}) is False
    worker.output_loop_idxs["r1"]["video_output"] = final
    assert worker.received_final_chunks("r1", {"video_output": final}) is True


def test_preprocess_failure_emits_error_chunk():
    """A request rejected during preprocessing (e.g. an invalid model kwarg)
    must produce an "error" chunk so the waiting client is released."""

    class _RejectingModel:
        def process_prompt(self, *args, **kwargs):
            raise ValueError("bad knob")

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.in_queue = queue.Queue()
    wt.out_queue = queue.Queue()
    wt.result_tensor_queue = queue.Queue()
    wt.cleanup_request_queue = queue.Queue()
    wt.stop_event = threading.Event()
    wt.communicator = SimpleNamespace(get_all_new_messages=lambda: [])
    cleaned = []
    wt.tensor_manager = SimpleNamespace(
        cleanup_request=cleaned.append, get_ready_tensors=lambda: {}
    )
    wt.model = _RejectingModel()
    wt.device = "cpu"
    wt.tensor_uuid_to_metadata_per_request = {}

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
