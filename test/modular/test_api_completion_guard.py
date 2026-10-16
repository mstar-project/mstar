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
from mstar.model.base import Model


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


def _preprocess_failure(model, file_paths=None):
    """Run one request through the preprocess loop and return its error chunk
    and the rids force-cleaned."""
    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.in_queue = queue.Queue()
    wt.out_queue = queue.Queue()
    wt.result_tensor_queue = queue.Queue()
    wt.cleanup_request_queue = queue.Queue()
    wt.abort_request_queue = queue.Queue()
    wt.reads_done_queue = queue.Queue()
    wt.discard_tensor_queue = queue.Queue()
    wt.stop_event = threading.Event()
    wt.communicator = SimpleNamespace(
        get_all_new_messages=lambda: [], send=lambda *args: None,
    )
    cleaned = []
    wt.tensor_manager = SimpleNamespace(
        force_cleanup_request=cleaned.append,
        has_inflight_reads=lambda rid: False,
        get_ready_tensors=lambda: {},
    )
    wt.model = model
    wt.device = "cpu"
    wt.tensor_uuid_to_metadata_per_request = {}
    wt.request_model_kwargs = {}
    wt.in_flight_requests = set()
    wt._draining_rids = set()
    wt._reads_done_sent = set()

    wt.in_queue.put(PreprocessInput(
        request_id="r1", text="x", file_paths=file_paths,
        input_modalities=[*(file_paths or {}), "text"], output_modalities=["video"],
        model_kwargs={},
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
    return chunk, cleaned


def test_preprocess_failure_emits_error_chunk():
    """A request rejected during preprocessing (e.g. an invalid model kwarg)
    must produce an "error" chunk so the waiting client is released."""

    class _RejectingModel:
        def process_prompt(self, *args, **kwargs):
            raise ValueError("bad knob")

    chunk, cleaned = _preprocess_failure(_RejectingModel())
    assert b"bad knob" in chunk.data
    assert chunk.metadata["status"] == 400
    assert cleaned == ["r1"]


class _ImageModel:
    def load_image(self, filepath, device):
        return Model.load_image(self, filepath, device)


def test_an_upload_that_will_not_decode_is_a_400(tmp_path):
    """torchvision raises RuntimeError on a corrupt image, which used to be a 500."""
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    chunk, cleaned = _preprocess_failure(_ImageModel(), {"image": [str(bad)]})
    assert b"could not decode the image input" in chunk.data
    assert chunk.metadata["status"] == 400
    assert cleaned == ["r1"]


def test_an_upload_our_cleanup_removed_is_a_500(tmp_path):
    chunk, _ = _preprocess_failure(_ImageModel(), {"image": [str(tmp_path / "gone.png")]})
    assert chunk.metadata["status"] == 500


def test_a_decoder_that_fails_to_import_is_a_500(tmp_path, monkeypatch):
    """torchcodec on a host without FFmpeg raises RuntimeError at import."""
    (tmp_path / "broken_decoder.py").write_text('raise RuntimeError("could not load libdecoder")\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    class _AudioModel:
        def load_audio(self, filepath, device):
            import broken_decoder  # noqa: F401

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    chunk, _ = _preprocess_failure(_AudioModel(), {"audio": [str(clip)]})
    assert b"could not load libdecoder" in chunk.data
    assert chunk.metadata["status"] == 500


def test_a_bug_in_a_loader_is_a_500(tmp_path):
    class _BuggyModel:
        def load_video(self, filepath, device):
            raise KeyError("average_fps")

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    chunk, _ = _preprocess_failure(_BuggyModel(), {"video": [str(clip)]})
    assert chunk.metadata["status"] == 500
