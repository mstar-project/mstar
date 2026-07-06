"""The API server must not treat a request with no reported outputs as
fully delivered.

An empty final_outputs dict made received_final_chunks vacuously true, so a
request whose walk emitted nothing (or whose completion message raced ahead
of every result) completed instantly with zero chunks instead of holding for
late results.
"""

from mstar.api_server.data_worker import PreprocessWorker
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
