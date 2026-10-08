"""The set-up heap freeze and the data worker's per-frame wake-up."""
import gc

from mstar.api_server.data_worker import PreprocessWorker
from mstar.utils.gc_freeze import freeze_after_setup


def test_freeze_after_setup_moves_the_heap_and_honours_the_knob(monkeypatch):
    gc.unfreeze()
    monkeypatch.setenv("MSTAR_GC_FREEZE", "0")
    assert freeze_after_setup("test") == 0 and gc.get_freeze_count() == 0
    monkeypatch.setenv("MSTAR_GC_FREEZE", "1")
    keep = [[i] for i in range(100)]  # live objects the collector tracks
    frozen = freeze_after_setup("test")
    try:
        assert frozen > 0 and gc.get_freeze_count() == frozen
        assert keep[0] == [0]  # frozen objects are still alive and usable
    finally:
        gc.unfreeze()
        assert gc.get_freeze_count() == 0


def test_a_frame_of_tokens_wakes_the_thread_once():
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.output_loop_idxs = {"r1": {}, "r2": {}}
    facade.per_request_reading_tensors = {"r1": 0, "r2": 0}

    class _Q:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    facade.result_tensor_input_queue = _Q()
    signals = []
    facade._signal = lambda: signals.append(1)
    facade.new_result_token("r1", 5, None, "text_inputs", "text", wake=False)
    facade.new_result_token("r2", 6, None, "text_inputs", "text", wake=False)
    assert len(facade.result_tensor_input_queue.items) == 2 and signals == []
    facade.wake()
    assert signals == [1]
    facade.new_result_token("r1", 7, None, "text_inputs", "text")
    assert signals == [1, 1]
