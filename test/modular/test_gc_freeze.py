"""The set-up heap freeze and the data worker's per-frame wake-up."""
import gc

from mstar.api_server.data_worker import PreprocessWorker
from mstar.utils.gc_freeze import freeze_after_setup


def test_freeze_after_setup_moves_the_heap():
    gc.unfreeze()
    keep = [[i] for i in range(100)]  # live objects the collector tracks
    keep_ids = {id(obj) for obj in keep}
    try:
        frozen = freeze_after_setup("test")
        assert frozen > 0
        # Refcounting can free other frozen objects, so the total can shrink.
        # Check our retained objects moved out of the collected generations.
        assert keep_ids.isdisjoint(map(id, gc.get_objects()))
        assert keep[0] == [0]  # frozen objects are still alive and usable
    finally:
        gc.unfreeze()
        assert gc.get_freeze_count() == 0
    assert keep_ids <= {id(obj) for obj in gc.get_objects()}


def test_a_frame_of_tokens_wakes_the_thread_once():
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.received_chunk_counts = {"r1": {}, "r2": {}}
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
