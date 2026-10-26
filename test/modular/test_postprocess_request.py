"""``Worker._get_postprocess_request`` routes each walk under its own output
signals."""
from types import SimpleNamespace

from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

NODE, MIXED = "LLM", "LLM_MIXED"
SIGNALS = {"decode": ["text_inputs"], "prefill_text": ["text_inputs", "new_token"]}


def _worker() -> Worker:
    w = Worker.__new__(Worker)
    w._graph_runtime = SimpleNamespace(
        get_output_signals=lambda node, walk: SIGNALS[walk], supports_inline_emit=False,
    )
    w._stop_token_signal = {}
    w._innermost_loop_by_node = {}
    w._device_loopback = {}
    w.enable_prof = False
    return w


def _pending(request_walks: dict[int, str]):
    batch = ScheduledBatch(
        node_name=NODE, graph_walk=MIXED,
        request_to_worker_graph=dict.fromkeys(request_walks, 0),
        # a merged batch keeps its first walk's names
        output_signals=SIGNALS["decode"], request_walks=dict(request_walks),
    )
    return SimpleNamespace(
        batch=batch, node_name=NODE, graph_walk=MIXED, partition="p",
        node_batch=SimpleNamespace(final_stream_rids=set(), stream_partition_done_rids=set()),
    )


def test_a_combined_batch_down_to_one_walk_routes_under_that_walks_signals():
    # the decode row left the batch (failed in prepare_inputs, say)
    req = _worker()._get_postprocess_request(_pending({2: "prefill_text"}), None, engine=None)
    assert (req.graph_walk, req.walks) == ("prefill_text", None)
    assert req.output_signals == SIGNALS["prefill_text"]


def test_a_combined_batch_of_several_walks_routes_under_their_union():
    req = _worker()._get_postprocess_request(
        _pending({1: "decode", 2: "prefill_text"}), None, engine=None,
    )
    assert req.walks == ["decode", "prefill_text"] and req.rid_walk_idx == [0, 1]
    assert req.output_signals == ["text_inputs", "new_token"]
