"""``StepExecutor._get_postprocess_request`` routes each walk under its own output
signals."""
from types import SimpleNamespace

from mstar.testing_utils.worker import make_worker
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.step_executor import StepExecutor

NODE, MIXED = "LLM", "LLM_MIXED"
SIGNALS = {"decode": ["text_inputs"], "prefill_text": ["text_inputs", "new_token"]}


def _executor() -> StepExecutor:
    return make_worker(
        graph_runtime=SimpleNamespace(
            get_output_signals=lambda node, walk: SIGNALS[walk], supports_inline_emit=False,
        ),
    ).executor


def _pending(request_walks: dict[int, str]):
    batch = ScheduledBatch(
        node_name=NODE, graph_walk=MIXED,
        request_to_worker_graph=dict.fromkeys(request_walks, 0),
        # a merged batch keeps its first walk's names
        output_signals=SIGNALS["decode"], request_walks=dict(request_walks),
    )
    return SimpleNamespace(
        batch=batch, node_name=NODE, graph_walk=MIXED, partition="p",
        node_batch=SimpleNamespace(final_stream_rids=set(), partition_done_rids=set()),
    )


def test_a_combined_batch_down_to_one_walk_routes_under_that_walks_signals():
    # the decode row left the batch (failed in prepare_inputs, say)
    req = _executor()._get_postprocess_request(_pending({2: "prefill_text"}), None, None, False, None, None)
    assert (req.graph_walk, req.walks) == ("prefill_text", None)
    assert req.output_signals == SIGNALS["prefill_text"]


def test_a_combined_batch_of_several_walks_routes_under_their_union():
    req = _executor()._get_postprocess_request(
        _pending({1: "decode", 2: "prefill_text"}), None, None, False, None, None,
    )
    assert req.walks == ["decode", "prefill_text"] and req.rid_walk_idx == [0, 1]
    assert req.output_signals == ["text_inputs", "new_token"]


def test_the_models_stops_ride_the_request_when_the_runtime_does_not_check():
    req = _executor()._get_postprocess_request(
        _pending({2: "prefill_text"}), None, None, False, None, {2: {"decode_loop"}},
    )
    assert not req.runtime_stops and req.stops == {2: {"decode_loop"}}
