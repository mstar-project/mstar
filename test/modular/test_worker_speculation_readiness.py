"""A speculative step continues a request onto another node only once that node
would run it, as the scheduler asks before it batches one.

Otherwise the request reaches admit without having asked, a KV pool that holds
it back refuses the whole batch, and every request in it waits a round. Left
out, it takes the queue, where readiness holds it like any other.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.worker.worker import Worker


class _Runtime:
    """The graph runtime as speculation sees it: a target, and the rids prepped."""

    def __init__(self, target: str):
        self.target = target
        self.prepped: list[int] | None = None

    def speculate_node(self, node_name, graph_walk, rid):
        del node_name, graph_walk, rid
        return [SimpleNamespace(node_name=self.target)]

    def prep_spec_rids(self, prep_input):
        self.prepped = list(prep_input.rids)
        return SimpleNamespace(
            consumed_streaming_edge_idxs=[], ready_rids=[], wg_ids=[],
            input_edges=SimpleNamespace(
                to_input_tensors=lambda get_tensor, rids: SimpleNamespace(by_rid={}),
            ),
        )


def _worker(target: str, not_ready: set[int]) -> tuple[Worker, _Runtime]:
    def check_ready(node_name, rid, request_info):
        del node_name, request_info
        return SimpleNamespace(ok=True, ready=rid not in not_ready)

    worker = Worker.__new__(Worker)
    worker._graph_runtime = runtime = _Runtime(target)
    worker._pending_removes = set()
    worker._poll_stream_buffers_for_speculation = lambda rid, node_name: []
    worker.scheduler = SimpleNamespace(room_for_continuing=lambda target: None)
    worker.tensor_manager = SimpleNamespace(get_tensor=None)
    worker.request_state = SimpleNamespace(
        get_partition_for_node=lambda node_name: 0,
        get_fwd_info=lambda rid, partition: None,
    )
    worker.engine_manager = SimpleNamespace(
        get_engine=lambda node_name: SimpleNamespace(check_ready=check_ready),
    )
    return worker, runtime


def _pending(node_name: str, rids: list[int]):
    return SimpleNamespace(
        batch=SimpleNamespace(
            node_name=node_name, graph_walk="walk",
            request_to_worker_graph={rid: rid for rid in rids},
        ),
        graph_walk="walk",
    )


def test_a_request_the_next_node_holds_back_is_not_continued_onto_it():
    worker, runtime = _worker("decoder", not_ready={2})

    worker._try_speculate_next(_pending("audio_encoder", [1, 2, 3]))

    assert runtime.prepped == [1, 3], (
        "a request the decoder would not run was carried onto it anyway"
    )

