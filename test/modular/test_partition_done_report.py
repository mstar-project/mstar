"""``mark_stream_partition_done`` reaches the conductor on WORKER_GRAPHS_DONE.

The flag the worker sets for a finished stream or an ``END_PARTITION`` stop
rides the next report the runtime builds, unless the node is in flight again
(speculatively scheduled), in which case it is held for the next real one.

Drives the real Python graph runtime through two prefill passes (the
``test_delivery_chunk_counts`` graph); ``test/rust/test_partition_done_report_rust.py``
runs the same script on the Rust runtime.
"""
import pytest

torch = pytest.importorskip("torch")

from test_delivery_chunk_counts import NODE, PREFILL, WORKER, python_runtime  # noqa: E402

from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SendInput  # noqa: E402
from mstar.utils.containers import ParallelList  # noqa: E402


def drive(runtime, put, speculative: bool) -> None:
    """Two prefill passes; the partition is marked done during the first,
    whose node is in flight again (a speculative successor) when
    ``speculative``."""
    rid = runtime.add_request(
        request_id="r1", partition="default", graph_walk=PREFILL,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
    )
    uuid = 100
    for first in (True, False):
        runtime.set_walk(rid, "default", PREFILL)
        block = ColumnarEdgeSpecs.empty()
        block.add_edge(rid, GraphEdge(name="text_inputs", next_node=NODE))
        assert runtime.ingest_inputs_batch(block, can_buffer=True) == []
        assert runtime.pop_rids(NODE, PREFILL, [rid]) is not None
        signals = runtime.get_output_signals(NODE, PREFILL)
        uuids = []
        for _ in signals:
            uuid += 1
            put(uuid)
            uuids.append(uuid)
        speculated = first and speculative
        if speculated:  # before the route: Rust reads it there, Python at the send
            runtime.set_in_flight(NODE, [rid], [0], True)
        out = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk=PREFILL, node_name=NODE, output_signals=signals,
            wg_ids=ParallelList([rid], [0]), tensors=uuids, num_tensors=[1] * len(signals),
        ))
        if first:
            runtime.mark_stream_partition_done(rid, "default")
        runtime.send_outputs(SendInput(
            completion_id=out.completion_id, per_request_info=ParallelList([rid], [None]),
            new_token_counts=ParallelList([rid], [{}]),
        ))
        if speculated:
            runtime.set_in_flight(NODE, [rid], [0], False)


# per pass: a speculative completion holds the flag back, and once reported it
# stays set (harmless, the conductor ends the partition on the first)
EXPECTED = {False: [True, True], True: [False, True]}


@pytest.mark.parametrize("speculative", [False, True], ids=["real", "speculative"])
def test_partition_done_rides_the_next_real_report(speculative):
    runtime, put, recorder = python_runtime(decode_steps=1)
    drive(runtime, put, speculative)
    assert [r.partition_done for r in recorder.reports()] == EXPECTED[speculative]
