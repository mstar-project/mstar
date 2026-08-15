"""A Loop whose inner node's inputs are ALL streaming (no loop-back) must have
that node re-armed into ``ready_for_streaming`` on every iteration.

Loop-back nodes are re-armed implicitly by their re-injected inputs; a
pure-streaming node has nothing to re-inject, so without an explicit re-arm it
falls out of ``ready_for_streaming`` after the first chunk and the loop stalls
(the Nemotron-Duplex talker / codec stream-consuming loops). This exercises the
generic ``Loop._advance_one_iter`` → ``WorkerGraphStateRegistry.rearm_streaming``
path, independent of any model.
"""
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.graph_io import WorkerGraphIO


def _pure_streaming_loop() -> WorkerGraphIO:
    node = GraphNode(
        name="consumer",
        input_names={"chunk"},
        outputs=[GraphEdge(next_node="downstream", name="out")],
    )
    node._register_streaming({"chunk"})            # input_names == streaming_inputs
    # _external_inputs empty: mimic _divide_into_worker_graphs, which filters the
    # streaming input out of the loop's re-injected external inputs.
    loop = Loop(
        name="consumer_loop", section=node, max_iters=8, outputs=[],
        _external_inputs=set(), _loop_back_inputs=set(),
    )
    return WorkerGraphIO(loop)


def test_pure_streaming_loop_node_rearmed_each_iteration():
    wg = _pure_streaming_loop()

    # Starts ready-for-streaming (all inputs streaming).
    assert "consumer" in wg.ready_for_streaming

    # Consume one streamed chunk -> node becomes fully ready and is dropped from
    # the streaming-ready set.
    assert wg.ingest_input(GraphEdge(next_node="consumer", name="chunk"))
    assert "consumer" not in wg.ready_for_streaming
    assert "consumer" in wg.ready_node_names

    # Finish the iteration -> loop advances -> node re-armed for the next chunk.
    wg.mark_node_complete("consumer")
    assert "consumer" in wg.ready_for_streaming

    # And it advanced rather than finishing (max_iters not reached).
    assert wg.loops["consumer_loop"].curr_iter == 1
    assert wg.loops["consumer_loop"].is_done is False
