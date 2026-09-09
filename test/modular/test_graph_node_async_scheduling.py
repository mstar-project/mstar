"""``GraphNode.enable_async_scheduling`` — the opt-out from speculation.

This used to assert the flag survived ``clone_for_next_iter``. Nodes are no
longer cloned per iteration (``ready_next_iter`` is a buffer on the node
itself), so that API is gone and the flag trivially persists on the one live
object. What still matters is the thing the flag exists for: the worker
refuses to build a speculative batch for a node that opted out
(``Worker._try_speculate_next``, which filters on
``wgio.nodes[info.node_name].enable_async_scheduling``).

The graph layer itself does NOT apply that filter — it reports readiness and
the worker decides — so these tests pin both halves: the flag survives on the
node the worker reads it from, and speculation readiness is computed
independently of it.
"""
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.graph_io import WorkerGraphIO


def _ar_graph(enable_async_scheduling: bool):
    return Sequential(sections=[
        GraphNode(
            name="prefill",
            input_names={"prompt"},
            outputs=[GraphEdge(name="token", next_node="ar_decode")],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode",
                input_names={"token"},
                outputs=[GraphEdge(name="token", next_node="ar_decode")],
                enable_async_scheduling=enable_async_scheduling,
            ),
            outputs=[],
            max_iters=5,
        ),
    ])


def test_flag_is_readable_off_the_live_node():
    """The worker reads the flag off ``wgio.nodes[name]``; it must survive
    graph construction and stay put across iterations."""
    io = WorkerGraphIO(_ar_graph(enable_async_scheduling=False))
    node = io.nodes["ar_decode"]
    assert node.enable_async_scheduling is False

    io.ingest_for_speculation(
        [GraphEdge(name="token", next_node="ar_decode")], "ar_decode"
    )
    io.clear_speculative_inputs()
    assert io.nodes["ar_decode"].enable_async_scheduling is False


def test_defaults_to_enabled():
    io = WorkerGraphIO(_ar_graph(enable_async_scheduling=True))
    assert io.nodes["ar_decode"].enable_async_scheduling is True
    assert io.nodes["prefill"].enable_async_scheduling is True


def test_graph_layer_reports_readiness_regardless_of_the_flag():
    """Opting out does not change what the graph layer reports — the filter
    lives in the worker. If the graph layer ever started applying it itself,
    the worker's filter would silently become dead code.
    """
    ready_by_flag = {}
    for flag in (True, False):
        io = WorkerGraphIO(_ar_graph(enable_async_scheduling=flag))
        ready = io.ingest_for_speculation(
            [GraphEdge(name="token", next_node="ar_decode")], "ar_decode"
        )
        ready_by_flag[flag] = [info.node_name for info in ready]

    assert ready_by_flag[True] == ["ar_decode"]
    assert ready_by_flag[False] == ["ar_decode"]
