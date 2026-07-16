"""Walk-core parity: the Rust walk core vs mstar's ``WorkerGraphIO``, driven
with identical event sequences over the same graphs. At every step the ready
sets must match; at the end, loop iteration counts and doneness must match.
Skipped unless the ``mstar_rust`` extension is installed."""
from copy import deepcopy

import pytest

pytest.importorskip("mstar_rust")

from mstar_rust import WalkSet

from mstar.graph.base import GraphEdge, GraphNode, Loop, Parallel, Sequential
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.rust_core import walks_to_json
from mstar.graph.special_destinations import EMIT_TO_CLIENT


def node(name, inputs, outputs):
    return GraphNode(
        name=name, input_names=set(inputs),
        outputs=[GraphEdge(next_node=d, name=n) for n, d in outputs],
    )


class Harness:
    """Drives both implementations in lockstep and asserts parity."""

    def __init__(self, section, seeds):
        self.py = WorkerGraphIO(deepcopy(section), wg_id="parity")
        rust_set = WalkSet.from_json(walks_to_json({"walk": section}))
        self.rs = rust_set.state("walk")
        for node_name, input_name in seeds:
            assert self.py.ingest_input(
                GraphEdge(next_node=node_name, name=input_name))
        self.rs.seed(seeds)
        self.assert_parity()

    def assert_parity(self):
        assert sorted(self.py.ready_node_names) == sorted(
            self.rs.ready_nodes()), "ready sets diverged"
        assert self.py.wg_state_registry.is_done == self.rs.is_done(), \
            "doneness diverged"
        assert dict(self.rs.loop_iters()) == self.py.get_loop_indices(), \
            "loop iteration counts diverged"

    def run_node(self, name):
        """Pop + execute + complete `name` in both; route Python's internal
        edges back (the Rust core routes internally)."""
        assert name in self.py.ready_node_names
        self.py.ready_node_names.discard(name)
        completion = self.py.mark_node_complete(name)
        for edge in completion.output_edges:
            if edge.next_node in self.py.nodes:
                self.py.ingest_input(edge)

        self.rs.schedule(name)
        self.rs.complete(name)
        self.assert_parity()

    def stop_loop(self, loop_name):
        self.py.register_loop_finish_signal(loop_name)
        self.rs.signal_loop_finish(loop_name)

    def run_until_done(self, max_steps=200):
        steps = 0
        while not self.rs.is_done():
            ready = sorted(self.rs.ready_nodes())
            assert ready, "stuck: nothing ready but not done"
            self.run_node(ready[0])
            steps += 1
            assert steps < max_steps, "walk did not terminate"

    def loop_parity(self):
        assert dict(self.rs.loop_iters()) == self.py.get_loop_indices(), \
            "loop iteration counts diverged"


def test_sequential_chain():
    section = Sequential([
        node("A", ["x"], [("h", "B")]),
        node("B", ["h"], [("h2", "C")]),
        node("C", ["h2"], [("out", EMIT_TO_CLIENT)]),
    ])
    h = Harness(section, [("A", "x")])
    h.run_until_done()


def test_parallel_fan_out_and_join():
    section = Sequential([
        node("A", ["x"], [("l", "B"), ("r", "C")]),
        Parallel([
            node("B", ["l"], [("lb", "D")]),
            node("C", ["r"], [("rc", "D")]),
        ]),
        node("D", ["lb", "rc"], [("out", EMIT_TO_CLIENT)]),
    ])
    h = Harness(section, [("A", "x")])
    # After A completes, both branches must light up in both implementations.
    h.run_node("A")
    assert sorted(h.rs.ready_nodes()) == ["B", "C"]
    h.run_until_done()


def _loop_graph(max_iters):
    # Loop `outputs` snapshot the body's values BY NAME, so the body node
    # declares a "final" edge (destination empty = value only captured by
    # the loop). Both loop inputs are seeded: "fb" is otherwise fed only by
    # the loop-back edge, which does not exist on iteration 0.
    from mstar.graph.special_destinations import EMPTY_DESTINATION
    body = node("L", ["seed", "fb"],
                [("fb", "L"), ("tok", EMIT_TO_CLIENT),
                 ("final", EMPTY_DESTINATION)])
    loop = Loop(section=body, max_iters=max_iters, outputs=[
        GraphEdge(next_node="post", name="final")], name="dec")
    return Sequential([loop, node("post", ["final"],
                                  [("out", EMIT_TO_CLIENT)])])


def test_loop_runs_to_max_iters():
    h = Harness(_loop_graph(4), [("L", "seed"), ("L", "fb")])
    h.run_until_done()
    h.loop_parity()


def test_loop_early_stop_signal():
    h = Harness(_loop_graph(50), [("L", "seed"), ("L", "fb")])
    h.run_node("L")
    h.run_node("L")
    h.stop_loop("dec")
    h.run_until_done()
    h.loop_parity()


def test_nested_loops():
    from mstar.graph.special_destinations import EMPTY_DESTINATION
    # inner: I runs 2 iters, its "chunk" snapshot feeds O; O restarts the
    # inner loop for the next outer iteration and produces "done", which the
    # outer loop snapshots for post.
    inner = Loop(
        section=node("I", ["iseed", "ifb"],
                     [("ifb", "I"), ("chunk", EMPTY_DESTINATION)]),
        max_iters=2, outputs=[GraphEdge(next_node="O", name="chunk")],
        name="inner")
    outer_body = Sequential([
        inner,
        node("O", ["chunk"],
             [("iseed", "I"), ("ifb", "I"), ("done", EMPTY_DESTINATION)]),
    ])
    outer = Loop(
        section=outer_body, max_iters=3,
        outputs=[GraphEdge(next_node="post", name="done")], name="outer")
    section = Sequential([
        outer,
        node("post", ["done"], [("out", EMIT_TO_CLIENT)]),
    ])
    h = Harness(section, [("I", "iseed"), ("I", "ifb")])
    h.run_until_done()
    h.loop_parity()


def test_shadow_mode_mirrors_and_detects(monkeypatch, caplog):
    """MSTAR_RUST_WALK=shadow: the wrapper mirrors real WorkerGraphIO events
    with no divergence on a healthy run, and STRICT mode raises when the
    states are forced apart."""
    import logging

    from mstar.graph.rust_core import wrap_worker_graph_io

    monkeypatch.setenv("MSTAR_RUST_WALK", "shadow")
    section = _loop_graph(3)
    io = wrap_worker_graph_io(
        WorkerGraphIO(deepcopy(section), wg_id="wg"), section, "wg")

    def drive(io):
        io.ingest_input(GraphEdge(next_node="L", name="seed"))
        io.ingest_input(GraphEdge(next_node="L", name="fb"))
        while not io.wg_state_registry.is_done:
            name = sorted(io.ready_node_names)[0]
            io.ready_node_names.discard(name)
            completion = io.mark_node_complete(name)
            for edge in completion.output_edges:
                if edge.next_node in io.nodes:
                    io.ingest_input(edge)

    with caplog.at_level(logging.ERROR):
        drive(io)
    assert io._suspended is None, io._suspended
    assert not [r for r in caplog.records if "divergence" in r.message]

    # Fault injection: desync the Rust state -> strict mode must raise.
    monkeypatch.setenv("MSTAR_RUST_WALK_STRICT", "1")
    io2 = wrap_worker_graph_io(
        WorkerGraphIO(deepcopy(section), wg_id="wg2"), section, "wg2")
    io2.ingest_input(GraphEdge(next_node="L", name="seed"))
    io2.ingest_input(GraphEdge(next_node="L", name="fb"))
    io2._rs.signal_loop_finish("dec")  # rust-only event = forced divergence
    io2.ready_node_names.discard("L")
    with pytest.raises(AssertionError, match="divergence"):
        # The check fires at the settle point (after the completion's
        # locally-destined edges are re-ingested), so drive a full step.
        completion = io2.mark_node_complete("L")
        for edge in completion.output_edges:
            if edge.next_node in io2.nodes:
                io2.ingest_input(edge)


def test_authority_mode_rust_drives(monkeypatch):
    """MSTAR_RUST_WALK=1: ready set / doneness / loop indices come from the
    Rust state; a full loop walk completes, and a forced divergence falls
    back to Python instead of breaking the request."""
    from mstar.graph.rust_core import wrap_worker_graph_io

    monkeypatch.setenv("MSTAR_RUST_WALK", "1")
    section = _loop_graph(3)
    io = wrap_worker_graph_io(
        WorkerGraphIO(deepcopy(section), wg_id="wg"), section, "wg")
    io.ingest_input(GraphEdge(next_node="L", name="seed"))
    io.ingest_input(GraphEdge(next_node="L", name="fb"))
    steps = 0
    while not io.wg_state_registry.is_done:
        name = sorted(io.ready_node_names)[0]
        io.ready_node_names.discard(name)  # -> rust schedule
        completion = io.mark_node_complete(name)
        for edge in completion.output_edges:
            if edge.next_node in io.nodes:
                io.ingest_input(edge)
        steps += 1
        assert steps < 20
    assert io._suspended is None
    assert io.get_loop_indices() == {"dec": 2}

    # Divergence -> per-request fallback to Python, request still completes.
    io2 = wrap_worker_graph_io(
        WorkerGraphIO(deepcopy(section), wg_id="wg2"), section, "wg2")
    io2.ingest_input(GraphEdge(next_node="L", name="seed"))
    io2.ingest_input(GraphEdge(next_node="L", name="fb"))
    io2._rs.signal_loop_finish("dec")  # force desync
    steps = 0
    while not io2.wg_state_registry.is_done:
        name = sorted(io2.ready_node_names)[0]
        io2.ready_node_names.discard(name)
        completion = io2.mark_node_complete(name)
        for edge in completion.output_edges:
            if edge.next_node in io2.nodes:
                io2.ingest_input(edge)
        steps += 1
        assert steps < 20
    assert io2._suspended is not None  # fell back, and the walk finished
