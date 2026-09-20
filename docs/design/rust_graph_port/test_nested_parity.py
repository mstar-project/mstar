"""Nested-loop parity: real Python WorkerGraphIO vs the Rust prototype.

Graph: Loop_outer[ pre -> Loop_inner[ step ] -> post ]  (see graph_fixture).
Drives both through the same event sequence and compares, after every node
completion: ready set, every loop's curr_iter, emitted-to-client count,
doneness, and num_times_run.

PYTHONPATH=docs/design/rust_graph_port/proto \
  python docs/design/rust_graph_port/test_nested_parity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "proto"))

from graph_fixture import _tinfo, make_nested_section
from mstar_graph_proto import GraphRuntime
from rust_spec import to_rust_spec

from mstar.graph.base import GraphEdge
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import SPECIAL_DESTINATIONS

OUTER, INNER = 3, 2


class Driver:
    """Runs the Python side; mirrors what the worker does around a node."""

    def __init__(self, section, auto_clear=True):
        self.io = WorkerGraphIO(section, wg_id="wg0")
        self.emits = 0
        self.completed = 0
        self.auto_clear = auto_clear

    def ingest(self, dest, name, uuid):
        self.io.ingest_input(GraphEdge(next_node=dest, name=name, tensor_info=[_tinfo(uuid)]))

    def complete(self, node_name, uuid_prefix):
        self.io.ready_node_names.discard(node_name)   # scheduler: pop_ready_nodes
        node = self.io.nodes[node_name]
        for i, e in enumerate(node.outputs):          # worker.py: store outputs
            e.tensor_info = [_tinfo(f"{uuid_prefix}-{i}")]
        out = self.io.mark_node_complete(node_name)
        for e in out.output_edges:                    # worker.py: route them
            if e.next_node in SPECIAL_DESTINATIONS:
                self.emits += 1
            elif e.next_node in self.io.nodes:
                self.io.ingest_input(e)
        if self.io.wg_state_registry.is_done:         # process_node_outputs: reset
            self.completed += 1
            if self.auto_clear:
                self.io.clear()
        return out

    def state(self):
        return (
            sorted(self.io.ready_node_names),
            dict(self.io.get_loop_indices()),
            self.emits,
            self.completed,
            self.io.num_times_run,
        )


class RsDriver:
    def __init__(self, section):
        self.rt = GraphRuntime(**to_rust_spec(section), workers=["w0"], me="w0")
        (self.h,) = self.rt.add_requests(["r0"])
        self.emits = 0
        self.completed = 0
        self._n_out = {n["name"]: len(n["outputs"]) for n in to_rust_spec(section)["nodes"]}

    def ingest(self, dest, name, uuid):
        self.rt.ingest_batch([self.h], dest, name, [hash(uuid) & 0xFFFFFFFF])

    def complete(self, node_name, uuid_prefix):
        self.rt.pop_ready(node_name, [self.h])        # scheduler: pop_ready_nodes
        n = self._n_out[node_name]
        uuids = [(hash(f"{uuid_prefix}-{i}") & 0xFFFFFFFF) for i in range(n)]
        r = self.rt.complete_and_route_batch(node_name, [self.h], uuids, [1] * n)
        self.emits += len(r.emit)
        self.completed += len(r.completed)
        return r

    def state(self):
        return (
            sorted(self.rt.ready_nodes(self.h)),
            dict(self.rt.loop_indices(self.h)),
            self.emits,
            self.completed,
            self.rt.num_times_run(self.h),
        )


def outer_last_entity_section():
    """Loop_outer[ pre -> Loop_inner[step] ] — the inner loop is the LAST
    entity of the outer body, and `acc` is a declared output of BOTH loops.
    Isolates the two Python nested-loop ordering bugs (README 7.5)."""
    from mstar.graph.base import GraphNode, Loop, Sequential
    step = GraphNode(name="step", input_names={"seed", "acc"},
                     outputs=[GraphEdge(next_node="step", name="acc")])
    inner = Loop(section=step, max_iters=2, name="inner",
                 outputs=[GraphEdge(next_node="sink", name="acc")])
    pre = GraphNode(name="pre", input_names={"x"},
                    outputs=[GraphEdge(next_node="step", name="seed"),
                             GraphEdge(next_node="step", name="acc")])
    return Loop(section=Sequential([pre, inner]), max_iters=1, name="outer",
                outputs=[GraphEdge(next_node="sink", name="acc")])


def check_known_divergence():
    """Documents where the prototype deliberately differs from Python.

    Python's `Loop.complete_iter` cascades into its parent BEFORE populating
    its own outputs' tensor_info, and DISCARDS the cascade's return value. So
    when an inner loop's termination also terminates the outer loop:
      (a) the outer caches the inner's still-empty tensor_info, keeping
          whatever stale value an earlier entity left under that name, and
      (b) the outer's declared output edges are never routed at all.
    """
    py = Driver(outer_last_entity_section(), auto_clear=False)
    py.ingest("pre", "x", "x0")
    for node, tag in (("pre", "p"), ("step", "s0"), ("step", "s1")):
        out = py.complete(node, tag)

    outer = py.io.loops["outer"]
    routed_names = {(e.name, e.next_node) for e in out.output_edges}
    outer_val = [t.uuid for t in outer.outputs[0].tensor_info]

    assert outer.is_done and py.io.wg_state_registry.is_done
    # (b) the outer loop's own output never reaches the router
    assert ("acc", "sink") in routed_names  # this one is the INNER loop's
    assert len([e for e in out.output_edges if e.name == "acc" and e.next_node == "sink"]) == 1
    # (a) and the value it holds came from `pre`, not from the inner loop
    assert outer_val == ["p-1"], outer_val  # pre's output edge #1, the `acc` one
    print("known Python divergence reproduced:")
    print(f"    outer loop declared output 'acc' -> sink carries {outer_val} "
          f"(pre's intermediate), not the inner loop's final value")
    print("    and is absent from the routed edges; wg reports done regardless")
    print("    prototype instead populates-then-cascades and appends outer's outputs")


def main():
    py = Driver(make_nested_section(OUTER, INNER))
    rs = RsDriver(make_nested_section(OUTER, INNER))

    def check(label):
        p, r = py.state(), rs.state()
        assert p == r, f"\n  at {label}\n  py={p}\n  rs={r}"
        return p

    py.ingest("pre", "x", "x0")
    rs.ingest("pre", "x", "x0")
    check("seed")

    steps = 0
    for o in range(OUTER):
        py.complete("pre", f"pre{o}")
        rs.complete("pre", f"pre{o}")
        check(f"outer{o}/pre")
        for i in range(INNER):
            py.complete("step", f"step{o}{i}")
            rs.complete("step", f"step{o}{i}")
            st = check(f"outer{o}/inner{i}/step")
            steps += 1
        py.complete("post", f"post{o}")
        rs.complete("post", f"post{o}")
        st = check(f"outer{o}/post")

    ready, idx, emits, ncomp, runs = st
    print(f"nested parity OK: {OUTER} outer x {INNER} inner iterations, {steps} step completions")
    print(f"  final: loop_indices={idx} emits={emits} completed={ncomp} num_times_run={runs}")

    # nested loop indices: outer -> inner chain, and a node outside any loop
    nl = rs.rt.nested_loop_idxs_for_node(rs.h, "step")
    print(f"  nested_loop_idxs_for_node('step'): order={nl.loop_name_order} "
          f"indices={dict(nl.loop_indices)} wg_fwd_pass_idx={nl.wg_fwd_pass_idx}")
    assert nl.loop_name_order == ["outer", "inner"], nl.loop_name_order

    print()
    check_known_divergence()


if __name__ == "__main__":
    main()
