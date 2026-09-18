"""WorkerGraphIO: node readiness, edge routing and loop iteration.

The generator builds the graph, and the ops drive it. A graph is a chain of
stages. Each stage holds one node, or a `Parallel` of several nodes. Each node
consumes one edge from every node of the previous stage, and each of those
edges has a different name. The generator can also put one continuous group of
stages into a `Loop`. The tail of that loop sends loop-carried edges back to
its head.

Ops: ingest, ingest_dup, complete, finish_loop, drain, clear.

Invariants
----------
graph.unrouted_edge_is_external           an edge for an unknown node is refused
graph.accept_fills_exactly_one_slot       an accepted edge fills one slot
graph.accept_does_not_disturb_other_slot  ... and leaves the other alone
graph.reject_is_inert                     a rejected edge changes no state
graph.reject_has_a_reason                 a declared input is never refused
                                          while neither slot holds it
graph.completion_respects_filter          a filtered edge is not also an output
graph.drain_terminates                    a graph always drains to done
graph.clear_empties_node_state            clear() empties every node slot
graph.clear_resets_loop                   clear() resets every loop
graph.clear_resets_registry               clear() resets the top registry
graph.ready_name_is_a_node                a ready name is a node of this graph
graph.ready_implies_all_inputs            a ready node holds all of its inputs
graph.slot_holds_only_declared_inputs     a slot holds only declared inputs
graph.slot_names_match_edges              slot names match the stored edges
graph.is_ready_matches_inputs             is_ready matches the slot contents
graph.completion_count_in_range           0 <= completed <= managed
graph.done_matches_completion_count       is_done matches the count
graph.loop_iter_within_bound              a loop stays within max_iters
graph.loop_done_has_a_cause               a finished loop had a signal or hit
                                          its last iteration

Not covered:

* what a node computes: an edge carries a name and no data
* streaming edges and `consumes_stream`
* speculative ingest
* a loop inside a loop
* a destination in another worker graph
* a `Parallel` loop body whose members feed each other:
  `Parallel.get_inputs_outputs` calls those edges internal, not loop-back

A drain returns as soon as the registry reports that it is done. Thus
`graph.drain_terminates` runs only when a drain fails to complete."""

from __future__ import annotations

import random
from collections.abc import Iterator

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier0 import _stubs  # noqa: F401
from mstar.graph.base import GraphEdge, GraphNode, Loop, Parallel, Sequential
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT

LOOP_NAME = "L"


def _node_name(stage: int, index: int) -> str:
    return f"n{stage}_{index}"


def _edge_name(stage: int, producer: int, consumer: int) -> str:
    """Name the edge from one node of a stage to one node of the next stage.

    The edge goes from node ``producer`` of the stage before ``stage``, into
    node ``consumer`` of ``stage``. Each (producer, consumer) pair gets a
    different name. A node with several producers therefore never sees the
    same input name two times.
    """
    return f"x{stage}_{producer}_{consumer}"


def gen_graph_config(rng: random.Random) -> dict:
    num_stages = rng.randint(1, 4)
    # widths[i] is the number of producers that feed stage i. widths[i + 1] is
    # the number of nodes in stage i. widths[0] counts the external sources.
    widths = [rng.randint(1, 2) for _ in range(num_stages + 1)]

    loop = None
    if rng.random() < 0.6:
        start = rng.randrange(num_stages)
        end = rng.randrange(start, num_stages)
        # A one-stage loop body must hold one node; see the module docstring.
        if start == end:
            widths[start + 1] = 1
        # Stage `start` must expect one producer per node of stage `end`.
        widths[start] = widths[end + 1]
        loop = {
            "start": start,
            "end": end,
            "max_iters": rng.randint(1, 4),
            "emit": rng.random() < 0.5,
        }

    return {"widths": widths, "loop": loop}


def build_graph(config: dict) -> tuple[Sequential | GraphNode, list[GraphEdge]]:
    """Make the graph from the config, with the external edges that start it."""
    widths: list[int] = list(config["widths"])
    loop_cfg = config.get("loop")
    num_stages = len(widths) - 1

    loop_start = loop_cfg["start"] if loop_cfg else None
    loop_end = loop_cfg["end"] if loop_cfg else None

    loopback_names: list[str] = []
    if loop_cfg:
        loopback_names = [
            _edge_name(loop_start, producer, consumer)
            for producer in range(widths[loop_start])
            for consumer in range(widths[loop_start + 1])
        ]

    def inputs_for(stage: int, index: int) -> set[str]:
        # The Loop's declared outputs feed the stage after it, under the
        # loop-carried names.
        if loop_cfg and stage == loop_end + 1:
            return {loopback_names[index % len(loopback_names)]}
        return {
            _edge_name(stage, producer, index) for producer in range(widths[stage])
        }

    def outputs_for(stage: int, index: int) -> list[GraphEdge]:
        if loop_cfg and stage == loop_end:
            # The loop tail emits loop-carried edges only; the rest leave
            # through Loop.outputs.
            return [
                GraphEdge(
                    name=_edge_name(loop_start, index, consumer),
                    next_node=_node_name(loop_start, consumer),
                )
                for consumer in range(widths[loop_start + 1])
            ]
        if stage == num_stages - 1:
            return [GraphEdge(name=f"out_{index}", next_node=EMIT_TO_CLIENT)]
        return [
            GraphEdge(
                name=_edge_name(stage + 1, index, consumer),
                next_node=_node_name(stage + 1, consumer),
            )
            for consumer in range(widths[stage + 2])
        ]

    def stage_section(stage: int):
        nodes = [
            GraphNode(
                name=_node_name(stage, index),
                input_names=inputs_for(stage, index),
                outputs=outputs_for(stage, index),
            )
            for index in range(widths[stage + 1])
        ]
        return nodes[0] if len(nodes) == 1 else Parallel(sections=nodes)

    stages = [stage_section(stage) for stage in range(num_stages)]

    if loop_cfg is None:
        sections = stages
    else:
        body = stages[loop_start:loop_end + 1]
        body_section = body[0] if len(body) == 1 else Sequential(sections=body)
        if loop_end + 1 < num_stages:
            loop_outputs = [
                GraphEdge(
                    name=loopback_names[consumer % len(loopback_names)],
                    next_node=_node_name(loop_end + 1, consumer),
                )
                for consumer in range(widths[loop_end + 2])
            ]
        elif loop_cfg["emit"]:
            loop_outputs = [
                GraphEdge(name=loopback_names[0], next_node=EMIT_TO_CLIENT)
            ]
        else:
            loop_outputs = []
        loop = Loop(
            name=LOOP_NAME,
            section=body_section,
            max_iters=loop_cfg["max_iters"],
            outputs=loop_outputs,
        )
        sections = stages[:loop_start] + [loop] + stages[loop_end + 1:]

    graph = sections[0] if len(sections) == 1 else Sequential(sections=sections)
    seed_edges = [
        GraphEdge(name=_edge_name(0, producer, consumer),
                  next_node=_node_name(0, consumer))
        for producer in range(widths[0])
        for consumer in range(widths[1])
    ]
    return graph, seed_edges


class GraphIOMachine(StateMachine):
    name = "graph_io"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        return gen_graph_config(rng)

    def __init__(self, config: dict) -> None:
        self.config = config
        graph, seed_edges = build_graph(config)
        self.io = WorkerGraphIO(graph, wg_id="wg0")
        self.seed_edges = seed_edges
        self.pending: list[GraphEdge] = list(seed_edges)
        self.ingested: list[GraphEdge] = []

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        choice = rng.random()
        if choice < 0.45:
            return Op("ingest", (rng.randrange(4),))
        if choice < 0.80:
            return Op("complete", (rng.randrange(4),))
        if choice < 0.86:
            return Op("ingest_dup", (rng.randrange(4),))
        if choice < 0.92:
            return Op("finish_loop", (rng.randrange(2),))
        if choice < 0.98:
            return Op("drain", ())
        return Op("clear", ())

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        if config.get("loop") and config["loop"]["max_iters"] > 1:
            smaller = dict(config["loop"])
            smaller["max_iters"] -= 1
            yield {"widths": list(config["widths"]), "loop": smaller}
        if config.get("loop") is not None:
            yield {"widths": list(config["widths"]), "loop": None}

    # -- execution -----------------------------------------------------------

    def _slots(self, node: GraphNode) -> tuple[frozenset, frozenset]:
        return (
            frozenset(node.ready_signals.ready_names),
            frozenset(node.ready_next_iter.ready_names),
        )

    def _ingest(self, edge: GraphEdge) -> None:
        """Route one edge. Check ``ingest_input`` against its contract."""
        if edge.next_node not in self.io.nodes:
            require(
                "graph.unrouted_edge_is_external",
                self.io.ingest_input(edge) is False,
                f"edge {edge.name} claims to have been ingested by a node "
                f"{edge.next_node!r} that is not in this graph",
            )
            return

        node = self.io.nodes[edge.next_node]
        before = self._slots(node)
        accepted = self.io.ingest_input(edge)
        after = self._slots(node)

        if accepted:
            self.ingested.append(edge)
            gained = (after[0] - before[0]) | (after[1] - before[1])
            require(
                "graph.accept_fills_exactly_one_slot",
                gained == {edge.name},
                f"ingesting {edge.name!r} into {node.name} was accepted but the "
                f"ready slots gained {sorted(gained)}",
            )
            require(
                "graph.accept_does_not_disturb_other_slot",
                (after[0] - before[0]) == set() or (after[1] - before[1]) == set(),
                f"ingesting {edge.name!r} into {node.name} touched both slots",
            )
        else:
            require(
                "graph.reject_is_inert",
                before == after,
                f"ingesting {edge.name!r} into {node.name} was rejected but "
                f"changed its ready state: {before} -> {after}",
            )
            legitimate = (
                edge.name not in node.input_names
                or (edge.name in before[0] and edge.name in before[1])
            )
            require(
                "graph.reject_has_a_reason",
                legitimate,
                f"{node.name} rejected {edge.name!r}, which it declares as an "
                f"input and holds in neither slot ({before}); the edge is lost "
                "and the request hangs",
            )

    def _complete(self, node_name: str) -> None:
        """Run one node, then route each edge that the node produced."""
        self.io.ready_node_names.discard(node_name)
        completion = self.io.mark_node_complete(node_name)
        for edge in completion.output_edges:
            require(
                "graph.completion_respects_filter",
                (edge.name, edge.next_node) not in completion.filtered_signals,
                f"{node_name} returned {edge.name} -> {edge.next_node} in "
                "output_edges while also listing it as filtered",
            )
            self.pending.append(edge)

    def _step(self) -> bool:
        """Make one step of progress.

        Ingest an edge if one is pending. If none is pending, run one ready
        node. Returns False if the graph can do neither.
        """
        if self.pending:
            self._ingest(self.pending.pop(0))
            return True
        ready = sorted(self.io.ready_node_names)
        if ready:
            self._complete(ready[0])
            return True
        return False

    def _drain_bound(self) -> int:
        """Give the maximum number of steps that a full drain can need.

        A drain that needs more steps than this does not make progress. The
        limit is generous, so it does not report a slow graph as a hung graph.
        """
        iters = self.config["loop"]["max_iters"] if self.config.get("loop") else 1
        return 16 + 8 * len(self.io.nodes) * (iters + 1) * max(
            1, len(self.seed_edges)
        )

    def execute(self, op: Op) -> None:
        if op.kind == "ingest":
            if not self.pending:
                return
            self._ingest(self.pending.pop(op.args[0] % len(self.pending)))

        elif op.kind == "ingest_dup":
            if not self.ingested:
                return
            original = self.ingested[op.args[0] % len(self.ingested)]
            self._ingest(original.clone())

        elif op.kind == "complete":
            ready = sorted(self.io.ready_node_names)
            if not ready:
                return
            self._complete(ready[op.args[0] % len(ready)])

        elif op.kind == "finish_loop":
            loops = sorted(self.io.loops)
            if not loops:
                return
            self.io.register_loop_finish_signal(loops[op.args[0] % len(loops)])

        elif op.kind == "drain":
            budget = self._drain_bound()
            for _ in range(budget):
                if self.io.wg_state_registry.is_done:
                    return
                if not self._step():
                    break
            require(
                "graph.drain_terminates",
                self.io.wg_state_registry.is_done,
                "the graph stopped making progress before completing: "
                f"{len(self.pending)} edge(s) pending, ready="
                f"{sorted(self.io.ready_node_names)}, completed "
                f"{self.io.wg_state_registry._num_completed_entities}"
                f"/{self.io.wg_state_registry._num_managed_entities}",
            )

        elif op.kind == "clear":
            self.io.clear()
            self._check_cleared()
            self.pending = list(self.seed_edges)
            self.ingested.clear()

        else:
            raise AssertionError(f"unknown op {op.kind}")

    # -- invariants ----------------------------------------------------------

    def _check_cleared(self) -> None:
        """Check that ``clear()`` returned every part of the graph to its start."""
        for node in self.io.nodes.values():
            for label, signals in (
                ("ready_signals", node.ready_signals),
                ("ready_next_iter", node.ready_next_iter),
                ("speculative_signals", node.speculative_signals),
            ):
                require(
                    "graph.clear_empties_node_state",
                    not signals.ready_names and not signals.ready_inputs,
                    f"after clear, {node.name}.{label} still holds "
                    f"{sorted(signals.ready_names)}",
                )
        for loop in self.io.loops.values():
            require(
                "graph.clear_resets_loop",
                loop.curr_iter == 0
                and not loop.is_done
                and not loop._finish_signal
                and not loop._ingested_external_inputs
                and not loop._accumulated_cache,
                f"after clear, loop {loop.name} still holds iter="
                f"{loop.curr_iter} done={loop.is_done} "
                f"finish={loop._finish_signal} "
                f"ext_inputs={len(loop._ingested_external_inputs)} "
                f"accum={len(loop._accumulated_cache)}",
            )
        registry = self.io.wg_state_registry
        require(
            "graph.clear_resets_registry",
            not registry.is_done
            and registry._num_completed_entities == 0
            and not registry.ready_names,
            "after clear the top-level registry is not back to its "
            f"initial state (done={registry.is_done}, completed="
            f"{registry._num_completed_entities}, ready={registry.ready_names})",
        )

    def check(self) -> None:
        registry = self.io.wg_state_registry

        for name in registry.ready_names:
            require(
                "graph.ready_name_is_a_node",
                name in self.io.nodes,
                f"ready set names {name!r}, which is not a node in this graph",
            )
            node = self.io.nodes[name]
            require(
                "graph.ready_implies_all_inputs",
                node.input_names <= node.ready_signals.ready_names,
                f"{name} is queued as ready but is missing input(s) "
                f"{sorted(node.input_names - node.ready_signals.ready_names)}; "
                "it would run on a partial input set",
            )

        for node in self.io.nodes.values():
            for label, signals in (
                ("ready_signals", node.ready_signals),
                ("ready_next_iter", node.ready_next_iter),
            ):
                stray = signals.ready_names - node.input_names
                require(
                    "graph.slot_holds_only_declared_inputs",
                    not stray,
                    f"{node.name}.{label} holds {sorted(stray)}, which the node "
                    "does not declare as inputs",
                )
                require(
                    "graph.slot_names_match_edges",
                    signals.ready_names == set(signals.ready_inputs),
                    f"{node.name}.{label} names {sorted(signals.ready_names)} but "
                    f"stores edges for {sorted(signals.ready_inputs)}",
                )
                require(
                    "graph.is_ready_matches_inputs",
                    signals.is_ready == (node.input_names <= signals.ready_names),
                    f"{node.name}.{label}.is_ready={signals.is_ready} disagrees "
                    f"with its contents {sorted(signals.ready_names)} vs "
                    f"required {sorted(node.input_names)}",
                )

        for reg in self._registries():
            require(
                "graph.completion_count_in_range",
                0 <= reg._num_completed_entities <= reg._num_managed_entities,
                f"registry completed {reg._num_completed_entities} of "
                f"{reg._num_managed_entities} entities",
            )
            require(
                "graph.done_matches_completion_count",
                reg.is_done
                == (reg._num_completed_entities == reg._num_managed_entities),
                f"registry is_done={reg.is_done} with "
                f"{reg._num_completed_entities}/{reg._num_managed_entities} done",
            )

        for loop in self.io.loops.values():
            require(
                "graph.loop_iter_within_bound",
                0 <= loop.curr_iter < max(1, loop.max_iters),
                f"loop {loop.name} is on iteration {loop.curr_iter} of "
                f"max_iters={loop.max_iters}",
            )
            require(
                "graph.loop_done_has_a_cause",
                not loop.is_done
                or loop._finish_signal
                or loop.curr_iter == loop.max_iters - 1,
                f"loop {loop.name} finished on iteration {loop.curr_iter} of "
                f"{loop.max_iters} with no finish signal",
            )

    def _registries(self):
        """Collect the top-level registry and the registry inside each loop."""
        seen = [self.io.wg_state_registry]
        for loop in self.io.loops.values():
            seen.append(loop.inner_registry)
        return seen

    def final_check(self) -> None:
        self.io.clear()
        self._check_cleared()
