"""The generated model: its shape, and the wiring that the shape makes.

A tier 1 case starts from a spec. A spec is a small tree of dictionaries. It
is JSON-able, thus it lives in the config of a case and it replays.

``plan`` turns a spec into a wiring. A wiring gives the input names of each
node, and the edges that each node sends. Two readers take that wiring:

* ``build_section`` makes the real ``GraphSection`` of mstar.
* ``fuzzer.tier1.reference`` interprets it, and gives the expected output.

The two readers take the same wiring. The wiring is the input of the test.
The system under test is the code that runs the wiring.

The spec
--------

A spec holds a chain of stages, the number of requests, and the stop rules::

    {
      "stages": [stage, ...],
      "num_requests": 1,
      "stops": [{"node": "s0n0", "run": 2, "loop": "s1L"}],
    }

A stage is one of two kinds::

    {"kind": "nodes", "width": 2, "fanin": 1}
    {"kind": "loop", "max_iters": 3, "ext": 1, "num_outputs": 1,
     "num_accum": 1, "body": [stage, ...]}

A stage of nodes holds ``width`` nodes. The stage is a ``GraphNode`` when the
width is 1. It is a ``Parallel`` when the width is more. Each node of the
stage takes ``fanin`` inputs. A loop stage holds a chain of its own.

A spec for ``kv_run`` holds one more section. It declares the resources of
the model::

    "kv": {
      "page_size": 2, "num_layers": 1, "num_kv_heads": 1, "head_dim": 1,
      "positions": true,                  # declare a position resource
      "labels": {"s0n0": "kv0"},          # the cache stream of each node
      "spans": {"s0n0": 2},               # the tokens each step adds
      "forks": [{"node": "s0n0", "to": "kv1", "when": "pre"}],
      "capture": {"bs": 2, "num_tokens": 6, "slots": 2},
      "preplan": true,
      "max_passes": 4, "max_seq_len": 24, "max_num_pages": 17,
    }

``gen_kv`` computes the last three fields from the rest. The cache then holds
every stream of every request for ``max_passes`` passes.

The rules of a chain
--------------------

The rules below keep every generated graph correct. A graph that breaks one
of them reports a failure of the generator, and not a failure of mstar.

* Each node takes a minimum of one input. A node with no input is never
  ready, so the graph never drains.
* The body of a loop starts and ends with a stage of nodes. Only a stage of
  nodes chooses the names that it sends. A loop sends the names of its own
  loop-back edges.
* Two loop stages are never neighbors. Neither of the two can send the names
  that the other one needs.
* A body of one stage holds one node. ``Parallel`` calls an edge between two
  of its own members internal. Such an edge is not a loop-back edge, so a
  loop of one wide stage never closes.

The names
---------

Every name comes from the position of its node. Thus the wiring is a pure
function of the spec. ``s1n0`` is node 0 of stage 1. ``s1L`` is the loop at
stage 1. ``s1L_`` starts every name inside the body of that loop.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT

__all__ = [
    "GraphPlan",
    "LoopPlan",
    "NodePlan",
    "StagePlan",
    "build_section",
    "gen_kv",
    "gen_spec",
    "is_valid",
    "rebuild_kv",
    "plan",
    "shrink_spec",
]

NodeAndDest = tuple[str, str]


# ---------------------------------------------------------------------------
# the wiring
# ---------------------------------------------------------------------------

@dataclass
class NodePlan:
    """One node: the names it takes, and the edges it sends."""

    name: str
    inputs: list[str]
    outputs: list[NodeAndDest]

    @property
    def output_names(self) -> list[str]:
        """The distinct names, in order. Two edges of one name carry one
        value, as ``store_and_populate_graph_edges`` does."""
        seen: list[str] = []
        for name, _dest in self.outputs:
            if name not in seen:
                seen.append(name)
        return seen


@dataclass
class LoopPlan:
    """One loop: its body, the names it sends, and the names it holds."""

    name: str
    max_iters: int
    body: list["StagePlan"]
    # The edges that close the loop. The tail of the body sends them, and the
    # feeder of the loop sends them one time, for the first iteration.
    entry: list[NodeAndDest] = field(default_factory=list)
    # The names that leave the loop. An output name carries the value of the
    # last iteration. An accumulated name carries the value of every
    # iteration, in order.
    out_names: list[str] = field(default_factory=list)
    accum_names: list[str] = field(default_factory=list)
    # The names that the loop takes from outside and holds. The loop sends
    # each of them into the body again at the start of every iteration.
    ext: list[NodeAndDest] = field(default_factory=list)
    # The edges that the loop sends onward, to the next stage or to the
    # client. Every name here is an output name or an accumulated name.
    downstream: list[NodeAndDest] = field(default_factory=list)

    @property
    def emitted_names(self) -> list[str]:
        return self.out_names + self.accum_names


@dataclass
class StagePlan:
    """One stage: several nodes, or one loop."""

    nodes: list[NodePlan] = field(default_factory=list)
    loop: LoopPlan | None = None

    @property
    def is_loop(self) -> bool:
        return self.loop is not None


@dataclass
class GraphPlan:
    """The whole graph: the stages, the edges that start it, and the names
    that reach the client."""

    stages: list[StagePlan]
    seeds: list[NodeAndDest]
    emits: list[NodeAndDest]

    def all_nodes(self) -> list[NodePlan]:
        out: list[NodePlan] = []

        def walk(stages: list[StagePlan]) -> None:
            for stage in stages:
                if stage.is_loop:
                    walk(stage.loop.body)
                else:
                    out.extend(stage.nodes)

        walk(self.stages)
        return out

    def all_loops(self) -> list[LoopPlan]:
        out: list[LoopPlan] = []

        def walk(stages: list[StagePlan]) -> None:
            for stage in stages:
                if stage.is_loop:
                    out.append(stage.loop)
                    walk(stage.loop.body)

        walk(self.stages)
        return out

    def enclosing_loops(self) -> dict[str, list[str]]:
        """Map each node to the loops that hold it, outer first."""
        out: dict[str, list[str]] = {}

        def walk(stages: list[StagePlan], loops: list[str]) -> None:
            for stage in stages:
                if stage.is_loop:
                    walk(stage.loop.body, loops + [stage.loop.name])
                else:
                    for node in stage.nodes:
                        out[node.name] = list(loops)

        walk(self.stages, [])
        return out


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def _node_name(prefix: str, stage: int, index: int) -> str:
    return f"{prefix}s{stage}n{index}"


def _loop_name(prefix: str, stage: int) -> str:
    return f"{prefix}s{stage}L"


def _body_prefix(prefix: str, stage: int) -> str:
    return f"{_loop_name(prefix, stage)}_"


# ---------------------------------------------------------------------------
# the wiring pass
# ---------------------------------------------------------------------------

def _head_nodes(prefix: str, stages: list[dict]) -> list[str]:
    """The nodes of a chain that take the edges from outside."""
    first = stages[0]
    assert first["kind"] == "nodes", "a chain starts with a stage of nodes"
    return [_node_name(prefix, 0, c) for c in range(first["width"])]


def _loop_split(stage: dict, body_prefix: str) -> tuple[list[str], list[str]]:
    """The names that a loop sends: the outputs, then the accumulated.

    Both groups come from the loop-back names of the body, because
    ``Loop.__post_init__`` refuses an output that the body does not produce.
    """
    names = [name for name, _dest in _chain_entry(body_prefix, stage["body"])]
    num_out = stage["num_outputs"]
    num_accum = stage["num_accum"]
    return names[:num_out], names[num_out:num_out + num_accum]


def _stage_entry(prefix: str, stages: list[dict], index: int) -> list[NodeAndDest]:
    """The edges that one stage needs from the stage before it."""
    stage = stages[index]

    if stage["kind"] == "loop":
        body_prefix = _body_prefix(prefix, index)
        entry = _chain_entry(body_prefix, stage["body"])
        heads = _head_nodes(body_prefix, stage["body"])
        entry += [
            (f"{body_prefix}e{j}_{c}", head)
            for c, head in enumerate(heads)
            for j in range(stage["ext"])
        ]
        return entry

    consumers = [_node_name(prefix, index, c) for c in range(stage["width"])]
    if index > 0 and stages[index - 1]["kind"] == "loop":
        # A loop sends the names of its own loop-back edges, so the stage
        # after it takes those names. Every consumer takes every name.
        out_names, accum_names = _loop_split(
            stages[index - 1], _body_prefix(prefix, index - 1),
        )
        names = out_names + accum_names
        return [(name, consumer) for consumer in consumers for name in names]

    return [
        (f"{prefix}s{index}i{j}_{c}", consumer)
        for c, consumer in enumerate(consumers)
        for j in range(stage["fanin"])
    ]


def _chain_entry(prefix: str, stages: list[dict]) -> list[NodeAndDest]:
    """The edges that a whole chain needs from outside."""
    return _stage_entry(prefix, stages, 0)


def _plan_chain(
    prefix: str,
    stages: list[dict],
    exit_edges: list[NodeAndDest],
    extra_head_inputs: list[NodeAndDest] | None = None,
) -> list[StagePlan]:
    """Wire one chain, from its tail to its head.

    ``exit_edges`` are the edges that the tail of the chain sends. The tail
    shares them out over its nodes, one after the other.
    """
    planned: list[StagePlan | None] = [None] * len(stages)
    downstream = list(exit_edges)

    for index in reversed(range(len(stages))):
        stage = stages[index]
        entry = _stage_entry(prefix, stages, index)
        if index == 0 and extra_head_inputs:
            entry = entry + list(extra_head_inputs)

        if stage["kind"] == "nodes":
            planned[index] = _plan_nodes_stage(prefix, index, stage, entry, downstream)
        else:
            planned[index] = _plan_loop_stage(prefix, index, stage, downstream)
        downstream = entry

    return list(planned)


def _plan_nodes_stage(
    prefix: str,
    index: int,
    stage: dict,
    entry: list[NodeAndDest],
    downstream: list[NodeAndDest],
) -> StagePlan:
    width = stage["width"]
    inputs: dict[str, list[str]] = {}
    for name, dest in entry:
        inputs.setdefault(dest, []).append(name)

    nodes = []
    for p in range(width):
        name = _node_name(prefix, index, p)
        nodes.append(NodePlan(
            name=name,
            inputs=sorted(inputs.get(name, [])),
            # Share the edges out one after the other, so every node of the
            # stage sends a part of the work of the stage.
            outputs=list(downstream[p::width]),
        ))
    return StagePlan(nodes=nodes)


def _plan_loop_stage(
    prefix: str, index: int, stage: dict, downstream: list[NodeAndDest],
) -> StagePlan:
    body_prefix = _body_prefix(prefix, index)
    body_stages = stage["body"]
    entry = _chain_entry(body_prefix, body_stages)
    heads = _head_nodes(body_prefix, body_stages)
    ext = [
        (f"{body_prefix}e{j}_{c}", head)
        for c, head in enumerate(heads)
        for j in range(stage["ext"])
    ]
    out_names, accum_names = _loop_split(stage, body_prefix)

    body = _plan_chain(body_prefix, body_stages, exit_edges=entry, extra_head_inputs=ext)
    return StagePlan(loop=LoopPlan(
        name=_loop_name(prefix, index),
        max_iters=stage["max_iters"],
        body=body,
        entry=entry,
        out_names=out_names,
        accum_names=accum_names,
        ext=ext,
        downstream=list(downstream),
    ))


def plan(spec: dict) -> GraphPlan:
    """Turn a spec into the explicit wiring of its graph."""
    stages = spec["stages"]
    last = stages[-1]
    if last["kind"] == "nodes":
        emits = [(f"out{p}", EMIT_TO_CLIENT) for p in range(last["width"])]
    else:
        out_names, accum_names = _loop_split(
            last, _body_prefix("", len(stages) - 1),
        )
        emits = [(name, EMIT_TO_CLIENT) for name in out_names + accum_names]

    return GraphPlan(
        stages=_plan_chain("", stages, exit_edges=emits),
        seeds=_chain_entry("", stages),
        emits=emits,
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

def _build_stages(stages: list[StagePlan]) -> GraphSection:
    sections = [_build_stage(stage) for stage in stages]
    return sections[0] if len(sections) == 1 else Sequential(sections=sections)


def _build_stage(stage: StagePlan) -> GraphSection:
    if stage.is_loop:
        loop = stage.loop
        return Loop(
            section=_build_stages(loop.body),
            max_iters=loop.max_iters,
            outputs=[
                GraphEdge(name=name, next_node=dest)
                for name, dest in _edges_named(loop, loop.out_names)
            ],
            accumulated_outputs=[
                GraphEdge(name=name, next_node=dest)
                for name, dest in _edges_named(loop, loop.accum_names)
            ],
            name=loop.name,
        )

    nodes = [
        GraphNode(
            name=node.name,
            input_names=set(node.inputs),
            outputs=[
                GraphEdge(name=name, next_node=dest) for name, dest in node.outputs
            ],
        )
        for node in stage.nodes
    ]
    return nodes[0] if len(nodes) == 1 else Parallel(sections=nodes)


def _edges_named(loop: LoopPlan, names: list[str]) -> list[NodeAndDest]:
    return [(name, dest) for name, dest in loop.downstream if name in names]


def build_section(graph_plan: GraphPlan) -> GraphSection:
    """Make the real ``GraphSection`` that the wiring describes."""
    return _build_stages(graph_plan.stages)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _gen_nodes_stage(rng: random.Random, width: int | None = None) -> dict:
    return {
        "kind": "nodes",
        "width": rng.randint(1, 2) if width is None else width,
        "fanin": rng.randint(1, 2),
    }


def _gen_body(rng: random.Random, depth: int) -> list[dict]:
    """Make the chain inside a loop. It starts and ends with nodes."""
    length = rng.randint(1, 3 if depth > 0 else 2)
    if length == 1:
        # One stage, thus one node: see the rules in the module docstring.
        return [_gen_nodes_stage(rng, width=1)]
    stages = [_gen_nodes_stage(rng)]
    for _ in range(length - 2):
        if depth > 0 and rng.random() < 0.5:
            stages.append(_gen_loop_stage(rng, depth - 1))
        else:
            stages.append(_gen_nodes_stage(rng))
    # The tail of the body holds one node, and it sends the loop-back edges.
    # With the rule of connection below, every other entity of the iteration
    # is an ancestor of the tail. The tail is therefore always the last
    # entity to complete. See "A limit of the generator" in ``model_run``.
    stages.append(_gen_nodes_stage(rng, width=1))
    return stages


def _gen_loop_stage(rng: random.Random, depth: int) -> dict:
    return {
        "kind": "loop",
        "max_iters": rng.randint(1, 3),
        "ext": rng.randint(0, 1),
        # ``_finalize`` sets these two, after the rule of connection fixes
        # the number of names that the body of the loop sends.
        "num_outputs": 1,
        "num_accum": 0,
        "body": _gen_body(rng, depth),
    }


def _connect(stages: list[dict]) -> None:
    """Give each stage one edge from every node of the stage before it.

    Without this rule a stage of several nodes can send fewer edges than it
    holds. One node of the stage then sends nothing. Such a node is not an
    ancestor of the tail of its loop. The tail therefore completes while that
    node is still pending. That shape meets the two open failures of
    ``tier0/graph_io``, which the corpus holds.
    """
    for index in range(1, len(stages)):
        before = stages[index - 1]
        if before["kind"] != "nodes":
            # A loop sends its own declared outputs, and every node of the
            # next stage takes all of them.
            continue
        stage = stages[index]
        head = stage if stage["kind"] == "nodes" else stage["body"][0]
        head["fanin"] = before["width"]


def _finalize(rng: random.Random, stages: list[dict]) -> None:
    """Connect each chain, then choose what each loop sends."""
    _connect(stages)
    for stage in stages:
        if stage["kind"] != "loop":
            continue
        _finalize(rng, stage["body"])
        head = stage["body"][0]
        num_names = head["width"] * head["fanin"]
        stage["num_outputs"] = rng.randint(1, num_names)
        stage["num_accum"] = rng.randint(0, num_names - stage["num_outputs"])


# How many forward passes one request runs before the machine removes it and
# a new request takes its place. It bounds the size of the cache that a case
# has to allocate.
MAX_PASSES = 4


def _stage_groups(graph_plan: GraphPlan) -> list[list[str]]:
    """The nodes of each stage, in execution order.

    Every stage completes before the next one starts. A stage is the only
    place where the order of two nodes is free. This order therefore decides
    what two nodes can share.
    """
    groups: list[list[str]] = []

    def walk(stages: list[StagePlan]) -> None:
        for stage in stages:
            if stage.is_loop:
                walk(stage.loop.body)
                continue
            groups.append([node.name for node in stage.nodes])

    walk(graph_plan.stages)
    return groups


def gen_forks(
    rng: random.Random, graph_plan: GraphPlan, labels: dict[str, str],
) -> list[dict]:
    """Sample a fork: one step copies its own stream onto another label.

    ``KVStep`` carries ``pre_forks`` and ``post_forks``. A pre-fork copies in
    ``plan``, before the spans of the step land. A post-fork copies in
    ``commit``, after them. Batched CFG uses them.

    The target is always a label that only later stages own. A label of the
    same stage makes the result depend on the order inside that stage, and
    the interpreter takes no order.
    """
    groups = _stage_groups(graph_plan)
    candidates: list[tuple[str, list[str]]] = []
    for index, group in enumerate(groups):
        later = {labels[name] for rest in groups[index + 1:] for name in rest}
        own = {labels[name] for name in group}
        targets = sorted(later - own)
        if targets:
            candidates.extend((name, targets) for name in group)
    if not candidates or rng.random() < 0.6:
        return []
    node, targets = rng.choice(candidates)
    return [{
        "node": node,
        "to": rng.choice(targets),
        "when": rng.choice(["pre", "post"]),
    }]


def _node_labels(graph_plan: GraphPlan) -> dict[str, str]:
    """Give each node a cache label from its position inside its stage.

    Two nodes of one stage take different labels. Two nodes of different
    stages can share one label.

    This rule keeps the oracle exact. The engine runs the nodes of one stage
    in any order, so two nodes that share a stream must be ordered by the
    graph. Every stage completes before the next one starts, and a stage is
    the only place where the order is free.
    """
    labels: dict[str, str] = {}

    def walk(stages: list[StagePlan]) -> None:
        for stage in stages:
            if stage.is_loop:
                walk(stage.loop.body)
                continue
            for index, node in enumerate(stage.nodes):
                labels[node.name] = f"kv{index}"

    walk(graph_plan.stages)
    return labels


def _size_kv(
    fields: dict, labels: dict[str, str], spans: dict[str, int],
    runs: dict[str, int], num_requests: int, forks: bool = False,
) -> dict:
    """Compute the sizes that the shape and the workload need.

    The cache holds every stream of every request for ``MAX_PASSES`` passes.
    A generated case therefore never runs out of pages. A case under real
    pressure needs an oracle for eviction, which this tier does not hold.

    A fork copies one stream onto another label. With a fork in the model,
    each label takes the size of all of the labels together.
    """
    page_size = fields["page_size"]
    tokens: dict[str, int] = {}
    for name, label in labels.items():
        tokens[label] = tokens.get(label, 0) + runs.get(name, 0) * spans[name]
    if forks:
        total = sum(tokens.values())
        tokens = dict.fromkeys(tokens, total)
    longest = max(tokens.values(), default=0) * MAX_PASSES
    pages = sum(
        -(-(count * MAX_PASSES) // page_size) + 1 for count in tokens.values()
    )
    return {
        **fields,
        "labels": labels,
        "spans": spans,
        "max_passes": MAX_PASSES,
        "max_seq_len": max(longest, 1),
        # one page for the sink, and four pages of slack
        "max_num_pages": 1 + pages * num_requests + 4,
    }


def _first_writers(labels: dict[str, str]) -> set[str]:
    """The node that writes each label first.

    ``labels`` comes from a walk in execution order, so the first node of a
    label is the first one to run.
    """
    seen: dict[str, str] = {}
    for name, label in labels.items():
        seen.setdefault(label, name)
    return set(seen.values())


def _fix_spans(labels: dict[str, str], spans: dict[str, int]) -> dict[str, int]:
    """Give the first writer of each label a span of one or more.

    A zero-span segment reads a stream without extending it. On a label that
    nothing has written, ``KVManager.admit`` returns OK and reserves no
    stream, and ``KVManager.plan`` then raises ``KeyError`` on that same step.
    ``corpus/kv_run/`` holds the case. The generator leaves it there, because
    one frequent failure hides every other failure.
    """
    first = _first_writers(labels)
    return {
        name: max(span, 1) if name in first else span
        for name, span in spans.items()
    }


def gen_kv(
    rng: random.Random, graph_plan: GraphPlan, runs: dict[str, int],
    num_requests: int,
) -> dict:
    """Sample the KV resource of a model, and size it for the case."""
    labels = _node_labels(graph_plan)
    # A span of zero is a step that reads its stream and does not extend it.
    # ``Segment`` names that case, and no released model reaches it.
    spans = _fix_spans(labels, {name: rng.randint(0, 3) for name in labels})
    fields = {
        "page_size": rng.choice([1, 2, 4]),
        "num_layers": rng.randint(1, 2),
        "num_kv_heads": rng.randint(1, 2),
        "head_dim": rng.randint(1, 2),
        # A model that also declares positions. The resource names the cache
        # in ``depends_on``, so the runner has to resolve and order the two.
        "positions": rng.random() < 0.5,
    }
    if rng.random() < 0.5:
        # The shape of a captured bucket. ``bs`` above the number of requests
        # gives the step padding rows, which is where a replay addresses the
        # sink page. Two slots are the double buffer that the pre-plan path
        # needs.
        fields["capture"] = {
            "bs": num_requests + rng.randint(0, 1),
            "num_tokens": num_requests * 3 + rng.randint(1, 3),
            "slots": rng.choice([1, 2]),
        }
        # The engine pre-plans only under a lease, so this follows capture.
        fields["preplan"] = rng.random() < 0.6
    forks = gen_forks(rng, graph_plan, labels)
    section = _size_kv(
        fields, labels, spans, runs, num_requests, forks=bool(forks),
    )
    section["forks"] = forks
    return section


def rebuild_kv(spec: dict, candidate: dict) -> dict:
    """The KV section of a shrunk spec.

    A shrink step adds and removes nodes, so the node-keyed parts of the
    section have to follow. The shape stays, a node that survives keeps its
    span, and a new one takes a span of one.
    """
    from fuzzer.tier1.reference import Reference

    kv = spec["kv"]
    graph_plan = plan(candidate)
    labels = _node_labels(graph_plan)
    spans = _fix_spans(
        labels, {name: kv["spans"].get(name, 1) for name in labels},
    )
    bare = {key: value for key, value in candidate.items() if key != "kv"}
    runs = Reference(bare, graph_plan).run_pass("r0", 0).runs
    fields = {
        key: kv[key]
        for key in (
            "page_size", "num_layers", "num_kv_heads", "head_dim", "positions",
        )
    }
    if kv.get("capture") is not None:
        fields["capture"] = dict(kv["capture"])
        fields["preplan"] = kv.get("preplan", False)
    # A fork whose node or whose target label is gone goes with them.
    forks = [
        fork for fork in kv.get("forks", [])
        if fork["node"] in labels and fork["to"] in set(labels.values())
    ]
    section = _size_kv(
        fields, labels, spans, runs, candidate.get("num_requests", 1),
        forks=bool(forks),
    )
    section["forks"] = forks
    return section


def gen_spec(rng: random.Random, with_resources: bool = False) -> dict:
    """Sample the shape of one model.

    With ``with_resources``, the model also declares a KV cache, and every
    node of it takes a real step against that cache.
    """
    length = rng.randint(1, 3)
    stages: list[dict] = []
    for _ in range(length):
        loop_allowed = not stages or stages[-1]["kind"] == "nodes"
        if loop_allowed and rng.random() < 0.5:
            stages.append(_gen_loop_stage(rng, depth=1))
        else:
            stages.append(_gen_nodes_stage(rng))
    _finalize(rng, stages)

    spec = {
        "stages": stages,
        "num_requests": rng.randint(1, 2),
        "stops": [],
    }

    graph_plan = plan(spec)
    loops = graph_plan.all_loops()
    enclosing = graph_plan.enclosing_loops()
    if loops and rng.random() < 0.5:
        loop = rng.choice(loops)
        inside = [name for name, names in enclosing.items() if loop.name in names]
        spec["stops"] = [{
            "node": rng.choice(sorted(inside)),
            "run": rng.randint(0, 2),
            "loop": loop.name,
        }]

    if with_resources:
        from fuzzer.tier1.reference import Reference

        runs = Reference(spec, graph_plan).run_pass("r0", 0).runs
        spec["kv"] = gen_kv(rng, graph_plan, runs, spec["num_requests"])
    return spec


# ---------------------------------------------------------------------------
# shrinking
# ---------------------------------------------------------------------------

def _valid_chain(stages: list[dict], body: bool) -> bool:
    """Say if a chain follows the rules in the module docstring."""
    if not stages:
        return False
    for before, after in zip(stages, stages[1:], strict=False):
        if before["kind"] == "loop" and after["kind"] == "loop":
            return False
    if body:
        if stages[0]["kind"] != "nodes" or stages[-1]["kind"] != "nodes":
            return False
        if len(stages) == 1 and stages[0]["width"] != 1:
            return False
    for stage in stages:
        if stage["kind"] != "loop":
            if stage["width"] < 1 or stage["fanin"] < 1:
                return False
            continue
        if stage["max_iters"] < 1 or stage["ext"] < 0:
            return False
        if not _valid_chain(stage["body"], body=True):
            return False
        head = stage["body"][0]
        names = head["width"] * head["fanin"]
        if stage["num_outputs"] < 1 or stage["num_outputs"] + stage["num_accum"] > names:
            return False
    return True


def is_valid(spec: dict) -> bool:
    """Say if a spec makes a correct graph."""
    return _valid_chain(spec.get("stages", []), body=False)


def _chain_mutations(stages: list[dict]) -> Iterator[list[dict]]:
    """Smaller chains to try in place of ``stages``, coarse first."""
    def swap(index: int, replacement: list[dict]) -> list[dict]:
        return stages[:index] + replacement + stages[index + 1:]

    for index in range(len(stages)):
        if len(stages) > 1:
            yield swap(index, [])

    for index, stage in enumerate(stages):
        if stage["kind"] == "loop":
            # A loop is the expensive part of a case. Try to remove it first.
            yield swap(index, [{"kind": "nodes", "width": 1, "fanin": 1}])
            for field in ("max_iters", "num_outputs"):
                if stage[field] > 1:
                    yield swap(index, [{**stage, field: stage[field] - 1}])
            for field in ("ext", "num_accum"):
                if stage[field] > 0:
                    yield swap(index, [{**stage, field: 0}])
            for body in _chain_mutations(stage["body"]):
                yield swap(index, [{**stage, "body": body}])
            continue
        for field in ("width", "fanin"):
            if stage[field] > 1:
                yield swap(index, [{**stage, field: stage[field] - 1}])


def shrink_spec(spec: dict) -> Iterator[dict]:
    """Smaller specs to try, coarse first.

    Every candidate keeps the rules of a chain. A stop rule that names a node
    or a loop that the candidate no longer holds goes away with it.
    """
    if spec.get("num_requests", 1) > 1:
        yield {**spec, "num_requests": 1}
    if spec.get("stops"):
        yield {**spec, "stops": []}

    for stages in _chain_mutations(spec["stages"]):
        candidate = {**spec, "stages": stages}
        if not is_valid(candidate):
            continue
        graph_plan = plan(candidate)
        nodes = {node.name for node in graph_plan.all_nodes()}
        loops = {loop.name for loop in graph_plan.all_loops()}
        candidate["stops"] = [
            stop for stop in spec.get("stops", [])
            if stop["node"] in nodes and stop["loop"] in loops
        ]
        if "kv" in spec:
            candidate["kv"] = rebuild_kv(spec, candidate)
        yield candidate
