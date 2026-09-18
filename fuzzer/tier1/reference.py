"""The oracle: a small interpreter of the wiring.

This module is the specification of one forward pass. It reads the wiring that
``spec.plan`` made, and it gives the values that must reach the client. It
holds no queue, no ready set and no registry. It runs each stage one after the
other, and it runs each loop as a Python loop.

The engine must agree with this interpreter. The engine chooses the order of
the nodes. It batches several requests together. It runs a loop through a
registry of signals. None of that may change one number.

The rules of a loop
-------------------

* A regular output of a loop carries the value of the last iteration.
  ``Loop._uncache_outputs`` empties the cache at each new iteration.
* An accumulated output carries the value of every iteration, in order.
  ``Loop._accumulated_cache`` keeps them. The loop sends them together when
  it ends.
* The external inputs of a loop do not change. The loop holds one copy, and
  sends it into the body again at each iteration.
* A loop ends after ``max_iters`` iterations. It also ends at the end of the
  iteration in which a node registered the finish signal.
* A loop inside a loop starts again for each iteration of the outer loop. Its
  iteration counter and its finish signal both reset.

The cache
---------

A spec that declares a KV resource makes each step read its own stream before
it writes to that stream. The interpreter holds each stream as the list of
tokens in it. It holds no page, no allocator and no layout. Thus the engine
must agree about the contents of a stream while it chooses the pages alone.

A stream lives for the whole life of a request, across forward passes, and
``remove_request`` ends it. Two nodes share a stream only when the graph
orders them. The interpreter therefore appends to a stream in stage order.
``spec._node_labels`` gives that rule.

A fork copies one stream onto another label, token for token. A pre-fork
copies before the tokens of its step land, and a post-fork copies after them.

Not covered
-----------

* several worker graphs, and the routing between them
* streaming edges and speculative execution
* the pages of a cache: the interpreter counts tokens and names no page
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fuzzer.tier1.spec import GraphPlan, LoopPlan, StagePlan, plan
from fuzzer.tier1.values import checksum, prefix_checksum, seed_value, token_value

__all__ = ["PassResult", "Reference"]

Values = tuple[str, ...]
Env = dict[str, Values]


# The output name under which a step's cache value is computed. It must match
# ``resources.CACHE_NAME``; the two modules agree on it and share nothing else.
CACHE_NAME = "<kv>"


@dataclass
class PassResult:
    """What one forward pass must produce."""

    # The values that reach the client, by the name of the edge.
    emitted: dict[str, Values]
    # The number of times each node ran during the pass.
    runs: dict[str, int] = field(default_factory=dict)


@dataclass
class _Context:
    """The state that one pass carries while the interpreter runs it."""

    request_id: str
    pass_index: int
    stops: list[dict]
    runs: dict[str, int] = field(default_factory=dict)
    signals: set[str] = field(default_factory=set)


class Reference:
    """The expected result of each forward pass of one generated model."""

    def __init__(self, spec: dict, graph_plan: GraphPlan | None = None) -> None:
        self.spec = spec
        self.plan = graph_plan if graph_plan is not None else plan(spec)
        self.kv = spec.get("kv")
        # node -> (target label, "pre" | "post")
        self.forks: dict[str, tuple[str, str]] = {
            fork["node"]: (fork["to"], fork["when"])
            for fork in (self.kv or {}).get("forks", [])
        }
        # (request, label) -> one entry for each token: the value of the step
        # that wrote it, and the position of the token inside that step.
        self.streams: dict[tuple[str, str], list[tuple[str, int]]] = {}

    def remove_request(self, request_id: str) -> None:
        """End every stream of a request, as ``remove_request`` does."""
        for key in [key for key in self.streams if key[0] == request_id]:
            del self.streams[key]

    def _cache_read(self, request_id: str, label: str) -> str:
        """The checksum of the prefix that a step reads before it writes."""
        tokens = self.streams.get((request_id, label), [])
        layers = self.kv["num_layers"]
        return prefix_checksum([
            token_value(value, index, layer)
            for layer in range(layers)
            for value, index in tokens
        ])

    def _fork(
        self, request_id: str, node_name: str, label: str, when: str,
    ) -> None:
        """Copy a stream onto its fork target, token for token."""
        fork = self.forks.get(node_name)
        if fork is None or fork[1] != when:
            return
        source = self.streams.get((request_id, label), [])
        self.streams[(request_id, fork[0])] = list(source)

    def _cache_write(
        self, request_id: str, label: str, value: str, span: int,
    ) -> None:
        """Append the tokens that a step writes."""
        self.streams.setdefault((request_id, label), []).extend(
            (value, index) for index in range(span)
        )

    def seed_env(self, request_id: str, pass_index: int) -> Env:
        """The values of the edges that start a forward pass."""
        return {
            name: (seed_value(request_id, pass_index, name),)
            for name, _dest in self.plan.seeds
        }

    def run_pass(self, request_id: str, pass_index: int) -> PassResult:
        """Interpret one forward pass of one request."""
        context = _Context(
            request_id=request_id,
            pass_index=pass_index,
            stops=self.spec.get("stops", []),
        )
        env = self._run_stages(
            self.plan.stages, self.seed_env(request_id, pass_index), context,
        )
        emitted = {name: env[name] for name, _dest in self.plan.emits}
        return PassResult(emitted=emitted, runs=dict(context.runs))

    # -- the interpreter -----------------------------------------------------

    def _run_stages(
        self, stages: list[StagePlan], env: Env, context: _Context,
    ) -> Env:
        """Run a chain. Each stage reads the values that the stage before it
        produced."""
        current = env
        for stage in stages:
            current = self._run_stage(stage, current, context)
        return current

    def _run_stage(self, stage: StagePlan, env: Env, context: _Context) -> Env:
        if stage.is_loop:
            return self._run_loop(stage.loop, env, context)

        produced: Env = {}
        for node in stage.nodes:
            inputs = [(name, env[name]) for name in node.inputs]
            run_index = context.runs.get(node.name, 0)
            context.runs[node.name] = run_index + 1

            # A step reads its own stream first. Every value of the step then
            # covers what it read.
            extra: tuple[str, ...] = ()
            if self.kv is not None:
                label = self.kv["labels"][node.name]
                self._fork(context.request_id, node.name, label, "pre")
                extra = (self._cache_read(context.request_id, label),)

            for output_name in node.output_names:
                produced[output_name] = (checksum(
                    context.request_id, context.pass_index, node.name,
                    output_name, run_index, inputs, extra=extra,
                ),)

            if self.kv is not None:
                self._cache_write(
                    context.request_id, label,
                    checksum(
                        context.request_id, context.pass_index, node.name,
                        CACHE_NAME, run_index, inputs, extra=extra,
                    ),
                    self.kv["spans"][node.name],
                )
                self._fork(context.request_id, node.name, label, "post")
            # The worker reads the stop of a step after the step gives its
            # outputs, and before it marks the node complete.
            self._register_stops(node.name, run_index, context)
        return produced

    def _run_loop(self, loop: LoopPlan, env: Env, context: _Context) -> Env:
        held = {name: env[name] for name, _dest in loop.ext}
        entry_names = _unique_names(loop.entry)
        carried = {name: env[name] for name in entry_names}

        cached: Env = {}
        accumulated: Env = {name: () for name in loop.accum_names}

        for _iteration in range(loop.max_iters):
            produced = self._run_stages(loop.body, {**held, **carried}, context)
            for name in loop.out_names:
                cached[name] = produced[name]
            for name in loop.accum_names:
                accumulated[name] = accumulated[name] + produced[name]
            if loop.name in context.signals:
                break
            carried = {name: produced[name] for name in entry_names}

        # An outer iteration resets an inner loop. The end of a pass resets a
        # top-level loop. The finish signal survives neither reset.
        context.signals.discard(loop.name)

        return {**cached, **accumulated}

    def _register_stops(
        self, node_name: str, run_index: int, context: _Context,
    ) -> None:
        for stop in context.stops:
            if stop["node"] == node_name and stop["run"] == run_index:
                context.signals.add(stop["loop"])


def _unique_names(edges: list[tuple[str, str]]) -> list[str]:
    """The distinct names of a list of edges, in order."""
    names: list[str] = []
    for name, _dest in edges:
        if name not in names:
            names.append(name)
    return names
