"""One generated model, run over the real graph layer, against the reference.

The generator makes the model. The ops make the schedule. The oracle is an
interpreter of the same model that holds no queue and no registry
(``fuzzer.tier1.reference``).

Each node returns a checksum. The checksum covers the identity of its step,
and every input that the step reads. Thus each of these faults changes a
number that a person can compare:

* a lost edge
* an old buffer
* an iteration in the wrong order
* a loop that ends too early

The machine needs no tolerance and no reference weights.

The ops of a case make a schedule. At each step the driver runs any node that
the graph reports ready. A client must not see that choice. The interpreter
therefore takes no order: it runs the stages one after the other. Every
schedule must give the same values. This is the metamorphic oracle of the
machine, and the oracle for the order of a batch.

Ops: start, step, drain, abort.

* ``start(rid)`` opens a forward pass and sends the first edges.
* ``step(rid, k)`` runs the k-th ready node of that request.
* ``drain(rid)`` runs ready nodes until the pass ends.
* ``abort(rid)`` drops the request, as ``remove_request`` does, and opens a
  new request in its place. The new request takes a new name, because the
  conductor never gives one name to two requests.

Invariants
----------
emit.matches_reference         the values that reach the client are the values
                               that the interpreter gives
emit.runs_match_reference      each node ran the number of times that the
                               interpreter gives
route.every_edge_lands         every edge that a node sends reaches a node of
                               this graph or the client
graph.pass_terminates          a pass that started always ends: no schedule
                               leaves it open with no ready node
state.clear_empties_the_graph  after a pass, every slot, every loop and the
                               registry are empty again

Not covered
-----------

* several worker graphs, and the routing between them. One worker graph runs
  the whole model here. A lost cross-worker edge is out of reach.
* the resources. This machine holds no batch, no cache and no slot. Two
  requests hold separate copies of the graph, so neither one reaches the data
  of the other. ``kv_run`` drives the resources.
* streaming edges, speculative execution and tensor parallelism.
* the refcounts of the transport layer. ``tier0/tensor_store`` holds those.

A limit of the generator
------------------------

The last stage of a loop body holds one node. The producer of the loop-back
edges is therefore always the last entity of the iteration to complete.

A wider tail meets the two open failures of ``tier0/graph_io``. The loop-back
edge of one member arrives while another member is still not complete. The
destination of that edge goes into the ready queue again, inside the
iteration that it already ran. A schedule that runs it there loses an entity
of the iteration.

``corpus/model_run/`` holds that shape, and the generator avoids it. One
frequent failure hides every other failure of the same oracle.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier1.executor import RequestRun
from fuzzer.tier1.reference import Reference
from fuzzer.tier1.spec import gen_spec, plan, shrink_spec
from fuzzer.tier1.values import checksum

# A pass that needs more steps than this many times its expected work is not
# making progress. The constant is generous: only a defect reaches it.
STEP_SLACK = 4
STEP_FLOOR = 50


class ModelRunMachine(StateMachine):
    name = "model_run"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        return gen_spec(rng)

    def __init__(self, config: dict) -> None:
        self.spec = config
        self.plan = plan(config)
        self.reference = Reference(config, self.plan)
        self.stops = config.get("stops", [])

        # A slot holds one request at a time. An abort puts a new request
        # into the slot, under a new name.
        self.slots = list(range(config.get("num_requests", 1)))
        self.request_ids = {slot: f"r{slot}" for slot in self.slots}
        # How many requests each slot held. It names them apart.
        self.incarnations = dict.fromkeys(self.slots, 0)
        self.runs = {slot: self._new_run(slot) for slot in self.slots}
        # The number of passes that the request in each slot finished. The
        # driver reads the counter of the system under test instead. A wrong
        # counter therefore makes the two numbers differ, and every checksum
        # of the pass changes.
        self.passes = dict.fromkeys(self.slots, 0)

        # A new interpreter. ``run_pass`` advances the cache streams of a
        # spec that declares resources, and this call wants the counts only.
        expected = Reference(config, self.plan).run_pass("r0", 0)
        self.step_bound = STEP_SLACK * sum(expected.runs.values()) + STEP_FLOOR
        for slot in self.slots:
            self._on_new_request(slot)

    def _new_run(self, slot: int) -> RequestRun:
        return RequestRun(self.plan, self.request_ids[slot], self.stops)

    def _run_node(self, slot: int, node_name: str) -> list[int]:
        """Run one node, and report the slots that ran.

        This machine holds no resource, so a step reads no cache. The value
        of a step comes from its identity and its inputs alone. ``kv_run``
        replaces this method, and takes the value from a real step of the
        resource layer.
        """
        run = self.runs[slot]
        step = run.begin_node(node_name)
        run.finish_node(node_name, {
            name: checksum(
                step.request_id, run.pass_index, step.node_name, name,
                step.run_index, step.inputs,
            )
            for name in step.output_names
        })
        return [slot]

    # -- hooks for a machine that holds resources ----------------------------

    def _on_new_request(self, slot: int) -> None:
        """A slot took a new request. Open whatever state it needs."""
        return

    def _on_drop_request(self, slot: int, request_id: str) -> None:
        """A slot dropped its request. Close whatever state it held."""
        return

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        request = rng.randrange(len(self.slots))
        choice = rng.random()
        if choice < 0.15:
            return Op("start", (request,))
        if choice < 0.80:
            return Op("step", (request, rng.randint(0, 3)))
        if choice < 0.95:
            return Op("drain", (request,))
        return Op("abort", (request,))

    # -- execution -----------------------------------------------------------

    def execute(self, op: Op) -> None:
        slot = self.slots[op.args[0] % len(self.slots)]
        run = self.runs[slot]

        if op.kind == "start":
            if not run.in_flight:
                run.start()
            return

        if op.kind == "step":
            if not run.in_flight:
                return
            ready = run.ready()
            if not ready:
                return
            for ran in self._run_node(slot, ready[op.args[1] % len(ready)]):
                self._settle(ran)
            return

        if op.kind == "drain":
            if not run.in_flight:
                return
            steps = 0
            while run.in_flight and steps < self.step_bound:
                ready = run.ready()
                if not ready:
                    break
                ran = self._run_node(slot, ready[0])
                steps += 1
                for touched in ran:
                    self._settle(touched)
            # Every input of this machine comes from inside the graph. A
            # pass that is still open can therefore never make progress
            # again. It used the whole bound, or it has no ready node and is
            # not done. Both cases are a request that hangs.
            require(
                "graph.pass_terminates",
                not run.in_flight,
                f"{run.request_id} did not end its pass after {steps} steps; "
                f"ready={run.ready()}",
            )
            return

        if op.kind == "abort":
            self._replace(slot)
            return

    def _replace(self, slot: int) -> None:
        """Drop the request of a slot, and open a new one in its place.

        The worker drops the whole per-request graph on a cleanup. The next
        request of the slot is a new request. It takes a new name, its own
        count of forward passes, and its own state in every resource.
        """
        self._on_drop_request(slot, self.request_ids[slot])
        self.incarnations[slot] += 1
        self.request_ids[slot] = f"r{slot}a{self.incarnations[slot]}"
        self.runs[slot] = self._new_run(slot)
        self.passes[slot] = 0
        self._on_new_request(slot)

    def check(self) -> None:
        """Every edge of this machine goes to this graph or to the client."""
        for slot in self.slots:
            run = self.runs[slot]
            require(
                "route.every_edge_lands",
                not run.rejected and not run.external,
                f"{run.request_id} sent edges that no node took: "
                f"rejected={[edge.name for edge in run.rejected]} "
                f"external={[edge.name for edge in run.external]}",
            )

    def _settle(self, slot: int) -> None:
        """End the pass if the graph reports that it is done, and compare."""
        run = self.runs[slot]
        request_id = run.request_id
        if not run.is_done():
            return

        pass_index = self.passes[slot]
        expected = self.reference.run_pass(request_id, pass_index)
        actual_runs = dict(run.runs)
        emitted = run.finish()
        self.passes[slot] = pass_index + 1

        # The number of runs comes first. It names the node that ran too many
        # times, or too few.
        require(
            "emit.runs_match_reference",
            actual_runs == expected.runs,
            f"{request_id} pass {pass_index} ran {actual_runs}; "
            f"the interpreter gives {expected.runs}",
        )
        require(
            "emit.matches_reference",
            emitted == expected.emitted,
            f"{request_id} pass {pass_index} emitted {_short(emitted)}; "
            f"the interpreter gives {_short(expected.emitted)}",
        )
        self._require_empty(request_id, run)

    def _require_empty(self, request_id: str, run: RequestRun) -> None:
        """After a pass, the graph must hold nothing of that pass."""
        leftovers: list[str] = []
        for node in run.io.nodes.values():
            for label, signals in (
                ("ready", node.ready_signals),
                ("next_iter", node.ready_next_iter),
                ("speculative", node.speculative_signals),
            ):
                if signals.ready_names:
                    leftovers.append(f"{node.name}.{label}={sorted(signals.ready_names)}")
        for loop in run.io.loops.values():
            if loop.curr_iter or loop.is_done or loop._finish_signal:
                leftovers.append(
                    f"{loop.name}: iter={loop.curr_iter} done={loop.is_done} "
                    f"signal={loop._finish_signal}"
                )
            if loop._cached_outputs or loop._accumulated_cache:
                leftovers.append(
                    f"{loop.name}: cached={sorted(loop._cached_outputs)} "
                    f"accumulated={sorted(loop._accumulated_cache)}"
                )
            if loop._ingested_external_inputs:
                leftovers.append(
                    f"{loop.name}: held={[e.name for e in loop._ingested_external_inputs]}"
                )
        registry = run.io.wg_state_registry
        if registry.is_done or registry._num_completed_entities:
            leftovers.append(
                f"registry: done={registry.is_done} "
                f"completed={registry._num_completed_entities}"
            )
        if registry.ready_names or registry.ready_next_iter:
            leftovers.append(
                f"registry queues: ready={sorted(registry.ready_names)} "
                f"next={sorted(registry.ready_next_iter)}"
            )
        require(
            "state.clear_empties_the_graph",
            not leftovers,
            f"{request_id} kept state after the pass: {leftovers}",
        )

    # -- the end of a case ---------------------------------------------------

    def final_check(self) -> None:
        """Drain every request, so a pass that no op finished still counts."""
        for slot in self.slots:
            run = self.runs[slot]
            if not run.in_flight:
                continue
            steps = 0
            while run.in_flight and steps < self.step_bound:
                ready = run.ready()
                if not ready:
                    break
                ran = self._run_node(slot, ready[0])
                steps += 1
                for touched in ran:
                    self._settle(touched)
            require(
                "graph.pass_terminates",
                not run.in_flight,
                f"{run.request_id} did not end its pass after {steps} steps at "
                f"the end of the case; ready={run.ready()}",
            )

    # -- shrinking -----------------------------------------------------------

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        return shrink_spec(config)


def _short(values: dict[str, tuple[str, ...]]) -> str:
    """Print a map of values so that a person can compare two of them."""
    return "{" + ", ".join(
        f"{name}: [{', '.join(value[:6] for value in group)}]"
        for name, group in sorted(values.items())
    ) + "}"
