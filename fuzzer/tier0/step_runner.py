"""StepRunner: the resource lifecycle of each step of a node.

This machine tests `StepRunner`. The resources are stubs, and each stub
records the calls that it receives. The machine therefore covers the order and
the scope of those calls. It does not cover the work that a resource does.

Ops: ingest, remove, admit, plan, commit, pre_admit, pre_plan, publish,
retrieve, set_fail_admit.

Invariants
----------
runner.rejects_preplan_dependency_gap      a pre-planner whose dependency does
                                           not pre-plan is refused at build
runner.ingest_reaches_every_resource       ingest_request reaches all of them
runner.remove_reaches_every_resource       remove_request leaves no state
runner.live_request_keeps_its_state        no resource drops a live request
runner.pre_admit_is_scoped_to_preplanners  pre_admit visits pre-planners only
runner.pre_plan_is_scoped_to_preplanners   pre_plan visits pre-planners only
runner.admit_sweeps_the_whole_step         a successful admit visits every key
runner.admit_names_the_failing_resource    a failed admit blames the refuser
runner.admit_short_circuits                a failed admit stops at the refuser
runner.plan_results_match_the_step         plan results cover exactly the step
runner.plan_results_land_on_the_context    plan returns ctx.plan_results itself
runner.plan_runs_after_its_dependencies    a plan sees its dependencies' output
runner.publish_is_scoped_to_the_node       publish stays inside the node
runner.retrieve_is_scoped_to_the_node      admit_retrieve stays inside the node
runner.order_covers_every_resource         the topo order covers every key
runner.order_is_topological                a dependency is swept first
runner.order_is_stable                     the cached order matches a fresh sort
runner.quiesce_leaves_no_request_state     removing every request leaves none

Not covered:

* what a resource does: a stub allocates nothing and returns no value
* concurrency: the worker runs `pre_plan` on `plan_executor` one step ahead
  of `gpu_executor` (`mstar/worker/worker.py:2682,2710`)
* the step order of the worker: this module generates ops in any order
* the size of a CUDA-graph buffer
* the double buffer and the pre-plan slot

`StepRunner` accepts a step that is not closed under its dependencies. This
module always generates a closed step."""

from __future__ import annotations

import random
from collections.abc import Iterator

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier0 import _stubs  # noqa: F401
from mstar.engine.resources.base import PublishedInfo, Resource
from mstar.engine.resources.runner import StepRunner, topo_sort
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AllocationFailed,
    ResourceStep,
    Segment,
    StepContext,
    SubmoduleStep,
)


class _Published(PublishedInfo):
    """What one stub publishes. It records each merge into itself."""

    def __init__(self, key: str, rid: str) -> None:
        self.key = key
        self.rid = rid
        self.merged: list[str] = []

    def update(self, other: "_Published") -> None:
        self.merged.append(other.key)


class _Stub(Resource):
    """A resource that records each call from the runner.

    It also holds the state of each request. The machine then sees if
    ``remove_request`` removes that state.
    """

    def __init__(self, key: str, deps: set[str], preplan: bool) -> None:
        self.key = key
        self._deps = set(deps)
        self._preplan = preplan
        self.rids: set[str] = set()
        self.calls: list[str] = []
        self.admit_calls = 0
        self.plan_calls = 0
        self.deps_seen: set[str] = set()
        self.fail_admit = False

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError("the fuzzer constructs stubs directly")

    def depends_on(self) -> set[str]:
        return set(self._deps)

    @property
    def supports_preplan(self) -> bool:
        return self._preplan

    def ingest_request(self, rid, overrides):
        self.calls.append(f"ingest:{rid}")
        self.rids.add(rid)

    def remove_request(self, rid):
        self.calls.append(f"remove:{rid}")
        self.rids.discard(rid)

    def admit(self, step, ctx):
        del step, ctx
        self.calls.append("admit")
        self.admit_calls += 1
        if self.fail_admit:
            return AdmitOutcome(
                ok=False, ready=False,
                reason=AllocationFailed("fuzz: out of pages", 1, "main", "?"),
            )
        return ADMIT_OK

    def plan(self, step, ctx):
        del step
        self.calls.append("plan")
        self.plan_calls += 1
        self.deps_seen = set(ctx.plan_results)
        return f"{self.key}-plan"

    def commit(self, step, ctx):
        del step, ctx
        self.calls.append("commit")


# StepRunner compares the class attribute against the one on `Resource` to
# find the resources that publish or retrieve. A stub must therefore define
# these on a class. StepRunner does not see an attribute on an instance, and
# never sweeps such a stub.

class _PublishMixin:
    def publish(self, request_id):
        self.calls.append(f"publish:{request_id}")
        return _Published(self.key, request_id)


class _RetrieveMixin:
    def admit_retrieve(self, rid, node_name, graph_walk, published):
        del node_name, graph_walk, published
        self.calls.append(f"retrieve:{rid}")
        return ADMIT_OK


class _PublishingStub(_PublishMixin, _Stub):
    pass


class _RetrievingStub(_RetrieveMixin, _Stub):
    pass


class _PublishingRetrievingStub(_PublishMixin, _RetrieveMixin, _Stub):
    pass


_STUB_CLASSES = {
    (False, False): _Stub,
    (True, False): _PublishingStub,
    (False, True): _RetrievingStub,
    (True, True): _PublishingRetrievingStub,
}


class StepRunnerMachine(StateMachine):
    name = "step_runner"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        num_resources = rng.randint(1, 4)
        resources = []
        for index in range(num_resources):
            # Deps point backwards, so the graph is acyclic by construction.
            deps = [
                f"R{j}" for j in range(index) if rng.random() < 0.4
            ]
            resources.append({
                "key": f"R{index}",
                "deps": deps,
                "preplan": rng.random() < 0.3,
                "publishes": rng.random() < 0.5,
                "retrieves": rng.random() < 0.5,
            })

        keys = [r["key"] for r in resources]
        num_nodes = rng.randint(1, 2)
        nodes = {
            f"node{i}": sorted(
                {key for key in keys if rng.random() < 0.7}
            )
            for i in range(num_nodes)
        }
        return {
            "resources": resources,
            "nodes": nodes,
            "scope_nodes": rng.random() < 0.7,
        }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.specs = {r["key"]: r for r in config["resources"]}
        self.keys = list(self.specs)
        self.node_names = sorted(config["nodes"])

        self.stubs = {
            key: _STUB_CLASSES[(spec["publishes"], spec["retrieves"])](
                key=key, deps=set(spec["deps"]), preplan=spec["preplan"],
            )
            for key, spec in self.specs.items()
        }
        # The runner must reject a pre-planner whose dependency does not.
        self.expect_rejection = any(
            spec["preplan"]
            and any(not self.specs[dep]["preplan"] for dep in spec["deps"])
            for spec in self.specs.values()
        )

        node_resources = config["nodes"] if config["scope_nodes"] else None
        self.runner: StepRunner | None = None
        self.build_error: Exception | None = None
        try:
            self.runner = StepRunner(self.stubs, node_resources=node_resources)
        except ValueError as exc:
            self.build_error = exc

        require(
            "runner.rejects_preplan_dependency_gap",
            self.expect_rejection == (self.build_error is not None),
            "a resource that pre-plans while a dependency does not was "
            f"{'accepted' if self.build_error is None else 'rejected'}, "
            f"expected the opposite (config={config['resources']})",
        )

        self.rids = ["a", "b"]
        self.live_rids: set[str] = set()

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        choice = rng.random()
        step_args = (rng.randrange(1 << len(self.keys)), rng.randrange(4))
        if choice < 0.14:
            return Op("ingest", (rng.randrange(2),))
        if choice < 0.22:
            return Op("remove", (rng.randrange(2),))
        if choice < 0.42:
            return Op("admit", step_args)
        if choice < 0.62:
            return Op("plan", step_args)
        if choice < 0.72:
            return Op("commit", step_args)
        if choice < 0.78:
            return Op("pre_admit", step_args)
        if choice < 0.84:
            return Op("pre_plan", step_args)
        if choice < 0.90:
            return Op("publish", (rng.randrange(2), rng.randrange(3)))
        if choice < 0.96:
            return Op("retrieve", (rng.randrange(2), rng.randrange(3)))
        return Op("set_fail_admit", (rng.randrange(4), rng.randint(0, 1)))

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        if len(config["resources"]) > 1:
            trimmed = config["resources"][:-1]
            dropped = config["resources"][-1]["key"]
            yield {
                "resources": [
                    {**r, "deps": [d for d in r["deps"] if d != dropped]}
                    for r in trimmed
                ],
                "nodes": {
                    node: [k for k in keys if k != dropped]
                    for node, keys in config["nodes"].items()
                },
                "scope_nodes": config["scope_nodes"],
            }

    # -- helpers -------------------------------------------------------------

    def _rid(self, index: int) -> str:
        """Map an index from an op onto a request ID."""
        return self.rids[index % len(self.rids)]

    def _node(self, index: int) -> str | None:
        """Map an index from an op onto a node name.

        The index 0 gives None, which means "no node". The runner then sweeps
        every resource instead of the resources of one node.
        """
        if index % (len(self.node_names) + 1) == 0:
            return None
        return self.node_names[(index % (len(self.node_names) + 1)) - 1]

    def _step_keys(self, mask: int) -> list[str]:
        """Choose the resource keys of one step.

        The result is never empty, and it holds every dependency of every key
        that it names.
        """
        chosen = {key for i, key in enumerate(self.keys) if mask >> i & 1}
        if not chosen:
            chosen = {self.keys[0]}
        # Close under deps: a plan must not read a result nothing produced.
        changed = True
        while changed:
            changed = False
            for key in list(chosen):
                missing = set(self.specs[key]["deps"]) - chosen
                if missing:
                    chosen |= missing
                    changed = True
        order = self.runner.order if self.runner is not None else self.keys
        return [key for key in order if key in chosen]

    def _make_step(self, mask: int, rid_mask: int) -> tuple[SubmoduleStep, list[str]]:
        """Build one step from two bit masks: the resources and the requests."""
        keys = self._step_keys(mask)
        rids = [
            rid for index, rid in enumerate(self.rids) if rid_mask >> index & 1
        ] or [self.rids[0]]
        step = SubmoduleStep(
            steps={key: ResourceStep() for key in keys},
            segments=[Segment(rid, "main", 1) for rid in rids],
        )
        step.set_ctx(
            StepContext(
                request_ids=list(rids), graph_walk="w0", slot=0, capture=False,
            )
        )
        return step, keys

    # -- execution -----------------------------------------------------------

    def execute(self, op: Op) -> None:
        if self.runner is None:
            return   # the config was correctly rejected; nothing to drive

        if op.kind == "ingest":
            rid = self._rid(op.args[0])
            self.runner.ingest_request(rid)
            self.live_rids.add(rid)
            missing = [key for key, stub in self.stubs.items() if rid not in stub.rids]
            require(
                "runner.ingest_reaches_every_resource",
                not missing,
                f"ingest_request({rid}) did not reach {missing}; those "
                "resources have no state for a request that is now live",
            )

        elif op.kind == "remove":
            rid = self._rid(op.args[0])
            self.runner.remove_request(rid)
            self.live_rids.discard(rid)
            leaked = [key for key, stub in self.stubs.items() if rid in stub.rids]
            require(
                "runner.remove_reaches_every_resource",
                not leaked,
                f"remove_request({rid}) left state behind in {leaked}",
            )

        elif op.kind == "admit":
            step, keys = self._make_step(*op.args)
            before = {key: self.stubs[key].admit_calls for key in keys}
            outcome = self.runner.admit(step)
            self._check_admit_sweep(keys, before, outcome)

        elif op.kind == "pre_admit":
            step, keys = self._make_step(*op.args)
            preplan_keys = [k for k in keys if self.stubs[k].supports_preplan]
            before = {key: self.stubs[key].admit_calls for key in keys}
            self.runner.pre_admit(step)
            touched = [k for k in keys if self.stubs[k].admit_calls > before[k]]
            require(
                "runner.pre_admit_is_scoped_to_preplanners",
                set(touched) <= set(preplan_keys),
                f"pre_admit called admit on {sorted(set(touched) - set(preplan_keys))}, "
                "which do not pre-plan",
            )

        elif op.kind == "plan":
            step, keys = self._make_step(*op.args)
            results = self.runner.plan(step)
            self._check_plan_sweep(step, keys, results)

        elif op.kind == "pre_plan":
            step, keys = self._make_step(*op.args)
            preplan_keys = [k for k in keys if self.stubs[k].supports_preplan]
            before = {key: self.stubs[key].plan_calls for key in keys}
            self.runner.pre_plan(step)
            touched = [k for k in keys if self.stubs[k].plan_calls > before[k]]
            require(
                "runner.pre_plan_is_scoped_to_preplanners",
                set(touched) <= set(preplan_keys),
                f"pre_plan called plan on {sorted(set(touched) - set(preplan_keys))}, "
                "which do not pre-plan",
            )

        elif op.kind == "commit":
            step, _keys = self._make_step(*op.args)
            self.runner.commit(step)

        elif op.kind == "publish":
            rid, node = self._rid(op.args[0]), self._node(op.args[1])
            out = self.runner.publish([rid], node_name=node)
            self._check_publish_scope(rid, node, out)

        elif op.kind == "retrieve":
            rid, node = self._rid(op.args[0]), self._node(op.args[1])
            before = {key: list(stub.calls) for key, stub in self.stubs.items()}
            self.runner.admit_retrieve(rid, node, "w0", None)
            self._check_retrieve_scope(node, before)

        elif op.kind == "set_fail_admit":
            key = self.keys[op.args[0] % len(self.keys)]
            self.stubs[key].fail_admit = bool(op.args[1])

        else:
            raise AssertionError(f"unknown op {op.kind}")

    # -- per-op invariants ---------------------------------------------------

    def _check_admit_sweep(self, keys, before, outcome) -> None:
        """Check which resources ``admit`` visited, and where it stopped."""
        called = [key for key in keys if self.stubs[key].admit_calls > before[key]]
        if outcome.ok:
            require(
                "runner.admit_sweeps_the_whole_step",
                called == keys,
                f"admit succeeded but only visited {called} of {keys}",
            )
            return
        failed = outcome.failed_resource
        require(
            "runner.admit_names_the_failing_resource",
            failed in keys and self.stubs[failed].fail_admit,
            f"admit failed and blamed {failed!r}, which is not the resource "
            f"that refused (refusers: "
            f"{[k for k in keys if self.stubs[k].fail_admit]})",
        )
        require(
            "runner.admit_short_circuits",
            called == keys[: keys.index(failed) + 1],
            f"admit failed at {failed} but went on to visit "
            f"{called[keys.index(failed) + 1:]}",
        )

    def _check_plan_sweep(self, step, keys, results) -> None:
        """Check the results of ``plan`` and the order of the calls."""
        require(
            "runner.plan_results_match_the_step",
            set(results) == set(keys),
            f"plan produced results for {sorted(results)} but the step declares "
            f"{keys}; a stale entry would be read as this step's",
        )
        require(
            "runner.plan_results_land_on_the_context",
            results is step.ctx.plan_results,
            "plan returned a dict that is not the one on the context, so a "
            "dependent resource would read a different set of results",
        )
        for key in keys:
            deps_in_step = set(self.specs[key]["deps"]) & set(keys)
            require(
                "runner.plan_runs_after_its_dependencies",
                deps_in_step <= self.stubs[key].deps_seen,
                f"{key}.plan ran before "
                f"{sorted(deps_in_step - self.stubs[key].deps_seen)}, whose "
                "output it is supposed to read",
            )

    def _expected_scope(self, node: str | None, predicate) -> set[str]:
        """Give the resources that one sweep must visit for one node."""
        eligible = {key for key in self.keys if predicate(key)}
        if node is None or not self.config["scope_nodes"]:
            return eligible
        return eligible & set(self.config["nodes"][node])

    def _check_publish_scope(self, rid, node, out) -> None:
        """Check that ``publish`` stayed inside the scope of the node."""
        expected = self._expected_scope(node, lambda k: self.specs[k]["publishes"])
        require(
            "runner.publish_is_scoped_to_the_node",
            set(out[rid]) == expected,
            f"publish for node {node!r} returned {sorted(out[rid])}, expected "
            f"{sorted(expected)}; publishing another node's state moves cache "
            "that this step never touched",
        )

    def _check_retrieve_scope(self, node, before) -> None:
        """Check that ``admit_retrieve`` stayed inside the scope of the node."""
        touched = {
            key for key, stub in self.stubs.items()
            if len(stub.calls) > len(before[key])
        }
        expected = self._expected_scope(node, lambda k: self.specs[k]["retrieves"])
        require(
            "runner.retrieve_is_scoped_to_the_node",
            touched == expected,
            f"admit_retrieve for node {node!r} visited {sorted(touched)}, "
            f"expected {sorted(expected)}",
        )

    # -- standing invariants -------------------------------------------------

    def check(self) -> None:
        if self.runner is None:
            return
        for rid in sorted(self.live_rids):
            dropped = sorted(
                key for key, stub in self.stubs.items() if rid not in stub.rids
            )
            require(
                "runner.live_request_keeps_its_state",
                not dropped,
                f"{rid} is live but {dropped} no longer hold state for it; "
                "the runner swept a removal that no op asked for",
            )
        order = self.runner.order
        require(
            "runner.order_covers_every_resource",
            set(order) == set(self.keys) and len(order) == len(self.keys),
            f"topo order {order} does not cover {self.keys} exactly",
        )
        position = {key: index for index, key in enumerate(order)}
        for key in self.keys:
            for dep in self.specs[key]["deps"]:
                require(
                    "runner.order_is_topological",
                    position[dep] < position[key],
                    f"{key} is swept before its dependency {dep} in {order}",
                )
        require(
            "runner.order_is_stable",
            self.runner.order == tuple(topo_sort(self.stubs)),
            "the runner's cached order no longer matches a fresh topo sort",
        )

    def final_check(self) -> None:
        if self.runner is None:
            return
        for rid in self.rids:
            self.runner.remove_request(rid)
        leaked = {
            key: sorted(stub.rids) for key, stub in self.stubs.items() if stub.rids
        }
        require(
            "runner.quiesce_leaves_no_request_state",
            not leaked,
            f"after removing every request these resources still hold state: "
            f"{leaked}",
        )
