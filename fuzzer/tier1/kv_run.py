"""A generated model over the real resources of mstar.

``model_run`` runs a generated model over the graph layer. It computes the
value of a step from the identity and the inputs of that step. This machine
runs the same generated model, with the same schedule and the same oracle. It
takes the value of a step from a real step of the resource layer:

* the resources are a real ``KVManager`` and, when the model declares one, a
  real ``PositionManager``. ``spec.resource_class.build`` makes them, which is
  the call that ``EngineManager.build`` makes. The position resource names the
  cache in ``depends_on``. A case therefore also drives
  ``resolve_spec_dependencies`` and the topological order of the runner.
* the cycle is a real ``StepRunner``: ``ingest_request``, ``admit``, ``plan``,
  ``commit``, ``publish``, ``remove_request``, and ``pre_admit``,
  ``pre_plan``, ``clear_preplan``
* the pages are real pages of a real ``KVCache``, on the CPU, in ``int32``

Tier 0 drives this same lifecycle against stubs. It covers the order and the
scope of the calls, and no behavior. This machine covers the behavior: what a
page holds, which page a stream gets, and whether a page goes back.

The checksum comes from the cache
---------------------------------

A step reads the tokens that its own stream holds before it writes. That
number is part of every value the step gives. A step that covers its identity
alone reads no page. A resource can then give one page to two requests,
address the wrong page, or keep a page across a free, and every oracle passes.

One step runs a batch
---------------------

One step of a node runs every request that is ready on that node. It puts one
``Segment`` for each request into one ``SubmoduleStep``, as the micro
scheduler does. The requests then share a plan, a packed token order and a
cache. Data that crosses between two requests is therefore something an
oracle sees here. It is not visible in ``model_run``, where each request holds
its own copy of the graph and nothing more.

Pre-planning, capture and forks
-------------------------------

Tier 0 reaches none of these three. It drives stubs, so it has no cache to
fork, no buffer to capture into, and no staged state to give back.

* Capture. A model may declare a bucket. Every step then runs under a
  ``SlotLease``. The slot's own rows pad the batch, and the plan goes through
  the static addressing of that slot. The rows that hold no token must
  address ``SINK_PAGE``. The machine writes those rows the way a replay does,
  and reports them when they reach a page of a live stream.
* Pre-planning. The ``preplan`` op stages a step through ``pre_admit`` and
  ``pre_plan``, one step ahead. The engine stages a step under a lease only,
  and so does the machine. A ``step`` op for the same node and the same batch
  then promotes the staged step. An ``abandon`` op drops it. A ``step`` op
  for anything else also drops it, which is what ``preplan_is_stale`` means.
* Forks. A step may carry a ``pre_fork`` or a ``post_fork``. Each one copies
  the stream of the step onto another label. A pre-fork copies in ``plan``,
  before the spans of the step land. A post-fork copies in ``commit``, after
  them.

The three meet in one place. A pre-fork inside a staged step copies pages
before anything knows whether the step runs. ``clear_preplan`` must give that
state back.

Ops: start, step, drain, abort, preplan, abandon.

Invariants
----------
emit.runs_match_reference      each node ran the number of times that the
                               interpreter gives
emit.matches_reference         the values that reach the client are the values
                               that the interpreter gives, and they cover what
                               each step read out of the cache
route.every_edge_lands         every edge that a node sends reaches a node of
                               this graph or the client
graph.pass_terminates          a pass that started always ends
state.clear_empties_the_graph  after a pass, every slot, every loop and the
                               registry are empty again
kv.admit_succeeds              the cache is sized for the whole case, so no
                               step is ever refused
kv.admit_gates_the_plan        a step that admit accepted plans and commits
                               without raising
kv.readback_matches_the_write  the slots a step wrote read back as written
kv.publish_matches_the_stream  what a step publishes is the length and the
                               pages that the stream holds
pos.ids_match_the_stream       the position ids of a step are the token
                               indices that its stream implies
kv.stream_matches_reference    after a pass, each stream holds the number of
                               tokens that the interpreter gives
capture.padding_stays_off_live_pages
                               the rows of a bucket that hold no token address
                               the sink page, not a page of a live stream
preplan.staging_does_not_raise
                               a step that is staged a step ahead is admitted
                               and planned without raising
preplan.abandon_restores_state
                               after a staged step is dropped, every stream
                               holds the length, the mark and the counter it
                               held before, keeps its label, and keeps or
                               grows its pages
kv.pages_are_not_shared        no page is held by two streams, and no held
                               page is also free
kv.pages_are_conserved         the held pages and the free pages together are
                               every page of the cache
kv.remove_frees_the_pages      after the last request goes, the cache holds
                               only its sink page

Not covered
-----------

* eviction and offload under real pressure. The cache takes a size that lets
  every admit succeed. An oracle for a refused admit needs a model of
  eviction, which this tier does not hold. ``kv.admit_succeeds`` holds the
  size, so pressure is a gap and not a silent pass.
* the attention resources over this cache, and the sampler. Those need
  kernels that the CPU does not have.
* retention, combined labels, and more than one cache in one model.
* tensor parallelism: one rank, one walk. Publish and retrieve never cross a
  device.
* the two threads of the pre-plan path. This machine stages a step and then
  promotes or drops it, on one thread. ``kv_race`` drives several threads.
"""

from __future__ import annotations

import random

from fuzzer.common.case import Op
from fuzzer.common.machine import require
from fuzzer.tier1.model_run import ModelRunMachine
from fuzzer.tier1.resources import ResourceRig
from fuzzer.tier1.spec import gen_spec


class KvRunMachine(ModelRunMachine):
    name = "kv_run"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        return gen_spec(rng, with_resources=True)

    def __init__(self, config: dict) -> None:
        self.rig = ResourceRig(
            config, [node.name for node in plan_nodes(config)],
        )
        self.max_passes = config["kv"]["max_passes"]
        super().__init__(config)

    # -- the lifecycle of a request -----------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        choice = rng.random()
        if choice < 0.12:
            return Op("preplan", (rng.randrange(len(self.slots)), rng.randint(0, 3)))
        if choice < 0.17:
            return Op("abandon", ())
        return super().gen_op(rng)

    def execute(self, op: Op) -> None:
        if op.kind == "preplan":
            self._stage(op.args[0] % len(self.slots), op.args[1])
            return
        if op.kind == "abandon":
            self._abandon()
            return
        super().execute(op)

    def _stage(self, slot: int, choice: int) -> None:
        """Stage the step of a ready node, a step ahead of its own run."""
        run = self.runs[slot]
        if not run.in_flight:
            return
        ready = run.ready()
        if not ready:
            return
        node_name = ready[choice % len(ready)]
        batch = self._batch_for(slot, node_name)
        request_ids = [self.runs[other].request_id for other in batch]
        # A staged step that makes a new label meets the open failure
        # ``abandoned-preplan-keeps-its-new-label``. ``clear_preplan`` keeps
        # that label and the pages it reserved. The corpus holds that shape,
        # and its config sets ``stage_new_labels``.
        if not self.spec["kv"].get("stage_new_labels"):
            label = self.rig.labels[node_name]
            lengths = self.rig.stream_lengths()
            if any((rid, label) not in lengths for rid in request_ids):
                return
        self.rig.stage_step(node_name, request_ids)
        require(
            "preplan.staging_does_not_raise",
            self.rig.staged_raised is None,
            f"staging a step of {node_name} raised: {self.rig.staged_raised}",
        )

    def _abandon(self) -> None:
        """Drop a staged step, and check the state that it gives back."""
        if self.rig.staged is None:
            return
        before = self.rig.before_stage
        rows = self.rig.staged_rows
        self.rig.clear_preplan()
        self._require_restored(before, rows)

    def _require_restored(
        self, before: dict | None, rows: tuple[str, ...] = (),
    ) -> None:
        """What a dropped pre-plan must give back.

        ``clear_preplan`` keeps the pages that a reservation took for a stream
        that already existed. ``page_indices`` is a high-water mark, and the
        comment there says so. ``_alloc`` also raises the generation of a
        stream that it grows. A generation that moved with the page count is
        therefore that same kept reservation.

        Everything else must go back:

        * the length of each stream
        * the in-flight mark of each stream
        * the counter of each position stream
        * every label that the staged step made
        """
        if before is None:
            return
        after = self.rig.snapshot(rows)
        differences: list[str] = []

        for key, value in after["streams"].items():
            was = before["streams"].get(key)
            if was is None:
                differences.append(f"{key}: a new label {value} was kept")
                continue
            length, generation, in_flight, pages = value
            old_length, old_generation, old_in_flight, old_pages = was
            if (length, in_flight) != (old_length, old_in_flight):
                differences.append(
                    f"{key}: length/mark {(old_length, old_in_flight)} -> "
                    f"{(length, in_flight)}"
                )
            if pages < old_pages:
                differences.append(f"{key}: pages {old_pages} -> {pages}")
            if generation != old_generation and pages == old_pages:
                differences.append(
                    f"{key}: generation {old_generation} -> {generation} with "
                    f"no page taken"
                )
        for key in before["streams"].keys() - after["streams"].keys():
            differences.append(f"{key}: the stream is gone")
        for key, value in after["positions"].items():
            if before["positions"].get(key) != value:
                differences.append(
                    f"{key}: counter {before['positions'].get(key)} -> {value}"
                )
        require(
            "preplan.abandon_restores_state",
            not differences,
            f"a staged step that never ran left state behind: {differences}",
        )

    def _batch_for(self, slot: int, node_name: str) -> list[int]:
        """Every slot whose request is ready on this node, as one batch."""
        batch = [
            other for other in self.slots
            if self.runs[other].in_flight
            and node_name in self.runs[other].ready()
        ]
        if slot not in batch:
            batch.append(slot)
        return batch

    def _on_new_request(self, slot: int) -> None:
        self.rig.ingest(self.request_ids[slot])

    def _on_drop_request(self, slot: int, request_id: str) -> None:
        # A request that goes while a step of it is staged takes the stage
        # with it. The staged plan names a stream that is about to go.
        if self.rig.staged is not None and request_id in self.rig.staged[1]:
            self.rig.clear_preplan()
        self.rig.remove(request_id)
        # The interpreter holds the streams of a request too.
        self.reference.remove_request(request_id)

    # -- one step, for every request that is ready on the node ---------------

    def _run_node(self, slot: int, node_name: str) -> list[int]:
        """Run one node for the whole batch that is ready on it."""
        batch = self._batch_for(slot, node_name)
        # A staged step that this one does not describe is stale. Drop it
        # here, before this step runs and changes the state itself.
        stale = self.rig.staged is not None and self.rig.staged != (
            node_name,
            tuple(self.runs[other].request_id for other in batch),
        )
        if stale:
            before = self.rig.before_stage
            rows = self.rig.staged_rows
            self.rig.clear_preplan()
            self._require_restored(before, rows)

        steps = [self.runs[other].begin_node(node_name) for other in batch]
        outcome = self.rig.run_step(
            node_name, steps,
            {self.runs[other].request_id: self.runs[other].pass_index
             for other in batch},
        )
        require(
            "kv.admit_succeeds",
            outcome.admitted,
            f"the cache refused a step of {node_name} and is {outcome.pages_short} "
            f"pages short, although it is sized for the whole case; "
            f"{self.rig.free_pages()} pages are free",
        )
        require(
            "kv.admit_gates_the_plan",
            outcome.raised is None,
            f"admit accepted a step of {node_name} and the resource then "
            f"raised: {outcome.raised}",
        )
        require(
            "kv.readback_matches_the_write",
            outcome.readback_ok,
            f"a step of {node_name} read back slots that hold values it did "
            f"not write",
        )
        require(
            "kv.publish_matches_the_stream",
            not outcome.publish_disagrees,
            f"after a step of {node_name}, publish and the stream disagree: "
            f"{outcome.publish_disagrees}",
        )
        require(
            "pos.ids_match_the_stream",
            not outcome.position_disagrees,
            f"a step of {node_name} planned positions that its streams do not "
            f"imply: {outcome.position_disagrees}",
        )
        for other, step in zip(batch, steps, strict=True):
            self.runs[other].finish_node(
                node_name, outcome.values[step.request_id],
            )
        return batch

    # -- the oracles of the cache -------------------------------------------

    def check(self) -> None:
        super().check()
        require(
            "capture.padding_stays_off_live_pages",
            not self.rig.padding_hits,
            f"a replay would have written into a live stream: "
            f"{self.rig.padding_hits}",
        )
        held = self.rig.held_pages()

        owner: dict[int, tuple[str, str]] = {}
        shared: list[str] = []
        for key, pages in held.items():
            for page in pages:
                if page in owner:
                    shared.append(f"page {page}: {owner[page]} and {key}")
                owner[page] = key
        require(
            "kv.pages_are_not_shared",
            not shared,
            f"a page reached two streams: {shared}",
        )

        free = self.rig.free_pages()
        total = self.rig.config.max_num_pages
        # Page 0 is the sink. The manager takes it out of circulation at
        # construction and no stream ever holds it.
        require(
            "kv.pages_are_conserved",
            len(owner) + free + 1 == total,
            f"{len(owner)} pages are held and {free} are free, with the sink "
            f"that is {len(owner) + free + 1} of {total}",
        )

    def _settle(self, slot: int) -> None:
        finished = self.runs[slot].is_done()
        request_id = self.runs[slot].request_id
        super()._settle(slot)
        if finished:
            self._require_streams_match(request_id)
        # A request lives for a bounded number of passes, so the cache that a
        # case has to allocate stays bounded. A new request takes the slot.
        if self.passes[slot] >= self.max_passes and not self.runs[slot].in_flight:
            self._replace(slot)

    def _require_streams_match(self, request_id: str) -> None:
        """After a pass, the streams of a request hold what the interpreter
        gives. The interpreter counts tokens, and it names no page."""
        wanted = {
            label: len(tokens)
            for (rid, label), tokens in self.reference.streams.items()
            if rid == request_id
        }
        got = {
            label: length
            for (rid, label), length in self.rig.stream_lengths().items()
            if rid == request_id and length > 0
        }
        require(
            "kv.stream_matches_reference",
            got == {label: n for label, n in wanted.items() if n > 0},
            f"{request_id} holds {got} tokens; the interpreter gives {wanted}",
        )

    def final_check(self) -> None:
        super().final_check()
        for slot in self.slots:
            self._on_drop_request(slot, self.request_ids[slot])
        require(
            "kv.remove_frees_the_pages",
            self.rig.free_pages() + 1 == self.rig.config.max_num_pages,
            f"after the last request went, {self.rig.free_pages()} of "
            f"{self.rig.config.max_num_pages} pages are free; the streams that "
            f"are left hold {self.rig.held_pages()}",
        )


def plan_nodes(config: dict):
    """The nodes of a config, without building the machine first."""
    from fuzzer.tier1.spec import plan

    return plan(config).all_nodes()
