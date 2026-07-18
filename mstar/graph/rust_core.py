"""The graph-core translation seam: translate ``GraphSection`` trees into the Rust
walk core's spec.

``walks_to_json`` produces the JSON that ``mstar_rust.WalkSet.from_json``
compiles; a ``WalkSet.state(walk)`` is then the Rust counterpart of a
per-request ``WorkerGraphIO`` — same readiness, completion-routing, loop
iteration, and termination semantics (asserted by
``test/rust/test_walk_parity.py``, which drives both implementations with
identical event sequences).

Scope: the walk state machine only. Streaming buffer semantics and the
worker/conductor wiring stay in Python for now — this module is the
translation seam those later steps build on.
"""

from __future__ import annotations

import json
import logging
import os
from itertools import count as _count

from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    NodeCompletionOutput,
    Parallel,
    Sequential,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION

# mstar's sentinel destinations -> the Rust core's.
_DEST = {EMIT_TO_CLIENT: "EMIT_TO_CLIENT", EMPTY_DESTINATION: "EMPTY_DESTINATION"}


def edge_to_spec(edge: GraphEdge) -> dict:
    return {
        "next_node": _DEST.get(edge.next_node, edge.next_node),
        "name": edge.name,
        "persist": bool(edge.persist),
        "output_modality": edge.output_modality or None,
    }


def section_to_spec(section: GraphSection) -> dict:
    if isinstance(section, GraphNode):
        return {
            "kind": "node",
            "name": section.name,
            "input_names": sorted(section.input_names),
            "outputs": [edge_to_spec(e) for e in section.outputs],
        }
    if isinstance(section, Loop):
        return {
            "kind": "loop",
            "name": section.name,
            "body": section_to_spec(section.section),
            "max_iters": int(section.max_iters),
            "outputs": [edge_to_spec(e) for e in section.outputs],
            "accumulated_outputs": [
                edge_to_spec(e) for e in section.accumulated_outputs
            ],
        }
    if isinstance(section, (Sequential, Parallel)):
        kind = "sequential" if isinstance(section, Sequential) else "parallel"
        return {
            "kind": kind,
            "sections": [section_to_spec(s) for s in section.sections],
        }
    raise TypeError(f"unknown GraphSection type: {type(section).__name__}")


def walks_to_json(walks: dict[str, GraphSection]) -> str:
    """``{walk_name: GraphSection}`` -> the WalkSet JSON spec."""
    return json.dumps({name: section_to_spec(s) for name, s in walks.items()})


# ---------------------------------------------------------------------------
# Shadow adoption (MSTAR_RUST_WALK=shadow): the Rust walk state runs in
# lockstep with the Python WorkerGraphIO on real traffic — Python stays
# authoritative, every event is mirrored, and ready-set / doneness / loop
# divergence is reported loudly. This is the pre-authority adoption step:
# it exercises the Rust core against every real model's walks in situ.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_WALKSET_CACHE: dict[int, tuple] = {}  # id(section) -> (section ref, WalkSet)


def rust_walk_mode() -> str:
    """``MSTAR_RUST_WALK``: ``0`` (default, off), ``shadow`` (mirror +
    compare, Python authoritative), or ``1`` (Rust decisions authoritative;
    Python keeps executing values; divergence falls back per-request)."""
    mode = os.getenv("MSTAR_RUST_WALK", "0").lower()
    if mode not in ("0", "shadow", "1", "pure"):
        raise ValueError(
            f"MSTAR_RUST_WALK must be 0, shadow, 1, or pure; got {mode!r}")
    return mode


class _RustReadyView(set):
    """The ready set served from the Rust state in authority mode. Callers
    mutate it (``pop_ready_nodes`` discards; OOM push-back adds) — discard
    marks the node scheduled in Rust; add cannot be modeled (no unschedule),
    so it falls the request back to Python."""

    def __init__(self, names, owner):
        super().__init__(names)
        self._owner = owner

    def discard(self, name):
        if name in self:
            self._owner._mark_scheduled(name)
        super().discard(name)

    def add(self, name):
        self._owner._suspend("ready push-back (OOM hold) not modeled")
        super().add(name)


class _RegistryShim:
    """`wg_state_registry.is_done` view honoring the current authority."""

    def __init__(self, owner):
        self._owner = owner

    @property
    def is_done(self):
        return self._owner._done()

    def __getattr__(self, name):
        return getattr(self._owner._io.wg_state_registry, name)


class ShadowedWorkerGraphIO:
    """A ``WorkerGraphIO`` with a Rust ``WalkState`` shadowing every event.

    Delegates everything to the Python io (authoritative). Mutating calls are
    mirrored into the Rust state; after each, ready sets, doneness, and loop
    indices are compared. A divergence logs an error (or raises with
    ``MSTAR_RUST_WALK_STRICT=1``). Events the Rust core does not model yet
    (streaming-buffer edges, speculative buffers) suspend comparison for the
    request with a single logged reason rather than false-positive.
    """

    def __init__(self, io, section, wg_id, authority: bool = False):
        from mstar_rust import WalkSet

        self._io = io
        self._authority = authority
        self._scheduled: set[str] = set()
        if authority:
            self.wg_state_registry = _RegistryShim(self)
        entry = _WALKSET_CACHE.get(id(section))
        if entry is None:
            walkset = WalkSet.from_json(walks_to_json({"walk": section}))
            _WALKSET_CACHE[id(section)] = (section, walkset)
        else:
            walkset = entry[1]
        self._walkset_key = id(section)
        self._rs = walkset.state("walk")
        self._wg_id = wg_id
        # (name, next_node) pairs produced INSIDE this graph: the Rust core
        # routes them itself on complete(); the worker re-ingests them into
        # the Python io, and mirroring that would double-ingest.
        self._internal: set[tuple[str, str]] = set()
        for node in io.nodes.values():
            for edge in node.outputs:
                self._internal.add((edge.name, edge.next_node))
        for loop in io.loops.values():
            for edge in list(loop.outputs) + list(loop.accumulated_outputs):
                self._internal.add((edge.name, edge.next_node))
        self._suspended: str | None = None
        # Internal edges from the last completion the worker has yet to
        # re-ingest: Rust routes them inside complete(), Python's worker
        # re-ingests them after — compare only once both have settled. A
        # multiset of (name, next_node): the same pair can also arrive as an
        # EXTERNAL seed (e.g. iteration 0 of a loop-back input), which must
        # be mirrored, so membership here is what says "already routed".
        self._pending_edges: list[tuple[str, str]] = []
        self._strict = os.getenv("MSTAR_RUST_WALK_STRICT") == "1"

    def __getattr__(self, name):
        return getattr(self._io, name)

    # -- authority views ------------------------------------------------------

    def _rust_drives(self) -> bool:
        return self._authority and self._suspended is None

    @property
    def ready_node_names(self):
        if self._rust_drives():
            return _RustReadyView(self._rs.ready_nodes(), self)
        return self._io.ready_node_names

    def _done(self) -> bool:
        if self._rust_drives():
            return self._rs.is_done()
        return self._io.wg_state_registry.is_done

    def get_loop_indices(self):
        if self._rust_drives():
            return dict(self._rs.loop_iters())
        return self._io.get_loop_indices()

    def _mark_scheduled(self, name: str) -> None:
        if self._suspended is None and name not in self._scheduled:
            try:
                self._rs.schedule(name)
                self._scheduled.add(name)
            except Exception as e:  # noqa: BLE001
                self._suspend(f"schedule rejected: {e!r}")
        # keep the Python io in step (it stays the value store)
        self._io.ready_node_names.discard(name)

    # -- mirroring -----------------------------------------------------------

    def _suspend(self, reason: str) -> None:
        if self._suspended is None:
            self._suspended = reason
            logger.info("rust-walk shadow suspended (%s): %s",
                        self._wg_id, reason)

    def _check(self, event: str) -> None:
        if self._suspended:
            return
        py_ready = sorted(self._io.ready_node_names)
        rs_ready = sorted(self._rs.ready_nodes())
        py_done = self._io.wg_state_registry.is_done
        rs_done = self._rs.is_done()
        rs_iters = dict(self._rs.loop_iters())
        py_iters = self._io.get_loop_indices()
        if py_ready != rs_ready or py_done != rs_done or py_iters != rs_iters:
            msg = (f"rust-walk shadow divergence after {event} ({self._wg_id}): "
                   f"ready py={py_ready} rs={rs_ready}; "
                   f"done py={py_done} rs={rs_done}; "
                   f"iters py={py_iters} rs={rs_iters}")
            if self._strict:
                raise AssertionError(msg)
            logger.error(msg)
            self._suspend("diverged; comparison stopped for this request")

    def ingest_input(self, graph_edge, can_buffer: bool = True) -> bool:
        claimed = self._io.ingest_input(graph_edge, can_buffer)
        if claimed:
            if getattr(graph_edge, "is_streaming", False):
                self._suspend("streaming edge (buffer semantics stay Python)")
            elif (pair := (graph_edge.name, graph_edge.next_node)) in \
                    self._pending_edges:
                # Rust already routed this inside complete(); Python is
                # catching up now.
                self._pending_edges.remove(pair)
            else:
                try:
                    self._rs.seed([(graph_edge.next_node, graph_edge.name)])
                except Exception as e:  # noqa: BLE001
                    self._suspend(f"seed rejected: {e!r}")
            if not self._pending_edges:
                self._check(
                    f"ingest {graph_edge.name}->{graph_edge.next_node}")
        return claimed

    def mark_node_complete(self, node_name: str):
        completion = self._io.mark_node_complete(node_name)
        if self._suspended is None:
            try:
                if node_name not in self._scheduled:
                    self._rs.schedule(node_name)
                self._scheduled.discard(node_name)
                self._rs.complete(node_name)
            except Exception as e:  # noqa: BLE001
                self._suspend(f"complete rejected: {e!r}")
            # EVERYTHING a completion hands back for local re-ingest is
            # already accounted for inside the Rust complete(): routed
            # internal edges, loop-back promotions, AND the re-injected
            # external inputs mstar re-emits on a loop advance.
            self._pending_edges = [
                (e.name, e.next_node) for e in completion.output_edges
                if e.next_node in self._io.nodes]
            if not self._pending_edges:
                self._check(f"complete {node_name}")
        return completion

    def register_loop_finish_signal(self, loop_name: str):
        self._io.register_loop_finish_signal(loop_name)
        if self._suspended is None and loop_name in self._io.loops:
            try:
                self._rs.signal_loop_finish(loop_name)
            except Exception as e:  # noqa: BLE001
                self._suspend(f"loop signal rejected: {e!r}")

    def clear(self):
        # End-of-pass reset for multi-forward-pass requests: fresh Rust
        # state (the worker re-ingests the next pass's inputs), comparison
        # re-armed.
        self._io.clear()
        walkset = _WALKSET_CACHE[self._walkset_key][1]
        self._rs = walkset.state("walk")
        self._suspended = None

    def ingest_for_speculation(self, edges, source_node):
        # Speculative buffers are a Python-side concern (not modeled yet);
        # they don't mutate real readiness, so no mirror and no suspend.
        return self._io.ingest_for_speculation(edges, source_node)


def wrap_worker_graph_io(io, section, wg_id):
    """The adoption seam: wrap a fresh per-request WorkerGraphIO according to
    ``MSTAR_RUST_WALK``. ``0`` returns it untouched; ``shadow`` mirrors with
    Python authoritative; ``1`` serves decisions (ready set, doneness, loop
    indices) from the Rust state — Python keeps executing values, comparison
    stays on, and any divergence or unmodeled event falls the request back
    to Python with an error logged."""
    mode = rust_walk_mode()
    if mode == "pure":
        # No Python registries at all. Streaming graphs and speculation are
        # not modeled yet: graphs with stream consumers fall back to
        # authority mode (Rust decisions, Python values) with a logged note.
        if any(getattr(n, "consumes_stream", False)
               for n in io.nodes.values()):
            logger.info("rust-walk pure: %s consumes streams; using "
                        "authority mode for it", wg_id)
            return ShadowedWorkerGraphIO(io, section, wg_id, authority=True)
        return PureRustWorkerGraphIO(section, wg_id)
    if mode in ("shadow", "1"):
        return ShadowedWorkerGraphIO(io, section, wg_id,
                                     authority=(mode == "1"))
    return io


# ---------------------------------------------------------------------------
# Pure mode (MSTAR_RUST_WALK=pure): no Python WorkerGraphIO at all — the Rust
# WalkState owns readiness/loops/termination, and this adapter keeps only the
# value store (uuid -> GraphEdge, the mstar-rs pattern) plus the node views
# the worker's execution path reads. Streaming graphs and speculation fall
# back / disable (documented; those ports are staged separately).
# ---------------------------------------------------------------------------

class _Slot:
    """ready_signals-shaped view: just the fields execution reads."""

    def __init__(self):
        self.ready_inputs: dict = {}
        self.ready_names: set = set()

    def clear(self):
        self.ready_inputs.clear()
        self.ready_names.clear()


class _NodeView:
    """Per-request node facade satisfying worker execution reads."""

    def __init__(self, node):
        self.name = node.name
        self.input_names = node.input_names
        self.outputs = node.outputs
        self.enable_async_scheduling = getattr(
            node, "enable_async_scheduling", True)
        self.consumes_stream = getattr(node, "consumes_stream", False)
        self.ready_signals = _Slot()
        self.ready_next_iter = _Slot()


class _PureRegistryShim:
    __slots__ = ("_io",)

    def __init__(self, io):
        self._io = io

    @property
    def is_done(self):
        return self._io._rs.is_done()


class PureRustWorkerGraphIO:
    """`WorkerGraphIO` with the Rust core as the ONLY state machine.

    Values: every ingested/produced edge gets a uuid; the Rust state carries
    uuids, this adapter maps them back to `GraphEdge`s (whose `tensor_info`
    the worker fills after execution — mutation is visible at loop
    termination when events return the captured uuids). Per (node, input)
    the adapter keeps a FIFO of pending edges; scheduling a node moves the
    oldest edge per input into its `ready_signals` view — FIFO order matches
    iteration order because completion re-emits loop externals exactly as
    mstar's registries do.
    """

    def __init__(self, section, wg_id):
        from mstar_rust import WalkSet

        entry = _WALKSET_CACHE.get(id(section))
        if entry is None:
            walkset = WalkSet.from_json(walks_to_json({"walk": section}))
            _WALKSET_CACHE[id(section)] = (section, walkset)
        else:
            walkset = entry[1]
        self._rs = walkset.state("walk")
        self.graph = section
        self.wg_id = wg_id
        self.num_times_run = 0
        self._graph_nodes = section.get_nodes()
        self.loops = section.get_loops()
        self.nodes = {n: _NodeView(g) for n, g in self._graph_nodes.items()}
        self._uuid = _count(1)
        self._store: dict[int, GraphEdge] = {}
        # (node, input) -> FIFO of pending ingested edges
        self._pending: dict[tuple[str, str], list[GraphEdge]] = {}
        self._scheduled: set[str] = set()
        # externals per loop, keyed by (node, input): re-emitted on each
        # advance (mstar's contract). Keyed — not appended — because the
        # re-emitted edges come back through ingest_input; appending would
        # double the list every iteration.
        self._loop_externals: dict[str, dict[tuple, GraphEdge]] = {}
        self._loop_members = {
            ln: set(lp.section.get_nodes().keys())
            for ln, lp in self.loops.items()
        }
        self._internal = {
            (e.name, e.next_node)
            for g in self._graph_nodes.values() for e in g.outputs
        }
        self.ready_for_streaming: set = set()
        self._registry_shim = _PureRegistryShim(self)
        # complete_full return values cached between completions; ingest and
        # schedule dirty the ready cache (recomputed lazily).
        self._loop_cache = {n: (i, t) for n, i, t in self._rs.loop_states()}
        self._ready_cache: list | None = []
        self._done_cache = False
        # ids of edges WE re-emitted on the last loop advance: Rust already
        # re-injected those values internally, so their re-ingest must only
        # refill the Python FIFO views — re-seeding Rust would append to the
        # loop's external_inputs every iteration (quadratic complete cost).
        self._reemitted_ids: set[int] = set()
        self._tm = None
        self._rid = None

    # -- registry-shaped surface ----------------------------------------------

    @property
    def ready_node_names(self):
        if self._ready_cache is None:
            self._ready_cache = self._rs.ready_nodes()
        return _RustReadyView(
            (n for n in self._ready_cache if n not in self._scheduled),
            self)

    @property
    def wg_state_registry(self):
        return self._registry_shim

    def get_loop_indices(self):
        return dict(self._rs.loop_iters())

    def register_communication_info(self, tm, rid):
        self._tm, self._rid = tm, rid

    def get_node(self, name):
        return self.nodes[name]

    # -- events ----------------------------------------------------------------

    def _mark_scheduled(self, name):
        if name in self._scheduled:
            return
        self._rs.schedule(name)
        self._scheduled.add(name)
        view = self.nodes[name]
        view.ready_signals.clear()
        for inp in view.input_names:
            fifo = self._pending.get((name, inp))
            if fifo:
                edge = fifo.pop(0)
                view.ready_signals.ready_inputs[inp] = edge
                view.ready_signals.ready_names.add(inp)

    def ingest_input(self, edge, can_buffer=True):
        if edge.next_node not in self.nodes:
            return False
        if edge.name not in self.nodes[edge.next_node].input_names:
            return False
        self._pending.setdefault((edge.next_node, edge.name), []).append(edge)
        if id(edge) in self._reemitted_ids:
            self._reemitted_ids.discard(id(edge))
        else:
            uuid = next(self._uuid)
            self._store[uuid] = edge
            self._rs.seed_with([(edge.next_node, edge.name, uuid)])
        self._ready_cache = None  # readiness may have changed
        # track loop externals for re-emission on advance
        pair = (edge.name, edge.next_node)
        if pair not in self._internal:
            for ln, members in self._loop_members.items():
                if edge.next_node in members:
                    self._loop_externals.setdefault(ln, {})[
                        (edge.next_node, edge.name)] = edge
        return True

    def mark_node_complete(self, node_name):
        before = self._loop_cache
        self._mark_scheduled(node_name)  # idempotent
        self._scheduled.discard(node_name)
        view = self.nodes[node_name]
        # Return the declared edge objects themselves — mstar's registries do
        # the same (the worker overwrites tensor_info each pass); fresh uuids
        # keep the Rust-side value identity per iteration. COPY the list
        # (sharing the objects): the loop-advance path extends the
        # completion's list, and sharing it would grow the node's declared
        # outputs every iteration.
        out_edges = list(view.outputs)
        outputs = []
        for e in out_edges:
            uuid = next(self._uuid)
            self._store[uuid] = e
            outputs.append((e.name, [uuid]))
        events, done, ready, loop_states = self._rs.complete_full(
            node_name, outputs)
        self._ready_cache = ready
        self._done_cache = done
        after = {n: (i, t) for n, i, t in loop_states}
        self._loop_cache = after

        completion = NodeCompletionOutput(output_edges=out_edges)
        for ln in self.loops:
            b_i, b_t = before.get(ln, (0, False))
            a_i, a_t = after.get(ln, (0, False))
            if a_t and not b_t:  # terminated now: filter its loop-backs
                completion.filtered_signals |= self.loops[ln]._loop_back_inputs
                # A dead loop's nodes must show no stale readiness (the EOS
                # contract: termination clears ready signals): drop their
                # buffered inputs and blank the views.
                for member in self._loop_members.get(ln, ()):
                    view = self.nodes.get(member)
                    if view is None:
                        continue
                    view.ready_signals.clear()
                    view.ready_next_iter.clear()
                    for inp in view.input_names:
                        self._pending.pop((member, inp), None)
                # loop outputs with captured values (uuids -> stored edges)
                for kind, name, _tgt, uuids in events:
                    del kind, name, uuids
                for le in self.loops[ln].outputs:
                    src = [e for e in out_edges if e.name == le.name]
                    lo = le.clone()
                    if src:
                        lo.tensor_info = src[0].tensor_info
                        lo._src_edge = src[0]
                    completion.output_edges.append(lo)
            elif a_i > b_i:  # advanced: re-emit external inputs
                reemit = list(self._loop_externals.get(ln, {}).values())
                completion.output_edges.extend(reemit)
        # EVERY locally-destined edge in this completion was already routed
        # (or re-injected) inside the Rust complete — their re-ingest by the
        # worker must only refill the Python FIFO views, never re-seed Rust
        # (a seed of a loop-member input appends to the loop's
        # external_inputs, growing every iteration).
        self._reemitted_ids.update(
            id(e) for e in completion.output_edges
            if e.next_node in self.nodes)
        return completion

    def register_loop_finish_signal(self, loop_name):
        if loop_name in self.loops:
            self._rs.signal_loop_finish(loop_name)

    def ingest_for_speculation(self, edges, source_node):
        return []  # speculation disabled in pure mode (staged separately)

    def clear_speculative_inputs(self):
        pass

    def clear(self):
        walkset = _WALKSET_CACHE[id(self.graph)][1]
        self._rs = walkset.state("walk")
        self._scheduled.clear()
        self._pending.clear()
        self._loop_externals.clear()
        for v in self.nodes.values():
            v.ready_signals.clear()
            v.ready_next_iter.clear()
        self.num_times_run += 1
