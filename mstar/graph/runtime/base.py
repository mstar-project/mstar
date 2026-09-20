"""The graph-runtime harness: one interface, a Python and a Rust implementation.

Scope is **one worker graph** — the same scope as ``WorkerGraphQueues`` and as
a compiled Rust ``GraphRuntime`` — holding the walk state of every request
registered with it.

Two call styles, deliberately:

* **per-request** (``ingest``, ``complete``) mirrors what ``worker.py`` does
  today, so the shadow implementation can mirror live traffic event-for-event.
* **batched** (``complete_and_route_batch``) is the target API, where one call
  serves the whole forward pass. The Python implementation provides it as a
  loop; the win only shows up under the Rust one.

``snapshot()`` is the parity primitive: everything an implementation is
expected to agree on, in a comparable form.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mstar.graph.base import (
    GraphEdge,
    NameAndDest,
    NodeCompletionOutput,
    SpeculativeNodeInfo,
)
from mstar.graph.loop_indices import NestedLoopIndices

#: Fields two backends are expected to agree on.
#:
#: ``ready_for_streaming`` is NOT among them. Python's
#: ``ReadySignals.is_ready_for_streaming`` (graph/base.py:183) guards with
#: ``input_names.issuperset(ready_names | streaming_inputs)``, but both of
#: those are always subsets of ``input_names``, so the test is trivially true
#: — every node is "ready for streaming" after its first ingested input,
#: streaming or not. The comment above it states the intent (``issubset``:
#: ready once the only missing inputs are streaming). The Rust core implements
#: the intent, so the two disagree by construction. Replicating the bug would
#: bake it into the new core; fixing Python is a separate, behavior-affecting
#: change that streaming models have to be re-validated against.
#: See docs/design/rust_graph_port §7.8.
COMPARED_FIELDS = ("ready", "loop_indices", "is_done", "num_times_run")
ALL_FIELDS = COMPARED_FIELDS + ("ready_for_streaming",)


@dataclass(frozen=True)
class InputSlot:
    """One ingested input, as the worker needs it.

    Only uuids, not ``TensorPointerInfo``: the caller looks the tensor up in
    the tensor manager by uuid, and the Rust backend carries interned handles
    rather than the full metadata.
    """
    uuids: list[str]
    final_stream_chunk: bool = False


@dataclass(frozen=True)
class RequestSnapshot:
    """Everything two implementations could agree on for one request.

    Deliberately excludes tensor identity: the Rust core carries interned
    handles, and which uuid sits in which slot is the tensor manager's
    business, not the walk machine's.
    """
    ready: frozenset[str]
    ready_for_streaming: frozenset[str]
    loop_indices: tuple[tuple[str, int], ...]
    is_done: bool
    num_times_run: int

    def diff(
        self, other: "RequestSnapshot", fields: tuple[str, ...] = COMPARED_FIELDS,
    ) -> list[str]:
        out = []
        for f in fields:
            a, b = getattr(self, f), getattr(other, f)
            if a != b:
                out.append(f"{f}: {a!r} != {b!r}")
        return out


@dataclass
class BatchRouting:
    """What a completed batch produced, grouped for batched consumption.

    ``to_workers`` holds edges for peer workers. Under the Rust runtime these
    can be handed over already encoded (see docs/design/rust_graph_port §3.1);
    today both implementations return ``GraphEdge``s so the two are comparable
    and the worker's send path is unchanged.
    """
    # rid -> edges that were re-ingested into this worker's own graphs
    routed_local: dict[str, list[GraphEdge]] = field(default_factory=dict)
    # worker id -> [(rid, edge)]
    to_workers: dict[str, list[tuple[str, GraphEdge]]] = field(default_factory=dict)
    persist: dict[str, list[GraphEdge]] = field(default_factory=dict)
    emit_to_client: dict[str, list[GraphEdge]] = field(default_factory=dict)
    new_token_outputs: dict[str, list[GraphEdge]] = field(default_factory=dict)
    # rids whose worker graph finished this pass
    completed: list[str] = field(default_factory=list)


class GraphRuntimeBase(ABC):
    """Walk state for every request on one worker graph."""

    #: Identifies the implementation in logs and divergence reports.
    backend: str = "?"

    # -- lifecycle ----------------------------------------------------------

    @abstractmethod
    def add_requests(self, rids: list[str]) -> None:
        """Register requests. Replaces ``WorkerGraphQueues.add_request``."""

    @abstractmethod
    def remove_requests(self, rids: list[str]) -> None:
        ...

    @abstractmethod
    def has_request(self, rid: str) -> bool:
        ...

    @abstractmethod
    def known_requests(self) -> list[str]:
        """Every registered rid. The graph layer's own registry, so callers
        need not consult ``per_request_queues`` (empty under the Rust
        backend)."""

    # -- ingest -------------------------------------------------------------

    @abstractmethod
    def ingest(self, rid: str, edges: list[GraphEdge], can_buffer: bool = True) -> list[GraphEdge]:
        """Route arriving edges into this graph's state.

        Returns the edges NOT claimed here, so the caller can try the next
        worker graph. Mirrors ``WorkerGraphQueues.process_new_inputs``.
        """

    # -- completion ---------------------------------------------------------

    @abstractmethod
    def complete(self, rid: str, node_name: str) -> NodeCompletionOutput:
        """Mark a node complete for one request.

        ``output_edges`` already has any finishing loop's loop-back edges
        removed; ``filtered_signals`` reports which (name, dest) pairs those
        were. Mirrors ``WorkerGraphIO.mark_node_complete`` — the caller routes.
        """

    @abstractmethod
    def complete_and_route_batch(
        self, node_name: str, rids: list[str], graph_walk: str,
    ) -> BatchRouting:
        """Complete ``node_name`` for every rid AND route the outputs.

        The batched path: one call per forward pass rather than one per
        request. Output tensor_info must already be on the nodes' output
        edges (``tensor_manager.store_and_populate_graph_edges``).
        """

    # -- scheduling ---------------------------------------------------------

    @abstractmethod
    def ready_nodes(self, rid: str) -> set[str]:
        ...

    @abstractmethod
    def ready_for_streaming(self, rid: str) -> set[str]:
        ...

    @abstractmethod
    def input_slots(
        self, rid: str, node_name: str, next_iter: bool = False,
    ) -> dict[str, InputSlot]:
        """The inputs a node has received, by signal name.

        ``next_iter=True`` reads the buffered next-iteration slot instead —
        what the speculation path runs against.
        """

    @abstractmethod
    def pop_ready(self, node_name: str, rids: list[str]) -> list[str]:
        """Take ``node_name`` off the ready set for these rids; returns the
        rids actually popped."""

    @abstractmethod
    def push_back(self, node_name: str, rids: list[str]) -> None:
        """Undo a pop (OOM hold)."""

    # -- loops --------------------------------------------------------------

    @abstractmethod
    def stop_loops(self, rid: str, loop_names: set[str]) -> set[NameAndDest]:
        """Register external finish signals; returns the union of the loops'
        loop-back (name, dest) pairs to drop from this iteration's routing."""

    @abstractmethod
    def loop_indices(self, rid: str) -> dict[str, int]:
        ...

    @abstractmethod
    def nested_loop_idxs_for_node(self, rid: str, node_name: str) -> NestedLoopIndices:
        ...

    @abstractmethod
    def nested_loop_idxs(self, rid: str, loop_name: str) -> NestedLoopIndices:
        """Snapshot of where execution sits across the loops enclosing
        ``loop_name`` (outer -> inner)."""

    @abstractmethod
    def loop_names(self) -> set[str]:
        """Every loop in this worker graph."""

    @abstractmethod
    def is_done(self, rid: str) -> bool:
        ...

    @abstractmethod
    def reset(self, rid: str) -> None:
        """End of a full forward pass; state is reusable for the next one."""

    @abstractmethod
    def num_times_run(self, rid: str) -> int:
        ...

    # -- speculation --------------------------------------------------------

    @abstractmethod
    def ingest_for_speculation(
        self, rid: str, edges: list[GraphEdge], source_node: str,
    ) -> list[SpeculativeNodeInfo]:
        ...

    @abstractmethod
    def clear_speculative_inputs(self, rid: str) -> None:
        ...

    # -- parity -------------------------------------------------------------

    def snapshot(self, rid: str) -> RequestSnapshot:
        return RequestSnapshot(
            ready=frozenset(self.ready_nodes(rid)),
            ready_for_streaming=frozenset(self.ready_for_streaming(rid)),
            loop_indices=tuple(sorted(self.loop_indices(rid).items())),
            is_done=self.is_done(rid),
            num_times_run=self.num_times_run(rid),
        )
