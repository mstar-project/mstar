"""The Rust-backed implementation of the graph-runtime harness.

One ``mstar_rust.GraphRuntime`` per **worker** owns every worker graph on it,
because routing has to see which *local* graph owns a destination node.
Python still talks to it one worker graph at a time: ``RustWorkerRuntimes``
holds the shared object and hands out a ``RustGraphRuntime`` view per graph,
each passing its own ``wg`` index. So the harness interface stays
per-worker-graph, matching ``WorkerGraphIO``.

Two translations happen here and nowhere else:

* **requests** become dense ``u32`` handles, worker-wide — one handle
  addresses a request in every graph it belongs to;
* **tensor uuids** (``str(uuid4())`` today) are interned to ``u64``.

The uuid interning is a bridge, not the destination: it lets the runtime be
adopted without touching the tensor manager, at the cost of a dict lookup per
tensor. Issuing ``u64`` handles from the tensor manager directly removes it —
see docs/design/rust_graph_port §4.
"""
import logging

from mstar.graph.base import (
    GraphEdge,
    GraphSection,
    NameAndDest,
    NodeCompletionOutput,
    SpeculativeNodeInfo,
    TensorPointerInfo,
)
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime.base import BatchRouting, GraphRuntimeBase, InputSlot
from mstar.graph.rust_core import try_compile_section

logger = logging.getLogger(__name__)


def rust_available() -> bool:
    try:
        import mstar_rust  # noqa: F401
    except ImportError:
        return False
    return hasattr(__import__("mstar_rust"), "GraphRuntime")


class RustWorkerRuntimes:
    """The one Rust object for a worker, plus a view per worker graph.

    Owns what is worker-wide: the compiled graphs, the request handles and
    the uuid interner.
    """

    def __init__(self, inner, wg_order: list[str]):
        self._rt = inner
        self._wg_order = wg_order
        self.handles: dict[str, int] = {}
        self.uuid_to_h: dict[str, int] = {}
        self.h_to_uuid: list[str] = []

    @classmethod
    def build(
        cls,
        sections: dict[str, GraphSection],
        worker_id: str,
        graph_walks: dict[str, set[str]],
        workers: list[str] | None = None,
        leader_nodes: set[str] | None = None,
    ) -> "RustWorkerRuntimes | None":
        """None when any graph uses something the Rust core does not model, or
        the extension is unavailable — the caller falls back to Python."""
        if not rust_available():
            logger.warning("MSTAR_RUST_GRAPH set but mstar_rust.GraphRuntime is missing")
            return None
        import mstar_rust

        specs = []
        wg_order = []
        for wg_id, section in sections.items():
            spec = try_compile_section(section, leader_nodes, wg_id=wg_id)
            if spec is None:
                return None  # all or nothing: routing needs every local graph
            specs.append({
                "wg_id": wg_id,
                "graph_walks": sorted(graph_walks.get(wg_id, ())),
                **spec,
            })
            wg_order.append(wg_id)
        inner = mstar_rust.GraphRuntime(
            worker_graphs=specs, workers=list(workers or [worker_id]), me=worker_id,
        )
        return cls(inner, wg_order)

    def view(self, wg_id: str, section: GraphSection) -> "RustGraphRuntime":
        return RustGraphRuntime(self, self._wg_order.index(wg_id), section, wg_id)

    def intern(self, uuid: str) -> int:
        h = self.uuid_to_h.get(uuid)
        if h is None:
            h = len(self.h_to_uuid)
            self.uuid_to_h[uuid] = h
            self.h_to_uuid.append(uuid)
        return h


class RustGraphRuntime(GraphRuntimeBase):
    """One worker graph's view onto the worker's shared Rust runtime."""

    backend = "rust"

    def __init__(self, shared: RustWorkerRuntimes, wg: int, section: GraphSection, wg_id: str):
        self._shared = shared
        self._rt = shared._rt
        self._wg = wg
        self._section = section
        self._wg_id = wg_id
        self._handles = shared.handles
        self._h_to_uuid = shared.h_to_uuid
        # Reconstructing a routed edge needs the declared GraphEdge it came
        # from; keyed (signal, dest) since a node may emit one signal to
        # several destinations.
        self._edge_by_key: dict[tuple[str, str], GraphEdge] = {}
        for node in section.get_nodes().values():
            for e in node.outputs:
                self._edge_by_key[(e.name, e.next_node)] = e
        for loop in section.get_loops().values():
            for e in list(loop.outputs) + list(loop.accumulated_outputs):
                self._edge_by_key[(e.name, e.next_node)] = e

    # -- interning ----------------------------------------------------------

    def _h(self, uuid: str) -> int:
        return self._shared.intern(uuid)

    def _tensors(self, infos: list[TensorPointerInfo]) -> list[tuple[int, int, int, int]]:
        return [
            (self._h(i.uuid), i.dims[0] if i.dims else 0, i.nbytes, i.offset)
            for i in infos
        ]

    def _edge_from_rust(self, name, dest, persist, new_token, streaming, modality,
                        persist_for_loop, n_tensors) -> GraphEdge:
        """Rebuild a GraphEdge from what Rust reports. tensor_info is left
        empty: the Rust core carries handles, and the authoritative tensor
        identity is Python's."""
        template = self._edge_by_key.get((name, dest))
        edge = template.clone() if template is not None else GraphEdge(
            next_node=dest, name=name, persist=persist,
            conductor_new_token=new_token, is_streaming=streaming,
            output_modality=modality,
        )
        edge.tensor_info = []
        edge._persist_for_loop = persist_for_loop
        return edge

    # -- lifecycle ----------------------------------------------------------

    def add_requests(self, rids: list[str]) -> None:
        for rid, h in zip(rids, self._rt.add_requests(self._wg, rids), strict=True):
            self._handles[rid] = h

    def remove_requests(self, rids: list[str]) -> None:
        handles = [self._handles.pop(r) for r in rids if r in self._handles]
        if handles:
            self._rt.remove_requests(self._wg, handles)

    def has_request(self, rid: str) -> bool:
        return rid in self._handles

    def known_requests(self) -> list[str]:
        return list(self._handles)

    # -- ingest -------------------------------------------------------------

    def ingest(self, rid: str, edges: list[GraphEdge], can_buffer: bool = True) -> list[GraphEdge]:
        h = self._handles[rid]
        return [
            e for e in edges
            if not self._rt.ingest(self._wg,
                h, e.next_node, e.name, self._tensors(e.tensor_info),
                can_buffer, e._final_stream_chunk,
            )
        ]

    def input_slots(self, rid, node_name, next_iter=False) -> dict[str, InputSlot]:
        return {
            name: InputSlot(
                uuids=[self._h_to_uuid[h] for h in handles],
                final_stream_chunk=final_chunk,
            )
            for name, handles, final_chunk in self._rt.input_slots(self._wg,
                self._handles[rid], node_name, next_iter
            )
        }

    # -- completion ---------------------------------------------------------

    def complete(self, rid: str, node_name: str) -> NodeCompletionOutput:
        node = self._section.get_nodes()[node_name]
        flat, tlens = [], []
        for e in node.outputs:
            t = self._tensors(e.tensor_info)
            flat.extend(t)
            tlens.append(len(t))
        raw, filtered = self._rt.complete_only(self._wg, self._handles[rid], node_name, flat, tlens)
        return NodeCompletionOutput(
            output_edges=[self._edge_from_rust(*r) for r in raw],
            filtered_signals={tuple(p) for p in filtered},
        )

    def complete_and_route_batch(
        self, node_name: str, rids: list[str], graph_walk: str,
    ) -> BatchRouting:
        node = self._section.get_nodes()[node_name]
        n_out = len(node.outputs)
        flat, tlens = [], []
        for _rid in rids:
            for e in node.outputs:
                t = self._tensors(e.tensor_info)
                flat.extend(t)
                tlens.append(len(t))
        handles = [self._handles[r] for r in rids]
        r = self._rt.complete_and_route_batch(self._wg, node_name, handles, flat, tlens)
        del n_out

        out = BatchRouting()
        for idx, name, modality, uuid_h in r.emit:
            edge = self._edge_from_rust(name, "emit_to_client", True, False, False,
                                        modality, False, 1)
            edge.tensor_info = [_stub_info(self._h_to_uuid[uuid_h])]
            out.emit_to_client.setdefault(rids[idx], []).append(edge)
        for idx, uuid_h in r.persist:
            out.persist.setdefault(rids[idx], []).append(
                GraphEdge(next_node="", name="", persist=True,
                          tensor_info=[_stub_info(self._h_to_uuid[uuid_h])])
            )
        for idx, name, uuid_h in r.new_tokens:
            out.new_token_outputs.setdefault(rids[idx], []).append(
                GraphEdge(next_node="", name=name, conductor_new_token=True,
                          tensor_info=[_stub_info(self._h_to_uuid[uuid_h])])
            )
        out.completed = [rids[i] for i in r.completed]
        # Peer edges arrive already encoded; the worker send path does not
        # consume them yet, so they are surfaced as-is for inspection.
        out.to_workers = {w: blob for w, blob in r.to_workers}
        return out

    # -- scheduling ---------------------------------------------------------

    def ready_nodes(self, rid: str) -> set[str]:
        return set(self._rt.ready_nodes(self._wg, self._handles[rid]))

    def ready_for_streaming(self, rid: str) -> set[str]:
        return set(self._rt.ready_for_streaming(self._wg, self._handles[rid]))

    def pop_ready(self, node_name: str, rids: list[str]) -> list[str]:
        handles = [self._handles[r] for r in rids if r in self._handles]
        popped = set(self._rt.pop_ready(self._wg, node_name, handles))
        return [r for r in rids if self._handles.get(r) in popped]

    def push_back(self, node_name: str, rids: list[str]) -> None:
        self._rt.push_back(self._wg, node_name, [self._handles[r] for r in rids if r in self._handles])

    # -- loops --------------------------------------------------------------

    def stop_loops(self, rid: str, loop_names: set[str]) -> set[NameAndDest]:
        h = self._handles[rid]
        signals: set[NameAndDest] = set()
        for name in loop_names:
            signals.update(tuple(p) for p in self._rt.stop_loops_batch(self._wg, [h], name))
        return signals

    def loop_indices(self, rid: str) -> dict[str, int]:
        return dict(self._rt.loop_indices(self._wg, self._handles[rid]))

    def nested_loop_idxs_for_node(self, rid: str, node_name: str) -> NestedLoopIndices:
        nl = self._rt.nested_loop_idxs_for_node(self._wg, self._handles[rid], node_name)
        return NestedLoopIndices(
            loop_name_order=list(nl.loop_name_order),
            loop_indices=dict(nl.loop_indices),
            wg_fwd_pass_idx=nl.wg_fwd_pass_idx,
        )

    def nested_loop_idxs(self, rid: str, loop_name: str) -> NestedLoopIndices:
        nl = self._rt.nested_loop_idxs_for_loop(self._wg, self._handles[rid], loop_name)
        return NestedLoopIndices(
            loop_name_order=list(nl.loop_name_order),
            loop_indices=dict(nl.loop_indices),
            wg_fwd_pass_idx=nl.wg_fwd_pass_idx,
        )

    def loop_names(self) -> set[str]:
        return set(self._section.get_loops())

    def is_done(self, rid: str) -> bool:
        return self._rt.is_done(self._wg, self._handles[rid])

    def reset(self, rid: str) -> None:
        self._rt.reset_request(self._wg, self._handles[rid])

    def num_times_run(self, rid: str) -> int:
        return self._rt.num_times_run(self._wg, self._handles[rid])

    # -- speculation --------------------------------------------------------

    def ingest_for_speculation(
        self, rid: str, edges: list[GraphEdge], source_node: str,
    ) -> list[SpeculativeNodeInfo]:
        # The Rust side speculates from the source node's declared outputs,
        # which is what the worker passes; `edges` is accepted for interface
        # symmetry with the Python implementation.
        del edges
        return [
            SpeculativeNodeInfo(node_name=n, is_new_loop_iter=new_iter, loop_name=loop)
            for n, new_iter, loop in self._rt.ingest_for_speculation(self._wg,
                self._handles[rid], source_node
            )
        ]

    def clear_speculative_inputs(self, rid: str) -> None:
        self._rt.clear_speculative_inputs(self._wg, self._handles[rid])


def _stub_info(uuid: str) -> TensorPointerInfo:
    """A TensorPointerInfo carrying only identity.

    The Rust core moves handles, not tensor metadata; the caller looks the
    real info up in the tensor manager by uuid.
    """
    return TensorPointerInfo(
        dims=[], dtype="", nbytes=0, address=0, stride=[], uuid=uuid,
        source_session_id="", source_entity="",
    )
