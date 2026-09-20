"""Reference implementation: the existing ``WorkerGraphIO`` behind the harness.

Carries no new semantics — it is today's behavior, so a divergence the shadow
reports is always the Rust side disagreeing with what ships.
"""
from copy import deepcopy

from mstar.graph.base import (
    GraphEdge,
    GraphSection,
    NameAndDest,
    NodeCompletionOutput,
    SpeculativeNodeInfo,
)
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime.base import BatchRouting, GraphRuntimeBase, InputSlot


class PythonGraphRuntime(GraphRuntimeBase):
    backend = "python"

    def __init__(
        self,
        section: GraphSection,
        wg_id: str,
        tensor_manager=None,
        sharding_config=None,
        worker_id: str = "",
        io_store: dict[str, WorkerGraphIO] | None = None,
    ):
        self._section = section
        self._wg_id = wg_id
        self._tensor_manager = tensor_manager
        self._sharding_config = sharding_config
        self._worker_id = worker_id
        # `io_store` lets the caller keep owning the dict — the worker passes
        # its `per_request_queues` so call sites that still reach into
        # WorkerGraphIO directly (micro_scheduler, the speculation path) see
        # the same objects this runtime drives.
        self._io: dict[str, WorkerGraphIO] = {} if io_store is None else io_store

    # -- lifecycle ----------------------------------------------------------

    def add_requests(self, rids: list[str]) -> None:
        for rid in rids:
            # The deepcopy the port exists to remove: spec and state live in
            # the same objects, so each request needs its own section.
            io = WorkerGraphIO(deepcopy(self._section), wg_id=self._wg_id)
            if self._tensor_manager is not None:
                io.register_communication_info(self._tensor_manager, rid)
            self._io[rid] = io

    def remove_requests(self, rids: list[str]) -> None:
        for rid in rids:
            self._io.pop(rid, None)

    def has_request(self, rid: str) -> bool:
        return rid in self._io

    def known_requests(self) -> list[str]:
        return list(self._io)

    # -- ingest -------------------------------------------------------------

    def ingest(self, rid: str, edges: list[GraphEdge], can_buffer: bool = True) -> list[GraphEdge]:
        io = self._io[rid]
        return [e for e in edges if not io.ingest_input(e, can_buffer)]

    # -- completion ---------------------------------------------------------

    def complete(self, rid: str, node_name: str) -> NodeCompletionOutput:
        return self._io[rid].mark_node_complete(node_name)

    def complete_and_route_batch(
        self, node_name: str, rids: list[str], graph_walk: str,
    ) -> BatchRouting:
        out = BatchRouting()
        for rid in rids:
            io = self._io.get(rid)
            if io is None:
                continue
            edges = [e.clone() for e in io.mark_node_complete(node_name).output_edges]
            local, external = [], []
            for edge in edges:
                if edge.persist:
                    out.persist.setdefault(rid, []).append(edge)
                if edge.conductor_new_token:
                    out.new_token_outputs.setdefault(rid, []).append(edge)
                (local if edge.next_node in io.nodes else external).append(edge)
            for edge in local:
                if io.ingest_input(edge, can_buffer=True):
                    out.routed_local.setdefault(rid, []).append(edge)
                else:
                    external.append(edge)
            for edge in external:
                self._route_external(out, rid, edge, node_name, graph_walk)
            if io.wg_state_registry.is_done:
                out.completed.append(rid)
        return out

    def _route_external(self, out: BatchRouting, rid, edge, node_name, graph_walk) -> None:
        from mstar.graph.special_destinations import EMIT_TO_CLIENT

        if edge.next_node == EMIT_TO_CLIENT:
            out.emit_to_client.setdefault(rid, []).append(edge)
            return
        if self._sharding_config is None:
            return
        fanout = self._sharding_config.fanout_graph_edges(
            edge, source_node=node_name, source_graph_walk=graph_walk,
            dest_graph_walk=graph_walk,
        )
        for worker, wkr_edge in fanout.items():
            out.to_workers.setdefault(worker, []).append((rid, wkr_edge))

    # -- scheduling ---------------------------------------------------------

    def ready_nodes(self, rid: str) -> set[str]:
        # The live set, not a copy: `get_ready_node_names` is called once per
        # request per scheduling step, and copying there was the old code's
        # cost too. Callers iterate; mutation goes through pop_ready.
        return self._io[rid].ready_node_names

    def ready_for_streaming(self, rid: str) -> set[str]:
        return self._io[rid].ready_for_streaming

    def input_slots(self, rid, node_name, next_iter=False) -> dict[str, InputSlot]:
        node = self._io[rid].nodes[node_name]
        slot = node.ready_next_iter if next_iter else node.ready_signals
        return {
            name: InputSlot(
                uuids=[info.uuid for info in edge.tensor_info],
                final_stream_chunk=edge._final_stream_chunk,
            )
            for name, edge in slot.ready_inputs.items()
        }

    def pop_ready(self, node_name: str, rids: list[str]) -> list[str]:
        popped = []
        for rid in rids:
            io = self._io.get(rid)
            if io is not None and node_name in io.ready_node_names:
                io.ready_node_names.discard(node_name)
                popped.append(rid)
        return popped

    def push_back(self, node_name: str, rids: list[str]) -> None:
        for rid in rids:
            io = self._io.get(rid)
            if io is not None:
                io.ready_node_names.add(node_name)

    # -- loops --------------------------------------------------------------

    def stop_loops(self, rid: str, loop_names: set[str]) -> set[NameAndDest]:
        io = self._io[rid]
        signals: set[NameAndDest] = set()
        for name in loop_names:
            if name not in io.loops:
                continue
            io.register_loop_finish_signal(name)
            signals.update(io.loops[name]._loop_back_inputs)
        return signals

    def loop_indices(self, rid: str) -> dict[str, int]:
        return self._io[rid].get_loop_indices()

    def nested_loop_idxs_for_node(self, rid: str, node_name: str) -> NestedLoopIndices:
        return self._io[rid].get_nested_loop_idxs_for_node(node_name)

    def nested_loop_idxs(self, rid: str, loop_name: str) -> NestedLoopIndices:
        return self._io[rid].get_nested_loop_idxs(target_loop_name=loop_name)

    def loop_names(self) -> set[str]:
        return set(self._section.get_loops())

    def is_done(self, rid: str) -> bool:
        return self._io[rid].wg_state_registry.is_done

    def reset(self, rid: str) -> None:
        self._io[rid].clear()

    def num_times_run(self, rid: str) -> int:
        return self._io[rid].num_times_run

    # -- speculation --------------------------------------------------------

    def ingest_for_speculation(
        self, rid: str, edges: list[GraphEdge], source_node: str,
    ) -> list[SpeculativeNodeInfo]:
        return self._io[rid].ingest_for_speculation(edges, source_node)

    def clear_speculative_inputs(self, rid: str) -> None:
        self._io[rid].clear_speculative_inputs()

    # -- escape hatch -------------------------------------------------------

    def io(self, rid: str) -> WorkerGraphIO:
        """The underlying ``WorkerGraphIO``.

        Only for call sites not yet expressed on the harness (the worker's
        speculation path reaches into node objects directly). Every use is a
        place the harness does not yet cover.
        """
        return self._io[rid]
