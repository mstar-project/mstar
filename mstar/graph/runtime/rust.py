"""``GraphRuntime`` over the Rust implementation in ``rust/``
(``mstar_rust.GraphRuntime``).

INCOMPLETE. The Rust side implements the bookkeeping and the structural
lookups; ingest, scheduling, routing, sending and speculation still raise. The
class exists so those land one method at a time behind a flag rather than as
one switchover, and so the seam that converts Python graph objects into the
Rust spec (:func:`worker_graph_args`) has a home.

Rust never sees a tensor. It holds a share of the same ``TensorBookkeeping``
Python gave ``TensorStore``, so descriptors and refcounts stay in one place.

Build: ``maturin develop --release`` in ``rust/`` (see ``docs/installation.rst``).
"""
from __future__ import annotations

from copy import deepcopy

from mstar_rust import GraphRuntime as _RustGraphRuntime

from mstar.communication.tensor_store import TensorBookkeeping
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphSection, Loop
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.runtime.base import (
    EdgeSpec,
    GraphRuntime,
    ParallelList,
    PopRidsOutput,
    ReadyNodeSpec,
    SpeculationOutput,
    SpeculationPrepInput,
    SpeculationPrepOutput,
)
from mstar.model.base import WorkerGraph


def _unported(name: str):
    """A method still on the Python side.

    Bound in the class body, not setattr'd afterwards: ABCMeta freezes
    __abstractmethods__ at class creation, so a later assignment leaves the
    class abstract and unusable. Raising by name beats an AttributeError three
    frames away.
    """
    def raise_unported(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            f"RustGraphRuntime.{name} is not ported yet; MSTAR_RUST_GRAPH "
            "cannot drive a full forward pass. See mstar/graph/runtime/rust.py."
        )
    raise_unported.__name__ = name
    return raise_unported


def _edge_args(section: GraphSection, edge) -> dict:
    del section
    return {
        "name": edge.name,
        "dest": edge.next_node,
        "persist": bool(edge.persist),
        "new_token": bool(edge.conductor_new_token),
        "streaming": bool(edge.is_streaming),
        "modality": edge.output_modality or "",
    }


def _loops_with_registries(section: GraphSection) -> list[Loop]:
    """The section's loops, with ``_managing_registry`` populated.

    A Loop only learns its enclosing registry when a ``WorkerGraphIO`` is built
    over the section, which normally happens per request. Compiling needs the
    loop nesting, so one is built here purely for that side effect -- over a
    COPY, since the object mutates the sections it walks and the pristine
    section is shared by every request.
    """
    io = WorkerGraphIO(deepcopy(section))
    return list(io.loops.values())


def _parent_loop_name(lp: Loop) -> str | None:
    """The enclosing loop, or None at the top level.

    ``_managing_registry`` is a LoopStateRegistry only when the loop is nested;
    at the top it is the worker graph's own registry, which has no ``loop``.
    """
    registry = lp._managing_registry
    parent = getattr(registry, "loop", None)
    return None if parent is None else parent.name


def worker_graph_args(worker_graph: WorkerGraph) -> dict:
    """One worker graph as the Rust compiler takes it.

    The compile seam: Python owns graph CONSTRUCTION, Rust owns the compiled
    form. ``loop_back`` / ``external_inputs`` are handed over verbatim from
    ``GraphSection.get_inputs_outputs()`` rather than re-derived, so the two
    cannot drift apart.
    """
    section = worker_graph.section
    nodes = section.get_nodes()
    loops = _loops_with_registries(section)
    loop_names = {lp.name for lp in loops}

    return {
        "wg_id": worker_graph.worker_graph_id,
        "graph_walks": sorted(worker_graph.graph_walks),
        "nodes": [
            {
                "name": name,
                "async_enabled": bool(node.enable_async_scheduling),
                "inputs": sorted(node.input_names),
                # _streaming_inputs, not streaming_inputs: the public name is
                # on ReadySignals. Populated during worker-graph construction
                # by _register_streaming, so it is set by the time we compile.
                "streaming_inputs": sorted(node._streaming_inputs),
                "outputs": [_edge_args(section, e) for e in node.outputs],
            }
            for name, node in nodes.items()
        ],
        "loops": [
            {
                "name": lp.name,
                "max_iters": lp.max_iters,
                "parent": _parent_loop_name(lp),
                # Directly-owned only: a node inside a child loop belongs there.
                "member_nodes": [
                    n for n in lp.section.get_nodes()
                    if n not in loop_names
                ],
                "outputs": [_edge_args(section, e) for e in lp.outputs],
                "accumulated": [
                    _edge_args(section, e) for e in lp.accumulated_outputs
                ],
                "loop_back": sorted(lp._loop_back_inputs or ()),
                "external_inputs": sorted(lp._external_inputs or ()),
            }
            for lp in loops
        ],
    }


def remote_worker_graph_args(
    wg_id: int,
    graph_walks: set[str],
    nodes: set[str],
    dyn_loops: set[str],
) -> dict:
    """A worker graph owned by another worker: enough to route to it."""
    return {
        "wg_id": wg_id,
        "graph_walks": sorted(graph_walks),
        "nodes": sorted(nodes),
        "dyn_loops": sorted(dyn_loops),
    }


def sharding_args(config: ShardingConfig) -> dict:
    """``ShardingConfig`` as the Rust template takes it.

    ``shard_dim`` goes over as pairs because a None value is meaningful in
    Python's type (`int | None`) even though it reads the same as an absent
    key; Rust drops the Nones on the way in.
    """
    return {
        "groups": [
            {
                "nodes": sorted(g.nodes),
                "tp_size": g.tp_size,
                "graph_walks": None if g.graph_walks is None
                else sorted(g.graph_walks),
                "tp_rank": g._tp_rank,
            }
            for g in config.groups
        ],
        "shard_dim": list(config.shard_dim.items()),
        "tp_enabled_nodes": sorted(config.tp_enabled_nodes),
        "sp_enabled_nodes": sorted(config.sp_enabled_nodes),
    }


class RustGraphRuntime(GraphRuntime):
    """Delegates to ``mstar_rust.GraphRuntime``.

    Handles, worker graph ids and tensor uuids are already integers on both
    sides, so they cross as-is; only node, walk and partition NAMES are
    strings, and Rust interns those on arrival.
    """

    def __init__(
        self,
        my_worker_id: str,
        my_worker_graphs: list[WorkerGraph],
        all_wg_ids_to_graph_walks: dict[int, set[str]],
        all_wg_ids_to_dyn_loops: dict[int, set[str]],
        all_wg_ids_to_nodes: dict[int, set[str]],
        node_to_partition: dict[str, str],
        sharding_config: ShardingConfig,
        bookkeeping: TensorBookkeeping,
    ):
        mine = {wg.worker_graph_id for wg in my_worker_graphs}
        self._rust = _RustGraphRuntime(
            worker_graphs=[worker_graph_args(wg) for wg in my_worker_graphs],
            remote_worker_graphs=[
                remote_worker_graph_args(
                    wg_id,
                    all_wg_ids_to_graph_walks.get(wg_id, set()),
                    all_wg_ids_to_nodes.get(wg_id, set()),
                    all_wg_ids_to_dyn_loops.get(wg_id, set()),
                )
                for wg_id in all_wg_ids_to_graph_walks
                if wg_id not in mine
            ],
            sharding=sharding_args(sharding_config),
            # The share, not a copy: routing adjusts the refcounts TensorStore
            # reads.
            bookkeeping=bookkeeping._rust,
            me=my_worker_id,
        )
        self._node_to_partition = node_to_partition

    # --------- Bookkeeping ----------

    def set_node_metadata(
        self, parallel_nodes: set[str],
        parallel_leader_nodes: set[str],
        tp_async_nodes: set[str],
    ):
        self._rust.set_node_metadata(
            sorted(parallel_nodes), sorted(parallel_leader_nodes),
            sorted(tp_async_nodes),
        )

    def add_request(
        self, request_id: str, partition: str, graph_walk: str,
        partition_worker_graph_ids: list[int],
        worker_graph_to_workers: ParallelList[int, list[str]],
    ) -> int:
        flat: list[str] = []
        counts: list[int] = []
        for _wg_id, workers in worker_graph_to_workers:
            flat.extend(workers)
            counts.append(len(workers))
        return self._rust.add_request(
            request_id, partition, graph_walk,
            list(worker_graph_to_workers.keys), flat, counts,
        )

    def remove_request(self, rid: int):
        self._rust.remove_request(rid)

    def get_rid_string(self, handle: int) -> str:
        return self._rust.get_rid_string(handle)

    def get_rid_handle(self, rid: str) -> int | None:
        return self._rust.get_rid_handle(rid)

    def set_walk(self, rid: int, partition: str, walk: str):
        self._rust.set_walk(rid, partition, walk)

    def set_speculatively_scheduled(
        self, node: str, wg_id: int, rids: list[int],
        speculatively_scheduled: bool,
    ):
        self._rust.set_speculatively_scheduled(
            node, wg_id, rids, speculatively_scheduled
        )

    def mark_stream_partition_done(self, rid: int, partition: str):
        self._rust.mark_stream_partition_done(rid, partition)

    def get_worker_graph_id_for_node(self, node: str, graph_walk: str) -> int:
        return self._rust.get_worker_graph_id_for_node(node, graph_walk)

    def is_async_schedulable(self, node_name: str, graph_walk: str) -> bool:
        return self._rust.is_async_schedulable(node_name, graph_walk)

    def get_output_signals(self, node_name: str, graph_walk: str) -> list[str]:
        return self._rust.get_output_signals(node_name, graph_walk)

    def get_consumed_edges(
        self, source_node: str, dest_node: str, graph_walk: str,
    ) -> set[tuple[str, str]]:
        return {
            (name, dest)
            for name, dest in self._rust.get_consumed_edges(
                source_node, dest_node, graph_walk
            )
        }

    def push_back_node(
        self, node_name: str, rids: list[int], wg_ids: list[int],
    ):
        self._rust.push_back_node(node_name, rids, wg_ids)

    # --------- pending loop stops ----------

    def has_pending_loop_stop(
        self, rid: int, graph_walk: str, loop_name: str,
    ) -> bool:
        return self._rust.has_pending_loop_stop(rid, graph_walk, loop_name)

    def pending_loop_stop_rids(
        self, graph_walk: str, loop_name: str,
    ) -> set[int]:
        return set(self._rust.pending_loop_stop_rids(graph_walk, loop_name))

    def clear_pending_loop_stops(self):
        self._rust.clear_pending_loop_stops()

    # --------- Inputs ----------

    def ingest_inputs_batch(
        self,
        signals: ParallelList[int, EdgeSpec],
        can_buffer: bool = True,
        is_streaming: bool = False,
    ) -> list[int]:
        return self._rust.ingest_inputs_batch(
            signals.keys,
            [
                {
                    "signal": s.signal,
                    "next_node": s.next_node,
                    "uuids": s.uuids,
                    "is_final_streaming_chunk": s.is_final_streaming_chunk,
                } for s in signals.values
            ],
            can_buffer,
            is_streaming,
        )

    def get_dynamic_loop_iters(
        self, request_ids: list[int], partition: str,
    ) -> ParallelList[int, dict[str, int]]:
        return ParallelList(
            list(request_ids),
            [
                dict(pairs)
                for pairs in self._rust.get_dynamic_loop_iters(
                    list(request_ids), partition
                )
            ],
        )

    def reset_outputs(
        self, node_name: str, rids: list[int], wg_ids: list[int],
    ):
        self._rust.reset_outputs(node_name, rids, wg_ids)

    def cleanup_consumed_inputs(
        self, node_name: str, rids: list[int], wg_ids: list[int],
    ):
        self._rust.cleanup_consumed_inputs(node_name, rids, wg_ids)

    # --------- Scheduling ----------

    def pop_rids(
        self, node_name: str, graph_walk: str, request_ids: list[int],
        check_ready: bool = False,
    ) -> PopRidsOutput | None:
        out = self._rust.pop_rids(
            node_name, graph_walk, list(request_ids), check_ready
        )
        if out is None:
            return None
        return PopRidsOutput(
            wg_ids=ParallelList(out.rids, out.wg_ids),
            input_edges=[
                EdgeSpec(
                    signal=signal, next_node=next_node, uuids=uuids,
                    is_final_streaming_chunk=final,
                )
                for signal, next_node, uuids, final in out.input_edges
            ],
            input_edges_per_rid=out.input_edges_per_rid,
        )

    def has_ready_excluding(
        self, exclude_rids: set[int],
        exclude_target: tuple[str, str] | None = None,
    ) -> bool:
        return self._rust.has_ready_excluding(
            sorted(exclude_rids), exclude_target
        )

    def get_ready_nodes(
        self, exclude_rids: set[int],
        target: tuple[str, str] | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> list[ReadyNodeSpec]:
        return [
            ReadyNodeSpec(node_name=n, graph_walk=w, rids=rids)
            for n, w, rids in self._rust.get_ready_nodes(
                sorted(exclude_rids), target, exclude_target
            )
        ]

    # --------- Speculation ----------

    def speculate_node(
        self, node_name: str, graph_walk: str, sample_rid: int,
    ) -> list[SpeculationOutput]:
        return [
            SpeculationOutput(
                node_name=n, graph_walk=w,
                is_new_loop_iter=new_iter, loop_name=loop_name,
            )
            for n, w, new_iter, loop_name in self._rust.speculate_node(
                node_name, graph_walk, sample_rid
            )
        ]

    def get_spec_target(
        self, curr_node_name: str, spec_node_name: str,
        graph_walk: str, sample_rid: int,
    ) -> SpeculationOutput | None:
        out = self._rust.get_spec_target(
            curr_node_name, spec_node_name, graph_walk, sample_rid
        )
        if out is None:
            return None
        node_name, walk, new_iter, loop_name = out
        return SpeculationOutput(
            node_name=node_name, graph_walk=walk,
            is_new_loop_iter=new_iter, loop_name=loop_name,
        )

    def _spec_prep_arg(self, input: SpeculationPrepInput) -> dict:
        return {
            "spec_node_name": input.spec_node_name,
            "curr_node_name": input.curr_node_name,
            "graph_walk": input.graph_walk,
            "rids": list(input.rids),
            "room_for_continuing": input.room_for_continuing,
            "streaming_edges": [
                {
                    "signal": e.signal, "next_node": e.next_node,
                    "uuids": e.uuids,
                    "is_final_streaming_chunk": e.is_final_streaming_chunk,
                } for e in input.streaming_edges
            ],
            "streaming_edges_per_rid": list(input.streaming_edges_per_rid),
        }

    @staticmethod
    def _spec_prep_out(out) -> SpeculationPrepOutput:
        return SpeculationPrepOutput(
            consumed_streaming_edge_idxs=out.consumed_streaming_edge_idxs,
            ready_rids=out.ready_rids,
            wg_ids=out.wg_ids,
            input_edges=[
                EdgeSpec(
                    signal=s, next_node=n, uuids=u,
                    is_final_streaming_chunk=f,
                ) for s, n, u, f in out.input_edges
            ],
            input_edges_per_rid=out.input_edges_per_rid,
        )

    def prep_spec_rids(
        self, input: SpeculationPrepInput,
    ) -> SpeculationPrepOutput:
        return self._spec_prep_out(
            self._rust.prep_spec_rids(self._spec_prep_arg(input))
        )

    def prep_follow_spec_rids(
        self, input: SpeculationPrepInput,
    ) -> SpeculationPrepOutput | None:
        out = self._rust.prep_follow_spec_rids(self._spec_prep_arg(input))
        return None if out is None else self._spec_prep_out(out)

    # --------- not ported yet ----------
    #
    # Everything a forward pass needs. Until these land, MSTAR_RUST_GRAPH=1
    # admits requests and answers the structural queries but cannot run a step.

    stop_loops_batched = _unported("stop_loops_batched")
    apply_peer_loop_stops = _unported("apply_peer_loop_stops")
    complete_and_route_batch = _unported("complete_and_route_batch")
    send_outputs = _unported("send_outputs")
