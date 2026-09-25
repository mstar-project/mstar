"""``GraphRuntime`` over the Rust implementation in ``rust/``
(``mstar_rust.GraphRuntime``).

The whole ``GraphRuntime`` contract is implemented. Graph state, scheduling,
routing and speculation run in Rust, and so does sending: Rust builds the
INPUT_SIGNALS / RESULT_TENSORS / WORKER_GRAPHS_DONE frames in ``wire.py``'s
typed msgpack (``rust/src/graph/frames.rs``) and sends them on a share of the
worker's transport. The payloads Python owns -- ``CurrentForwardPassInfo`` and
profiling -- are encoded here and spliced in as opaque bytes.

:func:`worker_graph_args` is the compile seam: Python owns graph
construction, Rust owns the compiled form.

Rust never sees a tensor. It holds a share of the same ``TensorBookkeeping``
Python gave ``TensorStore``, so descriptors and refcounts stay in one place.

Build: ``maturin develop --release`` in ``rust/`` (see ``docs/installation.rst``).
"""
from __future__ import annotations

import time as _time
from copy import deepcopy
from typing import get_args

from mstar_rust import GraphRuntime as _RustGraphRuntime

from mstar.communication import wire
from mstar.communication.tensor_store import TensorBookkeeping
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphNode, GraphSection, Loop
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime import sharding
from mstar.graph.runtime.base import (
    ColumnarEdgeSpecs,
    FreedTensors,
    GraphRuntime,
    ParallelList,
    PopRidsOutput,
    Profiling,
    ReadyNodeSpec,
    RouteInput,
    RouteOutput,
    SendInput,
    SpeculationOutput,
    SpeculationPrepInput,
    SpeculationPrepOutput,
)
from mstar.model.base import WorkerGraph
from mstar.utils.profiler import PHASE_PERIOD, phase_record

# SendInput.profiling's (rx_info, tx_info, graph_timings), by declared type.
_PROFILING_HINTS = get_args(Profiling)


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


def _member_nodes(lp: Loop) -> list[str]:
    """Nodes this loop owns directly; ``section.get_nodes()`` recurses, so
    every ancestor claimed its descendants and nested loops deadlocked."""
    return [
        name for name, entity in lp.inner_registry.managed_entities.items()
        if isinstance(entity, GraphNode)
    ]


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
                "member_nodes": _member_nodes(lp),
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


def _columns(out) -> ColumnarEdgeSpecs:
    """The edge columns off a pop or a prep result.

    Six list conversions for the whole batch, where the per-edge form was a
    tuple plus two Rust ``String`` allocations each. Rust flattens the columns
    onto its result rather than nesting them, because a ``#[pyo3(get)]`` on a
    pyclass field clones it -- hence the names being read off ``out`` here.
    """
    return ColumnarEdgeSpecs(
        signal_names=out.signal_names,
        uuids=out.uuids,
        tensors_per_edge=out.tensors_per_edge,
        signal_name_idxs=out.signal_name_idxs,
        rids=out.edge_rids,
        is_final_streaming_chunk=out.is_final_streaming_chunk,
    )


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
        communicator=None,
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
            # A share of the same transport, so the frames Rust decides on can
            # be sent from there rather than handed back one at a time. None
            # under the pyzmq communicator, which has no shareable object --
            # the worker's flag gate refuses that pairing anyway.
            communicator=getattr(communicator, "_inner", None),
        )
        self._node_to_partition = node_to_partition
        self._communicator = communicator
        self._bookkeeping = bookkeeping
        # Rust derives its own per-request sharding for routing, but that copy
        # cannot come back out: register_request wants the Python object and
        # the TP fan-out paths read group._workers. Derived here by the same
        # rule from the same input, so the two cannot disagree.
        self._sharding_base = sharding_config
        self._all_wg_ids_to_graph_walks = all_wg_ids_to_graph_walks
        self._all_wg_ids_to_nodes = all_wg_ids_to_nodes
        self._sharding: dict[int, ShardingConfig] = {}

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
        handle = self._rust.add_request(
            request_id, partition, graph_walk,
            # Two distinct lists: the partition's worker graphs decide which
            # of this worker's graphs the request opens and which are live in
            # the walk, while the full map is what the node -> workers routing
            # is derived from.
            list(partition_worker_graph_ids),
            list(worker_graph_to_workers.keys), flat, counts,
        )
        # One NewRequest arrives PER PARTITION, all carrying the same map, so
        # the first one settles it.
        if handle not in self._sharding:
            self._sharding[handle] = sharding.for_request(
                self._sharding_base, worker_graph_to_workers,
                self._all_wg_ids_to_graph_walks, self._all_wg_ids_to_nodes,
            )
        return handle

    def remove_request(self, rid: int):
        # Handles are recycled, so anything left keyed by this one attaches to
        # a DIFFERENT request later.
        self._sharding.pop(rid, None)
        self._rust.remove_request(rid)

    def get_sharding_config(self, rid: int) -> ShardingConfig | None:
        """None for a rid this rank does not know: the teardown and TP-fanout
        callers can legitimately race a removal."""
        return self._sharding.get(rid)

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

    def is_speculatively_scheduled(
        self, node: str, wg_id: int, rid: int,
    ) -> bool:
        return self._rust.is_speculatively_scheduled(node, wg_id, rid)

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
        signals: ColumnarEdgeSpecs,
        can_buffer: bool = True,
        is_streaming: bool = False,
    ) -> list[int]:
        # The block goes straight over: Rust reads the columns off the
        # dataclass, so nothing is marshalled per arriving signal.
        return self._rust.ingest_inputs_batch(
            signals, can_buffer, is_streaming,
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
    ) -> FreedTensors:
        return FreedTensors(
            *self._rust.cleanup_consumed_inputs(node_name, rids, wg_ids)
        )

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
            input_edges=_columns(out),
            output_signals=tuple(out.output_signals),
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
                output_signals=tuple(signals),
            )
            for n, w, new_iter, loop_name, signals
            in self._rust.speculate_node(node_name, graph_walk, sample_rid)
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
        node_name, walk, new_iter, loop_name, signals = out
        return SpeculationOutput(
            node_name=node_name, graph_walk=walk,
            is_new_loop_iter=new_iter, loop_name=loop_name,
            output_signals=tuple(signals),
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
                    # next_node is not sent: the poll filtered on it, so it
                    # is spec_node_name for every one of these.
                    "signal": e.signal,
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
            input_edges=_columns(out),
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

    # --------- Postprocess ----------

    def complete_and_route_batch(
        self, input: RouteInput, tensor_store,
    ) -> RouteOutput:
        # tensor_store is unused: Rust holds a share of the same bookkeeper,
        # so it reads descriptors and settles refcounts without it.
        del tensor_store
        out = self._rust.complete_and_route_batch({
            "partition": input.partition,
            "graph_walk": input.graph_walk,
            "node_name": input.node_name,
            "output_signals": list(input.output_signals),
            "rids": list(input.wg_ids.keys),
            "tensors": list(input.tensors),
            "num_tensors": list(input.num_tensors),
        })
        return RouteOutput(
            completion_id=out.completion_id,
            register_uuids=out.register_uuids,
            register_rids=out.register_rids,
            new_token_output_idxs=out.new_token_output_idxs,
            # Rust hands the groups over as three parallel columns, one entry
            # per stream; the dict is rebuilt here so both runtimes return the
            # same shape.
            local_streaming_by_signal={
                name: ParallelList(rids, uuids)
                for name, rids, uuids in zip(
                    out.local_streaming_signals,
                    out.local_streaming_rids,
                    out.local_streaming_uuids,
                    strict=True,
                )
            },
            rids_needing_request_info=frozenset(out.rids_needing_request_info),
            freed_inputs=FreedTensors(
                out.freed_input_uuids, out.freed_input_registered,
            ),
        )

    def stop_loops_batched(
        self, partition: str, graph_walk: str, last_node_run: str,
        loop_names: ParallelList[int, list[str]],
    ):
        """Rust stops the loops, decides who to tell, and sends."""
        self._rust.stop_loops_batched(
            partition, graph_walk, last_node_run,
            list(loop_names.keys), [list(v) for v in loop_names.values],
        )

    def apply_peer_loop_stops(
        self, rid: int, partition: str,
        loop_stop_times: dict[str, NestedLoopIndices],
    ):
        names = list(loop_stop_times)
        self._rust.apply_peer_loop_stops(
            rid, partition, names,
            [
                {
                    "loop_name_order": list(loop_stop_times[n].loop_name_order),
                    "loop_indices": list(loop_stop_times[n].loop_indices.items()),
                    "wg_fwd_pass_idx": loop_stop_times[n].wg_fwd_pass_idx,
                } for n in names
            ],
        )

    def _loop_stop_times(self, rid: int) -> dict[str, NestedLoopIndices]:
        """Only used for testing.
        """
        return {
            name: NestedLoopIndices(
                loop_name_order=order,
                loop_indices=dict(indices),
                wg_fwd_pass_idx=fwd,
            )
            for name, order, indices, fwd in self._rust.get_loop_stop_times(rid)
        }

    def send_outputs(self, input: SendInput):
        """A thin wrapper: Rust decides what goes where AND builds the frames.

        Only the Python-owned payloads are prepared here.
        ``CurrentForwardPassInfo`` is encoded rather than handed over, so Rust
        never owns the type -- and with it ``resource_publish_info``, whose
        ``PublishedInfo`` is abstract and would otherwise mean a Rust change
        for every new resource. Both splice in untouched.

        Encoded every pass, never cached: ``engine.finalize_batch`` folds each
        pass's published resource state into the SAME object the worker keeps,
        so a cache keyed on identity would splice the first pass's bytes into
        every later frame.

        The loop context each frame reports is not passed: Rust snapshotted it
        at route time, before the completion advanced it.
        """
        # The wrapper's own per-rid rebuilding, timed apart from the crossing.
        _t0 = _time.perf_counter() if PHASE_PERIOD else 0.0
        # Straight through as struct-of-arrays. `ParallelList` already holds
        # keys and values as two flat lists -- which is the whole reason it
        # exists -- so zipping them into tuples here only to have Rust unzip
        # them is a per-rid interpreted loop per argument, per forward pass.
        # The count dicts go over as dicts; PyO3 extracts them.
        args = dict(
            completion_id=input.completion_id,
            info_rids=input.per_request_info.keys,
            request_infos=[
                None if i is None else wire.encode_field(
                    i, CurrentForwardPassInfo,
                )
                for i in input.per_request_info.values
            ],
            ntc_rids=input.new_token_counts.keys,
            new_token_counts=input.new_token_counts.values,
            consumed_rids=(
                [] if input.stream_tokens_consumed is None
                else input.stream_tokens_consumed.keys
            ),
            stream_tokens_consumed=(
                [] if input.stream_tokens_consumed is None
                else input.stream_tokens_consumed.values
            ),
            prof_rids=[] if input.profiling is None else input.profiling.keys,
            # encode_fields, not encode: a bare tuple has no wire tag, so
            # encode would pickle it, and Rust cannot split a pickle into
            # WORKER_GRAPHS_DONE's three fields.
            profiling=[] if input.profiling is None else [
                wire.encode_fields(p, _PROFILING_HINTS)
                for p in input.profiling.values
            ],
        )
        if PHASE_PERIOD:
            _t1 = _time.perf_counter()
            phase_record("rust.send_outputs.marshal", _t1 - _t0)
            self._rust.send_outputs(**args)
            phase_record("rust.send_outputs.call", _time.perf_counter() - _t1)
        else:
            self._rust.send_outputs(**args)
