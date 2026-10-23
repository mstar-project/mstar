"""A scheduling pass split at the engine's readiness check: plan -> check -> commit.

The check (``Engine.check_ready``) is per-rank resource code with side effects
(a KV reload, a transfer kickoff), so it stays Python. Everything around it --
which rows to check and in what order, and what to do with the verdicts -- is
the scheduler core's. A core hands the caller a ``CheckRequest``, the caller
answers it with one verdict per row (``MicroScheduler._run_checks``), and the
core either asks again or returns the step. Two cores speak this protocol:
``MicroScheduler``'s own pass (a generator that yields its requests) and
``RustSchedCore`` (the graph runtime's), behind the same driver.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, NamedTuple

from mstar.graph.runtime.base import ColumnarEdgeSpecs

if TYPE_CHECKING:
    from mstar.worker.micro_scheduler import ScheduledBatch

# One verdict per requested row
NOT_READY, READY, FAILED, UNCHECKED = 0, 1, 2, 3


class CheckRequest(NamedTuple):
    """Rows whose engine readiness a scheduling pass needs, in check order."""
    rids: list[int]
    # row i is node ``nodes[node_idx[i]]``; None when every row is ``nodes[0]``
    node_idx: list[int] | None
    nodes: list[str]
    # stop once this many rows were found ready (and counted); -1 checks all.
    # Rows past the stop are UNCHECKED.
    limit: int = -1
    # Python core only: whether a ready row counts toward ``limit`` (a
    # backlog scan counts only rows of the step's capture group)
    counts: Callable[[int], bool] | None = None


class RustSchedCore:
    """The scheduling pass in the Rust graph runtime: the backlog and the
    round-robin table live there, next to the ready masks, pops and in-flight
    flags it works on, and a pass crosses once per check request.

    Used for a worker the runtime can schedule alone (``supported``): no node
    splits batches by capture key and no speculation chain mixes walks.
    """

    def __init__(self, runtime, keys: Iterable[tuple[str, str, list[str], dict, dict]],
                 leader_nodes: Iterable[str]):
        self._rt = runtime._rust
        self._rt.sched_configure(
            [(node, label, list(walks), list(caps.items()), list(row_caps.items()))
             for node, label, walks, caps, row_caps in keys],
            sorted(leader_nodes),
        )

    def run(
        self, check: Callable[[CheckRequest], bytes],
        target, exclude_target, pre_existing_batch_size: int,
        waiting: Iterable[int], failed: Iterable[int], scan_exclude: Iterable[int],
    ) -> ScheduledBatch | None:
        """One pass: begin, answer each check request with ``check``, until
        the runtime returns the step (or None)."""
        done, request, batch = self._rt.sched_begin(
            target, exclude_target, pre_existing_batch_size,
            list(waiting), list(failed), list(scan_exclude),
        )
        while not done:
            done, request, batch = self._rt.sched_resume(check(CheckRequest(*request)))
        return None if batch is None else _batch(batch)

    def room_for_continuing(self, node: str, walk: str) -> int | None:
        return self._rt.sched_room_for_continuing(node, walk)

    def backlog_has_schedulable(
        self, exclude_key, exclude_walk, waiting: Iterable[int], failed: Iterable[int],
    ) -> bool:
        return self._rt.sched_backlog_has_schedulable(
            exclude_key, exclude_walk, list(waiting), list(failed),
        )

    def backlog_rows(self, exclude: Iterable[int]) -> list[tuple[str, int]]:
        return self._rt.sched_backlog_rows(list(exclude))

    def drop_rids(self, rids: Iterable[int]) -> None:
        self._rt.sched_drop_rids(list(rids))

    def state(self):
        """``(backlog, rr)`` for tests: backlog as ``[(node, label, rids,
        walks)]`` in order, rr as ``{(node, label): batch number}`` plus the
        counter."""
        return self._rt.sched_state()


def _batch(cols) -> ScheduledBatch:
    from mstar.worker.micro_scheduler import ScheduledBatch
    (node, label, rids, wg_ids, walk_names, walk_idx, out_sigs, walk_out_sigs,
     (sig_names, sig_idx, edge_rids, per_edge, finals, uuids)) = cols
    batch = ScheduledBatch(
        node_name=node, graph_walk=label,
        request_to_worker_graph=dict(zip(rids, wg_ids, strict=True)),
        input_edges=ColumnarEdgeSpecs(
            signal_names=sig_names, uuids=uuids, tensors_per_edge=per_edge,
            signal_name_idxs=sig_idx, rids=edge_rids, is_final_streaming_chunk=finals,
        ),
        output_signals=out_sigs,
    )
    if walk_idx is not None:
        batch.request_walks = {
            rid: walk_names[w] for rid, w in zip(rids, walk_idx, strict=True)
        }
        batch.walk_output_signals = dict(walk_out_sigs)
    return batch
