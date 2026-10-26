"""The scheduling pass in the Rust graph runtime (see ``RustSchedCore``)."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.worker.sched_core.base import (
    CheckRequest,
    CheckResult,
    PassRequest,
    PassStep,
    SchedCore,
)

if TYPE_CHECKING:
    from mstar.worker.micro_scheduler import ScheduledBatch


class RustSchedCore(SchedCore):
    """The scheduling pass in the Rust graph runtime: the backlog and the
    round-robin table live there, next to the ready masks, pops and in-flight
    flags it works on, and a pass crosses once per check request."""

    def __init__(self, runtime, keys: Iterable[tuple[str, str, list[str], dict, dict]],
                 leader_nodes: Iterable[str], mixing_chains: Iterable[tuple[str, str]],
                 batch_builder: str):
        self._rt = runtime._rust
        self._rt.sched_configure(
            keys=[(node, label, list(walks), list(caps.items()), list(row_caps.items()))
                  for node, label, walks, caps, row_caps in keys],
            leaders=sorted(leader_nodes),
            mixing=sorted(mixing_chains),
            builder=batch_builder,
        )

    def begin(self, request: PassRequest) -> PassStep:
        return _step(*self._rt.sched_begin(
            target=request.target, exclude_target=request.exclude_target,
            pre_existing=request.pre_existing_batch_size,
            capture_group_of=request.capture_group_of,
            waiting=list(request.waiting), failed=list(request.failed),
        ))

    def resume(self, result: CheckResult) -> PassStep:
        return _step(*self._rt.sched_resume(
            verdicts=result.verdicts, groups=result.groups, anchor_group=result.anchor_group,
        ))

    def room_for_continuing(self, target: tuple[str, str]) -> int | None:
        return self._rt.sched_room_for_continuing(*target)

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

    def live_backlog(
        self, key: tuple[str, str], waiting: Iterable[int], failed: Iterable[int],
    ) -> list[int]:
        return self._rt.sched_live_backlog(*key, list(waiting), list(failed))

    def state(self):
        """``(backlog, rr)`` for tests: backlog as ``[(node, label, rids,
        walks)]`` in order, rr as ``{(node, label): batch number}`` plus the
        counter."""
        return self._rt.sched_state()


def _step(done: bool, asked, batch) -> PassStep:
    if not done:
        return PassStep(CheckRequest(*asked))
    return PassStep(None, None if batch is None else _batch(batch))


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
