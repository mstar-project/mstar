"""MicroScheduler round-robin over walks with no room left.

A walk whose cap leaves no room this step takes nothing, so the round-robin
passes it over rather than picking it and scheduling nothing.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.engine.resources.step import FULL_ADMIT_OK
from mstar.graph.runtime.base import ColumnarEdgeSpecs, PopRidsOutput, ReadyNodeSpec
from mstar.utils.containers import ParallelList
from mstar.worker.micro_scheduler import MicroScheduler

NODE = "LLM"


class _Runtime:
    """One worker graph: each rid's walk and one input tensor."""

    def __init__(self, walks: dict[str, str]):
        self.walks = walks
        self.ready = set(walks)

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        grouped: dict[str, list[str]] = {}
        for rid in self.walks:  # arrival order
            if rid in self.ready and rid not in exclude_rids:
                grouped.setdefault(self.walks[rid], []).append(rid)
        return [ReadyNodeSpec(NODE, walk, rids) for walk, rids in grouped.items()]

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        return "wg0"

    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        rids = [rid for rid in request_ids if rid in self.ready]
        self.ready -= set(rids)
        edges = ColumnarEdgeSpecs.empty()
        for i, rid in enumerate(rids):
            edges.add(rid, "input_ids", [i], False)
        return PopRidsOutput(
            wg_ids=ParallelList(rids, ["wg0"] * len(rids)), input_edges=edges,
        )

    def push_back_node(self, node_name, rids, wg_ids):
        self.ready |= set(rids)

    def set_in_flight(self, node_name, wg_id, rids, in_flight):
        pass


class _State:
    """Stands in for RequestStateManager."""

    def __init__(self, runtime: _Runtime):
        self.runtime = runtime
        self.per_request_info = dict.fromkeys(runtime.walks, object())

    def get_partition_for_node(self, node_name):
        return "default"

    def get_graph_walk(self, rid, partition):
        return self.runtime.walks[rid]

    def get_fwd_info(self, rid, partition):
        return object()


class _Engine:
    def __init__(self, max_bs):
        self.max_bs = max_bs  # an int for every walk, or walk -> int

    def get_max_batch_size(self, node_name, graph_walk):
        return self.max_bs.get(graph_walk) if isinstance(self.max_bs, dict) else self.max_bs

    def get_max_rows_in_combined(self, node_name, graph_walk):
        return None

    def splits_by_capture_key(self, node_name, graph_walk):
        return False

    def capture_group(self, node_name, graph_walk, rid, fwd_info, input_metadata=None):
        return None

    def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
        return FULL_ADMIT_OK


def _setup(walks, max_bs):
    engine = _Engine(max_bs)
    runtime = _Runtime(walks)
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine),
        parallel_leader_nodes={NODE},
    )
    sched.runtime = runtime
    return sched, runtime, _State(runtime)


def _rids(batch) -> list[str]:
    return list(batch.request_to_worker_graph)


def test_a_walk_with_no_room_does_not_starve_the_others():
    """Prefill with no free state slot takes nothing, and stays the least recently
    scheduled walk. If it blocked the pick, decode would never run, no request would
    finish, and no slot would ever free: a burst above the pool hung for good."""
    walks = {"p0": "prefill", "d0": "decode", "d1": "decode"}
    sched, runtime, state = _setup(walks, max_bs={"prefill": 0, "decode": 16})

    batch = sched.get_next_batch(state)

    assert (batch.graph_walk, _rids(batch)) == ("decode", ["d0", "d1"])
    assert "p0" in runtime.ready, "the prefill stays ready for when a slot frees"


def test_a_full_caller_batch_still_takes_nothing():
    """The other zero: a caller already holding a full batch for its target gets
    None, not a batch of some other walk."""
    sched, _, state = _setup(dict.fromkeys(("d0", "d1"), "decode"), max_bs=2)

    assert sched.get_next_batch(state, target=(NODE, "decode"), pre_existing_batch_size=2) is None


def test_a_cap_that_moves_is_read_on_every_pass():
    """A cap can follow free state (GLM-5.3's prefill takes only the KDA slots free
    now). A pass that kept an earlier pass's cap would hand prefill rows it has no
    slot for: refused on one rank, a divergence error on a TP follower."""
    max_bs = {"prefill": 2}
    sched, _, state = _setup({f"p{i}": "prefill" for i in range(4)}, max_bs=max_bs)

    assert _rids(sched.get_next_batch(state)) == ["p0", "p1"]
    max_bs["prefill"] = 1
    assert _rids(sched.get_next_batch(state)) == ["p2"]
    max_bs["prefill"] = 0
    assert sched.get_next_batch(state) is None
    max_bs["prefill"] = 1  # a slot frees
    assert _rids(sched.get_next_batch(state)) == ["p3"]
