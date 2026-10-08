"""MicroScheduler token budget per step (``max_step_tokens``).

A walk with a budget gets its oldest ready requests up to that many input
tokens, at least one. The rest go back to ready rather than to the backlog, so
the round-robin can run decode between prefill chunks.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.engine.resources.step import FULL_ADMIT_OK
from mstar.graph.runtime.base import ColumnarEdgeSpecs, PopRidsOutput, ReadyNodeSpec
from mstar.utils.containers import ParallelList
from mstar.worker.micro_scheduler import MicroScheduler, _input_tokens

NODE = "LLM"


class _Runtime:
    """One worker graph: each rid's walk and one input tensor of ``tokens[rid]``
    rows, whose uuid is the rid's index."""

    def __init__(self, tokens: dict[str, int], walks: dict[str, str]):
        self.tokens, self.walks = tokens, walks
        self.ready = set(tokens)
        self.uuid = {rid: i for i, rid in enumerate(tokens)}
        self.rows = {i: n for i, n in enumerate(tokens.values())}

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        grouped: dict[str, list[str]] = {}
        for rid in self.tokens:  # arrival order
            if rid in self.ready and rid not in exclude_rids:
                grouped.setdefault(self.walks[rid], []).append(rid)
        return [ReadyNodeSpec(NODE, walk, rids) for walk, rids in grouped.items()]

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        return "wg0"

    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        rids = [rid for rid in request_ids if rid in self.ready]
        self.ready -= set(rids)
        edges = ColumnarEdgeSpecs.empty()
        for rid in rids:
            edges.add(rid, "input_ids", [self.uuid[rid]], False)
        return PopRidsOutput(
            wg_ids=ParallelList(rids, ["wg0"] * len(rids)), input_edges=edges,
        )

    def push_back_node(self, node_name, rids, wg_ids):
        self.ready |= set(rids)

    def add_first(self, rid: str, tokens: int, walk: str = "prefill"):
        """A new request whose handle scans before every live one, as the Rust
        runtime's recycled handles can."""
        self.tokens = {rid: tokens, **self.tokens}
        self.walks[rid] = walk
        self.ready.add(rid)
        self.uuid[rid] = len(self.rows)
        self.rows[len(self.rows)] = tokens


class _State:
    """Stands in for RequestStateManager."""

    def __init__(self, runtime: _Runtime):
        self.runtime = runtime
        self.per_request_info = dict.fromkeys(runtime.tokens, object())

    def get_partition_for_node(self, node_name):
        return "default"

    def get_graph_walk(self, rid, partition):
        return self.runtime.walks[rid]

    def get_fwd_info(self, rid, partition):
        return object()


class _Engine:
    def __init__(self, max_bs=None):
        self.max_bs = max_bs  # an int for every walk, or walk -> int

    def get_max_batch_size(self, node_name, graph_walk):
        return self.max_bs.get(graph_walk) if isinstance(self.max_bs, dict) else self.max_bs

    def capture_group(self, node_name, graph_walk, rid, fwd_info):
        return None

    def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
        return FULL_ADMIT_OK


def _setup(tokens, walks, budget, max_bs=None):
    engine = _Engine(max_bs)
    runtime = _Runtime(tokens, walks)
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine),
        parallel_leader_nodes={NODE},
        max_step_tokens=None if budget is None else lambda node, walk: budget.get(walk),
        tensor_rows=runtime.rows.__getitem__,
    )
    sched.runtime = runtime
    return sched, runtime, _State(runtime)


def _prefills(*tokens: int, budget=None):
    rids = {f"p{i}": n for i, n in enumerate(tokens)}
    return _setup(rids, dict.fromkeys(rids, "prefill"), budget)


def _rids(batch) -> list[str]:
    return list(batch.request_to_worker_graph)


def test_a_prefill_batch_stops_at_the_token_budget():
    sched, runtime, state = _prefills(1100, 1100, 1100, 1100, 1100, budget={"prefill": 4096})

    batch = sched.get_next_batch(state)

    assert _rids(batch) == ["p0", "p1", "p2"], "3 x 1100 fits 4096, a 4th does not"
    assert runtime.ready == {"p3", "p4"}, "the rest are ready again"
    assert set(batch.input_edges.rids) == {"p0", "p1", "p2"}
    assert sched.backlog == {}


def test_a_request_over_the_budget_still_runs_alone():
    sched, runtime, state = _prefills(5000, 100, budget={"prefill": 4096})

    assert _rids(sched.get_next_batch(state)) == ["p0"]
    assert runtime.ready == {"p1"}


def test_the_budget_takes_requests_in_arrival_order():
    """A small late request does not jump a large early one."""
    sched, _, state = _prefills(1500, 1000, 100, budget={"prefill": 2048})

    assert _rids(sched.get_next_batch(state)) == ["p0"]
    assert _rids(sched.get_next_batch(state)) == ["p1", "p2"]


def test_an_older_request_is_not_passed_over_for_a_recycled_handle():
    """Ready order is handle order in the Rust runtime and handles are recycled, so a
    newer request can scan first; the budget still takes the oldest, or a large one
    is passed over under sustained arrivals."""
    sched, runtime, state = _prefills(3000, 3000, budget={"prefill": 4096})
    assert _rids(sched.get_next_batch(state)) == ["p0"]
    runtime.add_first("p2", 3000)
    state.per_request_info["p2"] = object()
    assert _rids(sched.get_next_batch(state)) == ["p1"]
    assert runtime.ready == {"p2"}


def test_decode_batches_are_unaffected():
    """Only the walk that declares a budget is cut; decode keeps its count cap
    and backlog."""
    rids = {f"d{i}": 1 for i in range(20)}
    sched, _, state = _setup(rids, dict.fromkeys(rids, "decode"), {"prefill": 4}, max_bs=16)

    batch = sched.get_next_batch(state)

    assert len(batch) == 16
    assert len(sched.backlog[(NODE, "decode")]) == 4


def test_decode_runs_between_prefill_chunks():
    """Leftover prefills go back to ready, so the round-robin reaches decode
    next. From the backlog they would go first and stall decode for the set."""
    tokens = {**{f"p{i}": 1100 for i in range(5)}, "d0": 1, "d1": 1}
    walks = {rid: "prefill" if rid.startswith("p") else "decode" for rid in tokens}
    sched, _, state = _setup(tokens, walks, {"prefill": 4096})

    steps = [sched.get_next_batch(state) for _ in range(3)]

    assert [(b.graph_walk, _rids(b)) for b in steps] == [
        ("prefill", ["p0", "p1", "p2"]),
        ("decode", ["d0", "d1"]),
        ("prefill", ["p3", "p4"]),
    ]


def test_a_budgeted_walk_leaves_rows_past_its_cap_ready():
    """Past the row cap too, the rest stay ready rather than backlogged, so a
    burst of short prompts still alternates with decode."""
    tokens = {**{f"p{i}": 100 for i in range(6)}, "d0": 1}
    walks = {rid: "prefill" if rid.startswith("p") else "decode" for rid in tokens}
    sched, runtime, state = _setup(tokens, walks, {"prefill": 4096}, max_bs=4)

    steps = [sched.get_next_batch(state) for _ in range(3)]

    assert [(b.graph_walk, _rids(b)) for b in steps] == [
        ("prefill", ["p0", "p1", "p2", "p3"]),
        ("decode", ["d0"]),
        ("prefill", ["p4", "p5"]),
    ]
    assert sched.backlog == {}


def test_no_budget_takes_the_whole_ready_set():
    sched, _, state = _prefills(1100, 1100, 1100, 1100, 1100)

    assert len(sched.get_next_batch(state)) == 5


def test_input_tokens_read_the_largest_input_edge():
    edges = ColumnarEdgeSpecs.empty()
    edges.add("a", "input_ids", [0, 1], False)  # one edge's tensors add up
    edges.add("a", "extra", [2], False)
    edges.add("b", "input_ids", [], False)
    rows = {0: 600, 1: 500, 2: 7}.__getitem__

    assert _input_tokens(edges, rows) == {"a": 1100, "b": 1}


def test_the_engine_reads_the_budget_off_the_submodule():
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import NodeSubmodule

    sub = SimpleNamespace(max_step_tokens=lambda walk: 4096 if walk == "prefill" else None)
    engine = object.__new__(Engine)
    engine._submodules = {NODE: SimpleNamespace(submodule=sub)}

    assert engine.get_max_step_tokens(NODE, "prefill") == 4096
    assert engine.get_max_step_tokens(NODE, "decode") is None
    assert NodeSubmodule.max_step_tokens(sub, "prefill") is None, "no budget by default"


def test_a_walk_with_no_room_does_not_starve_the_others():
    """Prefill with no free state slot takes nothing, and stays the least recently
    scheduled walk. If it blocked the pick, decode would never run, no request would
    finish, and no slot would ever free: a burst above the pool hung for good."""
    tokens = {"p0": 1000, "d0": 1, "d1": 1}
    walks = {"p0": "prefill", "d0": "decode", "d1": "decode"}
    sched, runtime, state = _setup(tokens, walks, None, max_bs={"prefill": 0, "decode": 16})

    batch = sched.get_next_batch(state)

    assert (batch.graph_walk, _rids(batch)) == ("decode", ["d0", "d1"])
    assert "p0" in runtime.ready, "the prefill stays ready for when a slot frees"


def test_a_full_caller_batch_still_takes_nothing():
    """The other zero: a caller already holding a full batch for its target gets
    None, not a batch of some other walk."""
    tokens = {"d0": 1, "d1": 1}
    sched, _, state = _setup(tokens, dict.fromkeys(tokens, "decode"), None, max_bs=2)

    assert sched.get_next_batch(state, target=(NODE, "decode"), pre_existing_batch_size=2) is None
