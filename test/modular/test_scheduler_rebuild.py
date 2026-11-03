"""A rebuild runs as rows mid-prefill that the worker's log holds, offered where a chunk in flight is.

Readiness answers a dropped request with a rebuild. The scheduler opens the rebuild in the log, and the request's
decode row waits until the log lands it, even if the request is let back in for room meanwhile. From then on the
log offers the rebuild's next chunk
the way the scheduler offers a speculation's chunk in flight, so the walk's caps, token budget and pick treat it
as a prompt mid-prefill. Nothing popped it, so nothing pushes it back or parks it in a backlog, and a chunk in
flight is offered again only as its step's continuation.
"""

from __future__ import annotations

import sys
from array import array
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_rebuild_policy import POLICY, _graphs
from test_scheduler_room import NODE, _rids, _Runtime
from test_scheduler_room import _Engine as _RoomEngine
from test_scheduler_room import _State as _RoomState

from mstar.engine.chunking import ChunkRange
from mstar.engine.resources.step import FULL_ADMIT_OK, FULL_ADMIT_REBUILD
from mstar.model.submodule_base import InputSeqLenInfo
from mstar.worker.host_pool import HostCopy, HostSlice
from mstar.worker.micro_scheduler import MicroScheduler, MidPrefillRow
from mstar.worker.rebuild import KeptIds, RebuildCursor, Rebuilds

PREFILL, DECODE, MIXED = POLICY.walk, "decode", "mixed"
R = "r"


class _State(_RoomState):
    """Forward-pass infos that name each request's walk, which a rebuild starts from."""

    def get_fwd_info(self, rid, partition):
        return SimpleNamespace(graph_walk=self.runtime.walks.get(rid))


class _Engine(_RoomEngine):
    """Readiness that answers ``dropped`` with a rebuild, a token budget, and ``cached`` tokens of every rebuild
    in the prefix cache."""

    def __init__(self, max_bs=8, budget=None, chunkable=True, cached=0, dropped=()):
        super().__init__(max_bs)
        self.budget, self.chunkable, self.cached, self.dropped = budget, chunkable, cached, set(dropped)
        self.asked, self.prepared = [], []

    def get_max_batch_tokens(self, node_name, graph_walk):
        return self.budget

    def supports_chunked_prefill(self, node_name, graph_walk):
        return self.chunkable

    def min_chunk_tokens(self, budget):
        return 1

    def walk_token_caps(self, node_name, graph_walk):
        return None

    def unserved_len(self, node_name, graph_walk, info, rid, length, end):
        return InputSeqLenInfo(end - self.cached, cached_prefix=self.cached)

    def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
        self.asked.append(rid)
        return FULL_ADMIT_REBUILD if rid in self.dropped else FULL_ADMIT_OK

    def prepare_rebuild(self, node_name, rid, rebuild):
        self.prepared.append((rid, list(rebuild.ids)))

    def admit_waiting(self, published_of):
        return []


def _scheduler(walks, combined=False, **engine):
    engine = _Engine(**engine)
    runtime = _Runtime(walks)
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine), parallel_leader_nodes={NODE},
        combined_walk_of={(NODE, PREFILL): MIXED, (NODE, DECODE): MIXED} if combined else None,
    )
    sched.runtime, sched.rebuilds = runtime, Rebuilds({NODE: POLICY}, _graphs())
    return sched, runtime, _State(runtime), engine


def _rebuilding(sched, rid=R, total=20, done=0):
    """``rid``'s rebuild on ``NODE``, ``done`` of its ``total`` tokens in."""
    sched.rebuilds._cursors[(NODE, rid)] = RebuildCursor(
        PREFILL, total, total, array("i", range(total + 1)), torch.int64, done,
    )


def test_a_dropped_requests_decode_row_waits_and_starts_one_rebuild():
    sched, runtime, state, engine = _scheduler({"d0": DECODE}, dropped={"d0"})
    prompt = torch.tensor([1, 2, 3], dtype=torch.int64)
    sched.rebuilds._kept[NODE]["d0"] = KeptIds(HostCopy(HostSlice(0, 24), prompt, None), array("i", [4]))

    sched.get_next_batch(state)
    sched.get_next_batch(state)

    assert engine.prepared == [("d0", [1, 2, 3, 4])], "the KV would lay the stream out twice, or not at all"
    assert sched.rebuilds.rebuilding("d0", NODE) and "d0" in runtime.ready, "the decode row ran before its KV"
    assert engine.asked.count("d0") == 1, "a waiting decode row asks the engine on every pass"


def test_a_decode_row_runs_once_its_rebuild_lands():
    sched, runtime, state, engine = _scheduler({"d0": DECODE})
    _rebuilding(sched, "d0")

    assert sched.get_next_batch(state).rebuild_rids == {"d0"} and not engine.asked
    sched.rebuilds.land_chunk("d0", NODE, None)

    assert _rids(sched.get_next_batch(state)) == ["d0"], "the rebuilt request would never decode again"


def test_a_request_let_back_in_for_room_mid_rebuild_starts_no_second_rebuild():
    sched, runtime, state, engine = _scheduler({"d0": DECODE}, dropped={"d0"})
    _rebuilding(sched, "d0")
    sched.parked[NODE] = {"d0"}  # it asked for room again, and the room is there now
    engine.admit_waiting = lambda published_of: [(NODE, "d0")]

    batch = sched.get_next_batch(state)

    assert batch.rebuild_rids == {"d0"} and not engine.asked, "the decode row would run before its KV is back"


def test_a_rebuild_row_shares_a_step_with_its_walks_decode_rows():
    sched, runtime, state, _ = _scheduler({"d0": DECODE}, combined=True)
    _rebuilding(sched)

    batch = sched.get_next_batch(state)

    assert _rids(batch) == [R, "d0"] and batch.rebuild_rids == {R}, "the rebuild would take a step of its own"
    assert batch.graph_rows() == {"d0": "wg0"}, "the graph runtime would be handed a row it holds no node of"


def test_a_long_rebuild_is_cut_by_the_walks_budget_and_resumes_where_it_stopped():
    sched, _, state, _ = _scheduler({}, budget=8)
    _rebuilding(sched, total=20)

    first = sched.get_next_batch(state)
    sched.rebuilds.land_chunk(R, NODE, first.chunk_ranges[R])
    second = sched.get_next_batch(state)

    assert first.chunk_ranges[R] == ChunkRange(0, 8, 20, 8) and first.incomplete_node_rids == {R}
    assert second.chunk_ranges[R] == ChunkRange(8, 16, 20, 8), "the rebuild would run past the budget, or again"


@pytest.mark.parametrize("engine", [{"budget": None}, {"budget": 8, "chunkable": False}])
def test_a_rebuild_of_a_walk_the_deployment_doesnt_chunk_runs_whole(engine):
    sched, _, state, _ = _scheduler({}, **engine)
    _rebuilding(sched, total=20)

    batch = sched.get_next_batch(state)

    assert _rids(batch) == [R] and not batch.chunk_ranges, "a walk that can't be cut would be split"


def test_a_rebuild_starts_past_its_cached_prefix():
    sched, _, state, _ = _scheduler({}, budget=8, cached=8)
    _rebuilding(sched, total=20)

    batch = sched.get_next_batch(state)

    assert batch.chunk_ranges[R] == ChunkRange(8, 16, 20, 16), "the rebuild would recompute what the cache holds"


class _Prefilling(_Runtime):
    """Every prompt stands 4 tokens into its 10 on its chunk Loop."""

    def loop_progress(self, node_name, rids, graph_walks, loop_names):
        return [4] * len(rids), [10] * len(rids)


def test_a_prompt_dropped_mid_prefill_runs_its_chunks_again_up_to_where_its_loop_stands():
    sched, _, _, engine = _scheduler({}, budget=8, dropped={R})
    sched.runtime = runtime = _Prefilling({R: PREFILL})
    sched.chunk_loop_walks = {NODE: frozenset({PREFILL})}
    state = _State(runtime)

    rebuild = sched.get_next_batch(state)

    assert sched.rebuilds.rebuilding(R, NODE) and not engine.prepared, (
        "the KV would lay the prompt out again over unkept ids"
    )
    assert rebuild.rebuild_rids == {R} and rebuild.chunk_ranges[R] == ChunkRange(0, 4, 4, 4), (
        "the rebuild would run past where the prompt stood, or whole, with no input of its own"
    )
    assert (R, NODE) in sched.chunk_started, "the prompt's next chunk would lose its place ahead of newer prompts"


def test_a_chunk_in_flight_is_offered_only_as_its_steps_continuation():
    sched, _, state, _ = _scheduler({}, budget=8)
    _rebuilding(sched, total=20, done=8)
    sched.in_flight_rids.add(R)

    assert sched.get_next_batch(state) is None, "a chunk in flight would run a second time"
    chain = sched.pick_chain_step(state, (NODE, PREFILL), [], [MidPrefillRow(R, "wg0", PREFILL, 16, 20)])
    batch = sched.take_chain_step(state, chain)

    assert batch.chunk_ranges[R] == ChunkRange(16, 20, 20, 4) and batch.rebuild_rids == {R}


@pytest.mark.parametrize("why", ["held", "failed", "removing"])
def test_a_rebuild_whose_request_the_scheduler_wouldnt_run_waits(why):
    sched, _, state, _ = _scheduler({})
    _rebuilding(sched)
    if why == "held":
        sched.hold_requests([R])
    elif why == "failed":
        sched.failed_rids.add(R)
    else:
        sched.pending_removes[R] = None

    assert sched.get_next_batch(state) is None, "a torn-down or held request would prefill again"
    assert sched.rebuilds.rebuilding(R, NODE), "the rebuild left the log before anything decided its request"


def test_a_rebuild_left_out_of_a_full_step_stays_in_the_log():
    sched, _, state, _ = _scheduler({"p0": PREFILL}, max_bs=1)
    sched.note_arrival("p0")
    sched.note_arrival(R)
    _rebuilding(sched)

    assert _rids(sched.get_next_batch(state)) == ["p0"]
    assert not any(R in batch.request_to_worker_graph for batch in sched.backlog.values()), (
        "a backlog would hold a row no queue gave it"
    )
    assert _rids(sched.get_next_batch(state)) == [R], "the rebuild would never run"


def test_a_mixing_chain_takes_a_waiting_rebuild_row():
    sched, _, state, _ = _scheduler({}, combined=True, budget=64)
    _rebuilding(sched)

    chain = sched.pick_chain_step(state, (NODE, DECODE), ["c0"])
    batch = sched.take_chain_step(state, chain, pre_existing_batch_size=1)

    assert [row.request_id for row in chain.rebuilds] == [R] and batch.rebuild_rids == {R}, (
        "decode would sit out every chunk of the rebuild"
    )


def test_a_chain_that_doesnt_mix_yields_to_a_rebuild_of_another_walk():
    sched, runtime, state, _ = _scheduler({})
    runtime.has_ready_excluding = lambda exclude, exclude_target: False  # nothing on the queues
    _rebuilding(sched)

    assert sched.has_ready_excluding(state, (NODE, DECODE)), "a decode chain would speculate past the rebuild"
    assert not sched.has_ready_excluding(state, (NODE, PREFILL)), "a chain of the rebuild's walk takes it itself"
    sched.in_flight_rids.add(R)
    assert not sched.has_ready_excluding(state, (NODE, DECODE)), "a chain would yield to a chunk in flight"


def test_no_chunk_loop_is_capped_for_a_rebuild_row():
    sched, runtime, state, _ = _scheduler({}, budget=8)
    sched.chunk_loop_walks = {NODE: frozenset({PREFILL})}
    capped = []
    runtime.set_loop_caps = lambda node, rids, walks, loops, caps: capped.extend(rids)
    _rebuilding(sched)

    sched.get_next_batch(state)

    assert not capped, "the walk's chunk Loop would be capped for a request that left it"


def test_pending_rows_and_a_refill_count_waiting_rebuilds():
    sched, _, state, _ = _scheduler({"p0": PREFILL})
    _rebuilding(sched)
    _rebuilding(sched, "r1")
    sched.in_flight_rids.add("r1")

    assert sched.pending_rows()[NODE] == 2, "the worker would report its node idle with a rebuild waiting"
    assert sched.could_refill(NODE, PREFILL, {"p0"}), "a refused step would hold its row though the rebuild refills it"


def test_a_chain_takes_a_rebuild_and_leaves_the_requests_parked_decode_row_queued():
    sched, runtime, state, _ = _scheduler({"c0": DECODE, "d0": DECODE, "d1": DECODE}, combined=True, budget=64)
    queues = runtime.get_ready_nodes
    runtime.get_ready_nodes = lambda exclude, target=None, exclude_target=None: queues(exclude)  # all one mixed key
    runtime.ready.discard("c0")  # the chain's own row, in flight
    sched.parked[NODE] = {"d0"}
    _rebuilding(sched, "d0")

    chain = sched.pick_chain_step(state, (NODE, DECODE), ["c0"])
    batch = sched.take_chain_step(state, chain, pre_existing_batch_size=1)

    assert _rids(batch) == ["d0", "d1"] and batch.rebuild_rids == {"d0"}
    assert "d0" in runtime.ready, "the parked decode row would be popped into its own rebuild's step"
