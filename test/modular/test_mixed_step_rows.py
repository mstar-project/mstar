"""Rows of one walk riding along in a step of another (``NodeSubmodule.mixed_step_walks``):
the scheduler appends the ready decode requests to a prefill step under their own walk, capped by
the decode walk's cap, only on a leader's own step; the walks survive splitting, merging and the
TP-follow message; the engine's batch answers each row's walk."""
from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.step import FULL_ADMIT_OK, StepContext
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch

NODE = "LLM"


class _Queue:
    def __init__(self, rids):
        self._ready = {rid: {NODE} for rid in rids}
        self.per_request_queues = {rid: SimpleNamespace(ready_node_names={NODE}) for rid in rids}

    def get_ready_node_names(self):
        return {rid: set(names) for rid, names in self._ready.items()}

    def pop_ready_nodes(self, rid, node_names):
        if rid not in self._ready:
            return []
        self._ready.pop(rid)
        self.per_request_queues.pop(rid)
        return [SimpleNamespace(name=node_names[0])]


class _Manager:
    """One worker graph serving both walks of the node; each request is on its own walk."""

    def __init__(self, walks: dict[str, str]):
        self.queues = {"wg0": _Queue(list(walks))}
        self.per_request_info = dict.fromkeys(walks, object())
        self._walks = dict(walks)

    def get_partition_for_node(self, node_name):
        del node_name
        return "default"

    def get_graph_walk(self, rid, partition):
        del partition
        return self._walks[rid]

    def get_fwd_info(self, rid, partition):
        del rid, partition
        return object()

    def get_worker_graph_id_for_node(self, rid, node_name, graph_walk=None):
        del rid, node_name, graph_walk
        return "wg0"


class _Engine:
    def __init__(self, caps: dict[str, int | None], mixed: dict[str, set[str]]):
        self._caps, self._mixed = caps, mixed

    def get_max_batch_size(self, node_name, graph_walk):
        del node_name
        return self._caps.get(graph_walk)

    def mixed_step_walks(self, node_name, graph_walk):
        del node_name
        return self._mixed.get(graph_walk, set())

    def check_ready(self, node_name, rid, fwd_info):
        del node_name, rid, fwd_info
        return FULL_ADMIT_OK


def _scheduler(engine) -> MicroScheduler:
    return MicroScheduler(engine_manager=SimpleNamespace(get_engine=lambda name: engine),
                          parallel_leader_nodes={NODE})


def _walks(n_prefill, n_decode):
    walks = {f"p{i}": "prefill" for i in range(n_prefill)}
    walks.update({f"d{i}": "decode" for i in range(n_decode)})
    return walks


def test_decode_rows_ride_along_in_a_prefill_step_under_their_own_walk():
    mgr = _Manager(_walks(2, 3))
    sched = _scheduler(_Engine(caps={"prefill": 8, "decode": 64}, mixed={"prefill": {"decode"}}))
    # the prefill walk has never run, the decode walk neither: round robin picks prefill first
    sched.node_and_walk_to_last_batch_num[(NODE, "decode")] = 5
    batch = sched.get_next_batch(mgr)
    assert batch.graph_walk == "prefill"
    assert set(batch.node_objects) == {"p0", "p1", "d0", "d1", "d2"}
    assert batch.request_walks == {"d0": "decode", "d1": "decode", "d2": "decode"}
    assert batch.walk_of("p0") == "prefill" and batch.walk_of("d1") == "decode"
    assert list(batch.node_objects)[:2] == ["p0", "p1"], "the step's own rows come first"
    # everything was popped: the next scan finds nothing, and both walks count as scheduled (the
    # riders' walk right after the step's own, so round robin sees both as just served)
    assert sched.get_next_batch(mgr) is None
    marks = sched.node_and_walk_to_last_batch_num
    assert marks[(NODE, "decode")] == marks[(NODE, "prefill")] + 1 == sched.batch_number


def test_riding_rows_respect_their_own_walks_cap():
    mgr = _Manager(_walks(1, 5))
    sched = _scheduler(_Engine(caps={"prefill": 8, "decode": 3}, mixed={"prefill": {"decode"}}))
    sched.node_and_walk_to_last_batch_num[(NODE, "decode")] = 5
    batch = sched.get_next_batch(mgr)
    assert batch.graph_walk == "prefill" and len(batch.request_walks) == 3
    # the other two decode requests are still ready for a decode step of their own
    nxt = sched.get_next_batch(mgr)
    assert nxt.graph_walk == "decode" and set(nxt.node_objects) == {"d3", "d4"} and not nxt.request_walks


def test_a_decode_step_takes_no_prefill_rows_and_a_targeted_call_mixes_nothing():
    mgr = _Manager(_walks(2, 2))
    sched = _scheduler(_Engine(caps={"prefill": 8, "decode": 64}, mixed={"prefill": {"decode"}}))
    sched.node_and_walk_to_last_batch_num[(NODE, "prefill")] = 5  # decode is least recent
    batch = sched.get_next_batch(mgr)
    assert batch.graph_walk == "decode" and set(batch.node_objects) == {"d0", "d1"} and not batch.request_walks
    mgr2 = _Manager(_walks(2, 2))
    sched2 = _scheduler(_Engine(caps={"prefill": 8, "decode": 64}, mixed={"prefill": {"decode"}}))
    targeted = sched2.get_next_batch(mgr2, target=(NODE, "prefill"))
    assert targeted.graph_walk == "prefill" and set(targeted.node_objects) == {"p0", "p1"}
    assert not targeted.request_walks


def test_backlogged_prefill_chunks_take_riders_too():
    mgr = _Manager(_walks(12, 2))
    sched = _scheduler(_Engine(caps={"prefill": 8, "decode": 64}, mixed={"prefill": {"decode"}}))
    sched.node_and_walk_to_last_batch_num[(NODE, "decode")] = 5
    first = sched.get_next_batch(mgr)
    assert first.graph_walk == "prefill" and len(first.node_objects) == 8 + 2
    assert (NODE, "prefill") in sched.backlog  # the four remaining prompts
    mgr._walks.update({"d5": "decode", "d6": "decode"})
    mgr.queues["wg0"]._ready.update({"d5": {NODE}, "d6": {NODE}})
    mgr.queues["wg0"].per_request_queues.update({r: SimpleNamespace(ready_node_names={NODE}) for r in ("d5", "d6")})
    mgr.per_request_info.update(dict.fromkeys(("d5", "d6"), object()))
    second = sched.get_next_batch(mgr)
    assert second.graph_walk == "prefill" and set(second.request_walks) == {"d5", "d6"}
    assert len(second.node_objects) == 4 + 2


def test_split_and_merge_carry_the_walks():
    b = ScheduledBatch(node_name=NODE, graph_walk="prefill", node_objects={r: object() for r in ("p0", "p1", "d0")},
                       request_to_worker_graph=dict.fromkeys(("p0", "p1", "d0"), "wg0"), request_walks={"d0": "decode"})
    head, rest = b.split_off_first(2)
    assert set(head.node_objects) == {"p0", "p1"} and head.request_walks == {}
    assert set(rest.node_objects) == {"d0"} and rest.request_walks == {"d0": "decode"}
    head.merge(rest)
    assert head.request_walks == {"d0": "decode"} and head.walk_of("d0") == "decode"


def test_a_tp_follower_pops_each_rider_under_its_own_walk():
    mgr = _Manager(_walks(1, 2))
    sched = _scheduler(_Engine(caps={"prefill": 8, "decode": 64}, mixed={"prefill": {"decode"}}))
    sched.register_tp_follow(ScheduleTPNode(node_name=NODE, graph_walk="prefill", request_ids=["p0", "d1", "d0"],
                                            spec_seq=7, request_walks={"d0": "decode", "d1": "decode"}))
    batch = sched.get_next_batch(mgr)
    assert batch.tp_seq == 7 and list(batch.node_objects) == ["p0", "d1", "d0"], "the leader's order"
    assert batch.request_walks == {"d0": "decode", "d1": "decode"}
    # a follow batch is reproduced as sent: nothing else is appended to it
    assert sched.get_next_batch(mgr) is None


def test_executing_batch_answers_each_rows_walk():
    batch = ExecutingBatch(node_name=NODE, per_request_info={}, request_walks={"d0": "decode"},
                           step_context=StepContext(request_ids=("p0", "d0"), graph_walk="prefill", slot=0,
                                                    capture=False))
    assert batch.walk_of("p0") == "prefill" and batch.walk_of("d0") == "decode"
