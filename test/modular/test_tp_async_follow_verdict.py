"""TP async scheduling — a follower's void verdict must reach a speculative
head whether the head is already queued when step N completes OR arrives
afterwards.

The leader broadcasts its speculation DURING its forward N; a follower can
finish N first (it is the faster rank, or the leader's build was slow). If the
verdict were applied only to the head-at-await, a head landing later would go
to the serial path with the leader's pre-verdict composition — and run a batch
the leader will not (allocation failure on N → the leader clears its
speculation), or with rids the leader dropped: a composition desync, the
NCCL-hang class. So the verdict is recorded per step and applied both at
await time (``_reconcile_tp_follow_head``) and on arrival
(``_register_tp_follow``).

Drives the real ``Worker`` methods unbound on a minimal stub, with a real
``MicroScheduler`` for the FIFO, the pattern ``test_worker_message_handling``
uses.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphEdge, GraphNode  # noqa: E402
from mstar.utils.ipc_format import ScheduleTPNode  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

NODE = "LLM"
WALK = "decode"


class _FakeEngineManager:
    def get_engine(self, node_name):
        return types.SimpleNamespace(check_ready=lambda *a: True)


def _node():
    # AR decode: two loop-back outputs feed the same node next iter.
    return GraphNode(
        name=NODE, input_names={"token", "kv_cache"},
        outputs=[
            GraphEdge(name="token", next_node=NODE),
            GraphEdge(name="kv_cache", next_node=NODE),
        ],
    )


def _stub():
    stub = types.SimpleNamespace()
    stub.worker_id = "w1"
    stub.tp_async_sched = True
    stub._tp_step_verdicts = {}
    stub._TP_VERDICTS_KEEP = Worker._TP_VERDICTS_KEEP
    stub.scheduler = MicroScheduler(
        engine_manager=_FakeEngineManager(), parallel_leader_nodes=set(),
    )
    # bound-method plumbing for the unbound calls below
    stub._apply_tp_step_verdict = lambda m: Worker._apply_tp_step_verdict(stub, m)
    stub._record_tp_step_verdict = lambda p, o: Worker._record_tp_step_verdict(stub, p, o)
    stub._register_tp_follow = lambda m: Worker._register_tp_follow(stub, m)
    stub._reconcile_tp_follow_head = lambda p, o: Worker._reconcile_tp_follow_head(stub, p, o)
    return stub


def _pending(rids, tp_seq):
    batch = types.SimpleNamespace(node_objects={r: _node() for r in rids})
    return types.SimpleNamespace(batch=batch, node_name=NODE, graph_walk=WALK, tp_seq=tp_seq)


def _output(ok_rids=(), allocation_failed=False, failed_requests=()):
    return types.SimpleNamespace(
        allocation_failed=allocation_failed,
        failed_requests=list(failed_requests),
        per_request_output_tensors={
            r: {"token": ["t"], "kv_cache": ["k"]} for r in ok_rids
        },
    )


def _head(rids, seq, from_seq):
    return ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=list(rids),
        speculative=True, spec_seq=seq, spec_from_seq=from_seq,
    )


# ---------------------------------------------------------- head already queued

def test_queued_head_dropped_when_parent_step_allocation_failed():
    s = _stub()
    s.scheduler.register_tp_follow(_head(["r0", "r1"], seq=5, from_seq=4))
    s._reconcile_tp_follow_head(_pending(["r0", "r1"], 4), _output(allocation_failed=True))
    assert s.scheduler.peek_tp_follow() is None


def test_queued_head_dropped_when_parent_step_had_failed_requests():
    s = _stub()
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    s._reconcile_tp_follow_head(_pending(["r0"], 4), _output(ok_rids=["r0"], failed_requests=["r0"]))
    assert s.scheduler.peek_tp_follow() is None


def test_queued_head_shrinks_to_survivors_keeps_fresh_and_order():
    s = _stub()
    # r1 (continuing) emits no loop-back output; f2 is fresh (not in N)
    s.scheduler.register_tp_follow(_head(["r0", "r1", "f2"], seq=5, from_seq=4))
    s._reconcile_tp_follow_head(_pending(["r0", "r1"], 4), _output(ok_rids=["r0"]))
    head = s.scheduler.peek_tp_follow()
    assert head is not None
    assert head.request_ids == ["r0", "f2"]
    assert head.spec_seq == 5 and head.spec_from_seq == 4 and head.speculative


def test_queued_head_untouched_when_everything_survives():
    s = _stub()
    original = _head(["r0", "r1"], seq=5, from_seq=4)
    s.scheduler.register_tp_follow(original)
    s._reconcile_tp_follow_head(_pending(["r0", "r1"], 4), _output(ok_rids=["r0", "r1"]))
    assert s.scheduler.peek_tp_follow() is original


def test_head_from_a_different_parent_is_left_alone_at_await():
    s = _stub()
    other = _head(["r0"], seq=9, from_seq=8)   # speculated from a later step
    s.scheduler.register_tp_follow(other)
    s._reconcile_tp_follow_head(_pending(["r0"], 4), _output(allocation_failed=True))
    assert s.scheduler.peek_tp_follow() is other


# ------------------------------------------------- head arrives AFTER the await

def test_late_head_dropped_on_arrival_when_parent_was_voided():
    s = _stub()
    # Follower finished step 4 (alloc failed) before the leader's head landed.
    s._reconcile_tp_follow_head(_pending(["r0"], 4), _output(allocation_failed=True))
    assert s.scheduler.peek_tp_follow() is None
    s._register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    assert s.scheduler.peek_tp_follow() is None, "voided head must never enter the FIFO"


def test_late_head_shrunk_on_arrival_to_the_recorded_survivors():
    s = _stub()
    s._reconcile_tp_follow_head(_pending(["r0", "r1"], 4), _output(ok_rids=["r1"]))
    s._register_tp_follow(_head(["r0", "r1", "f2"], seq=5, from_seq=4))
    head = s.scheduler.peek_tp_follow()
    assert head is not None and head.request_ids == ["r1", "f2"]


def test_late_head_with_no_survivors_never_enters_fifo():
    s = _stub()
    s._reconcile_tp_follow_head(_pending(["r0"], 4), _output(ok_rids=[]))
    s._register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    assert s.scheduler.peek_tp_follow() is None


def test_head_for_an_in_flight_parent_registers_unchanged():
    s = _stub()
    h = _head(["r0"], seq=5, from_seq=4)   # step 4 has no verdict yet
    s._register_tp_follow(h)
    assert s.scheduler.peek_tp_follow() is h


def test_non_speculative_message_never_touched_by_verdicts():
    s = _stub()
    s._reconcile_tp_follow_head(_pending(["r0"], 4), _output(allocation_failed=True))
    plain = ScheduleTPNode(NODE, WALK, ["r0"], spec_seq=5)   # serial broadcast
    s._register_tp_follow(plain)
    assert s.scheduler.peek_tp_follow() is plain


def test_flag_off_registers_verbatim():
    s = _stub()
    s.tp_async_sched = False
    s._tp_step_verdicts[4] = (frozenset({"r0"}), frozenset(), True)
    h = _head(["r0"], seq=5, from_seq=4)
    s._register_tp_follow(h)
    assert s.scheduler.peek_tp_follow() is h


def test_verdict_store_is_bounded():
    s = _stub()
    for seq in range(Worker._TP_VERDICTS_KEEP + 40):
        s._reconcile_tp_follow_head(_pending(["r0"], seq), _output(ok_rids=["r0"]))
    assert len(s._tp_step_verdicts) == Worker._TP_VERDICTS_KEEP
    assert min(s._tp_step_verdicts) == 40   # oldest evicted first
