"""A TP leader's early speculation takes a step's decision exactly once: an
early build that finds a head broadcasts it; one that finds nothing sends no
marker and schedules no yield-away (the top of the next iteration decides).
The TP1 early build still decides on the spot."""
from types import SimpleNamespace

from mstar.worker.worker import Worker, _SpecFairness


class _Scheduler:
    def __init__(self, other_ready=False, yield_batch=None):
        self.other_ready = other_ready
        self.yield_batch = yield_batch
        self.get_next_calls = 0

    def has_ready_excluding(self, request_state, target):
        return self.other_ready

    def get_next_batch(self, request_state, exclude_target=None):
        self.get_next_calls += 1
        return self.yield_batch


def _leader(head=None, other_ready=False):
    w = Worker.__new__(Worker)
    w.enable_nvtx = False
    w._phase_period = 0
    w.request_state = object()
    w.scheduler = _Scheduler(other_ready=other_ready)
    w.markers = []
    w.heads = []
    w._can_speculate = lambda batch: True
    w._try_speculate_next = lambda pending: head
    w._tp_lead_needs_marker = lambda pending, spec: spec is None or spec.tp_seq < 0
    w._broadcast_tp_nospec = lambda pending: w.markers.append(pending.tp_seq)

    def _send(node_batch, speculative=False, spec_from_seq=None):
        w.heads.append((speculative, spec_from_seq))
        return 7

    w.maybe_send_zmq_to_tp_followers = _send
    return w


def _pending(seq=3):
    return SimpleNamespace(
        tp_seq=seq, node_name="LLM", graph_walk="decode",
        batch=SimpleNamespace(request_to_worker_graph={1: 0}),
        node_batch=SimpleNamespace(admit_error=None),
    )


def _fair(**kw):
    return _SpecFairness(peek=True, hold_s=0.0, max_consecutive=1024, **kw)


def test_an_early_build_with_no_head_sends_nothing_and_leaves_the_decision():
    w = _leader(head=None)
    spec, target = w._build_speculation(_pending(), 1, _fair(), decide=False)
    assert spec is None and target is None
    assert w.markers == [] and w.heads == [] and w.scheduler.get_next_calls == 0
    # the full call afterwards takes exactly one decision: the marker
    spec, target = w._build_speculation(_pending(), 1, _fair(), decide=True)
    assert spec is None and w.markers == [3] and w.heads == []
    assert w.scheduler.get_next_calls == 1


def test_an_early_build_with_a_head_commits_it():
    head = SimpleNamespace(node_batch=object(), tp_seq=-1)
    w = _leader(head=head)
    spec, target = w._build_speculation(_pending(), 1, _fair(), decide=False)
    assert spec is head and spec.tp_seq == 7 and target is None
    assert w.heads == [(True, 3)] and w.markers == []


def test_a_fairness_yield_in_the_early_slot_is_deferred_on_a_parallel_node():
    w = _leader(head=SimpleNamespace(node_batch=object(), tp_seq=-1), other_ready=True)
    fair = _fair()
    spec, target = w._build_speculation(_pending(), 1, fair, decide=False)
    # other work is ready and the hold is 0: a yield would be due, but the
    # early build is non-committal, so nothing is sent or scheduled
    assert spec is None and target is None
    assert w.heads == [] and w.markers == [] and w.scheduler.get_next_calls == 0
    # the deciding call yields: marker first, then the yield-away schedule
    spec, target = w._build_speculation(_pending(), 1, fair, decide=True)
    assert spec is None and target == ("LLM", "decode")
    assert w.markers == [3] and w.scheduler.get_next_calls == 1


def test_a_tp1_early_build_decides_on_the_spot():
    w = _leader(head=None)
    w._tp_lead_needs_marker = lambda pending, spec: False  # not a parallel node
    spec, target = w._build_speculation(_pending(), 1, _fair(), decide=True)
    assert spec is None and w.markers == [] and w.scheduler.get_next_calls == 1
