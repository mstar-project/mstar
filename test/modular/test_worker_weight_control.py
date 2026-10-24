"""Pause and weight reload across the ranks of a TP2 instance, worker side.

Two bare ``Worker`` objects play rank 0 and its follower, each with a real
engine holding its TP shard of a tiny Orpheus language model. They are wired to
the conductor's real ``GenerationControl`` over an in-memory bus: a message to a
worker goes through its ``_process_message_list``, and the worker runs the step
from ``_run_weight_op`` as its loop does once nothing is in flight. What the
graph runtime reports idle is stubbed here; both runtimes' counts are compared in
``test/rust/test_runtime_parity.py``.
"""
from __future__ import annotations

import sys
from collections import deque
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_weight_reload import _checkpoint, _engine, _lm, _orpheus, _state

from mstar.conductor.generation_control import GenerationControl
from mstar.distributed.communication import CommGroup
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.utils.ipc_format import (
    ContinueGeneration,
    PauseGeneration,
    UpdateWeightsFromDisk,
    WeightControl,
    WeightOp,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.worker import Worker

WORKERS = ["worker_0", "worker_1"]


class _Runtime:
    """``idle_counts`` only: requests held, routes parked, ingests staged."""

    def __init__(self):
        self.counts = (0, 0, 0)

    def idle_counts(self):
        return self.counts


class _FailingAfterCopy(OrpheusModel):
    """Copies the checkpoint, then fails, as a read error partway would."""

    fail = True

    def reload_weights(self, path, submodules, *, check_only=False):
        super().reload_weights(path, submodules, check_only=check_only)
        if not check_only and self.fail:
            raise OSError("read error partway through the checkpoint")


class _Deployment:
    def __init__(self, engines):
        self.bus: deque = deque()
        self.answers = []
        self.now = 0.0
        # workers whose loop is stuck elsewhere: they take messages but run no step
        self.stalled: set[str] = set()
        self.workers = {wid: self._worker(wid, engine) for wid, engine in zip(WORKERS, engines, strict=True)}
        self.control = GenerationControl(
            workers=WORKERS, send=self._from_conductor,
            reject_queued=lambda: None, abort_in_flight=lambda: None,
            timeout_s=10.0, clock=lambda: self.now,
        )

    def _worker(self, worker_id, engine):
        w = Worker.__new__(Worker)
        w.worker_id = worker_id
        w.communicator = SimpleNamespace(send=lambda peer, msg: self.bus.append((peer, msg)))
        w._graph_runtime = _Runtime()
        w._in_flight_rids = set()
        w._pending_removes = {}
        w._pending_drains = set()
        w._removes_awaiting_step = {}
        w.scheduler = SimpleNamespace(tp_batches_pending_schedule=deque())
        w.engine_manager = SimpleNamespace(engine=engine)
        return w

    def _from_conductor(self, peer, message):
        if peer == "api_server":
            self.answers.append(message.body)
        else:
            self.bus.append((peer, message))

    def run(self, passes=3):
        """Deliver everything, then give each worker a loop pass; repeat."""
        for _ in range(passes):
            while self.bus:
                peer, message = self.bus.popleft()
                if peer == "conductor":
                    self.control.ack(message.body)
                else:
                    self.workers[peer]._process_message_list([message])
            for worker_id, worker in self.workers.items():
                if worker._weight_op is not None and worker_id not in self.stalled:
                    worker._run_weight_op()
            self.control.tick(0)

    def answer(self, op_id):
        return [a for a in self.answers if a.op_id == op_id][-1]

    def pause(self):
        self.control.pause(PauseGeneration(op_id="pause", mode="wait"))
        self.run()
        return self.answer("pause")


def _rank(i):
    return CommGroup(my_global_rank=i, my_group_rank=i, group_members=[0, 1])


@pytest.fixture
def ckpts(tmp_path):
    a, b = _state(seed=0), _state(seed=1)
    return SimpleNamespace(
        a=a, b=b, a_dir=_checkpoint(tmp_path / "a", a), b_dir=_checkpoint(tmp_path / "b", b),
    )


def _tp2(ckpts, follower_model=None):
    lms = [_lm(ckpts.a_dir, comm_group=_rank(i)) for i in range(2)]
    models = [_orpheus(ckpts.a_dir), follower_model or _orpheus(ckpts.a_dir)]
    return lms, _Deployment([_engine(m, lm) for m, lm in zip(models, lms, strict=True)])


def _shard(state, rank):
    q, kv = 32 // 2, 8  # q rows and KV rows per rank, see test_weight_reload._config
    p = "model.layers.0.self_attn."
    return torch.cat([
        state[p + "q_proj.weight"][rank * q:(rank + 1) * q],
        state[p + "k_proj.weight"][rank * kv:(rank + 1) * kv],
        state[p + "v_proj.weight"][rank * kv:(rank + 1) * kv],
    ])


# ── idle ────────────────────────────────────────────────────────────────────


def test_the_follower_confirms_idle_only_once_rank_0s_stamped_removal_lands(ckpts):
    _lms, d = _tp2(ckpts)
    follower = d.workers["worker_1"]
    # rank 0 released "r" after its step 5, which this rank has not run yet
    follower._graph_runtime.counts = (1, 0, 0)
    follower._removes_awaiting_step = {5: ["r"]}
    follower.scheduler.tp_batches_pending_schedule.append(object())

    d.control.pause(PauseGeneration(op_id="pause", mode="wait"))
    d.run()
    assert [a for a in d.answers if a.op_id == "pause"] == []
    assert follower._weight_op is not None, "the follower answered while holding work"
    assert follower._idle_blockers() == [
        "1 request(s) still held", "1 teardown(s) not yet applied",
        "1 committed TP step(s) not yet run",
    ]

    # the step ran, and the removal stamped after it was applied
    follower.scheduler.tp_batches_pending_schedule.clear()
    follower._removes_awaiting_step = {}
    follower._graph_runtime.counts = (0, 0, 0)
    d.run()
    assert d.answer("pause").status == 200


@pytest.mark.parametrize("blocker, expected", [
    (lambda w: w._in_flight_rids.add(3), "a step in flight"),
    (lambda w: setattr(w._graph_runtime, "counts", (0, 1, 0)), "step output(s) not yet sent"),
    (lambda w: setattr(w._graph_runtime, "counts", (0, 0, 2)), "staged speculative ingest"),
    (lambda w: w._pending_removes.setdefault(4), "teardown(s) not yet applied"),
])
def test_a_busy_rank_refuses_to_copy(ckpts, blocker, expected):
    lms, d = _tp2(ckpts)
    worker = d.workers["worker_0"]
    blocker(worker)
    worker._process_message_list([WorkerMessage(
        message_type=WorkerMessageType.WEIGHT_CONTROL,
        body=WeightControl(op=WeightOp.APPLY.value, epoch=1, model_path=ckpts.b_dir, weight_version="v1"),
    )])
    worker._run_weight_op()
    ack = d.bus.popleft()[1].body
    assert ack.error.startswith("not idle") and expected in ack.error
    assert not ack.dirty and ack.weight_version is None
    assert torch.equal(lms[0].model.layers[0].self_attn.qkv_proj.weight, _shard(ckpts.a, 0))


# ── reload ──────────────────────────────────────────────────────────────────


def test_both_ranks_reload_their_shards_in_one_epoch_and_commit(ckpts):
    lms, d = _tp2(ckpts)
    pointers = [{n: p.data_ptr() for n, p in lm.named_parameters()} for lm in lms]
    assert d.pause().status == 200

    d.control.update(UpdateWeightsFromDisk(op_id="u", model_path=ckpts.b_dir, weight_version="v1"))
    d.run()
    answer = d.answer("u")
    assert (answer.status, answer.weight_version) == (200, "v1")
    assert answer.rank_versions == dict.fromkeys(WORKERS, "v1")
    for rank, lm in enumerate(lms):
        assert {n: p.data_ptr() for n, p in lm.named_parameters()} == pointers[rank]
        assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _shard(ckpts.b, rank))

    d.control.resume(ContinueGeneration(op_id="c"))
    d.run()
    assert d.answer("c").status == 200 and not d.control.paused


def test_a_follower_that_fails_mid_copy_keeps_the_deployment_paused(ckpts):
    failing = _FailingAfterCopy.__new__(_FailingAfterCopy)
    failing.config, failing._llm_weights_dir = _orpheus(ckpts.a_dir).config, ckpts.a_dir
    lms, d = _tp2(ckpts, follower_model=failing)
    assert d.pause().status == 200

    d.control.update(UpdateWeightsFromDisk(op_id="u", model_path=ckpts.b_dir, weight_version="v1"))
    d.run()
    answer = d.answer("u")
    assert answer.status == 500 and answer.paused
    assert answer.ranks["worker_0"] == "ok"
    assert answer.ranks["worker_1"].startswith("OSError: read error partway")
    assert answer.rank_versions == {"worker_0": "v1", "worker_1": None}
    assert d.workers["worker_1"]._weights_dirty
    assert d.control.weight_version is None

    d.control.resume(ContinueGeneration(op_id="c1"))
    d.run()
    assert d.answer("c1").status == 409 and d.control.paused, "resumed with ranks on different weights"

    failing.fail = False
    d.control.update(UpdateWeightsFromDisk(op_id="u2", model_path=ckpts.b_dir, weight_version="v1"))
    d.run()
    assert d.answer("u2").status == 200
    d.control.resume(ContinueGeneration(op_id="c2"))
    d.run()
    assert d.answer("c2").status == 200


def test_a_rank_that_misses_a_copy_under_a_reused_label_keeps_the_deployment_paused(ckpts, tmp_path):
    """v1 is committed from B; C is then copied under the same label while the
    follower is stuck, so its apply times out and a later verify supersedes it.
    Both ranks say "v1" but hold different weights: continue must refuse."""
    c_state = _state(seed=2)
    c_dir = _checkpoint(tmp_path / "c", c_state)
    lms, d = _tp2(ckpts)
    assert d.pause().status == 200
    d.control.update(UpdateWeightsFromDisk(op_id="u1", model_path=ckpts.b_dir, weight_version="v1"))
    d.run()
    assert d.answer("u1").status == 200

    d.control.update(UpdateWeightsFromDisk(op_id="u2", model_path=c_dir, weight_version="v1"))
    d.run(passes=1)  # both ranks check C
    d.stalled.add("worker_1")
    d.run()
    d.now += 11.0
    d.run(passes=1)
    assert d.answer("u2").status == 504

    d.control.resume(ContinueGeneration(op_id="c"))
    d.stalled.clear()
    d.run()
    assert torch.equal(lms[0].model.layers[0].self_attn.qkv_proj.weight, _shard(c_state, 0))
    assert torch.equal(lms[1].model.layers[0].self_attn.qkv_proj.weight, _shard(ckpts.b, 1))
    assert d.answer("c").status == 409 and d.control.paused, "resumed with B on one rank and C on the other"


def test_a_checkpoint_one_rank_refuses_copies_nothing_anywhere(ckpts, tmp_path):
    lms, d = _tp2(ckpts)
    # the follower was loaded from a deeper config, so it refuses this one
    deeper = _checkpoint(tmp_path / "deeper", ckpts.a, extra_config={"num_hidden_layers": 3})
    d.workers["worker_1"].engine_manager.engine._model._llm_weights_dir = deeper
    assert d.pause().status == 200

    d.control.update(UpdateWeightsFromDisk(op_id="u", model_path=ckpts.b_dir, weight_version="v1"))
    d.run()
    answer = d.answer("u")
    assert answer.status == 400 and answer.ranks["worker_0"] == "ok"
    assert "num_hidden_layers" in answer.ranks["worker_1"]
    for rank, lm in enumerate(lms):
        assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _shard(ckpts.a, rank))
    assert not any(w._weights_dirty for w in d.workers.values())

    d.control.resume(ContinueGeneration(op_id="c"))
    d.run()
    assert d.answer("c").status == 200, "a refused update must not block serving the old weights"
