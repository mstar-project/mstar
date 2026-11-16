"""Pause and weight reload across two TP ranks' weight-control components."""
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
from mstar.model.loader.reload import reload_manifest
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.utils.ipc_format import (
    CheckWeights,
    ContinueGeneration,
    PauseGeneration,
    UpdateWeightsFromDisk,
    WeightControl,
    WeightOp,
)
from mstar.worker.weight_ops import WeightOps

WORKERS = ["worker_0", "worker_1"]


class _Runtime:
    """``idle_counts`` (requests held, routes parked, ingests staged) and the
    version the worker hands it for result frames."""

    def __init__(self):
        self.counts = (0, 0, 0)
        self.weight_version = None

    def idle_counts(self):
        return self.counts

    def set_weight_version(self, version):
        self.weight_version = version


class _Lifecycle:
    def __init__(self):
        self.pending_removes = {}
        self.pending_drains = set()
        self._removes_awaiting_step = {}

    def pending_teardown_count(self):
        return (
            len(self.pending_removes) + len(self.pending_drains)
            + sum(len(rids) for rids in self._removes_awaiting_step.values())
        )


class _Lockstep:
    def __init__(self):
        self._follow_fifo = deque()

    def pending_committed_steps(self):
        return len(self._follow_fifo)


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
        runtime = _Runtime()
        lifecycle = _Lifecycle()
        lockstep = _Lockstep()
        ops = WeightOps(
            SimpleNamespace(
                worker_id=worker_id, graph_runtime=runtime,
                engine_manager=SimpleNamespace(engine=engine),
                communicator=SimpleNamespace(send=lambda peer, msg: self.bus.append((peer, msg))),
                rebuilds=SimpleNamespace(copies_queued=False),
            ),
            lifecycle, lockstep,
        )
        return SimpleNamespace(runtime=runtime, lifecycle=lifecycle, lockstep=lockstep,
                               in_flight=set(), ops=ops)

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
                    self.workers[peer].ops.enqueue(message.body)
            for worker_id, worker in self.workers.items():
                if worker.ops.pending is not None and worker_id not in self.stalled:
                    worker.ops.process(worker.in_flight)
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
    follower.runtime.counts = (1, 0, 0)
    follower.lifecycle._removes_awaiting_step = {5: ["r"]}
    follower.lockstep._follow_fifo.append(object())

    d.control.pause(PauseGeneration(op_id="pause", mode="wait"))
    d.run()
    assert [a for a in d.answers if a.op_id == "pause"] == []
    assert follower.ops.pending is not None, "the follower answered while holding work"
    assert follower.ops.idle_blockers(follower.in_flight) == [
        "1 request(s) still held", "1 teardown(s) not yet applied",
        "1 committed TP step(s) not yet run",
    ]

    # the step ran, and the removal stamped after it was applied
    follower.lockstep._follow_fifo.clear()
    follower.lifecycle._removes_awaiting_step = {}
    follower.runtime.counts = (0, 0, 0)
    d.run()
    assert d.answer("pause").status == 200


@pytest.mark.parametrize("blocker, expected", [
    (lambda w: w.in_flight.add(3), "a step in flight"),
    (lambda w: setattr(w.runtime, "counts", (0, 1, 0)), "step output(s) not yet sent"),
    (lambda w: setattr(w.runtime, "counts", (0, 0, 2)), "staged speculative ingest"),
    (lambda w: w.lifecycle.pending_removes.setdefault(4), "teardown(s) not yet applied"),
])
def test_a_busy_rank_refuses_to_copy(ckpts, blocker, expected):
    lms, d = _tp2(ckpts)
    worker = d.workers["worker_0"]
    blocker(worker)
    worker.ops.enqueue(WeightControl(op=WeightOp.APPLY.value, epoch=1, model_path=ckpts.b_dir, weight_version="v1"))
    worker.ops.process(worker.in_flight)
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
    assert {w: d.workers[w].runtime.weight_version for w in WORKERS} == dict.fromkeys(WORKERS, "v1")
    for rank, lm in enumerate(lms):
        assert {n: p.data_ptr() for n, p in lm.named_parameters()} == pointers[rank]
        assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _shard(ckpts.b, rank))

    d.control.check_weights(CheckWeights(op_id="after-reload", mode="checksum"))
    d.run()
    checked = d.answer("after-reload")
    assert checked.status == 200 and checked.weight_version == "v1"
    assert set(checked.rank_checksums) == set(WORKERS)

    d.control.resume(ContinueGeneration(op_id="c"))
    d.run()
    assert d.answer("c").status == 200 and not d.control.paused


def test_checker_hashes_resident_rank_tensors_and_compares(ckpts):
    lms, d = _tp2(ckpts)
    assert d.pause().status == 200
    d.control.check_weights(CheckWeights(op_id="first", mode="checksum"))
    d.run()
    first = d.answer("first")
    assert first.status == 200 and first.weight_version is None
    assert set(first.rank_checksums) == set(WORKERS)
    assert all(len(value) == 64 for value in first.rank_checksums.values())
    assert first.rank_checksums["worker_0"] != first.rank_checksums["worker_1"]

    d.control.check_weights(CheckWeights(op_id="match", mode="compare", checksums=first.rank_checksums))
    d.run()
    assert d.answer("match").status == 200

    # A TP follower's loaded parameter, and a quantized/fused weight buffer,
    # must both affect its own rank's digest.
    with torch.no_grad():
        next(lms[1].named_parameters())[1].view(-1)[0].add_(1)
    d.control.check_weights(CheckWeights(op_id="different", mode="compare", checksums=first.rank_checksums))
    d.run()
    changed = d.answer("different")
    assert changed.status == 409 and changed.paused
    assert changed.rank_checksums["worker_0"] == first.rank_checksums["worker_0"]
    assert changed.rank_checksums["worker_1"] != first.rank_checksums["worker_1"]


def test_checksum_includes_nonpersistent_fused_weight_buffers():
    from mstar.engine.engine import Engine

    module = torch.nn.Module()
    module.register_buffer("qkv_proj_weight", torch.tensor([1.0, 2.0]), persistent=False)
    engine = Engine.__new__(Engine)
    engine._device = torch.device("cpu")
    engine._submodules = {"attention": SimpleNamespace(submodule=module)}
    before = engine.weight_checksum()
    module.qkv_proj_weight[0] = 3.0
    assert engine.weight_checksum() != before


def test_dirty_rank_refuses_weight_checksum(ckpts):
    _, d = _tp2(ckpts)
    assert d.pause().status == 200
    d.workers["worker_1"].ops.dirty = True
    d.control.check_weights(CheckWeights(op_id="dirty", mode="checksum"))
    d.run()
    answer = d.answer("dirty")
    assert answer.status == 500 and answer.paused
    assert "worker_1" not in answer.rank_checksums
    assert "partially copied" in answer.ranks["worker_1"]


def test_a_follower_that_fails_mid_copy_keeps_the_deployment_paused(ckpts):
    failing = _FailingAfterCopy.__new__(_FailingAfterCopy)
    failing.config, failing._llm_weights_dir = _orpheus(ckpts.a_dir).config, ckpts.a_dir
    failing._llm_reload_manifest = reload_manifest(ckpts.a_dir)
    lms, d = _tp2(ckpts, follower_model=failing)
    assert d.pause().status == 200

    d.control.update(UpdateWeightsFromDisk(op_id="u", model_path=ckpts.b_dir, weight_version="v1"))
    d.run()
    answer = d.answer("u")
    assert answer.status == 500 and answer.paused
    assert answer.ranks["worker_0"] == "ok"
    assert answer.ranks["worker_1"].startswith("OSError: read error partway")
    assert answer.rank_versions == {"worker_0": "v1", "worker_1": None}
    assert d.workers["worker_1"].ops.dirty
    # a failed copy changes no stamp: here, the first reload, the follower still
    # stamps nothing (after an earlier reload it would keep that one's version,
    # harmless since a dirty rank never serves); the rank that copied stamps
    # the new one, but stays paused
    assert d.workers["worker_1"].runtime.weight_version is None
    assert d.workers["worker_0"].runtime.weight_version == "v1"
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
    d.workers["worker_1"].ops._engine._model._llm_reload_manifest = reload_manifest(deeper)
    assert d.pause().status == 200

    d.control.update(UpdateWeightsFromDisk(op_id="u", model_path=ckpts.b_dir, weight_version="v1"))
    d.run()
    answer = d.answer("u")
    assert answer.status == 400 and answer.ranks["worker_0"] == "ok"
    assert "num_hidden_layers" in answer.ranks["worker_1"]
    for rank, lm in enumerate(lms):
        assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _shard(ckpts.a, rank))
    assert not any(w.ops.dirty for w in d.workers.values())

    d.control.resume(ContinueGeneration(op_id="c"))
    d.run()
    assert d.answer("c").status == 200, "a refused update must not block serving the old weights"
