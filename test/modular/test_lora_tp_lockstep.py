"""TP lockstep for the LoRA pool: every rank makes the same slot decisions.

The leader's broadcast orders the admits, the worker lands each teardown between the same two
admits on every rank (``after_tp_seq`` plus the admit fence), and a load is agreed across the
instance before it is installed. Ranks run as threads, the collective as a barrier.
"""

from __future__ import annotations

import os
import queue
import random
import sys
import threading
import time
from datetime import timedelta

import pytest
import torch

pytest.importorskip("safetensors")

try:
    from .test_lora_pool import ctx, ingest, make_spec, step, write_adapter
except ImportError:  # collected without package context
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_lora_pool import ctx, ingest, make_spec, step, write_adapter

from mstar.engine.resources.lora.pool import LoraAdapterPool  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
)

ADAPTERS = ("a1", "a2", "a3", "a4")


def _write_adapters(root, names):
    for i, name in enumerate(names):
        write_adapter(root / f"{name}.safetensors", rank=2 + i % 3, seed=i + 1)
    return {name: str(root / f"{name}.safetensors") for name in names}


class _Barrier:
    """``agree`` for ranks in threads: a MIN over a barrier; a missing peer raises."""

    def __init__(self, n: int, timeout: float = 5.0):
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._flags = [True] * n
        self.calls = [0] * n

    def for_rank(self, rank: int):
        def agree(flag: bool) -> bool:
            self.calls[rank] += 1
            self._flags[rank] = flag
            self._barrier.wait()
            result = all(self._flags)
            self._barrier.wait()
            return result
        return agree


def _verdict(outcome):
    if outcome.ok:
        return "ok"
    if isinstance(outcome.reason, AdmitRuntimeError):
        return ("load", tuple(sorted(outcome.reason.request_ids)))
    return ("full", outcome.reason.request_id)


def _settle_reads(pool: LoraAdapterPool, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while pool._reads:
        assert time.monotonic() < deadline, f"reads never completed: {list(pool._reads)}"
        time.sleep(0.001)
    with pool._lock:  # a completion stores its copy under the lock
        pass


class _Rank:
    def __init__(self, paths, rank: int, agree, seed: int):
        self.pool = LoraAdapterPool(make_spec(paths, max_slots=2, cpu_capacity=1), "cpu", rank=rank,
                                    world_size=2, agree=agree, screens=rank == 0)
        self.rng = random.Random(seed * 2 + rank)
        self.log: list = []  # (seq, rows, verdict, slot table) per admit
        self.pending: list = []

    def admit(self, seq: int, rows: list[str]):
        self.land_commit()
        s, c = step(rows), ctx(rows)
        outcome = self.pool.admit(s, c)
        if outcome.ok:
            self.pending.append((s, c))
        else:
            self.pool.rollback_admit(s, c)
        slots = map(self.pool.slot_state, range(self.pool.spec.num_slots))
        self.log.append((seq, tuple(rows), _verdict(outcome),
                         tuple((x.adapter, x.refcount, x.last_used_step) for x in slots)))
        return outcome

    def land_commit(self, maybe: bool = False):
        """Commit runs on the GPU thread, anywhere before the next admit."""
        if self.pending and not (maybe and self.rng.random() < 0.5):
            self.pool.commit(*self.pending.pop())


def _simulate(paths, seed: int, screened: bool, fenced: bool = True, steps: int = 60, timeout: float = 5.0):
    """A leader sending ingests, steps, stamped removes and an unload to a follower in FIFO
    order. ``screened`` picks rows as the scheduler does; ``fenced=False`` lands some removes
    on the leader before the admit of the step they are stamped behind."""
    barrier = _Barrier(2, timeout)
    leader, follower = (_Rank(paths, r, barrier.for_rank(r), seed) for r in (0, 1))
    wire: queue.Queue = queue.Queue()
    errors: list = []
    rng = random.Random(seed)

    def run_leader():
        active: dict[str, int] = {}
        waiting: list[str] = []
        late: list[str] = []
        names, n = (*ADAPTERS, None), 0
        try:
            for seq in range(steps):
                for _ in range(rng.choice((0, 0, 1, 2))):
                    rid, adapter = f"r{n}", rng.choice(names)
                    n += 1
                    ingest(leader.pool, rid, adapter)
                    wire.put(("ingest", rid, adapter))
                    waiting.append(rid)
                fresh = waiting[:3]
                if screened:
                    _settle_reads(leader.pool)  # a step outlasts a small adapter's read
                    fresh = []
                    for rid in waiting:
                        if len(fresh) == 3:
                            break  # a step's worth; the screen claims slots for what it passes
                        if leader.pool.admit_retrieve(rid, "LLM", "decode", None).ready:
                            fresh.append(rid)
                rows = list(active) + fresh
                if not rows:
                    continue
                wire.put(("step", seq, rows))
                if seq >= steps // 3 and "a4" in names:
                    # the leader unloads before this admit, the follower after it
                    names = (*ADAPTERS[:3], None)
                    leader.pool.unload_adapter("a4")
                    wire.put(("unload", "a4"))
                for rid in late:
                    leader.pool.remove_request(rid)
                    wire.put(("remove", rid))
                late = []
                outcome = leader.admit(seq, rows)
                done: list[str] = []
                if outcome.ok:
                    for rid in rows:
                        if rid in waiting:
                            waiting.remove(rid)
                            active[rid] = rng.randint(1, 6)
                        active[rid] -= 1
                        if active[rid] == 0:
                            done.append(rid)
                elif (blamed := outcome.reason.request_id) in waiting:
                    waiting.remove(blamed)  # held back; retried last
                    waiting.append(blamed)
                leader.land_commit(maybe=True)
                for rid in done:
                    active.pop(rid)
                    if not fenced and rng.random() < 0.5:
                        late.append(rid)
                        continue
                    leader.pool.remove_request(rid)
                    wire.put(("remove", rid))
                leader.land_commit(maybe=True)
            leader.land_commit()
        except Exception as exc:  # noqa: BLE001
            errors.append(("leader", exc))
        finally:
            wire.put(None)

    def run_follower():
        try:
            while (msg := wire.get()) is not None:
                kind, *body = msg
                if kind == "ingest":
                    ingest(follower.pool, *body)
                elif kind == "step":
                    follower.admit(*body)
                elif kind == "unload":
                    follower.pool.unload_adapter(*body)
                else:
                    follower.pool.remove_request(*body)
                follower.land_commit(maybe=True)
            follower.land_commit()
        except Exception as exc:  # noqa: BLE001
            errors.append(("follower", exc))

    threads = [threading.Thread(target=run_leader), threading.Thread(target=run_follower)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a rank is stuck"
    return leader, follower, barrier, errors


@pytest.mark.parametrize("screened", [False, True])
def test_ranks_agree_step_by_step_under_adapter_churn(tmp_path, screened):
    """Four adapters through two slots, one unloaded, each rank landing commits and the
    unload at its own moment: every admit has the same verdict and slot table on both ranks.
    Screened, no step is refused for want of a slot."""
    paths = _write_adapters(tmp_path, ADAPTERS)
    full = evict = 0
    for seed in range(12):
        leader, follower, barrier, errors = _simulate(paths, seed, screened)
        assert not errors, (seed, errors)
        assert leader.log == follower.log, seed
        assert barrier.calls[0] == barrier.calls[1]
        tables = [entry[3] for entry in leader.log]
        evict += sum(a[0] is not None and a[0] != b[0]
                     for t0, t1 in zip(tables, tables[1:], strict=False) for a, b in zip(t0, t1, strict=True))
        refused = sum(entry[2] != "ok" for entry in leader.log)
        full += refused
        if screened:
            assert refused == 0, seed
            assert not follower.pool._claims and not follower.pool._waiting_since
    assert evict > 0 and (screened or full > 0), (evict, full)  # not vacuous


def test_the_churn_run_diverges_without_the_fence(tmp_path):
    paths = _write_adapters(tmp_path, ADAPTERS)
    for seed in range(12):
        leader, follower, _, errors = _simulate(paths, seed, screened=False, fenced=False, timeout=0.5)
        if errors or leader.log != follower.log:
            return
    pytest.fail("no seed diverged with the leader's teardowns unfenced")


# --- the worker's admit fence --------------------------------------------------------------

class _PicksSlotPerMiss(LoraAdapterPool):
    """Mechanism off: each miss picks its slot after the previous miss's agreement."""

    def admit(self, step, ctx):
        if ctx.capture:
            return ADMIT_OK
        self._admits += 1
        misses: dict[str, list] = {}
        with self._lock:
            for seg in step.segments or ():
                if seg.request_id in self._awaiting and seg.span > 0:
                    name = self._adapter_of[seg.request_id]
                    slot = self._resident.get(name)
                    if slot is None:
                        misses.setdefault(name, []).append(seg.request_id)
                    else:
                        self._take(seg.request_id, slot)
        for name, rids in misses.items():
            with self._lock:
                slot = self._pick_slot()
            if slot is None:
                return AdmitOutcome(ok=False, reason=AllocationFailed(
                    message="full", pages_short=1, label="lora", request_id=rids[0]))
            outcome = self._load_miss(name, rids, slot)
            if not outcome.ok:
                return outcome
        return ADMIT_OK


def _slots(pool):
    return tuple((s.adapter, s.refcount, s.last_used_step)
                 for s in (pool.slot_state(i) for i in range(pool.spec.num_slots)))


def _on_both(*fns):
    results, errors = [None, None], []

    def run(i, fn):
        try:
            results[i] = fn()
        except Exception as exc:  # noqa: BLE001
            errors.append((i, exc))

    threads = [threading.Thread(target=run, args=(i, fn)) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads), "a rank is stuck"
    return results, errors


@pytest.mark.parametrize("pool_cls,splits", [(LoraAdapterPool, False), (_PicksSlotPerMiss, True)])
def test_a_teardown_inside_a_two_miss_admit_does_not_split_the_ranks(tmp_path, pool_cls, splits):
    """A follower's slow read holds the leader's admit in the load agreement while a
    teardown lands on the leader alone. Every miss's slot is picked when the admit starts,
    so the second miss cannot see the freed slot and both ranks refuse the step alike."""
    paths = _write_adapters(tmp_path, ADAPTERS)
    barrier = _Barrier(2, timeout=2.0)
    mid_admit: list = []  # a teardown the leader's main thread lands once, mid-admit

    def leader_agree(flag):
        agreed = barrier.for_rank(0)(flag)
        while mid_admit:
            remove = threading.Thread(target=mid_admit.pop())
            remove.start()
            remove.join()
        return agreed

    leader, follower = ranks = [
        pool_cls(make_spec(paths, max_slots=2, cpu_capacity=1), "cpu", rank=r, world_size=2,
                 agree=agree, screens=r == 0)
        for r, agree in ((0, leader_agree), (1, barrier.for_rank(1)))
    ]

    def admit(pool, rows):
        s, c = step(rows), ctx(rows)
        outcome = pool.admit(s, c)
        (pool.commit if outcome.ok else pool.rollback_admit)(s, c)
        return _verdict(outcome)

    for pool in ranks:  # a1's slot unheld, a4's held by "k"
        ingest(pool, "h", "a1")
        ingest(pool, "k", "a4")
        _settle_reads(pool)
    (v1, v2), errors = _on_both(lambda: admit(leader, ["h", "k"]), lambda: admit(follower, ["h", "k"]))
    assert not errors and v1 == v2 == "ok"
    for pool in ranks:
        pool.remove_request("h")
    gate, read = threading.Event(), follower.store._read
    follower.store._read = lambda name, path: (name != "a2" or gate.wait(10)) and read(name, path)
    for pool in ranks:
        ingest(pool, "rA", "a2")
        ingest(pool, "rB", "a3")
    _settle_reads(leader)
    mid_admit.append(lambda: leader.remove_request("k"))
    threading.Timer(0.2, gate.set).start()

    def follower_step():
        verdict = admit(follower, ["rA", "rB"])
        follower.remove_request("k")  # FIFO: after its admit
        return verdict

    (v1, v2), errors = _on_both(lambda: admit(leader, ["rA", "rB"]), follower_step)
    while mid_admit:  # no agreement ran: the leader lands it once the admit settles
        mid_admit.pop()()
    if splits:
        assert errors or v1 != v2 or _slots(leader) != _slots(follower)
        return
    assert not errors and v1 == v2 == ("full", "rB") and _slots(leader) == _slots(follower)
    _settle_reads(follower)  # the retry, k gone on both, loads both adapters on both
    (v1, v2), errors = _on_both(lambda: admit(leader, ["rA", "rB"]), lambda: admit(follower, ["rA", "rB"]))
    assert not errors and v1 == v2 == "ok" and _slots(leader) == _slots(follower)
    assert {leader.slot_state(i).adapter for i in (1, 2)} == {"a2", "a3"}


# --- the agreement over real gloo ----------------------------------------------------------

def _gloo_worker(rank: int, store: str, out_dir: str, lora_paths: dict):
    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.engine.resources.base import EngineResourceInfo

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    tp = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    tp.device_group = tp.cpu_group = dist.group.WORLD
    tp.initialized = True
    tp.init_all_reduce()
    joint = JointGroups(tp_group=tp, sp_group=CommGroup.trivial())
    pool = LoraAdapterPool.build(make_spec(lora_paths, max_slots=2, cpu_capacity=1),
                                 EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=joint))
    if rank == 1:
        read = pool.store._read

        def flaky(name, path):
            if name == "a2":
                raise OSError("read error on rank 1 only")
            return read(name, path)
        pool.store._read = flaky
    for rid, adapter in (("ok", "a1"), ("bad", "a2")):
        ingest(pool, rid, adapter)
    result = {"agree": [_verdict(pool.admit(step([rid]), ctx([rid]))) for rid in ("ok", "bad")],
              "slots": [pool.slot_state(i).adapter for i in range(3)]}
    torch.save(result, os.path.join(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def test_gloo_a_load_failing_on_one_rank_is_refused_on_both(tmp_path):
    import torch.multiprocessing as mp

    lora_paths = _write_adapters(tmp_path, ("a1", "a2"))
    mp.spawn(_gloo_worker, args=(str(tmp_path / "store"), str(tmp_path), lora_paths), nprocs=2, join=True)
    for r in range(2):
        result = torch.load(tmp_path / f"rank{r}.pt")
        assert result["agree"] == ["ok", ("load", ("bad",))]
        assert "a2" not in result["slots"]  # rank 0's good read was not installed

