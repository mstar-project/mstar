"""A TP follower's input read fails after rank 0 scheduled the row from its own
copy (the TP side of #418).

Rank 0 marks a step whose rows take an input the followers read from elsewhere
(``ScheduleTPNode.read_check``). A follower builds such a step once each row's
read landed or failed, leaving out the rows it could not read, and before any
rank prepares the step one all-reduce of a flag per row tells every rank which
rows to take out. Those rows never run, so nothing of them is computed,
committed or sent, and every rank runs the same collectives.

- The two-rank runs reuse the acceptance harness
  (``test_tp_read_failure_acceptance.py``) with an embedding that runs a
  collective per row inside the step, as Qwen3.5's vocab-parallel embedding
  does in its forward, and with a step whose every row failed.
- TP4 and TP2 x SP2: four real gloo ranks run the check itself, on the
  instance group ``GlobalParallelConfig`` ranks and ``init_dist`` creates;
  ``JointGroups.broadcast`` runs on the same four ranks.
- The follower's scheduling (serial and ``MSTAR_TP_ASYNC_SCHED``) and rank 0's
  marking run the real ``MicroScheduler``, ``Lockstep`` and ``Speculator`` on stubs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from array import array
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import test_tp_read_failure_acceptance as acc  # the two-rank harness
import torch
import torch.nn.functional as F
from test_failed_tensor_read import _failed_future, _read, _shm_manager
from test_micro_scheduler import NODE, WALK, _Engine, _Manager

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensors import StoredOutputs
from mstar.communication.wire import decode, encode
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import Rebuild, Residency
from mstar.engine.resources.step import FULL_ADMIT_REBUILD, StepContext
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs, SpeculationPrepOutput
from mstar.model.submodule_base import chunk_loop_name
from mstar.testing_utils.worker import make_scheduler, make_worker, tp_sharding
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode, WorkerMessageType
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.rebuild import Prefilling
from mstar.worker.speculation import Speculation
from mstar.worker.stream_buffers import StreamBufferManager

# ── two ranks through Worker.run (the acceptance harness) ─────────────────


def _vocab_parallel_submodule_class(harness_class):
    """The harness's Qwen3.5 LLM submodule, with an embedding that all-reduces
    over the TP group as ``VocabParallelEmbedding`` does. Qwen3.5 embeds a
    row's ids inside the step, the harness once per row, so a rank whose step
    holds a row another rank's does not runs a collective with no partner."""
    base = harness_class()

    class _Embedding(torch.nn.Module):
        def __init__(self, weight, group, rank):
            super().__init__()
            self.weight, self.group, self.rank = weight, group, rank

        def forward(self, ids):
            partial = F.embedding(ids, self.weight) * ((self.rank + 1) / 3.0)
            return self.group.all_reduce(partial)

    class _LLM(base):
        def __init__(self, group, rank):
            super().__init__(group, rank)
            self.model.model.embed_tokens = _Embedding(
                self.model.model.embed_tokens.weight, group, rank,
            )

    return _LLM


def _rank_main_vocab_parallel(rank, scenario, inbox, outbox):
    harness_class = acc._submodule_class
    acc._submodule_class = lambda: _vocab_parallel_submodule_class(harness_class)
    acc._rank_main(rank, scenario, inbox, outbox)


class _EveryLateReadFails(acc._Reads):
    """The follower's reads of both requests resolve late, and its read of
    each one's failing input fails: rank 0's step holds only rows this rank
    could not read."""

    def __init__(self, tm, kind, rank, scenario):
        from mstar.communication.tensors import FutureAndPointers
        from mstar.graph.base import GraphEdge as Edge
        self.tm, self.late, self.fired = tm, [], False
        uuid = 1000 * (rank + 1)
        for rid in ("r0", "r1"):
            for signal, tensors in acc._inputs(kind, rid).items():
                infos, buffers = [], []
                for t in tensors:
                    uuid += 1
                    info = acc._pointer(t, uuid)
                    buf = torch.zeros_like(t)
                    tm.tensor_store.put_tensor(rid, uuid, buf, info)
                    infos.append(info)
                    buffers.append((buf, t))
                future = Future()
                tm.pending.append(FutureAndPointers(
                    future=future, graph_edges=[Edge(next_node=acc.NODE, name=signal, tensor_info=infos)],
                    request_id=rid,
                ))
                if rank == 1:
                    self.late.append((future, buffers, signal == acc._FAILING[kind]))
                else:
                    self._land(future, buffers)


def _rank_main_every_row_fails(rank, scenario, inbox, outbox):
    acc._Reads = _EveryLateReadFails
    acc._rank_main(rank, scenario, inbox, outbox)


@pytest.mark.parametrize("kind", acc.KINDS)
def test_a_step_that_runs_a_collective_per_row_stays_in_step(tmp_path, monkeypatch, kind):
    """With the embedding all-reducing once per row inside the step, r1's
    failed follower read still fails only r1, and r0's outputs equal a run
    with no failure: the check takes r1 out of every rank's step before any
    rank runs it, so no rank runs r1's embedding collectives."""
    monkeypatch.setattr(acc, "_rank_main", _rank_main_vocab_parallel)
    clean = acc._run(tmp_path / "clean", acc.Scenario(kind=kind, fail=False))
    acc._healthy_ok(clean)
    assert clean.completed("r1") and not clean.failed, clean.summary()
    run = acc._run(tmp_path / "failed", acc.Scenario(kind=kind))
    acc._healthy_ok(run)
    assert set(run.failed) == {"r1"} and run.torn_down("r1"), run.summary()
    for rank in (0, 1):
        assert "r1" not in run.routed[rank], run.summary()
        assert run.routed[rank]["r0"] == clean.routed[rank]["r0"], f"rank {rank}"


def test_a_step_whose_every_row_failed_runs_empty_on_every_rank(tmp_path, monkeypatch):
    """Rank 0 sends a step of rows the follower could read none of: both ranks
    run it empty, both requests fail and drain, and nothing is routed."""
    monkeypatch.setattr(acc, "_rank_main", _rank_main_every_row_fails)
    run = acc._run(tmp_path, acc.Scenario(kind="text"))
    assert not run.crashed, run.summary() + "\n" + "\n".join(map(str, run.crashed.values()))
    assert not run.hung, run.summary()
    assert set(run.failed) == {"r0", "r1"}, run.summary()
    assert run.torn_down("r0") and run.torn_down("r1"), run.summary()
    assert run.routed == {0: {}, 1: {}}, run.summary()
    assert not run.collective_failures, run.summary()   # no rank sat in one the other never joined


# ── the check itself over a four-rank instance (four gloo ranks) ──────────

ROWS = (10, 11, 12, 13)


def _four_rank_groups(rank, sizes, port):
    """This rank's groups of a four-rank instance, as ``GlobalParallelConfig``
    lays them out and the real ``init_dist`` creates them, over gloo."""
    import torch.distributed as dist

    from mstar.distributed.communication import GlobalParallelConfig
    from mstar.model.base import WorkerGraph

    wg = WorkerGraph(
        section=SimpleNamespace(get_nodes=lambda: {NODE: None}), graph_walks=set(),
        ranks=[0, 1, 2, 3], **sizes,
    )
    workers = [f"worker_{r}" for r in range(4)]
    # covers the rendezvous too: on a loaded box the ranks reach it seconds apart
    groups = GlobalParallelConfig({NODE: wg}, workers, dist_timeout_s=acc.RENDEZVOUS_S).per_worker_config[workers[rank]]
    init = dist.init_process_group
    # torch binds a world to an accelerator only: it refuses a CPU device_id
    dist.init_process_group = lambda device_id, **kwargs: init(**kwargs)
    groups.init_dist(init_method=f"tcp://127.0.0.1:{port}", device=torch.device("cpu"))
    return groups


def _check_rank(rank, sizes, port, failed_here, out):
    import torch.distributed as dist
    try:
        w = make_worker(
            streams=StreamBufferManager(SimpleNamespace(per_request_info={}), None, []),
            parallel_groups=_four_rank_groups(rank, sizes, port),
        )
        w.lockstep.read_failed_rids.update(failed_here)
        # a follower that could not read a row left it out; the others hold it
        rows = [r for r in ROWS if r not in failed_here]
        batch = ExecutingBatch(
            node_name=NODE, per_request_info=dict.fromkeys(rows, SimpleNamespace()),
            step_context=StepContext(request_ids=tuple(rows), graph_walk=WALK, slot=0, capture=False),
        )
        batch.read_check_rids = ROWS
        w.lockstep.drop_rows_whose_read_failed(batch)
        out.put((rank, list(batch.request_ids), sorted(batch.failed_requests)))
        dist.destroy_process_group()
    except BaseException:
        out.put((rank, "crash", traceback.format_exc()))


def _broadcast_rank(rank, sizes, port, out):
    import torch.distributed as dist
    try:
        joint = _four_rank_groups(rank, sizes, port).get_joint_group_for_node(NODE)
        out.put((rank, joint.broadcast(torch.full((3,), 10.0 + rank)).tolist()))
        dist.destroy_process_group()
    except BaseException:
        out.put((rank, "crash", traceback.format_exc()))


def _run_four_ranks(target, args) -> dict:
    """Run ``target(rank, *args(rank, port), out)`` on four spawned ranks, each
    of which puts ``(rank, *result)`` on ``out``; return the results by rank."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=target, args=(r, *args(r, port), out)) for r in range(4)]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, *rest = out.get(timeout=acc.RENDEZVOUS_S + acc.DEADLINE_S)
            results[rank] = rest
    except queue.Empty:
        pass
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    assert sorted(results) == [0, 1, 2, 3], results
    for rest in results.values():
        assert rest[0] != "crash", rest[1]
    return results


@pytest.mark.parametrize("sizes", [{"tp_size": 4}, {"tp_size": 2, "sp_size": 2}, {"cfg_size": 2, "tp_size": 2}],
                         ids=["tp4", "tp2sp2", "cfg2tp2"])
def test_every_rank_takes_out_the_rows_any_rank_could_not_read(sizes):
    """Rank 2's read of row 11 and rank 3's of row 13 failed. Every rank, rank
    0 and rank 1 included, takes out both rows and fails both, and keeps the
    other two in rank 0's order. Under CFG2 x TP2 the vote has to cross the CFG
    axis, which a vote over TP then SP misses: it runs on the instance's own
    gloo group."""
    failed = {2: {11}, 3: {13}}
    results = _run_four_ranks(_check_rank, lambda r, port: (sizes, port, failed.get(r, set())))
    for rank, rest in results.items():
        assert rest == [[10, 12], [11, 13]], (rank, rest)


@pytest.mark.parametrize("sizes", [{"tp_size": 2, "sp_size": 2}, {"cfg_size": 2, "sp_size": 2}],
                         ids=["tp2sp2", "cfg2sp2"])
def test_a_joint_broadcast_reaches_every_rank_from_instance_rank_0(sizes):
    """Each rank starts with its own tensor, and ends with instance rank 0's:
    one broadcast per axis carries it to ranks that share no axis group with
    rank 0."""
    results = _run_four_ranks(_broadcast_rank, lambda r, port: (sizes, port))
    assert results == {rank: [[10.0] * 3] for rank in range(4)}


# ── a follower builds rank 0's step without the rows it could not read ────


class _DroppedHere(_Engine):
    """`_Engine` on a rank whose KV dropped ``dropped``, as its replay of rank 0's drop left it. Records the
    rebuilds the rank starts."""

    def __init__(self, dropped):
        super().__init__()
        self.dropped, self.started = set(dropped), []

    def residency(self, node_name, rid):
        return Residency.DROPPED if rid in self.dropped else Residency.RESIDENT

    def in_step(self, node_name, rid):
        return False

    def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
        if rid in self.dropped:
            return FULL_ADMIT_REBUILD
        return super().check_ready(node_name, rid, fwd_info, allow_reload)

    def prepare_rebuild(self, node_name, rid, rebuild):
        self.started.append(rid)


def _log(kept):
    """A rebuild log keeping ids for ``kept``, which starts each rebuild once, and from its prompt's chunks when
    the request was mid-prefill in its first prefill step."""
    cursors = {}

    def start(rid, node, prefilling):
        assert (node, rid) not in cursors, f"{rid}'s rebuild started twice"
        cursors[(node, rid)] = prefilling
        return None if prefilling is not None else Rebuild(array("i", [1, 2]))

    return SimpleNamespace(
        kept=lambda rid, node: rid if rid in kept else None, start=start,
        rebuilding=lambda rid, node: (node, rid) in cursors, cursors=cursors.items, nodes=frozenset({NODE}),
        cursor=lambda rid, node: SimpleNamespace(resumes_prefill=cursors[(node, rid)] is not None),
    )


def _follower_scheduler(ready, read_failed=(), dropped=(), kept=None, chunk_loops=None):
    manager = _Manager(["r0", "r1"])
    manager.queues["wg0"]._ready = {rid: {NODE} for rid in ready}
    engine = _DroppedHere(dropped)
    sched = make_scheduler(
        engine_manager=SimpleNamespace(get_engine=lambda n: engine),
        parallel_leader_nodes=set(), parallel_nodes={NODE}, request_state=manager, graph_runtime=manager.runtime,
        rebuilds=_log(dropped if kept is None else kept), chunk_loops=chunk_loops,
    )
    sched.lockstep.read_failed_rids.update(read_failed)
    return sched, manager


def _head(read_check, rids=("r0", "r1"), **kwargs):
    return ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=list(rids), spec_seq=0,
        read_check=read_check, **kwargs,
    )


def test_a_follower_builds_a_read_checked_step_with_the_row_it_could_not_read_empty():
    """r1's read failed here, so r1 never readies. The follower takes the head
    without waiting: it pops r0 alone and puts r1 in with no inputs, so the step
    has rank 0's rows, in its order, until the check takes r1 out. The head no
    longer holds r1's drain."""
    sched, manager = _follower_scheduler(ready=["r0"], read_failed=["r1"])
    sched.lockstep.enqueue_follow(_head(read_check=True, rids=("r1", "r0")))

    batch = sched.next_batch()

    assert list(batch.request_to_worker_graph) == ["r1", "r0"]
    assert batch.read_check_rids == ("r1", "r0")
    assert manager.pops[-1][1] == ["r0"]
    assert list(batch.input_edges.rids) == ["r0"]
    assert sched.lockstep.follow_head() is None
    assert "r1" not in sched.lockstep._follow_count


def test_a_follower_takes_a_step_whose_every_row_it_could_not_read():
    """The step still has rank 0's rows, so this rank submits it and joins its
    check, as rank 0 does."""
    sched, manager = _follower_scheduler(ready=[], read_failed=["r0", "r1"])
    sched.lockstep.enqueue_follow(_head(read_check=True))

    batch = sched.next_batch()

    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert batch.read_check_rids == ("r0", "r1")
    assert sched.lockstep.follow_head() is None


def test_a_follower_waits_on_a_row_whose_read_is_still_running():
    sched, manager = _follower_scheduler(ready=["r0"])
    sched.lockstep.enqueue_follow(_head(read_check=True))

    assert sched.next_batch() is None
    assert sched.lockstep.follow_head() is not None

    manager.ingest("r1")
    batch = sched.next_batch()
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]


def test_without_a_read_check_a_follower_pops_the_row_like_any_other(caplog):
    """No rank would take the row out, so the follower does not either: a row
    whose failed read was another step's input runs; one whose input never
    lands waits, as before the check, and says why once."""
    sched, manager = _follower_scheduler(ready=["r0", "r1"], read_failed=["r1"])
    sched.lockstep.enqueue_follow(_head(read_check=False))
    batch = sched.next_batch()
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert batch.read_check_rids is None

    sched, manager = _follower_scheduler(ready=["r0"], read_failed=["r1"])
    sched.lockstep.enqueue_follow(_head(read_check=False))
    with caplog.at_level("ERROR"):
        assert sched.next_batch() is None
        assert sched.next_batch() is None
    assert sum("no read check" in r.getMessage() for r in caplog.records) == 1


def test_a_read_that_fails_after_its_request_failed_still_goes_to_the_check(tmp_path):
    """This rank already failed r1 for another reason, and r1's read then fails
    before the conductor's removal lands. The read is not reported again, but it
    is recorded for the check: rank 0's read-checked step naming r1 is built
    with r1 empty, so this rank joins the step's check and the check takes r1
    out, rather than this rank waiting on r1 with no warning while rank 0 waits
    in the check."""
    sched, manager = _follower_scheduler(ready=["r0"])
    sched.lifecycle.add("r1", "r1")
    sched.lifecycle.fail({"r1": "failed for another reason"})
    sched.lifecycle.take_failures()
    tm, _ = _shm_manager(tmp_path)
    w = make_worker(
        streams=StreamBufferManager(SimpleNamespace(per_request_info={}), tm, []),
        scheduler=sched,
        lockstep=sched.lockstep,
        admission=sched.admission,
        request_state=manager,
        tensor_manager=tm,
        wakeup_event=SimpleNamespace(drain=lambda: None),
    )
    tm.pending.append(_read("r1", _input(17, "text_inputs"), _failed_future()))
    w.ingest.check_ready_tensors()
    assert w.lifecycle.take_failures() == {}, "r1 was already reported failed"
    assert sched.lockstep.read_failed_rids == {"r1"}

    sched.lockstep.enqueue_follow(_head(read_check=True))
    batch = sched.next_batch()

    assert batch is not None, "the follower waits on r1, which never readies"
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert batch.read_check_rids == ("r0", "r1")
    assert manager.pops[-1][1] == ["r0"]


def test_a_read_that_raises_as_it_starts_goes_to_the_check_and_fails_once():
    """r1's read raises as this rank starts it (a CUDA OOM in the H2D copy,
    say), and again for an input replayed behind it. r1 is reported failed once
    and recorded for the check, so rank 0's read-checked step naming r1 is built
    with r1 empty rather than this rank waiting on r1 for good."""
    sched, manager = _follower_scheduler(ready=["r0"])
    sched.lifecycle.add("r1", "r1")
    manager.per_request_info["r1"] = SimpleNamespace(stream_buffers={"audio": None})
    reads = []
    w = make_worker(
        worker_id="w1", scheduler=sched, lockstep=sched.lockstep, admission=sched.admission,
        request_state=manager,
        graph_runtime=SimpleNamespace(get_rid_handle=lambda r: r, get_rid_string=lambda h: h),
        tensor_manager=SimpleNamespace(start_read_tensors=lambda *a, **kw: reads.append(a[0]) or (_ for _ in ()).throw(
            RuntimeError("CUDA out of memory"),
        )),
        wakeup_event=SimpleNamespace(register_futures=lambda futures: None),
    )
    edge = _input(18, "audio")
    edge.is_streaming = True
    reported = []
    for _ in range(2):
        w.ingest._process_new_inputs(SimpleNamespace(
            request_id="r1", inputs=[edge], request_info=None, producer_done=set(),
        ))
        reported.append(list(w.lifecycle.take_failures()))
    assert reads == ["r1", "r1"]  # the read itself raised, both times
    assert reported == [["r1"], []]
    assert sched.lockstep.read_failed_rids == {"r1"}

    sched.lockstep.enqueue_follow(_head(read_check=True))
    batch = sched.next_batch()

    assert batch is not None, "the follower waits on r1, which never readies"
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]


def test_a_removed_requests_read_failure_is_forgotten():
    """The flag is keyed by the handle, which a later request can get."""
    sched, _ = _follower_scheduler(ready=[], read_failed=["r1"])
    sched.lockstep.clear("r1", "r1")
    assert "r1" not in sched.lockstep.read_failed_rids


def _spec_follower(read_failed, continuing, fresh, head_rids, dropped=()):
    """An async follower with step N (``continuing``) in flight and rank 0's
    speculative head for N+1 queued. The graph runtime's prep and the
    speculation's assembly are recorded, not run."""
    # a row whose read failed never readies
    sched, manager = _follower_scheduler(
        ready=[r for r in fresh if r not in read_failed], read_failed=read_failed, dropped=dropped,
    )

    def prep(prep_input):
        w.preps.append(list(prep_input.rids))
        return SpeculationPrepOutput([], list(prep_input.rids), ["wg0"] * len(prep_input.rids),
                                     ColumnarEdgeSpecs.empty())

    tensor_manager = SimpleNamespace(
        get_tensor=lambda uuid: torch.zeros(1),
        get_tensors=lambda uuids: [torch.zeros(1) for _ in uuids],
    )
    w = make_worker(
        scheduler=sched, request_state=manager, tensor_manager=tensor_manager,
        graph_runtime=SimpleNamespace(
            get_spec_target=lambda *a: SimpleNamespace(),
            prep_follow_spec_rids=prep,
            push_back_node=lambda node, rids, wg_ids: (
                w.pushed_back.extend(rids), manager.runtime.push_back_node(node, rids, wg_ids),
            ),
            commit_speculation=lambda *a, **k: None,
            get_worker_graph_id_for_node=manager.runtime.get_worker_graph_id_for_node,
        ),
        engine_manager=SimpleNamespace(get_engine=lambda n: SimpleNamespace(
            release_cached_prefix=lambda batch: w.released.extend(batch.request_ids),
        )),
        streams=StreamBufferManager(SimpleNamespace(per_request_info={}), tensor_manager, []),  # no streams
    )
    w.preps, w.assembled, w.pushed_back, w.released, w.continuing, w.inputs = [], [], [], [], [], []

    def assemble(pending, info, rows, inputs, streamed, continuing, fresh_final, rebuild_rids=frozenset(), **kwargs):
        w.assembled.append(list(rows))
        w.continuing.append(list(continuing))
        w.inputs.append(dict(inputs))
        rids = list(rows)
        return Speculation(
            scheduled_batch=ScheduledBatch(
                node_name=NODE, graph_walk=WALK, request_to_worker_graph=dict(rows), rebuild_rids=rebuild_rids,
            ),
            node_batch=ExecutingBatch(
                node_name=NODE, per_request_info=dict.fromkeys(rids, SimpleNamespace()),
                step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
            ),
            consumed_edges=set(), continuing_rids=set(), partition="default",
            is_new_iter=False, is_same_node=True,
        )

    w.speculator._spec_builder = SimpleNamespace(assemble=assemble)
    pending = SimpleNamespace(
        tp_seq=0, node_name=NODE, graph_walk=WALK,
        batch=ScheduledBatch(node_name=NODE, graph_walk=WALK,
                             request_to_worker_graph=dict.fromkeys(continuing, "wg0")),
        node_batch=SimpleNamespace(commit_done=SimpleNamespace(is_set=lambda: True)),
    )
    w.lockstep.enqueue_follow(ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=list(head_rids), spec_seq=1,
        speculative=True, spec_from_seq=0, read_check=True,
    ))
    return w, pending


@pytest.mark.parametrize("unread", ["fresh", "continuing", "every"])
def test_an_async_follower_builds_a_read_checked_head_with_the_row_empty(unread):
    """``MSTAR_TP_ASYNC_SCHED``: the follower builds a speculative head's
    continuing rows (prepped from N) and fresh rows (popped) apart. A row it
    could not read is neither: it goes in with no inputs, so the speculation
    has rank 0's rows (a continuing one still takes N's loop-back, and its
    loop count), and is submitted, voided and settled as rank 0's is."""
    head = ["c0", "c1", "f0", "f1"]
    failed = {"fresh": ["f1"], "continuing": ["c1"], "every": head}[unread]
    w, pending = _spec_follower(
        read_failed=failed, continuing=["c0", "c1"], fresh=["f0", "f1"], head_rids=head,
    )

    spec = w.speculator.follow(pending)

    assert spec is not None, "the head waits on a row this rank will never read"
    assert w.preps == [[r for r in ("c0", "c1") if r not in failed]]
    assert w.assembled == [head]
    assert w.continuing == [["c0", "c1"]]
    assert all(w.inputs[-1][r] == {} for r in failed)
    assert spec.node_batch.read_check_rids == tuple(head)
    assert w.lockstep.follow_head() is None
    # a read-checked step is admitted and planned inline on every rank, after the check
    assert not w.speculator.plans_ahead(spec)

    # N's verdict voids the head: rank 0 gives back the leases of every row it
    # sent, and so does this rank, the unread rows' included
    w.speculator.settle(spec, success=False)
    assert sorted(w.released) == sorted(head)


def test_an_async_follower_keeps_a_rebuild_row_of_rank_0s_head_out_of_the_pop():
    """Rank 0's speculative head takes rb, a rebuild its log offered. The graph runtime holds no row for rb, so the
    follower knows rb from its own KV, starts its rebuild, and puts it in with no inputs for its log to fill at the
    build. Popping it would wait for a row that never readies."""
    w, pending = _spec_follower(
        read_failed=(), continuing=["c0"], fresh=["f0"], head_rids=["c0", "rb", "f0"], dropped=["rb"],
    )

    spec = w.speculator.follow(pending)

    assert spec is not None, "the head waits on a rebuild row as if the graph runtime held it"
    assert w.assembled == [["c0", "rb", "f0"]] and w.inputs[-1]["rb"] == {}
    assert spec.scheduled_batch.rebuild_rids == {"rb"}, "the follower would route the rebuild row's outputs"
    assert w.scheduler.engine_manager.get_engine(NODE).started == ["rb"]


def test_an_async_follower_takes_a_rebuilds_next_chunk_after_n_as_a_rebuild_row():
    """rb's rebuild ran its first chunk in N. Rank 0's speculative head takes rb's next chunk as one of N's rows
    mid-prefill, and rank 0 marks it as a rebuild row from its log."""
    w, pending = _spec_follower(
        read_failed=(), continuing=["c0", "rb"], fresh=[], head_rids=["c0", "rb"], dropped=["rb"],
    )
    pending.batch.incomplete_node_rids = {"rb"}
    w.scheduler.rebuilds.start("rb", NODE, None)

    spec = w.speculator.follow(pending)

    assert spec is not None
    assert spec.scheduled_batch.rebuild_rids == {"rb"}, "rb's rebuild would not land with rank 0's"
    assert w.preps == [["c0"]] and spec.continuing_rids == {"rb"}


def test_a_follower_checking_a_dropped_requests_row_starts_no_rebuild():
    """A follower's own rebuild of rb has landed and rank 0's land of it hasn't replayed yet, so rb is still dropped
    here. A readiness check of rb's row in that gap finds it not ready and starts nothing, or rb's rebuild would run
    a second time on this rank alone."""
    sched, _ = _follower_scheduler(ready=[], dropped=["rb"])

    assert not sched.admission.check_ready(NODE, "rb", SimpleNamespace(graph_walk=WALK))
    assert sched.engine_manager.get_engine(NODE).started == [] and not sched.rebuilds.rebuilding("rb", NODE)


def test_a_follower_builds_rank_0s_rebuild_row_from_its_own_log():
    """rb is a rebuild row of rank 0's step. The follower starts rb's rebuild from its own kept ids once, pops r0
    alone, and builds the step with rank 0's rows in rank 0's order."""
    sched, manager = _follower_scheduler(ready=["r0"], dropped=["rb"])
    sched.lockstep.enqueue_follow(_head(read_check=False, rids=("rb", "r0")))

    batch = sched.next_batch()

    assert list(batch.request_to_worker_graph) == ["rb", "r0"], "the follower's step isn't rank 0's"
    assert batch.rebuild_rids == {"rb"} and sched.engine_manager.get_engine(NODE).started == ["rb"]


def test_a_follower_rebuilds_a_request_dropped_mid_prefill_from_its_prompts_chunks():
    """rb was dropped after 2 of its prompt's 4 chunks and keeps no ids. The follower knows rb's row from its own
    KV and starts the rebuild from rb's chunk Loop, as rank 0 did, since the Loop runs in lockstep."""
    sched, manager = _follower_scheduler(ready=["r0"], dropped=["rb"], kept=(), chunk_loops=ChunkLoops([(NODE, WALK)]))
    manager.loops[("rb", chunk_loop_name(NODE, WALK))] = [2, 4]
    sched.lockstep.enqueue_follow(_head(read_check=False, rids=("rb", "r0")))

    batch = sched.next_batch()

    assert list(batch.request_to_worker_graph) == ["rb", "r0"] and batch.rebuild_rids == {"rb"}, (
        "the follower took rb's row as one of its own"
    )
    assert dict(sched.rebuilds.cursors()) == {(NODE, "rb"): Prefilling(WALK, 2, 4)}, "rb's rebuild has another source"


def test_rank_0_and_a_follower_vote_on_the_same_rows_when_a_step_has_a_rebuild_row(tmp_path):
    """Before a step whose rows read inputs from elsewhere, every TP rank votes one flag per row on whose read
    failed, in one all-reduce. rb is a rebuild row of the step, and r1's prompt landed on rank 0 by a read. A
    follower that voted on fewer rows than rank 0 would give the all-reduce a different number of values."""
    w = _leader(tmp_path)
    _land(w, "r1", _input(11, "text_inputs"))
    batch = _step({"rb": ["text_inputs"], "r0": ["text_inputs"], "r1": ["text_inputs"]})
    batch.rebuild_rids = frozenset({"rb"})
    w.lockstep.broadcast(batch)
    (message,) = [m.body for _, m in w.sent if m.message_type == WorkerMessageType.SCHEDULE_TP]
    assert message.read_check and message.request_ids == ["rb", "r0", "r1"]

    sched, manager = _follower_scheduler(ready=["r0", "r1"], dropped=["rb"])
    sched.lockstep.enqueue_follow(message)
    followed = sched.next_batch()

    assert followed.read_check_rids == batch.read_check_rids, (
        f"rank 0 votes on {batch.read_check_rids} and the follower on {followed.read_check_rids}"
    )


def test_an_async_follower_that_cannot_prep_gives_back_only_the_rows_it_popped():
    """The continuing rows' prep fails (a chunk not landed yet): the follower
    hands back the fresh row it popped, not the row it could not read, which it
    never popped."""
    w, pending = _spec_follower(
        read_failed=["f1"], continuing=["c0"], fresh=["f0", "f1"], head_rids=["c0", "f0", "f1"],
    )
    w._graph_runtime.prep_follow_spec_rids = lambda prep_input: None

    assert w.speculator.follow(pending) is None
    assert w.pushed_back == ["f0"]
    assert w.lockstep.follow_head() is not None


def test_a_voided_speculations_unread_row_stays_out_of_the_next_read_checked_head():
    """f1's read failed here and f1 is not in N, so it is no continuing row of
    the speculation: voiding it (``StepPipeline._clear_speculation``) pushes f1 back with the
    fresh rows, though it was never popped, and the graph then holds f1 as
    ready with no input. Rank 0 sends the step again, read-checked, since it
    never ran: the follower still pops only the rows it read. Only the read
    check keeps f1 out: a head without one would pop it."""
    w, pending = _spec_follower(
        read_failed=["f1"], continuing=["c0"], fresh=["f0", "f1"], head_rids=["c0", "f0", "f1"],
    )
    spec = w.speculator.follow(pending)
    spec.continuing_rids = set(w.continuing[-1])  # as ``SpeculationBuilder.assemble`` sets it
    manager = w.request_state

    # what ``_clear_speculation`` gives back
    sb = spec.scheduled_batch
    fresh = [rid for rid in sb.request_to_worker_graph if rid not in spec.continuing_rids]
    assert fresh == ["f0", "f1"]
    w._graph_runtime.push_back_node(NODE, fresh, [sb.request_to_worker_graph[r] for r in fresh])
    assert NODE in manager.queues["wg0"]._ready["f1"]

    manager.ingest("c0")  # N's routing readies its continuing row
    w.lockstep.enqueue_follow(_head(read_check=True, rids=("c0", "f0", "f1")))
    batch = w.scheduler.next_batch()
    assert list(batch.request_to_worker_graph) == ["c0", "f0", "f1"]
    assert manager.pops[-1][1] == ["c0", "f0"]

    w.lockstep.enqueue_follow(_head(read_check=False, rids=("f1",)))
    assert w.scheduler.next_batch() is not None
    assert manager.pops[-1][1] == ["f1"]


# ── rank 0 marks the steps whose rows take an input a follower reads ──────

FOLLOWERS = ["w0", "w1"]


def _leader(tmp_path, streamed=(), nodes=(NODE,), consumed=None, leads=True):
    """Rank 0 leading ``nodes``, or with ``leads`` off a rank in them that leads
    nothing; ``consumed[(source, dest)]`` names the signals an edge carries from
    ``source`` to ``dest``."""
    consumed = consumed or {}
    tm, _ = _shm_manager(tmp_path)
    sent = []
    sharding = tp_sharding(*FOLLOWERS, tp_rank=0, nodes={NODE})
    engine_manager = SimpleNamespace(get_engine=lambda n: SimpleNamespace(
        take_resident_delta=OffloadDelta.new,
        decide_cached_prefix=lambda b: {}, page_trails=lambda n: {},
    ))
    w = make_worker(
        streams=StreamBufferManager(SimpleNamespace(per_request_info={}), tm, []),
        parallel_nodes=set(nodes),
        parallel_leader_nodes=set(nodes) if leads else set(),
        streaming_edge_names=set(streamed),
        communicator=SimpleNamespace(
            send=lambda dest, msg=None: sent.append((dest, decode(encode(msg)))),
        ),
        graph_runtime=SimpleNamespace(
            get_sharding_config=lambda r: sharding, get_rid_string=lambda h: h,
            get_rid_handle=lambda r: r, ingest_inputs_batch=lambda block, can_buffer: None,
            get_consumed_edges=lambda source, dest, walk: {(sig, dest) for sig in consumed.get((source, dest), ())},
        ),
        engine_manager=engine_manager,
        tensor_manager=tm,
        request_state=SimpleNamespace(per_request_info={"r0": None, "r1": None}),
        wakeup_event=SimpleNamespace(drain=lambda: None),
    )
    w.lockstep._followers = dict.fromkeys(nodes, FOLLOWERS[1:]) if leads else {}
    w.sent = sent
    return w


def _input(uuid, name, source_entity="api_server", source_node=None, node=NODE):
    t = torch.ones(3, dtype=torch.int64)
    info = TensorPointerInfo(
        list(t.shape), t.dtype, t.nbytes, 0, list(t.stride()), uuid, "s", source_entity,
        _source_node_name=source_node,
    )
    return GraphEdge(next_node=node, name=name, tensor_info=[info])


def _land(w, rid, edge):
    """``edge`` lands on ``w`` through its real read path."""
    done = Future()
    done.set_result(None)
    for info in edge.tensor_info:
        w.tensor_manager.tensor_store.put_tensor(rid, info.uuid, torch.zeros(3, dtype=torch.int64), info)
    w.tensor_manager.pending.append(_read(rid, edge, done))
    w.ingest.check_ready_tensors()


def _step(rows, node=NODE):
    """A step of ``node`` whose rows take ``rows[rid]``'s input names."""
    rids = list(rows)
    return ExecutingBatch(
        node_name=node, per_request_info=dict.fromkeys(rids, SimpleNamespace()),
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
        per_request_input_tensors={
            rid: {name: [torch.zeros(1)] for name in names} for rid, names in rows.items()
        },
    )


def _broadcast(w, batch):
    w.lockstep.broadcast(batch)
    (msg,) = [m for d, m in w.sent if m.message_type == WorkerMessageType.SCHEDULE_TP]
    w.sent.clear()
    return msg.body.read_check, batch.read_check_rids


def test_a_step_taking_an_input_read_from_elsewhere_is_checked_until_it_runs(tmp_path):
    """r1's prompt landed on rank 0 by a read; the step that takes it goes out
    checked, with its rows in order. A step that never ran (a voided
    speculation, a refused admit) leaves the row to a later one, which is
    checked too. Once one ran, the next step's input is the node's own output,
    which no rank reads."""
    w = _leader(tmp_path)
    _land(w, "r1", _input(11, "text_inputs"))
    rows = {"r0": ["text_inputs"], "r1": ["text_inputs"]}

    assert _broadcast(w, _step(rows)) == (True, ("r0", "r1"))
    ran = _step(rows)
    assert _broadcast(w, ran) == (True, ("r0", "r1"))
    w.lockstep.consume_reads(ran)
    assert _broadcast(w, _step(rows)) == (False, None)


def test_the_nodes_own_output_coming_back_is_no_read(tmp_path):
    """A persisted token returning as the decode walk's input is every rank's
    own copy; an input for another node, or with no tensor, has no read either."""
    w = _leader(tmp_path)
    me = w.tensor_manager.my_entity_id
    _land(w, "r1", _input(12, "text_inputs", source_entity=me, source_node=NODE))
    _land(w, "r1", _input(13, "text_inputs", node="vision_encoder"))
    _land(w, "r1", GraphEdge(next_node=NODE, name="text_inputs"))

    assert _broadcast(w, _step({"r1": ["text_inputs"]})) == (False, None)


def test_a_colocated_encoders_output_is_a_followers_read(tmp_path):
    """An encoder beside rank 0 hands its output to rank 0's node without a
    read; each follower reads it. Another led node it is not routed to has
    nothing to spend, so once the step taking it ran nothing is left pending."""
    w = _leader(tmp_path, nodes=(NODE, "Talker"), consumed={("vision_encoder", NODE): {"vision_embeds"}})
    w.lockstep.note_local_inputs(
        StoredOutputs(flat_uuids=[21], flat_rids=["r1"], signal_idxs=[0], num_tensors=[1]),
        ["vision_embeds"], "vision_encoder", ["r1"], [WALK], None,
    )

    ran = _step({"r0": [], "r1": ["vision_embeds"]})
    assert _broadcast(w, ran) == (True, ("r0", "r1"))
    w.lockstep.consume_reads(ran)
    assert not w.lockstep._unread


def test_one_lockstep_node_taking_an_input_leaves_another_nodes_copy_checked(tmp_path):
    """One producer beside rank 0 feeds the same signal to two lockstep nodes:
    the first node's step spends only its own read, so the second's goes out
    checked too."""
    other = "LLM_2"
    w = _leader(tmp_path, nodes=(NODE, other), consumed={
        ("producer", NODE): {"latents"}, ("producer", other): {"latents"},
    })
    w.lockstep.note_local_inputs(
        StoredOutputs(flat_uuids=[22], flat_rids=["r1"], signal_idxs=[0], num_tensors=[1]),
        ["latents"], "producer", ["r1"], [WALK], None,
    )
    ran = _step({"r1": ["latents"]})
    assert _broadcast(w, ran) == (True, ("r1",))
    w.lockstep.consume_reads(ran)

    assert _broadcast(w, _step({"r1": ["latents"]}, node=other)) == (True, ("r1",))
    assert _broadcast(w, _step({"r1": ["latents"]})) == (False, None)


def test_a_streamed_input_is_checked_on_every_step(tmp_path):
    w = _leader(tmp_path, streamed={"thinker_states"})
    for _ in range(2):
        assert _broadcast(w, _step({"r0": ["thinker_states"]})) == (True, ("r0",))
    assert _broadcast(w, _step({"r0": ["text_inputs"]})) == (False, None)


def test_rank_0_records_its_own_failed_reads_for_the_check(tmp_path):
    w = _leader(tmp_path)
    w.tensor_manager.pending.append(_read("r1", _input(15, "text_inputs"), _failed_future()))
    w.ingest.check_ready_tensors()
    assert w.lockstep.read_failed_rids == {"r1"}


def test_single_rank_workers_mark_nothing(tmp_path):
    w = _leader(tmp_path, leads=False)
    _land(w, "r1", _input(16, "text_inputs"))
    assert not w.lockstep._unread


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(pytest.main([__file__, "-q"]))
