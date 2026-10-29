"""A TP follower's input read fails after rank 0 scheduled the row from its own
copy (the TP side of #418).

Rank 0 marks a step whose rows take an input the followers read from elsewhere
(``ScheduleTPNode.read_check``). A follower builds such a step once each row's
read landed or failed, leaving out the rows it could not read, and before any
rank prepares the step one all-reduce of a flag per row tells every rank which
rows to take out. Those rows never run, so nothing of them is computed,
committed or sent, and every rank runs the same collectives.

- The two-rank runs reuse the acceptance harness
  (``test_tp_read_failure_acceptance.py``) with a ``prepare_inputs`` that runs a
  collective per text part, as Qwen3.5's vocab-parallel embedding does, and
  with a step whose every row failed.
- TP4 and TP2 x SP2: four real gloo ranks run the check itself, on the
  instance group ``GlobalParallelConfig`` ranks and ``init_dist`` creates;
  ``JointGroups.broadcast`` runs on the same four ranks.
- The follower's scheduling (serial and ``MSTAR_TP_ASYNC_SCHED``) and rank 0's
  marking run the real ``MicroScheduler`` and ``Worker`` methods on stubs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
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
from mstar.engine.resources.step import StepContext
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs, SpeculationPrepOutput
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode, WorkerMessageType
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import Speculation, Worker

# ── two ranks through Worker.run (the acceptance harness) ─────────────────


def _vocab_parallel_submodule_class(harness_class):
    """The harness's Qwen3.5 LLM submodule, with an embedding that all-reduces
    over the TP group as ``VocabParallelEmbedding`` does. Qwen3.5's
    ``prefill_vision`` embeds each text part (and its vision sentinels) inside
    ``prepare_inputs``, so a rank that prepares a row another rank does not
    runs a collective with no partner."""
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
def test_a_prepare_that_runs_a_collective_per_row_stays_in_step(tmp_path, monkeypatch, kind):
    """With the embedding all-reducing inside ``prepare_inputs``, r1's failed
    follower read still fails only r1, and r0's outputs equal a run with no
    failure: no rank prepares r1, so no rank runs r1's embedding collectives."""
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
    clean = acc._run(tmp_path / "clean", acc.Scenario(kind="text", fail=False))
    assert not clean.crashed and not clean.hung, clean.summary()
    monkeypatch.setattr(acc, "_rank_main", _rank_main_every_row_fails)
    run = acc._run(tmp_path / "failed", acc.Scenario(kind="text"))
    assert not run.crashed, run.summary() + "\n" + "\n".join(map(str, run.crashed.values()))
    assert not run.hung, run.summary()
    assert set(run.failed) == {"r0", "r1"}, run.summary()
    assert run.torn_down("r0") and run.torn_down("r1"), run.summary()
    assert run.routed == {0: {}, 1: {}}, run.summary()
    # No rank sat in a collective the other never joined, which costs a
    # COLLECTIVE_S wait. Measured against the run with no failure, not as a
    # fixed bound: starting the two ranks alone takes seconds on a slow host.
    assert run.elapsed < clean.elapsed + acc.COLLECTIVE_S, (run.summary(), clean.summary())


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
    groups = GlobalParallelConfig({NODE: wg}, workers, dist_timeout_s=10).per_worker_config[workers[rank]]
    init = dist.init_process_group
    # torch binds a world to an accelerator only: it refuses a CPU device_id
    dist.init_process_group = lambda device_id, **kwargs: init(**kwargs)
    groups.init_dist(init_method=f"tcp://127.0.0.1:{port}", device=torch.device("cpu"))
    return groups


def _check_rank(rank, sizes, port, failed_here, out):
    import torch.distributed as dist
    try:
        w = Worker.__new__(Worker)
        w.parallel_groups = _four_rank_groups(rank, sizes, port)
        w._walk_drivers = {}
        w._walk_driven_edges = {}
        w.scheduler = SimpleNamespace(read_failed_rids=set(failed_here))
        # a follower that could not read a row left it out; the others hold it
        rows = [r for r in ROWS if r not in failed_here]
        batch = ExecutingBatch(
            node_name=NODE, per_request_info=dict.fromkeys(rows, SimpleNamespace()),
            step_context=StepContext(request_ids=tuple(rows), graph_walk=WALK, slot=0, capture=False),
        )
        batch.read_check_rids = ROWS
        w._drop_rows_whose_read_failed(batch)
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
            rank, *rest = out.get(timeout=60)
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


def _follower_scheduler(ready, read_failed=()):
    manager = _Manager(["r0", "r1"])
    manager.queues["wg0"]._ready = {rid: {NODE} for rid in ready}
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda n: _Engine()),
        parallel_leader_nodes=set(),
    )
    sched.runtime = manager.runtime
    sched.read_failed_rids.update(read_failed)
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
    sched.register_tp_follow(_head(read_check=True, rids=("r1", "r0")))

    batch = sched.get_next_batch(manager)

    assert list(batch.request_to_worker_graph) == ["r1", "r0"]
    assert batch.read_check_rids == ("r1", "r0")
    assert manager.pops[-1][1] == ["r0"]
    assert list(batch.input_edges.rids) == ["r0"]
    assert sched.peek_tp_follow() is None
    assert "r1" not in sched.pending_tp_follow_count


def test_a_follower_takes_a_step_whose_every_row_it_could_not_read():
    """The step still has rank 0's rows, so this rank submits it and joins its
    check, as rank 0 does."""
    sched, manager = _follower_scheduler(ready=[], read_failed=["r0", "r1"])
    sched.register_tp_follow(_head(read_check=True))

    batch = sched.get_next_batch(manager)

    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert batch.read_check_rids == ("r0", "r1")
    assert sched.peek_tp_follow() is None


def test_a_follower_waits_on_a_row_whose_read_is_still_running():
    sched, manager = _follower_scheduler(ready=["r0"])
    sched.register_tp_follow(_head(read_check=True))

    assert sched.get_next_batch(manager) is None
    assert sched.peek_tp_follow() is not None

    manager.ingest("r1")
    batch = sched.get_next_batch(manager)
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]


def test_without_a_read_check_a_follower_pops_the_row_like_any_other(caplog):
    """No rank would take the row out, so the follower does not either: a row
    whose failed read was another step's input runs; one whose input never
    lands waits, as before the check, and says why once."""
    sched, manager = _follower_scheduler(ready=["r0", "r1"], read_failed=["r1"])
    sched.register_tp_follow(_head(read_check=False))
    batch = sched.get_next_batch(manager)
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert batch.read_check_rids is None

    sched, manager = _follower_scheduler(ready=["r0"], read_failed=["r1"])
    sched.register_tp_follow(_head(read_check=False))
    with caplog.at_level("ERROR"):
        assert sched.get_next_batch(manager) is None
        assert sched.get_next_batch(manager) is None
    assert sum("no read check" in r.getMessage() for r in caplog.records) == 1


def test_a_read_that_fails_after_its_request_failed_still_goes_to_the_check(tmp_path):
    """This rank already failed r1 for another reason, and r1's read then fails
    before the conductor's removal lands. The read is not reported again, but it
    is recorded for the check: rank 0's read-checked step naming r1 is built
    with r1 empty, so this rank joins the step's check and the check takes r1
    out, rather than this rank waiting on r1 with no warning while rank 0 waits
    in the check."""
    sched, manager = _follower_scheduler(ready=["r0"])
    sched.fail_rids({"r1"})
    tm, _ = _shm_manager(tmp_path)
    w = Worker.__new__(Worker)
    w._walk_drivers = {}
    w._walk_driven_edges = {}
    w.scheduler, w.request_state, w.tensor_manager = sched, manager, tm
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    w.enable_nvtx = False
    reported = []
    w._fail_requests = reported.append
    tm.pending.append(_read("r1", _input(17, "text_inputs"), _failed_future()))
    w._check_ready_tensors()
    assert reported == [], "r1 was already reported failed"
    assert sched.read_failed_rids == {"r1"}

    sched.register_tp_follow(_head(read_check=True))
    batch = sched.get_next_batch(manager)

    assert batch is not None, "the follower waits on r1, which never readies"
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert batch.read_check_rids == ("r0", "r1")
    assert manager.pops[-1][1] == ["r0"]


def test_a_removed_requests_read_failure_is_forgotten():
    """The flag is keyed by the handle, which a later request can get."""
    sched, _ = _follower_scheduler(ready=[], read_failed=["r1"])
    sched.clear_rid("r1", "r1")
    assert "r1" not in sched.read_failed_rids


def _spec_follower(read_failed, continuing, fresh, head_rids):
    """An async follower with step N (``continuing``) in flight and rank 0's
    speculative head for N+1 queued. The graph runtime's prep and the
    speculation's assembly are recorded, not run."""
    # a row whose read failed never readies
    sched, manager = _follower_scheduler(
        ready=[r for r in fresh if r not in read_failed], read_failed=read_failed,
    )
    w = Worker.__new__(Worker)
    w._walk_drivers = {}
    w._walk_driven_edges = {}
    w.scheduler, w.request_state, w.enable_nvtx = sched, manager, False
    w._in_flight_rids = set()
    w.preps, w.assembled = [], []

    def prep(prep_input):
        w.preps.append(list(prep_input.rids))
        return SpeculationPrepOutput([], list(prep_input.rids), ["wg0"] * len(prep_input.rids),
                                     ColumnarEdgeSpecs.empty())

    w._graph_runtime = SimpleNamespace(
        get_spec_target=lambda *a: SimpleNamespace(),
        prep_follow_spec_rids=prep,
        push_back_node=lambda node, rids, wg_ids: (
            w.pushed_back.extend(rids), manager.runtime.push_back_node(node, rids, wg_ids),
        ),
        commit_speculation=lambda *a, **k: None,
        get_worker_graph_id_for_node=manager.runtime.get_worker_graph_id_for_node,
    )
    w.pushed_back = []
    w.released = []
    w.engine_manager = SimpleNamespace(get_engine=lambda n: SimpleNamespace(
        release_cached_prefix=lambda batch: w.released.extend(batch.request_ids),
    ))
    w.tensor_manager = SimpleNamespace(get_tensor=lambda uuid: torch.zeros(1))

    w.continuing, w.inputs = [], []

    def assemble(pending, info, rows, inputs, streamed, continuing, fresh_final, **kwargs):
        w.assembled.append(list(rows))
        w.continuing.append(list(continuing))
        w.inputs.append(dict(inputs))
        rids = list(rows)
        return Speculation(
            scheduled_batch=ScheduledBatch(node_name=NODE, graph_walk=WALK, request_to_worker_graph=dict(rows)),
            node_batch=ExecutingBatch(
                node_name=NODE, per_request_info=dict.fromkeys(rids, SimpleNamespace()),
                step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
            ),
            consumed_edges=set(), continuing_rids=set(), partition="default",
            is_new_iter=False, is_same_node=True,
        )

    w._assemble_speculation = assemble
    pending = SimpleNamespace(
        tp_seq=0, node_name=NODE, graph_walk=WALK,
        batch=ScheduledBatch(node_name=NODE, graph_walk=WALK,
                             request_to_worker_graph=dict.fromkeys(continuing, "wg0")),
        node_batch=SimpleNamespace(commit_done=SimpleNamespace(is_set=lambda: True)),
    )
    sched.register_tp_follow(ScheduleTPNode(
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

    spec = Worker._try_follow_speculation(w, pending)

    assert spec is not None, "the head waits on a row this rank will never read"
    assert w.preps == [[r for r in ("c0", "c1") if r not in failed]]
    assert w.assembled == [head]
    assert w.continuing == [["c0", "c1"]]
    assert all(w.inputs[-1][r] == {} for r in failed)
    assert spec.node_batch.read_check_rids == tuple(head)
    assert w.scheduler.peek_tp_follow() is None
    # a read-checked step is admitted and planned inline on every rank, after the check
    assert not Worker._plans_ahead(w, spec)

    # N's verdict voids the head: rank 0 gives back the leases of every row it
    # sent, and so does this rank, the unread rows' included
    Worker._settle_speculation(w, spec, success=False)
    assert sorted(w.released) == sorted(head)


def test_an_async_follower_that_cannot_prep_gives_back_only_the_rows_it_popped():
    """The continuing rows' prep fails (a chunk not landed yet): the follower
    hands back the fresh row it popped, not the row it could not read, which it
    never popped."""
    w, pending = _spec_follower(
        read_failed=["f1"], continuing=["c0"], fresh=["f0", "f1"], head_rids=["c0", "f0", "f1"],
    )
    w._graph_runtime.prep_follow_spec_rids = lambda prep_input: None

    assert Worker._try_follow_speculation(w, pending) is None
    assert w.pushed_back == ["f0"]
    assert w.scheduler.peek_tp_follow() is not None


def test_a_voided_speculations_unread_row_stays_out_of_the_next_read_checked_head():
    """f1's read failed here and f1 is not in N, so it is no continuing row of
    the speculation: voiding it (``_maybe_clear_spec``) pushes f1 back with the
    fresh rows, though it was never popped, and the graph then holds f1 as
    ready with no input. Rank 0 sends the step again, read-checked, since it
    never ran: the follower still pops only the rows it read. Only the read
    check keeps f1 out: a head without one would pop it."""
    w, pending = _spec_follower(
        read_failed=["f1"], continuing=["c0"], fresh=["f0", "f1"], head_rids=["c0", "f0", "f1"],
    )
    spec = Worker._try_follow_speculation(w, pending)
    spec.continuing_rids = set(w.continuing[-1])  # as ``_assemble_speculation`` sets it
    manager = w.request_state

    # what ``_maybe_clear_spec`` gives back
    sb = spec.scheduled_batch
    fresh = [rid for rid in sb.request_to_worker_graph if rid not in spec.continuing_rids]
    assert fresh == ["f0", "f1"]
    w._graph_runtime.push_back_node(NODE, fresh, [sb.request_to_worker_graph[r] for r in fresh])
    assert NODE in manager.queues["wg0"]._ready["f1"]

    manager.ingest("c0")  # N's routing readies its continuing row
    w.scheduler.register_tp_follow(_head(read_check=True, rids=("c0", "f0", "f1")))
    batch = w.scheduler.get_next_batch(manager)
    assert list(batch.request_to_worker_graph) == ["c0", "f0", "f1"]
    assert manager.pops[-1][1] == ["c0", "f0"]

    w.scheduler.register_tp_follow(_head(read_check=False, rids=("f1",)))
    assert w.scheduler.get_next_batch(manager) is not None
    assert manager.pops[-1][1] == ["f1"]


# ── rank 0 marks the steps whose rows take an input a follower reads ──────

FOLLOWERS = ["w0", "w1"]


def _leader(tmp_path, streamed=()):
    tm, _ = _shm_manager(tmp_path)
    w = Worker.__new__(Worker)
    w._walk_drivers = {}
    w._walk_driven_edges = {}
    w.worker_id = "w0"
    w.parallel_nodes, w.parallel_leader_nodes = {NODE}, {NODE}
    w.is_tp_follower, w.enable_nvtx = False, False
    w._tp_broadcast_seq = 0
    w.sent = []
    w.communicator = SimpleNamespace(
        send=lambda dest, msg=None: w.sent.append((dest, decode(encode(msg)))),
    )
    group = SimpleNamespace(nodes={NODE}, tp_size=2, _tp_rank=0, _workers=FOLLOWERS)
    sharding = SimpleNamespace(groups=[group], get_sharding_group=lambda n, g: group)
    w._graph_runtime = SimpleNamespace(
        get_sharding_config=lambda r: sharding, get_rid_string=lambda h: h,
        get_rid_handle=lambda r: r, ingest_inputs_batch=lambda block, can_buffer: None,
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda n: SimpleNamespace(
        take_resident_delta=lambda n: OffloadDelta.new(),
        decide_cached_prefix=lambda b: {}, page_trails=lambda n: {},
    ))
    w.tensor_manager = tm
    w.scheduler = MicroScheduler(
        engine_manager=w.engine_manager, parallel_leader_nodes={NODE},
    )
    w.request_state = SimpleNamespace(per_request_info={"r0": None, "r1": None})
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    w._streaming_edge_names = set(streamed)
    w._setup_tp_read_check()
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
    w._check_ready_tensors()


def _step(rows):
    """A step whose rows take ``rows[rid]``'s input names."""
    rids = list(rows)
    return ExecutingBatch(
        node_name=NODE, per_request_info=dict.fromkeys(rids, SimpleNamespace()),
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
        per_request_input_tensors={
            rid: {name: [torch.zeros(1)] for name in names} for rid, names in rows.items()
        },
    )


def _broadcast(w, batch):
    w.maybe_send_zmq_to_tp_followers(batch)
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
    w._consume_lockstep_reads(ran)
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
    read; each follower reads it."""
    w = _leader(tmp_path)
    w._note_local_lockstep_inputs(
        StoredOutputs(flat_uuids=[21], flat_rids=["r1"], signal_idxs=[0], num_tensors=[1]),
        ["vision_embeds"],
    )

    assert _broadcast(w, _step({"r0": [], "r1": ["vision_embeds"]})) == (True, ("r0", "r1"))


def test_a_streamed_input_is_checked_on_every_step(tmp_path):
    w = _leader(tmp_path, streamed={"thinker_states"})
    for _ in range(2):
        assert _broadcast(w, _step({"r0": ["thinker_states"]})) == (True, ("r0",))
    assert _broadcast(w, _step({"r0": ["text_inputs"]})) == (False, None)


def test_rank_0_records_its_own_failed_reads_for_the_check(tmp_path):
    w = _leader(tmp_path)
    w._fail_requests = lambda errors: None
    w.tensor_manager.pending.append(_read("r1", _input(15, "text_inputs"), _failed_future()))
    w._check_ready_tensors()
    assert w.scheduler.read_failed_rids == {"r1"}


def test_single_rank_workers_mark_nothing(tmp_path):
    w = _leader(tmp_path)
    w.parallel_leader_nodes = set()
    w._setup_tp_read_check()
    _land(w, "r1", _input(16, "text_inputs"))
    assert not w._lockstep_unread


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(pytest.main([__file__, "-q"]))
