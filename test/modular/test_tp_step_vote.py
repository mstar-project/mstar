"""The ranks of a lockstep instance agree on each step's verdict under TP async
(``Worker._agree_on_step``), over real gloo ranks in their own processes.

Each rank finishes step N with its own ``admit_error`` and ``failed_requests``,
and drops or runs the head built on N by them. A row that failed after the
forward on one rank fails on every rank, so the head is dropped everywhere or
nowhere. A step that was admitted on some ranks only, or ran different rows,
has already mismatched the ranks' collectives, and every rank stops with the
same reason instead of one waiting forever in the next collective.

Handles differ per rank on purpose (each rank mints its own), so the vote can
only line rows up by what crosses the wire. All cases of a world run in one
set of processes, one vote after another, as the steps of a run do: every rank
takes the same branch at every vote, stops included, so the votes keep pairing.
"""
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from test_tp_async_teardown_order import HANDLES, NODE, _rank, _Stop

WALK = "decode"
ROWS = ["a", "b", "c"]


def _step(seq=7, rows=ROWS, failed=(), refused=False, raised=False):
    """One rank's view of step N once its GPU thread is done with it: the rows
    its forward ran (wire ids), the rows it failed before or after the forward,
    and whether the step raised instead."""
    return {"seq": seq, "rows": list(rows), "failed": list(failed), "refused": refused, "raised": raised}


TWO_RANKS = {
    "clean": {"*": _step()},
    # rank 1's postprocess raised for "b"
    "post_one": {0: _step(), 1: _step(failed=["b"])},
    "post_each": {0: _step(failed=["a"]), 1: _step(failed=["c"])},
    # the injected one-rank admit failure
    "refused_one": {0: _step(), 1: _step(refused=True)},
    "refused_all": {"*": _step(refused=True)},
    # rank 1's prepare_inputs raised for "b": its forward ran two rows to rank 0's three
    "pre_one": {0: _step(), 1: _step(rows=["a", "c"], failed=["b"])},
    # as many rows on each rank, but not the same ones
    "pre_swapped": {0: _step(rows=["a", "c"], failed=["b"]), 1: _step(rows=["a", "b"], failed=["c"])},
    # rank 0 could not resolve "d"'s input and sent the step without it
    "unsent": {0: _step(failed=["d"]), 1: _step()},
    "seq": {0: _step(seq=7), 1: _step(seq=8)},
    # rank 1's step raised (on the GPU thread, or in the loop after it)
    "raised_one": {0: _step(), 1: _step(raised=True)},
    "raised_all": {"*": _step(raised=True)},
    "after": {"*": _step(seq=9, rows=["b", "c"])},
}
GRID = {f"rank{r}": {r: _step(failed=["c"]), "*": _step()} for r in (1, 2, 3)}


def _rank_main(rank, world, tp, store, cases, out_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=world,
        timeout=timedelta(seconds=60),
    )
    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.worker.worker import PendingBatch, Worker

    # every rank creates every subgroup, in the same order
    sp = world // tp
    tp_groups = [list(range(r * tp, (r + 1) * tp)) for r in range(sp)]
    sp_groups = [list(range(c, world, tp)) for c in range(tp)]
    made = {tuple(m): dist.new_group(m, backend="gloo") for m in tp_groups + sp_groups if len(m) > 1}

    def _group(members):
        if len(members) == 1:
            return CommGroup.trivial()
        group = CommGroup(rank, members.index(rank), members)
        group.cpu_group = made[tuple(members)]
        return group

    joint = JointGroups(
        tp_group=_group(next(m for m in tp_groups if rank in m)),
        sp_group=_group(next(m for m in sp_groups if rank in m)),
    )
    # a handle per wire id, minted differently on every rank
    wire = {100 * rank + i: rid for i, rid in enumerate("abcdefgh")}
    handle = {rid: h for h, rid in wire.items()}

    w = Worker.__new__(Worker)
    w.worker_id = f"rank{rank}"
    w.enable_nvtx, w._phase_period, w._phase_buf = False, 0, {}
    w.parallel_groups = SimpleNamespace(get_joint_group_for_node=lambda node: joint)
    w._graph_runtime = SimpleNamespace(get_rid_string=wire.get)

    results = {}
    for name, per_rank in cases.items():
        mine = per_rank.get(rank, per_rank.get("*"))
        node_batch = SimpleNamespace(
            request_ids=[handle[r] for r in mine["rows"]],
            admit_error="kv cache: out of pages" if mine["refused"] else None,
            failed_requests={handle[r]: "boom" for r in mine["failed"]},
        )
        pending = PendingBatch(
            batch=None, node_batch=node_batch, node_name=NODE, partition="p",
            graph_walk=WALK, future=None, tp_seq=mine["seq"],
        )
        stopped = None
        try:
            Worker._agree_on_step(w, pending, raised=mine["raised"])
        except SystemExit as stop:
            stopped = str(stop)
        results[name] = {
            "stopped": stopped,
            "failed": sorted(wire[h] for h in node_batch.failed_requests),
            "void": Worker._step_voids_head(pending),
            "voted": pending.voted,
        }
    Path(out_dir, f"{rank}.json").write_text(json.dumps(results))
    dist.destroy_process_group()


def _spawn(out_dir: Path, cases, world, tp):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        _rank_main, args=(world, tp, str(out_dir / "store"), cases, str(out_dir)),
        nprocs=world, join=True,
    )
    return [json.loads((out_dir / f"{rank}.json").read_text()) for rank in range(world)]


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    return _spawn(tmp_path_factory.mktemp("tp2"), TWO_RANKS, world=2, tp=2)


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    return _spawn(tmp_path_factory.mktemp("tp2xsp2"), GRID, world=4, tp=2)


def _same(ranks, case):
    """Every rank's outcome of ``case``, which the vote makes one."""
    outcomes = [got[case] for got in ranks]
    assert all(o == outcomes[0] for o in outcomes), outcomes
    return outcomes[0]


def _stops(ranks, case):
    """The reason every rank stops on ``case``, the same on each. The rest of a
    rank's outcome is its own (its step, its failed rows): it is about to exit."""
    stopped = [got[case]["stopped"] for got in ranks]
    assert all(s is not None for s in stopped), stopped
    reasons = {s.split(": ", 1)[1] for s in stopped}
    assert len(reasons) == 1, stopped
    return stopped[0]


def test_a_clean_step_changes_nothing(two):
    assert _same(two, "clean") == {"stopped": None, "failed": [], "void": None, "voted": True}


def test_a_row_one_rank_failed_after_the_forward_fails_on_every_rank(two):
    """Rank 0 fails "b" too, so both drop the head built on N, and "b" leaves
    the next step on both."""
    assert _same(two, "post_one") == {"stopped": None, "failed": ["b"], "void": "failed rids", "voted": True}


def test_rows_each_rank_failed_after_the_forward_fail_on_every_rank(two):
    assert _same(two, "post_each")["failed"] == ["a", "c"]


def test_a_step_one_rank_refused_stops_every_rank(two):
    """Rank 0's forward ran and waits in its first all-reduce for a rank that never
    launched. No verdict repairs that, so the instance stops, with the reason."""
    stopped = _stops(two, "refused_one")
    assert "TP ranks diverged at step seq 7 of LLM/decode" in stopped
    assert "admitted on some ranks and refused on others" in stopped


def test_a_step_every_rank_refused_is_a_refusal_everywhere(two):
    assert _same(two, "refused_all") == {"stopped": None, "failed": [], "void": "admit_error", "voted": True}


def test_a_row_one_rank_failed_before_the_forward_stops_every_rank(two):
    assert "it ran 2 rows on one rank and 3 on another" in _stops(two, "pre_one")


def test_different_rows_failed_before_the_forward_stop_every_rank(two):
    assert "ran different rows on different ranks" in _stops(two, "pre_swapped")


def test_a_row_rank_0_failed_before_sending_the_step_voids_the_head_everywhere(two):
    """"d" was never in any other rank's step, so only rank 0 fails it; but rank 0
    drops its speculation over it, so the other ranks drop theirs too."""
    assert two[0]["unsent"] == {"stopped": None, "failed": ["d"], "void": "failed rids", "voted": True}
    assert two[1]["unsent"] == {"stopped": None, "failed": [], "void": "failed rids", "voted": True}


def test_ranks_voting_on_different_steps_stop(two):
    assert "its ranks voted on steps 7 and 8" in _stops(two, "seq")


def test_a_step_that_raised_on_one_rank_stops_every_rank(two):
    """The rank that raised fails the step's rows as a batch; the others would
    run on with them, into a collective it never joins."""
    assert "it raised on some ranks and not on others" in _stops(two, "raised_one")


def test_a_step_that_raised_on_every_rank_fails_there_as_before(two):
    """Every rank fails the batch in its error handler, which keeps them in step."""
    got = _same(two, "raised_all")
    assert got["stopped"] is None and got["voted"]


def test_the_votes_after_a_stop_still_pair(two):
    assert _same(two, "after") == {"stopped": None, "failed": [], "void": None, "voted": True}


@pytest.mark.parametrize("failing", ["rank1", "rank2", "rank3"])
def test_over_tp2_x_sp2_a_row_any_rank_failed_fails_on_all_four(grid, failing):
    assert _same(grid, failing) == {"stopped": None, "failed": ["c"], "void": "failed rids", "voted": True}


# ------------------------------------------------- in the run loop, one rank

def _voting_rank(monkeypatch, *, tp_rank, flag=True, verdict=None):
    """A rank of the teardown-order harness under TP async, its vote replaced by
    one that records each step voted on and, for step N (seq 0), lands
    ``verdict`` on it: what the other ranks' votes made of N."""
    w, log = _rank(monkeypatch, tp_rank=tp_rank, inbox=[])
    w.parallel_nodes = {NODE}
    w.tp_async_sched = flag
    voted = []

    def _agree_on_step(pending, raised=False):
        voted.append(pending.tp_seq)
        pending.voted = True
        if pending.tp_seq == 0 and verdict == "failed elsewhere":
            pending.failed_on_a_rank = True

    def _stop(timeout):
        raise _Stop

    w._agree_on_step = _agree_on_step
    w.communicator.wait_for_work = _stop
    w.engine_manager.get_engine = lambda node: SimpleNamespace(
        prefix_cache_open=lambda node: False, drop_cached_prefix=lambda head: None,
    )
    w._graph_runtime.push_back_node = lambda *args: None
    w._reset_skip_plan_flags = lambda node_batch: None
    follow = w._try_follow_speculation
    build = w._build_speculation

    def _follow(pending):
        spec = follow(pending)
        spec.continuing_rids = {HANDLES["kept"]}
        return spec

    def _build(pending, consecutive, fair):
        spec, away = build(pending, consecutive, fair)
        if spec is not None:
            spec.continuing_rids = {HANDLES["kept"]}
        return spec, away

    w._try_follow_speculation = _follow
    w._build_speculation = _build
    return w, log, voted


def _run_rank(w):
    try:
        w.run()
    except _Stop:
        return
    raise AssertionError("run returned")


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("flag", [True, False])
def test_each_lockstep_step_is_voted_on_once_under_the_flag_and_never_without(monkeypatch, tp_rank, flag):
    w, log, voted = _voting_rank(monkeypatch, tp_rank=tp_rank, flag=flag)
    _run_rank(w)
    assert ("admit", 1) in log, log
    assert voted == ([0] if flag else [])


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_a_rank_drops_its_n_plus_1_when_another_rank_failed_a_row_of_n(monkeypatch, tp_rank):
    """N ran clean here; the vote says a row of it failed on another rank, which
    drops the head built on N. Rank 0 clears its speculation and a follower drops
    the head, so neither runs N+1."""
    w, log, voted = _voting_rank(monkeypatch, tp_rank=tp_rank, verdict="failed elsewhere")
    _run_rank(w)
    assert voted == [0]
    assert ("admit", 0) in log and ("admit", 1) not in log, log
