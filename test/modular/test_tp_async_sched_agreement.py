"""``MSTAR_TP_ASYNC_SCHED`` must agree across the ranks of a lockstep instance.

The flag is per-rank env. A leader without it never sends a decision the
async follower waits for; a leader with it sends heads a serial follower
cannot always build. ``Worker._verify_tp_async_sched_agrees`` votes on the
per-node verdict at startup and refuses a mismatch.
"""

import sys
import threading
import types
from pathlib import Path

import pytest
from torch.distributed import ReduceOp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402
from mstar.platforms import CudaPlatform, RocmPlatform  # noqa: E402
from mstar.worker import worker as worker_module  # noqa: E402
from mstar.worker.worker import Worker, _tp_async_sched_setting  # noqa: E402


class _Group:
    """One rank's view of a group whose ranks run ``agree`` on threads."""

    def __init__(self, posted, barrier, rank):
        self.world_size = len(posted)
        self.group_members = list(range(len(posted)))
        self._posted, self._barrier, self._rank = posted, barrier, rank

    def agree(self, values, op):
        self._posted[self._rank] = list(values)
        self._barrier.wait()
        pick = min if op == ReduceOp.MIN else max
        out = [pick(column) for column in zip(*self._posted, strict=True)]
        self._barrier.wait()
        return out


def _run_ranks(async_per_rank, axis):
    """Run the check on one worker per rank of a ``node`` parallel over ``axis``;
    return each rank's exception, or None."""
    posted = [None] * len(async_per_rank)
    barrier = threading.Barrier(len(async_per_rank), timeout=10)
    errors = [None] * len(async_per_rank)

    def rank(i):
        group = _Group(posted, barrier, i)
        joint = JointGroups(
            tp_group=group if axis == "tp" else CommGroup.trivial(),
            sp_group=group if axis == "sp" else CommGroup.trivial(),
        )
        worker = types.SimpleNamespace(
            parallel_nodes={"node"},
            parallel_groups=types.SimpleNamespace(get_joint_group_for_node=lambda n: joint),
            _tp_async_for=lambda n: async_per_rank[i],
        )
        try:
            Worker._verify_tp_async_sched_agrees(worker)
        except RuntimeError as exc:
            errors[i] = exc

    threads = [threading.Thread(target=rank, args=(i,)) for i in range(len(async_per_rank))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


@pytest.mark.parametrize("axis", ["tp", "sp"])
@pytest.mark.parametrize("flag", [True, False])
def test_agreeing_ranks_pass(axis, flag):
    assert _run_ranks([flag, flag], axis) == [None, None]


@pytest.mark.parametrize("axis", ["tp", "sp"])
def test_mismatch_is_refused_on_every_rank(axis):
    errors = _run_ranks([True, False], axis)
    assert all(exc is not None and "disagrees" in str(exc) for exc in errors)


def _worker_with_tp_sizes(tp_sizes):
    worker = Worker.__new__(Worker)
    worker.parallel_nodes = set(tp_sizes)
    groups = {node: types.SimpleNamespace(world_size=size) for node, size in tp_sizes.items()}
    worker.parallel_groups = types.SimpleNamespace(get_tp_config_for_node=groups.__getitem__)
    return worker


@pytest.mark.parametrize("platform, want", [
    (RocmPlatform(), frozenset({"tp2", "tp2_sp2"})),
    (CudaPlatform(), frozenset()),
])
def test_the_platform_default_covers_tp_nodes_on_amd_gpus_only(monkeypatch, platform, want):
    """Sequence-parallel-only nodes stay serial; the sizes come from the
    config, so every rank of an instance gets the same answer."""
    monkeypatch.setattr(worker_module, "current_platform", platform)
    worker = _worker_with_tp_sizes({"tp2": 2, "tp2_sp2": 2, "sp2": 1})
    assert worker._tp_async_default_nodes() == want


@pytest.mark.parametrize("env, config, want", [
    (None, None, (True, frozenset({"LLM"}))),
    ("", None, (True, frozenset({"LLM"}))),
    ("0", None, (False, None)),  # the env var turns the default off
    (None, False, (False, None)),  # and so does the config
    (None, ["talker"], (True, frozenset({"talker"}))),
])
def test_the_default_applies_only_when_nothing_sets_the_flag(env, config, want):
    assert _tp_async_sched_setting(env, config, frozenset({"LLM"})) == want
