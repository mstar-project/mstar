"""``MSTAR_TP_ASYNC_SCHED`` must agree across the ranks of a lockstep instance.

The flag is per-rank env. A leader without it never sends a decision the
async follower waits for; a leader with it sends heads a serial follower
cannot always build. ``Worker._verify_tp_async_sched_agrees`` votes on the
per-node verdict at startup and refuses a mismatch. With it on,
``MSTAR_PRE_PLAN_SPEC`` has to agree too, since a follower builds a head that
N voids as rank 0 pre-planned it.
"""

import sys
import threading
import types
from pathlib import Path

import pytest
from torch.distributed import ReduceOp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402
from mstar.utils import knobs  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

# each rank's thread sets its own pre-plan setting
_RANK = threading.local()


@pytest.fixture(autouse=True)
def _pre_plan_per_rank(monkeypatch):
    monkeypatch.setattr(knobs, "pre_plan_enabled", lambda: getattr(_RANK, "pre_plan", True))


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


def _run_ranks(async_per_rank, axis, pre_plan=None):
    """Run the check on one worker per rank of a ``node`` parallel over ``axis``;
    return each rank's exception, or None."""
    posted = [None] * len(async_per_rank)
    barrier = threading.Barrier(len(async_per_rank), timeout=10)
    errors = [None] * len(async_per_rank)

    def rank(i):
        _RANK.pre_plan = True if pre_plan is None else pre_plan[i]
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


def test_a_pre_plan_mismatch_is_refused():
    """A follower builds a head that N voids, matching the leader's pre-plan, so the ranks have to pre-plan alike."""
    errors = _run_ranks([True, True], "tp", pre_plan=[True, False])
    assert all(exc is not None and "MSTAR_PRE_PLAN_SPEC disagrees" in str(exc) for exc in errors)


def test_a_pre_plan_mismatch_is_allowed_with_async_scheduling_off():
    """The serial protocol pre-plans no TP step, so ranks that pre-plan differently still start."""
    assert _run_ranks([False, False], "tp", pre_plan=[True, False]) == [None, None]
