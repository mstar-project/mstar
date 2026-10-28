"""``MSTAR_TP_ASYNC_SCHED`` must agree across the ranks of a lockstep instance.

The flag is per-rank env. A leader without it never sends a decision the
async follower waits for; a leader with it sends heads a serial follower
cannot always build. ``Worker._verify_tp_async_sched_agrees`` reduces
the per-node verdict over the instance at startup and refuses a mismatch.
"""

import sys
import types
from pathlib import Path

import pytest
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.distributed.communication import JointGroups  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402


class _Group:
    """A CommGroup stand-in whose ranks hold preset values, one each."""

    def __init__(self, members, values):
        self.world_size = len(members)
        self.group_members = list(members)
        self._values = values

    def host_all_reduce(self, values, op):
        return [(min if op == dist.ReduceOp.MIN else max)(self._values)]


def _worker(node, *, async_on, instance):
    """The check reads only the node's lockstep instance group."""
    joint = JointGroups(tp_group=None, sp_group=None, instance=instance)
    groups = types.SimpleNamespace(get_joint_group_for_node=lambda n: joint)
    return types.SimpleNamespace(
        parallel_nodes={node},
        parallel_groups=groups,
        _tp_async_for=lambda n: async_on,
    )


def _check(worker):
    Worker._verify_tp_async_sched_agrees(worker)


def test_agreeing_ranks_pass():
    _check(_worker("thinker", async_on=True, instance=_Group([0, 1], [1, 1])))
    _check(_worker("thinker", async_on=False, instance=_Group([0, 1], [0, 0])))


def test_mismatch_is_refused():
    with pytest.raises(RuntimeError, match="disagrees"):
        _check(_worker("thinker", async_on=True, instance=_Group([0, 1], [1, 0])))

