"""``MSTAR_TP_ASYNC_SCHED`` must agree across the ranks of a lockstep instance.

The flag is per-rank env. A leader without it never sends a decision the
async follower waits for; a leader with it sends heads a serial follower
cannot always build. ``Worker._verify_tp_async_sched_agrees`` all_gathers
the per-node verdict at startup and refuses a mismatch.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.worker.worker import Worker  # noqa: E402


class _Group:
    """A CommGroup stand-in whose all_gather returns preset per-rank values."""

    def __init__(self, members, gathered):
        self.world_size = len(members)
        self.group_members = list(members)
        self._gathered = gathered

    def all_gather(self, input_, dim=0):
        return torch.tensor(self._gathered, dtype=input_.dtype)


def _worker(node, *, async_on, tp_group, sp_group=None):
    groups = types.SimpleNamespace(
        get_tp_config_for_node=lambda n: tp_group,
        get_sp_config_for_node=lambda n: sp_group or _Group([0], [int(async_on)]),
    )
    return types.SimpleNamespace(
        parallel_nodes={node},
        parallel_groups=groups,
        device="cpu",
        _tp_async_for=lambda n: async_on,
    )


def _check(worker):
    Worker._verify_tp_async_sched_agrees(worker)


def test_agreeing_ranks_pass():
    _check(_worker("thinker", async_on=True, tp_group=_Group([0, 1], [1, 1])))
    _check(_worker("thinker", async_on=False, tp_group=_Group([0, 1], [0, 0])))


def test_non_parallel_group_is_skipped():
    _check(_worker("thinker", async_on=True, tp_group=_Group([0], [1])))


def test_tp_mismatch_is_refused():
    with pytest.raises(RuntimeError, match="disagrees"):
        _check(_worker("thinker", async_on=True, tp_group=_Group([0, 1], [1, 0])))


def test_sp_mismatch_is_refused():
    with pytest.raises(RuntimeError, match="disagrees"):
        _check(_worker(
            "dit", async_on=True,
            tp_group=_Group([0], [1]),
            sp_group=_Group([0, 1], [1, 0]),
        ))


# --- the submodule's default (MSTAR_TP_ASYNC_SCHED unset) --------------------

def _resolving_worker(prefers, *, env_unset=True, already_on=False):
    submodule = types.SimpleNamespace(prefers_tp_async_scheduling=prefers)
    engine = types.SimpleNamespace(submodule=lambda n: submodule)
    return types.SimpleNamespace(
        parallel_nodes={"LLM"}, tp_async_sched=already_on, tp_async_nodes=None,
        _tp_async_from_model=env_unset, worker_id="w0", is_tp_follower=False,
        engine_manager=types.SimpleNamespace(get_engine=lambda n: engine),
    )


def test_submodule_default_turns_async_on_when_the_variable_is_unset():
    w = _resolving_worker(True)
    Worker._resolve_tp_async_default(w)
    assert w.tp_async_sched and w.tp_async_nodes == frozenset({"LLM"})


def test_a_set_variable_wins_over_the_submodule_default():
    w = _resolving_worker(True, env_unset=False)  # MSTAR_TP_ASYNC_SCHED=0
    Worker._resolve_tp_async_default(w)
    assert not w.tp_async_sched and w.tp_async_nodes is None


def test_no_preference_keeps_the_serial_protocol():
    w = _resolving_worker(False)
    Worker._resolve_tp_async_default(w)
    assert not w.tp_async_sched
    # a node whose engine has no loaded submodule is skipped, not an error
    w = _resolving_worker(True)
    w.engine_manager = types.SimpleNamespace(get_engine=lambda n: (_ for _ in ()).throw(KeyError(n)))
    Worker._resolve_tp_async_default(w)
    assert not w.tp_async_sched
