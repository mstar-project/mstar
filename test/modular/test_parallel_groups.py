"""``WorkerParallelGroups.all_in_same_group``: may these nodes share one resource?

A resource spec naming several nodes (BAGEL's LLM + its two CFG branches share
one KV cache) is only legal when those nodes sit in the same (tp, sp) group.
The check has to answer per dimension: SP is usually unregistered, and the
lazy getters mint a fresh single-rank group per node, so comparing those would
reject every TP-only deployment.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup, WorkerParallelGroups


def _groups() -> WorkerParallelGroups:
    return WorkerParallelGroups(num_workers=1, global_rank=0)


def _tp(members: list[int], rank: int = 0) -> CommGroup:
    return CommGroup(
        my_global_rank=members[rank], my_group_rank=rank, group_members=members
    )


def test_unregistered_nodes_are_in_the_same_group():
    assert _groups().all_in_same_group(["a", "b"]) is True


def test_shared_tp_group_with_no_sp():
    """The regression: SP falls back to a per-node trivial group, so an
    identity check saw two different objects and rejected a valid spec."""
    groups = _groups()
    tp = _tp([0, 1])
    groups.add("llm", tp)
    groups.add("llm_cfg", tp)

    assert groups.all_in_same_group(["llm", "llm_cfg"]) is True


def test_equivalent_tp_groups_registered_separately():
    """Same membership, distinct objects — still the same group."""
    groups = _groups()
    groups.add("llm", _tp([0, 1]))
    groups.add("llm_cfg", _tp([0, 1]))

    assert groups.all_in_same_group(["llm", "llm_cfg"]) is True


def test_different_tp_groups_are_rejected():
    groups = _groups()
    groups.add("llm", _tp([0, 1]))
    groups.add("codec", _tp([2, 3]))

    assert groups.all_in_same_group(["llm", "codec"]) is False


def test_tp_node_and_unparallelized_node_are_rejected():
    groups = _groups()
    groups.add("llm", _tp([0, 1]))

    assert groups.all_in_same_group(["llm", "codec"]) is False


def test_sp_is_checked_too():
    groups = _groups()
    tp = _tp([0, 1])
    groups.add("a", tp)
    groups.add("b", tp)
    groups.add_sp("a", _tp([0, 1]))
    groups.add_sp("b", _tp([2, 3]))

    assert groups.all_in_same_group(["a", "b"]) is False


def test_the_check_does_not_cache_groups_for_remote_nodes():
    """``spec.nodes`` names nodes this worker may not host; asking about them
    must not leave a trivial group behind for one that registers later."""
    groups = _groups()
    groups.add("llm", _tp([0, 1]))

    groups.all_in_same_group(["llm", "elsewhere"])

    assert "elsewhere" not in groups.node_to_tp_group
    assert "elsewhere" not in groups.node_to_sp_group
