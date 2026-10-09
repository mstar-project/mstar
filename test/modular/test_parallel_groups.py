"""``WorkerParallelGroups.all_in_same_group``: may local nodes share one resource?

A logical resource spec may span remote replicas (BAGEL's LLM + its two CFG
branches), but one worker's physical resource instance can only serve local
nodes in the same (tp, sp) group. The check has to answer per dimension: SP is
usually unregistered, and the lazy getters mint a fresh single-rank group per
node, so comparing those would reject every TP-only deployment.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.distributed.communication import (
    CommGroup,
    GlobalParallelConfig,
    WorkerParallelGroups,
)


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


def test_remote_replicas_are_excluded_before_group_validation():
    groups = _groups()
    groups.add("llm", _tp([0, 1]))
    groups.add("llm_cfg_text", _tp([2, 3]))
    groups.add("llm_cfg_img", _tp([4, 5]))

    spec_nodes = {"llm", "llm_cfg_text", "llm_cfg_img"}
    local_nodes = spec_nodes & {"llm"}

    assert groups.all_in_same_group(spec_nodes) is False
    assert groups.all_in_same_group(local_nodes) is True


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


def test_remote_replicas_with_matching_parallel_shapes_are_compatible():
    groups = _groups()
    groups.node_to_parallel_shapes = {
        "llm": frozenset({(2, 1)}),
        "llm_cfg": frozenset({(2, 1)}),
    }

    assert groups.all_have_compatible_parallel_shape({"llm", "llm_cfg"})


def test_remote_replicas_with_different_parallel_shapes_are_rejected():
    groups = _groups()
    groups.node_to_parallel_shapes = {
        "llm": frozenset({(2, 1)}),
        "llm_cfg": frozenset({(4, 1)}),
    }

    assert not groups.all_have_compatible_parallel_shape({"llm", "llm_cfg"})


def test_global_instance_groups_determine_remote_transfer_need():
    groups = _groups()
    groups.node_to_instance_groups = {
        "llm": frozenset({(0, 1)}),
        "llm_cfg": frozenset({(2, 3)}),
    }

    assert groups.resource_needs_remote_transfer(
        {"llm", "llm_cfg"}, {"llm"}
    )

    groups.node_to_instance_groups["llm_cfg"] = frozenset({(0, 1)})
    assert not groups.resource_needs_remote_transfer(
        {"llm", "llm_cfg"}, {"llm", "llm_cfg"}
    )


def test_global_config_exposes_remote_replica_shapes_to_every_worker():
    def worker_graph(node, ranks, tp_size):
        return SimpleNamespace(
            section=SimpleNamespace(get_nodes=lambda: {node: None}),
            ranks=ranks,
            tp_size=tp_size,
            sp_size=1,
            _tp_comm_size=tp_size,
            _tp_ranks=[ranks] if tp_size > 1 else [],
            _sp_ranks=[],
            _instance_ranks=[ranks] if tp_size > 1 else [],
        )

    config = GlobalParallelConfig(
        worker_graphs={
            "main": worker_graph("llm", [0, 1], 2),
            "cfg": worker_graph("llm_cfg", [2], 1),
        },
        worker_ids=["worker-0", "worker-1", "worker-2"],
    )

    worker_view = config.per_worker_config["worker-0"]
    assert not worker_view.all_have_compatible_parallel_shape(
        {"llm", "llm_cfg"}
    )
