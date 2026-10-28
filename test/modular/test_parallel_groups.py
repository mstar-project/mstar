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

import pytest
import torch

sys.path.insert(0, ".")

from mstar.distributed.communication import (
    CommGroup,
    GlobalParallelConfig,
    WorkerParallelGroups,
)
from mstar.model.base import WorkerGraph


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
    groups.add("llm", tp, "tp")
    groups.add("llm_cfg", tp, "tp")

    assert groups.all_in_same_group(["llm", "llm_cfg"]) is True


def test_equivalent_tp_groups_registered_separately():
    """Same membership, distinct objects — still the same group."""
    groups = _groups()
    groups.add("llm", _tp([0, 1]), "tp")
    groups.add("llm_cfg", _tp([0, 1]), "tp")

    assert groups.all_in_same_group(["llm", "llm_cfg"]) is True


def test_different_tp_groups_are_rejected():
    groups = _groups()
    groups.add("llm", _tp([0, 1]), "tp")
    groups.add("codec", _tp([2, 3]), "tp")

    assert groups.all_in_same_group(["llm", "codec"]) is False


def test_remote_replicas_are_excluded_before_group_validation():
    groups = _groups()
    groups.add("llm", _tp([0, 1]), "tp")
    groups.add("llm_cfg_text", _tp([2, 3]), "tp")
    groups.add("llm_cfg_img", _tp([4, 5]), "tp")

    spec_nodes = {"llm", "llm_cfg_text", "llm_cfg_img"}
    local_nodes = spec_nodes & {"llm"}

    assert groups.all_in_same_group(spec_nodes) is False
    assert groups.all_in_same_group(local_nodes) is True


def test_tp_node_and_unparallelized_node_are_rejected():
    groups = _groups()
    groups.add("llm", _tp([0, 1]), "tp")

    assert groups.all_in_same_group(["llm", "codec"]) is False


def test_sp_is_checked_too():
    groups = _groups()
    tp = _tp([0, 1])
    groups.add("a", tp, "tp")
    groups.add("b", tp, "tp")
    groups.add("a", _tp([0, 1]), "sp")
    groups.add("b", _tp([2, 3]), "sp")

    assert groups.all_in_same_group(["a", "b"]) is False


def test_the_check_does_not_cache_groups_for_remote_nodes():
    """``spec.nodes`` names nodes this worker may not host; asking about them
    must not leave a trivial group behind for one that registers later."""
    groups = _groups()
    groups.add("llm", _tp([0, 1]), "tp")

    groups.all_in_same_group(["llm", "elsewhere"])

    assert "elsewhere" not in groups.axis_groups["tp"]
    assert "elsewhere" not in groups.axis_groups["sp"]


def test_remote_replicas_with_matching_parallel_shapes_are_compatible():
    groups = _groups()
    groups.node_to_parallel_shapes = {
        "llm": frozenset({(1, 2)}),
        "llm_cfg": frozenset({(1, 2)}),
    }

    assert groups.all_have_compatible_parallel_shape({"llm", "llm_cfg"})


def test_remote_replicas_with_different_parallel_shapes_are_rejected():
    groups = _groups()
    groups.node_to_parallel_shapes = {
        "llm": frozenset({(1, 2)}),
        "llm_cfg": frozenset({(1, 4)}),
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
        return WorkerGraph(
            section=SimpleNamespace(get_nodes=lambda: {node: None}),
            graph_walks=set(), ranks=ranks, tp_size=tp_size,
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


@pytest.mark.parametrize(("sizes", "groups"), [
    ({"tp_size": 2}, [(0, 1), (2, 3)]),
    ({"tp_size": 2, "sp_size": 2}, [(0, 1), (0, 2), (1, 3), (2, 3)]),
])
def test_every_rank_of_an_instance_holds_the_whole_instance_as_one_group(sizes, groups):
    """Every agreement site (barriers, capture votes, the KV symmetry check,
    the read-check vote) runs one host-side collective on
    ``JointGroups.instance``, so it must span every axis, and rank it in the
    order the conductor and the leader election use: the instance's ranks as
    listed. The instance is not one of the world's device groups: it takes
    no NCCL communicator."""
    wg = WorkerGraph(
        section=SimpleNamespace(get_nodes=lambda: {"dit": None}),
        graph_walks=set(), ranks=[0, 1, 2, 3], **sizes,
    )
    config = GlobalParallelConfig({"dit": wg}, [f"worker-{r}" for r in range(4)])

    for rank in range(4):
        view = config.per_worker_config[f"worker-{rank}"]
        joint = view.get_joint_group_for_node("dit")
        block = next(block for block in wg.replica_ranks if rank in block)
        assert joint.instance.group_members == block
        assert joint.rank == block.index(rank)
        assert view.world_parallel_groups == groups


def test_a_lockstep_instance_refuses_a_device_collective():
    """The instance has no device group, and torch would run ``group=None`` on
    the whole world: a hang when the world is wider than the instance."""
    wg = WorkerGraph(
        section=SimpleNamespace(get_nodes=lambda: {"dit": None}),
        graph_walks=set(), ranks=[0, 1, 2, 3], tp_size=2, sp_size=2,
    )
    config = GlobalParallelConfig({"dit": wg}, [f"worker-{r}" for r in range(4)])
    instance = config.per_worker_config["worker-0"].get_joint_group_for_node("dit").instance
    with pytest.raises(ValueError, match="host-only"):
        instance.all_gather(torch.zeros(2))


@pytest.mark.parametrize(
    ("device_type", "device_backend", "expected_backend"),
    [
        ("cpu", "gloo", "gloo"),
        ("cuda", "nccl", "nccl"),
        ("xpu", "xccl", "cpu:gloo,xpu:xccl"),
    ],
)
def test_init_dist_uses_mixed_groups_only_on_xpu(
    monkeypatch, device_type, device_backend, expected_backend,
):
    from mstar.distributed import communication

    groups = WorkerParallelGroups(
        num_workers=4, global_rank=0, any_parallelism=True,
        world_parallel_groups=[(0, 1), (2, 3)],
    )
    tp = _tp([0, 1])
    sp = _tp([0, 1])
    groups.add("llm", tp, "tp")
    groups.add("llm_cfg", tp, "tp")
    groups.add("llm", sp, "sp")
    singleton = CommGroup.trivial()
    groups.add("codec", singleton, "tp")
    # a one-axis instance shares its axis's gloo group; a wider one gets its own
    pair, block = _tp([0, 1]), _tp([0, 1, 2, 3])
    groups.node_to_instance_comm.update(llm_cfg=pair, llm=block)
    groups.node_to_instance_groups.update(llm_cfg=frozenset({(0, 1)}), llm=frozenset({(0, 1, 2, 3)}))
    calls = []
    handles = {}
    devices = []
    store = object()
    store_calls = []

    def create_store(init_method, rank, world_size, timeout):
        store_calls.append((init_method, rank, world_size, timeout))
        return store

    monkeypatch.setattr(communication, "rendezvous_store", create_store)
    monkeypatch.setattr(torch.accelerator, "set_device_index", devices.append)
    monkeypatch.setattr(
        communication.dist, "get_default_backend_for_device", lambda device: device_backend,
    )
    monkeypatch.setattr(
        communication.dist, "init_process_group", lambda **kwargs: calls.append(kwargs),
    )

    def new_group(ranks, **kwargs):
        calls.append((tuple(ranks), kwargs))
        key = (tuple(ranks), kwargs.get("backend"))
        handles[key] = object()
        return handles[key]

    monkeypatch.setattr(communication.dist, "new_group", new_group)
    device = torch.device(device_type, 0) if device_type != "cpu" else torch.device("cpu")
    groups.init_dist(init_method="test://world", device=device)
    timeout = communication.default_group_timeout(expected_backend)
    assert store_calls == [("test://world", 0, 4, timeout)]

    expected_group_calls = []
    for ranks in ((0, 1), (2, 3)):
        expected_group_calls.append((ranks, {}))
        if device_type != "xpu":
            expected_group_calls.append((
                ranks, {"backend": "gloo", "use_local_synchronization": True},
            ))
    assert calls == [
        {
            "backend": expected_backend, "store": store, "timeout": timeout,
            "world_size": 4, "rank": 0,
            "device_id": device,
        },
        # the instance first: gloo only, or on XPU the world's mixed group
        # (like the XPU axis groups)
        ((0, 1, 2, 3), {} if device_type == "xpu" else {"backend": "gloo", "use_local_synchronization": True}),
        *expected_group_calls,
    ]
    instance_backend = None if device_type == "xpu" else "gloo"
    assert pair.cpu_group is tp.cpu_group and pair.device_group is None
    assert block.cpu_group is handles[((0, 1, 2, 3), instance_backend)] and block.device_group is None
    assert devices == ([] if device_type == "cpu" else [device])
    for group in (tp, sp):
        assert group.initialized
        assert group.device_group is handles[((0, 1), None)]
        if device_type == "xpu":
            assert group.cpu_group is group.device_group
        else:
            assert group.cpu_group is handles[((0, 1), "gloo")]
            assert group.cpu_group is not group.device_group
    assert singleton.initialized
    assert singleton.device_group is None


def test_every_rank_makes_the_instance_gloo_groups_first_and_alike(monkeypatch):
    """torch names a members-only group after its ranks and how many groups the
    process holds, so the members of an instance agree on its name only if
    every rank has made the same number of groups before it: the instances go
    first, and a rank outside one makes a group of just itself."""
    from mstar.distributed import communication

    wg = WorkerGraph(
        section=SimpleNamespace(get_nodes=lambda: {"dit": None}),
        graph_walks=set(), ranks=list(range(8)), tp_size=2, sp_size=2,
    )
    config = GlobalParallelConfig({"dit": wg}, [f"worker-{r}" for r in range(8)])
    monkeypatch.setattr(communication, "rendezvous_store", lambda *args: object())
    monkeypatch.setattr(communication.dist, "get_default_backend_for_device", lambda device: "gloo")
    monkeypatch.setattr(communication.dist, "init_process_group", lambda **kwargs: None)
    calls = []

    def new_group(ranks, **kwargs):
        calls.append((tuple(ranks), kwargs.get("backend")))
        return object()

    monkeypatch.setattr(communication.dist, "new_group", new_group)
    for rank in range(8):
        calls.clear()
        config.per_worker_config[f"worker-{rank}"].init_dist(device=torch.device("cpu"))
        mine = next(block for block in wg.replica_ranks if rank in block)
        assert calls[:2] == [
            (tuple(block) if block == mine else (rank,), "gloo") for block in wg.replica_ranks
        ], rank
