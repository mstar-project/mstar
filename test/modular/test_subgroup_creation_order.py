"""Checks the subgroups init_dist creates on each rank: the device and gloo
groups of every mesh axis, and the gloo group of a multi-axis lockstep instance.

With the default group bound to a device, every device subgroup is an
ncclCommSplit that all ranks join, while a gloo twin only waits for its
members. torch names a members-only group after its ranks and the number of
groups this process has registered, so the members of a twin have to reach it
with the same count or they rendezvous under different store prefixes and
wait forever. A 2x2 TP x SP mesh hung at startup that way. The fake dist below
names groups the way torch 2.9 to 2.12 do and logs each rank's calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.distributed import communication
from mstar.distributed.communication import GlobalParallelConfig
from mstar.model.base import WorkerGraph

_NON_MEMBER = object()


class _FakeDist:
    """Enough of torch.distributed for init_dist."""

    def __init__(self, rank: int):
        self.rank = rank
        self.group_count = 0  # torch's _world.group_count
        self.registered = 0  # len(torch's _world.pg_names)
        self.calls: list[tuple[str, str, tuple[int, ...], str]] = []

    def get_default_backend_for_device(self, device_type):
        return "nccl"

    def init_process_group(self, **kwargs):
        name = str(self.group_count)
        self.group_count += 1
        self.registered += 1
        self.calls.append(("all", "init", (), name))

    def new_group(self, ranks, backend=None, use_local_synchronization=False):
        members = tuple(sorted(ranks))
        backend = backend or "nccl"
        if use_local_synchronization:
            if self.rank not in members:
                return None
            # _hash_ranks_to_str, before the sha1
            name = "_".join(map(str, members)) + f"_{self.registered}"
            kind = "members"
        else:
            name = str(self.group_count)
            self.group_count += 1
            kind = "all"
        self.calls.append((kind, backend, members, name))
        if self.rank not in members:
            return _NON_MEMBER
        self.registered += 1
        return SimpleNamespace(backend=backend, ranks=members, name=name)


def _mesh(tp: int, sp: int, cfg: int = 1) -> GlobalParallelConfig:
    """One node on one lockstep instance of cfg * sp * tp ranks."""
    graph = WorkerGraph(
        section=SimpleNamespace(get_nodes=lambda: {"dit": None}), graph_walks=set(),
        ranks=list(range(tp * sp * cfg)), tp_size=tp, sp_size=sp, cfg_size=cfg,
    )
    return GlobalParallelConfig(
        worker_graphs={"main": graph}, worker_ids=[f"worker-{r}" for r in graph.ranks],
    )


def _run(monkeypatch, tp: int, sp: int, cfg: int = 1):
    config = _mesh(tp, sp, cfg)
    # these need a real torch.distributed (and rank 0 binds a socket)
    monkeypatch.setattr(communication, "rendezvous_store", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(communication, "default_group_timeout", lambda backend: None, raising=False)
    fakes, views = {}, {}
    for rank in range(tp * sp * cfg):
        fake = _FakeDist(rank)
        monkeypatch.setattr(communication, "dist", fake)
        view = config.per_worker_config[f"worker-{rank}"]
        view.init_dist(init_method="tcp://127.0.0.1:1", device=torch.device("cpu"))
        fakes[rank], views[rank] = fake, view
    return fakes, views


def _twins(fake: _FakeDist):
    return [c for c in fake.calls if c[0] == "members" and len(c[2]) > 1]


# (tp, sp, cfg); a mesh with two axes or more also makes an instance gloo group
MESHES = [(2, 2, 1), (2, 1, 1), (1, 2, 1), (4, 1, 1), (2, 4, 1), (4, 2, 1), (1, 2, 2), (2, 1, 2), (2, 2, 2)]


def _gloo_tuples(views) -> list[tuple[int, ...]]:
    """Every gloo group's ranks, sorted: each axis group's and each instance's."""
    view = next(iter(views.values()))
    blocks = {block for blocks in view.node_to_instance_groups.values() for block in blocks}
    return sorted(set(view.world_parallel_groups) | {b for b in blocks if len(b) > 1})


@pytest.mark.parametrize(("tp", "sp", "cfg"), MESHES)
def test_members_of_a_gloo_twin_agree_on_its_name(monkeypatch, tp, sp, cfg):
    fakes, _ = _run(monkeypatch, tp, sp, cfg)
    names: dict[tuple[int, ...], set[str]] = {}
    for fake in fakes.values():
        for _, _, members, name in _twins(fake):
            names.setdefault(members, set()).add(name)
    assert names and all(len(n) == 1 for n in names.values()), names


@pytest.mark.parametrize(("tp", "sp", "cfg"), MESHES)
def test_every_rank_makes_the_same_all_rank_calls_in_the_same_order(monkeypatch, tp, sp, cfg):
    fakes, _ = _run(monkeypatch, tp, sp, cfg)
    sequences = {r: [c for c in f.calls if c[0] == "all"] for r, f in fakes.items()}
    first = sequences[0]
    assert all(seq == first for seq in sequences.values())
    # one split per distinct rank tuple
    assert len(first) == 1 + len({c[2] for c in first[1:]})


@pytest.mark.parametrize(("tp", "sp", "cfg"), MESHES)
def test_each_gloo_twin_is_made_by_exactly_its_members(monkeypatch, tp, sp, cfg):
    fakes, views = _run(monkeypatch, tp, sp, cfg)
    for members in _gloo_tuples(views):
        callers = {r for r, f in fakes.items() if any(c[2] == members for c in _twins(f))}
        assert callers == set(members)


@pytest.mark.parametrize(("tp", "sp", "cfg"), MESHES)
def test_every_rank_makes_its_gloo_twins_in_one_global_order(monkeypatch, tp, sp, cfg):
    # torch's rule for overlapping members-only groups, so no wait-for cycle
    fakes, views = _run(monkeypatch, tp, sp, cfg)
    order = _gloo_tuples(views)
    for rank, fake in fakes.items():
        assert [c[2] for c in _twins(fake)] == [t for t in order if rank in t]


def test_the_groups_land_on_the_right_comm_groups(monkeypatch):
    _, views = _run(monkeypatch, 2, 2)
    for rank, view in views.items():
        tp, sp = view.axis_groups["tp"]["dit"], view.axis_groups["sp"]["dit"]
        assert tp.device_group.ranks == tuple(tp.group_members) == tp.cpu_group.ranks
        assert sp.device_group.ranks == tuple(sp.group_members) == sp.cpu_group.ranks
        assert tp.cpu_group.backend == "gloo" and tp.device_group.backend == "nccl"
        assert rank in tp.group_members and rank in sp.group_members
        instance = view.node_to_instance_comm["dit"]
        assert instance.cpu_group.ranks == (0, 1, 2, 3) and instance.cpu_group.backend == "gloo"
        assert instance.device_group is None
