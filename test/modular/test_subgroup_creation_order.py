"""Checks the TP / SP subgroups init_dist creates on each rank.

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


def _mesh(tp: int, sp: int) -> GlobalParallelConfig:
    """One node on tp * sp ranks, laid out [sp][tp] as WorkerGraph does."""
    ranks = list(range(tp * sp))
    graph = SimpleNamespace(
        section=SimpleNamespace(get_nodes=lambda: {"dit": None}),
        ranks=ranks,
        tp_size=tp,
        sp_size=sp,
        cfg_size=1,
        _tp_comm_size=tp,
        _tp_ranks=[ranks[s * tp:(s + 1) * tp] for s in range(sp)] if tp > 1 else [],
        _sp_ranks=[ranks[t::tp] for t in range(tp)] if sp > 1 else [],
        _cfg_ranks=[],
        _instance_ranks=[ranks],
    )
    return GlobalParallelConfig(
        worker_graphs={"main": graph}, worker_ids=[f"worker-{r}" for r in ranks],
    )


def _run(monkeypatch, tp: int, sp: int):
    config = _mesh(tp, sp)
    # these need a real torch.distributed (and rank 0 binds a socket)
    monkeypatch.setattr(communication, "rendezvous_store", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(communication, "default_group_timeout", lambda backend: None, raising=False)
    fakes, views = {}, {}
    for rank in range(tp * sp):
        fake = _FakeDist(rank)
        monkeypatch.setattr(communication, "dist", fake)
        view = config.per_worker_config[f"worker-{rank}"]
        view.init_dist(init_method="tcp://127.0.0.1:1", device=torch.device("cpu"))
        fakes[rank], views[rank] = fake, view
    return fakes, views


def _twins(fake: _FakeDist):
    return [c for c in fake.calls if c[0] == "members" and len(c[2]) > 1]


MESHES = [(2, 2), (2, 1), (1, 2), (4, 1), (2, 4), (4, 2)]


@pytest.mark.parametrize(("tp", "sp"), MESHES)
def test_members_of_a_gloo_twin_agree_on_its_name(monkeypatch, tp, sp):
    fakes, _ = _run(monkeypatch, tp, sp)
    names: dict[tuple[int, ...], set[str]] = {}
    for fake in fakes.values():
        for _, _, members, name in _twins(fake):
            names.setdefault(members, set()).add(name)
    assert names and all(len(n) == 1 for n in names.values()), names


@pytest.mark.parametrize(("tp", "sp"), MESHES)
def test_every_rank_makes_the_same_all_rank_calls_in_the_same_order(monkeypatch, tp, sp):
    fakes, _ = _run(monkeypatch, tp, sp)
    sequences = {r: [c for c in f.calls if c[0] == "all"] for r, f in fakes.items()}
    first = sequences[0]
    assert all(seq == first for seq in sequences.values())
    # one split per distinct rank tuple
    assert len(first) == 1 + len({c[2] for c in first[1:]})


@pytest.mark.parametrize(("tp", "sp"), MESHES)
def test_each_gloo_twin_is_made_by_exactly_its_members(monkeypatch, tp, sp):
    fakes, _ = _run(monkeypatch, tp, sp)
    tuples = {c[2] for c in fakes[0].calls if c[0] == "all" and c[1] == "nccl"}
    for members in tuples:
        callers = {r for r, f in fakes.items() if any(c[2] == members for c in _twins(f))}
        assert callers == set(members)


@pytest.mark.parametrize(("tp", "sp"), MESHES)
def test_every_rank_makes_its_gloo_twins_in_one_global_order(monkeypatch, tp, sp):
    # torch's rule for overlapping members-only groups, so no wait-for cycle
    fakes, _ = _run(monkeypatch, tp, sp)
    order = [c[2] for c in fakes[0].calls if c[0] == "all" and c[1] == "nccl"]
    for rank, fake in fakes.items():
        assert [c[2] for c in _twins(fake)] == [t for t in order if rank in t]


def test_the_groups_land_on_the_right_comm_groups(monkeypatch):
    _, views = _run(monkeypatch, 2, 2)
    for rank, view in views.items():
        tp, sp = view.node_to_tp_group["dit"], view.node_to_sp_group["dit"]
        assert tp.device_group.ranks == tuple(tp.group_members) == tp.cpu_group.ranks
        assert sp.device_group.ranks == tuple(sp.group_members) == sp.cpu_group.ranks
        assert tp.cpu_group.backend == "gloo" and tp.device_group.backend == "nccl"
        assert rank in tp.group_members and rank in sp.group_members
