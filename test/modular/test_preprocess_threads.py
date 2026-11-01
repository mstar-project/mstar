"""The API server's intra-op thread count: the cgroup quota it is derived from, and
which of config, model and default wins."""

from __future__ import annotations

import pytest

from mstar.api_server import entrypoint
from mstar.utils import cpus


def _cgroup_tree(tmp_path, limits: dict[str, str], own: str):
    root = tmp_path / "cgroup"
    for rel, limit in limits.items():
        level = root / rel.lstrip("/")
        level.mkdir(parents=True, exist_ok=True)
        (level / "cpu.max").write_text(limit + "\n")
    (root / own.lstrip("/")).mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{own}\n")
    return root, str(proc)


def test_the_tightest_quota_over_the_cgroup_and_its_ancestors(tmp_path):
    root, proc = _cgroup_tree(tmp_path, {
        "/": "max 100000", "/pod": "5440000 100000", "/pod/ctr": "8000000 100000",
    }, own="/pod/ctr")
    assert cpus.cgroup_cpu_quota(root, proc) == pytest.approx(54.4)


def test_no_quota_anywhere_is_none(tmp_path):
    root, proc = _cgroup_tree(tmp_path, {"/": "max 100000"}, own="/svc")
    assert cpus.cgroup_cpu_quota(root, proc) is None
    assert cpus.cgroup_cpu_quota(tmp_path / "absent", str(tmp_path / "absent_proc")) is None


def test_usable_cpus_is_the_affinity_capped_by_the_quota(monkeypatch):
    monkeypatch.setattr(cpus.os, "sched_getaffinity", lambda pid: set(range(256)))
    monkeypatch.setattr(cpus, "cgroup_cpu_quota", lambda: 40.8)
    assert cpus.usable_cpus() == 40
    monkeypatch.setattr(cpus, "cgroup_cpu_quota", lambda: 0.5)
    assert cpus.usable_cpus() == 1
    monkeypatch.setattr(cpus, "cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(cpus.os, "sched_getaffinity", lambda pid: set(range(32)))
    assert cpus.usable_cpus() == 32


@pytest.mark.parametrize(("usable", "threads"), [(1, 1), (7, 1), (8, 2), (12, 3), (16, 4), (40, 4), (256, 4)])
def test_default_preprocess_threads(usable, threads):
    assert entrypoint.default_preprocess_threads(usable) == threads


class _Model:
    PREPROCESS_TORCH_THREADS = None


class _Pinned:
    PREPROCESS_TORCH_THREADS = 6


@pytest.mark.parametrize(("model", "config", "expect"), [
    (_Model, {}, 3),                                 # the quota-derived default
    (_Pinned, {}, 6),                                # the model's own count
    (_Pinned, {"preprocess_torch_threads": 2}, 2),   # the deployment's count
    (_Pinned, {"preprocess_torch_threads": None}, None),  # null: torch's default, untouched
])
def test_config_then_model_then_default(monkeypatch, model, config, expect):
    calls = []
    monkeypatch.setattr("torch.set_num_threads", calls.append)
    monkeypatch.setattr(entrypoint, "default_preprocess_threads", lambda: 3)
    entrypoint._set_preprocess_threads(model, config)
    assert calls == ([] if expect is None else [expect])
