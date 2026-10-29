"""The API server's preprocessing pool for a model that takes audio, and the
CPU split a deployment can ask for.

Torch's intra-op pool defaults to a thread per core of the process's affinity.
For a model whose prompt processing is a burst of small torch ops per request
(Whisper's mel extraction), that pool shares the cores with the engine worker's
busy threads and each op waits on a descheduled thread, so an audio model gets
a small pool. ``preprocess_cpus`` also pins it to CPUs of its own and starts
the conductor, whose workers inherit it, on the rest.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from mstar.api_server import data_worker as data_worker_mod
from mstar.api_server import entrypoint
from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.utils import cpus as cpus_mod
from mstar.utils.cpus import MIN_ENGINE_CPUS, preprocess_cpu_split


def test_the_pool_takes_the_last_cpus_and_the_engine_the_rest():
    assert preprocess_cpu_split(2, cpus=range(32, 40)) == ([38, 39], [32, 33, 34, 35, 36, 37])


def test_no_split_that_would_starve_the_engine():
    assert preprocess_cpu_split(1, cpus=range(MIN_ENGINE_CPUS + 1)) is not None
    assert preprocess_cpu_split(2, cpus=range(MIN_ENGINE_CPUS + 1)) is None
    assert preprocess_cpu_split(0, cpus=range(32)) is None


def _model(inputs: dict, threads: int | None = None):
    return SimpleNamespace(max_input_files=lambda: inputs, PREPROCESS_TORCH_THREADS=threads)


@pytest.fixture
def eight_cpus(monkeypatch):
    """A server pinned to 8 CPUs, under a quota that lets it keep them busy."""
    monkeypatch.setattr(cpus_mod.os, "sched_getaffinity", lambda pid: set(range(32, 40)))
    monkeypatch.setattr(cpus_mod, "cgroup_cpu_quota", lambda *a, **k: 54.4)


def test_the_split_is_the_serving_configs_to_ask_for(eight_cpus):
    assert entrypoint._preprocess_split({}) is None
    assert entrypoint._preprocess_split({"preprocess_cpus": 0}) is None
    assert entrypoint._preprocess_split({"preprocess_cpus": 2}) == ([38, 39], list(range(32, 38)))


@pytest.fixture
def thread_counts(monkeypatch):
    import torch

    sizes = []
    monkeypatch.setattr(torch, "set_num_threads", sizes.append)
    return sizes


def test_an_audio_model_gets_a_quarter_of_the_cpus(eight_cpus, thread_counts):
    entrypoint._set_preprocess_threads(_model({"audio": 1}), {})
    assert thread_counts == [2]


def test_a_model_without_audio_keeps_torchs_default(eight_cpus, thread_counts):
    entrypoint._set_preprocess_threads(_model({}), {})
    entrypoint._set_preprocess_threads(_model({"image": 4}), {})
    assert thread_counts == []


def test_the_config_the_model_and_a_split_size_the_pool_in_that_order(eight_cpus, thread_counts):
    audio = {"audio": 1}
    entrypoint._set_preprocess_threads(_model(audio), {"preprocess_torch_threads": 4}, [38, 39])
    entrypoint._set_preprocess_threads(_model(audio, threads=3), {}, [38, 39])
    entrypoint._set_preprocess_threads(_model(audio), {}, [39])
    assert thread_counts == [4, 3, 1]


def test_the_data_worker_moves_to_its_cpus_before_its_first_torch_op(monkeypatch):
    """Its intra-op team is made by its first parallel op, and the team's
    threads inherit its CPUs then."""
    calls = []
    monkeypatch.setattr(data_worker_mod.os, "sched_setaffinity", lambda pid, cpus: calls.append(("pin", pid, cpus)))
    monkeypatch.setattr(data_worker_mod, "warm_up_model", lambda model: calls.append(("warm",)))
    thread = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    thread.cpus = [38, 39]
    thread.model = None
    thread.stop_event = threading.Event()
    thread.stop_event.set()
    thread._idle_wait = lambda: None
    thread.in_flight_requests = set()
    thread.run()
    assert calls[0] == ("pin", 0, [38, 39])
