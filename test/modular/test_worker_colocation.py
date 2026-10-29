"""``mps: true``: the platform says how workers sharing a GPU get their kernels
to overlap (``Platform.colocated_workers``), and the conductor runs that."""
import logging
import os
from types import SimpleNamespace

import pytest

from mstar.conductor.mps import MpsDaemon
from mstar.platforms import (
    CpuPlatform,
    CudaPlatform,
    HardwareQueueColocation,
    RocmPlatform,
    WorkerColocation,
    XpuPlatform,
)


def test_nvidia_gets_an_mps_daemon_of_the_conductors_own():
    colocation = CudaPlatform().colocated_workers()
    assert isinstance(colocation, MpsDaemon) and isinstance(colocation, WorkerColocation)
    assert set(colocation.client_env) == {"CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"}
    colocation.stop()  # never started: nothing to quit, its directory goes


def test_rocm_starts_nothing_and_says_so(caplog):
    colocation = RocmPlatform().colocated_workers()
    assert isinstance(colocation, HardwareQueueColocation)
    with caplog.at_level(logging.INFO, logger="mstar.platforms.interface"):
        colocation.start()
    assert "hardware scheduler" in caplog.text
    assert colocation.client_env == {}
    colocation.stop()


@pytest.mark.parametrize("platform", [XpuPlatform(), CpuPlatform()])
def test_platforms_without_a_way_refuse(platform):
    with pytest.raises(ValueError, match="mps: true"):
        platform.colocated_workers()


def _spawner(mps: bool):
    workers = ["w0", "w1"]
    return SimpleNamespace(
        _sorted_ranks=[0, 1], worker_ids=workers, _worker_processes=[],
        _per_worker_graphs=dict.fromkeys(workers), per_worker_sharding_config=dict.fromkeys(workers),
        parallel_config=SimpleNamespace(per_worker_config=dict.fromkeys(workers)),
        model_config={}, _all_worker_graph_ids_to_graph_walks={}, _all_worker_graph_ids_to_nodes={},
        _all_worker_graph_ids_to_dyn_loops={}, hostname="host", socket_path_prefix="/tmp/x",
        _dist_init_method="", model=None, enable_nvtx=False, enable_prof=False, device_type="cuda",
        rank_devices={1: 0}, log_level="INFO", tensor_comm_protocol=None, tcp_transfer_device="",
        _status_interval_s=None, _mps_requested=mps, _colocation=None, shutdown=None,
    )


def test_the_conductor_runs_the_platforms_colocation(monkeypatch):
    from mstar.conductor import conductor

    envs, started = [], []

    class Process:
        def __init__(self, **kwargs):
            envs.append(kwargs["kwargs"]["env"])

        def start(self):
            pass

    class Colocation(WorkerColocation):
        client_env = {"COLOCATION": "1"}

        def start(self):
            started.append("start")

        def stop(self):
            started.append("stop")

    monkeypatch.setattr(conductor, "mp", SimpleNamespace(get_context=lambda method: SimpleNamespace(Process=Process)))
    monkeypatch.setattr(conductor, "atexit", SimpleNamespace(register=lambda fn: None))
    monkeypatch.setattr(conductor, "current_platform", SimpleNamespace(colocated_workers=Colocation))
    spawner = _spawner(mps=True)
    conductor.Conductor._launch_workers(spawner)
    assert envs == [{"COLOCATION": "1"}] * 2 and started == ["start"]
    # shutdown stops it after the workers
    spawner._stop_workers = lambda: started.append("workers stopped")
    conductor.Conductor.shutdown(spawner)
    assert started == ["start", "workers stopped", "stop"]

    # without mps: true the platform is not asked and the workers get no env
    envs.clear()
    monkeypatch.setattr(conductor, "current_platform", SimpleNamespace())
    conductor.Conductor._launch_workers(_spawner(mps=False))
    assert envs == [None, None]


def test_a_platform_without_a_way_fails_the_launch(monkeypatch):
    from mstar.conductor import conductor

    monkeypatch.setattr(conductor, "mp", SimpleNamespace(get_context=lambda method: SimpleNamespace()))
    monkeypatch.setattr(conductor, "atexit", SimpleNamespace(register=lambda fn: None))
    monkeypatch.setattr(conductor, "current_platform", CpuPlatform())
    with pytest.raises(ValueError, match="mps: true"):
        conductor.Conductor._launch_workers(_spawner(mps=True))
    assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
