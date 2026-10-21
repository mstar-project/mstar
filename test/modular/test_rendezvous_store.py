"""The TP/SP rendezvous store listens on the init_method host only."""
import multiprocessing as mp
import socket
import warnings
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout

from mstar.distributed import communication
from mstar.distributed.communication import WorkerParallelGroups, default_group_timeout, rendezvous_store


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _non_loopback_ipv4() -> str | None:
    """This host's outbound IPv4 address; a UDP connect sends nothing."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))
        except OSError:
            return None
        ip = s.getsockname()[0]
    return None if ip.startswith("127.") or ip == "0.0.0.0" else ip


def _allow_cpu_device_id() -> None:
    """``init_dist`` passes ``device_id``, which torch accepts only for accelerators."""
    real = dist.init_process_group

    def init_process_group(*args, device_id=None, **kwargs):
        if device_id is not None and device_id.type != "cpu":
            kwargs["device_id"] = device_id
        return real(*args, **kwargs)

    dist.init_process_group = init_process_group


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def test_store_is_not_reachable_on_other_interfaces(monkeypatch):
    monkeypatch.setattr(dist, "init_process_group", dist.init_process_group)
    _allow_cpu_device_id()
    ip = _non_loopback_ipv4()
    if ip is None:
        pytest.skip("no non-loopback IPv4 address on this host")
    port = _free_port()
    groups = WorkerParallelGroups(num_workers=1, global_rank=0, any_parallelism=True)
    groups.init_dist(init_method=f"tcp://localhost:{port}", device=torch.device("cpu"))
    try:
        assert _reachable("127.0.0.1", port)
        assert not _reachable(ip, port), f"the rendezvous store answers on {ip}:{port}"
        # gloo's group default, which init_method also gave the store
        assert dist.distributed_c10d._get_default_store().timeout == default_pg_timeout
    finally:
        dist.destroy_process_group()


class _StopInitError(Exception):
    pass


def test_store_and_group_get_the_nccl_timeout(monkeypatch):
    """init_method used to set the store to the group's timeout (10 min for NCCL); store= does not."""
    nccl = timedelta(minutes=10)
    monkeypatch.setattr(communication, "default_pg_nccl_timeout", nccl)  # None on CPU-only builds
    monkeypatch.setattr(dist, "get_default_backend_for_device", lambda device_type: "nccl")
    seen = {}

    def init_process_group(**kwargs):
        seen.update(kwargs)
        raise _StopInitError

    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    groups = WorkerParallelGroups(num_workers=1, global_rank=0, any_parallelism=True)
    with pytest.raises(_StopInitError):
        groups.init_dist(init_method=f"tcp://localhost:{_free_port()}", device=torch.device("cpu"))
    assert seen["backend"] == "nccl"
    assert seen["timeout"] == nccl
    assert seen["store"].timeout == nccl


@pytest.mark.parametrize("backend", ["nccl", "gloo"])
def test_default_timeout_matches_torch(backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # CPU-only builds warn that NCCL is not compiled
        expected = dist.distributed_c10d._get_default_timeout(dist.Backend(backend))
    assert default_group_timeout(backend) == expected


def _rank(rank: int, port: int, out) -> None:
    _allow_cpu_device_id()
    groups = WorkerParallelGroups(num_workers=2, global_rank=rank, any_parallelism=True)
    groups.init_dist(init_method=f"tcp://localhost:{port}", device=torch.device("cpu"))
    value = torch.tensor([rank + 1.0])
    dist.all_reduce(value)
    out.put((rank, value.item()))
    dist.destroy_process_group()


def test_two_ranks_meet_through_the_store():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_rank, args=(rank, port, out)) for rank in (1, 0)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    assert sorted(out.get(timeout=5) for _ in procs) == [(0, 3.0), (1, 3.0)]


@pytest.mark.parametrize("init_method", ["env://", "tcp://localhost", "file:///tmp/x"])
def test_only_tcp_host_port_is_accepted(init_method):
    with pytest.raises(ValueError, match="tcp://host:port"):
        rendezvous_store(init_method, 0, 1, default_pg_timeout)
