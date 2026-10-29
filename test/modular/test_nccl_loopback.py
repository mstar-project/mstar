"""NCCL's and Gloo's sockets stay on loopback in a single-host deployment (#529).

``init_process_group`` and the per-group ``new_group`` calls open NCCL's
bootstrap and proxy listeners and Gloo's pair listeners separately from the
rendezvous store. Unset, NCCL picks a routable interface and Gloo binds the
address the hostname resolves to, and they stay open while the deployment
runs. Every worker runs on the conductor's host, so the conductor points them
at ``lo`` unless the user chose an interface (or, for NCCL, a family).
"""

import multiprocessing
import os
import socket
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor import conductor
from mstar.conductor.conductor import keep_bootstrap_sockets_on_loopback

_LOOPBACK = {"127.0.0.1", "::1"}


@pytest.fixture
def clean_env(monkeypatch):
    # setenv first so the undo restores the original value even after the
    # code under test writes os.environ itself
    for key in ("NCCL_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY", "GLOO_SOCKET_IFNAME"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


def test_unset_defaults_to_loopback(clean_env):
    keep_bootstrap_sockets_on_loopback()
    assert os.environ["NCCL_SOCKET_IFNAME"] == "lo"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "lo"


def test_a_user_set_interface_wins(clean_env, monkeypatch):
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth1")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth2")
    keep_bootstrap_sockets_on_loopback()
    assert os.environ["NCCL_SOCKET_IFNAME"] == "eth1"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "eth2"


def test_a_user_set_family_leaves_the_nccl_interface_alone(clean_env, monkeypatch):
    monkeypatch.setenv("NCCL_SOCKET_FAMILY", "AF_INET6")
    keep_bootstrap_sockets_on_loopback()
    assert "NCCL_SOCKET_IFNAME" not in os.environ
    assert os.environ["GLOO_SOCKET_IFNAME"] == "lo"


def test_the_conductor_sets_it_before_spawning_workers(clean_env, monkeypatch):
    """A spawned worker inherits the parent's environment at ``start``."""
    seen = []

    class Process:
        def __init__(self, **kwargs):
            pass

        def start(self):
            seen.append((os.environ.get("NCCL_SOCKET_IFNAME"), os.environ.get("GLOO_SOCKET_IFNAME")))

    monkeypatch.setattr(conductor, "mp", SimpleNamespace(get_context=lambda method: SimpleNamespace(Process=Process)))
    monkeypatch.setattr(conductor, "atexit", SimpleNamespace(register=lambda fn: None))
    workers = ["w0", "w1"]
    spawner = SimpleNamespace(
        _sorted_ranks=[0, 1], worker_ids=workers, _worker_processes=[],
        _per_worker_graphs=dict.fromkeys(workers), per_worker_sharding_config=dict.fromkeys(workers),
        parallel_config=SimpleNamespace(per_worker_config=dict.fromkeys(workers)),
        model_config={}, _all_worker_graph_ids_to_graph_walks={}, _all_worker_graph_ids_to_nodes={},
        _all_worker_graph_ids_to_dyn_loops={}, hostname="localhost", socket_path_prefix="/tmp/x",
        _dist_init_method="", model=None, enable_nvtx=False, enable_prof=False, device_type="cuda",
        rank_devices=None, log_level="INFO", tensor_comm_protocol=None, tcp_transfer_device="",
        _status_interval_s=None,
        _mps_requested=False, _mps=None,
        shutdown=None,
    )
    conductor.Conductor._launch_workers(spawner)
    assert seen == [("lo", "lo"), ("lo", "lo")]


def _decode(hex_addr: str, v6: bool) -> str:
    addr, port = hex_addr.split(":")
    raw = bytes.fromhex(addr)
    # /proc/net/tcp{,6} prints the address as native-endian 32-bit words
    raw = b"".join(raw[i:i + 4][::-1] for i in range(0, len(raw), 4))
    return f"{socket.inet_ntop(socket.AF_INET6 if v6 else socket.AF_INET, raw)}:{int(port, 16)}"


def _own_listeners() -> list[str]:
    """``addr:port`` of every TCP socket this process listens on."""
    inodes = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            link = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if link.startswith("socket:["):
            inodes.add(link[len("socket:["):-1])
    found = []
    for path, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        with open(path) as f:
            for line in f.read().splitlines()[1:]:
                cols = line.split()
                if cols[3] == "0A" and cols[9] in inodes:  # 0A: LISTEN
                    found.append(_decode(cols[1], v6))
    return found


def _rank(rank: int, store_port: int, backend: str, queue) -> None:
    import torch.distributed as dist

    kwargs = {}
    if backend == "nccl":
        torch.cuda.set_device(rank)
        kwargs["device_id"] = torch.device("cuda", rank)
    dist.init_process_group(
        backend, init_method=f"tcp://127.0.0.1:{store_port}", world_size=2, rank=rank, **kwargs,
    )
    device = "cuda" if backend == "nccl" else "cpu"
    x = torch.ones(1, device=device)
    dist.all_reduce(x)
    # the CPU side group each TP/SP group gets (mstar.distributed.communication)
    cpu_group = dist.new_group(ranks=[0, 1], backend="gloo", use_local_synchronization=True)
    y = torch.ones(1)
    dist.all_reduce(y, group=cpu_group)
    store = f":{store_port}"
    queue.put((rank, x.item() + y.item(), [a for a in _own_listeners() if not a.endswith(store)]))
    dist.barrier()
    dist.destroy_process_group()


def _run_two_ranks(backend: str) -> None:
    keep_bootstrap_sockets_on_loopback()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        store_port = s.getsockname()[1]
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_rank, args=(r, store_port, backend, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(queue.get(timeout=300) for _ in procs)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    for rank, total, listeners in results:
        assert total == 4.0
        assert listeners, f"rank {rank}: no listener found"
        hosts = {a.rsplit(":", 1)[0] for a in listeners}
        assert hosts <= _LOOPBACK, f"rank {rank} listens on {listeners}"


def test_gloo_listens_on_loopback_only(clean_env):
    _run_two_ranks("gloo")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 CUDA GPUs")
def test_nccl_bootstrap_listens_on_loopback_only(clean_env):
    _run_two_ranks("nccl")
