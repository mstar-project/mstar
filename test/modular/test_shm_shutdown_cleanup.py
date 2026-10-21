"""A worker stopped with requests in flight unlinks the tensor files it owns.

A sent tensor's file stays in the shm dir until the consumer's TENSOR_RECEIVED
or the conductor's REMOVE_REQUEST. A server stopped with requests in flight
sends neither, so each worker's in-flight files stayed in ``/dev/shm`` (RAM)
until reboot. The worker processes here run the real
``_worker_process_target`` (its SIGTERM handling and orphan watchdog) around a
stand-in ``Worker``. Files are counted under the test deployment's namespace
only, as the e2e stress leak check does.
"""

import importlib.util
import multiprocessing as mp
import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    SharedMemoryCommunicationManager,
    _deployment_namespace,
    create_tensor_communication_manager,
)
from mstar.conductor.conductor import Conductor
from mstar.distributed.base import ShardingConfig

N_IN_FLIGHT = 9
ARENA_SPILLS = 3  # of 6 tensors of 300 KB, one 1 MiB segment holds three


class _DeploymentCommunicator(BaseCommunicator):
    """No mesh traffic; the protocol and prefix give the file namespace."""

    protocol = CommProtocol.IPC

    def __init__(self, prefix: str):
        self.ipc_socket_path_prefix = prefix

    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


def _namespace_files(shm_dir, prefix) -> list[str]:
    ns = _deployment_namespace(_DeploymentCommunicator(prefix))
    return sorted(n for n in os.listdir(shm_dir) if n.startswith(f"mstar_{ns}_"))


def _send_unacked(mgr, n: int, make=lambda i: torch.tensor([i])) -> list[str]:
    """Send one tensor per request and never ack it, as a worker mid-decode
    holds them; returns the files' paths."""
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
    cfg.setup({})
    paths = []
    for i in range(n):
        rid = f"req{i}"
        mgr.register_request(rid, cfg)
        [info] = mgr.store_and_return_tensor_info(rid, {"out": [make(i)]})["out"]
        mgr.increment_ref(info.uuid, 1)  # the consumer's pending TENSOR_RECEIVED
        mgr.register_for_send(rid, [info])
        paths.append(mgr._shm_path(mgr.my_entity_id, info.uuid))
    return paths


class _InFlightWorker:
    """Stands in for ``Worker``: a file-transport manager holding unacked
    per-request tensors, and a ``run()`` that serves until stopped."""

    shm_dir = arena = ready = None

    def __init__(self, *, worker_id, socket_path_prefix, tensor_comm_protocol, **_):
        cls = type(self)
        if cls.arena:
            os.environ.update(MSTAR_SHM_ARENA="1", MSTAR_SHM_ARENA_SEGMENT_MB="1",
                              MSTAR_SHM_ARENA_MAX_SEGMENTS="1", MSTAR_SHM_ARENA_SPILL_AFTER_S="0.01")
        self.tensor_manager = create_tensor_communication_manager(
            tensor_comm_protocol, my_entity_id=worker_id, hostname="localhost", device="cpu",
            communicator=_DeploymentCommunicator(socket_path_prefix), shm_dir=cls.shm_dir,
        )
        if cls.arena:
            _send_unacked(self.tensor_manager, 6, make=lambda i: torch.full((300_000,), i, dtype=torch.uint8))
        else:
            _send_unacked(self.tensor_manager, N_IN_FLIGHT)

    def run(self):
        type(self).ready.set()
        while True:
            time.sleep(0.05)


def _spawned_worker(shm_dir: str, prefix: str, arena: bool, ready) -> None:
    import mstar.worker.worker
    from mstar.conductor.conductor import _worker_process_target

    _InFlightWorker.shm_dir, _InFlightWorker.arena, _InFlightWorker.ready = shm_dir, arena, ready
    mstar.worker.worker.Worker = _InFlightWorker
    _worker_process_target(
        worker_id="worker_0", worker_ids=["worker_0"], my_worker_graphs=[], model_config={},
        all_worker_graph_ids_to_graph_walks={}, all_worker_graph_ids_to_nodes={},
        all_worker_graph_ids_to_dyn_loops={}, sharding_config=None, parallel_groups=None,
        hostname="localhost", socket_path_prefix=prefix, dist_init_method="", device="cpu",
        tensor_comm_protocol=CommProtocol.SHM,
    )


def _conductor_stand_in(shm_dir: str, prefix: str, ready, worker_pid) -> None:
    """The worker's parent, so the test can kill it out from under the worker."""
    p = mp.get_context("spawn").Process(target=_spawned_worker, args=(shm_dir, prefix, False, ready))
    p.start()
    worker_pid.put(p.pid)
    p.join()


def _dirs(tmp_path) -> tuple[str, str]:
    shm_dir, prefix = str(tmp_path / "shm"), str(tmp_path / "ipc")
    os.makedirs(shm_dir)
    return shm_dir, prefix


def _gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("arena", [
    False,
    pytest.param(True, id="arena-spill", marks=pytest.mark.skipif(
        importlib.util.find_spec("mstar_rust") is None, reason="needs mstar_rust")),
])
@pytest.mark.parametrize("how", ["shutdown", "ctrl-c"])
def test_a_worker_stopped_with_requests_in_flight_leaves_no_tensor_files(tmp_path, how, arena):
    shm_dir, prefix = _dirs(tmp_path)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_spawned_worker, args=(shm_dir, prefix, arena, ready))
    p.start()
    try:
        assert ready.wait(120), "worker did not start"
        held = _namespace_files(shm_dir, prefix)
        assert len(held) == (ARENA_SPILLS if arena else N_IN_FLIGHT), held
        if how == "ctrl-c":
            # The whole process group gets the SIGINT, so the conductor's
            # SIGTERM follows at once.
            os.kill(p.pid, signal.SIGINT)
    finally:
        Conductor.shutdown(SimpleNamespace(_worker_processes=[p]))
    assert _namespace_files(shm_dir, prefix) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_an_orphaned_worker_leaves_its_files_to_the_names_next_owner(tmp_path):
    """With its conductor gone the deployment's lock and ports are free, and a
    new deployment on the same prefix may already be writing these names."""
    shm_dir, prefix = _dirs(tmp_path)
    ctx = mp.get_context("spawn")
    ready, worker_pid = ctx.Event(), ctx.Queue()
    conductor = ctx.Process(target=_conductor_stand_in, args=(shm_dir, prefix, ready, worker_pid))
    conductor.start()
    try:
        assert ready.wait(120), "worker did not start"
        pid = worker_pid.get(timeout=10)
        held = _namespace_files(shm_dir, prefix)
        assert len(held) == N_IN_FLIGHT
    finally:
        conductor.kill()
        conductor.join(timeout=10)
    deadline = time.monotonic() + 30
    while not _gone(pid):  # the worker's watchdog stops it once the conductor is gone
        assert time.monotonic() < deadline, "orphaned worker did not exit"
        time.sleep(0.1)
    assert _namespace_files(shm_dir, prefix) == held


def test_unlink_owned_files_skips_what_is_gone_and_leaves_peers_alone(tmp_path):
    shm_dir, prefix = _dirs(tmp_path)

    def manager(entity_id):
        return SharedMemoryCommunicationManager(
            my_entity_id=entity_id, hostname="localhost", device="cpu",
            communicator=_DeploymentCommunicator(prefix), shm_dir=shm_dir,
        )

    mine, peer = manager("worker_0"), manager("worker_1")
    already_gone, *_ = _send_unacked(mine, 3)
    peer_files = sorted(os.path.basename(p) for p in _send_unacked(peer, 2))
    os.unlink(already_gone)

    mine.unlink_owned_files()
    mine.unlink_owned_files()
    assert _namespace_files(shm_dir, prefix) == peer_files
