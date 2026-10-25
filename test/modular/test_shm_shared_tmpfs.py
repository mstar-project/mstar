"""Per-uuid SHM tensor files when two deployments share a tmpfs.

Containers that share ``/dev/shm`` (``--ipc=host``, hostIPC) but not ``/tmp``
or the network namespace can run the same socket prefix and ports, so the
conductor endpoint alone does not tell their tensor files apart.
"""

import fcntl
import os
import tempfile

import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge

TOKEN_ENV = "MSTAR_DEPLOYMENT_TOKEN"


class _Communicator(BaseCommunicator):
    """Each container has its own /tmp, so both use the default prefix."""

    def __init__(self, prefix: str = "/tmp/mstar/"):
        self.protocol = CommProtocol.IPC
        self.ipc_socket_path_prefix = prefix

    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


def _manager(shm_dir, entity_id="worker_0", prefix="/tmp/mstar/"):
    mgr = SharedMemoryCommunicationManager(
        my_entity_id=entity_id, hostname="localhost", device="cpu",
        communicator=_Communicator(prefix), shm_dir=shm_dir,
    )
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
    cfg.setup({})
    mgr.register_request("req1", cfg)
    return mgr


def test_two_containers_sharing_a_tmpfs_do_not_swap_tensors(monkeypatch):
    """With the endpoint alone both containers derived one namespace, and the
    second one's write unlinked the first's live file and put its own bytes
    under the name. Each launch's token keeps them apart."""
    with tempfile.TemporaryDirectory() as shm:
        monkeypatch.setenv(TOKEN_ENV, "a" * 16)
        a = _manager(shm)
        consumer = _manager(shm, "worker_1")
        monkeypatch.setenv(TOKEN_ENV, "b" * 16)
        b = _manager(shm)

        [ia] = a.store_and_return_tensor_info(
            "req1", {"x": [torch.full((4,), 1.0)]})["x"]
        a.register_for_send("req1", [ia])
        [ib] = b.store_and_return_tensor_info(
            "req1", {"x": [torch.full((4,), 2.0)]})["x"]
        assert ib.uuid == ia.uuid, "precondition: the uuids really do collide"
        b.register_for_send("req1", [ib])

        edges = [GraphEdge(next_node="LLM", name="x", tensor_info=[ia])]
        consumer.start_read_tensors("req1", edges, graph_walk="decode")
        consumer.get_ready_tensors(graph_walk="decode")
        got = consumer.get_tensor(ia.uuid)
        assert torch.equal(got, torch.full((4,), 1.0)), (
            f"container A's consumer read container B's tensor: {got.tolist()}")


def test_a_spawned_child_inherits_the_launch_token(monkeypatch):
    """The API server mints the token; the processes it spawns derive the
    same namespace, and the next launch a different one."""
    import multiprocessing

    from mstar.communication.tensors import (
        _deployment_namespace,
        ensure_deployment_token,
    )

    monkeypatch.delenv(TOKEN_ENV, raising=False)
    comm = _Communicator()
    endpoint_only = _deployment_namespace(comm)
    monkeypatch.setenv(TOKEN_ENV, ensure_deployment_token())
    ours = _deployment_namespace(comm)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        assert pool.apply(_deployment_namespace, (comm,)) == ours
    assert ours != endpoint_only
    monkeypatch.delenv(TOKEN_ENV)
    ensure_deployment_token()
    assert _deployment_namespace(comm) not in (ours, endpoint_only)


def test_a_dead_deployments_files_are_swept_and_a_live_ones_kept(monkeypatch):
    """A per-launch namespace means a restart no longer collides with, and so
    no longer reclaims, what a crashed run left. A namespace whose lock no
    process holds is gone, and the next start deletes its files; one whose
    lock is held is left alone."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with tempfile.TemporaryDirectory() as shm:
        dead, live = "d" * 12, "e" * 12
        for ns in (dead, live):
            open(os.path.join(shm, f"mstar_{ns}.lock"), "wb").close()
            with open(os.path.join(shm, f"mstar_{ns}_worker_0_1"), "wb") as f:
                f.write(b"tensor")
        held = os.open(os.path.join(shm, f"mstar_{live}.lock"), os.O_RDONLY)
        try:
            fcntl.flock(held, fcntl.LOCK_SH)
            mgr = _manager(shm, prefix="/tmp/mstar_sweeper/")
            left = set(os.listdir(shm))
        finally:
            os.close(held)
        assert f"mstar_{dead}_worker_0_1" not in left
        assert f"mstar_{dead}.lock" not in left
        assert {f"mstar_{live}_worker_0_1", f"mstar_{live}.lock"} <= left
        assert f"mstar_{mgr._shm_namespace}.lock" in left
