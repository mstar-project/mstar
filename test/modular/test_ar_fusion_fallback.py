"""create_workspace falls back from FlashInfer's "auto" backend to "trtllm",
and every TP rank ends up with the same backend (or none), over a real gloo
group with a stubbed ``flashinfer.comm``."""
import os
import sys
import types

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2


class _FakeWorkspace:
    def __init__(self, backend):
        self.backend = backend
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def _install_fake_flashinfer(fails: dict[str, set[int]], rank: int, built: list):
    def create(*, backend, **_):
        if rank in fails.get(backend, set()):
            raise RuntimeError(f"{backend} unavailable on rank {rank}")
        ws = _FakeWorkspace(backend)
        built.append(ws)
        return ws

    comm = types.ModuleType("flashinfer.comm")
    comm.create_allreduce_fusion_workspace = create
    comm_backend = types.ModuleType("flashinfer.comm.comm_backend")
    comm_backend.TorchDistBackend = lambda group: None
    pkg = types.ModuleType("flashinfer")
    pkg.comm = comm
    sys.modules.update({"flashinfer": pkg, "flashinfer.comm": comm,
                        "flashinfer.comm.comm_backend": comm_backend})


def _worker(rank, store_path, fails, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{store_path}",
                            rank=rank, world_size=WORLD)
    built: list = []
    _install_fake_flashinfer(fails, rank, built)
    from mstar.distributed import ar_fusion
    from mstar.distributed.communication import CommGroup

    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(WORLD)))
    group.cpu_group = dist.group.WORLD
    ws_id = ar_fusion.create_workspace(group, 128, 64, None)
    chosen = None if ws_id is None else ar_fusion._WORKSPACES[ws_id].backend
    leaked = [ws.backend for ws in built if not ws.destroyed and ws.backend != chosen]
    with open(os.path.join(out_dir, str(rank)), "w") as f:
        f.write(f"{chosen}|{','.join(leaked)}")
    dist.destroy_process_group()


@pytest.mark.parametrize("fails, expected", [
    ({}, "auto"),                                   # auto works everywhere
    ({"auto": {0, 1}}, "trtllm"),                   # multicast fails on every rank
    ({"auto": {1}}, "trtllm"),                      # one rank fails: all must move on
    ({"auto": {0, 1}, "trtllm": {0}}, None),        # nothing works on rank 0
])
def test_every_rank_agrees_on_backend(tmp_path, fails, expected):
    mp.spawn(_worker, args=(str(tmp_path / "store"), fails, str(tmp_path)),
             nprocs=WORLD, join=True)
    for rank in range(WORLD):
        chosen, leaked = (tmp_path / str(rank)).read_text().split("|")
        assert chosen == str(expected), f"rank {rank} chose {chosen}"
        # a workspace built on some ranks but not all is destroyed, not kept
        assert leaked == "", f"rank {rank} kept {leaked}"


def test_a_second_width_on_a_shared_group_keeps_the_nccl_path(monkeypatch):
    """Two lockstep nodes of one instance share its group. The workspace is
    built for the first width and dtype set up; an input of another width or
    dtype takes NCCL rather than overrunning it."""
    import torch

    from mstar.distributed import ar_fusion
    from mstar.distributed.communication import CommGroup

    used = []
    monkeypatch.setattr(ar_fusion, "create_workspace", lambda group, max_tokens, hidden, dtype: 0)
    monkeypatch.setattr(
        ar_fusion, "allreduce_add_rmsnorm",
        lambda x, residual, weight, eps, bias, ws_id: used.append(x.shape[-1]) or (x, residual),
    )
    group = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    monkeypatch.setattr(group, "all_reduce", lambda x: used.append("nccl") or x)
    assert group.init_allreduce_fusion(16, torch.float32)
    assert not group.init_allreduce_fusion(8, torch.float32)
    assert not group.init_allreduce_fusion(16, torch.bfloat16)

    for hidden, dtype in ((16, torch.float32), (8, torch.float32), (16, torch.bfloat16)):
        x = torch.ones(4, hidden, dtype=dtype)
        group.allreduce_add_rmsnorm(x, torch.zeros_like(x), torch.ones(hidden, dtype=dtype), 1e-6)

    assert used == [16, "nccl", "nccl"]
