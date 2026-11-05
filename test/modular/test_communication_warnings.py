from mstar.distributed import communication as comm


def test_barrier_uses_explicit_device_for_nccl(monkeypatch):
    calls = {}
    group = object()

    monkeypatch.setattr(comm.dist, "get_backend", lambda _group=None: comm.dist.Backend.NCCL)
    monkeypatch.setattr(comm.torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(comm.dist, "barrier", lambda **kwargs: calls.update(kwargs))

    comm._barrier(group)

    assert calls == {"group": group, "device_ids": [7]}


def test_barrier_leaves_non_nccl_groups_alone(monkeypatch):
    calls = {}
    group = object()

    monkeypatch.setattr(comm.dist, "get_backend", lambda _group=None: comm.dist.Backend.GLOO)
    monkeypatch.setattr(comm.dist, "barrier", lambda **kwargs: calls.update(kwargs))

    comm._barrier(group)

    assert calls == {"group": group}
