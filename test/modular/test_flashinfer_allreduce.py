"""``mstar.distributed.flashinfer_allreduce``: NCCL/FlashInfer backend switch.

CPU-only: ``flashinfer.comm`` imports fine without a GPU (verified), but its
workspace creation and kernels do not run here, so every test that exercises
the FlashInfer path mocks ``flashinfer.comm``'s entry points and
``torch.cuda.is_current_stream_capturing`` (which itself requires a CUDA
device to call for real).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch
import torch.distributed as dist

from mstar.distributed import flashinfer_allreduce as fia
from mstar.distributed.communication import CommGroup


class _FakeWorkspace:
    def __init__(self, sufficient: bool = True):
        self.sufficient = sufficient

    def is_buffer_size_sufficient(self, tp_size, num_tokens, hidden_dim, dtype):
        return self.sufficient


def _group(world_size: int) -> CommGroup:
    group = CommGroup(
        my_global_rank=0, my_group_rank=0, group_members=list(range(world_size)),
    )
    if world_size > 1:
        # What ``ParallelConfig.init_dist`` does once the process group exists.
        group.device_group = SimpleNamespace(group_name=f"fake_pg_{world_size}")
        group.allreduce_handle = fia.register(group)
    return group


def _enable(monkeypatch, capturing: bool = False):
    monkeypatch.setattr(fia, "_ENABLED", True)
    # The real call needs a CUDA device; this test host has none.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)


def _mock_flashinfer(monkeypatch, workspace=None):
    """Replace FlashInfer's entry points; returns the recorded calls."""
    from flashinfer import comm

    calls = {"create": [], "fusion": []}

    def create(**kwargs):
        calls["create"].append(kwargs)
        return workspace or _FakeWorkspace()

    def fusion(input_, ws, pattern, **kwargs):
        calls["fusion"].append((input_, ws, pattern))
        return input_ * 2

    monkeypatch.setattr(comm, "create_allreduce_fusion_workspace", create)
    monkeypatch.setattr(comm, "allreduce_fusion", fusion)
    return calls


def _mock_nccl(monkeypatch):
    calls = []
    monkeypatch.setattr(dist, "all_reduce", lambda t, group=None: calls.append(t))
    return calls


def setup_function(_fn):
    fia._workspaces.clear()
    fia._backend_logged = False
    fia._capture_warned = False
    fia._flashinfer_available = None


def test_default_env_uses_nccl_in_place(monkeypatch):
    monkeypatch.setattr(fia, "_ENABLED", False)
    nccl = _mock_nccl(monkeypatch)

    group = _group(2)
    x = torch.randn(4, 8)
    out = group.all_reduce(x)

    assert out is x
    assert len(nccl) == 1
    assert fia._workspaces == {}


def test_register_keys_the_group_by_process_group_name():
    group = _group(2)
    assert group.allreduce_handle == "fake_pg_2"
    assert fia._groups["fake_pg_2"] is group


def test_flashinfer_path_creates_workspace_then_reduces(monkeypatch):
    _enable(monkeypatch)
    nccl = _mock_nccl(monkeypatch)
    fi = _mock_flashinfer(monkeypatch)

    group = _group(4)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    out = group.all_reduce(x)

    assert not nccl
    assert len(fi["create"]) == 1
    create = fi["create"][0]
    assert create["backend"] == "trtllm"
    assert create["world_size"] == 4 and create["rank"] == 0
    assert create["hidden_dim"] == 8 and create["dtype"] == torch.bfloat16
    assert create["max_token_num"] == fia._MAX_TOKENS
    assert len(fi["fusion"]) == 1
    assert fi["fusion"][0][0].shape == (6, 8)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x * 2)
    assert fia._workspaces[group.allreduce_handle] is fi["fusion"][0][1]

    # Second call reuses the workspace.
    group.all_reduce(x)
    assert len(fi["create"]) == 1 and len(fi["fusion"]) == 2


def test_capture_without_workspace_falls_back_to_nccl(monkeypatch):
    _enable(monkeypatch, capturing=True)
    nccl = _mock_nccl(monkeypatch)
    fi = _mock_flashinfer(monkeypatch)

    group = _group(2)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    out = group.all_reduce(x)

    assert not fi["create"] and not fi["fusion"]
    assert fia._workspaces == {}
    # Out of place: the op may not return an alias of its input.
    assert out is not x
    assert len(nccl) == 1 and nccl[0] is not x
    torch.testing.assert_close(out, x)


def test_insufficient_workspace_falls_back_to_nccl(monkeypatch):
    _enable(monkeypatch)
    nccl = _mock_nccl(monkeypatch)
    fi = _mock_flashinfer(monkeypatch, workspace=_FakeWorkspace(sufficient=False))

    group = _group(2)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    group.all_reduce(x)

    assert len(fi["create"]) == 1
    assert not fi["fusion"]
    assert len(nccl) == 1


def test_token_cap_skips_the_op(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(fia, "_MAX_TOKENS", 4)
    nccl = _mock_nccl(monkeypatch)
    fi = _mock_flashinfer(monkeypatch)

    group = _group(2)
    x = torch.randn(5, 8, dtype=torch.bfloat16)
    out = group.all_reduce(x)

    assert out is x
    assert len(nccl) == 1 and nccl[0] is x
    assert not fi["create"] and not fi["fusion"]


def test_unsupported_dtype_uses_nccl_in_place(monkeypatch):
    _enable(monkeypatch)
    nccl = _mock_nccl(monkeypatch)
    fi = _mock_flashinfer(monkeypatch)

    group = _group(2)
    x = torch.randn(4, 8, dtype=torch.float32)
    out = group.all_reduce(x)

    assert out is x
    assert len(nccl) == 1
    assert not fi["create"]


def test_world_size_one_is_identity_no_calls(monkeypatch):
    _enable(monkeypatch)
    nccl = _mock_nccl(monkeypatch)

    group = _group(1)
    x = torch.randn(1, 8, dtype=torch.bfloat16)
    out = group.all_reduce(x)

    assert out is x
    assert not nccl
    assert fia._workspaces == {}


def test_custom_op_fake_matches_real_shape(monkeypatch):
    """``torch.library.opcheck`` exercises the schema, the fake kernel, and
    AOTAutograd dispatch consistency between them."""
    _enable(monkeypatch)
    _mock_nccl(monkeypatch)
    _mock_flashinfer(monkeypatch)

    group = _group(2)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    torch.library.opcheck(
        torch.ops.mstar.flashinfer_all_reduce, (x, group.allreduce_handle),
    )
