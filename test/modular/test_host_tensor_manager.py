"""A tensor manager on the host never creates a CUDA context.

The API server stores each request's inputs through a manager built on ``device="cpu"`` and
registers them for send, and both calls synchronized the default stream whenever CUDA was
available. On SHM nothing else in the API server touches the GPU, so that sync created the
process's context, about 600 MiB, at the first request, after the workers had filled the GPU:
on Qwen3-Omni colocated every request failed ``preprocessing failed: CUDA error: out of memory``.
``torch.cuda.is_initialized()`` cannot tell, since importing flashinfer initializes torch's CUDA
state without a context, so the context tests run in a fresh interpreter and ask the driver; the
rest check on CPU which managers sync at all.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
    TensorCommunicationManager,
)

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="asks the driver for a context")
_HAS_ARENA = importlib.util.find_spec("mstar_rust") is not None

_PROBE = """
import ctypes, sys, tempfile
import torch
from mstar.communication.communicator import BaseCommunicator

class _StubCommunicator(BaseCommunicator):
    def send(self, entity_id, msg):
        pass
    def get_all_new_messages(self):
        return []

def primary_context_active():
    cuda = ctypes.CDLL("libcuda.so.1")
    dev, flags, active = ctypes.c_int(), ctypes.c_uint(), ctypes.c_int()
    assert cuda.cuInit(0) == 0 and cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    assert cuda.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active)) == 0
    return bool(active.value)

# the state importing flashinfer leaves: torch's CUDA initialized, no context yet
torch.cuda.init()
before = torch.cuda.is_initialized() and not primary_context_active()
module, cls = sys.argv[1].rsplit(".", 1)
manager = getattr(__import__(module, fromlist=[cls]), cls)
with tempfile.TemporaryDirectory() as shm_dir:
    mgr = manager(
        my_entity_id="api_server_preprocess_worker", hostname="localhost", device="cpu",
        communicator=_StubCommunicator(), shm_dir=shm_dir,
    )
    infos = mgr.store_and_return_tensor_info("req1", {"text_inputs": [torch.arange(8)]})
    mgr.register_for_send("req1", sum(infos.values(), start=[]))
print(before, primary_context_active())
"""


def _context_before_and_after(manager: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, manager], capture_output=True, text=True, timeout=300, check=False,
        # the tree this file sits in, not whichever mstar the cwd or the venv would import
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()[-2:]


@requires_cuda
@pytest.mark.parametrize("manager", [
    "mstar.communication.tensors.SharedMemoryCommunicationManager",
    pytest.param(
        "mstar.communication.arena.ArenaShmCommunicationManager",
        marks=pytest.mark.skipif(not _HAS_ARENA, reason="the arena needs the mstar_rust extension"),
    ),
])
def test_host_manager_stores_and_registers_without_a_cuda_context(manager):
    before, after = _context_before_and_after(manager)
    assert before == "True", "the probe must start with torch's CUDA initialized and no context"
    assert after == "False", "the API server would create a CUDA context on a GPU the workers filled"


class _StubCommunicator(BaseCommunicator):
    """Drops every message."""

    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


class _StubTransferEngine:
    """Registers memory without Mooncake."""

    def register_memory(self, address, nbytes):
        return 0


class _StubStream:
    """Records syncs instead of touching CUDA."""

    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


def _manager(cls, device: str):
    mgr = object.__new__(cls)
    TensorCommunicationManager.__init__(
        mgr, my_entity_id="api_server_preprocess_worker", my_session_id="local", device=device,
        communicator=_StubCommunicator(), transfer_engine=_StubTransferEngine(),
    )
    return mgr


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_a_device_manager_syncs_the_gpu(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mgr = _manager(SharedMemoryCommunicationManager, device)
    assert mgr._on_cuda, f"a {device} manager would send tensors its GPU has not finished writing"


def test_a_host_manager_never_syncs_the_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mgr = _manager(SharedMemoryCommunicationManager, "cpu")
    assert not mgr._on_cuda, "a host manager would create a CUDA context at its first sync"


def test_host_mooncake_manager_registers_without_a_sync(monkeypatch):
    mgr = _manager(MooncakeCommunicationManager, "cpu")
    mgr.protocol = CommProtocol.TCP
    infos = mgr.store_and_return_tensor_info("req1", {"text_inputs": [torch.arange(8)]})
    stream = _StubStream()
    monkeypatch.setattr(torch.cuda, "default_stream", lambda *args, **kwargs: stream)
    mgr.register_for_send("req1", infos["text_inputs"])
    assert stream.syncs == 0, "an RDMA API server would create a CUDA context at its first send"
