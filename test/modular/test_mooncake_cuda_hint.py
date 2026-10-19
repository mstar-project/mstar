"""The TCP/RDMA startup error names the Mooncake wheel that loads next to this torch.

Mooncake's default wheel is built for CUDA 12. On a CUDA 13 torch its import fails
with the dlopen error below, and the old advice ("Install mooncake-transfer-engine")
pointed at the wheel that was already installed.
"""

import pytest
import torch

from mstar.communication import tensors
from mstar.communication.communicator import CommProtocol

_NO_CUDART_12 = ImportError("libcudart.so.12: cannot open shared object file: No such file or directory")
_NO_CUDART_13 = ImportError("libcudart.so.13: cannot open shared object file: No such file or directory")


def _not_installed():
    return ModuleNotFoundError("No module named 'mooncake'", name="mooncake")


def _start_error(monkeypatch, import_error, torch_cuda) -> str:
    monkeypatch.setattr(tensors, "TransferEngine", None)
    monkeypatch.setattr(tensors, "MOONCAKE_IMPORT_ERROR", import_error)
    monkeypatch.setattr(torch.version, "cuda", torch_cuda)
    with pytest.raises(RuntimeError) as exc:
        tensors.MooncakeTransferEngine(hostname="localhost", protocol=CommProtocol.TCP)
    message = str(exc.value)
    assert f"Failed to load mooncake: {type(import_error).__name__}: {import_error}." in message
    assert message.endswith("or use SHM protocol.")
    return message


@pytest.mark.parametrize("torch_cuda", ["13.0", "13.2"])
def test_cuda12_wheel_on_cuda13_torch_names_the_cuda13_wheel(monkeypatch, torch_cuda):
    message = _start_error(monkeypatch, _NO_CUDART_12, torch_cuda)
    assert f"needs the CUDA 12 runtime, but torch is built for CUDA {torch_cuda}" in message
    assert (
        "`pip uninstall -y mooncake-transfer-engine && "
        "pip install --force-reinstall --no-deps mooncake-transfer-engine-cuda13`"
    ) in message
    assert "nvidia-cuda-runtime" not in message


def test_cuda13_wheel_on_cuda12_torch_names_the_default_wheel(monkeypatch):
    message = _start_error(monkeypatch, _NO_CUDART_13, "12.8")
    assert "needs the CUDA 13 runtime, but torch is built for CUDA 12.8" in message
    assert (
        "`pip uninstall -y mooncake-transfer-engine-cuda13 && "
        "pip install --force-reinstall --no-deps mooncake-transfer-engine`"
    ) in message


def test_cuda_wheel_on_cpu_torch_names_the_non_cuda_wheel(monkeypatch):
    message = _start_error(monkeypatch, _NO_CUDART_12, None)
    assert "but torch is built without CUDA" in message
    assert "pip install --force-reinstall --no-deps mooncake-transfer-engine-non-cuda`" in message


def test_missing_runtime_for_the_same_cuda_major_names_the_runtime(monkeypatch):
    message = _start_error(monkeypatch, _NO_CUDART_12, "12.8")
    assert "`pip install nvidia-cuda-runtime-cu12`" in message
    assert "pip uninstall" not in message


@pytest.mark.parametrize(
    ("torch_cuda", "wheel"),
    [
        ("13.0", "mooncake-transfer-engine-cuda13"),
        ("12.8", "mooncake-transfer-engine"),
        (None, "mooncake-transfer-engine-non-cuda"),
    ],
)
def test_not_installed_names_the_wheel_for_this_torch(monkeypatch, torch_cuda, wheel):
    message = _start_error(monkeypatch, _not_installed(), torch_cuda)
    assert message.endswith(f"Install {wheel} or use SHM protocol.")


@pytest.mark.parametrize(
    "import_error",
    [
        ImportError("libibverbs.so.1: cannot open shared object file: No such file or directory"),
        ModuleNotFoundError("No module named 'aiohttp'", name="aiohttp"),
    ],
)
def test_other_import_failures_keep_the_generic_advice(monkeypatch, import_error):
    message = _start_error(monkeypatch, import_error, "13.0")
    assert message.endswith("Install mooncake-transfer-engine or use SHM protocol.")
