"""``device_key``: an AMD GPU passed through SR-IOV reads the bare GPU's table."""

from types import SimpleNamespace

import pytest
import torch

from mstar.kernels import table


@pytest.mark.parametrize(("gpu", "rocm", "expected"), [
    ("AMD Instinct MI325X VF", True, "AMD Instinct MI325X"),
    ("AMD Instinct MI325X", True, "AMD Instinct MI325X"),
    ("AMD Instinct MI300X VF", True, "AMD Instinct MI300X"),
    # only AMD names drop the suffix
    ("Some GPU VF", False, "Some GPU VF"),
])
def test_a_virtual_function_reads_its_gpus_table(monkeypatch, gpu, rocm, expected):
    from mstar import platforms

    platform = SimpleNamespace(
        is_rocm=lambda: rocm, is_cuda=lambda: not rocm, arch=lambda device=None: "gfx942" if rocm else None,
    )
    monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: gpu)
    key = table.device_key(0)
    assert key.name == expected
    assert key.stems()[0] == table.file_stem(expected)
