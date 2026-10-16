"""The conductor builds the fused-MoE align op before spawning workers only for a
model that asks: every other model paid the nvcc build, or hung on its lock."""
import sys
import types
from types import SimpleNamespace

import pytest

from mstar.conductor.conductor import Conductor


@pytest.mark.parametrize("prebuild", [False, True])
def test_only_a_model_that_asks_builds_the_op(monkeypatch, prebuild):
    built = []
    align = types.ModuleType("mstar.utils.fused_moe.align")
    align._cuda_op_available = lambda: built.append(1)
    monkeypatch.setitem(sys.modules, "mstar.utils.fused_moe.align", align)
    monkeypatch.setattr(Conductor, "shutdown", lambda self: None)  # its atexit hook
    conductor = object.__new__(Conductor)
    conductor.model = SimpleNamespace(prebuild_fused_moe=prebuild) if prebuild else object()
    conductor._sorted_ranks, conductor.worker_ids = [], []
    conductor._launch_workers()
    assert built == ([1] if prebuild else [])
