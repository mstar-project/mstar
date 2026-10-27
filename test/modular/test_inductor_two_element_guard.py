"""The sm_103 guard that keeps Inductor's pointwise kernels off launch configs
giving each thread two elements, which ptxas before 13.1.115 can miscompile.

The CUDA test compiles the fused interleaved mRoPE table at a size whose only
pointwise config holds two elements per thread; on sm_103 with Triton 3.7.1 it
comes out wrong without the guard.
"""

import pickle
from types import SimpleNamespace

import pytest
import torch

from mstar import platforms
from mstar.engine import accelerator_graph_runner as agr
from mstar.platforms import two_element_guard
from mstar.platforms.interface import CudaPlatform


def _config(num_warps: int, **blocks) -> SimpleNamespace:
    return SimpleNamespace(kwargs=dict(blocks), num_warps=num_warps)


@pytest.mark.parametrize(
    ("blocks", "warps", "want_warps"),
    [
        ({"XBLOCK": 64}, 1, 2),
        ({"XBLOCK": 128}, 2, 4),
        ({"XBLOCK": 256}, 4, 8),
        ({"XBLOCK": 512}, 8, 4),
        ({"XBLOCK": 1024}, 16, 8),
        ({"YBLOCK": 16, "XBLOCK": 16}, 4, 8),
        ({"XBLOCK": 128}, 4, 4),  # one element per thread
        ({"XBLOCK": 512}, 4, 4),  # four
    ],
)
def test_only_two_elements_per_thread_is_rewritten(blocks, warps, want_warps):
    config = _config(warps, **blocks)
    rewritten = two_element_guard.without_two_per_thread(config)
    assert rewritten.num_warps == want_warps
    assert rewritten.kwargs == blocks
    assert two_element_guard.elems_per_thread(rewritten) != 2
    assert config.num_warps == warps


class _Blackwell(CudaPlatform):
    def __init__(self, arch: str):
        self._arch = arch

    def arch(self, device=None) -> str:
        return self._arch


@pytest.mark.parametrize(
    ("arch", "affected", "device", "env", "want"),
    [
        ("sm_103", True, "cuda", None, True),
        ("sm_103", False, "cuda", None, False),
        ("sm_100", True, "cuda", None, False),
        ("sm_90", True, "cuda", None, False),
        ("sm_103", True, "cpu", None, False),
        ("sm_103", True, "cuda", "0", False),
        ("sm_90", False, "cuda", "1", True),
        ("sm_103", True, "cuda", "yes", True),
    ],
)
def test_on_only_for_sm_103_with_an_affected_ptxas_unless_forced(monkeypatch, arch, affected, device, env, want):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Blackwell(arch))
    monkeypatch.setattr(two_element_guard, "ptxas_affected", lambda: affected)
    if env is None:
        monkeypatch.delenv("MSTAR_INDUCTOR_TWO_ELEMENT_GUARD", raising=False)
    else:
        monkeypatch.setenv("MSTAR_INDUCTOR_TWO_ELEMENT_GUARD", env)
    assert agr.two_element_guard_enabled(torch.device(device)) is want


@pytest.mark.parametrize(
    ("version", "affected"),
    [((13, 1, 80), True), ((13, 1, 115), False), ((13, 3, 1), False), (None, True)],
)
def test_ptxas_before_13_1_115_is_affected(monkeypatch, version, affected):
    monkeypatch.setattr(two_element_guard, "blackwell_ptxas_version", lambda: version)
    assert two_element_guard.ptxas_affected() is affected


def test_the_worker_task_pickles_by_reference():
    # Inductor's compile workers unpickle the task, which imports this module there
    task = two_element_guard._worker_compile_triton
    assert pickle.loads(pickle.dumps(task)) is task


@pytest.fixture
def guard_installed(monkeypatch):
    """Install the guard for one test and put Inductor back afterwards."""
    pytest.importorskip("triton")
    from torch._inductor import async_compile
    from torch._inductor.runtime import coordinate_descent_tuner, triton_heuristics
    from torch.compiler import config as compiler_config

    for owner, name in [
        (triton_heuristics, "cached_autotune"),
        (triton_heuristics, "check_autotune_cache"),
        (coordinate_descent_tuner.CoordescTuner, "is_valid_config"),
        (async_compile, "_worker_compile_triton"),
        (compiler_config, "cache_key_tag"),
        (two_element_guard, "_installed"),
    ]:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    two_element_guard.install()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Inductor's Triton backend)")
@torch.no_grad()
def test_compiled_interleaved_mrope_matches_eager(guard_installed, monkeypatch):
    from torch._inductor.utils import fresh_cache

    from mstar.model.cosmos3.components.transformer import Cosmos3RotaryEmbedding

    rot = Cosmos3RotaryEmbedding(128, 1e8, (24, 20, 20))
    n = 200  # 200 x 64 elements: one pointwise config, 256 elements over 4 warps
    pos = torch.stack([torch.arange(n) // 50, (torch.arange(n) // 10) % 5, torch.arange(n) % 10]).float().cuda() + 40

    def table(p):
        cos, sin = rot(p.unsqueeze(1), p.device, torch.float32)
        return cos.squeeze(0), sin.squeeze(0)

    monkeypatch.setattr(torch._inductor.config.triton, "autotune_pointwise", False)
    cos_e, sin_e = table(pos)
    torch._dynamo.reset()
    with fresh_cache():
        cos_c, sin_c = torch.compile(table, dynamic=False)(pos)
    torch._dynamo.reset()
    torch.testing.assert_close(cos_c, cos_e, rtol=0, atol=1e-5)
    torch.testing.assert_close(sin_c, sin_e, rtol=0, atol=1e-5)
