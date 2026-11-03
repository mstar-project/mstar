"""The process env default that points Triton at torch's ptxas for Blackwell.

ptxas before 13.1.115 (Triton 3.7.1 bundles 13.1.80 for Blackwell) miscompiles a
floor division of a periodic index under a range select when a thread holds two
elements. The CUDA tests run such kernels at that launch config with the
default applied; on sm_103 they fail with TRITON_PTXAS_BLACKWELL_PATH set to
Triton's bundled ptxas.
"""

import os
import stat

import pytest
import torch
import triton
import triton.language as tl

from mstar import platforms
from mstar.platforms import ptxas
from mstar.platforms.interface import CudaPlatform

BUNDLED = "/site-packages/triton/backends/nvidia/bin/ptxas-blackwell"
TORCH_PTXAS = os.path.join(os.path.dirname(torch.__file__), "bin", "ptxas")
USER_PTXAS = "/opt/cuda/bin/ptxas"
TAG = "TORCH_COMPILE_CACHE_KEY_TAG"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")


@pytest.fixture(autouse=True)
def _undecided():
    ptxas.blackwell_ptxas_env.cache_clear()
    yield
    ptxas.blackwell_ptxas_env.cache_clear()


@pytest.mark.parametrize(
    ("blackwell", "bundled", "torch_version", "user", "want_torch_ptxas"),
    [
        (True, (13, 1, 80), (13, 2, 78), False, True),
        (False, (13, 1, 80), (13, 2, 78), False, False),
        (True, (13, 3, 33), (13, 2, 78), False, False),  # Triton 3.8
        (True, (13, 1, 115), (13, 2, 78), False, False),
        (True, (13, 1, 80), (13, 1, 80), False, False),
        (True, (13, 1, 80), None, False, False),  # no ptxas in torch
        (True, None, (13, 2, 78), False, False),
        (True, (13, 1, 80), (13, 2, 78), True, False),
    ],
    ids=["affected", "not blackwell", "triton 3.8", "bundled fixed", "torch's too old", "torch has none",
         "bundled unknown", "already set"],
)
def test_torch_ptxas_only_on_blackwell_with_an_older_triton_ptxas(
    monkeypatch, blackwell, bundled, torch_version, user, want_torch_ptxas,
):
    versions = {BUNDLED: bundled, TORCH_PTXAS: torch_version, USER_PTXAS: (13, 2, 51)}
    monkeypatch.setattr(ptxas, "_has_blackwell_gpu", lambda: blackwell)
    monkeypatch.setattr(ptxas, "_triton_blackwell_ptxas", lambda: BUNDLED)
    monkeypatch.setattr(ptxas, "ptxas_version", versions.get)
    if user:
        monkeypatch.setenv(ptxas.BLACKWELL_PTXAS_ENV, USER_PTXAS)
    else:
        monkeypatch.delenv(ptxas.BLACKWELL_PTXAS_ENV, raising=False)

    want = {ptxas.BLACKWELL_PTXAS_ENV: TORCH_PTXAS, TAG: "ptxas-blackwell-13.2.78"} if want_torch_ptxas else {}
    if user:
        want = {TAG: "ptxas-blackwell-13.2.51"}  # the tag names the ptxas the user chose
    defaults = {k: v(None) if callable(v) else v for k, v in CudaPlatform().process_env_defaults().items()}
    assert {k: v for k, v in defaults.items() if k in (ptxas.BLACKWELL_PTXAS_ENV, TAG)} == want


@pytest.mark.parametrize(
    ("current", "want"),
    [
        (None, "ptxas-blackwell-13.2.78"),
        ("mine", "mine-ptxas-blackwell-13.2.78"),
        ("mine-ptxas-blackwell-13.2.78", "mine-ptxas-blackwell-13.2.78"),  # a worker applies it again
    ],
)
def test_the_ptxas_is_appended_once_to_a_tag_the_user_set(current, want):
    (tag,) = ptxas._cache_tag((13, 2, 78)).values()
    assert tag(current) == want


def test_ptxas_version_reads_the_build(tmp_path):
    fake = tmp_path / "ptxas"
    fake.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.1, V13.1.80'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    assert ptxas.ptxas_version(str(fake)) == (13, 1, 80)
    assert ptxas.ptxas_version(str(tmp_path / "missing")) is None


@pytest.fixture
def ptxas_default(monkeypatch, tmp_path):
    """M*'s process env defaults applied as a worker applies them, with a fresh
    Triton cache so no kernel another ptxas compiled is reused."""
    for key in (ptxas.BLACKWELL_PTXAS_ENV, TAG):
        # keep a value already set; otherwise set then delete, so teardown removes it
        monkeypatch.setenv(key, os.environ.get(key, ""))
        if not os.environ[key]:
            monkeypatch.delenv(key)
    platforms.apply_process_env_defaults()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


@triton.jit
def _floor_div_under_range_select(out_ptr, n, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    x0 = xindex % 64
    a = x0 - 1
    q = a // 3
    floor = tl.where((a % 3 != 0) & (a < 0), q - 1, q)  # Inductor's div_floor_integer
    tl.store(out_ptr + xindex, tl.where((x0 >= 1) & (x0 < 60), floor, 0), xindex < n)


@needs_cuda
@pytest.mark.parametrize(("xblock", "num_warps"), [(64, 1), (256, 4), (1024, 16)])  # two elements per thread
def test_floor_division_under_a_range_select(ptxas_default, xblock, num_warps):
    n = 64 * 1024
    x0 = torch.arange(n, device="cuda") % 64
    want = torch.where((x0 >= 1) & (x0 < 60), torch.div(x0 - 1, 3, rounding_mode="floor"), 0).int()
    out = torch.empty(n, device="cuda", dtype=torch.int32)
    _floor_div_under_range_select[(triton.cdiv(n, xblock),)](out, n, XBLOCK=xblock, num_warps=num_warps)
    assert torch.equal(out, want)


def _elems_per_thread(config) -> float:
    block = 1
    for key in ("XBLOCK", "YBLOCK", "ZBLOCK"):
        block *= config.kwargs.get(key, 1)
    return block / (32 * config.num_warps)


@pytest.fixture
def two_elements_per_thread(monkeypatch):
    """Inductor's pointwise kernels limited to their candidate configs giving
    each thread two elements; returns a list with one entry per limited kernel."""
    from torch._inductor.runtime import triton_heuristics
    from torch._inductor.runtime.hints import HeuristicType

    cached_autotune = triton_heuristics.cached_autotune
    limited = []

    def two_only(size_hints, configs, *args, **kwargs):
        two = [c for c in configs if _elems_per_thread(c) == 2]
        if kwargs.get("heuristic_type") == HeuristicType.POINTWISE and two:
            configs = two
            limited.append(len(two))
        return cached_autotune(size_hints, configs, *args, **kwargs)

    monkeypatch.setattr(triton_heuristics, "cached_autotune", two_only)
    # compile in this process, where the limit applies
    monkeypatch.setattr(torch._inductor.config, "compile_threads", 1)
    return limited


@needs_cuda
@pytest.mark.parametrize("positions", ["121-frame video", "small table"])
@torch.no_grad()
def test_compiled_interleaved_mrope_matches_eager(ptxas_default, two_elements_per_thread, positions):
    from torch._inductor.utils import fresh_cache

    from mstar.model.cosmos3.components.packing import build_static_inputs
    from mstar.model.cosmos3.components.transformer import Cosmos3RotaryEmbedding
    from mstar.model.cosmos3.config import Cosmos3Config

    cfg = Cosmos3Config(rope_theta=1e8)  # Edge's mRoPE
    rot = Cosmos3RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.rope_axes_dim)
    if positions == "121-frame video":  # 832x480 after a short prompt, as served
        pos = build_static_inputs(list(range(40)), (1, 48, 31, 30, 52), cfg, 4, 24.0, "cuda")["vision_mrope_ids"]
    else:  # 200 x 64 elements, where every pointwise candidate holds two per thread
        n = 200
        pos = torch.stack([torch.arange(n) // 50, (torch.arange(n) // 10) % 5, torch.arange(n) % 10]) + 40
        pos = pos.float().cuda()

    def table(p):
        cos, sin = rot(p.unsqueeze(1), p.device, torch.float32)
        return cos.squeeze(0), sin.squeeze(0)

    cos_e, sin_e = table(pos)
    torch._dynamo.reset()
    with fresh_cache():
        cos_c, sin_c = torch.compile(table, dynamic=False)(pos)
    torch._dynamo.reset()
    assert two_elements_per_thread, "no pointwise kernel ran at two elements per thread"
    # rounding differs by up to ~1e-4 at video positions; the miscompile is off by up to 2
    torch.testing.assert_close(cos_c, cos_e, rtol=0, atol=1e-3)
    torch.testing.assert_close(sin_c, sin_e, rtol=0, atol=1e-3)
