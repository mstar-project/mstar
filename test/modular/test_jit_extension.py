"""``mstar.kernels.hip.build.JitExtension`` without a compiler: how a build is
named and staged, which architectures it targets, and how a failed build is
retried, kept and reported. ``torch.utils.cpp_extension.load`` is stubbed."""

import itertools
import os

import pytest
import torch
from torch.utils import cpp_extension

from mstar import platforms
from mstar.kernels.hip import build
from mstar.kernels.hip.build import JitExtension, clear_stale_build_lock, hip_runtime_ldflags
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform

_unique = itertools.count()


@pytest.fixture
def ext_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    monkeypatch.setattr(platforms, "resolve_platform", CpuPlatform)
    return tmp_path


def _source(root, checkout, text, name="op.cu"):
    path = root / checkout / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_two_checkouts_of_one_source_share_a_build(ext_root):
    a = JitExtension("_mstar_test", [_source(ext_root, "wt_a", "// v1\n")])
    b = JitExtension("_mstar_test", [_source(ext_root, "wt_b", "// v1\n")])
    assert a.build_name() == b.build_name() and a.build_name().startswith("_mstar_test_")
    assert a.stage() == b.stage()
    (staged,) = a.stage()
    assert open(staged).read() == "// v1\n" and "wt_a" not in staged and staged.endswith("/op.cu")


def test_what_a_build_is_named_by(ext_root):
    source = _source(ext_root, "wt", "// v1\n")
    names = {
        JitExtension("_mstar_test", [source]).build_name(),
        JitExtension("_mstar_test", [source], extra_cuda_cflags=["-O3"]).build_name(),
        JitExtension("_mstar_test", [source], archs=["gfx942"]).build_name(),
        JitExtension("_mstar_test", [source], archs=["gfx942", "gfx950"]).build_name(),
        JitExtension("_mstar_test", [_source(ext_root, "wt", "// v2\n", "other.cu")]).build_name(),
        JitExtension("_mstar_test", [source], headers=[_source(ext_root, "wt", "#pragma once\n", "op.cuh")])
        .build_name(),
    }
    assert len(names) == 6


def test_headers_are_staged_beside_the_sources_under_their_names(ext_root):
    source = _source(ext_root, "wt", '#include "op.cuh"\n')
    header = _source(ext_root, "wt", "#pragma once\n", "op.cuh")
    extension = JitExtension("_mstar_test", [source], headers=[header])
    (staged,) = extension.stage()
    assert sorted(os.listdir(os.path.dirname(staged))) == ["op.cu", "op.cuh"]


class _Rocm(RocmPlatform):
    def arch(self, device=None):
        return "gfx942"


class _Hopper(CudaPlatform):
    def arch(self, device=None):
        return "sm_90"


def test_rocm_builds_for_the_device_or_the_archs_named(ext_root, monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", _Rocm)
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    source = _source(ext_root, "wt", "// v1\n")
    assert JitExtension("_mstar_test", [source])._arch_flags() == ["--offload-arch=gfx942"]
    fat = JitExtension("_mstar_test", [source], archs=["gfx942", "gfx950"])
    assert fat._arch_flags() == ["--offload-arch=gfx942", "--offload-arch=gfx950"]
    assert fat.target_archs() == ("gfx942", "gfx950")


def test_cuda_keeps_its_own_arch_choice_unless_archs_are_named(ext_root, monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", _Hopper)
    monkeypatch.setattr(torch.version, "hip", None)
    source = _source(ext_root, "wt", "// v1\n")
    plain = JitExtension("_mstar_test", [source])
    assert plain._arch_flags() == []
    assert plain.target_archs() == ("sm_90",), "the build is still named by the device's arch"
    named = JitExtension("_mstar_test", [source], archs=["sm_90a", "sm_100"])
    assert named._arch_flags() == [
        "-gencode=arch=compute_90a,code=sm_90a", "-gencode=arch=compute_100,code=sm_100",
    ]


class _Load:
    """Stands in for cpp_extension.load: fails ``failures`` times, then returns a path."""

    def __init__(self, failures=0):
        self.failures, self.calls = failures, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise RuntimeError(f"Error building extension {kwargs['name']!r}: hipcc failed")
        return f"/built/{kwargs['name']}.so"


def _extension(ext_root, **kwargs):
    # a source of its own, so no other test's build result is cached under its name
    text = f"// test {next(_unique)}\n"
    return JitExtension("_mstar_test", [_source(ext_root, f"wt{next(_unique)}", text)], **kwargs)


def test_a_build_runs_once_per_process(ext_root, monkeypatch):
    load = _Load()
    monkeypatch.setattr(cpp_extension, "load", load)
    monkeypatch.setattr(build.time, "sleep", lambda s: None)
    checked = []
    extension = _extension(ext_root, check=checked.append)
    assert extension.unavailable_reason() is None
    assert extension.load() == f"/built/{extension.build_name()}.so"
    assert len(load.calls) == 1 and checked == [f"/built/{extension.build_name()}.so"]
    assert load.calls[0]["name"] == extension.build_name()
    assert load.calls[0]["sources"] == extension.stage()


def test_a_failed_build_is_retried_once_then_kept_as_the_reason(ext_root, monkeypatch):
    monkeypatch.setattr(build.time, "sleep", lambda s: None)
    flaky = _Load(failures=1)
    monkeypatch.setattr(cpp_extension, "load", flaky)
    assert _extension(ext_root).unavailable_reason() is None
    assert len(flaky.calls) == 2

    broken = _Load(failures=5)
    monkeypatch.setattr(cpp_extension, "load", broken)
    extension = _extension(ext_root)
    reason = extension.unavailable_reason()
    assert reason.startswith("RuntimeError: Error building extension") and "hipcc failed" in reason
    with pytest.raises(RuntimeError, match="did not build"):
        extension.load()
    assert len(broken.calls) == 2, "two attempts, then the failure is kept"


def test_a_check_that_fails_makes_the_build_unavailable(ext_root, monkeypatch):
    monkeypatch.setattr(cpp_extension, "load", _Load())
    monkeypatch.setattr(build.time, "sleep", lambda s: None)

    def missing_op(library):
        raise AttributeError("no op moe_align_block_size")

    reason = _extension(ext_root, check=missing_op).unavailable_reason()
    assert reason == "AttributeError: no op moe_align_block_size"


def test_rocm_links_the_hip_runtime_already_loaded(ext_root, monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", _Rocm)
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(build, "hip_runtime_ldflags", lambda: ["-L/venv/rocm/lib"])
    load = _Load()
    monkeypatch.setattr(cpp_extension, "load", load)
    extension = _extension(ext_root, extra_ldflags=["-lfoo"])
    extension.load()
    assert load.calls[0]["extra_ldflags"] == ["-lfoo", "-L/venv/rocm/lib"]
    assert load.calls[0]["extra_cuda_cflags"] == ["--offload-arch=gfx942"]


def test_hip_runtime_ldflags_reads_the_mapped_library(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f10 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n"
        "7f20-7f30 r-xp 00000000 08:01 2 /venv/torch/lib/libamdhip64.so.7\n"
    )
    assert hip_runtime_ldflags(str(maps)) == ["-L/venv/torch/lib", "-Wl,-rpath,/venv/torch/lib"]
    assert hip_runtime_ldflags(str(tmp_path / "missing")) == []


def test_a_lock_nobody_holds_is_removed_and_a_held_one_kept(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("")
    with open(lock) as held:
        clear_stale_build_lock(tmp_path)
        assert lock.exists(), "this process holds it open"
        assert held.readable()
    clear_stale_build_lock(tmp_path)
    assert not lock.exists()
