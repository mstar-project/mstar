"""Choosing an implementation per platform, and the platform's inductor settings.

Both resolve against ``current_platform``, which these tests point at a chosen
``Platform`` instance, so they run without any accelerator.
"""

import threading
import time

import pytest
import torch._inductor.config as inductor_config

from mstar import platforms
from mstar.platforms import CpuPlatform, CudaPlatform, Platform, RocmPlatform, XpuPlatform
from mstar.platforms.compile import with_inductor_overrides
from mstar.platforms.dispatch import PlatformDispatch


@pytest.fixture
def on_platform(monkeypatch):
    """Make ``current_platform`` resolve to the given instance."""

    def use(platform: Platform) -> Platform:
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        return platform

    return use


class _Counting:
    """A factory that counts how often it ran; its implementation returns ``tag``."""

    def __init__(self, tag: str):
        self.tag = tag
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return lambda *args, **kwargs: (self.tag, args, kwargs)


def _dispatch(**factories) -> PlatformDispatch:
    return PlatformDispatch("mstar::example", **factories)


# --- PlatformDispatch -------------------------------------------------------


@pytest.mark.parametrize(
    "platform", [CudaPlatform(), RocmPlatform(), XpuPlatform(), CpuPlatform()], ids=repr,
)
def test_each_platform_gets_its_own_implementation(on_platform, platform):
    on_platform(platform)
    factories = {name: _Counting(name) for name in ("cuda", "rocm", "xpu", "cpu")}
    dispatch = _dispatch(**factories)
    assert dispatch(1, key=2) == (platform.name, (1,), {"key": 2})
    assert {name: f.calls for name, f in factories.items()} == {
        name: int(name == platform.name) for name in factories
    }


def test_default_serves_a_platform_without_its_own(on_platform):
    on_platform(RocmPlatform())
    cuda, default = _Counting("cuda"), _Counting("default")
    assert _dispatch(cuda=cuda, default=default)()[0] == "default"
    assert cuda.calls == 0


def test_the_factory_runs_once(on_platform):
    on_platform(CudaPlatform())
    cuda = _Counting("cuda")
    dispatch = _dispatch(cuda=cuda)
    for _ in range(3):
        dispatch()
    assert cuda.calls == 1


def test_concurrent_first_calls_resolve_once(on_platform):
    on_platform(CudaPlatform())
    calls = 0

    def slow_factory():
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return lambda: "cuda"

    dispatch = _dispatch(cuda=slow_factory)
    start = threading.Barrier(8)
    results = []

    def call():
        start.wait()
        results.append(dispatch())

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == ["cuda"] * 8
    assert calls == 1


def test_a_platform_with_no_implementation_is_named(on_platform):
    on_platform(RocmPlatform())
    dispatch = PlatformDispatch("mstar::rms_norm", cuda=_Counting("cuda"))
    with pytest.raises(NotImplementedError) as excinfo:
        dispatch()
    assert str(excinfo.value) == "mstar::rms_norm has no implementation for platform rocm"


def test_constructing_resolves_nothing(monkeypatch):
    def no_device():
        raise AssertionError("resolved the platform")

    monkeypatch.setattr(platforms, "resolve_platform", no_device)
    cuda = _Counting("cuda")
    dispatch = _dispatch(cuda=cuda)
    assert cuda.calls == 0
    with pytest.raises(AssertionError, match="resolved the platform"):
        dispatch()


def test_reset_resolves_again(on_platform):
    on_platform(CudaPlatform())
    cuda, rocm = _Counting("cuda"), _Counting("rocm")
    dispatch = _dispatch(cuda=cuda, rocm=rocm)
    assert dispatch()[0] == "cuda"
    on_platform(RocmPlatform())
    assert dispatch()[0] == "cuda", "the implementation is cached until reset"
    dispatch.reset()
    assert dispatch()[0] == "rocm"
    assert (cuda.calls, rocm.calls) == (1, 1)


# --- with_inductor_overrides ------------------------------------------------


class _SmallGemmsOnAten(RocmPlatform):
    """Overrides inductor for small token counts only, and records each query."""

    def __init__(self):
        self.queries: list[int | None] = []

    def inductor_overrides(self, num_tokens=None):
        self.queries.append(num_tokens)
        if num_tokens is not None and num_tokens <= 32:
            return {"max_autotune_gemm_backends": "ATEN"}
        return {}


def _read_gemm_backends(*args, **kwargs):
    return inductor_config.max_autotune_gemm_backends, args, kwargs


@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform()], ids=repr)
def test_no_overrides_returns_the_callable_itself(on_platform, platform):
    on_platform(platform)
    assert with_inductor_overrides(_read_gemm_backends, 16) is _read_gemm_backends
    assert with_inductor_overrides(_read_gemm_backends) is _read_gemm_backends


def test_overrides_hold_inside_the_call_only(on_platform):
    platform = on_platform(_SmallGemmsOnAten())
    before = inductor_config.max_autotune_gemm_backends
    assert before != "ATEN"

    wrapped = with_inductor_overrides(_read_gemm_backends, 16)
    assert wrapped is not _read_gemm_backends
    assert wrapped(1, key=2) == ("ATEN", (1,), {"key": 2})
    assert inductor_config.max_autotune_gemm_backends == before
    assert with_inductor_overrides(_read_gemm_backends, 64) is _read_gemm_backends
    assert platform.queries == [16, 64]


def test_overrides_are_undone_when_the_call_raises(on_platform):
    on_platform(_SmallGemmsOnAten())
    before = inductor_config.max_autotune_gemm_backends

    def fails():
        assert inductor_config.max_autotune_gemm_backends == "ATEN"
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        with_inductor_overrides(fails, 8)()
    assert inductor_config.max_autotune_gemm_backends == before
