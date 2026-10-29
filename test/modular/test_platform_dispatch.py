"""``PlatformDispatch``, and the platform's inductor overrides at the engine's
compile sites. ``current_platform`` is pointed at a chosen ``Platform``, so no
accelerator is needed.
"""

import functools
import threading
import time
from types import SimpleNamespace

import pytest
import torch
import torch._inductor.config as inductor_config

from mstar import platforms
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.engine import Engine
from mstar.platforms import CpuPlatform, CudaPlatform, Platform, RocmPlatform, XpuPlatform, rocm_gemm
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


class _OnArch(RocmPlatform):
    """A ROCm process whose GPU is ``arch``; counts the times it is asked."""

    def __init__(self, arch: str):
        self._arch = arch
        self.asked = 0

    def arch(self, device=None) -> str:
        self.asked += 1
        return self._arch


@pytest.mark.parametrize(("arch", "expected"), [("gfx950", "rocm:gfx950"), ("gfx942", "rocm")])
def test_an_arch_factory_wins_on_its_arch_only(on_platform, arch, expected):
    on_platform(_OnArch(arch))
    factories = {"rocm": _Counting("rocm"), "rocm:gfx950": _Counting("rocm:gfx950")}
    dispatch = PlatformDispatch(
        "mstar::example", rocm=factories["rocm"], archs={"rocm:gfx950": factories["rocm:gfx950"]},
    )
    assert dispatch.selected_key() == expected
    assert dispatch()[0] == expected
    assert {key: f.calls for key, f in factories.items()} == {key: int(key == expected) for key in factories}


def test_the_arch_is_asked_only_where_the_platform_has_arch_factories(on_platform):
    platform = on_platform(_OnArch("gfx942"))
    dispatch = PlatformDispatch("mstar::example", rocm=_Counting("rocm"), archs={"cuda:sm_100": _Counting("sm_100")})
    assert dispatch()[0] == "rocm"
    assert platform.asked == 0


def test_an_arch_factory_without_a_platform_factory_falls_to_default(on_platform):
    on_platform(_OnArch("gfx942"))
    dispatch = PlatformDispatch(
        "mstar::example", default=_Counting("default"), archs={"rocm:gfx950": _Counting("rocm:gfx950")},
    )
    assert dispatch.selected_key() == "default"
    assert dispatch()[0] == "default"


@pytest.mark.parametrize("key", ["gfx950", "rocm", "tpu:v5", "rocm:"])
def test_a_malformed_arch_key_is_refused(key):
    with pytest.raises(ValueError, match="'<platform>:<arch>'"):
        PlatformDispatch("mstar::example", archs={key: _Counting(key)})


def test_selecting_runs_no_factory_and_resolve_runs_it_once(on_platform):
    on_platform(CudaPlatform())
    cuda = _Counting("cuda")
    dispatch = _dispatch(cuda=cuda)
    assert dispatch.selected_key() == "cuda" and cuda.calls == 0
    impl = dispatch.resolve()
    assert dispatch.resolve() is impl and cuda.calls == 1
    assert _dispatch().selected_key() is None


def test_reset_all_resets_every_instance(on_platform):
    on_platform(CudaPlatform())
    dispatches = [_dispatch(cuda=_Counting("cuda"), rocm=_Counting("rocm")) for _ in range(2)]
    assert [d()[0] for d in dispatches] == ["cuda", "cuda"]
    on_platform(RocmPlatform())
    PlatformDispatch.reset_all()
    assert [d()[0] for d in dispatches] == ["rocm", "rocm"]


# --- with_inductor_overrides ------------------------------------------------


class _SmallGemmsOnAten(RocmPlatform):
    """Overrides inductor for small token counts only, and records each query."""

    def __init__(self):
        self.queries: list[int | None] = []

    def inductor_overrides(self, num_tokens=None, rows=None):
        self.queries.append(num_tokens)
        if num_tokens is not None and num_tokens <= 32:
            return {"max_autotune_gemm_backends": "ATEN"}
        return {}


def _read_gemm_backends(*args, **kwargs):
    return inductor_config.max_autotune_gemm_backends, args, kwargs


@pytest.mark.parametrize(
    ("platform", "num_tokens"),
    [(CudaPlatform(), 16), (CudaPlatform(), None), (RocmPlatform(), 64), (RocmPlatform(), None)],
    ids=repr,
)
def test_no_overrides_returns_the_callable_itself(on_platform, platform, num_tokens):
    on_platform(platform)
    assert with_inductor_overrides(_read_gemm_backends, num_tokens) is _read_gemm_backends


@pytest.mark.parametrize(
    ("num_tokens", "rows", "aten_only"),
    [
        (1, 1, True), (32, 32, True), (32, 4, True), (33, None, False),
        (64, 64, True), (128, 128, True), (129, 129, False),  # decode buckets
        (64, 4, False), (128, 16, False), (1024, 16, False),  # packed prefill
        (None, None, False),
    ],
)
def test_rocm_keeps_small_and_decode_buckets_gemms_on_aten(monkeypatch, num_tokens, rows, aten_only):
    """Small buckets and one-token-per-row buckets up to 128 rows autotune
    GEMMs over hipBLASLt only. Bigger ones, packed prefills past 32 tokens,
    and forwards compiled for any token count keep inductor's defaults."""
    monkeypatch.setattr(rocm_gemm, "skinny_gemm_passes", lambda: None)
    overrides = RocmPlatform().inductor_overrides(num_tokens, rows)
    assert overrides == ({"max_autotune_gemm_backends": "ATEN"} if aten_only else {})


def test_cuda_keeps_inductors_defaults():
    assert all(CudaPlatform().inductor_overrides(n) == {} for n in (1, 32, 1024, None))


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


# --- the engine's compile sites ---------------------------------------------


class _EverythingOnAten(_SmallGemmsOnAten):
    def inductor_overrides(self, num_tokens=None, rows=None):
        self.queries.append(num_tokens)
        return {"max_autotune_gemm_backends": "ATEN"}


class _Submodule:
    def forward(self, *args, **kwargs):
        return _read_gemm_backends(*args, **kwargs)

    def forward_batched(self, *args, **kwargs):
        return _read_gemm_backends(*args, **kwargs)


@pytest.fixture
def fake_compile(monkeypatch):
    """Swap ``torch.compile`` for one that returns a new eager callable, so a
    call sees the inductor config it ran under. Gives the list of compiled fns."""
    compiled = []

    def compile_(fn, **kwargs):
        compiled.append(fn)
        return functools.partial(fn)

    monkeypatch.setattr(torch, "compile", compile_)
    return compiled


class _GraphRunner:
    _forward_for = AcceleratorGraphRunner._forward_for

    def __init__(self):
        self._submodule = _Submodule()
        self._compiled_forwards = {}


def _bucket(num_tokens: int, bs: int | None = None):
    return SimpleNamespace(
        config=SimpleNamespace(
            compile=True, compile_mode=None, capture_forward_method="forward_batched", compile_dynamic=False,
        ),
        config_idx=0,
        num_tokens=num_tokens,
        bs=num_tokens if bs is None else bs,
    )


def test_a_capture_without_overrides_gets_the_compiled_forward(on_platform, fake_compile):
    on_platform(CudaPlatform())
    runner = _GraphRunner()
    first = runner._forward_for(_bucket(64))
    assert first is runner._compiled_forwards[0]
    assert runner._forward_for(_bucket(8)) is first
    assert len(fake_compile) == 1


def test_each_capture_runs_under_its_buckets_overrides(on_platform, fake_compile):
    platform = on_platform(_SmallGemmsOnAten())
    runner = _GraphRunner()
    large = runner._forward_for(_bucket(64))
    small = runner._forward_for(_bucket(16))
    assert large is runner._compiled_forwards[0]
    assert small()[0] == "ATEN"
    assert platform.queries == [64, 16]
    assert len(fake_compile) == 1, "buckets share one compiled forward"


@pytest.mark.parametrize("overrides", [False, True])
def test_compiled_submodule_forwards_take_overrides_for_any_token_count(
    monkeypatch, on_platform, fake_compile, overrides,
):
    platform = on_platform(_EverythingOnAten() if overrides else CudaPlatform())
    engine = Engine(graph_runtime=None, device_capability=SimpleNamespace(is_available=True))
    engine._submodules = {
        "node": SimpleNamespace(submodule=_Submodule(), forward=None, forward_batched=None),
    }
    engine._item_encoders = set()
    engine._compile_submodules()
    mgmt = engine._submodules["node"]
    expected = "ATEN" if overrides else inductor_config.max_autotune_gemm_backends
    assert mgmt.forward()[0] == expected
    assert mgmt.forward_batched()[0] == expected
    assert len(fake_compile) == 2
    if overrides:
        assert platform.queries == [None, None]
    else:
        assert isinstance(mgmt.forward, functools.partial)
