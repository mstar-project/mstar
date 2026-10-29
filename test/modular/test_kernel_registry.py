"""The kernel registry: one ``KernelOp`` per kernel with its schema and fake
defined once, implementations picked per platform and architecture, the
reference for CPU tensors and as the fallback, tuned configs per device, and
the listing. ``current_platform`` is pointed at a chosen ``Platform``, so no
accelerator is needed: an implementation is reached through
``KernelOp.implementation()``, which is what accelerator tensors dispatch to.
"""

import ast
import itertools
import json
import logging
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mstar import platforms
from mstar.engine.resources.rms_norm import run_rms_norm
from mstar.kernels import KernelOp, ShapePolicy, Tolerance, configs, registry
from mstar.kernels.numerics import as_tensors
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform
from mstar.platforms.dispatch import PlatformDispatch
from mstar.platforms.facts import DeviceFacts

REPO = Path(__file__).resolve().parents[2]
_names = itertools.count()


def _scale(x, scale):
    return x * scale


def _scale_fake(x, scale):
    return torch.empty_like(x)


@pytest.fixture
def declare():
    """Declare ops under fresh names; they leave the registry after the test
    (torch keeps their definitions, so a name is never reused)."""
    declared = []

    def make(schema="(Tensor x, float scale) -> Tensor", *, reference=_scale, fake=_scale_fake, **kwargs):
        name = f"_test_op_{next(_names)}"
        op = KernelOp(name, schema, fake=fake, reference=reference, **kwargs)
        declared.append(name)
        return op

    yield make
    for name in declared:
        registry._OPS.pop(name, None)


@pytest.fixture
def on_platform(monkeypatch):
    def use(platform):
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        return platform

    return use


class _Impl:
    """A factory whose implementation reports ``tag``; counts its runs and can fail."""

    def __init__(self, tag, error=None):
        self.tag, self.error, self.calls = tag, error, 0

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return lambda x, scale: (self.tag, x, scale)


class _OnArch(RocmPlatform):
    def __init__(self, arch):
        self._arch = arch

    def arch(self, device=None):
        return self._arch


# --- declaring ---------------------------------------------------------------


def test_declaring_runs_no_factory_and_resolves_no_platform(declare, monkeypatch):
    def no_device():
        raise AssertionError("resolved the platform")

    monkeypatch.setattr(platforms, "resolve_platform", no_device)
    impl = _Impl("cuda")
    op = declare(impls={"cuda": impl, "rocm:gfx950": _Impl("gfx950")})
    assert impl.calls == 0
    assert registry.get(op.name) is op and registry.get(op.qualname) is op


def test_the_op_gets_kernels_for_accelerators_and_a_composite_reference(declare):
    op = declare(impls={"cuda": _Impl("cuda")})
    for key in ("CUDA", "XPU", "CompositeExplicitAutograd"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(op.qualname, key), key
    assert torch.Tag.pt2_compliant_tag in op.op.default.tags


@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform(), XpuPlatform(), CpuPlatform()], ids=repr)
def test_cpu_tensors_run_the_reference_on_every_platform(declare, on_platform, platform):
    on_platform(platform)
    impls = {name: _Impl(name) for name in ("cuda", "rocm", "xpu")}
    op = declare(impls=impls)
    x = torch.arange(6.0).reshape(2, 3)
    torch.testing.assert_close(op(x, 2.0), x * 2)
    assert all(f.calls == 0 for f in impls.values())


def test_the_fake_gives_shapes_to_fake_and_meta_tensors(declare):
    op = declare()
    with FakeTensorMode():
        out = op(torch.empty(4, 8), 3.0)
    assert out.shape == (4, 8)
    assert op(torch.empty(2, 5, device="meta"), 3.0).shape == (2, 5)


def test_one_node_under_dynamo(declare):
    op = declare()

    def fn(x):
        return op.op(x + 1, 2.0) - 1

    x = torch.randn(3, 4)
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), fn(x))


def test_a_name_is_declared_once_and_keys_are_checked(declare):
    op = declare()
    with pytest.raises(ValueError, match="declared twice"):
        KernelOp(op.name, op.schema, fake=_scale_fake, reference=_scale)
    with pytest.raises(ValueError, match="unknown implementation keys"):
        declare(impls={"tpu": _Impl("tpu")})
    with pytest.raises(ValueError, match="'<platform>:<arch>'"):
        declare(impls={"rocm:": _Impl("rocm")})


def test_mutated_args_come_from_the_schema(declare):
    def fill(out, x):
        out.copy_(x)

    op = declare("(Tensor(a!) out, Tensor x) -> ()", reference=fill, fake=lambda out, x: None)
    assert op.mutated_args() == (0,)
    out = torch.zeros(3)
    op(out, torch.ones(3))
    assert torch.equal(out, torch.ones(3))
    assert declare().mutated_args() == ()


# --- which implementation runs ---------------------------------------------------


@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform(), XpuPlatform()], ids=repr)
def test_each_platform_resolves_its_own_implementation_once(declare, on_platform, platform):
    on_platform(platform)
    impls = {name: _Impl(name) for name in ("cuda", "rocm", "xpu")}
    op = declare(impls=impls)
    assert op.implementation_key() == platform.name
    assert op.implementation()(None, 1.0)[0] == platform.name
    assert op.implementation() is op.implementation()
    assert {name: f.calls for name, f in impls.items()} == {name: int(name == platform.name) for name in impls}
    assert op.unavailable_reason() is None


@pytest.mark.parametrize(("arch", "expected"), [("gfx950", "rocm:gfx950"), ("gfx942", "rocm")])
def test_an_arch_implementation_wins_on_its_arch(declare, on_platform, arch, expected):
    on_platform(_OnArch(arch))
    op = declare(impls={"rocm": _Impl("rocm"), "rocm:gfx950": _Impl("rocm:gfx950")})
    assert op.implementation_key() == expected
    assert op.implementation()(None, 1.0)[0] == expected


def test_a_failing_factory_names_the_op_platform_and_cause(declare, on_platform):
    on_platform(RocmPlatform())
    cause = ImportError("No module named 'flashinfer'")
    factory = _Impl("rocm", error=cause)
    op = declare(impls={"rocm": factory})
    assert op.unavailable_reason() == "ImportError: No module named 'flashinfer'"
    with pytest.raises(RuntimeError, match=f"{op.qualname} cannot run on rocm: ImportError") as excinfo:
        op.implementation()(None, 1.0)
    assert excinfo.value.__cause__ is cause
    op.unavailable_reason()
    assert factory.calls == 1, "a failure is kept, not retried per call"


def test_a_failing_factory_with_fallback_runs_the_reference_and_says_why(declare, on_platform, caplog):
    on_platform(CudaPlatform())
    op = declare(impls={"cuda": _Impl("cuda", error=OSError("libfoo.so"))}, fallback_to_reference=True)
    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        assert op.implementation() is _scale
        op.implementation()
    assert op.unavailable_reason() == "OSError: libfoo.so"
    assert caplog.text.count("runs its torch reference") == 1
    assert op.describe(probe=True)["runs"] == "reference"


def test_a_platform_without_an_implementation(declare, on_platform):
    on_platform(XpuPlatform())
    strict = declare(impls={"cuda": _Impl("cuda")})
    assert strict.implementation_key() is None
    assert strict.unavailable_reason() == "no implementation for platform xpu"
    with pytest.raises(NotImplementedError, match="no implementation for platform xpu"):
        strict.implementation()
    lenient = declare(impls={"cuda": _Impl("cuda")}, fallback_to_reference=True)
    assert lenient.implementation_key() == "reference"
    assert "runs the reference" in lenient.unavailable_reason()
    assert lenient.implementation() is _scale


def test_the_dispatch_is_what_accelerator_tensors_reach(declare, on_platform):
    on_platform(RocmPlatform())
    op = declare(impls={"rocm": _Impl("rocm")})
    assert op.dispatch(None, 2.0) == ("rocm", None, 2.0)
    assert op.dispatch.resolve() is op.implementation()


def test_a_default_implementation_covers_other_platforms(declare, on_platform):
    on_platform(XpuPlatform())
    op = declare(impls={"cuda": _Impl("cuda"), "default": _Impl("default")})
    assert op.implementation_key() == "default"
    assert op.implementation()(None, 1.0)[0] == "default"


def test_resetting_the_dispatches_resolves_again(declare, on_platform):
    on_platform(CudaPlatform())
    op = declare(impls={"cuda": _Impl("cuda"), "rocm": _Impl("rocm")})
    assert op.implementation()(None, 1.0)[0] == "cuda"
    on_platform(RocmPlatform())
    PlatformDispatch.reset_all()
    assert op.implementation()(None, 1.0)[0] == "rocm"


# --- shapes, warm-up and configs ------------------------------------------------


def _policy(warm=({"n": 2}, {"n": 300})):
    return ShapePolicy(
        dims=("n",),
        key=lambda x, scale: {"n": x.numel()},
        inputs=lambda shape, dtype, device, generator=None: (torch.ones(shape["n"], dtype=dtype, device=device), 2.0),
        classes=(("small", lambda s: s["n"] <= 128), ("large", lambda s: True)),
        warm=warm,
        dtypes=(torch.float32,),
    )


def test_shape_classes_take_the_first_match():
    policy = _policy()
    assert [policy.shape_class({"n": n}) for n in (1, 128, 129)] == ["small", "small", "large"]
    assert policy.key(torch.ones(2, 3), 1.0) == {"n": 6}


def test_warm_runs_each_shape(declare):
    seen = []

    def reference(x, scale):
        seen.append(x.numel())
        return x * scale

    op = declare(reference=reference, shapes=_policy())
    assert op.warm(device="cpu") == [{"n": 2}, {"n": 300}]
    assert op.warm([{"n": 5}], device="cpu") == [{"n": 5}]
    assert seen == [2, 300, 5]
    assert declare().warm(device="cpu") == []


def test_warm_resolves_the_implementation_for_an_accelerator_device(declare, on_platform):
    on_platform(CudaPlatform())
    factory = _Impl("cuda")
    op = declare(impls={"cuda": factory}, shapes=_policy())
    op.warm(device="cpu")
    assert factory.calls == 0, "CPU tensors run the reference"
    # meta tensors stand in for the GPU: the op runs its fake on them
    assert op.warm(device="meta") == [{"n": 2}, {"n": 300}]
    assert factory.calls == 1


class _Mi325x(RocmPlatform):
    def device_facts(self, device=None):
        return DeviceFacts(platform="rocm", arch="gfx942", product="mi325x")


def test_a_config_is_looked_up_by_arch_product_and_shape_class(declare, on_platform, tmp_path, monkeypatch):
    on_platform(_Mi325x())
    op = declare(shapes=_policy())
    monkeypatch.setattr(configs, "CONFIG_ROOT", tmp_path)
    configs._read.cache_clear()
    try:
        assert op.config({"n": 4}) is None
        arch_file = configs.config_path(op.name, "gfx942")
        arch_file.parent.mkdir(parents=True)
        arch_file.write_text(json.dumps({"classes": {"small": {"BLOCK": 64}, "default": {"BLOCK": 256}}}))
        configs._read.cache_clear()
        assert op.config({"n": 4}) == {"BLOCK": 64}
        assert op.config({"n": 4096}) == {"BLOCK": 256}
        product_file = configs.config_path(op.name, "gfx942", "mi325x")
        product_file.parent.mkdir(parents=True)
        product_file.write_text(json.dumps({"classes": {"small": {"BLOCK": 32}}}))
        configs._read.cache_clear()
        assert op.config({"n": 4}) == {"BLOCK": 32}
        assert op.config({"n": 4096}) is None, "a product's config replaces its arch's"
        assert configs.load_config(op.name, "gfx950", None) is None
        bad = configs.config_path("broken", "gfx942")
        bad.write_text(json.dumps({"BLOCK": 1}))
        with pytest.raises(ValueError, match="'classes' table"):
            configs.load_config("broken", "gfx942")
    finally:
        configs._read.cache_clear()


def test_describe_lists_what_a_listing_shows(declare, on_platform):
    on_platform(RocmPlatform())
    op = declare(impls={"rocm": _Impl("rocm")}, shapes=_policy(), tolerance=Tolerance.accumulated())
    info = op.describe()
    assert info["op"] == op.qualname and info["impls"] == ["rocm"]
    assert info["warm_shapes"] == [{"n": 2}, {"n": 300}] and info["shape_classes"] == ["small", "large"]
    assert info["tolerance"]["kind"] == "accumulated" and info["reference"].endswith("_scale")
    probed = op.describe(probe=True)
    assert (probed["platform"], probed["implementation"], probed["runs"]) == ("rocm", "rocm", "rocm")
    assert probed["unavailable_reason"] is None


# --- the declared ops --------------------------------------------------------------


def test_every_declared_op_has_a_reference_fake_and_warm_shapes():
    ops = registry.load_all()
    assert "mstar::rms_norm" in [op.qualname for op in ops]
    for op in ops:
        assert op.shapes is not None and op.shapes.warm, op
        for shape in op.shapes.warm:
            for dtype in op.shapes.dtypes:
                args = op.shapes.inputs(shape, dtype, torch.device("cpu"), None)
                assert op.shapes.key(*args) == dict(shape), (op, shape)
                outs, fakes = as_tensors(op.reference(*args)), as_tensors(op.fake(*args))
                assert [(t.shape, t.dtype) for t in fakes] == [(t.shape, t.dtype) for t in outs], (op, shape)


def _kernel_op_modules() -> set[str]:
    """Modules under mstar/ that construct a KernelOp."""
    found = set()
    for path in (REPO / "mstar").rglob("*.py"):
        if path.parent == REPO / "mstar" / "kernels" and path.name == "registry.py":
            continue
        tree = ast.parse(path.read_text())
        if any(isinstance(node, ast.Call) and getattr(node.func, "id", None) == "KernelOp"
               for node in ast.walk(tree)):
            found.add(".".join(path.relative_to(REPO).with_suffix("").parts))
    return found


def test_every_module_declaring_an_op_is_listed():
    """`python -m mstar.kernels` lists what `load_all` imports."""
    assert _kernel_op_modules() == set(registry.DECLARING_MODULES)


def test_rms_norm_is_declared_for_every_platform_it_ran_on():
    op = registry.get("rms_norm")
    assert set(op.impl_keys) == {"cuda", "rocm", "xpu"}
    assert not op.fallback_to_reference
    assert op.tolerance == Tolerance.rounded_once()
    assert op.shapes.shape_class({"rows": 64, "hidden": 4096}) == "decode"
    assert op.shapes.shape_class({"rows": 512, "hidden": 4096}) == "prefill"


def test_rms_norm_is_one_node_under_dynamo():
    weight = torch.randn(64)

    def fn(x):
        return run_rms_norm(x * 2, weight, 1e-6) + 1

    x = torch.randn(5, 64)
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), fn(x))


def test_selection_names_what_runs_and_why_not(declare, on_platform):
    on_platform(RocmPlatform())
    plain = declare(impls={"rocm": _Impl("fork")})
    assert plain.selection() == "rocm"

    def tuned():
        impl = _Impl("tuned")()
        impl.kernel_selection = "triton for decode"
        return impl

    assert declare(impls={"rocm": tuned}).selection() == "rocm: triton for decode"
    broken = declare(impls={"rocm": _Impl("x", error=ImportError("no aiter"))})
    assert broken.selection() == "rocm unavailable (ImportError: no aiter)"
    fallback = declare(impls={"rocm": _Impl("x", error=ImportError("no aiter"))}, fallback_to_reference=True)
    assert fallback.selection().endswith("; runs the reference")
    assert declare(impls={"cuda": _Impl("nv")}, fallback_to_reference=True).selection().startswith("reference")


def test_warm_all_resolves_every_op_once(declare, on_platform, monkeypatch):
    on_platform(RocmPlatform())
    factory = _Impl("fork")
    op = declare(impls={"rocm": factory})
    monkeypatch.setattr(registry, "DECLARING_MODULES", ())
    monkeypatch.setattr(registry, "_OPS", {op.name: op})
    lines = registry.warm_all()
    assert lines[op.qualname] == "rocm"
    assert factory.calls == 1
    registry.warm_all()
    assert factory.calls == 1, "a resolved implementation is kept"
