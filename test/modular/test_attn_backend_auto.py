"""``AUTO`` attention backends, resolved when the resource is built.

``AUTO`` resolves to FlashInfer on every platform, the old default, and a
backend named in a spec or YAML is built as named. Builds use the CPU device
since the managers allocate nothing until their first plan.
"""

import logging
from pathlib import Path

import pytest
import torch

from mstar import platforms
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    CrossAttentionConfig,
    CrossAttentionSpec,
    KVSpec,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
    RaggedBlockCausalAttentionSpec,
    RaggedCrossAttentionSpec,
    apply_yaml_overrides,
)
from mstar.engine.resources.attn import backends as attn_backends
from mstar.engine.resources.attn import base as attn_base
from mstar.engine.resources.attn import dense as dense_module
from mstar.engine.resources.attn.backends import ATTN_BACKENDS, AttnBackendEntry, AttnBackendRegistry
from mstar.engine.resources.attn.base import AttentionManager, resolve_attn_backend
from mstar.engine.resources.attn.cross import CrossAttentionManager, FlashInferCrossManager
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.attn.ragged.base import resolve_ragged_attn_backend
from mstar.engine.resources.attn.ragged.dense import (
    DenseRaggedCrossManager,
    DenseRaggedManager,
)
from mstar.engine.resources.attn.ragged.flashinfer import (
    FlashInferRaggedBlockCausalManager,
    FlashInferRaggedCrossManager,
    FlashInferRaggedManager,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.config import KVConfig, PagedKVConfig, RingKVConfig, RingKVLayerConfig
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform

CPU = torch.device("cpu")


def paged_config() -> PagedKVConfig:
    return PagedKVConfig(
        num_layers=2, num_kv_heads=2, head_dim=64, num_qo_heads=4,
        max_seq_len=1024, max_num_pages=16, page_size=16,
    )


def ring_config() -> RingKVConfig:
    return RingKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=64, num_qo_heads=4, tokens_per_frame=128,
        layers=(RingKVLayerConfig(ring_frames=3, ring_buckets=3, pinned_dilation=1),),
    )


def kv_info(kv_config) -> EngineResourceInfo:
    kv_spec = KVSpec(resource_key="kv", nodes={"node"}, config=kv_config)
    return EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16, dependencies={"kv": kv_spec})


def attention_spec(**config) -> AttentionSpec:
    return AttentionSpec(resource_key="attn", nodes={"node"}, config=AttentionConfig(kv_cache="kv", **config))


def cross_spec(**config) -> CrossAttentionSpec:
    return CrossAttentionSpec(
        resource_key="cross", nodes={"node"}, config=CrossAttentionConfig(kv_cache="kv", **config),
    )


def ragged_config(**config) -> RaggedAttentionConfig:
    return RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=64, **config)


# --- what a config declares when it names nothing ----------------------------


def test_every_attention_config_defaults_to_auto():
    assert AttentionConfig(kv_cache="kv").backend is AttnBackend.AUTO
    assert CrossAttentionConfig(kv_cache="kv").backend is AttnBackend.AUTO
    assert ragged_config().backend is RaggedAttnBackend.AUTO


# --- resolution ----------------------------------------------------------------


def test_auto_resolves_to_flashinfer():
    assert resolve_attn_backend(AttnBackend.AUTO, CPU) is AttnBackend.FLASHINFER
    assert resolve_ragged_attn_backend(RaggedAttnBackend.AUTO, CPU) is RaggedAttnBackend.FLASHINFER


@pytest.mark.parametrize("backend", [b for b in AttnBackend if b is not AttnBackend.AUTO], ids=str)
def test_an_explicit_backend_is_kept(backend):
    assert resolve_attn_backend(backend, CPU) is backend


@pytest.mark.parametrize("backend", [b for b in RaggedAttnBackend if b is not RaggedAttnBackend.AUTO], ids=str)
def test_an_explicit_ragged_backend_is_kept(backend):
    assert resolve_ragged_attn_backend(backend, CPU) is backend


# --- builds ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", [AttnBackend.AUTO, AttnBackend.FLASHINFER], ids=str)
def test_paged_attention_builds_flashinfer(backend):
    manager = AttentionManager.build(attention_spec(backend=backend), kv_info(paged_config()))
    assert type(manager) is FlashInferManager


def test_auto_is_resolved_before_the_kv_config_check():
    """The KV config a backend requires is looked up for the resolved backend,
    so AUTO over a ring cache fails the way FlashInfer would."""
    with pytest.raises(TypeError, match="'flashinfer' requires a PagedKVConfig"):
        AttentionManager.build(attention_spec(backend=AttnBackend.AUTO), kv_info(ring_config()))


def test_dense_without_a_varlen_kernel_falls_back_to_autos_backend(monkeypatch, caplog):
    monkeypatch.setattr(dense_module, "_VARLEN_IMPORT_ERROR", "ImportError: not installed")
    dense_module._fa3_unavailable_reason.cache_clear()
    dense_module._warn_dense_fallback.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger=dense_module.logger.name):
            manager = AttentionManager.build(attention_spec(backend=AttnBackend.DENSE), kv_info(paged_config()))
    finally:
        dense_module._fa3_unavailable_reason.cache_clear()
        dense_module._warn_dense_fallback.cache_clear()
    assert type(manager) is FlashInferManager
    assert "using the paged 'flashinfer' backend" in caplog.text


class _FakeCudaPlatform:
    def __init__(self, capability):
        self.capability = capability

    def cuda_capability(self, device=None):
        return self.capability


@pytest.mark.parametrize(
    ("capability", "has_kernel"),
    [((8, 0), True), ((8, 9), True), ((9, 0), True), ((7, 5), False), ((10, 0), False), ((10, 3), False),
     ((12, 0), False)],
    ids=str,
)
def test_dense_needs_a_kernel_for_the_gpus_capability(capability, has_kernel, monkeypatch):
    """fa3-fwd imports on any NVIDIA GPU but has kernels only for sm_80 and sm_90a."""
    monkeypatch.setattr(dense_module, "_VARLEN_IMPORT_ERROR", None)
    monkeypatch.setattr(dense_module, "current_platform", _FakeCudaPlatform(capability))
    dense_module._fa3_unavailable_reason.cache_clear()
    try:
        reason = dense_module._fa3_unavailable_reason(torch.device("cuda", 0))
    finally:
        dense_module._fa3_unavailable_reason.cache_clear()
    if has_kernel:
        assert reason is None
    else:
        assert reason == f"fa3-fwd has no kernel for compute capability {capability[0]}.{capability[1]}"


@pytest.mark.parametrize("backend", [AttnBackend.AUTO, AttnBackend.FLASHINFER], ids=str)
def test_cross_attention_builds_flashinfer(backend):
    manager = CrossAttentionManager.build(cross_spec(backend=backend), kv_info(paged_config()))
    assert type(manager) is FlashInferCrossManager


def test_cross_attention_rejects_a_backend_it_does_not_implement():
    with pytest.raises(ValueError, match="Unknown cross attention backend"):
        CrossAttentionManager.build(cross_spec(backend=AttnBackend.FLEX), kv_info(paged_config()))


@pytest.mark.parametrize("backend", [RaggedAttnBackend.AUTO, RaggedAttnBackend.FLASHINFER], ids=str)
@pytest.mark.parametrize(
    ("spec_class", "manager_class"),
    [
        (RaggedAttentionSpec, FlashInferRaggedManager),
        (RaggedCrossAttentionSpec, FlashInferRaggedCrossManager),
    ],
)
def test_ragged_attention_builds_flashinfer(spec_class, manager_class, backend):
    spec = spec_class(resource_key="ragged", nodes={"node"}, config=ragged_config(backend=backend))
    manager = build_resource(spec, EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    assert type(manager) is manager_class


def _clear_fa3_caches():
    dense_module._fa3_unavailable_reason.cache_clear()
    from mstar.engine.resources.attn.ragged import dense as ragged_dense

    ragged_dense._warn_ragged_dense_fallback.cache_clear()
    ragged_dense._warn_dense_on_flashinfer.cache_clear()


@pytest.mark.parametrize(
    ("spec_class", "manager_class"),
    [
        (RaggedAttentionSpec, DenseRaggedManager),
        (RaggedCrossAttentionSpec, DenseRaggedCrossManager),
    ],
)
def test_explicit_ragged_dense_builds_when_the_kernel_is_present(spec_class, manager_class, monkeypatch):
    monkeypatch.setattr(dense_module, "_VARLEN_IMPORT_ERROR", None)
    _clear_fa3_caches()
    try:
        spec = spec_class(
            resource_key="ragged", nodes={"node"},
            config=ragged_config(backend=RaggedAttnBackend.DENSE),
        )
        manager = build_resource(spec, EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    finally:
        _clear_fa3_caches()
    assert type(manager) is manager_class


@pytest.mark.parametrize(
    ("spec_class", "manager_class"),
    [
        (RaggedAttentionSpec, FlashInferRaggedManager),
        (RaggedCrossAttentionSpec, FlashInferRaggedCrossManager),
    ],
)
def test_explicit_ragged_dense_falls_back_when_the_kernel_is_missing(
    spec_class, manager_class, monkeypatch, caplog,
):
    monkeypatch.setattr(dense_module, "_VARLEN_IMPORT_ERROR", "ImportError: not installed")
    _clear_fa3_caches()
    try:
        spec = spec_class(
            resource_key="ragged", nodes={"node"},
            config=ragged_config(backend=RaggedAttnBackend.DENSE),
        )
        with caplog.at_level(logging.WARNING, logger="mstar.engine.resources.attn.ragged.dense"):
            manager = build_resource(spec, EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    finally:
        _clear_fa3_caches()
    assert type(manager) is manager_class
    assert "running it on FlashInfer's ragged prefill" in caplog.text


def test_block_causal_dense_falls_back_even_with_the_kernel_present(monkeypatch, caplog):
    monkeypatch.setattr(dense_module, "_VARLEN_IMPORT_ERROR", None)
    _clear_fa3_caches()
    try:
        spec = RaggedBlockCausalAttentionSpec(
            resource_key="ragged", nodes={"node"}, block_size=16,
            config=ragged_config(backend=RaggedAttnBackend.DENSE),
        )
        with caplog.at_level(logging.WARNING, logger="mstar.engine.resources.attn.ragged.dense"):
            manager = build_resource(spec, EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    finally:
        _clear_fa3_caches()
    assert type(manager) is FlashInferRaggedBlockCausalManager
    assert "no block-causal kernel" in caplog.text


# --- deployment overrides --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("auto", AttnBackend.AUTO), ("flashinfer", AttnBackend.FLASHINFER), ("dense", AttnBackend.DENSE)],
)
def test_yaml_names_an_attention_backend(value, expected):
    spec, cross = attention_spec(backend=AttnBackend.DENSE), cross_spec()
    kv = KVSpec(resource_key="kv", nodes={"node"}, config=paged_config())
    apply_yaml_overrides(
        [kv, spec, cross], {"resources": {"attn": {"backend": value}, "cross": {"backend": value}}},
    )
    assert spec.config.backend is expected
    assert cross.config.backend is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", RaggedAttnBackend.AUTO),
        ("flashinfer", RaggedAttnBackend.FLASHINFER),
        ("dense", RaggedAttnBackend.DENSE),
    ],
)
def test_yaml_names_a_ragged_backend(value, expected):
    spec = RaggedAttentionSpec(resource_key="ragged", nodes={"node"}, config=ragged_config())
    apply_yaml_overrides([spec], {"resources": {"ragged": {"backend": value}}})
    assert spec.config.backend is expected


def test_yaml_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="'aiter' is not a valid AttnBackend"):
        attention_spec().apply_yaml_overrides(backend="aiter")
    with pytest.raises(ValueError, match="'aiter' is not a valid AttnBackend"):
        cross_spec().apply_yaml_overrides(backend="aiter")
    ragged = RaggedAttentionSpec(resource_key="ragged", nodes={"node"}, config=ragged_config())
    with pytest.raises(ValueError, match="'aiter' is not a valid RaggedAttnBackend"):
        ragged.apply_yaml_overrides(backend="aiter")


# --- the backend table ---------------------------------------------------------


@pytest.mark.parametrize("backend", [b for b in AttnBackend if b is not AttnBackend.AUTO], ids=str)
def test_every_backend_has_an_entry_whose_builder_imports(backend):
    entry = ATTN_BACKENDS.entry(backend)
    assert entry.backend is backend and issubclass(entry.kv_config, KVConfig)
    assert callable(attn_backends._load(entry.builder))
    assert entry.probe is None or callable(attn_backends._load(entry.probe))


@pytest.mark.parametrize("platform", [CudaPlatform, RocmPlatform, XpuPlatform, CpuPlatform], ids=str)
def test_auto_resolves_to_a_registered_backend_on_every_platform(monkeypatch, platform):
    monkeypatch.setattr(platforms, "resolve_platform", platform)
    for page_size in (None, 16, 128):
        resolved = ATTN_BACKENDS.resolve(AttnBackend.AUTO, page_size)
        assert resolved is not AttnBackend.AUTO and resolved in ATTN_BACKENDS


def test_where_each_backend_runs():
    devices = {entry.backend: entry.devices for entry in ATTN_BACKENDS}
    assert devices.pop(AttnBackend.XPU_PAGED) == ("xpu",)
    assert set(devices.values()) == {()}
    assert {entry.platforms for entry in ATTN_BACKENDS} == {()}, "every backend here runs on any platform"


def test_options_name_the_backends_that_honour_them():
    assert ATTN_BACKENDS.supporting("sliding_window") == ["flashinfer"]
    assert ATTN_BACKENDS.supporting("sm_scale") == ["flashinfer_mla"]


def test_sliding_window_elsewhere_is_refused_before_the_kv_dependency_is_read():
    with pytest.raises(ValueError, match="sliding_window is unsupported .* use the flashinfer backend"):
        AttentionManager.build(attention_spec(backend=AttnBackend.DENSE, sliding_window=64), kv_info(paged_config()))
    no_kv = EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="sliding_window is unsupported .*FLEX"):
        AttentionManager.build(attention_spec(backend=AttnBackend.FLEX, sliding_window=64), no_kv)


def test_sm_scale_is_read_by_mla_only():
    with pytest.raises(ValueError, match="'flashinfer' ignores sm_scale; only 'flashinfer_mla' reads it"):
        AttentionManager.build(attention_spec(sm_scale=0.125), kv_info(paged_config()))


def test_xpu_paged_needs_an_xpu_device():
    with pytest.raises(RuntimeError, match="'xpu_paged' requires an XPU device; got cpu"):
        AttentionManager.build(attention_spec(backend=AttnBackend.XPU_PAGED), kv_info(paged_config()))


def test_the_attention_base_is_platform_neutral():
    """Platform policy lives in the table and beside the kernels, not in base.py."""
    source = Path(attn_base.__file__).read_text()
    for word in ("current_platform", "is_rocm", "is_cuda", "torch.version.hip"):
        assert word not in source, word


# a registry of stubs, each named by "module:function" like the real entries
_HERE = __name__
_probe_reason: dict[str, str | None] = {"value": None}
_fallbacks: list = []


def _stub_build(spec, info, kv_config):
    return ("built", spec.config.backend, type(kv_config).__name__)


def _stub_build_on_rocm(spec, info, kv_config):
    return ("built on rocm", spec.config.backend, type(kv_config).__name__)


def _stub_probe(device):
    return _probe_reason["value"]


def _stub_fallback_warning(reason, backend):
    _fallbacks.append((reason, backend))


def _stub_rocm_policy(page_size):
    return AttnBackend.FLASHINFER_MLA if page_size == 128 else AttnBackend.FLASHINFER


@pytest.fixture
def stub_registry():
    registry = AttnBackendRegistry("stub attention", auto={"default": AttnBackend.FLASHINFER})
    registry.register(AttnBackendEntry(
        AttnBackend.FLASHINFER, builder={"default": f"{_HERE}:_stub_build", "rocm": f"{_HERE}:_stub_build_on_rocm"},
        kv_config=PagedKVConfig,
    ))
    registry.register(AttnBackendEntry(
        AttnBackend.FLASHINFER_MLA, builder=f"{_HERE}:_stub_build", kv_config=PagedKVConfig,
    ))
    registry.register(AttnBackendEntry(
        AttnBackend.DENSE, builder=f"{_HERE}:_stub_build", kv_config=PagedKVConfig, probe=f"{_HERE}:_stub_probe",
        falls_back_to_auto=True, on_fallback=f"{_HERE}:_stub_fallback_warning",
    ))
    registry.register(AttnBackendEntry(
        AttnBackend.XPU_PAGED, builder=f"{_HERE}:_stub_build", kv_config=PagedKVConfig,
        probe=f"{_HERE}:_stub_probe", needs="stub library",
    ))
    _probe_reason["value"] = None
    _fallbacks.clear()
    return registry


def test_a_platform_policy_decides_auto_on_that_platform_only(stub_registry, monkeypatch):
    def no_device():
        raise AssertionError("resolved the platform")

    monkeypatch.setattr(platforms, "resolve_platform", no_device)
    assert stub_registry.resolve(AttnBackend.AUTO, 128) is AttnBackend.FLASHINFER, "one policy asks no platform"
    stub_registry.set_auto("rocm", f"{_HERE}:_stub_rocm_policy")
    monkeypatch.setattr(platforms, "resolve_platform", RocmPlatform)
    assert stub_registry.resolve(AttnBackend.AUTO, 128) is AttnBackend.FLASHINFER_MLA
    assert stub_registry.resolve(AttnBackend.AUTO, 16) is AttnBackend.FLASHINFER
    assert stub_registry.resolve(AttnBackend.DENSE, 128) is AttnBackend.DENSE
    monkeypatch.setattr(platforms, "resolve_platform", CudaPlatform)
    assert stub_registry.resolve(AttnBackend.AUTO, 128) is AttnBackend.FLASHINFER


def test_a_builder_can_differ_per_platform(stub_registry, monkeypatch):
    spec, info = attention_spec(), kv_info(paged_config())
    assert stub_registry.build(spec, info)[0] == "built"
    monkeypatch.setattr(platforms, "resolve_platform", RocmPlatform)
    assert stub_registry.build(spec, info)[0] == "built on rocm"


def test_a_failed_probe_falls_back_to_auto_or_raises(stub_registry):
    info = kv_info(paged_config())
    assert stub_registry.build(attention_spec(backend=AttnBackend.DENSE), info)[1] is AttnBackend.DENSE
    _probe_reason["value"] = "ImportError: no kernel"
    assert stub_registry.build(attention_spec(backend=AttnBackend.DENSE), info)[0] == "built"
    assert _fallbacks == [("ImportError: no kernel", AttnBackend.FLASHINFER)]
    xpu = kv_info(paged_config())
    xpu = EngineResourceInfo(device=torch.device("xpu"), kv_dtype=torch.bfloat16, dependencies=xpu.dependencies)
    with pytest.raises(RuntimeError, match=r"'xpu_paged' requires a working stub library \(ImportError: no kernel\)"):
        stub_registry.build(attention_spec(backend=AttnBackend.XPU_PAGED), xpu)


def test_an_entry_for_one_platform_is_refused_on_another(stub_registry, monkeypatch):
    stub_registry.register(AttnBackendEntry(
        AttnBackend.FLEX, builder=f"{_HERE}:_stub_build", kv_config=RingKVConfig, platforms=("rocm",),
    ))
    spec, info = attention_spec(backend=AttnBackend.FLEX), kv_info(ring_config())
    monkeypatch.setattr(platforms, "resolve_platform", CudaPlatform)
    with pytest.raises(RuntimeError, match="'flex' runs on rocm; this process runs on cuda"):
        stub_registry.build(spec, info)
    monkeypatch.setattr(platforms, "resolve_platform", RocmPlatform)
    assert stub_registry.build(spec, info)[1] is AttnBackend.FLEX


def test_the_table_refuses_what_it_cannot_serve(stub_registry):
    with pytest.raises(ValueError, match="Unknown stub attention backend"):
        stub_registry.entry(AttnBackend.FLEX)
    with pytest.raises(ValueError, match="cannot register"):
        stub_registry.register(AttnBackendEntry(AttnBackend.DENSE, builder="x:y", kv_config=PagedKVConfig))
    with pytest.raises(ValueError, match="cannot register"):
        stub_registry.register(AttnBackendEntry(AttnBackend.AUTO, builder="x:y", kv_config=PagedKVConfig))
    with pytest.raises(ValueError, match="needs a default policy"):
        AttnBackendRegistry("stub", auto={"rocm": AttnBackend.FLASHINFER})
