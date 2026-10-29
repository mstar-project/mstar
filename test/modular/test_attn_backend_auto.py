"""``AUTO`` attention backends, resolved when the resource is built.

``AUTO`` resolves to FlashInfer on NVIDIA, Intel and CPU processes, the old
default. On ROCm it resolves to AITER, or to AMD's FlashInfer build where AITER
is unsafe or missing. A backend named in a spec or YAML is built as named.
Builds use the CPU device since the managers allocate nothing until their
first plan.
"""

import dataclasses
import inspect
import logging
import types
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
    RaggedCrossAttentionSpec,
    apply_yaml_overrides,
)
from mstar.engine.resources.attn import aiter as rocm_attn
from mstar.engine.resources.attn import backends as attn_backends
from mstar.engine.resources.attn import base as attn_base
from mstar.engine.resources.attn import dense as dense_module
from mstar.engine.resources.attn import flashinfer as flashinfer_module
from mstar.engine.resources.attn.aiter import AiterCrossManager, AiterPagedAttentionManager
from mstar.engine.resources.attn.backends import (
    ATTN_BACKENDS,
    AttnBackendEntry,
    AttnBackendRegistry,
    flashinfer_kwargs,
)
from mstar.engine.resources.attn.base import AttentionManager, resolve_attn_backend
from mstar.engine.resources.attn.cross import CrossAttentionManager, FlashInferCrossManager
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.attn.ragged import aiter as rocm_ragged
from mstar.engine.resources.attn.ragged.aiter import (
    AiterRaggedCrossManager,
    AiterRaggedManager,
)
from mstar.engine.resources.attn.ragged.base import resolve_ragged_attn_backend
from mstar.engine.resources.attn.ragged.flashinfer import (
    FlashInferRaggedCrossManager,
    FlashInferRaggedManager,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.config import KVConfig, PagedKVConfig, RingKVConfig, RingKVLayerConfig
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import BucketKey, SlotLease
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def cpu_process(monkeypatch):
    """Resolve as a process without a GPU unless a test says otherwise, so the
    suite answers the same on a GPU node."""
    monkeypatch.setattr(platforms, "resolve_platform", CpuPlatform)


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


def test_an_explicit_ragged_backend_is_kept():
    assert resolve_ragged_attn_backend(RaggedAttnBackend.FLASHINFER, CPU) is RaggedAttnBackend.FLASHINFER


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
    dense_module._varlen_unavailable_reason.cache_clear()
    dense_module._warn_dense_fallback.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger=dense_module.logger.name):
            manager = AttentionManager.build(attention_spec(backend=AttnBackend.DENSE), kv_info(paged_config()))
    finally:
        dense_module._varlen_unavailable_reason.cache_clear()
        dense_module._warn_dense_fallback.cache_clear()
    assert type(manager) is FlashInferManager
    assert "using the paged 'flashinfer' backend" in caplog.text


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
    ("value", "expected"), [("auto", RaggedAttnBackend.AUTO), ("flashinfer", RaggedAttnBackend.FLASHINFER)],
)
def test_yaml_names_a_ragged_backend(value, expected):
    spec = RaggedAttentionSpec(resource_key="ragged", nodes={"node"}, config=ragged_config())
    apply_yaml_overrides([spec], {"resources": {"ragged": {"backend": value}}})
    assert spec.config.backend is expected


def test_yaml_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="'aitre' is not a valid AttnBackend"):
        attention_spec().apply_yaml_overrides(backend="aitre")
    with pytest.raises(ValueError, match="'aitre' is not a valid AttnBackend"):
        cross_spec().apply_yaml_overrides(backend="aitre")
    ragged = RaggedAttentionSpec(resource_key="ragged", nodes={"node"}, config=ragged_config())
    with pytest.raises(ValueError, match="'aitre' is not a valid RaggedAttnBackend"):
        ragged.apply_yaml_overrides(backend="aitre")


# --- ROCm: AITER by default, FlashInfer's ROCm build where AITER would be unsafe ---



@pytest.fixture
def rocm(monkeypatch):
    """A ROCm process whose AITER and FlashInfer-ROCm installs each import or not."""

    def setup(aiter: bool = True, flashinfer: bool = True, varlen: bool = True):
        monkeypatch.setattr(platforms, "resolve_platform", RocmPlatform)
        monkeypatch.setattr(
            rocm_attn, "aiter_paged_unavailable_reason",
            lambda: None if aiter else "ImportError: no aiter",
        )
        monkeypatch.setattr(
            rocm_attn, "flashinfer_rocm_unavailable_reason",
            lambda: None if flashinfer else "ImportError: no flashinfer.rocm",
        )
        monkeypatch.setattr(
            rocm_ragged, "aiter_varlen_unavailable_reason",
            lambda: None if varlen else "ImportError: no aiter",
        )

    return setup


def paged(page_size: int, num_qo_heads: int = 4, num_kv_heads: int = 2) -> PagedKVConfig:
    return PagedKVConfig(
        num_layers=2, num_kv_heads=num_kv_heads, head_dim=64, num_qo_heads=num_qo_heads,
        max_seq_len=1024, max_num_pages=16, page_size=page_size,
    )


@pytest.mark.parametrize("page_size", [128, 256, 96])
def test_rocm_auto_is_aiter_where_it_masks_page_tails(rocm, page_size):
    rocm()
    assert resolve_attn_backend(AttnBackend.AUTO, CPU, page_size) is AttnBackend.AITER


@pytest.mark.parametrize("page_size", [16, 32, 64])
def test_rocm_auto_prefers_flashinfer_where_aiter_reads_page_tails_unmasked(rocm, page_size):
    rocm()
    assert resolve_attn_backend(AttnBackend.AUTO, CPU, page_size) is AttnBackend.FLASHINFER
    rocm(flashinfer=False)
    assert resolve_attn_backend(AttnBackend.AUTO, CPU, page_size) is AttnBackend.AITER


def test_rocm_auto_without_aiter_is_flashinfer(rocm):
    rocm(aiter=False)
    assert resolve_attn_backend(AttnBackend.AUTO, CPU, 128) is AttnBackend.FLASHINFER


def test_rocm_auto_with_neither_is_a_startup_error(rocm):
    rocm(aiter=False, flashinfer=False)
    with pytest.raises(RuntimeError, match="amd-aiter.*amd-flashinfer"):
        resolve_attn_backend(AttnBackend.AUTO, CPU, 128)


def test_an_explicit_backend_is_kept_on_rocm(rocm):
    rocm(aiter=False, flashinfer=False)
    for backend in AttnBackend:
        if backend is not AttnBackend.AUTO:
            assert resolve_attn_backend(backend, CPU, 128) is backend


def test_cuda_never_asks_for_the_rocm_kernels(monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", CudaPlatform)

    def unexpected_import():
        raise AssertionError("a CUDA process checked for a ROCm kernel library")

    monkeypatch.setattr(rocm_attn, "aiter_paged_unavailable_reason", unexpected_import)
    monkeypatch.setattr(rocm_attn, "flashinfer_rocm_unavailable_reason", unexpected_import)
    monkeypatch.setattr(rocm_ragged, "aiter_varlen_unavailable_reason", unexpected_import)
    assert resolve_attn_backend(AttnBackend.AUTO, CPU, 64) is AttnBackend.FLASHINFER
    assert resolve_ragged_attn_backend(RaggedAttnBackend.AUTO, CPU) is RaggedAttnBackend.FLASHINFER
    assert flashinfer_kwargs(paged(128), "auto") == {"backend": "auto"}


@pytest.mark.parametrize(
    ("heads", "tensor_cores"), [((4, 2), False), ((24, 8), False), ((8, 8), False), ((12, 2), True)],
)
def test_flashinfer_on_rocm_is_fa2_without_tensor_cores_where_it_dispatches(rocm, heads, tensor_cores):
    rocm()
    kwargs = flashinfer_kwargs(paged(128, *heads), "auto")
    assert kwargs == {"backend": "fa2", "decode_tensor_cores": tensor_cores}
    # a backend the deployment named is kept
    assert flashinfer_kwargs(paged(128, *heads), "aiter")["backend"] == "aiter"


def test_flashinfer_on_rocm_without_the_amd_build_names_the_alternatives(rocm):
    rocm(flashinfer=False)
    with pytest.raises(RuntimeError, match="'aiter' or 'auto'"):
        AttentionManager.build(attention_spec(backend=AttnBackend.FLASHINFER), kv_info(paged(128)))


def test_rocm_builds_aiter_for_paged_and_cross_attention(rocm):
    rocm()
    manager = AttentionManager.build(attention_spec(), kv_info(paged(128)))
    assert type(manager) is AiterPagedAttentionManager
    cross = CrossAttentionManager.build(cross_spec(), kv_info(paged(128)))
    assert type(cross) is AiterCrossManager


def test_rocm_flashinfer_manager_is_pinned_to_fa2(rocm):
    rocm()
    manager = AttentionManager.build(attention_spec(), kv_info(paged(64)))
    assert type(manager) is FlashInferManager
    assert manager._wrapper_kv_kwargs["backend"] == "fa2"
    assert manager._decode_tensor_cores is False


def test_rocm_flashinfer_splits_a_combined_walk_with_its_decode_kernel_choice(rocm, monkeypatch):
    """A split walk on ROCm's FlashInfer gets the mixed wrapper, sized from the
    scheduler's caps, whose decode half takes the same tensor-core choice as a
    decode step."""
    rocm()
    info = dataclasses.replace(kv_info(paged(64)), max_rows_in_combined={"prefill": 2, "decode": None})
    split = CombinedSplit("mixed", prefill_walks=("prefill",))
    manager = AttentionManager.build(attention_spec(splits=(split,)), info)
    assert type(manager) is FlashInferManager
    built = {}
    monkeypatch.setattr(flashinfer_module, "FlashInferMixedWrapper", lambda **kw: built.update(kw))
    manager._workspaces = types.SimpleNamespace(get=lambda label, slot: None)
    bucket = BucketKey(graph_walk="mixed", bs=4, num_tokens=16)
    manager._cg_wrapper(SlotLease(slot=0, bucket=bucket), "main", 4, force_prefill=True)
    assert (built["use_tensor_cores"], built["backend"], built["max_prefill_rows"]) == (False, "fa2", 2)


def test_aiter_builds_for_a_split_walk(rocm):
    rocm()
    split = CombinedSplit("mixed", prefill_walks=("prefill",))
    manager = AttentionManager.build(attention_spec(splits=(split,)), kv_info(paged(128)))
    assert type(manager) is AiterPagedAttentionManager


def test_aiter_is_refused_off_rocm(monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", CudaPlatform)
    with pytest.raises(RuntimeError, match="'aiter' runs on rocm; this process runs on cuda"):
        AttentionManager.build(attention_spec(backend=AttnBackend.AITER), kv_info(paged(128)))


def test_aiter_refuses_what_its_kernel_was_not_checked_for(rocm):
    rocm()
    big = PagedKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=512, max_seq_len=64, max_num_pages=4, page_size=128,
    )
    with pytest.raises(ValueError, match="head_dim 512"):
        AttentionManager.build(attention_spec(backend=AttnBackend.AITER), kv_info(big))
    info = EngineResourceInfo(
        device=CPU, kv_dtype=torch.float32,
        dependencies={"kv": KVSpec(resource_key="kv", nodes={"node"}, config=paged(128))},
    )
    with pytest.raises(ValueError, match="KV dtype"):
        AttentionManager.build(attention_spec(backend=AttnBackend.AITER), info)


@pytest.mark.parametrize(
    ("varlen", "flashinfer", "expected"),
    [(True, True, RaggedAttnBackend.AITER), (False, True, RaggedAttnBackend.FLASHINFER)],
)
def test_rocm_ragged_auto(rocm, varlen, flashinfer, expected):
    rocm(varlen=varlen, flashinfer=flashinfer)
    assert resolve_ragged_attn_backend(RaggedAttnBackend.AUTO, CPU) is expected


def test_rocm_ragged_with_neither_is_a_startup_error(rocm):
    rocm(varlen=False, flashinfer=False)
    with pytest.raises(RuntimeError, match="No ragged attention backend"):
        resolve_ragged_attn_backend(RaggedAttnBackend.AUTO, CPU)


@pytest.mark.parametrize(
    ("spec_class", "manager_class"),
    [(RaggedAttentionSpec, AiterRaggedManager), (RaggedCrossAttentionSpec, AiterRaggedCrossManager)],
)
def test_rocm_ragged_builds_aiter(rocm, spec_class, manager_class):
    rocm()
    spec = spec_class(resource_key="ragged", nodes={"node"}, config=ragged_config())
    manager = build_resource(spec, EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    assert type(manager) is manager_class


def public_surface(manager) -> dict:
    """What a model or the engine can reach on ``manager``: every public
    property or attribute with its value, every method with its parameters."""
    surface = {}
    for name in dir(manager):
        if name.startswith("_"):
            continue
        value = getattr(manager, name)
        if callable(value) and not isinstance(inspect.getattr_static(manager, name), property):
            value = [(p.name, p.kind, p.default) for p in inspect.signature(value).parameters.values()]
        surface[name] = value
    return surface


@pytest.mark.parametrize(
    ("spec_class", "manager_class"),
    [(RaggedAttentionSpec, AiterRaggedManager), (RaggedCrossAttentionSpec, AiterRaggedCrossManager)],
)
def test_rocm_ragged_managers_are_drop_ins_for_flashinfers(rocm, spec_class, manager_class):
    """Models are written against the FlashInfer managers, so whatever AUTO
    builds on ROCm has to offer the same properties, with the same values for
    the same config, and the same methods."""
    rocm()

    def build(backend):
        spec = spec_class(resource_key="ragged", nodes={"node"}, config=ragged_config(backend=backend))
        apply_yaml_overrides([spec], {"resources": {"ragged": {"max_segments_per_request": 3}}})
        return build_resource(spec, EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))

    flashinfer, rocm_default = build(RaggedAttnBackend.FLASHINFER), build(RaggedAttnBackend.AUTO)
    assert type(rocm_default) is manager_class
    expected = public_surface(flashinfer)
    # the FlashInfer wrapper type the manager builds, not something callers use
    del expected["wrapper_class"]
    assert expected["max_segments_per_request"] == 3
    assert public_surface(rocm_default) == expected


def test_yaml_names_the_aiter_backends():
    spec = attention_spec()
    spec.apply_yaml_overrides(backend="aiter")
    assert spec.config.backend is AttnBackend.AITER
    ragged = RaggedAttentionSpec(resource_key="ragged", nodes={"node"}, config=ragged_config())
    ragged.apply_yaml_overrides(backend="aiter")
    assert ragged.config.backend is RaggedAttnBackend.AITER


# --- the backend table ---------------------------------------------------------


@pytest.mark.parametrize("backend", [b for b in AttnBackend if b is not AttnBackend.AUTO], ids=str)
def test_every_backend_has_an_entry_whose_builder_imports(backend):
    entry = ATTN_BACKENDS.entry(backend)
    assert entry.backend is backend and issubclass(entry.kv_config, KVConfig)
    assert callable(attn_backends._load(entry.builder))
    assert entry.probe is None or callable(attn_backends._load(entry.probe))


@pytest.mark.parametrize("platform", [CudaPlatform, RocmPlatform, XpuPlatform, CpuPlatform], ids=str)
def test_auto_resolves_to_a_registered_backend_on_every_platform(monkeypatch, rocm, platform):
    if platform is RocmPlatform:
        rocm()
    monkeypatch.setattr(platforms, "resolve_platform", platform)
    for page_size in (None, 16, 128):
        resolved = ATTN_BACKENDS.resolve(AttnBackend.AUTO, page_size)
        assert resolved is not AttnBackend.AUTO and resolved in ATTN_BACKENDS


def test_which_device_types_each_backend_runs_on():
    devices = {entry.backend: entry.devices for entry in ATTN_BACKENDS}
    assert devices.pop(AttnBackend.XPU_PAGED) == ("xpu",)
    assert set(devices.values()) == {()}


def test_options_name_the_backends_that_honour_them():
    assert ATTN_BACKENDS.supporting("sliding_window") == ["flashinfer"]
    assert ATTN_BACKENDS.supporting("sm_scale") == ["flashinfer_mla"]


def test_sliding_window_elsewhere_is_refused():
    with pytest.raises(ValueError, match="sliding_window is unsupported .* use the flashinfer backend"):
        AttentionManager.build(attention_spec(backend=AttnBackend.DENSE, sliding_window=64), kv_info(paged_config()))


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


def test_the_table_refuses_what_it_cannot_serve(stub_registry):
    with pytest.raises(ValueError, match="Unknown stub attention backend"):
        stub_registry.entry(AttnBackend.FLEX)
    with pytest.raises(ValueError, match="cannot register"):
        stub_registry.register(AttnBackendEntry(AttnBackend.DENSE, builder="x:y", kv_config=PagedKVConfig))
    with pytest.raises(ValueError, match="cannot register"):
        stub_registry.register(AttnBackendEntry(AttnBackend.AUTO, builder="x:y", kv_config=PagedKVConfig))
    with pytest.raises(ValueError, match="needs a default policy"):
        AttnBackendRegistry("stub", auto={"rocm": AttnBackend.FLASHINFER})
