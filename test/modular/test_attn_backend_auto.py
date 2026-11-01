"""``AUTO`` attention backends, resolved when the resource is built.

``AUTO`` resolves to FlashInfer on every platform, the old default, and a
backend named in a spec or YAML is built as named. Builds use the CPU device
since the managers allocate nothing until their first plan.
"""

import logging

import pytest
import torch

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
from mstar.engine.resources.attn import dense as dense_module
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
from mstar.engine.resources.kv.config import PagedKVConfig, RingKVConfig, RingKVLayerConfig

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


# --- the kernel a verify step runs under device rewind ---------------------------


def rewinding_manager(rewind_kernel: str, reason: str | None, monkeypatch) -> FlashInferManager:
    """A FlashInfer manager of a node that takes its verdicts back on the device, on a GPU where
    paged FA3 can't run for ``reason`` (None: it can). Nothing allocates before a plan."""
    from mstar.engine.resources.attn import fa3_paged

    monkeypatch.setattr(fa3_paged, "fa3_paged_unavailable_reason", lambda *args: reason)
    manager = FlashInferManager(
        kv_cache="kv", device=torch.device("cuda"), dtype=torch.bfloat16, kv_config=paged_config(),
        rewind_kernel=rewind_kernel,
    )
    manager.device_correct = True
    return manager


@pytest.mark.parametrize(
    ("rewind_kernel", "reason", "kernel"),
    [("auto", None, "fa3_paged"), ("auto", "no wheel", "fa2"), ("fa3", None, "fa3_paged"), ("fa2", None, "fa2")],
)
def test_a_verify_step_under_device_rewind_runs_the_kernel_rewind_kernel_names(
    rewind_kernel, reason, kernel, monkeypatch,
):
    manager = rewinding_manager(rewind_kernel, reason, monkeypatch)
    assert manager.corrects_on_device
    assert manager._prefill_backend(32, 4, verify=True) == kernel
    # a step that verifies nothing: FA2 for a short block, else the configured backend
    assert manager._prefill_backend(32, 4) == "fa2"
    assert manager._prefill_backend(4096, 4) == "auto"


def test_rewind_kernel_fa3_is_refused_where_paged_fa3_cant_run(monkeypatch):
    manager = rewinding_manager("fa3", "flash_attn_3 doesn't import", monkeypatch)
    with pytest.raises(ValueError, match="rewind_kernel fa3, but paged FA3 can't run: flash_attn_3 doesn't import"):
        manager._prefill_backend(32, 4, verify=True)


@pytest.mark.parametrize(("rewind_kernel", "corrects"), [("auto", False), ("fa3", True)])
def test_whether_a_node_rewinds_on_the_device_doesnt_depend_on_the_rank(rewind_kernel, corrects, monkeypatch):
    """With FlashInfer's FA3 named, the node rewinds on the device only if rewind_kernel names paged
    FA3 too, whether or not this rank can run it: every rank of a lockstep instance answers alike."""
    for reason in (None, "no wheel"):
        manager = rewinding_manager(rewind_kernel, reason, monkeypatch)
        manager._wrapper_kv_kwargs["backend"] = "fa3"
        assert manager.corrects_on_device is corrects


def test_yaml_names_a_rewind_kernel():
    spec = attention_spec()
    assert spec.config.rewind_kernel == "auto"
    spec.apply_yaml_overrides(rewind_kernel="fa2")
    assert spec.config.rewind_kernel == "fa2"
    with pytest.raises(ValueError, match="rewind_kernel must be one of"):
        spec.apply_yaml_overrides(rewind_kernel="fa4")
