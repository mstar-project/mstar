"""KV-cache attention-backend selection: ``KVCacheConfig.attention_backend``
names the ``BatchedCacheManager`` subclass ``create_cache_manager``
instantiates, unknown names are rejected, and the base class is abstract."""

import pytest

from mstar.engine.cache_manager import (
    ATTENTION_BACKENDS,
    BatchedCacheManager,
    DenseGenCacheManager,
    FlashInferCacheManager,
    create_cache_manager,
)
from mstar.engine.kv_store import KVCacheConfig


def _cfg(backend: str | None = None) -> KVCacheConfig:
    kwargs = {} if backend is None else {"attention_backend": backend}
    return KVCacheConfig(
        num_layers=2, num_kv_heads=1, head_dim=8, max_seq_len=64, **kwargs
    )


def _make(cfg: KVCacheConfig) -> BatchedCacheManager:
    return create_cache_manager(
        request_ids=["r0"],
        active_labels_per_request={"r0": "main"},
        kv_cache=None,
        alloc_manager=None,
        buffer_manager=None,
        kv_cache_config=cfg,
        device="cpu",
    )


def test_default_backend_is_flashinfer():
    assert type(_make(_cfg())) is FlashInferCacheManager


def test_dense_gen_backend_selected_by_config(monkeypatch):
    import mstar.engine.cache_manager as cm_mod

    monkeypatch.setattr(cm_mod, "_fa3_unavailable_reason", lambda: None)
    cm = _make(_cfg("dense_gen"))
    assert isinstance(cm, DenseGenCacheManager)
    # The dense backend extends the paged one: prefill, captured graphs, and
    # multi-request batches fall through to the inherited FlashInfer paths.
    assert isinstance(cm, FlashInferCacheManager)


def test_dense_gen_falls_back_to_paged_without_fa3(monkeypatch):
    import mstar.engine.cache_manager as cm_mod

    monkeypatch.setattr(
        cm_mod, "_fa3_unavailable_reason", lambda: "ImportError: mocked"
    )
    cm = _make(_cfg("dense_gen"))
    assert type(cm) is FlashInferCacheManager


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="attention backend"):
        _make(_cfg("nope"))


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        BatchedCacheManager(
            request_ids=[],
            active_labels_per_request={},
            kv_cache=None,
            alloc_manager=None,
            buffer_manager=None,
            kv_cache_config=_cfg(),
            device="cpu",
        )


def test_registry_names():
    assert set(ATTENTION_BACKENDS) == {"flashinfer", "dense_gen"}
