"""KV-cache attention-backend selection: ``KVCacheConfig.attention_backend``
names the ``BatchedCacheManager`` subclass ``create_cache_manager``
instantiates, unknown names are rejected, and the base class is abstract.

Also covers the ``mla_absorb`` kernel predicate and the CUDA-graph capture
decision it drives — the two must agree, or capture builds a paged wrapper that
cannot read the 4-D latent cache."""

import pytest
import torch

from mstar.engine.cache_manager import (
    ATTENTION_BACKENDS,
    BatchedCacheManager,
    DenseGenCacheManager,
    FlashInferCacheManager,
    create_cache_manager,
    mla_kernel_available_for,
)
from mstar.engine.cuda_graph_runner import mla_absorb_capture_blocked
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
    assert set(ATTENTION_BACKENDS) == {"flashinfer", "dense_gen", "mla_absorb"}


# --- mla_absorb: kernel predicate + capture decision ----------------------

# Real Kimi latent dims; flashinfer's MLA wrapper is hard-locked to these.
_REAL_CKV, _REAL_KPE = 512, 64
_CUDA = torch.device("cuda:0")  # constructible without a GPU present


def _mla_cfg(mla_ckv_dim: int | None, head_dim: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=2, num_kv_heads=1, head_dim=head_dim, max_seq_len=64,
        attention_backend="mla_absorb", mla_ckv_dim=mla_ckv_dim,
    )


@pytest.fixture
def sm90(monkeypatch):
    """Pretend we are on a Hopper GPU, without needing one."""
    import mstar.engine.cache_manager as cm_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (9, 0))
    # _mla_kernel_available is functools.cache'd and imports flashinfer; stub the
    # module attribute so the predicate's own logic is what's under test.
    monkeypatch.setattr(
        cm_mod, "_mla_kernel_available",
        lambda ckv, kpe, sm: (ckv, kpe, sm) == (_REAL_CKV, _REAL_KPE, 9),
    )


def test_kernel_predicate_true_only_for_real_dims_on_sm90(sm90):
    cfg = _mla_cfg(_REAL_CKV, _REAL_CKV + _REAL_KPE)
    assert mla_kernel_available_for(cfg, _CUDA) is True


def test_kernel_predicate_false_for_reduced_dims(sm90):
    # The reduced test config: right backend, wrong latent width.
    assert mla_kernel_available_for(_mla_cfg(32, 40), _CUDA) is False


def test_kernel_predicate_false_without_ckv_dim(sm90):
    assert mla_kernel_available_for(_mla_cfg(None, 40), _CUDA) is False


def test_kernel_predicate_is_total_on_cpu(sm90):
    """Must return False, not raise: torch.cuda.get_device_capability rejects a
    CPU device, and the absorbed-SDPA fallback has to stay reachable there."""
    cfg = _mla_cfg(_REAL_CKV, _REAL_CKV + _REAL_KPE)
    assert mla_kernel_available_for(cfg, torch.device("cpu")) is False


def test_capture_not_blocked_for_other_backends(sm90):
    for backend in ("flashinfer", "dense_gen"):
        assert mla_absorb_capture_blocked(_cfg(backend), _CUDA) is None


def test_capture_not_blocked_when_kernel_serves_the_dims(sm90):
    cfg = _mla_cfg(_REAL_CKV, _REAL_CKV + _REAL_KPE)
    assert mla_absorb_capture_blocked(cfg, _CUDA) is None


@pytest.mark.parametrize(
    "ckv, head_dim", [(32, 40), (None, 40), (_REAL_CKV, _REAL_CKV + 128)]
)
def test_capture_blocked_when_absorbed_falls_back_to_sdpa(sm90, ckv, head_dim):
    """No wrapper is planned on the SDPA path, so a captured graph could only
    hold a paged wrapper — capture must be cancelled, not attempted."""
    reason = mla_absorb_capture_blocked(_mla_cfg(ckv, head_dim), _CUDA)
    assert reason is not None
    assert "mla_absorb" in reason and "eager" in reason


def test_capture_blocked_on_cpu_device(sm90):
    cfg = _mla_cfg(_REAL_CKV, _REAL_CKV + _REAL_KPE)
    assert mla_absorb_capture_blocked(cfg, torch.device("cpu")) is not None
