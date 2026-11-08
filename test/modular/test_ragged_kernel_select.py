"""Which kernel a cacheless self-attention runs: the ``ragged_attention`` table
by GPU architecture and head geometry, the backends a config names over it,
and the cuDNN wrapper's layout rules. CPU only: the GPU is a stand-in.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
    RaggedCrossAttentionSpec,
    apply_yaml_overrides,
)
from mstar.engine.resources.attn.ragged import kernels as kernels_module
from mstar.engine.resources.attn.ragged.cudnn_wrappers import CudnnRaggedPrefillWrapper, uniform_segments
from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedCrossManager, FlashInferRaggedManager
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.kernels import Choice, DeviceKey
from mstar.kernels import registry as registry_module
from mstar.kernels.registry import reset_lookups
from mstar.kernels.table import find_entries

GPU = torch.device("cuda", 0)  # a stand-in: nothing here touches it
B300 = DeviceKey("NVIDIA B300 SXM6 AC", "sm_103")
GB300 = DeviceKey("NVIDIA GB300", "sm_103")
H100 = DeviceKey("NVIDIA H100 80GB HBM3", "sm_90")
# (heads, head_dim) of klein 4B, Z-Image Turbo and klein 9B's dit attention
DIT_GEOMETRIES = ((24, 128), (30, 128), (32, 128))


class _FakeCuda:
    def is_cuda(self):
        return True


@pytest.fixture
def gpu(monkeypatch):
    """Point the table lookups at a stand-in GPU; call with the device key."""
    monkeypatch.setattr(kernels_module, "current_platform", _FakeCuda())
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
    monkeypatch.setattr(kernels_module, "_flashinfer_unavailable", lambda target: None)
    reset_lookups()

    def use(key: DeviceKey):
        monkeypatch.setattr(registry_module, "device_key", lambda device=None: key)
        reset_lookups()

    yield use
    reset_lookups()


def spec(heads=24, head_dim=128, cls=RaggedAttentionSpec, **config):
    return cls(resource_key="dit_attn", nodes={"dit"},
               config=RaggedAttentionConfig(num_qo_heads=heads, num_kv_heads=heads, head_dim=head_dim,
                                            dtype=torch.bfloat16, **config))


def build(s, device=GPU):
    return build_resource(s, EngineResourceInfo(device=device, kv_dtype=torch.bfloat16))


def cu(*lens) -> torch.Tensor:
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


# --- the shipped table -------------------------------------------------------------------------


@pytest.mark.parametrize(("heads", "head_dim"), DIT_GEOMETRIES)
def test_sm_103_table_names_cudnn_for_the_dit_geometries(heads, head_dim):
    shape = (("head_dim", head_dim), ("num_kv_heads", heads), ("num_qo_heads", heads))
    for key in (B300, GB300):
        entry = next(find_entries("ragged_attention", key, "bfloat16", shape))
        assert entry.source.endswith("ragged_attention/sm_103.json")
        assert [c.impl for _, c in entry.rows] == ["cudnn"]
        # what it was chosen over, by the harness that wrote it
        assert all(m["cudnn_us"] < m["flashinfer_fa2_us"] for m in entry.measured.values())


@pytest.mark.parametrize(("heads", "head_dim"), DIT_GEOMETRIES)
@pytest.mark.parametrize("seq_len", [96, 4096, 4192, 4608])
def test_sm_103_picks_cudnn_for_the_dit_sizes(gpu, heads, head_dim, seq_len):
    gpu(B300)
    manager = build(spec(heads, head_dim))
    assert type(manager) is FlashInferRaggedManager
    assert kernels_module.choice_for(manager._kernel_rows, seq_len) == Choice("cudnn")


@pytest.mark.parametrize("key", [H100, DeviceKey("NVIDIA B200", "sm_100")], ids=["sm_90", "sm_100"])
def test_a_gpu_without_a_table_keeps_flashinfer(gpu, key):
    gpu(key)
    manager = build(spec())
    assert type(manager) is FlashInferRaggedManager
    assert manager._kernel_rows is None
    assert manager._kwargs["backend"] == "auto"


def test_a_geometry_without_an_entry_keeps_flashinfer(gpu):
    gpu(B300)
    assert build(spec(heads=16, head_dim=72))._kernel_rows is None


def test_the_table_is_read_for_the_ranks_heads(gpu):
    """The shape key is one rank's slice: 48 heads over 2 ranks is the 24-head entry."""
    gpu(B300)
    s = spec(heads=48)
    two_ranks = SimpleNamespace(tp_group=SimpleNamespace(world_size=2), sp_group=SimpleNamespace(world_size=1))
    manager = build_resource(s, EngineResourceInfo(device=GPU, kv_dtype=torch.bfloat16, joint_comm_group=two_ranks))
    assert manager._config.num_qo_heads == 24
    assert [c.impl for _, c in manager._kernel_rows] == ["cudnn"]


def test_rows_that_cant_run_here_keep_flashinfer(gpu, monkeypatch, caplog):
    gpu(B300)
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger=kernels_module.logger.name):
        assert build(spec())._kernel_rows is None
    assert "keeping FlashInfer" in caplog.text


def test_off_gpu_there_is_no_table():
    assert build(spec(), device=torch.device("cpu"))._kernel_rows is None


# --- what a config names wins ------------------------------------------------------------------


@pytest.mark.parametrize("override", [{"backend": "flashinfer"}, {"flashinfer_backend": "fa2"}],
                         ids=["backend", "flashinfer_backend"])
def test_a_named_backend_or_kernel_wins_over_the_table(gpu, override):
    gpu(B300)
    s = spec()
    apply_yaml_overrides([s], {"resources": {"dit_attn": override}})
    manager = build(s)
    assert manager._kernel_rows is None
    assert manager._kwargs["backend"] == override.get("flashinfer_backend", "auto")


@pytest.mark.parametrize("key", [B300, H100], ids=["sm_103", "sm_90"])
def test_backend_cudnn_runs_cudnn_wherever_named(gpu, key):
    gpu(key)
    s = spec()
    apply_yaml_overrides([s], {"resources": {"dit_attn": {"backend": "cudnn"}}})
    assert s.config.backend is RaggedAttnBackend.CUDNN
    assert build(s)._kernel_rows == ((None, Choice("cudnn")),)


def test_cross_attention_never_takes_cudnn(gpu, caplog):
    gpu(B300)
    from mstar.engine.resources.attn.ragged import dense as ragged_dense

    ragged_dense._warn_ragged_cudnn_fallback.cache_clear()
    with caplog.at_level(logging.WARNING, logger=ragged_dense.logger.name):
        manager = build(spec(cls=RaggedCrossAttentionSpec, backend=RaggedAttnBackend.CUDNN))
    assert type(manager) is FlashInferRaggedCrossManager
    assert "self-attention only" in caplog.text
    # and AUTO cross-attention reads no table
    assert not hasattr(build(spec(cls=RaggedCrossAttentionSpec)), "_kernel_rows")


# --- which wrapper a layout gets ---------------------------------------------------------------


class _StubFlashInfer:
    def __init__(self, workspace_buffer=None, backend="auto", **kwargs):
        self.backend = backend
        self.kwargs = kwargs


@pytest.fixture
def stub_flashinfer(monkeypatch):
    monkeypatch.setattr(FlashInferRaggedManager, "wrapper_class", _StubFlashInfer)


def manager_with(rows, device=torch.device("cpu")):
    return FlashInferRaggedManager(device=device, dtype=torch.bfloat16, key="dit_attn",
                                   config=RaggedAttentionConfig(num_qo_heads=24, num_kv_heads=24, head_dim=128),
                                   kernel_rows=rows)


CUDNN_ROWS = ((None, Choice("cudnn")),)


def test_equal_segments_get_the_cudnn_wrapper(stub_flashinfer):
    w = manager_with(CUDNN_ROWS)._build_wrapper("main", 0, cu(4608, 4608))
    assert type(w) is CudnnRaggedPrefillWrapper


def test_unequal_segments_get_flashinfer(stub_flashinfer):
    w = manager_with(CUDNN_ROWS)._build_wrapper("main", 0, cu(4608, 1024))
    assert type(w) is _StubFlashInfer
    assert w.backend == "auto"


def test_a_row_per_size_class(stub_flashinfer):
    rows = ((256, Choice("flashinfer", {"backend": "fa2"})), (None, Choice("cudnn")))
    m = manager_with(rows)
    short = m._build_wrapper("caption", 0, cu(96, 96))
    assert type(short) is _StubFlashInfer and short.backend == "fa2"
    assert type(m._build_wrapper("main", 0, cu(4192, 4192))) is CudnnRaggedPrefillWrapper


def test_a_captured_wrapper_is_sized_like_the_flashinfer_one(stub_flashinfer):
    w = manager_with(CUDNN_ROWS)._build_wrapper("main", 1, cu(4608), accelerator_graph=True,
                                                max_num_segments=1, max_total_tokens=4608)
    assert type(w) is CudnnRaggedPrefillWrapper
    assert w.accelerator_graph and w.max_total_tokens == 4608
    assert w._make_fallback is None


def test_an_eager_cudnn_wrapper_falls_back_for_unequal_segments(stub_flashinfer):
    w = manager_with(CUDNN_ROWS)._build_wrapper("main", 0, cu(64, 64))
    planned = []
    fallback = w._make_fallback()
    assert type(fallback) is _StubFlashInfer
    fallback.plan = lambda cu_seqlens, causal: planned.append((cu_seqlens.tolist(), causal))
    w._make_fallback = lambda: fallback
    w.plan(cu(64, 32))
    assert planned == [([0, 64, 96], False)] and w._on_fallback
    w.plan(cu(32, 32))
    assert not w._on_fallback and w._layout == (2, 32, False)


# --- the cuDNN wrapper's layouts ---------------------------------------------------------------


@pytest.mark.parametrize(("lens", "want"), [
    ((4608,), (1, 4608)),
    ((96, 96, 96), (3, 96)),
    ((4096, 4096, 0, 0), (2, 4096)),  # a graph wrapper's padding
    ((4096, 0, 4096), None),
    ((4096, 4192), None),
    ((0,), None),
])
def test_uniform_segments(lens, want):
    assert uniform_segments(cu(*lens)) == want


def graph_cudnn(max_segments=2, max_tokens=2 * 4608):
    return CudnnRaggedPrefillWrapper(num_qo_heads=24, num_kv_heads=24, head_dim=128, device=torch.device("cpu"),
                                     accelerator_graph=True, max_num_segments=max_segments,
                                     max_total_tokens=max_tokens)


def test_a_captured_cudnn_wrapper_replays_only_its_layout():
    w = graph_cudnn()
    w.plan(cu(4608, 4608))
    w.plan(cu(4608, 4608))  # the same layout: nothing to do
    with pytest.raises(ValueError, match="replays the 2 segments of 4608 tokens"):
        w.plan(cu(4608))
    with pytest.raises(ValueError, match="replays the 2 segments of 4608 tokens"):
        w.plan(cu(4608, 4608), causal=True)


def test_a_captured_cudnn_wrapper_needs_equal_segments():
    with pytest.raises(ValueError, match="needs equal-length segments"):
        graph_cudnn().plan(cu(4608, 1024))


def test_the_cudnn_wrapper_runs_mha_only():
    with pytest.raises(ValueError, match="MHA only"):
        CudnnRaggedPrefillWrapper(num_qo_heads=24, num_kv_heads=8, head_dim=128, device=torch.device("cpu"))
