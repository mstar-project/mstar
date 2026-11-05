"""The ragged resource's ``auto`` backend on uniform segments: which manager the table builds per
GPU, what overrides it, and which kernel each plan of the wrapper picks. No GPU: the GPU and
FlashInfer are stand-ins; parity of the kernels is in ``test_ragged_attention_select_gpu.py``."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.resources import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.attn.ragged.base import RaggedAttnManager
from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedManager
from mstar.engine.resources.attn.ragged.select import (
    RAGGED_ATTENTION,
    SelectedRaggedManager,
    SelectedRaggedPrefillWrapper,
    table_rows,
)
from mstar.engine.resources.attn.workspace import Workspace
from mstar.kernels import Choice, DeviceKey, registry, table

B300 = DeviceKey("NVIDIA B300 SXM6 AC", "sm_103")
H100 = DeviceKey("NVIDIA H100 80GB HBM3", "sm_90")
GPU = torch.device("cuda", 0)
# one rank's heads of the image DiTs, all with head dim 128: klein 4B, Z-Image, klein 9B
DIT_HEADS = {"klein 4B": 24, "Z-Image": 30, "klein 9B": 32}
# klein's [512 text | 4096 image], Z-Image's main layers at a 64-token caption, its image and caption refiners
DIT_SEGMENTS = (4608, 4160, 4096, 64)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(table.TABLES_ENV, raising=False)
    registry.reset_lookups()
    yield
    registry.reset_lookups()


@pytest.fixture
def on(monkeypatch):
    """Lookups as if on the given GPU, reading the package's shipped tables."""
    def use(key: DeviceKey):
        monkeypatch.setattr(registry, "_device", lambda device: GPU)
        monkeypatch.setattr(registry, "device_key", lambda device: key)
        registry.reset_lookups()
    return use


def _config(heads=24, **kw) -> RaggedAttentionConfig:
    return RaggedAttentionConfig(num_qo_heads=heads, num_kv_heads=heads, head_dim=128,
                                 max_segments_per_request=1, dtype=torch.bfloat16,
                                 uniform_segments=kw.pop("uniform_segments", True), **kw)


def _build(config: RaggedAttentionConfig):
    spec = RaggedAttentionSpec(resource_key="dit_attn", nodes={"dit"}, config=config)
    info = SimpleNamespace(device=torch.device("cpu"), joint_comm_group=None, head_shards=1, kv_dtype=None)
    return RaggedAttnManager.build(spec, info)


@pytest.mark.parametrize("model", DIT_HEADS)
def test_sm103_picks_cudnn_for_the_image_dits(on, model):
    on(B300)
    rows = table_rows(DIT_HEADS[model], DIT_HEADS[model], 128, torch.bfloat16, GPU)
    assert rows is not None
    for seq in DIT_SEGMENTS:
        assert RAGGED_ATTENTION.row_for(rows, seq).impl == "cudnn", seq
    manager = _build(_config(DIT_HEADS[model]))
    assert type(manager) is SelectedRaggedManager
    assert manager._kwargs["rows"] == rows


@pytest.mark.parametrize("key", [H100, DeviceKey("NVIDIA B200", "sm_100")], ids=["sm_90", "sm_100"])
def test_a_gpu_without_a_table_keeps_flashinfer(on, key):
    on(key)
    assert table_rows(24, 24, 128, torch.bfloat16, GPU) is None
    assert type(_build(_config())) is FlashInferRaggedManager


def test_heads_no_table_has_keep_flashinfer_on_sm103(on):
    on(B300)
    assert table_rows(16, 16, 128, torch.bfloat16, GPU) is None
    assert type(_build(_config(16))) is FlashInferRaggedManager


@pytest.mark.parametrize("override", [
    dict(backend=RaggedAttnBackend.FLASHINFER),
    dict(flashinfer_backend="fa2"),
    dict(uniform_segments=False),
], ids=["backend", "flashinfer_backend", "varlen layout"])
def test_an_explicit_backend_or_a_varlen_layout_wins_over_the_table(on, override):
    on(B300)
    assert type(_build(_config(**override))) is FlashInferRaggedManager


def test_the_yaml_override_reaches_the_built_manager(on):
    on(B300)
    spec = RaggedAttentionSpec(resource_key="dit_attn", nodes={"dit"}, config=_config())
    spec.apply_yaml_overrides(backend="flashinfer")
    info = SimpleNamespace(device=torch.device("cpu"), joint_comm_group=None, head_shards=1, kv_dtype=None)
    assert type(RaggedAttnManager.build(spec, info)) is FlashInferRaggedManager


def test_the_image_dits_declare_uniform_segments():
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
    from mstar.model.z_image.z_image_model import ZImageModel

    for cls, cfg in ((Flux2KleinModel, dict(num_attention_heads=24, attention_head_dim=128)),
                     (ZImageModel, dict(n_heads=30, head_dim=128))):
        model = cls(skip_weight_loading=True, attention_backend="flashinfer")
        model._config = SimpleNamespace(transformer=SimpleNamespace(**cfg))
        (spec,) = model.get_node_resources()
        assert spec.config.uniform_segments and spec.config.backend is RaggedAttnBackend.AUTO
        assert cls(skip_weight_loading=True, attention_backend="sdpa").get_node_resources() == []


# --- the wrapper: a kernel per plan, FlashInfer stood in --------------------------------------

@pytest.fixture
def fake_flashinfer(monkeypatch):
    native = Mock(side_effect=lambda *a, **kw: SimpleNamespace(plan=Mock(), run=Mock(), kw=kw))
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=native))
    return native


ROWS = ((64, Choice("flashinfer", {"backend": "fa2"})), (None, Choice("cudnn")))


def _wrapper(graph=False, segments=2, tokens=2 * 4608):
    kw = dict(accelerator_graph=True, max_num_segments=segments, max_total_tokens=tokens) if graph else {}
    return SelectedRaggedPrefillWrapper(
        lambda: Workspace(torch.empty(8, dtype=torch.uint8), "test"), 24, 24, 128, rows=ROWS,
        device=torch.device("cpu"), **kw,
    )


def _cu(*lens) -> torch.Tensor:
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return torch.tensor(cu, dtype=torch.int32)


def test_each_plan_takes_its_rows_kernel(fake_flashinfer):
    w = _wrapper()
    w.plan(_cu(4608, 4608))
    assert w.impl == "cudnn" and w._dense.layout == (2, 4608, False) and w.num_segments == 2
    w.plan(_cu(64))
    assert w.impl == "flashinfer:fa2"
    assert fake_flashinfer.call_args.kwargs["backend"] == "fa2"
    w.plan(_cu(4160), causal=True)
    assert w.impl == "cudnn" and w._dense.layout == (1, 4160, True)


def test_uneven_segments_run_flashinfer(fake_flashinfer):
    w = _wrapper()
    w.plan(_cu(4608, 4096))
    assert w.impl == "flashinfer:auto"


def test_a_captured_wrapper_keeps_its_first_kernel_and_layout(fake_flashinfer):
    w = _wrapper(graph=True)
    w.plan(_cu(4608, 4608))
    assert w.impl == "cudnn"
    w.plan(_cu(4608, 4608))
    with pytest.raises(ValueError, match="captured on cuDNN"):
        w.plan(_cu(4096, 4096))
    with pytest.raises(ValueError, match="captured on cuDNN"):
        w.plan(_cu(4608, 4000))


def test_a_wrapper_captured_on_flashinfer_stays_on_it(fake_flashinfer):
    w = _wrapper(graph=True, tokens=2 * 64)
    w.plan(_cu(64, 64))
    assert w.impl == "flashinfer:fa2"
    w.plan(_cu(100, 28))
    assert w.impl == "flashinfer:fa2"
    assert fake_flashinfer.call_count == 1
