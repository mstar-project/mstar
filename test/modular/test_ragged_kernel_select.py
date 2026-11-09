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
    AttentionStep,
    BucketKey,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
    RaggedBlockCausalAttentionSpec,
    RaggedCrossAttentionSpec,
    Segment,
    SlotLease,
    StepContext,
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
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule
from mstar.model.submodule_base import NodeInputs

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


def spec(heads=24, head_dim=128, cls=RaggedAttentionSpec, block_size=None, **config):
    config = {"num_kv_heads": heads, "dtype": torch.bfloat16, **config}
    extra = {} if block_size is None else {"block_size": block_size}
    return cls(resource_key="dit_attn", nodes={"dit"},
               config=RaggedAttentionConfig(num_qo_heads=heads, head_dim=head_dim, **config), **extra)


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
        assert entry.rows == ((None, Choice("cudnn")),)
        # what it was chosen over, by the harness that wrote it
        assert all(m["cudnn_us"] < m["flashinfer_fa2_us"] for m in entry.measured.values())


@pytest.mark.parametrize(("heads", "head_dim"), DIT_GEOMETRIES)
@pytest.mark.parametrize("seq_len", [96, 4096, 4192, 4608, 8160])
def test_sm_103_picks_cudnn_for_the_dit_sizes(gpu, heads, head_dim, seq_len):
    """The image DiTs' sizes, and past the last swept one, LTX-2.5's 8160-token video self-attention."""
    gpu(B300)
    manager = build(spec(heads, head_dim))
    assert type(manager) is FlashInferRaggedManager
    assert kernels_module.RAGGED_ATTENTION.row_for(manager._kernel_rows, seq_len) == Choice("cudnn")


@pytest.mark.parametrize("key", [H100, DeviceKey("NVIDIA B200", "sm_100")], ids=["sm_90", "sm_100"])
def test_a_gpu_without_a_table_keeps_flashinfer(gpu, key):
    gpu(key)
    manager = build(spec())
    assert type(manager) is FlashInferRaggedManager
    assert manager._kernel_rows is None
    assert manager._flashinfer_backend == "auto"


def test_a_geometry_without_an_entry_keeps_flashinfer(gpu):
    gpu(B300)
    assert build(spec(heads=16, head_dim=72))._kernel_rows is None


def test_the_table_is_read_for_the_ranks_heads(gpu):
    """The table key is one rank's slice: 48 heads over 2 ranks is the 24-head entry."""
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


@pytest.mark.parametrize("override", [{"backend": "flashinfer"}, {"flashinfer_backend": "fa2"},
                                      {"flashinfer_backend": "auto"}],
                         ids=["backend", "flashinfer_backend", "flashinfer_backend_auto"])
def test_a_named_backend_or_kernel_wins_over_the_table(gpu, override):
    """Any flashinfer_backend set in YAML, ``auto`` included, is FlashInfer's own choice and no table."""
    gpu(B300)
    s = spec()
    apply_yaml_overrides([s], {"resources": {"dit_attn": override}})
    manager = build(s)
    assert manager._kernel_rows is None
    assert manager._flashinfer_backend == override.get("flashinfer_backend", "auto")


@pytest.mark.parametrize("key", [B300, H100], ids=["sm_103", "sm_90"])
def test_backend_cudnn_runs_cudnn_wherever_named(gpu, key):
    gpu(key)
    s = spec()
    apply_yaml_overrides([s], {"resources": {"dit_attn": {"backend": "cudnn"}}})
    assert s.config.backend is RaggedAttnBackend.CUDNN
    assert build(s)._kernel_rows == ((None, Choice("cudnn")),)


@pytest.mark.parametrize(("config", "why"), [
    (dict(num_kv_heads=8, heads=16), "MHA only"),
    (dict(head_dim=72), "padded to 128"),
    (dict(dtype=torch.float32), "bf16 / fp16"),
], ids=["gqa", "padded_head_dim", "fp32"])
def test_backend_cudnn_is_refused_at_build_where_it_cant_run(gpu, config, why):
    gpu(B300)
    s = spec(backend=RaggedAttnBackend.CUDNN, **config)
    with pytest.raises(ValueError, match=f"names backend 'cudnn'.*{why}"):
        build(s)


@pytest.mark.parametrize("cls", [RaggedCrossAttentionSpec, RaggedBlockCausalAttentionSpec],
                         ids=["cross", "block_causal"])
def test_backend_cudnn_is_refused_at_build_off_self_attention(gpu, cls):
    gpu(B300)
    extra = {"block_size": 64} if cls is RaggedBlockCausalAttentionSpec else {}
    with pytest.raises(ValueError, match="runs self-attention only"):
        build(spec(cls=cls, backend=RaggedAttnBackend.CUDNN, **extra))


def test_cudnn_cant_run_a_padded_head_dim(gpu):
    """So a table row naming cudnn there is refused like the explicit backend (``kernel_rows``)."""
    gpu(B300)
    target = kernels_module.RAGGED_ATTENTION.target(
        torch.bfloat16, {"num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 72}, GPU)
    assert "padded to 128" in kernels_module.RAGGED_ATTENTION.unavailable(target)["cudnn"]


def test_cross_attention_reads_no_table(gpu):
    gpu(B300)
    manager = build(spec(cls=RaggedCrossAttentionSpec))
    assert type(manager) is FlashInferRaggedCrossManager
    assert not hasattr(manager, "_kernel_rows")


# --- which wrapper a layout gets ---------------------------------------------------------------


class _StubFlashInfer:
    def __init__(self, workspace_buffer=None, backend="auto", **kwargs):
        self.backend = backend
        self.kwargs = kwargs
        self.planned = []

    def plan(self, cu_seqlens, causal=False):
        self.planned.append(cu_seqlens.tolist())


@pytest.fixture
def stub_flashinfer(monkeypatch):
    monkeypatch.setattr(FlashInferRaggedManager, "wrapper_class", _StubFlashInfer)


def manager_with(rows):
    config = RaggedAttentionConfig(num_qo_heads=24, num_kv_heads=24, head_dim=128)
    return FlashInferRaggedManager(device=torch.device("cpu"), dtype=torch.bfloat16, key="dit_attn", config=config,
                                   kernel_rows=rows)


def plan(manager, step, lease=None):
    rids = sorted({seg.request_id for seg in step.segments})
    manager.plan(step, StepContext(request_ids=rids, graph_walk="image_gen", slot=0 if lease is None else lease.slot,
                                   capture=False, slot_lease=lease))


def planned(manager, *spans, label="main", lease=None, fixed_layout=True):
    """The wrapper ``manager`` plans one segment per span under ``label`` on."""
    segments = tuple(Segment(f"r{i}", label, n) for i, n in enumerate(spans))
    plan(manager, AttentionStep(segments=segments, causal=False, fixed_layout=fixed_layout), lease)
    return manager._current_plan_states[label]


def lease(bs, tokens, slot=1):
    return SlotLease(slot=slot, bucket=BucketKey(graph_walk="image_gen", bs=bs, num_tokens=tokens))


CUDNN_ROWS = ((None, Choice("cudnn")),)


def test_equal_segments_get_the_cudnn_wrapper(stub_flashinfer):
    assert type(planned(manager_with(CUDNN_ROWS), 4608, 4608)) is CudnnRaggedPrefillWrapper


def test_unequal_segments_get_flashinfer(stub_flashinfer):
    w = planned(manager_with(CUDNN_ROWS), 4608, 1024)
    assert type(w) is _StubFlashInfer
    assert w.backend == "auto"


def test_a_row_per_size_class(stub_flashinfer):
    rows = ((256, Choice("flashinfer", {"backend": "fa2"})), (None, Choice("cudnn")))
    m = manager_with(rows)
    short = planned(m, 96, 96, label="caption")
    assert type(short) is _StubFlashInfer and short.backend == "fa2"
    assert type(planned(m, 4192, 4192)) is CudnnRaggedPrefillWrapper


BOUNDED_ROWS = ((256, Choice("flashinfer", {"backend": "fa2"})), (4608, Choice("cudnn")), (None, Choice("flashinfer")))


def test_one_eager_slot_runs_each_layouts_row(stub_flashinfer):
    """Every eager plan picks its row, and a (label, slot) keeps a wrapper per implementation."""
    m = manager_with(BOUNDED_ROWS)
    fa2 = planned(m, 96, 96)
    assert type(fa2) is _StubFlashInfer and fa2.backend == "fa2"
    cudnn = planned(m, 4192, 4192)
    assert type(cudnn) is CudnnRaggedPrefillWrapper
    past = planned(m, 8192)
    assert type(past) is _StubFlashInfer and past.backend == "auto"
    assert planned(m, 96, 96) is fa2
    assert planned(m, 4192, 4192) is cudnn


def test_an_eager_slot_alternates_cudnn_and_flashinfer(stub_flashinfer):
    """Equal segments run cuDNN, the next unequal layout FlashInfer, and equal ones cuDNN again."""
    m = manager_with(CUDNN_ROWS)
    cudnn = planned(m, 64, 64)
    assert type(cudnn) is CudnnRaggedPrefillWrapper and cudnn._layout == (2, 64, False)
    fallback = planned(m, 64, 32)
    assert type(fallback) is _StubFlashInfer and fallback.backend == "auto" and fallback.planned == [[0, 64, 96]]
    again = planned(m, 32, 32)
    assert again is cudnn and cudnn._layout == (2, 32, False)


def test_a_size_past_the_last_row_runs_the_default(stub_flashinfer):
    """A table bounds its measured sizes with a ``"*"`` row naming flashinfer: past them a layout runs what
    a resource without a table does, on the wrapper a layout cuDNN can't take falls back to."""
    m = manager_with(BOUNDED_ROWS)
    past = planned(m, 8192, 8192)
    assert type(past) is _StubFlashInfer and past.backend == "auto"
    assert planned(m, 4096, 1024) is past
    no_table = planned(manager_with(None), 8192, 8192)
    assert type(no_table) is _StubFlashInfer and no_table.backend == past.backend and no_table.kwargs == past.kwargs


def test_a_captured_bucket_picks_its_row_once(stub_flashinfer):
    m = manager_with(BOUNDED_ROWS)
    bucket = lease(2, 2 * 4608)
    w = planned(m, 4608, 4608, lease=bucket)
    assert type(w) is CudnnRaggedPrefillWrapper and w.accelerator_graph
    assert planned(m, 4608, 4608, lease=bucket) is w
    with pytest.raises(ValueError, match="replays the 2 segments of 4608 tokens"):
        planned(m, 4608, lease=bucket)


def test_a_captured_wrapper_is_sized_like_the_flashinfer_one(stub_flashinfer):
    w = planned(manager_with(CUDNN_ROWS), 4608, lease=lease(1, 4608))
    assert type(w) is CudnnRaggedPrefillWrapper
    assert w.accelerator_graph and w.max_total_tokens == 4608 and w.max_num_segments == 1


def test_a_step_without_a_fixed_layout_keeps_a_captured_bucket_on_flashinfer(stub_flashinfer):
    """A captured cuDNN wrapper bakes its layout in, so only a step that fixes its bucket's
    layout captures one; any other step's bucket stays on FlashInfer, and runs cuDNN eagerly."""
    m = manager_with(CUDNN_ROWS)
    assert type(planned(m, 4608, 4608, lease=lease(2, 2 * 4608), fixed_layout=False)) is _StubFlashInfer
    assert type(planned(m, 4608, 4608, fixed_layout=False)) is CudnnRaggedPrefillWrapper
    other_slot = lease(2, 2 * 4608, slot=0)
    assert type(planned(m, 4608, 4608, lease=other_slot, fixed_layout=True)) is CudnnRaggedPrefillWrapper


class _Dit(DenoiseLoopSubmodule):
    """The scaffold's step declaration; a bucket key is its token count."""

    def num_tokens(self, bucket_key):
        return bucket_key


def test_the_scaffolds_step_captures_its_buckets_on_cudnn(stub_flashinfer):
    dit = _Dit(loop_name="denoise_loop", attn_resource_key="dit_attn")
    declared = dit.declare_step("image_gen", ["r0", "r1"], [NodeInputs(resource_step_info=4608)] * 2)
    step = declared.steps["dit_attn"]
    assert step.fixed_layout and [seg.span for seg in step.segments] == [4608, 4608]
    m = manager_with(CUDNN_ROWS)
    plan(m, step, lease(2, 2 * 4608))
    assert type(m._current_plan_states["main"]) is CudnnRaggedPrefillWrapper


def _image_dit(model: str):
    if model == "flux2_klein":
        from mstar.model.flux2_klein.config import Flux2KleinConfig
        from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel

        m = Flux2KleinModel(model_path_hf="test/flux2_klein", skip_weight_loading=True, attention_backend="flashinfer")
        m.set_config(Flux2KleinConfig())
        return m
    from mstar.model.z_image.config import ZImageConfig
    from mstar.model.z_image.z_image_model import ZImageModel

    m = ZImageModel(model_path_hf="test/z_image", skip_weight_loading=True, attention_backend="flashinfer")
    m.set_config(ZImageConfig())
    return m


@pytest.mark.parametrize("model", ["flux2_klein", "z_image"])
def test_the_image_dits_leave_their_kernel_to_the_table(model):
    (resource,) = _image_dit(model).get_node_resources()
    assert resource.config.backend is RaggedAttnBackend.AUTO and resource.config.flashinfer_backend is None


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


@pytest.mark.parametrize("graph", [True, False], ids=["captured", "eager"])
def test_the_cudnn_wrapper_needs_equal_segments(graph):
    w = graph_cudnn() if graph else CudnnRaggedPrefillWrapper(num_qo_heads=24, num_kv_heads=24, head_dim=128,
                                                              device=torch.device("cpu"))
    with pytest.raises(ValueError, match="needs equal-length segments"):
        w.plan(cu(4608, 1024))


def test_the_cudnn_wrapper_runs_mha_only():
    with pytest.raises(ValueError, match="MHA only"):
        CudnnRaggedPrefillWrapper(num_qo_heads=24, num_kv_heads=8, head_dim=128, device=torch.device("cpu"))
