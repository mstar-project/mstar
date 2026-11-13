"""The shared layers call state handles where a model declares the array, and resources where it doesn't.

Unported models keep running on these same layers, so a layer picks its path at bind from what it receives, and a layer
with no handle must behave exactly as it did. A ported layer names itself with its own model layer id, so it needs no
cursor.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, ".")

import pytest
import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.accelerator_graph_config import PiecewisePackedConfig
from mstar.engine.resources import AttentionStep, Segment, SubmoduleStep
from mstar.engine.resources.convenience import AttentionCallable
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Block,
    Causal,
    CausalConv,
    Entry,
    GatedDeltaNet,
    KVEntry,
    Positions,
    Read,
    Rope,
    TensorType,
    Value,
)
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle, state_steps
from mstar.engine.state.calls import ArrayHandle
from mstar.model.components import Attention
from mstar.model.components.decoder_layer import DecoderLayer, eager_frame, no_cursor, run_layers
from mstar.model.components.distributed.attention import ParallelAttention
from mstar.model.components.distributed.linear_attn import ParallelGatedDeltaNet
from mstar.model.components.qwen3_vit import VisionAttention, VisionBlock

HIDDEN, HEADS, DIM, TOKENS = 16, 2, 8, 3


class _StubKV:
    """Hands back pages and records what the layer wrote."""

    def __init__(self):
        self.pages = torch.zeros(4, 8, 2, HEADS, DIM)
        self.writes = []

    def layer_view(self, i):
        return self.pages[i]

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.writes.append(layer_idx)

    def opens_label(self, rid, label):
        return True


class _StubAttn:
    """Attends by returning the query, and records the layer it ran on."""

    requires_kv_write = True

    def __init__(self):
        self.runs = []

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        self.runs.append(layer_idx)
        return q.clone()


class _StubPositions:
    """Rotates by marking the query, so a test can tell it ran."""

    def apply_qk(self, q, k, label=None, rope_dtype=None):
        return q + 1, k


def _begin(*handles) -> None:
    step = SubmoduleStep(segments=[Segment(1, "main", TOKENS)], state={h.name: ArrayStep() for h in handles})
    step.set_ctx(SimpleNamespace(replays=False, slot_lease=None, capture=False, padded_request_ids=(1,)))
    for handle in handles:
        handle.begin_step(step)


def _kv_array(positions=None) -> Array:
    return Array(
        "kv", nodes={"LLM"}, layers={0, 1, 2, 3}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=8,
        reads={"attn": Read(mask=Causal(), op=Attend(q_heads=HEADS))},
        positions=positions,
    )


def _kv_handle(positions=None) -> ArrayHandle:
    array = _kv_array(positions)
    resources = {"kv": _StubKV(), "attn": _StubAttn(), "rope": _StubPositions()}
    return build_handle(array, resources, _resolve([array], StateSetup()))


def _attention(pos_key="rope", cls=Attention, **kwargs):
    return cls(hidden_size=HIDDEN, num_heads=HEADS, num_kv_heads=HEADS, head_dim=DIM, attn_key="attn", kv_key="kv",
               pos_key=pos_key, **kwargs)


def _parallel(pos_key="rope", **kwargs):
    return _attention(pos_key, ParallelAttention, comm_group=CommGroup.trivial(), **kwargs)


@pytest.mark.parametrize("build", [_attention, _parallel], ids=["Attention", "ParallelAttention"])
class TestAttentionPaths:
    def test_a_layer_with_a_handle_adds_and_reads_under_its_own_id(self, build):
        layer = build(layer=2)
        h = _kv_handle(Positions(rope=Rope()))
        layer.bind_state({"kv": h})
        _begin(h)
        layer(torch.randn(TOKENS, HIDDEN))
        assert h._kv.writes == [2] and h.first_read.runs == [2]

    def test_a_layer_with_a_position_key_rotates_through_the_handle(self, build):
        layer = build(layer=2)
        h = _kv_handle(Positions(rope=Rope()))
        h.positions.apply_qk = Mock(side_effect=lambda q, k, **kwargs: (q, k))
        layer.bind_state({"kv": h})
        _begin(h)
        layer(torch.randn(TOKENS, HIDDEN))
        h.positions.apply_qk.assert_called_once()

    def test_a_layer_with_no_position_key_does_not_rotate(self, build):
        layer = build(pos_key=None, layer=2)
        h = _kv_handle()
        layer.bind_state({"kv": h})
        _begin(h)
        layer(torch.randn(TOKENS, HIDDEN))
        assert h._kv.writes == [2], "a global layer still adds and reads"

    def test_a_layer_with_no_handle_keeps_its_resources_path(self, build):
        layer = build(layer=None)
        kv, attn = _StubKV(), _StubAttn()
        resources = {"kv": kv, "attn": attn}
        kv.default_label = "main"
        kv.layer_view = lambda i=None: kv.pages[0]
        layer.bind_resources(resources)
        layer.bind_state({})
        assert layer.state is None
        layer._apply_rope = Mock(side_effect=lambda q, k, label: (q, k))
        layer(torch.randn(TOKENS, HIDDEN))
        layer._apply_rope.assert_called_once()
        assert isinstance(layer.attend, AttentionCallable) and attn.runs == [None]


class _StubLinear:
    """Runs the two kernels as identities on the shapes the layer expects."""

    def __init__(self):
        self.calls = []

    def run_conv(self, x, conv_layer, weight, bias, label):
        self.calls.append("conv")
        return x.clone()

    def run(self, q, k, v, a, b, state_layer, a_log, dt_bias, label):
        self.calls.append("delta")
        return v.clone()


class _StubPool:
    """Hands back a block per field and layer."""

    def block(self, name, i):
        return torch.zeros(4, 2, DIM, DIM) if name == "state" else torch.zeros(4, 64, 3)


def _gdn(**kwargs) -> ParallelGatedDeltaNet:
    layer = ParallelGatedDeltaNet(
        hidden_size=HIDDEN, num_k_heads=HEADS, num_v_heads=HEADS, head_k_dim=DIM, head_v_dim=DIM, conv_kernel_size=4,
        state_key="gdn", comm_group=CommGroup.trivial(), **kwargs,
    )
    nn.init.normal_(layer.in_proj_fused.weight)
    return layer


def _gdn_handle() -> ArrayHandle:
    entry = Entry({
        "state": TensorType((HEADS, DIM, DIM), dtype=torch.float32, shard=0),
        "conv": TensorType((3 * HEADS * DIM, 3), dtype=None, shard=0),
    })
    array = Value("gdn", nodes={"LLM"}, layers={1, 2}, entry=entry,
                  updates={"conv": CausalConv(), "linear": GatedDeltaNet()})
    resolved = _resolve([_kv_array(), array], StateSetup())
    return build_handle(array, {"gdn": _StubPool(), "linear": _StubLinear()}, resolved)


def test_a_delta_net_with_a_handle_runs_both_updates_under_its_own_id():
    layer = _gdn(layer=2)
    h = _gdn_handle()
    layer.bind_state({"gdn": h})
    _begin(h)
    out = layer(torch.randn(TOKENS, HIDDEN))
    assert h._runners[GatedDeltaNet].calls == ["conv", "delta"] and out.shape == (TOKENS, HIDDEN)
    assert h._updated == {("conv", 2), ("state", 2)}


class _ViTConfig:
    hidden_size = 16
    intermediate_size = 32
    num_heads = 2
    head_dim = 8
    hidden_act = "gelu_pytorch_tanh"
    rope_theta = 10000.0


class _StubRagged:
    """Attends by adding, and records that it was called."""

    def __init__(self):
        self.runs = 0

    def run(self, q, k, v, label=None):
        self.runs += 1
        return q + k + v


def _vision_handle() -> ArrayHandle:
    array = Array("tokens", nodes={"vit"}, layers={0, 1}, entry=KVEntry(heads=2, head_dim=8), max_len=0,
                  reads={"vision": Read(mask=Block(), op=Attend(q_heads=2))})
    return build_handle(array, {"vision": _StubRagged()}, _resolve([array], StateSetup()))


def _vision_inputs():
    return torch.randn(TOKENS, 16), torch.ones(TOKENS, 8), torch.zeros(TOKENS, 8)


def test_a_vision_block_with_a_handle_adds_and_reads_and_leaves_no_hold():
    block = VisionBlock(_ViTConfig, "vision", array="tokens", layer=1)
    h = _vision_handle()
    block.attn.bind_state({"tokens": h})
    _begin(h)
    block(*_vision_inputs())
    assert h.first_read.runs == 1 and h._held is None


def test_a_vision_block_with_no_handle_keeps_its_resource():
    block = VisionBlock(_ViTConfig, "vision")
    ragged = _StubRagged()
    block.attn.bind_resources({"vision": ragged})
    block.attn.bind_state({})
    block(*_vision_inputs())
    assert ragged.runs == 1


def test_a_vision_attention_with_neither_says_what_is_missing():
    attn = VisionAttention(_ViTConfig, "vision")
    with pytest.raises(ValueError, match="declare a RaggedAttentionSpec under 'vision'"):
        attn(*_vision_inputs())


def _region(**kwargs) -> PiecewisePackedConfig:
    return PiecewisePackedConfig(
        capture_fn=lambda inp: {}, make_static_inputs=lambda shape: {}, total_tokens=[8], **kwargs,
    )


def test_a_region_on_an_array_declares_the_step_its_attn_key_did():
    h = _vision_handle()
    on_array = state_steps(_region(array="tokens").region_step([1, 2], [3, 5]), {"tokens": h})
    on_key = _region(attn_key="vision").region_step([1, 2], [3, 5])
    assert on_array.steps == on_key.steps
    assert on_array.segments == on_key.segments, "with_segment_steps reads the step's segments"
    assert on_array.steps["vision"] == AttentionStep(segments=on_key.segments, causal=False)


def test_a_one_row_region_on_an_array_lays_every_segment_on_its_one_row():
    config = _region(array="tokens", lease_before_step=True, one_row_per_step=True, capture_batch_sizes=[1])
    step = config.region_step([7], [3, 5])
    assert step.segments == [Segment(7, "main", 3), Segment(7, "main", 5)]


@pytest.mark.parametrize("kwargs", [{"attn_key": "a"}, {"declare_step": lambda rids, lens: None}])
def test_a_region_on_an_array_may_not_also_name_another_step(kwargs):
    with pytest.raises(ValueError, match="array declares the region's step"):
        _region(array="tokens", **kwargs)


class _NoAttend(nn.Module):
    """Has no cursor to move, as a layer on a state handle has none."""

    def forward(self, h):
        return h


@pytest.mark.parametrize("make", [
    lambda: Attention(hidden_size=HIDDEN, num_heads=HEADS, num_kv_heads=HEADS, head_dim=DIM, layer=3),
    lambda: ParallelAttention(hidden_size=HIDDEN, num_heads=HEADS, num_kv_heads=HEADS, head_dim=DIM, layer=3),
], ids=["Attention", "ParallelAttention"])
def test_a_layer_holds_its_id_as_a_cpu_tensor_that_a_move_leaves_alone(make):
    """A compiled frame would otherwise guard on the int, and a buffer would move to the GPU and sync at capture."""
    layer = make().to("meta").layer
    assert layer.ndim == 0 and layer.dtype == torch.int64 and layer.device.type == "cpu" and int(layer) == 3


def test_a_stack_with_no_cursor_never_reaches_for_one():
    layers = nn.ModuleList(
        DecoderLayer(_NoAttend(), nn.Identity(), _Norm(), _Norm()) for _ in range(2)
    )
    out = run_layers(layers, torch.ones(2, 4), _Norm(), no_cursor)
    assert out.shape == (2, 4)


@torch.no_grad()
def test_a_stack_with_no_cursor_keeps_its_loop_eager_with_no_break_in_its_layers():
    """A model's stack traced whole would hand every layer one graph, and `run_layers` promises a frame per layer."""
    torch._dynamo.reset()
    layers = nn.ModuleList(
        DecoderLayer(_NoAttend(), nn.Identity(), _Norm(), _Norm()) for _ in range(3)
    )
    graphs = []

    def backend(graph_module, inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def stack(x):
        return run_layers(layers, x * 2, _Norm(), no_cursor)

    torch._dynamo.optimize(backend, dynamic=False)(stack)(torch.ones(2, 4))
    assert len(graphs) > 1, "the loop should break at each layer's cursor"


class _Norm(nn.Module):
    """Adds into the residual stream without a kernel."""

    def forward(self, x):
        return x

    def forward_residual(self, x, residual, comm_group=None):
        return x, x if residual is None else x + residual


# The frame around a stack


@torch.no_grad()
def test_a_frame_marked_eager_is_skipped_and_what_it_calls_compiles():
    """Dynamo would otherwise compile a frame around `run_layers` and recompile it on every shape for nothing."""
    torch._dynamo.reset()

    def inner(x):
        return x * 2

    @eager_frame
    def marked(x):
        return inner(x + 1)

    compiled_ops = []

    def backend(graph_module, inputs):
        compiled_ops.extend(node.target.__name__ for node in graph_module.graph.nodes if node.op == "call_function")
        return graph_module.forward

    optimized = torch._dynamo.optimize(backend)(marked)
    assert optimized(torch.ones(2)).tolist() == [4.0, 4.0]
    assert "add" not in compiled_ops, "the marked frame's own work should run eagerly"
    assert "mul" in compiled_ops, "what the marked frame calls should still compile"
    with torch.compiler.set_stance("fail_on_recompile"):
        assert optimized(torch.ones(2)).tolist() == [4.0, 4.0], "a capture's stance should hold on the marked frame"
