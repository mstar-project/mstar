"""Speculative decoding: ``run_layers`` hands a drafter's taps the residual
stream entering each tapped layer, checked against an unfused reference loop
(the fused add+norm updates the stream in place, so a late copy would be wrong)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, ".")

from mstar.engine.resources.draft.taps import TapResource, TapStep, step_taps  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep  # noqa: E402
from mstar.model.components.decoder_layer import DecoderLayer, run_layers  # noqa: E402
from mstar.model.components.norm import RMSNorm  # noqa: E402

HIDDEN, LAYERS = 16, 4


class _Mix(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        return torch.tanh(self.proj(x))


def _stack():
    torch.manual_seed(0)
    return nn.ModuleList(
        DecoderLayer(_Mix(), _Mix(), RMSNorm(HIDDEN), RMSNorm(HIDDEN)) for _ in range(LAYERS)
    ), RMSNorm(HIDDEN)


def _reference_streams(layers, x):
    """The residual stream entering each layer, and after the last, unfused."""
    streams = [x.clone()]
    residual = x.clone()
    for layer in layers:
        residual = residual + layer.self_attn(layer.input_layernorm(residual))
        residual = residual + layer.mlp(layer.post_attention_layernorm(residual))
        streams.append(residual.clone())
    return streams


def _taps(layers_tapped, rows):
    taps = TapResource(layers_tapped, HIDDEN, torch.float32, torch.device("cpu"))
    step = TapStep(segments=(Segment(1, "main", rows),))
    taps.plan(step, StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False))
    return taps, step


@pytest.mark.parametrize("tapped", [(0,), (1, 3), (0, 2, 4), (4,)])
def test_taps_hold_the_stream_entering_each_layer(tapped):
    layers, norm = _stack()
    x = torch.randn(5, HIDDEN)
    taps, _ = _taps(tapped, 5)
    with torch.no_grad():
        out = run_layers(layers, x, norm, set_cursor=lambda i, attn: None, taps=taps)
        plain = run_layers(layers, x, norm, set_cursor=lambda i, attn: None)
        reference = _reference_streams(layers, x)
    read = taps.read()
    assert read.shape == (5, len(tapped), HIDDEN)
    for col, layer in enumerate(tapped):
        torch.testing.assert_close(read[:, col], reference[layer], rtol=1e-5, atol=1e-5)
    # tapping changes nothing the target computes
    torch.testing.assert_close(out, plain)


def test_a_step_that_declared_no_taps_writes_none():
    layers, norm = _stack()
    taps, step = _taps((1,), 3)
    taps.commit(step, None)
    with torch.no_grad():
        run_layers(layers, torch.randn(3, HIDDEN), norm, set_cursor=lambda i, attn: None, taps=taps)
    with pytest.raises(RuntimeError, match="declared none"):
        taps.read()


def test_taps_reach_the_target_only_on_steps_that_declared_them():
    taps = object()
    with_taps = SimpleNamespace(step=SubmoduleStep(steps={"t": TapStep()}), resources={"t": taps})
    without = SimpleNamespace(step=SubmoduleStep(steps={}), resources={"t": taps})
    assert step_taps(with_taps, "t") is taps
    assert step_taps(without, "t") is None
    assert step_taps(SimpleNamespace(step=None, resources={"t": taps}), "t") is None


def test_taps_verify_without_correction():
    from mstar.engine.resources.draft.taps import TapSpec

    spec = TapSpec("taps", {"LLM"}, layers=(2,), hidden_size=HIDDEN)
    assert spec.verify_refusal(None) is None and spec.async_verify_refusal(None) is None


def test_a_node_with_taps_can_run_on_resident_tables():
    """Taps keep nothing per request, so they never keep a node off resident planning."""
    from mstar.engine.resources.resident import build_planner

    taps = TapResource((1,), HIDDEN, torch.float32, torch.device("cpu"))
    assert taps.resident_refusal() is None
    planner, refused = build_planner(
        "n", {"decode"}, {"taps": taps}, {"taps": frozenset({"n"})}, torch.device("cpu"), max_bs=2, max_span=4,
    )
    assert planner is not None, refused


def test_tap_buffers_grow_until_sealed():
    """A later runner may ask more rows of a slot before any capture."""
    from types import SimpleNamespace

    taps = TapResource((1,), HIDDEN, torch.float32, torch.device("cpu"))
    taps.build_accelerator_graph_buffers([SimpleNamespace(slot=0, num_tokens=8)], 2, 64)
    taps.build_accelerator_graph_buffers([SimpleNamespace(slot=0, num_tokens=32)], 8, 64)
    assert taps._static[0].shape[0] == 32
    taps.seal_capture_buffers()
    with pytest.raises(RuntimeError, match="after sealing"):
        taps.build_accelerator_graph_buffers([SimpleNamespace(slot=0, num_tokens=64)], 16, 64)


def test_a_tap_past_the_stack_raises():
    layers, norm = _stack()
    taps, _ = _taps((LAYERS + 1,), 3)
    with pytest.raises(ValueError, match="taps name layer"):
        run_layers(layers, torch.randn(3, HIDDEN), norm, set_cursor=lambda i, attn: None, taps=taps)
