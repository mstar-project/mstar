"""A compressor source keeps partial groups in the engine's per-request Value."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import build_specs, node_handles, state_steps
from mstar.model.components.deepseek_v41.compressor import DeepSeekV41Compressor
from mstar.model.components.deepseek_v41.compressor_source import (
    DeepSeekV41CompressorSource,
    compressor_partial_value,
    load_compressor_weights,
)


def _weights(model: DeepSeekV41Compressor, layer: int, seed: int = 300) -> list[tuple[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    return [
        (f"layers.{layer}.attn.compressor.{name}", torch.randn(value.shape, generator=gen, dtype=value.dtype))
        for name, value in model.named_parameters()
    ]


def test_loader_accepts_ratio_one_bf16_and_ratio_two_fp32_without_scales():
    for layer, ratio in ((20, 1), (2, 2)):
        model = DeepSeekV41Compressor(16, 8, ratio)
        tensors = _weights(model, layer)
        load_compressor_weights(model, layer, tensors)
        assert [name for name, _ in tensors] == [
            f"layers.{layer}.attn.compressor.{name}" for name, _ in model.named_parameters()
        ]
        for (name, got), (key, want) in zip(model.named_parameters(), tensors, strict=True):
            assert key == f"layers.{layer}.attn.compressor.{name}"
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert model.wkv.weight.dtype == (torch.bfloat16 if ratio == 1 else torch.float32)
        assert not any("scale" in name for name, _ in tensors)


def test_ratio_one_source_needs_no_partial_value():
    model = DeepSeekV41Compressor(16, 8, 1).eval()
    gen = torch.Generator().manual_seed(301)
    with torch.no_grad():
        model.wkv.weight.copy_(torch.randn(model.wkv.weight.shape, generator=gen, dtype=model.wkv.weight.dtype))
    source = DeepSeekV41CompressorSource(20, model, "unused")
    source.bind_state({})
    tokens = torch.randn(1, 2, 16, generator=gen, dtype=torch.bfloat16)
    expected, state = model(tokens, 0)
    assert state is None
    torch.testing.assert_close(source(tokens, 0), expected, rtol=0, atol=0)


def test_ratio_two_source_refuses_mixed_position_batch_without_a_per_row_schedule():
    source = DeepSeekV41CompressorSource(2, _loaded_compressor(1.0), "partial")
    source.state = object()
    with pytest.raises(ValueError, match="one request per step"):
        source(torch.zeros(2, 1, 16, dtype=torch.bfloat16), 1)


@pytest.mark.parametrize("fault,expected", [
    ("missing", "missing DeepSeek compressor weights"),
    ("duplicate", "unexpected or duplicate"),
    ("wrong_layer", "wrong DeepSeek compressor checkpoint key"),
    ("wrong_dtype", "expected"),
    ("scale", "unexpected or duplicate"),
])
def test_loader_refuses_bad_or_duplicate_compressor_weights(fault, expected):
    model = DeepSeekV41Compressor(16, 8, 2)
    tensors = _weights(model, 2)
    if fault == "missing":
        tensors.pop()
    elif fault == "duplicate":
        tensors.append(tensors[0])
    elif fault == "wrong_layer":
        name, value = tensors[0]
        tensors[0] = (name.replace("layers.2.", "layers.8."), value)
    elif fault == "wrong_dtype":
        name, value = tensors[0]
        tensors[0] = (name, value.bfloat16())
    else:
        tensors.append(("layers.2.attn.compressor.wkv.scale", torch.ones(1)))
    with pytest.raises(ValueError, match=expected):
        load_compressor_weights(model, 2, tensors)


def _loaded_compressor(scale: float) -> DeepSeekV41Compressor:
    golden = json.loads((Path(__file__).parent / "data/deepseek_v41_compressor_golden.json").read_text())
    model = DeepSeekV41Compressor(16, 8, 2).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.tensor(golden["weights"][f"two.{name}"], dtype=parameter.dtype) * scale)
    return model


def test_two_sources_keep_interleaved_requests_in_distinct_engine_slots():
    value = compressor_partial_value("partial", node="LLM", source_layers=frozenset({2, 8}), ratio=2, head_dim=8)
    # Sink plus two live slots: c must reuse a's released slot.
    (spec,) = build_specs([value], kernel_settings={"partial": {"max_slots": 3}})
    pool = RecurrentStatePool(torch.device("cpu"), spec.config)
    for rid in ("a", "b"):
        pool.ingest_request(rid)
    handle = node_handles(SimpleNamespace(declare_state=lambda: [value]), "LLM", {"partial": pool})["partial"]
    sources = {layer: DeepSeekV41CompressorSource(layer, _loaded_compressor(scale), "partial")
               for layer, scale in ((2, 1.0), (8, 0.5))}
    references = {layer: _loaded_compressor(scale) for layer, scale in ((2, 1.0), (8, 0.5))}
    for source in sources.values():
        source.bind_state({"partial": handle})
    assert pool._start is not None
    assert all(torch.isneginf(pool._start[("score", i)]).all() for i in (0, 1))

    golden = json.loads((Path(__file__).parent / "data/deepseek_v41_compressor_golden.json").read_text())
    inputs = torch.tensor(golden["input"], dtype=torch.bfloat16)
    carried: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def run(rid: str, x: torch.Tensor, start: int, *, fresh: bool = False) -> None:
        # The Value writes one partial-group entry even for a multi-token prefill.
        segment = Segment(rid, "main", 1)
        step = state_steps(SubmoduleStep(segments=[segment], state={"partial": ArrayStep()}), {"partial": handle})
        plan = step.steps["partial"]
        assert isinstance(plan, RecurrentStep) and tuple(plan.segments) == (segment,)
        ctx = StepContext(request_ids=[rid], graph_walk="decode", slot=0, capture=False)
        assert pool.admit(plan, ctx).ok
        pool.plan(plan, ctx)
        step.set_ctx(ctx)
        step.begin_calls()
        if fresh:
            for layer in (2, 8):
                fields = handle.read(layer).fields
                assert torch.count_nonzero(fields["kv"]) == 0
                assert torch.isneginf(fields["score"]).all()
        for layer in (2, 8):
            actual = sources[layer](x, start)
            expected, state = references[layer](x, start, carried.get((rid, layer)))
            assert state is not None
            carried[rid, layer] = state
            assert (actual is None) == (expected is None)
            if expected is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        step.check_calls()
        pool.commit(plan, ctx)
        step.end_calls()

    run("a", inputs[0:1, :5], 0, fresh=True)
    run("b", inputs[1:2, :1], 0, fresh=True)
    run("a", inputs[0:1, 5:6], 5)
    run("b", inputs[1:2, 1:2], 1)
    run("a", inputs[0:1, 6:7], 6)
    run("b", inputs[1:2, 2:3], 2)
    run("a", inputs[0:1, 7:8], 7)
    run("b", inputs[1:2, 3:4], 3)

    pool.remove_request("a")
    pool.ingest_request("c")
    run("c", inputs[0:1, :1], 0, fresh=True)
    run("c", inputs[0:1, 1:2], 1)
    run("b", inputs[1:2, 4:5], 4)
