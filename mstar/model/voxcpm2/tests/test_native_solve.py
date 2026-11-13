"""Adapter coverage; the real-weight capture test needs H200 and VOXCPM2_PATH."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mstar.utils.voxcpm2_solve import PatchSolve, pack_gate, prepare, solve


def test_gate_packing_preserves_every_projection_row():
    rows = torch.arange(8192, dtype=torch.int32)[:, None].expand(-1, 1024)
    packed = pack_gate(rows)
    expected = torch.stack((torch.arange(4096).reshape(512, 8),
                            torch.arange(4096, 8192).reshape(512, 8)), dim=1).flatten()
    assert packed.is_contiguous()
    assert torch.equal(packed[:, 0], expected)
    assert torch.equal(packed, expected[:, None].expand_as(packed))
    restored = packed.reshape(512, 2, 8, 1024).transpose(0, 1).reshape(8192, 1024)
    assert torch.equal(restored, rows)


def test_cpu_model_does_not_load_cuda_extension():
    model = SimpleNamespace(config=SimpleNamespace(dit=None),
                            lm_to_dit_proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)))
    assert prepare(model) is None


def test_worker_dtype_cast_preserves_materialized_rotary_values():
    obj = PatchSolve.__new__(PatchSolve)
    torch.nn.Module.__init__(obj)
    table = torch.linspace(0, 1, 128, dtype=torch.float32)
    obj.register_buffer("cosine", table.clone(), persistent=False)
    obj.register_buffer("sine", table.clone(), persistent=False)
    obj.register_buffer("w0", torch.ones(1), persistent=False)
    obj.bfloat16()
    assert obj.w0.dtype == torch.bfloat16
    assert obj.cosine.dtype == obj.sine.dtype == torch.float32
    assert torch.equal(obj.cosine, table) and torch.equal(obj.sine, table)


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_dispatch_uses_actual_tensor_layout_and_batch(batch):
    with FakeTensorMode():
        obj = PatchSolve.__new__(PatchSolve)
        torch.nn.Module.__init__(obj)
        obj.register_buffer("w0", torch.empty((1024, 2048), device="cuda", dtype=torch.bfloat16))
        inputs = [torch.empty(s, device="cuda", dtype=torch.bfloat16)
                  for s in ((batch, 2048), (batch, 2048), (batch, 4, 64), (batch, 64, 4), (batch,), (11,))]
        inputs[4] = inputs[4].float()
        assert obj.supports(*inputs) == (batch == 1)
        if batch != 1:
            return
        changed = list(inputs)
        changed[0] = torch.empty((2, 2048), device="cuda", dtype=torch.bfloat16)
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[2] = torch.empty((batch, 64, 4), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[5] = torch.empty((9,), device="cuda", dtype=torch.bfloat16)
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[4] = inputs[4].bfloat16()
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[0] = inputs[0].cpu()
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[0] = torch.empty((), device="cuda", dtype=torch.bfloat16)
        assert not obj.supports(*changed)
        output = solve(*inputs, [], [], [], inputs[0], inputs[0], 1e-5)
        assert output.shape == (batch, 4, 64)
        assert output.dtype == torch.bfloat16 and output.device == inputs[0].device
        assert output.is_contiguous()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CUTLASS_ROOT")
    or not os.environ.get("VOXCPM2_PATH"),
    reason="needs H200, pinned CUTLASS_ROOT and cached VOXCPM2_PATH",
)
@torch.inference_mode()
def test_real_weights_compile_capture_and_fallback(monkeypatch, record_property):
    from mstar.model.voxcpm2.components.local import cfm_t_span
    from mstar.model.voxcpm2.config import VoxCPM2Config
    from mstar.model.voxcpm2.voxcpm2_model import load_generator

    if "H200" not in torch.cuda.get_device_name():
        pytest.skip("the measured specialization is H200-only")
    checkpoint = Path(os.environ["VOXCPM2_PATH"])
    if not (checkpoint / "model.safetensors").is_file():
        pytest.skip("checkpoint is not cached")
    model = load_generator(VoxCPM2Config.from_pretrained(checkpoint), str(checkpoint),
                           "cuda", torch.bfloat16).to(dtype=torch.bfloat16)
    assert model.patch_solver is not None
    rng = torch.Generator(device="cuda").manual_seed(314159)

    def inputs(batch, steps=10):
        return (
            *(torch.randn(shape, generator=rng, device="cuda", dtype=torch.bfloat16)
              for shape in ((batch, 2048), (batch, 2048), (batch, 4, 64), (batch, 64, 4))),
            torch.linspace(0.0, 3.0, batch, device="cuda", dtype=torch.float32),
            cfm_t_span(steps, "cuda", torch.bfloat16),
        )

    compiled = torch.compile(model.predict_patch, fullgraph=True)
    # Capture on a warmed stream, exactly as the engine does. Both invocations
    # remain live; a shared mutable output/scratch buffer would fail this check.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outputs, expected = [], [], []
    with torch.cuda.stream(stream):
        for _ in range(2):
            ys = inputs(1)
            expected.append(model.predict_patch(*ys).clone())
            for _ in range(2):
                compiled(*ys)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = compiled(*ys)
            graphs.append((graph, ys))
            outputs.append(out)
        for _ in range(3):
            for graph, _ in graphs:
                graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert all(torch.equal(a, b) for a, b in zip(outputs, expected, strict=True))

    # Replays must read current inputs, including CFG and the timestep schedule, without
    # recapture. Keep every graph/output live while refreshing its inputs.
    with torch.cuda.stream(stream):
        expected = []
        for graph, ys in graphs:
            for dst, src in zip(ys, inputs(ys[0].shape[0]), strict=True):
                dst.copy_(src)
            ys[4].add_(0.25)
            ys[5].copy_(cfm_t_span(10, "cuda", torch.bfloat16, sway_sampling_coef=0.5))
            expected.append(model.predict_patch(*ys).clone())
            graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert all(torch.equal(a, b) for a, b in zip(outputs, expected, strict=True))

    # Unsupported batches and schedules still execute the original function.
    solver = model.patch_solver
    for case in (inputs(2), inputs(8), inputs(1, 8)):
        assert not solver.supports(*case)
        with pytest.raises(RuntimeError, match="invalid tensor shape"):
            solver(*case)
        actual = model.predict_patch(*case)
        model.patch_solver = None
        reference = model.predict_patch(*case)
        model.patch_solver = solver
        assert torch.equal(actual, reference)

    # Compare the logical solve in FP32, starting from the same BF16 weights.
    # Shared norm's CUDA fallback otherwise rounds FP32 inputs back to BF16.
    from mstar.model.components import norm

    cases = [inputs(1)]
    actual = [model.predict_patch(*case).clone() for case in cases]
    model.patch_solver = None
    model.float()

    def fp32_norm(x, weight, eps=1e-6, rms_norm_dtype=None):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight

    def fp32_add_norm(hidden, residual, weight, eps, gemma_mode=False):
        assert not gemma_mode
        summed = hidden + residual
        return fp32_norm(summed, weight, eps), summed

    monkeypatch.setattr(norm, "run_rms_norm", fp32_norm)
    monkeypatch.setattr(norm, "add_rms_norm", fp32_add_norm)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for case, output in zip(cases, actual, strict=True):
            reference = model.predict_patch(*(x.float() for x in case)).double().flatten(1)
            delta = output.double().flatten(1) - reference
            rms = reference.square().mean(1).sqrt().clamp_min(1e-30)
            # Trajectory error is diagnostic: valid BF16 upstream implementations
            # also exceed the old B1-calibrated envelope on some inputs.
            assert torch.isfinite(output).all() and torch.isfinite(reference).all()
            batch = case[0].shape[0]
            record_property(f"b{batch}_relative_rms", (delta.square().mean(1).sqrt() / rms).max().item())
            record_property(f"b{batch}_max_over_rms", (delta.abs().amax(1) / rms).max().item())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
