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


def test_dispatch_uses_actual_tensor_layout_and_batch():
    with FakeTensorMode():
        obj = PatchSolve.__new__(PatchSolve)
        torch.nn.Module.__init__(obj)
        obj.register_buffer("w0", torch.empty((1024, 2048), device="cuda", dtype=torch.bfloat16))
        inputs = [torch.empty(s, device="cuda", dtype=torch.bfloat16)
                  for s in ((1, 2048), (1, 2048), (1, 4, 64), (1, 64, 4), (1,), (11,))]
        inputs[4] = inputs[4].float()
        assert obj.supports(*inputs)
        changed = list(inputs)
        changed[0] = torch.empty((2, 2048), device="cuda", dtype=torch.bfloat16)
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[2] = torch.empty((1, 64, 4), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[5] = inputs[5][:9]
        assert not obj.supports(*changed)
        changed = list(inputs)
        changed[4] = inputs[4].bfloat16()
        assert not obj.supports(*changed)
        output = solve(*inputs, [], [], [], inputs[0], inputs[0], 1e-5)
        assert output.shape == (1, 4, 64)
        assert output.dtype == torch.bfloat16 and output.device == inputs[0].device
        assert output.is_contiguous()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CUTLASS_ROOT")
    or not os.environ.get("VOXCPM2_PATH"),
    reason="needs H200, pinned CUTLASS_ROOT and cached VOXCPM2_PATH",
)
@torch.inference_mode()
def test_real_weights_compile_capture_and_fallback():
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
            torch.full((batch,), 2.0, device="cuda", dtype=torch.float32),
            cfm_t_span(steps, "cuda", torch.bfloat16),
        )

    xs = inputs(1)
    compiled = torch.compile(model.predict_patch, fullgraph=True)
    # Capture on a warmed stream, exactly as the engine does. Both invocations
    # remain live; a shared mutable output/scratch buffer would fail this check.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outputs, expected = [], [], []
    with torch.cuda.stream(stream):
        for _ in range(2):
            ys = tuple(x.clone() for x in xs)
            ys[3].normal_(generator=rng)
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

    # Unsupported batches and schedules still execute the original function.
    solver = model.patch_solver
    for case in (inputs(2), inputs(1, 8)):
        assert not solver.supports(*case)
        actual = model.predict_patch(*case)
        model.patch_solver = None
        reference = model.predict_patch(*case)
        model.patch_solver = solver
        assert torch.equal(actual, reference)
