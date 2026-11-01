"""Every kernel op in the registry against its own reference on the GPU.

Generated from the declarations, so a new op is checked as soon as it is
declared: on each of its warm shapes and dtypes the implementation this
platform runs is within the op's tolerance of the reference evaluated in
fp64, and a captured graph of it replays to what eager computed.
"""

import pytest
import torch

from mstar.kernels import registry
from mstar.kernels.numerics import as_tensors, compare, reference_outputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

OPS = registry.load_all()
CASES = [(op, shape, dtype) for op in OPS for shape in op.shapes.warm for dtype in op.shapes.dtypes]


def _case_id(case) -> str:
    op, shape, dtype = case
    return f"{op.name}-{'x'.join(str(v) for v in shape.values())}-{str(dtype).removeprefix('torch.')}"


def _skip_if_unavailable(op) -> None:
    reason = op.unavailable_reason()
    if reason is not None:
        pytest.skip(f"{op.qualname}: {reason}")


def _outputs(op, result, args) -> list[torch.Tensor]:
    return [*as_tensors(result), *(args[i] for i in op.mutated_args())]


def _clone(args):
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


@pytest.mark.parametrize(("op", "shape", "dtype"), CASES, ids=[_case_id(c) for c in CASES])
def test_op_matches_its_reference(op, shape, dtype):
    _skip_if_unavailable(op)
    generator = torch.Generator(device="cuda").manual_seed(0)
    args = op.shapes.inputs(shape, dtype, torch.device("cuda"), generator)
    expected = reference_outputs(op.reference, args, op.mutated_args())
    got = _outputs(op, op.op(*args), args)
    for out, ref in zip(got, expected, strict=True):
        result = compare(out, ref, op.tolerance)
        assert result.ok, f"{op.qualname} {dict(shape)} {dtype}: {result.summary()}"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_a_captured_replay_matches_eager(op):
    _skip_if_unavailable(op)
    generator = torch.Generator(device="cuda").manual_seed(1)
    inputs = op.shapes.inputs(op.shapes.warm[0], op.shapes.dtypes[0], torch.device("cuda"), generator)
    eager_args = _clone(inputs)
    eager = _outputs(op, op.op(*eager_args), eager_args)

    static = _clone(inputs)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op.op(*_clone(inputs))
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _outputs(op, op.op(*static), static)
    for buffer, value in zip(static, inputs, strict=True):
        if isinstance(buffer, torch.Tensor):
            buffer.copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    for out, ref in zip(captured, eager, strict=True):
        assert torch.equal(out, ref), op.qualname
