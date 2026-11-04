"""The block-scaled formats' kernels on a GPU that has them: the activation quantizers against the host reference, the
ops against fp32 within the format's error, and a ``QuantLinear`` through a compiled forward and a CUDA graph."""
import pytest
import torch
from torch import nn

from mstar.model.components.block_scaled import MXFP8, NVFP4
from mstar.model.components.diffusion.compile_utils import with_inductor_rounding
from mstar.model.components.quant_linear import QuantLinear, linear_format
from mstar.platforms import current_platform
from mstar.utils.fp8_block import swizzle_block_scales
from mstar.utils.quant_fp8 import quant_mxfp8, quant_nvfp4

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.has_block_scaled_gemm()),
    reason="needs a GPU with cuBLASLt's block-scaled GEMMs",
)

DEVICE = "cuda"
# rel L2 from fp32 on rows and weights with this spread; the MXFP8 bound is e4m3's rounding on both operands, NVFP4's
# the e2m1 equivalent. Measured on real klein activations: MXFP8 1.8-3.6%, NVFP4 7-14%.
LIMITS = {"mxfp8": 0.05, "nvfp4": 0.25}


def _matrix(rows, k, seed=0, std=1.0):
    g = torch.Generator(DEVICE).manual_seed(seed)
    return (torch.randn(rows, k, device=DEVICE, generator=g) * std).to(torch.bfloat16)


def _live_scale_mask(rows, blocks):
    """Scales past the matrix's rows or blocks are padding the kernel doesn't write."""
    return swizzle_block_scales(torch.ones(rows, blocks, dtype=torch.uint8, device=DEVICE)).bool()


@pytest.mark.parametrize(("rows", "k"), [(4096, 3072), (512, 9216), (70, 3072), (32, 128), (4608, 12288)])
def test_mxfp8_quant_matches_the_host_reference_bytes(rows, k):
    x = _matrix(rows, k, seed=rows)
    x[0, :32] = 0  # an all-zero block keeps a finite scale
    q, ref = quant_mxfp8(x), MXFP8.quantize(x)
    assert torch.equal(q.data.view(torch.uint8), ref.data.view(torch.uint8))
    live = _live_scale_mask(rows, k // 32)
    assert torch.equal(q.scale.view(torch.uint8)[live], ref.scale.view(torch.uint8)[live])


@pytest.mark.parametrize("tensor_scale", [1.0, 4.0])
@pytest.mark.parametrize(("rows", "k"), [(4096, 3072), (512, 9216), (70, 3072), (32, 128), (4608, 12288)])
def test_nvfp4_quant_matches_the_host_reference_within_fp32_ties(rows, k, tensor_scale):
    # past a unit scale's range, so a non-unit one is what keeps the top
    x = _matrix(rows, k, seed=rows, std=1000.0 * tensor_scale)
    scale = torch.full((1,), tensor_scale, device=DEVICE)
    q, ref = quant_nvfp4(x, scale), NVFP4.quantize(x, scale)
    live = _live_scale_mask(rows, k // 16)
    assert torch.equal(q.scale.view(torch.uint8)[live], ref.scale.view(torch.uint8)[live])
    # the device's fp32 division isn't correctly rounded, so a value right on a bin edge can land in the other bin
    same = (q.data.view(torch.uint8) == ref.data.view(torch.uint8)).float().mean().item()
    assert same > 0.995
    torch.testing.assert_close(NVFP4.dequantize(q), NVFP4.dequantize(ref), rtol=0.5, atol=2 ** -8 * tensor_scale)


@pytest.mark.parametrize("name", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize(("m", "n", "k"), [(4096, 9216, 3072), (512, 3072, 9216), (70, 3072, 3072)])
def test_op_against_fp32(monkeypatch, name, m, n, k):
    fmt = linear_format(name)
    x, w = _matrix(m, k, seed=1), _matrix(n, k, seed=2, std=0.03)
    weights = fmt.quantize_weight(w, input_amax=None)
    out = fmt.linear(x, *(weights[buffer] for buffer in fmt.buffers))
    assert out.dtype == torch.bfloat16 and out.shape == (m, n)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    ref = x.float() @ w.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert 0 < rel < LIMITS[name]


def test_nvfp4_keeps_an_input_past_the_unit_range_with_its_declared_amax(monkeypatch):
    """A 16-block above 2688 clips under a unit tensor scale; the declared amax's scale holds it."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    x, w = _matrix(512, 3072, seed=6), _matrix(256, 3072, seed=7, std=0.03)
    x[:, 5] = 4000.0  # one massive channel, as klein's text feed-forward output has
    ref = x.float() @ w.float().t()
    errors = {}
    for amax in (None, 4000.0):
        linear = QuantLinear(w, None, NVFP4, input_amax=amax)
        errors[amax] = ((linear(x).float() - ref).norm() / ref.norm()).item()
    assert errors[4000.0] < LIMITS["nvfp4"] < errors[None]


@pytest.mark.parametrize("name", ["mxfp8", "nvfp4"])
def test_a_strided_input_reads_its_own_values(name):
    linear = QuantLinear(_matrix(512, 3072, seed=8, std=0.03), None, linear_format(name))
    wide = _matrix(256, 6144, seed=9)
    assert torch.equal(linear(wide[:, :3072]), linear(wide[:, :3072].contiguous()))


class _Forward(nn.Module):
    """A norm kept eager, as compile_exact_ops [norms] has it, around one quantized linear."""

    def __init__(self, weight, fmt):
        super().__init__()
        self.norm = nn.LayerNorm(weight.shape[1], elementwise_affine=False)
        self.linear = QuantLinear(weight, None, fmt, input_amax=5000.0)
        self.norm.forward = torch._dynamo.disable(self.norm.forward)

    def forward(self, x):
        return self.linear(self.norm(x) * 1.5 + 0.25) * 0.5


@pytest.mark.parametrize("name", ["mxfp8", "nvfp4"])
def test_compile_and_capture_keep_the_output(name):
    fmt = linear_format(name)
    w = _matrix(3072, 3072, seed=3, std=0.03)
    x = _matrix(1024, 3072, seed=4).view(1, 1024, 3072)
    module = _Forward(w, fmt)
    with torch.inference_mode():
        eager = module(x)
        # the linear is one op, no graph break of its own
        torch.testing.assert_close(torch.compile(module.linear, fullgraph=True)(x), module.linear(x), rtol=0, atol=0)
        # rounded where eager rounds, as compile_transformer_forward compiles a DiT
        compiled_fn = with_inductor_rounding(
            torch.compile(module.forward, fullgraph=False, dynamic=False), eager_rounding=True,
        )
        compiled = compiled_fn(x)
        torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled_fn(x)
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        # the graph reads the activation's bytes at replay: a new input gives a new output
        y = _matrix(1024, 3072, seed=5).view(1, 1024, 3072)
        x.copy_(y)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, module(y), rtol=0, atol=0)
