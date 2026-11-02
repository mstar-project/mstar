"""The block-scaled GEMM and its activation quantizer on a GPU that has them: the quantizer against the host reference,
the GEMM against fp32 within the format's error, and both through a compiled forward and a CUDA graph."""
import pytest
import torch
from torch import nn

from mstar.model.components.linear import BlockScaledLinear
from mstar.platforms import current_platform
from mstar.platforms.block_gemm import block_scaled_gemm
from mstar.utils.fp8_block import BLOCK_FORMATS, MXFP8, NVFP4, quantize_block
from mstar.utils.quant_fp8 import quant_block

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


@pytest.mark.parametrize(("rows", "k"), [(4096, 3072), (512, 9216), (70, 3072), (32, 128), (4608, 12288)])
def test_mxfp8_quant_matches_the_host_reference_bytes(rows, k):
    x = _matrix(rows, k, seed=rows)
    x[0, :32] = 0  # an all-zero block keeps a finite scale
    q, ref = quant_block(x, MXFP8), quantize_block(x, MXFP8)
    assert torch.equal(q.data.view(torch.uint8), ref.data.view(torch.uint8))
    live = _live_scale_mask(rows, k // 32, ref.scale.numel())
    assert torch.equal(q.scale.view(torch.uint8)[live], ref.scale.view(torch.uint8)[live])


@pytest.mark.parametrize(("rows", "k"), [(4096, 3072), (512, 9216), (70, 3072), (32, 128), (4608, 12288)])
def test_nvfp4_quant_matches_the_host_reference_within_fp32_ties(rows, k):
    x = _matrix(rows, k, seed=rows)
    q, ref = quant_block(x, NVFP4), quantize_block(x, NVFP4, tensor_scale=1.0)
    live = _live_scale_mask(rows, k // 16, ref.scale.numel())
    assert torch.equal(q.scale.view(torch.uint8)[live], ref.scale.view(torch.uint8)[live])
    # the device divides y by the scale, the host by a product it rounded: bins differ only at bin edges
    same = (q.data.view(torch.uint8) == ref.data.view(torch.uint8)).float().mean().item()
    assert same > 0.999


def _live_scale_mask(rows, blocks, numel):
    """Scales past the matrix's rows or blocks are padding the kernel doesn't write."""
    from mstar.utils.fp8_block import swizzle_block_scales

    return swizzle_block_scales(torch.ones(rows, blocks, dtype=torch.uint8, device=DEVICE)).bool()


@pytest.mark.parametrize("name", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize(("m", "n", "k"), [(4096, 9216, 3072), (512, 3072, 9216), (70, 3072, 3072)])
def test_gemm_against_fp32(name, m, n, k):
    fmt = BLOCK_FORMATS[name]
    x, w = _matrix(m, k, seed=1), _matrix(n, k, seed=2, std=0.03)
    out = block_scaled_gemm(quant_block(x, fmt), quantize_block(w, fmt), fmt)
    assert out.dtype == torch.bfloat16 and out.shape == (m, n)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = x.float() @ w.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert 0 < rel < LIMITS[name]


class _Forward(nn.Module):
    """A norm kept eager, as compile_exact_ops [norms] has it, around one block-scaled linear."""

    def __init__(self, weight, fmt):
        super().__init__()
        self.norm = nn.LayerNorm(weight.shape[1], elementwise_affine=False)
        self.linear = BlockScaledLinear(weight, fmt)
        self.norm.forward = torch._dynamo.disable(self.norm.forward)

    def forward(self, x):
        return self.linear(self.norm(x)) * 0.5


@pytest.mark.parametrize("name", ["mxfp8", "nvfp4"])
def test_compile_and_capture_keep_the_output(name):
    fmt = BLOCK_FORMATS[name]
    w = _matrix(3072, 3072, seed=3, std=0.03)
    x = _matrix(1024, 3072, seed=4).view(1, 1024, 3072)
    module = _Forward(w, fmt)
    with torch.inference_mode():
        eager = module(x)
        compiled_fn = torch.compile(module.forward, fullgraph=False, dynamic=False)
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
