"""AITER's skinny GEMM behind ``mstar::skinny_gemm`` and the pass that routes
decode-sized linears to it, on an AMD GPU.

The op must match a fp32 reference within one bf16 rounding of the output, a
compiled 1-2 row bucket must reach it (and match eager), and a captured graph
of that bucket must replay to the same result.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

from mstar.platforms import current_platform
from mstar.platforms.compile import with_inductor_overrides
from mstar.platforms.rocm_gemm import skinny_gemm_passes

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or skinny_gemm_passes() is None,
    reason="needs an AMD GPU with AITER",
)


def _close(out: torch.Tensor, ref: torch.Tensor) -> None:
    # one bf16 rounding of a fp32-accumulated result, plus summation order
    torch.testing.assert_close(out.float(), ref, rtol=1.6e-2, atol=1e-2 * ref.abs().max().item())


def _qwen35(hidden, inter, heads, kv_heads, v_heads):
    """(n, k) of every linear in a Qwen3.5 dense decode step (head_dim 256,
    GDN heads of 128 with 16 k-heads, vocab 248320)."""
    key, value = 16 * 128, v_heads * 128
    return [
        (2 * key + 2 * value + 2 * v_heads + (-v_heads % 16), hidden),  # GDN in_proj
        (hidden, value),                                                # GDN out_proj
        (2 * heads * 256 + 2 * kv_heads * 256, hidden),                 # q(+gate)kv
        (hidden, heads * 256),                                          # o_proj
        (2 * inter, hidden), (hidden, inter),                           # MLP
        (248320, hidden),                                               # lm_head
    ]


SHAPES = sorted({
    (16, 64),
    *_qwen35(1024, 3584, 8, 2, 16),     # 0.8B
    *_qwen35(2048, 6144, 8, 2, 16),     # 2B
    *_qwen35(2560, 9216, 16, 4, 32),    # 4B
    *_qwen35(4096, 12288, 16, 4, 32),   # 9B
    # Orpheus 3B (Llama): qkv, o, gate_up, down, head
    (5120, 3072), (3072, 3072), (16384, 3072), (3072, 8192), (156940, 3072),
})


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize(("n", "k"), SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_op_matches_fp32(rows, n, k, dtype):
    g = torch.Generator(device="cuda").manual_seed(rows * 7 + n + k)
    x = torch.randn(rows, k, device="cuda", dtype=dtype, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=dtype, generator=g) / k ** 0.5
    out = torch.ops.mstar.skinny_gemm(x, w, None)
    assert out.shape == (rows, n) and out.dtype == dtype
    _close(out, x.float() @ w.float().T)


def test_op_bias_and_strided_rows():
    x = torch.randn(4, 2 * 1024, device="cuda", dtype=torch.bfloat16)[:, :1024]
    w = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) / 32
    b = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.mstar.skinny_gemm(x, w, b)
    _close(out, x.float() @ w.float().T + b.float())
    residual = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.mstar.skinny_gemm(x, w, residual)
    _close(out, x.float() @ w.float().T + residual.float())


class _Block(torch.nn.Module):
    """A gated MLP plus a biased head, linears as models write them."""

    def __init__(self, hidden=1024, inter=2816, vocab=4096):
        super().__init__()
        self.gate_up = torch.nn.Linear(hidden, 2 * inter, bias=False)
        self.down = torch.nn.Linear(inter, hidden, bias=False)
        self.head = torch.nn.Linear(hidden, vocab, bias=True)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        h = x + self.down(F.silu(g) * u)
        return self.head(h)


def _compiled(model, rows):
    return with_inductor_overrides(
        torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False), rows,
    )


@pytest.mark.parametrize("rows", [1, 2])
def test_small_bucket_compiles_to_the_skinny_op(rows):
    """Every linear reaches the skinny op, the one-row ones before inductor
    turns them into its own GEMV reduction."""
    model = _Block().to("cuda", torch.bfloat16)
    x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    fn = _compiled(model, rows)
    with torch.no_grad():
        out, code = run_and_get_code(fn, x)
        ref = model(x)
    calls = sum(len(re.findall(r"= torch\.ops\.mstar\.skinny_gemm(_any)?\.default\(", c)) for c in code)
    assert calls == 3
    assert not any("extern_kernels.mm(" in c or "extern_kernels.addmm(" in c for c in code)
    _close(out, ref.float())


@pytest.mark.parametrize("rows", [4, 8])
def test_larger_bucket_keeps_hipblaslt(rows):
    model = _Block().to("cuda", torch.bfloat16)
    x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        _, code = run_and_get_code(_compiled(model, rows), x)
    assert not any("mstar.skinny_gemm" in c for c in code)  # neither op


def test_one_row_gated_norm_feeding_the_gemm_is_one_kernel():
    """The layout-free op lets inductor finish the norm in a single kernel."""
    from mstar.model.components.norm import RMSNormGated

    norm = RMSNormGated(128).cuda()
    w = torch.randn(1024, 32 * 128, device="cuda", dtype=torch.bfloat16) / 64

    def fn(core, z):
        return F.linear(norm(core, z).reshape(1, 32 * 128), w)

    core = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(1, 2 * 32 * 128, device="cuda", dtype=torch.bfloat16)[:, 32 * 128:].reshape(1, 32, 128)
    with torch.no_grad():
        out, code = run_and_get_code(_compiled(fn, 1), core, z)
    kernels = set(re.findall(r"def (triton_\w+)\(", "\n".join(code)))
    assert len(kernels) == 1, kernels
    _close(out, F.linear(norm(core, z).reshape(1, -1), w).float())


def test_captured_bucket_replays():
    model = _Block().to("cuda", torch.bfloat16)
    fn = _compiled(model, 2)
    static_x = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(2):
            fn(static_x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fn(static_x)
        for seed in range(3):
            static_x.copy_(torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16,
                                       generator=torch.Generator(device="cuda").manual_seed(seed)))
            graph.replay()
            torch.cuda.synchronize()
            _close(static_out, model(static_x).float())
