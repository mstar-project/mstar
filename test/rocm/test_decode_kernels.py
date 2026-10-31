"""The Triton decode kernels on an AMD GPU: the fused residual add + RMSNorm,
the decode GEMV (``mstar::decode_gemv``) and its gated-MLP epilogue.

Each must meet its op's bar against the fp64 reference (one rounding for the
norm, a rounded fp32 sum for the GEMV) at every knob family the search tries,
a split-K launch must replay to the same bits in a captured graph, and a
compiled bucket must reach the fused gate/up op once a config
selects it.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

from mstar.kernels import configs
from mstar.kernels.numerics import Tolerance, compare, reference_outputs
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"


def _fused_inputs(rows, hidden, dtype, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = torch.randn(rows, hidden, device=DEV, dtype=dtype, generator=g)
    r = torch.randn(rows, hidden, device=DEV, dtype=dtype, generator=g) * 4
    w = torch.randn(hidden, device=DEV, dtype=dtype, generator=g) * 0.5
    return h, r, w


@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("hidden", [256, 1024, 2880, 4096])
@pytest.mark.parametrize("rows", [1, 7, 128, 513])
@pytest.mark.parametrize(
    "knobs",
    [
        {"elems_per_thread": 16, "rows_per_program": 1, "waves_per_eu": 0},
        {"elems_per_thread": 2, "rows_per_program": 4, "waves_per_eu": 2},
    ],
    ids=["ept16-r1", "ept2-r4"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_add_rms_norm_is_rounded_once(gemma, hidden, rows, knobs, dtype):
    from mstar.engine.resources.rms_norm.rms_norm import _fused_add_rms_norm_reference
    from mstar.kernels.triton.norm import fused_add_rms_norm

    h, r, w = _fused_inputs(rows, hidden, dtype, rows * hidden)
    expected = reference_outputs(_fused_add_rms_norm_reference, (h, r, w, 1e-6, gemma), (0, 1))
    fused_add_rms_norm(h, r, w, 1e-6, gemma, knobs)
    for out, ref in zip((h, r), expected, strict=True):
        result = compare(out, ref, Tolerance.rounded_once())
        assert result.ok, result.summary()


def test_fused_add_rms_norm_replays_like_eager():
    from mstar.kernels.triton.norm import fused_add_rms_norm

    h, r, w = _fused_inputs(16, 4096, torch.bfloat16, 3)
    h_eager, r_eager = h.clone(), r.clone()
    fused_add_rms_norm(h_eager, r_eager, w, 1e-6, True)
    h_graph, r_graph = h.clone(), r.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_add_rms_norm(h_graph, r_graph, w, 1e-6, True)
    h_graph.copy_(h)
    r_graph.copy_(r)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(h_graph, h_eager) and torch.equal(r_graph, r_eager)


def _gemv_inputs(rows, n, k, dtype, seed, bias=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, k, device=DEV, dtype=dtype, generator=g)
    w = torch.randn(n, k, device=DEV, dtype=dtype, generator=g) * k ** -0.5
    b = None
    if bias == "n":
        b = torch.randn(n, device=DEV, dtype=dtype, generator=g)
    elif bias == "rows":
        b = torch.randn(rows, n, device=DEV, dtype=dtype, generator=g)
    return x, w, b


def _assert_accumulated(out, ref):
    result = compare(out, ref, Tolerance.accumulated())
    assert result.ok, result.summary()


KNOBS = [
    {"block_n": 8, "block_k": 512, "split_k": 1, "num_warps": 4, "use_dot": False},
    {"block_n": 4, "block_k": 1024, "split_k": 4, "num_warps": 2, "use_dot": False, "xcd_remap": True},
    {"block_n": 16, "block_k": 256, "split_k": 1, "num_warps": 4, "use_dot": True},
    {"block_n": 32, "block_k": 128, "split_k": 8, "num_warps": 4, "use_dot": True, "xcd_remap": True,
     "num_stages": 2},
]
KNOB_IDS = ["fma", "fma-splitk4-xcd", "dot", "dot-splitk8-xcd"]


@pytest.mark.parametrize("knobs", KNOBS, ids=KNOB_IDS)
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 13, 16])
@pytest.mark.parametrize(("n", "k"), [(4096, 4096), (12352, 4096), (1000, 1032), (2048, 6144)])
@pytest.mark.parametrize("bias", [None, "n", "rows"])
def test_decode_gemv_matches_the_reference(knobs, rows, n, k, bias):
    from mstar.kernels.triton.gemv import decode_gemv

    if knobs["use_dot"] is False and rows > 8 and knobs["block_n"] * knobs["block_k"] > 4096:
        pytest.skip("an FMA tile this size spills at 16 rows; the search never pairs them")
    x, w, b = _gemv_inputs(rows, n, k, torch.bfloat16, rows + n + k, bias)
    ref = (x.double() @ w.double().T) + (0 if b is None else b.double())
    _assert_accumulated(decode_gemv(x, w, b, knobs), ref)


@pytest.mark.parametrize("knobs", KNOBS, ids=KNOB_IDS)
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_silu_mul_epilogue_matches_the_unfused_pair(knobs, rows):
    from mstar.kernels.triton.gemv import decode_gemv_silu_mul

    x, w, _ = _gemv_inputs(rows, 2 * 3072, 2048, torch.bfloat16, rows)
    gate, up = (x.double() @ w.double().T).chunk(2, dim=-1)
    _assert_accumulated(decode_gemv_silu_mul(x, w, knobs), F.silu(gate) * up)


def test_split_k_replays_to_the_same_bits():
    from mstar.kernels.triton import gemv

    knobs = {"block_n": 16, "block_k": 256, "split_k": 8, "num_warps": 4, "use_dot": True}
    x, w, _ = _gemv_inputs(4, 4096, 4096, torch.bfloat16, 11)
    eager = gemv.decode_gemv(x, w, None, knobs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gemv.decode_gemv(x, w, None, knobs)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), "the partials are added in a fixed order"


@pytest.fixture
def triton_config(tmp_path, monkeypatch):
    """A ``decode_gemm`` table sending every bucket's linears to Triton and folding its gated activations."""
    facts = current_platform.device_facts()
    knobs = {"block_n": 8, "block_k": 512, "split_k": 2, "num_warps": 4}
    path = configs.config_path("decode_gemm", facts.arch, facts.product, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    triton = {"impl": "triton", "candidates": {"triton": knobs}}
    classes = dict.fromkeys(("rows1", "rows2", "rows4", "rows8", "rows16"), triton)
    gated = {
        f"gated1-k1024-n5632-rows{rows}": {"impl": "fused", "candidates": {"fused": knobs}} for rows in (1, 4, 16)
    }
    path.write_text(json.dumps({"kernel": "decode_gemm", "classes": classes, "shapes": gated}))
    monkeypatch.setattr(configs, "CONFIG_ROOT", tmp_path)
    configs._read.cache_clear()
    yield
    configs._read.cache_clear()


class _GatedMLP(torch.nn.Module):
    def __init__(self, hidden=1024, inter=2816):
        super().__init__()
        bf16 = dict(dtype=torch.bfloat16, device=DEV)
        self.gate_up = torch.nn.Parameter(torch.randn(2 * inter, hidden, **bf16) * hidden ** -0.5)
        self.down = torch.nn.Parameter(torch.randn(hidden, inter, **bf16) * inter ** -0.5)
        self.inter = inter

    def forward(self, x):
        gate, up = F.linear(x, self.gate_up).split(self.inter, dim=-1)
        return F.linear(F.silu(gate) * up, self.down)


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_a_compiled_bucket_reaches_the_fused_gate_up(triton_config, rows):
    from mstar.platforms import rocm_gemm
    from mstar.platforms.compile import with_inductor_overrides
    from mstar.platforms.rocm_gemm import skinny_gemm_passes

    torch._dynamo.reset()
    # the ops pick their kernels per shape once; start from this config
    for op in (rocm_gemm.DECODE_GEMV, rocm_gemm.DECODE_GEMV_SILU_MUL):
        op.dispatch.reset()
    assert skinny_gemm_passes() is not None
    mlp = _GatedMLP()
    x = torch.randn(rows, 1024, device=DEV, dtype=torch.bfloat16)
    eager = mlp(x)
    compiled = with_inductor_overrides(torch.compile(mlp, dynamic=False), num_tokens=rows, rows=rows)
    out, code = run_and_get_code(compiled, x)
    source = "\n".join(code)
    assert "decode_gemv_silu_mul" in source
    assert "mstar.decode_gemv.default" in source, "the down projection runs on the Triton GEMV"
    torch.testing.assert_close(out.float(), eager.float(), rtol=3e-2, atol=3e-2 * eager.abs().max().item())
