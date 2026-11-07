"""dense_fp8 on Blackwell: neither SM90 path (flashinfer's swapAB GEMM, cuBLAS's block-scaled scaled_mm) runs on
sm_100 / sm_103, so the dispatch takes flashinfer's groupwise GEMM where its probe passed and the shape is in whole
128-row blocks, else w8a16 as before; on CUDA (sm_100 only) the groupwise path matches the dequantized GEMM."""

import pytest
import torch
import torch.nn.functional as F

from mstar.utils import dense_fp8
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight

DEV = torch.device("cuda", 0)


@pytest.fixture
def blackwell(monkeypatch):
    """A device whose probes found only the groupwise GEMM (sm_100), with no table rows for it."""
    monkeypatch.setitem(dense_fp8._BW_OK, DEV, True)
    monkeypatch.setitem(dense_fp8._FI_OK, DEV, False)
    monkeypatch.setitem(dense_fp8._W8A8_OK, DEV, False)
    monkeypatch.setattr(dense_fp8, "_table_rows", lambda *a: None)


@pytest.mark.parametrize("m", [1, 16, 64, 1024])
def test_blackwell_takes_the_groupwise_gemm_at_every_token_count(blackwell, m):
    assert dense_fp8._choose(DEV, 6144, 2048, False, m) == ("groupwise", None)
    assert dense_fp8._choose(DEV, 1536, 6144, True, m) == ("groupwise", None)  # GLM-5.3's TP8 MLP, SwiGLU fused


def test_a_partial_row_block_and_another_block_keep_w8a16(blackwell):
    assert dense_fp8._choose(DEV, 2624, 6144, False, 8) == ("w8a16", None)  # GLM's q_a + kv_a, 2624 rows
    assert dense_fp8._choose(DEV, 6144, 2048, False, 8, block_size=(32, 32)) == ("w8a16", None)


def test_without_the_groupwise_gemm_the_dispatch_is_unchanged(monkeypatch):
    monkeypatch.setitem(dense_fp8._BW_OK, DEV, False)
    monkeypatch.setitem(dense_fp8._FI_OK, DEV, True)
    monkeypatch.setitem(dense_fp8._W8A8_OK, DEV, True)
    monkeypatch.setattr(dense_fp8, "_table_rows", lambda *a: None)
    assert dense_fp8._choose(DEV, 6144, 2048, False, 8)[0] == "w8a16"
    assert dense_fp8._choose(DEV, 6144, 2048, False, 256)[0] == "flashinfer"
    assert dense_fp8._choose(DEV, 6144, 2048, False, 4096)[0] == "cublas"


def test_a_table_row_names_groupwise_only_where_it_runs(blackwell, monkeypatch):
    from mstar.kernels.table import Choice

    monkeypatch.setattr(dense_fp8, "_table_rows", lambda *a: ((8,), (Choice("groupwise"), Choice("w8a16"))))
    assert dense_fp8._choose(DEV, 6144, 2048, False, 4) == ("groupwise", None)
    monkeypatch.setitem(dense_fp8._BW_OK, DEV, False)
    assert dense_fp8._choose(DEV, 6144, 2048, False, 4)[0] == "w8a16"  # the built-in rule then


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                    reason="flashinfer's groupwise fp8 GEMM is sm_100 / sm_103")
@pytest.mark.parametrize("glu", [False, True])
def test_the_groupwise_gemm_matches_the_dequantized_one(glu):
    from benchmark.kernels.fp8_linear_impls import _block_quant

    dense_fp8.reserve(DEV)
    assert dense_fp8._BW_OK[DEV], "the probe failed on a Blackwell GPU"
    torch.manual_seed(0)
    n, k = 512, 1024
    q, s = _block_quant((torch.randn(2 * n if glu else n, k, device=DEV) * 0.02).to(torch.bfloat16), FP8_DTYPE)
    ref_w = dequantize_fp8_block_weight(q, s, (128, 128)).to(torch.bfloat16)
    for m in (1, 7, 64, 300):
        x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
        want = F.linear(x, ref_w)
        if glu:
            g, u = want.chunk(2, dim=-1)
            want = F.silu(g) * u
        got = dense_fp8.linear(x, q.view(torch.uint8), s, (128, 128), glu)
        # x is quantized to e4m3 per 128 columns, as in the SM90 paths
        assert ((got.float() - want.float()).norm() / want.float().norm()).item() < 0.06, (m, glu)
