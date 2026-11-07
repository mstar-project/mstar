"""dense_fp8 on Blackwell without a block-fp8 GEMM: above _DEQUANT_MIN_TOKENS the weight is dequantized to bf16 per
call and cuBLAS runs it (w8a16 is ~10x slower than bf16 there at prefill), below it w8a16 as before; other GPUs and
a table's rows are unchanged (CPU). The dequant kernel compiles for sm_90 and sm_100 with the arguments dequantize
passes (CPU), and on CUDA the dequantized linear matches the dequantized reference."""

import pytest
import torch
import torch.nn.functional as F

from mstar.utils import dense_fp8
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight

DEV = torch.device("cuda", 0)


@pytest.fixture
def blackwell(monkeypatch):
    """A device whose probes found only what Blackwell has on main: no SM90 GEMM, no table rows."""
    monkeypatch.setitem(dense_fp8._BLACKWELL, DEV, True)
    monkeypatch.setitem(dense_fp8._FI_OK, DEV, False)
    monkeypatch.setitem(dense_fp8._W8A8_OK, DEV, False)
    monkeypatch.setattr(dense_fp8, "_table_rows", lambda *a: None)


def test_blackwell_dequantizes_above_the_threshold_and_keeps_w8a16_below(blackwell):
    edge = dense_fp8._DEQUANT_MIN_TOKENS
    for n, k, glu in ((6144, 2048, False), (1536, 6144, True), (2624, 6144, False)):
        assert [dense_fp8._choose(DEV, n, k, glu, m)[0] for m in (1, edge, edge + 1, 4096)] == [
            "w8a16", "w8a16", "dequant", "dequant"], (n, k, glu)
    assert dense_fp8._choose(DEV, 6144, 2048, False, 512, block_size=(32, 32))[0] == "dequant"
    assert dense_fp8._choose(DEV, 6144, 2048, False, 8, block_size=(32, 32))[0] == "w8a16"


def test_other_gpus_keep_their_dispatch(monkeypatch):
    monkeypatch.setitem(dense_fp8._BLACKWELL, DEV, False)
    monkeypatch.setitem(dense_fp8._FI_OK, DEV, True)
    monkeypatch.setitem(dense_fp8._W8A8_OK, DEV, True)
    monkeypatch.setattr(dense_fp8, "_table_rows", lambda *a: None)
    assert [dense_fp8._choose(DEV, 6144, 2048, False, m)[0] for m in (8, 256, 4096)] == ["w8a16", "flashinfer",
                                                                                          "cublas"]


def test_a_table_row_can_name_dequant(blackwell, monkeypatch):
    from mstar.kernels import Choice

    monkeypatch.setattr(dense_fp8, "_table_rows", lambda *a: ((16,), (Choice("w8a16"), Choice("dequant"))))
    assert dense_fp8._choose(DEV, 6144, 2048, False, 32) == ("dequant", None)
    assert "dequant" in [impl.name for impl in dense_fp8.FP8_LINEAR.impls]


class _Launches:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((args, kwargs))


@pytest.mark.parametrize("arch", [90, 100])
def test_the_dequant_kernel_compiles(monkeypatch, arch):
    pytest.importorskip("triton.backends.nvidia")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    launches = _Launches(dense_fp8._dequant_kernel)
    monkeypatch.setattr(dense_fp8, "_dequant_kernel", launches)
    for rows, k in ((2624, 6144), (3072, 6144)):
        dense_fp8.dequantize(torch.zeros(rows, k, dtype=torch.uint8), torch.ones(-(-rows // 128), k // 128),
                             (128, 128))
    types = {torch.uint8: "*u8", torch.float32: "*fp32", torch.bfloat16: "*bf16"}
    for args, kwargs in launches.calls:
        names = launches.fn.arg_names
        values = dict(zip(names, args, strict=False)) | {k: v for k, v in kwargs.items() if k in names}
        consts = {names[i]: values[names[i]] for i in launches.fn.constexprs}
        sig = {n: "constexpr" if n in consts else types[v.dtype] if isinstance(v, torch.Tensor) else "i32"
               for n, v in values.items()}
        try:
            triton.compile(ASTSource(fn=launches.fn, signature=sig, constexprs=consts),
                           target=GPUTarget("cuda", arch, 32))
        except (RuntimeError, FileNotFoundError) as exc:
            if "ptxas" in str(exc).lower():
                pytest.skip(f"this host can't run ptxas for sm_{arch}: {exc}")
            raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("glu", [False, True])
def test_the_dequantized_linear_matches_the_reference(glu):
    torch.manual_seed(0)
    n, k = 640, 1024
    rows = 2 * n if glu else n
    w = torch.randn(rows, k, device=DEV) * 0.02
    tiles = w.reshape(rows // 128, 128, k // 128, 128)
    s = (tiles.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12).contiguous()
    q = (tiles / s[:, None, :, None]).to(FP8_DTYPE).reshape(rows, k)
    ref_w = dequantize_fp8_block_weight(q, s, (128, 128)).to(torch.bfloat16)
    assert torch.equal(dense_fp8.dequantize(q.view(torch.uint8), s, (128, 128)), ref_w)
    for m in (65, 300):
        x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
        want = F.linear(x, ref_w)
        if glu:
            g, u = want.chunk(2, dim=-1)
            want = F.silu(g) * u
        got = dense_fp8._dequant_linear(x, q.view(torch.uint8), s, (128, 128), glu)
        assert ((got.float() - want.float()).norm() / want.float().norm()).item() < 0.01, (m, glu)
