"""Tuned-config lookup and precomputed alignment for the fused MoE."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.kernels import DeviceKey, registry, table
from mstar.utils.fused_moe import kernels

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE requires CUDA")


@pytest.fixture
def tuned_table(tmp_path, monkeypatch):
    """The moe_gemm table file of a fake GPU, the only one the lookup sees."""
    monkeypatch.setenv(table.TABLES_ENV, str(tmp_path))
    monkeypatch.setattr(table, "TABLE_DIR", tmp_path / "package")
    monkeypatch.setattr(registry, "_device_index", lambda device: 0)
    monkeypatch.setattr(registry, "device_key", lambda index: DeviceKey("Test GPU", "sm_00"))
    registry.reset_lookups()
    kernels._TILE_ROWS.clear()
    yield tmp_path / "moe_gemm" / "Test_GPU.json"
    registry.reset_lookups()
    kernels._TILE_ROWS.clear()


def _write(path, rows, top_k=2, shape=None, dtype="bfloat16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 1, "op": "moe_gemm", "entries": [{
        "dtype": dtype, "shape": shape or {"E": 8, "N": 64, "K": 32, "top_k": top_k}, "backend": "triton",
        "provenance": {"device": "Test GPU", "date": "2026-10-28", "harness": "test"}, "rows": rows}]}))


def test_get_config_falls_back_to_default(tuned_table):
    _write(tuned_table, {"1": {"BLOCK_SIZE_M": 64}}, top_k=4)  # another shape's entry
    up, down = kernels.get_config(M=4, E=8, N=64, K=32, top_k=2)
    assert up == down == kernels.get_default_config(4, 8, 64, 32, 2)


def test_get_config_uses_nearest_batch_and_down_overrides(tuned_table):
    tile = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
    _write(tuned_table, {
        "1": {**tile, "down": {"BLOCK_SIZE_K": 64, "BLOCK_SIZE_M": 64}},
        "64": {**tile, "BLOCK_SIZE_M": 64},
    })
    up, down = kernels.get_config(M=2, E=8, N=64, K=32, top_k=2)
    assert up == tile
    # The down GEMM shares the up GEMM's alignment, so BLOCK_SIZE_M is not overridable.
    assert down == {**tile, "BLOCK_SIZE_K": 64}
    assert kernels.get_config(M=50, E=8, N=64, K=32, top_k=2)[0]["BLOCK_SIZE_M"] == 64
    # fp16 activations have no entry here
    assert kernels.get_config(M=2, E=8, N=64, K=32, top_k=2, dtype=torch.float16)[0] == \
        kernels.get_default_config(2, 8, 64, 32, 2)


FP8_SHAPE = {"E": 8, "N": 256, "K": 256, "top_k": 2}


def test_fp8_tiles_without_a_table_are_the_previous_defaults(tuned_table):
    """No float8_e4m3fn entry: fused_experts_fp8 launches what it always did (GLM-5.2's decode
    tiles up to 16 tokens, the block-fp8 defaults above), BLOCK_SIZE_K the scale block's."""
    _write(tuned_table, {"1": {"BLOCK_SIZE_M": 64}}, shape=FP8_SHAPE)  # bf16's entry, not fp8's
    up, down = kernels.get_config_fp8(M=4, **FP8_SHAPE)
    assert up == {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8,
                  "num_warps": 4, "num_stages": 5}
    assert down == {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
                    "num_warps": 4, "num_stages": 2}
    up, down = kernels.get_config_fp8(M=300, **FP8_SHAPE)
    assert up == down == {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32,
                          "num_warps": 4, "num_stages": 3}
    assert kernels.get_config_fp8(M=4, **FP8_SHAPE, block_k=64)[0]["BLOCK_SIZE_K"] == 64


def test_fp8_tiles_come_from_the_tables_fp8_entry(tuned_table):
    tile = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 4}
    _write(tuned_table, shape=FP8_SHAPE, dtype="float8_e4m3fn", rows={
        "1": {**tile, "down": {"BLOCK_SIZE_N": 64, "num_stages": 3}},
        # a row can't move BLOCK_SIZE_K off the scale block: the kernel rescales per K tile
        "256": {**tile, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
    })
    up, down = kernels.get_config_fp8(M=2, **FP8_SHAPE)
    assert up == tile and down == {**tile, "BLOCK_SIZE_N": 64, "num_stages": 3}
    up, down = kernels.get_config_fp8(M=200, **FP8_SHAPE)
    assert up["BLOCK_SIZE_M"] == down["BLOCK_SIZE_M"] == 64
    assert up["BLOCK_SIZE_K"] == down["BLOCK_SIZE_K"] == 128
    # the fp8 entry serves neither the bf16 kernel nor another scale block
    assert kernels.get_config(M=2, **FP8_SHAPE)[0] == kernels.get_default_config(2, *FP8_SHAPE.values())
    assert kernels.get_config_fp8(M=2, **FP8_SHAPE, block_k=64)[0]["BLOCK_SIZE_N"] == 64


def test_the_triton_backend_takes_fp8_experts_in_whole_scale_blocks():
    from mstar.utils.quant_fp8 import FP8_DTYPE

    triton = kernels.MOE_GEMM.backend("triton")
    cuda = torch.device("cuda")
    # Qwen3-Coder-30B-A3B's experts at TP1 and TP2
    assert triton.unavailable(cuda, FP8_DTYPE, {"E": 128, "N": 1536, "K": 2048, "top_k": 8}) is None
    assert triton.unavailable(cuda, FP8_DTYPE, {"E": 128, "N": 768, "K": 2048, "top_k": 8}) is None
    # TP4 cuts a 128-wide scale block
    assert "scale blocks" in triton.unavailable(cuda, FP8_DTYPE, {"E": 128, "N": 384, "K": 2048, "top_k": 8})
    assert triton.unavailable(cuda, torch.float32, FP8_SHAPE) is not None
    assert triton.unavailable(torch.device("cpu"), FP8_DTYPE, FP8_SHAPE) is not None


def test_command_a_plus_keeps_its_gb200_tiles():
    """The GB200 table moved into the registry's format with the same rows (#377's file)."""
    path = table.TABLE_DIR / "moe_gemm" / "NVIDIA_GB200.json"
    (entry,) = table.load_table(path, "moe_gemm")
    assert dict(entry.shape) == {"E": 128, "N": 2048, "K": 4096, "top_k": 8}
    assert entry.buckets() == (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512)
    assert dict(entry.row(1)) == {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4,
        "num_stages": 5, "down": {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4,
                                  "num_stages": 5}}


@CUDA
@pytest.mark.parametrize("cuda_align", [True, False], ids=["cuda_align", "torch_align"])
@pytest.mark.parametrize("dynamic", [False, None], ids=["captured", "eager_forward"])
def test_compiled_fused_experts_takes_the_tables_tiles(tuned_table, monkeypatch, dynamic, cuda_align):
    """Compiled as the engine compiles (dynamic=False for a captured bucket; dynamic=None for
    the eager forward, where the token count turns symbolic after a few sizes, under the engine's
    dynamo settings), fused_experts runs the table's row for the nearest M and matches the eager
    call. The tiles must reach the Triton launch as constants: dynamo refuses a symbolic
    num_warps. The alignment's torch fallback breaks the graph, so there the default tiles serve."""
    import mstar.engine  # noqa: F401 - the engine's process-wide dynamo settings
    from mstar.utils.fused_moe import align, fused_experts

    if not cuda_align:
        monkeypatch.setattr(align, "_cuda_op_available", lambda: False)
    elif not align._cuda_op_available():
        pytest.skip("the CUDA moe_align_block_size op did not build here")
    torch._dynamo.reset()

    experts, hidden, inter, top_k = 8, 256, 128, 2
    _write(tuned_table, shape={"E": experts, "N": 2 * inter, "K": hidden, "top_k": top_k}, rows={
        "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
              "num_warps": 4, "num_stages": 2, "down": {"BLOCK_SIZE_N": 32}},
        "64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
               "num_warps": 8, "num_stages": 3},
    })
    assert kernels.get_config(40, experts, 2 * inter, hidden, top_k)[0]["BLOCK_SIZE_M"] == 32

    class Experts(torch.nn.Module):
        """Router and experts as a model has them: the expert shapes and top_k are static under
        any compile, and only the token count can be symbolic."""

        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(hidden, experts, bias=False, device="cuda", dtype=torch.bfloat16)
            self.w1 = torch.nn.Parameter(
                torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
            self.w2 = torch.nn.Parameter(
                torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02)
            self.top_k = top_k

        def forward(self, x):
            weights, ids = torch.softmax(self.gate(x).float(), dim=-1).topk(self.top_k, dim=-1)
            return fused_experts(x, self.w1, self.w2, weights.to(x.dtype), ids) * 2

    torch.manual_seed(0)
    module = Experts()
    compiled = torch.compile(module, dynamic=dynamic)
    with torch.no_grad():
        for tokens in (3, 40, 50, 300, 1000):
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(compiled(x), module(x), atol=1e-2, rtol=1e-2)


@CUDA
def test_compiled_fused_experts_is_one_graph():
    """fused_experts compiles without a graph break: the tile lookup and the align op's build
    check are constants to dynamo. Main broke the graph at both (torch.cuda.get_device_name and
    a file read in each), so a tuned config crossed into the next graph as data."""
    from mstar.utils.fused_moe import align, fused_experts

    if not align._cuda_op_available():
        pytest.skip("the CUDA moe_align_block_size op did not build here; its torch fallback breaks the graph")
    torch._dynamo.reset()

    experts, hidden, inter, top_k = 8, 256, 128, 2
    x = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    weights, ids = torch.rand(4, experts, device="cuda").topk(top_k, dim=-1)
    explain = torch._dynamo.explain(lambda *a: fused_experts(*a) * 2)(x, w1, w2, weights.bfloat16(), ids)
    assert explain.graph_break_count == 0, [str(b.reason) for b in explain.break_reasons]
    assert explain.graph_count == 1


@CUDA
@pytest.mark.parametrize("num_tokens", [1, 3, 16, 64])
def test_fused_experts_matches_naive_and_precomputed_alignment(num_tokens):
    from mstar.model.components.moe import dispatch_experts_fused
    from mstar.utils.fused_moe import MoeAlignment, fused_experts, moe_block_m
    from mstar.utils.fused_moe.align import moe_align_block_size

    # The Command A+ expert shape, which has a tuned table on GB200.
    experts, hidden, inter, top_k = 128, 4096, 1024, 8
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    weights, ids = torch.rand(num_tokens, experts, device="cuda").topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)

    out = fused_experts(x, w1, w2, weights, ids)
    naive = dispatch_experts_fused(x, w1, w2, experts, ids, weights)
    torch.testing.assert_close(out, naive, atol=2e-2, rtol=2e-2)

    block_m = moe_block_m(num_tokens, w1, top_k)
    alignment = MoeAlignment(*moe_align_block_size(ids.to(torch.int32), block_m, experts), block_m)
    precomputed = fused_experts(x, w1, w2, weights, ids, alignment=alignment)
    torch.testing.assert_close(precomputed, out, atol=0, rtol=0)

    with pytest.raises(AssertionError, match="BLOCK_SIZE_M"):
        fused_experts(x, w1, w2, weights, ids, alignment=alignment._replace(block_m=block_m * 2))
