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
    monkeypatch.setattr(registry, "_device", lambda device: torch.device("cuda", 0))
    monkeypatch.setattr(registry, "device_key", lambda device: DeviceKey("Test GPU", "sm_00"))
    registry.reset_lookups()
    kernels._TILE_ROWS.clear()
    yield tmp_path / "moe_gemm" / "Test_GPU.json"
    registry.reset_lookups()
    kernels._TILE_ROWS.clear()


def _write(path, rows, top_k=2, shape=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 2, "op": "moe_gemm", "entries": [{
        "dtype": "bfloat16", "shape": shape or {"E": 8, "N": 64, "K": 32, "top_k": top_k},
        "provenance": {"device": "Test GPU", "date": "2026-10-28", "harness": "test"},
        "rows": {key: {"impl": "triton", "config": config} for key, config in rows.items()}}]}))


def test_get_config_falls_back_to_default(tuned_table):
    _write(tuned_table, {"*": {"BLOCK_SIZE_M": 64}}, top_k=4)  # another shape's entry
    up, down = kernels.get_config(M=4, E=8, N=64, K=32, top_k=2)
    assert up == down == kernels.get_default_config(4, 8, 64, 32, 2)


def test_get_config_takes_the_row_up_to_m_and_down_overrides(tuned_table):
    tile = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
    _write(tuned_table, {
        "2": {**tile, "down": {"BLOCK_SIZE_K": 64, "BLOCK_SIZE_M": 64}},
        "*": {**tile, "BLOCK_SIZE_M": 64},
    })
    up, down = kernels.get_config(M=2, E=8, N=64, K=32, top_k=2)
    assert up == tile
    # The down GEMM shares the up GEMM's alignment, so BLOCK_SIZE_M is not overridable.
    assert down == {**tile, "BLOCK_SIZE_K": 64}
    assert [kernels.get_config(M=m, E=8, N=64, K=32, top_k=2)[0]["BLOCK_SIZE_M"] for m in (0, 1, 3, 5000)] == \
        [16, 16, 64, 64]
    # fp16 activations have no entry here
    assert kernels.get_config(M=2, E=8, N=64, K=32, top_k=2, dtype=torch.float16)[0] == \
        kernels.get_default_config(2, 8, 64, 32, 2)


def test_a_row_without_its_tiles_is_refused(tuned_table):
    """The default tiles depend on the token count, so a row can't defer to them."""
    _write(tuned_table, {"*": None})
    with pytest.raises(ValueError, match="needs its tiles"):
        kernels.get_config(M=4, E=8, N=64, K=32, top_k=2)


def test_command_a_plus_keeps_its_gb200_tiles():
    """#377's GB200 table took the nearest tuned batch size (a tie to the smaller); its rows
    here take the counts up to the midpoint of their batch size and the next one, which picks the
    same tiles at every count."""
    path = table.TABLE_DIR / "moe_gemm" / "NVIDIA_GB200.json"
    (entry,) = table.load_table(path, "moe_gemm")
    assert dict(entry.shape) == {"E": 128, "N": 2048, "K": 4096, "top_k": 8}
    tuned = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512)
    assert entry.buckets() == (*((a + b) // 2 for a, b in zip(tuned, tuned[1:], strict=False)), None)
    assert entry.rows[0][1].impl == "triton"
    assert dict(entry.rows[0][1].config) == {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4,
        "num_stages": 5, "down": {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4,
                                  "num_stages": 5}}


@pytest.mark.parametrize("compiling", [False, True], ids=["eager", "compiling"])
@pytest.mark.parametrize("cuda_align", [True, False], ids=["cuda_align", "torch_align"])
def test_fused_experts_asks_for_the_tuned_tiles_unless_a_compile_would_lose_them(
        monkeypatch, compiling, cuda_align):
    """Only a compiled call without the CUDA align op takes the default tiles: the op's torch
    fallback breaks the graph between the lookup and the launch. An eager call takes the
    table's tiles either way, as does a caller that passes its own alignment."""
    from mstar.utils.fused_moe import align, runner

    monkeypatch.setattr(align, "_cuda_op_available", lambda: cuda_align)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    asked = []

    class LookupReachedError(Exception):
        pass

    def get_config(*args, tuned=True, **kwargs):
        asked.append(tuned)
        raise LookupReachedError

    monkeypatch.setattr(runner, "get_config", get_config)
    x = torch.zeros(2, 32, dtype=torch.bfloat16)
    w1, w2 = torch.zeros(4, 16, 32, dtype=torch.bfloat16), torch.zeros(4, 32, 8, dtype=torch.bfloat16)
    weights, ids = torch.ones(2, 2, dtype=torch.bfloat16), torch.zeros(2, 2, dtype=torch.int64)
    with pytest.raises(LookupReachedError):
        runner.fused_experts(x, w1, w2, weights, ids)
    with pytest.raises(LookupReachedError):
        runner.fused_experts(x, w1, w2, weights, ids, alignment=runner.MoeAlignment(None, None, None, 16))
    assert asked == [cuda_align or not compiling, True]


@CUDA
@pytest.mark.parametrize("cuda_align", [True, False], ids=["cuda_align", "torch_align"])
@pytest.mark.parametrize("dynamic", [False, None], ids=["captured", "eager_forward"])
def test_compiled_fused_experts_takes_the_tables_tiles(tuned_table, monkeypatch, dynamic, cuda_align):
    """Compiled as the engine compiles (dynamic=False for a captured bucket; dynamic=None for
    the eager forward, where the token count turns symbolic after a few sizes, under the engine's
    dynamo settings), fused_experts runs the table's row for M and matches the eager call. The
    tiles must reach the Triton launch as constants: dynamo refuses a symbolic num_warps. The
    alignment's torch fallback breaks the graph, so there a compiled call takes the default
    tiles."""
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
        "*": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
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
    check are constants to dynamo, so a tuned config reaches the launch as a constant."""
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
