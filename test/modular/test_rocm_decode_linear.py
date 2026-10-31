"""What the ``decode_gemm`` table says about ``mstar::decode_linear`` (a tuned hipBLASLt solution per shape): the
solution a shape reads, and that a table from another stack keeps its kernels but not its solutions. CPU only."""

from __future__ import annotations

import pytest

from mstar.kernels import configs
from mstar.platforms import rocm_gemm

ENTRY = {"impl": "hipblaslt_tuned", "candidates": {"hipblaslt_tuned": {"us": 1.0, "solution": 7},
                                                   "triton": {"us": 2.0, "block_n": 32}}}
TRITON = {"impl": "triton", "candidates": {"triton": {"us": 1.0, "block_n": 16},
                                           "hipblaslt_tuned": {"us": 2.0, "solution": 9}}}


@pytest.fixture
def table(monkeypatch):
    facts = type("F", (), {"arch": "gfx942", "product": "mi325x"})()
    platform = type("P", (), {"device_facts": lambda self, device=None: facts})()
    monkeypatch.setattr("mstar.kernels.registry.current_platform", platform)
    config = {"classes": {"rows8": {"impl": "hipblaslt"}},
              "shapes": {"k32-n64-rows8": ENTRY, "k32-n128-rows8": TRITON},
              "provenance": {"stack": "torch-0-hip-0-triton-0"}}
    monkeypatch.setattr(configs, "load_config", lambda kernel, arch, product=None, root=None: config)
    monkeypatch.setattr(rocm_gemm, "_have_aiter", lambda: True)
    monkeypatch.setattr(rocm_gemm, "_stack", lambda: "torch-0-hip-0-triton-0")
    for cached in (rocm_gemm._solutions_apply, rocm_gemm._decode_solution):
        cached.cache_clear()
    yield config
    for cached in (rocm_gemm._solutions_apply, rocm_gemm._decode_solution):
        cached.cache_clear()


def test_a_shape_reads_its_solution_only_where_the_table_chose_it(table):
    assert rocm_gemm.gemv_impl(8, 64, 32) == "hipblaslt_tuned"
    assert rocm_gemm._decode_solution(8, 64, 32) == 7
    assert rocm_gemm.gemv_impl(8, 128, 32) == "triton"
    assert rocm_gemm._decode_solution(8, 128, 32) is None
    assert rocm_gemm.gemv_impl(8, 256, 32) == "hipblaslt"
    assert rocm_gemm._decode_solution(8, 256, 32) is None


def test_a_table_from_another_stack_keeps_its_kernels_but_not_its_solutions(table, monkeypatch):
    monkeypatch.setattr(rocm_gemm, "_stack", lambda: "torch-1-hip-1-triton-1")
    assert rocm_gemm.gemv_impl(8, 64, 32) == "hipblaslt"
    assert rocm_gemm.gemv_impl(8, 128, 32) == "triton"


def test_the_losers_knobs_stay_readable(table):
    assert rocm_gemm.DECODE_GEMM.params({"rows": 8, "n": 64, "k": 32}, "triton")["block_n"] == 32
    assert rocm_gemm.DECODE_GEMM.choose({"rows": 8, "n": 64, "k": 32}).params["solution"] == 7


def test_wide_buckets_need_a_tuned_entry(table, monkeypatch):
    monkeypatch.setattr(rocm_gemm, "skinny_gemm_passes", lambda: ("pre", "post"))
    rocm_gemm.tunes_wide_decode.cache_clear()
    assert rocm_gemm.tunes_wide_decode()
    table["shapes"] = {"k32-n128-rows8": TRITON}
    rocm_gemm.tunes_wide_decode.cache_clear()
    assert not rocm_gemm.tunes_wide_decode()
    rocm_gemm.tunes_wide_decode.cache_clear()
