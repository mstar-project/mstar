"""The fp8_linear table: bench_linears turns per-kernel timings into per-token-count rows; the H200 table has swept
rows for every Qwen3.5 FP8 checkpoint linear; a row naming DeepGEMM (an optional import) runs the built-in rule's
W8A8 GEMM where DeepGEMM doesn't."""

from pathlib import Path

import pytest

from benchmark.quantization.bench_linears import _write_tables
from mstar.kernels.table import load_table

TABLE = Path(__file__).resolve().parents[2] / "mstar/kernels/tables/fp8_linear/NVIDIA_H200.json"
IMPLS = ["w8a16", "flashinfer", "cublas"]


def _rows(layer, n, times):
    rows = []
    for m, per in times.items():
        rows.append({"layer": layer, "n": n, "k": 512, "tokens": m, "format": "bf16", "us": 100.0})
        for impl, (us, err) in per.items():
            rows.append({"layer": layer, "n": n, "k": 512, "tokens": m, "format": f"fp8_block[{impl}]", "us": us,
                         "rel_l2": err})
    return rows


def _entry(path, n, glu):
    (entry,) = [e for e in load_table(path, "fp8_linear") if dict(e.shape) == {"N": n, "K": 512, "glu": glu}]
    return {bucket: choice.impl for bucket, choice in entry.rows}


def test_each_token_count_gets_its_fastest_kernel(tmp_path):
    times = {1: {"w8a16": (10, 0.03), "flashinfer": (12, 0.05)},
             64: {"w8a16": (50, 0.03), "flashinfer": (20, 0.05), "cublas": (30, 0.05)},
             1024: {"w8a16": (900, 0.03), "flashinfer": (210, 0.05), "cublas": (200, 0.05)}}
    (path,) = _write_tables(_rows("qkv", 256, times), tmp_path, IMPLS, "NVIDIA H200", "sm_90")
    assert _entry(path, 256, 0) == {1: "w8a16", 64: "flashinfer", None: "cublas"}


def test_a_swiglu_with_fp8_activations_counts_and_a_wrong_kernel_does_not(tmp_path):
    # 1438's table forced w8a16 on the fused SwiGLU at every count: FlashInfer's 5.3% error was over the old
    # 5% cut, and c=64 lost 23%
    times = {64: {"w8a16": (154, 0.038), "flashinfer": (69, 0.053), "cublas": (60, 0.9)}}
    (path,) = _write_tables(_rows("gate_up_glu", 128, times), tmp_path, IMPLS, "NVIDIA H200", "sm_90")
    assert _entry(path, 128, 1) == {None: "flashinfer"}


def test_the_h200_table_covers_the_qwen3_5_fp8_checkpoints():
    """Every block-fp8 linear Qwen3.5-27B-FP8 (TP1) and 397B-A17B-FP8 (TP4) dispatch has swept H200 rows: a shape
    without them takes the built-in rule, whose untuned w8a16 tile ran 2.3x flashinfer's time on 397B's 5120 x 4096
    at 2-16 tokens (sweep in slurm-1 job 1815)."""
    entries = {tuple(sorted(dict(e.shape).items())): e for e in load_table(TABLE, "fp8_linear")}
    shapes = ((14336, 5120, 0), (16384, 5120, 0), (17408, 5120, 1), (5120, 17408, 0), (5120, 6144, 0),  # 27B TP1
              (5120, 4096, 0), (4608, 4096, 0), (4096, 2048, 0), (256, 4096, 1), (4096, 256, 0))  # 397B TP4
    for n, k, glu in shapes:
        entry = entries.get(tuple(sorted({"N": n, "K": k, "glu": glu}.items())))
        assert entry is not None and "sweep" in entry.provenance["harness"], (n, k, glu)


def test_without_deepgemm_its_rows_run_the_builtin_w8a8_gemm(monkeypatch):
    """DeepGEMM is optional: where it doesn't import, the probe says so (without touching the GPU) and a table row
    naming it runs the built-in rule's W8A8 GEMM for the count, flashinfer's or cuBLAS's."""
    pytest.importorskip("triton")
    import sys

    import torch

    from mstar.kernels import Choice
    from mstar.utils import dense_fp8

    dev = torch.device("cuda", 0)
    monkeypatch.setattr(dense_fp8.current_platform, "cuda_capability", lambda device: (9, 0))
    monkeypatch.setitem(sys.modules, "deep_gemm", None)  # not installed: the import raises
    probed = dense_fp8._probe_dg(dev)
    assert probed is False

    monkeypatch.setattr(dense_fp8, "_table_rows", lambda *a: ((16,), (Choice("triton8"), Choice("deepgemm"))))
    monkeypatch.setitem(dense_fp8._FI_OK, dev, True)
    monkeypatch.setitem(dense_fp8._W8A8_OK, dev, True)
    monkeypatch.setitem(dense_fp8._DG_OK, dev, probed)
    assert [dense_fp8._choose(dev, 5120, 17408, False, m)[0] for m in (8, 32, 4096)] == ["triton8", "flashinfer",
                                                                                       "cublas"]
    monkeypatch.setitem(dense_fp8._DG_OK, dev, True)
    assert dense_fp8._choose(dev, 5120, 17408, False, 32) == ("deepgemm", None)
