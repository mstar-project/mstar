"""The fp8_linear table: bench_linears turns per-kernel timings into per-token-count rows, and the committed
H200 entries for Qwen3.5-27B's shapes pick the kernels job 1438 measured fastest (FlashInfer's swapAB GEMM, the
one vLLM runs below 32 tokens, where the built-in rule ran w8a16)."""

from pathlib import Path

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


def test_the_h200_table_runs_flashinfer_where_vllm_does():
    entries = {tuple(sorted(dict(e.shape).items())): {b: c.impl for b, c in e.rows}
               for e in load_table(TABLE, "fp8_linear")}
    for n, k, glu in ((14336, 5120, 0), (5120, 6144, 0), (5120, 17408, 0), (17408, 5120, 1)):
        rows = entries[tuple(sorted({"N": n, "K": k, "glu": glu}.items()))]
        assert all(rows[m] == "flashinfer" for m in (2, 8, 16, 24, 48, 64)), (n, k, glu, rows)
