"""fp8_linear rows for a model's own shapes: the registry's boot log names a sweep that exists (fp8_linear is one
of benchmark.kernels.sweep's ops), fp8_rows reads the shapes back out of that very log line, and turns logs and
named per-rank sets into one sweep command (CPU)."""

import logging

import pytest
import torch

pytest.importorskip("triton")

from benchmark.kernels import fp8_linear as sweep_fp8  # noqa: E402
from benchmark.kernels.sweep import _sweeper  # noqa: E402
from benchmark.quantization import fp8_rows  # noqa: E402
from mstar.kernels import DeviceKey, registry, table  # noqa: E402
from mstar.utils import dense_fp8  # noqa: E402


def test_the_boot_log_for_a_shape_without_rows_names_a_sweep_that_exists_and_reads_back(tmp_path, monkeypatch,
                                                                                         caplog):
    monkeypatch.setenv(table.TABLES_ENV, str(tmp_path))
    monkeypatch.setattr(table, "TABLE_DIR", tmp_path / "package")
    monkeypatch.setattr(registry, "_device", lambda device: torch.device("cuda", 0))
    monkeypatch.setattr(registry, "device_key", lambda device: DeviceKey("Test GPU", "sm_00"))
    registry.reset_lookups()
    with caplog.at_level(logging.INFO, logger=registry.__name__):
        assert dense_fp8.FP8_LINEAR.lookup(dense_fp8.FP8_DTYPE, {"N": 5120, "K": 2048, "glu": 0}) is None
        assert dense_fp8.FP8_LINEAR.lookup(dense_fp8.FP8_DTYPE, {"N": 1536, "K": 6144, "glu": 1}) is None
    registry.reset_lookups()
    log = tmp_path / "server.log"
    log.write_text(caplog.text + caplog.text)  # a shape logged by several ranks is one shape
    assert fp8_rows.shapes_from_logs([log]) == [(5120, 2048, 0), (1536, 6144, 1)]
    assert "benchmark.kernels.sweep fp8_linear --dtype float8_e4m3fn --shape N=5120,K=2048,glu=0" in caplog.text
    assert _sweeper("fp8_linear") is sweep_fp8.Fp8LinearSweep


def test_logs_sets_and_shapes_become_one_sweep_command(tmp_path):
    log = tmp_path / "srv.log"
    log.write_text("x kernel table: no fp8_linear float8_e4m3fn N=2048,K=4096,glu=0 entry for NVIDIA B300; ...\n")
    cmd = fp8_rows.sweep_command(fp8_rows.shapes_from_logs([log]) + fp8_rows.SETS["qwen3_omni_thinker_tp1"],
                                 "/t", False, "omni")
    assert cmd[1:6] == ["-m", "benchmark.kernels.sweep", "fp8_linear", "--dtype", "float8_e4m3fn"]
    assert cmd.count("--shape") == 3 and cmd[cmd.index("--out") + 1] == "/t"
    assert fp8_rows.sweep_command([(1, 2, 0)], None, True, None)[-1] == "--write"


def test_every_named_set_is_per_rank_block_fp8():
    for name, shapes in fp8_rows.SETS.items():
        for n, k, glu in shapes:
            assert n % 128 == 0 or name == "glm53_tp8", (name, n)  # GLM's 2624-row q_a + kv_a: a partial block
            assert k % 128 == 0 and glu in (0, 1), (name, n, k)


def test_w8a16_candidates_split_k_evenly():
    for m in (1, 4, 8, 64):
        for k in (1536, 2048, 6144):
            cands = list(sweep_fp8.w8a16_candidates(m, k))
            assert cands and all(k % (c["SPLIT"] * c["BK"]) == 0 for c in cands)
            assert all(c["DOT"] == (m > 4) for c in cands)
