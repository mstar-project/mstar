"""The transport microbenchmark: argument parsing, the latency fit, and a CPU
smoke run of both roles on localhost (control and SHM always, Mooncake TCP
when it imports)."""
import json
import math
import socket

import pytest

from benchmark.transport import microbench
from benchmark.transport.fit import (
    LinearFit,
    fit_crossover,
    fit_latency,
    measured_crossover,
    parse_size,
    parse_sizes,
    percentile,
)
from mstar.communication.tensors import MOONCAKE_IMPORT_ERROR


def test_sizes_parse_as_binary_units_and_log_ranges():
    assert parse_size("64") == 64
    assert parse_size("4K") == 4096
    assert parse_size("16KiB") == 16384
    assert parse_size("4M") == 4 << 20
    assert parse_sizes("64:4M:x4") == [64 * 4 ** i for i in range(9)]
    assert parse_sizes("1K,64,1K") == [64, 1024]
    for bad in ("", "4X", "0", "64:4M", "64:4M:x1", "1M:64:x4"):
        with pytest.raises(ValueError):
            parse_sizes(bad)


def test_arguments_are_checked():
    args = microbench.parse_args(["--paths", "control,tcp", "--sizes", "64,1K", "--inflight", "4"])
    assert args.paths == ["control", "tcp"]
    assert args.sizes == [64, 1024]
    assert args.receiver_device == "cpu"
    with pytest.raises(SystemExit):
        microbench.parse_args(["--paths", "carrier-pigeon"])
    with pytest.raises(SystemExit):
        microbench.parse_args(["--inflight", "0"])
    with pytest.raises(SystemExit):
        microbench.parse_args(["--dtype", "float7"])


def test_fit_recovers_fixed_cost_and_bandwidth():
    sizes = [64 * 4 ** i for i in range(9)]
    lat = [20e-6 + s / 2e9 for s in sizes]
    fit = fit_latency(sizes, lat)
    assert fit.fixed_s == pytest.approx(20e-6, rel=1e-6)
    assert fit.bandwidth_bps == pytest.approx(2e9, rel=1e-6)
    assert fit.r2 == pytest.approx(1.0)
    assert fit_latency([64], [1e-5]) is None
    flat = fit_latency([64, 1024], [1e-5, 1e-5])
    assert math.isinf(flat.bandwidth_bps) and flat.to_json()["bandwidth_bps"] is None


def test_crossover_of_two_lines():
    cheap_fixed = LinearFit(fixed_s=10e-6, bandwidth_bps=1e9, r2=1.0)
    cheap_bytes = LinearFit(fixed_s=100e-6, bandwidth_bps=10e9, r2=1.0)
    # 10us + S/1e9 == 100us + S/1e10  ->  S = 90e-6 / (1e-9 - 1e-10) = 100 kB
    assert fit_crossover(cheap_fixed, cheap_bytes) == pytest.approx(1e5)
    assert fit_crossover(cheap_fixed, cheap_fixed) is None
    sizes = [1, 2, 4, 8]
    assert measured_crossover(sizes, [1, 2, 3, 4], [3, 3, 3, 3]) == 4
    assert measured_crossover(sizes, [1, 1, 1, 1], [2, 2, 2, 2]) is None
    assert percentile([3, 1, 2], 50) == 2
    assert math.isnan(percentile([], 99))


def _free_port_pair() -> int:
    for _ in range(50):
        with socket.socket() as a, socket.socket() as b:
            a.bind(("127.0.0.1", 0))
            port = a.getsockname()[1]
            try:
                b.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port
    pytest.skip("no free port pair")


def test_smoke_both_roles_on_localhost(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    paths = ["control", "shm"] + (["tcp"] if MOONCAKE_IMPORT_ERROR is None else [])
    out = tmp_path / "r.json"
    results = microbench.main([
        "--paths", ",".join(paths), "--sizes", "64,4K", "--iters", "5", "--warmup", "1",
        "--port", str(_free_port_pair()), "--tcp-device", "lo", "--shm-dir", str(tmp_path),
        "--json", str(out), "--quiet",
    ])
    saved = json.loads(out.read_text())
    assert saved["paths"].keys() == results["paths"].keys() == set(paths)
    for path in paths:
        rows = saved["paths"][path]["results"]
        assert [r["size"] for r in rows] == [64, 4096]
        assert all(r["iters"] == 5 and r["p50_us"] > 0 for r in rows)
        assert saved["paths"][path]["fit"] is not None
    assert {(c["a"], c["b"]) for c in saved["crossovers"]} == {
        (a, b) for i, a in enumerate(paths) for b in paths[i + 1:]
    }
    # every SHM file was unlinked once its ack came back
    assert not list(tmp_path.glob("mstar_*"))


def test_interference_reports_extra_p99(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    results = microbench.main([
        "--interference", "--bg-sizes", "1K", "--bg-streams", "4", "--bg-rate", "50",
        "--duration", "0.5", "--port", str(_free_port_pair()), "--quiet",
    ])
    inter = results["interference"]
    points = inter["points"]
    assert [p["bg_size"] for p in points] == [0, 1024]
    assert points[1]["burst_bytes"] == 4096 and points[1]["bg_received"] > 0
    assert all(p["probes"] > 0 and p["after_burst_p99_us"] > 0 for p in points)
    assert inter["idle"]["probes"] > 0 and inter["idle"]["bg_received"] == 0
