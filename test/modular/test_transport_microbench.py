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
    with pytest.raises(SystemExit, match="not whole float32"):
        microbench.parse_args(["--dtype", "float32", "--sizes", "1,5"])
    assert microbench.parse_args(["--dtype", "float32", "--sizes", "4,64"]).sizes == [4, 64]
    for flag in ("--duration", "--bg-rate", "--probe-interval-ms", "--timeout-s"):
        for bad in ("nan", "inf", "0", "-1"):
            with pytest.raises(SystemExit, match="finite number > 0"):
                microbench.parse_args([flag, bad])
    with pytest.raises(SystemExit, match="larger than --max-bytes-per-size"):
        microbench.parse_args(["--sizes", "63,64,65", "--max-bytes-per-size", "64"])
    assert microbench.parse_args(["--sizes", "63,64", "--max-bytes-per-size", "64"]).sizes == [63, 64]


def test_unanswered_probes_count_toward_the_percentiles():
    rtts, after_burst = [0.001], [0.001]
    sent_at, tagged = {7: 10.0, 8: 14.0}, {8}
    assert microbench.count_unanswered(sent_at, tagged, rtts, after_burst, now=16.0) == 2
    assert rtts == [0.001, 6.0, 2.0] and after_burst == [0.001, 2.0]
    assert microbench.percentile(rtts, 99) > 1.0


class _ReachedMakeCommError(Exception):
    pass


@pytest.mark.parametrize(("role", "devices", "made_current"), [
    ("Sender", ["--device", "cuda:1", "--receiver-device", "cuda:2"], ["cuda:1"]),
    ("Receiver", ["--device", "cuda:1", "--receiver-device", "cuda:2"], ["cuda:2"]),
    ("Sender", ["--device", "cuda"], []),
    ("Receiver", ["--device", "cpu"], []),
])
def test_each_role_makes_its_device_current_before_building(monkeypatch, role, devices, made_current):
    """The transports record their CUDA events on the current device's
    stream, so ``cuda:N`` other than 0 has to be current before any is made.
    ``cuda`` alone already means the current one."""
    args = microbench.parse_args(devices)
    current = []
    monkeypatch.setattr(microbench.torch.cuda, "set_device", lambda d: current.append(str(d)))

    def make_comm(my_id, args):
        raise _ReachedMakeCommError(list(current))

    monkeypatch.setattr(microbench, "make_comm", make_comm)
    with pytest.raises(_ReachedMakeCommError) as built:
        getattr(microbench, role)(args)
    assert built.value.args[0] == made_current


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
    # latency falling with size is no bandwidth: the flat model, not a
    # negative slope's intercept with a perfect R^2
    falling = fit_latency([64, 1024], [20e-6, 10e-6])
    assert math.isinf(falling.bandwidth_bps)
    assert falling.fixed_s == pytest.approx(falling.predict(1024))
    assert 10e-6 < falling.fixed_s < 20e-6 and falling.r2 < 1


def test_crossover_of_two_lines():
    cheap_fixed = LinearFit(fixed_s=10e-6, bandwidth_bps=1e9, r2=1.0)
    cheap_bytes = LinearFit(fixed_s=100e-6, bandwidth_bps=10e9, r2=1.0)
    # 10us + S/1e9 == 100us + S/1e10  ->  S = 90e-6 / (1e-9 - 1e-10) = 100 kB
    assert fit_crossover(cheap_fixed, cheap_bytes) == pytest.approx(1e5)
    assert fit_crossover(cheap_fixed, cheap_fixed) is None
    sizes = [1, 2, 4, 8]
    assert measured_crossover(sizes, [1, 2, 3, 4], [3, 3, 3, 3]) == 8  # equal at 4, swapped at 8
    assert measured_crossover(sizes, [1, 1, 1, 1], [2, 2, 2, 2]) is None
    # a tie is not a change of winner
    assert measured_crossover([64, 1024], [1, 1], [2, 1]) is None
    assert measured_crossover([1, 2, 3], [1, 1, 3], [2, 1, 2]) == 3
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


def _main_on_free_ports(argv):
    """``microbench.main`` on a fresh port pair, retried if another process
    took one of the ports between the check and the bind."""
    for attempt in range(3):
        try:
            return microbench.main([*argv, "--port", str(_free_port_pair()), "--timeout-s", "30"])
        except Exception as exc:  # noqa: BLE001 - the bind error type depends on the mesh
            if attempt == 2 or not ("in use" in str(exc) or "receiver exited" in str(exc)):
                raise


def test_smoke_both_roles_on_localhost(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    paths = ["control", "shm"] + (["tcp", "tcp-pool"] if MOONCAKE_IMPORT_ERROR is None else [])
    out = tmp_path / "r.json"
    # two in flight, so the pool path's slots are reused out of order
    results = _main_on_free_ports([
        "--paths", ",".join(paths), "--sizes", "64,4K", "--iters", "5", "--warmup", "3",
        "--inflight", "2", "--tcp-device", "lo", "--shm-dir", str(tmp_path),
        "--json", str(out), "--quiet",
    ])
    saved = json.loads(out.read_text())
    assert saved["paths"].keys() == results["paths"].keys() == set(paths)
    for path in paths:
        rows = saved["paths"][path]["results"]
        assert [r["size"] for r in rows] == [64, 4096]
        assert all(r["iters"] == 5 and r["p50_us"] > 0 for r in rows)
        # the receiver's phases cover the timed sends only, not the warmup
        assert all(set(r["phases_n"].values()) == {5} for r in rows), rows
        assert saved["paths"][path]["fit"] is not None
    assert {(c["a"], c["b"]) for c in saved["crossovers"]} == {
        (a, b) for i, a in enumerate(paths) for b in paths[i + 1:]
    }
    # every SHM file was unlinked once its ack came back; the namespace lock
    # is held for the life of the process
    assert not [p for p in tmp_path.glob("mstar_*") if p.suffix != ".lock"]


def test_interference_reports_extra_p99(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    results = _main_on_free_ports([
        "--interference", "--bg-sizes", "1K", "--bg-streams", "4", "--bg-rate", "50",
        "--duration", "0.5", "--quiet",
    ])
    inter = results["interference"]
    points = inter["points"]
    assert [p["bg_size"] for p in points] == [0, 1024]
    assert points[1]["burst_bytes"] == 4096 and points[1]["bg_received"] > 0
    assert all(p["probes"] > 0 and p["after_burst_p99_us"] > 0 for p in points)
    assert inter["idle"]["probes"] > 0 and inter["idle"]["bg_received"] == 0
    assert points[1]["bursts"] > 0
    assert points[1]["achieved_rate_hz"] == pytest.approx(points[1]["bursts"] / 0.5)


class _PollRecorder:
    """Records how ``recv`` waits: the Rust communicator's blocking receive
    waits in 50 ms slices, so a short wait has to go through the poll."""

    def __init__(self, ready):
        self.ready, self.polls, self.blocking = ready, [], False

    def get_all_new_messages(self, blocking=False, timeout_s=None):
        self.blocking = self.blocking or blocking
        return ["m"] if self.ready and self.polls else []

    def poll_for_messages(self, timeout_ms=20):
        self.polls.append(timeout_ms)
        return self.ready


def test_recv_waits_no_longer_than_asked():
    comm = _PollRecorder(ready=False)
    assert microbench.recv(comm, 0.002) == []
    assert comm.polls == [2] and not comm.blocking
    comm = _PollRecorder(ready=True)
    assert microbench.recv(comm, 0.002) == ["m"]
    comm = _PollRecorder(ready=False)
    assert microbench.recv(comm, 0.0) == [] and comm.polls == []


def test_crossover_labels_follow_the_fit_and_the_points_separately():
    def path(fixed, bw, p50s):
        return {"fit": {"fixed_s": fixed, "bandwidth_bps": bw, "r2": 1.0},
                "results": [{"size": s, "p50_us": v} for s, v in p50s]}
    # A = 1us + S/1e9 and B = 2us + S/2e9 cross at 2000 B; measured only above it
    paths = {"A": path(1e-6, 1e9, [(4096, 5.1), (16384, 17.4)]),
             "B": path(2e-6, 2e9, [(4096, 4.0), (16384, 10.2)])}
    (c,) = microbench.crossovers(paths)
    assert c["fit_bytes"] == pytest.approx(2000)
    assert c["fit_cheaper_below"] == "A"
    assert c["measured_cheaper_first"] == "B" and c["measured_bytes"] is None
