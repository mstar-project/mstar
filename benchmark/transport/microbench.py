"""Per-tensor latency and throughput of M*'s tensor paths, across sizes.

One process sends, one receives, over M*'s own control mesh (the ZMQ
communicator and wire codec) and transport managers, so the numbers are what
a worker pays. See README.md next to this file.

    python -m benchmark.transport.microbench                       # both roles here
    python -m benchmark.transport.microbench --paths control,tcp --sizes 64:4M:x4
    python -m benchmark.transport.microbench --interference --bg-sizes 1K,4K,16K,64K
    # two machines: the same arguments on each, with --role
    python -m benchmark.transport.microbench --role receiver --sender-host A --receiver-host B
    python -m benchmark.transport.microbench --role sender   --sender-host A --receiver-host B
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch

from benchmark.transport.fit import (
    LinearFit,
    fit_crossover,
    fit_latency,
    human_size,
    measured_crossover,
    parse_size,
    parse_sizes,
    percentile,
)

SENDER, RECEIVER = "worker_0", "worker_1"

#: control: the bytes ride in a control message (serialize, send, deserialize).
#: tcp / rdma: M*'s Mooncake path (register, descriptor, read, ack, cleanup).
#: shm / arena: M*'s same-host paths (per-uuid file, Rust arena).
#: tcp-pool / rdma-pool: Mooncake reads from a buffer registered once, copied
#: into (or, with --pool-mode zerocopy, produced in) per send. Not an M* path
#: today; the contrast with tcp shows what per-tensor registration costs.
PATHS = ("control", "tcp", "rdma", "shm", "arena", "tcp-pool", "rdma-pool")
SAME_HOST_PATHS = ("shm", "arena")


# -- arguments ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmark.transport.microbench",
        description="Per-tensor latency and throughput of M*'s tensor paths.",
    )
    p.add_argument("--role", choices=("both", "sender", "receiver"), default="both",
                   help="both: spawn the receiver on this host (default)")
    p.add_argument("--paths", default="control,tcp,shm",
                   help=f"comma list of {', '.join(PATHS)}; unavailable ones are skipped with a note")
    p.add_argument("--sizes", default="64:4M:x4", help="64:4M:x4 (log-spaced) or 64,4K,1M")
    p.add_argument("--iters", type=int, default=200, help="timed sends per size and path")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max-bytes-per-size", default="1G",
                   help="cap iters so one size moves at most this much")
    p.add_argument("--inflight", type=int, default=1, help="tensors in flight at once")
    p.add_argument("--device", default="cpu", help="sender buffers: cpu or cuda:N")
    p.add_argument("--receiver-device", default=None, help="default: same as --device")
    p.add_argument("--dtype", default="uint8")
    p.add_argument("--pool-mode", choices=("copy", "zerocopy"), default="copy",
                   help="*-pool paths: copy each tensor into the registered pool, or produce it there")
    p.add_argument("--sender-host", default="127.0.0.1", help="address the sender binds and advertises")
    p.add_argument("--receiver-host", default="127.0.0.1", help="address the receiver binds and advertises")
    p.add_argument("--port", type=int, default=19700, help="sender port; the receiver uses port+1")
    p.add_argument("--tcp-device", default="", help="Mooncake TCP device (e.g. lo for loopback)")
    p.add_argument("--rdma-device", default="", help="RDMA_DEVICE_NAME for the rdma paths")
    p.add_argument("--shm-dir", default=None)
    p.add_argument("--interference", action="store_true",
                   help="instead of the sweep: extra p99 of small control messages under payload load")
    p.add_argument("--bg-sizes", default="1K,4K,16K,64K", help="interference payload sizes")
    p.add_argument("--bg-streams", type=int, default=32, help="payloads per burst (one per stream)")
    p.add_argument("--bg-rate", type=float, default=60.0, help="bursts per second (decode steps/s)")
    p.add_argument("--probe-interval-ms", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=5.0, help="seconds per interference point")
    p.add_argument("--p99-budget-ms", type=float, default=1.0,
                   help="interference: report the largest burst whose extra p99 stays under this")
    p.add_argument("--gpu-busy-ms", type=float, default=0.0,
                   help="keep each process's default CUDA stream this many ms ahead with spin kernels, "
                        "as a worker's next step is queued while it moves tensors (cuda devices only)")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="torch intra-op threads in both roles (default: torch's, as M* workers keep). "
                        "On a busy host the pool's threads inflate host copies of 32K+ elements")
    p.add_argument("--timeout-s", type=float, default=60.0,
                   help="give up when no send completes for this long (a receiver that died)")
    p.add_argument("--json", default=None, help="write results here")
    p.add_argument("--quiet", action="store_true")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.paths = [s.strip() for s in args.paths.split(",") if s.strip()]
    bad = [x for x in args.paths if x not in PATHS]
    if bad:
        raise SystemExit(f"unknown path(s) {bad}; choose from {', '.join(PATHS)}")
    args.sizes = parse_sizes(args.sizes)
    args.bg_sizes = parse_sizes(args.bg_sizes)
    args.max_bytes_per_size = parse_size(args.max_bytes_per_size)
    args.receiver_device = args.receiver_device or args.device
    if args.iters < 1 or args.inflight < 1 or args.warmup < 0:
        raise SystemExit("--iters and --inflight must be >= 1, --warmup >= 0")
    if not hasattr(torch, args.dtype) or not isinstance(getattr(torch, args.dtype), torch.dtype):
        raise SystemExit(f"unknown dtype {args.dtype!r}")
    itemsize = getattr(torch, args.dtype).itemsize
    odd = [s for s in [*args.sizes, *args.bg_sizes] if s % itemsize]
    if odd:
        raise SystemExit(f"sizes {odd} are not whole {args.dtype} tensors ({itemsize} bytes each)")
    if args.gpu_busy_ms < 0:
        raise SystemExit("--gpu-busy-ms must be >= 0")
    # NaN passes every comparison the run loops make, so it would never end
    for flag in ("bg_rate", "probe_interval_ms", "duration", "timeout_s"):
        if not math.isfinite(getattr(args, flag)) or getattr(args, flag) <= 0:
            raise SystemExit(f"--{flag.replace('_', '-')} must be a finite number > 0")
    if not math.isfinite(args.p99_budget_ms) or args.p99_budget_ms < 0:
        raise SystemExit("--p99-budget-ms must be a finite number >= 0")
    over = [s for s in args.sizes if s > args.max_bytes_per_size]
    if over:
        raise SystemExit(f"sizes {over} are larger than --max-bytes-per-size ({args.max_bytes_per_size})")
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise SystemExit("--torch-threads must be >= 1")
        torch.set_num_threads(args.torch_threads)
    return args


# -- availability ------------------------------------------------------------

def rdma_devices() -> list[str]:
    try:
        return sorted(os.listdir("/sys/class/infiniband"))
    except OSError:
        return []


def unavailable_reason(path: str, args) -> str | None:
    """Why ``path`` cannot run here, or None."""
    from mstar.communication.tensors import MOONCAKE_IMPORT_ERROR
    if path in ("tcp", "rdma", "tcp-pool", "rdma-pool") and MOONCAKE_IMPORT_ERROR is not None:
        return f"mooncake not importable ({type(MOONCAKE_IMPORT_ERROR).__name__})"
    if path.startswith("rdma") and not rdma_devices():
        return "no RDMA device under /sys/class/infiniband"
    if path in SAME_HOST_PATHS and args.sender_host != args.receiver_host:
        return "same-host path, but sender and receiver hosts differ"
    if path == "arena":
        try:
            from mstar_rust import SegmentedShmArena, ShmArena  # noqa: F401
        except ImportError:
            return "mstar_rust (the arena) not installed"
    for dev in {args.device, args.receiver_device}:
        if dev.startswith("cuda") and not torch.cuda.is_available():
            return "CUDA requested but not available"
    return None


# -- the control mesh ----------------------------------------------------------

def make_comm(my_id: str, args):
    """M*'s communicator (Rust or pyzmq, as ``make_communicator`` picks), with
    each side's endpoint taken from the arguments so the two can be on
    different machines."""
    from mstar.communication.communicator import CommProtocol, ZMQCommunicator
    base = ZMQCommunicator
    if os.getenv("MSTAR_RUST_ZMQ", "AUTO").upper() != "0":
        try:
            from mstar.communication.rust_communicator import RustZMQCommunicator, _RustZmq
            if _RustZmq is not None:
                base = RustZMQCommunicator
        except ImportError:
            pass
    endpoints = {
        SENDER: f"tcp://{args.sender_host}:{args.port}",
        RECEIVER: f"tcp://{args.receiver_host}:{args.port + 1}",
        # only names the SHM file namespace; never bound
        "conductor": f"tcp://{args.sender_host}:{args.port + 2}",
    }

    class BenchCommunicator(base):
        def _endpoint(self, entity_id: str) -> str:
            return endpoints[entity_id]

    os.environ.pop("MSTAR_ZMQ_TRANSPORT", None)
    peer = RECEIVER if my_id == SENDER else SENDER
    return BenchCommunicator(my_id, [peer], protocol=CommProtocol.TCP)


def recv(comm, timeout_s: float) -> list:
    """Messages within ``timeout_s``. Polls rather than blocking receives: the
    Rust communicator's blocking receive waits in 50 ms slices, which would
    overrun the interference schedule."""
    msgs = comm.get_all_new_messages()
    if msgs or timeout_s < 1e-3:
        return msgs
    if comm.poll_for_messages(timeout_ms=int(timeout_s * 1000)):
        return comm.get_all_new_messages()
    return []


# -- tensor managers -------------------------------------------------------------

def make_manager(path: str, my_id: str, device: str, comm, args):
    from mstar.communication.communicator import CommProtocol
    from mstar.communication.tensors import create_tensor_communication_manager
    host = args.sender_host if my_id == SENDER else args.receiver_host
    if path in ("shm", "arena"):
        os.environ["MSTAR_SHM_ARENA"] = "1" if path == "arena" else "0"
        return create_tensor_communication_manager(
            protocol=CommProtocol.SHM, my_entity_id=my_id, hostname=host,
            device=device, communicator=comm, shm_dir=args.shm_dir,
        )
    if path in ("rdma", "rdma-pool") and args.rdma_device:
        os.environ["RDMA_DEVICE_NAME"] = args.rdma_device
    return create_tensor_communication_manager(
        protocol=CommProtocol.RDMA if path.startswith("rdma") else CommProtocol.TCP,
        my_entity_id=my_id, hostname=host, device=device, communicator=comm,
        tcp_transfer_device=args.tcp_device,
    )


def _fwd_info(rid: str):
    from mstar.conductor.request_info import CurrentForwardPassInfo
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="bench", fwd_index=0, random_seed=0, max_tokens=1,
    )


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _use_device(device: str) -> None:
    """Make ``device`` current, as a worker does when it is built: the
    transports record their CUDA events on the current device's stream.
    ``cuda`` alone means the current device, so it is left as is."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev)


class HostCopier:
    """Device <-> host copies on a side stream, waited on alone, the way a
    worker keeps them from queueing behind its next step."""

    def __init__(self, device: str):
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None

    def to_bytes(self, t: torch.Tensor) -> bytes:
        flat = t.reshape(-1).view(torch.uint8)
        if self.stream is None:
            return flat.numpy().tobytes()
        host = torch.empty(flat.numel(), dtype=torch.uint8, pin_memory=True)
        with torch.cuda.stream(self.stream):
            host.copy_(flat, non_blocking=True)
        self.stream.synchronize()
        return host.numpy().tobytes()

    def from_bytes(self, data: bytes, dtype: torch.dtype, shape) -> torch.Tensor:
        if not data:
            return torch.empty(shape, dtype=dtype, device=self.device)
        host = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype).reshape(shape)
        if self.stream is None:
            return host
        with torch.cuda.stream(self.stream):
            out = host.to(self.device)
        self.stream.synchronize()
        return out


class GpuBusy:
    """Spin kernels on the default stream, about ``ms`` queued at any time.

    A worker moves step N's tensors while step N+1 is already queued on its
    default stream, so anything that waits on that stream (a read whose event
    is recorded there, a legacy-stream copy inside a transport) waits for it.
    """

    def __init__(self, device: str, ms: float):
        self.device = torch.device(device)
        self.ms = ms
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if ms > 0 and self.device.type == "cuda":
            self.cycles = self._calibrate()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _calibrate(self) -> int:
        with torch.cuda.device(self.device):
            probe = 20_000_000
            torch.cuda._sleep(probe // 10)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda._sleep(probe)
            end.record()
            end.synchronize()
            return int(probe * self.ms / start.elapsed_time(end))

    def _run(self) -> None:
        with torch.cuda.device(self.device):
            prev = None
            while not self._stop.is_set():
                torch.cuda._sleep(self.cycles)
                ev = torch.cuda.Event()
                ev.record()
                if prev is not None:
                    prev.synchronize()
                prev = ev

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# -- sender ------------------------------------------------------------------------

@dataclass
class SizeResult:
    size: int
    iters: int
    lat_s: list[float] = field(default_factory=list)
    phases: dict[str, list[float]] = field(default_factory=dict)
    wall_s: float = 0.0

    def add_phase(self, name: str, value: float) -> None:
        self.phases.setdefault(name, []).append(value)

    def summary(self) -> dict:
        return {
            "size": self.size,
            "iters": self.iters,
            "p50_us": percentile(self.lat_s, 50) * 1e6,
            "p99_us": percentile(self.lat_s, 99) * 1e6,
            "mean_us": sum(self.lat_s) / len(self.lat_s) * 1e6 if self.lat_s else float("nan"),
            "throughput_MBps": self.size * len(self.lat_s) / self.wall_s / 1e6 if self.wall_s else 0.0,
            "phases_p50_us": {k: percentile(v, 50) * 1e6 for k, v in self.phases.items()},
            "phases_n": {k: len(v) for k, v in self.phases.items()},
        }


class Sender:
    def __init__(self, args):
        self.args = args
        _use_device(args.device)
        self.comm = make_comm(SENDER, args)
        self.dtype = getattr(torch, args.dtype)
        self.copier = HostCopier(args.device)
        self._held: deque = deque()
        self.busy = GpuBusy(args.device, args.gpu_busy_ms)

    def _await(self, want, timeout_s: float = 30.0):
        """Messages until ``want(msg)`` returns something; stray ones are kept."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            while self._held:
                got = want(self._held.popleft())
                if got is not None:
                    return got
            for msg in recv(self.comm, 0.05):
                self._held.append(msg)
        raise TimeoutError("receiver did not answer; is it running with the same arguments?")

    def _ctl(self, kind: str, **kw) -> None:
        self.comm.send(RECEIVER, {"kind": kind, **kw})

    def _reply(self, kind: str):
        return lambda m: m if isinstance(m, dict) and m.get("kind") == kind else None

    def handshake(self, receiver_alive=lambda: True) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if not receiver_alive():
                raise RuntimeError("the receiver exited before answering (see its output)")
            self._ctl("hello")
            try:
                self._await(self._reply("ready"), timeout_s=1.0)
                return
            except TimeoutError:
                continue
        raise TimeoutError("no receiver")

    def iters_for(self, size: int) -> int:
        return max(1, min(self.args.iters, self.args.max_bytes_per_size // size))

    def _tensor(self, size: int) -> torch.Tensor:
        n = max(1, size // self.dtype.itemsize)
        return torch.randint(0, 100, (n,), dtype=torch.int64).to(self.dtype).to(self.args.device)

    # A window of ``inflight`` sends. ``start(seq, t0)`` issues send number
    # ``seq`` (counted from ``base``) and returns its key; ``finish(msg)``
    # returns the keys a reply completes.
    def _windowed(self, total: int, start, finish, result: SizeResult | None, base: int = 0) -> None:
        t_start: dict = {}
        issued = done = 0
        last_progress = time.monotonic()
        while done < total:
            if time.monotonic() - last_progress > self.args.timeout_s:
                raise TimeoutError(
                    f"no send completed for {self.args.timeout_s:.0f} s "
                    f"({done} of {total} done); did the receiver die?")
            while issued < total and len(t_start) < self.args.inflight:
                t0 = time.perf_counter()
                t_start[start(base + issued, t0)] = t0
                issued += 1
            msgs = list(self._held)
            self._held.clear()
            msgs += recv(self.comm, 0.0 if msgs else 0.05)
            for msg in msgs:
                for key in finish(msg):
                    t0 = t_start.pop(key, None)
                    if t0 is None:
                        continue
                    done += 1
                    last_progress = time.monotonic()
                    if result is not None:
                        result.lat_s.append(time.perf_counter() - t0)

    def run_path(self, path: str) -> list[dict]:
        self._ctl("phase", path=path)
        self._await(self._reply("phase_ready"), timeout_s=60)
        runner = {
            "control": self._run_control,
            "tcp": self._run_manager, "rdma": self._run_manager,
            "shm": self._run_manager, "arena": self._run_manager,
            "tcp-pool": self._run_pool, "rdma-pool": self._run_pool,
        }[path]
        out = []
        mgr = make_manager(path, SENDER, self.args.device, self.comm, self.args) \
            if path in ("tcp", "rdma", "shm", "arena") else None
        for size in self.args.sizes:
            res = SizeResult(size=size, iters=self.iters_for(size))
            runner(path, size, res, mgr)
            self._ctl("size_done")
            report = self._await(self._reply("report"))
            for k, v in report["phases"].items():
                res.phases[k] = v
            out.append(res.summary())
            if not self.args.quiet:
                s = out[-1]
                print(f"  {path:9s} {human_size(size):>9s}  p50 {s['p50_us']:9.1f} us  "
                      f"p99 {s['p99_us']:9.1f} us  {s['throughput_MBps']:9.1f} MB/s", flush=True)
        if mgr is not None and hasattr(mgr, "_async_reader"):
            mgr._async_reader.shutdown()
        self._ctl("phase_done")
        self._await(self._reply("phase_closed"))
        return out

    def _run_control(self, path, size, res: SizeResult, mgr) -> None:
        from mstar.utils.ipc_format import WorkerMessageType
        tensors = [self._tensor(size) for _ in range(self.args.inflight)]
        _sync(self.args.device)

        warm = self.args.warmup

        def start(i, t0):
            t = tensors[i % len(tensors)]
            data = self.copier.to_bytes(t)
            t1 = time.perf_counter()
            self._ctl("payload", seq=i, dtype=self.args.dtype, shape=list(t.shape), data=data)
            if i >= warm:
                res.add_phase("serialize", t1 - t0)
                res.add_phase("send", time.perf_counter() - t1)
            return i

        def finish(msg):
            if getattr(msg, "message_type", None) == WorkerMessageType.TENSOR_RECEIVED:
                return list(msg.body.successful_tensors)
            return []

        self._windowed(warm, start, finish, None)
        self._ctl("timing_start")
        t_wall = time.perf_counter()
        self._windowed(res.iters, start, finish, res, base=warm)
        res.wall_s = time.perf_counter() - t_wall

    def _run_manager(self, path, size, res: SizeResult, mgr) -> None:
        from mstar.graph.base import GraphEdge
        from mstar.utils.ipc_format import InputSignals, WorkerMessage, WorkerMessageType
        tensors = [self._tensor(size) for _ in range(self.args.inflight)]
        _sync(self.args.device)
        free = deque(range(len(tensors)))
        slot_of: dict[int, int] = {}
        rid_of: dict[int, str] = {}
        timed = [False]

        def start(i, t0):
            slot = free.popleft()
            rid = f"b{size}_{i}_{time.monotonic_ns()}"
            infos = mgr.store_and_return_tensor_info(rid, {"x": [tensors[slot]]}, skip_cuda_sync=True)["x"]
            for info in infos:
                mgr.tensor_store.increment_ref(info.uuid, 1)
            t1 = time.perf_counter()
            mgr.register_for_send(rid, infos, skip_cuda_sync=True)
            t2 = time.perf_counter()
            edge = GraphEdge(next_node="sink", name="x", tensor_info=infos)
            self.comm.send(RECEIVER, WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(request_id=rid, inputs=[edge], request_info=_fwd_info(rid)),
            ))
            if timed[0]:
                res.add_phase("store", t1 - t0)
                res.add_phase("register", t2 - t1)
                res.add_phase("send", time.perf_counter() - t2)
            uuid = infos[0].uuid
            slot_of[uuid], rid_of[uuid] = slot, rid
            return uuid

        def finish(msg):
            if getattr(msg, "message_type", None) != WorkerMessageType.TENSOR_RECEIVED:
                return []
            acked = msg.body.successful_tensors
            t0 = time.perf_counter()
            mgr.dereference_batch(list(acked), list(acked.values()))
            # an ack covers whatever reads finished together, so it names uuids, not a request
            for uuid in acked:
                if uuid in rid_of:
                    mgr.cleanup_request(rid_of[uuid])
            if timed[0]:
                res.add_phase("cleanup", time.perf_counter() - t0)
            for uuid in acked:
                if uuid in slot_of:
                    free.append(slot_of.pop(uuid))
                    rid_of.pop(uuid, None)
            return list(acked)

        self._windowed(self.args.warmup, start, finish, None)
        self._ctl("timing_start")
        timed[0] = True
        t_wall = time.perf_counter()
        self._windowed(res.iters, start, finish, res)
        res.wall_s = time.perf_counter() - t_wall

    def _run_pool(self, path, size, res: SizeResult, mgr) -> None:
        engine = self._pool_engine(path)
        n = len(self._pool_slots)
        tensors = [self._tensor(size) for _ in range(n)]
        _sync(self.args.device)
        if self.args.pool_mode == "zerocopy":
            for slot, t in zip(self._pool_slots, tensors, strict=True):
                slot[: t.nbytes].copy_(t.reshape(-1).view(torch.uint8))
            _sync(self.args.device)
        free = deque(range(n))
        slot_of: dict[int, int] = {}
        timed = [False]

        def start(i, t0):
            slot = free.popleft()
            t = tensors[slot]
            buf = self._pool_slots[slot]
            if self.args.pool_mode == "copy":
                buf[: t.nbytes].copy_(t.reshape(-1).view(torch.uint8))
                _sync(self.args.device)
            t1 = time.perf_counter()
            self._ctl("pool", seq=i, slot=slot, addr=buf.data_ptr(), nbytes=t.nbytes,
                      session=engine.get_session_id())
            if timed[0]:
                res.add_phase("copy", t1 - t0)
                res.add_phase("send", time.perf_counter() - t1)
            slot_of[i] = slot
            return i

        def finish(msg):
            if isinstance(msg, dict) and msg.get("kind") == "pool_ack":
                free.append(slot_of.pop(msg["seq"]))
                return [msg["seq"]]
            return []

        self._windowed(self.args.warmup, start, finish, None)
        self._ctl("timing_start")
        timed[0] = True
        t_wall = time.perf_counter()
        self._windowed(res.iters, start, finish, res, base=self.args.warmup)
        res.wall_s = time.perf_counter() - t_wall

    def _pool_engine(self, path):
        if getattr(self, "_pool_path", None) != path:
            from mstar.communication.communicator import CommProtocol
            from mstar.communication.tensors import MooncakeTransferEngine
            proto = CommProtocol.RDMA if path.startswith("rdma") else CommProtocol.TCP
            self._engine = MooncakeTransferEngine(
                hostname=self.args.sender_host, protocol=proto,
                tcp_transfer_device=self.args.tcp_device,
            )
            cap = max(self.args.sizes)
            self._pool = torch.empty(cap * self.args.inflight, dtype=torch.uint8, device=self.args.device)
            if self._engine.register_memory(self._pool.data_ptr(), self._pool.nbytes) != 0:
                raise RuntimeError("could not register the sender pool")
            self._pool_slots = [self._pool[i * cap:(i + 1) * cap] for i in range(self.args.inflight)]
            self._pool_path = path
        return self._engine

    # -- interference ----------------------------------------------------------

    def run_interference(self) -> dict:
        a = self.args
        self._ctl("phase", path="control")
        self._await(self._reply("phase_ready"), timeout_s=60)
        # The frames of a step go out whether or not they carry bytes, so the
        # baseline is a burst of empty ones; "idle" (no bursts) is for scale.
        idle = {"bg_size": None, **self._interference_point(0, burst=False)}
        points = [{"bg_size": size, **self._interference_point(size)} for size in [0, *a.bg_sizes]]
        self._ctl("phase_done")
        self._await(self._reply("phase_closed"))
        base = points[0]
        budget_us = a.p99_budget_ms * 1e3
        for p in points:
            p["burst_bytes"] = p["bg_size"] * a.bg_streams
            p["extra_p99_us"] = p["probe_p99_us"] - base["probe_p99_us"]
            p["extra_after_burst_p99_us"] = p["after_burst_p99_us"] - base["after_burst_p99_us"]
        # a point whose probes did not all come back has only a lower bound on its p99
        ok = [p["burst_bytes"] for p in points[1:]
              if not p["unanswered_probes"] and not base["unanswered_probes"]
              and max(p["extra_p99_us"], p["extra_after_burst_p99_us"]) <= budget_us]
        return {
            "idle": idle,
            "bg_streams": a.bg_streams, "bg_rate_hz": a.bg_rate,
            "probe_interval_ms": a.probe_interval_ms, "duration_s": a.duration,
            "p99_budget_ms": a.p99_budget_ms, "points": points,
            "largest_burst_within_budget": max(ok) if ok else 0,
        }

    def _interference_point(self, bg_size: int, burst: bool = True) -> dict:
        a = self.args
        payload = self.copier.to_bytes(self._tensor(bg_size)) if bg_size else b""
        sent_at: dict[int, float] = {}
        bursts = 0
        rtts: list[float] = []
        after_burst: list[float] = []
        tagged: set[int] = set()
        seq = 0
        burst_period = 1.0 / a.bg_rate
        probe_period = a.probe_interval_ms / 1e3
        now = time.perf_counter()
        next_burst = next_probe = now
        end = now + a.duration

        def handle(msgs):
            for msg in msgs:
                if isinstance(msg, dict) and msg.get("kind") == "echo":
                    t0 = sent_at.pop(msg["seq"], None)
                    if t0 is not None:
                        rtt = time.perf_counter() - t0
                        rtts.append(rtt)
                        if msg["seq"] in tagged:
                            after_burst.append(rtt)

        while True:
            now = time.perf_counter()
            if now >= end:
                break
            if burst and now >= next_burst:
                # A completion right behind a step's frames: timed from before
                # the frames, so encoding them counts as it would in a worker.
                t_burst = time.perf_counter()
                for s in range(a.bg_streams):
                    self._ctl("bg", stream=s, data=payload)
                bursts += 1
                sent_at[seq] = t_burst
                tagged.add(seq)
                self._ctl("probe", seq=seq)
                seq += 1
                next_burst = max(next_burst + burst_period, now - burst_period)
            if now >= next_probe:
                sent_at[seq] = time.perf_counter()
                self._ctl("probe", seq=seq)
                seq += 1
                next_probe = max(next_probe + probe_period, now - probe_period)
            wake = min(next_probe, next_burst if burst else end, end)
            handle(recv(self.comm, min(max(0.0, wake - time.perf_counter()), 0.002)))
        deadline = time.perf_counter() + 5.0
        while sent_at and time.perf_counter() < deadline:
            handle(recv(self.comm, 0.05))
        unanswered = count_unanswered(sent_at, tagged, rtts, after_burst, time.perf_counter())
        self._ctl("size_done")
        report = self._await(self._reply("report"))
        out = {
            "probes": len(rtts),
            "unanswered_probes": unanswered,
            "probe_p50_us": percentile(rtts, 50) * 1e6,
            "probe_p99_us": percentile(rtts, 99) * 1e6,
            "after_burst_p50_us": percentile(after_burst, 50) * 1e6 if after_burst else 0.0,
            "after_burst_p99_us": percentile(after_burst, 99) * 1e6 if after_burst else 0.0,
            "bg_received": report.get("bg_received", 0),
            "bursts": bursts,
            "achieved_rate_hz": bursts / a.duration,
        }
        if not a.quiet:
            what = human_size(bg_size) if burst else "idle"
            print(f"  interference bg {what:>9s} x{a.bg_streams}"
                  f" @ {a.bg_rate:.0f} Hz  probe p50 {out['probe_p50_us']:8.1f} us  "
                  f"p99 {out['probe_p99_us']:8.1f} us", flush=True)
        return out

    def close(self) -> None:
        self._ctl("stop")
        self.busy.stop()


def count_unanswered(sent_at: dict[int, float], tagged: set[int], rtts: list[float],
                     after_burst: list[float], now: float) -> int:
    """Count the probes still unanswered at the drain deadline and add each one's
    age so far to the samples: a lower bound on its round trip, so the slowest
    probes are not dropped from the percentiles."""
    for seq, t0 in sent_at.items():
        rtts.append(now - t0)
        if seq in tagged:
            after_burst.append(now - t0)
    return len(sent_at)


# -- receiver ----------------------------------------------------------------------

class Receiver:
    def __init__(self, args):
        self.args = args
        _use_device(args.receiver_device)
        self.comm = make_comm(RECEIVER, args)
        self.copier = HostCopier(args.receiver_device)
        self.mgr = None
        self.path = None
        self.phases: dict[str, list[float]] = {}
        self.t_recv: dict[str, float] = {}
        self.bg_received = 0
        self.pool_reader = None
        self.busy = GpuBusy(args.receiver_device, args.gpu_busy_ms)

    def _send(self, msg) -> None:
        self.comm.send(SENDER, msg)

    def _phase(self, name: str, value: float) -> None:
        self.phases.setdefault(name, []).append(value)

    def serve(self) -> None:
        while True:
            busy = bool(self.mgr is not None and self.mgr.pending) or bool(
                self.pool_reader is not None and self.pool_pending)
            # with reads in flight, poll like a worker loop instead of blocking
            msgs = self.comm.get_all_new_messages() if busy else recv(self.comm, 0.05)
            for msg in msgs:
                if self._handle(msg) == "stop":
                    return
            if self.mgr is not None and self.mgr.pending:
                self._drain_ready()
            if self.pool_reader is not None:
                self._drain_pool()

    def _handle(self, msg):
        from mstar.utils.ipc_format import TensorReceived, WorkerMessage, WorkerMessageType
        if not isinstance(msg, dict):
            if msg.message_type == WorkerMessageType.INPUT_SIGNALS:
                body = msg.body
                self.t_recv[body.request_id] = time.perf_counter()
                self.mgr.start_read_tensors(body.request_id, body.inputs)
            return None
        kind = msg.get("kind")
        if kind == "hello":
            self._send({"kind": "ready"})
        elif kind == "phase":
            self.path = msg["path"]
            if self.path in ("tcp", "rdma", "shm", "arena"):
                self.mgr = make_manager(self.path, RECEIVER, self.args.receiver_device, self.comm, self.args)
            elif self.path.endswith("-pool"):
                self._setup_pool()
            self._send({"kind": "phase_ready"})
        elif kind == "payload":
            t0 = time.perf_counter()
            tensor = self.copier.from_bytes(msg["data"], getattr(torch, msg["dtype"]), msg["shape"])
            self._phase("deserialize", time.perf_counter() - t0)
            del tensor
            self._send(WorkerMessage(
                message_type=WorkerMessageType.TENSOR_RECEIVED,
                body=TensorReceived(successful_tensors={msg["seq"]: 1},
                                    failed_tensor_ids=[]),
            ))
        elif kind == "pool":
            from mstar.communication.tensors import TransferReadInfo
            # the sender's slot: a later send cannot reuse it before this one's ack
            local = self.pool_slots[msg["slot"]]
            fut = self.pool_reader.submit([TransferReadInfo(
                source_session_id=msg["session"], local_ptr=local.data_ptr(),
                remote_ptr=msg["addr"], nbytes=msg["nbytes"],
            )])
            self.pool_pending.append((msg["seq"], fut, time.perf_counter()))
        elif kind == "bg":
            self.copier.from_bytes(msg["data"], torch.uint8, [len(msg["data"])])
            self.bg_received += 1
        elif kind == "timing_start":
            self.phases = {}
        elif kind == "probe":
            self._send({"kind": "echo", "seq": msg["seq"]})
        elif kind == "size_done":
            while self.mgr is not None and self.mgr.pending:
                self._drain_ready()
            self._send({"kind": "report",
                        "phases": {k: v for k, v in self.phases.items()},
                        "bg_received": self.bg_received})
            self.phases = {}
            self.bg_received = 0
        elif kind == "phase_done":
            if self.mgr is not None and hasattr(self.mgr, "_async_reader"):
                self.mgr._async_reader.shutdown()
            self.mgr = None
            self.pool_reader = None
            self._send({"kind": "phase_closed"})
        elif kind == "stop":
            return "stop"
        return None

    def _drain_ready(self) -> None:
        for rid, edges in self.mgr.get_ready_tensors().items():
            t0 = self.t_recv.pop(rid, None)
            if t0 is not None:
                self._phase("read", time.perf_counter() - t0)
            for edge in edges:
                self.mgr.dereference_batch_uniform([i.uuid for i in edge.tensor_info])
            self.mgr.cleanup_request(rid)

    def _setup_pool(self) -> None:
        from mstar.communication.communicator import CommProtocol
        from mstar.communication.tensors import AsyncMooncakeReader, MooncakeTransferEngine
        proto = CommProtocol.RDMA if self.path.startswith("rdma") else CommProtocol.TCP
        self.engine = MooncakeTransferEngine(
            hostname=self.args.receiver_host, protocol=proto, tcp_transfer_device=self.args.tcp_device,
        )
        cap = max(self.args.sizes)
        self.pool = torch.empty(cap * self.args.inflight, dtype=torch.uint8, device=self.args.receiver_device)
        if self.engine.register_memory(self.pool.data_ptr(), self.pool.nbytes) != 0:
            raise RuntimeError("could not register the receiver pool")
        self.pool_slots = [self.pool[i * cap:(i + 1) * cap] for i in range(self.args.inflight)]
        self.pool_reader = AsyncMooncakeReader(self.engine._engine, device=self.args.receiver_device)
        self.pool_pending: list = []

    def _drain_pool(self) -> None:
        still = []
        for seq, fut, t0 in self.pool_pending:
            if fut is None or fut.done():
                if fut is not None:
                    fut.result()
                self._phase("read", time.perf_counter() - t0)
                self._send({"kind": "pool_ack", "seq": seq})
            else:
                still.append((seq, fut, t0))
        self.pool_pending = still


def receiver_main(argv: list[str]) -> None:
    Receiver(parse_args(argv)).serve()


# -- report ------------------------------------------------------------------------

def crossovers(paths: dict) -> list[dict]:
    names = [p for p, r in paths.items() if r.get("fit")]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = paths[a]["fit"], paths[b]["fit"]
            la = LinearFit(fa["fixed_s"], fa["bandwidth_bps"] or float("inf"), fa["r2"])
            lb = LinearFit(fb["fixed_s"], fb["bandwidth_bps"] or float("inf"), fb["r2"])
            ra, rb = paths[a]["results"], paths[b]["results"]
            common = sorted({r["size"] for r in ra} & {r["size"] for r in rb})
            pa = {r["size"]: r["p50_us"] for r in ra}
            pb = {r["size"]: r["p50_us"] for r in rb}
            first = next((s for s in common if pa[s] != pb[s]), None)
            out.append({
                "a": a, "b": b,
                "fit_bytes": fit_crossover(la, lb),
                # below the fitted crossover the line with the smaller fixed cost wins
                "fit_cheaper_below": a if la.fixed_s < lb.fixed_s else b,
                "measured_bytes": measured_crossover(common, [pa[s] for s in common], [pb[s] for s in common]),
                # at the smallest size where they differ
                "measured_cheaper_first": None if first is None else (a if pa[first] < pb[first] else b),
            })
    return out


def render(results: dict) -> str:
    lines = [f"host {results['host']}  sender {results['sender_device']}  receiver {results['receiver_device']}"
             f"  inflight {results['inflight']}  mesh {results['mesh']}  mooncake {results.get('mooncake')}"
             f"  load {results['loadavg_start']:.1f} on {results['cpus']} cpus"
             f"  torch threads {results['torch_threads']}"
             + (f"  gpu busy {results['gpu_busy_ms']} ms" if results.get("gpu_busy_ms") else ""), ""]
    paths = results["paths"]
    names = [p for p in paths if paths[p].get("results")]
    if names:
        sizes = sorted({r["size"] for p in names for r in paths[p]["results"]})
        head = "size        " + "".join(f"{p + ' p50/p99 us':>26s}" for p in names)
        lines += [head, "-" * len(head)]
        for s in sizes:
            row = f"{human_size(s):<12s}"
            for p in names:
                r = next((r for r in paths[p]["results"] if r["size"] == s), None)
                row += f"{(f'{r['p50_us']:.1f} / {r['p99_us']:.1f}' if r else '-'):>26s}"
            lines.append(row)
        lines += ["", "fit: latency(S) = fixed + S / bandwidth (p50)"]
        for p in names:
            f = paths[p].get("fit")
            if f:
                bw = f"{f['bandwidth_bps'] / 1e6:.0f} MB/s" if f["bandwidth_bps"] else "flat"
                lines.append(f"  {p:10s} fixed {f['fixed_s'] * 1e6:8.1f} us  bandwidth {bw:>12s}  r2 {f['r2']:.3f}")
        for c in results.get("crossovers", []):
            fb = (f"{human_size(c['fit_bytes'])} ({c['fit_cheaper_below']} cheaper below)"
                  if c["fit_bytes"] else "none")
            mb = human_size(c["measured_bytes"]) if c["measured_bytes"] else "none in range"
            first = c["measured_cheaper_first"] or "neither"
            lines.append(f"  crossover {c['a']} vs {c['b']}: fit {fb}; measured {mb} "
                         f"({first} cheaper at the smallest size)")
    for p, r in paths.items():
        if r.get("skipped"):
            lines.append(f"  skipped {p}: {r['skipped']}")
    inter = results.get("interference")
    if inter:
        lines += ["", f"interference: {inter['bg_streams']} payloads per burst at {inter['bg_rate_hz']:.0f} Hz,"
                      f" probe every {inter['probe_interval_ms']} ms"]
        idle = inter["idle"]
        lines.append(f"idle (no bursts): probe p50 {idle['probe_p50_us']:.1f} us, p99 {idle['probe_p99_us']:.1f} us")
        lines.append(f"{'payload':>10s} {'burst':>10s} {'probe p50/p99 us':>18s} {'behind burst p50/p99 us':>24s}"
                     f" {'extra p99 us (probe / behind burst)':>36s} {'bursts/s':>9s}")
        for pt in inter["points"]:
            lines.append(f"{human_size(pt['bg_size']) if pt['bg_size'] else 'empty':>10s} "
                         f"{human_size(pt['burst_bytes']) if pt['burst_bytes'] else '-':>10s} "
                         f"{pt['probe_p50_us']:8.1f} / {pt['probe_p99_us']:7.1f} "
                         f"{pt['after_burst_p50_us']:12.1f} / {pt['after_burst_p99_us']:9.1f} "
                         f"{pt['extra_p99_us']:20.1f} / {pt['extra_after_burst_p99_us']:9.1f}"
                         f" {pt['achieved_rate_hz']:9.1f}")
        lines.append(f"largest burst with extra p99 <= {inter['p99_budget_ms']} ms: "
                     f"{human_size(inter['largest_burst_within_budget'])}")
    return "\n".join(lines)


def run_sender(args, receiver_alive=None) -> dict:
    sender = Sender(args)
    sender.handshake(receiver_alive or (lambda: True))
    results = {
        "host": f"{socket.gethostname()} ({platform.processor() or platform.machine()})",
        "sender_host": args.sender_host, "receiver_host": args.receiver_host,
        "sender_device": args.device, "receiver_device": args.receiver_device,
        "inflight": args.inflight, "dtype": args.dtype, "pool_mode": args.pool_mode,
        "mesh": type(sender.comm).__mro__[1].__name__, "paths": {},
        "torch": torch.__version__,
        "loadavg_start": os.getloadavg()[0], "cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(), "gpu_busy_ms": args.gpu_busy_ms,
    }
    try:
        import importlib.metadata as md
        results["mooncake"] = next((d.version for d in md.distributions()
                                    if (d.metadata["Name"] or "").startswith("mooncake")), None)
    except Exception:  # noqa: BLE001
        results["mooncake"] = None
    try:
        if args.interference:
            results["interference"] = sender.run_interference()
        else:
            for path in args.paths:
                why = unavailable_reason(path, args)
                if why:
                    results["paths"][path] = {"skipped": why}
                    if not args.quiet:
                        print(f"  skipping {path}: {why}", flush=True)
                    continue
                rs = sender.run_path(path)
                fit = fit_latency([r["size"] for r in rs], [r["p50_us"] / 1e6 for r in rs])
                results["paths"][path] = {"results": rs, "fit": fit.to_json() if fit else None}
            results["crossovers"] = crossovers(results["paths"])
    finally:
        sender.close()
    results["loadavg_end"] = os.getloadavg()[0]
    return results


def main(argv: list[str] | None = None) -> dict | None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.role == "receiver":
        receiver_main(argv)
        return None
    child = None
    if args.role == "both":
        child = mp.get_context("spawn").Process(target=receiver_main, args=(argv,), daemon=True)
        child.start()
    try:
        results = run_sender(args, receiver_alive=(child.is_alive if child is not None else None))
    finally:
        if child is not None:
            child.join(timeout=10)
            if child.is_alive():
                child.terminate()
    text = render(results)
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
