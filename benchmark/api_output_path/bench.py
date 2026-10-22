"""CPU cost of the API server's output path for inline decode tokens.

Drives the real code from a RESULT_TOKENS frame to the SSE text a
``/v1/chat/completions`` stream yields, with no GPU, conductor or worker:

  decode   ``wire.decode`` of the frame (what the communicator does per frame)
  route    one ``APIServer._process_messages`` pass carrying the frame
           (the frame's values onto the data worker's queue)
  emit     the data worker thread's ``run`` pass: the model's postprocess
           (detokenize) and the reorder buffer
  drain    one ``_process_messages`` pass with no frame: ``get_result_chunks``,
           append to each ``PendingRequest``, ``notify_many``
  sse      the event loop running every handler's ``_stream_events`` until
           each has yielded this step's token (no HTTP write; see ``--http``)
  relay    instead of ``sse`` with ``--frontend rust``: the Rust-frontend
           bridge's ``_relay`` per chunk (msgpack + a real Rust ZMQ send)

``stages`` runs them one after another on one thread and times each. ``threaded``
runs the real threads (message thread, data worker thread, event loop) over a
real Rust ZMQ socket with frames paced at a target token rate, and reports
per-thread CPU and frame-to-SSE latency.

    python -m benchmark.api_output_path.bench stages --rows 128 --steps 400
    python -m benchmark.api_output_path.bench threaded --rows 128 --rate 16000 --seconds 10
"""
from __future__ import annotations

import argparse
import asyncio
import os
import queue
import statistics
import sys
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path

import torch

from mstar.api_server import data_worker as dw
from mstar.api_server import entrypoint as ep
from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.openai import serving_chat
from mstar.api_server.request_types import APIServerMessage, ResultTokens
from mstar.communication import wire
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.gc_freeze import freeze_after_setup

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test" / "modular"))
from api_server_helpers import make_test_api_server  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

PASSAGE = (
    "The history of computing is a story of abstraction layered on abstraction. "
    "Early machines were programmed by rewiring panels; later, stored programs "
    "let the same hardware run arbitrary instructions. Compilers turned human-"
    "readable code into machine instructions, operating systems multiplexed "
    "scarce processors among many users, and networks stitched machines into "
    "systems larger than any one of them. Each layer hid the one beneath it, "
    "and each, in turn, became the foundation for the next. Today a request to "
    "a language model passes through load balancers, HTTP servers, schedulers, "
    "tensor kernels and back, and every one of those layers adds a little "
    "latency and a little CPU work. When a thousand tokens a second become a "
    "hundred thousand, the small costs add up, and the layers that were "
    "invisible start to matter. Profiling them, one by one, is how a system "
    "stays fast as it grows: measure, find the largest cost, remove it, and "
    "measure again. Résumé, naïve café, 東京 and 🙂 appear too. "
) * 4


# -- the pieces ---------------------------------------------------------------


class _StubTensorManager:
    """The data worker's tensor manager on the inline path: never read."""

    def get_ready_tensors(self):
        return {}

    def has_inflight_reads(self, request_id):
        return False

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        pass


class _OnePassComm:
    """The data worker's communicator: ends ``run`` after one full pass."""

    def __init__(self, stop: threading.Event):
        self.stop = stop
        self.calls = 0

    def get_all_new_messages(self):
        self.calls += 1
        if self.calls >= 2:
            self.stop.set()
        return []

    def send(self, entity, msg):
        pass


class _FrameComm:
    """The API server's communicator: hands out the frames queued on it."""

    def __init__(self):
        self.pending: list = []

    def get_all_new_messages(self):
        out, self.pending = self.pending, []
        return out


def load_model(stub: bool):
    if stub:
        class _M:
            def postprocess(self, output, modality, request_kwargs=None):
                return str(int(output[0])).encode()

            def postprocess_inline(self, values, modality, request_kwargs):
                return [str(v).encode() for v in values]
        return _M(), list(range(1000, 1600))
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = Qwen3_5DenseModel(MODEL_NAME)
    ids = model.tokenizer.encode(PASSAGE, add_special_tokens=False)
    return model, ids


def build_server(model, data_worker_comm=None, wakeup=None):
    """A real ``APIServer`` (built as the modular tests build it) whose
    ``PreprocessWorker`` facade and ``PreprocessWorkerThread`` share queues as
    ``PreprocessWorker.__init__`` wires them."""
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.request_input_queue = queue.Queue()
    facade.result_tensor_input_queue = queue.Queue()
    facade.cleanup_request_queue = queue.Queue()
    facade.abort_request_queue = queue.Queue()
    facade.reads_done_queue = queue.Queue()
    facade.discard_tensor_queue = queue.Queue()
    facade.output_queue = queue.Queue()
    facade.profile_queue = queue.Queue()
    facade.per_request_reading_tensors = {}
    facade.received_chunk_counts = {}
    facade.wakeup = wakeup
    stop = threading.Event()
    facade.stop_event = stop
    thread = PreprocessWorkerThread(
        in_queue=facade.request_input_queue,
        result_tensor_queue=facade.result_tensor_input_queue,
        out_queue=facade.output_queue,
        profile_queue=facade.profile_queue,
        cleanup_request_queue=facade.cleanup_request_queue,
        abort_request_queue=facade.abort_request_queue,
        reads_done_queue=facade.reads_done_queue,
        discard_tensor_queue=facade.discard_tensor_queue,
        stop_event=stop,
        communicator=data_worker_comm or _OnePassComm(stop),
        tensor_manager=_StubTensorManager(),
        model=model,
        wakeup=wakeup,
    )
    server = make_test_api_server(preprocess_worker=facade, model=model, model_name=MODEL_NAME)
    server.timeout_seconds = 3600.0
    return server, facade, thread


def submit(server, rows: int) -> tuple[list[str], list[str]]:
    """Submit ``rows`` streaming chat requests; their client ids and keys."""
    rids, keys = [], []
    for _ in range(rows):
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        server.submit_request(
            text="hi", input_modalities=["text"], output_modalities=["text"],
            model_kwargs={}, streaming=True, request_id=rid,
        )
        rids.append(rid)
        keys.append(server.submission_key(rid))
    # Skip preprocessing (prompt -> conductor): this measures the output side
    # only. Leave the data worker's state as a submitted request leaves it.
    q = server.preprocess_worker.request_input_queue
    while not q.empty():
        q.get_nowait()
    return rids, keys


def mark_live(thread, keys):
    for key in keys:
        thread.in_flight_requests.add(key)
        thread._live_requests[key] = 1


def frame_bytes(keys: list[str], token_ids: list[int], step: int) -> bytes:
    """One decode step's RESULT_TOKENS frame, one value per request."""
    n = len(token_ids)
    msg = APIServerMessage(
        message_type="result_tokens",
        body=ResultTokens(
            request_ids=keys,
            values=[token_ids[(step + 37 * i) % n] for i in range(len(keys))],
            loop_indices=[
                NestedLoopIndices(["decode_loop"], {"decode_loop": step}, step + 1)
                for _ in keys
            ],
            signal="text_inputs", modality="text",
        ),
    )
    return wire.encode(msg)


def one_message_pass(server) -> None:
    """Run ``_process_messages`` for exactly one pass: its end-of-pass sleep
    stops the loop."""
    shim = types.SimpleNamespace(
        monotonic=time.monotonic, time=time.time, perf_counter=time.perf_counter,
        sleep=lambda _s: setattr(server, "running", False),
    )
    real = ep.time
    ep.time = shim
    try:
        server.running = True
        server._process_messages()
    finally:
        ep.time = real


class Consumers:
    """One ``_stream_events`` (or bridge ``_relay``) task per request."""

    def __init__(self, server, rids, frontend: str, bridge=None):
        self.events = 0  # SSE events (or relayed chunks) seen
        self.bytes = 0
        self.tasks = []
        self.first_seen: list[float] = []  # per token, perf_counter when yielded
        self.track = False
        for rid in rids:
            if frontend == "python":
                gen = serving_chat._stream(server, MODEL_NAME, rid, 24000)
                self.tasks.append(asyncio.ensure_future(self._consume_sse(gen)))
            else:
                key = server.submission_key(rid)
                bridge._keys[id(rid)] = key
                chunks = server.iter_result_chunks(rid)
                self.tasks.append(asyncio.ensure_future(
                    bridge._relay(rid, id(rid), chunks)))

    async def _consume_sse(self, gen):
        async for s in gen:
            n = s.count("data: ")
            self.events += n
            self.bytes += len(s)
            if self.track:
                t = time.perf_counter()
                self.first_seen.extend([t] * n)


class _CountingSend:
    """Bridge send that counts the chunks the bridge relays, on top of the
    real codec + Rust ZMQ send."""

    def __init__(self, comm, consumers_ref):
        self.comm = comm
        self.ref = consumers_ref

    def __call__(self, msg):
        self.comm.send("frontend", msg)
        c = self.ref[0]
        if c is not None and msg.get("t") == "chunk":
            c.events += 1
            if c.track:
                c.first_seen.append(time.perf_counter())


def make_bridge(server, sock_dir: str, consumers_ref):
    from mstar.api_server.rust_frontend import RustFrontendBridge
    from mstar.communication.communicator import CommProtocol
    from mstar.communication.rust_communicator import MsgpackCodec, RustZMQCommunicator

    sink = RustZMQCommunicator("frontend", push_ids=[], protocol=CommProtocol.IPC,
                               ipc_socket_path_prefix=sock_dir, codec=MsgpackCodec)
    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = server
    bridge.comm = RustZMQCommunicator("bridge", push_ids=["frontend"], protocol=CommProtocol.IPC,
                                      ipc_socket_path_prefix=sock_dir, codec=MsgpackCodec)
    bridge.running = True
    bridge._keys = {}
    bridge._send = _CountingSend(bridge.comm, consumers_ref)
    return bridge, sink


class GCStats:
    """Time spent in the cyclic GC, by generation, from ``gc.callbacks``."""

    def __init__(self):
        import gc
        self.t = {0: 0.0, 1: 0.0, 2: 0.0}
        self.n = {0: 0, 1: 0, 2: 0}
        self._start = 0.0
        gc.callbacks.append(self._cb)

    def _cb(self, phase, info):
        if phase == "start":
            self._start = time.perf_counter()
        else:
            g = info["generation"]
            self.t[g] += time.perf_counter() - self._start
            self.n[g] += 1

    def report(self, per: float, label: str):
        parts = [f"gen{g}: {self.n[g]}x {self.t[g] * 1e3:.1f} ms" for g in (0, 1, 2)]
        total = sum(self.t.values())
        print(f"gc ({label}): " + ", ".join(parts) + f"; {total / per * 1e3:.3f} ms per step")


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


# -- stages: one thread, stage by stage ---------------------------------------


async def run_stages(args):
    model, token_ids = load_model(args.stub_tokenizer)
    # as the data worker's run() would, minus the per-call warm-up
    dw.warm_up_model(model)
    dw.warm_up_model = lambda _m: None
    server, facade, thread = build_server(model)
    rids, keys = submit(server, args.rows)
    mark_live(thread, keys)
    comm = _FrameComm()
    server.communicator = comm
    sock_dir = tempfile.mkdtemp(prefix="aop", dir="/tmp")
    ref = [None]
    bridge = sink = None
    if args.frontend == "rust":
        bridge, sink = make_bridge(server, sock_dir, ref)
    cons = Consumers(server, rids, args.frontend, bridge)
    ref[0] = cons
    await asyncio.sleep(0)  # let every handler bind its wake event

    frames = [frame_bytes(keys, token_ids, s) for s in range(args.steps)]
    # as the API server does once the workers report ready
    freeze_after_setup("harness")
    gcs = GCStats()
    names = ("decode", "route", "emit", "drain", args.frontend == "python" and "sse" or "relay")
    times = {n: [] for n in names}
    # thread CPU, not wall: the data worker's pass ends in a 1 ms idle sleep
    clock = time.thread_time
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    for step, raw in enumerate(frames):
        target = (step + 1) * args.rows
        t0 = clock()
        msg = wire.decode(raw)
        t1 = clock()
        comm.pending = [msg]
        one_message_pass(server)
        t2 = clock()
        thread.stop_event.clear()
        thread.communicator.calls = 0
        thread.run()
        t3 = clock()
        one_message_pass(server)
        t4 = clock()
        while cons.events < target:
            await asyncio.sleep(0)
            if sink is not None:
                sink._inner.drain()
        t5 = clock()
        for n, a, b in zip(names, (t0, t1, t2, t3, t4), (t1, t2, t3, t4, t5), strict=True):
            times[n].append(b - a)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0

    warm = args.steps // 10
    print(f"rows={args.rows} steps={args.steps} frontend={args.frontend} "
          f"tokenizer={'stub' if args.stub_tokenizer else 'qwen3.5'} "
          f"(first {warm} steps dropped)")
    print(f"{'stage':8s} {'mean_ms/step':>13s} {'p50':>8s} {'p95':>8s} {'us/token':>9s}")
    total = 0.0
    for n in names:
        xs = times[n][warm:]
        m = statistics.mean(xs)
        total += m
        print(f"{n:8s} {m * 1e3:13.3f} {_pct(xs, 50) * 1e3:8.3f} {_pct(xs, 95) * 1e3:8.3f} "
              f"{m / args.rows * 1e6:9.2f}")
    per_tok = total / args.rows
    print(f"{'total':8s} {total * 1e3:13.3f} {'':8s} {'':8s} {per_tok * 1e6:9.2f}")
    for rate in (16000, 32000, 64000):
        print(f"  core fraction at {rate} tok/s: {per_tok * rate:.2f}")
    print(f"process cpu/wall over the run: {cpu / wall:.2f}  (the rest is the 1 ms idle sleep "
          "ending each data worker pass)")
    gcs.report(args.steps, "whole run, included in the stage times")
    for t in cons.tasks:
        t.cancel()


# -- threaded: the real threads over a real socket ----------------------------


def _thread_cpu(tid: int) -> float:
    with open(f"/proc/self/task/{tid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    tck = os.sysconf("SC_CLK_TCK")
    return (int(parts[11]) + int(parts[12])) / tck


async def run_threaded(args):
    from mstar.communication.communicator import CommProtocol
    from mstar.communication.event import EventWakeup
    from mstar.communication.rust_communicator import RustZMQCommunicator

    model, token_ids = load_model(args.stub_tokenizer)
    sock_dir = tempfile.mkdtemp(prefix="aop", dir="/tmp")
    wakeup = EventWakeup()
    dw_comm = RustZMQCommunicator("api_server_preprocess_worker", push_ids=["conductor"],
                                  protocol=CommProtocol.IPC, ipc_socket_path_prefix=sock_dir)
    server, facade, thread = build_server(model, data_worker_comm=dw_comm, wakeup=wakeup)
    server.communicator = RustZMQCommunicator("api_server", push_ids=["conductor"],
                                              protocol=CommProtocol.IPC,
                                              ipc_socket_path_prefix=sock_dir)
    sender = RustZMQCommunicator("worker_0", push_ids=["api_server"],
                                 protocol=CommProtocol.IPC, ipc_socket_path_prefix=sock_dir)
    rids, keys = submit(server, args.rows)
    mark_live(thread, keys)
    ref = [None]
    bridge = sink = None
    if args.frontend == "rust":
        bridge, sink = make_bridge(server, sock_dir, ref)
    cons = Consumers(server, rids, args.frontend, bridge)
    ref[0] = cons
    await asyncio.sleep(0)

    step_period = args.rows / args.rate
    steps = int(args.seconds / step_period)
    frames = [frame_bytes(keys, token_ids, s) for s in range(steps)]
    freeze_after_setup("harness")
    gcs = GCStats()
    sent_at: list[float] = []

    tids = {}

    def msg_thread():
        tids["message"] = threading.get_native_id()
        server._process_messages()

    def data_thread():
        tids["data_worker"] = threading.get_native_id()
        thread.run()

    def feeder():
        tids["feeder"] = threading.get_native_id()
        t_next = time.perf_counter() + 0.2
        for raw in frames:
            while True:
                now = time.perf_counter()
                if now >= t_next:
                    break
                time.sleep(min(0.0005, t_next - now))
            sent_at.append(time.perf_counter())
            sender._inner.send("api_server", raw)
            t_next += step_period

    threads = [threading.Thread(target=f, daemon=True) for f in (msg_thread, data_thread)]
    for t in threads:
        t.start()
    await asyncio.sleep(0.2)
    tids["event_loop"] = threading.get_native_id()
    cons.track = True
    feed = threading.Thread(target=feeder, daemon=True)
    cpu_before = None
    feed.start()
    await asyncio.sleep(0.25)
    cpu_before = {k: _thread_cpu(v) for k, v in tids.items()}
    wall0 = time.perf_counter()
    target = steps * args.rows
    deadline = time.monotonic() + args.seconds * 4 + 10
    while cons.events < target and time.monotonic() < deadline:
        await asyncio.sleep(0.001)
        if sink is not None:
            sink._inner.drain()
    wall = time.perf_counter() - wall0
    cpu_after = {k: _thread_cpu(v) for k, v in tids.items() if k != "feeder"}
    feed.join()
    server.running = False
    thread.stop_event.set()
    wakeup.signal()

    print(f"rows={args.rows} target={args.rate} tok/s step={step_period * 1e3:.2f} ms "
          f"steps={steps} frontend={args.frontend} "
          f"tokenizer={'stub' if args.stub_tokenizer else 'qwen3.5'}")
    got = cons.events
    print(f"delivered {got}/{target} tokens")
    if len(sent_at) > 1:
        achieved = (len(sent_at) - 1) * args.rows / (sent_at[-1] - sent_at[0])
        print(f"feeder achieved {achieved:.0f} tok/s")
    # token j of a request was in step j; map each delivered token to its step
    per_req = args.rows
    lat = []
    seen = cons.first_seen
    # tokens arrive step-major closely enough: the k-th delivered token across
    # all requests belongs to step k // rows, give or take within-step order
    for k, t in enumerate(seen):
        s = k // per_req
        if s < len(sent_at):
            lat.append(t - sent_at[s])
    if lat:
        lat_ms = [x * 1e3 for x in lat[len(lat) // 10:]]
        print(f"frame->yield latency ms: p50 {_pct(lat_ms, 50):.2f}  p90 {_pct(lat_ms, 90):.2f}  "
              f"p99 {_pct(lat_ms, 99):.2f}  max {max(lat_ms):.2f}")
    print("thread CPU over the measured window (fraction of one core):")
    for k, after in cpu_after.items():
        if k in cpu_before:
            print(f"  {k:12s} {(after - cpu_before[k]) / wall:.2f}")
    total = sum(cpu_after[k] - cpu_before[k] for k in cpu_after) / wall
    print(f"  {'total':12s} {total:.2f}")
    gcs.report(steps, "whole run, all threads")
    for t in cons.tasks:
        t.cancel()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["stages", "threaded"])
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--rate", type=float, default=16000, help="threaded: tokens/s")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--frontend", choices=["python", "rust"], default="python")
    p.add_argument("--stub-tokenizer", action="store_true")
    args = p.parse_args()
    import logging
    logging.basicConfig(level=logging.WARNING)
    torch.set_num_threads(1)
    asyncio.run(run_stages(args) if args.mode == "stages" else run_threaded(args))
    sys.stdout.flush()
    # the Rust sockets and the daemon threads blocked in them make interpreter
    # teardown hang
    os._exit(0)


if __name__ == "__main__":
    main()
