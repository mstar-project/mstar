"""The API server's output path end to end over HTTP, with a fake engine.

One process runs the real API server pieces -- the FastAPI ``app`` under
uvicorn, ``APIServer``'s message thread, the data worker thread (which also
preprocesses each new request with the real Qwen3.5 chat template) -- built
as ``bench.py`` builds them. A fake engine process stands in for conductor
and workers: it takes each NEW_REQUEST, sends one RESULT_TOKENS frame per
step for every live request over the real Rust ZMQ transport, completes a
request after ``--tokens`` tokens, and answers READS_DONE with REMOVE_REQUEST.
Client processes run closed-loop streaming ``/v1/chat/completions``.

    python -m benchmark.api_output_path.http_e2e --concurrency 128 --step-ms 8

The engine's step period is fixed, so the offered rate is about
``concurrency / step``; the client-side token rate says whether the API
server kept up, and the thread CPU says how close it came to a full core
(all of it shares one GIL).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

PY = sys.executable
MODEL_KEY = "qwen3_5_0.8b"


# -- fake engine ---------------------------------------------------------------


def run_engine(args):
    from mstar.api_server.request_types import APIServerMessage, RequestComplete, ResultTokens
    from mstar.communication import wire
    from mstar.communication.communicator import CommProtocol
    from mstar.communication.rust_communicator import RustZMQCommunicator
    from mstar.graph.loop_indices import NestedLoopIndices
    from mstar.utils.ipc_format import (
        ConductorMessageType,
        RemoveRequest,
        WorkerMessage,
        WorkerMessageType,
    )

    comm = RustZMQCommunicator("conductor", push_ids=["api_server"], protocol=CommProtocol.IPC,
                               ipc_socket_path_prefix=args.sock_dir)
    ids = json.loads(args.token_ids)
    n_ids = len(ids)
    active: dict[str, int] = {}
    order: list[str] = []
    period = args.step_ms / 1e3
    t_next = time.perf_counter()
    end = time.time() + args.seconds
    steps = 0
    rows_total = 0
    while time.time() < end:
        for msg in comm.get_all_new_messages():
            mt = msg.message_type
            if mt == ConductorMessageType.NEW_REQUEST:
                active[msg.body.request_id] = 0
                order.append(msg.body.request_id)
            elif mt == ConductorMessageType.READS_DONE:
                comm.send("api_server_preprocess_worker", WorkerMessage(
                    message_type=WorkerMessageType.REMOVE_REQUEST,
                    body=RemoveRequest(request_id=msg.body.request_id)))
            elif mt == ConductorMessageType.ABORT_REQUEST:
                active.pop(msg.body.request_id, None)
        now = time.perf_counter()
        if now < t_next:
            time.sleep(min(0.0005, t_next - now))
            continue
        t_next += period
        if t_next < now - period:  # never burst to catch up
            t_next = now + period
        keys = [k for k in order if k in active]
        order = keys
        if not keys:
            continue
        steps += 1
        rows_total += len(keys)
        values, loops = [], []
        for i, k in enumerate(keys):
            j = active[k]
            values.append(ids[(j + 37 * i) % n_ids])
            loops.append(NestedLoopIndices(["decode_loop"], {"decode_loop": j}, j + 1))
            active[k] = j + 1
        comm._inner.send("api_server", wire.encode(APIServerMessage(
            message_type="result_tokens",
            body=ResultTokens(request_ids=keys, values=values, loop_indices=loops,
                              signal="text_inputs", modality="text"))))
        t = time.perf_counter()
        for k in keys:
            if active[k] >= args.tokens:
                del active[k]
                comm.send("api_server", APIServerMessage(
                    message_type="request_complete",
                    body=RequestComplete(request_id=k, output_chunk_counts={"text_inputs": args.tokens},
                                         conductor_ingest_time=t, conductor_finish_time=t)))
    print(json.dumps({"steps": steps, "mean_batch": rows_total / max(steps, 1)}), flush=True)
    os._exit(0)


# -- client --------------------------------------------------------------------


def run_client(args):
    import aiohttp

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    body = {"model": MODEL_KEY, "stream": True,
            "messages": [{"role": "user", "content": "Write a long story about a lighthouse."}]}
    t_meas0 = args.t0 + args.warmup
    t_meas1 = args.t0 + args.seconds - 1.0
    tokens_in_window = 0
    itls: list[float] = []
    ttfts: list[float] = []
    done_reqs = 0
    # completed streams that did not carry exactly --tokens content deltas
    # (the token ids are ASCII words, so each token is one delta)
    wrong_count = 0

    async def one(session):
        nonlocal tokens_in_window, done_reqs, wrong_count
        while time.time() < t_meas1:
            start = time.time()
            last = None
            n = 0
            async with session.post(url, json=body) as resp:
                async for line in resp.content:
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    if b'"content"' not in line:
                        continue
                    n += 1
                    now = time.time()
                    if last is None:
                        if t_meas0 <= now <= t_meas1:
                            ttfts.append(now - start)
                    elif t_meas0 <= now <= t_meas1:
                        itls.append(now - last)
                    if t_meas0 <= now <= t_meas1:
                        tokens_in_window += 1
                    last = now
            done_reqs += 1
            wrong_count += n != args.tokens

    async def main():
        conn = aiohttp.TCPConnector(limit=0)
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
            await asyncio.gather(*(one(s) for _ in range(args.streams)))

    asyncio.run(main())
    cpu = time.process_time()
    print(json.dumps({
        "tokens": tokens_in_window, "window": t_meas1 - t_meas0, "itls": itls,
        "ttfts": ttfts, "requests": done_reqs, "cpu": cpu, "wrong_count": wrong_count,
    }), flush=True)
    os._exit(0)


# -- server + orchestration ----------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoSendTensorManager:
    """The data worker's tensor manager without SHM: the prompt's tensors are
    not stored (nothing reads them), every output rides inline."""

    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass

    def set_persist(self, uuid, persist):
        pass

    def get_ready_tensors(self):
        return {}

    def has_inflight_reads(self, request_id):
        return False

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        pass


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def run_server(args):
    import torch
    import uvicorn

    from benchmark.api_output_path import bench
    from mstar.api_server import entrypoint as ep
    from mstar.communication.communicator import CommProtocol
    from mstar.communication.event import EventWakeup
    from mstar.communication.rust_communicator import RustZMQCommunicator
    from mstar.utils.gc_freeze import freeze_after_setup

    torch.set_num_threads(1)
    model, token_ids = bench.load_model(stub=False)
    sock_dir = tempfile.mkdtemp(prefix="aoh", dir="/tmp")
    wakeup = EventWakeup()
    dw_comm = RustZMQCommunicator("api_server_preprocess_worker", push_ids=["conductor"],
                                  protocol=CommProtocol.IPC, ipc_socket_path_prefix=sock_dir)
    server, facade, thread = bench.build_server(model, data_worker_comm=dw_comm, wakeup=wakeup)
    thread.tensor_manager = _NoSendTensorManager()
    server.model_name = MODEL_KEY
    server.communicator = RustZMQCommunicator("api_server", push_ids=["conductor"],
                                              protocol=CommProtocol.IPC,
                                              ipc_socket_path_prefix=sock_dir)
    ep.api_server = server
    port = args.port or _free_port()

    tids: dict[str, int] = {}

    def msg_thread():
        tids["message"] = threading.get_native_id()
        server._process_messages()

    def data_thread():
        tids["data_worker"] = threading.get_native_id()
        thread.run()

    for f in (msg_thread, data_thread):
        threading.Thread(target=f, daemon=True).start()

    config = uvicorn.Config(ep.app, host="127.0.0.1", port=port, access_log=False,
                            log_level="warning", loop=args.loop, http=args.http)
    userver = uvicorn.Server(config)

    async def orchestrate():
        tids["event_loop"] = threading.get_native_id()
        while not userver.started:
            await asyncio.sleep(0.05)
        freeze_after_setup("harness")
        gcs = bench.GCStats()
        env = dict(os.environ)
        t0 = time.time() + 1.0
        engine = subprocess.Popen(
            [PY, "-m", "benchmark.api_output_path.http_e2e", "engine",
             "--sock-dir", sock_dir, "--step-ms", str(args.step_ms), "--tokens", str(args.tokens),
             "--seconds", str(args.seconds + 2), "--token-ids", json.dumps(token_ids)],
            stdout=subprocess.PIPE, env=env)
        per = [args.concurrency // args.clients + (1 if i < args.concurrency % args.clients else 0)
               for i in range(args.clients)]
        clients = [subprocess.Popen(
            [PY, "-m", "benchmark.api_output_path.http_e2e", "client", "--port", str(port),
             "--streams", str(n), "--seconds", str(args.seconds), "--warmup", str(args.warmup),
             "--t0", str(t0), "--tokens", str(args.tokens)],
            stdout=subprocess.PIPE, env=env) for n in per if n > 0]
        await asyncio.sleep(max(0.0, t0 + args.warmup - time.time()))
        cpu0 = {k: bench._thread_cpu(v) for k, v in tids.items()}
        p0 = time.process_time()
        w0 = time.perf_counter()
        await asyncio.sleep(args.seconds - args.warmup - 1.0)
        wall = time.perf_counter() - w0
        cpu1 = {k: bench._thread_cpu(v) for k, v in tids.items()}
        p1 = time.process_time()
        outs = await asyncio.gather(*(asyncio.to_thread(c.communicate) for c in clients))
        eng = json.loads((await asyncio.to_thread(engine.communicate))[0])
        res = [json.loads(o[0]) for o in outs]
        tokens = sum(r["tokens"] for r in res)
        window = res[0]["window"]
        itls = [x for r in res for x in r["itls"]]
        ttfts = [x for r in res for x in r["ttfts"]]
        offered = eng["mean_batch"] * 1e3 / args.step_ms
        print(f"c={args.concurrency} step={args.step_ms} ms tokens/req={args.tokens} "
              f"frontend={args.frontend} loop={args.loop} http={args.http} clients={len(clients)}")
        print(f"engine: mean batch {eng['mean_batch']:.1f} -> offered {offered:.0f} tok/s")
        print(f"client: {tokens / window:.0f} tok/s delivered, "
              f"{sum(r['requests'] for r in res)} requests done, "
              f"{sum(r['wrong_count'] for r in res)} without exactly {args.tokens} tokens")
        print(f"ITL ms p50 {_pct(itls, 50) * 1e3:.2f} p90 {_pct(itls, 90) * 1e3:.2f} "
              f"p99 {_pct(itls, 99) * 1e3:.2f}; TTFT ms p50 {_pct(ttfts, 50) * 1e3:.1f} "
              f"p99 {_pct(ttfts, 99) * 1e3:.1f}")
        print("API server thread CPU (fraction of one core):")
        for k, after in cpu1.items():
            print(f"  {k:12s} {(after - cpu0[k]) / wall:.2f}")
        print(f"  {'process':12s} {(p1 - p0) / wall:.2f}")
        print(f"client CPU (s, per process over the run): {[round(r['cpu'], 1) for r in res]}")
        gcs.report(eng["steps"], "API server process, per engine step")
        sys.stdout.flush()
        if args.frontend == "rust":
            proc.terminate()
        os._exit(0)

    async def serve():
        asyncio.get_running_loop().create_task(orchestrate())
        await userver.serve()

    if args.frontend == "rust":
        from mstar.api_server.rust_frontend import (
            RustFrontendBridge,
            find_server_binary,
            launch_rust_server,
        )
        bridge_dir = tempfile.mkdtemp(prefix="aob", dir="/tmp")
        upload_dir = tempfile.mkdtemp(prefix="aou", dir="/tmp")
        proc = launch_rust_server(find_server_binary(None), MODEL_KEY, port, bridge_dir, upload_dir)
        bridge = RustFrontendBridge(server, bridge_dir)

        class _Started:
            started = False

        userver = _Started()

        async def serve_rust():
            loop = asyncio.get_running_loop()
            loop.create_task(orchestrate())
            # mstar-server answers /health only once the bridge does
            async def wait_up():
                import aiohttp
                async with aiohttp.ClientSession() as s:
                    while True:
                        try:
                            async with s.get(f"http://127.0.0.1:{port}/health") as r:
                                if r.status == 200:
                                    userver.started = True
                                    return
                        except Exception:  # noqa: BLE001
                            pass
                        await asyncio.sleep(0.1)
            loop.create_task(wait_up())
            try:
                await bridge._serve()
            finally:
                proc.terminate()

        try:
            asyncio.run(serve_rust()) if args.loop == "asyncio" else _run_uvloop(serve_rust)
        finally:
            proc.terminate()
        return

    asyncio.run(serve()) if args.loop == "asyncio" else _run_uvloop(serve)


def _run_uvloop(coro_fn):
    import uvloop
    uvloop.run(coro_fn())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("role", nargs="?", default="server", choices=["server", "engine", "client"])
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--clients", type=int, default=8, help="client processes")
    p.add_argument("--step-ms", type=float, default=8.0)
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--warmup", type=float, default=6.0)
    p.add_argument("--loop", default="asyncio")
    p.add_argument("--http", default="h11")
    p.add_argument("--frontend", choices=["python", "rust"], default="python",
                   help="rust: mstar-server (MSTAR_SERVER_BIN or rust/server/target) + the bridge")
    p.add_argument("--port", type=int, default=0)
    # engine / client
    p.add_argument("--sock-dir")
    p.add_argument("--token-ids")
    p.add_argument("--streams", type=int)
    p.add_argument("--t0", type=float)
    args = p.parse_args()
    import logging
    logging.basicConfig(level=logging.ERROR)
    {"server": run_server, "engine": run_engine, "client": run_client}[args.role](args)


if __name__ == "__main__":
    main()
