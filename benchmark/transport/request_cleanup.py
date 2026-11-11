"""CPU cost of completion reports and teardown through real handlers and ZMQ.

No GPU, engine resources or tensor payloads: completing requests have no reads
in flight; --other-reads adds pending reads for unrelated requests. The API ACK
is delivered locally to isolate the worker drain cost.
Run the same script with PYTHONPATH pointing to each checkout for an A/B.
This measures teardown, not inference throughput or latency under GPU load.
"""

import argparse
import hashlib
import inspect
import json
import logging
import multiprocessing as mp
import os
import platform
import statistics
import tempfile
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from mstar.communication.communicator import make_communicator
from mstar.communication.tensors import FutureAndPointers, SharedMemoryCommunicationManager
from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import PartitionState
from mstar.distributed.base import ShardingConfig
from mstar.testing_utils.worker import deliver, make_worker
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    ReadsDone,
    WorkerGraphsDone,
    WorkerMessageType,
)
from mstar.worker.input_ingest import InputIngest
from mstar.worker.lockstep import Lockstep
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.rid_table import RidTable
from mstar.worker.step_pipeline import StepPipeline
from mstar.worker.stream_buffers import StreamBufferManager


def _worker_loop(worker_id, prefix, other_reads):
    logging.disable(logging.INFO)
    comm = make_communicator(worker_id, ["conductor"], ipc_socket_path_prefix=prefix)
    rids = RidTable()
    runtime = SimpleNamespace(
        get_rid_handle=rids.handle, get_rid_string=rids.name,
        get_sharding_config=lambda rid: SimpleNamespace(groups=[]),
        remove_request=rids.release,
    )
    tensors = SharedMemoryCommunicationManager(
        my_entity_id=worker_id, hostname="localhost", device="cpu",
        communicator=comm, shm_dir=prefix,
    )
    tensors.rid_to_str = rids.name
    requests = {}
    request_state = SimpleNamespace(per_request_info=requests, remove_request=requests.pop)
    w = make_worker(
        worker_id=worker_id, graph_runtime=runtime, communicator=comm,
        tensor_manager=tensors, request_state=request_state,
        engine_manager=SimpleNamespace(remove_request=lambda rid: None),
        streams=StreamBufferManager(request_state, tensors, []),
        scheduler=SimpleNamespace(clear_rid=lambda rid: None, drop_backlogged=lambda rids: None),
    )
    config = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
    config.setup({})
    if other_reads:
        other = rids.intern("unrelated")
        tensors.register_request(other, config)
        tensors.pending = [
            FutureAndPointers(future=Future(), graph_edges=[], request_id=other)
            for _ in range(other_reads)
        ]
    remaining = 0
    started = 0
    while True:
        for msg in comm.get_all_new_messages(blocking=True, timeout_s=30):
            if isinstance(msg, dict):
                if msg["op"] == "stop":
                    return
                if msg["op"] == "prepare":
                    names = msg["rids"]
                    handles = [rids.intern(name) for name in names]
                    for name, handle in zip(names, handles, strict=True):
                        tensors.register_request(handle, config)
                        requests[handle] = SimpleNamespace(stream_buffers={})
                        w.lifecycle.add(handle, name)
                    remaining = len(names)
                    comm.send("conductor", {"op": "ready"})
                elif msg["op"] == "complete":
                    started = time.process_time()
                    for name in names:
                        comm.send("conductor", ConductorMessage(
                            message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
                            body=WorkerGraphsDone(
                                request_id=name, worker_graph_ids=[0],
                                is_first_tp_rank=worker_id == "worker_0",
                            ),
                        ))
                else:
                    raise ValueError(msg)
            elif msg.message_type == WorkerMessageType.DRAIN_REQUEST:
                deliver(w, msg)
            elif msg.message_type == WorkerMessageType.REMOVE_REQUEST:
                deliver(w, msg)
                remaining -= 1
                if remaining == 0:
                    cpu_us = (time.process_time() - started) * 1e6
                    assert not w.lifecycle.teardowns_pending() and not w.lifecycle.arrivals
                    assert not requests
                    assert len(rids) == bool(other_reads)
                    comm.send("conductor", {"op": "removed", "cpu_us": cpu_us})
        w.pipeline._apply_drains(set())


class _ApiLoopback:
    def __init__(self, comm):
        self.comm = comm
        self.completed = []

    def send(self, entity_id, msg):
        if entity_id == "api_server":
            self.completed.append(msg.body.request_id)
        elif entity_id != Conductor.PREPROCESS_WORKER:
            self.comm.send(entity_id, msg)


def _receive(comm):
    messages = comm.get_all_new_messages(blocking=True, timeout_s=30)
    if not messages:
        raise TimeoutError("teardown benchmark peer did not respond within 30s")
    return messages


def _wave(c, comm, workers, concurrency, sequence):
    names = [f"r{sequence}_{i}" for i in range(concurrency)]
    for worker in workers:
        comm.send(worker, {"op": "prepare", "rids": names})
    ready = 0
    while ready < len(workers):
        for msg in _receive(comm):
            assert msg["op"] == "ready"
            ready += 1
    for name in names:
        c.requests[name] = RequestData(
            persist_signals={}, persist_signal_ref_cnt={},
            worker_graph_to_workers={0: workers}, all_worker_graph_ids={0},
            max_output_tokens=1, random_seed=0, resource_configs={},
            partition_states={"default": PartitionState(
                partition_name="default", graph_walk="decode",
                current_worker_graph_ids={0},
            )},
        )
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for worker in workers:
        comm.send(worker, {"op": "complete"})
    removed, worker_cpu = 0, 0
    while removed < len(workers):
        for msg in _receive(comm):
            if isinstance(msg, dict):
                assert msg["op"] == "removed"
                removed += 1
                worker_cpu += msg["cpu_us"]
            elif msg.message_type == ConductorMessageType.WORKER_GRAPHS_DONE:
                if c._process_worker_graphs_done(msg.body):
                    c._process_request_done(msg.body.request_id)
            elif msg.message_type == ConductorMessageType.READS_DONE:
                c._handle_reads_done(msg.body)
            else:
                raise ValueError(msg)
        for name in c.communicator.completed:
            c._handle_reads_done(ReadsDone(request_id=name, entity_id=c.PREPROCESS_WORKER))
        c.communicator.completed.clear()
    wall_us = (time.perf_counter() - wall_start) * 1e6
    cpu_us = (time.process_time() - cpu_start) * 1e6
    assert not c.requests and not c.draining
    return wall_us / concurrency, cpu_us / concurrency, worker_cpu / concurrency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--waves", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--other-reads", type=int, default=0,
                        help="synthetic pending read records for another request (scan cost only)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.workers, args.waves, *args.concurrency) < 1 or min(args.warmup, args.other_reads) < 0:
        parser.error("workers, waves and concurrency must be positive; warmup must be nonnegative")
    logging.disable(logging.INFO)
    ctx = mp.get_context("spawn")
    workers = [f"worker_{i}" for i in range(args.workers)]
    # A short prefix also fits macOS's Unix-domain socket limit.
    with tempfile.TemporaryDirectory(prefix="m244_", dir="/tmp") as prefix:
        comm = make_communicator("conductor", workers, ipc_socket_path_prefix=prefix)
        processes = [
            ctx.Process(target=_worker_loop, args=(worker, prefix, args.other_reads))
            for worker in workers
        ]
        for process in processes:
            process.start()
        try:
            c = Conductor.__new__(Conductor)
            c.communicator = _ApiLoopback(comm)
            c.requests, c.draining, c._early_reads_done = {}, {}, {}
            c.enable_prof = False
            c.producer_triggered_partitions = set()
            c._client_streams = {}
            c._drain_ttl_s = 120
            c._draining_deadlines = deque()
            c._early_reads_done_deadlines = deque()
            c._try_admit_waiting = lambda: None
            results = {
                "platform": platform.platform(), "cpu": platform.processor(),
                "mesh": type(comm).__name__, "codec": comm.codec.__name__,
                "workers": args.workers, "waves": args.waves,
                "other_reads": args.other_reads,
                "loadavg": os.getloadavg(), "results": [],
            }
            results["sources"] = {
                cls.__name__: {
                    "path": inspect.getfile(cls),
                    "sha256": hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest(),
                } for cls in (
                    Conductor, InputIngest, RequestLifecycle, StepPipeline, Lockstep,
                    SharedMemoryCommunicationManager, WorkerGraphsDone, make_worker,
                )
            }
            sequence = 0
            for concurrency in args.concurrency:
                rows = []
                for wave in range(args.warmup + args.waves):
                    row = _wave(c, comm, workers, concurrency, sequence)
                    sequence += 1
                    if wave >= args.warmup:
                        rows.append(row)
                result = {"concurrency": concurrency}
                for col, name in enumerate(("wall_us", "conductor_cpu_us", "worker_cpu_us")):
                    result[name] = statistics.median(row[col] for row in rows)
                result["total_cpu_us"] = statistics.median(row[1] + row[2] for row in rows)
                results["results"].append(result)
                print(json.dumps(result), flush=True)
            if args.json:
                args.json.write_text(json.dumps(results, indent=2) + "\n")
        finally:
            for worker in workers:
                comm.send(worker, {"op": "stop"})
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            for process in processes:
                if process.exitcode:
                    raise RuntimeError(f"worker benchmark process exited with {process.exitcode}")


if __name__ == "__main__":
    main()
