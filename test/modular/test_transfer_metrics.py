"""Per-edge tensor transfer metrics: ``mstar_transfer_bytes_total``,
``mstar_transfers_total`` and ``mstar_transfer_seconds``.

Real reads between real tensor managers (SHM files, and Mooncake's TCP path
with an engine that copies the bytes in-process), counted by the consumer;
a worker that leads no node still reporting them on the worker status (the
path itself is ``test_worker_metrics``); the data worker's loop publishing
what it reads. The conductor and ``/metrics`` are in ``test_serving_metrics``.
The Rust graph runtime, bookkeeper and SHM arena are in
``test/rust/test_transfer_metrics_rust.py``.
"""
import contextlib
import ctypes
import queue
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.api_server import data_worker as data_worker_mod  # noqa: E402
from mstar.api_server.data_worker import PreprocessWorkerThread, ResultQueue  # noqa: E402
from mstar.api_server.request_types import ResultTensors  # noqa: E402
from mstar.communication.communicator import CommProtocol  # noqa: E402
from mstar.communication.event import EventWakeup  # noqa: E402
from mstar.communication.tensors import (  # noqa: E402
    AsyncMooncakeReader,
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
    TensorCommunicationManager,
)
from mstar.distributed.base import ShardingConfig  # noqa: E402
from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.graph.loop_indices import NestedLoopIndices  # noqa: E402
from mstar.observability import catalog  # noqa: E402
from mstar.observability.recorders import EdgeTransfers  # noqa: E402
from mstar.streaming.topology import PartitionTopology  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessageType,
    EnableMetrics,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.worker import Worker  # noqa: E402

DATA_WORKER = "api_server_preprocess_worker"
TRANSFER_FAMILIES = {f.name for f in (catalog.TRANSFER_BYTES, catalog.TRANSFERS, catalog.TRANSFER_SECONDS)}


class _Sink:
    """The control mesh: every message (the consumer's acks) is kept."""

    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    # the data worker's outbox encodes with this: identity, so sends arrive as objects
    codec = SimpleNamespace(encode=lambda msg: msg)

    def try_send_bytes(self, entity, msg):
        self.send(entity, msg)
        return True

    def get_all_new_messages(self):
        return []


def _shm(entity, tmp_path):
    return SharedMemoryCommunicationManager(
        my_entity_id=entity, hostname="localhost", device="cpu",
        communicator=_Sink(), shm_dir=str(tmp_path),
    )


def _series(snapshots):
    """{(src, dst, transport): (bytes, tensors, reads, seconds)} of one
    manager's (or one status message's) transfer families."""
    by_name = {s.name: s for s in snapshots}
    out = {}
    nbytes = by_name[catalog.TRANSFER_BYTES.name]
    tensors = by_name[catalog.TRANSFERS.name]
    seconds = by_name[catalog.TRANSFER_SECONDS.name]
    for row, b in zip(nbytes.label_values, nbytes.values, strict=True):
        out[tuple(row)] = [b]
    for row, n in zip(tensors.label_values, tensors.values, strict=True):
        out[tuple(row)].append(n)
    for row, counts, total in zip(seconds.label_values, seconds.bucket_counts, seconds.sums, strict=True):
        out[tuple(row)] += [counts[-1], total]
    return {k: tuple(v) for k, v in out.items()}


def _send(producer, rid, tensors, node="enc"):
    """The producer stores one node's outputs and makes them readable."""
    infos = producer.store_and_return_tensor_info(rid, tensors, node_name=node, graph_walk="prefill")
    producer.register_for_send(rid, [i for il in infos.values() for i in il])
    return infos


def _read(consumer, rid, edges, timeout=5.0):
    """Start the reads and poll until every edge is delivered."""
    consumer.start_read_tensors(rid, edges)
    deadline = time.monotonic() + timeout
    got = []
    while len(got) < len(edges):
        got += consumer.get_ready_tensors().get(rid, [])
        assert time.monotonic() < deadline, "the reads never completed"
        time.sleep(0.001)


# ── what one read counts ─────────────────────────────────────────────────────

def test_a_read_edge_counts_its_bytes_and_tensors_and_one_timed_read(tmp_path):
    producer, consumer = _shm("worker_0", tmp_path), _shm("worker_1", tmp_path)
    stats = consumer.enable_transfer_metrics()
    a, b = torch.randn(4, 8), torch.arange(5)
    infos = _send(producer, "r1", {"h": [a, b]})

    consumer.start_read_tensors("r1", [GraphEdge(next_node="llm", name="h", tensor_info=infos["h"])])
    (rx_time,) = [ep.rx_time for ep in consumer.pending]
    consumer.get_ready_tensors()

    assert torch.equal(consumer.get_tensor(infos["h"][0].uuid), a)
    assert _series(stats.snapshot()) == {
        ("enc", "llm", "SHM"): (a.nbytes + b.nbytes, 2, 1, rx_time),
    }
    assert rx_time > 0

    # a second request's read of the same edge adds to the same series
    infos = _send(producer, "r2", {"h": [b]})
    _read(consumer, "r2", [GraphEdge(next_node="llm", name="h", tensor_info=infos["h"])])
    (row,) = _series(stats.snapshot()).values()
    assert row[:3] == (a.nbytes + 2 * b.nbytes, 3, 2)


def test_edges_into_and_out_of_the_graph_are_named_api_server(tmp_path):
    data_worker, worker = _shm(DATA_WORKER, tmp_path), _shm("worker_0", tmp_path)
    worker_stats = worker.enable_transfer_metrics()
    data_worker_stats = data_worker.enable_transfer_metrics()
    prompt = torch.arange(6, dtype=torch.uint8)

    # the request's inputs: minted by the data worker outside any node
    infos = data_worker.store_and_return_tensor_info("r1", {"text_inputs": [prompt]})
    data_worker.register_for_send("r1", infos["text_inputs"])
    _read(worker, "r1", [GraphEdge(next_node="enc", name="text_inputs", tensor_info=infos["text_inputs"])])
    # a result: the data worker reads it, whatever node the edge names
    out = _send(worker, "r1", {"new_token": [torch.tensor([7])]}, node="llm")
    _read(data_worker, "r1", [GraphEdge(next_node="emit_to_client", name="text_output",
                                        tensor_info=out["new_token"])])

    assert list(_series(worker_stats.snapshot())) == [("api_server", "enc", "SHM")]
    assert list(_series(data_worker_stats.snapshot())) == [("llm", "api_server", "SHM")]


def test_only_tensors_read_over_the_transport_count(tmp_path):
    """A tensor the consumer made itself, or one already in its store (sent to
    two of its nodes), moves no bytes."""
    producer, consumer = _shm("worker_0", tmp_path), _shm("worker_1", tmp_path)
    stats = consumer.enable_transfer_metrics()
    shared = _send(producer, "r1", {"h": [torch.randn(16)]})["h"]
    own = consumer.store_and_return_tensor_info("r1", {"loop": [torch.randn(3)]}, node_name="llm")["loop"]
    consumer.register_request("r1", ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}))

    _read(consumer, "r1", [
        GraphEdge(next_node="llm", name="h", tensor_info=list(shared)),
        GraphEdge(next_node="vocoder", name="h", tensor_info=list(shared)),
        GraphEdge(next_node="llm", name="loop", tensor_info=own),
    ])

    assert _series(stats.snapshot()).keys() == {("enc", "llm", "SHM")}
    assert _series(stats.snapshot())[("enc", "llm", "SHM")][:3] == (64, 1, 1)


def test_with_metrics_off_a_read_records_and_stores_nothing_more(tmp_path):
    producer, consumer = _shm("worker_0", tmp_path), _shm("worker_1", tmp_path)
    infos = _send(producer, "r1", {"h": [torch.randn(4)]})
    consumer.start_read_tensors("r1", [GraphEdge(next_node="llm", name="h", tensor_info=infos["h"])])
    (ep,) = consumer.pending
    consumer.get_ready_tensors()

    assert consumer.transfer_stats is None
    assert not {"read_src", "read_bytes", "read_tensors", "read_more"} & vars(ep).keys()


def test_an_edge_joined_from_two_producers_counts_each_producer(tmp_path):
    """The conductor joins one input name produced by several nodes (or
    walks) into one edge; each producer's tensors count under its own src,
    and the edge's one read time is split between them by bytes."""
    producer, consumer = _shm("worker_0", tmp_path), _shm("worker_1", tmp_path)
    stats = consumer.enable_transfer_metrics()
    a = _send(producer, "r1", {"h": [torch.randn(4)]}, node="enc_a")["h"]
    b = _send(producer, "r1", {"h": [torch.randn(8), torch.randn(4)]}, node="enc_b")["h"]
    consumer.start_read_tensors("r1", [GraphEdge(next_node="llm", name="h", tensor_info=a + b)])
    (rx_time,) = [ep.rx_time for ep in consumer.pending]
    consumer.get_ready_tensors()

    got = _series(stats.snapshot())
    assert {k: v[:3] for k, v in got.items()} == {
        ("enc_a", "llm", "SHM"): (16, 1, 1),
        ("enc_b", "llm", "SHM"): (48, 2, 1),
    }
    assert got[("enc_a", "llm", "SHM")][3] == pytest.approx(rx_time / 4)
    assert got[("enc_b", "llm", "SHM")][3] == pytest.approx(rx_time * 3 / 4)


class _MemmoveEngine:
    """Mooncake's engine with the network replaced by a local copy: the read
    really moves the bytes, and takes at least ``delay`` seconds."""

    delay = 0.003

    def batch_transfer_sync_read(self, session_id, local_ptrs, remote_ptrs, nbytes):
        time.sleep(self.delay)
        for dst, src, n in zip(local_ptrs, remote_ptrs, nbytes, strict=True):
            ctypes.memmove(dst, src, n)
        return 0


class _NoopRegistry:
    def register_memory(self, ptr, nbytes):
        return 0

    def unregister_memory(self, ptr):
        return 0


def _mooncake(entity):
    mgr = object.__new__(MooncakeCommunicationManager)
    TensorCommunicationManager.__init__(
        mgr, my_entity_id=entity, my_session_id=f"{entity}:1", device="cpu",
        communicator=_Sink(), transfer_engine=_NoopRegistry(),
    )
    mgr.protocol = CommProtocol.TCP
    mgr.needs_flat_layout = True
    mgr._async_reader = AsyncMooncakeReader(_MemmoveEngine(), device="cpu")
    return mgr


def test_a_tcp_read_is_timed_by_the_reader_and_labelled_tcp():
    """Mooncake reads finish on the reader's thread, so its time comes from
    there; it was only measured under --log-stats before."""
    producer, consumer = _mooncake("worker_0"), _mooncake("worker_1")
    try:
        stats = consumer.enable_transfer_metrics()
        a = torch.randn(3, 5)
        infos = _send(producer, "r1", {"h": [a, torch.empty(0)]})
        _read(consumer, "r1", [GraphEdge(next_node="llm", name="h", tensor_info=infos["h"])])

        assert torch.equal(consumer.get_tensor(infos["h"][0].uuid), a)
        ((labels, (nbytes, tensors, reads, seconds)),) = _series(stats.snapshot()).items()
        assert labels == ("enc", "llm", "TCP")
        # the empty tensor arrives without a read but still crossed over
        assert (nbytes, tensors, reads) == (a.nbytes, 2, 1)
        assert seconds >= _MemmoveEngine.delay
    finally:
        producer._async_reader.shutdown()
        consumer._async_reader.shutdown()


class _FailSecondRead(_MemmoveEngine):
    delay = 0.0

    def __init__(self):
        self.calls = 0

    def batch_transfer_sync_read(self, *args):
        self.calls += 1
        return -1 if self.calls == 2 else super().batch_transfer_sync_read(*args)


def test_a_failed_read_does_not_count_the_reads_before_it_again():
    """A read whose future raises is dropped and handed to take_failed_reads;
    the read delivered before it counts once, the failed one not at all."""
    producer, consumer = _mooncake("worker_0"), _mooncake("worker_1")
    consumer._async_reader = AsyncMooncakeReader(_FailSecondRead(), device="cpu")
    try:
        stats = consumer.enable_transfer_metrics()
        ok = _send(producer, "r1", {"h": [torch.randn(4)]})["h"]
        bad = _send(producer, "r1", {"g": [torch.randn(4)]})["g"]
        consumer.start_read_tensors("r1", [
            GraphEdge(next_node="llm", name="h", tensor_info=ok),
            GraphEdge(next_node="llm", name="g", tensor_info=bad),
        ])
        for future in [ep.future for ep in consumer.pending]:
            with contextlib.suppress(RuntimeError):
                future.result(timeout=5)
        assert [e.name for e in consumer.get_ready_tensors()["r1"]] == ["h"]
        for _ in range(2):
            assert consumer.get_ready_tensors() == {}
        (failed,) = consumer.take_failed_reads()
        assert failed.request_id == "r1" and [e.name for e in failed.graph_edges] == ["g"]
        assert _series(stats.snapshot())[("enc", "llm", "TCP")][:3] == (16, 1, 1)
    finally:
        producer._async_reader.shutdown()
        consumer._async_reader._executor.shutdown(wait=True)


# ── the recorder ─────────────────────────────────────────────────────────────

def test_seconds_land_in_inclusive_buckets_and_the_counts_are_cumulative():
    stats = EdgeTransfers("SHM")
    bounds = catalog.TRANSFER_SECONDS.buckets
    for seconds in (bounds[0], bounds[0] * 1.5, 100.0):
        stats.record("enc", "llm", 10, 1, seconds)
    hist = stats.snapshot()[2]
    (counts,) = hist.bucket_counts
    assert counts[0] == 1 and counts[1] == 2  # le is inclusive
    assert counts[-2] == 2 and counts[-1] == 3  # +Inf holds the 100 s read
    assert hist.sums == [bounds[0] * 2.5 + 100.0]


# ── worker -> conductor ──────────────────────────────────────────────────────

def _bare_worker(tmp_path):
    """The attributes ``Worker._start_metrics`` reads, for a rank leading no
    node, with a real SHM manager."""
    worker = Worker.__new__(Worker)
    worker.worker_id = "worker_1"
    worker.parallel_leader_nodes = set()
    worker.engine_manager = SimpleNamespace(engine=SimpleNamespace(paged_kv_caches=list))
    worker.communicator = _Sink()
    worker.tensor_manager = _shm("worker_1", tmp_path)
    return worker


def test_a_rank_leading_no_node_reports_its_transfers_once_per_interval(tmp_path):
    """A tensor parallel follower reads its own slice, so it reports its
    transfers on the worker status; its lead rank reports the engine
    families. A read before ENABLE_METRICS (a warmup request's) is not
    counted."""
    worker = _bare_worker(tmp_path)

    def not_on_a_follower():
        raise AssertionError("a rank leading nothing has no pending rows to report")

    worker.scheduler = SimpleNamespace(pending_rows=not_on_a_follower)
    metrics = worker._metrics = worker._start_metrics(3600.0)
    producer = _shm("worker_0", tmp_path)
    warmup = _send(producer, "warmup-0", {"h": [torch.randn(8, 8)]})
    _read(worker.tensor_manager, "warmup-0", [GraphEdge(next_node="llm", name="h", tensor_info=warmup["h"])])
    worker._process_message_list([WorkerMessage(WorkerMessageType.ENABLE_METRICS, EnableMetrics())])
    infos = _send(producer, "r1", {"h": [torch.randn(4, 4)]})
    _read(worker.tensor_manager, "r1", [GraphEdge(next_node="llm", name="h", tensor_info=infos["h"])])

    metrics.maybe_report()
    metrics.maybe_report()

    ((to, message),) = worker.communicator.sent
    assert to == "conductor"
    assert message.message_type == ConductorMessageType.WORKER_STATUS
    assert message.body.worker_id == "worker_1"
    assert [s.name for s in message.body.metrics] == list(catalog.WORKER_FAMILIES)
    engine = [s for s in message.body.metrics if s.name not in TRANSFER_FAMILIES]
    assert all(s.label_values == [] for s in engine)
    ((labels, row),) = _series(message.body.metrics).items()
    assert labels == ("enc", "llm", "SHM") and row[:3] == (64, 1, 1)


# ── the data worker ──────────────────────────────────────────────────────────

class _TextModel:
    def postprocess(self, output, modality, request_kwargs=None):
        return f"<{int(output[0])}>".encode()

    def get_partition_topology(self):
        return PartitionTopology(partitions=["default"])


def test_the_data_worker_loop_publishes_the_results_it_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(data_worker_mod, "STATUS_INTERVAL_S", 0.01)
    worker, manager = _shm("worker_1", tmp_path), _shm(DATA_WORKER, tmp_path)
    transfers = manager.enable_transfer_metrics()
    stop = threading.Event()
    thread = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=ResultQueue(EventWakeup()),
        profile_queue=ResultQueue(EventWakeup()), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop, communicator=_Sink(),
        tensor_manager=manager, model=_TextModel(),
    )
    token = torch.tensor([7])
    infos = _send(worker, "r1", {"new_token": [token]}, node="llm")
    thread.result_tensor_queue.put(ResultTensors(
        request_id="r1", modality="text",
        graph_edge=GraphEdge(next_node="emit_to_client", name="new_token", tensor_info=infos["new_token"]),
        loop_indices=NestedLoopIndices(loop_name_order=[], loop_indices={}, wg_fwd_pass_idx=0),
    ))
    runner = threading.Thread(target=thread.run)
    runner.start()
    try:
        deadline = time.monotonic() + 5
        while not _series(transfers.latest):
            assert time.monotonic() < deadline, "the data worker never published its read"
            time.sleep(0.005)
    finally:
        stop.set()
        runner.join(timeout=5)
    assert thread.out_queue.get(timeout=1).data == b"<7>"
    ((labels, row),) = _series(transfers.latest).items()
    assert labels == ("llm", "api_server", "SHM") and row[:3] == (token.nbytes, 1, 1)
