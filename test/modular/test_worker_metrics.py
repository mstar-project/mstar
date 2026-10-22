"""The engine workers' metrics: KV pages, steps and batch sizes, prefix-cache
lookups and pending rows, recorded by ``WorkerMetrics`` and carried to
``/metrics`` through the conductor's status message.

Drives the real ``KVManager`` (on the CPU), ``Engine.prepare_inputs`` and
prefix probe, ``MicroScheduler`` (with the fakes of ``test_micro_scheduler``),
the real ``Conductor.run`` (the harness of ``test_conductor_status``) and the
API server's message thread (``test_serving_metrics``).
"""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import msgpack
import pytest

torch = pytest.importorskip("torch")

from test_conductor_status import _conductor, _request, _to_api_server  # noqa: E402
from test_ingest_unpersist import _run  # noqa: E402
from test_kv_prefix_lease import ROOT, _grow, _keys, _layout, _seed, _StubTransfer  # noqa: E402
from test_micro_scheduler import _Engine, _Manager, _next_batch, _scheduler  # noqa: E402
from test_serving_metrics import _labels, _serve, _wait  # noqa: E402

import mstar.communication.wire_types  # noqa: E402,F401  (registers the wire tags)
import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.api_server.request_types import APIServerMessage  # noqa: E402
from mstar.communication import wire  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402
from mstar.observability import catalog  # noqa: E402
from mstar.observability.recorders import EdgeTransfers, MetricSnapshot  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessage,
    ConductorMessageType,
    EnableMetrics,
    WorkerMessage,
    WorkerMessageType,
    WorkerStatus,
)
from mstar.worker import worker_metrics  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402
from mstar.worker.worker_metrics import WorkerMetrics  # noqa: E402

NODE, PREFILL, DECODE = "LLM", "prefill", "decode"


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _kv(max_num_pages=32, prefix=False) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=16,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    if prefix:
        kv.enable_prefix_cache(ROOT)
    return kv


class _Sub:
    """An AR node: each row's prompt length is its fwd_info's ``n``; rows in
    ``decline`` are skipped by prepare_inputs."""

    def __init__(self, decline=()):
        self.decline = set(decline)

    def uniform_row_inputs(self, graph_walk, has_loopback=False):
        return None

    def prepare_inputs(self, **kw):
        info = kw["fwd_info"].info
        return None if info.rid in self.decline else ARNodeInputs(input_seq_len=info.n)

    def can_batch(self, batch, model_inputs):
        return True

    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        return ARNodeInputs(input_seq_len=end - start)


def _hold(kv: KVManager, rid: str, tokens: int) -> None:
    """A request that admitted and committed a ``tokens``-long prompt."""
    kv.ingest_request(rid)
    _grow(kv, rid, tokens)


def _engine(resources=(), sub=None, keyed=False) -> Engine:
    """The real ``Engine`` attributes its prefix probe and
    ``paged_kv_caches`` read; every resource here sits on ``NODE``."""
    eng = Engine.__new__(Engine)
    eng._resources = {r.name: r for r in resources}
    eng._resource_nodes = {r.name: frozenset({NODE}) for r in resources}
    eng._submodules = {NODE: SimpleNamespace(submodule=sub or _Sub(), resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    eng._runner = StepRunner(eng._resources, node_resources={NODE: list(eng._resources)})
    eng._keyed_walks = {NODE: {PREFILL}} if keyed else {}
    eng._prefix_model = "Toy"
    return eng


def _prepare(eng: Engine, walk: str, **lengths) -> ExecutingBatch:
    """One step's real ``prepare_inputs`` over rows ``rid -> prompt length``."""
    batch = ExecutingBatch(
        node_name=NODE,
        per_request_info={rid: SimpleNamespace(rid=rid, n=n) for rid, n in lengths.items()},
        step_context=StepContext(
            request_ids=tuple(lengths), graph_walk=walk, slot=0, capture=False,
        ),
    )
    eng.prepare_inputs(batch)
    return batch


def _metrics(eng=None, scheduler=None, components=(NODE,), sent=None, counting=True) -> WorkerMetrics:
    """``counting``: as after the API server's warmup requests."""
    if scheduler is None:
        scheduler = SimpleNamespace(pending_rows=lambda: {})
    m = WorkerMetrics(
        "worker_0", 3600.0, frozenset(components), eng or _engine(), scheduler,
        send=(sent if sent is not None else []).append, enable_transfers=lambda: EdgeTransfers("SHM"),
    )
    if counting:
        m.start_counting()
    return m


def _by_name(snapshots: list[MetricSnapshot]) -> dict[str, MetricSnapshot]:
    return {s.name: s for s in snapshots}


def _values(snapshots, name) -> dict[tuple, float]:
    snap = _by_name(snapshots)[name]
    return {tuple(row): v for row, v in zip(snap.label_values, snap.values, strict=True)}


def _hist(snapshots, name) -> dict[tuple, tuple[list[int], float]]:
    snap = _by_name(snapshots)[name]
    return {tuple(row): (counts, total) for row, counts, total in zip(
        snap.label_values, snap.bucket_counts, snap.sums, strict=True)}


def _kv_view(m: WorkerMetrics) -> tuple[float, float, float]:
    snaps = m.snapshot()
    key = (NODE, "kv")
    return (_values(snaps, "mstar_kv_pages_used")[key], _values(snaps, "mstar_kv_pages_total")[key],
            _values(snaps, "mstar_kv_pages_cached")[key])


# ── KV pages ─────────────────────────────────────────────────────────────────

def test_kv_pages_follow_admits_and_frees():
    kv = _kv()
    m = _metrics(_engine([kv]))
    assert _kv_view(m) == (0, 31, 0), "31 pages: one of the 32 is the reserved sink"

    _hold(kv, "a", 20)  # two pages of 16
    _hold(kv, "b", 40)  # three
    assert _kv_view(m) == (5, 31, 0)

    kv.remove_request("a")
    assert _kv_view(m) == (3, 31, 0)
    kv.remove_request("b")
    assert _kv_view(m) == (0, 31, 0)


def test_pages_the_prefix_cache_keeps_stay_used_and_show_as_cached():
    kv = _kv(prefix=True)
    m = _metrics(_engine([kv]))
    _seed(kv, list(range(100)))  # 7 pages written, the 6 full ones indexed
    assert _kv_view(m) == (7, 31, 6)

    kv.remove_request("seed")
    assert _kv_view(m) == (6, 31, 6), "the index still holds its 6 pages"


def test_a_cache_only_follower_nodes_use_is_not_reported():
    """A lockstep instance's other ranks hold the same pages; its lead rank reports them."""
    kv = _kv()
    m = _metrics(_engine([kv]), components=("talker",))
    snaps = m.snapshot()
    assert _values(snaps, "mstar_kv_pages_total") == {}


# ── steps and batch sizes ───────────────────────────────────────────────────

class _Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(worker_metrics, "time", c)
    return c


def _bucket(counts: list[int], le: float) -> int:
    """The cumulative count at bucket ``le`` of ``mstar_batch_size``."""
    return counts[catalog.BATCH_SIZE.buckets.index(le)]


def test_a_step_counts_from_its_hand_off_or_the_previous_step_whichever_is_later(clock):
    m = _metrics()
    eng = _engine()
    # idle worker: the step counts from its hand-off to the GPU thread
    clock.now = 1.010
    m.step_done(NODE, DECODE, _prepare(eng, DECODE, r1=1, r2=1).inputs, submitted_at=1.000)
    # pipelined: handed off at 1.005, during the first step; counted from 1.010
    clock.now = 1.014
    m.step_done(NODE, DECODE, _prepare(eng, DECODE, r1=1, r2=1, r3=1).inputs, submitted_at=1.005)

    step = _hist(m.snapshot(), "mstar_step_seconds")[(NODE, DECODE)]
    assert step[0][-1] == 2
    assert step[1] == pytest.approx(0.010 + 0.004)
    batch_counts, batch_sum = _hist(m.snapshot(), "mstar_batch_size")[(NODE, DECODE)]
    assert batch_sum == 5
    assert (_bucket(batch_counts, 2.0), _bucket(batch_counts, 4.0)) == (1, 2)


def test_batch_size_is_the_rows_the_forward_ran(clock):
    m = _metrics()
    eng = _engine(sub=_Sub(decline={"r2"}))
    clock.now = 1.0
    m.step_done(NODE, DECODE, _prepare(eng, DECODE, r1=1, r2=1, r3=1).inputs, submitted_at=0.9)
    counts, total = _hist(m.snapshot(), "mstar_batch_size")[(NODE, DECODE)]
    assert total == 2 and _bucket(counts, 2.0) == 1


def test_a_step_with_no_forward_or_on_a_followed_node_is_not_recorded(clock):
    """It still ends the worker's previous step: the next one counts from it."""
    m = _metrics()
    eng = _engine(sub=_Sub(decline={"r1"}))
    clock.now = 1.0
    m.step_done(NODE, DECODE, _prepare(eng, DECODE, r1=1).inputs, submitted_at=0.5)
    clock.now = 2.0
    m.step_done("talker", DECODE, _prepare(_engine(), DECODE, r2=1).inputs, submitted_at=1.5)
    assert _hist(m.snapshot(), "mstar_step_seconds") == {}

    clock.now = 2.25
    m.step_done(NODE, PREFILL, _prepare(_engine(), PREFILL, r3=8).inputs, submitted_at=1.75)
    assert _hist(m.snapshot(), "mstar_step_seconds")[(NODE, PREFILL)][1] == pytest.approx(0.25)


class _StopLoopError(BaseException):
    """Ends ``Worker.run``, whose loop handles only ``Exception``."""


def _looping_worker(monkeypatch, schedule, speculate, refuse):
    """A worker whose real ``run`` loop drives these steps; everything it
    calls on the worker is stubbed except the loop itself. ``schedule`` is
    what the micro-scheduler hands out per call, ``speculate`` maps a loop
    iteration to the yield-away step it speculates, and ``refuse`` names the
    steps whose admit fails."""
    import mstar.worker.worker as worker_mod
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import Speculation

    monkeypatch.setenv("MSTAR_PRE_PLAN_SPEC", "0")
    monkeypatch.setattr(worker_mod, "gc", SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0))
    eng = _engine()
    sent, postprocessed, iteration = [], [], [0]

    def scheduled(name, rows):
        return ScheduledBatch(node_name=NODE, graph_walk=name,
                              request_to_worker_graph={f"{name}.{i}": 0 for i in range(rows)})

    def executing(batch):
        return ExecutingBatch(
            node_name=NODE,
            per_request_info={rid: SimpleNamespace(rid=rid, n=1) for rid in batch.request_to_worker_graph},
            step_context=StepContext(request_ids=tuple(batch.request_to_worker_graph),
                                     graph_walk=batch.graph_walk, slot=0, capture=False),
        )

    def next_batch(request_state, exclude_target=None):
        if not schedule:
            raise _StopLoopError
        step = schedule.pop(0)
        return None if step is None else scheduled(*step)

    def gpu(batch, node_batch, plan_future):
        eng.prepare_inputs(node_batch)
        if batch.graph_walk in refuse:
            node_batch.register_admit_error("out of pages")
        if node_batch.launch_started_event is not None:
            node_batch.launch_started_event.set()
        return {}

    def build_speculation(pending, consecutive, fair):
        iteration[0] += 1
        step = speculate.get(iteration[0])
        if step is None:
            return None, None
        batch = scheduled(*step)
        return Speculation(
            scheduled_batch=batch, node_batch=executing(batch), consumed_edges=set(),
            continuing_rids=set(), partition="p", is_new_iter=False, is_same_node=False,
            is_yield_away=True,
        ), None

    def stub(*a, **k):
        return None

    w = Worker.__new__(Worker)
    w.worker_id = "worker_0"
    w.enable_nvtx = False
    w._phase_period, w._phase_buf = 0, {}
    w._max_consecutive_spec, w._launch_wait_s = 1024, 0.0
    w.tp_async_sched, w.is_tp_follower = False, False
    w._in_flight_rids = set()
    w.parallel_groups = SimpleNamespace(barrier_all=stub, restore_default_timeout=stub)
    w.engine_manager = SimpleNamespace(warmup_all=stub, get_engine=lambda node: eng)
    w.communicator = SimpleNamespace(send=stub, wait_for_work=stub)
    w.wakeup_event = SimpleNamespace(register_future=stub)
    w.request_state = SimpleNamespace(get_partition_for_node=lambda node: "p")
    w.scheduler = SimpleNamespace(take_admit_errors=dict, get_next_batch=next_batch)
    for name in ("_verify_tp_async_sched_agrees", "_init_engine_thread", "_await_admit_settled",
                 "_apply_pending_removes_safe_to_drop", "_fail_requests", "_apply_pending_drains",
                 "_process_messages", "_apply_removes_whose_step_landed", "_check_ready_tensors",
                 "_poll_stream_buffers", "_set_in_flight_flag", "_clear_in_flight_flag",
                 "_settle_speculation", "_drop_rows_whose_loop_stopped", "_handle_admit_failure"):
        setattr(w, name, stub)
    w.maybe_send_zmq_to_tp_followers = lambda node_batch: -1
    w._build_executing_batch = executing
    w._build_speculation = build_speculation
    w._execute_on_gpu_thread = gpu
    w._postprocess_batch = lambda pending, outputs: postprocessed.append(pending.graph_walk)

    def fail_loudly(error, *args):
        raise error

    w._handle_main_loop_error = fail_loudly
    w._metrics = WorkerMetrics("worker_0", 0.0, frozenset({NODE}), eng,
                               SimpleNamespace(pending_rows=dict), send=sent.append,
                               enable_transfers=lambda: EdgeTransfers("SHM"))
    w._metrics.start_counting()
    return w, sent, postprocessed, iteration


def test_the_run_loop_records_each_step_that_ran_once_on_both_submit_paths(monkeypatch):
    # iteration 1 submits "a" (two rows) from the scheduler; iteration 2
    # speculates the yield-away step "b" (three rows) and postprocesses "a";
    # iteration 3 postprocesses "b" and submits "c", whose admit is refused
    # in iteration 4, so it never postprocesses
    w, sent, postprocessed, iteration = _looping_worker(
        monkeypatch, schedule=[("a", 2), ("c", 1), None], speculate={2: ("b", 3)}, refuse={"c"},
    )
    with pytest.raises(_StopLoopError):
        w.run()

    assert postprocessed == ["a", "b"]
    steps = _hist(w._metrics.snapshot(), "mstar_step_seconds")
    sizes = _hist(w._metrics.snapshot(), "mstar_batch_size")
    assert sorted(steps) == [(NODE, "a"), (NODE, "b")]
    assert {walk: (counts[-1], total) for (_, walk), (counts, total) in sizes.items()} == {
        "a": (1, 2), "b": (1, 3)}
    assert all(0 < total < 5 for _, total in steps.values())
    # an interval of 0 reports on every pass through stage 1
    assert len(sent) == iteration[0]


def test_with_metrics_off_the_run_loop_records_and_sends_nothing(monkeypatch):
    w, sent, postprocessed, _ = _looping_worker(
        monkeypatch, schedule=[("a", 2), None], speculate={}, refuse=set(),
    )
    metrics, w._metrics = w._metrics, None
    with pytest.raises(_StopLoopError):
        w.run()
    assert postprocessed == ["a"]
    assert sent == [] and _hist(metrics.snapshot(), "mstar_step_seconds") == {}


# ── prefix cache ─────────────────────────────────────────────────────────────

def _prefix(m):
    snaps = m.snapshot()
    return (_values(snaps, "mstar_prefix_cache_query_tokens_total"),
            _values(snaps, "mstar_prefix_cache_hit_tokens_total"))


def test_prefix_lookups_count_each_request_once_through_the_engines_probe():
    kv = _kv(prefix=True)
    eng = _engine([kv], keyed=True)
    m = _metrics(eng)
    prompt = list(range(100))
    keys = _seed(kv, prompt)
    kv.ingest_request("hit", KVReqConfig(prefix_keys={"main": keys}, prefix_layout=_layout(100)))
    other = _keys(list(range(1000, 1100)))
    kv.ingest_request("miss", KVReqConfig(prefix_keys={"main": other}, prefix_layout=_layout(100)))
    assert _prefix(m) == ({}, {})

    batch = _prepare(eng, PREFILL, hit=100, miss=100)
    assert [i.input_seq_len for i in batch.inputs] == [4, 100], "the hit skips its 96 cached tokens"
    assert _prefix(m) == ({(NODE,): 200}, {(NODE,): 96})

    # a refused admit sends the step through prepare_inputs again: both
    # requests are probed again, and neither is counted twice
    _prepare(eng, PREFILL, hit=100, miss=100)
    assert _prefix(m) == ({(NODE,): 200}, {(NODE,): 96})


def test_an_unkeyed_request_is_not_a_lookup():
    kv = _kv(prefix=True)
    eng = _engine([kv], keyed=True)
    kv.ingest_request("plain", KVReqConfig())
    _prepare(eng, PREFILL, plain=50)
    assert _prefix(_metrics(eng)) == ({}, {})


# ── warmup requests ──────────────────────────────────────────────────────────

def test_before_start_counting_gauges_are_live_and_steps_are_not_counted(clock):
    """The warmup requests run first; a step whose time starts before
    ``start_counting`` is not counted even when it finishes after it."""
    kv = _kv()
    eng = _engine([kv])
    sent = []
    m = _metrics(eng, sent=sent, counting=False)
    _hold(kv, "warmup-0", 20)
    clock.now = 1.0
    m.step_done(NODE, PREFILL, _prepare(eng, PREFILL, w=20).inputs, submitted_at=0.5)
    m.maybe_report()
    (status,) = sent
    assert [s.name for s in status.metrics] == list(catalog.WORKER_FAMILIES)
    assert _values(status.metrics, "mstar_kv_pages_used") == {(NODE, "kv"): 2}
    assert _hist(status.metrics, "mstar_step_seconds") == {}

    clock.now = 2.0
    m.start_counting()
    clock.now = 2.5
    m.step_done(NODE, DECODE, _prepare(eng, DECODE, w=1).inputs, submitted_at=1.5)
    assert _hist(m.snapshot(), "mstar_step_seconds") == {}
    clock.now = 2.75
    m.step_done(NODE, DECODE, _prepare(eng, DECODE, r1=1, r2=1).inputs, submitted_at=2.6)
    step = _hist(m.snapshot(), "mstar_step_seconds")
    assert list(step) == [(NODE, DECODE)]
    assert step[(NODE, DECODE)][0][-1] == 1 and step[(NODE, DECODE)][1] == pytest.approx(0.15)
    assert _hist(m.snapshot(), "mstar_batch_size")[(NODE, DECODE)][1] == 2
    # a speculated step carries no hand-off time: it counts from the step before
    clock.now = 2.85
    m.step_done(NODE, DECODE, _prepare(eng, DECODE, r1=1, r2=1).inputs, submitted_at=0.0)
    assert _hist(m.snapshot(), "mstar_step_seconds")[(NODE, DECODE)][1] == pytest.approx(0.25)


def test_prefix_lookups_before_start_counting_are_not_counted():
    kv = _kv(prefix=True)
    eng = _engine([kv], keyed=True)
    m = _metrics(eng, counting=False)
    keys = _seed(kv, list(range(100)))
    kv.ingest_request("warmup-0", KVReqConfig(prefix_keys={"main": keys}, prefix_layout=_layout(100)))
    _prepare(eng, PREFILL, **{"warmup-0": 100})
    assert _prefix(m) == ({}, {})

    m.start_counting()
    assert _prefix(m) == ({(NODE,): 0}, {(NODE,): 0})
    other = _keys(list(range(1000, 1100)))
    kv.ingest_request("miss", KVReqConfig(prefix_keys={"main": other}, prefix_layout=_layout(100)))
    kv.ingest_request("hit", KVReqConfig(prefix_keys={"main": keys}, prefix_layout=_layout(100)))
    _prepare(eng, PREFILL, miss=100, hit=100)
    assert _prefix(m) == ({(NODE,): 200}, {(NODE,): 96})


def test_the_worker_starts_counting_on_enable_metrics(clock):
    w = Worker.__new__(Worker)
    w._metrics = _metrics(counting=False)
    clock.now = 1.0
    w._process_message_list([WorkerMessage(WorkerMessageType.ENABLE_METRICS, EnableMetrics())])
    clock.now = 1.25
    w._metrics.step_done(NODE, DECODE, _prepare(_engine(), DECODE, r1=1).inputs, submitted_at=1.0)
    assert _hist(w._metrics.snapshot(), "mstar_step_seconds")[(NODE, DECODE)][1] == pytest.approx(0.25)


def test_the_conductor_forwards_enable_metrics_to_every_worker_before_the_next_request(
        tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, status_interval_s=0.0)
    enable = ConductorMessage(ConductorMessageType.ENABLE_METRICS, EnableMetrics())
    _run(c, [[wire.decode(wire.encode(enable)), _request("r1")]])
    for worker_id in c.worker_ids:
        kinds = [m.message_type for e, m in c.comm.sent if e == worker_id]
        assert kinds.count(WorkerMessageType.ENABLE_METRICS) == 1, worker_id
    first = [m.message_type for e, m in c.comm.sent if e == "worker_0"][:2]
    assert first == [WorkerMessageType.ENABLE_METRICS, WorkerMessageType.NEW_REQUEST]


def test_enable_metrics_crosses_the_wire_both_ways():
    for message in (ConductorMessage(ConductorMessageType.ENABLE_METRICS, EnableMetrics()),
                    WorkerMessage(WorkerMessageType.ENABLE_METRICS, EnableMetrics())):
        frame = wire.encode(message)
        _, payload = msgpack.unpackb(frame, raw=False, strict_map_key=False)
        assert payload["body"] == ["enable_metrics", {}]
        assert wire.decode(frame) == message


def test_the_api_server_starts_the_workers_counting_when_it_enables_metrics(monkeypatch, tmp_path):
    """``main()`` calls ``enable_metrics`` after the warmup requests."""
    pytest.importorskip("prometheus_client")
    with _serve(monkeypatch, tmp_path, metrics=False) as served:
        assert not served.data_worker.worker_metrics
        served.server.enable_metrics()
        assert served.data_worker.worker_metrics


def test_the_data_worker_sends_enable_metrics_before_any_later_request(tmp_path):
    """The API server's warmups are done when it asks; the thread sends the
    message on the socket that carries every submission, ahead of the next."""
    from test_serving_metrics import TEXT, _StubTensorManager

    from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
    from mstar.api_server.request_types import PreprocessInput

    class _Model:
        def process_prompt(self, *args, **kwargs):
            return None

    sent = []
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.worker_metrics_requested = threading.Event()
    facade.wakeup = None
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop,
        # codec / try_send_bytes: the thread's outbox sends through them
        communicator=SimpleNamespace(get_all_new_messages=lambda: [],
                                     codec=SimpleNamespace(encode=lambda msg: msg),
                                     try_send_bytes=lambda entity, msg: sent.append((entity, msg)) or True),
        tensor_manager=_StubTensorManager(), model=_Model(),
        worker_metrics_requested=facade.worker_metrics_requested,
    )

    def submit(key):
        worker.in_queue.put(PreprocessInput(
            request_id=key, client_request_id=key, text="hi", file_paths=None,
            model_kwargs={}, **TEXT))

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        submit("warmup-0")
        _wait(lambda: len(sent) == 1)
        facade.enable_worker_metrics()
        submit("k1")
        _wait(lambda: len(sent) == 3)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert {entity for entity, _ in sent} == {"conductor"}
    assert [(m.message_type, getattr(m.body, "request_id", None)) for _, m in sent] == [
        (ConductorMessageType.NEW_REQUEST, "warmup-0"),
        (ConductorMessageType.ENABLE_METRICS, None),
        (ConductorMessageType.NEW_REQUEST, "k1"),
    ]


# ── pending rows ─────────────────────────────────────────────────────────────

def test_pending_rows_are_the_backlog_and_the_ready_rows_not_in_a_step():
    sched = _scheduler(_Engine(max_bs=2))
    manager = _Manager([f"r{i}" for i in range(7)])
    queue = manager.queues["wg0"]._ready
    del queue["r5"], queue["r6"]
    m = _metrics(scheduler=sched, components=(NODE, "vision_encoder"))

    batch = _next_batch(sched, manager)  # cap 2: r0, r1 run, r2..r4 wait in the backlog
    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert _values(m.snapshot(), "mstar_component_pending_steps") == {
        (NODE,): 3, ("vision_encoder",): 0}

    manager.ingest("r5")  # ready, still on its queue
    queue["r6"] = {"talker"}  # ready for a node this rank does not schedule
    assert _values(m.snapshot(), "mstar_component_pending_steps")[(NODE,)] == 4

    sched.fail_rids({"r2"})
    sched.pending_removes["r5"] = None
    assert _values(m.snapshot(), "mstar_component_pending_steps")[(NODE,)] == 2


# ── reporting ────────────────────────────────────────────────────────────────

def test_one_report_per_interval_with_every_worker_family():
    sent = []
    m = _metrics(sent=sent)
    m.maybe_report()
    m.maybe_report()
    (status,) = sent
    assert status.worker_id == "worker_0"
    assert [s.name for s in status.metrics] == list(catalog.WORKER_FAMILIES)


def test_a_failed_report_is_skipped_and_never_reaches_the_run_loop():
    """The run loop's error handler fails every request in flight."""
    sent, broken = [], [True]

    def pending_rows():
        if broken[0]:
            raise RuntimeError("boom")
        return {}

    m = _metrics(scheduler=SimpleNamespace(pending_rows=pending_rows), sent=sent)
    m.maybe_report()
    assert sent == []
    broken[0] = False
    m._next_report = 0.0  # the next interval
    m.maybe_report()
    assert len(sent) == 1


def test_the_worker_status_crosses_the_wire_typed():
    status = WorkerStatus("worker_0", [
        MetricSnapshot("mstar_kv_pages_used", [[NODE, "kv"]], [3.0]),
        MetricSnapshot("mstar_step_seconds", [[NODE, DECODE]], bucket_counts=[[0, 1, 1]], sums=[0.5]),
    ])
    message = ConductorMessage(ConductorMessageType.WORKER_STATUS, status)
    frame = wire.encode(message)
    tag, payload = msgpack.unpackb(frame, raw=False, strict_map_key=False)
    assert tag == "conductor_msg"
    body_tag, body = payload["body"]
    assert body_tag == "worker_status"
    # the dynamic path wraps a dataclass in an extension; the typed one is a plain map
    assert body["metrics"][1] == {"name": "mstar_step_seconds", "label_values": [[NODE, DECODE]],
                                  "values": [], "bucket_counts": [[0, 1, 1]], "sums": [0.5]}
    assert wire.decode(frame) == message


def test_metrics_are_off_on_a_worker_built_without_an_interval():
    assert Worker.__new__(Worker)._metrics is None


def test_every_worker_process_gets_the_status_interval(tmp_path, monkeypatch):
    """The conductor spawns every rank, a lockstep instance's followers too."""
    spawned = []
    monkeypatch.setattr(conductor_mod.mp, "get_context", lambda method: SimpleNamespace(
        Process=lambda **kw: spawned.append(kw["kwargs"]) or SimpleNamespace(start=lambda: None)))
    monkeypatch.setattr(conductor_mod, "apply_process_env_defaults", lambda: None)
    monkeypatch.setattr(conductor_mod, "keep_bootstrap_sockets_on_loopback", lambda: None)
    monkeypatch.setattr(conductor_mod.atexit, "register", lambda fn: None)
    launch = conductor_mod.Conductor._launch_workers  # the harness stubs it out
    for interval in (0.5, None):
        spawned.clear()
        c = _conductor(tmp_path, monkeypatch, status_interval_s=interval)
        launch(c)
        assert [kw["status_interval_s"] for kw in spawned] == [interval] * len(c.worker_ids)


def test_the_worker_process_hands_the_interval_to_the_worker(monkeypatch):
    import mstar.worker.worker as worker_mod

    built = []

    class _Worker:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def run(self):
            pass

    monkeypatch.setattr(worker_mod, "Worker", _Worker)
    monkeypatch.setattr(conductor_mod, "apply_process_env_defaults", lambda: None)
    monkeypatch.setattr(conductor_mod, "_exit_when_orphaned", lambda worker_id: None)
    monkeypatch.setattr(conductor_mod.signal, "signal", lambda *a: None)
    monkeypatch.setattr(conductor_mod.logging, "basicConfig", lambda **k: None)
    conductor_mod._worker_process_target(
        "worker_0", ["worker_0"], [], {}, {}, {}, {}, None, None, "localhost", "/tmp/x",
        "tcp://localhost:1", device="cpu", status_interval_s=0.5,
    )
    assert [b["status_interval_s"] for b in built] == [0.5]


# ── through the conductor to /metrics ─────────────────────────────────────────

def _worker_status(worker_id, used, steps=(), pending=0):
    """What a worker running ``NODE`` sends, after the wire codec."""
    kv = _kv()
    for i in range(used):
        _hold(kv, f"r{i}", 16)
    clock = _Clock()
    m = WorkerMetrics(
        worker_id, 3600.0, frozenset({NODE}), _engine([kv]),
        SimpleNamespace(pending_rows=lambda: {NODE: pending}), send=None,
        enable_transfers=lambda: EdgeTransfers("SHM"),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_metrics, "time", clock)
        m.start_counting()
    for submitted, done in steps:
        clock.now = done
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(worker_metrics, "time", clock)
            m.step_done(NODE, DECODE, [ARNodeInputs(input_seq_len=1)] * 4, submitted)
    message = ConductorMessage(ConductorMessageType.WORKER_STATUS, WorkerStatus(worker_id, m.snapshot()))
    return wire.decode(wire.encode(message))


def test_the_conductor_sums_its_workers_latest_reports(tmp_path, monkeypatch):
    """Two workers each run a replica of ``NODE``; a newer report from one
    replaces its older one rather than adding to it."""
    c = _conductor(tmp_path, monkeypatch, status_interval_s=0.0)
    _run(c, [
        [_worker_status("worker_0", used=1, steps=[(0.0, 0.010)])],
        [_worker_status("worker_1", used=2, steps=[(0.0, 0.020)], pending=3)],
        [_worker_status("worker_0", used=4, steps=[(0.0, 0.010), (0.0, 0.015)], pending=1)],
    ])
    statuses = _to_api_server(c, "conductor_status")
    assert [s.name for s in statuses[0].metrics] == list(catalog.STATUS_FAMILIES)
    used = [_values(s.metrics, "mstar_kv_pages_used")[(NODE, "kv")] for s in statuses]
    total = [_values(s.metrics, "mstar_kv_pages_total")[(NODE, "kv")] for s in statuses]
    assert (used, total) == ([1, 3, 6], [31, 62, 62])
    assert _values(statuses[2].metrics, "mstar_component_pending_steps") == {(NODE,): 4}
    counts, step_sum = _hist(statuses[2].metrics, "mstar_step_seconds")[(NODE, DECODE)]
    assert counts[-1] == 3 and step_sum == pytest.approx(0.010 + 0.005 + 0.020)


def test_worker_families_reach_metrics(tmp_path, monkeypatch):
    pytest.importorskip("prometheus_client")
    c = _conductor(tmp_path, monkeypatch, status_interval_s=0.0)
    _run(c, [[_worker_status("worker_0", used=2, steps=[(0.0, 0.004)], pending=5)]])
    (status,) = _to_api_server(c, "conductor_status")

    with _serve(monkeypatch, tmp_path) as served:
        served.conductor.inbox.put(wire.decode(wire.encode(APIServerMessage("conductor_status", status))))
        key = ("mstar_kv_pages_used", _labels(component=NODE, resource="kv"))
        _wait(lambda: key in served.scrape())
        samples = served.scrape()

    assert samples[key] == 2
    assert samples[("mstar_kv_pages_total", _labels(component=NODE, resource="kv"))] == 31
    assert samples[("mstar_component_pending_steps", _labels(component=NODE))] == 5
    step = _labels(component=NODE, graph_walk=DECODE)
    assert samples[("mstar_step_seconds_count", step)] == 1
    assert samples[("mstar_step_seconds_bucket", _labels(component=NODE, graph_walk=DECODE, le="0.005"))] == 1
    assert samples[("mstar_step_seconds_bucket", _labels(component=NODE, graph_walk=DECODE, le="0.003"))] == 0
    assert samples[("mstar_batch_size_sum", step)] == 4
