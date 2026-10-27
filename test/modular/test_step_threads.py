"""The CPU a worker's main, GPU and plan threads share: how each worker picks
it (``choose_step_core``), the knob, the GPU's local CPU list from sysfs, and
the real ``Worker.run`` moving its three step threads onto it. The knob
itself is in ``test_knobs``."""
from __future__ import annotations

import logging
import os
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.engine.engine import ExecutingBatch  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.platforms import (  # noqa: E402
    CpuPlatform,
    CudaPlatform,
    RocmPlatform,
    interface,  # noqa: E402
)
from mstar.utils.cpu_list import format_cpu_list, parse_cpu_list  # noqa: E402
from mstar.worker import step_threads  # noqa: E402
from mstar.worker.step_threads import StepCore, choose_step_core  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

NODE0, NODE1 = parse_cpu_list("0-31,128-159"), parse_cpu_list("32-63,160-191")
HOST = frozenset(range(256))


def smt(cpu: int) -> tuple[int, int]:
    """Sibling order of the host above: CPU n and n + 128 share a core."""
    return cpu // 128, cpu


def _cpus(setting, local_cpus, allowed, order=smt):
    return [choose_step_core(setting, slot, local_cpus, allowed, order).cpu
            for slot in range(len(local_cpus))]


# ── choice ──────────────────────────────────────────────────────────────────

def test_auto_takes_a_cpu_local_to_the_gpu_one_per_worker():
    assert _cpus("auto", [NODE0], HOST) == [0]
    # two ranks on one GPU, or on two GPUs of one NUMA node
    assert _cpus("auto", [NODE0, NODE0], HOST) == [0, 1]
    assert _cpus("auto", [NODE0, NODE1, NODE0], HOST) == [0, 32, 1]
    core = choose_step_core("auto", 0, [NODE0], HOST, smt)
    assert core.source == "local to its device: 0-31,128-159"


def test_smt_siblings_come_after_every_first_thread():
    two_cores = parse_cpu_list("0-1,128-129")
    assert _cpus("auto", [two_cores] * 3, HOST) == [0, 1, 128]
    # without topology the order is numeric
    assert _cpus("auto", [parse_cpu_list("0,128")] * 2, HOST, order=None) == [0, 128]


def test_a_taskset_off_the_gpu_node_falls_back_to_the_process_mask():
    allowed = parse_cpu_list("32-39")
    assert _cpus("auto", [NODE0], allowed) == [32]
    assert "process affinity" in choose_step_core("auto", 0, [NODE0], allowed).source
    # no local list at all (no sysfs entry, a CPU device, macOS)
    assert _cpus("auto", [None, None], allowed) == [32, 33]


def test_workers_never_share_when_only_some_lists_meet_the_mask():
    # worker 0's GPU is on node 1, worker 1's on node 0; only CPU 8 of node 0
    # is allowed, so worker 1 must not fall back onto worker 0's CPU 32
    allowed = parse_cpu_list("8,32-39")
    assert _cpus("auto", [NODE1, NODE0], allowed) == [32, 8]
    assert _cpus("auto", [NODE1, NODE0, NODE0], allowed) == [32, 8, 33]
    picks = _cpus("auto", [NODE0, None, NODE1, NODE0], parse_cpu_list("0-1,32-33,40"))
    assert len(set(picks)) == 4


def test_too_few_allowed_cpus_skips_rather_than_shares():
    # every worker needs its own CPU and one is left for everything else
    assert _cpus("auto", [NODE0, NODE0], parse_cpu_list("0-1")) == [None, None]
    assert "skipped" in choose_step_core("auto", 0, [NODE0, NODE0], parse_cpu_list("0-1")).source
    assert _cpus("auto", [NODE0, NODE0], parse_cpu_list("0-2")) == [0, 1]
    assert _cpus("auto", [None], {5}) == [None]


def test_off_and_explicit_values():
    assert choose_step_core("off", 0, [NODE0], HOST) == StepCore(None, "off")
    allowed = parse_cpu_list("32-39")
    # the first worker on that CPU, the others on the allowed CPUs upward of it
    assert _cpus(34, [NODE0, NODE0], allowed) == [34, 35]
    assert _cpus(38, [None, None, None], allowed) == [38, 39, None]
    # a CPU outside the mask is refused, not forced
    assert _cpus(8, [NODE0], allowed) == [None]
    assert _cpus(0, [None, None], parse_cpu_list("0-1")) == [None, None]


# ── local CPU list ─────────────────────────────────────────────────────────

def test_cpu_lists_round_trip():
    assert parse_cpu_list("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    assert parse_cpu_list("") == frozenset()
    assert format_cpu_list({11, 0, 1, 2, 3, 8, 10}) == "0-3,8,10-11"


def test_local_cpus_follow_the_pci_address_on_cuda_and_rocm(tmp_path, monkeypatch):
    real_open = open
    sysfs = tmp_path / "0000:4a:00.0"
    sysfs.mkdir()
    (sysfs / "local_cpulist").write_text("0-31,128-159\n")

    def fake_open(path, *args, **kwargs):
        prefix = "/sys/bus/pci/devices/"
        if str(path).startswith(prefix):
            path = tmp_path / str(path)[len(prefix):]
        return real_open(path, *args, **kwargs)

    props = {0: SimpleNamespace(pci_domain_id=0, pci_bus_id=0x4A, pci_device_id=0),
             1: SimpleNamespace(pci_domain_id=0, pci_bus_id=0x9A, pci_device_id=0)}

    def get_props(index):
        if index not in props:
            raise AssertionError("Invalid device id")
        return props[index]

    monkeypatch.setattr(interface, "open", fake_open, raising=False)
    monkeypatch.setattr(interface.torch.cuda, "get_device_properties", get_props)
    for platform in (CudaPlatform(), RocmPlatform()):
        assert platform.local_cpus(0) == NODE0
        assert platform.local_cpus(1) is None  # no sysfs entry
        assert platform.local_cpus(5) is None  # no such device
    assert CpuPlatform().local_cpus(0) is None


# ── worker ──────────────────────────────────────────────────────────────────

def _bare_worker(worker_id="worker_1", device="cuda:1"):
    w = Worker.__new__(Worker)
    w.worker_id = worker_id
    w.device = torch.device(device)
    return w


def test_the_worker_picks_its_row_and_logs_it(monkeypatch, caplog):
    import mstar.worker.worker as worker_mod

    local = {0: NODE0, 1: NODE0, 2: NODE1}
    monkeypatch.setattr(worker_mod, "current_platform", SimpleNamespace(local_cpus=local.get))
    monkeypatch.setattr(worker_mod, "PINNABLE", True)
    monkeypatch.setattr(worker_mod, "smt_order", smt)
    monkeypatch.setattr(worker_mod.os, "sched_getaffinity", lambda pid: set(HOST), raising=False)
    monkeypatch.delenv("MSTAR_STEP_THREAD_CORE", raising=False)
    w = _bare_worker()
    with caplog.at_level(logging.INFO, logger="mstar.worker.worker"):
        w._choose_step_core(["worker_0", "worker_1", "worker_2"], ["cuda:0", "cuda:1", "cuda:2"])
    assert w._step_core.cpu == 1 and w._process_cpus == HOST
    assert "step threads (main, GPU, plan) on CPU 1 (local to its device: 0-31,128-159; " \
           "cuda:1, worker 2 of 3, allowed 0-255)" in caplog.text

    # standalone (no conductor list): the only worker
    w = _bare_worker()
    w._choose_step_core(["worker_1"], None)
    assert w._step_core.cpu == 0

    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "off")
    w = _bare_worker()
    with caplog.at_level(logging.INFO, logger="mstar.worker.worker"):
        w._choose_step_core(["worker_1"], None)
    assert w._step_core.cpu is None and "step threads not pinned (off)" in caplog.text


def test_without_sched_setaffinity_the_worker_pins_nothing(monkeypatch, caplog):
    import mstar.worker.worker as worker_mod

    monkeypatch.setattr(worker_mod, "PINNABLE", False)
    w = _bare_worker()
    with caplog.at_level(logging.INFO, logger="mstar.worker.worker"):
        w._choose_step_core(["worker_1"], None)
    assert w._step_core.cpu is None and "no sched_setaffinity" in caplog.text
    w._pin_step_thread()  # a no-op, never reaches os.sched_setaffinity


class _StopLoopError(BaseException):
    """Ends ``Worker.run``, whose loop handles only ``Exception``."""


@pytest.mark.skipif(not step_threads.PINNABLE, reason="needs os.sched_setaffinity (Linux)")
def test_run_moves_the_main_gpu_and_plan_threads_to_the_chosen_cpu(monkeypatch):
    """The real ``Worker.run``, every collaborator stubbed: one scheduled step,
    then a speculated one the plan thread pre-plans. Each of the three threads
    reports its mask from inside its own work; a thread one of them starts
    afterwards for intra-op work keeps the process mask."""
    import mstar.worker.worker as worker_mod
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import Speculation

    allowed = frozenset(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two allowed CPUs")
    cpu = max(allowed)
    monkeypatch.setenv("MSTAR_PRE_PLAN_SPEC", "1")
    monkeypatch.setattr(worker_mod, "gc", SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0))
    seen: dict[str, tuple[int, set[int]]] = {}
    schedule = [("a", 2), None]
    iteration = [0]

    def mask(name):
        seen[name] = threading.get_native_id(), os.sched_getaffinity(0)

    def scheduled(name, rows):
        return ScheduledBatch(node_name="LLM", graph_walk=name,
                              request_to_worker_graph={f"{name}.{i}": 0 for i in range(rows)})

    def executing(batch):
        return ExecutingBatch(
            node_name="LLM",
            per_request_info={rid: SimpleNamespace(rid=rid, n=1) for rid in batch.request_to_worker_graph},
            step_context=StepContext(request_ids=tuple(batch.request_to_worker_graph),
                                     graph_walk=batch.graph_walk, slot=0, capture=False),
        )

    def next_batch(request_state, exclude_target=None):
        mask("main")
        if not schedule:
            raise _StopLoopError
        step = schedule.pop(0)
        return None if step is None else scheduled(*step)

    def gpu(batch, node_batch, plan_future):
        mask("gpu")
        if plan_future is not None:
            plan_future.result()
        if node_batch.launch_started_event is not None:
            node_batch.launch_started_event.set()
        return {}

    def preplan(pending, spec):
        mask("plan")
        # a pool this thread starts now inherits its CPU; one it started
        # before moving does not
        torch.ones(torch.get_num_threads() * 65536).sum()
        return True

    def build_speculation(pending, consecutive, fair):
        iteration[0] += 1
        if iteration[0] != 2:
            return None, None
        batch = scheduled("b", 1)
        return Speculation(
            scheduled_batch=batch, node_batch=executing(batch), consumed_edges=set(),
            continuing_rids=set(), partition="p", is_new_iter=False, is_same_node=False,
            is_yield_away=True,
        ), None

    def stub(*a, **k):
        return None

    w = Worker.__new__(Worker)
    w.worker_id = "worker_0"
    w.device = torch.device("cpu")
    w.enable_nvtx = False
    w._phase_period, w._phase_buf = 0, {}
    w._max_consecutive_spec, w._launch_wait_s = 1024, 0.0
    w.tp_async_sched, w.is_tp_follower = False, False
    w._in_flight_rids = set()
    w._metrics = None
    w._step_core, w._process_cpus = StepCore(cpu, "test"), allowed
    w.parallel_groups = SimpleNamespace(barrier_all=stub, restore_default_timeout=stub)
    w.engine_manager = SimpleNamespace(warmup_all=stub, get_engine=lambda node: None)
    w.communicator = SimpleNamespace(send=stub, wait_for_work=stub)
    w.wakeup_event = SimpleNamespace(register_future=stub)
    w.request_state = SimpleNamespace(get_partition_for_node=lambda node: "p")
    w.scheduler = SimpleNamespace(take_admit_errors=dict, get_next_batch=next_batch, clear_batch_limit=stub)
    for name in ("_verify_tp_async_sched_agrees", "_await_admit_settled",
                 "_apply_pending_removes_safe_to_drop", "_fail_requests", "_apply_pending_drains",
                 "_process_messages", "_apply_removes_whose_step_landed", "_check_ready_tensors",
                 "_poll_stream_buffers", "_set_in_flight_flag", "_clear_in_flight_flag",
                 "_settle_speculation", "_drop_rows_whose_loop_stopped", "_handle_admit_failure",
                 "_thread_outputs_to_speculative", "_postprocess_batch"):
        setattr(w, name, stub)
    w.maybe_send_zmq_to_tp_followers = lambda node_batch: -1
    w._build_executing_batch = executing
    w._build_speculation = build_speculation
    w._plans_ahead = lambda spec: True
    w._preplan_spec = preplan
    w._execute_on_gpu_thread = gpu

    def fail_loudly(error, *args):
        raise error

    w._handle_main_loop_error = fail_loudly
    errors = []

    def run():
        try:
            w.run()
        except _StopLoopError:
            pass
        except BaseException as e:  # noqa: BLE001 — reported below
            errors.append(e)

    # run() pins the thread that calls it; keep that off pytest's own thread
    main = threading.Thread(target=run)
    main.start()
    main.join(60)
    assert not main.is_alive() and not errors, errors
    assert {name: m for name, (_, m) in seen.items()} == {"main": {cpu}, "gpu": {cpu}, "plan": {cpu}}
    step_tids = {tid for tid, _ in seen.values()}
    assert len(step_tids) == 3
    # no other thread of the process landed on the CPU, the intra-op pool the
    # plan thread just used included
    stray = []
    for tid in map(int, os.listdir("/proc/self/task")):
        try:
            if tid not in step_tids and os.sched_getaffinity(tid) == {cpu}:
                stray.append(tid)
        except OSError:  # exited meanwhile
            pass
    assert stray == []
