"""Pinning the step threads (``step_thread_core``): the CPU plan every process
builds (``mstar.utils.cpus``) from sysfs and the process mask, the lock that
keeps a second server off a GPU, the device list the API server and the
conductor share, the startup failures, the pool sizes, the thread helpers
(``mstar.utils.threads``) and the ruff rule that enforces them, and the real
``Worker.run`` moving its three step threads onto their CPUs. The knob itself is
in ``test_knobs``."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from mstar.conductor.workers import deployment_workers  # noqa: E402
from mstar.engine.engine import ExecutingBatch  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, interface  # noqa: E402
from mstar.utils import cpus, threads  # noqa: E402
from mstar.utils.cpus import CpuPlan, CpuPlanError, StepCpus, deployment_plan, parse_cpu_list  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NODES = [parse_cpu_list(f"{32 * n}-{32 * n + 31},{128 + 32 * n}-{128 + 32 * n + 31}") for n in range(4)]
HOST = frozenset(range(256))
# this test host's 8 GPUs, two per NUMA node, by PCI address
GPUS = {"0000:1a:00.0": 0, "0000:40:00.0": 0, "0000:62:00.0": 1, "0000:73:00.0": 1,
        "0000:9a:00.0": 2, "0000:da:00.0": 2, "0000:f3:00.0": 3, "0000:fa:00.0": 3}


def smt(cpu: int) -> frozenset[int]:
    """CPU n and n + 128 share a core."""
    return frozenset({cpu % 128, cpu % 128 + 128})


def _sysfs(tmp_path, gpus=GPUS, nodes=NODES) -> Path:
    """A fake /sys/bus/pci/devices: the GPUs, plus a BMC display adapter and a
    NIC that are not GPUs."""
    root = tmp_path / "pci"
    devices = {address: ("0x10de", "0x030200", node) for address, node in gpus.items()}
    devices |= {"0000:2c:00.0": ("0x1a03", "0x030000", 0), "0000:3b:00.0": ("0x15b3", "0x020700", 0)}
    for address, (vendor, cls, node) in devices.items():
        d = root / address
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
        (d / "class").write_text(cls + "\n")
        (d / "numa_node").write_text(f"{node}\n")
        local = cpus.format_cpu_list(nodes[node]) if node >= 0 else cpus.format_cpu_list(HOST)
        (d / "local_cpulist").write_text(local + "\n")
    return root


def _plan(setting, addresses, mask, root, usable=None):
    return deployment_plan(setting, addresses, mask=mask, usable=usable or len(mask), pci_root=root, siblings=smt)


def _one(cpu) -> StepCpus:
    return StepCpus(frozenset({cpu}), "")


def _cpus(plan: CpuPlan) -> list[int | None]:
    """Each worker's first step CPU (its core's first hardware thread)."""
    return [None if w is None else min(w.cpus) for w in plan.workers]


# ── where a GPU sits ────────────────────────────────────────────────────────

def test_the_hosts_gpus_are_the_functions_of_one_vendor_and_class_by_address(tmp_path):
    root = _sysfs(tmp_path)
    gpus = cpus.host_gpus("0000:40:00.0", root)
    # the display adapter on node 0 (another vendor and class) and the NIC are not GPUs
    assert [g.address for g in gpus] == list(GPUS)
    assert gpus[1] == cpus.HostGpu("0000:40:00.0", "0", NODES[0])
    assert cpus.host_gpus("0000:ff:00.0", root) == []   # no such device
    assert cpus.host_gpus("0000:40:00.0", tmp_path / "absent") == []


def test_pci_addresses_come_from_torch_on_cuda_and_rocm(monkeypatch):
    props = {0: SimpleNamespace(pci_domain_id=0, pci_bus_id=0x40, pci_device_id=0)}

    def get_props(index):
        if index not in props:
            raise AssertionError("Invalid device id")
        return props[index]

    monkeypatch.setattr(interface.torch.cuda, "get_device_properties", get_props)
    for platform in (CudaPlatform(), RocmPlatform()):
        assert platform.pci_address(0) == "0000:40:00.0"
        assert platform.pci_address(5) is None
    assert CpuPlatform().pci_address(0) is None


# ── auto ────────────────────────────────────────────────────────────────────

def test_auto_takes_the_cores_at_the_gpus_rank_on_its_node(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("auto", ["0000:40:00.0"], HOST, root)
    # GPU 2 of 2 on node 0: the worker the second core from the bottom, the
    # API server the second from the top
    # (both hardware threads of the worker's core, one CPU for the API server)
    assert plan.workers[0].cpus == {1, 129} and plan.result == 30
    assert "GPU 2 of 2 on its NUMA node" in plan.workers[0].why
    plan = _plan("auto", ["0000:62:00.0"], HOST, root)
    assert _cpus(plan) == [32] and plan.result == 63


@pytest.mark.parametrize(("mask", "a", "b"), [
    (HOST, "0000:1a:00.0", "0000:40:00.0"),                          # one node, no taskset
    (NODES[0] | NODES[1], "0000:40:00.0", "0000:62:00.0"),           # two nodes under one taskset
    (parse_cpu_list("0-19,128-147"), "0000:1a:00.0", "0000:40:00.0"),  # a 40-CPU taskset on their node
    (parse_cpu_list("32-39,96-103"), "0000:40:00.0", "0000:fa:00.0"),  # one GPU's node left out
    (parse_cpu_list("32-47"), "0000:40:00.0", "0000:fa:00.0"),         # both left out
])
def test_two_servers_one_per_gpu_under_one_taskset_take_different_cpus(tmp_path, mask, a, b):
    """Each server knows only its own GPU, and no two pick a CPU in common."""
    root = _sysfs(tmp_path)
    first, second = _plan("auto", [a], mask, root), _plan("auto", [b], mask, root)
    assert first.pinned and second.pinned
    assert not first.pinned & second.pinned


def test_a_vm_without_numa_nodes_still_ranks_its_gpus(tmp_path):
    root = _sysfs(tmp_path, gpus={"0000:07:00.0": -1, "0000:08:00.0": -1})
    mask = parse_cpu_list("0-39")
    one, two = (_plan("auto", [a], mask, root) for a in ("0000:07:00.0", "0000:08:00.0"))
    assert (_cpus(one), one.result, _cpus(two), two.result) == ([0], 39, [1], 38)


def test_workers_of_one_server_on_one_gpu_take_distinct_cores(tmp_path):
    root = _sysfs(tmp_path)
    # a split config (two workers on GPU 0) plus a worker on the node's other GPU
    plan = _plan("auto", ["0000:40:00.0", "0000:40:00.0", "0000:1a:00.0"], HOST, root)
    assert _cpus(plan) == [1, 3, 0] and plan.result == 30


def test_a_mask_off_the_gpus_node_falls_back_to_its_rank_among_the_gpus_off_theirs(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("auto", ["0000:40:00.0"], parse_cpu_list("32-47"), root)
    # node 1's two GPUs would take 32, 33, 46 and 47; six GPUs have no allowed
    # CPU on their node, and the second of them takes from 34-45
    assert _cpus(plan) == [35] and plan.result == 44
    assert "no allowed CPU on its NUMA node, so GPU 2 of 6 without one" in plan.workers[0].why
    # a device sysfs doesn't place (a CPU worker) takes the mask from the bottom
    plan = _plan("auto", [None, None], parse_cpu_list("32-39"), root)
    assert _cpus(plan) == [32, 33] and plan.result == 39


def test_auto_pins_nothing_when_the_mask_is_too_small(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("auto", [None], {32, 33}, root)   # a worker and the API server would take both
    assert plan.workers == () and plan.result is None
    assert plan.why.startswith("not pinned: ") and "leaving none for the rest" in plan.why
    plan = _plan("auto", [None] * 3, {32, 33, 34}, root)
    assert plan.pinned == frozenset() and "more cores than the mask has" in plan.why


@pytest.mark.parametrize(("mask", "address", "gpus"), [
    # node 1's GPUs take 32, 33, 38 and 39 and leave 4 cores for the 6 others:
    # rank 0 and rank 4 would both land on 34 and 37
    (parse_cpu_list("32-39"), "0000:1a:00.0", 6),
    (parse_cpu_list("32-39"), "0000:9a:00.0", 6),
    (parse_cpu_list("32-39,160-167"), "0000:40:00.0", 6),   # a core's sibling is the same core
    (parse_cpu_list("32-34"), "0000:62:00.0", 2),           # on their node: 3 cores for 2 GPUs
    (parse_cpu_list("32-34"), "0000:73:00.0", 2),
])
def test_auto_pins_nothing_with_fewer_than_two_cores_per_gpu_in_a_pool(tmp_path, mask, address, gpus):
    plan = _plan("auto", [address], mask, _sysfs(tmp_path))
    assert plan.pinned == frozenset() and plan.why.startswith("not pinned: ")
    assert f"for {gpus} GPU(s), and each needs two" in plan.why


def test_two_cores_per_gpu_is_enough(tmp_path):
    root = _sysfs(tmp_path)
    assert _plan("auto", ["0000:62:00.0"], parse_cpu_list("32-35"), root).pinned == {32, 35}
    assert _plan("auto", ["0000:73:00.0"], parse_cpu_list("32-35"), root).pinned == {33, 34}
    # 12 cores left for the 6 GPUs off their nodes
    assert _plan("auto", ["0000:1a:00.0"], parse_cpu_list("32-47"), root).pinned == {34, 45}
    assert _plan("auto", ["0000:fa:00.0"], parse_cpu_list("32-47"), root).pinned == {39, 40}


@pytest.mark.parametrize("mask", [
    HOST, NODES[1], parse_cpu_list("32-39"), parse_cpu_list("32-47"), parse_cpu_list("32-35"),
    parse_cpu_list("32-39,96-103"), parse_cpu_list("0-3,32-47"), parse_cpu_list("0-2,32-47"),
])
def test_no_two_gpus_of_the_host_share_a_cpu_under_one_mask(tmp_path, mask):
    """Whichever GPU a server gets, and whichever others are in use."""
    root = _sysfs(tmp_path)
    plans = [_plan("auto", [address], mask, root) for address in GPUS]
    for i, one in enumerate(plans):
        for other in plans[i + 1:]:
            assert not one.pinned & other.pinned


def test_workers_that_outgrow_their_pool_pin_nothing_rather_than_wrap(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-39")
    # GPU 1 of 2 on the node: 32, 34 and 36, then 38 is the other GPU's API server
    assert _cpus(_plan("auto", ["0000:62:00.0"] * 3, mask, root)) == [32, 34, 36]
    plan = _plan("auto", ["0000:62:00.0"] * 4, mask, root)
    assert plan.pinned == frozenset() and "more cores than the mask has" in plan.why


def test_a_worker_takes_both_hardware_threads_of_its_core(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-35,160-163")
    plan = _plan("auto", [None, None], mask, root)
    assert [w.cpus for w in plan.workers] == [{32, 160}, {33, 161}] and plan.result == 35
    assert plan.pinned == {32, 33, 35, 160, 161} and plan.left == 3
    # a mask with one hardware thread of the core: that one
    assert [w.cpus for w in _plan("auto", [None], parse_cpu_list("32-35"), root).workers] == [{32}]


# ── a CPU number, and off ───────────────────────────────────────────────────

def test_a_number_is_the_first_workers_cpu_and_the_api_takes_the_last_core(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-39")
    plan = _plan(34, ["0000:40:00.0", "0000:40:00.0"], mask, root)
    assert _cpus(plan) == [34, 35] and plan.result == 39
    # the number names a core: the worker takes its hardware threads in the mask
    assert _plan(160, [None], HOST, root).workers[0].cpus == {32, 160}


@pytest.mark.parametrize(("setting", "n", "message"), [
    (8, 1, "step_thread_core 8 is outside the process mask 32-39"),
    (38, 2, "step_thread_core 38 leaves 1 core(s) at or above it for 2 worker(s) in the process mask 32-39 "
            "(the last, 39, is the API server's)"),
    (39, 1, "leaves 0 core(s)"),
])
def test_a_number_the_mask_cannot_honour_fails_at_startup(tmp_path, setting, n, message):
    with pytest.raises(CpuPlanError, match=message.replace("(", r"\(").replace(")", r"\)")):
        _plan(setting, [None] * n, parse_cpu_list("32-39"), _sysfs(tmp_path))


def test_off_pins_nothing_and_the_pools_get_every_usable_cpu(tmp_path):
    plan = _plan("off", ["0000:40:00.0"], parse_cpu_list("32-39"), _sysfs(tmp_path))
    assert plan == CpuPlan(parse_cpu_list("32-39"), 8, why="step_thread_core off")
    assert plan.left == 8


# ── pool sizes ──────────────────────────────────────────────────────────────

def test_the_pools_are_sized_from_the_cpus_left_after_the_pins():
    mask = parse_cpu_list("32-47")
    free = CpuPlan(mask, usable=16)
    pinned = CpuPlan(mask, usable=16, workers=(_one(32), _one(33)), result=47)
    assert (free.left, free.preprocess_threads, free.png_threads, free.encode_workers) == (16, 4, 8, 2)
    assert (pinned.left, pinned.preprocess_threads, pinned.png_threads, pinned.encode_workers) == (13, 3, 8, 1)
    # the cgroup quota caps the count; a pinned CPU outside the mask isn't taken from it
    assert CpuPlan(mask, usable=6, workers=(_one(32),), result=99).left == 5


def test_the_api_server_reads_the_deployments_plan_from_its_config(monkeypatch):
    from mstar.api_server import entrypoint

    monkeypatch.setattr(entrypoint, "accelerator_type", lambda: "cuda")

    seen = {}

    def plan(setting, addresses, **kw):
        seen.update(setting=setting, addresses=addresses)
        return CpuPlan(frozenset(range(8)), 8)

    monkeypatch.setattr(entrypoint.cpus, "deployment_plan", plan)
    monkeypatch.setattr(entrypoint.cpus, "device_addresses", lambda devices: [f"at {d}" for d in devices])
    monkeypatch.delenv("MSTAR_STEP_THREAD_CORE", raising=False)
    config = {"knobs": {"step_thread_core": "auto"}, "rank_devices": {2: 0},
              "node_groups": [{"node_names": ["a"], "ranks": [2]}, {"node_names": ["b"], "ranks": [0, 1]}]}
    entrypoint.deployment_cpu_plan(config)
    # ranks in order, rank 2 placed on device 0 as the conductor places it
    assert seen == {"setting": "auto", "addresses": ["at cuda:0", "at cuda:1", "at cuda:0"]}
    entrypoint.deployment_cpu_plan({**config, "knobs": {"step_thread_core": 33}})
    assert seen == {"setting": 33, "addresses": [None, None, None]}
    entrypoint.knobs.configure(None)


def test_device_addresses_read_the_worker_device_strings(monkeypatch):
    from mstar import platforms

    monkeypatch.setattr(platforms.current_platform, "pci_address", lambda index: f"gpu {index}", raising=False)
    assert cpus.device_addresses(["cuda:0", "xpu:3", "cuda", "cpu"]) == ["gpu 0", "gpu 3", None, None]


# ── one list of workers ─────────────────────────────────────────────────────

def _shipped(name) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text())


def test_deployment_workers_skips_what_the_conductor_reports():
    groups = [3, {"ranks": "x"}, {"ranks": [2, "a", 0]}, {"node_names": []}]
    assert deployment_workers({"node_groups": groups}, "cuda") == ([0, 2], ["cuda:0", "cuda:2"])
    assert deployment_workers({"node_groups": "x"}, "cuda") == ([], [])
    assert deployment_workers({}, "cuda") == ([], [])
    assert deployment_workers({"node_groups": [{"ranks": [1]}], "rank_devices": {1: 0}}, "cpu").devices == ["cpu"]


@pytest.mark.parametrize(("name", "ranks", "devices"), [
    ("qwen3tts_1p7b_split.yaml", [0, 1], ["cuda:0", "cuda:0"]),                  # a split config
    ("qwen3_5_27b_tp4.yaml", [0, 1, 2, 3], ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]),  # TP
])
def test_the_api_server_and_the_conductors_workers_build_one_plan(
    name, ranks, devices, tmp_path, monkeypatch, gpu_locks,
):
    """From the shipped config: the real conductor's worker list and what it
    hands each worker, the real API server plan and the real worker plan. No
    worker's step core holds the API server's result CPU."""
    from test_startup_graph_validation import _llm, _start_conductor, _ToyModel

    from mstar.api_server import entrypoint
    from mstar.conductor import conductor as conductor_mod
    from mstar.conductor.conductor import Conductor

    config = _shipped(name)
    groups = config["node_groups"]
    nodes = {node: group.get("graph_walks") or [f"{node} walk"] for group in groups for node in group["node_names"]}
    model = _ToyModel(lambda: {walk: _llm(node) for node, walks in nodes.items() for walk in walks},
                      tp_enabled_nodes=[node for group in groups if group.get("tp_size", 1) > 1
                                        for node in group["node_names"]])
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(
        {"max_seq_len": 64, "node_groups": groups, "rank_devices": config.get("rank_devices")}))

    launch = Conductor._launch_workers
    monkeypatch.setattr(conductor_mod, "accelerator_type", lambda: "cuda")
    conductor = _start_conductor(model, str(path), tmp_path, monkeypatch)
    assert conductor._sorted_ranks == ranks and conductor._worker_devices == devices
    # every rank has graphs: the list from the config is what the walks use
    assert ranks == sorted({rank for wg in conductor.worker_graphs.values() for rank in wg.ranks})

    spawned = []

    class Process:
        def __init__(self, target, kwargs, daemon):
            spawned.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(
        conductor_mod, "mp", SimpleNamespace(get_context=lambda method: SimpleNamespace(Process=Process)))
    monkeypatch.setattr(conductor_mod, "atexit", SimpleNamespace(register=lambda fn: None))
    launch(conductor)
    assert [kw["device"] for kw in spawned] == devices

    root = _sysfs(tmp_path)
    at = {f"cuda:{i}": address for i, address in enumerate(list(GPUS)[:4])}
    plan = deployment_plan

    def placed(setting, addresses):
        return plan(setting, addresses, mask=HOST, usable=len(HOST), pci_root=root, siblings=smt)

    monkeypatch.setattr(cpus, "device_addresses", lambda ds: [at[d] for d in ds])
    monkeypatch.setattr(cpus, "deployment_plan", placed)
    monkeypatch.setattr(entrypoint, "accelerator_type", lambda: "cuda")
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    api = entrypoint.deployment_cpu_plan(config)
    assert len(api.workers) == len(ranks) and api.result is not None
    assert all(api.result not in step.cpus for step in api.workers)
    for kw, step in zip(spawned, api.workers, strict=True):
        worker = _bare_worker(kw["worker_id"], kw["device"])
        worker._plan_step_cpus(kw["worker_ids"], kw["local_worker_devices"])
        assert cpus.plan() == api and worker._step_cpus == step
    cpus.install(None)
    entrypoint.knobs.configure(None)


# ── one server per GPU ──────────────────────────────────────────────────────

# what a second server does: take the locks, say how that went, stay up until told to go
_OTHER_SERVER = """
import sys
from pathlib import Path
from mstar.utils import cpus
cpus._GPU_LOCK_DIR = Path(sys.argv[1])
print(cpus.claim_gpus(sys.argv[2:]), flush=True)
sys.stdin.read()
"""


@pytest.fixture
def gpu_locks(tmp_path, monkeypatch):
    """Locks in a directory of the test's, held by this process until it ends."""
    held: dict[str, int] = {}
    monkeypatch.setattr(cpus, "_GPU_LOCK_DIR", tmp_path)
    monkeypatch.setattr(cpus, "_gpu_locks", held)
    yield tmp_path
    for fd in held.values():
        os.close(fd)


@pytest.fixture
def other_server(gpu_locks):
    """Starts a process that claims the GPUs it's given; gives back the process
    and what its claim returned."""
    started = []

    def start(*addresses):
        p = subprocess.Popen(
            [sys.executable, "-c", _OTHER_SERVER, str(gpu_locks), *addresses],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        started.append(p)
        return p, p.stdout.readline().strip()

    yield start
    for p in started:
        p.kill()
        p.communicate()


A, B = "0000:40:00.0", "0000:fa:00.0"


def test_a_second_server_on_a_gpu_is_refused_until_the_first_exits(other_server):
    first, claimed = other_server(A)
    assert claimed == "None"
    refused = cpus.claim_gpus([A])
    assert f"another server holds GPU {A}" in refused and cpus._gpu_locks == {}
    # another GPU is free, and so is this one once the first has exited
    assert cpus.claim_gpus([B]) is None
    first.communicate("")
    assert first.returncode == 0
    assert cpus.claim_gpus([A]) is None


def test_a_killed_server_does_not_block_the_next(other_server):
    first, claimed = other_server(A)
    assert claimed == "None"
    first.kill()
    first.communicate()
    assert cpus.claim_gpus([A]) is None
    # and now it's this process that holds it
    assert "another server holds" in other_server(A)[1]


def test_a_refused_claim_holds_nothing(other_server):
    assert other_server(B)[1] == "None"
    assert f"another server holds GPU {B}" in cpus.claim_gpus([A, B])
    assert cpus._gpu_locks == {}
    assert other_server(A)[1] == "None"   # the lock on A was let go


def test_one_server_claims_its_gpus_once(gpu_locks):
    assert cpus.claim_gpus([A, None, A]) is None   # a split config, and a device sysfs doesn't place
    fd = cpus._gpu_locks[A]
    assert cpus.claim_gpus([A]) is None and cpus._gpu_locks == {A: fd}
    assert [path.name for path in gpu_locks.iterdir()] == [f"mstar-gpu-{A}.lock"]
    # a child doesn't inherit the fd, so it can't keep a dead server's lock
    assert not os.get_inheritable(fd)


def test_a_lock_that_cannot_be_opened_is_a_refusal(gpu_locks, monkeypatch):
    monkeypatch.setattr(cpus, "_GPU_LOCK_DIR", gpu_locks / "absent")
    assert f"can't lock GPU {A}" in cpus.claim_gpus([A])


def test_an_existing_lock_file_of_another_user_still_locks(gpu_locks, monkeypatch):
    real_open = os.open

    def refuse_create(path, flags, *mode):
        # the kernel answers O_EXCL on an existing file before the sticky check
        if flags & os.O_CREAT and not flags & os.O_EXCL:
            raise PermissionError("protected_regular")
        return real_open(path, flags, *mode)

    (gpu_locks / f"mstar-gpu-{A}.lock").touch()
    monkeypatch.setattr(cpus.os, "open", refuse_create)
    assert cpus.claim_gpus([A]) is None


def test_the_lock_file_is_readable_by_other_users_whatever_the_umask(gpu_locks):
    umask = os.umask(0o077)
    try:
        assert cpus.claim_gpus([A]) is None
    finally:
        os.umask(umask)
    mode = (gpu_locks / f"mstar-gpu-{A}.lock").stat().st_mode & 0o777
    assert mode == 0o644, "other users' servers couldn't open the lock, so none of them would pin on this GPU"


@pytest.fixture
def api_plan(tmp_path, monkeypatch):
    """``entrypoint.deployment_cpu_plan`` for a deployment on GPU ``A``, on the
    fake host, from a mask of 16 cores and their siblings."""
    from mstar.api_server import entrypoint

    root = _sysfs(tmp_path)
    plan = deployment_plan

    def placed(setting, addresses):
        return plan(setting, addresses, mask=parse_cpu_list("32-47,160-175"), usable=32, pci_root=root, siblings=smt)

    monkeypatch.setattr(cpus, "device_addresses", lambda devices: [A] * len(devices))
    monkeypatch.setattr(cpus, "deployment_plan", placed)
    monkeypatch.setattr(entrypoint, "accelerator_type", lambda: "cuda")
    config = {"node_groups": [{"node_names": ["llm"], "ranks": [0]}]}
    # back to this after the test: the refusal below writes the knob's env var
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    yield lambda: entrypoint.deployment_cpu_plan(config)
    cpus.install(None)
    entrypoint.knobs.configure(None)


def test_the_first_server_pins_and_the_second_pins_nothing_with_a_warning(other_server, api_plan, caplog):
    other, claimed = other_server(A)
    assert claimed == "None"
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        second = api_plan()
    assert second.pinned == frozenset() and f"not pinned: another server holds GPU {A}" in second.why
    assert f"nothing pinned, another server holds GPU {A}" in caplog.text
    # the pools get the whole mask
    assert second.left == 32
    # its workers read the knob, so they pin nothing either
    worker = _bare_worker("worker_0", "cuda:0")
    worker._plan_step_cpus(["worker_0"], ["cuda:0"])
    assert worker._step_cpus is None and cpus.plan().pinned == frozenset()

    other.communicate("")
    os.environ["MSTAR_STEP_THREAD_CORE"] = "auto"
    first = api_plan()
    assert first.pinned == {35, 163, 44} and A in cpus._gpu_locks
    worker._plan_step_cpus(["worker_0"], ["cuda:0"])
    assert worker._step_cpus.cpus == {35, 163}


def test_only_an_auto_plan_that_pins_takes_the_lock(api_plan, monkeypatch, gpu_locks):
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "off")
    assert api_plan().pinned == frozenset()
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "36")
    assert api_plan().pinned
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    monkeypatch.setattr(cpus, "device_addresses", lambda devices: ["0000:ff:00.0"] * len(devices))
    real = cpus.deployment_plan
    monkeypatch.setattr(cpus, "deployment_plan", lambda s, a: CpuPlan(frozenset({32}), 1, why="not pinned: too small"))
    assert api_plan().pinned == frozenset()
    monkeypatch.setattr(cpus, "deployment_plan", real)
    assert cpus._gpu_locks == {} and not list(gpu_locks.glob("mstar-gpu-*"))


# ── worker ──────────────────────────────────────────────────────────────────

def _bare_worker(worker_id="worker_1", device="cuda:1"):
    w = Worker.__new__(Worker)
    w.worker_id = worker_id
    w.device = torch.device(device)
    return w


def test_the_worker_takes_its_row_and_logs_it(tmp_path, monkeypatch, caplog):
    import mstar.worker.worker as worker_mod

    root = _sysfs(tmp_path)
    addresses = {"cuda:0": "0000:1a:00.0", "cuda:1": "0000:40:00.0", "cuda:2": "0000:62:00.0"}

    def plan(setting, addrs):
        return deployment_plan(setting, addrs, mask=HOST, usable=54, pci_root=root, siblings=smt)

    monkeypatch.setattr(worker_mod.cpus, "device_addresses", lambda devices: [addresses.get(d) for d in devices])
    monkeypatch.setattr(worker_mod.cpus, "deployment_plan", plan)
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    w = _bare_worker()
    with caplog.at_level(logging.INFO, logger="mstar.worker.worker"):
        w._plan_step_cpus(["worker_0", "worker_1", "worker_2"], ["cuda:0", "cuda:1", "cuda:2"])
    assert w._step_cpus.cpus == {1, 129} and w._process_cpus == HOST
    # three workers' cores and the API server's CPU leave 54 - 7 for the pools
    assert cpus.plan().result == 31 and cpus.plan().left == 54 - 7
    assert ("step threads (main, GPU, plan) on CPUs 1,129 (auto: GPU 2 of 2 on its NUMA node, over its allowed "
            "CPUs 0-31,128-159; cuda:1, worker 2 of 3, mask 0-255)") in caplog.text

    # standalone (no conductor list): the only worker
    w = _bare_worker()
    w._plan_step_cpus(["worker_1"], None)
    assert w._step_cpus.cpus == {1, 129}

    # a number the mask can't honour stops the worker
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "300")
    with pytest.raises(CpuPlanError, match="outside the process mask"):
        _bare_worker()._plan_step_cpus(["worker_1"], None)

    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "off")
    w = _bare_worker()
    with caplog.at_level(logging.INFO, logger="mstar.worker.worker"):
        w._plan_step_cpus(["worker_1"], None)
    assert w._step_cpus is None and "step threads not pinned (step_thread_core off)" in caplog.text
    w._pin_step_thread()  # a no-op
    cpus.install(None)


# ── threads a pinned thread starts ─────────────────────────────────────────

needs_affinity = pytest.mark.skipif(not cpus.PINNABLE, reason="needs os.sched_setaffinity (Linux)")


@pytest.fixture
def intra_op_pool():
    """At least two intra-op threads, so a pool exists to start, under any
    OMP_NUM_THREADS (CI sets 1)."""
    before = torch.get_num_threads()
    torch.set_num_threads(max(2, before))
    yield
    torch.set_num_threads(before)


def _two_cpus():
    allowed = frozenset(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two allowed CPUs")
    return allowed, max(allowed)


def _in_thread(target):
    """Run ``target`` on a fresh thread: pinning pins the caller, and pytest's
    own thread must keep its mask."""
    errors = []

    def run():
        try:
            target()
        except BaseException as e:  # noqa: BLE001 — reported below
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    t.join(60)
    assert not t.is_alive() and not errors, errors


@needs_affinity
def test_threads_and_pools_a_pinned_thread_starts_run_on_the_process_mask():
    """A new thread copies its creator's mask: a bare pool started after the
    pin sits on the pinned CPU, the helpers' threads do not."""
    allowed, cpu = _two_cpus()
    seen = {}

    def pinned():
        threads.pin_thread({cpu}, allowed)
        with ThreadPoolExecutor(1) as pool:
            seen["bare pool"] = pool.submit(os.sched_getaffinity, 0).result()
        with threads.ThreadPool(1, initializer=lambda: seen.setdefault("init", os.sched_getaffinity(0))) as pool:
            seen["pool"] = pool.submit(os.sched_getaffinity, 0).result()
        t = threads.Thread(target=lambda: seen.setdefault("thread", os.sched_getaffinity(0)))
        t.start()
        t.join()

    _in_thread(pinned)
    assert seen == {"bare pool": {cpu}, "init": allowed, "pool": allowed, "thread": allowed}


@needs_affinity
def test_a_thread_runs_on_the_process_mask_whatever_its_run_is():
    """No cooperation asked of the subclass; ``run`` called directly is the
    caller's own and stays on its mask."""
    allowed, cpu = _two_cpus()
    seen = {}

    class Override(threads.Thread):
        def run(self):
            seen["override"] = os.sched_getaffinity(0)

    class Below(Override):
        def run(self):
            seen["below"] = os.sched_getaffinity(0)

    class Mixin:
        def run(self):
            seen["mixin"] = os.sched_getaffinity(0)

    class Mixed(Mixin, threads.Thread):
        pass

    assigned = threads.Thread()
    assigned.run = lambda: seen.setdefault("assigned", os.sched_getaffinity(0))

    def pinned():
        threads.pin_thread({cpu}, allowed)
        for thread in (Override(), Below(), Mixed(), assigned):
            thread.start()
            thread.join()
        Override().run()
        seen["called directly"] = seen.pop("override")

    _in_thread(pinned)
    assert seen == {"below": allowed, "mixin": allowed, "assigned": allowed, "called directly": {cpu}}


@needs_affinity
def test_an_override_started_from_a_pinned_thread_is_on_the_process_mask():
    allowed, cpu = _two_cpus()
    seen = {}

    class Override(threads.Thread):
        def run(self):
            seen["override"] = os.sched_getaffinity(0)

    def pinned():
        threads.pin_thread({cpu}, allowed)
        thread = Override()
        thread.start()
        thread.join()

    _in_thread(pinned)
    assert seen == {"override": allowed}


def _tids() -> set[int]:
    return set(map(int, os.listdir("/proc/self/task")))


def _confined_to(cpus: int | frozenset[int], besides: set[int]) -> list[int]:
    """This process's threads, other than ``besides``, whose mask is just
    ``cpus``. ``besides`` holds the threads a test started with: an earlier
    test's idle pool threads may still be on their way out."""
    stray = []
    for tid in map(int, os.listdir("/proc/self/task")):
        try:
            if tid not in besides and os.sched_getaffinity(tid) == ({cpus} if isinstance(cpus, int) else cpus):
                stray.append(tid)
        except OSError:  # exited meanwhile
            pass
    return stray


@needs_affinity
def test_the_pin_starts_the_intra_op_pool_on_the_process_mask(intra_op_pool):
    """Without the warm-up op in ``pin_thread`` the pinned thread's first
    parallel op would start its pool on the pinned CPU."""
    allowed, cpu = _two_cpus()
    seen = {}
    before = _tids()

    def pinned():
        threads.pin_thread({cpu}, allowed)
        torch.ones(torch.get_num_threads() * 65536).sum()
        # the pool lives as long as its master thread, so look now
        seen["stray"] = _confined_to(cpu, before | {threading.get_native_id()})

    _in_thread(pinned)
    assert seen == {"stray": []}


# ── the ruff rule ───────────────────────────────────────────────────────────

def _ruff() -> str:
    path = shutil.which("ruff")
    if path is None:
        try:
            from ruff.__main__ import find_ruff_bin
            path = find_ruff_bin()
        except (ImportError, FileNotFoundError):
            pytest.skip("ruff is not installed")
    return path


@pytest.mark.parametrize(("path", "flagged"), [
    ("mstar/worker/new_module.py", True),
    ("mstar/utils/threads.py", False),             # the helpers
    ("mstar/model/foo/tests/test_foo.py", False),  # tests
    ("test/modular/test_foo.py", False),
    ("benchmark/foo.py", False),
])
def test_bare_threads_are_refused_under_mstar_only(path, flagged):
    source = ("import threading\nfrom concurrent.futures import ThreadPoolExecutor\n\n"
              "threading.Thread(target=print).start()\nThreadPoolExecutor(1).shutdown()\n")
    out = subprocess.run([_ruff(), "check", "--no-cache", "--select", "TID251", "--stdin-filename", path, "-"],
                         input=source, capture_output=True, text=True, cwd=REPO, check=False)
    assert ("TID251" in out.stdout) == flagged, out.stdout + out.stderr


# ── API server ─────────────────────────────────────────────────────────────

@needs_affinity
def test_the_api_message_thread_and_the_bridge_move_to_the_result_cpu(monkeypatch):
    from api_server_helpers import StandInSocket, make_test_api_server

    from mstar.api_server.rust_frontend import RustFrontendBridge
    from mstar.communication.event import EventWakeup

    allowed, cpu = _two_cpus()
    exited = SimpleNamespace(pid=5, is_alive=lambda: False, exitcode=1)
    server = make_test_api_server(
        conductor_proc=exited, _liveness_interval_s=0.0, _result_cpu=cpu, _process_cpus=allowed,
        communicator=StandInSocket(), preprocess_worker=SimpleNamespace(results_wakeup=EventWakeup()),
    )
    seen = {}

    def message_thread():
        server._process_messages()  # finds the conductor gone and returns
        seen["message"] = os.sched_getaffinity(0)

    async def serve():
        seen["bridge"] = os.sched_getaffinity(0)

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = server
    monkeypatch.setattr(bridge, "_serve", serve, raising=False)
    for target in (message_thread, bridge.run):
        _in_thread(target)
    assert seen == {"message": {cpu}, "bridge": {cpu}}
    # a server built without a CPU (tests, ``off``, macOS) pins nothing
    make_test_api_server().pin_result_thread()


def test_the_api_server_takes_its_cpu_from_the_plan_and_logs_it(monkeypatch, caplog):
    from api_server_helpers import make_test_api_server

    mask = parse_cpu_list("32-39")
    monkeypatch.setattr(cpus, "_plan", CpuPlan(mask, 8, (_one(33),), 38, "auto: test"))
    server = make_test_api_server()
    with caplog.at_level(logging.INFO, logger="mstar.api_server.entrypoint"):
        server._take_result_cpu()
    assert (server._result_cpu, server._process_cpus) == (38, mask)
    assert "message thread and frontend bridge on CPU 38 (auto: test; mask 32-39)" in caplog.text
    monkeypatch.setattr(cpus, "_plan", CpuPlan(mask, 8, why="not pinned: too small"))
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        server._take_result_cpu()
    assert server._result_cpu is None and "result threads not pinned (not pinned: too small)" in caplog.text


class _StopLoopError(BaseException):
    """Ends ``Worker.run``, whose loop handles only ``Exception``."""


@needs_affinity
@pytest.mark.parametrize("width", [1, 2])
def test_run_moves_the_main_gpu_and_plan_threads_to_their_cpus(monkeypatch, intra_op_pool, width):
    """The real ``Worker.run``, every collaborator stubbed: one scheduled step,
    then a speculated one the plan thread pre-plans. Each of the three threads
    reports its mask from inside its own work; a thread one of them starts
    afterwards for intra-op work keeps the process mask."""
    import mstar.worker.worker as worker_mod
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import Speculation

    allowed, cpu = _two_cpus()
    step = frozenset({cpu, min(allowed)} if width == 2 else {cpu})
    before = _tids()
    monkeypatch.setenv("MSTAR_PRE_PLAN_SPEC", "1")
    monkeypatch.setattr(worker_mod, "gc", SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0))
    seen: dict[str, tuple[int, set[int]]] = {}
    schedule = [("a", 2), None]
    iteration = [0]

    def mask(name, stray=()):
        seen[name] = threading.get_native_id(), os.sched_getaffinity(0), stray

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
        # a pool this thread started before moving keeps the process mask; one
        # it started now would sit on its CPU
        torch.ones(torch.get_num_threads() * 65536).sum()
        # by name: the GPU thread may not have reported yet
        gpu_thread = {t.native_id for t in threading.enumerate() if t.name.startswith("mstar-gpu-")}
        steps = before | {seen["main"][0], threading.get_native_id()} | gpu_thread
        mask("plan", _confined_to(step, steps))
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
    w._victims = {}
    w._logged_pools = []
    w._metrics = None
    w._step_cpus = StepCpus(step, "test")
    w._process_cpus = allowed
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
    assert {name: m for name, (_, m, _) in seen.items()} == {"main": step, "gpu": step, "plan": step}
    assert len({tid for tid, _, _ in seen.values()}) == 3
    # no other thread of the process landed on a step CPU, the intra-op pool
    # the plan thread just used included (looked at while it was alive)
    assert seen["plan"][2] == []
