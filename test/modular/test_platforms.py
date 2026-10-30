"""``mstar.platforms``: which hardware the process runs on."""

import copy
import multiprocessing
import os
import pickle
from types import SimpleNamespace

import pytest
import torch

from mstar import platforms
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform, current_platform


@pytest.fixture
def resolve(monkeypatch):
    """Resolve afresh for a given torch build and visible devices."""

    def run(*, cuda: bool, hip: str | None = None, xpu: bool = False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: xpu))
        platforms.resolve_platform.cache_clear()
        return platforms.resolve_platform()

    yield run
    platforms.resolve_platform.cache_clear()


def test_a_rocm_build_is_rocm_though_torch_calls_it_cuda(resolve):
    platform = resolve(cuda=True, hip="7.0")
    assert platform.is_rocm() and not platform.is_cuda()


def test_a_cuda_build_is_cuda(resolve):
    platform = resolve(cuda=True)
    assert platform.is_cuda() and not platform.is_rocm()


def test_xpu_and_cpu(resolve):
    assert resolve(cuda=False, xpu=True).is_xpu()
    assert resolve(cuda=False).is_cpu()


def test_current_platform_follows_resolution(resolve):
    resolve(cuda=False)
    assert current_platform.is_cpu()
    assert repr(current_platform) == "CpuPlatform()"


def test_resolving_caches_the_platform():
    platforms.resolve_platform()
    assert platforms.resolve_platform.cache_info().currsize == 1


def test_the_next_test_starts_with_no_platform_cached():
    # test/modular/conftest.py clears the cache around every test
    assert platforms.resolve_platform.cache_info().currsize == 0


def test_current_platform_pickles_and_copies_as_itself(resolve):
    resolve(cuda=False)
    assert pickle.loads(pickle.dumps(current_platform)) is current_platform
    assert copy.copy(current_platform) is current_platform
    assert copy.deepcopy(current_platform) is current_platform


def test_an_mi300x_is_not_a_hopper(monkeypatch):
    """ROCm reports the GFX version where CUDA reports compute capability."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 4))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
    )
    platform = RocmPlatform()
    assert platform.arch() == "gfx942"
    assert platform.cuda_capability() is None
    assert not platform.has_cuda_capability(9)


@pytest.mark.parametrize(
    ("capability", "arch", "hopper"),
    [((8, 0), "sm_80", False), ((9, 0), "sm_90", True), ((12, 0), "sm_120", True)],
)
def test_cuda_arch_and_capability(monkeypatch, capability, arch, hopper):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    platform = CudaPlatform()
    assert platform.arch() == arch
    assert platform.has_cuda_capability(9) is hopper
    assert platform.has_cuda_capability(*capability)


@pytest.mark.parametrize(
    ("hip", "cuda", "xpu", "expected"),
    [
        ("7.0", None, None, RocmPlatform),
        (None, "12.8", None, CudaPlatform),
        (None, None, "20250101", XpuPlatform),
        (None, None, None, CpuPlatform),
    ],
)
def test_env_defaults_come_from_the_build_without_a_device_call(
    monkeypatch, hip, cuda, xpu, expected,
):
    """Resolving ``current_platform`` would initialize the runtime, which reads
    its environment at init."""

    def device_call():
        raise AssertionError("a device call before the environment was set")

    monkeypatch.setattr(torch.cuda, "is_available", device_call)
    monkeypatch.setattr(platforms, "resolve_platform", device_call)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "xpu", xpu, raising=False)
    monkeypatch.setattr(
        expected, "process_env_defaults", lambda self: {"PLATFORM_ENV_TEST": type(self).__name__},
    )
    # set then deleted, so teardown removes what the call sets
    monkeypatch.setenv("PLATFORM_ENV_TEST", "")
    monkeypatch.delenv("PLATFORM_ENV_TEST")

    platforms.apply_process_env_defaults()
    assert os.environ["PLATFORM_ENV_TEST"] == expected.__name__


def test_env_defaults_keep_a_value_already_set(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(
        RocmPlatform,
        "process_env_defaults",
        lambda self: {"PLATFORM_ENV_TEST_UNSET": "default", "PLATFORM_ENV_TEST_SET": "default"},
    )
    monkeypatch.setenv("PLATFORM_ENV_TEST_UNSET", "")
    monkeypatch.delenv("PLATFORM_ENV_TEST_UNSET")
    monkeypatch.setenv("PLATFORM_ENV_TEST_SET", "user")

    platforms.apply_process_env_defaults()
    assert os.environ["PLATFORM_ENV_TEST_UNSET"] == "default"
    assert os.environ["PLATFORM_ENV_TEST_SET"] == "user"


def test_a_callable_env_default_adds_to_the_value_already_set(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(
        RocmPlatform, "process_env_defaults",
        lambda self: {"PLATFORM_ENV_TEST_SET": lambda current: f"{current}-default"},
    )
    monkeypatch.setenv("PLATFORM_ENV_TEST_SET", "user")

    platforms.apply_process_env_defaults()
    assert os.environ["PLATFORM_ENV_TEST_SET"] == "user-default"


def test_a_worker_sets_its_env_defaults_before_anything_else(monkeypatch):
    from mstar.conductor import conductor

    class AppliedError(Exception):
        """Raised by the stand-in, so nothing after it runs."""

    def apply():
        raise AppliedError

    monkeypatch.setattr(conductor, "apply_process_env_defaults", apply)
    with pytest.raises(AppliedError):
        conductor._worker_process_target(*[None] * 12)


@pytest.fixture
def rocm_env_default(monkeypatch):
    """A ROCm build whose platform defaults PLATFORM_ENV_TEST, which is unset."""
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(RocmPlatform, "process_env_defaults", lambda self: {"PLATFORM_ENV_TEST": "default"})
    # set then deleted, so teardown removes what the test sets
    monkeypatch.setenv("PLATFORM_ENV_TEST", "")
    monkeypatch.delenv("PLATFORM_ENV_TEST")
    return "PLATFORM_ENV_TEST"


def test_a_spawned_child_inherits_the_env_defaults(rocm_env_default):
    platforms.apply_process_env_defaults()
    check = f"import os, sys; sys.exit(0 if os.environ.get({rocm_env_default!r}) == 'default' else 3)"
    child = multiprocessing.get_context("spawn").Process(target=exec, args=(check,))
    child.start()
    child.join(timeout=120)
    assert child.exitcode == 0


def test_the_conductor_sets_env_defaults_before_spawning_workers(monkeypatch, rocm_env_default):
    """A spawned worker imports the model's modules while unpickling, before
    its target runs, so the defaults have to be in the parent's environment."""
    from mstar.conductor import conductor

    seen = []

    class Process:
        def __init__(self, **kwargs):
            seen.append(os.environ.get(rocm_env_default))

        def start(self):
            seen.append(os.environ.get(rocm_env_default))

    monkeypatch.setattr(conductor, "mp", SimpleNamespace(get_context=lambda method: SimpleNamespace(Process=Process)))
    monkeypatch.setattr(conductor, "atexit", SimpleNamespace(register=lambda fn: None))
    workers = ["w0", "w1"]
    spawner = SimpleNamespace(
        _sorted_ranks=[0, 1], worker_ids=workers, _worker_processes=[],
        _per_worker_graphs=dict.fromkeys(workers), per_worker_sharding_config=dict.fromkeys(workers),
        parallel_config=SimpleNamespace(per_worker_config=dict.fromkeys(workers)),
        model_config={}, _all_worker_graph_ids_to_graph_walks={}, _all_worker_graph_ids_to_nodes={},
        _all_worker_graph_ids_to_dyn_loops={}, hostname="host", socket_path_prefix="/tmp/x",
        _dist_init_method="", model=None, enable_nvtx=False, enable_prof=False, device_type="cuda",
        rank_devices=None, log_level="INFO", tensor_comm_protocol=None, tcp_transfer_device="",
        _status_interval_s=None,
        _mps_requested=False, _colocation=None,
        shutdown=None,
    )
    conductor.Conductor._launch_workers(spawner)
    assert seen == ["default"] * 4
    assert len(spawner._worker_processes) == 2


# forward convs, and backward-data convs (what a transposed conv runs)
MIOPEN_SPLIT_K = (
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC",
)


@pytest.mark.parametrize("solver", MIOPEN_SPLIT_K)
def test_only_rocm_turns_off_miopens_split_k_conv_solvers(solver):
    assert RocmPlatform().process_env_defaults()[solver] == "0"
    for platform in (CudaPlatform(), XpuPlatform(), CpuPlatform()):
        assert solver not in platform.process_env_defaults()


IN_GRAPH_KNOBS = ("ingraph_decode_rope", "sampler_ingraph_scatter")


def test_only_rocm_builds_decode_rope_and_sampler_scatter_in_the_graph():
    assert RocmPlatform().knob_defaults() == {
        **{knob: "1" for knob in IN_GRAPH_KNOBS}, "step_thread_core": "off", "dit_exact_ops": "0",
    }
    for platform in (CudaPlatform(), XpuPlatform(), CpuPlatform()):
        assert platform.knob_defaults() == {}


def test_only_rocm_turns_tp_async_scheduling_on_by_default():
    assert RocmPlatform().tp_async_sched_default()
    for platform in (CudaPlatform(), XpuPlatform(), CpuPlatform()):
        assert not platform.tp_async_sched_default()


@pytest.mark.parametrize("solver", MIOPEN_SPLIT_K)
@pytest.mark.parametrize(("exported", "expected"), [(None, "0"), ("1", "1")])
def test_the_miopen_solver_default_keeps_an_exported_value(monkeypatch, solver, exported, expected):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    # set first, so teardown restores the variable whatever the call does
    for name in MIOPEN_SPLIT_K:
        monkeypatch.setenv(name, "")
    if exported is None:
        monkeypatch.delenv(solver)
    else:
        monkeypatch.setenv(solver, exported)
    platforms.apply_process_env_defaults()
    assert os.environ[solver] == expected


# --- device selection (mstar serve --gpus) ----------------------------------


@pytest.fixture
def device_env(monkeypatch):
    """Starts with no device variables set. Teardown restores what the test sets."""
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    return monkeypatch


@pytest.mark.parametrize(
    ("hip", "cuda", "xpu", "expected"),
    [
        (None, "12.8", None, {"CUDA_VISIBLE_DEVICES": "0,1"}),
        (None, None, "20250101", {"CUDA_VISIBLE_DEVICES": "0,1"}),
        ("7.0", None, None, {"CUDA_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1"}),
    ],
)
def test_gpus_select_devices_without_a_device_call(device_env, hip, cuda, xpu, expected):
    """ROCm sets both variables since HIP reads HIP_VISIBLE_DEVICES first.
    Other builds set only CUDA_VISIBLE_DEVICES, as before."""

    def device_call():
        raise AssertionError("a device call before the devices were selected")

    device_env.setattr(torch.cuda, "is_available", device_call)
    device_env.setattr(platforms, "resolve_platform", device_call)
    device_env.setattr(torch.version, "hip", hip)
    device_env.setattr(torch.version, "cuda", cuda)
    device_env.setattr(torch.version, "xpu", xpu, raising=False)

    platforms.apply_visible_devices("0,1")
    selected = {
        key: os.environ[key]
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        if key in os.environ
    }
    assert selected == expected


def test_gpus_replace_a_leftover_hip_selection_on_rocm(device_env):
    device_env.setattr(torch.version, "hip", "7.0")
    device_env.setenv("HIP_VISIBLE_DEVICES", "3")
    platforms.apply_visible_devices("0")
    assert os.environ["HIP_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize(
    ("hip_env", "cuda_env"), [(None, None), ("1", None), (None, "1"), ("0,1", "0,1")],
)
def test_without_gpus_a_consistent_environment_stands(device_env, hip_env, cuda_env):
    device_env.setattr(torch.version, "hip", "7.0")
    if hip_env is not None:
        device_env.setenv("HIP_VISIBLE_DEVICES", hip_env)
    if cuda_env is not None:
        device_env.setenv("CUDA_VISIBLE_DEVICES", cuda_env)
    platforms.apply_visible_devices(None)
    assert os.environ.get("HIP_VISIBLE_DEVICES") == hip_env
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == cuda_env


def test_without_gpus_conflicting_selections_fail_on_rocm(device_env):
    device_env.setattr(torch.version, "hip", "7.0")
    device_env.setenv("HIP_VISIBLE_DEVICES", "1")
    device_env.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="HIP_VISIBLE_DEVICES='1'"):
        platforms.apply_visible_devices(None)


def test_without_gpus_cuda_ignores_hip_selection(device_env):
    device_env.setattr(torch.version, "hip", None)
    device_env.setattr(torch.version, "cuda", "12.8")
    device_env.setenv("HIP_VISIBLE_DEVICES", "1")
    device_env.setenv("CUDA_VISIBLE_DEVICES", "0")
    platforms.apply_visible_devices(None)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_serve_selects_gpus_through_the_platform(device_env):
    from mstar.cli import main as cli

    selected = []

    class SelectedError(Exception):
        """Raised by the stand-in, so the server never starts."""

    def apply(devices):
        selected.append(devices)
        raise SelectedError

    device_env.setattr(platforms, "apply_visible_devices", apply)
    with pytest.raises(SelectedError):
        cli.main(["serve", "kokoro", "--gpus", "2,3"])
    assert selected == ["2,3"]


def test_cuda_keeps_nccl_graph_registration_off(monkeypatch):
    """NCCL registering the buffers captured in CUDA graphs gave wrong Cosmos3
    SP4 images (#915); a branch that rewrites the platform defaults keeps it."""
    from mstar.platforms import interface

    monkeypatch.setattr(interface, "blackwell_ptxas_env", lambda: {})
    assert CudaPlatform().process_env_defaults() == {"NCCL_GRAPH_REGISTER": "0"}


@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform(), XpuPlatform(), CpuPlatform()], ids=repr)
def test_env_defaults_are_strings_or_callables(platform):
    """What ``apply_process_env_defaults`` knows how to apply."""
    for key, value in platform.process_env_defaults().items():
        assert isinstance(key, str) and (isinstance(value, str) or callable(value)), key

def _fake_runtime(api: str, register_rc: int = 0, unregister_rc: int = 0):
    """libcudart / libamdhip64 as far as ``HostRegistrar`` uses it: the three
    functions, returning scripted codes and recording each call."""
    calls = []

    def register(ptr, nbytes, flags):
        calls.append(("register", ptr, nbytes, flags))
        return register_rc

    def unregister(ptr):
        calls.append(("unregister", ptr))
        return unregister_rc

    def get_last_error():
        calls.append(("get_last_error",))
        return 0

    lib = SimpleNamespace(**{
        f"{api}HostRegister": register,
        f"{api}HostUnregister": unregister,
        f"{api}GetLastError": get_last_error,
    })
    return lib, calls


@pytest.mark.parametrize("platform", [CpuPlatform(), XpuPlatform()])
def test_only_gpu_platforms_page_lock_host_memory(platform):
    assert not platform.host_register(1 << 20, 4096)
    assert not platform.host_unregister(1 << 20)


@pytest.mark.parametrize("api", ["cuda", "hip"])
@pytest.mark.parametrize(
    ("rc", "registered", "error_read"),
    [(0, True, False), (712, True, True), (1, False, True)],
)
def test_host_register_accepts_a_range_already_registered(api, rc, registered, error_read):
    """712 is the already-registered code of both runtimes. Any nonzero code
    is read back, or the thread's next kernel launch would raise it."""
    from mstar.platforms.host_memory import HostRegistrar

    lib, calls = _fake_runtime(api, register_rc=rc)
    registrar = HostRegistrar(api, lambda: lib)
    assert registrar.register(0x1000, 4096) is registered
    assert calls[0] == ("register", 0x1000, 4096, 0)
    assert (("get_last_error",) in calls) is error_read


def test_host_unregister_of_a_range_never_registered_fails_and_clears_the_error():
    from mstar.platforms.host_memory import HostRegistrar

    lib, calls = _fake_runtime("hip", unregister_rc=713)
    assert not HostRegistrar("hip", lambda: lib).unregister(0x1000)
    assert calls == [("unregister", 0x1000), ("get_last_error",)]


def test_a_runtime_that_cannot_load_is_reported_once(caplog):
    from mstar.platforms.host_memory import HostRegistrar

    loads = []

    def load():
        loads.append(1)
        raise OSError("not here")

    registrar = HostRegistrar("hip", load)
    with caplog.at_level("WARNING", logger="mstar.platforms.host_memory"):
        assert not registrar.register(0x1000, 4096)
        assert not registrar.register(0x2000, 4096)
        assert not registrar.unregister(0x1000)
    assert len(loads) == 1
    assert [r.getMessage() for r in caplog.records if "cannot load" in r.getMessage()] == [
        "cannot load the hip runtime to page-lock host memory (not here); copies through such memory stay pageable"
    ]


def test_rocm_binds_only_the_hip_runtime_torch_already_loaded(monkeypatch):
    """Loading a second HIP runtime (the system ROCm's) would register memory
    with a runtime torch's copies never consult."""
    from mstar.platforms import host_memory

    opened = []

    def cdll(name, mode=0):
        opened.append((name, mode))
        raise OSError(f"{name}: not loaded")

    monkeypatch.setattr(torch.version, "hip", "7.2.53211")
    monkeypatch.setattr(host_memory.ctypes, "CDLL", cdll)
    with pytest.raises(OSError, match="is loaded in this process"):
        host_memory._load_amdhip()
    assert [name for name, _ in opened] == ["libamdhip64.so", "libamdhip64.so.7"]
    assert all(mode & os.RTLD_NOLOAD for _, mode in opened)


@pytest.mark.parametrize(("platform", "registrar"), [(RocmPlatform(), "HIP_HOST"), (CudaPlatform(), "CUDA_HOST")])
def test_each_gpu_platform_registers_through_its_own_runtime(monkeypatch, platform, registrar):
    from mstar.platforms import host_memory

    api = "hip" if registrar == "HIP_HOST" else "cuda"
    lib, calls = _fake_runtime(api)
    monkeypatch.setattr(host_memory, registrar, host_memory.HostRegistrar(api, lambda: lib))
    assert platform.host_register(0x1000, 8192)
    assert platform.host_unregister(0x1000)
    assert calls == [("register", 0x1000, 8192, 0), ("unregister", 0x1000)]
