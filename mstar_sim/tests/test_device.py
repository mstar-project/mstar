"""The emulated GPU keeps the CUDA behaviours mstar's code depends on.

Each case runs in a fresh interpreter with the emulator installed, as
``python -m mstar_sim`` would start one.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_sim(code: str, gpu: str = "h100", **extra_env: str) -> str:
    env = dict(os.environ, MSTAR_SIM="1", MSTAR_SIM_GPU=gpu, **extra_env,
               PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "mstar_sim", "_site"), ROOT,
                                           os.environ.get("PYTHONPATH", "")]))
    with tempfile.TemporaryDirectory() as tmp:  # a file, so spawned ranks can import it
        script = os.path.join(tmp, "case.py")
        with open(script, "w") as f:
            f.write(textwrap.dedent(code))
        proc = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True,
                              timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


def test_cuda_semantics():
    out = run_sim("""
        import torch
        assert torch.cuda.is_available() and torch.cuda.get_device_name().startswith("NVIDIA H100")
        m = torch.nn.Linear(8, 4, device="meta").to_empty(device="cuda")
        x = torch.randn(3, 8, device="cuda")
        assert m(x).is_cuda and m.weight.device == torch.device("cuda", 0)
        assert isinstance(m.weight, torch.nn.Parameter)
        m.register_buffer("table", torch.arange(4.0), persistent=False)
        m = m.to(device="cuda:0", dtype=torch.bfloat16)  # kwargs form, as mstar's engine calls it
        assert m.table.is_cuda and m.table.dtype == torch.bfloat16
        emb, head = torch.nn.Embedding(10, 4), torch.nn.Linear(4, 10, bias=False)
        head.weight = emb.weight  # tied, as a model ties its lm_head
        tied = torch.nn.ModuleList([emb, head]).to("cuda")
        assert tied[1].weight is tied[0].weight  # a loader's write to the embedding reaches the head
        host = torch.arange(3)
        dev = host.to(device="cuda", dtype=torch.int32, copy=True)
        assert dev.is_cuda and dev.dtype == torch.int32 and dev.tolist() == [0, 1, 2]
        staged = torch.zeros_like(dev, device="cpu", pin_memory=True)  # a host staging buffer
        assert type(staged) is torch.Tensor and staged.device.type == "cpu"
        try:
            x + torch.ones(3, 8)
            raise SystemExit("a cuda/cpu mix was accepted")
        except RuntimeError as e:
            assert "same device" in str(e)
        assert (x + torch.tensor(1.0)).is_cuda          # CPU scalars are fine
        assert x[torch.tensor([0, 2])].shape == (2, 8)  # so is a CPU index
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert m(x).dtype == torch.bfloat16
        # small host floats keep their values; beside a never-computed operand they join it
        v, g = torch.empty(512, 256, 3, device="cuda"), torch.randn(512, 1, 1).cuda()
        assert torch._weight_norm(v, g, 0).shape == v.shape  # a composite's same-device check
        assert torch.stack([x.sum(), torch.ones((), device="cuda")]).shape == (2,)
        torch.cuda.manual_seed_all(3)  # dynamo saves and restores this state
        assert [gen.initial_seed() for gen in torch.cuda.default_generators] == [3] * torch.cuda.device_count()
        torch.cuda.set_rng_state(torch.cuda.get_rng_state(1), 1)
        print("ok")
    """)
    assert out.strip() == "ok"


def test_graph_replay_costs_its_kernels_and_reprices_value_dependent_ones():
    """A replay costs what its kernels cost eagerly; a kernel priced from the
    values in its arguments (attention over the latest plan) is priced again
    on every replay; a host read during capture raises, as on CUDA."""
    out = run_sim("""
        import torch
        from mstar_sim import clock

        # priced from the lengths in `lens` (small ints keep values), as attention is
        @clock.kernel_fn("test.rows", cost=lambda lens, out: (0.0, 1e9 * float(lens._data.sum())))
        def rows(lens, out):
            return out

        def timed(fn):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize()
            return start.elapsed_time(end) / 1000

        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        lens = torch.ones(2, dtype=torch.int32, device="cuda")
        buf = torch.empty(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            c = a @ a
            rows(lens, buf)
        eager = timed(lambda: (a @ a, rows(lens, buf)))
        short = timed(g.replay)
        assert 0.9 < short / eager < 1.05, (short, eager)  # the graph saves only launch gaps
        lens.copy_(torch.tensor([3, 5], dtype=torch.int32))
        long = timed(g.replay)
        gemm = timed(lambda: a @ a)
        assert 3.5 < (long - gemm) / (short - gemm) < 4.5, (long, short, gemm)
        # values the emulator keeps (small ints) run at replay, not at capture, as on CUDA:
        # an in-graph counter, a staged H2D and a D2H into a host mailbox
        step = torch.zeros(1, dtype=torch.int64, device="cuda")
        staged, mailbox = torch.tensor([7]), torch.zeros(1, dtype=torch.int64)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            lens2 = torch.zeros(1, dtype=torch.int64, device="cuda")
            lens2.copy_(staged, non_blocking=True)
            step += 1
            mailbox.copy_(lens2 + step, non_blocking=True)
        assert step.item() == 0 and mailbox.tolist() == [0]
        staged.fill_(10); g2.replay(); g2.replay(); torch.cuda.synchronize()
        assert step.item() == 2 and mailbox.tolist() == [12], (step.item(), mailbox.tolist())
        try:
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                lens.sum().item()
            raise SystemExit("a host read during capture was accepted")
        except RuntimeError as e:
            assert "capturing" in str(e)
        print("ok")
    """)
    assert out.strip() == "ok"


@pytest.mark.parametrize("gpu", ["h100", "b200"])
def test_device_time_follows_the_cost_model_not_the_host(gpu):
    """A big GEMM takes its simulated time on the device, while the host that
    launched it waits only at a sync."""
    out = run_sim("""
        import os, time, torch
        from mstar_sim import clock, cost
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        outs = [a @ a for _ in range(20)]  # kept: freeing them is host time too
        end.record()
        launched = time.perf_counter() - t0
        end.synchronize()
        print(start.elapsed_time(end) / 1000, launched)
    """, gpu)
    device_s, _ = map(float, out.split())
    flops = 20 * 2 * 4096**3
    # the device time is the roofline estimate's order of magnitude for this card
    peak = {"h100": 989e12, "b200": 2250e12}[gpu]
    assert flops / peak < device_s < 20 * flops / peak


def test_timing_only_holds_shapes_not_values():
    """Nothing is computed: a 40 GB tensor costs no host memory
    but counts against the card; small integer tensors (what mstar plans with)
    keep their values; a value that was never computed reads as zeros."""
    out = run_sim("""
        import torch
        free0, total = torch.cuda.mem_get_info()
        big = torch.empty(10 * 2**30, dtype=torch.float32, device="cuda")
        free1, _ = torch.cuda.mem_get_info()
        assert total == torch.cuda.get_device_properties().total_memory
        assert free0 - free1 == 40 * 2**30
        a, b = torch.empty(1000, device="cuda"), torch.empty(3000, device="cuda")
        sa, sb = a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()
        assert sa and sb and abs(sa - sb) >= 4000  # distinct storages: mstar keys bytes by pointer
        assert a[10:].data_ptr() == sa + 40
        pos = torch.arange(5, device="cuda") + 1
        assert pos.tolist() == [1, 2, 3, 4, 5]  # small ints are real
        logits = torch.randn(4, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
        assert logits.argmax(-1).tolist() == [0, 0, 0, 0]  # never computed
        before = torch.cuda.mem_get_info()[0]
        del big
        assert torch.cuda.mem_get_info()[0] - before == 40 * 2**30
        on0 = torch.cuda.memory_allocated(0)
        other = torch.empty(2**20, device="cuda:1")  # each device counts its own
        assert torch.cuda.memory_allocated(1) == 4 * 2**20 and torch.cuda.memory_allocated(0) == on0
        assert torch.cuda.mem_get_info(1)[0] == total - 4 * 2**20
        print("ok")
    """)
    assert out.strip() == "ok"


def test_memory_tracking_lock_is_reentrant():
    """A tensor freed by a garbage collection that runs while the tracker holds
    its lock releases its storage on the same thread: a plain lock deadlocks."""
    from mstar_sim import device

    with device._track_lock:
        assert device._track_lock.acquire(blocking=False)
        device._track_lock.release()


def test_tensor_parallel_collectives_are_timed_not_sent():
    """Two ranks ask for NCCL, as mstar does: they get gloo, and an all-reduce of
    a value-less GPU tensor moves nothing but costs its NCCL time."""
    out = run_sim("""
        import os, torch, torch.distributed as dist, torch.multiprocessing as mp

        def rank(r):
            dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29573", rank=r,
                                    world_size=2, device_id=torch.device("cuda", r))
            torch.cuda.set_device(r)
            x = torch.empty(2**30, dtype=torch.bfloat16, device="cuda")  # 2 GiB, no data
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); dist.all_reduce(x); end.record(); end.synchronize()
            assert dist.get_backend() == "gloo" and start.elapsed_time(end) > 1.0, start.elapsed_time(end)
            dist.destroy_process_group()

        if __name__ == "__main__":
            mp.spawn(rank, nprocs=2)
            print("ok")
    """)
    assert out.strip().endswith("ok")


def test_fast_mode_skips_device_time_but_keeps_host_semantics():
    """--fast: a host read after a long kernel doesn't wait, graphs still capture
    and replay, and a host read during capture is still refused."""
    out = run_sim("""
        import time, torch
        a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        t0 = time.perf_counter()
        for _ in range(50):
            a @ a  # ~0.1 s of simulated H100 time
        torch.cuda.synchronize()
        assert time.perf_counter() - t0 < 0.05, time.perf_counter() - t0
        pos = torch.arange(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = pos + 1
        g.replay()
        assert out.tolist() == [1, 2, 3, 4]
        try:
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                pos.sum().item()
            raise SystemExit("a host read during capture was accepted")
        except RuntimeError as e:
            assert "capturing" in str(e)
        print("ok")
    """, MSTAR_SIM_FAST="1")
    assert out.strip() == "ok"


def test_every_cuda_seam_mstar_reaches_is_emulated():
    """The contract with mstar: every CUDA entry point, kernel-library name, ctypes
    library and torch.ops namespace mstar's source reaches is a stand-in, or is
    listed with a reason in the table of the module that owns that seam. A new one
    fails here, in the mstar change that adds it."""
    out = run_sim("""
        import ast, importlib, pathlib, re, torch
        from mstar_sim import cuda, device, libcuda, patches

        assert not patches.SKIPPED, f"patches no longer apply to this mstar: {patches.SKIPPED}"

        def dotted(node):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None

        found = {}
        for path in pathlib.Path(%r, "mstar").rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            uses_ctypes = "ctypes" in path.read_text()
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Attribute) and (d := dotted(node)):
                    p = d.split(".")
                    if p[:2] in (["torch", "cuda"], ["torch", "accelerator"]) and len(p) > 2:
                        names.append(".".join(p[:3]))
                    elif p[:2] == ["torch", "_C"] and len(p) > 2 and p[2].startswith("_cuda"):
                        names.append(".".join(p[:3]))
                    elif p[:2] == ["torch", "ops"] and len(p) > 2:
                        names.append("ops:" + p[2])
                    elif p[0] == "flashinfer" and len(p) > 1:
                        names.append(d)
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    top = node.module.split(".")[0]
                    if top == "flashinfer":
                        names += [f"{node.module}.{a.name}" for a in node.names]
                    elif top == "cuda":
                        names.append("module:cuda")
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.split(".")[0] == "cuda":
                            names.append("module:cuda")
                        elif a.name.startswith("flashinfer."):
                            names.append(a.name)
                elif uses_ctypes and isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names += ["lib:" + m for m in re.findall(r"\\b(lib[a-z0-9_]+?)(?:\\.so|$)", node.value)]
                for n in names:
                    found.setdefault(n, f"{path}:{node.lineno}")

        import flashinfer  # under the emulator, the whole package is the stand-in
        assert "fake_libs" in flashinfer.__file__, flashinfer.__file__

        def resolves(name):
            obj = importlib.import_module(name.split(".")[0])
            for part in name.split(".")[1:]:
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    try:
                        obj = importlib.import_module(f"{obj.__name__}.{part}")
                    except ImportError:
                        return None
            return obj

        missing = []
        for name, where in sorted(found.items()):
            if name.startswith("ops:"):
                ok = name[4:] in device.OPS_NAMESPACES
            elif name.startswith("module:"):
                ok = name[7:] in device.ABSENT_MODULES
            elif name.startswith("lib:"):
                ok = name[4:] in libcuda.LIBRARIES
            elif name.startswith("flashinfer"):
                try:
                    ok = resolves(name) is not None
                except (AttributeError, ImportError):
                    ok = False
            else:
                obj = resolves(name)
                ok = name in cuda.PASSTHROUGH or (
                    obj is not None and (getattr(obj, "__module__", "") or "").startswith("mstar_sim"))
            if not ok:
                missing.append(f"{name} (first at {where})")
        assert not missing, (
            "mstar now reaches CUDA through names the emulator doesn't cover; add a stand-in in "
            "mstar_sim/ (or a row with the reason in that module's table):\\n  " + "\\n  ".join(missing))
        print("ok", len(found))
    """ % ROOT)
    assert out.startswith("ok")
