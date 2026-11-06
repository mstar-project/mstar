"""The emulated GPU keeps the CUDA behaviours mstar's code depends on.

Each case runs in a fresh interpreter with the emulator installed, as
``python -m mstar_sim`` would start one.
"""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def caller_env() -> dict[str, str]:
    """The test run's environment, but for a mask of its own: a case sees all 8 emulated GPUs."""
    return {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}


def start_sim(code: str, gpu: str = "h100", **extra_env: str) -> subprocess.CompletedProcess:
    env = dict(caller_env(), MSTAR_SIM="1", MSTAR_SIM_GPU=gpu, **extra_env,
               PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "mstar_sim", "_site"), ROOT,
                                           os.environ.get("PYTHONPATH", "")]))
    with tempfile.TemporaryDirectory() as tmp:  # a file, so spawned ranks can import it
        script = os.path.join(tmp, "case.py")
        with open(script, "w") as f:
            f.write(textwrap.dedent(code))
        return subprocess.run([sys.executable, script], env=env, capture_output=True, text=True,
                              timeout=300, check=False)


def run_sim(code: str, gpu: str = "h100", **extra_env: str) -> str:
    proc = start_sim(code, gpu, **extra_env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


def test_cuda_semantics():
    out = run_sim("""
        import itertools, subprocess, sys, torch, torch.distributed as dist
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
        like = torch.tensor(1.0).to(dev), torch.tensor(1.0).to(tensor=dev, non_blocking=True)  # as torchaudio does
        assert all(t.is_cuda and t.dtype == torch.int32 for t in like), like
        staged = torch.zeros_like(dev, device="cpu", pin_memory=True)  # a host staging buffer
        assert type(staged) is torch.Tensor and staged.device.type == "cpu"
        try:
            x + torch.ones(3, 8)
            raise SystemExit("a cuda/cpu mix was accepted")
        except RuntimeError as e:
            assert "same device" in str(e)
        assert x.to("cuda:1").device == torch.device("cuda", 1)  # a peer copy lands on the peer
        # a device argument beside a cuda operand wins over the operand's
        peer = x.to("cuda:1").device
        assert torch.zeros_like(x, device="cuda:1").device == x.new_zeros(2, device="cuda:1").device == peer
        side = torch.cuda.Stream()
        for past in (lambda: torch.zeros(2, device="cuda:8"), lambda: x.to("cuda:8"),  # 8 emulated GPUs
                     lambda: torch.cuda.set_device(8), lambda: torch.cuda.Stream(8),
                     lambda: torch.zeros_like(x, device="cuda:8"), lambda: x.new_zeros(2, device="cuda:8"),
                     lambda: torch.cuda.synchronize(8), lambda: torch.cuda.current_stream(8),
                     lambda: torch.cuda.ExternalStream(side.cuda_stream, device="cuda:8"),
                     lambda: torch._C._cuda_getCurrentRawStream(8), lambda: dist.barrier(device_ids=[8]),
                     lambda: dist.init_process_group("nccl", device_id=torch.device("cuda", 8))):
            try:
                past()
                raise SystemExit("an index past the emulated devices was accepted")
            except torch.AcceleratorError:
                pass
        torch.cuda.set_device(-1)  # a negative index is ignored, as torch ignores it
        assert torch.cuda.current_device() == 0 and torch.zeros(1, device="cuda").device == torch.device("cuda", 0)
        properties = torch.cuda.get_device_name, torch.cuda.get_device_capability, torch.cuda.get_device_properties
        for past, index in itertools.product(properties, (8, -1)):
            try:
                past(index)
                raise SystemExit("a past index's properties were read")
            except AssertionError as e:
                assert str(e) == "Invalid device id"
        for figure in (torch.cuda.memory_allocated, torch.cuda.memory_reserved, torch.cuda.max_memory_reserved):
            try:
                figure(-1)
                raise SystemExit("a negative index's memory was read")
            except RuntimeError as e:
                assert "Invalid device argument -1" in str(e)
        assert torch.cuda.mem_get_info(-1) == torch.cuda.mem_get_info()  # CUDAGuard(-1) stays on the current device
        from mstar_sim.device import _rebuild
        try:
            _rebuild(torch.ones(1), 8)
            raise SystemExit("a tensor from a device past the emulated ones was rebuilt")
        except torch.AcceleratorError:
            pass
        try:
            x + x.to("cuda:1")
            raise SystemExit("a cuda:0/cuda:1 mix was accepted")
        except RuntimeError as e:
            assert "cuda:0 and cuda:1" in str(e)
        assert (x + torch.tensor(1.0)).is_cuda          # CPU scalars are fine
        assert x[torch.tensor([0, 2])].shape == (2, 8)  # so is a CPU index
        # indices CUDA accepts raise no device-side assert: negative ones where the kernel wraps
        # them, and an index after a mask, which counts against the dimension after the mask's
        ids = torch.arange(6, device="cuda")
        assert torch.take(ids, torch.tensor([-1], device="cuda")).tolist() == [5]
        assert ids.clone().index_fill_(0, torch.tensor([-1], device="cuda"), 9).tolist()[-1] == 9
        cube = torch.zeros(2, 3, 7, dtype=torch.long, device="cuda")
        assert cube[torch.ones(2, 3, dtype=torch.bool, device="cuda"), torch.tensor([5], device="cuda")].shape == (6,)
        torch.cuda.synchronize()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert m(x).dtype == torch.bfloat16
        # small host floats keep their values; beside a never-computed operand they join it
        v, g = torch.empty(512, 256, 3, device="cuda"), torch.randn(512, 1, 1).cuda()
        assert torch._weight_norm(v, g, 0).shape == v.shape  # a composite's same-device check
        assert torch.stack([x.sum(), torch.ones((), device="cuda")]).shape == (2,)
        # a block-scaled fp8 GEMM takes a partial last 128-block of N; no other bad scale
        F, fp8 = torch.nn.functional, torch.float8_e4m3fn
        a, w = torch.empty(128, 256, dtype=fp8, device="cuda"), torch.empty(208, 256, dtype=fp8, device="cuda").t()
        rows, blocks = F.ScalingType.BlockWise1x128.value, F.ScalingType.BlockWise128x128.value
        none = F.SwizzleType.NO_SWIZZLE.value
        a_scale = torch.empty_strided((128, 2), (1, 128), dtype=torch.float32, device="cuda")
        w_scale = torch.empty_strided((4, 2), (1, 4), dtype=torch.float32, device="cuda")

        def scaled_mm(scale):
            return torch._scaled_mm_v2(a, w, [a_scale], [rows], [none], [scale], [blocks], [none], None, torch.bfloat16)

        assert scaled_mm(w_scale).shape == (128, 208)
        try:
            scaled_mm(w_scale.t())
            raise SystemExit("a transposed scale was accepted")
        except RuntimeError as e:
            assert "blockwise scaling" in str(e)
        rowwise = F.ScalingType.RowWise.value
        try:  # a recipe with no 128-blocks takes no padded scale
            torch._scaled_mm_v2(a, w, [torch.empty(128, 1, device="cuda")], [rowwise], [none],
                                [torch.empty(256, device="cuda")], [rowwise], [none], None, torch.bfloat16)
            raise SystemExit("a scale for a rounded-up N was accepted")
        except RuntimeError as e:
            assert "208 elements" in str(e)
        # mstar's custom ops run on the emulated GPU, kept operands included (their torch.empty
        # scratch is a cuda buffer, not host garbage), and a kept storage shares like any other
        @torch.library.custom_op("mstar_sim_test::plus_one", mutates_args=())
        def plus_one(t: torch.Tensor) -> torch.Tensor:
            assert t.is_cuda
            return t + 1
        assert torch.ops.mstar_sim_test.plus_one(torch.ones(2, device="cuda")).tolist() == [2.0, 2.0]
        shared = torch.ones(2, device="cuda")
        share = shared.untyped_storage()._share_cuda_()
        code = (f"import torch; s = torch.UntypedStorage._new_shared_cuda(*{share!r}); "
                "t = torch.empty(0).set_(s, 0, (2,), (1,)); t[1] = 5")
        subprocess.run([sys.executable, "-c", code], check=True)
        assert shared.tolist() == [1, 5]
        # two processes' value-less KV caches, opened as mstar's KV transfer does (one twice)
        import ast
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        exported = [ast.literal_eval(subprocess.run(
            [sys.executable, "-c", f"import torch; kv = torch.empty(1 << 20, device='cuda:{i}'); "
                                   "print(kv.untyped_storage()._share_cuda_())"], check=True, capture_output=True,
            text=True).stdout)
            for i in (0, 1)]
        opened = [rebuild_cuda_tensor(torch.Tensor, (1 << 20,), (1,), 0, torch.UntypedStorage, torch.float32,
                                      *exported[i][:4], False, *exported[i][4:]) for i in (0, 0, 1)]
        assert [str(t.device) for t in opened] == ["cuda:0", "cuda:0", "cuda:1"], opened
        assert opened[0].data_ptr() == opened[1].data_ptr() != opened[2].data_ptr()
        torch.empty(1 << 20, device="cuda:1").copy_(opened[2])
        torch.cuda.manual_seed_all(3)  # dynamo saves and restores this state
        assert [gen.initial_seed() for gen in torch.cuda.default_generators] == [3] * torch.cuda.device_count()
        torch.cuda.set_rng_state(torch.cuda.get_rng_state(1), 1)
        from torch._dynamo.testing import CompileCounter
        counter = CompileCounter()  # a test's own backend compiles for real, without tracing the emulator
        for layer in [torch.nn.Linear(4, 4) for _ in range(3)]:
            layer.forward = torch.compile(lambda x, f=layer.forward: f(x).float().to(x.dtype),
                                          backend=counter, dynamic=True)
            layer(torch.randn(2, 4))
        assert counter.frame_count == 1, counter.frame_count
        print("ok")
    """)
    assert out.strip() == "ok"


def test_a_parameter_given_another_gpus_data_is_on_that_gpu():
    run_sim("""
        import torch
        p = torch.nn.Parameter(torch.ones(2, device="cuda:0"))
        p.data = torch.ones(2, device="cuda:1")
        assert p.device.index == 1 and (p + torch.ones(2, device="cuda:1")).device.index == 1
    """)


def test_a_mask_narrows_the_launchers_gpus_and_leaves_these_tests_alone(monkeypatch):
    """``CUDA_VISIBLE_DEVICES`` can hide emulated GPUs but not add any, and the caller's own
    doesn't reach a case that doesn't ask for it."""
    count = "import torch; print(torch.cuda.device_count())"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert run_sim(count).strip() == "8"
    assert run_sim(count, CUDA_VISIBLE_DEVICES="0,1,2,3", MSTAR_SIM_NUM_GPUS="2").strip() == "2"
    assert run_sim(count, CUDA_VISIBLE_DEVICES="0", MSTAR_SIM_NUM_GPUS="2").strip() == "1"


@pytest.mark.parametrize("command, table", [
    (["-m", "mstar.cli.main", "serve", "m", "--port", "9000"], "9000"),
    (["-m", "mstar.cli.main", "serve", "m"], "8000"),
    (["-m", "mstar.api_server.entrypoint", "--config", "c.yaml"], "8000"),  # the README's: no port in its argv
    (["-m", "benchmark.runner", "--url=http://localhost:9000/v1"], "9000"),
    (["-m", "x", "--dataset-url", "http://h:443/d", "--port=9000"], "9000"),  # no other flag's URL names it
    (["-m", "x", "--url", "http://localhost/v1"], "the launcher's pid"),  # no port: the launch is its own
    (["my_script.py"], "the launcher's pid"),
    (["-m", "pytest", "-n", "4"], None),  # a test run's processes aren't one deployment
])
def test_a_launch_names_its_processes_clock_table(command, table):
    from mstar_sim.__main__ import deployment
    assert deployment(command) == (str(os.getpid()) if table == "the launcher's pid" else table)


def test_graph_replay_costs_its_kernels_and_reprices_value_dependent_ones():
    """A replay costs what its kernels cost eagerly; a kernel priced from the
    values in its arguments (attention over the latest plan) is priced again
    on every replay; a host read during capture raises, as on CUDA."""
    out = run_sim("""
        import torch
        from mstar_sim import clock

        # priced from the lengths in `lens` (small ints keep values), as attention is
        @clock.kernel_fn("test.rows", cost=lambda lens, out: lambda: (0.0, 1e9 * float(lens._data.sum())))
        def rows(lens, out):
            return out

        def timed(fn, queued=0.1):
            # device time of fn's launches, queued behind other work: on an idle stream the
            # events also time a host stalled by load, as they would on a GPU
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            clock.charge(queued)
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
        # a node costs no eager launch gap, though its op was priced eagerly first (a warmup
        # forward); and a composite's later calls charge what its first did (rms_norm: 8 kernels)
        for warm, cold, op in ((a[:2048], a.view(8192, 2048)[:4096], lambda x: x + x),  # cold: same bytes
                               (a[:512], a[512:1024], lambda x: torch.rms_norm(x, (4096,)))):
            first, again = timed(lambda: op(warm)), timed(lambda: op(warm))
            assert abs(again - first) < 1e-9, (again, first)  # (a launch gap is 1e-6)
            graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
            for graph, x in zip(graphs, (warm, cold)):
                with torch.cuda.graph(graph):
                    op(x)
            nodes = [timed(graph.replay) for graph in graphs]
            assert abs(nodes[0] - nodes[1]) < 1e-9, nodes
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
            cast = lens2.to(torch.int32)
        assert step.item() == 0 and mailbox.tolist() == [0]
        staged.fill_(10); g2.replay(); g2.replay(); torch.cuda.synchronize()
        assert step.item() == 2 and mailbox.tolist() == [12], (step.item(), mailbox.tolist())
        assert cast.tolist() == [10], cast.tolist()
        # so does a reshape that has to copy: a view function only when it can be one
        source = torch.arange(3, device="cuda")
        g7 = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g7):  # (the engine's: composites arrive whole)
            copied = source[:, None].expand(3, 2).reshape(-1)
            packed = source[:, None].expand(3, 2).contiguous()
        source.add_(1); g7.replay()
        assert copied.tolist() == [1, 1, 2, 2, 3, 3] and packed.tolist() == [[1, 1], [2, 2], [3, 3]], packed.tolist()
        # a result too large to keep is value-less, so a replay has nothing to re-run for it
        wide, g9 = torch.ones(4096, device="cuda"), torch.cuda.CUDAGraph()
        with torch.cuda.graph(g9):
            outer = wide[:, None] * wide[None, :]
        assert not [run for _, run in g9.tape.nodes if run]
        # a stand-in kernel's work on kept values re-runs at each replay too (timed as under --fast),
        # and capture writes nothing: here the top-k transform's slots from the latest page table
        import flashinfer
        scores = torch.empty(2, 64, device="cuda")
        table = torch.arange(16, dtype=torch.int32, device="cuda").view(2, 8)
        lengths, rows = torch.tensor([3, 1], device="cuda"), torch.arange(2, device="cuda")
        raw = torch.full((2, 4), 7, dtype=torch.int32, device="cuda")
        g6 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g6):
            slots = flashinfer.top_k_page_table_transform(scores, table, lengths, 4, row_to_batch=rows,
                                                          out_raw_indices=raw)
        assert raw.tolist() == [[7] * 4] * 2, raw.tolist()
        table.add_(100); lengths.copy_(torch.tensor([1, 2]))
        g6.replay()
        assert slots.tolist() == [[100, -1, -1, -1], [108, 109, -1, -1]], slots.tolist()
        assert raw.tolist() == [[0, -1, -1, -1], [0, 1, -1, -1]], raw.tolist()
        # the emulator's own work at a replay (re-running ops on kept values) is not host time
        work = lambda *_: sum(range(10_000_000))  # ~0.2 s of the emulator's own CPU
        g5 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g5):
            clock.on_replay(work)
            a @ a
        took = timed(g5.replay, queued=0.0)  # idle: a late launch would show
        assert took < 0.05, took
        # nor is what it does around a stand-in kernel
        from mstar_sim import device
        count_storages, device.count_storages = device.count_storages, work
        before = clock.now()
        clock.kernel_fn("test.idle")(lambda: None)()
        assert clock.now() - before < 0.05, clock.now() - before
        device.count_storages = count_storages
        try:
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                lens.sum().item()
            raise SystemExit("a host read during capture was accepted")
        except RuntimeError as e:
            assert "capturing" in str(e)
        # a captured kernel's index bounds are checked at replay, on that step's indices
        table, ids = torch.empty(50, 8, device="cuda"), torch.tensor([60, 10], device="cuda")
        g3 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g3):
            torch.nn.functional.embedding((ids - 50).clamp_(0, 49), table)  # a TP vocab shard's clamp
            torch.where(ids > 20, torch.arange(2, device="cuda").expand(2, 2), -1)  # an expanded operand
        g3.replay()
        weight, g8 = torch.nn.Parameter(torch.zeros(4, device="cuda")), torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g8):
            weight[:2].add_(1)
        g8.replay()  # autograd doesn't see a replay, whatever the caller's grad mode
        assert weight.tolist() == [1, 1, 0, 0], weight.tolist()
        # capture runs nothing: an index that is out of range only before an in-place clamp
        # is fine, and an op whose output shape depends on values is refused, as on CUDA
        table_ids, wide = torch.arange(50, device="cuda"), torch.tensor([200, 10], device="cuda")
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            table_ids[(wide - 50).clamp_(0, 49)]
            torch.rms_norm(torch.ones(2, 4, device="cuda", dtype=torch.bfloat16), (4,))  # casts inside
            torch._assert(wide.max() < 1000, "a check torch.compile leaves out of a capture on a GPU")
            torch.nn.functional.one_hot(table_ids, 50)  # its range is the scatter's device assert
            # so are synchronous copies from host memory: a blocking H2D copy, a tensor from host data
            for sync in (lambda: torch.nonzero(ids), lambda: ids.copy_(torch.tensor([1, 2])),
                         lambda: torch.tensor([1, 2], device="cuda"), lambda: torch.tensor([1, 2]).cuda()):
                try:
                    sync()
                    raise SystemExit("a synchronizing call was captured")
                except RuntimeError as e:
                    assert "capturing" in str(e)
        g4, bad = torch.cuda.CUDAGraph(), torch.tensor([1, 2], device="cuda")
        with torch.cuda.graph(g4):
            torch.nn.functional.embedding(bad + 0, table)
        bad.copy_(torch.tensor([3, 77]))
        g4.replay()  # asynchronous, as on CUDA: the next sync reports it
        try:
            torch.cuda.synchronize()
            raise SystemExit("an index past the table replayed")
        except RuntimeError as e:
            assert "device-side assert" in str(e)
        print("ok")
    """)
    assert out.strip() == "ok"


def test_flashinfers_planner_runs_short_of_a_small_workspace():
    """A plan lays its split-KV partials out in the float workspace as FlashInfer's planner
    does, so one too small for them raises FlashInfer's overflow, and one that holds what
    ``workspace_size`` counts plans."""
    run_sim("""
        import torch, flashinfer
        from flashinfer.prefill import get_batch_prefill_module
        i32 = dict(dtype=torch.int32, device="cuda")
        workspace = torch.empty(1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", use_cuda_graph=True, qo_indptr_buf=torch.zeros(3, **i32),
            paged_kv_indptr_buf=torch.zeros(3, **i32), paged_kv_indices_buf=torch.zeros(8, **i32),
            paged_kv_last_page_len_buf=torch.ones(2, **i32), backend="fa2")
        qo, kv = torch.tensor([0, 16, 32], dtype=torch.int32), torch.tensor([0, 2, 4], dtype=torch.int32)

        def plan(**extras):
            wrapper.plan(qo, kv, torch.arange(4, dtype=torch.int32), torch.tensor([16, 16], dtype=torch.int32),
                         8, 8, 64, 16, causal=True, q_data_type=torch.bfloat16, **extras)

        need, _ = get_batch_prefill_module("fa2").workspace_size(
            workspace, qo, kv, None, 32, 2, 8, 8, 16, True, 64, 64, True, -1, -1, False, 0, 0)
        for size, short_of in ((1024, "batch_prefill_tmp_v"), (need - 1, "batch_prefill_tmp_s")):
            workspace.data = torch.empty(size, dtype=torch.uint8, device="cuda")
            try:
                plan()
                raise SystemExit(f"a {size}-byte workspace held a plan that needs {need}")
            except RuntimeError as e:
                assert f"allocating memory for {short_of}" in str(e), e

        def unplanned():  # a failed plan is no plan
            try:
                wrapper.run(torch.zeros(32, 8, 64, device="cuda"), None)
                raise SystemExit("run() after a failed plan")
            except RuntimeError as e:
                assert "plan() must be called" in str(e), e

        unplanned()
        plan(disable_split_kv=True)  # no split, no partials: any workspace holds it
        ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend="fa2")
        one_chunk = torch.tensor([0, 4096, 8192], dtype=torch.int32)
        ragged.plan(one_chunk, one_chunk, 8, 8, 64, q_data_type=torch.bfloat16, disable_split_kv=True)
        try:  # a split it fixes writes partials: too many for this workspace
            ragged.plan(one_chunk, one_chunk, 8, 8, 64, q_data_type=torch.bfloat16, fixed_split_size=128)
            raise SystemExit("a fixed split kept no partials")
        except RuntimeError as e:
            assert "batch_prefill_tmp_v" in str(e), e
            try:  # and leaves the wrapper unplanned
                ragged.run(torch.zeros(8192, 8, 64, device="cuda"), None, None)
                raise SystemExit("ragged run() after a failed plan")
            except RuntimeError as e:
                assert "plan() must be called" in str(e), e
        workspace.data = torch.empty(need, dtype=torch.uint8, device="cuda")
        plan()
        workspace.data = torch.empty(need - 1, dtype=torch.uint8, device="cuda")  # and a failed re-plan undoes one
        try:
            plan()
            raise SystemExit("a short workspace re-planned")
        except RuntimeError:
            unplanned()
        # auto takes fa3, which keeps no partials, only where its kernels serve the plan
        rows, pages = torch.tensor([0, 16, 32], dtype=torch.int32), torch.tensor([0, 2, 4], dtype=torch.int32)
        for head_dim, kv_dtype, overflows in ((128, torch.bfloat16, False), (80, torch.bfloat16, True),
                                              (128, torch.float8_e4m3fn, True)):
            auto = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(1024, dtype=torch.uint8, device="cuda"), "NHD", use_cuda_graph=True,
                qo_indptr_buf=torch.zeros(3, **i32), paged_kv_indptr_buf=torch.zeros(3, **i32),
                paged_kv_indices_buf=torch.zeros(8, **i32), paged_kv_last_page_len_buf=torch.ones(2, **i32))
            try:
                auto.plan(rows, pages, torch.arange(4, dtype=torch.int32), torch.tensor([16, 16], dtype=torch.int32),
                          8, 8, head_dim, 16, causal=True, q_data_type=torch.bfloat16, kv_data_type=kv_dtype)
                assert not overflows, (head_dim, kv_dtype)
            except RuntimeError as e:
                assert overflows and "allocating memory for batch_prefill_tmp_v" in str(e), e
        # tensor-core decode's auto is fa2, or the rule's answer when fp8 is in play
        for dtype, head_dim, expected in ((torch.bfloat16, 128, "fa2"), (torch.float8_e4m3fn, 80, "fa2"),
                                          (torch.float8_e4m3fn, 128, "fa3")):
            decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                torch.empty(1 << 26, dtype=torch.uint8, device="cuda"), "NHD", use_tensor_cores=True)
            decode.plan(torch.tensor([0, 1], **i32), torch.zeros(1, **i32), torch.tensor([3], **i32), 8, 8, head_dim,
                        16, q_data_type=dtype, kv_data_type=dtype)
            assert decode._backend == expected, decode._backend
        # fa3's work arrays are int, one set per head under a graph: many heads overrun the int workspace
        fa3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", use_cuda_graph=True, qo_indptr_buf=torch.zeros(2, **i32),
            paged_kv_indptr_buf=torch.zeros(2, **i32), paged_kv_indices_buf=torch.zeros(1, **i32),
            paged_kv_last_page_len_buf=torch.ones(1, **i32), backend="fa3")
        for heads, overflows in ((8, False), (80, True)):
            try:
                fa3.plan(torch.tensor([0, 500_000], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32),
                         torch.zeros(1, dtype=torch.int32), torch.tensor([16], dtype=torch.int32), heads, 8, 128, 16,
                         q_data_type=torch.bfloat16)
                assert not overflows, heads
            except RuntimeError as e:
                assert overflows and "batch_prefill_sm90" in str(e), e
        cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=False)
        try:
            cores.plan(torch.tensor([0, 1], **i32), torch.zeros(1, **i32), torch.tensor([3], **i32), 8, 8, 128, 16,
                       fixed_split_size=4)
            raise SystemExit("CUDA-core decode took a fixed split")
        except ValueError as e:
            assert "tensor core decode" in str(e), e
        # a plan's Python is the emulator's time, not the host's: FlashInfer's own is a C++ loop
        import time

        def shown(call):  # the least share of call's wall time the clock showed, over a few tries:
            # on a loaded box a preemption the clock can't place raises one
            shares = []
            for _ in range(5):
                wall, began = time.clock_gettime(time.CLOCK_MONOTONIC), time.perf_counter()
                call()
                shares.append((time.perf_counter() - began) / (time.clock_gettime(time.CLOCK_MONOTONIC) - wall))
            return min(shares)

        batch = torch.arange(257, dtype=torch.int32)
        share = shown(lambda: get_batch_prefill_module("fa2").plan(
            torch.empty(1 << 30, dtype=torch.uint8, device="cuda"), None, None, batch, batch * 64,
            torch.full((256,), 1024, dtype=torch.int32), 256, 256, 8, 8, 16, False, 64, 64, True, -1, -1, False, 0, 0))
        assert share < 1 / 2, share
        rows = torch.arange(4097, dtype=torch.int32)  # a ragged wrapper prices in its own plan
        wide = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(1 << 30, dtype=torch.uint8, device="cuda"), "NHD", backend="fa2")
        share = shown(lambda: wide.plan(rows, rows * 64, 8, 8, 64, causal=True, q_data_type=torch.bfloat16,
                                        disable_split_kv=True))
        assert share < 1 / 10, share
        # what FlashInfer's plan refuses: a group size with no kernel, a negative length, and
        # several queries a row where the kernels take one
        one_row = torch.tensor([0, 1], **i32), torch.zeros(1, **i32), torch.tensor([3], **i32)
        backwards = torch.tensor([0, 8, 4], dtype=torch.int32)
        refused = ((lambda: cores.plan(*one_row, 7, 1, 128, 16), RuntimeError, "Unsupported group_size: 7"),
                   (lambda: get_batch_prefill_module("fa2").plan(
                       workspace, None, None, backwards, kv, torch.tensor([16, 16], dtype=torch.int32), 8, 2, 8, 8, 16,
                       False, 64, 64, True, -1, -1, False, 0, 0), RuntimeError, "qo_indptr[2]4 - qo_indptr[1]8"),
                   (lambda: cores.plan(*one_row, 8, 8, 128, 16, q_len_per_req=2), ValueError, "requires tensor-core"),
                   (lambda: cores.plan(*one_row, 8, 8, 512, 16), NotImplementedError, "no stand-in yet"),
                   (lambda: decode.plan(*one_row, 8, 8, 128, 16, q_data_type=dtype, q_len_per_req=2),
                    NotImplementedError, "fa2 tensor-core"))
        for call, kind, text in refused:
            try:
                call()
                raise SystemExit(f"nothing refused it: {text}")
            except kind as e:
                assert text in str(e), e
    """)


def test_flashinfers_planner_port_answers_as_flashinfers_own():
    """The stand-in's planner port answers every case FlashInfer's own planner answered on the
    device properties the golden stores (``tests/planner_golden.json``, from ``python -m
    mstar_sim.planner_golden``): the same workspace bytes, the same overflow one byte short of
    each, the same refusals; and never less than it answered for a kernel of a given register
    count, which the port doesn't take. The golden is of the stand-in's FlashInfer version."""
    golden = os.path.join(ROOT, "mstar_sim", "tests", "planner_golden.json")
    out = run_sim(f"""
        import json, types, torch, flashinfer
        from flashinfer._paged import allocate, decode_allocations, fa3_allocations, prefill_allocations
        from flashinfer.mla import mla_allocations
        golden = json.load(open({golden!r}))
        assert golden["flashinfer"] == flashinfer.__version__, (golden["flashinfer"], flashinfer.__version__)
        kv_dtype = {{1: torch.float8_e4m3fn, 2: torch.bfloat16}}

        def allocations(c):
            batch = len(c["kv"]) - 1
            if c["planner"] == "fa2":
                return prefill_allocations(c["qo"], c["kv"], c["rows"], batch, *c["heads"], c["page"], c["graph"],
                                           *c["hd"], c["window"], c["fixed"], c["disable"], c["colocated"],
                                           c["uniform"], kv_bytes=c["kv_bytes"])
            if c["planner"] == "fa3":
                return fa3_allocations(c["qo"], c["rows"], batch, c["heads"][0], c["graph"], c["hd"][1])
            if c["planner"] == "mla":
                return mla_allocations(c["qo"], batch, c["heads"], c["hd_o"])
            return decode_allocations(c["kv"], batch, *c["heads"], c["page"], c["graph"], c["head_dim"],
                                      kv_dtype[c["kv_bytes"]])

        def short_of(out, float_bytes, int_bytes):
            try:
                allocate(out, float_bytes, int_bytes)
                return "no overflow"
            except RuntimeError as e:
                return str(e)

        wrong = []
        for gpu, props in golden["devices"].items():
            torch.cuda.get_device_properties = lambda device=None, p=props: types.SimpleNamespace(**p)
            for c in golden["cases"]:
                want = c["answers"][gpu]
                try:
                    out = allocations(c)
                except RuntimeError as e:
                    got = {{"error": str(e)}}
                else:
                    got = dict(zip(("float", "int"), allocate(out, None, None)))
                    if c.get("regs"):  # the port sizes without registers, which only shrink the grid
                        if got["float"] < want["float"] or got["int"] < want["int"]:
                            wrong.append(f"{{c['name']}} on {{gpu}}: port {{got}} under FlashInfer's {{want}}")
                        continue
                    if got["float"]:
                        got["float_short"] = short_of(out, got["float"] - 1, None)
                    if got["int"]:
                        got["int_short"] = short_of(out, None, got["int"] - 1)
                if "error" in want and "error" in got and want["error"] in got["error"]:
                    continue
                if got != want:
                    wrong.append(f"{{c['name']}} on {{gpu}} ({{c['pins']}}): port {{got}}, FlashInfer {{want}}")
        print(len(wrong), "wrong")
        print("\\n".join(wrong[:20]))
    """)
    assert out.startswith("0 wrong"), out
    # the golden's properties are its own: this emulated Ada SM holds its 1536 threads, 12 of 128,
    # and a module plans by its own KV size: 1-byte KV fits the tile 2-byte KV doesn't on 99 KiB
    run_sim("""
        import torch
        from flashinfer._paged import blocks_per_sm
        from flashinfer.prefill import get_batch_prefill_module
        assert blocks_per_sm(128, 0) == 12, blocks_per_sm(128, 0)
        qo, kv = torch.tensor([0, 1, 2], dtype=torch.int32), torch.tensor([0, 50, 100], dtype=torch.int32)
        floats = [get_batch_prefill_module("fa2", torch.bfloat16, dtype).workspace_size(
            None, qo, kv, None, 2, 2, 16, 16, 16, False, 448, 256, True, -1, -1, False, 0, 0)[0]
            for dtype in (torch.bfloat16, torch.float8_e4m3fn)]
        assert floats[0] == 4 * floats[1], floats
    """, gpu="l40s")


def test_a_jit_built_cuda_extension_registers_capturable_ops(tmp_path):
    """``cpp_extension.load`` of CUDA sources registers their ``TORCH_LIBRARY`` ops as
    kernels that write their mutable arguments, which a graph captures, whichever of its
    sources defines them; an op whose schema comes from C++ fails the build, so mstar keeps
    its fallback."""
    (tmp_path / "op.cu").write_text('''
        TORCH_LIBRARY(mstar_sim_jit, m) {
          m.def("fill(Tensor x, int n, "
                "Tensor! out) -> ()");  // a schema in literals, as mstar's moe_align_block_size
        }''')
    (tmp_path / "inferred.cu").write_text("TORCH_LIBRARY(mstar_sim_inferred, m) { m.def(\"f\", &f); }")
    (tmp_path / "kernel.cu").write_text("__global__ void fill_kernel(float* x) {}")  # the ops' kernels, apart
    run_sim(f"""
        import torch
        from torch.utils.cpp_extension import load
        from mstar_sim.device import never_computed
        load(name="jit", sources=["{tmp_path}/op.cu", "{tmp_path}/kernel.cu"], is_python_module=False)
        x, out = torch.ones(4, device="cuda"), torch.zeros(8, dtype=torch.int32, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.ops.mstar_sim_jit.fill(x, 3, out)
        assert not never_computed(out._data)  # capture runs nothing
        graph.replay()
        assert never_computed(out._data) and not never_computed(x._data)
        try:
            load(name="inferred", sources=["{tmp_path}/inferred.cu"], is_python_module=False)
            raise SystemExit("a schema inferred from C++ was accepted")
        except RuntimeError as e:
            assert "Error building extension" in str(e)
    """)


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
    # a host read is a D2H copy: it takes device time even on an idle stream (slowed down
    # 1000x to stand clear of the host's own time between the events)
    out = run_sim("""
        import torch
        from mstar_sim import cost
        small, start, end = torch.ones(1, device="cuda"), torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        start.record(); small.item(); end.record(); end.synchronize()
        assert start.elapsed_time(end) / 1000 > 1000 * cost.copy_seconds(4, "d2h"), start.elapsed_time(end)
        # a captured copy is scaled once, at its replay
        host, graph = torch.ones(4), torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            host.to("cuda", non_blocking=True)
        start.record(); graph.replay(); end.record(); end.synchronize()
        once = 1000 * (cost.copy_seconds(16, "h2d") + cost.graph_node_seconds() + cost.graph_launch_seconds())
        # (scaled twice it is 1000x that: the bound leaves room for a host stalled by load)
        assert start.elapsed_time(end) / 1000 < 100 * once, (start.elapsed_time(end), once)
    """, gpu, MSTAR_SIM_TIME_SCALE="1000")
    flops = 20 * 2 * 4096**3
    # the device time is the roofline estimate's order of magnitude for this card
    peak = {"h100": 989e12, "b200": 2250e12}[gpu]
    assert flops / peak < device_s < 20 * flops / peak


@pytest.mark.parametrize("gpu", ["h100", "b200"])
def test_a_wait_on_the_gpu_threads_step_spans_its_device_time_not_the_emulators(gpu):
    """A worker's main thread waits for the GPU thread's step, then syncs on its event: the
    span is the step's device time, however long the emulator took to launch it, with the
    parent watch asleep beside them as in a worker. So a faster GPU shows, host-bound box or
    not."""
    out = run_sim("""
        import multiprocessing, time, torch
        from concurrent.futures import ThreadPoolExecutor
        from mstar_sim import clock
        from mstar.utils import orphan

        multiprocessing.parent_process = lambda: type("Alive", (), {"is_alive": lambda self: True})()
        orphan.watch_parent("test", "parent")

        @clock.kernel_fn("test.step", cost=lambda: (0.0, 4e9))  # ~1.6 ms of HBM traffic on an H100
        def kernel():
            sum(range(300000))  # the emulator's own work: longer than the kernel

        def step():
            for _ in range(50):
                kernel()
            done = torch.cuda.Event()
            done.record()
            return done

        gpu = ThreadPoolExecutor(1)
        gpu.submit(int).result()  # its thread is up
        torch.cuda.synchronize()
        busy, t0 = clock.DEFAULT.busy, time.perf_counter()
        gpu.submit(step).result().synchronize()  # the worker's await and event_sync
        print(time.perf_counter() - t0, clock.DEFAULT.busy - busy)
    """, gpu)
    span, device = map(float, out.split())
    assert device <= span < 1.25 * device, (span, device)


def test_a_device_wait_jumps_the_clock_instead_of_sleeping():
    out = run_sim("""
        import time, torch
        from mstar_sim import clock
        wall, t0 = time.clock_gettime(time.CLOCK_MONOTONIC), time.perf_counter()
        clock.charge(1.0)
        torch.cuda.synchronize()
        clock.charge(1.0)
        torch.ones(1, device="cuda").item()  # a host read's D2H is the same wait
        print(time.perf_counter() - t0, time.clock_gettime(time.CLOCK_MONOTONIC) - wall)
    """)
    moved, wall = map(float, out.split())
    assert 1.999 < moved < 2.01 and wall < 0.2, (moved, wall)


def test_a_region_alone_holds_the_clock_still_whatever_took_its_time():
    """With no other thread running, a region's whole wall time is the emulator's, CPU or not: a
    loaded box moves no clock. (So native work on an uncounted thread beside it is hidden too.)"""
    out = run_sim("""
        import time
        from mstar_sim import clock, waits
        t0 = time.perf_counter()
        with clock.computing():
            waits._real_sleep(0.3)  # as if the box had taken the CPU away
        print(time.perf_counter() - t0)
    """)
    assert float(out) < 0.05, out


def test_a_thread_woken_by_another_runs_before_the_clock_can_jump():
    """An executor's thread does the host's work and sets an event: its waiter's span is that
    work, though the waker blocks at once and another thread waits for the device 10 s on."""
    out = run_sim("""
        import threading, time, torch
        from concurrent.futures import ThreadPoolExecutor
        from mstar_sim import clock

        far, ready = threading.Event(), threading.Event()

        def device_waiter():
            clock.charge(10.0)
            torch.cuda.synchronize()
            far.set()

        def work():
            end = time.perf_counter() + 0.2  # the host's own work
            while time.perf_counter() < end:
                pass
            ready.set()

        from mstar_sim.device import SimMode
        left, leave = [], SimMode.__exit__
        SimMode.__exit__ = lambda self, *exc: (left.append(self), leave(self, *exc))[1]
        pool = ThreadPoolExecutor(1)
        pool.submit(int).result()
        started = threading.Thread(target=device_waiter)
        started.start()
        wall, t0 = time.clock_gettime(time.CLOCK_MONOTONIC), time.perf_counter()
        pool.submit(work)
        assert ready.wait(timeout=60)
        span = time.perf_counter() - t0
        far.wait()
        print(span, time.clock_gettime(time.CLOCK_MONOTONIC) - wall)
        started.join()
        assert left  # a thread leaves its mode in Python, not in its thread-local teardown
    """)
    span, wall = map(float, out.split())
    assert 0.2 <= span < 1 and wall < 5, (span, wall)


def test_a_notify_as_a_wait_opens_leaves_the_mover_alive_and_a_dead_mover_stops_the_run():
    """A notify between a Condition's first try and its wait being listed: the mover must not end
    that wait too, as releasing an unlocked lock killed it. And should it die of anything, every
    device wait would hang, so the process aborts instead."""
    out = run_sim("""
        import threading
        from mstar_sim import waits
        enter, lock, go, done = waits.enter, threading._allocate_lock(), waits._real_lock(), waits._real_lock()
        lock.acquire()  # a Condition's waiter lock
        go.acquire()
        done.acquire()

        def notify():
            go.acquire()
            try:
                lock.release()
            except RuntimeError:  # the mover got there first: Condition.notify allows for it
                pass
            done.release()

        threading.Thread(target=notify, name=waits.OWN + "notify", daemon=True).start()

        def entered(*args, **kwargs):
            w = enter(*args, **kwargs)
            go.release()
            done.acquire(True, 0.1)
            w.due = True  # its deadline passes before it can look
            waits.kick()
            waits._real_sleep(0.1)
            return w

        waits.enter = entered
        lock.acquire(True, 0.001)
        waits.enter = enter
        waits._real_sleep(0.1)
        print(any(t.name == waits.OWN + "mover" for t in threading.enumerate()))
    """)
    assert out.strip() == "True", out
    dead = start_sim("""
        import time
        from mstar_sim import waits
        waits._move = lambda: 1 / 0
        waits.kick()
        time.sleep(30)
    """)
    assert dead.returncode == -6 and "ZeroDivisionError" in dead.stderr, (dead.returncode, dead.stderr[-1500:])


def run_deployment(*codes: str) -> list[str]:
    """Each of ``codes`` as one process of a deployment of its own (a fresh table), all at once.
    Each gets ``barrier()``, which returns once all have joined the table. The last one out
    takes the table with it."""
    with tempfile.TemporaryDirectory() as tmp:
        node = os.path.basename(tmp)
        env = dict(caller_env(), MSTAR_SIM="1", MSTAR_SIM_GPU="h100", MSTAR_SIM_NODE=node,
                   PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "mstar_sim", "_site"), ROOT]))
        os.mkdir(os.path.join(tmp, "joined"))
        prelude = f"""
import os, sys, time, threading, torch
from mstar_sim import clock, waits
TMP = {tmp!r}
def barrier():
    open(TMP + f"/joined/{{os.getpid()}}", "w").close()
    while len(os.listdir(TMP + "/joined")) < {len(codes)}:
        time.perf_counter()  # (a sign of life: a process silent for a look is parked)
        waits._real_sleep(0.01)
    waits.TABLE.bound()
"""
        scripts = [prelude + textwrap.dedent(code) for code in codes]
        procs = [subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for script in scripts]
        try:
            outs = [proc.communicate(timeout=120) for proc in procs]
        finally:
            for proc in procs:
                proc.kill()
    for proc, (_, err) in zip(procs, outs, strict=True):
        assert proc.returncode == 0, err[-3000:]
    assert not os.path.exists(f"/dev/shm/mstar_sim_clock_{os.getuid()}_{node}")
    return [out for out, _ in outs]


def test_a_blocked_process_follows_the_deployments_time():
    """A process with nothing to do but poll keeps up with a worker's jumps, in far less wall
    time than the time it follows."""
    worker, follower = run_deployment("""
        barrier()
        start = time.perf_counter()
        clock.charge(2.0)
        torch.cuda.synchronize()
        print(time.perf_counter() - start)
        time.sleep(0.05)  # the follower's last sleep ends with this one
    """, """
        barrier()
        start = wall = time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC)
        while time.perf_counter() - start[0] < 2.0:
            time.sleep(0.05)
        print(time.perf_counter() - start[0], time.clock_gettime(time.CLOCK_MONOTONIC) - start[1], waits.TABLE._q[1])
    """)
    moved, wall, slots = map(float, follower.split())
    assert 1.99 < float(worker) < 2.1 and 2.0 <= moved < 2.1 and wall < 1, (worker, moved, wall)
    assert slots == 2, slots  # a read scans what was used, not all 64


def test_two_blocked_processes_move_together_to_the_earlier_deadline():
    """Both blocked on host deadlines: the clock goes to the earlier one, then the later, each
    on time on the deployment's clock, in far less wall time than the deadlines are apart."""
    early, late = run_deployment("""
        barrier()
        start = time.perf_counter()
        time.sleep(1.0)
        print(time.perf_counter() - start, time.perf_counter())
        time.sleep(3.0)  # a peer until the other is done
    """, """
        barrier()
        start, wall = time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC)
        time.sleep(2.0)
        print(time.perf_counter() - start, time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC) - wall)
    """)
    first, woke = map(float, early.split())
    second, ended, wall = map(float, late.split())
    assert 1.0 <= first < 1.05 and 2.0 <= second < 2.05 and woke < ended and wall < 1, (first, second, wall)


def test_a_fixed_rate_client_keeps_up_with_a_worker_a_message_is_in_flight_to():
    """Round-3's deadlock: a message in flight to a running worker holds no one else back. The
    client moves to each arrival as the worker's clock passes it, and the worker doesn't run
    more than W past the client."""
    worker, client = run_deployment("""
        barrier()
        start = time.perf_counter()
        ahead = 0.0
        while time.perf_counter() - start < 0.3:
            ahead = max(ahead, time.perf_counter() - waits.TABLE.bound())
        print(time.perf_counter() - start, ahead)
    """, """
        barrier()
        start = time.perf_counter()
        for _ in range(30):  # 100 arrivals a second
            time.sleep(0.01)
        print(time.perf_counter() - start)
    """)
    span, ahead = map(float, worker.split())
    assert 0.3 <= span < 0.5 and 0.3 <= float(client) < 0.5 and ahead < 0.002, (span, ahead, client)


def test_a_blocked_process_publishes_its_next_event_as_its_waits_end():
    """A deadline gone from a process that stays blocked (a thread the registry doesn't count) is
    not left in its slot: peers' bounds stand on that, and clause 2 would hold them for ever."""
    out = run_sim("""
        import threading, time
        from mstar_sim import clock, table, waits
        base, seen, done = clock.now(), [], threading.Event()

        def uncounted():
            time.sleep(0.2)
            waits._real_sleep(0.3)  # the main thread is blocked throughout
            seen.append(waits.TABLE._d[waits.TABLE._mine + table._NEXT] - base)
            done.set()

        threading.Thread(target=uncounted, name=waits.OWN + "test", daemon=True).start()
        done.wait(5.0)
        print(seen[0])
    """)
    assert 4.9 < float(out) < 5.1, out


def test_a_clock_read_is_held_only_where_it_can_be_let_go():
    """mstar reads the clock from a GC callback, which can run while the emulator holds its lock,
    or at interpreter exit, when the mover is gone: a read there must not wait on the mover. (The
    bound is behind until it is put back.)"""
    out = run_sim("""
        import threading, time
        from mstar_sim import clock, waits
        waits.HELD_LIMIT = 2.0
        bound = waits.TABLE.bound
        waits.TABLE.bound = lambda: 0.0
        waits._ahead[0] = 0.0
        with clock.LOCK:
            time.perf_counter()
        waits._move = lambda: 0.05  # the mover lets no one go, as when it is gone
        threading.Thread(target=lambda: (waits._real_sleep(0.5), setattr(waits.TABLE, "bound", bound)),
                         name=waits.OWN + "test").start()
        time.perf_counter()
        print("returned")
    """)
    assert out.strip() == "returned", out


def test_an_exiting_process_keeps_its_slot_and_its_table_until_it_is_gone():
    """A process in finalization is still a peer, at real rate: two idle processes don't carry a
    third's ``join(timeout)`` past it. And a table isn't unlinked under a process about to claim
    a slot in it, which would then share no time with one that comes later."""
    exiting, waiter, _ = run_deployment("""
        class Slow:
            def __del__(self, real=waits._real_sleep):
                real(1.0)
        slow = Slow()
        barrier()
    """, """
        barrier()
        waits._real_sleep(0.2)  # the first is in finalization
        wall = time.clock_gettime(time.CLOCK_MONOTONIC)
        time.sleep(0.5)
        print(time.clock_gettime(time.CLOCK_MONOTONIC) - wall)
        open(TMP + "/done", "w").close()
    """, """
        barrier()
        while not os.path.exists(TMP + "/done"):
            time.sleep(0.05)
    """)
    assert float(waiter) >= 0.4, waiter
    out = run_sim("""
        import fcntl, os, threading
        from mstar_sim import table, waits
        path = table.path_for(f"test-{os.getpid()}-leave")
        first, late = table.Table(path), []
        fcntl.fcntl(first._fd, table._F_OFD_SETLK, table._flock(fcntl.F_WRLCK, 0))  # as a claim in progress
        claim = threading.Thread(target=lambda: late.append(table.Table(path)))
        claim.start()
        waits._real_sleep(0.3)  # it has the file open, and waits to claim a slot
        first.leave()
        fcntl.fcntl(first._fd, table._F_OFD_SETLK, table._flock(fcntl.F_UNLCK, 0))
        claim.join()
        print(os.path.exists(path))
        late[0].leave()
    """)
    assert out.strip() == "True", out


def test_a_native_spinner_doesnt_hold_the_gate_shut():
    """A task of a process with nothing running that never leaves the CPU (an OpenMP pool's spin
    wait) isn't a message on its way: past a limit the process says it is blocked, so its frozen
    clock can't hold a peer to the abort. And a scan that can't tell keeps the gate shut."""
    spinner, held = run_deployment("""
        import hashlib
        spin = lambda: hashlib.pbkdf2_hmac("sha256", b"x", b"y", 2**31 - 1)  # (no GIL, no clock)
        threading.Thread(target=spin, name=waits.OWN + "spin", daemon=True).start()
        barrier()
        time.sleep(2.0)
    """, """
        waits.HELD_LIMIT = 3.0
        barrier()
        clock.charge(1.0)
        torch.cuda.synchronize()
        time.perf_counter()
        print("passed")
    """)
    assert held.strip() == "passed", held
    out = run_sim("""
        import errno, os
        from mstar_sim import waits
        real, tasks = os.open, waits._Tasks("self")

        def out_of_fds(*args, **kwargs):
            raise OSError(errno.EMFILE, "too many open files")

        os.open = out_of_fds
        print(tasks.on_cpu(waits._mover_tid[0]))
    """)
    assert out.strip() == "True", out


def test_a_device_wait_needs_no_one():
    """A device wait beside another thread's wait that it can't see the end of ends all the same.
    Beside a sleep (nothing bounds it, the sleep is earlier): at once, not as the sleep passes in
    real time. And beside a thread at a meet, whose peer may be waiting for what comes after."""
    out = run_sim("""
        import threading, time, torch
        from mstar_sim import clock
        threading.Thread(target=time.sleep, args=(0.5,), daemon=True).start()
        wall = time.clock_gettime(time.CLOCK_MONOTONIC)
        clock.charge(1.0)
        torch.cuda.synchronize()
        print(time.clock_gettime(time.CLOCK_MONOTONIC) - wall)
    """)
    assert float(out) < 0.4, out
    run_deployment("""
        import datetime
        import torch.distributed as dist
        dist.init_process_group("nccl", init_method="file://" + TMP + "/group", rank=0, world_size=2,
                                timeout=datetime.timedelta(seconds=20))
        x = torch.empty(1 << 14, device="cuda")
        barrier()

        def after_the_device():
            clock.charge(1.0)
            torch.cuda.synchronize()
            open(TMP + "/passed", "w").close()

        thread = threading.Thread(target=after_the_device)
        thread.start()
        dist.all_reduce(x)
        thread.join()
        dist.destroy_process_group()
    """, """
        import datetime
        import torch.distributed as dist
        dist.init_process_group("nccl", init_method="file://" + TMP + "/group", rank=1, world_size=2,
                                timeout=datetime.timedelta(seconds=20))
        x = torch.empty(1 << 14, device="cuda")
        barrier()
        while not os.path.exists(TMP + "/passed"):
            time.perf_counter()  # it runs, so the bound is its clock
            waits._real_sleep(0.01)
        dist.all_reduce(x)
        dist.destroy_process_group()
    """)


def test_a_real_wake_of_the_rust_poll_survives_a_poke_for_another_wait():
    """The mover's eventfd is shared by every poll of the process, and a poll's wake drains mstar's
    own event: dropped here, the data worker's flush would sit out the poll. A poll with no event
    has no wake to give its caller: that one was the mover's."""
    out = run_sim("""
        import collections, time
        from mstar_sim import clock, waits
        from mstar.communication import rust_communicator as rust
        from mstar.communication.event import EventWakeup

        class Inner:
            def recv_or_wake(self, timeout_ms):
                if timeout_ms <= 0 or not self.woken:
                    return "timeout", None
                self.woken = False
                return "wake", None

        def poll(event):
            comm = rust.RustZMQCommunicator.__new__(rust.RustZMQCommunicator)
            comm._inner, comm._buffered, comm.event = Inner(), collections.deque(), event
            comm._inner.woken = True
            return comm._poll_once(100)

        waits._poked.add(object())  # a selector's wait the mover is ending
        print(poll(EventWakeup()), poll(None))
        waits._poked.clear()
    """)
    assert out.split() == ["wake", "timeout"], out


def test_a_process_alone_sees_the_real_time_of_a_wait_it_cant_see_the_end_of():
    """A client with no peer in its table, waiting on a server's socket: nothing else can move its
    clock, so the wait's real time is what it shows, or a client's TTFT would read its CPU alone."""
    out = run_sim("""
        import os, selectors, subprocess, sys, time
        read, write = os.pipe()
        reply = f"import os, time; time.sleep(0.3); os.write({write}, b'x')"
        subprocess.Popen([sys.executable, "-S", "-c", reply], pass_fds=[write], env={})
        selector = selectors.DefaultSelector()
        selector.register(read, selectors.EVENT_READ)
        start = time.perf_counter()
        selector.select(30)
        print(time.perf_counter() - start)
    """)
    assert 0.25 < float(out) < 5, out


def test_the_mover_waits_on_no_lock_of_threadings():
    """It holds the emulator's lock, and a thread holding ``threading``'s can run a free that wants
    that one."""
    out = run_sim("""
        import threading
        from mstar_sim import clock, waits
        go, done = waits._real_lock(), waits._real_lock()
        go.acquire()
        done.acquire()
        w = waits.Wait(waits.DEVICE, 0.0, True, None, None)

        def report():
            go.acquire()
            waits._slept(w, 0.0)
            done.release()

        threading.Thread(target=report, name=waits.OWN + "report", daemon=True).start()
        with threading._active_limbo_lock:
            go.release()
            print(done.acquire(True, 2.0))
    """)
    assert out.strip() == "True", out


def test_a_read_held_behind_a_rising_bound_doesnt_abort():
    """The limit is on a bound that stands still, not on a hold: a device wait can put a process
    far past a peer that runs at real rate, and it then waits for the peer to get there."""
    jumper, spinner = run_deployment("""
        waits.HELD_LIMIT = 0.5
        barrier()
        clock.charge(2.0)
        torch.cuda.synchronize()
        time.perf_counter()
        print("passed")
    """, """
        barrier()
        end = time.perf_counter() + 2.5
        while time.perf_counter() < end:
            pass
    """)
    assert jumper.strip() == "passed", jumper


def test_a_read_sees_a_bound_only_the_mover_had_stale():
    """The mover stores a bound it read before a change of state, over the one a read just worked
    out: the read after that looks again. And of two reads that see a process parked, the second
    must not catch up a process the first has set running."""
    out = run_sim("""
        import threading, time
        from mstar_sim import clock, waits
        table, low = waits.TABLE, clock.now() - 1.0
        stale = lambda: threading.current_thread().name == waits.OWN + "mover" or waits.REPORT["ahead_stalls"]
        table.bound = lambda: 1e9 if stale() else low
        waits._seen[0] = table.epochs()  # the change came before this read looked
        waits.kick()
        waits._real_sleep(0.1)
        time.perf_counter()  # past the true bound, though the mover's says not
        print(waits.REPORT["ahead_stalls"])
        table.bound, table.latest = lambda: 1e9, lambda: 1e6
        go = waits._real_lock()
        go.acquire()
        threading.Thread(target=lambda: (go.acquire(), time.perf_counter()), name=waits.OWN + "reader").start()
        with clock.LOCK:
            clock.PARKED[0] = True
            go.release()
            waits._real_sleep(0.05)  # it has seen the process parked, and waits for this lock
            clock.PARKED[0] = False
        waits._real_sleep(0.05)
        print(clock.now() < 1e6)
    """)
    assert out.split() == ["1", "True"], out


def test_a_process_in_a_wait_the_emulator_cant_see_doesnt_pace_the_deployment():
    """A thread in an unknown wait counts as running, but a process doing nothing at all (no task
    on a CPU, no CPU used, no clock read) is parked: a peer's device wait still jumps."""
    reader, waiter = run_deployment("""
        os.mkfifo(TMP + "/fifo")
        fd = os.open(TMP + "/fifo", os.O_RDWR)  # (a writer of its own, so the read waits)
        barrier()
        os.read(fd, 1)  # a pipe read: no patch sees it
        print("read")
    """, """
        barrier()
        waits._real_sleep(0.2)  # the reader is in its read, and a look has seen it idle
        wall = time.clock_gettime(time.CLOCK_MONOTONIC)
        clock.charge(3.0)
        torch.cuda.synchronize()
        time.perf_counter()  # a read 3 s on: clause 2 holds it while the reader counts as running
        print(time.clock_gettime(time.CLOCK_MONOTONIC) - wall)
        os.write(os.open(TMP + "/fifo", os.O_WRONLY), b"x")
    """)
    assert reader.strip() == "read" and float(waiter) < 0.5, (reader, waiter)


def test_a_stalled_deployment_aborts_saying_who_holds_it():
    """Clause 2 never lets go, so a peer that stops must not mean a silent hang: past the limit
    the held process prints the table and aborts, as an unpaired collective does."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(caller_env(), MSTAR_SIM="1", MSTAR_SIM_GPU="h100", MSTAR_SIM_NODE=os.path.basename(tmp),
                   PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "mstar_sim", "_site"), ROOT]))
        held = subprocess.run([sys.executable, "-c", textwrap.dedent("""
            import os, subprocess, sys, time, torch
            from mstar_sim import clock, waits
            waits.HELD_LIMIT = 1.0
            peer = subprocess.Popen([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSTOP)"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.waitid(os.P_PID, peer.pid, os.WSTOPPED | os.WNOWAIT)
            print(peer.pid, flush=True)
            clock.charge(5.0)
            torch.cuda.synchronize()
            time.perf_counter()
            print("passed")
        """)], env=env, capture_output=True, text=True, timeout=120, check=False)
        os.kill(int(held.stdout.split()[0]), signal.SIGKILL)
        os.remove(f"/dev/shm/mstar_sim_clock_{os.getuid()}_{os.path.basename(tmp)}")
    assert held.returncode == -6 and "passed" not in held.stdout, (held.returncode, held.stderr[-1500:])
    assert "past the bound" in held.stderr and "running" in held.stderr, held.stderr[-1500:]


def test_a_process_beside_two_ranks_stays_within_w_of_them():
    """Both ranks at a meet need no one, so they hold their place in the bound: a poller beside
    them (an API server) stays with the leading rank, at its reads, through a run of collectives."""
    rank = """
        import torch.distributed as dist
        from mstar_sim import dist as emulated
        dist.init_process_group("nccl", init_method="file://" + TMP + "/group", rank=%d, world_size=2)
        x = torch.empty(1 << 14, device="cuda")
        watched = emulated._watched  # each meet takes 20 ms of real time, as gloo can on a loaded box
        emulated._watched = lambda wait, *args, **kwargs: (waits._real_sleep(0.02), watched(wait, *args, **kwargs))
        barrier()
        for _ in range(40):
            with clock.computing():
                sum(range(100000))  # the emulator's own work
            time.perf_counter()
            dist.all_reduce(x)
        open(TMP + "/ranks-done-%d", "w").close()
        dist.destroy_process_group()
    """
    _, _, poller = run_deployment(rank % (0, 0), rank % (1, 1), """
        leave = waits.leave  # a loaded box, slow to run a thread the mover has let go
        late = lambda w: w is not None and w.kind is waits.AHEAD
        waits.leave = lambda w: (waits._real_sleep(0.006) if late(w) else None, leave(w))[1]
        barrier()
        table, ahead = waits.TABLE, 0.0
        while len([f for f in os.listdir(TMP) if f.startswith("ranks-done")]) < 2:
            time.sleep(0.002)
            waits._real_sleep(0.001)  # the box takes the thread a moment, past W: this read is held
            now = time.perf_counter()  # a read, as mstar's: clause 2 bounds a process at its reads
            latest = table.latest()
            if latest:  # (0: the ranks have gone)
                ahead = max(ahead, now - latest)
        print(ahead)
    """)
    assert float(poller) < 2 * 0.5e-3 + 1e-3, poller


def test_a_collective_starts_at_the_latest_ranks_stream():
    """Ranks reach a collective with their streams apart: it starts when the last stream gets
    there, on every rank, and a host reading the result on either sees that."""
    node = f"test-{os.getpid()}-collective"
    out = run_sim("""
        import sys, tempfile, time, torch, torch.distributed as dist, torch.multiprocessing as mp
        from mstar_sim import clock

        def rank(r, path):
            dist.init_process_group("nccl", init_method=f"file://{path}", rank=r, world_size=2)
            torch.cuda.set_device(r)
            x = torch.empty(1 << 14, device="cuda")
            dist.barrier()
            start = time.perf_counter()
            if r == 0:
                clock.charge(1.0)
            dist.all_reduce(x)
            torch.cuda.synchronize()
            sys.stdout.write(f"rank {r} {time.perf_counter() - start} {clock.DEFAULT.free_at}\\n")  # one write
            sys.stdout.flush()
            dist.destroy_process_group()

        if __name__ == "__main__":
            mp.spawn(rank, args=(tempfile.mkdtemp() + "/group",), nprocs=2)
    """, MSTAR_SIM_NODE=node)
    rows = [line.split() for line in out.splitlines() if line.startswith("rank")]
    spans, ends = {r[1]: float(r[2]) for r in rows}, [float(r[3]) for r in rows]
    assert len(rows) == 2 and 1.0 <= spans["0"] < 1.2 and abs(ends[0] - ends[1]) < 1e-6, out


def test_a_native_process_holds_the_deployment_only_while_it_works():
    """The Rust frontend's shape: a child no patch reaches, watched by the process that started
    it. Its work takes real time on the deployment's clock, a peer's deadline can't pass it,
    and a wait for the device costs no wall time once it is idle."""
    owner, peer = run_deployment("""
        import subprocess
        work = "import sys, time; sys.stdin.readline(); end = time.process_time() + 0.3\\n" \\
               "while time.process_time() < end: pass\\nsys.stdin.readline()"
        child = subprocess.Popen([sys.executable, "-S", "-c", work], stdin=subprocess.PIPE, env={})
        waits.watch(child.pid)
        barrier()
        start, real = time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC)
        cpu = waits._native[child.pid][0]
        idle = time.clock_gettime(cpu)
        child.stdin.write(b"go\\n")
        child.stdin.flush()
        while time.clock_gettime(cpu) == idle:  # (its onset can beat a look: README Limits)
            waits._real_sleep(0.0001)
        while not waits._native_busy[0]:
            time.sleep(0.001)
        while waits._native_busy[0]:
            time.sleep(0.01)
        print(time.perf_counter() - start, time.clock_gettime(time.CLOCK_MONOTONIC) - real)
        wall = time.clock_gettime(time.CLOCK_MONOTONIC)
        clock.charge(1.0)
        torch.cuda.synchronize()
        print(time.clock_gettime(time.CLOCK_MONOTONIC) - wall)
        child.kill()
    """, """
        barrier()
        start = time.perf_counter()
        time.sleep(0.1)
        print(time.perf_counter() - start)
    """)
    worked, real, wall = map(float, owner.split())
    assert 0.3 <= worked <= real and wall < 0.2 and 0.1 <= float(peer) < 0.2, (worked, real, wall, peer)


def test_a_watched_child_takes_its_fds_with_it():
    """Each native child's task fds go when it does: a media run starts one per request."""
    out = run_sim("""
        import os, subprocess
        from mstar_sim import waits
        count = lambda: len(os.listdir("/proc/self/fd")) - len(waits._own.fds)  # (the process's own stay)
        before = count()
        for _ in range(10):
            child = subprocess.Popen(["sleep", "0.02"])
            while not waits._native[child.pid][2].fds:  # a look has seen it
                waits._real_sleep(0.001)
            child.wait()
            while child.pid in waits._native:
                waits._real_sleep(0.01)
        print(count() - before)
    """)
    assert int(out) <= 1, out


def test_a_blocking_h2d_holds_the_host_until_the_copy_is_done():
    """As ``cudaMemcpy``: after what is queued on the stream, and for the copy itself."""
    out = run_sim("""
        import time, torch
        from mstar_sim import clock
        host = torch.zeros(1 << 30, dtype=torch.uint8)
        device = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
        for blocking in (host.cuda, lambda: device.copy_(host)):
            torch.cuda.synchronize()
            clock.charge(0.1)
            t0 = time.perf_counter()
            blocking()
            print(time.perf_counter() - t0, clock.DEFAULT.free_at - t0)
    """)
    for line in out.splitlines():
        span, done = map(float, line.split())
        assert done > 0.1 and done <= span < done + 0.01, (span, done)


def test_a_copy_from_pageable_memory_blocks_the_host_and_is_slower_than_from_pinned():
    out = run_sim("""
        import time, torch
        from mstar_sim import clock
        pageable = torch.zeros(1 << 30, dtype=torch.uint8)
        pinned = torch.zeros(1 << 30, dtype=torch.uint8, pin_memory=True)
        assert pinned.is_pinned() and not pageable.is_pinned() and pageable.pin_memory().is_pinned()
        assert pinned[8:].is_pinned() and not pageable.clone().is_pinned()
        for source in (pinned, pageable):
            torch.cuda.synchronize()
            busy, t0 = clock.ENGINES["h2d"].free_at, time.perf_counter()
            source.to("cuda", non_blocking=True)
            print(time.perf_counter() - t0, clock.DEFAULT.free_at - t0)
    """)
    (host_pinned, device_pinned), (host_pageable, device_pageable) = (map(float, l.split()) for l in out.splitlines())
    assert host_pinned < 0.1 * device_pinned, (host_pinned, device_pinned)
    assert host_pageable >= 0.9 * device_pageable and device_pageable > 1.5 * device_pinned, out


def test_concurrent_copies_share_a_copy_engine_and_overlap_compute():
    """Two H2D copies on two streams take twice one; a kernel on a third stream runs beside them,
    and a D2H has an engine of its own."""
    out = run_sim("""
        import torch
        from mstar_sim import clock, cost
        host = torch.zeros(1 << 30, dtype=torch.uint8, pin_memory=True)
        one = cost.copy_seconds(1 << 30, "h2d")
        streams = [torch.cuda.Stream() for _ in range(3)]
        out = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        start = clock.now()
        for stream in streams[:2]:
            with torch.cuda.stream(stream):
                host.to("cuda", non_blocking=True)
        with torch.cuda.stream(streams[2]):
            clock.charge(one)  # compute beside the copies
        print((streams[1].timeline.free_at - start) / one, (streams[2].timeline.free_at - start) / one)
        with torch.cuda.stream(streams[2]):
            torch.cuda.synchronize()
            start = clock.now()
            host.copy_(out, non_blocking=True)
        print((streams[2].timeline.free_at - start) / one)
    """)
    (both, compute), (d2h,) = (map(float, line.split()) for line in out.splitlines())
    assert 1.95 < both < 2.1 and compute < 1.1 and d2h < 1.1, out


def test_a_cached_call_answers_as_the_call_would():
    """The shape cache tells ``1`` from ``1.0``, whose results differ in dtype, and a launch counts
    the kernels it stands for, cached or not (``rms_norm``: 8)."""
    out = run_sim("""
        import torch
        from mstar_sim import clock
        ints = torch.empty(1 << 21, dtype=torch.int64, device="cuda")  # (too many to keep values)
        print(*((ints * value).dtype for value in (1, 1.0, True)))
        x = torch.empty(512, 4096, device="cuda")
        counts = [clock.DEFAULT.kernels]
        for _ in range(3):
            torch.rms_norm(x, (4096,))
            counts.append(clock.DEFAULT.kernels)
        print(*(b - a for a, b in zip(counts, counts[1:])))
    """)
    dtypes, kernels = out.splitlines()
    assert dtypes.split() == ["torch.int64", "torch.float32", "torch.int64"], out
    assert len(set(kernels.split())) == 1 and int(kernels.split()[0]) > 1, out


def test_timing_only_holds_shapes_not_values():
    """Nothing is computed: a 40 GB tensor costs no host memory
    but counts against the card; small integer tensors (what mstar plans with)
    keep their values; a value that was never computed reads as zeros."""
    out = run_sim("""
        import torch
        from mstar_sim import clock
        free0, total = torch.cuda.mem_get_info()
        big = torch.empty(10 * 2**30, dtype=torch.float32, device="cuda")
        free1, _ = torch.cuda.mem_get_info()
        assert total == torch.cuda.get_device_properties().total_memory
        assert free0 - free1 == 40 * 2**30
        a, b = torch.empty(1000, device="cuda"), torch.empty(3000, device="cuda")
        sa, sb = a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()
        assert sa and sb and abs(sa - sb) >= 4000  # distinct storages: mstar keys bytes by pointer
        assert a[10:].data_ptr() == sa + 40
        stream = clock.current_timeline()
        launched = stream.kernels
        torch.empty_like(a), a.new_empty(8)
        assert stream.kernels == launched  # an allocation is no kernel
        pos = torch.arange(5, device="cuda") + 1
        assert pos.tolist() == [1, 2, 3, 4, 5]  # small ints are real
        ones = torch.ones(4096, device="cuda")
        assert (ones[:, None] * ones[None, :]).sum().item() == 0  # size decides, for a result too
        raw = torch.zeros(1 << 16, dtype=torch.uint8, device="cuda")
        floats = raw.view(torch.float32)  # no new result: a view, however large it reads as floats
        floats[0] = 1.0
        assert raw[:4].tolist() == [0, 0, 128, 63] and floats.data_ptr() == raw.data_ptr()
        logits = torch.randn(4, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
        assert logits.argmax(-1).tolist() == [0, 0, 0, 0]  # never computed
        # a small tensor keeps the values last written to it, however it was made: a host
        # float staged into a preallocated buffer, and a never-computed write over a kept one
        buf = torch.empty(1, device="cuda")
        buf.copy_(torch.tensor([0.5]))
        assert buf.item() == 0.5
        buf.mul_(logits[0, :1])
        assert buf.item() == 0.0
        # such a write lands on the elements it writes, whatever it is; captured, at each replay
        ids, one = torch.arange(4, device="cuda"), torch.tensor([1], device="cuda")
        ids[:1] = logits[:1, 0].long()
        ids.index_copy_(0, one, logits[:1, 0].long())
        assert ids.tolist() == [0, 0, 2, 3]
        host_cast = torch.ones(2, device="cuda")
        host_cast.copy_(torch.empty(2, device="meta"))  # a header-only weight
        assert host_cast.tolist() == [0, 0]
        staged, g = torch.ones(2, device="cuda"), torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            staged.mul_(logits[0, :2])
        assert staged.tolist() == [1, 1]
        g.replay()
        assert staged.tolist() == [0, 0]
        # no kernel computed what a stand-in wrote: a write over part of it leaves the rest so, one
        # that adds to it reads it, and only a write of all of it, with computed values, ends that
        import flashinfer
        from mstar_sim.device import never_computed
        stand_in = lambda: flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1)
        row, mask = torch.tensor([0], device="cuda"), torch.eye(2, 4, dtype=torch.bool, device="cuda")
        for write in (lambda t: t[1:].zero_(), lambda t: t[1:].copy_(torch.zeros(1, 4)),
                      lambda t: t.masked_fill_(mask, 0.0), lambda t: t.index_fill_(0, row, 0.0),
                      lambda t: t.index_put_((row,), torch.ones(1, device="cuda"), accumulate=True),
                      lambda t: t.scatter_(0, row[:, None], 1.0, reduce="add")):
            written = stand_in()
            write(written)
            assert never_computed(written._data), write
        for write in (lambda t: t.zero_(), lambda t: t.copy_(torch.ones(2, 4, device="cuda")),
                      lambda t: t.copy_(torch.ones(2, 4)), lambda t: torch.neg(torch.ones(2, 4, device="cuda"), out=t),
                      lambda t: torch.sort(torch.ones(2, 4, device="cuda"), out=(t, row.new_zeros(2, 4)))):
            written = stand_in()
            write(written)
            assert not never_computed(written._data), write
        # and an op that takes only a shape from a tensor reads none of it
        for make in (torch.zeros_like, torch.ones_like, lambda t: torch.full_like(t, 2.0), lambda t: t.new_zeros(2),
                     lambda t: t.new_ones(2), lambda t: t.new_full((2,), 2.0)):
            assert not never_computed(make(stand_in())._data), make
        # an op that refused a stand-in's values leaves what it writes as no kernel computed
        out = row.new_full((2, 1), 99)
        try:
            torch.multinomial(stand_in() * 0, 1, out=out)
        except RuntimeError:
            assert never_computed(out._data)
        else:
            raise AssertionError("multinomial of zeros passed")
        # so does one refused at a replay, whose probe runs under the caller's default device
        from mstar_sim import device as emulated
        zeros, eight, graph = stand_in().long(), torch.full((2, 4), 8, device="cuda"), torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            eight // zeros
        refused = emulated.VALUE_SHAPES[0]
        with torch.device("meta"):
            try:
                graph.replay()
                raise AssertionError("a divide by zeros replayed")
            except RuntimeError as e:
                assert "ZeroDivision" in str(e), e
        assert emulated.VALUE_SHAPES[0] == refused + 1
        # a computed mask selects its rows of a value-less tensor, each mask its own
        rows = torch.empty(8192, 16, device="cuda")
        for step in (2, 4):
            assert rows[torch.arange(8192, device="cuda") % step == 0].shape == (8192 // step, 16)
        # an integer index shapes nothing: a kept table read through a value-less one has no values
        assert never_computed(pos[torch.empty(8192, device="cuda").long()]._data)
        # a stand-in's write in place flags what it writes, and only that
        import types
        from flashinfer import comm
        from mstar_sim import kernels
        q, k, state, weight, residual, normed, summed = (torch.empty(2, 8, device="cuda") for _ in range(7))
        flashinfer.fused_add_rmsnorm(q, k, weight[0])
        assert never_computed(q._data) and never_computed(k._data) and not never_computed(weight._data)
        q, k = torch.empty(2, 1, 8, device="cuda"), torch.empty(2, 1, 8, device="cuda")
        flashinfer.apply_rope_pos_ids_inplace(q, k, torch.arange(2, device="cuda"))
        assert never_computed(q._data) and never_computed(k._data)
        kernels.causal_conv1d_update(weight.clone(), state, weight)
        assert never_computed(state._data)
        ws = types.SimpleNamespace(world_size=1)
        comm.allreduce_fusion(weight, ws, residual_in=weight, residual_out=residual, norm_out=normed)
        comm.allreduce_fusion(input=weight, workspace=ws, pattern=comm.AllReduceFusionPattern.kAllReduce, output=summed)
        assert all(never_computed(t._data) for t in (residual, normed, summed)) and not never_computed(weight._data)
        torch.cuda.empty_cache()
        before = torch.cuda.mem_get_info()[0]
        del big
        assert torch.cuda.mem_get_info()[0] == before  # cached, as torch's allocator keeps it
        torch.cuda.empty_cache()
        assert torch.cuda.mem_get_info()[0] - before == 40 * 2**30
        on0 = torch.cuda.memory_allocated(0)
        other = torch.empty(2**20, device="cuda:1")  # each device counts its own
        assert torch.cuda.memory_allocated(1) == 4 * 2**20 and torch.cuda.memory_allocated(0) == on0
        assert torch.cuda.mem_get_info(1)[0] == total - 20 * 2**20  # the allocator's segment for a 4 MiB block
        torch.cuda.set_device(1)  # a bare "cuda" names the current device
        assert torch.cuda.memory_allocated(torch.device("cuda")) == torch.cuda.memory_allocated(1)
        torch.cuda.set_device(0)
        print("ok")
    """)
    assert out.strip() == "ok"


def test_device_side_assert_surfaces_once_the_device_reaches_it_and_stays():
    """An out-of-bounds index faults the device, not the launch: the host, ahead of the
    device, launches on; the next sync raises, and so does every later CUDA call."""
    out = run_sim("""
        import torch
        from mstar_sim import clock
        table, ids = torch.empty(50, 8, device="cuda"), torch.tensor([3, 77], device="cuda")
        torch.cuda.synchronize()
        clock.charge(2)  # device work queued ahead of the bad index
        torch.nn.functional.embedding(ids, table)
        (ids + 1)  # launched before the device got to the fault
        stream, event = torch.cuda.Stream(), torch.cuda.Event()
        for call in (torch.cuda.synchronize, lambda: torch.empty(1, device="cuda"), torch.cuda.mem_get_info,
                     lambda: stream.wait_stream(torch.cuda.current_stream()), lambda: stream.wait_event(event),
                     torch.cuda.empty_cache, lambda: torch.cuda.CUDAGraph().capture_begin()):
            try:
                call()
                raise SystemExit("a CUDA call after the fault succeeded")
            except RuntimeError as e:
                assert "device-side assert" in str(e), e
        assert ids.view(2, 1).data_ptr() and ids.data.shape == (2,)  # no CUDA call: nothing to fail
        print("ok")
    """)
    assert out.strip() == "ok"
    # an index no kernel computed (a sampled token) can't fault: its value is unknown, not 0
    out = run_sim("""
        import flashinfer, torch
        tokens = flashinfer.sampling_from_probs(torch.empty(2, 512, device="cuda"))
        torch.nn.functional.embedding(tokens - 10, torch.empty(50, 8, device="cuda"))
        torch.cuda.synchronize()
        assert tokens.tolist() == [0, 0]
        print("ok")
    """)
    assert out.strip() == "ok"


def test_graph_capture_reserves_its_pool_and_the_card_runs_out():
    """A capture holds its private pool's high-water mark (not 0, not every tensor it
    saw) until the graph goes; a reservation past the card is out of memory."""
    out = run_sim("""
        import torch
        from mstar_sim import clock

        @clock.kernel_fn("test.read", cost=lambda x, n: lambda: (0.0, float(n.sum())))
        def read(x, n):
            return None

        GiB = 2**30
        n = torch.tensor([1], device="cuda")
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            scratch = torch.empty(GiB // 4, device="cuda")  # 1 GiB, freed inside the capture
            read(scratch, n)  # a replay-priced kernel: the graph keeps its cost, not its tensors
            del scratch
            out = torch.empty(GiB // 16, device="cuda")  # 256 MiB, reuses the freed block
        took = torch.cuda.memory_reserved() - before
        assert took == GiB, took
        pool = [s for s in torch.cuda.memory_snapshot() if s["segment_pool_id"] == g.pool()]
        assert sum(s["total_size"] for s in pool) == GiB
        torch.cuda.empty_cache()
        assert torch.cuda.memory_reserved() - before == GiB  # the graph holds its pool
        g.reset()
        del out
        torch.cuda.empty_cache()
        assert torch.cuda.memory_reserved() == before
        # a pool any caller begins allocating to takes this thread's allocations on its device
        handle = torch.cuda.graph_pool_handle()
        torch._C._cuda_beginAllocateToPool(0, handle)
        held = torch.empty(GiB // 8, device="cuda")
        torch._C._cuda_endAllocateToPool(0, handle)
        assert sum(s["total_size"] for s in torch.cuda.memory_snapshot() if s["segment_pool_id"] == handle) == GiB // 2
        assert "Allocated memory" in torch.cuda.memory_summary()
        del held
        torch._C._cuda_releasePool(0, handle)

        # a stand-in kernel allocates its outputs only, not what the emulator makes them with
        import flashinfer
        rows, k = 8192, 2048
        scores, table = torch.empty(rows, 4096, device="cuda"), torch.empty(rows, 64, dtype=torch.int32, device="cuda")
        lengths = torch.full((rows,), 4096, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        slots = flashinfer.top_k_page_table_transform(scores, table, lengths, k, page_size=64)
        assert torch.cuda.max_memory_allocated() - base == slots.numel() * 4, torch.cuda.max_memory_allocated() - base
        del slots

        total = torch.cuda.get_device_properties().total_memory
        x = torch.empty(total // 16 // 4096, 4096, device="cuda")  # a quarter of the card
        torch.rms_norm(x, (4096,))  # its output, a quarter more, comes and goes
        big = torch.empty(total // 6, device="cuda")  # two thirds of the card
        del big
        again = torch.empty(total // 6, device="cuda")  # reuses the cached block
        for _ in range(20):  # a retry after out of memory, as mstar's workspace growth does
            try:
                torch.empty(total // 6, device="cuda")
                raise SystemExit("an allocation past the card succeeded")
            except torch.cuda.OutOfMemoryError as e:
                assert "Tried to allocate" in str(e)
        try:  # a call seen before still allocates its output, past the card now
            torch.rms_norm(x, (4096,))
            raise SystemExit("a repeated call's output past the card succeeded")
        except torch.cuda.OutOfMemoryError:
            pass
        print("ok")
    """, MSTAR_SIM_MEMORY_GB="80")
    assert out.strip() == "ok"


def test_a_long_run_over_changing_shapes_holds_the_emulators_memory_flat():
    """The emulator's caches are bounded: a model whose shapes change step to step (a TTS, a
    prefill mix) doesn't grow the worker for as long as it serves."""
    out = run_sim("""
        import tracemalloc, torch
        from mstar_sim import cost
        cost.CACHE_ENTRIES = 256  # small, so this run outgrows it
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)

        def step(tokens):
            x = torch.randn(tokens, 1024, device="cuda", dtype=torch.bfloat16)
            torch.rms_norm(torch.nn.functional.linear(x, w) + x, (1024,))

        tracemalloc.start()
        for tokens in range(8, 208):  # until every cache entry is one tracemalloc saw made
            step(tokens)
        before = tracemalloc.get_traced_memory()[0]
        for tokens in range(208, 408):
            step(tokens)
        grown = tracemalloc.get_traced_memory()[0] - before
        assert grown < 500_000, grown  # (bounded, it moves +-140 KB; with no bound, 2.9 MB)
        print("ok")
    """)
    assert out.strip() == "ok"


def test_small_dev_shm_answers_every_statvfs_form():
    """``--shm-mb``: ``os.statvfs`` of ``/dev/shm`` by str, bytes or path is the small
    tmpfs, indexable like the real result; a file descriptor is the real call's."""
    out = run_sim("""
        import os, pathlib
        for path in ("/dev/shm", b"/dev/shm/", pathlib.Path("/dev/shm")):
            st = os.statvfs(path)
            assert st.f_blocks * st.f_frsize == 64 << 20 and st[2] == st.f_blocks, (path, st)
        fd = os.open("/dev/shm", os.O_RDONLY)
        assert os.statvfs(fd).f_blocks == os.fstatvfs(fd).f_blocks
        print("ok")
    """, MSTAR_SIM_SHM_MB="64")
    assert out.strip() == "ok"


def test_mps_daemon_runs_its_lifecycle_on_the_drivers_stand_in(tmp_path):
    """mstar's own MPS daemon, under the launcher: the control tool comes up, answers while
    it runs, quits on ``quit``, and fails a command once it is gone."""
    script = tmp_path / "mps.py"
    script.write_text(textwrap.dedent("""
        from mstar.conductor.mps import MpsDaemon
        daemon = MpsDaemon()
        daemon.start(timeout_s=120)  # each Python it starts installs the emulator: slow on a loaded box
        assert daemon._command("get_server_list")
        proc = daemon._proc
        daemon.stop()
        assert proc.returncode == 0 and not daemon._command("get_server_list"), proc.returncode
        print("ok")
    """))
    proc = subprocess.run([sys.executable, "-m", "mstar_sim", "--", str(script)], cwd=ROOT, capture_output=True,
                          text=True, timeout=120, check=False, env=dict(os.environ, PYTHONPATH=ROOT))
    assert proc.stdout.strip() == "ok", proc.stderr[-3000:]


def test_memory_tracking_lock_is_reentrant():
    """A tensor freed by a garbage collection that runs while the tracker holds
    its lock releases its storage on the same thread: a plain lock deadlocks."""
    from mstar_sim import memory

    with memory._track_lock:
        assert memory._track_lock.acquire(blocking=False)
        memory._track_lock.release()


def test_tensor_parallel_ranks_meet_at_each_collective():
    """Two ranks ask for NCCL, as mstar does: they get gloo, and an all-reduce of
    a value-less GPU tensor moves nothing but costs its NCCL time. Data moves only
    if every rank holds it."""
    out = run_sim("""
        import tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp

        def rank(r, path):
            dist.init_process_group("nccl", init_method=f"file://{path}", rank=r,
                                    world_size=2, device_id=torch.device("cuda", r))
            torch.cuda.set_device(r)
            x = torch.empty(2**30, dtype=torch.bfloat16, device="cuda")  # 2 GiB, no data
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); dist.all_reduce(x); end.record(); end.synchronize()
            assert dist.get_backend() == "gloo" and start.elapsed_time(end) > 1.0, start.elapsed_time(end)
            # kept values are sent for real: what ranks agreed on was computed, unless a rank's wasn't
            import flashinfer
            from mstar_sim.device import never_computed
            lengths = torch.empty(4, dtype=torch.long, device="cuda")
            dist.all_gather_into_tensor(lengths, torch.arange(2, device="cuda") + 2 * r)
            assert lengths.tolist() == [0, 1, 2, 3] and not never_computed(lengths._data)
            gathered = torch.zeros(4, 4, device="cuda")
            dist.all_gather_into_tensor(gathered, flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1))
            assert never_computed(gathered._data)
            # a value one rank never computed taints what every rank received
            part = torch.ones(1, 4, device="cuda")
            whole = torch.zeros(2, 4, device="cuda")
            dist.all_gather_into_tensor(whole, flashinfer.top_k_renorm_probs(part, 1) if r else part)
            assert never_computed(whole._data)
            # a broadcast reads only its root's tensor: a peer's stale buffer leaves the root's as it
            # was and is overwritten from it
            stale = flashinfer.top_k_renorm_probs(torch.ones(1, 4, device="cuda"), 1).view(4)
            cast = torch.arange(4, device="cuda").float() if r == 0 else stale
            dist.broadcast(cast, src=0)
            assert not never_computed(cast._data) and cast.tolist() == [0, 1, 2, 3]
            # it reads its source and writes its destination: a whole write of computed values
            # ends the fact there, and what didn't move leaves a source computed
            src = torch.arange(4, device="cuda").float()
            dist.all_gather_into_tensor(gathered.view(-1)[:8], src)  # part of it: the rest is as it was
            assert never_computed(gathered._data) and not never_computed(src._data)
            stale = flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1)
            dist.all_gather_into_tensor(stale.view(-1), src)
            assert not never_computed(stale._data) and stale.view(-1).tolist() == [0, 1, 2, 3] * 2
            dist.all_gather_into_tensor(torch.empty(1 << 14, device="cuda")[:8], src)
            assert not never_computed(src._data)
            # an uneven all-to-all held on rank 0 and value-less on rank 1 moves nothing on either
            sends, receives = [[2048, 2048], [2048, 4096]][r], [2048, [2048, 4096][r]]
            out = torch.zeros(sum(receives), device="cuda")
            dist.all_to_all_single(out, torch.ones(sum(sends), device="cuda"), receives, sends)
            assert never_computed(out._data) if r == 0 else out._data.device.type == "meta"
            # point to point moves only if both ranks hold values: a value-less send against a
            # kept receive moves nothing, where the receive alone would wait for it forever
            kept, valueless = torch.zeros(4, device="cuda"), torch.empty(1 << 14, device="cuda")[:4] * 2
            if r == 0:
                dist.send(valueless, dst=1)
                dist.isend(valueless, dst=1).wait()
            else:
                assert dist.recv(kept, src=0) == 0 and never_computed(kept._data)
                dist.irecv(kept, src=0).wait()
                assert never_computed(kept._data)
            # and a kept send's payload against a value-less receive has nowhere to land
            if r == 0:
                dist.send(torch.ones(4, device="cuda"), dst=1)
            else:
                dist.recv(valueless, src=0)
            # kept on both, each isend before its irecv, whichever way the peer is named
            mine, theirs = torch.full((4,), float(r), device="cuda"), torch.zeros(4, device="cuda")
            sends = dist.isend(mine, 1 - r), dist.isend(mine * 3, group_dst=1 - r, group=dist.group.WORLD, tag=7)
            late = torch.zeros(4, device="cuda")
            recvs = dist.irecv(theirs, 1 - r), dist.irecv(late, 1 - r, tag=7)
            for work in (*recvs[::-1], *sends):  # the waits needn't follow the posts
                work.wait()
            assert theirs.tolist() == [1.0 - r] * 4 and late.tolist() == [3.0 - 3 * r] * 4
            assert not never_computed(theirs._data)
            # NCCL has no receive from any source (gloo, which stands in for it, has)
            for receive in (dist.recv, dist.irecv):
                try:
                    receive(theirs)
                    raise SystemExit("a receive from any source was accepted")
                except NotImplementedError as e:
                    assert "does not support recvAnysource" in str(e) and e._mstar_sim_gpu
            # tags pair apart from the order they're posted in, even when only one payload moves
            a, b = torch.zeros(4, device="cuda"), torch.zeros(4, device="cuda")
            if r == 0:
                works = dist.isend(valueless, dst=1, tag=1), dist.isend(mine + 2, dst=1, tag=2)
            else:
                works = dist.irecv(b, src=0, tag=2), dist.irecv(a, src=0, tag=1)
            for work in works:
                work.wait()
            assert r == 0 or (b.tolist() == [2.0] * 4 and never_computed(a._data) and not never_computed(b._data))
            # captured, nothing moves, at each replay
            y, graph = torch.zeros(4, device="cuda"), torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y.copy_(src * 2)
                dist.all_reduce(y)
                dist.isend(y, 1 - r).wait()
                assert dist.recv(y, 1 - r) == dist.recv(y, group_src=1 - r) == 1 - r
            graph.replay()
            assert never_computed(y._data)
            # a rank's wait for a late peer is a host wait, not the emulator's own time
            import time
            from mstar_sim import clock
            emulator = clock.STATS["emulator_s"]
            time.sleep(r)
            dist.all_reduce(y)
            assert clock.STATS["emulator_s"] - emulator < 0.5, clock.STATS
            time.sleep(r)
            dist.recv(kept, src=1) if r == 0 else dist.send(mine, dst=0)
            assert clock.STATS["emulator_s"] - emulator < 0.5, clock.STATS
            # so is gloo's transfer of held values, which NCCL's enqueue doesn't wait for: a peer late
            # to it doesn't move mstar's clock. It spins, as gloo can, for a thread the emulator
            # sees on a CPU, where a sleep would be parked (a wait it can't see) with no help
            if r == 0:
                def late():
                    end = clock._real() + 1
                    while clock._real() < end:
                        pass

                pg_broadcast = dist.ProcessGroup.broadcast
                dist.ProcessGroup.broadcast = lambda *args: (late(), pg_broadcast(*args))[1]
            before = clock.now()
            dist.broadcast(kept, src=0)
            assert clock.now() - before < 0.5, clock.now() - before
            dist.destroy_process_group()

        if __name__ == "__main__":
            mp.spawn(rank, args=(tempfile.mkdtemp() + "/group",), nprocs=2)
            print("ok")
    """)
    assert out.strip().endswith("ok")
    # a TP x SP mesh, its groups made as init_dist makes them: a graph's replay meets each
    # group on the collectives over that group's ranks, though its peers hold other groups
    out = run_sim("""
        import tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp

        def rank(r, path):
            dist.init_process_group("nccl", init_method=f"file://{path}", rank=r, world_size=4)
            torch.cuda.set_device(r)
            tps = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
            tp = tps[r // 2]
            sp = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])][r % 2]
            x, y = torch.empty(1 << 14, device="cuda"), torch.empty(1 << 14, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                dist.all_reduce(x, group=tp)
                dist.all_to_all_single(y, x, group=sp)
            graph.replay()
            # a reduce writes only its destination's: rank 1's stale tensor taints rank 0's sum, and the
            # others' buffers stay as they were (gloo's reduce can leave partial sums in them)
            from mstar_sim.device import never_computed, set_never_computed
            # a ring of isends and irecvs: no rank waits for its peer to post
            nxt, prv = torch.full((4,), float(r), device="cuda"), torch.zeros(4, device="cuda")
            for work in (dist.isend(nxt, (r + 1) % 4), dist.irecv(prv, (r - 1) % 4)):
                work.wait()
            assert prv.tolist() == [float((r - 1) % 4)] * 4
            # in a group: its rank names the peer, and a receive returns the global one
            if r % 2:
                dist.send(nxt, group_dst=0, group=tp)
            else:
                assert dist.recv(prv, group_src=1, group=tp) == r + 1 and prv.tolist() == [r + 1.0] * 4
            # on a group this rank isn't in nothing runs, and torch answers
            outside = tps[1 - r // 2]
            assert dist.irecv(prv, group_src=0, group=outside) is None
            assert dist.recv(prv, group_src=0, group=outside) == -1
            assert dist.isend(prv, group_dst=0, group=outside) is None
            assert dist.all_reduce(prv, group=outside) is None
            # what a call's thread raises reaches its wait, not just stderr
            from mstar_sim.dist import _Posted
            failed = _Posted(lambda: 1 // 0)
            try:
                failed.wait()
                raise SystemExit("a failed call reported success")
            except ZeroDivisionError:
                assert not failed.is_success()
            summed = torch.full((4,), r + 1.0, device="cuda")
            set_never_computed(summed._data, r == 1)
            dist.reduce(summed, dst=0)
            assert never_computed(summed._data) == (r < 2), r
            assert summed.tolist() == [[10.0] * 4, [r + 1.0] * 4][r > 0], summed.tolist()
            # a fused all-reduce's workspace, on ranks that hold more groups than their peers (only
            # TP 0-1 takes the fused path): its channel meets eagerly and replayed, with its group
            from flashinfer import comm
            spaces = [comm.create_allreduce_fusion_workspace(world_size=2, rank=r % 2, group=tp)] if r < 2 else []
            spaces.append(comm.create_allreduce_fusion_workspace(world_size=2, rank=r // 2, group=sp))
            fused = dict(pattern=comm.AllReduceFusionPattern.kAllReduce, output=torch.empty(8, 64, device="cuda"))
            # with its group's timeout, which mstar shortens after load
            import datetime
            dist.distributed_c10d._set_pg_timeout(datetime.timedelta(minutes=7), sp)
            comm.allreduce_fusion(torch.empty(8, 64, device="cuda"), spaces[-1], **fused)
            assert spaces[-1].group.options._timeout == datetime.timedelta(minutes=7)
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
                comm.allreduce_fusion(torch.empty(8, 64, device="cuda"), spaces[-1], **fused)
                dist.all_reduce(x, group=sp)
            graph.replay()
            for space in spaces:
                space.destroy()
            dist.destroy_process_group()

        if __name__ == "__main__":
            mp.spawn(rank, args=(tempfile.mkdtemp() + "/group",), nprocs=4)
            print("ok")
    """)
    assert out.strip().endswith("ok")


@pytest.mark.parametrize("case", ["mismatch", "graph_vs_eager", "bucket", "extra_replay", "skip", "fused_skip",
                                  "send_size", "recv_skip", "held", "held_debug"])
def test_ranks_whose_collectives_dont_pair_abort(case):
    """A rank whose collective doesn't pair with its peer's (a barrier against an all-reduce,
    a graph replayed against eager calls or against one that captured another size) aborts
    with both ranks' descriptors; one whose peer never comes (to a collective, or to a fused
    all-reduce on its own channel) aborts at the group's timeout, as NCCL's watchdog does, or
    hangs in a replay. So for point to point: a receive of another size, whose rank has both
    descriptors, and a receive with no sender. A tensor gloo's worker never lets go of aborts at
    the timeout too, with torch's debug wrapper around the backend or not."""
    out = run_sim(f"""
        import datetime, os, tempfile, time, torch, torch.distributed as dist, torch.multiprocessing as mp

        def rank(r, path):
            os.dup2(1, 2)
            dist.init_process_group("nccl", init_method=f"file://{{path}}", rank=r, world_size=2)
            # short for the collectives only: a rank that starts late on a loaded box still joins
            dist.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=5), dist.group.WORLD)
            if "{case}" == "held_debug":  # torch's setter doesn't see through its wrapper
                gloo = dist.group.WORLD._get_backend(torch.device("cpu")).wrapped_pg
                gloo._set_default_timeout(datetime.timedelta(seconds=5))
            torch.cuda.set_device(r)
            x = torch.empty(1 << 14, device="cuda")
            if "{case}" == "fused_skip":
                from flashinfer import comm
                space = comm.create_allreduce_fusion_workspace(world_size=2, rank=r)
            if "{case}" == "mismatch":
                dist.all_reduce(x) if r else dist.barrier()
            elif "{case}" == "send_size":  # value-less sides: gloo's own size check never sees it
                dist.recv(x, src=0) if r else dist.send(x[:8] * 2, dst=1)
            elif "{case}" == "recv_skip":
                dist.recv(x, src=1) if r == 0 else time.sleep(7)
            elif "{case}" in ("held", "held_debug"):  # autograd holds it, as a gloo that kept its work would
                from mstar_sim.dist import _let_go, _waiting
                held = torch.ones(1, requires_grad=True) * 2
                kept = held * held
                if r == 0:
                    with _waiting():  # where the clock stands still: the timeout is real time
                        _let_go(dist.group.WORLD, held)
                else:
                    time.sleep(7)
            elif "{case}" in ("graph_vs_eager", "bucket", "extra_replay"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    dist.all_reduce(x[: 1 << 13] if r and "{case}" == "bucket" else x)  # a wrong bucket
                graph.replay()
                if "{case}" == "graph_vs_eager":  # the same all-reduces, which NCCL would pair
                    graph.replay() if r else dist.all_reduce(x)
                elif "{case}" == "extra_replay" and r:
                    open(path + ".again", "w").close()
                    graph.replay()
            elif r == 0:
                if "{case}" == "skip":
                    dist.all_reduce(x)
                else:
                    comm.allreduce_fusion(x, space, pattern=comm.AllReduceFusionPattern.kAllReduce, output=x)
            else:
                time.sleep(7)
            print(f"rank {{r}} passed", flush=True)

        if __name__ == "__main__":
            path = tempfile.mkdtemp() + "/group"
            ctx = mp.spawn(rank, args=(path,), nprocs=2, join=False)
            ctx.processes[0].join()
            if "{case}" == "extra_replay":
                while not os.path.exists(path + ".again"):
                    time.sleep(0.1)
                ctx.processes[1].join(10)  # past the group's timeout: the replay hangs
            else:
                ctx.processes[1].join()
            print("exits", [p.exitcode for p in ctx.processes], flush=True)
            for p in ctx.processes:
                p.kill()
    """, **({"TORCH_DISTRIBUTED_DEBUG": "DETAIL"} if case == "held_debug" else {}))
    if case in ("skip", "fused_skip", "recv_skip", "held", "held_debug"):
        assert "exits [-6, 0]" in out, out
    elif case == "send_size":
        assert "exits [0, -6]" in out and "numel=16384" in out and "numel=8" in out, out
    elif case == "extra_replay":
        assert "exits [0, None]" in out and "rank 1 passed" not in out, out
    else:
        assert "exits [-6, -6]" in out and "rank 0: " in out and "rank 1: " in out, out  # each rank's row
        assert "passed" not in out, out


def test_fast_mode_skips_device_time_but_keeps_host_semantics():
    """--fast: a host read after a long kernel doesn't wait, graphs still capture
    and replay, and a host read during capture is still refused."""
    out = run_sim("""
        import time, torch
        a = torch.randn(32768, 32768, device="cuda", dtype=torch.bfloat16)
        t0 = time.perf_counter()
        for _ in range(50):
            a @ a  # ~5 s of simulated H100 time; the host work is ~5 ms, so load can't reach the bound
        torch.cuda.synchronize()
        assert time.perf_counter() - t0 < 1.0, time.perf_counter() - t0
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


def run_pytest(cwd, *args, emulated: bool = False, **env: str) -> str:
    """pytest's output for ``args`` in ``cwd``, on the emulated GPU when ``emulated``."""
    if emulated:
        env = dict(env, MSTAR_SIM="1", PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "mstar_sim", "_site"), ROOT]))
    env = dict(caller_env(), **{"PYTHONPATH": ROOT, **env})
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300, check=False)
    return proc.stdout


def test_plugin_files_gpu_errors_as_mstar_failures_and_missing_stand_ins_as_gaps(tmp_path):
    """What the emulator raises as CUDA would (a host read during capture) is mstar's
    failure; a name an emulated object lacks is the emulator's gap; mstar's own error
    stays its own, whatever library it names. An assert is a value check, the one failure
    the plugin expects, only once the test has read a never-computed value: not for making a
    scratch buffer, or exporting a storage. A pass isn't reported for a test whose own code
    read a never-computed value: that check can't answer."""
    (tmp_path / "test_cases.py").write_text(textwrap.dedent("""
        import pytest, torch
        def test_host_read_in_capture():
            pos = torch.arange(4, device="cuda")
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                pos.sum().item()
        def test_stand_in_lacks_a_name():
            torch.cuda.Stream().no_such_method()
        def test_own_error_names_a_library():
            raise AttributeError("the flashinfer wrapper has no plan")
        def test_tensor_lacks_an_attribute():
            torch.ones(2, device="cuda").no_such_attribute
        def test_scratch_buffer_filled_by_rows():
            buf = torch.empty(4, device="cuda")
            buf[:2] = 1.0
            assert buf.tolist() == [2.0] * 4
        def test_exported_storage():
            t = torch.arange(4, device="cuda")
            t.untyped_storage()._share_cuda_()
            assert t.sum().item() == 7
        def test_captured_stand_in_read_after_a_replay():
            import flashinfer
            graph, probs = torch.cuda.CUDAGraph(), torch.ones(2, 4, device="cuda")
            with torch.cuda.graph(graph):
                rows = flashinfer.top_k_renorm_probs(probs, 1) * 2
            graph.replay()
            torch.testing.assert_close(rows, torch.zeros(2, 4, device="cuda"))
        def test_never_computed_write():
            kept = torch.arange(4, device="cuda")
            kept[:2] = torch.empty(1 << 21, dtype=torch.long, device="cuda")[:2]
            assert kept.tolist() == [5, 6, 2, 3]
        def test_parity_of_two_never_computed_values():
            import flashinfer
            rows = flashinfer.top_k_renorm_probs(torch.empty(2, 4, device="cuda"), 1)  # small: kept
            torch.testing.assert_close(rows * 2, torch.zeros(2, 4, device="cuda"))
        def test_mstar_reads_a_sampled_token_and_the_test_checks_a_shape():
            import flashinfer
            tokens = flashinfer.sampling_from_probs(torch.empty(2, 512, device="cuda"))
            helper = eval(compile("lambda t: t.tolist()", "/mstar/worker.py", "eval"))
            assert len(helper(tokens)) == 2
        def test_lifecycle_assert_after_mstar_reads_a_sampled_token():
            import flashinfer
            tokens = flashinfer.sampling_from_probs(torch.empty(2, 512, device="cuda"))
            eval(compile("lambda t: t.tolist()", "/mstar/worker.py", "eval"))(tokens)
            assert len([]) == 1
        def test_shape_mistake_on_cuda_tensors():
            torch.ones(3, device="cuda") + torch.ones(4, device="cuda")
        def test_assert_after_a_never_computed_mask_of_a_value_less_tensor():
            import flashinfer
            mask = flashinfer.top_k_renorm_probs(torch.ones(2, 2048, device="cuda"), 1).reshape(-1) >= 0
            assert torch.empty(4096, 16, device="cuda")[mask].shape[0] == 1
        def test_index_past_the_end():
            torch.arange(3, device="cuda")[7]
        def test_shape_mistake_in_a_collective(tmp_path):
            import torch.distributed as dist
            dist.init_process_group("nccl", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
            try:
                dist.all_gather_into_tensor(torch.zeros(3, device="cuda"), torch.zeros(2, device="cuda"))
            finally:
                dist.destroy_process_group()
        def test_shape_mistake_inside_a_stand_in():
            stand_in = eval(compile("lambda a, b: a + b", "/mstar_sim/fake_libs/x.py", "eval"))
            stand_in(torch.ones(3, device="cuda"), torch.ones(4, device="cuda"))
        def test_index_into_what_a_never_computed_mask_selected():
            import flashinfer
            flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1).gt(0.5).nonzero()[0]
        def test_torch_refuses_a_stand_ins_zeros():
            import flashinfer
            torch.multinomial(flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1), 1)
        def test_torch_refuses_a_singular_matrix_of_a_stand_ins_zeros():
            import flashinfer
            torch.inverse(flashinfer.top_k_renorm_probs(torch.ones(4, 4, device="cuda"), 1))
        def test_torch_refuses_an_integer_divide_by_a_stand_ins_zeros():
            import flashinfer
            zeros = flashinfer.top_k_renorm_probs(torch.ones(4, 4, device="cuda"), 1).long()
            torch.full((4, 4), 8, device="cuda") // zeros
        def test_shape_mistake_on_a_stand_ins_output():
            import flashinfer
            flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1) + torch.ones(3, device="cuda")
        def test_dtype_mistake_on_a_stand_ins_output():
            import flashinfer
            flashinfer.top_k_renorm_probs(torch.ones(4, 4, device="cuda"), 1).half() @ torch.ones(4, 4, device="cuda")
        def test_cpu_kernel_limit():
            torch.fft.rfft(torch.ones(8, device="cuda", dtype=torch.float16))
        def test_mstars_own_raise_after_a_never_computed_mask():
            import flashinfer
            rows = flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1)
            rows[rows >= 0]
            raise RuntimeError("mstar's own lifecycle error")
        def test_assert_after_an_integer_index_of_a_never_computed_value():
            import flashinfer
            rows = flashinfer.top_k_renorm_probs(torch.ones(2, 4, device="cuda"), 1)
            assert rows[torch.tensor([0], device="cuda")].shape == (2, 4)
        def test_inplace_on_an_expanded_stand_ins_output():  # an overlap, which a GPU refuses too
            import flashinfer
            flashinfer.top_k_renorm_probs(torch.ones(4, 4, device="cuda"), 1)[:, :1].expand(4, 4).add_(1)
        def test_passing_value_check():
            assert (torch.arange(3, device="cuda") + 1).tolist() == [1, 2, 3]
        def test_dynamo_traces_a_device():
            import torch._dynamo
            torch._dynamo.explain(lambda x: x + torch.device(0).index)(torch.ones(2, device="cuda"))
    """))
    out = run_pytest(tmp_path, "-p", "mstar_sim.pytest_plugin", "-rfxp", "test_cases.py", emulated=True,
                     MSTAR_SIM_FAST="1")
    outcomes = {line.split()[1].split("::")[1]: line.split()[0] for line in out.splitlines()
                if line.startswith(("FAILED", "XFAIL", "PASSED"))}
    assert outcomes == {"test_host_read_in_capture": "FAILED", "test_stand_in_lacks_a_name": "XFAIL",
                        "test_own_error_names_a_library": "FAILED", "test_tensor_lacks_an_attribute": "FAILED",
                        "test_scratch_buffer_filled_by_rows": "FAILED",
                        "test_exported_storage": "FAILED", "test_captured_stand_in_read_after_a_replay": "XFAIL",
                        "test_never_computed_write": "XFAIL", "test_parity_of_two_never_computed_values": "XFAIL",
                        "test_mstar_reads_a_sampled_token_and_the_test_checks_a_shape": "PASSED",
                        "test_lifecycle_assert_after_mstar_reads_a_sampled_token": "FAILED",
                        "test_shape_mistake_on_cuda_tensors": "FAILED",
                        "test_shape_mistake_in_a_collective": "FAILED",
                        "test_index_past_the_end": "FAILED",
                        "test_assert_after_a_never_computed_mask_of_a_value_less_tensor": "XFAIL",
                        "test_shape_mistake_inside_a_stand_in": "XFAIL",
                        "test_index_into_what_a_never_computed_mask_selected": "XFAIL",
                        "test_mstars_own_raise_after_a_never_computed_mask": "FAILED",
                        "test_torch_refuses_a_stand_ins_zeros": "XFAIL",
                        "test_torch_refuses_a_singular_matrix_of_a_stand_ins_zeros": "XFAIL",
                        "test_torch_refuses_an_integer_divide_by_a_stand_ins_zeros": "XFAIL",
                        "test_shape_mistake_on_a_stand_ins_output": "FAILED",
                        "test_dtype_mistake_on_a_stand_ins_output": "FAILED",
                        "test_cpu_kernel_limit": "XFAIL",
                        "test_assert_after_an_integer_index_of_a_never_computed_value": "FAILED",
                        "test_inplace_on_an_expanded_stand_ins_output": "FAILED",
                        "test_passing_value_check": "PASSED", "test_dynamo_traces_a_device": "XFAIL"}, out


def test_skipped_in_selects_what_a_plain_run_skipped(tmp_path):
    """``--skipped-in``: the tests a plain CPU run's junit report skipped, in a class or in
    a module skipped at collection, and those whose own skipif checks for a GPU; none
    other it ran."""
    pkg = tmp_path / "test" / "a"
    pkg.mkdir(parents=True)
    (pkg / "test_classes.py").write_text(textwrap.dedent("""
        import os, pytest, torch
        gpu = pytest.mark.skipif(bool(os.environ.get("PLAIN")), reason="needs a GPU")
        class TestC:
            @gpu
            def test_skipped(self): pass
            def test_ran(self): pass
        @pytest.mark.skipif(not torch.cuda.is_available(), reason="")
        def test_checks_for_a_gpu(): pass
        @pytest.mark.skipif(os.name == "nt", reason="posix only")
        def test_other_skipif(): pass
        requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="")
        @requires_cuda
        def test_named_gpu_skipif(): pass
    """))
    (pkg / "test_module.py").write_text(textwrap.dedent("""
        import os, pytest
        if os.environ.get("PLAIN"):
            pytest.skip("needs a GPU", allow_module_level=True)
        def test_one(): pass
    """))

    run_pytest(tmp_path, "--junitxml=plain.xml", "test", PLAIN="1")
    for selection in ("--skipped-in=plain.xml", "--gpu-only"):  # the same tests, from the run or the source
        out = run_pytest(tmp_path, "-p", "mstar_sim.pytest_plugin", selection, "--collect-only", "-q", "test")
        selected = {line for line in out.splitlines() if "::" in line}
        assert selected == {"test/a/test_classes.py::TestC::test_skipped", "test/a/test_module.py::test_one",
                            "test/a/test_classes.py::test_checks_for_a_gpu",
                            "test/a/test_classes.py::test_named_gpu_skipif"}, (selection, out)


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
        from mstar.engine.resources.attn import wrappers  # mstar's pin moves, so must the stand-in
        assert flashinfer.__version__ in wrappers.DIRECT_PLAN_FLASHINFER_VERSIONS, flashinfer.__version__

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
