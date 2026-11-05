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
        import subprocess, sys, torch
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
        for past in (lambda: torch.zeros(2, device="cuda:8"), lambda: x.to("cuda:8"),  # 8 emulated GPUs
                     lambda: torch.cuda.set_device(8), lambda: torch.cuda.Stream(8)):
            try:
                past()
                raise SystemExit("an index past the emulated devices was accepted")
            except torch.AcceleratorError:
                pass
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
        import time
        g5 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g5):
            clock.on_replay(lambda: time.sleep(1))
            a @ a
        took = timed(g5.replay, queued=0.0)  # idle: a late launch would show
        assert took < 0.5, took
        # nor is what it does around a stand-in kernel
        from mstar_sim import device
        count_storages, device.count_storages = device.count_storages, lambda out: time.sleep(1)
        before = clock.vnow()
        clock.kernel_fn("test.idle")(lambda: None)()
        assert clock.vnow() - before < 0.5, clock.vnow() - before
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


def test_an_event_on_an_idle_stream_holds_no_thread_back():
    """An event is complete when its stream drains, not at the recording thread's clock:
    a waiter whose host time lags (the emulator's own time on its thread) isn't delayed."""
    out = run_sim("""
        import threading, time, torch
        from mstar_sim import clock
        event, stream = torch.cuda.Event(), torch.cuda.Stream()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with clock.computing():
            time.sleep(1)
        recorder = threading.Thread(target=event.record)
        recorder.start(); recorder.join()
        with torch.cuda.stream(stream):
            start.record(); stream.wait_event(event); torch.empty(8, device="cuda").add_(1); end.record()
        end.synchronize()
        assert start.elapsed_time(end) < 500, start.elapsed_time(end)
        print("ok")
    """)
    assert out.strip() == "ok"


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
        big = torch.empty(total // 6, device="cuda")  # two thirds of the card
        del big
        again = torch.empty(total // 6, device="cuda")  # reuses the cached block
        for _ in range(20):  # a retry after out of memory, as mstar's workspace growth does
            try:
                torch.empty(total // 6, device="cuda")
                raise SystemExit("an allocation past the card succeeded")
            except torch.cuda.OutOfMemoryError as e:
                assert "Tried to allocate" in str(e)
        print("ok")
    """, MSTAR_SIM_MEMORY_GB="80")
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
        import os, tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp

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
            # captured, nothing moves, at each replay
            y, graph = torch.zeros(4, device="cuda"), torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y.copy_(src * 2)
                dist.all_reduce(y)
            graph.replay()
            assert never_computed(y._data)
            # a rank's wait for a late peer is a host wait, not the emulator's own time
            import time
            from mstar_sim import clock
            emulator = clock.STATS["host_compute"]
            time.sleep(r)
            dist.all_reduce(y)
            assert clock.STATS["host_compute"] - emulator < 0.5, clock.STATS
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
            tp = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])][r // 2]
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


@pytest.mark.parametrize("case", ["mismatch", "graph_vs_eager", "bucket", "extra_replay", "skip", "fused_skip"])
def test_ranks_whose_collectives_dont_pair_abort(case):
    """A rank whose collective doesn't pair with its peer's (a barrier against an all-reduce,
    a graph replayed against eager calls or against one that captured another size) aborts
    with both ranks' descriptors; one whose peer never comes (to a collective, or to a fused
    all-reduce on its own channel) aborts at the group's timeout, as NCCL's watchdog does, or
    hangs in a replay."""
    out = run_sim(f"""
        import datetime, os, tempfile, time, torch, torch.distributed as dist, torch.multiprocessing as mp

        def rank(r, path):
            os.dup2(1, 2)
            dist.init_process_group("nccl", init_method=f"file://{{path}}", rank=r, world_size=2)
            # short for the collectives only: a rank that starts late on a loaded box still joins
            dist.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=5), dist.group.WORLD)
            torch.cuda.set_device(r)
            x = torch.empty(1 << 14, device="cuda")
            if "{case}" == "fused_skip":
                from flashinfer import comm
                space = comm.create_allreduce_fusion_workspace(world_size=2, rank=r)
            if "{case}" == "mismatch":
                dist.all_reduce(x) if r else dist.barrier()
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
    """)
    if case in ("skip", "fused_skip"):
        assert "exits [-6, 0]" in out, out
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
    env = dict(os.environ, **{"PYTHONPATH": ROOT, **env})
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300, check=False)
    return proc.stdout


def test_plugin_files_gpu_errors_as_mstar_failures_and_missing_stand_ins_as_gaps(tmp_path):
    """What the emulator raises as CUDA would (a host read during capture) is mstar's
    failure; a name an emulated object lacks is the emulator's gap; mstar's own error
    stays its own, whatever library it names. An assert is a value check, the one failure
    the plugin expects, only once the test has read a never-computed value: not for making a
    scratch buffer, or exporting a storage. A pass isn't reported for a test whose own code
    read a never-computed value: that check can't answer. A test listed as needing what a
    stand-in lacks is a gap, and fails the run once it passes."""
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
        @pytest.mark.parametrize("fails", [True, False])
        def test_a_sealed_buffer_reports_flashinfers_number(fails):
            assert not fails
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
                        "test_a_sealed_buffer_reports_flashinfers_number[True]": "XFAIL",
                        "test_a_sealed_buffer_reports_flashinfers_number[False]": "FAILED",
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
