"""A run on the device (``mstar.engine.device_run``): when a WHILE run ends, and
what its body does each iteration. The node itself needs CUDA; this drives the
same body through the rule in torch, with a backend that loops on its condition."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine import accelerator_graph_runner as runner_mod
from mstar.engine.accelerator_graph_backend import GraphConditional
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.device_run import (
    RUN_COUNTS,
    RUN_DESCRIPTORS,
    RUN_RAN,
    RUN_REPLAYS,
    RUN_SKIPPED,
    CudaWhile,
    DeviceRun,
    IterationWindow,
    RunAsk,
    RunState,
    TokenStream,
)
from mstar.engine.resources.iteration_tables import IterationTables, tables_key
from mstar.engine.resources.step import BucketKey, SlotLease
from mstar.model.submodule_base import BatchedModelOutput, CarryPair
from mstar.worker.step_repeats import RepeatConfig

CPU = torch.device("cpu")
WALK = "decode"
EOS = 9


class _Loop:
    """What ``CudaWhile`` does, off the GPU: ``body`` until the condition clears."""

    def __init__(self, on_iteration=lambda i: None):
        self.body = None
        self.capturing = lambda body: None
        self.on_iteration = on_iteration
        self.state = None

    def _load_kernels(self):
        pass

    def start(self, state, stream):
        skip = int(state.limit) > 1 and int(state.generation[0]) <= int(state.yield_until[0])
        state.started[0] = RUN_SKIPPED if skip else RUN_RAN
        state.continuation[0] = not skip

    def stop(self, state, tokens, stream):
        """The test double of the ``run_stop`` kernel, which only CUDA can run."""
        self.state = state
        if state.descriptors is not None:
            identities = torch.where(state.marks, 0, state.identities)
            state.descriptors[:, int(state.index)] = identities
            if state.stream is not None:
                state.stream.identities[:, int(state.index)] = identities
        state.counts += ~state.marks
        token = None if tokens is None else tokens.reshape(len(state.marks), -1)[:, 0]
        if token is not None:
            state.marks |= (token.unsqueeze(-1) == state.stop_ids).any(-1) & ~state.immune
        if state.stream is not None:
            state.stream.tokens[:, int(state.index)] = token
            state.stream.landed[0] = int(state.index) + 1
        state.index += 1
        done = int(state.marks.sum())
        state.continuation[0] = (
            done < len(state.marks) and int(state.index) < int(state.limit)
            and int(state.generation[0]) > int(state.yield_until[0])
            and (int(state.refill) == 0 or done < int(state.refill))
        )

    def node_refusal(self, graph):
        return None

    def compose(self, ops):
        def replay(sequence):
            for op in sequence:
                if isinstance(op, GraphConditional):
                    iteration = 0
                    while op.predicate.item():
                        self.on_iteration(iteration)
                        replay(op.body)
                        iteration += 1
                        if op.kind == "if":
                            break
                else:
                    op.replay()
        return SimpleNamespace(replay=lambda: replay(ops))


def _runner(
    monkeypatch, loop, tokens, bs=2, limit=6, refill_at=None, carries=None, stream_tokens=False, counted=False,
    progress=False, membership=None, unmanaged=False, ask=None,
):
    """A runner holding a captured run whose forward samples ``tokens[iteration]``."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._device, runner._memory_pool, runner._autocast_dtype = CPU, None, None
    runner._submodule_name, runner._capture_stream, runner._resident = "n", None, None
    runner._run_asks, runner._run_failures, runner._generator_walks = {}, {}, set()
    runner._member = ""
    runner._resources = {tables_key("n"): IterationTables(CPU)}
    runner._submodule = SimpleNamespace(ITERATION_INPUTS={"cond": SimpleNamespace(capture_fill=0)})
    runner._graph_backend = SimpleNamespace(
        while_refusal=lambda: None, while_runtime=lambda: loop,
        raw_stream=lambda: 0, synchronize=lambda: None, compose=loop.compose,
    )
    resident = layout = None
    if progress:
        from mstar.engine.resources.loopback_rows import LoopbackRows
        from mstar.graph.loop_indices import LoopLayout

        layout = LoopLayout("decode", (), ())
        capacity = bs if membership is None else max(bs, membership.capacity)
        rows = runner._rows = LoopbackRows(CPU, capacity=capacity)
        for rid in range(1, capacity + 1):
            rows.add(rid)
        buffers = SimpleNamespace(rows=torch.arange(1, bs + 1), slots=torch.arange(1, bs + 1))
        planner = runner._resident = SimpleNamespace(
            store=torch.tensor([0] + [rows._row_of[rid] for rid in range(1, capacity + 1)]),
            table=lambda: SimpleNamespace(stride=1, col=0), covers=lambda bucket: True,
            drop_rows=lambda bucket, slot: buffers, max_slots=capacity + 1,
            bindings={},
        )
        rows.resident_binding().bind(planner, "rows")
        resident = SimpleNamespace(bucket=BucketKey(WALK, bs, bs), slot=0, buffers=buffers, slots=buffers.slots)
        runner.test_resource = torch.zeros(capacity + 1)
    runner.ask_run(WALK, ask or RunAsk(None if counted else "new_token", frozenset() if counted else frozenset({EOS}),
                                       False, limit, carries or {}, ("frame",), ("cond",),
                                       refill_at, stream_tokens, layout, membership))
    static = {"cond": torch.zeros(bs)}
    if membership is None or unmanaged:
        static["h"] = torch.zeros(bs)
    calls = []

    def forward(in_run=False, select_inputs=None):
        if resident is not None:
            runner._rows.resident_binding().gather(resident)
            static["cond"].fill_(-55)  # input selection must follow resource gathering
        if select_inputs is not None:
            select_inputs()
        i = min(len(calls), len(tokens) - 1)
        calls.append(static["cond"].clone())
        token = static["cond"].long() if progress else torch.tensor(tokens[i])
        if resident is not None:
            runner.test_resource.index_add_(0, resident.slots, static["cond"])
        return BatchedModelOutput(
            check_stop_buffers={} if counted else {"new_token": token},
            row_outputs={"frame": token.float() * 2, "h": static.get("h", torch.zeros(bs)) + 1},
        )

    def capture(body, *args, **kwargs):
        if loop.body is None:
            loop.body = body
            loop.capturing(body)
        out = body()
        calls.clear()  # capture records; it doesn't run
        if "h" in static:
            static["h"].zero_()
        return SimpleNamespace(raw_cuda_graph=lambda: 7, replay=body), out

    monkeypatch.setattr(runner_mod, "capture_into_graph", capture)
    spec = SimpleNamespace(bucket=BucketKey(WALK, bs, bs), slot=0, config=SimpleNamespace(get_config_type=lambda: None))
    warmed = SimpleNamespace(run=forward, static_inputs=static, static_input_keys=tuple(static),
                             row_heads={}, output=None, resident=resident)
    run, _, _ = runner._capture_run(spec, warmed)
    if run is not None:
        run.test_inputs = (runner._resources[tables_key("n")], SlotLease(0, spec.bucket), list(range(1, bs + 1)))
    runner._buckets = {spec.bucket: SimpleNamespace(slots=[SimpleNamespace(run=run)])}
    return runner, run, static, calls


def _stage_inputs(run, columns):
    if run.window is None:
        run.stage_inputs(columns)
    else:
        owner, lease, slots = run.test_inputs
        owner.stage_window(lease, slots, columns)


def _run(run, real, repeats, immune=None, refill_at=None, columns=None):
    run.state.stage(real, immune or [False] * real, repeats, refill_at)
    _stage_inputs(run, columns)
    run.launch()
    return run.results(repeats)


def test_a_long_row_beside_a_short_one_runs_to_the_runs_limit(monkeypatch):
    """A done row doesn't end the run for the row beside it."""
    tokens = [[1, 1], [EOS, 2], [3, 3], [4, 4]]
    _, run, _, calls = _runner(monkeypatch, _Loop(), tokens)
    stops, rows, _ = _run(run, 2, 4)
    assert len(calls) == 4 and stops[RUN_REPLAYS].tolist() == [4, 4]
    assert stops[RUN_COUNTS].tolist() == [2, 4]
    assert stops["new_token"].tolist() == [[1, EOS, 3, 4], [1, 2, 3, 4]]
    assert rows["frame"].tolist() == [[2, 18, 6, 8], [2, 4, 6, 8]]
    assert run.state.marks.tolist() == [True, False]


def test_a_run_ends_with_its_last_row_and_padding_keeps_nothing_alive(monkeypatch):
    _, run, _, calls = _runner(monkeypatch, _Loop(), [[1, 0], [EOS, 0], [3, 0]])
    stops, _, _ = _run(run, 1, 5)
    assert len(calls) == 2 and stops[RUN_REPLAYS].tolist() == [2, 2]
    assert stops[RUN_COUNTS].tolist() == [2, 0]


def test_a_stop_from_before_a_run_does_not_end_it_and_ignore_eos_never_marks(monkeypatch):
    _, run, _, calls = _runner(monkeypatch, _Loop(), [[EOS, EOS]])
    _run(run, 2, 3)
    assert len(calls) == 1
    calls.clear()
    stops, _, _ = _run(run, 2, 3, immune=[True, True])
    assert len(calls) == 3 and not run.state.marks.any(), "a new run starts unmarked"


def test_the_hosts_yield_word_ends_a_run_and_its_row_goes_on_in_the_next(monkeypatch):
    """Ready work ends a run at its next iteration; the live row hands on where
    it stands, its carried value intact."""
    holder = {}
    loop = _Loop(on_iteration=lambda i: holder["runner"].yield_runs() if i == 2 else None)
    carries = {"h": CarryPair("h", "h")}
    runner, run, static, calls = _runner(monkeypatch, loop, [[1, 1]], carries=carries)
    holder["runner"] = runner
    stops, _, _ = _run(run, 2, 6)
    assert len(calls) == 3 and stops[RUN_REPLAYS].tolist() == [3, 3]
    assert static["h"].tolist() == [3, 3]
    loop.on_iteration = lambda i: None
    stops, _, _ = _run(run, 2, 2)
    assert stops[RUN_REPLAYS].tolist() == [2, 2], "the next launch is newer than the yield"
    assert static["h"].tolist() == [5, 5]


@pytest.mark.parametrize("limit, ran, begun", [(6, 0, RUN_SKIPPED), (1, 1, RUN_RAN)])
def test_a_run_staged_with_its_word_set_runs_nothing_unless_it_is_one_iteration(monkeypatch, limit, ran, begun):
    """The step behind a run that was asked to end would otherwise hold up what asked
    for one iteration. A step of one iteration always runs: it's the one taking in new
    rows, or the first after a skip, so skips can't follow each other."""
    runner, run, _, calls = _runner(monkeypatch, _Loop(), [[1, 1]])
    run.state.stage(2, [False, False], limit)
    run.state.request_yield()
    run.launch()
    stops, _, _ = run.results(limit)
    assert len(calls) == ran and int(run.state.started[0]) == begun
    assert stops[RUN_REPLAYS].tolist() == [ran, ran]
    assert stops[RUN_COUNTS].tolist() == [ran, ran]


@pytest.mark.parametrize("yield_after_stage", [False, True])
def test_a_yield_names_only_already_staged_runs_on_a_shared_state(monkeypatch, yield_after_stage):
    """N is live while N+1 stages on the same state. A yield before N+1's
    stage must survive for N without cancelling the new row merged into N+1;
    a yield after that stage skips N+1 too. Later identical stages run again."""
    from torch.utils._python_dispatch import TorchDispatchMode

    loop = _Loop()
    runner, run, _, calls = _runner(monkeypatch, loop, [[1, 1]])
    state = run.state
    queued = []
    # CPU stand-in for the device execution stream: stage's device writes
    # wait behind N, while its host writes happen immediately.
    device_words = [
        t for t in vars(state).values()
        if isinstance(t, torch.Tensor) and t is not state.yield_until and t is not state.started
    ]
    pointers = {t.untyped_storage().data_ptr() for t in device_words}

    class QueueDeviceWrites(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            target = kwargs.get("out", args[0] if args else None)
            if (func._schema.is_mutable and isinstance(target, torch.Tensor)
                    and target.untyped_storage().data_ptr() in pointers):
                queued.append((func, args, kwargs))
                return target
            return func(*args, **kwargs)

    def stage_successor(i):
        if i != 0:
            return
        if not yield_after_stage:
            runner.yield_runs()
        with QueueDeviceWrites():
            state.stage(2, [False, False], 4)
        if yield_after_stage:
            runner.yield_runs()

    loop.on_iteration = stage_successor
    _run(run, 1, 4)
    assert len(calls) == 1, "staging the successor must not erase the live run's yield"
    loop.on_iteration = lambda i: None
    for func, args, kwargs in queued:
        func(*args, **kwargs)
    calls.clear()
    run.launch()
    assert len(calls) == (0 if yield_after_stage else 4)
    assert int(state.started[0]) == (RUN_SKIPPED if yield_after_stage else RUN_RAN)
    assert not state.marks.any(), "the successor includes its newly admitted row"
    calls.clear()
    _run(run, 2, 4)  # identical configuration, but a new launch
    assert len(calls) == 4, "a later launch must not inherit an earlier yield"


def test_refill_at_ends_a_run_once_that_share_of_its_rows_is_done(monkeypatch):
    tokens = [[EOS, 1, 1, 1], [0, EOS, 1, 1], [0, 0, 1, 1]]
    _, run, _, calls = _runner(monkeypatch, _Loop(), tokens, bs=4)
    stops, _, _ = _run(run, 4, 6, refill_at=0.5)
    assert stops[RUN_REPLAYS].tolist() == [2] * 4
    assert stops[RUN_COUNTS].tolist() == [1, 2, 2, 2]


def test_each_replay_reads_its_column_of_an_iteration_input(monkeypatch):
    _, run, _, calls = _runner(monkeypatch, _Loop(), [[1, 1]])
    _run(run, 2, 3, columns={"cond": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})
    assert [c.tolist() for c in calls] == [[1, 4], [2, 5], [3, 6]]


@pytest.mark.parametrize(("kinds", "why"), [
    ([0, 1], None),
    ([0, 3], "a host node"),
    ([10], "a memory alloc node"),
    ([7], "a event record node"),
    ([13], "a conditional node"),
], ids=["kernels", "host", "alloc", "event", "if"])
def test_a_step_graph_a_while_body_cant_hold_is_refused(kinds, why):
    import ctypes

    cuda = CudaWhile.__new__(CudaWhile)

    def get_nodes(graph, nodes, count):
        count._obj.value = len(kinds)
        if nodes is not None:
            for i in range(len(kinds)):
                nodes[i] = i + 1
        return 0

    def get_type(node, kind):
        kind._obj.value = kinds[node.value - 1]
        return 1 if kinds[node.value - 1] == 13 else 0

    cuda._rt = SimpleNamespace(cudaGraphGetNodes=get_nodes, cudaGraphNodeGetType=get_type)
    refusal = cuda.node_refusal(ctypes.c_void_p(1).value)
    assert refusal is None if why is None else why in refusal



@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA captured body reuse")
def test_one_run_body_reuses_standalone_and_composed_parents():
    from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
    from mstar.engine.accelerator_graph_runner import capture_into_graph

    backend = get_accelerator_graph_backend(torch.device("cuda:0"))
    cuda, device = backend.while_runtime(), backend.device
    state = RunState(2, "token", {3, 14}, device)
    token = torch.zeros(2, dtype=torch.long, device=device)
    history = torch.zeros((2, 6), dtype=torch.long, device=device)
    offsets = torch.tensor([1, 11], device=device)
    tail = torch.zeros_like(state.index)
    pool = backend.graph_pool_handle()
    cuda._load_kernels()

    def body():
        token.copy_(state.index + offsets)
        history.scatter_(1, state.index.clamp(0, 5).expand(2, 1), token[:, None])
        cuda.stop(state, token, backend.raw_stream())
        tail.copy_(state.index)

    captured = capture_into_graph(body, pool, device, None, keep_graph=True)[0]
    entry = capture_into_graph(lambda: cuda.start(state, backend.raw_stream()), pool, device, None,
                               keep_graph=True)[0]
    standalone = backend.compose((entry, GraphConditional("while", state.continuation, (captured,))))
    # Both parents own fresh conditional handles but reuse the same entry/body
    # and exclusive bound storage. Entry executes once, never inside the loop.
    composed = backend.compose((entry, GraphConditional("while", state.continuation, (
        GraphConditional("if", state.continuation, (captured,)),
    ))))
    try:
        cases = [(standalone, False, 6), (composed, False, 6),
                 (standalone, True, 6), (composed, True, 6),
                 (composed, False, 6), (standalone, False, 6),
                 (standalone, True, 1), (composed, True, 1)]
        for parent, yielded, width in cases:
            state.stage(2, [False, False], width)
            history.zero_()
            tail.zero_()
            if yielded:
                state.request_yield()
            parent.replay()
            backend.synchronize()
            skip = yielded and width > 1
            count = 0 if skip else min(width, 4)
            assert state.counts.tolist() == [min(count, 3), count]
            assert state.index.item() == tail.item() == count
            assert state.continuation.item() == 0
            assert state.started.item() == (RUN_SKIPPED if skip else RUN_RAN)
            assert history.tolist() == [
                list(range(1, count + 1)) + [0] * (6 - count),
                list(range(11, count + 11)) + [0] * (6 - count),
            ]
    finally:
        backend.synchronize()
        composed.reset()
        standalone.reset()


def test_a_run_refuses_a_generator_draw_and_a_walk_with_no_while_node():
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._run_asks, runner._run_failures, runner._generator_walks = {WALK: object()}, {}, {WALK}
    runner._graph_backend = SimpleNamespace(while_refusal=lambda: None)
    spec = SimpleNamespace(bucket=BucketKey(WALK, 1, 1), config=SimpleNamespace(get_config_type=lambda: None))
    assert runner._capture_run(spec, None) == (None, None, None)
    runner._buckets = {}
    assert "torch generator" in runner.run_refusal(WALK)
    runner._graph_backend = SimpleNamespace(while_refusal=lambda: "no node")
    assert runner.run_refusal(WALK) == "backend: no node"


def test_a_runs_tokens_land_as_its_replays_finish_and_a_row_ends_at_its_stop(monkeypatch):
    """What leaves during the run is what the landing would have sent: a row
    that stops in its first column sends nothing after it."""
    seen = []

    def to_stop(rid, column, token):
        return True, token == EOS

    loop = _Loop(on_iteration=lambda i: seen.append(run.state.stream.take(to_stop)))
    runner, run, _, _ = _runner(monkeypatch, loop, [[EOS, 1], [5, 2], [6, 3]], limit=3, stream_tokens=True)
    run.state.stream.begin([10, 11])
    _run(run, 2, 3)
    assert seen == [([], []), ([10, 11], [EOS, 1]), ([11], [2])]
    assert run.state.stream.take(lambda rid, column, token: (True, False)) == ([11], [3])
    assert run.state.stream.sent == {10: 1, 11: 3}


def test_the_worker_sends_a_runs_tokens_for_live_rows_up_to_their_limits():
    """A row that stopped before the wait sends nothing, and an iteration
    limit or an end token ends a row as the landing's stop would."""
    from mstar.model.submodule_base import StopKind, StopPolicy
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    sent = []
    worker._graph_runtime = SimpleNamespace(
        send_tokens=lambda *args: sent.append(args),
        # the runtime's counts, which a speculated step's own metadata lags
        loop_progress=lambda node, rids, walks, loops: ([{1: 8, 4: 0}[rid] for rid in rids], [100] * len(rids)),
    )
    policy = StopPolicy(StopKind.AR, "t", {EOS}, sampler="s")
    worker._streamed = {("n", WALK): (policy, "text", "text")}
    worker._innermost_loop_by_node = {(WALK, "n"): "loop"}

    def info(max_tokens):
        return SimpleNamespace(
            max_tokens=max_tokens, dynamic_loop_iter_counts={"loop": 0},
            resource_configs={"s": SimpleNamespace(ignore_eos=False)},
        )

    stream = TokenStream(3, 3, CPU)
    stream.begin([1, 2, 4])
    stream.tokens.copy_(torch.tensor([[5, 6, 7], [8, 9, 10], [14, EOS, 15]]))
    stream.landed[0] = 3
    done = iter([False, True])
    worker._stream_tokens(SimpleNamespace(
        node_name="n", graph_walk=WALK,
        node_batch=SimpleNamespace(
            token_stream=stream, request_ids=[1, 4], completion_event=SimpleNamespace(query=lambda: next(done)),
            per_request_info_wrapped={1: info(10), 4: info(9)},
        ),
    ))
    # row 1 is at 8 of 10
    assert sent == [([1, 4, 1, 4], [5, 14, 6, EOS], "text", "text")] and stream.sent == {1: 2, 4: 2}


@pytest.mark.parametrize("routed, withholding, follower, why", [
    ({"hidden"}, {}, False, "node: its tokens don't ride inline"),
    (set(), {("n", WALK): object()}, False, "node: it withholds an output, so its landing doesn't ride inline"),
    (set(), {}, True, "TP: the leader sends the tokens"),
    (set(), {}, False, None),
])
def test_only_a_landing_that_rides_inline_streams(routed, withholding, follower, why):
    """Only the inline landing skips what a run sent, so any other would send
    it twice; a TP follower's landing sends nothing, so its stream would."""
    from mstar.graph.special_destinations import EMIT_TO_CLIENT
    from mstar.model.submodule_base import StopKind, StopPolicy
    from mstar.worker.step_repeats import RepeatConfig
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    edges = [SimpleNamespace(name="text", next_node=EMIT_TO_CLIENT, output_modality="text")]
    edges += [SimpleNamespace(name=name, next_node="other", output_modality=None) for name in routed]
    node = SimpleNamespace(outputs=edges)
    worker._walk_graphs = {WALK: SimpleNamespace(get_nodes=lambda: {"n": node})}
    worker._withholding, worker._streamed, worker._device_loopback = withholding, {}, {}
    worker._graph_runtime = SimpleNamespace(supports_inline_emit=True)
    worker.parallel_nodes, worker.parallel_leader_nodes = ({"n"}, set()) if follower else (set(), set())
    worker._step_repeats = SimpleNamespace(
        config=lambda name: RepeatConfig(max=4, stream_tokens=True), may_repeat=lambda name, walk: True,
    )
    asked = []
    engine = SimpleNamespace(
        inline_client_signals=lambda name, walk: {"text": "new_token"},
        get_stop_policies=lambda name, walk: {"loop": StopPolicy(StopKind.AR, "new_token", {EOS})},
        device_loopback_signals=lambda name, walk: ["text"],
        stream_tokens=lambda *args: asked.append(args),
    )
    worker.engine_manager = SimpleNamespace(get_engine=lambda name: engine)
    assert worker._stream_refusal("n", WALK) == why
    assert asked == ([] if why else [("n", WALK, "new_token", 4)])


def test_a_runs_columns_are_allocated_before_its_capture(monkeypatch):
    """A column allocated inside the capture took a block the forward had freed
    there, and each iteration's scratch overwrote it: only the last survived."""
    runs = []
    keep = DeviceRun.keep

    def record(self, name_of, kept):
        runs.append(self)
        return keep(self, name_of, kept)

    monkeypatch.setattr(DeviceRun, "keep", record)
    loop, seen = _Loop(), {}
    loop.capturing = lambda body: seen.update({name: id(column) for name, column in runs[0].stops.items()})
    _, run, _, _ = _runner(monkeypatch, loop, [[1, 2], [3, 4]])
    assert seen and {name: id(column) for name, column in run.stops.items()} == seen and run.sealed


@pytest.mark.parametrize("progress", [False, True])
def test_a_failed_while_capture_settles_its_walk_and_a_plain_capture_serves(monkeypatch, progress):
    """One failure is the walk's answer: trying each bucket cost a second capture
    of every one of them."""
    class Failing(_Loop):
        attempts = 0

        def stop(self, state, tokens, stream):
            Failing.attempts += 1
            raise RuntimeError("no such stream")

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    runner, run, _, _ = _runner(monkeypatch, Failing(), [[1, 2]], progress=progress)
    assert not runner._resources[tables_key("n")]._windows
    assert run is None and "no such stream" in runner.run_refusal(WALK)
    spec = SimpleNamespace(bucket=BucketKey(WALK, 2, 2), slot=1, config=SimpleNamespace(get_config_type=lambda: None))
    assert runner._capture_run(spec, None) == (None, None, None) and Failing.attempts == 1


def test_a_raw_launch_takes_the_cuda_handle_torch_stream_doesnt_have(monkeypatch):
    """``torch.accelerator.current_stream`` gives a ``torch.Stream``, which has
    no ``cuda_stream``."""
    from mstar.engine.accelerator_graph_backend import CUDAGraphBackend

    backend = CUDAGraphBackend.__new__(CUDAGraphBackend)
    backend.device = torch.device("cuda:0")
    backend._runtime = SimpleNamespace(current_stream=lambda device: SimpleNamespace(cuda_stream=77))
    backend._runtime_entry = lambda name: 5
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda device: SimpleNamespace())
    graph = SimpleNamespace(raw_cuda_graph_exec=lambda: 9)
    assert backend.raw_launch(graph) == (5, 5, 9, 77)


@pytest.mark.parametrize("ran", [3, 4])
def test_the_worker_settles_each_row_even_if_another_reaches_the_full_bound(monkeypatch, ran):
    from mstar.utils import profiler
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    ended = []
    engine = SimpleNamespace(settle_run=lambda batch, ran, counts: ended.append((ran, counts)))
    batch = SimpleNamespace(batch=SimpleNamespace(repeats=4),
                            node_batch=SimpleNamespace(stream_windows={}), graph_walk=WALK)
    monkeypatch.setattr(profiler, "PHASE_PERIOD", 1)
    profiler.phase_buffer().clear()
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])  # the last column was never written
    rows = SimpleNamespace(request_ids=(7, 9), buffers={
        RUN_REPLAYS: torch.tensor([ran, ran]), RUN_COUNTS: torch.tensor([1, ran]),
        "t": tokens, "one": torch.zeros(2),
    })
    worker._settle_run(batch, rows, engine)
    assert ended == [(ran, {7: 1, 9: ran})] and batch.batch.repeats == ran
    assert RUN_REPLAYS not in rows.buffers and RUN_COUNTS not in rows.buffers
    # what keys the prefix chain reads next: nothing the run didn't produce
    assert torch.equal(rows.buffers["t"], tokens[:, :ran])
    assert rows.buffers["one"].shape == (2,)
    # a yield's cut shows nowhere else: the walk's line still says "on" (review-r7 G5)
    assert profiler.phase_buffer()[f"repeats.ran.{WALK}#"] == [ran]
    profiler.phase_buffer().clear()


def _skipped_step(worker, log, words, rows=(1, 2), **node_batch):
    """A step in ``_landed_skipped``, real batch and all, on a runtime that logs what it gets back."""
    from mstar.worker.micro_scheduler import ScheduledBatch

    class Word:
        value = None

        def __getitem__(self, index):
            self.value = next(words, self.value)
            return self.value

    worker._clear_in_flight_flag = lambda batch: log.append(("flag", sorted(batch.request_to_worker_graph)))
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        settle_run=lambda node_batch, ran: log.append(("rewound", ran)),
        ring_rows=lambda node: SimpleNamespace(
            ring_skip=lambda signal, rid, positions: log.append(("ring", signal, rid, list(positions))),
        ),
    ))
    worker._graph_runtime = SimpleNamespace(
        pending_loop_stop_rids=lambda rids, loop: [],
        push_back_node=lambda node, rids, wgs: log.append(("queued", rids)),
    )
    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph=dict.fromkeys(rows, 0))
    return SimpleNamespace(
        node_name="n", graph_walk=WALK, batch=batch, speculative_new_iter=True, loop_name="loop", group=None,
        future=SimpleNamespace(done=lambda: False, result=lambda: "outputs"),
        node_batch=SimpleNamespace(**{
            "run_started": Word(), "step_context": SimpleNamespace(repeats=8), "completion_event": None,
            "request_ids": list(rows), "cached_rids": set(), "failed_requests": {}, "ring_bases": {}, "ring_width": 8,
            "stream_windows": {}, "run_rows": {},
            **node_batch,
        }),
    )


@pytest.mark.parametrize("begun, skipped", [(RUN_SKIPPED, True), (RUN_RAN, False)])
@pytest.mark.parametrize("owner", ["ordinary", "membership", "unit", "refused_unit"])
def test_a_skipped_run_lands_as_a_step_that_never_ran(begun, skipped, owner):
    """Nothing it counted happened and it has no outputs: what admit counted goes back
    and its rows wait with the inputs they had, with no postprocess at all. The word
    can still be 0 at first: the step before has only just landed."""
    from mstar.worker.worker import Worker

    worker, log = Worker.__new__(Worker), []
    step = _skipped_step(worker, log, iter([0, begun]), run_rows={1: object()} if owner == "membership" else {})
    if owner in {"unit", "refused_unit"}:
        step.group = SimpleNamespace(submitted=owner == "unit")
    skipped = skipped and owner in {"ordinary", "refused_unit"}  # logical closure owns its reader/pin release
    assert worker._landed_skipped(step) is skipped
    assert log == ([("rewound", 0), ("flag", [1, 2]), ("queued", [1, 2])] if skipped else [])


def test_a_skipped_run_tells_auto_nothing():
    """Its near-empty graph isn't a run's time, and the gap after it is the host build
    the skip caused: read as idle time, each cut would push runs longer."""
    from collections import deque

    from mstar.worker.step_repeats import AutoRepeats
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    key = ("n", WALK)
    auto = AutoRepeats(limit=8)
    seen = []
    auto.observe = lambda gap, gpu, repeats: seen.append((gap, gpu, repeats))
    worker._step_repeats = SimpleNamespace(auto={key: auto})

    def event(at):
        return SimpleNamespace(
            query=lambda: True, at=at, elapsed_time=lambda end: (end.at - at) * 1000,
        )

    skipped, ran = torch.tensor([RUN_SKIPPED]), torch.tensor([RUN_RAN])
    worker._auto_prev_end = event(0.0)
    worker._auto_spans = deque([
        (event(1.0), event(1.001), key, 8, skipped), (event(5.0), event(6.0), key, 8, ran),
    ])
    worker._drain_auto_spans()
    assert seen == [(0.0, 1.0, 8)]


def test_a_skip_is_not_missed_for_a_run_that_began_as_the_host_looked():
    """The launch tail sets the event after the admit fence lifts, and the word can
    change between two reads, so "it never launched" takes a fresh look at all three."""
    from mstar.worker.worker import Worker

    worker, log = Worker.__new__(Worker), []
    step = _skipped_step(worker, log, iter([0, RUN_SKIPPED]))
    step.future = SimpleNamespace(done=lambda: True, result=lambda: "outputs")
    assert worker._landed_skipped(step) is True

    # the event wasn't set yet as the wait began, and the graph is still to run
    step = _skipped_step(worker, log, iter([0, 0, RUN_SKIPPED]))

    def _tail_done():
        step.node_batch.completion_event = SimpleNamespace(query=lambda: False)
        return True

    step.future = SimpleNamespace(done=_tail_done, result=lambda: "outputs")
    assert worker._landed_skipped(step) is True


def test_a_step_that_never_launched_is_no_skip():
    from mstar.worker.worker import Worker

    worker, log = Worker.__new__(Worker), []
    step = _skipped_step(worker, log, iter([0]))
    step.future = SimpleNamespace(done=lambda: True, result=lambda: "outputs")
    assert worker._landed_skipped(step) is False and log == []


def test_a_skipped_steps_ring_positions_hold_no_room():
    """The consumer never gets them, and only postprocess gave them back."""
    from mstar.worker.worker import Worker

    worker, log = Worker.__new__(Worker), []
    step = _skipped_step(worker, log, iter([RUN_SKIPPED]), ring_bases={"codes": {1: 16, 2: 24}}, ring_width=2)
    assert worker._landed_skipped(step) is True
    rings = [entry for entry in log if entry[0] == "ring"]
    assert rings == [("ring", "codes", 1, [16, 17]), ("ring", "codes", 2, [24, 25])]


@pytest.mark.parametrize("unaccepted_descriptor", [False, True])
def test_a_skipped_step_gives_back_only_the_rows_a_refused_one_would(unaccepted_descriptor):
    """The step was built before N landed. A row whose loop stopped at N has an emptied
    node, which would pop with no inputs and run past its stop. One prepare_inputs
    vetoed has spent its inputs, and one a stage blamed fails."""
    from mstar.worker.worker import Worker

    worker, log = Worker.__new__(Worker), []
    step = _skipped_step(
        worker, log, iter([RUN_SKIPPED]), rows=(1, 2, 3, 4), request_ids=[1, 2, 4], failed_requests={4: "boom"},
    )
    worker._graph_runtime.pending_loop_stop_rids = lambda rids, loop: [2]
    from mstar.streaming.stream_buffer import StreamChunkInfo, StreamingEdge, StreamReservation

    step.node_batch.stream_windows = {rid: [StreamingEdge("text", "n", [], StreamChunkInfo(
        0, 0, 8, False, StreamReservation(0, 4, 4),
    ))] for rid in (1, 2, 3, 4)}
    retired = []
    worker._wait_for_step = lambda step: retired.append("completed")
    worker.streams = SimpleNamespace(settle_window=lambda rid, edge, n: retired.append((rid, n)))
    failed = []
    worker._fail_step_rows = lambda step, outputs: failed.append((dict(step.node_batch.failed_requests), outputs))
    if unaccepted_descriptor:
        # A closed membership window already settled its logical receipt to
        # zero. It has no RUN_SKIPPED word, but owns the same unread graph work.
        step.node_batch.run_started = None
        worker._return_unexecuted(step, "outputs")
        assert not any(entry[0] == "rewound" for entry in log)
    else:
        assert worker._landed_skipped(step) is True
    assert ("queued", [1, 4]) in log, log
    assert failed == [({4: "boom"}, "outputs")]
    assert retired == ["completed", (2, 0), (3, 0), (4, 0)]
    assert set(step.node_batch.stream_windows) == {1}, "only a retried row retains its original reader"


def test_worker_closes_reused_physical_rows_as_separate_logical_receipts():
    from concurrent.futures import ThreadPoolExecutor

    from mstar.engine.engine import Engine, ExecutingBatch
    from mstar.engine.resources.run_rows import RunRows
    from mstar.engine.resources.step import StepContext
    from mstar.model.submodule_base import NodeSubmodule
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import PendingBatch, Worker

    membership = RunRows(CPU, 2, 4, 2)
    original = membership.reserve_row(1, 1, 0, (1, 0))
    zero = membership.reserve_row(2, 2, 1, (2, 0))
    for row in (original, zero):
        membership.publish_row(row)
    ids, bindings = torch.zeros(2, dtype=torch.long), torch.zeros((2, 2), dtype=torch.long)
    membership.boundary(ids, bindings)
    membership.stop(ids, torch.tensor([True, False]))
    membership.acknowledge(membership.control.tolist())
    late = membership.reserve_row(3, 3, 0, (3, 0))
    membership.publish_row(late)
    log = []

    class Node(NodeSubmodule):
        def prepare_inputs(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            pass

        def postprocess(self, request_id, request_info, outputs, inputs):
            log.append(("tail", request_id, inputs))
            assert outputs["final"][0].item() == {1: 101, 3: 302}[request_id]

    engine = Engine.__new__(Engine)
    engine._enable_nvtx, engine._loopback_steps = False, {}
    engine._submodules = {"n": SimpleNamespace(
        submodule=Node.__new__(Node), rings={}, run_columns={WALK: frozenset({"frame"})}, carries={},
        device_stops={}, accelerator_graph_runner=SimpleNamespace(
            slot_for=lambda lease: SimpleNamespace(run=SimpleNamespace(state=SimpleNamespace(membership=membership))),
            release=lambda lease, count: log.append(("release", count)),
        ),
    )}
    def settle(step, counts):
        log.append(("settle", dict(counts)))
        step.ctx.run_counts = dict(counts)
    engine._runner = SimpleNamespace(settle_run=settle)
    engine.finalize_batch = lambda *args, **kwargs: {}
    resident = SimpleNamespace(unpin=lambda slots: log.append(("unpin", slots)))
    def receipt(rows):
        rids = [row.request_id for row in rows]
        ctx = StepContext(rids, WALK, 0, False, repeats=4, resident=resident,
                          slot_lease=SlotLease(0, BucketKey(WALK, 2, 2)))
        batch = ExecutingBatch("n", {}, ctx, inputs=[f"input{rid}" for rid in rids])
        batch.per_request_info = dict.fromkeys(rids, SimpleNamespace())
        batch.per_request_info_wrapped = dict(batch.per_request_info)
        batch.run_rows = {row.request_id: row for row in rows}
        batch.step = SimpleNamespace(ctx=ctx)
        batch.run_started = torch.tensor([RUN_RAN])
        scheduled = ScheduledBatch("n", WALK, dict.fromkeys(rids, 0), repeats=4)
        return PendingBatch(scheduled, batch, "n", "partition", WALK, None)
    active, joined = receipt((original, zero)), receipt((late,))
    active.joined.append(joined)
    active.node_batch.outputs = BatchedModelOutput(
        row_outputs={
            "frame": torch.tensor([[11, 31, 32, -1], [-1, -1, -1, -1]]),
            "final": torch.tensor([[101, 301, 302, -1], [-1, -1, -1, -1]]),
        },
        check_stop_buffers={RUN_DESCRIPTORS: torch.tensor([[original.handle, late.handle, late.handle, 0], [0]*4])},
    )
    worker = Worker.__new__(Worker)
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    worker.device, worker.device_capability = CPU, SimpleNamespace(is_accelerator=False)
    worker._wait_for_step = lambda part: None
    worker._stage_host_rows = lambda *args: None
    worker._publishable_request_ids = lambda batch: batch.request_ids
    worker._clear_in_flight_flag = lambda batch: log.append(("flag", list(batch.request_to_worker_graph)))
    worker._drop_rows_whose_loop_stopped = lambda step: None
    worker._push_back_batch = lambda batch: log.append(("retry", list(batch.request_to_worker_graph)))
    def finish(part, outputs):
        log.append(("route", part.batch.repeats, {
            rid: out["frame"][0].tolist() for rid, out in outputs.per_rid_outputs.items()
        }))
    worker._finish_step = finish
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker._finish_live_run(active, executor)
    assert log[:2] == [("settle", {1: 1, 2: 0}), ("settle", {3: 2})]
    assert ("retry", [2]) in log
    assert [event for event in log if event[0] == "tail"] == [("tail", 1, "input1"), ("tail", 3, "input3")]
    assert [event for event in log if event[0] == "route"] == [
        ("route", 1, {1: [[11]]}), ("route", 2, {3: [[31, 32]]}),
    ]
    assert log[-4:] == [("release", 2), ("unpin", (1,)), ("unpin", (2,)), ("unpin", (3,))]
    assert membership.idle and not active.node_batch.run_rows and not joined.node_batch.run_rows


@pytest.mark.parametrize("admitted", [False, True])
def test_ready_row_is_owned_by_the_graph_before_live_publication(admitted):
    from concurrent.futures import ThreadPoolExecutor

    from mstar.engine.engine import ExecutingBatch
    from mstar.engine.resources.run_rows import RunRows
    from mstar.engine.resources.step import StepContext
    from mstar.worker.batch_builder import StepPick
    from mstar.worker.micro_scheduler import ChainPick, ScheduledBatch
    from mstar.worker.worker import PendingBatch, Worker

    owner = RunRows(CPU, 2, 3, 2)
    original = owner.reserve_row(1, 1, 0, (1, 0))
    owner.publish_row(original)
    ctx = StepContext([1], WALK, 0, False, repeats=4, slot_lease=SlotLease(0, BucketKey(WALK, 2, 2)))
    batch = ExecutingBatch("n", {}, ctx)
    batch.run_rows = {1: original}
    batch.completion_event = SimpleNamespace(query=lambda: False)
    active = PendingBatch(ScheduledBatch("n", WALK, {1: 0}, repeats=4), batch, "n", "p", WALK, None)
    late = ExecutingBatch("n", {}, StepContext([2], WALK, 0, False, repeats=4))
    worker = Worker.__new__(Worker)
    owned, holds, ready = set(), set(), {2, 3}
    worker._is_tearing_down = lambda rid: False
    worker.request_state = object()
    worker._build_executing_batch = lambda scheduled: late
    worker._set_in_flight_flag = lambda scheduled, value: owned.update(scheduled.request_to_worker_graph)
    worker._hold_rows = lambda lane, rids: (holds.clear(), holds.update(rids))
    worker._all_on_device = lambda *args: False
    worker._count_turn = lambda receipt: None
    def retry(receipt, output):
        owned.difference_update(receipt.batch.request_to_worker_graph)
        ready.update(receipt.batch.request_to_worker_graph)
    worker._return_unexecuted = retry
    def take(state, chain):
        assert chain.pick.taken == [2] and chain.pick.held_back == {3}
        ready.difference_update(chain.pick.taken)
        return ScheduledBatch("n", WALK, {2: 0}, repeats=4)
    worker.scheduler = SimpleNamespace(
        pick_chain_step=lambda *args: ChainPick(("n", WALK), StepPick([2, 3], [], None, set(), False), []),
        take_chain_step=take,
    )
    def admit(candidate, incumbent, row):
        assert owned == {2} and holds == {1, 2} and ready == {3}
        assert active.joined[0].node_batch is candidate and row == 1
        assert owner.control[row].item() == 0  # no publication before commit
        if not admitted:
            return None
        descriptor = owner.reserve_row(2, 2, row, (2, 0))
        candidate.run_rows[2] = descriptor
        candidate.committed = object()
        return descriptor
    engine = SimpleNamespace(run_membership=lambda batch: owner, prepare_inputs=lambda batch: None,
                             admit_run_row=admit)
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert worker._offer_run_row(active, executor) is admitted
    if admitted:
        ids, bindings = torch.zeros(2, dtype=torch.long), torch.zeros((2, 2), dtype=torch.long)
        owner.boundary(ids, bindings)
        assert ids[1].item() == late.run_rows[2].handle and bindings[1].tolist() == [2, 0]
        assert ready == {3} and owned == {2} and len(active.joined) == 1
    else:
        assert ready == {2, 3} and not owned and not active.joined and holds == {1}
        assert owner.control[1].item() == 0


@pytest.mark.parametrize("later_tokens", [0, 4])
def test_a_short_run_takes_back_a_decode_snapshot_past_where_it_ended(later_tokens):
    """Filed once the chain keys its page, a snapshot from past the row's real
    tokens would serve state at the wrong length; its predecessor stands again."""
    import threading

    from mstar.engine.resources.recurrent.pool import DecodeSnapshot, RecurrentStatePool

    pool = RecurrentStatePool.__new__(RecurrentStatePool)
    pool._lock = threading.RLock()
    held = SimpleNamespace(tokens=12 + later_tokens)
    pool._slots = {1: {"main": held}}
    earlier = DecodeSnapshot(3, 8)
    pool._decode = {(1, "main"): DecodeSnapshot(4, 12, replaces=earlier)}
    pool._unkeyed = {(1, "main")}
    freed = []
    pool._drop_decode = lambda key, decode: (pool._unkeyed.discard(key), freed.append(decode.slot))
    pool.rewind_run(SimpleNamespace(segments=[SimpleNamespace(request_id=1, label="main", span=1)]), None, 2, {1})
    assert held.tokens == 10 + later_tokens and freed == [4] and pool._decode[(1, "main")] is earlier


def test_a_walk_that_drops_rows_refuses_a_prefix_chain():
    """A dropped row's tokens past its stop hold no KV, which a chain would key."""
    from mstar.engine.engine import Engine

    asked = []
    engine = Engine.__new__(Engine)
    engine._runner = SimpleNamespace(prefix_chain_keys=lambda node: ["kv"])
    engine._submodules = {"n": SimpleNamespace(
        joint_comm_group=SimpleNamespace(world_size=1), device_stops={}, refills={}, loopback_rows=None,
        submodule=SimpleNamespace(repeat_carry=lambda walk, signals: {}, ITERATION_INPUTS={}), run_columns={},
    )}
    engine.get_stop_policies = lambda node, walk: {"loop": SimpleNamespace(
        reads_token=True, token_signal="t", eos_tokens=frozenset({9}),
    )}
    runner = SimpleNamespace(
        ask_run=lambda walk, ask: asked.append(ask), refuse_run=lambda walk, why: asked.append(why),
    )
    engine._ask_run("n", runner, WALK, RepeatConfig(device_stop="drop_rows", max=4), ())
    assert len(asked) == 1 and "prefix" in asked[0]


def test_a_run_that_ends_short_repeats_its_last_iteration_in_the_columns_it_never_wrote():
    """A run on the device can end before its count (a stop, a yield): whatever
    reads a column's last (the step speculated meanwhile, a node's postprocess)
    reads the last iteration that ran, not a column nothing wrote (GPU finding
    14, review-r2 F2)."""
    from mstar.engine.device_run import RUN_REPLAYS, DeviceRun

    state = RunState(1, "t", {9}, CPU)
    run = DeviceRun(state, {}, lambda: None, 4)
    run.rows["t"] = torch.tensor([[5, 6, 1 << 40, 1 << 40]])
    run.stops["t"] = run.rows["t"].clone()
    state.index.fill_(2)
    stops, rows, _ = run.results(4)
    assert rows["t"].tolist() == [[5, 6, 6, 6]] and stops["t"].tolist() == [[5, 6, 6, 6]]
    assert stops[RUN_REPLAYS].tolist() == [2]
    state.index.fill_(4)
    assert stops[RUN_REPLAYS].tolist() == [2]  # a staged successor cannot rewrite this receipt
    run.rows["t"] = torch.tensor([[1, 2, 3, 4]])
    assert run.results(4)[1]["t"].tolist() == [[1, 2, 3, 4]]


def test_joining_results_only_expose_views_while_the_device_run_is_active():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class ViewsOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert not func._schema.is_mutable, "results must not queue device writes"
            result = func(*args, **(kwargs or {}))
            inputs = [v for v in tree_leaves((args, kwargs)) if isinstance(v, torch.Tensor)]
            for value in tree_leaves(result):
                if isinstance(value, torch.Tensor):
                    assert any(torch._C._is_alias_of(value, old) for old in inputs), "results allocated device data"
            return result

    state = RunState(2, "t", {9}, CPU)
    state.descriptors = torch.tensor([[1, 2, 0, 0], [3, 3, 0, 0]])
    state.index.fill_(2)
    state.counts.fill_(2)
    run = DeviceRun(state, {}, lambda: None, 4)
    run.rows["t"] = torch.tensor([[5, 6, -1, -2], [7, 8, -3, -4]])
    run.stops["t"] = run.rows["t"].clone()
    run.spans["t"] = torch.ones(2, 4, dtype=torch.long)
    with ViewsOnly():
        stops, rows, spans = run.results(4)
    assert rows["t"].tolist() == [[5, 6, -1, -2], [7, 8, -3, -4]]
    assert stops[RUN_DESCRIPTORS].tolist() == [[1, 2, 0, 0], [3, 3, 0, 0]]
    # The metadata was requested before completion; the already-returned views
    # must observe the device's later writes without another results call.
    state.index.fill_(3)
    state.counts[1] = 3
    state.descriptors[1, 2] = 3
    run.rows["t"][1, 2] = 9
    assert stops[RUN_REPLAYS].tolist() == [3, 3]
    assert stops[RUN_COUNTS].tolist() == [2, 3]
    assert rows["t"][1, 2].item() == 9 and spans["t"].shape == (2, 4)


@pytest.mark.parametrize("refusal", [
    None, "cap_only", "custom", "missing_loop", "other_stop", "past_cap", "prefix", "rows", "resident", "ranks",
])
def test_iteration_inputs_enable_only_a_resident_single_loop_run(refusal):
    from mstar.engine.engine import Engine
    from mstar.graph.loop_indices import LoopLayout
    from mstar.model.submodule_base import StopKind, StopPolicy

    asked, refused = [], []
    engine = Engine.__new__(Engine)
    loop = LoopLayout("loop")
    engine._loops = {} if refusal == "missing_loop" else {("n", WALK): loop}
    engine._runner = SimpleNamespace(prefix_chain_keys=lambda node: ["kv"] if refusal == "prefix" else [])
    engine._submodules = {"n": SimpleNamespace(
        joint_comm_group=SimpleNamespace(world_size=2 if refusal == "ranks" else 1), device_stops={}, refills={},
        loopback_rows=None if refusal == "rows" else object(),
        submodule=SimpleNamespace(repeat_carry=lambda walk, signals: {}, ITERATION_INPUTS={"x": object()}),
        run_columns={},
    )}
    engine.get_stop_policies = lambda node, walk: {
        "other" if refusal == "other_stop" else "loop": StopPolicy(StopKind.ITER),
    }
    if refusal in {"cap_only", "custom"}:
        engine.get_stop_policies = lambda node, walk: {} if refusal == "cap_only" else None
    engine.resident_refusal = lambda *args: "mixed capture walks" if refusal == "resident" else None
    engine._row_pairs = lambda *args: ({}, {})
    runner = SimpleNamespace(
        ask_run=lambda walk, ask: asked.append(ask), refuse_run=lambda walk, why: refused.append(why),
    )
    engine._ask_run("n", runner, WALK, RepeatConfig(
        device_stop="skip", max=4, refill_at=None, stream_tokens=False, past_cap=refusal == "past_cap",
    ), ())
    if refusal in {None, "cap_only"}:
        assert len(asked) == 1 and asked[0].loop is loop and asked[0].token_signal is None
    else:
        assert not asked
        assert refused or refusal in {"ranks", "custom"}


def test_a_decode_wrapper_reads_its_lengths_off_the_table():
    """A plan for longer rows serves what's there: the next step after a short run."""
    from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper

    wrapper = FlashInferDecodeWrapper.__new__(FlashInferDecodeWrapper)
    wrapper.page_size = 4
    wrapper._paged_kv_last_page_len_buf = torch.zeros(3, dtype=torch.int32)
    wrapper.stage_page_base(torch.tensor([0, 3, 4]))
    wrapper.gather_lengths(torch.tensor([9, 2, 0]))
    assert wrapper._paged_kv_last_page_len_buf.tolist()[:2] == [2, 3]


CHAIN, OTHER = ("n", WALK), ("n", "prefill")


def _waiting_on_a_run(monkeypatch, ready):
    """A worker in ``_await_run``, and what it did. ``ready``: the (node, walk) keys
    ready as the wait starts, then after each pass's messages; the last stay."""
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    log = []
    looks = iter(ready)
    scheduler = SimpleNamespace(ready=next(looks))
    scheduler.has_ready_excluding = lambda state, key: bool(scheduler.ready - {key})
    worker.scheduler = scheduler
    worker.request_state = None
    worker.engine_manager = SimpleNamespace(
        get_engine=lambda node: SimpleNamespace(yield_runs=lambda node: log.append("yield")),
    )

    def _messages():
        log.append("messages")
        scheduler.ready = next(looks, scheduler.ready)
        return True

    worker._process_messages = _messages
    worker.wakeup_event = SimpleNamespace(drain=lambda: False)
    worker._check_ready_tensors = lambda: None
    worker._poll_stream_buffers = lambda: None
    monkeypatch.setattr("mstar.worker.repeating_steps.time.sleep", lambda s: None)
    return worker, log


def _run_step(rows, looks, **node_batch):
    landed = iter([False] * looks + [True])
    return SimpleNamespace(
        node_name="n", graph_walk=WALK, group=None, batch=SimpleNamespace(request_to_worker_graph=dict.fromkeys(rows)),
        node_batch=SimpleNamespace(completion_event=SimpleNamespace(query=lambda: next(landed)),
                                   run_rows={}, **node_batch),
    )


@pytest.mark.parametrize("peek, yields", [(True, 1), (False, 0)])
def test_without_the_fairness_peek_ready_work_leaves_runs_alone(monkeypatch, peek, yields):
    """The chain yields to other work only at its consecutive-step cap then, so
    ending a run for it serves the work no sooner (review-r7 G4)."""
    worker, log = _waiting_on_a_run(monkeypatch, [{OTHER}])
    worker._hold_rows = lambda lane, rows: None
    worker._spec_peek = peek
    worker._await_run(_run_step([1], 2, token_stream=None), None, [], None)
    assert log.count("yield") == yields


def test_a_runs_wait_moves_nothing_until_the_step_behind_it_has_admitted(monkeypatch):
    """The loop's own fence covers this work at the top of a pass, not here."""
    worker, log = _waiting_on_a_run(monkeypatch, [set()])
    worker._hold_rows = lambda lane, rows: None
    behind = _run_step([2], 0, launch_started_event=SimpleNamespace(
        is_set=lambda: False, wait=lambda timeout: log.append("admitted") or True,
    ))
    worker._await_run(_run_step([1], 1, token_stream=None), behind, [], None)
    assert log == ["admitted", "messages"]


def test_a_run_serves_the_lanes_and_ends_once_a_row_turns_ready(monkeypatch):
    """The request that arrives mid-run ends nothing: its prefill goes on a lane beside
    the run, and the run ends when the row it readies does, one ask for the two runs
    it reaches. N+1's rows count as in flight."""
    worker, log = _waiting_on_a_run(monkeypatch, [set(), set(), {CHAIN}])
    held = {}
    worker._hold_rows = held.__setitem__
    lane = SimpleNamespace(finish=lambda: log.append("finish"), launch=lambda pool: log.append("launch"))
    worker._await_run(
        _run_step([1], 3, token_stream=None),
        _run_step([2], 0, launch_started_event=SimpleNamespace(is_set=lambda: True)), [lane], None,
    )
    served = ["messages", "finish", "launch"]
    assert log == [*served, *served, "yield", *served]
    assert held == {0: {1, 2}}


def test_a_pass_where_nothing_happened_scans_nothing(monkeypatch):
    """A row mostly turns ready as a message, a read or a lane's step lands, so the
    other 100 us passes of a run's wait pay for those three looks alone."""
    worker, log = _waiting_on_a_run(monkeypatch, [set()])
    monkeypatch.setattr("mstar.worker.repeating_steps.time.monotonic", lambda: 0.0)  # no quiet scan due
    worker._hold_rows = lambda lane, rows: None
    worker._process_messages = lambda: False
    woke, landed = iter([False, True, False, False]), iter([False, False, True, False])
    worker.wakeup_event = SimpleNamespace(drain=lambda: next(woke))
    worker._check_ready_tensors = lambda: log.append("tensors")
    lane = SimpleNamespace(finish=lambda: next(landed), launch=lambda pool: log.append("launch"))
    worker._await_run(_run_step([1], 4, token_stream=None), None, [lane], None)
    assert log == ["tensors", "launch"] * 2


def test_a_row_ready_with_nothing_landing_is_still_seen(monkeypatch):
    """A KV copy from the host tier, or an OOM hold running out, readies a row only as a
    scan looks; nothing wakes the wait for either, so a slower tick scans anyway."""
    worker, log = _waiting_on_a_run(monkeypatch, [set()])
    worker._hold_rows = lambda lane, rows: None
    passes = iter(range(100))

    def _quiet():
        if next(passes) == 3:
            worker.scheduler.ready = {OTHER}  # a copy landed, unannounced
        return False

    worker._process_messages = _quiet
    clock = iter(range(0, 10**6, 300))
    monkeypatch.setattr("mstar.worker.repeating_steps.time.monotonic", lambda: next(clock) * 1e-6)
    worker._await_run(_run_step([1], 12, token_stream=None), None, [], None)
    assert log == ["yield"]


@pytest.mark.parametrize("ready, yields", [
    # a step too full to take its own walk's rows would otherwise end every run early
    ([{CHAIN}], 0),
    # that row hides nothing else, and other work needs no room in the step
    ([{CHAIN}, {CHAIN}, {CHAIN, OTHER}], 1),
    # A row ready since speculation gets its yield here, after N+1 stages
    ([{OTHER}], 1),
])
def test_rows_ready_as_a_run_waits_end_it_unless_its_walk_had_rows_waiting(monkeypatch, ready, yields):
    worker, log = _waiting_on_a_run(monkeypatch, ready)
    worker._hold_rows = lambda lane, rows: None
    worker._await_run(_run_step([1], 3, token_stream=None), None, [], None)
    assert log.count("yield") == yields


def test_a_run_that_streams_its_tokens_serves_between_looks(monkeypatch):
    from mstar.model.submodule_base import StopKind, StopPolicy

    worker, log = _waiting_on_a_run(monkeypatch, [set(), {CHAIN}])
    worker._hold_rows = lambda lane, rows: None
    worker._graph_runtime = SimpleNamespace(
        send_tokens=lambda *args: None,
        loop_progress=lambda node, rids, walks, loops: ([0] * len(rids), [9] * len(rids)),
    )
    worker._streamed = {("n", WALK): (StopPolicy(StopKind.AR, "t", {EOS}), "text", "text")}
    worker._innermost_loop_by_node = {}
    stream = TokenStream(1, 4, CPU)
    stream.begin([1])
    step = _run_step(
        [1], 1, token_stream=stream, request_ids=[1],
        per_request_info_wrapped={1: SimpleNamespace(max_tokens=9, resource_configs={})},
    )
    worker._await_run(step, None, [], None)
    assert log == ["messages", "yield"]


@pytest.mark.parametrize("slots, why", [(1, "one capture slot"), (2, None)])
def test_a_single_capture_slot_does_not_stream_tokens(slots, why):
    """A slot's landing area is the slot's: with one slot the next step resets
    it while this one's tokens are still being sent."""
    from mstar.engine.engine import Engine

    engine = Engine.__new__(Engine)
    mgmt = SimpleNamespace(accelerator_graph_runner=SimpleNamespace(num_slots=slots), device_stream_tokens={})
    engine._submodules = {"n": mgmt}
    refusal = engine.stream_tokens("n", WALK, "new_token", 4)
    assert (refusal is None) is (why is None) and (why is None or why in refusal)
    assert bool(mgmt.device_stream_tokens) is (why is None)


def test_a_loop_back_aliasing_a_run_column_goes_on_as_its_last_iteration():
    """A node whose postprocess loops its sampled token back by name
    (``text_inputs = new_token``) would hand a run's whole column on as the
    next input; the engine hands on the last iteration, for every model
    (review-r2 F5)."""
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import BatchedModelOutput, CarryPair, NodeSubmodule

    class _Aliasing(NodeSubmodule):
        prepare_inputs = forward = None

        def postprocess(self, request_id, request_info, outputs, **kwargs):
            outputs["text_inputs"] = outputs["new_token"]

    engine = Engine.__new__(Engine)
    engine._submodules = {"n": SimpleNamespace(
        submodule=_Aliasing(), run_columns={WALK: frozenset({"new_token"})},
        carries={WALK: {"text_inputs": CarryPair("new_token", "input_ids")}},
    )}
    ctx = SimpleNamespace(repeats=3)
    batch = SimpleNamespace(
        node_name="n", graph_walk=WALK, request_ids=[1], inputs=[None], failed_requests={},
        per_request_info_wrapped={1: None}, step_context=ctx,
    )
    outputs = BatchedModelOutput(per_rid_outputs={1: {"new_token": [torch.tensor([[4, 5, 6]])]}})
    engine._postprocess_batch(batch, outputs)
    token = outputs.get(1)["text_inputs"][0]
    assert token.tolist() == [6] and token.stride() == (1,), "a routed edge's byte view takes it"
    assert token.data_ptr() == outputs.get(1)["new_token"][0][0, 2:].data_ptr(), "a view: no kernel a row"
    assert outputs.get(1)["new_token"][0].tolist() == [[4, 5, 6]], "the client's items are the whole column"
    ctx.repeats = 1
    outputs = BatchedModelOutput(per_rid_outputs={1: {"new_token": [torch.tensor([7])]}})
    engine._postprocess_batch(batch, outputs)
    assert outputs.get(1)["text_inputs"][0].tolist() == [7]


@pytest.mark.parametrize("yield_after", [None, 2])
def test_counted_run_needs_no_token_output_and_honors_limit_and_yield(monkeypatch, yield_after):
    loop = _Loop()
    _, run, _, calls = _runner(monkeypatch, loop, [[EOS, EOS]], counted=True, limit=4)
    if yield_after is not None:
        loop.on_iteration = lambda i: run.state.request_yield() if i + 1 == yield_after else None
    stops, rows, _ = _run(run, 1, 4)
    ran = 4 if yield_after is None else yield_after
    assert len(calls) == ran
    assert stops[RUN_REPLAYS].tolist() == [ran, ran]
    assert "new_token" not in stops
    assert rows["frame"][0].tolist() == [2 * EOS] * 4


def test_counted_run_cannot_request_token_streaming(monkeypatch):
    runner, run, _, _ = _runner(monkeypatch, _Loop(), [[1, 1]], counted=True, stream_tokens=True)
    assert run is None
    assert "no token signal" in runner.run_refusal(WALK)


def test_engine_asks_for_a_counted_device_run_without_a_token_signal():
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import StopKind, StopPolicy

    asked = []
    policy = StopPolicy(StopKind.ITER)
    engine = Engine.__new__(Engine)
    engine._submodules = {"n": SimpleNamespace(
        joint_comm_group=SimpleNamespace(world_size=1), device_stops={}, refills={}, loopback_rows=None,
        submodule=SimpleNamespace(repeat_carry=lambda walk, signals: {}), run_columns={},
    )}
    engine.get_stop_policies = lambda node, walk: {"loop": policy}
    runner = SimpleNamespace(ask_run=lambda walk, ask: asked.append(ask))
    engine._ask_run("n", runner, WALK, RepeatConfig(
        device_stop="skip", max=4, refill_at=None, stream_tokens=False,
    ), ())
    assert len(asked) == 1 and asked[0].token_signal is None
    assert engine._submodules["n"].device_stops[WALK] is policy


def test_a_width_one_device_launch_returns_zero_row_counts_without_changing_output_shapes(monkeypatch):
    from mstar.engine.engine import Engine
    from mstar.engine.resources.step import SlotLease, StepContext

    loop = _Loop()
    runner, run, _, _ = _runner(monkeypatch, loop, [[1, 1]])
    loop.on_iteration = lambda i: run.state.marks.fill_(True)
    slot = next(iter(runner._buckets.values())).slots[0]
    slot.row_heads = {}
    raw = BatchedModelOutput(check_stop_buffers={"token": torch.tensor([1, 1])})
    runner.run_forward = lambda *args, **kwargs: (run.launch(), raw)[1]
    ctx = StepContext(request_ids=(7, 9), graph_walk=WALK, slot=0, capture=True)
    batch = SimpleNamespace(step_context=ctx, iteration_columns=None, preplan_event=None)
    mgmt = SimpleNamespace(
        device_stops={WALK: SimpleNamespace(limits=lambda info: SimpleNamespace(ignore_eos=False))},
        refills={WALK: None}, device_stream_tokens={},
    )
    lease = SlotLease(slot=0, bucket=next(iter(runner._buckets)))
    out = Engine._forward(
        Engine.__new__(Engine), batch, mgmt, runner,
        SimpleNamespace(per_request_info={7: None, 9: None}), {}, lease, True, [7, 9], None,
    )
    assert out.check_stop_buffers["token"].shape == (2,)
    assert out.check_stop_buffers[RUN_REPLAYS].tolist() == [1, 1]
    assert out.check_stop_buffers[RUN_COUNTS].tolist() == [0, 0]


@pytest.mark.parametrize("first_count", [0, 1, 3, 4])
def test_a_queued_window_reads_actual_progress_and_suppresses_predecessor_stops(monkeypatch, first_count):
    """N+1's snapshot predates N. A short/zero N chooses its actual next input;
    an EOS row never mutates the resource again, including in a later launch.
    """
    from mstar.engine.resources.loopback_rows import LoopbackRows

    loop = _Loop()

    def entering_capture(body):
        # GPU capture records allocation/initialization ops once; replay does
        # not rerun Python's lazy-table branch. Such a reset would recur in
        # every WHILE iteration and erase the predecessor's terminal state.
        def no_new_table(*args):
            pytest.fail("resident progress storage must be initialized before capture")
        monkeypatch.setattr(LoopbackRows, "_table", no_new_table)

    loop.capturing = entering_capture
    runner, run, _, calls = _runner(monkeypatch, loop, [[1, 1]], limit=4, progress=True)
    runner.test_resource.zero_()
    entries = torch.tensor([[1, 0], [1, 0]])
    columns = {"cond": torch.tensor([[7., 8., 9., 10., 11., 12., 13., 14.],
                                     [20., 21., 22., 23., 24., 25., 26., 27.]])}
    # The immutable successor already exists before the first launch.
    successor = IterationWindow(columns, torch.zeros(2, dtype=torch.long), entries)
    run.state.stage(2, [False, False], 4)
    _stage_inputs(run, successor)
    if first_count == 0:
        run.state.request_yield()
    else:
        loop.on_iteration = lambda i: run.state.request_yield() if i == first_count - 1 else None
    run.launch()
    assert run.state.counts.tolist() == [min(first_count, 3), first_count]
    calls.clear()
    loop.on_iteration = lambda i: None
    run.state.stage(2, [False, False], 4)
    _stage_inputs(run, successor)
    run.launch()
    assert run.state.counts.tolist() == [max(0, 3 - first_count), 4]
    assert calls[0][1].item() == 20 + first_count
    assert runner.test_resource[1].item() == 7 + 8 + 9, "stale successor cannot overwrite recurrent state"
    assert run.window.columns["cond"].shape[1] == 8 and run.rows["frame"].shape[1] == 4
    # A new graph pass is a new entry, even when the resident row/slot is unchanged.
    run.state.stage(2, [False, False], 1)
    _stage_inputs(run, IterationWindow(columns, torch.zeros(2, dtype=torch.long), entries + torch.tensor([0, 1])))
    run.launch()
    assert run.state.counts.tolist() == [1, 1]
    assert calls[-1].tolist() == [7, 20]


@pytest.mark.parametrize("plain_capture", [False, True])
def test_ordinary_and_device_steps_publish_the_same_actual_loop_progress(monkeypatch, plain_capture):
    from mstar.engine.engine import Engine, ExecutingBatch
    from mstar.engine.resources.step import StepContext
    from mstar.graph.loop_indices import LoopLayout
    from mstar.model.submodule_base import InputMetadata, StopKind, StopPolicy

    loop = _Loop()
    runner, run, _, calls = _runner(monkeypatch, loop, [[1, 1]], limit=4, progress=True)
    policy = StopPolicy(StopKind.EOS, token_signal="new_token", eos_tokens=frozenset({EOS}))
    node = SimpleNamespace(ITERATION_INPUTS={"cond": object()})
    mgmt = SimpleNamespace(
        submodule=node, device_stops={WALK: policy}, loopback_rows=runner._rows, accelerator_graph_runner=None,
        forward_batched=lambda *args, **kwargs: BatchedModelOutput(
            check_stop_buffers={"new_token": torch.tensor([EOS, 2])},
        ),
        forward=lambda *args, **kwargs: {"raw_sample": [torch.tensor([EOS])]},
    )
    engine = Engine.__new__(Engine)
    engine._device = CPU
    engine._submodules = {"n": mgmt}
    engine._loops = {("n", WALK): LoopLayout("decode")}
    engine._runner = SimpleNamespace(request_generation=lambda rid: 1)
    engine._stage_lock = None

    def postprocess(batch, output):
        for values in output.per_rid_outputs.values():
            if "raw_sample" in values:
                values["new_token"] = values.pop("raw_sample")

    engine.postprocess_batch = postprocess

    def eager(rids, counters, passes, walks=None):
        batch = ExecutingBatch(
            node_name="n", per_request_info={rid: SimpleNamespace() for rid in rids},
            per_request_input_metadata={rid: InputMetadata(
                dynamic_loop_iter_counts={"decode": k}, graph_pass=p, iteration_origin=k,
            ) for rid, k, p in zip(rids, counters, passes, strict=True)},
            step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False,
                                     request_walks=walks or {}),
        )
        lease = object() if plain_capture and len(rids) > 1 else None
        ordinary = SimpleNamespace(
            slot_for=lambda lease: SimpleNamespace(row_heads={}, run=None),
            run_forward=lambda *args, **kwargs: mgmt.forward_batched(),
        )
        mgmt.device_stream_tokens = {}
        engine.exec = lambda current: engine._forward(
            current, mgmt, ordinary,
            SimpleNamespace(request_ids=rids, per_request_info=current.per_request_info_wrapped),
            {}, lease, len(rids) > 1, rids, None,
        )
        return engine.exec_and_postprocess(batch)

    # Its successor's window was prepared at zero before this ordinary step.
    window = IterationWindow(
        {"cond": torch.tensor([[7., 8., 9., 10.], [20., 21., 22., 23.]])},
        torch.zeros(2, dtype=torch.long), torch.tensor([[1, 0], [1, 0]]),
    )
    eager([1, 2], [0, 0], [0, 0])
    run.state.stage(2, [False, False], 2)
    _stage_inputs(run, window)
    calls.clear()
    run.launch()
    assert run.state.counts.tolist() == [0, 2], "eager EOS must suppress an already queued captured successor"
    assert [row[1].item() for row in calls] == [21, 22]
    # A new entry is allowed; a row of another walk in the same eager batch
    # must leave this Loop's cursor alone.
    eager([1, 2], [0, 3], [1, 0], walks={1: WALK, 2: "prefill"})
    eager([2], [3], [0])  # the unbatched output representation stops row 2
    run.state.stage(2, [False, False], 1)
    _stage_inputs(run, window._replace(entries=torch.tensor([[1, 1], [1, 0]])))
    run.launch()
    assert run.state.counts.tolist() == [0, 0]
    indices = runner._rows._index([1, 2])
    positions, _ = runner._rows.iteration_head(WALK, engine._loops["n", WALK], indices, run.window.entries[1:])
    assert positions.tolist() == [1, 4]

    failed = ExecutingBatch(
        node_name="n", per_request_info={1: SimpleNamespace(), 2: SimpleNamespace()},
        per_request_input_metadata={rid: InputMetadata(
            dynamic_loop_iter_counts={"decode": 0}, graph_pass=2, iteration_origin=0,
        ) for rid in (1, 2)},
        step_context=StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False),
    )
    engine.exec = lambda batch: BatchedModelOutput(per_rid_outputs={1: {}, 2: {"new_token": [torch.tensor([1])]}})
    engine.postprocess_batch = lambda batch, output: batch.register_failure(1, RuntimeError("bad tail"))
    engine.exec_and_postprocess(failed)
    assert 1 in failed.failed_requests and 2 not in failed.failed_requests
    assert 1 in failed.per_request_info_wrapped, "failure bookkeeping retains its wrapper until worker cleanup"
    positions, live = runner._rows.iteration_head(
        WALK, engine._loops["n", WALK], indices, torch.tensor([[1, 1], [1, 2]]),
    )
    assert positions.tolist() == [1, 1] and live.tolist() == [False, True]

    # A model whose Loop cap is its only stop must not acquire an EOS stop
    # from an ordinary fallback's incidental output, nor use max_tokens as
    # its denoising budget. This fixture has no max_tokens field at all.
    mgmt.device_stops[WALK] = None
    eager([2], [0], [3])
    positions, live = runner._rows.iteration_head(
        WALK, engine._loops["n", WALK], indices, torch.tensor([[1, 1], [1, 3]]),
    )
    assert positions.tolist() == [1, 1] and live.tolist() == [False, True]


@pytest.mark.parametrize("streamed", [False, True])
def test_engine_windows_survive_two_short_runs_with_a_prepared_successor(monkeypatch, streamed):
    """The engine's real immutable columns feed the actual capture body;
    source settlement follows per-row counts, including a real-to-tail crossing.
    """
    from mstar.engine.engine import Engine
    from mstar.engine.resources.iteration_tables import IterationTables
    from mstar.engine.resources.step import StepContext
    from mstar.graph.loop_indices import LoopLayout
    from mstar.model.submodule_base import InputMetadata, IterationInput
    from mstar.streaming.chunk_policy import FixedChunkPolicy
    from mstar.streaming.stream_buffer import StreamBuffer

    loop = _Loop()
    runner, run, _, calls = _runner(monkeypatch, loop, [[1]], bs=1, limit=4, counted=True, progress=True)
    layout = LoopLayout("decode")
    node = SimpleNamespace(
        ITERATION_INPUTS={"cond": IterationInput("source", streamed=streamed, pad="pad" if streamed else None)},
        iteration_tables=IterationTables(CPU),
        request_states={1: {"source": torch.arange(20., 32.), "pad": torch.tensor([-1., -2.])}},
    )
    engine = Engine.__new__(Engine)
    engine._device = CPU
    engine._loops = {("n", WALK): layout}
    engine._runner = SimpleNamespace(request_generation=lambda rid: 1)
    source = StreamBuffer(1, "source", "producer", FixedChunkPolicy(1, True))
    for i in range(3):
        source.pre_read_register(i)
        source.put(i, torch.tensor(float(20 + i)))
    source.signal_done(3)

    def window(k, origin, width):
        snapshot = source.reserve_window(width) if streamed else None
        chunks = {} if snapshot is None else {"source": snapshot.info}
        inputs = {} if snapshot is None or not snapshot.num_items else {1: {"source": [snapshot.data["data"]]}}
        info = SimpleNamespace(metadata=InputMetadata(
            dynamic_loop_iter_counts={"decode": k}, graph_pass=0, iteration_origin=origin, stream_chunks=chunks,
        ))
        ctx = StepContext(request_ids=(1,), graph_walk=WALK, slot=0, capture=False, repeats=width)
        return engine._iteration_columns("n", node, ctx, [1], {1: info}, inputs, actual_loop=layout), snapshot

    first, a = window(0, 0, 4)
    second, b = window(4, 0, 4)  # built before the first actual count exists
    for columns, snapshot, actual in [(first, a, 2), (second, b, 3)]:
        loop.on_iteration = lambda i, actual=actual: run.state.request_yield() if i == actual - 1 else None
        stops, _, _ = _run(run, 1, 4, columns=columns)
        assert stops[RUN_COUNTS].tolist() == [actual]
        if streamed:
            source.settle_window(snapshot.reservation, actual)
    third, c = window(5, 5, 1)
    loop.on_iteration = lambda i: None
    _run(run, 1, 1, columns=third)
    if streamed:
        source.settle_window(c.reservation, 1)
        assert source._consumed == 3 and source._tail_consumed == 3 and not source._reservations
    assert [row.item() for row in calls] == ([20, 21, 22, -1, -2, -2] if streamed else list(range(20, 26)))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_run_membership_keeps_old_readers_across_captured_row_reuse(device_name):
    from contextlib import nullcontext

    from mstar.engine.resources.run_rows import RunRows

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA membership publication")
    device = torch.device(device_name, 0) if device_name == "cuda" else CPU
    rows = RunRows(device, rows=1, capacity=2, binding_width=2)
    from mstar.engine.resources.runner import StepRunner

    assert StepRunner({"membership": rows}).publish([7]) == {7: {}}
    snapshots = [(torch.zeros(1, dtype=torch.long, device=device),
                  torch.zeros((1, 2), dtype=torch.long, device=device)) for _ in range(2)]
    stopped = torch.ones(1, dtype=torch.bool, device=device)
    graphs = []
    for ids, bindings in snapshots:
        rows.boundary(ids, bindings)
        if device.type == "cuda":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                rows.boundary(ids, bindings)
            graphs.append(graph)

    def publication():
        return torch.cuda.stream(rows.publication_stream) if device.type == "cuda" else nullcontext()

    def execute(index):
        if graphs:
            torch.cuda.current_stream().wait_stream(rows.publication_stream)
            graphs[index].replay()
        else:
            rows.boundary(*snapshots[index])
        return snapshots[index]

    def retire(descriptor, ids):
        rows.stop(ids, stopped)
        rows.acknowledge(rows.control.tolist())
        if device.type == "cuda":
            rows.publication_stream.wait_stream(torch.cuda.current_stream())
        return rows.release(descriptor)

    first = rows.reserve_row(7, 1, 0, (11, 21))
    rows.retain(first)  # the first run's output reader outlives later runs
    with publication():
        rows.publish_row(first)
    ids, bindings = execute(0)
    old_ids, old_bindings = ids.clone(), bindings.clone()
    assert ids.tolist() == [first.handle] and bindings.tolist() == [[11, 21]]
    assert rows.copied[(first.handle - 1) % rows.capacity].item() == first.handle
    assert not retire(first, ids)

    # Both captures existed before either generation was published. A long-lived
    # output reader retains one credit while the other cycles through new rows.
    for generation in (2, 3, 4):
        next_row = rows.reserve_row(7, generation, 0, (10 + generation, 20 + generation))
        assert next_row is not None and next_row.handle != first.handle
        with publication():
            rows.publish_row(next_row)
        ids, bindings = execute(generation % 2)
        assert ids.tolist() == [next_row.handle]
        assert bindings.tolist() == [[10 + generation, 20 + generation]]
        rows.retain(next_row)
        assert not retire(next_row, ids)
        assert rows.reserve_row(8, 1, 0, (8, 8)) is None  # readers hold both credits
        assert rows.release(next_row)
    assert old_ids.tolist() == [first.handle] and old_bindings.tolist() == [[11, 21]]
    assert rows.release(first)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("cancel_at", ["unpublished", "offered", "active", "closed"])
def test_run_membership_retracts_zero_work_and_cancels_accepted_rows(cancel_at, device_name):
    from contextlib import nullcontext

    from mstar.engine.resources.run_rows import RunRows

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA membership publication")
    device = torch.device(device_name, 0) if device_name == "cuda" else CPU
    rows = RunRows(device, rows=1, capacity=1, binding_width=1)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    bindings = torch.zeros((1, 1), dtype=torch.long, device=device)

    def publish_call(method, *args):
        with torch.cuda.stream(rows.publication_stream) if device.type == "cuda" else nullcontext():
            method(*args)
        if device.type == "cuda":
            torch.cuda.current_stream().wait_stream(rows.publication_stream)
    descriptor = rows.reserve_row(3, 1, 0, (17,))
    if cancel_at != "unpublished":
        publish_call(rows.publish_row, descriptor)
    if cancel_at == "active":
        rows.boundary(ids, bindings)
        assert ids.tolist() == [descriptor.handle]
    if cancel_at == "closed":
        # Every submitted window ended before the offer became visible.
        publish_call(rows.drain)
    else:
        publish_call(rows.cancel, descriptor)
        if cancel_at != "unpublished":
            rows.boundary(ids, bindings)
            assert ids.tolist() == [0] and bindings.tolist() == [[0]]
    rows.acknowledge(rows.control.tolist())
    assert rows.release(descriptor)
    replacement = rows.reserve_row(3, 2, 0, (23,))
    assert replacement.handle != descriptor.handle
    publish_call(rows.publish_row, replacement)
    rows.boundary(ids, bindings)
    assert ids.tolist() == [replacement.handle] and bindings.tolist() == [[23]]
    with pytest.raises(ValueError):
        publish_call(rows.cancel, descriptor)


def test_a_joining_capture_refuses_an_unbound_owner_before_serving(monkeypatch):
    from mstar.engine.resources.resident_hooks import unbound_horizon
    from mstar.engine.resources.run_rows import RunRows

    original = AcceleratorGraphRunner.ask_run

    def ask(runner, *args):
        original(runner, *args)
        runner._resident.bindings["context"] = SimpleNamespace(check_horizon=unbound_horizon)

    monkeypatch.setattr(AcceleratorGraphRunner, "ask_run", ask)
    membership = RunRows(CPU, rows=2, capacity=4, binding_width=2)
    runner, run, _, calls = _runner(monkeypatch, _Loop(), [[0, 0]], progress=True, membership=membership)
    assert run is None and not calls
    assert "context" in runner._run_failures[WALK]


def test_captured_membership_uses_one_owner_snapshot_and_keeps_output_identity_through_eos_reuse(monkeypatch):
    from mstar.engine.resources.run_rows import RunRows

    membership = RunRows(CPU, rows=3, capacity=4, binding_width=2)
    _, refused, _, _ = _runner(monkeypatch, _Loop(), [[0, 0]], bs=2, limit=5,
                               progress=True, membership=membership, unmanaged=True)
    assert refused is None  # a live join must not inherit an incumbent's private static input
    loop = _Loop()
    runner, run, _, _ = _runner(monkeypatch, loop, [[0, 0]], bs=2, limit=5,
                              progress=True, stream_tokens=True, membership=membership)
    assert run is not None
    owner, lease, _ = run.test_inputs
    runner.test_resource.zero_()  # discard the warm-up forward's owner writes
    # The first physical row is empty. The other one keeps the run alive.
    incumbent = membership.reserve_row(1, 1, 1, (1, 0))
    late = membership.reserve_row(2, 1, 0, (2, 0))
    membership.retain(late)  # its outputs survive replacement on the physical row
    membership.publish_row(incumbent)
    columns = IterationWindow(
        {"cond": torch.tensor([[1., 2., 3., 4., 5.], [7., EOS, 0., 0., 0.]])},
        torch.tensor([0, 0]), torch.tensor([[1, 0], [1, 0]]),
    )
    replacement_columns = IterationWindow(
        {"cond": torch.tensor([[float(EOS), 12., 13., 14., 15.], [0., 0., 0., 0., 0.]])},
        torch.tensor([0, 0]), torch.tensor([[2, 0], [0, 0]]),
    )
    owner.stage_window(lease, [3], replacement_columns)
    replacement = []

    def arrive(iteration):
        if iteration == 1:
            membership.publish_row(late)
        if iteration == 3:
            membership.acknowledge(membership.control.tolist())
            assert not membership.release(late)
            row = membership.reserve_row(3, 2, 0, (3, 1))
            membership.publish_row(row)
            replacement.append(row)

    loop.on_iteration = arrive
    run.state.stream.begin([1])  # dynamic rows are not limited by the original rid list
    stops, outputs, _ = _run(run, 1, 5, columns=columns)
    current = replacement[0]
    assert outputs["h"].shape == (2, 5)  # final-only values also survive physical row reuse
    assert stops[RUN_DESCRIPTORS].tolist() == [
        [0, late.handle, late.handle, current.handle, current.handle], [incumbent.handle] * 5,
    ]
    assert outputs["frame"][0, 1:].tolist() == [14., 18., 18., 24.]
    assert runner.test_resource[1:4].tolist() == [15., 16., 21.]
    seen = []

    def take(identity, ordinal, token):
        seen.append((identity, ordinal, token))
        return True, identity == late.handle and token == EOS

    ids, tokens = run.state.stream.take(take)
    assert [(ordinal, token) for identity, ordinal, token in seen if identity == current.handle] == [(0, EOS), (1, 12)]
    assert len(ids) == len(tokens) == 9 and 0 not in ids
    assert run.state.stream.take(take) == ([], [])
    assert membership.release(late)
    # Unexecuted columns are identity zero even when ordinary values pad the last replay.
    run.state.stage(1, [False], 5)
    run.state.request_yield()
    run.launch()
    assert not run.results(5)[0][RUN_DESCRIPTORS].any()
    # A descriptor can be accepted but its same Loop entry is already stopped.
    # Such an iteration reads trash and never publishes an output identity.
    membership.drain()
    membership.acknowledge(membership.control.tolist())
    stopped = membership.reserve_row(2, 1, 0, (2, 0))
    membership.publish_row(stopped)
    run.state.stream.begin([])
    before = runner.test_resource[1:].clone()
    stopped_outputs = _run(run, 1, 5, columns=columns)[0]
    assert not stopped_outputs[RUN_DESCRIPTORS].any()
    assert not stopped_outputs[RUN_COUNTS].any()
    assert torch.equal(runner.test_resource[1:], before)
    assert run.state.stream.take(take) == ([], [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA publication kernel loading")
def test_publication_never_loads_a_new_cas_kernel_for_row_or_descriptor_values():
    from triton import knobs

    from mstar.engine.resources.run_rows import RunRows

    membership = RunRows(torch.device("cuda", 0), rows=18, capacity=18, binding_width=1)
    membership._generation[0] = 1 << 30  # descriptor words require the same 64-bit ABI at all ages
    descriptors = [membership.reserve_row(row, 1, row, (row + 1,)) for row in (0, 1, 17)]

    def no_late_load(*args):
        raise AssertionError("publication loaded a CUDA module while runs could already be active")

    knobs.runtime.kernel_load_start_hook.add(no_late_load)
    try:
        with torch.cuda.stream(membership.publication_stream):
            for descriptor in descriptors:
                membership.publish_row(descriptor)
                membership.cancel(descriptor)
        membership.publication_stream.synchronize()
    finally:
        knobs.runtime.kernel_load_start_hook.remove(no_late_load)
    assert membership.control.cpu().tolist() == [
        (next(descriptor.handle for descriptor in descriptors if descriptor.row == row) << 2) | 2
        if row in (0, 1, 17) else 0 for row in range(18)
    ]


@pytest.mark.parametrize("case", ["connected", "absent", "custom", "explicit_stop"])
def test_connected_capture_selection_preserves_standalone_rules_and_prepares_owners(case, monkeypatch):
    from mstar.engine.engine import Engine
    from mstar.worker.step_repeats import StepRepeats
    from mstar.worker.worker import Worker

    unit = SimpleNamespace(nodes=("a", "n"), graph_walks=("produce", WALK),
                           streaming_edges=(("a", "n", "x"),))
    config = RepeatConfig(max=4, device_stop="skip") if case == "explicit_stop" else RepeatConfig()
    worker = Worker.__new__(Worker)
    worker._step_repeats = StepRepeats({"n": config}, {("n", WALK): object()})
    worker.scheduler = SimpleNamespace(execution_units={} if case == "absent" else {("a", "produce"): unit})
    requests = worker._capture_run_requests()
    ordinary = worker._step_repeats.device_stops()
    assert ordinary == ({("n", WALK): config} if case == "explicit_stop" else {})
    assert worker._step_repeats.config("n") == config and not worker._step_repeats.may_repeat("n", WALK)
    if case == "absent":
        assert not requests
    else:
        assert set(requests) == {("a", "produce"), ("n", WALK)}
        assert requests["n", WALK] == config
    asks, prepared = [], []
    owner = SimpleNamespace(ask_repeats=prepared.append)
    engine = Engine.__new__(Engine)
    engine._device = torch.device("cuda")
    engine._walk_runs = frozenset() if case == "absent" else frozenset({"produce"})
    engine._graph_runtime = SimpleNamespace(execution_units=lambda walk: (unit,))
    engine._submodules = {"n": SimpleNamespace(
        resources={"owner": owner}, joint_comm_group=SimpleNamespace(world_size=1),
        device_stops={}, refills={}, loopback_rows=None, run_columns={WALK: frozenset({"frame"})},
        submodule=SimpleNamespace(repeat_carry=lambda walk, signals: {}),
    )}
    def stop_after_owner_preparation():
        raise RuntimeError()
    engine._compile_regions = stop_after_owner_preparation
    with pytest.raises(RuntimeError):
        engine.warmup(device_stops=requests)
    assert prepared == ([] if case == "absent" else [WALK])
    engine.get_stop_policies = lambda node, walk: None if case == "custom" else {}
    runner = SimpleNamespace(ask_run=lambda walk, ask: asks.append(ask))
    engine._ask_run("n", runner, WALK, config, ())
    if case in {"absent", "custom"}:
        assert not asks
    else:
        assert len(asks) == 1 and asks[0].token_signal is None and asks[0].limit == config.max
        _, run, _, calls = _runner(monkeypatch, _Loop(), [[1]], bs=1, counted=True, ask=asks[0])
        stops, output, _ = _run(run, 1, 1)
        assert len(calls) == 1 and stops[RUN_COUNTS].tolist() == [1]
        assert output["frame"].tolist() == [[2]]
    if case == "explicit_stop":
        worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
        with pytest.raises(ValueError):
            worker._step_repeats.settle(lambda *a: None, lambda *a: "stream input needs its policy",
                                        {"device stop": worker._device_stop_refusal})


@pytest.mark.parametrize("mismatch", [None, "producer", "producer_walk", "consumer_walk", "signal"])
def test_a_connected_ring_exemption_is_scoped_to_its_exact_members_and_walks(mismatch):
    from mstar.streaming.chunk_policy import FixedChunkPolicy
    from mstar.worker.step_repeats import StepRepeats
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    worker._step_repeats = StepRepeats({"n": RepeatConfig(max=4)}, {("n", WALK): SimpleNamespace(streamed=True)})
    unit = SimpleNamespace(
        nodes=("other" if mismatch == "producer" else "a", "n"),
        graph_walks=("other" if mismatch == "producer_walk" else "produce",
                     "other" if mismatch == "consumer_walk" else WALK),
        streaming_edges=(("other" if mismatch == "producer" else "a", "n",
                          "other" if mismatch == "signal" else "x"),),
    )
    worker.scheduler = SimpleNamespace(execution_units={("a", "produce"): unit})
    worker.streams = SimpleNamespace(
        connections=[SimpleNamespace(edge_name="x", join_dim=None, chunk_policy_factory=lambda: FixedChunkPolicy(2))],
        consumer_node={"x": "n"},
    )
    refusal = worker._ring_refusal_for("a", "x", 8, {"produce"})
    assert (refusal is None) is (mismatch is None)



def test_a_streamed_unit_waits_and_yields_for_its_nontrigger_members(monkeypatch):
    from mstar.worker.step_repeats import StepRepeats

    worker, log = _waiting_on_a_run(monkeypatch, [{OTHER}])
    worker._step_repeats = StepRepeats({"a": RepeatConfig(max=4), "n": RepeatConfig(max=4, ready_waits=True)}, {})
    worker._device_runs = frozenset()  # neither body was requested as a standalone device run
    worker._spec_peek = True
    worker._hold_rows = lambda *args: None
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(yield_runs=log.append))
    pending = _run_step([1], 1, token_stream=None, step_context=SimpleNamespace(repeats=1))
    pending.group = SimpleNamespace(
        submitted=True, unit=SimpleNamespace(streaming_edges=(("a", "n", "x"),)),
        members=((SimpleNamespace(node_name="a"), SimpleNamespace(step_context=SimpleNamespace(repeats=4))),
                 (SimpleNamespace(node_name="n"), pending.node_batch)),
    )
    assert worker._is_device_run(pending)
    assert not worker._await_run(pending, None, [], None)
    assert log.count("a") == 1 and "n" not in log
