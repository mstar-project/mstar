"""A run on the device (``mstar.engine.device_run``): when a WHILE run ends, and
what its body does each iteration. The node itself needs CUDA; this drives the
same body through the rule in torch, with a backend that loops on its condition."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine import accelerator_graph_runner as runner_mod
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.device_run import RUN_REPLAYS, CudaWhile, DeviceRun, RunAsk, RunState, TokenStream
from mstar.engine.resources.step import BucketKey
from mstar.model.submodule_base import BatchedModelOutput, CarryPair

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

    def new_loop(self, device):
        return 1, 2, torch.zeros(1)

    def stop(self, state, tokens, handle, stream):
        """The test double of the ``run_stop`` kernel, which only CUDA can run."""
        self.state = state
        token = tokens.reshape(len(state.marks), -1)[:, 0]
        state.marks |= (token.unsqueeze(-1) == state.stop_ids).any(-1) & ~state.immune
        if state.stream is not None:
            state.stream.tokens[:, int(state.index)] = token
            state.stream.landed[0] = int(state.index) + 1
        state.index += 1
        done = int(state.marks.sum())
        self.go = (
            done < len(state.marks) and int(state.index) < int(state.limit) and int(state.yield_word[0]) == 0
            and (int(state.refill) == 0 or done < int(state.refill))
        )

    def node_refusal(self, graph):
        return None

    def finish(self, outer, handle, graph):
        return 3

    def launch(self, graph_exec, stream):
        i = 0
        while True:
            self.on_iteration(i)
            self.body()
            i += 1
            if not self.go:
                return


def _runner(monkeypatch, loop, tokens, bs=2, limit=6, refill_at=None, carries=None, stream_tokens=False):
    """A runner holding a captured run whose forward samples ``tokens[iteration]``."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._device, runner._memory_pool, runner._autocast_dtype = CPU, None, None
    runner._submodule_name, runner._capture_stream, runner._resident = "n", None, None
    runner._run_asks, runner._run_failures, runner._generator_walks = {}, {}, set()
    runner._graph_backend = SimpleNamespace(
        while_refusal=lambda: None, while_runtime=lambda: loop,
        raw_stream=lambda: 0, synchronize=lambda: None,
    )
    runner.ask_run(WALK, RunAsk("new_token", frozenset({EOS}), False, limit, carries or {}, ("frame",), ("cond",),
                                refill_at, stream_tokens))
    static = {"cond": torch.zeros(bs), "h": torch.zeros(bs)}
    calls = []

    def forward(in_run=False):
        i = min(len(calls), len(tokens) - 1)
        calls.append(static["cond"].clone())
        token = torch.tensor(tokens[i])
        return BatchedModelOutput(
            check_stop_buffers={"new_token": token},
            row_outputs={"frame": token.float() * 2, "h": static["h"] + 1},
        )

    def capture(body, *args, **kwargs):
        loop.body = body
        loop.capturing(body)
        out = body()
        calls.clear()  # capture records; it doesn't run
        static["h"].zero_()
        return SimpleNamespace(raw_cuda_graph=lambda: 7), out

    monkeypatch.setattr(runner_mod, "capture_into_graph", capture)
    spec = SimpleNamespace(bucket=BucketKey(WALK, bs, bs), slot=0, config=SimpleNamespace(get_config_type=lambda: None))
    warmed = SimpleNamespace(run=forward, static_inputs=static, output=None)
    run, _, _ = runner._capture_run(spec, warmed)
    runner._buckets = {spec.bucket: SimpleNamespace(slots=[SimpleNamespace(run=run)])}
    return runner, run, static, calls


def _run(run, real, repeats, immune=None, refill_at=None, columns=None):
    run.state.stage(real, immune or [False] * real, repeats, refill_at)
    run.stage_columns(columns)
    run.launch()
    return run.results(repeats)


def test_a_long_row_beside_a_short_one_runs_to_the_runs_limit(monkeypatch):
    """A done row doesn't end the run for the row beside it."""
    tokens = [[1, 1], [EOS, 2], [3, 3], [4, 4]]
    _, run, _, calls = _runner(monkeypatch, _Loop(), tokens)
    stops, rows, _ = _run(run, 2, 4)
    assert len(calls) == 4 and stops[RUN_REPLAYS].tolist() == [4, 4]
    assert stops["new_token"].tolist() == [[1, EOS, 3, 4], [1, 2, 3, 4]]
    assert rows["frame"].tolist() == [[2, 18, 6, 8], [2, 4, 6, 8]]
    assert run.state.marks.tolist() == [True, False]


def test_a_run_ends_with_its_last_row_and_padding_keeps_nothing_alive(monkeypatch):
    _, run, _, calls = _runner(monkeypatch, _Loop(), [[1, 0], [EOS, 0], [3, 0]])
    stops, _, _ = _run(run, 1, 5)
    assert len(calls) == 2 and stops[RUN_REPLAYS].tolist() == [2, 2]


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
    assert stops[RUN_REPLAYS].tolist() == [2, 2], "the next run clears the word"
    assert static["h"].tolist() == [5, 5]


def test_a_yield_outlives_the_stage_of_the_step_behind_the_run():
    """N+1 stages while N still runs, so a yield asked before that has to be N's own."""
    running, behind = (RunState(2, "new_token", {EOS}, CPU) for _ in range(2))
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    slots = [SimpleNamespace(run=SimpleNamespace(state=state)) for state in (running, behind)]
    runner._buckets = {"bucket": SimpleNamespace(slots=[*slots, SimpleNamespace(run=None)])}
    runner.yield_runs()
    behind.stage(2, [False, False], 4)
    assert int(running.yield_word[0]) == 1 and int(behind.yield_word[0]) == 0
    runner.yield_runs()
    assert int(behind.yield_word[0]) == 1, "a yield asked once it has staged ends it too"


def test_refill_at_ends_a_run_once_that_share_of_its_rows_is_done(monkeypatch):
    tokens = [[EOS, 1, 1, 1], [0, EOS, 1, 1], [0, 0, 1, 1]]
    _, run, _, calls = _runner(monkeypatch, _Loop(), tokens, bs=4)
    stops, _, _ = _run(run, 4, 6, refill_at=0.5)
    assert stops[RUN_REPLAYS].tolist() == [2] * 4


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


def test_a_run_holds_the_device_handle_its_graph_reads(monkeypatch):
    """The stop kernel reads the WHILE handle by address: freed after capture,
    its block went to the next allocation and the kernel trapped on the GPU."""
    import gc
    import weakref

    handles = []

    class Kept(_Loop):
        def new_loop(self, device):
            outer, handle, on_device = super().new_loop(device)
            handles.append(weakref.ref(on_device))
            return outer, handle, on_device

    loop = Kept()
    _, run, _, _ = _runner(monkeypatch, loop, [[1, 2]])
    loop.body = None  # a real capture keeps no Python body
    gc.collect()
    assert handles[0]() is not None and run is not None


def test_a_failed_while_capture_settles_its_walk_and_a_plain_capture_serves(monkeypatch):
    """One failure is the walk's answer: trying each bucket cost a second capture
    of every one of them."""
    class Failing(_Loop):
        loops = 0

        def new_loop(self, device):
            Failing.loops += 1
            return super().new_loop(device)

        def stop(self, state, tokens, handle, stream):
            raise RuntimeError("no such stream")

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    runner, run, _, _ = _runner(monkeypatch, Failing(), [[1, 2]])
    assert run is None and "no such stream" in runner.run_refusal(WALK)
    spec = SimpleNamespace(bucket=BucketKey(WALK, 2, 2), slot=1, config=SimpleNamespace(get_config_type=lambda: None))
    assert runner._capture_run(spec, None) == (None, None, None) and Failing.loops == 1


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


def test_the_worker_settles_a_run_that_ended_short():
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    ended = []
    engine = SimpleNamespace(end_run_early=lambda batch, ran: ended.append(ran))
    batch = SimpleNamespace(batch=SimpleNamespace(repeats=4), node_batch=object())
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])  # the last column was never written
    rows = SimpleNamespace(buffers={RUN_REPLAYS: torch.tensor([3, 3]), "t": tokens, "one": torch.zeros(2)})
    worker._settle_run(batch, rows, engine)
    assert ended == [3] and batch.batch.repeats == 3 and RUN_REPLAYS not in rows.buffers
    # what keys the prefix chain reads next: nothing the run didn't produce
    assert rows.buffers["t"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert rows.buffers["one"].shape == (2,)


def test_a_short_run_takes_back_a_decode_snapshot_past_where_it_ended():
    """Filed once the chain keys its page, a snapshot from past the row's real
    tokens would serve state at the wrong length; its predecessor stands again."""
    import threading

    from mstar.engine.resources.recurrent.pool import DecodeSnapshot, RecurrentStatePool

    pool = RecurrentStatePool.__new__(RecurrentStatePool)
    pool._lock = threading.RLock()
    held = SimpleNamespace(tokens=12)
    pool._slots = {1: {"main": held}}
    earlier = DecodeSnapshot(3, 8)
    pool._decode = {(1, "main"): DecodeSnapshot(4, 12, replaces=earlier)}
    pool._unkeyed = {(1, "main")}
    freed = []
    pool._drop_decode = lambda key, decode: (pool._unkeyed.discard(key), freed.append(decode.slot))
    pool.rewind_run(SimpleNamespace(segments=[SimpleNamespace(request_id=1, label="main", span=1)]), None, 2, {1})
    assert held.tokens == 10 and freed == [4] and pool._decode[(1, "main")] is earlier


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
    engine._ask_run("n", runner, WALK, SimpleNamespace(device_stop="drop_rows", max=4, refill_at=None), ())
    assert len(asked) == 1 and "prefix" in asked[0]


def test_a_run_that_ends_short_repeats_its_last_iteration_in_the_columns_it_never_wrote():
    """A run on the device can end before its count (a stop, a yield): whatever
    reads a column's last (the step speculated meanwhile, a node's postprocess)
    reads the last iteration that ran, not a column nothing wrote (GPU finding
    14, review-r2 F2)."""
    from mstar.engine.device_run import RUN_REPLAYS, DeviceRun, RunState

    state = RunState(1, "t", {9}, CPU)
    run = DeviceRun(state, {}, lambda: None, 4, torch.zeros(1, dtype=torch.long))
    run.rows["t"] = torch.tensor([[5, 6, 1 << 40, 1 << 40]])
    run.stops["t"] = run.rows["t"].clone()
    state.index.fill_(2)
    stops, rows, _ = run.results(4)
    assert rows["t"].tolist() == [[5, 6, 6, 6]] and stops["t"].tolist() == [[5, 6, 6, 6]]
    assert stops[RUN_REPLAYS].tolist() == [2]
    state.index.fill_(4)
    run.rows["t"] = torch.tensor([[1, 2, 3, 4]])
    assert run.results(4)[1]["t"].tolist() == [[1, 2, 3, 4]]


def test_a_walk_with_iteration_inputs_stays_off_the_device():
    """A run on the device can end short of its count, and the step speculated
    during it took iteration inputs cut on the host for the whole count
    (review-r2 F2): such a walk keeps the host's loop, with the reason."""
    from mstar.engine.engine import Engine

    asked = []
    engine = Engine.__new__(Engine)
    engine._submodules = {"n": SimpleNamespace(
        joint_comm_group=SimpleNamespace(world_size=1), device_stops={}, refills={}, loopback_rows=None,
        submodule=SimpleNamespace(repeat_carry=lambda walk, signals: {}, ITERATION_INPUTS={"x": object()}),
        run_columns={},
    )}
    engine.get_stop_policies = lambda node, walk: {"loop": SimpleNamespace(
        reads_token=True, token_signal="t", eos_tokens=frozenset({9}),
    )}
    runner = SimpleNamespace(
        ask_run=lambda walk, ask: asked.append(ask), refuse_run=lambda walk, why: asked.append(why),
    )
    engine._ask_run("n", runner, WALK, SimpleNamespace(device_stop="skip", max=4, refill_at=None), ())
    assert len(asked) == 1 and "iteration inputs" in asked[0]
    assert "n" in engine._submodules and WALK not in engine._submodules["n"].device_stops


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
        node_name="n", graph_walk=WALK, batch=SimpleNamespace(request_to_worker_graph=dict.fromkeys(rows)),
        node_batch=SimpleNamespace(completion_event=SimpleNamespace(query=lambda: next(landed)), **node_batch),
    )


def test_a_runs_wait_moves_nothing_until_the_step_behind_it_has_admitted(monkeypatch):
    """The loop's own fence covers this work at the top of a pass, not here."""
    worker, log = _waiting_on_a_run(monkeypatch, [set()])
    worker._hold_rows = lambda lane, rows: None
    behind = _run_step([2], 0, launch_started_event=SimpleNamespace(wait=lambda: log.append("admitted")))
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
        _run_step([2], 0, launch_started_event=SimpleNamespace(wait=lambda: None)), [lane], None,
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
    # N+1's stage used to clear this one's yield; a row ready since the speculation gets its ask here
    ([{OTHER}], 1),
])
def test_rows_ready_as_a_run_waits_end_it_only_if_they_can_use_the_room(monkeypatch, ready, yields):
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
