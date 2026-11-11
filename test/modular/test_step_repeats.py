"""Which decode steps may stand for several iterations (``step_repeats:``)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.device_run import CudaWhile, RunState, WalkAsk
from mstar.engine.engine import Engine
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import (
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    HostRows,
    StopKind,
    StopPolicy,
    TokenStop,
)
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.worker.services import _step_repeats  # noqa: PLC2701
from mstar.worker.step_executor import StepExecutor
from mstar.worker.step_repeats import RepeatConfig, StepRepeats, loop_signals, parse_step_repeats

NODE, WALK = "decoder", "decode"


def _loop(*outputs: GraphEdge) -> Loop:
    edges = [
        GraphEdge(next_node=NODE, name="text_inputs"), GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token"), *outputs,
    ]
    node = GraphNode(name=NODE, input_names=["text_inputs"], outputs=edges)
    return Loop(name="decode_loop", section=node, max_iters=8, outputs=[])


@pytest.mark.parametrize(("block", "match"), [
    ({NODE: {"max": 0}}, "at least 1"),
    ({NODE: {"max": 8, "auto": True}}, "not settings"),
    ({NODE: {"max": 1, "ready_waits": True}}, "needs max"),
    ({NODE: {"max": 1, "device_stop": True}}, "needs max"),
    ({"other": {"max": 8}}, "not nodes"),
])
def test_a_setting_that_cannot_run_is_refused_at_load(block, match):
    with pytest.raises(ValueError, match=match):
        parse_step_repeats(block, {NODE})


def test_a_lockstep_node_is_refused_before_any_rank_launches():
    model = SimpleNamespace(get_graph_walk_graphs=lambda: {WALK: _loop()})
    with pytest.raises(ValueError, match="lockstep"):
        _step_repeats(model, {NODE: {"max": 8}}, {(NODE, WALK)}, {NODE}, ())


def test_a_per_iteration_output_other_than_the_client_token_refuses_its_walk():
    good = loop_signals(_loop(), NODE, ())
    assert good.refusal is None and good.loop_back == {"text_inputs"}
    bad = loop_signals(_loop(GraphEdge(next_node="vocoder", name="frames")), NODE, ())
    assert "read once an iteration" in bad.refusal


def test_a_node_the_config_names_with_no_walk_that_can_repeat_stops_the_worker():
    repeats = StepRepeats({NODE: RepeatConfig(max=8)}, {(NODE, WALK): loop_signals(_loop(), NODE, ())})
    with pytest.raises(ValueError, match="no walk of it can"):
        repeats.settle(lambda node, walk: "resource 'attn' allows one iteration a step")


@pytest.mark.parametrize(("ready_waits", "walks", "repeats", "owed", "want"), [
    (True, [WALK], 1, False, True),   # the run the waiting rows are owed
    (True, [WALK], 1, True, False),   # once: a step narrowed to 1 can't keep them out
    (True, [WALK], 8, False, False),  # it just ran long
    (True, ["prefill"], 1, False, False),  # a prompt step of the node can never repeat
    (True, ["prefill", WALK], 1, False, True),  # a mixed step has decode rows
    (False, [WALK], 1, False, False),  # the policy is off: today's behaviour
])
def test_ready_rows_wait_only_for_the_one_run_owed(ready_waits, walks, repeats, owed, want):
    config = RepeatConfig(max=8, ready_waits=ready_waits)
    steps = StepRepeats({NODE: config}, {})
    steps._walks.add((NODE, WALK))
    assert steps.owes_run(NODE, walks, repeats, owed) is want


def test_a_device_run_starts_with_only_real_rows_live_and_keeps_a_column_an_iteration():
    state = RunState(4, 3, {2}, torch.device("cpu"))
    state.stage((7, 8), [False, True], 3, None)
    assert state.marks.tolist() == [False, False, True, True], "padding rows start stopped"
    assert (state.continuation.item(), state.limit.item(), state.immune.tolist()) == (1, 3, [False, True, False, False])

    state.tokens[:] = 0
    for token in ([[5], [6], [0], [0]], [[7], [8], [0], [0]]):
        state.keep_token(torch.tensor(token).reshape(-1))
        state.index += 1
    results = state.results(3, ("new_token",))
    assert results["new_token"][:2, :2].tolist() == [[5, 7], [6, 8]]
    assert state.index.item() == 2


@pytest.mark.parametrize("before_bucket", [4, 8])  # the same bucket, or a wider one
@pytest.mark.parametrize("rows, seeded", [
    ((7, 8), [True, False]),
    ((8, 7), [False, False]),  # moved
    ((7,), [False]),  # one left
    ((7, 8, 9), [False, False, False]),  # one joined
])
def test_a_device_run_after_a_run_of_its_rows_starts_with_that_runs_marks(rows, seeded, before_bucket):
    """A step speculated past a row's stop ends once the rest stop, whatever the
    model samples after one; rows that moved, left or joined start unmarked."""
    before, run = (RunState(bucket, 3, {2}, torch.device("cpu")) for bucket in (before_bucket, 4))
    before.stage((7, 8), [False, False], 3, None)
    before.marks[0] = True  # the run's kernel saw request 7 stop

    run.stage(rows, [False] * len(rows), 3, before)
    want = seeded + [True] * (4 - len(rows))
    assert run.marks.tolist() == want, "a stopped row keeps its mark, padding rows theirs"
    run.stage(rows, [False] * len(rows), 3, run)
    assert run.marks.tolist() == want, "a slot after its own run keeps its marks"
    run.stage(rows, [False] * len(rows), 3, None)
    assert run.marks.tolist() == [False] * len(rows) + [True] * (4 - len(rows)), "no step before: unmarked"


@pytest.mark.parametrize("is_new_iter", [False, True])
def test_only_a_step_that_repeats_continues_the_run_before_it(is_new_iter):
    """A step that repeats runs every row of N one iteration on in N's Loop, so a
    row N saw stop is one it drops. The first step of a Loop that starts again
    for the same request is no new iteration of N's: it starts from no marks."""
    from mstar.worker.speculation import Speculator

    repeats = StepRepeats({NODE: RepeatConfig(max=8)}, {})
    repeats._walks.add((NODE, WALK))
    speculator = Speculator.__new__(Speculator)
    speculator._consecutive = 1
    speculator._step_repeats = repeats
    speculator._graph_runtime = SimpleNamespace(loop_room=lambda *args: 8)
    speculator._innermost_loop_by_node = {(WALK, NODE): "decode_loop"}
    ctx = SimpleNamespace(repeats=1, continues=None)
    speculation = SimpleNamespace(
        scheduled_batch=SimpleNamespace(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph={1: 0}, chunk_ranges=None,
        ),
        node_batch=SimpleNamespace(step_context=ctx), is_new_iter=is_new_iter, is_same_node=True,
        continuing_rids=[1],
    )
    pending = SimpleNamespace(node_batch=SimpleNamespace(step_context=SimpleNamespace(repeats=8)))

    speculator._stamp_repeats(speculation, pending, owes_run=False)
    assert ctx.continues is (pending.node_batch.step_context if is_new_iter else None)


def test_a_device_run_that_ended_short_settles_its_width_before_anything_reads_it():
    """A run ended short wrote only the columns it ran, so the host cuts them
    to that width and reads the step as the width it really had."""
    rows = HostRows((1, 2), {"new_token": torch.tensor([[5, 9, 0], [6, 2, 0]])}, torch.tensor([2]))
    ctx = SimpleNamespace(repeats=3)
    batch_n = SimpleNamespace(node_batch=SimpleNamespace(step_context=ctx), graph_walk=WALK)

    StepExecutor._settle_device_run(batch_n, rows)

    assert rows.buffers["new_token"].tolist() == [[5, 9], [6, 2]]
    assert ctx.repeats == 2


def test_a_device_run_leaves_a_resources_host_row_whole():
    """A one-iteration run with a log-prob row: that row is one value a row, not a column."""
    rows = HostRows(
        (1, 2), {"new_token": torch.tensor([[5], [6]]), "sampled_logprob": torch.tensor([-0.5, -0.7])},
        torch.tensor([1]),
    )
    ctx = SimpleNamespace(repeats=1)
    batch_n = SimpleNamespace(node_batch=SimpleNamespace(step_context=ctx), graph_walk=WALK)

    StepExecutor._settle_device_run(batch_n, rows)

    assert rows.buffers["sampled_logprob"].tolist() == pytest.approx([-0.5, -0.7])
    assert rows.buffers["new_token"].tolist() == [[5], [6]]


def test_a_capture_another_walk_replays_is_refused_at_load_for_a_device_run():
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._run_asks, runner._run_failures = {WALK: object()}, {}
    spec = SimpleNamespace(
        bucket=SimpleNamespace(graph_walk=WALK), config=SimpleNamespace(replay_graph_walks=[WALK, "alias"]),
    )

    assert runner._capture_run(spec, None) is None
    assert "alias" in runner.device_run_refusal(WALK)

    # an ask of its own for the other walk, with other stop ids, is no cover
    runner._run_asks, runner._run_failures = {WALK: object(), "alias": object()}, {}
    assert runner._capture_run(spec, None) is None and "alias" in runner.device_run_refusal(WALK)


def test_a_slot_with_a_device_run_never_replays_unstaged():
    """A walk that shares the capture of an asked walk gets its WHILE graph but
    no stop policy: it must fail, not replay on the last run's device state."""
    ctx = SimpleNamespace(graph_walk="alias", resident=None, repeats=1, continues=None)
    batch = SimpleNamespace(step_context=ctx, preplan_event=None)
    run = SimpleNamespace(stage=lambda *args: None)
    runner = SimpleNamespace(slot_run=lambda lease: run, run_forward=lambda *args, **kwargs: BatchedModelOutput())
    inputs = SimpleNamespace(per_request_info={})

    with pytest.raises(KeyError, match="alias"):
        make_engine()._forward(batch, make_mgmt(), runner, inputs, {}, object(), True, [1])


def test_a_step_stages_its_device_run_after_the_run_of_the_step_it_continues():
    """The engine hands ``stage`` the run of the step ``ctx.continues`` names, and clears the link."""
    previous_run, staged = object(), []
    ctx = SimpleNamespace(
        graph_walk="alias", resident=None, repeats=1, continues=SimpleNamespace(slot_lease="before"),
    )
    batch = SimpleNamespace(step_context=ctx, preplan_event=None)
    run = SimpleNamespace(stage=lambda *args: staged.append(args))
    runner = SimpleNamespace(
        slot_run=lambda lease: previous_run if lease == "before" else run,
        run_forward=lambda *args, **kwargs: BatchedModelOutput(), repeat=lambda lease, out, repeats: out,
    )
    inputs = SimpleNamespace(per_request_info={})

    engine = make_engine()
    engine._ignore_eos = lambda *args: [False]
    engine._forward(batch, make_mgmt(), runner, inputs, {}, object(), True, [1])

    assert staged[0][-1] is previous_run
    assert ctx.continues is None


def test_a_device_run_refuses_a_walk_with_two_token_stops():
    """The run's kernel ends on one policy's ids; a second policy's would never end it."""
    policies = {
        loop: StopPolicy(StopKind.AR, token_signal="new_token", eos_tokens={eos}) for loop, eos in (("a", 2), ("b", 3))
    }
    submodule = SimpleNamespace(
        get_stop_policies=lambda walk: policies, repeat_carry=lambda walk, signals: {"new_token": "input_ids"},
    )

    with pytest.raises(ValueError, match="two token stops|token stop policies"):
        Engine._run_ask(NODE, make_mgmt(submodule), WALK, WalkAsk(frozenset({"text_inputs"}), 8))


def test_a_plain_ar_node_is_refused_until_it_declares_what_a_repeat_reads():
    """Only a node that says so reads its token off the sampler and its rows off
    the resident tables; a TokenStop alone vouches for neither. A stop that
    withholds a frame keeps check_stop, which is what withholds it."""

    class Plain(ARNodeSubmodule):
        prepare_inputs = preprocess = forward = forward_batched = None

    node = Plain()
    node.token_stop = TokenStop(loop="decode_loop", buffer="new_token", stop_ids=frozenset({2}), sampler="sampler")
    sampler = SimpleNamespace(
        enable_device_loopback_reader=lambda: None,
        lean_capture=False, force_double_buffer=False, supports_preplan=False,
    )
    engine = make_engine(_submodules={NODE: make_mgmt(node, resources={"sampler": sampler})})

    engine.enable_repeats(NODE, {WALK: WalkAsk(frozenset({"text_inputs"}))})

    assert not node.resident_walks() and node.device_loopback is None
    assert node.get_stop_policies(WALK) is None, "a walk that can't repeat keeps check_stop"

    class Declared(Plain):
        device_loopback = DeviceLoopback(WALK, "text_inputs", "sampler", "new_token")

        def resident_walks(self):
            return frozenset({WALK})

    declared = Declared()
    declared.token_stop = node.token_stop
    declared.enable_repeats({WALK})
    assert declared.get_stop_policies(WALK), "a walk that can repeat reads its stop in the runtime"
    declared.token_stop = TokenStop(
        loop="decode_loop", buffer="new_token", stop_ids=frozenset({2}), sampler="sampler", withhold=("frame",),
    )
    assert declared.get_stop_policies(WALK) is None


def test_the_spec_cap_counts_steps_not_the_iterations_a_step_asked_for():
    from mstar.worker.speculation import Speculator

    speculator = Speculator.__new__(Speculator)
    speculator._consecutive = 0
    launched = SimpleNamespace(is_yield_away=False, node_batch=SimpleNamespace(step_context=SimpleNamespace(repeats=8)))
    speculator.note_launched(launched)
    assert speculator._consecutive == 1


@pytest.mark.parametrize(
    ("which", "name", "sibling"),
    [(0, "libnvJitLink.so.13", "nvjitlink"), (1, "libcudadevrt.a", "cuda_runtime")],
    ids=["nvjitlink", "devrt"],
)
def test_the_link_libraries_are_looked_for_next_to_nvrtc_before_the_toolkit(
    tmp_path, monkeypatch, which, name, sibling,
):
    """Both have to come from one CUDA release, the NVRTC's own before a toolkit that may not be."""
    link_name, devrt_name = "libnvJitLink.so.13", "libcudadevrt.a"
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    real_exists = os.path.exists  # this host's /usr/local/cuda must not answer for the fake dirs
    monkeypatch.setattr(os.path, "exists", lambda p: str(p).startswith(str(tmp_path)) and real_exists(p))
    nvrtc, toolkit = tmp_path / "cu13" / "lib", tmp_path / "cuda" / "lib64"
    for d in (nvrtc, toolkit):
        d.mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    # the wheels that split NVRTC and the rest into directories: ``nvidia/<pkg>/lib``
    split = tmp_path / "nvidia" / "cuda_nvrtc" / "lib"
    split.mkdir(parents=True)
    for pkg in ("nvjitlink", "cuda_runtime"):
        (tmp_path / "nvidia" / pkg / "lib").mkdir(parents=True)

    def found(nvrtc_dir):
        return CudaWhile.link_libraries(str(nvrtc_dir))

    other, other_sibling = (devrt_name, "cuda_runtime") if which == 0 else (link_name, "nvjitlink")
    (nvrtc / other).touch()
    (toolkit / name).touch()
    assert found(nvrtc) == (None, None), "a toolkit with only one of them is not the release of the other"
    (toolkit / other).touch()
    assert found(nvrtc) == (str(toolkit / link_name), str(toolkit / devrt_name))
    (tmp_path / "nvidia" / sibling / "lib" / name).touch()
    (tmp_path / "nvidia" / other_sibling / "lib" / other).touch()
    split_dir = str(split / ".." / ".." / "%s" / "lib")
    assert found(split) == (split_dir % "nvjitlink" + "/" + link_name, split_dir % "cuda_runtime" + "/" + devrt_name)
    (nvrtc / name).touch()
    assert found(nvrtc) == (str(nvrtc / link_name), str(nvrtc / devrt_name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiles and launches the stop kernel")
@pytest.mark.parametrize("bs", [3, 130, 1100])  # a warp, several, more than one block holds
def test_the_stop_kernel_links_to_sass_and_marks_the_rows_that_sampled_a_stop_id(bs):
    """``load_kernels`` compiles for the device's arch and links with nvJitLink, with no JIT of PTX by the
    driver; a thread a row marks each row, an immune row never, and the run goes on while any row is live."""
    state = RunState(bs, 4, {2}, torch.device("cuda"))
    real = bs - 1  # the last row pads
    state.stage(tuple(range(real)), [row == 1 for row in range(real)], 4, None)
    stream = torch.cuda.current_stream().cuda_stream
    cuda = CudaWhile()
    tokens = torch.ones(bs, 2, dtype=torch.long, device="cuda")  # a stride of 2: the token is column 0
    tokens[::2, 0] = 2

    cuda.stop(state, tokens, stream)
    torch.cuda.synchronize()
    want = [row % 2 == 0 or row == real for row in range(bs)]
    assert (state.marks.tolist(), state.index.item(), state.continuation.item()) == (want, 1, 1)

    tokens[:, 0] = 2
    cuda.stop(state, tokens, stream)
    torch.cuda.synchronize()
    assert state.marks.tolist() == [row != 1 for row in range(bs)], "the immune row is never marked"
    assert (state.index.item(), state.continuation.item()) == (2, 1), "it keeps the run going"

    for _ in range(2):
        cuda.stop(state, tokens, stream)
    torch.cuda.synchronize()
    assert (state.index.item(), state.continuation.item()) == (4, 0), "the limit ends it"


class _CudaRun:
    """A captured WHILE run of ``bs`` rows whose every row samples stop id 2 once
    ``stop_at`` iterations have run (never while it is 0), and 1 before."""

    def __init__(self, bs: int, stop_at: torch.Tensor):
        from mstar.engine.device_run import WhileGraph

        cuda = CudaWhile()
        cuda.load_kernels()
        self.state = RunState(bs, 8, {2}, torch.device("cuda"))
        self.counter = torch.zeros(1, dtype=torch.long, device="cuda")
        child, side = torch.cuda.CUDAGraph(keep_graph=True), torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(child, stream=side):
            self.counter.add_(1)
            token = torch.where(self.counter == stop_at, 2, 1).repeat(bs)
            self.state.keep_token(token)
            cuda.stop(self.state, token, torch.cuda.current_stream().cuda_stream)
        self.graph = WhileGraph(cuda, child, self.state.continuation, lambda: torch.cuda.current_stream().cuda_stream)

    def replay(self) -> None:
        self.counter.zero_()
        self.graph.replay()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures and launches a WHILE run")
def test_a_device_run_ends_once_every_row_stopped_and_the_next_step_of_its_rows_after_one_iteration():
    """A run whose rows all sample a stop at iteration k reads ``index`` k + 1. The
    step staged straight after it (no host sync: speculated) for the same rows
    ends after one iteration, though its model never samples a stop; one for
    other rows runs to its limit."""
    stop_at = torch.zeros(1, dtype=torch.long, device="cuda")
    first, second = _CudaRun(3, stop_at), _CudaRun(3, stop_at)
    rows, k = (7, 8), 2

    stop_at.fill_(k + 1)
    first.state.stage(rows, [False, False], 8, None)
    first.replay()
    stop_at.fill_(0)
    second.state.stage(rows, [False, False], 8, first.state)
    second.replay()
    torch.cuda.synchronize()
    assert first.state.index.item() == k + 1
    assert first.state.results(k + 1, ("t",))["t"][:2].tolist() == [[1] * k + [2]] * 2
    assert second.state.index.item() == 1

    first.state.stage((8, 9), [False, False], 8, second.state)
    first.replay()
    torch.cuda.synchronize()
    assert first.state.index.item() == 8
def test_a_node_no_deployment_asked_to_repeat_replays_as_main():
    """With ``step_repeats`` unset a decode step's forward is main's: the replay
    alone, nothing of the repeat path around it (Qwen3.5 at c64 spent ~30 us a
    step there)."""
    replayed = BatchedModelOutput()

    class Runner:
        def __init__(self):
            self.calls = []

        def run_forward(self, lease, preprocessed, **kwargs):
            self.calls.append("run_forward")
            return replayed

        def __getattr__(self, name):
            self.calls.append(name)
            return lambda *args: None

    ctx = SimpleNamespace(graph_walk=WALK, resident=None, repeats=1, continues=None)
    batch = SimpleNamespace(step_context=ctx, preplan_event=None)
    inputs = SimpleNamespace(per_request_info={})
    runner = Runner()
    make_engine()._forward(batch, make_mgmt(), runner, inputs, {}, object(), True, [1])
    assert runner.calls == ["run_forward"]

    # a node a deployment asked to repeat takes its iterations from the repeat path
    runner = Runner()
    make_engine()._forward(batch, make_mgmt(may_repeat=True), runner, inputs, {}, object(), True, [1])
    assert "repeat" in runner.calls
