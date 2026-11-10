"""Which decode steps may stand for several iterations (``step_repeats:``)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.device_run import RunState, WalkAsk
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


@pytest.mark.parametrize(("ready_waits", "repeats", "owed", "want"), [
    (True, 1, False, True),   # the run the waiting rows are owed
    (True, 1, True, False),   # once: a step narrowed to 1 can't keep them out
    (True, 8, False, False),  # it just ran long
    (False, 1, False, False),  # the policy is off: today's behaviour
])
def test_ready_rows_wait_only_for_the_one_run_owed(ready_waits, repeats, owed, want):
    config = RepeatConfig(max=8, ready_waits=ready_waits)
    assert StepRepeats({NODE: config}, {}).owes_run(NODE, repeats, owed) is want


def test_a_device_run_starts_with_only_real_rows_live_and_keeps_a_column_an_iteration():
    state = RunState(4, 3, {2}, torch.device("cpu"))
    state.ran[0] = 3
    state.stage(2, [False, True], limit=3)
    assert state.marks.tolist() == [False, False, True, True], "padding rows start stopped"
    assert (state.continuation.item(), state.limit.item(), state.immune.tolist()) == (1, 3, [False, True, False, False])
    assert state.ran.item() == 0, "the last run on the slot no longer counts"

    # as ``run_stop`` lands them: an iteration's column, then the count
    state.tokens[0], state.tokens[1] = torch.tensor([5, 6, 0, 0]), torch.tensor([7, 8, 0, 0])
    state.ran[0] = 2
    state.names = ("new_token",)
    results, landing = state.results(2)
    assert results["new_token"][:2].tolist() == [[5, 7], [6, 8]]
    assert landing is state


def test_a_device_run_that_ended_short_settles_its_width_before_anything_reads_it():
    """A run ended short wrote only the columns it ran, so the host cuts them
    to that width and reads the step as the width it really had."""
    rows = HostRows((1, 2), {"new_token": torch.tensor([[5, 9, 0], [6, 2, 0]])})
    ctx = SimpleNamespace(repeats=3)
    batch_n = SimpleNamespace(node_batch=SimpleNamespace(step_context=ctx), graph_walk=WALK)
    run = RunState(2, 3, {2}, torch.device("cpu"))
    run.ran[0] = 2

    StepExecutor._settle_run(batch_n, rows, run)

    assert rows.buffers["new_token"].tolist() == [[5, 9], [6, 2]]
    assert ctx.repeats == 2


def test_a_slot_with_a_device_run_never_replays_unstaged():
    """A walk that shares the capture of an asked walk gets its WHILE graph but
    no stop policy: it must fail, not replay on the last run's device state."""
    ctx = SimpleNamespace(graph_walk="alias", resident=None, repeats=1)
    batch = SimpleNamespace(step_context=ctx, preplan_event=None)
    run = SimpleNamespace(stage=lambda *args: None)
    runner = SimpleNamespace(slot_run=lambda lease: run, run_forward=lambda *args, **kwargs: BatchedModelOutput())
    inputs = SimpleNamespace(per_request_info={})

    with pytest.raises(KeyError, match="alias"):
        make_engine()._forward(batch, make_mgmt(), runner, inputs, {}, object(), True, [1])


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
