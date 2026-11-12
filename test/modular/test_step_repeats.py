"""Which decode steps may stand for several iterations (``step_repeats:``)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import (
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    TokenStop,
)
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.worker.services import _step_repeats  # noqa: PLC2701
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
    mgmt = make_mgmt(node, resources={"sampler": sampler})
    engine = make_engine(_submodules={NODE: mgmt})

    engine.enable_repeats(NODE, {WALK})

    assert mgmt.may_repeat

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


def test_a_node_no_deployment_asked_to_repeat_replays_as_main():
    """With ``step_repeats`` unset a decode step's forward is the replay alone:
    neither ``repeat`` nor ``slot_run`` is called."""
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
    assert "repeat" not in runner.calls and "slot_run" not in runner.calls
