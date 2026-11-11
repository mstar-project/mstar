"""Which decode steps may stand for several iterations (``step_repeats:``)."""
from __future__ import annotations

from types import SimpleNamespace

from mstar.model.submodule_base import (
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    TokenStop,
)
from mstar.testing_utils.engine import make_engine, make_mgmt

NODE, WALK = "decoder", "decode"


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
