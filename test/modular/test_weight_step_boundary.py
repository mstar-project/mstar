"""A weight step breaks an async TP chain at the settled pipeline boundary."""
from types import SimpleNamespace

from mstar.worker.step_pipeline import StepPipeline


def test_weight_control_sends_no_head_before_finishing_an_async_tp_step():
    events = []
    pipeline = StepPipeline.__new__(StepPipeline)
    pending = SimpleNamespace(node_name="LLM")
    pipeline.pending = pending
    pipeline.side_pending = None
    pipeline.timer = SimpleNamespace(period=False)
    pipeline.weight_ops = SimpleNamespace(
        pending=object(), process=lambda in_flight: events.append("weight") or True,
    )
    pipeline.lockstep = SimpleNamespace(
        maybe_broadcast_nospec=lambda step, head: events.append(("no_head", step, head)),
    )
    pipeline.speculator = SimpleNamespace(
        build=lambda step: events.append("unexpected speculation"),
        chain_ended=lambda: events.append("chain_ended"),
    )
    pipeline.in_flight_rids = set()
    pipeline.is_tp_follower = False
    pipeline._deferred_teardown = lambda laps: None
    pipeline._preamble = lambda laps: None
    pipeline._finish_chain_step = lambda laps: events.append("finish")
    pipeline._finish_side = lambda block: events.append(("side", block))

    pipeline._pass()

    assert events == [
        ("no_head", pending, None), "finish", "chain_ended", ("side", True), "weight",
    ]
