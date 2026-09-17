"""The engine tells a stream consumer which step carries the final chunk.

A node fed by a ``StreamBuffer`` withholds state between chunks (a vocoder's
crossfade tail, a token encoder's look-ahead frames) and must flush it on the
last chunk. The worker already knows that chunk (``ExecutingBatch.final_stream_rids``);
this pins that ``prepare_inputs`` receives it as ``is_final_stream_chunk``, per
request, so no consumer has to infer the end of its stream from the data.
"""

from types import SimpleNamespace

from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.model.submodule_base import NodeInputs, NodeSubmodule


class _RecordingSubmodule(NodeSubmodule):
    def __init__(self):
        super().__init__()
        self.seen: dict[str, bool] = {}

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        self.seen[fwd_info.request_id] = kwargs["is_final_stream_chunk"]
        return NodeInputs()

    def forward(self, graph_walk, engine_inputs, **kwargs):  # pragma: no cover - not run
        return {}


def _engine_with(submodule) -> Engine:
    engine = object.__new__(Engine)
    engine._submodules = {"vocoder": SimpleNamespace(submodule=submodule, resources={})}
    engine._enable_nvtx = False
    return engine


def _batch(rids, final):
    return ExecutingBatch(
        node_name="vocoder",
        per_request_info={rid: SimpleNamespace(request_id=rid) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk="chunk", slot=0, capture=False),
        per_request_input_tensors={rid: {} for rid in rids},
        final_stream_rids=set(final),
    )


def test_prepare_inputs_receives_the_final_stream_flag_per_request():
    submodule = _RecordingSubmodule()
    engine = _engine_with(submodule)

    engine.prepare_inputs(_batch(["a", "b", "c"], final={"b"}))

    assert submodule.seen == {"a": False, "b": True, "c": False}


def test_flag_is_false_when_no_stream_ends_this_step():
    submodule = _RecordingSubmodule()
    engine = _engine_with(submodule)

    engine.prepare_inputs(_batch(["a"], final=()))

    assert submodule.seen == {"a": False}
