"""The batched stop check reads each request's loop counts from the batch's own
snapshot (``InputMetadata``), like the per-request path, not from the shared
forward-pass info the worker may already have advanced for the next step."""
from types import SimpleNamespace

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.model.submodule_base import InputMetadata


class _Submodule:
    def __init__(self):
        self.seen = {}

    def check_stop_batched(self, request_ids, request_infos, host_rows):
        del host_rows
        self.seen = {rid: dict(request_infos[rid].dynamic_loop_iter_counts) for rid in request_ids}
        return {}


def test_batched_check_stop_reads_the_batch_snapshot():
    info = CurrentForwardPassInfo(
        request_id="wire-1", rid_handle=1, graph_walk="decode", fwd_index=0, random_seed=0,
        max_tokens=8, dynamic_loop_iter_counts={"decode_loop": 5},
    )
    batch = ExecutingBatch(
        node_name="LLM", per_request_info={1: info},
        per_request_input_metadata={1: InputMetadata(dynamic_loop_iter_counts={"decode_loop": 3})},
        step_context=StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False),
    )
    sub = _Submodule()
    engine = SimpleNamespace(_submodules={"LLM": SimpleNamespace(submodule=sub)})
    assert Engine.check_stop_for_batch(engine, batch, {}, host_rows=object()) == {}
    assert sub.seen == {1: {"decode_loop": 3}}
